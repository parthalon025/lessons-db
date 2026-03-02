"""Tests for FSRS spaced repetition module."""

from datetime import date, timedelta

import pytest
from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db, insert_lesson
from lessons_db.fsrs import (
    GRADE_AGAIN,
    GRADE_EASY,
    GRADE_GOOD,
    GRADE_HARD,
    INITIAL_S,
    NEGATIVE_INITIAL_STABILITY,
    OUTCOME_TO_GRADE,
    POSITIVE_INITIAL_DIFFICULTY,
    POSITIVE_INITIAL_STABILITY,
    POSITIVE_OUTCOME_TO_GRADE,
    backfill_fsrs_defaults,
    compute_retrievability,
    enforce_positive_ratio,
    ensure_fsrs_columns,
    get_due_lessons,
    get_fading_level,
    interleave_due_lessons,
    record_review,
    update_difficulty,
    update_stability,
)

# ---------------------------------------------------------------------------
# compute_retrievability
# ---------------------------------------------------------------------------


class TestComputeRetrievability:
    """R = (1 + F*t/S)^DECAY where F=19/81, DECAY=-0.5"""

    def test_r_at_zero_days(self):
        """t=0 -> R=1.0 (just reviewed, fully remembered)."""
        assert compute_retrievability(stability=1.0, days_since_review=0.0) == 1.0

    def test_r_at_stability_equals_ninety_percent(self):
        """t=S -> R=0.9 (FSRS defines S as the time for 90% retention)."""
        r = compute_retrievability(stability=1.0, days_since_review=1.0)
        assert abs(r - 0.9) < 1e-9

    def test_r_at_nine_days(self):
        """t=9, S=1 -> R ≈ 0.567 (well below 0.9 threshold)."""
        r = compute_retrievability(stability=1.0, days_since_review=9.0)
        assert abs(r - 0.5669) < 0.001

    def test_r_decreases_over_time(self):
        """R should monotonically decrease as days increase."""
        r1 = compute_retrievability(stability=2.0, days_since_review=1.0)
        r5 = compute_retrievability(stability=2.0, days_since_review=5.0)
        r10 = compute_retrievability(stability=2.0, days_since_review=10.0)
        assert r1 > r5 > r10

    def test_higher_stability_slower_decay(self):
        """Higher S means slower forgetting at same elapsed time."""
        r_low_s = compute_retrievability(stability=1.0, days_since_review=5.0)
        r_high_s = compute_retrievability(stability=5.0, days_since_review=5.0)
        assert r_high_s > r_low_s

    def test_r_always_between_zero_and_one(self):
        """R should always be in (0, 1]."""
        for s in [0.1, 1.0, 10.0, 100.0]:
            for t in [0.0, 1.0, 10.0, 100.0, 1000.0]:
                r = compute_retrievability(stability=s, days_since_review=t)
                assert 0.0 < r <= 1.0, f"R={r} out of range for S={s}, t={t}"

    def test_r_with_large_stability(self):
        """Very high S means almost no forgetting at moderate t."""
        r = compute_retrievability(stability=100.0, days_since_review=10.0)
        assert r > 0.95


# ---------------------------------------------------------------------------
# update_stability
# ---------------------------------------------------------------------------


class TestUpdateStability:
    def test_good_grade_increases_stability(self):
        """Good review should increase stability."""
        old_s = 2.0
        new_s = update_stability(old_S=old_s, old_D=5.0, grade=GRADE_GOOD, R=0.9)
        assert new_s > old_s

    def test_easy_grade_still_increases_stability(self):
        """Easy review should still increase stability (just less than Good).

        In FSRS-6, w15 (Easy modifier) is 0.25 — a damping factor. Easy cards
        get a *smaller* stability boost per review because the algorithm assumes
        easy material doesn't need as large a boost. The Easy advantage is
        through reduced difficulty, which compounds over many reviews.
        """
        old_s = 2.0
        s_easy = update_stability(old_S=old_s, old_D=5.0, grade=GRADE_EASY, R=0.9)
        assert s_easy > old_s

    def test_again_grade_decreases_stability(self):
        """Again (lapse) should decrease stability significantly."""
        old_s = 10.0
        new_s = update_stability(old_S=old_s, old_D=5.0, grade=GRADE_AGAIN, R=0.9)
        assert new_s < old_s

    def test_stability_always_positive(self):
        """Stability must never go to zero or negative."""
        new_s = update_stability(old_S=0.4, old_D=10.0, grade=GRADE_AGAIN, R=0.1)
        assert new_s > 0.0

    def test_low_retrievability_bigger_boost(self):
        """Successful review at low R should boost stability more (desirable difficulty)."""
        s_high_r = update_stability(old_S=5.0, old_D=5.0, grade=GRADE_GOOD, R=0.9)
        s_low_r = update_stability(old_S=5.0, old_D=5.0, grade=GRADE_GOOD, R=0.5)
        assert s_low_r > s_high_r


# ---------------------------------------------------------------------------
# update_difficulty
# ---------------------------------------------------------------------------


class TestUpdateDifficulty:
    def test_again_increases_difficulty(self):
        """Failing a review should make it harder."""
        old_d = 5.0
        new_d = update_difficulty(old_D=old_d, grade=GRADE_AGAIN)
        assert new_d > old_d

    def test_easy_decreases_difficulty(self):
        """Easy review should make it less difficult."""
        old_d = 5.0
        new_d = update_difficulty(old_D=old_d, grade=GRADE_EASY)
        assert new_d < old_d

    def test_difficulty_bounded_low(self):
        """Difficulty should not go below 1.0."""
        d = 1.5
        for _ in range(50):
            d = update_difficulty(old_D=d, grade=GRADE_EASY)
        assert d >= 1.0

    def test_difficulty_bounded_high(self):
        """Difficulty should not go above 10.0."""
        d = 8.0
        for _ in range(50):
            d = update_difficulty(old_D=d, grade=GRADE_AGAIN)
        assert d <= 10.0

    def test_good_grade_mild_change(self):
        """Good grade should produce modest difficulty change."""
        old_d = 5.0
        new_d = update_difficulty(old_D=old_d, grade=GRADE_GOOD)
        assert abs(new_d - old_d) < 2.0  # not a drastic shift


# ---------------------------------------------------------------------------
# DB integration: get_due_lessons
# ---------------------------------------------------------------------------


@pytest.fixture
def fsrs_db(tmp_path):
    """Create a test DB with FSRS columns and some lessons."""
    db_path = tmp_path / "test_fsrs.db"
    conn = init_db(db_path)
    ensure_fsrs_columns(conn)
    return conn


def _insert_reviewed_lesson(conn, title, stability, difficulty, days_ago, polarity="negative"):
    """Helper: insert a lesson with FSRS fields set."""
    lesson_id = insert_lesson(conn, {"title": title, "created_date": date.today().isoformat(), "polarity": polarity})
    review_date = (date.today() - timedelta(days=days_ago)).isoformat()
    conn.execute(
        "UPDATE lessons SET stability = ?, difficulty = ?, last_review_date = ? WHERE id = ?",
        [stability, difficulty, review_date, lesson_id],
    )
    conn.commit()
    return lesson_id


class TestGetDueLessons:
    def test_returns_lessons_below_threshold(self, fsrs_db):
        """Lessons with R < threshold should be returned."""
        # S=1.0, 20 days ago -> R very low (well below 0.9)
        _insert_reviewed_lesson(fsrs_db, "forgotten lesson", stability=1.0, difficulty=5.0, days_ago=20)
        # S=100, 1 day ago -> R very high (above 0.9)
        _insert_reviewed_lesson(fsrs_db, "fresh lesson", stability=100.0, difficulty=5.0, days_ago=1)

        due = get_due_lessons(fsrs_db, threshold=0.9)
        titles = [d["title"] for d in due]
        assert "forgotten lesson" in titles
        assert "fresh lesson" not in titles

    def test_excludes_never_reviewed(self, fsrs_db):
        """Lessons without last_review_date should NOT appear (never reviewed = not yet in rotation)."""
        insert_lesson(fsrs_db, {"title": "unreviewed", "created_date": date.today().isoformat()})
        due = get_due_lessons(fsrs_db, threshold=0.9)
        assert len(due) == 0

    def test_empty_db(self, fsrs_db):
        """No lessons -> empty list."""
        due = get_due_lessons(fsrs_db, threshold=0.9)
        assert due == []

    def test_returns_computed_retrievability(self, fsrs_db):
        """Each result should include the computed R value."""
        _insert_reviewed_lesson(fsrs_db, "due", stability=1.0, difficulty=5.0, days_ago=15)
        due = get_due_lessons(fsrs_db, threshold=0.9)
        assert len(due) == 1
        assert "retrievability" in due[0]
        assert 0.0 < due[0]["retrievability"] < 0.9


# ---------------------------------------------------------------------------
# DB integration: record_review
# ---------------------------------------------------------------------------


class TestRecordReview:
    def test_first_review_sets_initial_values(self, fsrs_db):
        """First review of a lesson should set initial S from INITIAL_S dict."""
        lesson_id = insert_lesson(
            fsrs_db,
            {
                "title": "new lesson",
                "created_date": date.today().isoformat(),
            },
        )
        ensure_fsrs_columns(fsrs_db)

        result = record_review(fsrs_db, lesson_id, GRADE_GOOD)
        assert result["stability"] == pytest.approx(INITIAL_S[GRADE_GOOD], abs=1e-6)
        assert result["difficulty"] is not None
        assert 1.0 <= result["difficulty"] <= 10.0
        assert result["last_review_date"] == date.today().isoformat()

    def test_subsequent_review_updates_values(self, fsrs_db):
        """Second review should use update equations, not initial values."""
        lid = _insert_reviewed_lesson(fsrs_db, "existing", stability=2.0, difficulty=5.0, days_ago=5)
        result = record_review(fsrs_db, lid, GRADE_GOOD)

        # Stability should have changed from 2.0
        assert result["stability"] != 2.0
        assert result["stability"] > 0

    def test_review_persists_to_db(self, fsrs_db):
        """record_review should write values back to the DB."""
        lid = _insert_reviewed_lesson(fsrs_db, "persist", stability=2.0, difficulty=5.0, days_ago=3)
        result = record_review(fsrs_db, lid, GRADE_HARD)

        # Read back from DB
        row = fsrs_db.execute(
            "SELECT stability, difficulty, last_review_date FROM lessons WHERE id = ?", [lid]
        ).fetchone()
        assert row["stability"] == pytest.approx(result["stability"], abs=1e-9)
        assert row["difficulty"] == pytest.approx(result["difficulty"], abs=1e-9)
        assert row["last_review_date"] == date.today().isoformat()

    def test_again_review_decreases_stability(self, fsrs_db):
        """AGAIN grade should lower stability."""
        lid = _insert_reviewed_lesson(fsrs_db, "failed", stability=10.0, difficulty=5.0, days_ago=5)
        result = record_review(fsrs_db, lid, GRADE_AGAIN)
        assert result["stability"] < 10.0

    def test_nonexistent_lesson_raises(self, fsrs_db):
        """Reviewing a non-existent lesson should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            record_review(fsrs_db, 99999, GRADE_GOOD)

    def test_invalid_grade_raises(self, fsrs_db):
        """Invalid grade should raise ValueError."""
        lid = insert_lesson(fsrs_db, {"title": "bad grade", "created_date": date.today().isoformat()})
        with pytest.raises(ValueError, match="[Gg]rade"):
            record_review(fsrs_db, lid, 0)
        with pytest.raises(ValueError, match="[Gg]rade"):
            record_review(fsrs_db, lid, 5)


# ---------------------------------------------------------------------------
# ensure_fsrs_columns idempotency
# ---------------------------------------------------------------------------


class TestEnsureFsrsColumns:
    def test_idempotent(self, tmp_path):
        """Calling ensure_fsrs_columns multiple times should not error."""
        conn = init_db(tmp_path / "test.db")
        ensure_fsrs_columns(conn)
        ensure_fsrs_columns(conn)  # second call should be fine

    def test_columns_exist_after_call(self, tmp_path):
        """After ensure_fsrs_columns, lessons table should have S, D, last_review_date."""
        conn = init_db(tmp_path / "test.db")
        ensure_fsrs_columns(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)").fetchall()}
        assert "stability" in cols
        assert "difficulty" in cols
        assert "last_review_date" in cols


# ---------------------------------------------------------------------------
# OUTCOME_TO_GRADE mapping
# ---------------------------------------------------------------------------


class TestOutcomeToGrade:
    def test_heeded_maps_to_good(self):
        """heeded outcome -> GRADE_GOOD (lesson applied correctly)."""
        assert OUTCOME_TO_GRADE["heeded"] == GRADE_GOOD

    def test_dismissed_maps_to_again(self):
        """dismissed outcome -> GRADE_AGAIN (lesson ignored)."""
        assert OUTCOME_TO_GRADE["dismissed"] == GRADE_AGAIN

    def test_false_positive_maps_to_easy(self):
        """false_positive outcome -> GRADE_EASY (surfaced incorrectly)."""
        assert OUTCOME_TO_GRADE["false_positive"] == GRADE_EASY

    def test_all_mapped_grades_are_valid(self):
        """Every mapped grade should be in the valid grade set."""
        valid = {GRADE_AGAIN, GRADE_HARD, GRADE_GOOD, GRADE_EASY}
        for outcome, grade in OUTCOME_TO_GRADE.items():
            assert grade in valid, f"Outcome '{outcome}' maps to invalid grade {grade}"

    def test_recurrence_not_mapped(self):
        """recurrence is not in OUTCOME_TO_GRADE — it's tracked separately."""
        assert "recurrence" not in OUTCOME_TO_GRADE


# ---------------------------------------------------------------------------
# backfill_fsrs_defaults
# ---------------------------------------------------------------------------


class TestBackfillFsrsDefaults:
    def test_backfills_null_stability(self, fsrs_db):
        """Lessons with stability IS NULL should get defaults."""
        lid = insert_lesson(fsrs_db, {"title": "no fsrs", "created_date": date.today().isoformat()})
        # Force NULL stability (override the DEFAULT 1.0 from schema)
        fsrs_db.execute("UPDATE lessons SET stability = NULL WHERE id = ?", [lid])
        fsrs_db.commit()

        count = backfill_fsrs_defaults(fsrs_db)
        assert count >= 1

        row = fsrs_db.execute(
            "SELECT stability, difficulty, retrievability FROM lessons WHERE id = ?", [lid]
        ).fetchone()
        assert row["stability"] == 1.0
        assert row["difficulty"] == 5.0
        assert row["retrievability"] == 1.0

    def test_does_not_overwrite_existing(self, fsrs_db):
        """Lessons with non-NULL stability should be left alone."""
        lid = _insert_reviewed_lesson(fsrs_db, "has values", stability=7.5, difficulty=3.0, days_ago=1)
        backfill_fsrs_defaults(fsrs_db)

        row = fsrs_db.execute("SELECT stability, difficulty FROM lessons WHERE id = ?", [lid]).fetchone()
        assert row["stability"] == 7.5
        assert row["difficulty"] == 3.0

    def test_returns_zero_when_nothing_to_backfill(self, fsrs_db):
        """If all lessons have stability set, count should be 0."""
        _insert_reviewed_lesson(fsrs_db, "ok", stability=2.0, difficulty=5.0, days_ago=1)
        count = backfill_fsrs_defaults(fsrs_db)
        assert count == 0


# ---------------------------------------------------------------------------
# Additional record_review tests
# ---------------------------------------------------------------------------


class TestRecordReviewExtended:
    def test_review_updates_retrievability_in_db(self, fsrs_db):
        """record_review should set retrievability=1.0 in the DB (just reviewed)."""
        lid = _insert_reviewed_lesson(fsrs_db, "r-check", stability=2.0, difficulty=5.0, days_ago=10)
        # Retrievability in DB should be stale (not 1.0) before review
        record_review(fsrs_db, lid, GRADE_GOOD)

        row = fsrs_db.execute("SELECT retrievability FROM lessons WHERE id = ?", [lid]).fetchone()
        assert row["retrievability"] == 1.0

    def test_first_review_each_grade(self, fsrs_db):
        """First review with each grade should produce valid initial values."""
        for grade in [GRADE_AGAIN, GRADE_HARD, GRADE_GOOD, GRADE_EASY]:
            lid = insert_lesson(
                fsrs_db,
                {"title": f"grade-{grade}", "created_date": date.today().isoformat()},
            )
            result = record_review(fsrs_db, lid, grade)
            assert result["stability"] == pytest.approx(INITIAL_S[grade], abs=1e-6)
            assert 1.0 <= result["difficulty"] <= 10.0
            assert result["retrievability"] == 1.0


# ---------------------------------------------------------------------------
# Additional get_due_lessons tests
# ---------------------------------------------------------------------------


class TestGetDueLessonsExtended:
    def test_sorted_most_forgotten_first(self, fsrs_db):
        """Results should be sorted by R ascending (most forgotten first)."""
        # Very forgotten (S=1, 30 days ago)
        _insert_reviewed_lesson(fsrs_db, "very forgotten", stability=1.0, difficulty=5.0, days_ago=30)
        # Somewhat forgotten (S=1, 5 days ago)
        _insert_reviewed_lesson(fsrs_db, "somewhat forgotten", stability=1.0, difficulty=5.0, days_ago=5)

        due = get_due_lessons(fsrs_db, threshold=0.9)
        assert len(due) == 2
        assert due[0]["title"] == "very forgotten"
        assert due[1]["title"] == "somewhat forgotten"
        assert due[0]["retrievability"] < due[1]["retrievability"]

    def test_includes_days_since_review(self, fsrs_db):
        """Each result should include days_since_review."""
        _insert_reviewed_lesson(fsrs_db, "with days", stability=1.0, difficulty=5.0, days_ago=10)
        due = get_due_lessons(fsrs_db, threshold=0.9)
        assert len(due) == 1
        assert due[0]["days_since_review"] == 10

    def test_custom_threshold(self, fsrs_db):
        """A lower threshold should return fewer lessons."""
        # S=1, 2 days ago -> R ≈ 0.82 (below 0.9 but above 0.5)
        _insert_reviewed_lesson(fsrs_db, "medium", stability=1.0, difficulty=5.0, days_ago=2)

        due_high = get_due_lessons(fsrs_db, threshold=0.9)
        due_low = get_due_lessons(fsrs_db, threshold=0.5)
        assert len(due_high) == 1
        assert len(due_low) == 0


# ---------------------------------------------------------------------------
# Additional compute_retrievability edge cases
# ---------------------------------------------------------------------------


class TestComputeRetrievabilityEdgeCases:
    def test_negative_days_returns_one(self):
        """Negative days_since_review should return 1.0 (future review date = just reviewed)."""
        r = compute_retrievability(stability=1.0, days_since_review=-5.0)
        assert r == 1.0

    def test_very_small_stability(self):
        """Very small stability should produce rapid decay but never negative R."""
        r = compute_retrievability(stability=0.01, days_since_review=1.0)
        assert 0.0 < r < 1.0


# ---------------------------------------------------------------------------
# Additional update_stability edge cases
# ---------------------------------------------------------------------------


class TestUpdateStabilityExtended:
    def test_hard_grade_increases_less_than_good(self):
        """Hard grade should increase stability less than Good."""
        old_s = 5.0
        s_hard = update_stability(old_S=old_s, old_D=5.0, grade=GRADE_HARD, R=0.9)
        s_good = update_stability(old_S=old_s, old_D=5.0, grade=GRADE_GOOD, R=0.9)
        assert s_hard > old_s  # Hard still increases
        assert s_hard < s_good  # but less than Good

    def test_lapse_capped_at_old_stability(self):
        """After lapse, new stability should not exceed old stability."""
        old_s = 3.0
        new_s = update_stability(old_S=old_s, old_D=5.0, grade=GRADE_AGAIN, R=0.5)
        assert new_s <= old_s


# ---------------------------------------------------------------------------
# get_fading_level — adaptive presentation based on stability
# ---------------------------------------------------------------------------


class TestGetFadingLevel:
    """Fading levels: full -> brief -> silent -> enforced as S increases."""

    def test_very_low_stability_is_full(self):
        """S < 2.0 -> 'full' (show full lesson text + code example)."""
        assert get_fading_level(0.5) == "full"
        assert get_fading_level(1.0) == "full"
        assert get_fading_level(1.99) == "full"

    def test_moderate_stability_is_brief(self):
        """2.0 <= S < 10.0 -> 'brief' (one-liner reminder only)."""
        assert get_fading_level(2.0) == "brief"
        assert get_fading_level(5.0) == "brief"
        assert get_fading_level(9.99) == "brief"

    def test_high_stability_is_silent(self):
        """10.0 <= S < 50.0 -> 'silent' (Semgrep rule only, no message)."""
        assert get_fading_level(10.0) == "silent"
        assert get_fading_level(20.0) == "silent"
        assert get_fading_level(49.99) == "silent"

    def test_very_high_stability_is_enforced(self):
        """S >= 50.0 -> 'enforced' (automated enforcement, never shown)."""
        assert get_fading_level(50.0) == "enforced"
        assert get_fading_level(100.0) == "enforced"
        assert get_fading_level(1000.0) == "enforced"

    def test_boundary_at_two(self):
        """Exact boundary: 1.999... -> full, 2.0 -> brief."""
        assert get_fading_level(1.9999) == "full"
        assert get_fading_level(2.0) == "brief"

    def test_boundary_at_ten(self):
        """Exact boundary: 9.999... -> brief, 10.0 -> silent."""
        assert get_fading_level(9.9999) == "brief"
        assert get_fading_level(10.0) == "silent"

    def test_boundary_at_fifty(self):
        """Exact boundary: 49.999... -> silent, 50.0 -> enforced."""
        assert get_fading_level(49.9999) == "silent"
        assert get_fading_level(50.0) == "enforced"

    def test_near_zero_stability(self):
        """Very small stability (near-new lesson) -> full."""
        assert get_fading_level(0.01) == "full"

    def test_all_levels_are_valid_strings(self):
        """Every fading level returned must be one of the four valid levels."""
        valid = {"full", "brief", "silent", "enforced"}
        for s in [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]:
            level = get_fading_level(s)
            assert level in valid, f"S={s} returned invalid level '{level}'"


# ---------------------------------------------------------------------------
# interleave_due_lessons — Bjork desirable difficulties interleaving
# ---------------------------------------------------------------------------


class TestInterleaveDueLessons:
    def test_empty_list(self):
        """Empty input returns empty output."""
        assert interleave_due_lessons([]) == []

    def test_single_lesson(self):
        """Single lesson returns unchanged."""
        lessons = [{"id": 1, "cluster": "A"}]
        result = interleave_due_lessons(lessons)
        assert len(result) == 1
        assert result[0]["id"] == 1

    def test_same_cluster_preserved_order(self):
        """All lessons from one cluster keep their original order."""
        lessons = [
            {"id": 1, "cluster": "A"},
            {"id": 2, "cluster": "A"},
            {"id": 3, "cluster": "A"},
        ]
        result = interleave_due_lessons(lessons)
        assert len(result) == 3
        assert [r["id"] for r in result] == [1, 2, 3]

    def test_two_clusters_interleaved(self):
        """Two clusters with equal count should alternate."""
        lessons = [
            {"id": 1, "cluster": "A"},
            {"id": 2, "cluster": "A"},
            {"id": 3, "cluster": "B"},
            {"id": 4, "cluster": "B"},
        ]
        result = interleave_due_lessons(lessons)
        assert len(result) == 4
        # No three consecutive from the same cluster
        for i in range(len(result) - 2):
            clusters = {result[i]["cluster"], result[i + 1]["cluster"], result[i + 2]["cluster"]}
            assert len(clusters) > 1, f"Three consecutive same-cluster at index {i}"

    def test_avoids_three_consecutive_same_cluster(self):
        """With multiple clusters, never 3+ consecutive from the same cluster."""
        lessons = [{"id": i, "cluster": c} for i, c in enumerate(["A", "A", "A", "A", "B", "B", "B", "C", "C"])]
        result = interleave_due_lessons(lessons)
        assert len(result) == 9
        for i in range(len(result) - 2):
            c0 = result[i]["cluster"]
            c1 = result[i + 1]["cluster"]
            c2 = result[i + 2]["cluster"]
            assert not (c0 == c1 == c2), f"Three consecutive '{c0}' at index {i}"

    def test_none_cluster_treated_as_group(self):
        """Lessons with None cluster are grouped together."""
        lessons = [
            {"id": 1, "cluster": None},
            {"id": 2, "cluster": "A"},
            {"id": 3, "cluster": None},
            {"id": 4, "cluster": "A"},
        ]
        result = interleave_due_lessons(lessons)
        assert len(result) == 4
        # All lessons present
        assert {r["id"] for r in result} == {1, 2, 3, 4}

    def test_empty_string_cluster_treated_as_none(self):
        """Empty string cluster treated same as None."""
        lessons = [
            {"id": 1, "cluster": ""},
            {"id": 2, "cluster": "A"},
        ]
        result = interleave_due_lessons(lessons)
        assert len(result) == 2

    def test_preserves_all_fields(self):
        """Interleaving should not lose any dict fields."""
        lessons = [
            {"id": 1, "cluster": "A", "title": "first", "stability": 1.0},
            {"id": 2, "cluster": "B", "title": "second", "stability": 2.0},
        ]
        result = interleave_due_lessons(lessons)
        for r in result:
            assert "id" in r
            assert "title" in r
            assert "stability" in r

    def test_sorted_by_retrievability_within_interleaving(self):
        """Due lessons sorted by retrievability ascending should remain that way per-cluster."""
        lessons = [
            {"id": 1, "cluster": "A", "retrievability": 0.3},
            {"id": 2, "cluster": "A", "retrievability": 0.5},
            {"id": 3, "cluster": "B", "retrievability": 0.2},
            {"id": 4, "cluster": "B", "retrievability": 0.7},
        ]
        result = interleave_due_lessons(lessons)
        # Within each cluster, order should be preserved
        a_items = [r for r in result if r["cluster"] == "A"]
        b_items = [r for r in result if r["cluster"] == "B"]
        assert a_items[0]["retrievability"] <= a_items[1]["retrievability"]
        assert b_items[0]["retrievability"] <= b_items[1]["retrievability"]


# ---------------------------------------------------------------------------
# CLI: fsrs due
# ---------------------------------------------------------------------------


class TestFsrsDueCli:
    def test_help(self, tmp_path):
        """fsrs due --help should succeed."""
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "fsrs", "due", "--help"])
        assert result.exit_code == 0
        assert "threshold" in result.output.lower()

    def test_no_lessons_due(self, tmp_path):
        """With no reviewed lessons, output should say none due."""
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "fsrs", "due"])
        assert result.exit_code == 0
        assert "No lessons due" in result.output

    def test_shows_due_lessons(self, tmp_path):
        """Lessons with low retrievability should appear in output."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        ensure_fsrs_columns(conn)
        # Insert a lesson that's overdue: S=1.0, reviewed 20 days ago
        lid = insert_lesson(conn, {"title": "overdue lesson", "created_date": date.today().isoformat()})
        review_date = (date.today() - timedelta(days=20)).isoformat()
        conn.execute(
            "UPDATE lessons SET stability = 1.0, difficulty = 5.0, last_review_date = ? WHERE id = ?",
            [review_date, lid],
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "fsrs", "due"])
        assert result.exit_code == 0
        assert "overdue lesson" in result.output
        assert "S=" in result.output
        assert "R=" in result.output

    def test_threshold_flag(self, tmp_path):
        """--threshold should control which lessons appear."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        ensure_fsrs_columns(conn)
        # S=1.0, 2 days ago -> R ≈ 0.82 (below 0.9 but above 0.5)
        lid = insert_lesson(conn, {"title": "medium lesson", "created_date": date.today().isoformat()})
        review_date = (date.today() - timedelta(days=2)).isoformat()
        conn.execute(
            "UPDATE lessons SET stability = 1.0, difficulty = 5.0, last_review_date = ? WHERE id = ?",
            [review_date, lid],
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        # With default threshold (0.9), lesson should appear
        result_high = runner.invoke(main, ["--db", str(db_path), "fsrs", "due"])
        assert "medium lesson" in result_high.output

        # With low threshold (0.5), lesson should NOT appear
        result_low = runner.invoke(main, ["--db", str(db_path), "fsrs", "due", "--threshold", "0.5"])
        assert "medium lesson" not in result_low.output


# ---------------------------------------------------------------------------
# CLI: fsrs stats
# ---------------------------------------------------------------------------


class TestFsrsStatsCli:
    def test_help(self, tmp_path):
        """fsrs stats --help should succeed."""
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "fsrs", "stats", "--help"])
        assert result.exit_code == 0

    def test_empty_db(self, tmp_path):
        """Stats on an empty DB should show all zeros."""
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "fsrs", "stats"])
        assert result.exit_code == 0
        assert "Stability distribution" in result.output
        assert "Review forecast" in result.output

    def test_stability_distribution(self, tmp_path):
        """Stats should show correct fading level counts."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        ensure_fsrs_columns(conn)
        # 2 full (S < 2), 1 brief (2 <= S < 10), 1 silent (10 <= S < 50)
        for title, stability in [("a", 0.5), ("b", 1.5), ("c", 5.0), ("d", 25.0)]:
            lid = insert_lesson(conn, {"title": title, "created_date": date.today().isoformat()})
            conn.execute("UPDATE lessons SET stability = ? WHERE id = ?", [stability, lid])
        conn.commit()
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "fsrs", "stats"])
        assert result.exit_code == 0
        assert "full" in result.output
        assert "brief" in result.output
        assert "silent" in result.output

    def test_review_forecast(self, tmp_path):
        """Stats should show forecast for 1, 3, 7, 14, 30 days."""
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        ensure_fsrs_columns(conn)
        # Insert a reviewed lesson
        lid = insert_lesson(conn, {"title": "reviewed", "created_date": date.today().isoformat()})
        conn.execute(
            "UPDATE lessons SET stability = 1.0, difficulty = 5.0, last_review_date = ? WHERE id = ?",
            [date.today().isoformat(), lid],
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "fsrs", "stats"])
        assert result.exit_code == 0
        assert "in  1 day(s)" in result.output
        assert "in  3 day(s)" in result.output
        assert "in  7 day(s)" in result.output
        assert "in 14 day(s)" in result.output
        assert "in 30 day(s)" in result.output


# ---------------------------------------------------------------------------
# Due lessons sorted by retrievability ascending
# ---------------------------------------------------------------------------


class TestDueLessonsSortedByRetrievability:
    def test_get_due_lessons_sorted_ascending(self, fsrs_db):
        """get_due_lessons should return results sorted by retrievability ascending."""
        # Create lessons with varying staleness
        for title, stability, days_ago in [
            ("least forgotten", 1.0, 3),
            ("most forgotten", 1.0, 30),
            ("middle forgotten", 1.0, 10),
        ]:
            _insert_reviewed_lesson(fsrs_db, title, stability=stability, difficulty=5.0, days_ago=days_ago)

        due = get_due_lessons(fsrs_db, threshold=0.9)
        # Verify ascending R order
        for i in range(len(due) - 1):
            assert (
                due[i]["retrievability"] <= due[i + 1]["retrievability"]
            ), f"Not sorted: R[{i}]={due[i]['retrievability']} > R[{i + 1}]={due[i + 1]['retrievability']}"


# ---------------------------------------------------------------------------
# Polarity-differentiated FSRS constants
# ---------------------------------------------------------------------------


class TestPolarityConstants:
    def test_positive_initial_stability_greater_than_negative(self):
        """Positive lessons should have higher initial stability than negative."""
        assert POSITIVE_INITIAL_STABILITY > NEGATIVE_INITIAL_STABILITY

    def test_negative_initial_stability_matches_default(self):
        """NEGATIVE_INITIAL_STABILITY should equal 1.0 (DEFAULT_STABILITY alias)."""
        from lessons_db.fsrs import DEFAULT_STABILITY

        assert NEGATIVE_INITIAL_STABILITY == DEFAULT_STABILITY

    def test_positive_initial_difficulty_lower_than_default(self):
        """Positive lessons start easier — lower initial difficulty."""
        from lessons_db.fsrs import DEFAULT_DIFFICULTY

        assert POSITIVE_INITIAL_DIFFICULTY < DEFAULT_DIFFICULTY

    def test_positive_outcome_to_grade_values(self):
        """Positive outcome mapping should use correct grades."""
        assert POSITIVE_OUTCOME_TO_GRADE["reused"] == GRADE_GOOD
        assert POSITIVE_OUTCOME_TO_GRADE["not_reused"] == GRADE_HARD
        assert POSITIVE_OUTCOME_TO_GRADE["never_reused"] == GRADE_AGAIN

    def test_positive_outcome_all_grades_valid(self):
        """Every positive outcome grade should be in the valid grade set."""
        valid = {GRADE_AGAIN, GRADE_HARD, GRADE_GOOD, GRADE_EASY}
        for outcome, grade in POSITIVE_OUTCOME_TO_GRADE.items():
            assert grade in valid, f"Positive outcome '{outcome}' maps to invalid grade {grade}"


# ---------------------------------------------------------------------------
# record_review with polarity parameter
# ---------------------------------------------------------------------------


class TestRecordReviewPolarity:
    def test_positive_first_review_uses_higher_stability(self, fsrs_db):
        """First review with polarity='positive' should use POSITIVE_INITIAL_STABILITY."""
        lid = insert_lesson(
            fsrs_db,
            {
                "title": "good pattern",
                "created_date": date.today().isoformat(),
                "polarity": "positive",
            },
        )
        result = record_review(fsrs_db, lid, GRADE_GOOD, polarity="positive")
        assert result["stability"] == pytest.approx(POSITIVE_INITIAL_STABILITY, abs=1e-6)

    def test_negative_first_review_uses_grade_stability(self, fsrs_db):
        """First review with polarity='negative' should use INITIAL_S[grade]."""
        lid = insert_lesson(
            fsrs_db,
            {"title": "anti-pattern", "created_date": date.today().isoformat()},
        )
        result = record_review(fsrs_db, lid, GRADE_GOOD, polarity="negative")
        assert result["stability"] == pytest.approx(INITIAL_S[GRADE_GOOD], abs=1e-6)

    def test_default_polarity_is_negative(self, fsrs_db):
        """Default polarity should be 'negative' (backward compatible)."""
        lid = insert_lesson(
            fsrs_db,
            {"title": "default polarity", "created_date": date.today().isoformat()},
        )
        result = record_review(fsrs_db, lid, GRADE_GOOD)
        assert result["stability"] == pytest.approx(INITIAL_S[GRADE_GOOD], abs=1e-6)

    def test_positive_first_review_lower_difficulty(self, fsrs_db):
        """Positive lessons should start with lower initial difficulty."""
        lid = insert_lesson(
            fsrs_db,
            {
                "title": "easy positive",
                "created_date": date.today().isoformat(),
                "polarity": "positive",
            },
        )
        result = record_review(fsrs_db, lid, GRADE_GOOD, polarity="positive")
        assert result["difficulty"] == pytest.approx(POSITIVE_INITIAL_DIFFICULTY, abs=1e-6)

    def test_positive_stability_higher_than_negative_again_grade(self, fsrs_db):
        """For AGAIN grade, positive first review yields much higher S than negative.

        Positive lessons always get POSITIVE_INITIAL_STABILITY (3.0) regardless
        of grade. Negative lessons use grade-specific INITIAL_S (0.40 for AGAIN).
        The biggest difference is at the AGAIN grade — exactly the case where
        positive framing matters most (the pattern wasn't reused this time, but
        the base knowledge is still there).
        """
        lid_pos = insert_lesson(
            fsrs_db,
            {
                "title": "positive",
                "created_date": date.today().isoformat(),
                "polarity": "positive",
            },
        )
        lid_neg = insert_lesson(
            fsrs_db,
            {"title": "negative", "created_date": date.today().isoformat()},
        )
        r_pos = record_review(fsrs_db, lid_pos, GRADE_AGAIN, polarity="positive")
        r_neg = record_review(fsrs_db, lid_neg, GRADE_AGAIN, polarity="negative")
        assert r_pos["stability"] > r_neg["stability"]
        # Positive: 3.0 vs Negative AGAIN: ~0.40
        assert r_pos["stability"] == pytest.approx(POSITIVE_INITIAL_STABILITY, abs=1e-6)

    def test_subsequent_review_ignores_polarity(self, fsrs_db):
        """After first review, polarity should not affect update equations."""
        lid = _insert_reviewed_lesson(
            fsrs_db, "existing positive", stability=3.0, difficulty=3.0, days_ago=5, polarity="positive"
        )
        result = record_review(fsrs_db, lid, GRADE_GOOD, polarity="positive")
        # Should use standard update equations, not initial values
        assert result["stability"] != POSITIVE_INITIAL_STABILITY


# ---------------------------------------------------------------------------
# get_due_lessons with polarity filter
# ---------------------------------------------------------------------------


class TestGetDueLessonsPolarity:
    def test_filter_positive_only(self, fsrs_db):
        """polarity='positive' should only return positive lessons."""
        _insert_reviewed_lesson(fsrs_db, "neg lesson", stability=1.0, difficulty=5.0, days_ago=20, polarity="negative")
        _insert_reviewed_lesson(fsrs_db, "pos lesson", stability=1.0, difficulty=5.0, days_ago=20, polarity="positive")

        due = get_due_lessons(fsrs_db, threshold=0.9, polarity="positive")
        titles = [d["title"] for d in due]
        assert "pos lesson" in titles
        assert "neg lesson" not in titles

    def test_filter_negative_only(self, fsrs_db):
        """polarity='negative' should only return negative lessons."""
        _insert_reviewed_lesson(fsrs_db, "neg lesson", stability=1.0, difficulty=5.0, days_ago=20, polarity="negative")
        _insert_reviewed_lesson(fsrs_db, "pos lesson", stability=1.0, difficulty=5.0, days_ago=20, polarity="positive")

        due = get_due_lessons(fsrs_db, threshold=0.9, polarity="negative")
        titles = [d["title"] for d in due]
        assert "neg lesson" in titles
        assert "pos lesson" not in titles

    def test_no_filter_returns_both(self, fsrs_db):
        """polarity=None should return both positive and negative lessons."""
        _insert_reviewed_lesson(fsrs_db, "neg lesson", stability=1.0, difficulty=5.0, days_ago=20, polarity="negative")
        _insert_reviewed_lesson(fsrs_db, "pos lesson", stability=1.0, difficulty=5.0, days_ago=20, polarity="positive")

        due = get_due_lessons(fsrs_db, threshold=0.9, polarity=None)
        titles = [d["title"] for d in due]
        assert "neg lesson" in titles
        assert "pos lesson" in titles

    def test_default_polarity_is_none(self, fsrs_db):
        """Default polarity filter should be None (return all)."""
        _insert_reviewed_lesson(fsrs_db, "neg lesson", stability=1.0, difficulty=5.0, days_ago=20, polarity="negative")
        _insert_reviewed_lesson(fsrs_db, "pos lesson", stability=1.0, difficulty=5.0, days_ago=20, polarity="positive")

        due = get_due_lessons(fsrs_db, threshold=0.9)
        assert len(due) == 2


# ---------------------------------------------------------------------------
# enforce_positive_ratio — dual-polarity interleaving
# ---------------------------------------------------------------------------


class TestEnforcePositiveRatio:
    def test_empty_list(self):
        """Empty input returns empty output."""
        assert enforce_positive_ratio([]) == []

    def test_all_negative(self):
        """All negative lessons returned unchanged."""
        lessons = [{"id": i, "polarity": "negative"} for i in range(5)]
        result = enforce_positive_ratio(lessons)
        assert len(result) == 5
        assert all(r["polarity"] == "negative" for r in result)

    def test_all_positive(self):
        """All positive lessons returned unchanged."""
        lessons = [{"id": i, "polarity": "positive"} for i in range(5)]
        result = enforce_positive_ratio(lessons)
        assert len(result) == 5
        assert all(r["polarity"] == "positive" for r in result)

    def test_interleaves_one_per_three(self):
        """With default ratio 0.25, should insert 1 positive after every 3 negatives."""
        negatives = [{"id": i, "polarity": "negative"} for i in range(9)]
        positives = [{"id": 100 + i, "polarity": "positive"} for i in range(3)]
        lessons = negatives + positives

        result = enforce_positive_ratio(lessons)
        assert len(result) == 12
        # Check that positives appear at indices 3, 7, 11 (after every 3 negatives)
        assert result[3]["polarity"] == "positive"
        assert result[7]["polarity"] == "positive"
        assert result[11]["polarity"] == "positive"

    def test_no_more_than_three_consecutive_negatives(self):
        """With default ratio 0.25, no more than 3 consecutive negatives should appear."""
        negatives = [{"id": i, "polarity": "negative"} for i in range(12)]
        positives = [{"id": 100 + i, "polarity": "positive"} for i in range(4)]
        lessons = negatives + positives

        result = enforce_positive_ratio(lessons)
        consecutive_neg = 0
        max_consecutive = 0
        for r in result:
            if r["polarity"] == "negative":
                consecutive_neg += 1
                max_consecutive = max(max_consecutive, consecutive_neg)
            else:
                consecutive_neg = 0
        assert max_consecutive <= 3, f"Found {max_consecutive} consecutive negatives"

    def test_preserves_all_lessons(self):
        """All input lessons should appear in the output."""
        negatives = [{"id": i, "polarity": "negative"} for i in range(6)]
        positives = [{"id": 100 + i, "polarity": "positive"} for i in range(2)]
        lessons = negatives + positives

        result = enforce_positive_ratio(lessons)
        result_ids = {r["id"] for r in result}
        input_ids = {l["id"] for l in lessons}
        assert result_ids == input_ids

    def test_not_enough_positives_distributes_evenly(self):
        """When there are fewer positives than needed, distribute what's available."""
        negatives = [{"id": i, "polarity": "negative"} for i in range(9)]
        positives = [{"id": 100, "polarity": "positive"}]
        lessons = negatives + positives

        result = enforce_positive_ratio(lessons)
        assert len(result) == 10
        # The single positive should appear after the first 3 negatives
        assert result[3]["polarity"] == "positive"

    def test_custom_ratio(self):
        """Custom ratio should change interleaving gap."""
        negatives = [{"id": i, "polarity": "negative"} for i in range(4)]
        positives = [{"id": 100 + i, "polarity": "positive"} for i in range(4)]
        lessons = negatives + positives

        # min_ratio=0.5 -> 1 positive per 1 negative
        result = enforce_positive_ratio(lessons, min_ratio=0.5)
        assert len(result) == 8
        # Should alternate: neg, pos, neg, pos, ...
        assert result[0]["polarity"] == "negative"
        assert result[1]["polarity"] == "positive"

    def test_excess_positives_appended(self):
        """Extra positives beyond what's needed should be appended at the end."""
        negatives = [{"id": i, "polarity": "negative"} for i in range(3)]
        positives = [{"id": 100 + i, "polarity": "positive"} for i in range(5)]
        lessons = negatives + positives

        result = enforce_positive_ratio(lessons)
        assert len(result) == 8
        # All lessons present
        assert {r["id"] for r in result} == {0, 1, 2, 100, 101, 102, 103, 104}

    def test_missing_polarity_treated_as_negative(self):
        """Lessons without polarity key should be treated as negative."""
        lessons = [
            {"id": 1},  # no polarity key
            {"id": 2, "polarity": "negative"},
            {"id": 3, "polarity": "negative"},
            {"id": 4, "polarity": "positive"},
        ]
        result = enforce_positive_ratio(lessons)
        assert len(result) == 4
        # All present
        assert {r["id"] for r in result} == {1, 2, 3, 4}
