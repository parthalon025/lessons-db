"""Tests for FSRS spaced repetition module."""

from datetime import date, timedelta

import pytest

from lessons_db.db import init_db, insert_lesson
from lessons_db.fsrs import (
    GRADE_AGAIN,
    GRADE_EASY,
    GRADE_GOOD,
    GRADE_HARD,
    INITIAL_S,
    OUTCOME_TO_GRADE,
    backfill_fsrs_defaults,
    compute_retrievability,
    ensure_fsrs_columns,
    get_due_lessons,
    get_fading_level,
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


def _insert_reviewed_lesson(conn, title, stability, difficulty, days_ago):
    """Helper: insert a lesson with FSRS fields set."""
    lesson_id = insert_lesson(conn, {"title": title, "created_date": date.today().isoformat()})
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
