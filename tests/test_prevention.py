"""Tests for the prevention pipeline — enforce, generate, track, report."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from lessons_db.db import init_db, insert_lesson
from lessons_db.prevention import (
    VELOCITY_THRESHOLD,
    EnforcementDecision,
    assess_and_enforce,
    bulk_generate_rules,
    check_content,
    generate_rule_for_lesson,
    prevention_report,
    resolve_outcomes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_lesson(conn, enforcement="documentation", recurrence_count=0, severity=3):
    return insert_lesson(
        conn,
        {
            "title": "Test lesson for prevention",
            "one_liner": "Do not use bare except",
            "enforcement": enforcement,
            "recurrence_count": recurrence_count,
            "severity": severity,
        },
    )


def _insert_pattern(
    conn,
    lesson_id,
    regex=r"except:",
    language="python",
    pattern_type="syntactic",
):
    conn.execute(
        "INSERT INTO detection_patterns (lesson_id, pattern_type, regex, language) VALUES (?, ?, ?, ?)",
        (lesson_id, pattern_type, regex, language),
    )
    conn.commit()


def _insert_recurrence(conn, lesson_id, hours_ago=0.5):
    """Insert a pre-existing recurrence event, bypassing assess_and_enforce."""
    ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    conn.execute(
        "INSERT INTO recurrence_events (lesson_id, timestamp, hook_point, trigger_type) " "VALUES (?, ?, ?, ?)",
        (lesson_id, ts, "edit", "test"),
    )
    conn.commit()


def _insert_surfacing(conn, lesson_id, hours_ago=2.0, outcome="unknown"):
    """Insert a surfacing event directly into the DB."""
    ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    conn.execute(
        "INSERT INTO surfacing_events (lesson_id, hook_point, outcome, timestamp) " "VALUES (?, ?, ?, ?)",
        (lesson_id, "pre-edit", outcome, ts),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


# ---------------------------------------------------------------------------
# Recurrence tracking (db layer)
# ---------------------------------------------------------------------------


class TestRecurrenceTracking:
    """insert_recurrence_event + get_recurrence_velocity + get_velocity_warnings."""

    def test_insert_returns_row_id(self, db_path):
        from lessons_db.db import insert_recurrence_event

        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        row_id = insert_recurrence_event(conn, lid, "edit", "regex_match")
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_velocity_counts_events_in_window(self, db_path):
        from lessons_db.db import get_recurrence_velocity, insert_recurrence_event

        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        insert_recurrence_event(conn, lid, "edit", "regex_match")
        insert_recurrence_event(conn, lid, "edit", "regex_match")
        assert get_recurrence_velocity(conn, lid, window_days=7) == 2

    def test_velocity_excludes_old_events(self, db_path):
        from lessons_db.db import get_recurrence_velocity

        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        old_ts = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        conn.execute(
            "INSERT INTO recurrence_events (lesson_id, timestamp, hook_point, trigger_type) " "VALUES (?, ?, ?, ?)",
            (lid, old_ts, "edit", "test"),
        )
        conn.commit()
        assert get_recurrence_velocity(conn, lid, window_days=7) == 0

    def test_velocity_independent_across_lessons(self, db_path):
        from lessons_db.db import get_recurrence_velocity, insert_recurrence_event

        conn = init_db(db_path)
        lid_a = _insert_lesson(conn)
        lid_b = _insert_lesson(conn)
        insert_recurrence_event(conn, lid_a, "edit", "regex_match")
        insert_recurrence_event(conn, lid_a, "edit", "regex_match")
        assert get_recurrence_velocity(conn, lid_b, window_days=7) == 0

    def test_velocity_warnings_returns_high_velocity_lessons(self, db_path):
        from lessons_db.db import get_velocity_warnings, insert_recurrence_event

        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        insert_recurrence_event(conn, lid, "edit", "regex_match")
        insert_recurrence_event(conn, lid, "edit", "regex_match")
        warnings = get_velocity_warnings(conn, window_days=7, threshold=2)
        assert any(w["lesson_id"] == lid for w in warnings)

    def test_velocity_warnings_excludes_low_velocity(self, db_path):
        from lessons_db.db import get_velocity_warnings, insert_recurrence_event

        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        insert_recurrence_event(conn, lid, "edit", "regex_match")  # only 1 hit
        warnings = get_velocity_warnings(conn, window_days=7, threshold=2)
        assert not any(w["lesson_id"] == lid for w in warnings)


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------


class TestGenerateRuleForLesson:
    """generate_rule_for_lesson — write YAML, upsert enforcement_rules."""

    def test_happy_path_writes_file_and_db_row(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_pattern(conn, lid)

        path = generate_rule_for_lesson(conn, lid, rules_dir=tmp_path, validate=False)

        assert path is not None
        assert path.exists()
        assert path.suffix == ".yaml"
        row = conn.execute("SELECT rule_id FROM enforcement_rules WHERE lesson_id = ?", (lid,)).fetchone()
        assert row is not None

    def test_idempotent_second_call_does_not_duplicate_db_row(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_pattern(conn, lid)

        p1 = generate_rule_for_lesson(conn, lid, rules_dir=tmp_path, validate=False)
        p2 = generate_rule_for_lesson(conn, lid, rules_dir=tmp_path, validate=False)

        assert p1 == p2
        count = conn.execute("SELECT COUNT(*) FROM enforcement_rules WHERE lesson_id = ?", (lid,)).fetchone()[0]
        assert count == 1  # upsert, not insert-on-top

    def test_no_patterns_returns_none(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)  # no detection_patterns inserted

        path = generate_rule_for_lesson(conn, lid, rules_dir=tmp_path, validate=False)
        assert path is None

    def test_lesson_not_found_returns_none(self, db_path, tmp_path):
        conn = init_db(db_path)
        path = generate_rule_for_lesson(conn, 9999, rules_dir=tmp_path, validate=False)
        assert path is None

    def test_rule_yaml_placed_in_language_subdir(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_pattern(conn, lid, language="python")

        path = generate_rule_for_lesson(conn, lid, rules_dir=tmp_path, validate=False)

        assert path is not None
        assert path.parent == tmp_path / "python"

    def test_rule_yaml_uses_any_dir_for_unknown_language(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_pattern(conn, lid, language="any")

        path = generate_rule_for_lesson(conn, lid, rules_dir=tmp_path, validate=False)

        assert path is not None
        assert path.parent == tmp_path / "any"


class TestBulkGenerateRules:
    """bulk_generate_rules — all lessons with detection_patterns."""

    def test_generates_rules_for_lessons_with_patterns(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid1 = _insert_lesson(conn)
        lid2 = _insert_lesson(conn)
        _insert_pattern(conn, lid1)
        _insert_pattern(conn, lid2, regex=r"eval\(")
        _insert_lesson(conn)  # lid3 — no patterns; query filters it out, never enters loop

        result = bulk_generate_rules(conn, rules_dir=tmp_path, validate=False)

        assert result["generated"] == 2
        assert result["skipped_no_patterns"] == 0  # lid3 never queried
        assert len(result["paths"]) == 2

    def test_only_enforcement_filter_restricts_generation(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid_warn = _insert_lesson(conn, enforcement="semgrep_warning")
        lid_doc = _insert_lesson(conn, enforcement="documentation")
        _insert_pattern(conn, lid_warn)
        _insert_pattern(conn, lid_doc, regex=r"eval\(")

        result = bulk_generate_rules(
            conn,
            rules_dir=tmp_path,
            only_enforcement=("semgrep_warning",),
            validate=False,
        )

        assert result["generated"] == 1

    def test_empty_db_returns_all_zeros(self, db_path, tmp_path):
        conn = init_db(db_path)
        result = bulk_generate_rules(conn, rules_dir=tmp_path, validate=False)
        assert result["generated"] == 0
        assert result["skipped_no_patterns"] == 0
        assert result["paths"] == []

    def test_result_paths_are_strings(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_pattern(conn, lid)

        result = bulk_generate_rules(conn, rules_dir=tmp_path, validate=False)

        assert all(isinstance(p, str) for p in result["paths"])


# ---------------------------------------------------------------------------
# Enforcement cycle
# ---------------------------------------------------------------------------


class TestAssessAndEnforce:
    """assess_and_enforce — log → velocity → escalate → rule → decision."""

    def test_logs_recurrence_event_to_db(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        assess_and_enforce(conn, lid, "edit", "regex_match")

        count = conn.execute("SELECT COUNT(*) FROM recurrence_events WHERE lesson_id = ?", (lid,)).fetchone()[0]
        assert count == 1

    def test_below_velocity_no_escalation(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, recurrence_count=0)

        # Only 1 event total (the one just logged) — below VELOCITY_THRESHOLD=2
        decision = assess_and_enforce(conn, lid, "edit", "regex_match")

        assert not decision.escalated
        assert decision.enforcement_level == "documentation"

    def test_at_velocity_threshold_triggers_escalation(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, recurrence_count=0)
        # Pre-insert 1 event so the assess call brings total to VELOCITY_THRESHOLD=2
        _insert_recurrence(conn, lid)

        decision = assess_and_enforce(conn, lid, "edit", "regex_match")

        assert decision.escalated
        assert decision.velocity >= VELOCITY_THRESHOLD

    def test_blocking_enforcement_sets_should_block(self, db_path):
        conn = init_db(db_path)
        # Pre-escalated lesson at semgrep_error tier
        lid = _insert_lesson(conn, enforcement="semgrep_error", recurrence_count=3)

        decision = assess_and_enforce(conn, lid, "edit", "regex_match")

        assert decision.should_block is True

    def test_nonblocking_enforcement_does_not_block(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, enforcement="semgrep_warning", recurrence_count=2)

        decision = assess_and_enforce(conn, lid, "edit", "regex_match")

        assert decision.should_block is False

    def test_missing_lesson_returns_safe_non_blocking_decision(self, db_path):
        conn = init_db(db_path)

        decision = assess_and_enforce(conn, 9999, "edit", "regex_match")

        assert not decision.should_block
        assert decision.enforcement_level == "documentation"
        assert not decision.escalated
        assert decision.velocity == 0

    def test_returns_enforcement_decision_dataclass(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        decision = assess_and_enforce(conn, lid, "edit", "regex_match")

        assert isinstance(decision, EnforcementDecision)
        assert decision.lesson_id == lid

    def test_to_dict_serializes_path_as_string_or_none(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        decision = assess_and_enforce(conn, lid, "edit", "regex_match", rules_dir=tmp_path)
        d = decision.to_dict()

        assert "rule_path" in d
        assert d["rule_path"] is None or isinstance(d["rule_path"], str)

    def test_file_path_stored_in_recurrence_event(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        assess_and_enforce(conn, lid, "edit", "regex_match", file_path="src/foo.py")

        row = conn.execute("SELECT file_path FROM recurrence_events WHERE lesson_id = ?", (lid,)).fetchone()
        assert row["file_path"] == "src/foo.py"


# ---------------------------------------------------------------------------
# Content checking
# ---------------------------------------------------------------------------


class TestCheckContent:
    """check_content — match content against patterns, run enforcement."""

    def test_no_matches_returns_no_block_and_empty_violations(self, db_path):
        conn = init_db(db_path)
        with patch("lessons_db.search.search_by_content", return_value=[]):
            result = check_content(conn, "clean code here")

        assert result["block"] is False
        assert result["violations"] == []
        assert result["message"] == ""

    def test_matching_content_returns_violation_entry(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, enforcement="semgrep_warning")

        with patch("lessons_db.search.search_by_content", return_value=[{"id": lid}]):
            result = check_content(conn, "try:\n    pass\nexcept:\n    pass")

        assert len(result["violations"]) == 1
        assert result["violations"][0]["lesson_id"] == lid

    def test_blocking_lesson_sets_block_true_with_message(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, enforcement="semgrep_error", recurrence_count=3)

        with patch("lessons_db.search.search_by_content", return_value=[{"id": lid}]):
            result = check_content(conn, "except:")

        assert result["block"] is True
        assert "BLOCKED" in result["message"]

    def test_non_blocking_lesson_does_not_set_block(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, enforcement="semgrep_warning", recurrence_count=2)

        with patch("lessons_db.search.search_by_content", return_value=[{"id": lid}]):
            result = check_content(conn, "except:")

        assert result["block"] is False
        assert len(result["violations"]) == 1

    def test_violation_contains_decision_dict(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, enforcement="documentation")

        with patch("lessons_db.search.search_by_content", return_value=[{"id": lid}]):
            result = check_content(conn, "some content")

        v = result["violations"][0]
        assert "decision" in v
        assert isinstance(v["decision"], dict)
        assert "should_block" in v["decision"]


# ---------------------------------------------------------------------------
# Outcome resolution
# ---------------------------------------------------------------------------


class TestResolveOutcomes:
    """resolve_outcomes — behavioral inference: recurred=dismissed, else heeded."""

    def test_heeds_lesson_with_no_subsequent_recurrence(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, hours_ago=3.0)

        result = resolve_outcomes(conn, max_age_hours=24)

        assert result["heeded"] == 1
        assert result["dismissed"] == 0

    def test_dismisses_lesson_with_recurrence_after_surfacing(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        # Surfacing 3h ago; recurrence 0.5h ago (after surfacing)
        _insert_surfacing(conn, lid, hours_ago=3.0)
        _insert_recurrence(conn, lid, hours_ago=0.5)

        result = resolve_outcomes(conn, max_age_hours=24)

        assert result["dismissed"] == 1
        assert result["heeded"] == 0

    def test_ignores_events_less_than_one_hour_old(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, hours_ago=0.5)  # too recent to resolve

        result = resolve_outcomes(conn, max_age_hours=24)

        assert result["resolved"] == 0

    def test_ignores_events_older_than_max_age(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, hours_ago=30.0)  # outside the 24h lookback

        result = resolve_outcomes(conn, max_age_hours=24)

        assert result["resolved"] == 0

    def test_already_resolved_events_skipped(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, hours_ago=3.0, outcome="heeded")  # already resolved

        result = resolve_outcomes(conn, max_age_hours=24)

        assert result["resolved"] == 0

    def test_returns_correct_heeded_and_dismissed_counts(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        # Surfacing A: 4h ago — recurrence at 2h ago is AFTER it → dismissed
        _insert_surfacing(conn, lid, hours_ago=4.0)
        _insert_recurrence(conn, lid, hours_ago=2.0)
        # Surfacing B: 1.5h ago — recurrence at 2h ago is BEFORE it → heeded
        _insert_surfacing(conn, lid, hours_ago=1.5)

        result = resolve_outcomes(conn, max_age_hours=24)

        assert result["resolved"] == 2
        assert result["heeded"] == 1
        assert result["dismissed"] == 1

    def test_updates_outcome_field_in_surfacing_events(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        eid = _insert_surfacing(conn, lid, hours_ago=3.0)

        resolve_outcomes(conn, max_age_hours=24)

        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id = ?", (eid,)).fetchone()
        assert row["outcome"] in ("heeded", "dismissed")


# ---------------------------------------------------------------------------
# Prevention report
# ---------------------------------------------------------------------------


class TestPreventionReport:
    """prevention_report — structured effectiveness report."""

    def test_report_has_all_expected_keys(self, db_path):
        conn = init_db(db_path)
        report = prevention_report(conn)

        assert "window_days" in report
        assert "enforcement_coverage" in report
        assert "total_lessons" in report
        assert "rules_generated" in report
        assert "lessons_without_patterns" in report
        assert "surfacing" in report
        assert "top_recurring" in report
        assert "velocity_alerts" in report
        assert "false_positive_hotspots" in report
        assert "hookify_candidates" in report

    def test_enforcement_coverage_counts_lessons(self, db_path):
        conn = init_db(db_path)
        _insert_lesson(conn, enforcement="documentation")
        _insert_lesson(conn, enforcement="documentation")
        _insert_lesson(conn, enforcement="semgrep_warning")

        report = prevention_report(conn)

        assert report["total_lessons"] == 3
        assert report["enforcement_coverage"].get("documentation", 0) == 2
        assert report["enforcement_coverage"].get("semgrep_warning", 0) == 1

    def test_lessons_without_patterns_counted_correctly(self, db_path):
        conn = init_db(db_path)
        lid_with = _insert_lesson(conn)
        _insert_pattern(conn, lid_with)
        _insert_lesson(conn)  # no patterns

        report = prevention_report(conn)

        assert report["lessons_without_patterns"] == 1

    def test_rules_generated_reflects_enforcement_rules(self, db_path, tmp_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_pattern(conn, lid)
        generate_rule_for_lesson(conn, lid, rules_dir=tmp_path, validate=False)

        report = prevention_report(conn)

        assert report["rules_generated"] == 1

    def test_top_recurring_includes_hit_lessons(self, db_path):
        from lessons_db.db import insert_recurrence_event

        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        insert_recurrence_event(conn, lid, "edit", "regex_match")
        insert_recurrence_event(conn, lid, "edit", "regex_match")

        report = prevention_report(conn)

        lesson_ids = [r["lesson_id"] for r in report["top_recurring"]]
        assert lid in lesson_ids

    def test_empty_db_report_has_zero_totals(self, db_path):
        conn = init_db(db_path)
        report = prevention_report(conn)

        assert report["total_lessons"] == 0
        assert report["rules_generated"] == 0
        assert report["top_recurring"] == []
