"""Tests for learning pipeline — outcome tracking and relevance scoring."""

import pytest
from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db, insert_lesson
from lessons_db.learn import (
    record_outcome,
    record_surfacing,
    relevance_score,
    should_surface_positive,
    surfacing_stats,
    update_win_streak,
)


@pytest.fixture
def conn_with_lesson(db_path):
    conn = init_db(db_path)
    lid = insert_lesson(
        conn,
        {
            "title": "Test lesson",
            "one_liner": "Always log before swallowing exceptions",
            "created_date": "2026-02-26",
        },
    )
    return conn, lid


class TestRecordSurfacing:
    def test_creates_surfacing_event(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="read", context="src/hub.py")
        assert event_id is not None
        row = conn.execute("SELECT * FROM surfacing_events WHERE id=?", [event_id]).fetchone()
        assert row["lesson_id"] == lid
        assert row["hook_point"] == "read"
        assert row["outcome"] == "unknown"

    def test_stores_context(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="plan", context="authentication refactor")
        row = conn.execute("SELECT context FROM surfacing_events WHERE id=?", [event_id]).fetchone()
        assert "authentication" in row["context"]


class TestRecordOutcome:
    def test_updates_outcome_to_heeded(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="read", context="hub.py")
        record_outcome(conn, event_id, "heeded")
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id=?", [event_id]).fetchone()
        assert row["outcome"] == "heeded"

    def test_updates_outcome_to_dismissed(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="read", context="hub.py")
        record_outcome(conn, event_id, "dismissed")
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id=?", [event_id]).fetchone()
        assert row["outcome"] == "dismissed"

    def test_rejects_invalid_outcome(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="read", context="hub.py")
        with pytest.raises(ValueError):
            record_outcome(conn, event_id, "ignored")

    def test_raises_on_nonexistent_event_id(self, conn_with_lesson):
        conn, _ = conn_with_lesson
        with pytest.raises(ValueError, match="No surfacing event found"):
            record_outcome(conn, event_id=9999, outcome="heeded")


class TestRelevanceScore:
    def test_cold_start_returns_half_semantic(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        # No history → outcome_rate=0.5, recurrence=0
        score = relevance_score(conn, lid, context="hub.py", semantic_sim=0.8)
        # 0.5*0.8 + 0.3*0.5 + 0.2*0.0 = 0.4 + 0.15 = 0.55
        assert abs(score - 0.55) < 0.01

    def test_heeded_history_boosts_score(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        for _ in range(3):
            eid = record_surfacing(conn, lid, "read", "hub.py")
            record_outcome(conn, eid, "heeded")
        score_with_history = relevance_score(conn, lid, context="hub.py", semantic_sim=0.8)
        score_cold = relevance_score(conn, lid, context="other.py", semantic_sim=0.8)
        assert score_with_history > score_cold

    def test_dismissed_history_lowers_score(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        for _ in range(3):
            eid = record_surfacing(conn, lid, "read", "hub.py")
            record_outcome(conn, eid, "dismissed")
        score = relevance_score(conn, lid, context="hub.py", semantic_sim=0.8)
        # outcome_rate=0.0 → 0.5*0.8 + 0.3*0.0 + 0.2*0 = 0.4
        assert score < 0.55

    def test_recurrence_boosts_score(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        # Add 10 near-misses to push recurrence to max
        for _ in range(10):
            conn.execute(
                "INSERT INTO near_misses (lesson_id, file_path, event_type, timestamp) "
                "VALUES (?, 'hub.py', 'hookify_warn', '2026-02-26T10:00:00')",
                [lid],
            )
        conn.commit()
        score = relevance_score(conn, lid, context="other.py", semantic_sim=0.5)
        # 0.5*0.5 + 0.3*0.5 + 0.2*1.0 = 0.25 + 0.15 + 0.20 = 0.60
        assert score > 0.55


class TestLearnRecordCLI:
    """CLI learn record creates a surfacing event."""

    def test_learn_record_cli(self, db_path):
        runner = CliRunner()
        # First create a lesson to record a surfacing event for
        conn = init_db(db_path)
        lid = insert_lesson(conn, {"title": "T", "one_liner": "X", "created_date": "2026-01-01"})
        conn.close()

        result = runner.invoke(
            main,
            [
                "--db",
                str(db_path),
                "learn",
                "record",
                "--lesson-id",
                str(lid),
                "--hook",
                "plan",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Recorded" in result.output

        # Verify the surfacing event was actually written to the DB
        conn2 = init_db(db_path)
        events = conn2.execute("SELECT * FROM surfacing_events").fetchall()
        assert len(events) == 1
        assert events[0]["hook_point"] == "plan"
        assert events[0]["lesson_id"] == lid

    def test_learn_record_with_context(self, db_path):
        runner = CliRunner()
        conn = init_db(db_path)
        lid = insert_lesson(conn, {"title": "T", "one_liner": "X", "created_date": "2026-01-01"})
        conn.close()

        result = runner.invoke(
            main,
            [
                "--db",
                str(db_path),
                "learn",
                "record",
                "--lesson-id",
                str(lid),
                "--hook",
                "bash",
                "--context",
                "test failure in hub.py",
            ],
        )
        assert result.exit_code == 0, result.output

        # Verify the event was stored
        conn2 = init_db(db_path)
        events = conn2.execute("SELECT * FROM surfacing_events").fetchall()
        assert len(events) == 1
        assert events[0]["hook_point"] == "bash"
        assert "hub.py" in events[0]["context"]


def test_record_outcome_false_positive(tmp_path):
    """record_outcome accepts false_positive outcome."""
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_outcome, record_surfacing

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(
        conn,
        {
            "title": "Test",
            "one_liner": "test",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    eid = record_surfacing(conn, lid, "plan", "ctx")
    record_outcome(conn, eid, "false_positive")
    row = conn.execute("SELECT outcome FROM surfacing_events WHERE id = ?", [eid]).fetchone()
    assert row["outcome"] == "false_positive"


def test_record_outcome_recurrence(tmp_path):
    """record_outcome accepts recurrence outcome."""
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_outcome, record_surfacing

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(
        conn,
        {
            "title": "Test2",
            "one_liner": "test2",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    eid = record_surfacing(conn, lid, "bash", "ctx")
    record_outcome(conn, eid, "recurrence")
    row = conn.execute("SELECT outcome FROM surfacing_events WHERE id = ?", [eid]).fetchone()
    assert row["outcome"] == "recurrence"


def test_record_outcome_rejects_invalid(tmp_path):
    """record_outcome raises ValueError on unknown outcome."""
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_outcome, record_surfacing

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(
        conn,
        {
            "title": "Test3",
            "one_liner": "test3",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    eid = record_surfacing(conn, lid, "plan", "ctx")
    with pytest.raises(ValueError, match="Invalid outcome"):
        record_outcome(conn, eid, "wrong")


class TestEvaluateCommit:
    """Tests for evaluate_commit — post-commit outcome evaluation."""

    def test_marks_heeded_when_antipattern_absent(self, db_path):
        """If the diff does NOT contain the anti-pattern, outcome should be 'heeded'."""
        from lessons_db.learn import evaluate_commit, record_surfacing

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "No bare except",
                "one_liner": "Always log before swallowing exceptions",
                "detection_pattern": "except:\n    pass",
                "created_date": "2026-02-26",
            },
        )
        eid = record_surfacing(conn, lid, "edit", "src/hub.py")

        # Diff that does NOT contain the anti-pattern
        diff_text = """\
diff --git a/src/hub.py b/src/hub.py
--- a/src/hub.py
+++ b/src/hub.py
@@ -10,3 +10,5 @@ def process():
+    try:
+        do_stuff()
+    except Exception as e:
+        logger.error("Failed: %s", e)
"""
        results = evaluate_commit(conn, diff_text, hours=24, dry_run=False)
        assert len(results) == 1
        assert results[0]["outcome"] == "heeded"
        assert results[0]["event_id"] == eid

        # Verify DB was updated
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id=?", [eid]).fetchone()
        assert row["outcome"] == "heeded"

    def test_marks_dismissed_when_antipattern_present(self, db_path):
        """If the diff contains the anti-pattern, outcome should be 'dismissed'."""
        from lessons_db.learn import evaluate_commit, record_surfacing

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "No bare except",
                "one_liner": "Always log before swallowing exceptions",
                "detection_pattern": r"except\s*:",
                "created_date": "2026-02-26",
            },
        )
        eid = record_surfacing(conn, lid, "edit", "src/hub.py")

        # Diff that DOES contain the anti-pattern
        diff_text = """\
diff --git a/src/hub.py b/src/hub.py
--- a/src/hub.py
+++ b/src/hub.py
@@ -10,3 +10,5 @@ def process():
+    try:
+        do_stuff()
+    except:
+        pass
"""
        results = evaluate_commit(conn, diff_text, hours=24, dry_run=False)
        assert len(results) == 1
        assert results[0]["outcome"] == "dismissed"

        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id=?", [eid]).fetchone()
        assert row["outcome"] == "dismissed"

    def test_uses_detection_patterns_table_regex(self, db_path):
        """Falls back to detection_patterns table regex when lesson.detection_pattern is empty."""
        from lessons_db.db import insert_detection_pattern
        from lessons_db.learn import evaluate_commit, record_surfacing

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "No bare except",
                "one_liner": "Always log before swallowing exceptions",
                "created_date": "2026-02-26",
            },
        )
        insert_detection_pattern(
            conn,
            {
                "lesson_id": lid,
                "pattern_type": "syntactic",
                "regex": r"except\s*:",
            },
        )
        eid = record_surfacing(conn, lid, "edit", "src/hub.py")

        diff_text = "+    except:\n+        pass\n"
        results = evaluate_commit(conn, diff_text, hours=24, dry_run=False)
        assert len(results) == 1
        assert results[0]["outcome"] == "dismissed"

    def test_skips_already_resolved_events(self, db_path):
        """Events with outcome != 'unknown' should not be re-evaluated."""
        from lessons_db.learn import evaluate_commit, record_outcome, record_surfacing

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Test",
                "one_liner": "test",
                "detection_pattern": "bad_pattern",
                "created_date": "2026-02-26",
            },
        )
        eid = record_surfacing(conn, lid, "edit", "src/hub.py")
        record_outcome(conn, eid, "heeded")  # already resolved

        results = evaluate_commit(conn, "+bad_pattern\n", hours=24, dry_run=False)
        assert len(results) == 0

    def test_dry_run_does_not_update_db(self, db_path):
        """With dry_run=True, outcomes should be computed but NOT written to DB."""
        from lessons_db.learn import evaluate_commit, record_surfacing

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Test",
                "one_liner": "test",
                "detection_pattern": "bad_pattern",
                "created_date": "2026-02-26",
            },
        )
        eid = record_surfacing(conn, lid, "edit", "src/hub.py")

        results = evaluate_commit(conn, "+bad_pattern\n", hours=24, dry_run=True)
        assert len(results) == 1
        assert results[0]["outcome"] == "dismissed"

        # DB should still show 'unknown'
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id=?", [eid]).fetchone()
        assert row["outcome"] == "unknown"

    def test_respects_hours_window(self, db_path):
        """Events older than the window should not be evaluated."""
        from datetime import UTC, datetime, timedelta

        from lessons_db.learn import evaluate_commit

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Test",
                "one_liner": "test",
                "detection_pattern": "bad_pattern",
                "created_date": "2026-02-26",
            },
        )
        # Insert a surfacing event from 48 hours ago
        old_ts = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        conn.execute(
            "INSERT INTO surfacing_events (lesson_id, hook_point, context, outcome, timestamp) "
            "VALUES (?, 'edit', 'ctx', 'unknown', ?)",
            [lid, old_ts],
        )
        conn.commit()

        results = evaluate_commit(conn, "+bad_pattern\n", hours=24, dry_run=False)
        assert len(results) == 0

    def test_no_pattern_skips_lesson(self, db_path):
        """Lessons with no detection pattern at all should be skipped (left unknown)."""
        from lessons_db.learn import evaluate_commit, record_surfacing

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Test",
                "one_liner": "test",
                "created_date": "2026-02-26",
            },
        )
        record_surfacing(conn, lid, "edit", "src/hub.py")

        results = evaluate_commit(conn, "+some code\n", hours=24, dry_run=False)
        assert len(results) == 0

    def test_multiple_events_evaluated(self, db_path):
        """Multiple unknown events for different lessons should all be evaluated."""
        from lessons_db.learn import evaluate_commit, record_surfacing

        conn = init_db(db_path)
        lid1 = insert_lesson(
            conn,
            {
                "title": "Lesson A",
                "one_liner": "A",
                "detection_pattern": "anti_a",
                "created_date": "2026-02-26",
            },
        )
        lid2 = insert_lesson(
            conn,
            {
                "title": "Lesson B",
                "one_liner": "B",
                "detection_pattern": "anti_b",
                "created_date": "2026-02-26",
            },
        )
        record_surfacing(conn, lid1, "edit", "ctx")
        record_surfacing(conn, lid2, "edit", "ctx")

        # Diff contains anti_a but not anti_b
        results = evaluate_commit(conn, "+anti_a found here\n", hours=24, dry_run=False)
        assert len(results) == 2
        outcomes = {r["lesson_id"]: r["outcome"] for r in results}
        assert outcomes[lid1] == "dismissed"
        assert outcomes[lid2] == "heeded"


class TestEvaluateCommitCLI:
    """CLI tests for 'learn evaluate-commit'."""

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["learn", "evaluate-commit", "--help"])
        assert result.exit_code == 0
        assert "evaluate-commit" in result.output or "Evaluate" in result.output

    def test_evaluate_commit_no_events(self, db_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "learn", "evaluate-commit"],
        )
        assert result.exit_code == 0
        assert "No unknown surfacing events" in result.output

    def test_evaluate_commit_dry_run(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Test",
                "one_liner": "test",
                "detection_pattern": "bad_pattern",
                "created_date": "2026-02-26",
            },
        )
        record_surfacing(conn, lid, "edit", "src/hub.py")
        conn.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "learn", "evaluate-commit", "--dry-run", "--diff-text", "+bad_pattern\n"],
        )
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output or "dry" in result.output.lower()

        # Verify DB still shows 'unknown'
        conn2 = init_db(db_path)
        row = conn2.execute("SELECT outcome FROM surfacing_events WHERE lesson_id=?", [lid]).fetchone()
        assert row["outcome"] == "unknown"

    def test_evaluate_commit_with_diff_text(self, db_path):
        """Passing --diff-text directly bypasses git."""
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Test",
                "one_liner": "test",
                "detection_pattern": "anti_pattern_x",
                "created_date": "2026-02-26",
            },
        )
        record_surfacing(conn, lid, "edit", "src/hub.py")
        conn.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "learn", "evaluate-commit", "--diff-text", "+anti_pattern_x here\n"],
        )
        assert result.exit_code == 0, result.output
        assert "dismissed" in result.output.lower()

    def test_evaluate_commit_hours_flag(self, db_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "learn", "evaluate-commit", "--hours", "1"],
        )
        assert result.exit_code == 0


class TestSurfacingStats:
    def test_returns_zero_counts_when_empty(self, db_path):
        conn = init_db(db_path)
        stats = surfacing_stats(conn)
        assert stats["total_surfacing_events"] == 0
        assert stats["heed_rate"] is None

    def test_counts_heeded_and_dismissed(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        e1 = record_surfacing(conn, lid, "read", "a.py")
        record_outcome(conn, e1, "heeded")
        e2 = record_surfacing(conn, lid, "read", "b.py")
        record_outcome(conn, e2, "dismissed")
        record_surfacing(conn, lid, "plan", "c")  # unknown
        stats = surfacing_stats(conn)
        assert stats["total_surfacing_events"] == 3
        assert stats["heeded"] == 1
        assert stats["dismissed"] == 1
        assert stats["unknown"] == 1
        assert stats["heed_rate"] == 0.33


class TestFeedbackLoopEndToEnd:
    """Integration test: full feedback loop from lesson creation through outcome evaluation."""

    def test_feedback_loop_end_to_end(self, db_path):
        """End-to-end: lesson → surfacing → evaluate_commit → outcome transitions.

        Path 1 (dismissed): diff contains the anti-pattern → outcome = 'dismissed'
        Path 2 (heeded): diff does NOT contain the anti-pattern → outcome = 'heeded'
        """
        from lessons_db.learn import evaluate_commit, record_surfacing

        # --- Setup: create a lesson with a regex detection pattern ---
        conn = init_db(db_path)
        lesson_id = insert_lesson(
            conn,
            {
                "title": "No bare except",
                "one_liner": "Always log before swallowing exceptions",
                "detection_pattern": r"except\s*:",
                "created_date": "2026-02-26",
            },
        )

        # === Path 1: DISMISSED (anti-pattern present in diff) ===

        # Step 1: Record a surfacing event — outcome starts as 'unknown'
        event_id_dismissed = record_surfacing(conn, lesson_id, hook_point="edit", context="src/hub.py")
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id = ?", [event_id_dismissed]).fetchone()
        assert row["outcome"] == "unknown", "Surfacing event should start as 'unknown'"

        # Step 2: Simulate a commit diff that CONTAINS the anti-pattern
        diff_with_antipattern = """\
diff --git a/src/hub.py b/src/hub.py
--- a/src/hub.py
+++ b/src/hub.py
@@ -10,3 +10,5 @@ def process():
+    try:
+        do_stuff()
+    except:
+        pass
"""

        # Step 3: Run evaluate_commit — should mark as 'dismissed'
        results = evaluate_commit(conn, diff_with_antipattern, hours=24, dry_run=False)
        assert len(results) == 1
        assert results[0]["event_id"] == event_id_dismissed
        assert results[0]["lesson_id"] == lesson_id
        assert results[0]["outcome"] == "dismissed"

        # Step 4: Verify the DB outcome changed from 'unknown' to 'dismissed'
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id = ?", [event_id_dismissed]).fetchone()
        assert row["outcome"] == "dismissed", (
            "After evaluate_commit with anti-pattern present, outcome must be 'dismissed'"
        )

        # === Path 2: HEEDED (anti-pattern absent from diff) ===

        # Step 5: Record a new surfacing event for the same lesson
        event_id_heeded = record_surfacing(conn, lesson_id, hook_point="plan", context="src/hub.py refactor")
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id = ?", [event_id_heeded]).fetchone()
        assert row["outcome"] == "unknown", "New surfacing event should start as 'unknown'"

        # Step 6: Simulate a commit diff WITHOUT the anti-pattern (proper exception handling)
        diff_without_antipattern = """\
diff --git a/src/hub.py b/src/hub.py
--- a/src/hub.py
+++ b/src/hub.py
@@ -10,3 +10,5 @@ def process():
+    try:
+        do_stuff()
+    except Exception as e:
+        logger.error("Failed: %s", e)
"""

        # Step 7: Run evaluate_commit — should mark as 'heeded'
        results = evaluate_commit(conn, diff_without_antipattern, hours=24, dry_run=False)
        assert len(results) == 1
        assert results[0]["event_id"] == event_id_heeded
        assert results[0]["lesson_id"] == lesson_id
        assert results[0]["outcome"] == "heeded"

        # Step 8: Verify the DB outcome changed from 'unknown' to 'heeded'
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id = ?", [event_id_heeded]).fetchone()
        assert row["outcome"] == "heeded", "After evaluate_commit with anti-pattern absent, outcome must be 'heeded'"

        # Step 9: Verify the first event is still 'dismissed' (not re-evaluated)
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id = ?", [event_id_dismissed]).fetchone()
        assert row["outcome"] == "dismissed", "Previously dismissed event must not be re-evaluated"


class TestWinStreaksTable:
    """Tests for win_streaks table creation."""

    def test_win_streaks_table_exists(self, db_path):
        conn = init_db(db_path)
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='win_streaks'").fetchall()
        assert len(rows) == 1

    def test_win_streaks_table_schema(self, db_path):
        conn = init_db(db_path)
        cols = conn.execute("PRAGMA table_info('win_streaks')").fetchall()
        col_names = {c["name"] for c in cols}
        assert {"id", "category", "current_streak", "longest_streak", "last_updated"} == col_names


class TestUpdateWinStreak:
    """Tests for update_win_streak — streak increment and reset."""

    def test_increment_on_win(self, db_path):
        conn = init_db(db_path)
        info = update_win_streak(conn, "async", won=True)
        assert info["current_streak"] == 1
        assert info["longest_streak"] == 1
        assert info["category"] == "async"

    def test_consecutive_wins_increment(self, db_path):
        conn = init_db(db_path)
        update_win_streak(conn, "async", won=True)
        update_win_streak(conn, "async", won=True)
        info = update_win_streak(conn, "async", won=True)
        assert info["current_streak"] == 3
        assert info["longest_streak"] == 3

    def test_resets_on_loss(self, db_path):
        conn = init_db(db_path)
        update_win_streak(conn, "async", won=True)
        update_win_streak(conn, "async", won=True)
        info = update_win_streak(conn, "async", won=False)
        assert info["current_streak"] == 0
        assert info["longest_streak"] == 2

    def test_longest_streak_preserved_after_reset(self, db_path):
        conn = init_db(db_path)
        for _ in range(5):
            update_win_streak(conn, "error-handling", won=True)
        update_win_streak(conn, "error-handling", won=False)
        update_win_streak(conn, "error-handling", won=True)
        info = update_win_streak(conn, "error-handling", won=True)
        assert info["current_streak"] == 2
        assert info["longest_streak"] == 5

    def test_loss_on_first_record(self, db_path):
        conn = init_db(db_path)
        info = update_win_streak(conn, "new-category", won=False)
        assert info["current_streak"] == 0
        assert info["longest_streak"] == 0

    def test_separate_categories_independent(self, db_path):
        conn = init_db(db_path)
        update_win_streak(conn, "alpha", won=True)
        update_win_streak(conn, "alpha", won=True)
        update_win_streak(conn, "beta", won=True)
        info_alpha = update_win_streak(conn, "alpha", won=True)
        info_beta = update_win_streak(conn, "beta", won=False)
        assert info_alpha["current_streak"] == 3
        assert info_beta["current_streak"] == 0


class TestShouldSurfacePositive:
    """Tests for should_surface_positive — variable-ratio probability gate."""

    def test_returns_tuple_bool_dict(self, db_path):
        conn = init_db(db_path)
        result = should_surface_positive(conn, "async")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], dict)

    def test_streak_info_keys(self, db_path):
        conn = init_db(db_path)
        _, info = should_surface_positive(conn, "async")
        assert "current_streak" in info
        assert "longest_streak" in info
        assert "category" in info
        assert info["category"] == "async"

    def test_cold_start_streak_info_zeros(self, db_path):
        conn = init_db(db_path)
        _, info = should_surface_positive(conn, "never-seen")
        assert info["current_streak"] == 0
        assert info["longest_streak"] == 0

    def test_reflects_updated_streak(self, db_path):
        conn = init_db(db_path)
        update_win_streak(conn, "async", won=True)
        update_win_streak(conn, "async", won=True)
        _, info = should_surface_positive(conn, "async")
        assert info["current_streak"] == 2
        assert info["longest_streak"] == 2

    def test_probability_distribution(self, db_path):
        """Over many samples, ~30% should return True (Skinner variable-ratio)."""
        import random as _random

        conn = init_db(db_path)
        _random.seed(42)
        results = [should_surface_positive(conn, "test")[0] for _ in range(1000)]
        ratio = sum(results) / len(results)
        # Allow +-5% tolerance around 0.3
        assert 0.25 <= ratio <= 0.35, f"Expected ~30% True, got {ratio:.1%}"


class TestFindExceptions:
    """Tests for find_exceptions — SFBT exception-finding for internalized anti-patterns."""

    def test_no_data_returns_empty(self, db_path):
        """With no surfacing events at all, find_exceptions returns empty list."""
        from lessons_db.learn import find_exceptions

        conn = init_db(db_path)
        result = find_exceptions(conn)
        assert result == []

    def test_identifies_absent_antipatterns(self, db_path):
        """A dismissed lesson absent from recent sessions is identified as an exception."""
        from lessons_db.learn import find_exceptions

        conn = init_db(db_path)
        # Create a negative lesson
        lid = insert_lesson(
            conn,
            {
                "title": "No bare except",
                "one_liner": "Always log before swallowing exceptions",
                "category": "error-handling",
                "polarity": "negative",
                "created_date": "2026-02-26",
            },
        )

        # Create an old session where this lesson was dismissed (far in the past)
        conn.execute(
            "INSERT INTO surfacing_events "
            "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
            "VALUES (?, 'edit', 'ctx', 'dismissed', '2026-01-01T10:00:00', 'old-session-1')",
            [lid],
        )

        # Create 5 recent sessions WITHOUT this lesson being dismissed.
        # Using 5 sessions ensures the old session falls outside the lookback window.
        for i in range(5):
            other_lid = insert_lesson(
                conn,
                {
                    "title": f"Other lesson {i}",
                    "one_liner": f"Other {i}",
                    "polarity": "negative",
                    "created_date": "2026-02-26",
                },
            )
            conn.execute(
                "INSERT INTO surfacing_events "
                "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
                "VALUES (?, 'read', 'ctx', 'heeded', ?, ?)",
                [other_lid, f"2026-02-2{3 + i}T10:00:00", f"recent-session-{i}"],
            )
        conn.commit()

        result = find_exceptions(conn, lookback_sessions=5)
        assert len(result) == 1
        assert result[0]["lesson_id"] == lid
        assert result[0]["title"] == "No bare except"
        assert result[0]["category"] == "error-handling"
        assert result[0]["absent_sessions"] == 5  # 5 recent sessions checked

    def test_excludes_patterns_that_appeared_recently(self, db_path):
        """A dismissed lesson that was also dismissed in a recent session is NOT an exception."""
        from lessons_db.learn import find_exceptions

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "No bare except",
                "one_liner": "Always log before swallowing exceptions",
                "category": "error-handling",
                "polarity": "negative",
                "created_date": "2026-02-26",
            },
        )

        # Create an old session where this lesson was dismissed
        conn.execute(
            "INSERT INTO surfacing_events "
            "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
            "VALUES (?, 'edit', 'ctx', 'dismissed', '2026-02-20T10:00:00', 'old-session-1')",
            [lid],
        )

        # Create a recent session where this lesson was ALSO dismissed
        conn.execute(
            "INSERT INTO surfacing_events "
            "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
            "VALUES (?, 'edit', 'ctx', 'dismissed', '2026-02-28T10:00:00', 'recent-session-1')",
            [lid],
        )
        conn.commit()

        result = find_exceptions(conn, lookback_sessions=5)
        assert len(result) == 0

    def test_positive_lessons_excluded(self, db_path):
        """Positive lessons should never appear as exceptions even if historically dismissed."""
        from lessons_db.learn import find_exceptions

        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Good pattern",
                "one_liner": "Use this pattern",
                "category": "patterns",
                "polarity": "positive",
                "created_date": "2026-02-26",
            },
        )

        # Create dismissed event for a positive lesson (unusual, but possible)
        conn.execute(
            "INSERT INTO surfacing_events "
            "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
            "VALUES (?, 'edit', 'ctx', 'dismissed', '2026-02-20T10:00:00', 'old-session-1')",
            [lid],
        )
        # Recent session without this lesson
        other_lid = insert_lesson(
            conn,
            {
                "title": "Other",
                "one_liner": "Other",
                "polarity": "negative",
                "created_date": "2026-02-26",
            },
        )
        conn.execute(
            "INSERT INTO surfacing_events "
            "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
            "VALUES (?, 'read', 'ctx', 'heeded', '2026-02-28T10:00:00', 'recent-session-1')",
            [other_lid],
        )
        conn.commit()

        result = find_exceptions(conn, lookback_sessions=5)
        assert len(result) == 0


class TestFindExceptionsCLI:
    """CLI tests for 'learn find-exceptions'."""

    def test_help_flag(self):
        runner = CliRunner()
        result = runner.invoke(main, ["learn", "find-exceptions", "--help"])
        assert result.exit_code == 0
        assert "find-exceptions" in result.output or "SFBT" in result.output

    def test_no_exceptions_message(self, db_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "learn", "find-exceptions"],
        )
        assert result.exit_code == 0
        assert "No exceptions found" in result.output

    def test_lookback_flag(self, db_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "learn", "find-exceptions", "--lookback", "3"],
        )
        assert result.exit_code == 0

    def test_reports_exceptions(self, db_path):
        """With proper data, CLI should report internalized patterns."""
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "No bare except",
                "one_liner": "Always log before swallowing exceptions",
                "category": "error-handling",
                "polarity": "negative",
                "created_date": "2026-02-26",
            },
        )
        # Old dismissed event (far in the past)
        conn.execute(
            "INSERT INTO surfacing_events "
            "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
            "VALUES (?, 'edit', 'ctx', 'dismissed', '2026-01-01T10:00:00', 'old-session-1')",
            [lid],
        )
        # Create 5 recent sessions without this lesson (pushes old session outside lookback)
        for i in range(5):
            other_lid = insert_lesson(
                conn,
                {
                    "title": f"Other {i}",
                    "one_liner": f"Other {i}",
                    "polarity": "negative",
                    "created_date": "2026-02-26",
                },
            )
            conn.execute(
                "INSERT INTO surfacing_events "
                "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
                "VALUES (?, 'read', 'ctx', 'heeded', ?, ?)",
                [other_lid, f"2026-02-2{3 + i}T10:00:00", f"recent-session-{i}"],
            )
        conn.commit()
        conn.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "learn", "find-exceptions"],
        )
        assert result.exit_code == 0, result.output
        assert "internalized" in result.output.lower()
        assert "error-handling" in result.output
        assert "No bare except" in result.output
