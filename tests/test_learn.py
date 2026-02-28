"""Tests for learning pipeline — outcome tracking and relevance scoring."""

import pytest
from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db, insert_lesson
from lessons_db.learn import (
    record_outcome,
    record_surfacing,
    relevance_score,
    surfacing_stats,
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
