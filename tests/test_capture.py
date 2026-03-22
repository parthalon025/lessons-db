"""Tests for positive knowledge capture."""

import json
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from lessons_db.capture import (
    capture_from_design_doc,
    capture_positive_manual,
    detect_wins,
    list_drafts,
    promote_draft,
    score_one_liner,
)
from lessons_db.db import get_lesson, init_db


class TestScoreOneLiner:
    """Ollama-based quality scoring."""

    def test_score_parses_integer_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "4"}
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            score = score_one_liner("Store subscriber refs on self for lifecycle cleanup")
        assert score == 4

    def test_score_returns_default_on_network_error(self):
        with patch("lessons_db.capture.requests.post", side_effect=Exception("timeout")):
            score = score_one_liner("anything")
        assert score == 3

    def test_score_returns_default_on_bad_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "not-a-number"}
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            score = score_one_liner("something")
        assert score == 3


class TestCaptureFromDesignDoc:
    """Auto-capture drafts from design doc content."""

    def test_creates_draft_in_db(self, db_path, tmp_path):
        doc = tmp_path / "design.md"
        doc.write_text("## Decision\nDual-axis testing outperforms single-axis in integration scenarios.")
        conn = init_db(db_path)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": json.dumps(
                {
                    "entries": [
                        {
                            "one_liner": "Dual-axis testing catches integration bugs",
                            "why": "Tests both horizontal and vertical",
                            "category": "testing-pattern",
                        }
                    ]
                }
            )
        }
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            drafts = capture_from_design_doc(doc, conn)

        assert len(drafts) == 1
        rows = conn.execute("SELECT * FROM capture_drafts WHERE status='pending'").fetchall()
        assert len(rows) == 1

    def test_returns_empty_on_ollama_failure(self, db_path, tmp_path):
        doc = tmp_path / "design.md"
        doc.write_text("Some content")
        conn = init_db(db_path)
        with patch("lessons_db.capture.requests.post", side_effect=Exception("timeout")):
            drafts = capture_from_design_doc(doc, conn)
        assert drafts == []

    def test_draft_has_pending_status(self, db_path, tmp_path):
        doc = tmp_path / "design.md"
        doc.write_text("Decision: use Thompson Sampling for routing")
        conn = init_db(db_path)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": json.dumps(
                {
                    "entries": [
                        {
                            "one_liner": "Thompson Sampling beats round-robin",
                            "why": "Adapts to observed performance",
                            "category": "architecture-pattern",
                        }
                    ]
                }
            )
        }
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            capture_from_design_doc(doc, conn)
        row = conn.execute("SELECT status FROM capture_drafts LIMIT 1").fetchone()
        assert row["status"] == "pending"


class TestPromoteDraft:
    """Promoting a draft to a live lesson."""

    def test_promote_inserts_lesson(self, db_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', ?, 'auto_design_doc')",
            [
                "raw",
                json.dumps({"one_liner": "Test pattern", "why": "Because", "category": "testing-pattern"}),
                date.today().isoformat(),
            ],
        )
        conn.commit()
        draft_id = conn.execute("SELECT id FROM capture_drafts LIMIT 1").fetchone()["id"]

        lesson_id = promote_draft(conn, draft_id)
        assert lesson_id is not None
        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "positive"
        assert lesson["tier"] == "noticed"

    def test_promote_marks_draft_approved(self, db_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', ?, 'auto_design_doc')",
            [
                "raw",
                json.dumps({"one_liner": "X", "why": "Y", "category": "architecture-pattern"}),
                date.today().isoformat(),
            ],
        )
        conn.commit()
        draft_id = conn.execute("SELECT id FROM capture_drafts LIMIT 1").fetchone()["id"]
        promote_draft(conn, draft_id)
        status = conn.execute("SELECT status FROM capture_drafts WHERE id=?", [draft_id]).fetchone()["status"]
        assert status == "approved"


class TestListDrafts:
    def test_list_returns_pending_drafts(self, db_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES ('raw', '{}', 'pending', '2026-02-26', 'auto_design_doc')"
        )
        conn.commit()
        drafts = list_drafts(conn)
        assert len(drafts) == 1
        assert drafts[0]["status"] == "pending"


class TestCaptureFromTranscript:
    @patch("lessons_db.capture.requests.post")
    def test_extracts_lessons_from_transcript(self, mock_post, db_path):
        from lessons_db.capture import capture_from_transcript
        from lessons_db.db import init_db

        mock_post.return_value = MagicMock(
            json=lambda: {
                "response": '{"lessons": [{"one_liner": "Always log before fallback", "cluster": "A", "tier": "lesson"}]}'
            },
            raise_for_status=lambda: None,
        )
        conn = init_db(db_path)
        result = capture_from_transcript("Session transcript text here. " * 10, conn)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["one_liner"] == "Always log before fallback"
        rows = conn.execute("SELECT * FROM capture_drafts").fetchall()
        assert len(rows) == 1
        assert rows[0]["source"] == "auto_transcript"

    @patch("lessons_db.capture.requests.post")
    def test_returns_empty_on_ollama_failure(self, mock_post, db_path):
        from lessons_db.capture import capture_from_transcript
        from lessons_db.db import init_db

        mock_post.side_effect = Exception("network error")
        conn = init_db(db_path)
        with pytest.raises(Exception):
            capture_from_transcript("transcript " * 20, conn)

    def test_returns_empty_on_short_transcript(self, db_path):
        from lessons_db.capture import capture_from_transcript
        from lessons_db.db import init_db

        conn = init_db(db_path)
        result = capture_from_transcript("too short", conn)
        assert result == []

    @patch("lessons_db.capture.requests.post")
    def test_draft_cap_truncates_at_50(self, mock_post, db_path):
        """LLM returning >50 lessons must be silently truncated to 50."""
        from lessons_db.capture import capture_from_transcript
        from lessons_db.db import init_db

        # Simulate LLM returning 100 lessons — should be capped at 50
        many_lessons = [{"one_liner": f"Lesson {i}", "cluster": "A", "tier": "lesson"} for i in range(100)]
        mock_post.return_value = MagicMock(
            json=lambda: {"response": json.dumps({"lessons": many_lessons})},
            raise_for_status=lambda: None,
        )
        # score_one_liner is called per lesson — mock it to always pass quality gate
        with patch("lessons_db.capture.score_one_liner", return_value=5):
            conn = init_db(db_path)
            result = capture_from_transcript("Session transcript text. " * 10, conn)

        assert len(result) <= 50, f"Expected ≤50 lessons, got {len(result)}"
        rows = conn.execute("SELECT COUNT(*) as cnt FROM capture_drafts").fetchone()
        assert rows["cnt"] <= 50

    @patch("lessons_db.capture.requests.post")
    def test_draft_cap_logs_warning_when_exceeded(self, mock_post, db_path, caplog):
        """When LLM returns >50 lessons, a WARNING log must be emitted."""
        import logging

        from lessons_db.capture import capture_from_transcript
        from lessons_db.db import init_db

        many_lessons = [{"one_liner": f"Lesson {i}", "cluster": "A", "tier": "lesson"} for i in range(75)]
        mock_post.return_value = MagicMock(
            json=lambda: {"response": json.dumps({"lessons": many_lessons})},
            raise_for_status=lambda: None,
        )
        conn = init_db(db_path)
        with patch("lessons_db.capture.score_one_liner", return_value=5):
            with caplog.at_level(logging.WARNING, logger="lessons_db.capture"):
                capture_from_transcript("Session transcript text. " * 10, conn)

        assert any("75" in r.message and "truncating" in r.message for r in caplog.records), (
            "Expected a WARNING log mentioning 75 and truncating"
        )

    @patch("lessons_db.capture.requests.post")
    def test_exactly_50_lessons_not_truncated(self, mock_post, db_path):
        """Exactly 50 lessons must pass through without truncation."""
        from lessons_db.capture import capture_from_transcript
        from lessons_db.db import init_db

        exactly_50 = [{"one_liner": f"Lesson {i}", "cluster": "A", "tier": "lesson"} for i in range(50)]
        mock_post.return_value = MagicMock(
            json=lambda: {"response": json.dumps({"lessons": exactly_50})},
            raise_for_status=lambda: None,
        )
        with patch("lessons_db.capture.score_one_liner", return_value=5):
            conn = init_db(db_path)
            result = capture_from_transcript("Session transcript text. " * 10, conn)

        assert len(result) == 50


class TestCaptureFromDiff:
    @patch("lessons_db.capture.requests.post")
    def test_extracts_lessons_from_diff(self, mock_post, db_path):
        from lessons_db.capture import capture_from_diff
        from lessons_db.db import init_db

        mock_post.return_value = MagicMock(
            json=lambda: {
                "response": '{"lessons": [{"one_liner": "Never use bare except", "cluster": "A", "tier": "lesson"}]}'
            },
            raise_for_status=lambda: None,
        )
        conn = init_db(db_path)
        result = capture_from_diff(
            "diff --git a/foo.py b/foo.py\n-except:\n-    pass\n+except Exception as e:\n+    log(e)", conn
        )
        assert isinstance(result, list)
        assert len(result) == 1
        rows = conn.execute("SELECT * FROM capture_drafts").fetchall()
        assert len(rows) == 1
        assert rows[0]["source"] == "auto_diff"

    @patch("lessons_db.capture.requests.post")
    def test_returns_empty_on_empty_diff(self, mock_post, db_path):
        from lessons_db.capture import capture_from_diff
        from lessons_db.db import init_db

        conn = init_db(db_path)
        result = capture_from_diff("", conn)
        assert result == []

    def test_returns_empty_on_short_diff(self, db_path):
        from lessons_db.capture import capture_from_diff
        from lessons_db.db import init_db

        conn = init_db(db_path)
        result = capture_from_diff("short diff txt", conn)
        assert result == []

    @patch("lessons_db.capture.requests.post")
    def test_returns_empty_on_ollama_failure(self, mock_post, db_path):
        from lessons_db.capture import capture_from_diff
        from lessons_db.db import init_db

        mock_post.side_effect = Exception("connection refused")
        conn = init_db(db_path)
        with pytest.raises(Exception):
            capture_from_diff("diff --git a/foo.py b/foo.py\n-except:\n+except Exception:", conn)


class TestPromoteDraftPolarity:
    def test_auto_transcript_promotes_as_negative(self, db_path):
        conn = init_db(db_path)
        entry = {"one_liner": "Never swallow exceptions silently", "cluster": "A", "tier": "lesson"}
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', '2026-02-27', 'auto_transcript')",
            ["raw", json.dumps(entry)],
        )
        conn.commit()
        draft_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        lesson_id = promote_draft(conn, draft_id)

        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "negative"
        assert lesson["source"] == "auto_transcript"

    def test_auto_transcript_positive_promotes_as_positive(self, db_path):
        conn = init_db(db_path)
        entry = {"one_liner": "Dual-axis testing catches integration bugs", "cluster": "", "tier": "lesson"}
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', '2026-02-27', 'auto_transcript_positive')",
            ["raw", json.dumps(entry)],
        )
        conn.commit()
        draft_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        lesson_id = promote_draft(conn, draft_id)

        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "positive"

    def test_auto_diff_promotes_as_negative(self, db_path):
        conn = init_db(db_path)
        entry = {"one_liner": "Always validate schema before inserting", "cluster": "B", "tier": "lesson"}
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', '2026-02-27', 'auto_diff')",
            ["raw", json.dumps(entry)],
        )
        conn.commit()
        draft_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        lesson_id = promote_draft(conn, draft_id)

        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "negative"

    def test_unknown_source_defaults_to_negative(self, db_path):
        conn = init_db(db_path)
        entry = {"one_liner": "Some lesson", "tier": "lesson"}
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', '2026-02-27', 'unknown_future_source')",
            ["raw", json.dumps(entry)],
        )
        conn.commit()
        draft_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        lesson_id = promote_draft(conn, draft_id)
        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "negative"
        assert lesson["entry_type"] == "lesson"


class TestCapturePositiveManual:
    """Quality gate and lesson creation for manual capture."""

    def test_rejects_below_quality_threshold(self, db_path):
        conn = init_db(db_path)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "2"}
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            result = capture_positive_manual(conn, "vague insight", "unclear", "testing-pattern")
        assert result is None

    def test_accepts_at_quality_threshold(self, db_path):
        conn = init_db(db_path)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "3"}
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            lesson_id = capture_positive_manual(
                conn, "Dual-axis testing catches integration bugs", "Tests both axes", "testing-pattern"
            )
        assert lesson_id is not None

    def test_lesson_has_correct_polarity_and_tier(self, db_path):
        conn = init_db(db_path)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "4"}
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            lesson_id = capture_positive_manual(
                conn,
                "Thompson Sampling beats round-robin routing",
                "Adapts to observed performance",
                "architecture-pattern",
            )
        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "positive"
        assert lesson["tier"] == "noticed"


class TestCapturePositiveCLI:
    """CLI capture positive subcommand is accessible and prompts correctly."""

    def test_capture_positive_subcommand_exists(self):
        from click.testing import CliRunner

        from lessons_db.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["capture", "positive", "--help"])
        assert result.exit_code == 0, result.output
        assert "positive" in result.output.lower()

    def test_capture_positive_prompts_and_creates_lesson(self, db_path):
        from click.testing import CliRunner

        from lessons_db.cli import main

        runner = CliRunner()
        # Trailing \n ensures Click's CliRunner correctly terminates the last prompt
        user_input = (
            "\n".join(
                [
                    "Dual-axis testing catches integration seam bugs",  # one_liner
                    "Tests both horizontal (surface) and vertical (depth) paths",  # why
                    "testing-pattern",  # category
                ]
            )
            + "\n"
        )
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "4"}  # quality score passes
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            result = runner.invoke(
                main,
                ["--db", str(db_path), "capture", "positive"],
                input=user_input,
            )
        assert result.exit_code == 0, result.output
        assert "Captured" in result.output

    def test_capture_positive_aborts_on_low_quality(self, db_path):
        from click.testing import CliRunner

        from lessons_db.cli import main

        runner = CliRunner()
        # Trailing \n ensures Click's CliRunner correctly terminates the last prompt
        user_input = "\n".join(["vague", "unclear", "testing-pattern"]) + "\n"
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "2"}  # score below threshold
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            result = runner.invoke(
                main,
                ["--db", str(db_path), "capture", "positive"],
                input=user_input,
            )
        assert result.exit_code == 0, result.output
        assert "aborted" in result.output.lower() or "failed" in result.output.lower()


class TestCaptureDesignDocCLI:
    """CLI capture design-doc command queues drafts from a markdown file."""

    def test_capture_design_doc_cli_success(self, db_path, tmp_path):
        from click.testing import CliRunner

        from lessons_db.cli import main

        doc = tmp_path / "test-design.md"
        doc.write_text("## Decision\nUsed dual-axis testing. Works well in integration scenarios.")

        runner = CliRunner()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": json.dumps(
                {
                    "entries": [
                        {
                            "one_liner": "Dual-axis testing finds integration bugs",
                            "why": "Tests both axes",
                            "category": "testing-pattern",
                        }
                    ]
                }
            )
        }
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            result = runner.invoke(main, ["--db", str(db_path), "capture", "design-doc", str(doc)])

        assert result.exit_code == 0, result.output
        assert "Queued" in result.output
        assert "draft" in result.output.lower()

    def test_capture_design_doc_cli_ollama_unavailable(self, db_path, tmp_path):
        """Exits 0 even when Ollama is unavailable (non-blocking)."""
        from click.testing import CliRunner

        from lessons_db.cli import main

        doc = tmp_path / "test-design.md"
        doc.write_text("## Decision\nSome content here.")

        runner = CliRunner()
        with patch("lessons_db.capture.requests.post", side_effect=Exception("connection refused")):
            result = runner.invoke(main, ["--db", str(db_path), "capture", "design-doc", str(doc)])

        assert result.exit_code == 0, result.output
        assert "No positive" in result.output


# ---------------------------------------------------------------------------
# detect_wins() tests
# ---------------------------------------------------------------------------


def _insert_lesson(conn, polarity="negative"):
    """Helper: insert a minimal lesson and return its id."""
    cursor = conn.execute(
        "INSERT INTO lessons (title, one_liner, tier, created_date, polarity) VALUES (?, ?, 'observation', ?, ?)",
        [f"Test lesson ({polarity})", "one-liner", date.today().isoformat(), polarity],
    )
    conn.commit()
    return cursor.lastrowid


def _insert_surfacing(conn, lesson_id, outcome, hours_ago=1):
    """Helper: insert a surfacing event with a timestamp hours_ago in the past."""
    ts = (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat()
    conn.execute(
        "INSERT INTO surfacing_events (lesson_id, hook_point, context, outcome, timestamp) "
        "VALUES (?, 'stop', 'test-ctx', ?, ?)",
        [lesson_id, outcome, ts],
    )
    conn.commit()


class TestDetectWins:
    """Win detection from session surfacing events."""

    def test_returns_empty_when_no_events(self, db_path):
        conn = init_db(db_path)
        wins = detect_wins(conn, lookback_hours=4)
        assert wins == []

    def test_all_heeded_win(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)
        _insert_surfacing(conn, lid, "heeded", hours_ago=2)

        wins = detect_wins(conn, lookback_hours=4)
        types = [w["win_type"] for w in wins]
        assert "all_heeded" in types

    def test_no_anti_pattern_hits_win(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)

        wins = detect_wins(conn, lookback_hours=4)
        types = [w["win_type"] for w in wins]
        assert "no_anti_pattern_hits" in types

    def test_dismissed_prevents_clean_session(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)
        _insert_surfacing(conn, lid, "dismissed", hours_ago=2)

        wins = detect_wins(conn, lookback_hours=4)
        types = [w["win_type"] for w in wins]
        assert "no_anti_pattern_hits" not in types

    def test_recurrence_prevents_clean_session(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)
        _insert_surfacing(conn, lid, "recurrence", hours_ago=2)

        wins = detect_wins(conn, lookback_hours=4)
        types = [w["win_type"] for w in wins]
        assert "no_anti_pattern_hits" not in types

    def test_low_heed_rate_no_all_heeded_win(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        # 1 heeded, 3 dismissed = 25% heed rate (below 70% threshold)
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)
        _insert_surfacing(conn, lid, "dismissed", hours_ago=1)
        _insert_surfacing(conn, lid, "dismissed", hours_ago=2)
        _insert_surfacing(conn, lid, "dismissed", hours_ago=2)

        wins = detect_wins(conn, lookback_hours=4)
        types = [w["win_type"] for w in wins]
        assert "all_heeded" not in types

    def test_positive_pattern_reused_win(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, polarity="positive")
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)

        wins = detect_wins(conn, lookback_hours=4)
        types = [w["win_type"] for w in wins]
        assert "positive_pattern_reused" in types
        # Verify lesson_ids are present
        reuse_win = next(w for w in wins if w["win_type"] == "positive_pattern_reused")
        assert lid in reuse_win["lesson_ids"]

    def test_negative_heeded_no_positive_reuse(self, db_path):
        """A heeded negative lesson should not trigger positive_pattern_reused."""
        conn = init_db(db_path)
        lid = _insert_lesson(conn, polarity="negative")
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)

        wins = detect_wins(conn, lookback_hours=4)
        types = [w["win_type"] for w in wins]
        assert "positive_pattern_reused" not in types

    def test_events_outside_lookback_ignored(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        # Event 10 hours ago, lookback is 4 hours
        _insert_surfacing(conn, lid, "heeded", hours_ago=10)

        wins = detect_wins(conn, lookback_hours=4)
        assert wins == []

    def test_win_detail_contains_counts(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)
        _insert_surfacing(conn, lid, "heeded", hours_ago=2)

        wins = detect_wins(conn, lookback_hours=4)
        heeded_win = next(w for w in wins if w["win_type"] == "all_heeded")
        assert "2/2" in heeded_win["detail"]

    def test_unknown_outcomes_count_toward_total(self, db_path):
        """Unknown outcomes dilute the heed rate but don't count as anti-pattern hits."""
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        _insert_surfacing(conn, lid, "heeded", hours_ago=1)
        _insert_surfacing(conn, lid, "unknown", hours_ago=2)
        _insert_surfacing(conn, lid, "unknown", hours_ago=2)
        _insert_surfacing(conn, lid, "unknown", hours_ago=3)

        wins = detect_wins(conn, lookback_hours=4)
        types = [w["win_type"] for w in wins]
        # 1/4 = 25% heed rate — below threshold
        assert "all_heeded" not in types
        # But no dismissed/recurrence, so clean session still applies
        assert "no_anti_pattern_hits" in types
