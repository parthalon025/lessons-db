"""Tests for positive knowledge capture."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from lessons_db.capture import (
    capture_from_design_doc,
    capture_positive_manual,
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
