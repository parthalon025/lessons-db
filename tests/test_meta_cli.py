"""Tests for the meta CLI command group (extract-principles)."""

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import get_lesson, init_db, insert_lesson


class TestMetaExtractPrinciplesHelp:
    """meta extract-principles --help must work."""

    def test_help_exits_zero(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "extract-principles", "--help"])
        assert result.exit_code == 0
        assert "Extract domain-independent principles" in result.output

    def test_meta_group_help(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "--help"])
        assert result.exit_code == 0
        assert "extract-principles" in result.output


class TestMetaExtractPrinciplesDryRun:
    """Dry-run mode should preview without updating the database."""

    def test_dry_run_does_not_update_db(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Subscriber lifecycle management",
                "one_liner": "Store callback ref on self, unsubscribe in shutdown",
                "description": "When subscribing to external events, store the callback reference and explicitly unsubscribe during shutdown.",
            },
        )
        conn.close()

        mock_response = json.dumps(
            {
                "response": "Resources acquired in callbacks must be explicitly released",
            }
        ).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(
                main,
                [
                    "--db",
                    str(db_path),
                    "meta",
                    "extract-principles",
                    "--dry-run",
                    "--batch-size",
                    "5",
                ],
            )

        assert result.exit_code == 0
        assert "Resources acquired in callbacks must be explicitly released" in result.output

        # Verify DB was NOT updated
        conn = init_db(db_path)
        row = get_lesson(conn, lid)
        assert row["principle"] is None
        conn.close()


class TestMetaExtractPrinciplesUpdate:
    """Non-dry-run mode should update the database."""

    def test_updates_principle_in_db(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Bare except swallowing",
                "one_liner": "Never use bare except without logging",
                "description": "Bare excepts hide real errors and make debugging impossible.",
            },
        )
        conn.close()

        mock_response = json.dumps(
            {
                "response": "Errors must be observable before being suppressed",
            }
        ).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(
                main,
                [
                    "--db",
                    str(db_path),
                    "meta",
                    "extract-principles",
                    "--batch-size",
                    "5",
                ],
            )

        assert result.exit_code == 0
        assert "Errors must be observable before being suppressed" in result.output
        assert "Updated: 1" in result.output

        # Verify DB was updated
        conn = init_db(db_path)
        row = get_lesson(conn, lid)
        assert row["principle"] == "Errors must be observable before being suppressed"
        conn.close()

    def test_skips_lessons_with_existing_principle(self, db_path):
        conn = init_db(db_path)
        # Lesson with principle already set
        lid1 = insert_lesson(
            conn,
            {
                "title": "Has principle",
                "one_liner": "Already enriched",
            },
        )
        conn.execute("UPDATE lessons SET principle = ? WHERE id = ?", ("Existing principle", lid1))
        conn.commit()

        # Lesson without principle
        lid2 = insert_lesson(
            conn,
            {
                "title": "Needs principle",
                "one_liner": "Not yet enriched",
            },
        )
        conn.close()

        mock_response = json.dumps(
            {
                "response": "New principle extracted",
            }
        ).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = runner.invoke(
                main,
                [
                    "--db",
                    str(db_path),
                    "meta",
                    "extract-principles",
                    "--batch-size",
                    "10",
                ],
            )

        assert result.exit_code == 0
        # Only 1 lesson should be processed (the one without a principle)
        assert "Processing 1 lessons" in result.output
        assert "Updated: 1" in result.output

        # Verify: lid1 still has its original principle, lid2 has the new one
        conn = init_db(db_path)
        assert get_lesson(conn, lid1)["principle"] == "Existing principle"
        assert get_lesson(conn, lid2)["principle"] == "New principle extracted"
        conn.close()


class TestMetaExtractPrinciplesNoLessons:
    """When all lessons have principles, should report nothing to do."""

    def test_no_lessons_without_principles(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "All done",
                "one_liner": "Already has principle",
            },
        )
        conn.execute("UPDATE lessons SET principle = ? WHERE id = ?", ("Already set", lid))
        conn.commit()
        conn.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--db",
                str(db_path),
                "meta",
                "extract-principles",
            ],
        )

        assert result.exit_code == 0
        assert "No lessons without principles found" in result.output


class TestMetaExtractPrinciplesErrorHandling:
    """Ollama errors should be handled gracefully."""

    def test_ollama_connection_error(self, db_path):
        conn = init_db(db_path)
        insert_lesson(
            conn,
            {
                "title": "Test lesson",
                "one_liner": "Will fail to connect",
            },
        )
        conn.close()

        import urllib.error

        runner = CliRunner()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            result = runner.invoke(
                main,
                [
                    "--db",
                    str(db_path),
                    "meta",
                    "extract-principles",
                ],
            )

        assert result.exit_code == 0
        assert "ERROR" in result.output
        assert "Errors: 1" in result.output

    def test_model_flag_overrides_default(self, db_path):
        conn = init_db(db_path)
        insert_lesson(
            conn,
            {
                "title": "Test model flag",
                "one_liner": "Check model override",
            },
        )
        conn.close()

        mock_response = json.dumps(
            {
                "response": "Test principle",
            }
        ).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
            result = runner.invoke(
                main,
                [
                    "--db",
                    str(db_path),
                    "meta",
                    "extract-principles",
                    "--model",
                    "qwen3:8b",
                ],
            )

        assert result.exit_code == 0
        assert "model: qwen3:8b" in result.output

        # Verify the payload sent to Ollama used the correct model
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        sent_payload = json.loads(req.data.decode("utf-8"))
        assert sent_payload["model"] == "qwen3:8b"
