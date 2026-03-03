"""Tests for eval-generate and eval-judge CLI commands."""

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db, insert_lesson


def _seed_eval_db(conn):
    """Create a minimal test DB with 2 clusters for eval testing."""
    for i, cat in enumerate(["integration", "testing", "monitoring"]):
        insert_lesson(
            conn,
            {
                "title": f"Cluster A lesson {i}",
                "one_liner": f"A one-liner {i}",
                "description": f"A description {i}",
                "cluster_seed": "A",
                "category": cat,
            },
        )
    for i, cat in enumerate(["data-model", "deployment", "integration"]):
        insert_lesson(
            conn,
            {
                "title": f"Cluster B lesson {i}",
                "one_liner": f"B one-liner {i}",
                "description": f"B description {i}",
                "cluster_seed": "B",
                "category": cat,
            },
        )


class TestEvalGenerateHelp:
    """eval-generate --help must work."""

    def test_help_exits_zero(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "eval-generate", "--help"])
        assert result.exit_code == 0
        assert "eval" in result.output.lower()

    def test_lists_in_meta_help(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "--help"])
        assert "eval-generate" in result.output


class TestEvalGenerateCommand:
    """eval-generate runs generation and produces results JSON."""

    def test_produces_results_file(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_eval_db(conn)
        conn.close()

        output_file = tmp_path / "results.json"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "Test principle generated."}).encode("utf-8")
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
                    "eval-generate",
                    "--variants",
                    "A",
                    "--per-cluster",
                    "1",
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert len(data["results"]) > 0

    def test_no_clusters_reports_empty(self, db_path, tmp_path):
        init_db(db_path)
        output_file = tmp_path / "results.json"

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--db",
                str(db_path),
                "meta",
                "eval-generate",
                "--variants",
                "A",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert "No source lessons" in result.output

    def test_resume_flag(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_eval_db(conn)
        conn.close()

        output_file = tmp_path / "results.json"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "Principle."}).encode("utf-8")
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
                    "eval-generate",
                    "--variants",
                    "A",
                    "--per-cluster",
                    "1",
                    "--output",
                    str(output_file),
                    "--resume",
                ],
            )
        assert result.exit_code == 0
