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

    def test_invalid_variant_exits_with_error(self, db_path, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--db",
                str(db_path),
                "meta",
                "eval-generate",
                "--variants",
                "Z",
                "--output",
                str(tmp_path / "results.json"),
            ],
        )
        assert result.exit_code != 0
        assert "Unknown variant" in result.output

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


# ---------------------------------------------------------------------------
# eval-judge CLI
# ---------------------------------------------------------------------------


class TestEvalJudgeHelp:
    """eval-judge --help must work."""

    def test_help_exits_zero(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "eval-judge", "--help"])
        assert result.exit_code == 0
        assert "judge" in result.output.lower() or "score" in result.output.lower()

    def test_lists_in_meta_help(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "--help"])
        assert "eval-judge" in result.output


class TestEvalJudgeCommand:
    """eval-judge reads results and produces a report."""

    def test_produces_report(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_eval_db(conn)
        conn.close()

        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [1]},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": 1,
                    "lesson_title": "Cluster A lesson 0",
                    "cluster_seed": "A",
                    "principle": "Silent fallbacks mask upstream failures.",
                    "model": "test-model",
                    "prompt_id": "baseline-fewshot",
                    "settings": {},
                    "generation_time_s": 1.0,
                    "error": None,
                }
            ],
        }
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(results_data))
        report_file = tmp_path / "report.md"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": '{"transfer": 4, "precision": 3, "actionability": 5}'}
        ).encode("utf-8")
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
                    "eval-judge",
                    str(results_file),
                    "--output",
                    str(report_file),
                ],
            )

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert report_file.exists()
        report_text = report_file.read_text()
        assert "Variant" in report_text

    def test_missing_results_file(self, db_path, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--db",
                str(db_path),
                "meta",
                "eval-judge",
                str(tmp_path / "nonexistent.json"),
                "--output",
                str(tmp_path / "report.md"),
            ],
        )
        assert result.exit_code != 0 or "not found" in result.output.lower() or "Error" in result.output
