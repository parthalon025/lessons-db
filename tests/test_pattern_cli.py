"""Tests for `lessons-db pattern` CLI commands."""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cli_db(db_path, monkeypatch):
    """Patch CLI to use test DB path."""
    monkeypatch.setenv("LESSONS_DB_PATH", str(db_path))
    init_db(db_path)
    return db_path


class TestPatternScan:
    def test_scan_invokes_pipeline(self, runner, cli_db):
        with (
            patch("lessons_db.cli.pattern_extract") as mock_extract,
            patch("lessons_db.cli.pattern_verify") as mock_verify,
            patch("lessons_db.cli.pattern_triage") as mock_triage,
        ):
            mock_extract.list_active_repos.return_value = []
            mock_extract.build_semgrep_patterns.return_value = []
            mock_extract.extract_python_candidates.return_value = []
            mock_extract.extract_nonpython_candidates.return_value = []

            result = runner.invoke(main, ["--db", str(cli_db), "pattern", "scan"])

        assert result.exit_code == 0

    def test_scan_writes_last_scan_timestamp(self, runner, cli_db):
        with (
            patch("lessons_db.cli.pattern_extract") as mock_extract,
            patch("lessons_db.cli.pattern_verify"),
            patch("lessons_db.cli.pattern_triage"),
        ):
            mock_extract.list_active_repos.return_value = []
            mock_extract.build_semgrep_patterns.return_value = []
            mock_extract.extract_python_candidates.return_value = []
            mock_extract.extract_nonpython_candidates.return_value = []

            runner.invoke(main, ["--db", str(cli_db), "pattern", "scan"])

        import sqlite3

        from lessons_db.db import get_scan_state

        conn = sqlite3.connect(str(cli_db))
        conn.row_factory = sqlite3.Row
        ts = get_scan_state(conn, "last_scan_timestamp")
        assert ts != "1970-01-01T00:00:00"

    def test_scan_verify_rejection_does_not_crash(self, runner, cli_db):
        """verify_candidate returns (None, gate_tag) on rejection — must not raise AttributeError."""
        from unittest.mock import MagicMock

        fake_candidate = MagicMock()

        with (
            patch("lessons_db.cli.pattern_extract") as mock_extract,
            patch("lessons_db.cli.pattern_verify") as mock_verify,
            patch("lessons_db.cli.pattern_triage") as mock_triage,
        ):
            mock_extract.list_active_repos.return_value = [MagicMock(name="repo")]
            mock_extract.build_semgrep_patterns.return_value = []
            mock_extract.extract_python_candidates.return_value = [fake_candidate]
            mock_extract.extract_nonpython_candidates.return_value = []
            # Simulate dedup rejection — returns tuple, not None
            mock_verify.verify_candidate.return_value = (None, "dedup")

            result = runner.invoke(main, ["--db", str(cli_db), "pattern", "scan"])

        assert result.exit_code == 0, result.output
        mock_triage.triage_candidate.assert_not_called()


class TestPatternReview:
    def test_review_shows_pending_drafts(self, runner, cli_db):
        import sqlite3

        conn = sqlite3.connect(str(cli_db))
        conn.execute(
            "INSERT INTO capture_drafts "
            "(raw_content, status, created_date, source, detection_source, confidence) "
            "VALUES ('test snippet', 'pending', '2026-02-26', "
            "        'test', 'cross_project_scan', 0.75)"
        )
        conn.commit()
        conn.close()

        result = runner.invoke(main, ["--db", str(cli_db), "pattern", "review"], input="s\n")
        assert result.exit_code == 0
        assert "test snippet" in result.output or "pending" in result.output

    def test_review_sorted_by_confidence_desc(self, runner, cli_db):
        import sqlite3

        conn = sqlite3.connect(str(cli_db))
        for conf in [0.70, 0.90, 0.80]:
            conn.execute(
                "INSERT INTO capture_drafts "
                "(raw_content, status, created_date, source, detection_source, confidence) "
                f"VALUES ('snippet-{conf}', 'pending', '2026-02-26', "
                f"        'test', 'cross_project_scan', {conf})"
            )
        conn.commit()
        conn.close()

        result = runner.invoke(main, ["--db", str(cli_db), "pattern", "review"], input="s\ns\ns\n")
        # Highest confidence shown first
        assert "0.90" in result.output, f"Expected 0.90 in output: {result.output}"
        assert "0.80" in result.output, f"Expected 0.80 in output: {result.output}"
        assert result.output.index("0.90") < result.output.index("0.80"), (
            "Expected 0.90 before 0.80 (sorted by confidence DESC)"
        )


class TestPatternStatus:
    def test_status_shows_counts(self, runner, cli_db):
        result = runner.invoke(main, ["--db", str(cli_db), "pattern", "status"])
        assert result.exit_code == 0
        assert "threshold" in result.output.lower() or "0.85" in result.output


class TestPatternCalibrate:
    def test_calibrate_shows_bands(self, runner, cli_db):
        result = runner.invoke(main, ["--db", str(cli_db), "pattern", "calibrate"])
        assert result.exit_code == 0

    def test_calibrate_apply_requires_sufficient_data(self, runner, cli_db):
        result = runner.invoke(main, ["--db", str(cli_db), "pattern", "calibrate", "--apply"])
        assert result.exit_code == 0
        assert "insufficient" in result.output.lower() or "data" in result.output.lower()
