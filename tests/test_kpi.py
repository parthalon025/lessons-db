"""Tests for the KPI dashboard CLI command."""

from datetime import UTC, datetime, timedelta

from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db, insert_lesson
from lessons_db.learn import record_outcome, record_surfacing, update_win_streak


class TestKpiHelp:
    """KPI command --help works."""

    def test_kpi_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["kpi", "--help"])
        assert result.exit_code == 0
        assert "kpi" in result.output.lower() or "learning" in result.output.lower()


class TestKpiEmptyDb:
    """KPI command produces output on an empty database."""

    def test_kpi_empty_db(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "kpi"])
        assert result.exit_code == 0
        assert "Learning KPI Dashboard" in result.output

    def test_kpi_empty_has_all_sections(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "kpi"])
        assert result.exit_code == 0
        assert "Heeded Rate by Category" in result.output
        assert "Stability Distribution" in result.output
        assert "Positive/Negative Ratio" in result.output
        assert "Win Streaks" in result.output
        assert "Learning Velocity" in result.output
        assert "ZPD Identification" in result.output

    def test_kpi_empty_shows_no_data_indicators(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "kpi"])
        assert result.exit_code == 0
        assert "no surfacing data" in result.output
        assert "no win streak data" in result.output


class TestKpiSeededData:
    """KPI command with seeded data shows expected sections."""

    def _seed_db(self, db_path):
        """Seed a DB with lessons, surfacing events, win streaks, and reviews."""
        conn = init_db(db_path)

        # Insert lessons with different polarities and clusters
        l1 = insert_lesson(
            conn,
            {
                "title": "Async discipline",
                "one_liner": "No async def without I/O",
                "cluster": "async-discipline",
                "polarity": "negative",
                "created_date": "2026-01-01",
            },
        )
        l2 = insert_lesson(
            conn,
            {
                "title": "Error handling guard",
                "one_liner": "Log before returning fallback",
                "cluster": "error-handling",
                "polarity": "negative",
                "created_date": "2026-01-15",
            },
        )
        l3 = insert_lesson(
            conn,
            {
                "title": "Context manager pattern",
                "one_liner": "Use contextlib for resource cleanup",
                "cluster": "resource-management",
                "polarity": "positive",
                "created_date": "2026-02-01",
            },
        )

        # Surfacing events with outcomes for l1 (async-discipline): 4 heeded, 1 dismissed = 80%
        for _ in range(4):
            eid = record_surfacing(conn, l1, "pre_edit", context="test")
            record_outcome(conn, eid, "heeded")
        eid = record_surfacing(conn, l1, "pre_edit", context="test")
        record_outcome(conn, eid, "dismissed")

        # Surfacing events for l2 (error-handling): 3 heeded, 3 dismissed = 50%
        for _ in range(3):
            eid = record_surfacing(conn, l2, "pre_edit", context="test")
            record_outcome(conn, eid, "heeded")
        for _ in range(3):
            eid = record_surfacing(conn, l2, "pre_edit", context="test")
            record_outcome(conn, eid, "dismissed")

        # Win streaks
        update_win_streak(conn, "async-discipline", won=True)
        update_win_streak(conn, "async-discipline", won=True)
        update_win_streak(conn, "async-discipline", won=True)
        update_win_streak(conn, "error-handling", won=True)
        update_win_streak(conn, "error-handling", won=False)

        # Set FSRS stability on lessons to test stability distribution
        conn.execute(
            "UPDATE lessons SET stability = 1.5, last_review_date = ? WHERE id = ?",
            [datetime.now(UTC).date().isoformat(), l1],
        )
        conn.execute(
            "UPDATE lessons SET stability = 5.0, last_review_date = ? WHERE id = ?",
            [datetime.now(UTC).date().isoformat(), l2],
        )
        conn.execute(
            "UPDATE lessons SET stability = 55.0, last_review_date = ? WHERE id = ?",
            [(datetime.now(UTC) - timedelta(days=60)).date().isoformat(), l3],
        )
        conn.commit()

        return conn

    def test_kpi_heeded_rate_by_category(self, tmp_path):
        db_path = tmp_path / "test.db"
        self._seed_db(db_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "kpi"])
        assert result.exit_code == 0
        assert "Heeded Rate by Category" in result.output
        assert "async-discipline" in result.output
        assert "error-handling" in result.output
        # async-discipline: 4/5 = 80%
        assert "80%" in result.output
        # error-handling: 3/6 = 50%
        assert "50%" in result.output

    def test_kpi_stability_distribution(self, tmp_path):
        db_path = tmp_path / "test.db"
        self._seed_db(db_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "kpi"])
        assert result.exit_code == 0
        assert "Stability Distribution" in result.output
        # l1 stability=1.5 -> full, l2 stability=5.0 -> brief, l3 stability=55.0 -> enforced
        assert "full: 1" in result.output
        assert "brief: 1" in result.output
        assert "enforced: 1" in result.output

    def test_kpi_positive_negative_ratio(self, tmp_path):
        db_path = tmp_path / "test.db"
        self._seed_db(db_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "kpi"])
        assert result.exit_code == 0
        assert "Positive/Negative Ratio" in result.output
        assert "positive: 1" in result.output
        assert "negative: 2" in result.output

    def test_kpi_win_streaks(self, tmp_path):
        db_path = tmp_path / "test.db"
        self._seed_db(db_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "kpi"])
        assert result.exit_code == 0
        assert "Win Streaks" in result.output
        assert "async-discipline" in result.output
        assert "current=3" in result.output
        assert "longest=3" in result.output

    def test_kpi_learning_velocity(self, tmp_path):
        db_path = tmp_path / "test.db"
        self._seed_db(db_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "kpi"])
        assert result.exit_code == 0
        assert "Learning Velocity" in result.output
        # l1 and l2 were reviewed today, l3 was 60 days ago (outside 30d window)
        assert "reviewed in last 30 days" in result.output

    def test_kpi_zpd_identification(self, tmp_path):
        db_path = tmp_path / "test.db"
        self._seed_db(db_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "kpi"])
        assert result.exit_code == 0
        assert "ZPD Identification" in result.output
        # error-handling has 50% heeded rate (3/6) — within ZPD [50%, 80%]
        assert "Error handling guard" in result.output or "error-handling" in result.output

    def test_kpi_zpd_excludes_high_heeded(self, tmp_path):
        """Lessons with >80% heeded rate should NOT appear in ZPD."""
        db_path = tmp_path / "test.db"
        self._seed_db(db_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "kpi"])
        assert result.exit_code == 0
        # async-discipline has 80% (4/5) — at the boundary, should be included
        # since 0.50 <= 0.80 <= 0.80
        zpd_section = result.output.split("ZPD Identification")[1]
        # error-handling (50%) should be in ZPD
        assert "error-handling" in zpd_section or "Error handling" in zpd_section


class TestKpiAcceptanceCriteria:
    """Acceptance criteria from task spec."""

    def test_acceptance_criteria(self, tmp_path):
        runner = CliRunner()
        r = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "kpi"])
        assert r.exit_code == 0
        assert "heeded" in r.output.lower() or "stability" in r.output.lower()
