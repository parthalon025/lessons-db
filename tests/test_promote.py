"""Tests for positive entry promotion ladder."""

import pytest
from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import get_lesson, init_db, insert_lesson
from lessons_db.promote import apply_template, list_templates, record_reuse


@pytest.fixture
def conn_with_positive(db_path):
    conn = init_db(db_path)
    lid = insert_lesson(
        conn,
        {
            "title": "Dual-axis pipeline testing",
            "one_liner": "Dual-axis testing catches integration bugs missed by unit tests",
            "polarity": "positive",
            "entry_type": "pattern",
            "tier": "noticed",
            "category": "testing-pattern",
            "created_date": "2026-02-26",
        },
    )
    return conn, lid


class TestRecordReuse:
    def test_first_reuse_promotes_to_tested(self, conn_with_positive):
        conn, lid = conn_with_positive
        new_tier = record_reuse(conn, lid)
        assert new_tier == "tested"
        lesson = get_lesson(conn, lid)
        assert lesson["reuse_count"] == 1
        assert lesson["tier"] == "tested"

    def test_second_reuse_promotes_to_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)  # → tested
        new_tier = record_reuse(conn, lid)  # → proven
        assert new_tier == "proven"
        lesson = get_lesson(conn, lid)
        assert lesson["tier"] == "proven"

    def test_second_reuse_generates_template(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        templates = list_templates(conn)
        assert len(templates) == 1
        assert templates[0]["lesson_id"] == lid

    def test_third_reuse_promotes_to_standard(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        new_tier = record_reuse(conn, lid)
        assert new_tier == "standard"

    def test_increments_reuse_count(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        lesson = get_lesson(conn, lid)
        assert lesson["reuse_count"] == 3

    def test_beyond_standard_stays_standard(self, conn_with_positive):
        """4th+ reuse: tier stays standard, no new template generated."""
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        record_reuse(conn, lid)  # → standard
        tier = record_reuse(conn, lid)  # 4th
        assert tier == "standard"
        lesson = get_lesson(conn, lid)
        assert lesson["reuse_count"] == 4
        assert lesson["tier"] == "standard"
        # Only one template row — idempotent guard worked
        assert len(list_templates(conn)) == 1


class TestListTemplates:
    def test_empty_initially(self, db_path):
        conn = init_db(db_path)
        assert list_templates(conn) == []

    def test_returns_template_after_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        templates = list_templates(conn)
        assert len(templates) == 1
        assert "one_liner" in templates[0]
        assert templates[0]["tier"] == "proven"


class TestApplyTemplate:
    def test_returns_none_before_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        assert apply_template(conn, lid) is None

    def test_returns_content_after_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        content = apply_template(conn, lid)
        assert content is not None
        assert "Dual-axis" in content

    def test_returns_most_recent_when_multiple_rows(self, conn_with_positive):
        """apply_template returns most recent (highest id) if duplicates exist."""
        conn, lid = conn_with_positive
        # Manually insert two template rows to simulate the duplicate scenario
        from datetime import date

        conn.execute(
            "INSERT INTO templates (lesson_id, template_type, content, created_date) "
            "VALUES (?, 'approach', 'first content', ?)",
            [lid, date.today().isoformat()],
        )
        conn.execute(
            "INSERT INTO templates (lesson_id, template_type, content, created_date) "
            "VALUES (?, 'approach', 'second content', ?)",
            [lid, date.today().isoformat()],
        )
        conn.commit()
        content = apply_template(conn, lid)
        assert content == "second content"


class TestReuseCLI:
    """Tests for the 'lessons-db reuse record' CLI command."""

    def test_reuse_record_help(self):
        """CLI wiring: 'reuse record --help' exits 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["reuse", "record", "--help"])
        assert result.exit_code == 0
        assert "LESSON_ID" in result.output

    def test_reuse_record_promotes_and_echoes_tier(self, db_path, conn_with_positive):
        """CLI records reuse, echoes new tier, and updates DB."""
        conn, lid = conn_with_positive
        conn.close()  # CLI opens its own connection

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "reuse", "record", str(lid)])
        assert result.exit_code == 0
        assert "tier: tested" in result.output
        assert f"lesson #{lid}" in result.output

    def test_reuse_record_shows_one_liner(self, db_path, conn_with_positive):
        """CLI echoes the lesson's one_liner after recording reuse."""
        conn, lid = conn_with_positive
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "reuse", "record", str(lid)])
        assert result.exit_code == 0
        assert "Dual-axis" in result.output

    def test_reuse_record_creates_surfacing_event(self, db_path, conn_with_positive):
        """CLI creates a surfacing event with outcome='heeded' for the learning pipeline."""
        conn, lid = conn_with_positive
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "reuse", "record", str(lid)])
        assert result.exit_code == 0

        # Reopen and verify surfacing event was created
        check_conn = init_db(db_path)
        events = check_conn.execute("SELECT * FROM surfacing_events WHERE lesson_id = ?", [lid]).fetchall()
        assert len(events) == 1
        assert events[0]["hook_point"] == "edit"
        assert events[0]["context"] == "positive_reuse"
        assert events[0]["outcome"] == "heeded"
        check_conn.close()

    def test_reuse_record_nonexistent_lesson(self, db_path):
        """CLI exits with error for a non-existent lesson ID."""
        conn = init_db(db_path)
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "reuse", "record", "99999"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_reuse_record_successive_promotions(self, db_path, conn_with_positive):
        """CLI correctly reports tier progression across successive calls."""
        conn, lid = conn_with_positive
        conn.close()

        runner = CliRunner()
        r1 = runner.invoke(main, ["--db", str(db_path), "reuse", "record", str(lid)])
        assert "tier: tested" in r1.output

        r2 = runner.invoke(main, ["--db", str(db_path), "reuse", "record", str(lid)])
        assert "tier: proven" in r2.output

        r3 = runner.invoke(main, ["--db", str(db_path), "reuse", "record", str(lid)])
        assert "tier: standard" in r3.output
