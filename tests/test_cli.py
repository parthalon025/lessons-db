"""Tests for CLI commands using Click CliRunner."""

from pathlib import Path

from click.testing import CliRunner

from lessons_db.cli import main


def test_main_help():
    """Main group help shows lessons-learned description."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "lessons-learned" in result.output


def test_status_command(tmp_path):
    """Status command runs against a fresh DB."""
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "status"])
    assert result.exit_code == 0
    assert "lessons" in result.output.lower()


def test_search_command(tmp_path):
    """Search command runs (returns 0 results on empty DB)."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["--db", str(tmp_path / "test.db"), "search", "subscriber lifecycle"]
    )
    assert result.exit_code == 0


def test_migrate_dry_run(tmp_path):
    """Migrate --dry-run lists found lesson files."""
    lesson_dir = tmp_path / "lessons"
    lesson_dir.mkdir()
    lesson_file = lesson_dir / "2026-01-01-test-lesson.md"
    lesson_file.write_text(
        "# Lesson #999: Test lesson title\n\n"
        "**Tier:** observation\n"
        "**Category:** testing\n"
        "**Cluster:** A\n\n"
        "## Observation\nSomething happened.\n"
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--db", str(tmp_path / "test.db"),
            "migrate",
            "--source", str(lesson_dir),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "1" in result.output
