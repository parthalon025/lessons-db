"""Tests for CLI commands using Click CliRunner."""

from pathlib import Path
from unittest.mock import patch

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


def test_index_seed_only_backfills_cluster_seed(tmp_path):
    """index --seed-only copies cluster → cluster_seed for existing lessons."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Log before returning fallback",
        "one_liner": "Never swallow exceptions silently",
        "cluster": "A",
        "tier": "lesson_learned",
        "created_date": "2026-01-01",
    })

    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "index", "--seed-only"])

    assert result.exit_code == 0
    assert "1" in result.output  # 1 row updated

    row = conn.execute("SELECT cluster_seed FROM lessons WHERE id = 1").fetchone()
    assert row["cluster_seed"] == "A"


def test_index_seed_only_skips_already_seeded(tmp_path):
    """index --seed-only does not overwrite cluster_seed that is already set."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Already seeded lesson",
        "cluster": "B",
        "cluster_seed": "existing",
        "tier": "observation",
        "created_date": "2026-01-01",
    })

    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "index", "--seed-only"])

    assert result.exit_code == 0
    assert "0" in result.output  # 0 rows updated

    row = conn.execute("SELECT cluster_seed FROM lessons WHERE id = 1").fetchone()
    assert row["cluster_seed"] == "existing"


@patch("lessons_db.vectors.get_embedding")
def test_index_generates_embeddings(mock_embed, tmp_path, lance_dir):
    """index command upserts embeddings for all lessons."""
    from lessons_db.db import init_db, insert_lesson

    mock_embed.return_value = [0.1] * 768

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Test lesson",
        "one_liner": "A lesson summary",
        "keywords": "testing, fixtures",
        "cluster": "A",
        "tier": "lesson",
        "created_date": "2026-01-01",
    })

    runner = CliRunner()
    with patch("lessons_db.cli.LANCE_DIR", lance_dir):
        result = runner.invoke(main, ["--db", str(db_path), "index"])

    assert result.exit_code == 0
    assert "Indexed: 1" in result.output
    assert "Failed: 0" in result.output
