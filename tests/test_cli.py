"""Tests for CLI commands using Click CliRunner."""

from unittest.mock import MagicMock, patch

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
    result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "search", "subscriber lifecycle"])
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
            "--db",
            str(tmp_path / "test.db"),
            "migrate",
            "--source",
            str(lesson_dir),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "1" in result.output


def test_migrate_idempotent(tmp_path):
    """Running migrate twice does not create duplicate lessons."""
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

    db_path = tmp_path / "test.db"
    runner = CliRunner()
    migrate_args = [
        "--db",
        str(db_path),
        "migrate",
        "--source",
        str(lesson_dir),
    ]

    # First run — should insert
    result1 = runner.invoke(main, migrate_args)
    assert result1.exit_code == 0
    assert "Migrated: 1" in result1.output

    # Second run — should skip, not duplicate
    result2 = runner.invoke(main, migrate_args)
    assert result2.exit_code == 0
    assert "Migrated: 0" in result2.output
    assert "Skipped: 1" in result2.output

    # Verify only 1 row in DB
    from lessons_db.db import init_db

    conn = init_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    assert count == 1


def test_migrate_adds_new_files_on_rerun(tmp_path):
    """Re-running migrate picks up new files without duplicating old ones."""
    lesson_dir = tmp_path / "lessons"
    lesson_dir.mkdir()
    first = lesson_dir / "2026-01-01-first-lesson.md"
    first.write_text("# Lesson: First lesson\n\n" "**Tier:** observation\n\n" "## Observation\nFirst.\n")

    db_path = tmp_path / "test.db"
    runner = CliRunner()
    migrate_args = [
        "--db",
        str(db_path),
        "migrate",
        "--source",
        str(lesson_dir),
    ]

    # First run
    runner.invoke(main, migrate_args)

    # Add a second file
    second = lesson_dir / "2026-01-02-second-lesson.md"
    second.write_text("# Lesson: Second lesson\n\n" "**Tier:** insight\n\n" "## Observation\nSecond.\n")

    # Second run — should add 1, skip 1
    result = runner.invoke(main, migrate_args)
    assert result.exit_code == 0
    assert "Migrated: 1" in result.output
    assert "Skipped: 1" in result.output

    from lessons_db.db import init_db

    conn = init_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    assert count == 2


def test_index_seed_only_backfills_cluster_seed(tmp_path):
    """index --seed-only copies cluster → cluster_seed for existing lessons."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(
        conn,
        {
            "title": "Log before returning fallback",
            "one_liner": "Never swallow exceptions silently",
            "cluster": "A",
            "tier": "lesson_learned",
            "created_date": "2026-01-01",
        },
    )

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
    insert_lesson(
        conn,
        {
            "title": "Already seeded lesson",
            "cluster": "B",
            "cluster_seed": "existing",
            "tier": "observation",
            "created_date": "2026-01-01",
        },
    )

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
    insert_lesson(
        conn,
        {
            "title": "Test lesson",
            "one_liner": "A lesson summary",
            "keywords": "testing, fixtures",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )

    runner = CliRunner()
    with patch("lessons_db.cli.LANCE_DIR", lance_dir):
        result = runner.invoke(main, ["--db", str(db_path), "index"])

    assert result.exit_code == 0
    assert "Indexed: 1" in result.output
    assert "Failed: 0" in result.output


def test_rule_generate_no_patterns(tmp_path):
    """rule generate exits cleanly when lesson has no detection patterns."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(
        conn,
        {
            "title": "Log before fallback",
            "one_liner": "Never swallow",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "rule", "generate", "1"])
    assert result.exit_code == 0
    assert "no detection patterns" in result.output.lower()


def test_rule_generate_with_patterns(tmp_path):
    """rule generate writes YAML to rules_dir when patterns exist."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    rules_dir = tmp_path / "rules"
    conn = init_db(db_path)
    lid = insert_lesson(
        conn,
        {
            "title": "Bare except swallows failures",
            "one_liner": "Never use bare except",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    conn.execute(
        "INSERT INTO detection_patterns (lesson_id, pattern_type, regex, description, language) VALUES (?,?,?,?,?)",
        [lid, "regex", r"except\s*:", "bare except", "python"],
    )
    conn.commit()
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--db", str(db_path), "rule", "generate", str(lid), "--rules-dir", str(rules_dir)],
    )
    assert result.exit_code == 0
    assert "generated" in result.output.lower()
    yaml_files = list(rules_dir.glob("**/*.yaml"))
    assert len(yaml_files) == 1


def test_rule_test_no_rules(tmp_path):
    """rule test exits cleanly when no rules exist."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--db", str(tmp_path / "test.db"), "rule", "test", "--rules-dir", str(tmp_path / "rules")],
    )
    assert result.exit_code == 0
    assert "no rules" in result.output.lower()


@patch("lessons_db.scan.subprocess.run")
def test_scan_command_runs(mock_run, tmp_path):
    """scan command parses SARIF findings and saves them to DB."""
    import json

    from lessons_db.db import init_db, insert_lesson

    rules_dir = tmp_path / "rules" / "python"
    rules_dir.mkdir(parents=True)
    (rules_dir / "bare-except-001.yaml").write_text("rules: []")

    sarif = {
        "version": "2.1.0",
        "runs": [
            {
                "results": [
                    {
                        "ruleId": "lessons-db.python.bare-except-001",
                        "message": {"text": "bare except"},
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "src/foo.py"},
                                    "region": {"startLine": 42},
                                }
                            }
                        ],
                    }
                ]
            }
        ],
    }
    mock_run.return_value = MagicMock(
        returncode=1,  # semgrep exits 1 when findings exist
        stdout=json.dumps(sarif),
        stderr="",
    )

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(
        conn,
        {
            "title": "Bare except swallows errors",
            "one_liner": "Never use bare except",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--db", str(db_path), "scan", "--rules-dir", str(tmp_path / "rules"), "--target", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "1 saved to DB" in result.output

    # Verify the finding was actually written
    row = conn.execute("SELECT * FROM scan_findings WHERE lesson_id = 1").fetchone()
    assert row is not None
    assert row["rule_id"] == "lessons-db.python.bare-except-001"


def test_scan_command_no_rules(tmp_path):
    """scan exits cleanly when rules dir is empty."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--db",
            str(tmp_path / "test.db"),
            "scan",
            "--rules-dir",
            str(tmp_path / "empty-rules"),
            "--target",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "no rules" in result.output.lower()


def test_export_command(tmp_path):
    """export outputs lesson markdown."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(
        conn,
        {
            "title": "Log every external failure",
            "one_liner": "Never swallow exceptions silently",
            "cluster": "A",
            "tier": "lesson_learned",
            "created_date": "2026-01-01",
        },
    )
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "export", str(lid)])
    assert result.exit_code == 0
    assert "Log every external failure" in result.output
    assert "Key Takeaway" in result.output


def test_export_missing_lesson(tmp_path):
    """export exits cleanly for unknown lesson ID."""
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "export", "999"])
    assert result.exit_code == 0
    assert "not found" in result.output.lower()


def test_summary_command(tmp_path):
    """summary writes SUMMARY.md to the output path."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(
        conn,
        {
            "title": "Log every failure",
            "one_liner": "Always log",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    out_file = tmp_path / "SUMMARY.md"
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "summary", "--output", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text()
    assert "Always log" in content


class TestCaptureReview:
    def test_review_dry_run_exits_cleanly(self, tmp_path):
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES ('raw', '{\"one_liner\": \"Never swallow exceptions without logging\"}', "
            "'pending', '2026-02-27', 'auto_transcript')"
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "capture", "review", "--dry-run"])

        assert result.exit_code == 0


class TestStatsEfficiency:
    """stats efficiency subcommand."""

    def test_stats_efficiency_shows_wasted_surfacings(self, tmp_path):
        """stats efficiency lists lessons surfaced but never heeded."""
        from lessons_db.db import init_db, insert_lesson
        from lessons_db.learn import record_surfacing

        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Wasted lesson",
                "one_liner": "Bare except swallows failures",
                "created_date": "2026-01-01",
            },
        )
        # 5 surfacings, 0 heeded
        for _ in range(5):
            record_surfacing(conn, lid, "read", "hub.py")
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "stats", "efficiency"])
        assert result.exit_code == 0, result.output
        assert "Bare except swallows failures" in result.output
        assert "Wasted" in result.output or "wasted" in result.output

    def test_stats_efficiency_shows_enforcement_candidates(self, tmp_path):
        """stats efficiency lists high-recurrence low-heed-rate lessons."""
        from lessons_db.db import init_db, insert_lesson
        from lessons_db.learn import record_outcome, record_surfacing

        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "High recurrence low heed",
                "one_liner": "SQLite closing context manager",
                "created_date": "2026-01-01",
                "recurrence_count": 12,
            },
        )
        # 4 surfacings, 1 heeded → heed_rate = 0.25
        for i in range(4):
            eid = record_surfacing(conn, lid, "read", "db.py")
            record_outcome(conn, eid, "heeded" if i == 0 else "dismissed")
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "stats", "efficiency"])
        assert result.exit_code == 0, result.output
        assert "SQLite closing context manager" in result.output

    def test_stats_efficiency_shows_average_outcome_rate(self, tmp_path):
        """stats efficiency prints average outcome rate across all lessons with surfacings."""
        from lessons_db.db import init_db, insert_lesson
        from lessons_db.learn import record_outcome, record_surfacing

        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Some lesson",
                "one_liner": "Do something right",
                "created_date": "2026-01-01",
            },
        )
        e1 = record_surfacing(conn, lid, "read", "x.py")
        record_outcome(conn, e1, "heeded")
        e2 = record_surfacing(conn, lid, "read", "y.py")
        record_outcome(conn, e2, "dismissed")
        conn.close()

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "stats", "efficiency"])
        assert result.exit_code == 0, result.output
        assert "outcome rate" in result.output.lower() or "Average" in result.output

    def test_stats_efficiency_no_data(self, tmp_path):
        """stats efficiency handles empty DB gracefully."""
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "empty.db"), "stats", "efficiency"])
        assert result.exit_code == 0, result.output
        # Should not crash, should show empty state or zeros


def test_learn_record_with_outcome_heeded(tmp_path):
    """learn record --outcome heeded records heeded surfacing event."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(
        conn,
        {
            "title": "Test",
            "one_liner": "t",
            "cluster": "A",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--db",
            str(db_path),
            "learn",
            "record",
            "--lesson-id",
            "1",
            "--hook",
            "plan",
            "--context",
            "ctx",
            "--outcome",
            "heeded",
        ],
    )
    assert result.exit_code == 0, result.output
    row = conn.execute("SELECT outcome FROM surfacing_events WHERE lesson_id = 1").fetchone()
    assert row["outcome"] == "heeded"


def test_learn_record_with_outcome_false_positive(tmp_path):
    """learn record --outcome false_positive records dismissal."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(
        conn,
        {
            "title": "Test2",
            "one_liner": "t",
            "cluster": "B",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--db",
            str(db_path),
            "learn",
            "record",
            "--lesson-id",
            "1",
            "--hook",
            "edit",
            "--context",
            "ctx",
            "--outcome",
            "false_positive",
        ],
    )
    assert result.exit_code == 0, result.output
    row = conn.execute("SELECT outcome FROM surfacing_events WHERE lesson_id = 1").fetchone()
    assert row["outcome"] == "false_positive"


def test_learn_record_without_outcome_defaults_to_unknown(tmp_path):
    """learn record without --outcome stores outcome='unknown'."""
    from lessons_db.db import init_db, insert_lesson

    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(
        conn,
        {
            "title": "Test3",
            "one_liner": "t",
            "cluster": "C",
            "tier": "lesson",
            "created_date": "2026-01-01",
        },
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--db",
            str(db_path),
            "learn",
            "record",
            "--lesson-id",
            "1",
            "--hook",
            "plan",
            "--context",
            "ctx",
        ],
    )
    assert result.exit_code == 0, result.output
    row = conn.execute("SELECT outcome FROM surfacing_events WHERE lesson_id = 1").fetchone()
    assert row["outcome"] == "unknown"
