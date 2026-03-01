"""Tests for SQLite schema and CRUD operations."""

from datetime import date, timedelta

from lessons_db.db import (
    get_lesson,
    get_near_miss_hotspots,
    get_open_findings,
    get_overdue_actions,
    init_db,
    insert_corrective_action,
    insert_detection_pattern,
    insert_lesson,
    insert_near_miss,
    insert_scan_finding,
    search_by_enforcement,
    search_by_file,
    update_lesson,
)

EXPECTED_TABLES = [
    "lessons",
    "corrective_actions",
    "affected_files",
    "enforcement_rules",
    "detection_patterns",
    "near_misses",
    "scan_findings",
]

EXPECTED_INDEXES = [
    "idx_affected_files_path",
    "idx_affected_files_project",
    "idx_lessons_enforcement",
    "idx_corrective_status",
    "idx_near_misses_lesson",
    "idx_detection_patterns_type",
    "idx_scan_findings_status",
]


class TestSchemaCreation:
    """Schema initialisation tests."""

    def test_init_creates_all_tables(self, db_path):
        conn = init_db(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row["name"] for row in cursor.fetchall()]
        for t in EXPECTED_TABLES:
            assert t in tables, f"Missing table: {t}"
        conn.close()

    def test_init_creates_indexes(self, db_path):
        conn = init_db(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
        indexes = [row["name"] for row in cursor.fetchall()]
        for idx in EXPECTED_INDEXES:
            assert idx in indexes, f"Missing index: {idx}"
        conn.close()

    def test_wal_mode_enabled(self, db_path):
        conn = init_db(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_init_is_idempotent(self, db_path):
        conn1 = init_db(db_path)
        conn1.close()
        conn2 = init_db(db_path)
        cursor = conn2.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row["name"] for row in cursor.fetchall()]
        for t in EXPECTED_TABLES:
            assert t in tables
        conn2.close()

    def test_add_extension_columns_idempotent(self, db_path):
        """_add_extension_columns must not raise on a second call (ALTER TABLE duplicate column guard)."""
        from lessons_db.db import _add_extension_columns

        conn = init_db(db_path)
        # init_db already called _add_extension_columns once; a second call must be a no-op
        _add_extension_columns(conn)  # must not raise OperationalError
        # Verify v5 mining_runs columns are present and correct
        cols = {row["name"] for row in conn.execute("PRAGMA table_info('mining_runs')").fetchall()}
        for expected in (
            "candidates_extracted",
            "diff_size_rejected",
            "gate2_rejected",
            "gate3_rejected",
            "gate4_rejected",
            "duration_seconds",
        ):
            assert expected in cols, f"missing column {expected!r} after double init"
        conn.close()


class TestLessonCRUD:
    """Lesson insert, get, update, and search."""

    def test_insert_and_get_lesson(self, db_path):
        conn = init_db(db_path)
        data = {
            "title": "bare-except swallowing",
            "one_liner": "Never use bare except without logging",
            "description": "Bare excepts hide real errors.",
            "cluster": "A",
            "category": "testing",
            "scope": "language:python",
            "keywords": "exception,logging",
            "source": "manual",
        }
        lesson_id = insert_lesson(conn, data)
        assert isinstance(lesson_id, int)
        assert lesson_id > 0

        lesson = get_lesson(conn, lesson_id)
        assert lesson is not None
        assert lesson["title"] == "bare-except swallowing"
        assert lesson["one_liner"] == "Never use bare except without logging"
        assert lesson["tier"] == "observation"  # default
        assert lesson["severity"] == 3  # default
        assert lesson["confidence"] == "emerging"  # default
        assert lesson["enforcement"] == "documentation"  # default
        assert lesson["recurrence_count"] == 0  # default
        assert lesson["created_date"] == date.today().isoformat()

        # Non-existent
        assert get_lesson(conn, 9999) is None
        conn.close()

    def test_update_enforcement(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "test lesson",
                "one_liner": "a test",
            },
        )
        update_lesson(conn, lid, {"enforcement": "semgrep_warning", "severity": 5})
        lesson = get_lesson(conn, lid)
        assert lesson["enforcement"] == "semgrep_warning"
        assert lesson["severity"] == 5
        conn.close()

    def test_search_by_file(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "subscriber lifecycle",
                "one_liner": "Store callback ref on self",
                "cluster": "B",
                "enforcement": "semgrep_error",
                "severity": 4,
            },
        )
        # Insert an affected file
        conn.execute(
            "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
            (lid, "src/aria/hub/presence.py", "ha-aria"),
        )
        conn.commit()

        results = search_by_file(conn, "presence.py")
        assert len(results) >= 1
        hit = results[0]
        assert hit["id"] == lid
        assert hit["one_liner"] == "Store callback ref on self"
        assert hit["cluster"] == "B"
        assert hit["enforcement"] == "semgrep_error"
        assert hit["severity"] == 4

        # No match
        assert search_by_file(conn, "nonexistent.py") == []
        conn.close()

    def test_search_by_enforcement(self, db_path):
        conn = init_db(db_path)
        insert_lesson(
            conn,
            {
                "title": "doc only",
                "one_liner": "low severity",
                "enforcement": "documentation",
            },
        )
        insert_lesson(
            conn,
            {
                "title": "warning level",
                "one_liner": "medium severity",
                "enforcement": "semgrep_warning",
            },
        )
        insert_lesson(
            conn,
            {
                "title": "also warning",
                "one_liner": "another medium",
                "enforcement": "semgrep_warning",
            },
        )

        results = search_by_enforcement(conn, "semgrep_warning")
        assert len(results) == 2
        titles = {r["title"] for r in results}
        assert titles == {"warning level", "also warning"}
        conn.close()


class TestCorrectiveActions:
    """Corrective action insert and overdue query."""

    def test_insert_and_get_overdue(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "test lesson",
                "one_liner": "for corrective",
            },
        )

        # Overdue action (due yesterday, no explicit due_date — auto +7 days won't
        # be overdue, so set explicitly)
        past = (date.today() - timedelta(days=1)).isoformat()
        insert_corrective_action(
            conn,
            {
                "lesson_id": lid,
                "action": "Add logging to exception handler",
                "status": "proposed",
                "due_date": past,
            },
        )

        # Not overdue (future date)
        future = (date.today() + timedelta(days=30)).isoformat()
        insert_corrective_action(
            conn,
            {
                "lesson_id": lid,
                "action": "Write semgrep rule",
                "status": "in_progress",
                "due_date": future,
            },
        )

        # Completed (should not appear even if overdue)
        insert_corrective_action(
            conn,
            {
                "lesson_id": lid,
                "action": "Deploy rule",
                "status": "completed",
                "due_date": past,
            },
        )

        overdue = get_overdue_actions(conn)
        assert len(overdue) == 1
        assert overdue[0]["action"] == "Add logging to exception handler"

        # Test auto due_date default (+7 days)
        aid = insert_corrective_action(
            conn,
            {
                "lesson_id": lid,
                "action": "Auto-dated action",
                "status": "proposed",
            },
        )
        row = conn.execute("SELECT due_date FROM corrective_actions WHERE id = ?", (aid,)).fetchone()
        expected_due = (date.today() + timedelta(days=7)).isoformat()
        assert row["due_date"] == expected_due
        conn.close()


class TestNearMisses:
    """Near miss insert and hotspot aggregation."""

    def test_insert_and_hotspots(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "bare except",
                "one_liner": "no bare except",
            },
        )

        # 3 hits on file A, 1 hit on file B
        for _ in range(3):
            insert_near_miss(
                conn,
                {
                    "lesson_id": lid,
                    "file_path": "src/engine/core.py",
                    "event_type": "block",
                    "rule_id": "bare-except",
                },
            )
        insert_near_miss(
            conn,
            {
                "lesson_id": lid,
                "file_path": "src/hub/main.py",
                "event_type": "warn",
                "rule_id": "bare-except",
            },
        )

        hotspots = get_near_miss_hotspots(conn, limit=10)
        assert len(hotspots) == 2
        assert hotspots[0]["file_path"] == "src/engine/core.py"
        assert hotspots[0]["count"] == 3
        assert hotspots[1]["file_path"] == "src/hub/main.py"
        assert hotspots[1]["count"] == 1
        conn.close()


class TestScanFindings:
    """Scan finding insert and open query."""

    def test_insert_and_get_open(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "async discipline",
                "one_liner": "no async without IO",
            },
        )

        insert_scan_finding(
            conn,
            {
                "lesson_id": lid,
                "rule_id": "async-no-io",
                "file_path": "src/capture.py",
                "line_number": 42,
                "snippet": "async def pure_func():",
                "status": "open",
            },
        )
        insert_scan_finding(
            conn,
            {
                "lesson_id": lid,
                "rule_id": "async-no-io",
                "file_path": "src/export.py",
                "line_number": 10,
                "snippet": "async def format():",
                "status": "resolved",
            },
        )

        open_findings = get_open_findings(conn)
        assert len(open_findings) == 1
        assert open_findings[0]["file_path"] == "src/capture.py"
        assert open_findings[0]["line_number"] == 42
        assert open_findings[0]["title"] == "async discipline"
        assert open_findings[0]["one_liner"] == "no async without IO"
        conn.close()

    def test_get_open_findings_includes_null_lesson_id(self, db_path):
        """get_open_findings must return findings with NULL lesson_id (scanner-discovered, not yet linked)."""
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO scan_findings (lesson_id, scan_date, rule_id, file_path, line_number, snippet) "
            "VALUES (NULL, ?, 'S105', 'src/foo.py', 10, 'secret = \"x\"')",
            (date.today().isoformat(),),
        )
        conn.commit()
        findings = get_open_findings(conn)
        assert len(findings) == 1
        assert findings[0]["rule_id"] == "S105"
        assert findings[0]["lesson_id"] is None
        conn.close()


class TestSchemaExtension:
    """Tests for v2 schema extension columns and tables."""

    def test_lessons_has_entry_type_column(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "entry_type" in cols

    def test_lessons_has_polarity_column(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "polarity" in cols

    def test_lessons_has_cluster_seed_column(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "cluster_seed" in cols

    def test_lessons_has_reuse_count_column(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "reuse_count" in cols

    def test_surfacing_events_table_exists(self, db_path):
        conn = init_db(db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "surfacing_events" in tables

    def test_templates_table_exists(self, db_path):
        conn = init_db(db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "templates" in tables

    def test_insert_lesson_with_polarity(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Test positive entry",
                "one_liner": "Dual-axis testing catches integration bugs",
                "created_date": "2026-02-26",
                "polarity": "positive",
                "entry_type": "pattern",
            },
        )
        row = get_lesson(conn, lid)
        assert row["polarity"] == "positive"
        assert row["entry_type"] == "pattern"

    def test_reuse_count_defaults_to_zero(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "T",
                "one_liner": "X",
                "created_date": "2026-02-26",
            },
        )
        row = get_lesson(conn, lid)
        assert row["reuse_count"] == 0

    def test_capture_drafts_table_exists(self, db_path):
        conn = init_db(db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "capture_drafts" in tables

    def test_cluster_runs_table_exists(self, db_path):
        conn = init_db(db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "cluster_runs" in tables


class TestInsertDetectionPattern:
    def test_inserts_pattern_for_lesson(self, db_path):
        conn = init_db(db_path)
        lesson_id = insert_lesson(
            conn, {"title": "bare-except", "one_liner": "Never swallow exceptions", "tier": "lesson"}
        )

        pattern_id = insert_detection_pattern(
            conn,
            {
                "lesson_id": lesson_id,
                "pattern_type": "regex",
                "regex": r"except\s*:",
                "description": "Bare except clause — always log before swallowing",
                "language": "python",
            },
        )

        row = conn.execute("SELECT * FROM detection_patterns WHERE id = ?", [pattern_id]).fetchone()
        assert row["lesson_id"] == lesson_id
        assert row["regex"] == r"except\s*:"
        assert row["language"] == "python"

    def test_requires_lesson_id(self, db_path):
        conn = init_db(db_path)
        import sqlite3

        import pytest

        with pytest.raises(sqlite3.IntegrityError):
            insert_detection_pattern(
                conn,
                {
                    "pattern_type": "regex",
                    "regex": r"except\s*:",
                },
            )

    def test_requires_pattern_type(self, db_path):
        conn = init_db(db_path)
        lesson_id = insert_lesson(conn, {"title": "test-lesson", "one_liner": "test", "tier": "lesson"})
        import sqlite3

        import pytest

        with pytest.raises(sqlite3.IntegrityError):
            insert_detection_pattern(
                conn,
                {
                    "lesson_id": lesson_id,
                    "regex": r"except\s*:",
                },
            )

    def test_defaults_language_to_any(self, db_path):
        conn = init_db(db_path)
        lesson_id = insert_lesson(conn, {"title": "test-lesson", "one_liner": "test", "tier": "lesson"})
        pattern_id = insert_detection_pattern(
            conn,
            {
                "lesson_id": lesson_id,
                "pattern_type": "regex",
                "regex": r"except\s*:",
            },
        )
        row = conn.execute("SELECT language FROM detection_patterns WHERE id = ?", [pattern_id]).fetchone()
        assert row["language"] == "any"


from lessons_db.db import get_mined_repo, insert_mined_repo, insert_mining_run, update_mined_repo


def test_insert_mined_repo(db_path):
    conn = init_db(db_path)
    repo_id = insert_mined_repo(conn, "owner/repo", topics=["asyncio", "python"])
    assert repo_id > 0
    repo = get_mined_repo(conn, "owner/repo")
    assert repo["repo_full_name"] == "owner/repo"
    assert repo["commit_count"] == 0
    assert repo["lessons_extracted"] == 0


def test_insert_mining_run(db_path):
    conn = init_db(db_path)
    run_id = insert_mining_run(
        conn,
        repos_searched=5,
        commits_analyzed=200,
        gate0_rejected=80,
        gate1_rejected=20,
        auto_approved=3,
        conflicts_flagged=1,
        error_count=2,
        duration_seconds=45.2,
    )
    assert run_id > 0
    row = conn.execute("SELECT * FROM mining_runs WHERE id=?", (run_id,)).fetchone()
    assert row["repos_searched"] == 5
    assert row["error_count"] == 2


def test_mined_repo_idempotent(db_path):
    conn = init_db(db_path)
    id1 = insert_mined_repo(conn, "owner/repo")
    id2 = insert_mined_repo(conn, "owner/repo")  # should upsert, not duplicate
    assert id1 == id2
    count = conn.execute("SELECT COUNT(*) FROM mined_repos WHERE repo_full_name='owner/repo'").fetchone()[0]
    assert count == 1


def test_update_mined_repo(db_path):
    conn = init_db(db_path)
    insert_mined_repo(conn, "owner/repo")
    # First update
    update_mined_repo(conn, "owner/repo", commit_count=10, lessons_extracted=2)
    repo = get_mined_repo(conn, "owner/repo")
    assert repo["commit_count"] == 10
    assert repo["lessons_extracted"] == 2
    assert repo["last_mined_date"] is not None
    # Second update — verify cumulative increment
    update_mined_repo(conn, "owner/repo", commit_count=5, lessons_extracted=1)
    repo = get_mined_repo(conn, "owner/repo")
    assert repo["commit_count"] == 15  # 10 + 5
    assert repo["lessons_extracted"] == 3  # 2 + 1


def test_scan_findings_lesson_id_nullable(db_path):
    """scan_findings.lesson_id must accept NULL (scanner findings not yet linked to a lesson)."""
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO scan_findings (lesson_id, scan_date, rule_id, file_path, line_number, snippet, status) "
        "VALUES (NULL, '2026-01-01', 'S105', 'src/foo.py', 42, 'password = \"x\"', 'open')"
    )
    conn.commit()
    row = conn.execute("SELECT lesson_id FROM scan_findings WHERE rule_id='S105'").fetchone()
    assert row["lesson_id"] is None


def test_scan_findings_lesson_id_nullable_migration(tmp_path):
    """Migration must fix a legacy NOT NULL lesson_id constraint on an existing DB."""
    import sqlite3 as _sqlite3

    db_file = tmp_path / "legacy.db"

    # Simulate the old schema with NOT NULL lesson_id
    legacy = _sqlite3.connect(str(db_file))
    legacy.row_factory = _sqlite3.Row
    legacy.execute("PRAGMA journal_mode=WAL")
    legacy.executescript("""
        CREATE TABLE lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            tier TEXT NOT NULL DEFAULT 'observation',
            severity INTEGER NOT NULL DEFAULT 3,
            confidence TEXT NOT NULL DEFAULT 'emerging',
            enforcement TEXT NOT NULL DEFAULT 'documentation',
            recurrence_count INTEGER NOT NULL DEFAULT 0,
            created_date TEXT NOT NULL
        );
        CREATE TABLE scan_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lesson_id INTEGER NOT NULL,
            rule_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_number INTEGER,
            snippet TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            scan_date TEXT NOT NULL
        );
    """)
    legacy.commit()
    legacy.close()

    # init_db must detect and fix the NOT NULL constraint
    conn = init_db(db_file)
    rows = conn.execute("PRAGMA table_info('scan_findings')").fetchall()
    col = next((r for r in rows if r["name"] == "lesson_id"), None)
    assert col is not None
    assert col["notnull"] == 0, "lesson_id should be nullable after migration"

    # Confirm NULL insert works post-migration
    conn.execute(
        "INSERT INTO scan_findings (lesson_id, scan_date, rule_id, file_path, status) "
        "VALUES (NULL, '2026-01-01', 'S105', 'src/bar.py', 'open')"
    )
    conn.commit()
    result = conn.execute("SELECT lesson_id FROM scan_findings WHERE rule_id='S105'").fetchone()
    assert result["lesson_id"] is None
