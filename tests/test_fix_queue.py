"""Tests for fix queue and GitHub issue creation."""

from unittest.mock import patch

from lessons_db.db import (
    add_to_fix_queue,
    get_fix_queue,
    get_next_fix,
    init_db,
    insert_lesson,
    update_fix_status,
)
from lessons_db.prevention import create_github_issues, populate_fix_queue

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_lesson(conn, enforcement="semgrep_warning", severity=4, recurrence_count=2):
    return insert_lesson(
        conn,
        {
            "title": "Bare except swallows exceptions",
            "one_liner": "Do not use bare except",
            "enforcement": enforcement,
            "recurrence_count": recurrence_count,
            "severity": severity,
            "corrective_action": "Replace with `except Exception as e: _log.warning(...)`",
        },
    )


def _insert_finding(conn, lesson_id, file_path="src/foo.py", line_number=42):
    """Insert a scan finding directly."""
    from datetime import date

    cursor = conn.execute(
        "INSERT INTO scan_findings (lesson_id, rule_id, file_path, line_number, snippet, status, scan_date) "
        "VALUES (?, ?, ?, ?, ?, 'open', ?)",
        (lesson_id, "test-rule", file_path, line_number, "except:", date.today().isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# add_to_fix_queue
# ---------------------------------------------------------------------------


class TestAddToFixQueue:
    """add_to_fix_queue — CRUD and deduplication."""

    def test_adds_entry_and_returns_id(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        row_id = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", line_number=42)

        assert isinstance(row_id, int)
        assert row_id > 0

    def test_duplicate_entry_returns_none(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", line_number=42)
        second = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", line_number=42)

        assert second is None

    def test_different_line_number_not_duplicate(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        id1 = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", line_number=10)
        id2 = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", line_number=20)

        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    def test_different_file_not_duplicate(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        id1 = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", line_number=42)
        id2 = add_to_fix_queue(conn, lesson_id=lid, file_path="src/bar.py", line_number=42)

        assert id1 is not None and id2 is not None

    def test_null_line_number_deduplicates(self, db_path):
        """Two inserts with same lesson_id + file_path and line_number=None must deduplicate.

        The UNIQUE INDEX uses COALESCE(line_number, -1) so NULL values are treated
        as equal rather than always distinct (SQL NULL != NULL semantics).
        """
        conn = init_db(db_path)
        lid = _insert_lesson(conn)

        id1 = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", line_number=None)
        id2 = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", line_number=None)

        assert id1 is not None
        assert id2 is None  # duplicate — unique index treats NULL as -1


# ---------------------------------------------------------------------------
# get_next_fix
# ---------------------------------------------------------------------------


class TestGetNextFix:
    """get_next_fix — priority ordering and queue management."""

    def test_returns_none_on_empty_queue(self, db_path):
        conn = init_db(db_path)
        assert get_next_fix(conn) is None

    def test_returns_pending_fix(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")

        fix = get_next_fix(conn)

        assert fix is not None
        assert fix["lesson_id"] == lid
        assert fix["status"] == "pending"

    def test_prioritizes_higher_severity(self, db_path):
        conn = init_db(db_path)
        lid_low = _insert_lesson(conn, severity=2)
        lid_high = _insert_lesson(conn, severity=5)
        add_to_fix_queue(conn, lesson_id=lid_low, file_path="src/a.py")
        add_to_fix_queue(conn, lesson_id=lid_high, file_path="src/b.py")

        fix = get_next_fix(conn)

        assert fix["lesson_id"] == lid_high

    def test_skips_non_pending_entries(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        fid = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")
        update_fix_status(conn, fid, "applied")

        fix = get_next_fix(conn)

        assert fix is None

    def test_includes_suggested_fix_from_lesson(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        add_to_fix_queue(
            conn,
            lesson_id=lid,
            file_path="src/foo.py",
            suggested_fix="Replace with except Exception as e:",
        )

        fix = get_next_fix(conn)

        assert fix["suggested_fix"] == "Replace with except Exception as e:"


# ---------------------------------------------------------------------------
# get_fix_queue
# ---------------------------------------------------------------------------


class TestGetFixQueue:
    """get_fix_queue — filtering by status."""

    def test_returns_only_pending_by_default(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        fid = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")
        add_to_fix_queue(conn, lesson_id=lid, file_path="src/bar.py")
        update_fix_status(conn, fid, "applied")

        items = get_fix_queue(conn, status="pending")

        assert len(items) == 1
        assert items[0]["file_path"] == "src/bar.py"

    def test_returns_applied_when_requested(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        fid = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")
        update_fix_status(conn, fid, "applied")

        items = get_fix_queue(conn, status="applied")

        assert len(items) == 1


# ---------------------------------------------------------------------------
# update_fix_status
# ---------------------------------------------------------------------------


class TestUpdateFixStatus:
    """update_fix_status — status transitions and URL storage."""

    def test_updates_status(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        fid = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")

        update_fix_status(conn, fid, "applied")

        row = conn.execute("SELECT status FROM fix_queue WHERE id = ?", (fid,)).fetchone()
        assert row["status"] == "applied"

    def test_stores_github_issue_url(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn)
        fid = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")

        update_fix_status(conn, fid, "issue_created", github_issue_url="https://github.com/org/repo/issues/1")

        row = conn.execute("SELECT github_issue_url FROM fix_queue WHERE id = ?", (fid,)).fetchone()
        assert row["github_issue_url"] == "https://github.com/org/repo/issues/1"


# ---------------------------------------------------------------------------
# populate_fix_queue
# ---------------------------------------------------------------------------


class TestPopulateFixQueue:
    """populate_fix_queue — builds queue from open scan findings."""

    def test_adds_high_severity_findings(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=4)
        _insert_finding(conn, lid)

        result = populate_fix_queue(conn, min_severity=3)

        assert result["added"] == 1
        items = get_fix_queue(conn)
        assert len(items) == 1

    def test_skips_below_min_severity(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=2)  # below threshold
        _insert_finding(conn, lid)

        result = populate_fix_queue(conn, min_severity=3)

        assert result["added"] == 0

    def test_deduplicates_on_second_call(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=4)
        _insert_finding(conn, lid)

        populate_fix_queue(conn)
        result2 = populate_fix_queue(conn)  # second call

        assert result2["added"] == 0
        assert result2["skipped_duplicate"] >= 1

    def test_copies_corrective_action_as_suggested_fix(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=4)
        _insert_finding(conn, lid)

        populate_fix_queue(conn)

        fix = get_next_fix(conn)
        assert fix["suggested_fix"] is not None
        assert "except" in fix["suggested_fix"].lower()

    def test_skips_findings_with_no_lesson(self, db_path):
        conn = init_db(db_path)
        # Insert finding without a valid lesson_id
        from datetime import date

        conn.execute(
            "INSERT INTO scan_findings (lesson_id, rule_id, file_path, status, scan_date) "
            "VALUES (NULL, 'orphan-rule', 'src/x.py', 'open', ?)",
            (date.today().isoformat(),),
        )
        conn.commit()

        result = populate_fix_queue(conn)

        assert result["skipped_no_lesson"] == 1


# ---------------------------------------------------------------------------
# create_github_issues
# ---------------------------------------------------------------------------


class TestCreateGithubIssues:
    """create_github_issues — calls gh, stores URL, deduplicates."""

    def test_dry_run_does_not_call_gh(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=5)
        add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", suggested_fix="Fix this")

        with patch("lessons_db.prevention._gh_create_issue") as mock_gh:
            result = create_github_issues(conn, dry_run=True, min_severity=4)

        mock_gh.assert_not_called()
        assert result["created"] == 1

    def test_creates_issue_for_pending_high_severity(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=5)
        fid = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py", suggested_fix="Fix this")

        with patch("lessons_db.prevention._gh_create_issue", return_value="https://github.com/org/repo/issues/99"):
            result = create_github_issues(conn, min_severity=4)

        assert result["created"] == 1
        row = conn.execute("SELECT github_issue_url, status FROM fix_queue WHERE id = ?", (fid,)).fetchone()
        assert row["github_issue_url"] == "https://github.com/org/repo/issues/99"
        assert row["status"] == "issue_created"

    def test_skips_pending_entries_with_existing_url(self, db_path):
        """skipped_existing: pending entry already has a URL (partial prior run)."""
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=5)
        fid = add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")
        # Simulate partial failure: URL written but status not yet updated
        conn.execute(
            "UPDATE fix_queue SET github_issue_url = ? WHERE id = ?",
            ("https://github.com/org/repo/issues/1", fid),
        )
        conn.commit()

        with patch("lessons_db.prevention._gh_create_issue") as mock_gh:
            result = create_github_issues(conn, min_severity=4)

        mock_gh.assert_not_called()
        assert result["skipped_existing"] == 1

    def test_skips_below_min_severity(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=2)  # below threshold
        add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")

        with patch("lessons_db.prevention._gh_create_issue") as mock_gh:
            result = create_github_issues(conn, min_severity=4)

        mock_gh.assert_not_called()
        assert result["skipped_severity"] == 1

    def test_gh_failure_increments_error_count(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, severity=5)
        add_to_fix_queue(conn, lesson_id=lid, file_path="src/foo.py")

        with patch("lessons_db.prevention._gh_create_issue", side_effect=RuntimeError("gh failed")):
            result = create_github_issues(conn, min_severity=4)

        assert result["errors"] == 1
        assert result["created"] == 0
