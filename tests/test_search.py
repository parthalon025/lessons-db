"""Tests for multi-strategy search module."""

import pytest

from lessons_db.db import init_db, insert_lesson
from lessons_db.search import (
    search_by_content,
    search_combined,
    search_for_file,
    search_semantic,
)


@pytest.fixture
def populated_db(db_path):
    """DB with one lesson, an affected file, and a detection pattern."""
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "bare-except swallowing",
        "one_liner": "Never use bare except without logging",
        "cluster": "A",
        "severity": 5,
        "enforcement": "semgrep_error",
    })
    # Affected file
    conn.execute(
        "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
        (lid, "src/hub/core.py", "ha-aria"),
    )
    # Detection pattern
    conn.execute(
        "INSERT INTO detection_patterns (lesson_id, pattern_type, regex, description, language) "
        "VALUES (?, ?, ?, ?, ?)",
        (lid, "syntactic", r"except.*:\s*pass", "Bare except with pass", "python"),
    )
    conn.commit()
    yield conn
    conn.close()


class TestFileSearch:
    """File path search strategy."""

    def test_finds_lesson_by_exact_path(self, populated_db):
        results = search_for_file(populated_db, "src/hub/core.py")
        assert len(results) == 1
        assert results[0]["one_liner"] == "Never use bare except without logging"

    def test_finds_lesson_by_partial_path(self, populated_db):
        results = search_for_file(populated_db, "hub/core.py")
        assert len(results) == 1
        assert results[0]["id"] > 0

    def test_returns_empty_for_unknown_file(self, populated_db):
        results = search_for_file(populated_db, "nonexistent/file.py")
        assert results == []


class TestContentSearch:
    """Content regex matching strategy."""

    def test_matches_syntactic_pattern(self, populated_db):
        code = """\
try:
    do_something()
except:  pass
"""
        results = search_by_content(populated_db, code, language="python")
        assert len(results) == 1
        assert results[0]["one_liner"] == "Never use bare except without logging"
        assert results[0]["matched_pattern"] == r"except.*:\s*pass"

    def test_no_match_for_safe_code(self, populated_db):
        code = """\
try:
    do_something()
except ValueError as e:
    logger.error("Failed: %s", e)
"""
        results = search_by_content(populated_db, code, language="python")
        assert results == []


class TestCombinedSearch:
    """Combined search deduplication."""

    def test_deduplicates_across_strategies(self, populated_db):
        code = "except:  pass"
        results = search_combined(
            populated_db,
            lance_db=None,
            file_path="src/hub/core.py",
            content=code,
            language="python",
        )
        # Both file and content match the same lesson — should appear once
        lesson_ids = [r["id"] for r in results]
        assert len(lesson_ids) == len(set(lesson_ids))
        assert len(results) == 1

    def test_polarity_filter_returns_only_positive(self, db_path):
        """search_combined with polarity='positive' excludes negative entries."""
        conn = init_db(db_path)
        neg_id = insert_lesson(conn, {
            "title": "negative lesson",
            "one_liner": "never swallow exceptions",
            "created_date": "2026-01-01",
            "polarity": "negative",
        })
        pos_id = insert_lesson(conn, {
            "title": "positive pattern",
            "one_liner": "dual-axis testing catches integration bugs",
            "created_date": "2026-01-01",
            "polarity": "positive",
        })
        # Use file path search to avoid needing LanceDB
        conn.execute(
            "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
            (neg_id, "src/hub.py", "ha-aria"),
        )
        conn.execute(
            "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
            (pos_id, "src/hub.py", "ha-aria"),
        )
        conn.commit()

        results = search_combined(conn, lance_db=None, file_path="src/hub.py", polarity="positive")
        ids = [r["id"] for r in results]
        assert pos_id in ids
        assert neg_id not in ids

    def test_polarity_filter_returns_only_negative(self, db_path):
        """search_combined with polarity='negative' excludes positive entries."""
        conn = init_db(db_path)
        neg_id = insert_lesson(conn, {
            "title": "negative lesson",
            "one_liner": "never swallow exceptions",
            "created_date": "2026-01-01",
            "polarity": "negative",
        })
        pos_id = insert_lesson(conn, {
            "title": "positive pattern",
            "one_liner": "dual-axis testing catches integration bugs",
            "created_date": "2026-01-01",
            "polarity": "positive",
        })
        conn.execute(
            "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
            (neg_id, "src/hub.py", "ha-aria"),
        )
        conn.execute(
            "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
            (pos_id, "src/hub.py", "ha-aria"),
        )
        conn.commit()

        results = search_combined(conn, lance_db=None, file_path="src/hub.py", polarity="negative")
        ids = [r["id"] for r in results]
        assert neg_id in ids
        assert pos_id not in ids

    def test_no_polarity_filter_returns_all(self, db_path):
        """search_combined without polarity returns both polarities."""
        conn = init_db(db_path)
        neg_id = insert_lesson(conn, {
            "title": "negative lesson", "one_liner": "X",
            "created_date": "2026-01-01", "polarity": "negative",
        })
        pos_id = insert_lesson(conn, {
            "title": "positive pattern", "one_liner": "Y",
            "created_date": "2026-01-01", "polarity": "positive",
        })
        conn.execute(
            "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
            (neg_id, "src/hub.py", "ha-aria"),
        )
        conn.execute(
            "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
            (pos_id, "src/hub.py", "ha-aria"),
        )
        conn.commit()

        results = search_combined(conn, lance_db=None, file_path="src/hub.py")
        ids = [r["id"] for r in results]
        assert neg_id in ids
        assert pos_id in ids
