"""Tests for multi-strategy search module."""

import pytest

from lessons_db.db import init_db, insert_lesson
from lessons_db.search import (
    search_by_content,
    search_combined,
    search_for_file,
    search_text_fallback,
)


@pytest.fixture
def populated_db(db_path):
    """DB with one lesson, an affected file, and a detection pattern."""
    conn = init_db(db_path)
    lid = insert_lesson(
        conn,
        {
            "title": "bare-except swallowing",
            "one_liner": "Never use bare except without logging",
            "cluster": "A",
            "severity": 5,
            "enforcement": "semgrep_error",
        },
    )
    # Affected file
    conn.execute(
        "INSERT INTO affected_files (lesson_id, file_path, project) VALUES (?, ?, ?)",
        (lid, "src/hub/core.py", "ha-aria"),
    )
    # Detection pattern
    conn.execute(
        "INSERT INTO detection_patterns (lesson_id, pattern_type, regex, description, language) VALUES (?, ?, ?, ?, ?)",
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
        neg_id = insert_lesson(
            conn,
            {
                "title": "negative lesson",
                "one_liner": "never swallow exceptions",
                "created_date": "2026-01-01",
                "polarity": "negative",
            },
        )
        pos_id = insert_lesson(
            conn,
            {
                "title": "positive pattern",
                "one_liner": "dual-axis testing catches integration bugs",
                "created_date": "2026-01-01",
                "polarity": "positive",
            },
        )
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
        neg_id = insert_lesson(
            conn,
            {
                "title": "negative lesson",
                "one_liner": "never swallow exceptions",
                "created_date": "2026-01-01",
                "polarity": "negative",
            },
        )
        pos_id = insert_lesson(
            conn,
            {
                "title": "positive pattern",
                "one_liner": "dual-axis testing catches integration bugs",
                "created_date": "2026-01-01",
                "polarity": "positive",
            },
        )
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
        neg_id = insert_lesson(
            conn,
            {
                "title": "negative lesson",
                "one_liner": "X",
                "created_date": "2026-01-01",
                "polarity": "negative",
            },
        )
        pos_id = insert_lesson(
            conn,
            {
                "title": "positive pattern",
                "one_liner": "Y",
                "created_date": "2026-01-01",
                "polarity": "positive",
            },
        )
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


class TestTextFallback:
    """SQLite LIKE fallback when LanceDB is unavailable."""

    def test_search_text_fallback_matches(self, populated_db):
        """Search for a word in one_liner — should find it."""
        results = search_text_fallback(populated_db, "bare")
        assert len(results) == 1
        assert results[0]["one_liner"] == "Never use bare except without logging"

    def test_search_text_fallback_no_match(self, populated_db):
        """Search for nonsense — should return empty."""
        results = search_text_fallback(populated_db, "xyzzyplugh")
        assert results == []

    def test_search_text_fallback_multi_term(self, db_path):
        """Multi-term search: only match lessons containing all terms."""
        conn = init_db(db_path)
        id1 = insert_lesson(
            conn,
            {
                "title": "subscriber lifecycle",
                "one_liner": "Store callback ref on self and unsubscribe in shutdown",
                "severity": 4,
            },
        )
        id2 = insert_lesson(
            conn,
            {
                "title": "async discipline",
                "one_liner": "No async def without IO",
                "severity": 3,
            },
        )
        conn.commit()

        # Both terms must match
        results = search_text_fallback(conn, "callback shutdown")
        ids = [r["id"] for r in results]
        assert id1 in ids
        assert id2 not in ids
        conn.close()

    def test_search_combined_falls_back_without_lance(self, db_path):
        """search_combined with lance_db=None and a text query uses text fallback."""
        conn = init_db(db_path)
        insert_lesson(
            conn,
            {
                "title": "schema changes",
                "one_liner": "Schema changes must update all consumers in same PR",
                "severity": 5,
            },
        )
        conn.commit()

        # No file_path, no content — only query. lance_db=None triggers fallback.
        results = search_combined(conn, lance_db=None, query="schema consumers")
        assert len(results) >= 1
        assert any("schema" in r["one_liner"].lower() for r in results)
        conn.close()
