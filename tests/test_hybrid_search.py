"""Tests for lessons-db hybrid-search subcommand."""

import json

import pytest
from click.testing import CliRunner

from lessons_db.cli import _bm25_search, _rrf_merge, main
from lessons_db.db import init_db, insert_lesson

# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestRrfMerge:
    """RRF merge correctness."""

    def test_bm25_only_returns_sorted_scores(self):
        """Without semantic input, scores are 1/(k+rank) per BM25 position."""
        bm25 = [(10, 5.0), (20, 3.0), (30, 1.0)]
        merged = _rrf_merge(bm25, semantic_ranked=None, k=60)

        ids = [m[0] for m in merged]
        assert ids == [10, 20, 30]

        # First-ranked entry: 1/(60+1) ≈ 0.016393
        assert abs(merged[0][1] - 1 / 61) < 1e-9

    def test_deduplicates_by_id(self):
        """Same id in both lists must appear once with combined score."""
        bm25 = [(42, 9.0), (7, 1.0)]
        semantic = [(42, 8.0), (99, 0.5)]
        merged = _rrf_merge(bm25, semantic_ranked=semantic, k=60)

        ids = [m[0] for m in merged]
        assert ids.count(42) == 1

    def test_higher_ranked_id_scores_higher(self):
        """An id ranked #1 in both lists should outscore an id ranked #1 in only one."""
        bm25 = [(1, 10.0), (2, 5.0)]
        semantic = [(1, 9.0), (3, 4.0)]
        merged = dict(_rrf_merge(bm25, semantic_ranked=semantic, k=60))

        # id=1 appears in both → larger combined score than id=3 (only in semantic)
        assert merged[1] > merged[3]

    def test_empty_bm25_returns_empty(self):
        assert _rrf_merge([], semantic_ranked=None) == []

    def test_empty_semantic_treated_same_as_none(self):
        bm25 = [(5, 3.0)]
        assert _rrf_merge(bm25, semantic_ranked=[]) == _rrf_merge(bm25, semantic_ranked=None)


# ---------------------------------------------------------------------------
# Integration tests via Click test runner
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_conn(db_path):
    """In-memory DB with three lessons for BM25 smoke testing."""
    conn = init_db(db_path)
    insert_lesson(
        conn,
        {
            "title": "async discipline",
            "description": "Always await coroutines — missing await creates silent no-op.",
            "one_liner": "No async def without I/O",
        },
    )
    insert_lesson(
        conn,
        {
            "title": "subscriber lifecycle cleanup",
            "description": "Store callback ref on self; unsubscribe in shutdown().",
            "one_liner": "Unsubscribe in shutdown",
        },
    )
    insert_lesson(
        conn,
        {
            "title": "bare except swallowing",
            "description": "Log before returning fallback — never swallow exceptions silently.",
            "one_liner": "Log before fallback",
        },
    )
    conn.commit()
    return conn


class TestBm25Search:
    """BM25 search over the in-process DB."""

    def test_returns_results_for_matching_query(self, populated_conn):
        results = _bm25_search(populated_conn, "async await coroutine")
        assert len(results) > 0
        ids = [r[0] for r in results]
        # The async lesson should surface
        row = populated_conn.execute("SELECT id FROM lessons WHERE title = 'async discipline'").fetchone()
        assert row["id"] in ids

    def test_returns_empty_for_empty_db(self, db_path):
        empty_conn = init_db(db_path)
        results = _bm25_search(empty_conn, "anything")
        assert results == []
        empty_conn.close()

    def test_zero_score_entries_excluded(self, populated_conn):
        """Query with no matching tokens should return empty list."""
        results = _bm25_search(populated_conn, "xyzzy quux zzz")
        assert results == []


class TestHybridSearchCommand:
    """CLI-level tests via Click test runner."""

    def _run(self, db_path, args):
        runner = CliRunner()
        return runner.invoke(main, ["--db", str(db_path)] + args, catch_exceptions=False)

    def test_returns_results_for_known_query(self, db_path, populated_conn):
        result = self._run(db_path, ["hybrid-search", "async discipline", "--top", "3"])
        assert result.exit_code == 0
        assert "#" in result.output  # at least one id printed

    def test_top_flag_limits_output(self, db_path, populated_conn):
        result = self._run(db_path, ["hybrid-search", "async", "--top", "1"])
        assert result.exit_code == 0
        lines = [ln for ln in result.output.strip().splitlines() if ln.startswith("[")]
        assert len(lines) <= 1

    def test_json_flag_produces_valid_json(self, db_path, populated_conn):
        result = self._run(db_path, ["hybrid-search", "subscriber lifecycle", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        if data:
            assert "rank" in data[0]
            assert "id" in data[0]
            assert "score" in data[0]
            assert "title" in data[0]

    def test_no_results_prints_message(self, db_path, populated_conn):
        result = self._run(db_path, ["hybrid-search", "xyzzy quux zzz"])
        assert result.exit_code == 0
        assert "No results found." in result.output

    def test_no_results_json_prints_empty_array(self, db_path, populated_conn):
        result = self._run(db_path, ["hybrid-search", "xyzzy quux zzz", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    def test_rank_field_is_sequential(self, db_path, populated_conn):
        result = self._run(db_path, ["hybrid-search", "async", "--top", "3", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        ranks = [item["rank"] for item in data]
        assert ranks == list(range(1, len(ranks) + 1))
