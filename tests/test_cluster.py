"""Tests for adaptive HDBSCAN clustering pipeline."""

from unittest.mock import patch

import pytest

from lessons_db.cluster import (
    apply_cluster_proposals,
    discover_clusters,
    extract_representative_terms,
    find_seed_overlap,
    get_cluster_history,
)
from lessons_db.db import init_db, insert_lesson


@pytest.fixture
def conn_with_lessons(db_path):
    """DB with 6 sample lessons across two topic areas."""
    conn = init_db(db_path)
    for i in range(3):
        insert_lesson(
            conn,
            {
                "title": f"Subscriber lesson {i}",
                "one_liner": f"Store subscriber refs on self for cleanup {i}",
                "cluster_seed": "A",
                "keywords": "subscriber, lifecycle, cleanup",
                "created_date": "2026-02-26",
            },
        )
    for i in range(3):
        insert_lesson(
            conn,
            {
                "title": f"Planning lesson {i}",
                "one_liner": f"Plan quality exceeds execution quality {i}",
                "cluster_seed": "F",
                "keywords": "planning, quality, execution",
                "created_date": "2026-02-26",
            },
        )
    return conn


class TestExtractRepresentativeTerms:
    def test_returns_top_terms_from_one_liners(self, conn_with_lessons):
        lesson_ids = conn_with_lessons.execute("SELECT id FROM lessons WHERE cluster_seed='A'").fetchall()
        ids = [r["id"] for r in lesson_ids]
        terms = extract_representative_terms(conn_with_lessons, ids)
        assert isinstance(terms, list)
        assert len(terms) > 0
        assert all(isinstance(t, str) for t in terms)

    def test_filters_stopwords(self, conn_with_lessons):
        ids = [r["id"] for r in conn_with_lessons.execute("SELECT id FROM lessons").fetchall()]
        terms = extract_representative_terms(conn_with_lessons, ids)
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "of", "to", "is"}
        assert not any(t in stopwords for t in terms)


class TestFindSeedOverlap:
    def test_finds_majority_seed(self, conn_with_lessons):
        ids = [r["id"] for r in conn_with_lessons.execute("SELECT id FROM lessons WHERE cluster_seed='A'").fetchall()]
        seed = find_seed_overlap(conn_with_lessons, ids)
        assert seed == "A"

    def test_returns_none_for_mixed_cluster(self, conn_with_lessons):
        # Mix of A and F — no majority
        ids = [r["id"] for r in conn_with_lessons.execute("SELECT id FROM lessons").fetchall()]
        seed = find_seed_overlap(conn_with_lessons, ids)
        assert seed is None  # 50/50 split — below 60% threshold


class TestApplyClusterProposals:
    def test_writes_cluster_label_to_db(self, conn_with_lessons):
        ids = [r["id"] for r in conn_with_lessons.execute("SELECT id FROM lessons WHERE cluster_seed='A'").fetchall()]
        proposals = [{"cluster_id": 0, "lesson_ids": ids, "suggested_name": "Subscriber Lifecycle"}]
        confirmed = {0: "Subscriber Lifecycle"}
        count = apply_cluster_proposals(conn_with_lessons, proposals, confirmed)
        assert count == len(ids)
        for lid in ids:
            row = conn_with_lessons.execute("SELECT cluster FROM lessons WHERE id=?", [lid]).fetchone()
            assert row["cluster"] == "Subscriber Lifecycle"

    def test_skips_unconfirmed_proposals(self, conn_with_lessons):
        ids = [r["id"] for r in conn_with_lessons.execute("SELECT id FROM lessons WHERE cluster_seed='F'").fetchall()]
        proposals = [{"cluster_id": 1, "lesson_ids": ids, "suggested_name": "Planning Quality"}]
        confirmed = {}  # Nothing confirmed
        count = apply_cluster_proposals(conn_with_lessons, proposals, confirmed)
        assert count == 0

    def test_records_cluster_run(self, conn_with_lessons):
        proposals = [{"cluster_id": 0, "lesson_ids": [1], "suggested_name": "Test Cluster"}]
        apply_cluster_proposals(conn_with_lessons, proposals, {0: "Test Cluster"})
        runs = get_cluster_history(conn_with_lessons)
        assert len(runs) == 1
        assert runs[0]["proposal_count"] == 1


class TestGetClusterHistory:
    def test_returns_empty_initially(self, db_path):
        conn = init_db(db_path)
        assert get_cluster_history(conn) == []


class TestDiscoverClusters:
    def test_raises_runtime_error_when_deps_missing(self, db_path):
        """RuntimeError with install instructions when optional deps unavailable."""
        conn = init_db(db_path)
        with patch.dict("sys.modules", {"umap": None, "hdbscan": None}):
            with pytest.raises(RuntimeError, match="pip install"):
                discover_clusters(conn)
