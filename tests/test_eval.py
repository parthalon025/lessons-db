"""Tests for eval pipeline: config, variant definitions, test set selection."""

from lessons_db.config import DATA_DIR, EVAL_DIR
from lessons_db.db import init_db, insert_lesson
from lessons_db.eval import (
    VARIANT_CONFIGS,
    _select_diverse,
    select_source_lessons,
    select_transfer_targets,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_clusters(conn, clusters=None):
    """Seed test DB with lessons in known clusters. Returns dict of cluster_seed -> list of lesson IDs."""
    if clusters is None:
        clusters = {
            "A": [
                ("Silent failure 0", "integration"),
                ("Silent failure 1", "testing"),
                ("Silent failure 2", "monitoring"),
                ("Silent failure 3", "error-handling"),
                ("Silent failure 4", "caching"),
            ],
            "B": [
                ("Boundary issue 0", "integration"),
                ("Boundary issue 1", "data-model"),
                ("Boundary issue 2", "testing"),
                ("Boundary issue 3", "deployment"),
                ("Boundary issue 4", "integration"),
                ("Boundary issue 5", "data-model"),
            ],
            "D": [
                ("Spec drift 0", "integration"),
                ("Spec drift 1", "specification-drift"),
                ("Spec drift 2", "specification-drift"),
                ("Spec drift 3", "integration"),
            ],
            "E": [
                ("Context issue 0", "context-retrieval"),
                ("Context issue 1", "context-retrieval"),
                ("Context issue 2", "context-retrieval"),
                ("Context issue 3", "context-retrieval"),
            ],
            "F": [
                ("Plan issue 0", "planning-control-flow"),
                ("Plan issue 1", "planning-control-flow"),
                ("Plan issue 2", "data-model"),
                ("Plan issue 3", "frontend"),
            ],
        }
    ids_by_cluster = {}
    for seed, lessons in clusters.items():
        ids = []
        for title, cat in lessons:
            lid = insert_lesson(
                conn,
                {
                    "title": title,
                    "one_liner": f"One-liner for {title}",
                    "description": f"Description for {title}",
                    "cluster_seed": seed,
                    "category": cat,
                },
            )
            ids.append(lid)
        ids_by_cluster[seed] = ids
    return ids_by_cluster


# ---------------------------------------------------------------------------
# TestEvalConfig
# ---------------------------------------------------------------------------


class TestEvalConfig:
    def test_eval_dir_equals_data_dir_eval(self):
        assert EVAL_DIR == DATA_DIR / "eval"


# ---------------------------------------------------------------------------
# TestVariantConfigs
# ---------------------------------------------------------------------------


class TestVariantConfigs:
    def test_has_five_variants(self):
        assert set(VARIANT_CONFIGS.keys()) == {"A", "B", "C", "D", "E"}

    def test_each_variant_has_required_fields(self):
        required = {"prompt_id", "model", "temperature", "num_ctx", "chunked"}
        for vid, cfg in VARIANT_CONFIGS.items():
            missing = required - set(cfg.keys())
            assert not missing, f"Variant {vid} missing fields: {missing}"

    def test_variant_a_is_baseline(self):
        a = VARIANT_CONFIGS["A"]
        assert a["prompt_id"] == "baseline-fewshot"
        assert a["model"] == "deepseek-r1:8b-0528-qwen3-q4_K_M"
        assert a["temperature"] == 0.7
        assert a["num_ctx"] == 4096
        assert a["chunked"] is False

    def test_chunked_variants(self):
        """Variants C and E are chunked; A, B, D are not."""
        assert VARIANT_CONFIGS["C"]["chunked"] is True
        assert VARIANT_CONFIGS["E"]["chunked"] is True
        assert VARIANT_CONFIGS["A"]["chunked"] is False
        assert VARIANT_CONFIGS["B"]["chunked"] is False
        assert VARIANT_CONFIGS["D"]["chunked"] is False


# ---------------------------------------------------------------------------
# TestSelectSourceLessons
# ---------------------------------------------------------------------------


class TestSelectSourceLessons:
    def test_returns_correct_count_per_cluster(self, db_path):
        conn = init_db(db_path)
        ids_by_cluster = _seed_clusters(conn)
        results = select_source_lessons(conn, per_cluster=4)
        # Group results by cluster_seed
        by_cluster = {}
        for r in results:
            by_cluster.setdefault(r["cluster_seed"], []).append(r)
        # Each qualifying cluster should have up to 4 lessons
        for _seed, items in by_cluster.items():
            assert len(items) <= 4

    def test_respects_per_cluster_limit(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        results = select_source_lessons(conn, per_cluster=2)
        by_cluster = {}
        for r in results:
            by_cluster.setdefault(r["cluster_seed"], []).append(r)
        for _seed, items in by_cluster.items():
            assert len(items) <= 2

    def test_maximizes_category_diversity(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        results = select_source_lessons(conn, per_cluster=4)
        by_cluster = {}
        for r in results:
            by_cluster.setdefault(r["cluster_seed"], []).append(r)
        # Cluster A has 5 unique categories — picking 4 should yield 4 distinct categories
        if "A" in by_cluster:
            cats = [r["category"] for r in by_cluster["A"]]
            assert len(set(cats)) == len(cats), f"Cluster A should have all unique categories but got: {cats}"

    def test_empty_db_returns_empty(self, db_path):
        conn = init_db(db_path)
        results = select_source_lessons(conn)
        assert results == []

    def test_excludes_double_loop_meta_lessons(self, db_path):
        """Only single-loop lessons are selected (loop_level IS NULL or 'single')."""
        conn = init_db(db_path)
        # Create a cluster with 4 lessons so that after marking 1 double-loop,
        # 3 single-loop remain (meeting the >= 3 threshold)
        clusters = {
            "X": [
                ("Meta lesson 0", "integration"),
                ("Meta lesson 1", "testing"),
                ("Meta lesson 2", "monitoring"),
                ("Meta lesson 3", "caching"),
            ],
        }
        ids = _seed_clusters(conn, clusters)
        # Mark one as double-loop
        conn.execute("UPDATE lessons SET loop_level = 'double' WHERE id = ?", (ids["X"][0],))
        conn.commit()
        # Mark another as single explicitly
        conn.execute("UPDATE lessons SET loop_level = 'single' WHERE id = ?", (ids["X"][1],))
        conn.commit()
        # Third and fourth have default 'single' from insert

        results = select_source_lessons(conn, per_cluster=4)
        result_ids = [r["id"] for r in results]
        # The double-loop lesson should be excluded
        assert ids["X"][0] not in result_ids
        # The single-loop lessons should be included
        assert ids["X"][1] in result_ids
        assert ids["X"][2] in result_ids
        assert ids["X"][3] in result_ids

    def test_returns_required_keys(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        results = select_source_lessons(conn, per_cluster=2)
        required_keys = {"id", "title", "one_liner", "description", "cluster_seed", "category"}
        for r in results:
            assert required_keys <= set(r.keys()), f"Missing keys: {required_keys - set(r.keys())}"


# ---------------------------------------------------------------------------
# TestSelectDiverse
# ---------------------------------------------------------------------------


class TestSelectDiverse:
    def test_none_category_treated_as_distinct(self):
        """Lessons with category=None should be treated as a single category group."""
        lessons = [
            {"id": 1, "category": None},
            {"id": 2, "category": None},
            {"id": 3, "category": "testing"},
        ]
        result = _select_diverse(lessons, limit=2)
        # Pass 1 picks one None and one "testing" (two distinct categories)
        assert len(result) == 2
        cats = [r["category"] for r in result]
        assert None in cats
        assert "testing" in cats

    def test_limit_less_than_unique_categories(self):
        """When limit < number of unique categories, only limit items are returned."""
        lessons = [
            {"id": 1, "category": "a"},
            {"id": 2, "category": "b"},
            {"id": 3, "category": "c"},
            {"id": 4, "category": "d"},
        ]
        result = _select_diverse(lessons, limit=2)
        assert len(result) == 2
        # Should still be diverse — 2 different categories
        cats = [r["category"] for r in result]
        assert len(set(cats)) == 2


# ---------------------------------------------------------------------------
# TestSelectTransferTargets
# ---------------------------------------------------------------------------


class TestSelectTransferTargets:
    def test_returns_correct_structure(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A")
        assert "same_cluster" in result
        assert "diff_cluster" in result

    def test_correct_count_same_and_diff(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_same=2, count_diff=2)
        assert len(result["same_cluster"]) == 2
        assert len(result["diff_cluster"]) == 2

    def test_same_cluster_excludes_source(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_same=4)
        same_ids = [r["id"] for r in result["same_cluster"]]
        assert source_id not in same_ids

    def test_diff_cluster_from_other_clusters(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_diff=3)
        for r in result["diff_cluster"]:
            assert r["cluster_seed"] != "A", "diff_cluster should not contain source cluster A"

    def test_prefers_different_category_in_same_cluster(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        # Source is ids["A"][0] which has category "integration"
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_same=3)
        # The first results should prefer different categories over "integration"
        same = result["same_cluster"]
        # Cluster A has categories: integration, testing, monitoring, error-handling, caching
        # Source is integration, so first picks should be non-integration
        cats = [r["category"] for r in same]
        # With 3 picks from 4 remaining (testing, monitoring, error-handling, caching + integration[4]),
        # all 3 should be different from "integration" since there are 4 non-integration options
        for cat in cats:
            assert cat != "integration", f"Expected different category from source 'integration' but got '{cat}'"

    def test_only_single_loop_in_targets(self, db_path):
        """Transfer targets should only include single-loop lessons."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        # Mark one lesson in cluster A as double-loop
        conn.execute("UPDATE lessons SET loop_level = 'double' WHERE id = ?", (ids["A"][1],))
        conn.commit()
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_same=4)
        same_ids = [r["id"] for r in result["same_cluster"]]
        assert ids["A"][1] not in same_ids
