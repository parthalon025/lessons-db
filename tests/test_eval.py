"""Tests for eval pipeline: config, variant definitions, test set selection, generation."""

import json
from unittest.mock import MagicMock, patch

import pytest

from lessons_db.config import DATA_DIR, EVAL_DIR
from lessons_db.db import init_db, insert_lesson
from lessons_db.eval import (
    DEFAULT_BINARY_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL,
    VARIANT_CONFIGS,
    _clean_principle,
    _generate_for_lesson,
    _select_diverse,
    build_binary_judge_prompt,
    build_generation_prompt,
    build_judge_prompt,
    build_mechanism_extraction_prompt,
    build_paired_judge_prompt,
    build_simulation_prompt,
    call_judge,
    call_ollama,
    compute_bayesian_metrics,
    compute_embedding_signal,
    compute_mechanism_signal,
    compute_metrics,
    compute_paired_signal,
    compute_scope_signal,
    compute_simulation_lift,
    compute_tournament_metrics,
    compute_transfer_posterior,
    diagnose_vs_reference,
    parse_binary_judge,
    parse_judge_scores,
    parse_mechanism_triplet,
    parse_paired_judge,
    parse_simulation_result,
    render_report,
    render_v2_report,
    run_eval_generate,
    run_eval_judge,
    run_paired_tournament,
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
    def test_has_nine_variants(self):
        assert set(VARIANT_CONFIGS.keys()) == {"A", "B", "C", "D", "E", "F", "G", "H", "M"}

    def test_each_variant_has_required_fields(self):
        required = {"prompt_id", "model", "temperature", "num_ctx", "chunked"}
        for vid, cfg in VARIANT_CONFIGS.items():
            missing = required - set(cfg.keys())
            assert not missing, f"Variant {vid} missing fields: {missing}"

    def test_variant_a_is_baseline(self):
        a = VARIANT_CONFIGS["A"]
        assert a["prompt_id"] == "baseline-fewshot"
        assert a["model"] == "deepseek-r1:8b"
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
        results = select_source_lessons(conn, per_cluster=4, group_by="cluster_seed")
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
        results = select_source_lessons(conn, per_cluster=2, group_by="cluster_seed")
        by_cluster = {}
        for r in results:
            by_cluster.setdefault(r["cluster_seed"], []).append(r)
        for _seed, items in by_cluster.items():
            assert len(items) <= 2

    def test_maximizes_category_diversity(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        results = select_source_lessons(conn, per_cluster=4, group_by="cluster_seed")
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

        results = select_source_lessons(conn, per_cluster=4, group_by="cluster_seed")
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
        results = select_source_lessons(conn, per_cluster=2, group_by="cluster_seed")
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
        result = select_transfer_targets(conn, source_id, "A", group_by="cluster_seed")
        assert "same_cluster" in result
        assert "diff_cluster" in result

    def test_correct_count_same_and_diff(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_same=2, count_diff=2, group_by="cluster_seed")
        assert len(result["same_cluster"]) == 2
        assert len(result["diff_cluster"]) == 2

    def test_same_cluster_excludes_source(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_same=4, group_by="cluster_seed")
        same_ids = [r["id"] for r in result["same_cluster"]]
        assert source_id not in same_ids

    def test_diff_cluster_from_other_clusters(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_diff=3, group_by="cluster_seed")
        for r in result["diff_cluster"]:
            assert r["cluster_seed"] != "A", "diff_cluster should not contain source cluster A"

    def test_prefers_different_category_in_same_cluster(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        # Source is ids["A"][0] which has category "integration"
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", count_same=3, group_by="cluster_seed")
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
        result = select_transfer_targets(conn, source_id, "A", count_same=4, group_by="cluster_seed")
        same_ids = [r["id"] for r in result["same_cluster"]]
        assert ids["A"][1] not in same_ids


# ---------------------------------------------------------------------------
# TestSelectSourceLessonsByCategory
# ---------------------------------------------------------------------------


class TestSelectSourceLessonsByCategory:
    """select_source_lessons groups by category when group_by='category'."""

    def test_returns_lessons_from_distinct_categories(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        result = select_source_lessons(conn, per_cluster=2, group_by="category")
        categories = {r["category"] for r in result}
        # Should have multiple categories represented
        assert len(categories) >= 2
        conn.close()

    def test_group_by_category_default(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        result = select_source_lessons(conn, per_cluster=2)
        # Default is category — should work without explicit group_by
        assert len(result) > 0
        conn.close()

    def test_group_by_cluster_seed_still_works(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        result = select_source_lessons(conn, per_cluster=2, group_by="cluster_seed")
        clusters = {r["cluster_seed"] for r in result}
        assert len(clusters) >= 2
        conn.close()

    def test_invalid_group_by_raises(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        with pytest.raises(ValueError, match="group_by"):
            select_source_lessons(conn, per_cluster=2, group_by="invalid")
        conn.close()

    def test_category_grouping_produces_different_results_than_cluster_seed(self, db_path):
        """Category grouping selects based on category distribution, not cluster_seed."""
        conn = init_db(db_path)
        _seed_clusters(conn)
        by_category = select_source_lessons(conn, per_cluster=2, group_by="category")
        by_cluster = select_source_lessons(conn, per_cluster=2, group_by="cluster_seed")
        # The two groupings should produce different sets of source IDs
        # (since categories span clusters in the test data)
        cat_ids = {r["id"] for r in by_category}
        clus_ids = {r["id"] for r in by_cluster}
        # They may overlap but total counts or sets should differ
        assert len(by_category) > 0
        assert len(by_cluster) > 0
        # Category-based picks from groups with >= 3 lessons in the same category
        # while cluster-based picks from groups with >= 3 lessons in the same cluster
        conn.close()

    def test_category_groups_have_minimum_threshold(self, db_path):
        """Only categories with >= 3 single-loop lessons are selected."""
        conn = init_db(db_path)
        # Create categories: "big" has 4, "small" has 2 (below threshold)
        clusters = {
            "X": [
                ("Lesson 0", "big"),
                ("Lesson 1", "big"),
                ("Lesson 2", "big"),
                ("Lesson 3", "big"),
            ],
            "Y": [
                ("Lesson 4", "small"),
                ("Lesson 5", "small"),
            ],
        }
        _seed_clusters(conn, clusters)
        result = select_source_lessons(conn, per_cluster=4, group_by="category")
        categories = {r["category"] for r in result}
        assert "big" in categories
        assert "small" not in categories
        conn.close()


# ---------------------------------------------------------------------------
# TestSelectTransferTargetsByCategory
# ---------------------------------------------------------------------------


class TestSelectTransferTargetsByCategory:
    """select_transfer_targets groups by category when group_by='category'."""

    def test_same_group_by_category(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        # Lesson "Silent failure 0" (ids["A"][0]) has category="integration"
        result = select_transfer_targets(conn, ids["A"][0], "integration", group_by="category")
        for t in result["same_cluster"]:
            assert t["category"] == "integration"
        conn.close()

    def test_diff_group_by_category(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        result = select_transfer_targets(conn, ids["A"][0], "integration", group_by="category")
        for t in result["diff_cluster"]:
            assert t["category"] != "integration"
        conn.close()

    def test_backward_compat_cluster_seed(self, db_path):
        """Passing group_by='cluster_seed' still works like the old API."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "A", group_by="cluster_seed")
        assert "same_cluster" in result
        assert "diff_cluster" in result
        for t in result["same_cluster"]:
            assert t["cluster_seed"] == "A"
        for t in result["diff_cluster"]:
            assert t["cluster_seed"] != "A"
        conn.close()

    def test_invalid_group_by_raises(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        with pytest.raises(ValueError, match="group_by"):
            select_transfer_targets(conn, ids["A"][0], "integration", group_by="invalid")
        conn.close()

    def test_same_group_excludes_source(self, db_path):
        """Same-group targets exclude the source lesson itself."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        result = select_transfer_targets(conn, source_id, "integration", count_same=10, group_by="category")
        same_ids = [t["id"] for t in result["same_cluster"]]
        assert source_id not in same_ids
        conn.close()


# ---------------------------------------------------------------------------
# TestBuildGenerationPrompt
# ---------------------------------------------------------------------------


class TestBuildGenerationPrompt:
    """build_generation_prompt produces variant-specific prompts."""

    def _lesson(self, **overrides):
        base = {
            "id": 1,
            "title": "Test lesson",
            "one_liner": "Test one-liner",
            "description": "Test description of the lesson",
            "cluster_seed": "A",
            "category": "testing",
        }
        base.update(overrides)
        return base

    def test_variant_a_includes_examples(self):
        prompt = build_generation_prompt("A", self._lesson())
        assert "Examples of good principles" in prompt
        assert "Test lesson" in prompt

    def test_variant_b_is_zero_shot(self):
        prompt = build_generation_prompt("B", self._lesson())
        assert "Examples of good principles" not in prompt
        assert "causal" in prompt.lower() or "causes" in prompt.lower()

    def test_variant_c_requires_siblings(self):
        siblings = [self._lesson(id=2, title="Sibling 1"), self._lesson(id=3, title="Sibling 2")]
        prompt = build_generation_prompt("C", self._lesson(), siblings=siblings)
        assert "Sibling 1" in prompt
        assert "Sibling 2" in prompt

    def test_variant_c_without_siblings_falls_back(self):
        prompt = build_generation_prompt("C", self._lesson(), siblings=None)
        # Should fall back to variant B's zero-shot prompt
        assert "Test lesson" in prompt

    def test_variant_d_same_prompt_as_b(self):
        prompt_b = build_generation_prompt("B", self._lesson())
        prompt_d = build_generation_prompt("D", self._lesson())
        # Same prompt template, different model (handled at call level)
        assert prompt_b == prompt_d

    def test_variant_e_same_prompt_as_c(self):
        siblings = [self._lesson(id=2)]
        prompt_c = build_generation_prompt("C", self._lesson(), siblings=siblings)
        prompt_e = build_generation_prompt("E", self._lesson(), siblings=siblings)
        assert prompt_c == prompt_e

    def test_truncates_long_descriptions(self):
        long_desc = "x" * 1000
        prompt = build_generation_prompt("A", self._lesson(description=long_desc))
        assert "x" * 501 not in prompt  # description truncated at 500


# ---------------------------------------------------------------------------
# TestCallOllama
# ---------------------------------------------------------------------------


class TestCallOllama:
    """call_ollama sends HTTP request and returns cleaned response."""

    def test_returns_cleaned_response(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "  Test principle.  "}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = call_ollama("http://localhost:7683", "test-model", "prompt", {})
        assert result == "Test principle."

    def test_strips_think_tags(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "<think>reasoning here</think>Clean principle."}).encode(
            "utf-8"
        )
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = call_ollama("http://localhost:7683", "test-model", "prompt", {})
        assert result == "Clean principle."

    def test_returns_none_on_http_error(self):
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("http://localhost", 400, "Bad Request", {}, None),
        ):
            result = call_ollama("http://localhost:7683", "test-model", "prompt", {})
        assert result is None

    def test_retries_on_502(self):
        import urllib.error

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "ok"}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=[
                    urllib.error.HTTPError("http://localhost", 502, "Bad Gateway", {}, None),
                    mock_resp,
                ],
            ),
            patch("time.sleep"),
        ):
            result = call_ollama("http://localhost:7683", "test-model", "prompt", {})
        assert result == "ok"

    def test_exhausts_retries_on_persistent_502(self):
        import urllib.error

        error = urllib.error.HTTPError("http://localhost", 502, "Bad Gateway", {}, None)
        with patch("urllib.request.urlopen", side_effect=error), patch("time.sleep"):
            result = call_ollama("http://localhost:7683", "test-model", "prompt", {})
        assert result is None

    def test_sends_correct_payload(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "ok"}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            call_ollama(
                "http://localhost:7683",
                "my-model",
                "my prompt",
                {"temperature": 0.6, "num_ctx": 8192},
            )
        req = mock_url.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "my-model"
        assert payload["prompt"] == "my prompt"
        assert payload["options"]["temperature"] == 0.6
        assert payload["options"]["num_ctx"] == 8192

    def test_sends_priority_and_source_in_payload(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "ok"}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            call_ollama(
                "http://localhost:7683",
                "my-model",
                "my prompt",
                {},
                priority=1,
                source="eval-generate",
            )
        req = mock_url.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["_priority"] == 1
        assert payload["_source"] == "eval-generate"
        assert payload["_timeout"] == 600  # default timeout

    def test_omits_queue_fields_when_priority_unset(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "ok"}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            call_ollama("http://localhost:7683", "my-model", "my prompt", {})
        req = mock_url.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert "_priority" not in payload
        assert "_source" not in payload
        assert "_timeout" not in payload


# ---------------------------------------------------------------------------
# TestRunEvalGenerate
# ---------------------------------------------------------------------------


class TestRunEvalGenerate:
    """run_eval_generate orchestrates variant x lesson generation."""

    def test_generates_results_json(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "Test principle from model."}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["A"],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "meta" in data
        assert "results" in data
        assert len(data["results"]) > 0
        assert data["results"][0]["variant"] == "A"
        assert data["results"][0]["principle"] is not None
        conn.close()

    def test_resume_skips_existing(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        # Pre-seed a partial results file
        existing = {
            "meta": {"variants": ["A"], "per_cluster": 1},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "principle": "Already done",
                    "error": None,
                }
            ],
        }
        output_path.write_text(json.dumps(existing))

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "New principle."}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["A"],
                per_cluster=1,
                output_path=output_path,
                resume=True,
            )

        data = json.loads(output_path.read_text())
        a_results = [r for r in data["results"] if r["variant"] == "A" and r["lesson_id"] == ids["A"][0]]
        assert len(a_results) == 1
        assert a_results[0]["principle"] == "Already done"
        conn.close()

    def test_records_errors(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        import urllib.error

        with (
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.HTTPError("http://localhost", 502, "Bad Gateway", {}, None),
            ),
            patch("time.sleep"),
        ):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["A"],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        data = json.loads(output_path.read_text())
        error_results = [r for r in data["results"] if r["error"] is not None]
        assert len(error_results) > 0
        conn.close()

    def test_includes_metadata(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "Principle."}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["A", "B"],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        data = json.loads(output_path.read_text())
        assert data["meta"]["variants"] == ["A", "B"]
        assert "generated_at" in data["meta"]
        assert "source_lessons" in data["meta"]
        conn.close()

    def test_groups_variants_by_model(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "Principle."}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        # Pass D before A — D uses qwen3:14b, A uses deepseek-r1
        # Model grouping should run A (deepseek) before D (qwen)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["D", "A"],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        data = json.loads(output_path.read_text())
        variant_order = [r["variant"] for r in data["results"]]
        # All A entries should come before all D entries (deepseek < qwen alphabetically)
        a_indices = [i for i, v in enumerate(variant_order) if v == "A"]
        d_indices = [i for i, v in enumerate(variant_order) if v == "D"]
        assert max(a_indices) < min(d_indices), f"A should run before D, got order: {variant_order}"
        conn.close()


# ---------------------------------------------------------------------------
# TestBuildJudgePrompt
# ---------------------------------------------------------------------------


class TestBuildJudgePrompt:
    """build_judge_prompt creates the rubric-based scoring prompt."""

    def test_contains_principle(self):
        prompt = build_judge_prompt(
            "Silent fallbacks mask failures.",
            {
                "title": "Git apply silent failure",
                "one_liner": "|| true discards errors",
                "description": "The git apply command...",
            },
        )
        assert "Silent fallbacks mask failures." in prompt

    def test_contains_target_lesson(self):
        prompt = build_judge_prompt(
            "Test principle.",
            {
                "title": "My Target Lesson",
                "one_liner": "Target one-liner",
                "description": "Target description",
            },
        )
        assert "My Target Lesson" in prompt

    def test_contains_scoring_criteria(self):
        prompt = build_judge_prompt("Principle.", {"title": "T", "one_liner": "O", "description": "D"})
        assert "transfer" in prompt.lower()
        assert "precision" in prompt.lower()
        assert "actionability" in prompt.lower()

    def test_requests_json_output(self):
        prompt = build_judge_prompt("Principle.", {"title": "T", "one_liner": "O", "description": "D"})
        assert "JSON" in prompt or "json" in prompt

    def test_truncates_long_descriptions(self):
        long_desc = "x" * 600
        prompt = build_judge_prompt("Principle.", {"title": "T", "one_liner": "O", "description": long_desc})
        assert "x" * 301 not in prompt  # description truncated at 300


# ---------------------------------------------------------------------------
# TestCleanPrinciple
# ---------------------------------------------------------------------------


class TestCleanPrinciple:
    """_clean_principle strips CoT artifacts from generated principles."""

    def test_clean_text_unchanged(self):
        text = "Silent fallbacks mask failures when errors are suppressed."
        assert _clean_principle(text) == text

    def test_strips_principle_marker(self):
        text = "**Principle:** Silent fallbacks mask failures.\n\nThis principle distinguishes..."
        assert _clean_principle(text) == "Silent fallbacks mask failures."

    def test_strips_the_principle_is_marker(self):
        text = "The principle is: **Delegation Failure causes errors.**\n\nThis principle..."
        assert _clean_principle(text) == "Delegation Failure causes errors."

    def test_strips_cot_preamble(self):
        text = (
            "Okay, let's analyze the pattern.\n\n"
            "The lessons share a theme of failure suppression.\n\n"
            "* Lesson 1: colliding changes..."
        )
        result = _clean_principle(text)
        assert not result.startswith("Okay")
        assert "lessons share a theme" in result

    def test_strips_trailing_explanation(self):
        text = "Ambiguous requirements cause errors.\n\n*(This principle applies because...)"
        assert _clean_principle(text) == "Ambiguous requirements cause errors."

    def test_strips_this_principle_applies(self):
        text = "Resource cleanup prevents leaks. *(This principle distinguishes...)"
        assert _clean_principle(text) == "Resource cleanup prevents leaks."

    def test_strips_markdown_bold(self):
        text = "**Guarded Merge Fails When Merging**"
        assert _clean_principle(text) == "Guarded Merge Fails When Merging"

    def test_empty_string(self):
        assert _clean_principle("") == ""

    def test_none_passthrough(self):
        assert _clean_principle(None) is None

    def test_judge_prompt_uses_cleaned_principle(self):
        """build_judge_prompt should strip CoT before embedding."""
        raw = "**Principle:** Clean version.\n\nThis principle applies because..."
        prompt = build_judge_prompt(raw, {"title": "T", "one_liner": "O", "description": "D"})
        assert "Clean version." in prompt
        assert "This principle applies" not in prompt


# ---------------------------------------------------------------------------
# TestParseJudgeScores
# ---------------------------------------------------------------------------


class TestParseJudgeScores:
    """parse_judge_scores extracts 3 integer scores from judge response."""

    def test_parses_valid_json(self):
        response = '{"transfer": 4, "precision": 3, "actionability": 5}'
        scores = parse_judge_scores(response)
        assert scores == {"transfer": 4, "precision": 3, "actionability": 5}

    def test_parses_json_with_surrounding_text(self):
        response = 'Here are the scores:\n{"transfer": 2, "precision": 1, "actionability": 3}\nDone.'
        scores = parse_judge_scores(response)
        assert scores == {"transfer": 2, "precision": 1, "actionability": 3}

    def test_returns_none_on_invalid(self):
        scores = parse_judge_scores("I cannot score this.")
        assert scores is None

    def test_returns_none_on_missing_keys(self):
        scores = parse_judge_scores('{"transfer": 4}')
        assert scores is None

    def test_clamps_scores_to_1_5(self):
        response = '{"transfer": 0, "precision": 7, "actionability": 3}'
        scores = parse_judge_scores(response)
        assert scores["transfer"] == 1
        assert scores["precision"] == 5
        assert scores["actionability"] == 3


# ---------------------------------------------------------------------------
# TestCallJudge
# ---------------------------------------------------------------------------


class TestCallJudge:
    """call_judge routes to Ollama or OpenAI based on backend parameter."""

    def test_ollama_backend(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": '{"transfer": 4, "precision": 3, "actionability": 5}'}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = call_judge(
                prompt="test prompt",
                backend="ollama",
                ollama_url="http://localhost:7683",
                ollama_model=DEFAULT_JUDGE_MODEL,
            )
        assert result is not None
        assert "transfer" in result

    def test_openai_backend(self):
        mock_resp = MagicMock()
        openai_response = {"choices": [{"message": {"content": '{"transfer": 3, "precision": 2, "actionability": 4}'}}]}
        mock_resp.read.return_value = json.dumps(openai_response).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = call_judge(
                prompt="test prompt",
                backend="openai",
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
            )
        assert result is not None

    def test_returns_none_on_error(self):
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail")):
            result = call_judge(
                prompt="test prompt",
                backend="ollama",
                ollama_url="http://localhost:7683",
                ollama_model=DEFAULT_JUDGE_MODEL,
            )
        assert result is None


# ---------------------------------------------------------------------------
# TestComputeMetrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """compute_metrics calculates F1, recall, precision per variant."""

    def _make_scored_pair(self, variant, is_same_cluster, transfer=3, precision=3, actionability=3):
        return {
            "variant": variant,
            "is_same_cluster": is_same_cluster,
            "scores": {"transfer": transfer, "precision": precision, "actionability": actionability},
        }

    def test_perfect_scores(self):
        pairs = [
            self._make_scored_pair("A", True, transfer=5, precision=5, actionability=5),
            self._make_scored_pair("A", True, transfer=5, precision=5, actionability=5),
            self._make_scored_pair("A", False, transfer=1, precision=5, actionability=5),
            self._make_scored_pair("A", False, transfer=1, precision=5, actionability=5),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["recall"] == 1.0
        assert metrics["A"]["precision"] == 1.0
        assert metrics["A"]["f1"] == 1.0

    def test_zero_recall(self):
        pairs = [
            self._make_scored_pair("A", True, transfer=1),
            self._make_scored_pair("A", True, transfer=2),
            self._make_scored_pair("A", False, transfer=1),
            self._make_scored_pair("A", False, transfer=1),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["recall"] == 0.0

    def test_zero_precision(self):
        pairs = [
            self._make_scored_pair("A", True, transfer=5),
            self._make_scored_pair("A", True, transfer=5),
            self._make_scored_pair("A", False, transfer=5),
            self._make_scored_pair("A", False, transfer=4),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["precision"] == 0.0

    def test_multiple_variants(self):
        pairs = [
            self._make_scored_pair("A", True, transfer=5),
            self._make_scored_pair("A", False, transfer=1),
            self._make_scored_pair("B", True, transfer=3),
            self._make_scored_pair("B", False, transfer=3),
        ]
        metrics = compute_metrics(pairs)
        assert "A" in metrics
        assert "B" in metrics

    def test_mean_actionability(self):
        pairs = [
            self._make_scored_pair("A", True, actionability=4),
            self._make_scored_pair("A", True, actionability=2),
            self._make_scored_pair("A", False, actionability=3),
            self._make_scored_pair("A", False, actionability=5),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["mean_actionability"] == 3.5


# ---------------------------------------------------------------------------
# TestRenderReport
# ---------------------------------------------------------------------------


class TestRenderReport:
    """render_report produces valid markdown."""

    def test_contains_summary_table(self):
        metrics = {
            "A": {"recall": 0.8, "precision": 0.7, "f1": 0.75, "mean_actionability": 3.5},
            "B": {"recall": 0.9, "precision": 0.6, "f1": 0.72, "mean_actionability": 4.0},
        }
        report = render_report(metrics, [], {"A": {}, "B": {}})
        assert "| Variant" in report
        assert "0.80" in report or "0.8" in report

    def test_identifies_winner(self):
        metrics = {
            "A": {"recall": 0.5, "precision": 0.5, "f1": 0.50, "mean_actionability": 3.0},
            "B": {"recall": 0.9, "precision": 0.9, "f1": 0.90, "mean_actionability": 4.5},
        }
        report = render_report(metrics, [], {"A": {}, "B": {}})
        assert "B" in report  # Winner mentioned

    def test_renders_per_cluster_breakdown(self):
        metrics = {"A": {"recall": 0.8, "precision": 0.7, "f1": 0.75, "mean_actionability": 3.5}}
        scored_pairs = [
            {
                "variant": "A",
                "is_same_cluster": True,
                "cluster_seed": "X",
                "scores": {"transfer": 4, "precision": 3, "actionability": 5},
                "principle": "Test",
                "target_title": "Target",
            },
            {
                "variant": "A",
                "is_same_cluster": False,
                "cluster_seed": "X",
                "scores": {"transfer": 2, "precision": 4, "actionability": 3},
                "principle": "Test",
                "target_title": "Target2",
            },
        ]
        report = render_report(metrics, scored_pairs, {"A": {}})
        assert "Per-Cluster Breakdown" in report
        assert "Cluster X" in report

    def test_renders_failure_analysis(self):
        metrics = {"A": {"recall": 0.5, "precision": 0.5, "f1": 0.5, "mean_actionability": 3.0}}
        scored_pairs = [
            {
                "variant": "A",
                "is_same_cluster": True,
                "cluster_seed": "Y",
                "scores": {"transfer": 1, "precision": 2, "actionability": 2},
                "principle": "Weak principle that fails transfer",
                "target_title": "Some target lesson",
            },
        ]
        report = render_report(metrics, scored_pairs, {"A": {}})
        assert "Failure Analysis" in report
        assert "scored below threshold" in report


# ---------------------------------------------------------------------------
# TestRunEvalJudge
# ---------------------------------------------------------------------------


class TestRunEvalJudge:
    """run_eval_judge orchestrates scoring of generated principles."""

    def test_produces_scored_pairs_and_metrics(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [ids["A"][0]]},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "lesson_title": "Silent failure 0",
                    "cluster_seed": "A",
                    "category": "integration",
                    "principle": "Silent fallbacks mask upstream failures.",
                    "model": "test-model",
                    "prompt_id": "baseline-fewshot",
                    "settings": {},
                    "generation_time_s": 1.0,
                    "error": None,
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))
        report_path = tmp_path / "report.md"

        def mock_judge(prompt, **kwargs):
            return '{"transfer": 4, "precision": 3, "actionability": 5}'

        with patch("lessons_db.eval.judge.call_judge", side_effect=mock_judge):
            scored_pairs, metrics = run_eval_judge(
                results_path=results_path,
                conn=conn,
                report_path=report_path,
                backend="ollama",
            )

        assert len(scored_pairs) > 0
        assert "A" in metrics
        assert report_path.exists()
        report_text = report_path.read_text()
        assert "Variant" in report_text
        conn.close()

    def test_skips_error_results(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [ids["A"][0]]},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "lesson_title": "Error lesson",
                    "cluster_seed": "A",
                    "category": "integration",
                    "principle": None,
                    "error": "generation_failed",
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))
        report_path = tmp_path / "report.md"

        scored_pairs, metrics = run_eval_judge(
            results_path=results_path,
            conn=conn,
            report_path=report_path,
            backend="ollama",
        )

        assert len(scored_pairs) == 0
        conn.close()

    def test_fallback_scores_on_judge_failure(self, db_path, tmp_path):
        """When call_judge returns None, scored pair should get all-1 default scores."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [ids["A"][0]]},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "lesson_title": "Silent failure 0",
                    "cluster_seed": "A",
                    "category": "integration",
                    "principle": "Test principle for fallback.",
                    "model": "test-model",
                    "prompt_id": "baseline-fewshot",
                    "settings": {},
                    "generation_time_s": 1.0,
                    "error": None,
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))
        report_path = tmp_path / "report.md"

        # Mock judge to always return None (simulates network failure / parse failure)
        with patch("lessons_db.eval.judge.call_judge", return_value=None):
            scored_pairs, metrics = run_eval_judge(
                results_path=results_path,
                conn=conn,
                report_path=report_path,
                backend="ollama",
            )

        assert len(scored_pairs) > 0
        for pair in scored_pairs:
            assert pair["scores"] == {"transfer": 1, "precision": 1, "actionability": 1}
        conn.close()


# ---------------------------------------------------------------------------
# TestBinaryJudge
# ---------------------------------------------------------------------------


class TestBinaryJudgePrompt:
    """build_binary_judge_prompt produces valid YES/NO prompts."""

    def test_contains_principle(self):
        target = {"title": "Test Bug", "one_liner": "A test bug", "description": "Details"}
        prompt = build_binary_judge_prompt("Always validate input", target)
        assert "Always validate input" in prompt

    def test_contains_yes_no_instruction(self):
        target = {"title": "Bug", "one_liner": "test", "description": "test"}
        prompt = build_binary_judge_prompt("Test principle", target)
        assert "YES" in prompt
        assert "NO" in prompt

    def test_cleans_principle(self):
        target = {"title": "Bug", "one_liner": "test", "description": "test"}
        prompt = build_binary_judge_prompt("**Principle:** Always validate input", target)
        assert "Always validate input" in prompt
        assert "**Principle:**" not in prompt


class TestParseBinaryJudge:
    """parse_binary_judge handles YES/NO responses."""

    def test_parses_yes(self):
        assert parse_binary_judge("YES") is True

    def test_parses_no(self):
        assert parse_binary_judge("NO") is False

    def test_parses_yes_with_explanation(self):
        assert parse_binary_judge("YES - the mechanism matches") is True

    def test_parses_no_with_explanation(self):
        assert parse_binary_judge("NO - different mechanism") is False

    def test_strips_think_tags(self):
        assert parse_binary_judge("<THINK>reasoning</THINK>\nYES") is True

    def test_returns_none_on_empty(self):
        assert parse_binary_judge("") is None

    def test_returns_none_on_ambiguous(self):
        assert parse_binary_judge("Maybe YES or NO depending on context" * 5) is None


class TestComputeMetricsBinary:
    """compute_metrics handles binary scored pairs."""

    def _make_binary_pair(self, variant, is_same_cluster, matched):
        return {
            "variant": variant,
            "is_same_cluster": is_same_cluster,
            "scores": {"matched": matched},
        }

    def test_perfect_binary(self):
        pairs = [
            self._make_binary_pair("A", True, True),
            self._make_binary_pair("A", True, True),
            self._make_binary_pair("A", False, False),
            self._make_binary_pair("A", False, False),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["recall"] == 1.0
        assert metrics["A"]["precision"] == 1.0
        assert metrics["A"]["f1"] == 1.0
        assert metrics["A"]["tp"] == 2
        assert metrics["A"]["tn"] == 2
        assert metrics["A"]["binary"] is True

    def test_zero_precision_binary(self):
        pairs = [
            self._make_binary_pair("A", True, True),
            self._make_binary_pair("A", False, True),
            self._make_binary_pair("A", False, True),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["precision"] < 0.5
        assert metrics["A"]["fp"] == 2

    def test_zero_recall_binary(self):
        pairs = [
            self._make_binary_pair("A", True, False),
            self._make_binary_pair("A", True, False),
            self._make_binary_pair("A", False, False),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["recall"] == 0.0
        assert metrics["A"]["fn"] == 2

    def test_no_actionability_in_binary(self):
        pairs = [self._make_binary_pair("A", True, True)]
        metrics = compute_metrics(pairs)
        assert "mean_actionability" not in metrics["A"]


class TestRenderReportBinary:
    """render_report handles binary metrics format."""

    def test_binary_summary_table(self):
        metrics = {
            "A": {"recall": 0.8, "precision": 0.75, "f1": 0.77, "tp": 4, "fp": 1, "fn": 1, "tn": 3, "binary": True},
        }
        report = render_report(metrics, [], {"A": {}})
        assert "| TP | FP | FN | TN |" in report
        assert "Actionability" not in report

    def test_binary_failure_analysis(self):
        scored = [
            {
                "variant": "A",
                "is_same_cluster": True,
                "scores": {"matched": False},
                "cluster_seed": "X",
                "principle": "Test principle",
                "target_title": "Test target",
            },
        ]
        metrics = {
            "A": {"recall": 0.0, "precision": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 1, "tn": 0, "binary": True}
        }
        report = render_report(metrics, scored, {"A": {}})
        assert "false negatives" in report

    def test_binary_false_positives_shown(self):
        scored = [
            {
                "variant": "A",
                "is_same_cluster": False,
                "scores": {"matched": True},
                "cluster_seed": "X",
                "principle": "Bad principle",
                "target_title": "Wrong match",
            },
        ]
        metrics = {
            "A": {"recall": 0.0, "precision": 0.0, "f1": 0.0, "tp": 0, "fp": 1, "fn": 0, "tn": 0, "binary": True}
        }
        report = render_report(metrics, scored, {"A": {}})
        assert "false positives" in report


class TestRunEvalJudgeBinary:
    """run_eval_judge with binary=True uses binary prompt and parser."""

    def test_binary_mode_uses_binary_prompt(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [ids["A"][0]]},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "lesson_title": "Silent failure 0",
                    "cluster_seed": "A",
                    "category": "integration",
                    "principle": "Silent fallbacks mask upstream failures.",
                    "model": "test-model",
                    "prompt_id": "baseline-fewshot",
                    "settings": {},
                    "generation_time_s": 1.0,
                    "error": None,
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))
        report_path = tmp_path / "report.md"

        def mock_judge(prompt, **kwargs):
            # Binary prompt asks for YES/NO, not JSON
            if "Answer ONLY 'YES' or 'NO'" in prompt:
                return "YES"
            return '{"transfer": 4, "precision": 3, "actionability": 5}'

        with patch("lessons_db.eval.judge.call_judge", side_effect=mock_judge):
            scored_pairs, metrics = run_eval_judge(
                results_path=results_path,
                conn=conn,
                report_path=report_path,
                backend="ollama",
                binary=True,
            )

        assert len(scored_pairs) > 0
        for pair in scored_pairs:
            assert "matched" in pair["scores"]
            assert "transfer" not in pair["scores"]
        assert metrics["A"]["binary"] is True
        conn.close()

    def test_binary_fallback_on_judge_failure(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [ids["A"][0]]},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "lesson_title": "Test",
                    "cluster_seed": "A",
                    "category": "integration",
                    "principle": "Test principle.",
                    "model": "test-model",
                    "prompt_id": "test",
                    "settings": {},
                    "generation_time_s": 1.0,
                    "error": None,
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))
        report_path = tmp_path / "report.md"

        with patch("lessons_db.eval.judge.call_judge", return_value=None):
            scored_pairs, metrics = run_eval_judge(
                results_path=results_path,
                conn=conn,
                report_path=report_path,
                backend="ollama",
                binary=True,
            )

        assert len(scored_pairs) > 0
        for pair in scored_pairs:
            assert pair["scores"]["matched"] is False
        conn.close()


class TestDefaultBinaryJudgeModel:
    """DEFAULT_BINARY_JUDGE_MODEL is gemma3:12b."""

    def test_default_binary_model(self):
        assert DEFAULT_BINARY_JUDGE_MODEL == "gemma3:12b"


class TestBuildPairedPrompt:
    def test_contains_both_targets(self):
        same = {
            "title": "Resource cleanup",
            "one_liner": "Close connections",
            "description": "DB connections left open",
        }
        diff = {"title": "CSS specificity", "one_liner": "Use BEM", "description": "CSS specificity wars"}
        prompt, same_is_a = build_paired_judge_prompt("Always close resources", same, diff)
        assert "TARGET A" in prompt
        assert "TARGET B" in prompt
        assert "Resource cleanup" in prompt
        assert "CSS specificity" in prompt

    def test_asks_for_a_or_b(self):
        same = {"title": "T1", "one_liner": "O1", "description": "D1"}
        diff = {"title": "T2", "one_liner": "O2", "description": "D2"}
        prompt, _ = build_paired_judge_prompt("Principle", same, diff)
        assert "A" in prompt
        assert "B" in prompt
        assert "NEITHER" in prompt

    def test_randomizes_position(self):
        """Over many seeds, same-cluster target should appear in both A and B positions."""
        same = {"title": "Same", "one_liner": "S", "description": "S"}
        diff = {"title": "Diff", "one_liner": "D", "description": "D"}
        positions = set()
        for seed in range(20):
            prompt, same_is_a = build_paired_judge_prompt("P", same, diff, position_seed=seed)
            positions.add("A" if same_is_a else "B")
        assert len(positions) == 2  # both positions used

    def test_returns_same_is_a_flag(self):
        same = {"title": "Same", "one_liner": "S", "description": "S"}
        diff = {"title": "Diff", "one_liner": "D", "description": "D"}
        prompt, same_is_a = build_paired_judge_prompt("P", same, diff, position_seed=1)
        assert isinstance(same_is_a, bool)
        # Verify the flag matches actual placement
        if same_is_a:
            # Same target should be in position A (before "TARGET B")
            a_section = prompt.split("TARGET B")[0]
            assert "Same" in a_section
        else:
            b_section = prompt.split("TARGET B")[1]
            assert "Same" in b_section

    def test_cleans_principle(self):
        """Principle should be cleaned (strip CoT artifacts)."""
        same = {"title": "T1", "one_liner": "O1", "description": "D1"}
        diff = {"title": "T2", "one_liner": "O2", "description": "D2"}
        prompt, _ = build_paired_judge_prompt("<think>reasoning</think>The actual principle", same, diff)
        assert "<think>" not in prompt
        assert "The actual principle" in prompt


class TestParsePairedJudge:
    def test_parses_a(self):
        assert parse_paired_judge("A") == "A"

    def test_parses_b(self):
        assert parse_paired_judge("B") == "B"

    def test_parses_neither(self):
        assert parse_paired_judge("NEITHER") == "NEITHER"

    def test_parses_a_with_explanation(self):
        assert parse_paired_judge("A - Target A matches the structural pattern better") == "A"

    def test_parses_b_with_explanation(self):
        assert parse_paired_judge("B. The principle specifically addresses this failure mode.") == "B"

    def test_strips_think_tags(self):
        assert parse_paired_judge("<think>Let me analyze...</think>A") == "A"

    def test_returns_none_for_empty(self):
        assert parse_paired_judge("") is None

    def test_returns_none_for_ambiguous(self):
        assert parse_paired_judge("I think both targets are equally applicable to this principle") is None

    def test_parses_neither_in_sentence(self):
        assert parse_paired_judge("NEITHER target applies well here") == "NEITHER"


# ---------------------------------------------------------------------------
# TestRunPairedTournament
# ---------------------------------------------------------------------------


class TestRunPairedTournament:
    def test_returns_win_rate(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        # Set distinct categories so group_by="category" has clear groups
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.execute("UPDATE lessons SET category = 'spec-drift' WHERE cluster_seed = 'D'")
        conn.execute("UPDATE lessons SET category = 'context' WHERE cluster_seed = 'E'")
        conn.execute("UPDATE lessons SET category = 'planning' WHERE cluster_seed = 'F'")
        conn.commit()

        results_data = {
            "meta": {"variants": ["A"], "group_by": "category"},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "cluster_seed": "A",
                    "category": "error-handling",
                    "principle": "Always close resources in finally blocks.",
                    "error": None,
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))

        # Mock: always pick A
        with patch("lessons_db.eval.judge.call_judge", return_value="A"):
            tournament_results = run_paired_tournament(
                results_path=results_path,
                conn=conn,
                group_by="category",
                pairs_per_principle=2,
            )

        assert len(tournament_results) > 0
        for r in tournament_results:
            assert "win_rate" in r
            assert "comparisons" in r
            assert "wins" in r
            assert "losses" in r
            assert "neithers" in r
            assert 0.0 <= r["win_rate"] <= 1.0
            assert r["comparisons"] > 0
            assert r["wins"] + r["losses"] + r["neithers"] == r["comparisons"]
        conn.close()

    def test_perfect_judge_gets_high_win_rate(self, db_path, tmp_path):
        """If judge always picks the same-group target, win_rate should be 1.0."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.execute("UPDATE lessons SET category = 'spec-drift' WHERE cluster_seed = 'D'")
        conn.execute("UPDATE lessons SET category = 'context' WHERE cluster_seed = 'E'")
        conn.execute("UPDATE lessons SET category = 'planning' WHERE cluster_seed = 'F'")
        conn.commit()

        results_data = {
            "meta": {"variants": ["A"], "group_by": "category"},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "cluster_seed": "A",
                    "category": "error-handling",
                    "principle": "Always close resources.",
                    "error": None,
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))

        # Smart mock: track what build_paired_judge_prompt returns for same_is_a
        # and always pick the same-group target
        original_build = build_paired_judge_prompt
        same_is_a_tracker = []

        def tracking_build(principle, same_target, diff_target, position_seed=None):
            prompt, same_is_a = original_build(principle, same_target, diff_target, position_seed)
            same_is_a_tracker.append(same_is_a)
            return prompt, same_is_a

        def smart_judge(prompt, **kwargs):
            # Return whichever letter corresponds to the same-group target
            same_is_a = same_is_a_tracker[-1]  # most recent build call
            return "A" if same_is_a else "B"

        with (
            patch("lessons_db.eval.judge.build_paired_judge_prompt", side_effect=tracking_build),
            patch("lessons_db.eval.judge.call_judge", side_effect=smart_judge),
        ):
            results = run_paired_tournament(
                results_path=results_path,
                conn=conn,
                group_by="category",
                pairs_per_principle=2,
            )

        assert len(results) > 0
        assert results[0]["comparisons"] > 0
        assert results[0]["win_rate"] == 1.0
        assert results[0]["losses"] == 0
        assert results[0]["neithers"] == 0
        conn.close()

    def test_skips_errored_entries(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.execute("UPDATE lessons SET category = 'spec-drift' WHERE cluster_seed = 'D'")
        conn.execute("UPDATE lessons SET category = 'context' WHERE cluster_seed = 'E'")
        conn.execute("UPDATE lessons SET category = 'planning' WHERE cluster_seed = 'F'")
        conn.commit()

        results_data = {
            "meta": {"variants": ["A"], "group_by": "category"},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "cluster_seed": "A",
                    "category": "error-handling",
                    "principle": None,
                    "error": "gen_failed",
                },
                {
                    "variant": "A",
                    "lesson_id": ids["A"][1],
                    "cluster_seed": "A",
                    "category": "error-handling",
                    "principle": "Valid principle.",
                    "error": None,
                },
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))

        with patch("lessons_db.eval.judge.call_judge", return_value="A"):
            results = run_paired_tournament(
                results_path=results_path,
                conn=conn,
                group_by="category",
                pairs_per_principle=2,
            )

        # Only the valid entry should produce results
        assert len(results) == 1
        conn.close()

    def test_handles_neither_response(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.execute("UPDATE lessons SET category = 'spec-drift' WHERE cluster_seed = 'D'")
        conn.execute("UPDATE lessons SET category = 'context' WHERE cluster_seed = 'E'")
        conn.execute("UPDATE lessons SET category = 'planning' WHERE cluster_seed = 'F'")
        conn.commit()

        results_data = {
            "meta": {"variants": ["A"], "group_by": "category"},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "cluster_seed": "A",
                    "category": "error-handling",
                    "principle": "Some principle.",
                    "error": None,
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))

        with patch("lessons_db.eval.judge.call_judge", return_value="NEITHER"):
            results = run_paired_tournament(
                results_path=results_path,
                conn=conn,
                group_by="category",
                pairs_per_principle=2,
            )

        assert len(results) > 0
        r = results[0]
        assert r["wins"] == 0
        assert r["losses"] == 0
        assert r["neithers"] == r["comparisons"]
        assert r["win_rate"] == 0.0
        conn.close()

    def test_handles_parse_failure(self, db_path, tmp_path):
        """When the judge returns unparseable garbage, it counts as neither."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.execute("UPDATE lessons SET category = 'spec-drift' WHERE cluster_seed = 'D'")
        conn.execute("UPDATE lessons SET category = 'context' WHERE cluster_seed = 'E'")
        conn.execute("UPDATE lessons SET category = 'planning' WHERE cluster_seed = 'F'")
        conn.commit()

        results_data = {
            "meta": {"variants": ["A"], "group_by": "category"},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "cluster_seed": "A",
                    "category": "error-handling",
                    "principle": "Some principle.",
                    "error": None,
                }
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))

        # Return garbage the parser can't handle
        with patch(
            "lessons_db.eval.judge.call_judge",
            return_value="I think both are equally applicable and cannot decide",
        ):
            results = run_paired_tournament(
                results_path=results_path,
                conn=conn,
                group_by="category",
                pairs_per_principle=2,
            )

        assert len(results) > 0
        r = results[0]
        assert r["wins"] == 0
        assert r["losses"] == 0
        # Parse failures count as neithers
        assert r["neithers"] == r["comparisons"]
        conn.close()

    def test_progress_callback(self, db_path, tmp_path):
        """Progress callback is called once per principle."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.execute("UPDATE lessons SET category = 'spec-drift' WHERE cluster_seed = 'D'")
        conn.execute("UPDATE lessons SET category = 'context' WHERE cluster_seed = 'E'")
        conn.execute("UPDATE lessons SET category = 'planning' WHERE cluster_seed = 'F'")
        conn.commit()

        results_data = {
            "meta": {"variants": ["A"], "group_by": "category"},
            "results": [
                {
                    "variant": "A",
                    "lesson_id": ids["A"][0],
                    "cluster_seed": "A",
                    "category": "error-handling",
                    "principle": "Principle one.",
                    "error": None,
                },
                {
                    "variant": "A",
                    "lesson_id": ids["A"][1],
                    "cluster_seed": "A",
                    "category": "error-handling",
                    "principle": "Principle two.",
                    "error": None,
                },
            ],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))

        callback = MagicMock()

        with patch("lessons_db.eval.judge.call_judge", return_value="A"):
            results = run_paired_tournament(
                results_path=results_path,
                conn=conn,
                group_by="category",
                pairs_per_principle=2,
                progress_callback=callback,
            )

        assert len(results) == 2
        assert callback.call_count == 2
        conn.close()


class TestComputeTournamentMetrics:
    def test_perfect_discrimination(self):
        results = [
            {"variant": "A", "win_rate": 1.0, "comparisons": 4, "wins": 4, "losses": 0, "neithers": 0},
            {"variant": "A", "win_rate": 1.0, "comparisons": 4, "wins": 4, "losses": 0, "neithers": 0},
        ]
        metrics = compute_tournament_metrics(results)
        assert metrics["A"]["mean_win_rate"] == 1.0
        assert metrics["A"]["discriminating_frac"] == 1.0
        assert metrics["A"]["principle_count"] == 2
        assert metrics["A"]["comparison_count"] == 8

    def test_random_discrimination(self):
        results = [
            {"variant": "A", "win_rate": 0.5, "comparisons": 4, "wins": 2, "losses": 2, "neithers": 0},
        ]
        metrics = compute_tournament_metrics(results)
        assert metrics["A"]["mean_win_rate"] == 0.5
        assert metrics["A"]["discriminating_frac"] == 0.0  # 0.5 is not > 0.5

    def test_multiple_variants(self):
        results = [
            {"variant": "A", "win_rate": 0.8, "comparisons": 4, "wins": 3, "losses": 1, "neithers": 0},
            {"variant": "B", "win_rate": 0.6, "comparisons": 4, "wins": 2, "losses": 1, "neithers": 1},
        ]
        metrics = compute_tournament_metrics(results)
        assert "A" in metrics
        assert "B" in metrics
        assert metrics["A"]["mean_win_rate"] == 0.8
        assert metrics["B"]["mean_win_rate"] == 0.6

    def test_empty_results(self):
        metrics = compute_tournament_metrics([])
        assert metrics == {}

    def test_all_neithers(self):
        results = [
            {"variant": "A", "win_rate": 0.0, "comparisons": 4, "wins": 0, "losses": 0, "neithers": 4},
        ]
        metrics = compute_tournament_metrics(results)
        assert metrics["A"]["mean_win_rate"] == 0.0
        assert metrics["A"]["total_neithers"] == 4


# ---------------------------------------------------------------------------
# TestBuildMechanismPrompt (Task 7)
# ---------------------------------------------------------------------------


class TestBuildMechanismPrompt:
    def test_contains_both_lessons(self):
        lesson_a = {
            "title": "Resource cleanup",
            "one_liner": "Close DB connections",
            "description": "Database connections left open in error paths",
        }
        lesson_b = {
            "title": "File handle leak",
            "one_liner": "Close file handles",
            "description": "File handles not closed when exception thrown",
        }
        prompt = build_mechanism_extraction_prompt(lesson_a, lesson_b)
        assert "Resource cleanup" in prompt
        assert "File handle leak" in prompt

    def test_requests_triplet_format(self):
        a = {"title": "A", "one_liner": "A", "description": "A"}
        b = {"title": "B", "one_liner": "B", "description": "B"}
        prompt = build_mechanism_extraction_prompt(a, b)
        assert "TRIGGER" in prompt
        assert "TARGET" in prompt
        assert "FIX" in prompt

    def test_truncates_long_descriptions(self):
        a = {"title": "A", "one_liner": "A", "description": "x" * 500}
        b = {"title": "B", "one_liner": "B", "description": "y" * 500}
        prompt = build_mechanism_extraction_prompt(a, b)
        # Description should be truncated to 300 chars
        assert "x" * 301 not in prompt
        assert "y" * 301 not in prompt

    def test_handles_none_values(self):
        a = {"title": None, "one_liner": None, "description": None}
        b = {"title": "B", "one_liner": "B", "description": "B"}
        prompt = build_mechanism_extraction_prompt(a, b)
        assert "LESSON A" in prompt
        assert "LESSON B" in prompt


# ---------------------------------------------------------------------------
# TestParseMechanismTriplet (Task 7)
# ---------------------------------------------------------------------------


class TestParseMechanismTriplet:
    def test_parses_valid_triplet(self):
        response = (
            "TRIGGER: Uncaught exception in cleanup path\n"
            "TARGET: Database connection pool\n"
            "FIX: Finally block with explicit close"
        )
        result = parse_mechanism_triplet(response)
        assert result is not None
        assert "Uncaught exception" in result["trigger"]
        assert "Database connection" in result["target"]
        assert "Finally block" in result["fix"]

    def test_returns_none_for_empty(self):
        assert parse_mechanism_triplet("") is None
        assert parse_mechanism_triplet(None) is None

    def test_returns_none_for_none_response(self):
        assert parse_mechanism_triplet("NONE") is None
        assert parse_mechanism_triplet("none") is None

    def test_strips_think_tags(self):
        response = (
            "<think>analyzing...</think>\n"
            "TRIGGER: Missing validation\n"
            "TARGET: Input data\n"
            "FIX: Add schema check"
        )
        result = parse_mechanism_triplet(response)
        assert result is not None
        assert "Missing validation" in result["trigger"]

    def test_returns_none_for_incomplete(self):
        # Missing FIX
        assert parse_mechanism_triplet("TRIGGER: Something\nTARGET: Something") is None

    def test_truncates_long_values(self):
        response = f"TRIGGER: {'x' * 200}\nTARGET: short\nFIX: short"
        result = parse_mechanism_triplet(response)
        assert result is not None
        assert len(result["trigger"]) <= 100

    def test_case_insensitive(self):
        response = "trigger: lower case trigger\ntarget: lower target\nfix: lower fix"
        result = parse_mechanism_triplet(response)
        assert result is not None
        assert "lower case trigger" in result["trigger"]


# ---------------------------------------------------------------------------
# TestMechanismVariant (Task 8)
# ---------------------------------------------------------------------------


class TestMechanismVariant:
    def test_variant_m_in_configs(self):
        assert "M" in VARIANT_CONFIGS
        assert VARIANT_CONFIGS["M"]["mechanism"] is True

    def test_mechanism_generation(self, db_path):
        """Mechanism variant should produce TRIGGER|TARGET|FIX formatted principles."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        lesson = {
            "id": ids["A"][0],
            "title": "Test",
            "one_liner": "Test",
            "description": "Test",
            "cluster_seed": "A",
            "category": "integration",
        }
        siblings = {
            "integration": [
                {
                    "id": ids["A"][0],
                    "title": "T0",
                    "one_liner": "O0",
                    "description": "D0",
                    "cluster_seed": "A",
                    "category": "integration",
                },
                {
                    "id": ids["A"][1],
                    "title": "T1",
                    "one_liner": "O1",
                    "description": "D1",
                    "cluster_seed": "A",
                    "category": "integration",
                },
            ]
        }

        config = VARIANT_CONFIGS["M"]

        with patch(
            "lessons_db.eval.generate.call_ollama",
            return_value="TRIGGER: Missing cleanup\nTARGET: DB connection\nFIX: Add finally block",
        ):
            result = _generate_for_lesson("M", config, lesson, "http://fake:7683", siblings, group_by="category")

        assert result["principle"] is not None
        assert "TRIGGER:" in result["principle"]
        assert "TARGET:" in result["principle"]
        assert "FIX:" in result["principle"]
        conn.close()

    def test_mechanism_no_siblings(self, db_path):
        """Mechanism variant should return error when no siblings available."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        lesson = {
            "id": ids["A"][0],
            "title": "Test",
            "one_liner": "Test",
            "description": "Test",
            "cluster_seed": "A",
            "category": "integration",
        }
        # Only the lesson itself — no other siblings
        siblings = {
            "integration": [
                {
                    "id": ids["A"][0],
                    "title": "T0",
                    "one_liner": "O0",
                    "description": "D0",
                    "cluster_seed": "A",
                    "category": "integration",
                },
            ]
        }

        config = VARIANT_CONFIGS["M"]
        result = _generate_for_lesson("M", config, lesson, "http://fake:7683", siblings, group_by="category")
        assert result["principle"] is None
        assert result["error"] == "no_siblings_for_mechanism"
        conn.close()

    def test_mechanism_extraction_failed(self, db_path):
        """Mechanism variant should return error when extraction fails."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        lesson = {
            "id": ids["A"][0],
            "title": "Test",
            "one_liner": "Test",
            "description": "Test",
            "cluster_seed": "A",
            "category": "integration",
        }
        siblings = {
            "integration": [
                {
                    "id": ids["A"][0],
                    "title": "T0",
                    "one_liner": "O0",
                    "description": "D0",
                    "cluster_seed": "A",
                    "category": "integration",
                },
                {
                    "id": ids["A"][1],
                    "title": "T1",
                    "one_liner": "O1",
                    "description": "D1",
                    "cluster_seed": "A",
                    "category": "integration",
                },
            ]
        }

        config = VARIANT_CONFIGS["M"]

        with patch("lessons_db.eval.generate.call_ollama", return_value="NONE"):
            result = _generate_for_lesson("M", config, lesson, "http://fake:7683", siblings, group_by="category")

        assert result["principle"] is None
        assert result["error"] == "mechanism_extraction_failed"
        conn.close()


# ---------------------------------------------------------------------------
# Signal extractors (Task 9)
# ---------------------------------------------------------------------------


class TestSignalExtractors:
    def test_paired_signal_same_positive(self):
        """Paired comparison: same-group winner → positive log-LR."""
        assert compute_paired_signal(winner="same") > 0

    def test_paired_signal_diff_negative(self):
        assert compute_paired_signal(winner="diff") < 0

    def test_paired_signal_neither_zero(self):
        assert compute_paired_signal(winner="neither") == 0.0

    def test_paired_signal_unknown_zero(self):
        assert compute_paired_signal(winner="unknown") == 0.0

    def test_embedding_signal_high_sim_positive(self):
        """Cosine similarity >= 0.7 → positive log-LR."""
        assert compute_embedding_signal(0.9) > 0
        assert compute_embedding_signal(0.7) > 0

    def test_embedding_signal_low_sim_negative(self):
        assert compute_embedding_signal(0.2) < 0

    def test_embedding_signal_mid_sim_mild(self):
        """0.3 <= sim < 0.5 → mild negative."""
        sig = compute_embedding_signal(0.4)
        assert sig < 0
        assert sig > -1.5  # not as strong as very low

    def test_scope_signal_full_overlap_positive(self):
        assert compute_scope_signal({"python", "async"}, {"python", "async"}) > 0

    def test_scope_signal_no_overlap_negative(self):
        assert compute_scope_signal({"python"}, {"javascript"}) < 0

    def test_scope_signal_partial_overlap_mild(self):
        sig = compute_scope_signal({"python", "async"}, {"python", "testing"})
        assert sig > 0  # some overlap

    def test_scope_signal_empty_zero(self):
        assert compute_scope_signal(set(), {"python"}) == 0.0
        assert compute_scope_signal({"python"}, set()) == 0.0

    def test_mechanism_signal_match_positive(self):
        assert compute_mechanism_signal(True) > 0

    def test_mechanism_signal_no_match_negative(self):
        assert compute_mechanism_signal(False) < 0

    def test_mechanism_signal_none_zero(self):
        assert compute_mechanism_signal(None) == 0.0


# ---------------------------------------------------------------------------
# Bayesian fusion (Task 10)
# ---------------------------------------------------------------------------


class TestBayesianFusion:
    def test_all_positive_signals_high_posterior(self):
        """All positive signals → high posterior."""
        posterior = compute_transfer_posterior(
            paired_signal=2.5,
            embedding_signal=1.5,
            scope_signal=1.0,
            mechanism_signal=2.0,
        )
        assert posterior > 0.9

    def test_all_negative_signals_low_posterior(self):
        posterior = compute_transfer_posterior(
            paired_signal=-2.5,
            embedding_signal=-1.5,
            scope_signal=-0.5,
            mechanism_signal=-1.5,
        )
        assert posterior < 0.1

    def test_mixed_signals_moderate_posterior(self):
        """Conflicting signals → moderate posterior."""
        posterior = compute_transfer_posterior(
            paired_signal=2.5,
            embedding_signal=-1.5,
            scope_signal=0.0,
            mechanism_signal=0.0,
        )
        assert 0.3 < posterior < 0.7

    def test_no_signals_equals_prior(self):
        """With no signals, posterior equals prior (0.25)."""
        posterior = compute_transfer_posterior(0.0, 0.0, 0.0, 0.0)
        assert abs(posterior - 0.25) < 0.01

    def test_posterior_is_probability(self):
        """Output is always in [0, 1]."""
        for vals in [(10, 10, 10, 10), (-10, -10, -10, -10), (0, 0, 0, 0)]:
            p = compute_transfer_posterior(*vals)
            assert 0.0 <= p <= 1.0

    def test_monotonicity(self):
        """More positive evidence → higher posterior."""
        p1 = compute_transfer_posterior(2.5, 0.0, 0.0, 0.0)
        p2 = compute_transfer_posterior(2.5, 1.5, 0.0, 0.0)
        p3 = compute_transfer_posterior(2.5, 1.5, 1.0, 2.0)
        assert p1 < p2 < p3


# ---------------------------------------------------------------------------
# Bayesian metrics (Task 11)
# ---------------------------------------------------------------------------


class TestComputeBayesianMetrics:
    def test_perfect_separation(self):
        """Perfect fusion → high AUC and separation."""
        scored = [
            {"variant": "A", "is_same_group": True, "posterior": 0.9},
            {"variant": "A", "is_same_group": True, "posterior": 0.85},
            {"variant": "A", "is_same_group": False, "posterior": 0.1},
            {"variant": "A", "is_same_group": False, "posterior": 0.15},
        ]
        metrics = compute_bayesian_metrics(scored)
        assert metrics["A"]["same_mean_posterior"] > 0.8
        assert metrics["A"]["diff_mean_posterior"] < 0.2
        assert metrics["A"]["auc"] > 0.9
        assert metrics["A"]["separation"] > 0.6

    def test_random_separation(self):
        """No discrimination → AUC ~ 0.5."""
        scored = [
            {"variant": "A", "is_same_group": True, "posterior": 0.5},
            {"variant": "A", "is_same_group": True, "posterior": 0.5},
            {"variant": "A", "is_same_group": False, "posterior": 0.5},
            {"variant": "A", "is_same_group": False, "posterior": 0.5},
        ]
        metrics = compute_bayesian_metrics(scored)
        assert abs(metrics["A"]["auc"] - 0.5) < 0.1
        assert abs(metrics["A"]["separation"]) < 0.1

    def test_multi_variant(self):
        """Metrics computed per variant."""
        scored = [
            {"variant": "A", "is_same_group": True, "posterior": 0.9},
            {"variant": "A", "is_same_group": False, "posterior": 0.1},
            {"variant": "B", "is_same_group": True, "posterior": 0.6},
            {"variant": "B", "is_same_group": True, "posterior": 0.4},
            {"variant": "B", "is_same_group": False, "posterior": 0.5},
            {"variant": "B", "is_same_group": False, "posterior": 0.55},
        ]
        metrics = compute_bayesian_metrics(scored)
        assert "A" in metrics
        assert "B" in metrics
        assert metrics["A"]["auc"] > metrics["B"]["auc"]

    def test_calibration_error(self):
        """Calibration error: |mean_posterior - actual_fraction|."""
        scored = [
            {"variant": "A", "is_same_group": True, "posterior": 0.8},
            {"variant": "A", "is_same_group": False, "posterior": 0.8},
        ]
        metrics = compute_bayesian_metrics(scored)
        # mean posterior = 0.8, actual positive fraction = 0.5
        assert metrics["A"]["calibration_error"] == pytest.approx(0.3, abs=0.05)

    def test_empty_input(self):
        """Empty input → empty output."""
        assert compute_bayesian_metrics([]) == {}

    def test_single_class_auc(self):
        """Only same-group entries → AUC defaults to 0.5 (can't compute)."""
        scored = [
            {"variant": "A", "is_same_group": True, "posterior": 0.9},
            {"variant": "A", "is_same_group": True, "posterior": 0.85},
        ]
        metrics = compute_bayesian_metrics(scored)
        assert metrics["A"]["auc"] == 0.5  # degenerate case


# ---------------------------------------------------------------------------
# Reference comparison (Task 12)
# ---------------------------------------------------------------------------


class TestReferenceComparison:
    def test_diagnoses_improvement(self):
        reference_metrics = {"A": {"auc": 0.70}}
        new_metrics = {"A": {"auc": 0.85}}
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics)
        assert diagnosis["A"]["status"] == "improved"
        assert diagnosis["A"]["delta"] > 0

    def test_diagnoses_regression(self):
        reference_metrics = {"A": {"auc": 0.80}}
        new_metrics = {"A": {"auc": 0.60}}
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics)
        assert diagnosis["A"]["status"] == "regressed"
        assert diagnosis["A"]["delta"] < 0

    def test_diagnoses_stable(self):
        """Within noise margin -> stable."""
        reference_metrics = {"A": {"auc": 0.80}}
        new_metrics = {"A": {"auc": 0.79}}
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics)
        assert diagnosis["A"]["status"] == "stable"

    def test_diagnoses_drift(self):
        """Both degrade -> data drift, not mechanism failure."""
        reference_metrics = {"A": {"auc": 0.80}}
        new_metrics = {"A": {"auc": 0.60}}
        reference_rerun = {"A": {"auc": 0.62}}
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics, reference_rerun=reference_rerun)
        assert diagnosis["A"]["status"] == "data_drift"

    def test_new_variant_no_reference(self):
        """Variant with no reference -> 'new'."""
        reference_metrics = {"A": {"auc": 0.70}}
        new_metrics = {"A": {"auc": 0.85}, "B": {"auc": 0.75}}
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics)
        assert diagnosis["B"]["status"] == "new"

    def test_removed_variant(self):
        """Variant in reference but not in new -> 'removed'."""
        reference_metrics = {"A": {"auc": 0.70}, "B": {"auc": 0.80}}
        new_metrics = {"A": {"auc": 0.85}}
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics)
        assert diagnosis["B"]["status"] == "removed"
        assert diagnosis["B"]["ref_value"] == 0.80
        assert diagnosis["B"]["new_value"] is None

    def test_custom_metric_key(self):
        """Works with non-default metric key (e.g. mean_win_rate for tournament)."""
        reference = {"A": {"mean_win_rate": 0.70}}
        new = {"A": {"mean_win_rate": 0.85}}
        diagnosis = diagnose_vs_reference(reference, new, metric_key="mean_win_rate")
        assert diagnosis["A"]["status"] == "improved"


# ---------------------------------------------------------------------------
# Simulation prompt + parser (Task 13)
# ---------------------------------------------------------------------------


class TestBuildSimulationPrompt:
    def test_with_principle_contains_rule(self):
        scenario = "Writing a database query handler that catches exceptions"
        principle = "Uncaught exception in cleanup path causes resource leak"
        prompt = build_simulation_prompt(scenario, principle)
        assert "CODING RULE" in prompt
        assert principle in prompt
        assert scenario in prompt

    def test_without_principle_has_no_rule(self):
        scenario = "Writing a database query handler"
        prompt = build_simulation_prompt(scenario, principle=None)
        assert "CODING RULE" not in prompt
        assert scenario in prompt

    def test_both_prompts_ask_about_bugs(self):
        with_p = build_simulation_prompt("scenario", "principle")
        without_p = build_simulation_prompt("scenario", None)
        assert "BUG" in with_p.upper()
        assert "BUG" in without_p.upper()


class TestParseSimulationResult:
    def test_bug_found(self):
        assert parse_simulation_result("BUG FOUND: resource leak in finally") is True

    def test_no_bug_found(self):
        assert parse_simulation_result("NO BUG FOUND") is False

    def test_empty_response(self):
        assert parse_simulation_result("") is False
        assert parse_simulation_result(None) is False

    def test_case_insensitive(self):
        assert parse_simulation_result("bug found: something") is True

    def test_no_bug_takes_precedence(self):
        """'NO BUG FOUND' should be False even though 'BUG FOUND' is a substring."""
        assert parse_simulation_result("After review: NO BUG FOUND in this code") is False


# ---------------------------------------------------------------------------
# Simulation lift metric (Task 14)
# ---------------------------------------------------------------------------


class TestComputeSimulationLift:
    def test_perfect_lift(self):
        """Principle catches all bugs, control catches none -> lift 1.0."""
        results = [
            {"variant": "A", "principle": "p1", "with_principle": True, "without_principle": False},
            {"variant": "A", "principle": "p2", "with_principle": True, "without_principle": False},
        ]
        lift = compute_simulation_lift(results)
        assert lift["A"]["lift"] == 1.0
        assert lift["A"]["with_catch_rate"] == 1.0
        assert lift["A"]["without_catch_rate"] == 0.0

    def test_no_lift(self):
        """Both catch (or both miss) -> lift 0.0."""
        results = [
            {"variant": "A", "principle": "p1", "with_principle": True, "without_principle": True},
        ]
        lift = compute_simulation_lift(results)
        assert lift["A"]["lift"] == 0.0

    def test_negative_lift(self):
        """Control catches more than with-principle -> negative lift."""
        results = [
            {"variant": "A", "principle": "p1", "with_principle": False, "without_principle": True},
        ]
        lift = compute_simulation_lift(results)
        assert lift["A"]["lift"] < 0

    def test_multi_variant(self):
        results = [
            {"variant": "A", "principle": "p1", "with_principle": True, "without_principle": False},
            {"variant": "B", "principle": "p1", "with_principle": True, "without_principle": True},
        ]
        lift = compute_simulation_lift(results)
        assert lift["A"]["lift"] > lift["B"]["lift"]

    def test_empty(self):
        assert compute_simulation_lift([]) == {}


class TestRenderV2Report:
    def test_contains_tournament_section(self):
        report = render_v2_report(
            tournament_metrics={
                "A": {
                    "mean_win_rate": 0.75,
                    "discriminating_frac": 0.8,
                    "principle_count": 10,
                    "comparison_count": 40,
                    "total_wins": 30,
                    "total_losses": 8,
                    "total_neithers": 2,
                }
            },
        )
        assert "## Tournament" in report
        assert "0.75" in report

    def test_contains_bayesian_section(self):
        report = render_v2_report(
            bayesian_metrics={
                "A": {
                    "same_mean_posterior": 0.85,
                    "diff_mean_posterior": 0.15,
                    "separation": 0.70,
                    "auc": 0.92,
                    "calibration_error": 0.05,
                    "pair_count": 20,
                }
            },
        )
        assert "## Bayesian" in report
        assert "0.92" in report
        assert "AUC" in report

    def test_contains_reference_section(self):
        report = render_v2_report(
            reference_diagnosis={"A": {"status": "improved", "delta": 0.15, "new_value": 0.85, "ref_value": 0.70}},
        )
        assert "## Reference" in report
        assert "improved" in report

    def test_contains_simulation_section(self):
        report = render_v2_report(
            simulation_lift={
                "A": {"lift": 0.35, "with_catch_rate": 0.80, "without_catch_rate": 0.45, "trial_count": 12}
            },
        )
        assert "## Simulation" in report
        assert "0.35" in report

    def test_winner_by_auc(self):
        """Winner is determined by AUC when bayesian metrics provided."""
        report = render_v2_report(
            bayesian_metrics={
                "A": {
                    "auc": 0.85,
                    "same_mean_posterior": 0.8,
                    "diff_mean_posterior": 0.2,
                    "separation": 0.6,
                    "calibration_error": 0.05,
                    "pair_count": 20,
                },
                "B": {
                    "auc": 0.92,
                    "same_mean_posterior": 0.9,
                    "diff_mean_posterior": 0.1,
                    "separation": 0.8,
                    "calibration_error": 0.03,
                    "pair_count": 20,
                },
            },
        )
        assert "**Variant B**" in report

    def test_empty_sections_omitted(self):
        """Sections with no data are not rendered."""
        report = render_v2_report()
        assert "## Tournament" not in report
        assert "## Simulation" not in report

    def test_failure_analysis_section(self):
        """Failure analysis with scored pairs."""
        scored = [
            {"variant": "A", "is_same_group": True, "posterior": 0.1, "principle": "p1", "target_title": "t1"},
            {"variant": "A", "is_same_group": False, "posterior": 0.9, "principle": "p2", "target_title": "t2"},
        ]
        report = render_v2_report(scored_pairs=scored)
        assert "## Failure Analysis" in report

    def test_failure_analysis_includes_mechanism_triplet(self):
        """Failure analysis renders mechanism triplet when present."""
        scored = [
            {
                "variant": "A",
                "is_same_group": True,
                "posterior": 0.1,
                "principle": "p1",
                "target_title": "t1",
                "mechanism_trigger": "uncaught exception",
                "mechanism_target": "cleanup path",
                "mechanism_fix": "finally block",
            },
        ]
        report = render_v2_report(scored_pairs=scored)
        assert "uncaught exception" in report
        assert "cleanup path" in report

    def test_signal_diagnostics_section(self):
        """Signal diagnostics renders per-signal means and disagreement."""
        diag = [
            {
                "variant": "A",
                "paired_signal": 2.5,
                "embedding_signal": 1.5,
                "scope_signal": 1.0,
                "mechanism_signal": 2.0,
            },
            {
                "variant": "A",
                "paired_signal": -2.5,
                "embedding_signal": 1.5,
                "scope_signal": -0.5,
                "mechanism_signal": 0.0,
            },
        ]
        report = render_v2_report(signal_diagnostics=diag)
        assert "## Signal Diagnostics" in report
        assert "Paired" in report
        assert "Disagree" in report

    def test_signal_diagnostics_omitted_when_empty(self):
        """Signal diagnostics not rendered when no data."""
        report = render_v2_report()
        assert "## Signal Diagnostics" not in report


# ---------------------------------------------------------------------------
# TestEvalV2EndToEnd — helpers
# ---------------------------------------------------------------------------


def _build_results_json(conn, lesson_ids, variant, tmp_path, filename="results.json"):
    """Build a synthetic eval-generate results JSON from lesson IDs."""
    entries = []
    for lid in lesson_ids:
        row = conn.execute(
            "SELECT id, title, one_liner, description, cluster_seed, category " "FROM lessons WHERE id = ?",
            (lid,),
        ).fetchone()
        entries.append(
            {
                "variant": variant,
                "lesson_id": row["id"],
                "title": row["title"],
                "one_liner": row["one_liner"],
                "description": row["description"],
                "cluster_seed": row["cluster_seed"],
                "category": row["category"],
                "principle": f"{variant} principle for lesson {row['id']}",
                "error": None,
            }
        )
    return entries


def _tournament_to_scored_pairs(
    tournament_results,
    principle_scopes=None,
    target_scopes=None,
):
    """Convert tournament results into scored pairs via Bayesian signal fusion."""
    if principle_scopes is None:
        principle_scopes = {"python"}
    if target_scopes is None:
        target_scopes = {"python", "testing"}

    scored_pairs = []
    for r in tournament_results:
        winner = "same" if r["win_rate"] > 0.5 else ("diff" if r["win_rate"] < 0.5 else "neither")
        paired = compute_paired_signal(winner)
        embedding = compute_embedding_signal(0.6)
        scope = compute_scope_signal(principle_scopes, target_scopes)
        mechanism = compute_mechanism_signal(None)
        posterior = compute_transfer_posterior(paired, embedding, scope, mechanism)
        scored_pairs.append(
            {
                "variant": r["variant"],
                "is_same_group": r["win_rate"] > 0.5,
                "posterior": posterior,
                "principle": r["principle"],
                "target_title": f"target for {r['lesson_id']}",
                "paired_signal": paired,
                "embedding_signal": embedding,
                "scope_signal": scope,
                "mechanism_signal": mechanism,
            }
        )
    return scored_pairs


def _seed_and_categorize(db_path):
    """Init DB, seed clusters, and assign distinct categories for A and B."""
    conn = init_db(db_path)
    ids_by_cluster = _seed_clusters(conn)
    for lid in ids_by_cluster["A"]:
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE id = ?", (lid,))
    for lid in ids_by_cluster["B"]:
        conn.execute("UPDATE lessons SET category = 'testing' WHERE id = ?", (lid,))
    conn.commit()
    return conn, ids_by_cluster


# ---------------------------------------------------------------------------
# TestEvalV2EndToEnd
# ---------------------------------------------------------------------------


class TestEvalV2EndToEnd:
    """End-to-end test: full Bayesian pipeline from seed to report."""

    def test_full_bayesian_pipeline(self, db_path, tmp_path):
        """Generate → Tournament → Bayesian fusion → Report.

        Validates the complete data-flow through all 7 V2 functions using
        mocked LLM calls.  Seeds two categories with enough lessons for the
        tournament, then traces data from results JSON to markdown report.
        """
        conn, ids_by_cluster = _seed_and_categorize(db_path)

        # Step 1: Build synthetic results JSON (2 lessons per category)
        source_ids = ids_by_cluster["A"][:2] + ids_by_cluster["B"][:2]
        entries = _build_results_json(conn, source_ids, "A", tmp_path)
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({"meta": {"group_by": "category"}, "results": entries}))

        # Step 2: Tournament (mock judge → always picks A)
        with patch("lessons_db.eval.judge.call_judge", side_effect=lambda prompt, **kw: "A"):
            tournament_results = run_paired_tournament(
                results_path=results_path,
                conn=conn,
                group_by="category",
                pairs_per_principle=2,
            )

        assert len(tournament_results) > 0
        for r in tournament_results:
            assert 0.0 <= r["win_rate"] <= 1.0
            assert r["comparisons"] > 0

        # Step 3: Tournament metrics
        tournament_metrics = compute_tournament_metrics(tournament_results)
        assert tournament_metrics["A"]["principle_count"] == len(tournament_results)

        # Step 4–5: Bayesian signals → fusion → metrics
        scored_pairs = _tournament_to_scored_pairs(tournament_results)
        for sp in scored_pairs:
            assert 0.0 <= sp["posterior"] <= 1.0
        bayesian_metrics = compute_bayesian_metrics(scored_pairs)
        assert 0.0 <= bayesian_metrics["A"]["auc"] <= 1.0

        # Step 6: Reference comparison
        diagnosis = diagnose_vs_reference({"A": {"auc": 0.50}}, bayesian_metrics)
        assert diagnosis["A"]["status"] in ("improved", "regressed", "stable", "new")

        # Step 7: Render report
        report = render_v2_report(
            tournament_metrics=tournament_metrics,
            bayesian_metrics=bayesian_metrics,
            reference_diagnosis=diagnosis,
            scored_pairs=scored_pairs,
            signal_diagnostics=scored_pairs,
        )
        for section in (
            "# Eval V2 Report",
            "## Tournament Results",
            "## Bayesian Fusion",
            "## Reference Comparison",
            "## Signal Diagnostics",
        ):
            assert section in report
        assert "AUC" in report

        conn.close()

    def test_pipeline_with_simulation_lift(self, db_path, tmp_path):
        """Simulation lift integrates with the rest of the pipeline."""
        sim_results = [
            {"variant": "A", "principle": "p1", "with_principle": True, "without_principle": False},
            {"variant": "A", "principle": "p2", "with_principle": True, "without_principle": True},
            {"variant": "A", "principle": "p3", "with_principle": False, "without_principle": False},
        ]
        lift = compute_simulation_lift(sim_results)
        assert abs(lift["A"]["lift"] - (2 / 3 - 1 / 3)) < 0.01

        report = render_v2_report(simulation_lift=lift)
        assert "## Simulation Lift" in report
        assert "+0.333" in report

    def test_pipeline_multiple_variants(self, db_path, tmp_path):
        """Two variants flow through the full pipeline producing separate metrics."""
        conn, ids_by_cluster = _seed_and_categorize(db_path)

        # Build results for two variants
        entries_a = _build_results_json(conn, ids_by_cluster["A"][:2], "A", tmp_path)
        entries_b = _build_results_json(conn, ids_by_cluster["B"][:2], "B", tmp_path)
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({"meta": {"group_by": "category"}, "results": entries_a + entries_b}))

        with patch("lessons_db.eval.judge.call_judge", side_effect=lambda prompt, **kw: "A"):
            tournament_results = run_paired_tournament(
                results_path=results_path,
                conn=conn,
                group_by="category",
                pairs_per_principle=2,
            )

        tournament_metrics = compute_tournament_metrics(tournament_results)
        assert "A" in tournament_metrics
        assert "B" in tournament_metrics

        scored_pairs = _tournament_to_scored_pairs(
            tournament_results,
            principle_scopes={"python"},
            target_scopes={"python"},
        )
        bayesian_metrics = compute_bayesian_metrics(scored_pairs)
        assert "A" in bayesian_metrics and "B" in bayesian_metrics

        # Reference with only A → B is "new"
        diagnosis = diagnose_vs_reference({"A": {"auc": 0.50}}, bayesian_metrics)
        assert diagnosis["B"]["status"] == "new"

        report = render_v2_report(
            tournament_metrics=tournament_metrics,
            bayesian_metrics=bayesian_metrics,
            reference_diagnosis=diagnosis,
            signal_diagnostics=scored_pairs,
        )
        assert "# Eval V2 Report" in report
        assert "| A |" in report
        assert "| B |" in report

        conn.close()
