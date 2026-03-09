"""Tests for eval pipeline: config, variant definitions, test set selection, generation."""

import json
from unittest.mock import MagicMock, patch

from lessons_db.config import DATA_DIR, EVAL_DIR
from lessons_db.db import init_db, insert_lesson
from lessons_db.eval import (
    DEFAULT_BINARY_JUDGE_MODEL,
    DEFAULT_JUDGE_MODEL,
    VARIANT_CONFIGS,
    _clean_principle,
    _select_diverse,
    build_binary_judge_prompt,
    build_generation_prompt,
    build_judge_prompt,
    call_judge,
    call_ollama,
    compute_metrics,
    parse_binary_judge,
    parse_judge_scores,
    render_report,
    run_eval_generate,
    run_eval_judge,
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
    def test_has_eight_variants(self):
        assert set(VARIANT_CONFIGS.keys()) == {"A", "B", "C", "D", "E", "F", "G", "H"}

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
        assert payload["_timeout"] == 300  # default timeout

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

        with patch("lessons_db.eval.call_judge", side_effect=mock_judge):
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
        with patch("lessons_db.eval.call_judge", return_value=None):
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

        with patch("lessons_db.eval.call_judge", side_effect=mock_judge):
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

        with patch("lessons_db.eval.call_judge", return_value=None):
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
