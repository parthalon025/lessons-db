"""Tests for eval/analysis.py — per-lesson breakdown and failure case extraction."""

from lessons_db.eval.analysis import (
    bootstrap_f1_ci,
    compute_per_lesson_breakdown,
    compute_stability,
    describe_prompt_diff,
    extract_failure_cases,
    propose_next_variant,
)

# ---------------------------------------------------------------------------
# Helpers — scored pair builders
# ---------------------------------------------------------------------------


def _binary_pair(
    variant: str,
    source_lesson_id: int,
    principle: str,
    target_id: int,
    target_title: str,
    is_same_cluster: bool,
    matched: bool,
    *,
    cluster_seed: str = "cluster-A",
    target_cluster_seed: str | None = None,
) -> dict:
    return {
        "variant": variant,
        "source_lesson_id": source_lesson_id,
        "principle": principle,
        "target_id": target_id,
        "target_title": target_title,
        "is_same_cluster": is_same_cluster,
        "cluster_seed": cluster_seed,
        "target_cluster_seed": target_cluster_seed or ("cluster-A" if is_same_cluster else "cluster-B"),
        "scores": {"matched": matched},
    }


def _rubric_pair(
    variant: str,
    source_lesson_id: int,
    principle: str,
    target_id: int,
    target_title: str,
    is_same_cluster: bool,
    transfer: int,
    *,
    precision: int = 3,
    actionability: int = 3,
    cluster_seed: str = "cluster-A",
    target_cluster_seed: str | None = None,
) -> dict:
    return {
        "variant": variant,
        "source_lesson_id": source_lesson_id,
        "principle": principle,
        "target_id": target_id,
        "target_title": target_title,
        "is_same_cluster": is_same_cluster,
        "cluster_seed": cluster_seed,
        "target_cluster_seed": target_cluster_seed or ("cluster-A" if is_same_cluster else "cluster-B"),
        "scores": {"transfer": transfer, "precision": precision, "actionability": actionability},
    }


# ---------------------------------------------------------------------------
# TestPerLessonBreakdown
# ---------------------------------------------------------------------------


class TestPerLessonBreakdown:
    def test_groups_by_source_lesson(self):
        """Each distinct source_lesson_id produces one breakdown entry."""
        pairs = [
            _binary_pair("A", 1, "principle-1", 10, "target-10", True, True),
            _binary_pair("A", 1, "principle-1", 11, "target-11", False, False),
            _binary_pair("A", 2, "principle-2", 12, "target-12", True, True),
        ]
        expected_lessons = {p["source_lesson_id"] for p in pairs}
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == len(expected_lessons)
        lesson_ids = {r["source_lesson_id"] for r in result}
        assert lesson_ids == expected_lessons

    def test_computes_per_lesson_f1(self):
        """Perfect pair: TP on same-cluster, TN on diff-cluster -> F1=1.0."""
        pairs = [
            _binary_pair("A", 1, "principle-1", 10, "target-10", True, True),  # TP
            _binary_pair("A", 1, "principle-1", 11, "target-11", False, False),  # TN
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 1
        entry = result[0]
        assert entry["f1"] == 1.0
        assert entry["tp"] == 1
        assert entry["fn"] == 0
        assert entry["fp"] == 0
        assert entry["total_pairs"] == 2

    def test_sorted_worst_first(self):
        """Lesson with F1=0.0 appears before lesson with F1=1.0."""
        pairs = [
            # Lesson 1: perfect (TP same, TN diff) -> F1=1.0
            _binary_pair("A", 1, "good-principle", 10, "t-10", True, True),
            _binary_pair("A", 1, "good-principle", 11, "t-11", False, False),
            # Lesson 2: total miss (FN same, FP diff) -> F1=0.0
            _binary_pair("A", 2, "bad-principle", 12, "t-12", True, False),
            _binary_pair("A", 2, "bad-principle", 13, "t-13", False, True),
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 2
        assert result[0]["f1"] == 0.0
        assert result[0]["source_lesson_id"] == 2
        assert result[1]["f1"] == 1.0
        assert result[1]["source_lesson_id"] == 1

    def test_includes_variant(self):
        """Result includes the variant field."""
        pairs = [
            _binary_pair("F", 5, "principle-F", 20, "t-20", True, True),
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 1
        assert result[0]["variant"] == "F"

    def test_only_diff_cluster_pairs(self):
        """Lesson with only diff-cluster pairs: recall=0.0 (no same-cluster ground truth)."""
        pairs = [
            _binary_pair("A", 1, "p1", 10, "t-10", False, True),  # FP
            _binary_pair("A", 1, "p1", 11, "t-11", False, False),  # TN
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 1
        assert result[0]["recall"] == 0.0
        assert result[0]["tp"] == 0

    def test_only_same_cluster_pairs(self):
        """Lesson with only same-cluster pairs: precision=0.0 guard (no FP possible)."""
        pairs = [
            _binary_pair("A", 1, "p1", 10, "t-10", True, True),  # TP
            _binary_pair("A", 1, "p1", 11, "t-11", True, False),  # FN
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 1
        # tp=1, fp=0 → precision = 1/(1+0) = 1.0 (not division by zero)
        assert result[0]["precision"] == 1.0
        assert result[0]["fp"] == 0

    def test_empty_pairs(self):
        """Empty input returns empty list."""
        assert compute_per_lesson_breakdown([]) == []


# ---------------------------------------------------------------------------
# TestExtractFailureCases
# ---------------------------------------------------------------------------


class TestExtractFailureCases:
    def test_finds_false_positives(self):
        """Diff-cluster pair with matched=True is a false positive."""
        pairs = [
            _binary_pair(
                "A", 1, "p1", 10, "t-10", False, True, cluster_seed="cluster-A", target_cluster_seed="cluster-B"
            ),
        ]
        result = extract_failure_cases(pairs)
        assert len(result) == 1
        assert result[0]["failure_type"] == "false_positive"
        assert result[0]["target_id"] == 10

    def test_finds_false_negatives(self):
        """Same-cluster pair with matched=False is a false negative."""
        pairs = [
            _binary_pair("A", 1, "p1", 10, "t-10", True, False),
        ]
        result = extract_failure_cases(pairs)
        assert len(result) == 1
        assert result[0]["failure_type"] == "false_negative"

    def test_correct_pairs_excluded(self):
        """TP (same+matched) and TN (diff+not matched) are not failures."""
        pairs = [
            _binary_pair("A", 1, "p1", 10, "t-10", True, True),  # TP
            _binary_pair("A", 1, "p1", 11, "t-11", False, False),  # TN
        ]
        result = extract_failure_cases(pairs)
        assert result == []

    def test_rubric_mode_threshold(self):
        """In rubric mode, transfer >= 3 on diff-cluster is a false positive."""
        pairs = [
            _rubric_pair("B", 3, "p3", 30, "t-30", False, transfer=4),  # FP (diff + transfer>=3)
            _rubric_pair("B", 3, "p3", 31, "t-31", True, transfer=2),  # FN (same + transfer<3)
            _rubric_pair("B", 3, "p3", 32, "t-32", True, transfer=3),  # TP (same + transfer>=3)
            _rubric_pair("B", 3, "p3", 33, "t-33", False, transfer=1),  # TN (diff + transfer<3)
        ]
        result = extract_failure_cases(pairs)
        assert len(result) == 2
        types = {r["failure_type"] for r in result}
        assert types == {"false_positive", "false_negative"}
        fp = next(r for r in result if r["failure_type"] == "false_positive")
        assert fp["target_id"] == 30
        fn = next(r for r in result if r["failure_type"] == "false_negative")
        assert fn["target_id"] == 31

    def test_empty_pairs(self):
        """Empty input returns empty list."""
        assert extract_failure_cases([]) == []


# ---------------------------------------------------------------------------
# TestBootstrapF1CI
# ---------------------------------------------------------------------------


class TestBootstrapF1CI:
    def test_returns_low_mid_high(self):
        scored_pairs = [
            {"variant": "A", "is_same_cluster": True, "scores": {"matched": True}},
            {"variant": "A", "is_same_cluster": True, "scores": {"matched": True}},
            {"variant": "A", "is_same_cluster": True, "scores": {"matched": False}},
            {"variant": "A", "is_same_cluster": False, "scores": {"matched": False}},
            {"variant": "A", "is_same_cluster": False, "scores": {"matched": False}},
            {"variant": "A", "is_same_cluster": False, "scores": {"matched": True}},
        ]
        result = bootstrap_f1_ci(scored_pairs, variant="A", n_bootstrap=100, seed=42)
        assert "low" in result
        assert "mid" in result
        assert "high" in result
        assert result["low"] <= result["mid"] <= result["high"]

    def test_perfect_score_narrow_ci(self):
        scored_pairs = [
            {"variant": "A", "is_same_cluster": True, "scores": {"matched": True}},
            {"variant": "A", "is_same_cluster": False, "scores": {"matched": False}},
        ] * 10  # 20 perfect pairs
        result = bootstrap_f1_ci(scored_pairs, variant="A", n_bootstrap=200, seed=42)
        assert result["low"] >= 0.9  # should be very tight around 1.0

    def test_filters_by_variant(self):
        scored_pairs = [
            {"variant": "A", "is_same_cluster": True, "scores": {"matched": True}},
            {"variant": "B", "is_same_cluster": True, "scores": {"matched": False}},
            {"variant": "A", "is_same_cluster": False, "scores": {"matched": False}},
            {"variant": "B", "is_same_cluster": False, "scores": {"matched": True}},
        ]
        result_a = bootstrap_f1_ci(scored_pairs, variant="A", n_bootstrap=100, seed=42)
        result_b = bootstrap_f1_ci(scored_pairs, variant="B", n_bootstrap=100, seed=42)
        assert result_a["mid"] > result_b["mid"]  # A is perfect, B is zero

    def test_empty_pairs_returns_zeros(self):
        result = bootstrap_f1_ci([], variant="A")
        assert result == {"low": 0.0, "mid": 0.0, "high": 0.0}

    def test_reproducible_with_seed(self):
        scored_pairs = [
            {"variant": "A", "is_same_cluster": True, "scores": {"matched": True}},
            {"variant": "A", "is_same_cluster": True, "scores": {"matched": False}},
            {"variant": "A", "is_same_cluster": False, "scores": {"matched": False}},
            {"variant": "A", "is_same_cluster": False, "scores": {"matched": True}},
        ] * 5
        r1 = bootstrap_f1_ci(scored_pairs, variant="A", seed=123)
        r2 = bootstrap_f1_ci(scored_pairs, variant="A", seed=123)
        assert r1 == r2

    def test_single_bootstrap_iteration(self):
        """n_bootstrap=1 should not crash or produce negative indices."""
        pairs = [
            {"variant": "A", "is_same_cluster": True, "scores": {"matched": True}},
            {"variant": "A", "is_same_cluster": False, "scores": {"matched": False}},
        ]
        result = bootstrap_f1_ci(pairs, variant="A", n_bootstrap=1, seed=0)
        assert result["low"] <= result["mid"] <= result["high"]
        assert all(0.0 <= v <= 1.0 for v in result.values())


# ---------------------------------------------------------------------------
# TestComputeStability
# ---------------------------------------------------------------------------


class TestComputeStability:
    def test_returns_stdev_per_variant(self):
        entries = [
            {"variant": "A", "f1": 0.28, "date": "2026-03-08"},
            {"variant": "A", "f1": 0.30, "date": "2026-03-09"},
            {"variant": "A", "f1": 0.25, "date": "2026-03-10"},
        ]
        stability = compute_stability(entries)
        assert "A" in stability
        assert "stdev" in stability["A"]
        assert stability["A"]["stdev"] > 0

    def test_single_run_zero_stdev(self):
        entries = [{"variant": "B", "f1": 0.40, "date": "2026-03-09"}]
        stability = compute_stability(entries)
        assert stability["B"]["stdev"] == 0.0

    def test_skips_ablation_entries(self):
        entries = [
            {"variant": "A", "f1": 0.28, "date": "2026-03-09"},
            {"type": "ablations", "date": "2026-03-09", "ablations": []},
        ]
        stability = compute_stability(entries)
        assert len(stability) == 1

    def test_flags_unstable_variants(self):
        entries = [
            {"variant": "A", "f1": 0.10, "date": "d1"},
            {"variant": "A", "f1": 0.90, "date": "d2"},
        ]
        stability = compute_stability(entries)
        assert stability["A"]["stable"] is False

    def test_flags_stable_variants(self):
        entries = [
            {"variant": "A", "f1": 0.50, "date": "d1"},
            {"variant": "A", "f1": 0.52, "date": "d2"},
            {"variant": "A", "f1": 0.49, "date": "d3"},
        ]
        stability = compute_stability(entries)
        assert stability["A"]["stable"] is True

    def test_skips_entries_without_f1(self):
        """Entries with missing or None f1 are silently skipped."""
        entries = [
            {"variant": "A", "f1": 0.50},
            {"variant": "A"},
            {"variant": "A", "f1": None},
        ]
        stability = compute_stability(entries)
        assert stability["A"]["n_runs"] == 1
        assert stability["A"]["stdev"] == 0.0


# ---------------------------------------------------------------------------
# TestDescribePromptDiff
# ---------------------------------------------------------------------------


class TestDescribePromptDiff:
    def test_describes_contrastive_addition(self):
        configs = {
            "B": {
                "prompt_id": "zero-shot-causal",
                "model": "deepseek-r1:8b",
                "temperature": 0.6,
                "num_ctx": 8192,
                "chunked": False,
            },
            "F": {
                "prompt_id": "contrastive",
                "model": "deepseek-r1:8b",
                "temperature": 0.6,
                "num_ctx": 8192,
                "chunked": False,
                "contrastive": True,
            },
        }
        diff = describe_prompt_diff("B", "F", configs)
        assert "contrastive" in diff.lower()
        assert "boundary" in diff.lower() or "scope" in diff.lower() or "not apply" in diff.lower()

    def test_describes_model_change(self):
        configs = {
            "B": {
                "prompt_id": "zero-shot-causal",
                "model": "deepseek-r1:8b",
                "temperature": 0.6,
                "num_ctx": 8192,
                "chunked": False,
            },
            "D": {
                "prompt_id": "zero-shot-causal",
                "model": "qwen3:14b",
                "temperature": 0.6,
                "num_ctx": 8192,
                "chunked": False,
            },
        }
        diff = describe_prompt_diff("B", "D", configs)
        assert "model" in diff.lower()

    def test_same_config_returns_identical(self):
        configs = {"A": {"prompt_id": "x", "model": "m", "temperature": 0.7}}
        diff = describe_prompt_diff("A", "A", configs)
        assert "identical" in diff.lower() or "same" in diff.lower()

    def test_unknown_variant_graceful(self):
        diff = describe_prompt_diff("A", "ZZ", {})
        assert "unknown" in diff.lower()


# ---------------------------------------------------------------------------
# TestProposeNextVariant
# ---------------------------------------------------------------------------


class TestProposeNextVariant:
    def test_returns_valid_config(self):
        best = {
            "variant": "F",
            "f1": 0.52,
            "config": {
                "prompt_id": "contrastive",
                "model": "deepseek-r1:8b",
                "temperature": 0.6,
                "num_ctx": 8192,
                "chunked": False,
                "contrastive": True,
            },
        }
        ablation_impacts = {"contrastive": [0.12, 0.08], "model": [-0.03]}
        proposal = propose_next_variant(best, ablation_impacts, existing_ids=["A", "B", "F"])
        assert "variant_id" in proposal
        assert "config" in proposal
        assert "hypothesis" in proposal
        assert proposal["variant_id"].startswith("X")

    def test_avoids_existing_ids(self):
        best = {
            "variant": "F",
            "f1": 0.52,
            "config": {
                "prompt_id": "contrastive",
                "model": "deepseek-r1:8b",
                "temperature": 0.6,
                "num_ctx": 8192,
                "chunked": False,
                "contrastive": True,
            },
        }
        existing = ["A", "B", "F", "X01", "X02"]
        proposal = propose_next_variant(best, {}, existing_ids=existing)
        assert proposal["variant_id"] not in existing

    def test_returns_none_with_no_best_config(self):
        best = {"variant": "F", "f1": 0.52}  # no "config" key
        proposal = propose_next_variant(best, {})
        assert proposal is None

    def test_hypothesis_mentions_change(self):
        best = {
            "variant": "F",
            "f1": 0.52,
            "config": {
                "prompt_id": "contrastive",
                "model": "deepseek-r1:8b",
                "temperature": 0.6,
                "num_ctx": 8192,
                "chunked": False,
                "contrastive": True,
            },
        }
        proposal = propose_next_variant(best, {}, existing_ids=[])
        assert len(proposal["hypothesis"]) > 10
