# Eval Analysis & Auto-Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 6 analysis capabilities to the eval pipeline and wire them into an autonomous improvement loop that proposes, tests, and promotes better variant configs without human intervention.

**Architecture:** All 6 features live in a new `eval/analysis.py` module. The key architectural change is widening `run_eval_learn()` to accept raw `scored_pairs` (not just aggregated metrics) — this unlocks per-lesson breakdown, confidence intervals, and failure case export. The auto-improvement loop lives in `scripts/autoresearch-loop.sh` and calls `eval-propose` CLI to generate the next variant.

**Tech Stack:** Python stdlib only (random for bootstrap, statistics for stdev). No new dependencies.

---

## Data Structures Reference

**scored_pairs** (from `judge.py:429-440`):
```python
{
    "variant": str,               # e.g. "A", "F"
    "source_lesson_id": int,
    "principle": str,
    "target_id": int,
    "target_title": str,
    "cluster_seed": str,
    "target_cluster_seed": str,
    "is_same_cluster": bool,
    "scores": {"matched": bool}   # binary mode
           or {"transfer": int, "precision": int, "actionability": int}  # rubric
}
```

**metrics_by_variant** (from `judge.py:compute_metrics`):
```python
{"A": {"recall": 0.93, "precision": 0.17, "f1": 0.28, "tp": 12, "fp": 45, ...}}
```

**VARIANT_CONFIGS** (from `variants.py:56-160`):
```python
{"A": {"prompt_id": "baseline-fewshot", "model": "deepseek-r1:8b", "temperature": 0.7,
       "num_ctx": 4096, "chunked": False}}
```

---

## Batch 1: Per-Lesson Breakdown + Failure Cases (scored_pairs consumers)

These two features both consume `scored_pairs` and are tightly related — per-lesson breakdown identifies *which* lessons fail, failure cases shows *why*.

### Task 1.1: Write failing tests for `compute_per_lesson_breakdown()`

**Files:**
- Create: `tests/test_eval_analysis.py`

**Step 1: Write the failing tests**

```python
"""Tests for eval/analysis.py — analysis capabilities for the eval pipeline."""

import pytest

from lessons_db.eval.analysis import compute_per_lesson_breakdown


class TestPerLessonBreakdown:
    def test_groups_by_source_lesson(self):
        scored_pairs = [
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": True,
             "scores": {"matched": True}, "principle": "p1", "target_id": 10},
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": False,
             "scores": {"matched": False}, "principle": "p1", "target_id": 20},
            {"variant": "A", "source_lesson_id": 2, "is_same_cluster": True,
             "scores": {"matched": False}, "principle": "p2", "target_id": 30},
        ]
        breakdown = compute_per_lesson_breakdown(scored_pairs)
        assert len(breakdown) == 2
        assert breakdown[0]["source_lesson_id"] in (1, 2)

    def test_computes_per_lesson_f1(self):
        scored_pairs = [
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": True,
             "scores": {"matched": True}, "principle": "p", "target_id": 10},
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": False,
             "scores": {"matched": False}, "principle": "p", "target_id": 20},
        ]
        breakdown = compute_per_lesson_breakdown(scored_pairs)
        # Perfect score: TP=1 on same, TN=1 on diff → F1=1.0
        assert breakdown[0]["f1"] == pytest.approx(1.0)

    def test_sorted_worst_first(self):
        scored_pairs = [
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": True,
             "scores": {"matched": True}, "principle": "p", "target_id": 10},
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": False,
             "scores": {"matched": False}, "principle": "p", "target_id": 20},
            {"variant": "A", "source_lesson_id": 2, "is_same_cluster": True,
             "scores": {"matched": False}, "principle": "p2", "target_id": 30},
            {"variant": "A", "source_lesson_id": 2, "is_same_cluster": False,
             "scores": {"matched": True}, "principle": "p2", "target_id": 40},
        ]
        breakdown = compute_per_lesson_breakdown(scored_pairs)
        # Lesson 2 has F1=0.0, lesson 1 has F1=1.0 → lesson 2 should be first
        assert breakdown[0]["source_lesson_id"] == 2
        assert breakdown[0]["f1"] == 0.0

    def test_includes_variant_in_breakdown(self):
        scored_pairs = [
            {"variant": "B", "source_lesson_id": 1, "is_same_cluster": True,
             "scores": {"matched": True}, "principle": "p", "target_id": 10},
        ]
        breakdown = compute_per_lesson_breakdown(scored_pairs)
        assert breakdown[0]["variant"] == "B"

    def test_empty_pairs(self):
        assert compute_per_lesson_breakdown([]) == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_analysis.py -x -q`
Expected: FAIL (module not found)

### Task 1.2: Implement `compute_per_lesson_breakdown()`

**Files:**
- Create: `src/lessons_db/eval/analysis.py`

**Step 1: Write minimal implementation**

```python
"""Analysis capabilities for the eval pipeline.

Per-lesson breakdown, confidence intervals, failure case export,
cross-run stability, prompt diff, and auto-variant generation.

All functions operate on scored_pairs (the raw judge output) or
learnings.jsonl (the cross-run audit trail). No LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any

_log = logging.getLogger(__name__)


def compute_per_lesson_breakdown(
    scored_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute per-lesson hit/miss rates from scored pairs.

    Groups pairs by (variant, source_lesson_id), computes per-lesson
    recall/precision/F1, and returns sorted worst-first so you can
    identify which lessons are hardest to generate principles for.
    """
    if not scored_pairs:
        return []

    # Group by (variant, source_lesson_id)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for pair in scored_pairs:
        key = (pair["variant"], pair["source_lesson_id"])
        groups.setdefault(key, []).append(pair)

    results: list[dict[str, Any]] = []
    for (variant, lesson_id), pairs in groups.items():
        same = [p for p in pairs if p["is_same_cluster"]]
        diff = [p for p in pairs if not p["is_same_cluster"]]

        is_binary = any("matched" in p.get("scores", {}) for p in pairs)
        if is_binary:
            tp = sum(1 for p in same if p["scores"].get("matched"))
            fn = sum(1 for p in same if not p["scores"].get("matched"))
            fp = sum(1 for p in diff if p["scores"].get("matched"))
        else:
            tp = sum(1 for p in same if p["scores"].get("transfer", 0) >= 3)
            fn = sum(1 for p in same if p["scores"].get("transfer", 0) < 3)
            fp = sum(1 for p in diff if p["scores"].get("transfer", 0) >= 3)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

        results.append({
            "variant": variant,
            "source_lesson_id": lesson_id,
            "principle": pairs[0].get("principle", ""),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "total_pairs": len(pairs),
        })

    # Sort worst-first (lowest F1)
    results.sort(key=lambda x: x["f1"])
    return results
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_eval_analysis.py -x -q`
Expected: 5 passed

**Step 3: Commit**

```bash
git add tests/test_eval_analysis.py src/lessons_db/eval/analysis.py
git commit -m "feat(eval): add per-lesson F1 breakdown"
```

### Task 1.3: Write failing tests for `extract_failure_cases()`

**Files:**
- Modify: `tests/test_eval_analysis.py`

**Step 1: Add failing tests**

```python
from lessons_db.eval.analysis import compute_per_lesson_breakdown, extract_failure_cases


class TestExtractFailureCases:
    def test_finds_false_positives(self):
        """diff-cluster pair where judge said YES → false positive."""
        scored_pairs = [
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": False,
             "scores": {"matched": True}, "principle": "too broad", "target_id": 20,
             "target_title": "unrelated", "cluster_seed": "c1", "target_cluster_seed": "c2"},
        ]
        failures = extract_failure_cases(scored_pairs)
        assert len(failures) == 1
        assert failures[0]["failure_type"] == "false_positive"

    def test_finds_false_negatives(self):
        """same-cluster pair where judge said NO → false negative."""
        scored_pairs = [
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": True,
             "scores": {"matched": False}, "principle": "too narrow", "target_id": 10,
             "target_title": "related", "cluster_seed": "c1", "target_cluster_seed": "c1"},
        ]
        failures = extract_failure_cases(scored_pairs)
        assert len(failures) == 1
        assert failures[0]["failure_type"] == "false_negative"

    def test_correct_pairs_excluded(self):
        scored_pairs = [
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": True,
             "scores": {"matched": True}, "principle": "good", "target_id": 10,
             "target_title": "t", "cluster_seed": "c1", "target_cluster_seed": "c1"},
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": False,
             "scores": {"matched": False}, "principle": "good", "target_id": 20,
             "target_title": "t", "cluster_seed": "c1", "target_cluster_seed": "c2"},
        ]
        assert extract_failure_cases(scored_pairs) == []

    def test_rubric_mode_threshold(self):
        """Rubric: transfer >= 3 on diff-cluster is a false positive."""
        scored_pairs = [
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": False,
             "scores": {"transfer": 4, "precision": 3, "actionability": 3},
             "principle": "broad", "target_id": 20,
             "target_title": "t", "cluster_seed": "c1", "target_cluster_seed": "c2"},
        ]
        failures = extract_failure_cases(scored_pairs)
        assert len(failures) == 1

    def test_empty_pairs(self):
        assert extract_failure_cases([]) == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_analysis.py::TestExtractFailureCases -x -q`
Expected: FAIL (import error)

### Task 1.4: Implement `extract_failure_cases()`

**Files:**
- Modify: `src/lessons_db/eval/analysis.py`

**Step 1: Add implementation after `compute_per_lesson_breakdown`**

```python
def extract_failure_cases(
    scored_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract misclassified pairs: false positives and false negatives.

    False positive: diff-cluster pair where judge said YES / transfer >= 3
    False negative: same-cluster pair where judge said NO / transfer < 3

    Returns list sorted by variant + lesson ID for easy inspection.
    """
    failures: list[dict[str, Any]] = []

    for pair in scored_pairs:
        scores = pair.get("scores", {})
        is_binary = "matched" in scores
        is_same = pair["is_same_cluster"]

        if is_binary:
            is_positive = bool(scores.get("matched"))
        else:
            is_positive = scores.get("transfer", 0) >= 3

        failure_type = None
        if is_same and not is_positive:
            failure_type = "false_negative"
        elif not is_same and is_positive:
            failure_type = "false_positive"

        if failure_type:
            failures.append({
                "failure_type": failure_type,
                "variant": pair["variant"],
                "source_lesson_id": pair["source_lesson_id"],
                "principle": pair.get("principle", ""),
                "target_id": pair["target_id"],
                "target_title": pair.get("target_title", ""),
                "cluster_seed": pair.get("cluster_seed", ""),
                "target_cluster_seed": pair.get("target_cluster_seed", ""),
                "scores": scores,
            })

    failures.sort(key=lambda x: (x["variant"], x["source_lesson_id"]))
    return failures
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_eval_analysis.py -x -q`
Expected: 10 passed

**Step 3: Commit**

```bash
git add tests/test_eval_analysis.py src/lessons_db/eval/analysis.py
git commit -m "feat(eval): add failure case extraction"
```

---

## Batch 2: Confidence Intervals + Cross-Run Stability (statistical rigor)

### Task 2.1: Write failing tests for `bootstrap_f1_ci()`

**Files:**
- Modify: `tests/test_eval_analysis.py`

**Step 1: Add failing tests**

```python
from lessons_db.eval.analysis import bootstrap_f1_ci


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_analysis.py::TestBootstrapF1CI -x -q`
Expected: FAIL

### Task 2.2: Implement `bootstrap_f1_ci()`

**Files:**
- Modify: `src/lessons_db/eval/analysis.py`

**Step 1: Add implementation**

```python
import random


def _compute_f1_from_pairs(pairs: list[dict[str, Any]]) -> float:
    """Compute F1 from a list of scored pairs (binary mode)."""
    same = [p for p in pairs if p["is_same_cluster"]]
    diff = [p for p in pairs if not p["is_same_cluster"]]

    is_binary = any("matched" in p.get("scores", {}) for p in pairs)
    if is_binary:
        tp = sum(1 for p in same if p["scores"].get("matched"))
        fn = sum(1 for p in same if not p["scores"].get("matched"))
        fp = sum(1 for p in diff if p["scores"].get("matched"))
    else:
        tp = sum(1 for p in same if p["scores"].get("transfer", 0) >= 3)
        fn = sum(1 for p in same if p["scores"].get("transfer", 0) < 3)
        fp = sum(1 for p in diff if p["scores"].get("transfer", 0) >= 3)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0


def bootstrap_f1_ci(
    scored_pairs: list[dict[str, Any]],
    variant: str,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a variant's F1.

    Resamples scored_pairs with replacement N times, computes F1 each time,
    returns the ci-percentile interval.

    Returns {"low": float, "mid": float, "high": float}.
    """
    pairs = [p for p in scored_pairs if p["variant"] == variant]
    if not pairs:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}

    rng = random.Random(seed)
    f1s: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(pairs, k=len(pairs))
        f1s.append(_compute_f1_from_pairs(sample))

    f1s.sort()
    alpha = (1 - ci) / 2
    low_idx = int(alpha * len(f1s))
    high_idx = int((1 - alpha) * len(f1s)) - 1
    return {
        "low": round(f1s[max(0, low_idx)], 4),
        "mid": round(f1s[len(f1s) // 2], 4),
        "high": round(f1s[min(high_idx, len(f1s) - 1)], 4),
    }
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_eval_analysis.py -x -q`
Expected: 15 passed

**Step 3: Commit**

```bash
git add tests/test_eval_analysis.py src/lessons_db/eval/analysis.py
git commit -m "feat(eval): add bootstrap F1 confidence intervals"
```

### Task 2.3: Write failing tests for `compute_stability()`

**Files:**
- Modify: `tests/test_eval_analysis.py`

**Step 1: Add failing tests**

```python
from lessons_db.eval.analysis import compute_stability


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
```

**Step 2: Run test to verify it fails**

### Task 2.4: Implement `compute_stability()`

**Files:**
- Modify: `src/lessons_db/eval/analysis.py`

**Step 1: Add implementation**

```python
import statistics


def compute_stability(
    entries: list[dict[str, Any]],
    instability_threshold: float = 0.10,
) -> dict[str, dict[str, Any]]:
    """Compute cross-run F1 stability per variant from learnings.jsonl entries.

    A variant is "stable" if its stdev(F1) < instability_threshold.
    High variance means model temperature or sample randomness dominates
    over prompt design — fix that before optimizing prompts.

    Returns {variant: {stdev, mean, n_runs, stable, f1s}}.
    """
    # Reuse compute_variant_trends logic inline to avoid circular import
    by_variant: dict[str, list[float]] = {}
    for e in entries:
        if e.get("type") == "ablations":
            continue
        vid = e.get("variant")
        f1 = e.get("f1")
        if vid and f1 is not None:
            by_variant.setdefault(vid, []).append(float(f1))

    result: dict[str, dict[str, Any]] = {}
    for vid, f1s in by_variant.items():
        stdev = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        result[vid] = {
            "stdev": round(stdev, 4),
            "mean": round(statistics.mean(f1s), 4),
            "n_runs": len(f1s),
            "stable": stdev < instability_threshold,
            "f1s": f1s,
        }
    return result
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_eval_analysis.py -x -q`
Expected: 20 passed

**Step 3: Commit**

```bash
git add tests/test_eval_analysis.py src/lessons_db/eval/analysis.py
git commit -m "feat(eval): add cross-run stability analysis"
```

---

## Batch 3: Prompt Diff + Auto-Variant Generation (closes the loop)

### Task 3.1: Write failing tests for `describe_prompt_diff()`

**Files:**
- Modify: `tests/test_eval_analysis.py`

**Step 1: Add failing tests**

```python
from lessons_db.eval.analysis import describe_prompt_diff


class TestDescribePromptDiff:
    def test_describes_contrastive_addition(self):
        configs = {
            "B": {"prompt_id": "zero-shot-causal", "model": "deepseek-r1:8b",
                   "temperature": 0.6, "num_ctx": 8192, "chunked": False},
            "F": {"prompt_id": "contrastive", "model": "deepseek-r1:8b",
                   "temperature": 0.6, "num_ctx": 8192, "chunked": False, "contrastive": True},
        }
        diff = describe_prompt_diff("B", "F", configs)
        assert "contrastive" in diff.lower()
        assert "boundary" in diff.lower() or "scope" in diff.lower() or "not apply" in diff.lower()

    def test_describes_model_change(self):
        configs = {
            "B": {"prompt_id": "zero-shot-causal", "model": "deepseek-r1:8b",
                   "temperature": 0.6, "num_ctx": 8192, "chunked": False},
            "D": {"prompt_id": "zero-shot-causal", "model": "qwen3:14b",
                   "temperature": 0.6, "num_ctx": 8192, "chunked": False},
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
```

**Step 2: Run test to verify it fails**

### Task 3.2: Implement `describe_prompt_diff()`

**Files:**
- Modify: `src/lessons_db/eval/analysis.py`

**Step 1: Add implementation**

```python
# Maps config flags to their prompt-level effect
_PROMPT_EFFECT_DESCRIPTIONS: dict[str, str] = {
    "contrastive": "adds 'when does this NOT apply?' instruction — forces boundary/scope specificity",
    "chunked": "splits lesson into ~512-token chunks — per-chunk focus vs whole-lesson coherence",
    "multi_stage": "two-pass pipeline: extract pattern then distill principle — more deliberate, 2x cost",
    "mechanism": "asks for root-cause mechanism (TRIGGER→FAILURE→CONSEQUENCE) instead of surface rule",
}


def describe_prompt_diff(
    variant_a: str,
    variant_b: str,
    variant_configs: dict[str, dict[str, Any]],
) -> str:
    """Describe what changes between two variant prompts in plain language.

    Goes beyond config flags to explain *what the model sees differently*.
    """
    cfg_a = variant_configs.get(variant_a)
    cfg_b = variant_configs.get(variant_b)

    if not cfg_a or not cfg_b:
        return f"unknown variant ('{variant_a}' or '{variant_b}' not in configs)"

    if variant_a == variant_b:
        return f"identical — same config ({variant_a})"

    changes: list[str] = []

    # Model change
    if cfg_a.get("model") != cfg_b.get("model"):
        changes.append(f"model: {cfg_a.get('model')} → {cfg_b.get('model')} (different capacity/training)")

    # Temperature change
    if cfg_a.get("temperature") != cfg_b.get("temperature"):
        ta, tb = cfg_a.get("temperature"), cfg_b.get("temperature")
        direction = "more deterministic" if tb < ta else "more creative"
        changes.append(f"temperature: {ta} → {tb} ({direction})")

    # Context window
    if cfg_a.get("num_ctx") != cfg_b.get("num_ctx"):
        changes.append(f"context window: {cfg_a.get('num_ctx')} → {cfg_b.get('num_ctx')} tokens")

    # Boolean flags — these change what the model actually sees in the prompt
    for flag, description in _PROMPT_EFFECT_DESCRIPTIONS.items():
        val_a = bool(cfg_a.get(flag))
        val_b = bool(cfg_b.get(flag))
        if val_a != val_b:
            action = "added" if val_b else "removed"
            changes.append(f"{flag} {action}: {description}")

    # Prompt ID change (different template entirely)
    if cfg_a.get("prompt_id") != cfg_b.get("prompt_id"):
        changes.append(f"prompt template: {cfg_a.get('prompt_id')} → {cfg_b.get('prompt_id')}")

    if not changes:
        return "identical effective config (no prompt-level differences)"

    return "; ".join(changes)
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_eval_analysis.py -x -q`
Expected: 24 passed

**Step 3: Commit**

```bash
git add tests/test_eval_analysis.py src/lessons_db/eval/analysis.py
git commit -m "feat(eval): add prompt diff descriptions"
```

### Task 3.3: Write failing tests for `propose_next_variant()`

**Files:**
- Modify: `tests/test_eval_analysis.py`

**Step 1: Add failing tests**

```python
from lessons_db.eval.analysis import propose_next_variant


class TestProposeNextVariant:
    def test_returns_valid_config(self):
        best = {"variant": "F", "f1": 0.52, "config": {
            "prompt_id": "contrastive", "model": "deepseek-r1:8b",
            "temperature": 0.6, "num_ctx": 8192, "chunked": False, "contrastive": True,
        }}
        ablation_impacts = {"contrastive": [0.12, 0.08], "model": [-0.03]}
        proposal = propose_next_variant(best, ablation_impacts, existing_ids=["A", "B", "F"])
        assert "variant_id" in proposal
        assert "config" in proposal
        assert "hypothesis" in proposal
        assert proposal["variant_id"].startswith("X")

    def test_avoids_existing_ids(self):
        best = {"variant": "F", "f1": 0.52, "config": {
            "prompt_id": "contrastive", "model": "deepseek-r1:8b",
            "temperature": 0.6, "num_ctx": 8192, "chunked": False, "contrastive": True,
        }}
        existing = ["A", "B", "F", "X01", "X02"]
        proposal = propose_next_variant(best, {}, existing_ids=existing)
        assert proposal["variant_id"] not in existing

    def test_returns_none_with_no_best_config(self):
        best = {"variant": "F", "f1": 0.52}  # no "config" key
        proposal = propose_next_variant(best, {})
        assert proposal is None

    def test_hypothesis_mentions_change(self):
        best = {"variant": "F", "f1": 0.52, "config": {
            "prompt_id": "contrastive", "model": "deepseek-r1:8b",
            "temperature": 0.6, "num_ctx": 8192, "chunked": False, "contrastive": True,
        }}
        proposal = propose_next_variant(best, {}, existing_ids=[])
        assert len(proposal["hypothesis"]) > 10
```

**Step 2: Run test to verify it fails**

### Task 3.4: Implement `propose_next_variant()`

**Files:**
- Modify: `src/lessons_db/eval/analysis.py`

**Step 1: Add implementation**

```python
import copy

# Exploration strategies: each is a (dimension, mutation, hypothesis_template)
_EXPLORATION_STRATEGIES: list[tuple[str, dict[str, Any], str]] = [
    ("temperature", {"temperature": 0.4}, "lower temperature ({val}) increases determinism — may sharpen precision"),
    ("temperature", {"temperature": 0.8}, "higher temperature ({val}) increases diversity — may improve recall"),
    ("num_ctx", {"num_ctx": 16384}, "larger context window ({val}) lets model see full lesson — tests context scaling"),
    ("num_ctx", {"num_ctx": 4096}, "smaller context window ({val}) forces conciseness — tests if brevity helps"),
    ("contrastive", {"contrastive": True}, "adding contrastive scope — forces boundary specificity"),
    ("multi_stage", {"multi_stage": True}, "adding multi-stage — two-pass pipeline for more deliberate output"),
    ("chunked", {"chunked": True}, "adding chunking — per-chunk focus may isolate failure patterns"),
    ("model", {"model": "qwen3:14b"}, "larger model (qwen3:14b) may follow instructions more faithfully"),
    ("model", {"model": "qwen3.5:9b"}, "qwen3.5:9b — newer architecture, may generalize differently"),
]


def propose_next_variant(
    best: dict[str, Any],
    ablation_impacts: dict[str, list[float]],
    existing_ids: list[str] | None = None,
) -> dict[str, Any] | None:
    """Propose the next variant config based on ablation trends and best-so-far.

    Strategy:
    1. Start from the best config (not control — build on what works)
    2. If ablation data exists, push the most impactful positive dimension further
    3. If no ablation data, try the next untested exploration strategy
    4. Never propose a config identical to an existing variant

    Returns {"variant_id": "X03", "config": {...}, "hypothesis": "..."} or None.
    """
    best_config = best.get("config")
    if not best_config:
        return None

    existing = set(existing_ids or [])

    # Generate next X-ID
    for i in range(1, 100):
        candidate_id = f"X{i:02d}"
        if candidate_id not in existing:
            break
    else:
        candidate_id = "X99"

    # Strategy 1: if ablation data shows a clear winner, push it
    if ablation_impacts:
        ranked = sorted(
            ablation_impacts.items(),
            key=lambda kv: sum(d for d in kv[1] if d > 0) / max(len(kv[1]), 1),
            reverse=True,
        )
        for dim, _deltas in ranked:
            for _strat_dim, mutation, hyp_template in _EXPLORATION_STRATEGIES:
                if _strat_dim != dim:
                    continue
                candidate = copy.deepcopy(best_config)
                candidate.update(mutation)
                # Check if this is already in the best config
                if all(candidate.get(k) == best_config.get(k) for k in mutation):
                    continue
                candidate["prompt_id"] = f"auto-{candidate_id}"
                return {
                    "variant_id": candidate_id,
                    "config": candidate,
                    "hypothesis": hyp_template.format(val=list(mutation.values())[0]),
                }

    # Strategy 2: systematic exploration — try each strategy until we find one not yet tested
    for _dim, mutation, hyp_template in _EXPLORATION_STRATEGIES:
        candidate = copy.deepcopy(best_config)
        candidate.update(mutation)
        if all(candidate.get(k) == best_config.get(k) for k in mutation):
            continue
        candidate["prompt_id"] = f"auto-{candidate_id}"
        return {
            "variant_id": candidate_id,
            "config": candidate,
            "hypothesis": hyp_template.format(val=list(mutation.values())[0]),
        }

    # Exhausted all strategies
    return {
        "variant_id": candidate_id,
        "config": copy.deepcopy(best_config),
        "hypothesis": "no new mutations available — rerunning best config for stability data",
    }
```

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_eval_analysis.py -x -q`
Expected: 28 passed

**Step 3: Commit**

```bash
git add tests/test_eval_analysis.py src/lessons_db/eval/analysis.py
git commit -m "feat(eval): add auto-variant proposal from ablation trends"
```

---

## Batch 4: Wire Into Pipeline (CLI + learn.py + autoresearch)

### Task 4.1: Widen `run_eval_learn()` to accept `scored_pairs`

**Files:**
- Modify: `src/lessons_db/eval/learn.py:394-421`
- Modify: `tests/test_eval_learn.py`

**Step 1: Write a failing test**

Add to `tests/test_eval_learn.py` in `TestRunEvalLearn`:

```python
    def test_analysis_included_when_scored_pairs_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"A": {"f1": 0.28, "recall": 0.93, "precision": 0.17}}
        scored_pairs = [
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": True,
             "scores": {"matched": True}, "principle": "p", "target_id": 10},
            {"variant": "A", "source_lesson_id": 1, "is_same_cluster": False,
             "scores": {"matched": True}, "principle": "p", "target_id": 20},
        ]
        result = run_eval_learn(metrics, _VARIANT_CONFIGS, scored_pairs=scored_pairs)
        insights, ablations = result[:2]
        assert len(insights) == 1
        # Should have analysis dict as third element
        assert len(result) == 3
        analysis = result[2]
        assert "per_lesson" in analysis
        assert "failure_cases" in analysis
        assert "confidence_intervals" in analysis
```

**Step 2: Update `run_eval_learn()` signature and return**

In `src/lessons_db/eval/learn.py`, change `run_eval_learn()`:

```python
def run_eval_learn(
    metrics_by_variant: dict[str, dict[str, float]],
    variant_configs: dict[str, dict[str, Any]],
    program_md_path: Path | None = None,
    run_date: str | None = None,
    scored_pairs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Derive insights + ablation + analysis, persist, and optionally update program.md.

    When scored_pairs is provided, also computes per-lesson breakdown,
    failure cases, and confidence intervals.

    Returns (insights, ablations, analysis).
    analysis is empty dict when scored_pairs is not provided.
    """
    insights = derive_insights(metrics_by_variant, variant_configs, run_date)
    ablations = compute_ablations(metrics_by_variant, variant_configs) if len(metrics_by_variant) > 1 else []

    analysis: dict[str, Any] = {}
    if scored_pairs:
        from lessons_db.eval.analysis import (
            bootstrap_f1_ci,
            compute_per_lesson_breakdown,
            extract_failure_cases,
        )
        analysis["per_lesson"] = compute_per_lesson_breakdown(scored_pairs)
        analysis["failure_cases"] = extract_failure_cases(scored_pairs)
        analysis["confidence_intervals"] = {
            vid: bootstrap_f1_ci(scored_pairs, variant=vid, n_bootstrap=500, seed=42)
            for vid in metrics_by_variant
        }

    try:
        save_learnings(insights, ablations or None)
    except Exception as exc:
        _log.warning("save_learnings failed (non-fatal): %s", exc)

    if program_md_path:
        try:
            append_to_program_md(insights, program_md_path, ablations or None)
        except Exception as exc:
            _log.warning("append_to_program_md failed (non-fatal): %s", exc)

    return insights, ablations, analysis
```

**Important:** This changes the return type from `tuple[list, list]` to `tuple[list, list, dict]`. All existing callers that unpack `insights, ablations = run_eval_learn(...)` must be updated to `insights, ablations, analysis = run_eval_learn(...)` or use `result = run_eval_learn(...); insights, ablations = result[0], result[1]`.

**Step 3: Update all callers**

In `src/lessons_db/cli.py`, update the two call sites:

1. Line ~2958 in `meta_eval_judge`:
```python
insights, ablations, analysis = run_eval_learn(
    metrics_by_variant=metrics,
    variant_configs=VARIANT_CONFIGS,
    program_md_path=program_md if program_md.exists() else None,
    scored_pairs=scored_pairs,  # NEW — pass raw pairs
)
```

2. Line ~3017 in `meta_eval_learn`:
```python
insights, ablations, _analysis = run_eval_learn(
    metrics_by_variant=metrics,
    variant_configs=VARIANT_CONFIGS,
    program_md_path=program_md if program_md.exists() else None,
)
```

**Step 4: Update existing tests** in `tests/test_eval_learn.py`:

All tests that unpack `insights, ablations = run_eval_learn(...)` need to become
`insights, ablations, _analysis = run_eval_learn(...)`.

Similarly `insights, _ablations = run_eval_learn(...)` → `insights, _ablations, _analysis = run_eval_learn(...)`.

**Step 5: Run tests**

Run: `pytest tests/test_eval_learn.py tests/test_eval_analysis.py -x -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/lessons_db/eval/learn.py src/lessons_db/cli.py tests/test_eval_learn.py
git commit -m "feat(eval): widen run_eval_learn to accept scored_pairs — unlocks analysis"
```

### Task 4.2: Add `eval-analyze` CLI command and display analysis in `eval-judge`

**Files:**
- Modify: `src/lessons_db/cli.py`

**Step 1: Add analysis display to `meta_eval_judge`** after the existing learn display block:

```python
        if analysis:
            if analysis.get("confidence_intervals"):
                click.echo("\nConfidence intervals (95%):")
                for vid, ci in sorted(analysis["confidence_intervals"].items()):
                    click.echo(f"  {vid}: F1 = {ci['mid']:.3f} [{ci['low']:.3f} – {ci['high']:.3f}]")
            if analysis.get("per_lesson"):
                worst = analysis["per_lesson"][:5]
                if worst:
                    click.echo(f"\nHardest lessons (lowest F1, top {len(worst)}):")
                    for pl in worst:
                        click.echo(f"  lesson #{pl['source_lesson_id']} [{pl['variant']}]: F1={pl['f1']:.3f} (TP={pl['tp']} FN={pl['fn']} FP={pl['fp']})")
            if analysis.get("failure_cases"):
                fp_count = sum(1 for f in analysis["failure_cases"] if f["failure_type"] == "false_positive")
                fn_count = sum(1 for f in analysis["failure_cases"] if f["failure_type"] == "false_negative")
                click.echo(f"\nFailure cases: {fp_count} false positives, {fn_count} false negatives")
```

**Step 2: Add `eval-propose` CLI command** (for auto-variant generation):

```python
@meta.command("eval-propose")
@click.option("--dry-run", is_flag=True, help="Show proposal without writing to variants.py.")
@click.pass_context
def meta_eval_propose(ctx, dry_run):
    """Propose the next variant config based on ablation trends and best-so-far."""
    from lessons_db.eval.analysis import propose_next_variant
    from lessons_db.eval.learn import compute_dimension_impacts, load_best, load_learnings
    from lessons_db.eval.variants import VARIANT_CONFIGS

    best = load_best()
    if not best:
        click.echo("No best.json found. Run eval-judge first.", err=True)
        ctx.exit(1)
        return

    entries = load_learnings()
    dim_impacts = compute_dimension_impacts(entries)

    proposal = propose_next_variant(
        best=best,
        ablation_impacts=dim_impacts,
        existing_ids=list(VARIANT_CONFIGS.keys()),
    )

    if not proposal:
        click.echo("Cannot propose: best.json has no config. Re-run eval-judge to populate.")
        return

    click.echo(f"Proposed: {proposal['variant_id']}")
    click.echo(f"  Hypothesis: {proposal['hypothesis']}")
    click.echo(f"  Config: {proposal['config']}")

    if not dry_run:
        # Write proposal to a JSON file for autoresearch-loop.sh to consume
        import json as json_mod
        proposal_path = Path.home() / ".local" / "share" / "lessons-db" / "eval" / "proposed_variant.json"
        proposal_path.parent.mkdir(parents=True, exist_ok=True)
        proposal_path.write_text(json_mod.dumps(proposal, indent=2))
        click.echo(f"  Written to: {proposal_path}")
```

**Step 3: Commit**

```bash
git add src/lessons_db/cli.py
git commit -m "feat(eval): add analysis display + eval-propose CLI command"
```

### Task 4.3: Update `__init__.py` exports

**Files:**
- Modify: `src/lessons_db/eval/__init__.py`

**Step 1: Add exports for analysis.py**

```python
from lessons_db.eval.analysis import (
    bootstrap_f1_ci as bootstrap_f1_ci,
)
from lessons_db.eval.analysis import (
    compute_per_lesson_breakdown as compute_per_lesson_breakdown,
)
from lessons_db.eval.analysis import (
    compute_stability as compute_stability,
)
from lessons_db.eval.analysis import (
    describe_prompt_diff as describe_prompt_diff,
)
from lessons_db.eval.analysis import (
    extract_failure_cases as extract_failure_cases,
)
from lessons_db.eval.analysis import (
    propose_next_variant as propose_next_variant,
)
```

**Step 2: Commit**

```bash
git add src/lessons_db/eval/__init__.py
git commit -m "feat(eval): export analysis functions from eval package"
```

### Task 4.4: Create `autoresearch-loop.sh`

**Files:**
- Create: `scripts/autoresearch-loop.sh`

**Step 1: Write the loop script**

```bash
#!/usr/bin/env bash
# autoresearch-loop.sh — autonomous improvement loop
#
# Usage: ./scripts/autoresearch-loop.sh [--max-runs N] [--per-cluster N]
#
# Runs the propose → generate → judge → learn cycle until:
#   - Max runs reached (default: 10)
#   - No improvement for 3 consecutive runs
#   - Proposal system exhausts strategies
#
# Each iteration:
#   1. lessons-db meta eval-propose → proposed_variant.json
#   2. Inject config into variants.py (temporary X-variant)
#   3. Run autoresearch-run.sh with the proposed variant
#   4. Record result, loop

set -euo pipefail

MAX_RUNS=10
PER_CLUSTER=2
STALE_LIMIT=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-runs) MAX_RUNS="$2"; shift 2 ;;
        --per-cluster) PER_CLUSTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROPOSAL_JSON="$HOME/.local/share/lessons-db/eval/proposed_variant.json"

cd "$PROJECT_DIR"
source .venv/bin/activate

stale_count=0

for run in $(seq 1 "$MAX_RUNS"); do
    echo ""
    echo "=== autoresearch-loop: run $run/$MAX_RUNS (stale=$stale_count/$STALE_LIMIT) ==="

    # 1. Propose next variant
    echo "[1/4] Proposing next variant..."
    if ! lessons-db meta eval-propose 2>/dev/null; then
        echo "Proposal failed — running best variant for stability data"
        VARIANT="$(python3 -c "import json; print(json.load(open('$HOME/.local/share/lessons-db/eval/best.json')).get('variant', 'A'))")"
    else
        VARIANT="$(python3 -c "import json; print(json.load(open('$PROPOSAL_JSON'))['variant_id'])")"

        # 2. Inject proposed config into variants.py
        echo "[2/4] Injecting $VARIANT into variants.py..."
        python3 -c "
import json, ast
proposal = json.load(open('$PROPOSAL_JSON'))
vid = proposal['variant_id']
config = proposal['config']

vfile = '$PROJECT_DIR/src/lessons_db/eval/variants.py'
with open(vfile) as f:
    content = f.read()

# Only inject if not already present
if f'\"$VARIANT\"' not in content:
    # Find the closing brace of VARIANT_CONFIGS
    marker = '}  # end VARIANT_CONFIGS'
    if marker not in content:
        # Fallback: insert before last closing brace
        idx = content.rindex('}')
        insert = f'    \"{vid}\": {json.dumps(config, indent=8)},\n'
        content = content[:idx] + insert + content[idx:]
    else:
        content = content.replace(marker, f'    \"{vid}\": {json.dumps(config, indent=8)},\n' + marker)
    with open(vfile, 'w') as f:
        f.write(content)
    print(f'Injected {vid} into variants.py')
else:
    print(f'{vid} already in variants.py')
"
    fi

    # 3. Run experiment
    echo "[3/4] Running experiment: $VARIANT --per-cluster $PER_CLUSTER..."
    EXIT_CODE=0
    "$SCRIPT_DIR/autoresearch-run.sh" "$VARIANT" --per-cluster "$PER_CLUSTER" || EXIT_CODE=$?

    # 4. Interpret result
    case $EXIT_CODE in
        0)
            echo "IMPROVED — $VARIANT is new best!"
            stale_count=0
            ;;
        1)
            echo "No improvement from $VARIANT"
            stale_count=$((stale_count + 1))
            ;;
        2)
            echo "CRASH — $VARIANT failed"
            stale_count=$((stale_count + 1))
            ;;
    esac

    if [[ $stale_count -ge $STALE_LIMIT ]]; then
        echo "Stopping: $STALE_LIMIT consecutive runs without improvement."
        break
    fi
done

echo ""
echo "=== autoresearch-loop complete: $run runs ==="
echo "Results: $PROJECT_DIR/results.tsv"
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/autoresearch-loop.sh
git add scripts/autoresearch-loop.sh
git commit -m "feat(eval): add autoresearch-loop.sh — autonomous improvement loop"
```

---

## Batch 5: Full Suite Verification

### Task 5.1: Run all tests

**Step 1:** Run full eval test suite

```bash
pytest tests/test_eval_learn.py tests/test_eval_analysis.py tests/test_eval.py --timeout=120 -x -q -n 6
```

Expected: All pass

**Step 2:** Run full project suite

```bash
pytest --timeout=120 -x -q -n 6
```

Expected: All pass (minus pre-existing flaky network tests)

### Task 5.2: Final commit and PR

```bash
git add -A
git commit -m "test: verify full suite after analysis pipeline"
git push origin autoresearch/mar09
```

Update PR #18 with the new commits, or create a new PR if branch was merged.

---

## Summary

| Batch | Tasks | Features | Tests Added |
|-------|-------|----------|------------|
| 1 | 1.1–1.4 | Per-lesson breakdown + failure cases | ~10 |
| 2 | 2.1–2.4 | Bootstrap CI + cross-run stability | ~10 |
| 3 | 3.1–3.4 | Prompt diff + auto-variant generation | ~8 |
| 4 | 4.1–4.4 | Pipeline wiring (learn.py, CLI, autoresearch-loop.sh) | ~5 |
| 5 | 5.1–5.2 | Full verification + PR | 0 |

**Total:** ~33 new tests, 6 new functions in `analysis.py`, 2 new CLI commands (`eval-propose`, display in `eval-judge`), 1 new script (`autoresearch-loop.sh`), 1 signature change (`run_eval_learn`).

**Key architectural decision:** The `run_eval_learn()` return type changes from `tuple[list, list]` to `tuple[list, list, dict]`. The third element is the analysis dict (empty when `scored_pairs` not provided). This is a backward-compatible change at the call sites because existing callers that unpack `insights, ablations = ...` will get an error and must be updated — but there are only 2 call sites (both in `cli.py`), plus test files.
