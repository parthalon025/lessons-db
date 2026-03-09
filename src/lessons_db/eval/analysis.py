"""Per-lesson breakdown and failure case extraction from scored eval pairs.

Provides two analysis functions that operate on scored pairs produced by the
judge stage of the eval pipeline:

- ``compute_per_lesson_breakdown`` — groups by (variant, source_lesson_id),
  computes per-lesson recall/precision/F1, returns sorted worst-first.
- ``extract_failure_cases`` — filters to misclassified pairs only (false
  positives and false negatives).

Both functions support binary mode (``scores.matched``) and rubric mode
(``scores.transfer`` threshold >= 3).
"""

from __future__ import annotations

import copy
import logging
import random
import statistics
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_positive(scores: dict[str, Any]) -> bool:
    """Return True if the judge considered this pair a match.

    Binary mode: ``matched`` is True.
    Rubric mode: ``transfer`` score >= 3.
    """
    if "matched" in scores:
        return bool(scores["matched"])
    raw = scores.get("transfer", 0)
    try:
        return int(raw) >= 3
    except (TypeError, ValueError):
        _log.warning("_is_positive: unexpected transfer value %r; treating as negative", raw)
        return False


def _compute_f1_from_pairs(pairs: list[dict[str, Any]]) -> float:
    """Compute F1 from a list of scored pairs using _is_positive for mode detection."""
    tp = fn = fp = 0
    for p in pairs:
        same = p["is_same_cluster"]
        positive = _is_positive(p["scores"])
        if same and positive:
            tp += 1
        elif same and not positive:
            fn += 1
        elif not same and positive:
            fp += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0


# ---------------------------------------------------------------------------
# compute_per_lesson_breakdown
# ---------------------------------------------------------------------------


def compute_per_lesson_breakdown(
    scored_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group scored pairs by (variant, source_lesson_id) and compute per-lesson metrics.

    Classification logic (same for binary and rubric modes):
      - TP: same-cluster pair where judge said YES
      - FN: same-cluster pair where judge said NO
      - FP: diff-cluster pair where judge said YES
      (TN: diff-cluster pair where judge said NO — not tracked, but counted in total)

    Returns a list of dicts sorted by F1 ascending (worst-performing lessons first):
      ``{variant, source_lesson_id, principle, recall, precision, f1,
        tp, fn, fp, total_pairs}``
    """
    if not scored_pairs:
        return []

    # Warn if mixed binary/rubric pairs — metrics will be unreliable
    modes = {"binary" if "matched" in p["scores"] else "rubric" for p in scored_pairs}
    if len(modes) > 1:
        _log.warning("Mixed binary/rubric scored pairs detected — metrics will be unreliable")

    # Group by (variant, source_lesson_id)
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for pair in scored_pairs:
        key = (pair["variant"], pair["source_lesson_id"])
        groups.setdefault(key, []).append(pair)

    results: list[dict[str, Any]] = []
    for (variant, source_lesson_id), pairs in groups.items():
        tp = fn = fp = 0
        # Use the principle from the first pair in the group
        principle = pairs[0].get("principle", "")

        for p in pairs:
            same = p["is_same_cluster"]
            positive = _is_positive(p["scores"])

            if same and positive:
                tp += 1
            elif same and not positive:
                fn += 1
            elif not same and positive:
                fp += 1
            # else: TN — correct rejection, not tracked

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results.append(
            {
                "variant": variant,
                "source_lesson_id": source_lesson_id,
                "principle": principle,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "total_pairs": len(pairs),
            }
        )

    # Sort worst-first (ascending F1), then by variant+lesson for stability
    results.sort(key=lambda r: (r["f1"], r["variant"], r["source_lesson_id"]))
    return results


# ---------------------------------------------------------------------------
# extract_failure_cases
# ---------------------------------------------------------------------------


def extract_failure_cases(
    scored_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter scored pairs to misclassified cases only.

    Failure types:
      - **false_positive**: diff-cluster pair where judge said YES
        (``matched=True`` or ``transfer >= 3``)
      - **false_negative**: same-cluster pair where judge said NO
        (``matched=False`` or ``transfer < 3``)

    Returns a list sorted by (variant, source_lesson_id):
      ``{failure_type, variant, source_lesson_id, principle, target_id,
        target_title, cluster_seed, target_cluster_seed, scores}``
    """
    if not scored_pairs:
        return []

    failures: list[dict[str, Any]] = []
    for pair in scored_pairs:
        same = pair["is_same_cluster"]
        positive = _is_positive(pair["scores"])

        if not same and positive:
            failure_type = "false_positive"
        elif same and not positive:
            failure_type = "false_negative"
        else:
            continue  # TP or TN — not a failure

        failures.append(
            {
                "failure_type": failure_type,
                "variant": pair["variant"],
                "source_lesson_id": pair["source_lesson_id"],
                "principle": pair.get("principle", ""),
                "target_id": pair["target_id"],
                "target_title": pair.get("target_title", ""),
                "cluster_seed": pair.get("cluster_seed", ""),
                "target_cluster_seed": pair.get("target_cluster_seed", ""),
                "scores": pair["scores"],
            }
        )

    failures.sort(key=lambda f: (f["variant"], f["source_lesson_id"]))
    return failures


# ---------------------------------------------------------------------------
# bootstrap_f1_ci
# ---------------------------------------------------------------------------


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

    rng = random.Random(seed)  # noqa: S311 — statistical resampling, not crypto
    f1s: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.choices(pairs, k=len(pairs))
        f1s.append(_compute_f1_from_pairs(sample))

    f1s.sort()
    alpha = (1 - ci) / 2
    low_idx = max(0, int(alpha * len(f1s)))
    high_idx = max(0, min(int((1 - alpha) * len(f1s)) - 1, len(f1s) - 1))
    return {
        "low": round(f1s[low_idx], 4),
        "mid": round(statistics.median(f1s), 4),
        "high": round(f1s[high_idx], 4),
    }


# ---------------------------------------------------------------------------
# compute_stability
# ---------------------------------------------------------------------------


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
        stdev_raw = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        stdev_rounded = round(stdev_raw, 4)
        result[vid] = {
            "stdev": stdev_rounded,
            "mean": round(statistics.mean(f1s), 4),
            "n_runs": len(f1s),
            "stable": stdev_rounded < instability_threshold,
            "f1s": f1s,
        }
    return result


# ---------------------------------------------------------------------------
# describe_prompt_diff
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# propose_next_variant
# ---------------------------------------------------------------------------

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


def _try_mutation(
    best_config: dict[str, Any],
    candidate_id: str,
    strategies: list[tuple[str, dict[str, Any], str]],
) -> dict[str, Any] | None:
    """Try each strategy against best_config; return first novel mutation or None."""
    for _dim, mutation, hyp_template in strategies:
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
    return None


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
            filtered = [s for s in _EXPLORATION_STRATEGIES if s[0] == dim]
            result = _try_mutation(best_config, candidate_id, filtered)
            if result:
                return result

    # Strategy 2: systematic exploration — try each strategy until we find one not yet tested
    result = _try_mutation(best_config, candidate_id, _EXPLORATION_STRATEGIES)
    if result:
        return result

    # Exhausted all strategies
    return {
        "variant_id": candidate_id,
        "config": copy.deepcopy(best_config),
        "hypothesis": "no new mutations available — rerunning best config for stability data",
    }
