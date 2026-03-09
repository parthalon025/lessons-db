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
    low_idx = int(alpha * len(f1s))
    high_idx = int((1 - alpha) * len(f1s)) - 1
    return {
        "low": round(f1s[max(0, low_idx)], 4),
        "mid": round(f1s[len(f1s) // 2], 4),
        "high": round(f1s[min(high_idx, len(f1s) - 1)], 4),
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
        stdev = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
        result[vid] = {
            "stdev": round(stdev, 4),
            "mean": round(statistics.mean(f1s), 4),
            "n_runs": len(f1s),
            "stable": stdev < instability_threshold,
            "f1s": f1s,
        }
    return result
