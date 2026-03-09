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
