"""Diagnostic utilities for eval pipeline analysis."""

from __future__ import annotations

from typing import Any


def build_confusion_matrix(
    scored_pairs: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build a cluster confusion matrix from scored pairs.

    For each (source_cluster, target_cluster) pair, computes the average
    transfer score. High cross-cluster scores indicate principle bleed —
    clusters that share structural patterns and may need merging or
    more discriminative prompts.

    scored_pairs: list of dicts with keys:
        cluster_seed (source cluster), target_cluster_seed (for diff-cluster),
        is_same_cluster, scores.transfer

    Returns: {(source_cluster, target_cluster): {avg_transfer, count}}
    """
    if not scored_pairs:
        return {}

    buckets: dict[tuple[str, str], list[int]] = {}

    for pair in scored_pairs:
        source_cluster = pair.get("cluster_seed", "")
        scores = pair.get("scores")
        if not scores:
            continue
        transfer = scores.get("transfer")
        if transfer is None:
            continue

        if pair.get("is_same_cluster"):
            key = (source_cluster, source_cluster)
        else:
            target_cluster = pair.get("target_cluster_seed", "unknown")
            key = (source_cluster, target_cluster)

        buckets.setdefault(key, []).append(transfer)

    matrix: dict[tuple[str, str], dict[str, Any]] = {}
    for key, scores_list in buckets.items():
        matrix[key] = {
            "avg_transfer": round(sum(scores_list) / len(scores_list), 2),
            "count": len(scores_list),
        }

    return matrix


def render_confusion_report(
    matrix: dict[tuple[str, str], dict[str, Any]],
) -> str:
    """Render confusion matrix as a markdown table."""
    if not matrix:
        return "No data.\n"

    clusters = sorted({k[0] for k in matrix} | {k[1] for k in matrix})
    lines = ["# Cluster Confusion Matrix\n"]
    lines.append("Average transfer score — high off-diagonal values " "indicate principle bleed.\n")

    header = "| Source \\ Target | " + " | ".join(clusters) + " |"
    sep = "|" + "---|" * (len(clusters) + 1)
    lines.append(header)
    lines.append(sep)

    for src in clusters:
        row = f"| **{src}** |"
        for tgt in clusters:
            cell = matrix.get((src, tgt))
            if cell:
                val = cell["avg_transfer"]
                marker = " !!" if src != tgt and val >= 3.0 else ""
                row += f" {val:.1f}{marker} |"
            else:
                row += " -- |"
        lines.append(row)

    lines.append("\n## Flagged Pairs\n")
    flagged = [(k, v) for k, v in matrix.items() if k[0] != k[1] and v["avg_transfer"] >= 3.0]
    if flagged:
        for (src, tgt), v in sorted(flagged, key=lambda x: -x[1]["avg_transfer"]):
            lines.append(
                f"- **{src} -> {tgt}**: avg transfer {v['avg_transfer']:.1f} "
                f"({v['count']} pairs) — principles from {src} "
                f"falsely match {tgt}"
            )
    else:
        lines.append("No cross-cluster pairs with avg transfer >= 3.0.")

    return "\n".join(lines) + "\n"


def compute_roc_curve(
    scored_pairs: list[dict[str, Any]],
    thresholds: list[int] | None = None,
) -> dict[int, dict[str, float]]:
    """Sweep transfer score thresholds and compute precision/recall at each.

    For each threshold t:
      recall    = fraction of same-cluster pairs with transfer >= t
      precision = fraction of diff-cluster pairs with transfer < t
      f1        = harmonic mean

    Helps identify the optimal threshold and visualize the precision-recall trade-off.
    """
    if thresholds is None:
        thresholds = [1, 2, 3, 4, 5]

    same = [p for p in scored_pairs if p.get("is_same_cluster") and p.get("scores", {}).get("transfer") is not None]
    diff = [p for p in scored_pairs if not p.get("is_same_cluster") and p.get("scores", {}).get("transfer") is not None]

    curve: dict[int, dict[str, float]] = {}
    for t in thresholds:
        recall = sum(1 for p in same if p["scores"]["transfer"] >= t) / len(same) if same else 0.0
        precision = sum(1 for p in diff if p["scores"]["transfer"] < t) / len(diff) if diff else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        curve[t] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
        }

    return curve


def render_roc_report(curve: dict[int, dict[str, float]]) -> str:
    """Render ROC curve as a text table."""
    if not curve:
        return "No data.\n"

    lines = ["Threshold | Recall | Precision | F1"]
    lines.append("----------|--------|-----------|----")

    best_f1 = max(v["f1"] for v in curve.values())
    for t in sorted(curve.keys()):
        c = curve[t]
        marker = " <-- best" if c["f1"] == best_f1 and best_f1 > 0 else ""
        lines.append(f"    {t}     | {c['recall']:.2f}   | {c['precision']:.2f}      | {c['f1']:.2f}{marker}")

    return "\n".join(lines) + "\n"
