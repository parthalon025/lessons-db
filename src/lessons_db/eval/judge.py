"""Judge orchestrator: score principles, compute metrics, run tournaments."""

import json as _json
import re as _re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from lessons_db.config import OLLAMA_QUEUE_URL
from lessons_db.eval.client import call_judge
from lessons_db.eval.prompts import (
    build_binary_judge_prompt,
    build_judge_prompt,
    build_paired_judge_prompt,
)
from lessons_db.eval.runs import record_eval_run
from lessons_db.eval.sampling import select_transfer_targets
from lessons_db.eval.variants import VARIANT_CONFIGS

# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def parse_judge_scores(response: str) -> dict[str, int] | None:
    """Extract transfer/precision/actionability scores from judge response.

    Returns dict with keys transfer, precision, actionability (ints 1-5),
    or None if parsing fails.
    """
    match = _re.search(r"\{[^}]+\}", response)
    if not match:
        return None

    try:
        data = _json.loads(match.group())
    except _json.JSONDecodeError:
        return None

    required = {"transfer", "precision", "actionability"}
    if not required.issubset(data.keys()):
        return None

    scores = {}
    for key in required:
        val = int(data[key])
        scores[key] = max(1, min(5, val))
    return scores


def parse_binary_judge(response: str) -> bool | None:
    """Parse YES/NO from binary judge response. Returns True/False/None."""
    if not response:
        return None
    text = response.strip().upper()
    # Strip think blocks
    text = _re.sub(r"<THINK>.*?</THINK>", "", text, flags=_re.DOTALL).strip()
    if text.startswith("YES"):
        return True
    if text.startswith("NO"):
        return False
    # Check for YES/NO anywhere in short response
    if len(text) < 50:
        if "YES" in text and "NO" not in text:
            return True
        if "NO" in text and "YES" not in text:
            return False
    return None


def parse_paired_judge(response: str) -> str | None:
    """Parse A/B/NEITHER from paired comparison response."""
    if not response:
        return None
    text = response.strip().upper()
    # Strip thinking tags (some models like deepseek-r1 emit these)
    text = _re.sub(r"<THINK>.*?</THINK>", "", text, flags=_re.DOTALL).strip()
    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    if "NEITHER" in text:
        return "NEITHER"
    # Fallback: look for single letter in short response
    for ch in ["A", "B"]:
        if ch in text and len(text) < 30:
            return ch
    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(scored_pairs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute per-variant aggregate metrics from scored pairs.

    Supports two score formats:
    - Rubric: scores = {transfer, precision, actionability} (1-5 scale)
    - Binary: scores = {matched: True/False}

    Returns dict[variant_id -> {recall, precision, f1, ...}].
    """
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for pair in scored_pairs:
        by_variant.setdefault(pair["variant"], []).append(pair)

    metrics: dict[str, dict[str, float]] = {}
    for variant, pairs in by_variant.items():
        same = [p for p in pairs if p["is_same_cluster"]]
        diff = [p for p in pairs if not p["is_same_cluster"]]

        # Detect binary mode from score keys
        is_binary = any("matched" in p.get("scores", {}) for p in pairs)

        if is_binary:
            # Standard classification: TP/FP/FN/TN
            tp = sum(1 for p in same if p["scores"].get("matched"))
            fn = sum(1 for p in same if not p["scores"].get("matched"))
            fp = sum(1 for p in diff if p["scores"].get("matched"))
            tn = sum(1 for p in diff if not p["scores"].get("matched"))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
            metrics[variant] = {
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "binary": True,
            }
        else:
            # Rubric mode (original)
            recall = sum(1 for p in same if p["scores"]["transfer"] >= 3) / len(same) if same else 0.0
            precision = sum(1 for p in diff if p["scores"]["transfer"] <= 2) / len(diff) if diff else 0.0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
            all_act = [p["scores"]["actionability"] for p in pairs]
            mean_act = sum(all_act) / len(all_act) if all_act else 0.0
            metrics[variant] = {
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f1": round(f1, 4),
                "mean_actionability": round(mean_act, 4),
            }

    return metrics


def compute_rank_metrics(scored_pairs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute rank-based metrics that are immune to judge score inflation.

    For each (variant, principle), groups all target scores and checks whether
    same-cluster targets rank above diff-cluster targets.  Uses AUC
    (area under ROC curve via Mann-Whitney U statistic) — 1.0 means perfect
    discrimination, 0.5 means random.
    """
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for pair in scored_pairs:
        by_variant.setdefault(pair["variant"], []).append(pair)

    metrics: dict[str, dict[str, float]] = {}
    for variant, pairs in by_variant.items():
        # Group by source principle (cluster_seed + principle combo)
        by_principle: dict[str, list[dict[str, Any]]] = {}
        for p in pairs:
            key = f"{p.get('cluster_seed', '')}|{p.get('principle', '')[:50]}"
            by_principle.setdefault(key, []).append(p)

        aucs: list[float] = []
        for _key, principle_pairs in by_principle.items():
            same_scores = [p["scores"]["transfer"] for p in principle_pairs if p["is_same_cluster"]]
            diff_scores = [p["scores"]["transfer"] for p in principle_pairs if not p["is_same_cluster"]]
            if not same_scores or not diff_scores:
                continue
            # Mann-Whitney U: proportion of (same, diff) pairs where same > diff
            u = sum(1 for s in same_scores for d in diff_scores if s > d)
            ties = sum(1 for s in same_scores for d in diff_scores if s == d)
            n = len(same_scores) * len(diff_scores)
            auc = (u + 0.5 * ties) / n if n > 0 else 0.5
            aucs.append(auc)

        mean_auc = sum(aucs) / len(aucs) if aucs else 0.5
        # Fraction of principles that discriminate (AUC > 0.5)
        discriminating = sum(1 for a in aucs if a > 0.5) / len(aucs) if aucs else 0.0

        metrics[variant] = {
            "mean_auc": round(mean_auc, 4),
            "discriminating_frac": round(discriminating, 4),
            "n_principles": len(aucs),
        }

    return metrics


# ---------------------------------------------------------------------------
# Paired tournament
# ---------------------------------------------------------------------------


def run_paired_tournament(
    results_path: Path,
    conn: sqlite3.Connection,
    backend: str = "ollama",
    ollama_url: str = OLLAMA_QUEUE_URL,
    ollama_model: str = "",
    group_by: str = "category",
    pairs_per_principle: int = 4,
    progress_callback: Any = None,
    priority: int | None = None,
) -> list[dict[str, Any]]:
    """Run paired tournament: for each principle, compare same-group vs diff-group targets.

    For each generated principle:
    1. Select same-group and diff-group transfer targets
    2. Create paired comparisons (one same + one diff per pair)
    3. Call judge with paired prompt, randomizing A/B position
    4. Track win rate (did judge pick the same-group target?)

    Returns list of dicts with keys:
        variant, lesson_id, principle, win_rate, comparisons, wins, losses, neithers
    """
    data = _json.loads(results_path.read_text())
    results = data.get("results", [])

    tournament_results: list[dict[str, Any]] = []

    for entry in results:
        principle = entry.get("principle")
        if not principle or entry.get("error"):
            continue

        variant = entry["variant"]
        lesson_id = entry["lesson_id"]
        group_value = entry.get(group_by, entry.get("cluster_seed", ""))

        # Get transfer targets
        targets = select_transfer_targets(
            conn,
            lesson_id,
            group_value,
            count_same=pairs_per_principle,
            count_diff=pairs_per_principle,
            group_by=group_by,
        )

        same_targets = targets["same_cluster"]
        diff_targets = targets["diff_cluster"]

        # Create paired comparisons (zip same + diff)
        wins = 0
        losses = 0
        neithers = 0
        comparisons = 0

        for i in range(min(len(same_targets), len(diff_targets))):
            same_t = same_targets[i]
            diff_t = diff_targets[i]

            prompt, same_is_a = build_paired_judge_prompt(principle, same_t, diff_t, position_seed=i)

            response = call_judge(
                prompt=prompt,
                backend=backend,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
                priority=priority,
            )

            answer = parse_paired_judge(response)
            comparisons += 1

            if answer == "NEITHER":
                neithers += 1
            elif answer is not None:
                # Did the judge pick the same-group target?
                picked_same = (answer == "A" and same_is_a) or (answer == "B" and not same_is_a)
                if picked_same:
                    wins += 1
                else:
                    losses += 1
            else:
                # None response (parse failure) counts as neither
                neithers += 1

        win_rate = wins / comparisons if comparisons > 0 else 0.0

        tournament_results.append(
            {
                "variant": variant,
                "lesson_id": lesson_id,
                "principle": principle[:200],
                "win_rate": win_rate,
                "comparisons": comparisons,
                "wins": wins,
                "losses": losses,
                "neithers": neithers,
            }
        )

        if progress_callback:
            progress_callback(variant, lesson_id, win_rate, comparisons)

    return tournament_results


def compute_tournament_metrics(
    tournament_results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute aggregate metrics from tournament results, grouped by variant.

    Returns dict of variant_id -> metrics dict with:
        mean_win_rate: average win rate across principles (approx AUC)
        discriminating_frac: fraction of principles with win_rate > 0.5
        principle_count: number of principles evaluated
        comparison_count: total comparisons made
        total_wins: total wins across all principles
        total_losses: total losses
        total_neithers: total neither responses
    """
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in tournament_results:
        by_variant[r["variant"]].append(r)

    metrics: dict[str, dict[str, float]] = {}
    for variant_id, results in sorted(by_variant.items()):
        win_rates = [r["win_rate"] for r in results]
        total_comparisons = sum(r["comparisons"] for r in results)
        total_wins = sum(r["wins"] for r in results)
        total_losses = sum(r["losses"] for r in results)
        total_neithers = sum(r["neithers"] for r in results)

        metrics[variant_id] = {
            "mean_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
            "discriminating_frac": (sum(1 for wr in win_rates if wr > 0.5) / len(win_rates) if win_rates else 0.0),
            "principle_count": len(results),
            "comparison_count": total_comparisons,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_neithers": total_neithers,
        }

    return metrics


# ---------------------------------------------------------------------------
# Judge orchestrator
# ---------------------------------------------------------------------------


def run_eval_judge(
    results_path: Path,
    conn: sqlite3.Connection,
    report_path: Path,
    backend: str = "ollama",
    ollama_url: str = OLLAMA_QUEUE_URL,
    ollama_model: str = "",
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    progress_callback: Any = None,
    priority: int | None = None,
    binary: bool = False,
    group_by: str = "category",
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Run eval-judge: score generated principles against transfer targets.

    Reads results JSON, constructs transfer test cases, scores each pair,
    computes metrics, and writes a markdown report.

    When binary=True, uses YES/NO discrimination instead of 1-5 rubric.

    Returns (scored_pairs, metrics_by_variant).
    """
    from lessons_db.eval.reports import render_report

    results_data = _json.loads(results_path.read_text())
    results = results_data.get("results", [])

    scored_pairs: list[dict[str, Any]] = []

    for entry in results:
        principle = entry.get("principle")
        if not principle or entry.get("error"):
            continue

        variant = entry["variant"]
        lesson_id = entry["lesson_id"]
        cluster_seed = entry.get("cluster_seed", "")
        group_value = entry.get(group_by, cluster_seed)

        targets = select_transfer_targets(
            conn,
            lesson_id,
            group_value,
            group_by=group_by,
        )

        for is_same, target_list in [
            (True, targets["same_cluster"]),
            (False, targets["diff_cluster"]),
        ]:
            for target in target_list:
                if binary:
                    prompt = build_binary_judge_prompt(principle, target)
                else:
                    prompt = build_judge_prompt(principle, target)

                response = call_judge(
                    prompt=prompt,
                    backend=backend,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    openai_api_key=openai_api_key,
                    openai_model=openai_model,
                    priority=priority,
                )

                if binary:
                    matched = parse_binary_judge(response) if response else None
                    scores = {"matched": matched if matched is not None else False}
                else:
                    scores = parse_judge_scores(response) if response else None
                    if scores is None:
                        scores = {"transfer": 1, "precision": 1, "actionability": 1}

                pair = {
                    "variant": variant,
                    "source_lesson_id": lesson_id,
                    "principle": principle,
                    "target_id": target["id"],
                    "target_title": target.get("title", ""),
                    "cluster_seed": cluster_seed,
                    "target_cluster_seed": target.get("cluster_seed", ""),
                    "is_same_cluster": is_same,
                    "scores": scores,
                }
                scored_pairs.append(pair)

                if progress_callback:
                    label = "TP" if is_same else "TN"
                    progress_callback(variant, target["id"], label, scores)

    metrics = compute_metrics(scored_pairs)

    report = render_report(metrics, scored_pairs, VARIANT_CONFIGS)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    # Save scored pairs for diagnostic tools (confusion matrix, etc.)
    scored_path = report_path.with_suffix(".scored.json")
    scored_path.write_text(_json.dumps(scored_pairs, indent=2))

    # Persist aggregate metrics for regression tracking and APO history
    for variant_id, m in metrics.items():
        variant_cfg = VARIANT_CONFIGS.get(variant_id, {})
        record_eval_run(
            conn,
            variant=variant_id,
            f1=m.get("f1", 0.0),
            recall=m.get("recall", 0.0),
            precision=m.get("precision", 0.0),
            model=variant_cfg.get("model"),
            judge_model=ollama_model or openai_model,
            prompt_id=variant_cfg.get("prompt_id"),
            results_file=str(results_path),
        )

    return scored_pairs, metrics
