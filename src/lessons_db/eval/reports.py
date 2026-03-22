"""Report renderers for eval pipeline: V1 markdown report, V2 report, diagnostics."""

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def parse_simulation_result(response: str | None) -> bool:
    """Parse whether the LLM found a bug in the simulation."""
    if not response:
        return False
    text = response.strip().upper()
    return "BUG FOUND" in text and "NO BUG FOUND" not in text


def compute_simulation_lift(
    simulation_results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute lift metric: with_principle catch rate minus without_principle catch rate.

    Lift > 0 means the principle actually helps the LLM catch bugs it would miss.
    Lift ~ 0 means the principle adds no value. Lift < 0 means it hurts.
    """
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for r in simulation_results:
        by_variant[r["variant"]].append(r)

    lift_metrics: dict[str, dict[str, float]] = {}
    for variant_id, results in sorted(by_variant.items()):
        n = len(results)
        with_catches = sum(1 for r in results if r["with_principle"])
        without_catches = sum(1 for r in results if r["without_principle"])
        with_rate = with_catches / n if n else 0.0
        without_rate = without_catches / n if n else 0.0
        lift_metrics[variant_id] = {
            "lift": with_rate - without_rate,
            "with_catch_rate": with_rate,
            "without_catch_rate": without_rate,
            "trial_count": n,
        }

    return lift_metrics


# ---------------------------------------------------------------------------
# Reference comparison
# ---------------------------------------------------------------------------


def diagnose_vs_reference(
    reference_metrics: dict[str, dict[str, float]],
    new_metrics: dict[str, dict[str, float]],
    reference_rerun: dict[str, dict[str, float]] | None = None,
    noise_margin: float = 0.03,
    metric_key: str = "auc",
) -> dict[str, dict[str, Any]]:
    """Compare new eval metrics against a frozen reference baseline.

    Distinguishes improvement/regression from data drift by optionally
    re-running the reference model on new data. If both degrade, it's drift.

    Args:
        metric_key: Which metric to compare. Default "auc" for Bayesian metrics,
            use "mean_win_rate" for tournament metrics.
    """
    diagnosis: dict[str, dict[str, Any]] = {}

    all_variants = set(new_metrics.keys()) | set(reference_metrics.keys())
    for variant in sorted(all_variants):
        if variant not in reference_metrics:
            diagnosis[variant] = {
                "status": "new",
                "delta": None,
                "new_value": new_metrics[variant][metric_key],
                "ref_value": None,
            }
            continue
        if variant not in new_metrics:
            diagnosis[variant] = {
                "status": "removed",
                "delta": None,
                "new_value": None,
                "ref_value": reference_metrics[variant][metric_key],
            }
            continue

        ref_val = reference_metrics[variant][metric_key]
        new_val = new_metrics[variant][metric_key]
        delta = new_val - ref_val

        # Check for data drift: if reference also degrades on new data
        if reference_rerun and variant in reference_rerun:
            rerun_val = reference_rerun[variant][metric_key]
            rerun_delta = rerun_val - ref_val
            if rerun_delta < -noise_margin and delta < -noise_margin:
                diagnosis[variant] = {
                    "status": "data_drift",
                    "delta": delta,
                    "new_value": new_val,
                    "ref_value": ref_val,
                    "rerun_value": rerun_val,
                }
                continue

        if abs(delta) <= noise_margin:
            status = "stable"
        elif delta > 0:
            status = "improved"
        else:
            status = "regressed"

        diagnosis[variant] = {"status": status, "delta": delta, "new_value": new_val, "ref_value": ref_val}

    return diagnosis


# ---------------------------------------------------------------------------
# V1 report renderers
# ---------------------------------------------------------------------------


def _render_failure_binary(scored_pairs: list[dict[str, Any]], lines: list[str]) -> None:
    """Render failure analysis for binary-judged pairs."""
    failures = [p for p in scored_pairs if p.get("is_same_cluster") and not p["scores"].get("matched")]
    if failures:
        lines.append(f"{len(failures)} same-cluster pairs judged NO (false negatives):\n")
        for f in failures[:10]:
            lines.append(
                f'- [{f.get("variant", "?")}] Principle: "{f.get("principle", "?")[:60]}..." '
                f'-> Target: "{f.get("target_title", "?")[:40]}"'
            )
    else:
        lines.append("No same-cluster failures (all judged YES).")
    false_pos = [p for p in scored_pairs if not p.get("is_same_cluster") and p["scores"].get("matched")]
    if false_pos:
        lines.append(f"\n{len(false_pos)} diff-cluster pairs judged YES (false positives):\n")
        for f in false_pos[:10]:
            lines.append(
                f'- [{f.get("variant", "?")}] Principle: "{f.get("principle", "?")[:60]}..." '
                f'-> Target: "{f.get("target_title", "?")[:40]}"'
            )


def _render_failure_rubric(scored_pairs: list[dict[str, Any]], lines: list[str]) -> None:
    """Render failure analysis for rubric-scored pairs."""
    failures = [p for p in scored_pairs if p.get("is_same_cluster") and p["scores"]["transfer"] < 3]
    if failures:
        lines.append(f"{len(failures)} same-cluster pairs scored below threshold:\n")
        for f in failures[:5]:
            lines.append(
                f'- [{f.get("variant", "?")}] Principle: "{f.get("principle", "?")[:60]}..." '
                f'-> Target: "{f.get("target_title", "?")[:40]}" (transfer={f["scores"]["transfer"]})'
            )
    else:
        lines.append("No same-cluster failures (all scored >= 3 on transfer).")


def _render_pair_sections(scored_pairs: list[dict[str, Any]], lines: list[str]) -> None:
    """Append per-cluster breakdown and failure analysis to report lines."""
    if not scored_pairs:
        return

    is_binary = any("matched" in p.get("scores", {}) for p in scored_pairs)

    # Per-cluster breakdown
    lines.append("\n## Per-Cluster Breakdown\n")
    clusters = sorted({p.get("cluster_seed", "") for p in scored_pairs})
    for cluster in clusters:
        if not cluster:
            continue
        cluster_pairs = [p for p in scored_pairs if p.get("cluster_seed") == cluster]
        if not cluster_pairs:
            continue
        if is_binary:
            same = [p for p in cluster_pairs if p.get("is_same_cluster")]
            tp = sum(1 for p in same if p["scores"].get("matched"))
            lines.append(f"- **Cluster {cluster}**: {tp}/{len(same)} same-cluster matched ({len(cluster_pairs)} pairs)")
        else:
            avg_transfer = sum(p["scores"]["transfer"] for p in cluster_pairs) / len(cluster_pairs)
            lines.append(f"- **Cluster {cluster}**: avg transfer = {avg_transfer:.1f} ({len(cluster_pairs)} pairs)")

    # Failure analysis
    lines.append("\n## Failure Analysis\n")
    if is_binary:
        _render_failure_binary(scored_pairs, lines)
    else:
        _render_failure_rubric(scored_pairs, lines)


def render_report(
    metrics: dict[str, dict[str, float]],
    scored_pairs: list[dict[str, Any]],
    variant_configs: dict[str, Any],
) -> str:
    """Render evaluation results as a markdown report."""
    lines: list[str] = []
    lines.append("# Transfer-Test Evaluation Report\n")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}\n")

    # Summary table — detect binary mode from metrics
    is_binary = any(m.get("binary") for m in metrics.values())
    lines.append("## Summary\n")
    if is_binary:
        lines.append("| Variant | Recall | Precision | F1 | TP | FP | FN | TN |")
        lines.append("|---------|--------|-----------|-----|----|----|----|----|")
        for vid in sorted(metrics.keys()):
            m = metrics[vid]
            lines.append(
                f"| {vid} | {m['recall']:.2f} | {m['precision']:.2f} "
                f"| {m['f1']:.2f} | {m.get('tp', 0)} | {m.get('fp', 0)} "
                f"| {m.get('fn', 0)} | {m.get('tn', 0)} |"
            )
    else:
        lines.append("| Variant | Recall | Precision | F1 | Actionability |")
        lines.append("|---------|--------|-----------|-----|---------------|")
        for vid in sorted(metrics.keys()):
            m = metrics[vid]
            lines.append(
                f"| {vid} | {m['recall']:.2f} | {m['precision']:.2f} | {m['f1']:.2f} | {m['mean_actionability']:.2f} |"
            )

    # Winner
    lines.append("\n## Winner\n")
    if metrics:
        winner = max(metrics.keys(), key=lambda v: metrics[v]["f1"])
        wm = metrics[winner]
        if is_binary:
            lines.append(
                f"**Variant {winner}** — F1: {wm['f1']:.2f} "
                f"(Recall: {wm['recall']:.2f}, Precision: {wm['precision']:.2f}, "
                f"TP={wm.get('tp', 0)} FP={wm.get('fp', 0)} "
                f"FN={wm.get('fn', 0)} TN={wm.get('tn', 0)})"
            )
        else:
            lines.append(
                f"**Variant {winner}** — F1: {wm['f1']:.2f} "
                f"(Recall: {wm['recall']:.2f}, Precision: {wm['precision']:.2f}, "
                f"Actionability: {wm['mean_actionability']:.2f})"
            )
        cfg = variant_configs.get(winner, {})
        if cfg:
            lines.append(f"\nModel: `{cfg.get('model', 'N/A')}`")
            lines.append(f"Prompt: `{cfg.get('prompt_id', 'N/A')}`")
            lines.append(f"Settings: temperature={cfg.get('temperature', 'N/A')}, num_ctx={cfg.get('num_ctx', 'N/A')}")

    _render_pair_sections(scored_pairs, lines)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# V2 report renderers
# ---------------------------------------------------------------------------


def _render_v2_failure_analysis(scored_pairs: list[dict[str, Any]], lines: list[str]) -> None:
    """Render failure analysis section for V2 report (false negatives + false positives)."""
    fn_pairs = [p for p in scored_pairs if p.get("is_same_group") and p.get("posterior", 1) < 0.5]
    fp_pairs = [p for p in scored_pairs if not p.get("is_same_group") and p.get("posterior", 0) > 0.5]
    if not fn_pairs and not fp_pairs:
        return
    lines.append("## Failure Analysis\n")
    for label, pairs in [
        ("false negatives (same-group, low posterior)", fn_pairs),
        ("false positives (diff-group, high posterior)", fp_pairs),
    ]:
        if not pairs:
            continue
        lines.append(f"{len(pairs)} {label}:\n")
        for p in pairs[:10]:
            mech = ""
            if p.get("mechanism_trigger"):
                mech = (
                    f" | Mechanism: {p['mechanism_trigger']}"
                    f"→{p.get('mechanism_target', '?')}"
                    f"→{p.get('mechanism_fix', '?')}"
                )
            lines.append(
                f"- [{p.get('variant', '?')}] P={p.get('posterior', 0):.2f} "
                f'Principle: "{p.get("principle", "")[:60]}" → '
                f'Target: "{p.get("target_title", "")[:40]}"{mech}'
            )
        lines.append("")


def _render_v2_tournament(tournament_metrics: dict[str, dict[str, float]], lines: list[str]) -> None:
    """Render tournament results table."""
    lines.append("## Tournament Results\n")
    lines.append("| Variant | Win Rate | Discriminating | Principles | Comparisons | W/L/N |")
    lines.append("|---------|----------|----------------|------------|-------------|-------|")
    for vid in sorted(tournament_metrics.keys()):
        m = tournament_metrics[vid]
        lines.append(
            f"| {vid} | {m['mean_win_rate']:.3f} | {m['discriminating_frac']:.2f} "
            f"| {m['principle_count']} | {m['comparison_count']} "
            f"| {m['total_wins']}/{m['total_losses']}/{m['total_neithers']} |"
        )
    lines.append("")


def _render_v2_bayesian(bayesian_metrics: dict[str, dict[str, float]], lines: list[str]) -> None:
    """Render Bayesian fusion table and winner."""
    lines.append("## Bayesian Fusion\n")
    lines.append("| Variant | AUC | Same Post. | Diff Post. | Separation | Cal. Error | Pairs |")
    lines.append("|---------|-----|------------|------------|------------|------------|-------|")
    for vid in sorted(bayesian_metrics.keys()):
        m = bayesian_metrics[vid]
        lines.append(
            f"| {vid} | {m['auc']:.3f} | {m['same_mean_posterior']:.3f} "
            f"| {m['diff_mean_posterior']:.3f} | {m['separation']:.3f} "
            f"| {m['calibration_error']:.3f} | {m['pair_count']} |"
        )
    lines.append("")
    lines.append("### Winner\n")
    winner = max(bayesian_metrics.keys(), key=lambda v: bayesian_metrics[v]["auc"])
    wm = bayesian_metrics[winner]
    lines.append(
        f"**Variant {winner}** — AUC: {wm['auc']:.3f} "
        f"(Separation: {wm['separation']:.3f}, "
        f"Cal. Error: {wm['calibration_error']:.3f})"
    )
    lines.append("")


def _render_v2_signal_diagnostics(signal_diagnostics: list[dict[str, Any]], lines: list[str]) -> None:
    """Render signal diagnostics section — shows per-signal contribution and disagreements."""
    lines.append("## Signal Diagnostics\n")
    # Aggregate signal means per variant
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for entry in signal_diagnostics:
        by_variant[entry.get("variant", "?")].append(entry)

    signal_keys = ["paired_signal", "embedding_signal", "scope_signal", "mechanism_signal"]
    lines.append("| Variant | Paired | Embedding | Scope | Mechanism | Disagree % |")
    lines.append("|---------|--------|-----------|-------|-----------|------------|")
    for vid in sorted(by_variant.keys()):
        entries = by_variant[vid]
        means = {}
        for key in signal_keys:
            vals = [e.get(key, 0.0) for e in entries]
            means[key] = sum(vals) / len(vals) if vals else 0.0
        # Disagreement: entries where signals have mixed signs
        disagree_count = 0
        for e in entries:
            signs = [e.get(k, 0.0) > 0 for k in signal_keys if e.get(k, 0.0) != 0]
            if signs and not (all(signs) or not any(signs)):
                disagree_count += 1
        disagree_pct = disagree_count / len(entries) * 100 if entries else 0
        lines.append(
            f"| {vid} | {means['paired_signal']:+.2f} | {means['embedding_signal']:+.2f} "
            f"| {means['scope_signal']:+.2f} | {means['mechanism_signal']:+.2f} "
            f"| {disagree_pct:.0f}% |"
        )
    lines.append("")


def render_v2_report(
    tournament_metrics: dict[str, dict[str, float]] | None = None,
    bayesian_metrics: dict[str, dict[str, float]] | None = None,
    reference_diagnosis: dict[str, dict[str, Any]] | None = None,
    simulation_lift: dict[str, dict[str, float]] | None = None,
    scored_pairs: list[dict[str, Any]] | None = None,
    signal_diagnostics: list[dict[str, Any]] | None = None,
) -> str:
    """Render unified V2 evaluation report as markdown.

    All sections are optional — only sections with data are rendered.
    """
    lines: list[str] = []
    lines.append("# Eval V2 Report\n")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}\n")

    if tournament_metrics:
        _render_v2_tournament(tournament_metrics, lines)
    if bayesian_metrics:
        _render_v2_bayesian(bayesian_metrics, lines)

    # 3. Reference Comparison
    if reference_diagnosis:
        lines.append("## Reference Comparison\n")
        for vid in sorted(reference_diagnosis.keys()):
            d = reference_diagnosis[vid]
            delta_str = f"Δ={d['delta']:+.3f}" if d.get("delta") is not None else "N/A"
            lines.append(f"- **{vid}**: {d['status']} ({delta_str})")
        lines.append("")

    # 4. Simulation Lift
    if simulation_lift:
        lines.append("## Simulation Lift\n")
        lines.append("| Variant | Lift | With Rate | Without Rate | Trials |")
        lines.append("|---------|------|-----------|--------------|--------|")
        for vid in sorted(simulation_lift.keys()):
            m = simulation_lift[vid]
            lines.append(
                f"| {vid} | {m['lift']:+.3f} | {m['with_catch_rate']:.2f} "
                f"| {m['without_catch_rate']:.2f} | {m['trial_count']} |"
            )
        lines.append("")

    # 5. Failure Analysis
    if scored_pairs:
        _render_v2_failure_analysis(scored_pairs, lines)

    # 6. Signal Diagnostics
    if signal_diagnostics:
        _render_v2_signal_diagnostics(signal_diagnostics, lines)

    return "\n".join(lines) + "\n"
