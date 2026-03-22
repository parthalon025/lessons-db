"""Post-judge learning: derive insights from eval metrics and feed them back.

Called automatically after every eval-judge run, regardless of whether any
variant improved. The goal is to always extract signal — even a crash or a
miss teaches something about the design space.

Outputs:
  - learnings.jsonl  — append-only audit trail (one JSON object per variant per run)
  - best.json        — current best F1 tracker (read by autoresearch-run.sh)
  - program.md       — "Learned so far" section updated in-place (if file exists in cwd)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

_EVAL_DIR = Path.home() / ".local" / "share" / "lessons-db" / "eval"
BEST_JSON = _EVAL_DIR / "best.json"
LEARNINGS_FILE = _EVAL_DIR / "learnings.jsonl"

# ---------------------------------------------------------------------------
# Mechanical diagnosis — no LLM needed
# ---------------------------------------------------------------------------

_HIGH_RECALL = 0.65
_LOW_RECALL = 0.45
_HIGH_PRECISION = 0.40
_LOW_PRECISION = 0.25


def _diagnose(recall: float, precision: float) -> tuple[str, str]:
    """Return (diagnosis, recommendation) based on precision/recall signature.

    The four quadrants map to distinct prompt-engineering interventions:

      High recall + low precision  → principle too broad
        Fix: add contrastive scope ("when does this NOT apply?")

      Low recall + high precision  → principle too narrow
        Fix: remove chunking, increase context window

      Both low                     → generation ineffective
        Fix: try a different model or prompt strategy entirely

      Both high                    → well-balanced
        Action: promote this config, try slight temperature reduction
    """
    if recall >= _HIGH_RECALL and precision <= _LOW_PRECISION:
        return (
            "principle too broad — high recall, low precision (false positives across clusters)",
            "add contrastive scope constraints; try contrastive=True or lower temperature",
        )
    if recall <= _LOW_RECALL and precision >= _HIGH_PRECISION:
        return (
            "principle too narrow — low recall, high precision (misses within-cluster matches)",
            "reduce chunking, increase context window, or use zero-shot-causal prompt",
        )
    if recall <= _LOW_RECALL and precision <= _LOW_PRECISION:
        return (
            "generation ineffective — both metrics low (model or prompt mismatch)",
            "try a different model or prompt strategy; check generation output for coherence",
        )
    # Middle zone: at least one metric is between thresholds — not extreme enough
    # to diagnose decisively, but not good enough to promote.
    if recall < _HIGH_RECALL or precision < _HIGH_PRECISION:
        return (
            "moderate — metrics in middle zone, needs targeted improvement",
            "run more variants to isolate bottleneck; check ablation results for which dimension to push",
        )
    return (
        "well-balanced — good precision-recall tradeoff",
        "promote this config; try lower temperature or larger model for marginal gains",
    )


def _config_diff_vs_control(variant_id: str, variant_configs: dict[str, Any]) -> str:
    """Describe what this variant changes relative to the control (A)."""
    if variant_id == "A":
        return "baseline (control)"
    control = variant_configs.get("A", {})
    variant = variant_configs.get(variant_id, {})
    if not variant:
        return "unknown config (not in VARIANT_CONFIGS)"
    changes = []
    for key in ("model", "temperature", "num_ctx"):
        if variant.get(key) != control.get(key):
            changes.append(f"{key}={variant.get(key)}")
    for flag in ("chunked", "contrastive", "multi_stage", "mechanism"):
        if variant.get(flag) and not control.get(flag):
            changes.append(f"{flag}=True")
        elif not variant.get(flag) and control.get(flag):
            changes.append(f"{flag}=False")
    return ", ".join(changes) if changes else "same as control"


# ---------------------------------------------------------------------------
# Ablation analysis — compare variants that differ by one config dimension
# ---------------------------------------------------------------------------

_ABLATION_KEYS = ("model", "temperature", "num_ctx", "chunked", "contrastive", "multi_stage", "mechanism")
_BOOLEAN_FLAGS = {"chunked", "contrastive", "multi_stage", "mechanism"}


def _normalize_cfg_value(key: str, value: Any) -> Any:
    """Normalize config values: treat None as False for boolean flags."""
    if key in _BOOLEAN_FLAGS:
        return bool(value)
    return value


def _config_fingerprint(cfg: dict[str, Any], exclude_key: str) -> tuple:
    """Hashable config fingerprint excluding one key, for finding ablation pairs."""
    return tuple((k, _normalize_cfg_value(k, cfg.get(k))) for k in _ABLATION_KEYS if k != exclude_key)


def _compare_pair(
    dim: str,
    va: str,
    vb: str,
    entries: dict[str, tuple[dict[str, Any], dict[str, float]]],
) -> dict[str, Any] | None:
    """Compare two variants on a single dimension. Returns ablation dict or None."""
    cfg_a, m_a = entries[va]
    cfg_b, m_b = entries[vb]
    val_a = _normalize_cfg_value(dim, cfg_a.get(dim))
    val_b = _normalize_cfg_value(dim, cfg_b.get(dim))
    if val_a == val_b:
        return None
    return {
        "dimension": dim,
        "from": val_a,
        "to": val_b,
        "delta_f1": round(float(m_b.get("f1", 0.0)) - float(m_a.get("f1", 0.0)), 4),
        "variant_a": va,
        "variant_b": vb,
        "f1_a": float(m_a.get("f1", 0.0)),
        "f1_b": float(m_b.get("f1", 0.0)),
    }


def _collect_ablations_for_dim(
    dim: str,
    entries: dict[str, tuple[dict[str, Any], dict[str, float]]],
    seen_pairs: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Find ablation pairs for a single config dimension."""
    groups: dict[tuple, list[str]] = {}
    for vid, (cfg, _) in entries.items():
        fp = _config_fingerprint(cfg, dim)
        groups.setdefault(fp, []).append(vid)

    results: list[dict[str, Any]] = []
    for _fp, vids in groups.items():
        if len(vids) < 2:
            continue
        for i, va in enumerate(vids):
            for vb in vids[i + 1 :]:
                pair_key = (min(va, vb), max(va, vb))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                result = _compare_pair(dim, va, vb, entries)
                if result:
                    results.append(result)
    return results


def compute_ablations(
    metrics_by_variant: dict[str, dict[str, float]],
    variant_configs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find pairs of variants that differ by exactly one config dimension.

    For each such pair, compute the delta in F1 and attribute it to the
    dimension that changed. Returns a list of ablation dicts sorted by
    absolute delta descending (most impactful dimension first).

    Example output::

        {"dimension": "contrastive", "from": False, "to": True,
         "delta_f1": +0.12, "variant_a": "B", "variant_b": "F",
         "f1_a": 0.45, "f1_b": 0.57}
    """
    entries = {}
    for vid in metrics_by_variant:
        cfg = variant_configs.get(vid)
        if cfg:
            entries[vid] = (cfg, metrics_by_variant[vid])

    seen_pairs: set[tuple[str, str]] = set()
    ablations: list[dict[str, Any]] = []
    for dim in _ABLATION_KEYS:
        ablations.extend(_collect_ablations_for_dim(dim, entries, seen_pairs))

    ablations.sort(key=lambda x: abs(x["delta_f1"]), reverse=True)
    return ablations


def format_ablation_summary(ablations: list[dict[str, Any]], top_n: int = 5) -> list[str]:
    """Format top-N ablation results as human-readable lines."""
    lines = []
    for ab in ablations[:top_n]:
        direction = "+" if ab["delta_f1"] > 0 else ""
        lines.append(
            f"{ab['dimension']}: {ab['from']} → {ab['to']} = "
            f"{direction}{ab['delta_f1']:.3f} F1 "
            f"({ab['variant_a']}={ab['f1_a']:.3f} vs {ab['variant_b']}={ab['f1_b']:.3f})"
        )
    return lines


# ---------------------------------------------------------------------------
# Best-F1 tracking
# ---------------------------------------------------------------------------


def load_best() -> dict[str, Any]:
    """Load the current best F1 record. Returns empty dict if none exists."""
    if BEST_JSON.exists():
        try:
            return json.loads(BEST_JSON.read_text())
        except Exception as exc:
            _log.warning("best.json unreadable: %s", exc)
    return {}


def save_best(
    variant_id: str,
    metrics: dict[str, float],
    variant_config: dict[str, Any] | None = None,
) -> None:
    """Persist a new best F1 record, including the full config for reproducibility."""
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "variant": variant_id,
        "f1": metrics["f1"],
        "recall": metrics.get("recall", 0.0),
        "precision": metrics.get("precision", 0.0),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    if variant_config:
        record["config"] = variant_config
    BEST_JSON.write_text(json.dumps(record, indent=2))


# ---------------------------------------------------------------------------
# Core: derive_insights — always runs, no LLM
# ---------------------------------------------------------------------------


def derive_insights(
    metrics_by_variant: dict[str, dict[str, float]],
    variant_configs: dict[str, dict[str, Any]],
    run_date: str | None = None,
) -> list[dict[str, Any]]:
    """Derive mechanical insights from eval metrics for every variant.

    Always runs regardless of whether any variant improved. Every experiment
    teaches something — even a miss narrows the design space.

    Returns a list of insight dicts, one per variant, in F1-descending order.
    """
    date = run_date or datetime.now(UTC).strftime("%Y-%m-%d")
    best_record = load_best()
    best_f1 = float(best_record.get("f1", 0.0))

    insights: list[dict[str, Any]] = []

    for variant_id, m in sorted(metrics_by_variant.items(), key=lambda kv: kv[1].get("f1", 0.0), reverse=True):
        f1 = float(m.get("f1", 0.0))
        recall = float(m.get("recall", 0.0))
        precision = float(m.get("precision", 0.0))

        diagnosis, recommendation = _diagnose(recall, precision)
        config_changes = _config_diff_vs_control(variant_id, variant_configs)

        delta = f1 - best_f1
        if f1 > best_f1:
            status = "win"
            summary = f"NEW BEST F1={f1:.3f} (+{delta:+.3f} vs prior {best_f1:.3f})"
            # Update rolling best immediately so subsequent variants in this run
            # compare against the new bar.
            best_f1 = f1
            save_best(variant_id, m, variant_configs.get(variant_id))
        elif delta >= -0.03:
            status = "near-miss"
            summary = f"near-miss F1={f1:.3f} ({delta:+.3f} vs best {best_f1:.3f})"
        else:
            status = "miss"
            summary = f"below best F1={f1:.3f} ({delta:+.3f} vs best {best_f1:.3f})"

        insights.append(
            {
                "date": date,
                "variant": variant_id,
                "f1": f1,
                "recall": recall,
                "precision": precision,
                "status": status,
                "diagnosis": diagnosis,
                "recommendation": recommendation,
                "config_changes": config_changes,
                "summary": summary,
            }
        )

    return insights


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_learnings(
    insights: list[dict[str, Any]],
    ablations: list[dict[str, Any]] | None = None,
) -> None:
    """Append insights + ablation data to the append-only learnings.jsonl audit trail."""
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with LEARNINGS_FILE.open("a") as fh:
        for ins in insights:
            fh.write(json.dumps(ins) + "\n")
        if ablations:
            run_date = insights[0]["date"] if insights else datetime.now(UTC).strftime("%Y-%m-%d")
            fh.write(json.dumps({"type": "ablations", "date": run_date, "ablations": ablations}) + "\n")


def append_to_program_md(
    insights: list[dict[str, Any]],
    program_md_path: Path,
    ablations: list[dict[str, Any]] | None = None,
) -> bool:
    """Append insights + top ablation to the '## Learned so far' section of program.md.

    Inserts new bullet points immediately after the section header (and its
    optional italic subtitle), so the newest entry always appears first.

    Returns True if the file was updated, False if the section was not found.
    """
    if not program_md_path.exists():
        return False

    content = program_md_path.read_text()
    marker = "## Learned so far"
    if marker not in content:
        return False

    lines = content.splitlines()
    insert_at: int | None = None
    for i, line in enumerate(lines):
        if line.strip() == marker:
            insert_at = i + 1
            # Skip the italic placeholder line if present
            if insert_at < len(lines) and lines[insert_at].startswith("*("):
                insert_at += 1
            break

    if insert_at is None:
        return False

    new_lines = []
    # Top ablation finding first (most actionable)
    if ablations:
        ab_lines = format_ablation_summary(ablations, top_n=3)
        date = insights[0]["date"] if insights else "unknown"
        new_lines.append(f"- {date}: [ABLATION] top impacts: {'; '.join(ab_lines)}")
    for ins in insights:
        note = (
            f"- {ins['date']}: [{ins['variant']}] {ins['summary']}. {ins['diagnosis']}. Next: {ins['recommendation']}."
        )
        new_lines.append(note)

    for j, note in enumerate(new_lines):
        lines.insert(insert_at + j, note)

    program_md_path.write_text("\n".join(lines) + "\n")
    _log.info("append_to_program_md: wrote %d insights to %s", len(insights), program_md_path)
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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
            vid: bootstrap_f1_ci(scored_pairs, variant=vid, n_bootstrap=500, seed=42) for vid in metrics_by_variant
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


# ---------------------------------------------------------------------------
# Trend analysis — read learnings.jsonl across runs
# ---------------------------------------------------------------------------


def load_learnings() -> list[dict[str, Any]]:
    """Load all entries from learnings.jsonl. Returns empty list if missing."""
    if not LEARNINGS_FILE.exists():
        return []
    entries = []
    for line in LEARNINGS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                _log.warning("Skipping malformed learnings line: %s", line[:80])
    return entries


def compute_variant_trends(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group per-variant insights by variant ID, ordered by date.

    Returns {variant_id: [{date, f1, recall, precision, status}, ...]}
    for each variant that has appeared in any run. Ablation entries are excluded.
    """
    trends: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        if e.get("type") == "ablations":
            continue
        vid = e.get("variant")
        if not vid:
            continue
        trends.setdefault(vid, []).append(
            {
                "date": e.get("date", ""),
                "f1": float(e.get("f1", 0.0)),
                "recall": float(e.get("recall", 0.0)),
                "precision": float(e.get("precision", 0.0)),
                "status": e.get("status", ""),
            }
        )
    return trends


def compute_dimension_impacts(entries: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Extract ablation deltas per dimension across all runs.

    Returns {dimension: [delta_f1, delta_f1, ...]} so you can compute
    mean/median impact of each config dimension over time.
    """
    impacts: dict[str, list[float]] = {}
    for e in entries:
        if e.get("type") != "ablations":
            continue
        for ab in e.get("ablations", []):
            dim = ab.get("dimension", "")
            if dim:
                impacts.setdefault(dim, []).append(float(ab.get("delta_f1", 0.0)))
    return impacts
