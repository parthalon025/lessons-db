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


def save_best(variant_id: str, metrics: dict[str, float]) -> None:
    """Persist a new best F1 record."""
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    BEST_JSON.write_text(
        json.dumps(
            {
                "variant": variant_id,
                "f1": metrics["f1"],
                "recall": metrics.get("recall", 0.0),
                "precision": metrics.get("precision", 0.0),
                "updated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )


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
            save_best(variant_id, m)
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


def save_learnings(insights: list[dict[str, Any]]) -> None:
    """Append insights to the append-only learnings.jsonl audit trail."""
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with LEARNINGS_FILE.open("a") as fh:
        for ins in insights:
            fh.write(json.dumps(ins) + "\n")


def append_to_program_md(insights: list[dict[str, Any]], program_md_path: Path) -> bool:
    """Append insights to the '## Learned so far' section of program.md.

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
    for ins in insights:
        note = (
            f"- {ins['date']}: [{ins['variant']}] {ins['summary']}. "
            f"{ins['diagnosis']}. Next: {ins['recommendation']}."
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
) -> list[dict[str, Any]]:
    """Derive insights, persist them, and optionally update program.md.

    This is the single call site used by the CLI. Always runs — no guard on
    whether variants improved or not. Every run teaches something.

    Returns the list of insight dicts for CLI display.
    """
    insights = derive_insights(metrics_by_variant, variant_configs, run_date)

    try:
        save_learnings(insights)
    except Exception as exc:
        _log.warning("save_learnings failed (non-fatal): %s", exc)

    if program_md_path:
        try:
            append_to_program_md(insights, program_md_path)
        except Exception as exc:
            _log.warning("append_to_program_md failed (non-fatal): %s", exc)

    return insights
