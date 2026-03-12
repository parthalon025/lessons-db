"""Automatic Prompt Optimization (APO) for the eval pipeline.

Three strategies:
  feedback   — analyze false positives, ask optimizer to fix instruction flaws
  opro       — OPRO meta-prompt (DeepMind ICLR 2024), requires 32B+ local model
  opro-api   — OPRO via API (Claude/GPT-4o-mini), most reliable
"""

from __future__ import annotations

import json as _json
import logging
import re as _re
import sqlite3
from datetime import UTC, datetime
from typing import Any

from lessons_db.eval.variants import VARIANT_CONFIGS

_log = logging.getLogger(__name__)


def load_all_variant_configs(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Merge hand-authored VARIANT_CONFIGS with DB-stored APO variants.

    Hand-authored variants (A-H, M) always take precedence — a DB row
    with variant_id='A' is silently skipped to prevent config corruption.
    """
    merged: dict[str, dict[str, Any]] = dict(VARIANT_CONFIGS)
    rows = conn.execute("SELECT variant_id, config_json, instruction_text FROM prompt_variants").fetchall()
    for row in rows:
        vid = row["variant_id"]
        if vid in VARIANT_CONFIGS:
            _log.warning("Skipping DB variant %s — hand-authored variant exists", vid)
            continue
        config = _json.loads(row["config_json"])
        config["_instruction_text"] = row["instruction_text"]
        config["_apo_generated"] = True
        merged[vid] = config
    return merged


def parse_optimizer_candidates(response: str | None) -> list[dict[str, str]]:
    """Parse optimizer LLM response into instruction candidates.

    Expects a JSON array of objects with 'instruction' and 'hypothesis' keys.
    Strips <think> blocks, extracts first JSON array from surrounding text.
    Returns empty list on parse failure.
    """
    if not response:
        return []

    # Strip think blocks
    text = _re.sub(r"<think>.*?</think>", "", response, flags=_re.DOTALL | _re.IGNORECASE).strip()

    # Find JSON array in response
    match = _re.search(r"\[.*\]", text, flags=_re.DOTALL)
    if not match:
        _log.warning("No JSON array found in optimizer response")
        return []

    try:
        data = _json.loads(match.group())
    except _json.JSONDecodeError:
        _log.warning("Failed to parse optimizer response as JSON")
        return []

    if not isinstance(data, list):
        return []

    # Filter to valid candidates
    return [c for c in data if isinstance(c, dict) and "instruction" in c]


def build_feedback_prompt(
    instruction_text: str,
    f1: float,
    false_positives: list[dict[str, str]],
    n_candidates: int = 3,
) -> str:
    """Build optimizer prompt for feedback strategy.

    Shows the current instruction + its worst false positives and asks
    the optimizer to fix the instruction to prevent them.
    """
    fp_lines = []
    for i, fp in enumerate(false_positives[:5], 1):
        fp_lines.append(
            f"  {i}. Principle: \"{fp.get('principle', '')}\"\n"
            f"     Wrongly matched: \"{fp.get('target_title', '')}\" "
            f"(cluster: {fp.get('target_cluster_seed', '?')})\n"
            f"     Source cluster: {fp.get('cluster_seed', '?')}"
        )
    fp_block = "\n".join(fp_lines) if fp_lines else "  (no false positives available)"

    return (
        "You are improving a principle-extraction prompt for a lessons-learned system.\n\n"
        f"Current instruction (F1={f1:.3f}):\n"
        "---\n"
        f"{instruction_text}\n"
        "---\n\n"
        "This instruction produces principles that are too broad. Here are the worst\n"
        "false positives — cases where a principle wrongly matched an unrelated lesson:\n\n"
        f"{fp_block}\n\n"
        "Analyze what about the current instruction causes these false matches.\n"
        f"Then generate {n_candidates} improved instructions that would prevent them.\n\n"
        "Each instruction must:\n"
        "- Be a complete replacement (not a diff/edit)\n"
        "- Be 50-200 words\n"
        "- Target precision improvement specifically\n\n"
        "Return JSON array:\n"
        '[{"instruction": "...", "hypothesis": "why this should reduce false positives"}]'
    )


def build_opro_prompt(
    history: list[dict[str, Any]],
    n_candidates: int = 3,
) -> str:
    """Build OPRO-style meta-prompt with past prompts sorted by score.

    Follows DeepMind OPRO (ICLR 2024): solution-score pairs sorted
    ascending so the best prompt appears last (recency bias favors it).
    """
    sorted_history = sorted(history, key=lambda h: h.get("f1", 0.0))
    prompt_lines = []
    for entry in sorted_history:
        f1 = entry.get("f1", 0.0)
        text = entry.get("instruction_text", "")
        prompt_lines.append(f'[Score: {f1:.3f}] "{text}"')
    history_block = "\n\n".join(prompt_lines)

    return (
        "You are optimizing a prompt instruction for a principle-extraction system.\n"
        "Below are past instructions sorted by F1 score (higher = better).\n\n"
        f"{history_block}\n\n"
        "The main failure mode: high recall (>0.9) but low precision (0.07-0.17).\n"
        "Principles match too broadly across unrelated bug categories.\n\n"
        f"Generate {n_candidates} new instructions that should score higher. Each must:\n"
        "- Be a complete instruction (not a diff/edit)\n"
        "- Target precision improvement specifically\n"
        "- Be 50-200 words\n\n"
        "Return JSON array:\n"
        '[{"instruction": "...", "hypothesis": "why this should score higher"}]'
    )


def next_x_id(conn: sqlite3.Connection) -> str:
    """Generate the next available X-ID (X01, X02, ...) checking both code and DB."""
    existing_db = {r[0] for r in conn.execute("SELECT variant_id FROM prompt_variants").fetchall()}
    existing = existing_db | set(VARIANT_CONFIGS.keys())
    for i in range(1, 100):
        candidate = f"X{i:02d}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("X-ID space exhausted (X01-X99 all taken)")


def register_apo_variant(
    conn: sqlite3.Connection,
    instruction_text: str,
    parent_variant: str,
    strategy: str,
    optimizer_model: str | None = None,
    hypothesis: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> str:
    """Register an APO-generated variant in the DB. Returns the new variant_id.

    Config is inherited from parent_variant (must be in VARIANT_CONFIGS or DB),
    with prompt_id set to 'apo-generated'. config_overrides can override
    specific fields (e.g. temperature).
    """
    # Build config from parent
    if parent_variant in VARIANT_CONFIGS:
        config = dict(VARIANT_CONFIGS[parent_variant])
    else:
        row = conn.execute(
            "SELECT config_json FROM prompt_variants WHERE variant_id = ?",
            (parent_variant,),
        ).fetchone()
        config = _json.loads(row["config_json"]) if row else dict(VARIANT_CONFIGS.get("D", {}))

    config["prompt_id"] = "apo-generated"
    if config_overrides:
        config.update(config_overrides)

    variant_id = next_x_id(conn)
    conn.execute(
        """INSERT INTO prompt_variants
           (variant_id, instruction_text, config_json, parent_variant,
            strategy, optimizer_model, hypothesis, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            variant_id,
            instruction_text,
            _json.dumps(config),
            parent_variant,
            strategy,
            optimizer_model,
            hypothesis,
            datetime.now(UTC).isoformat(),
        ),
    )
    conn.commit()
    return variant_id
