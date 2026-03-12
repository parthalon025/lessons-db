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
