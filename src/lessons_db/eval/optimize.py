"""Automatic Prompt Optimization (APO) for the eval pipeline.

Three strategies:
  feedback   — analyze false positives, ask optimizer to fix instruction flaws
  opro       — OPRO meta-prompt (DeepMind ICLR 2024), requires 32B+ local model
  opro-api   — OPRO via API (Claude/GPT-4o-mini), most reliable
"""

from __future__ import annotations

import json as _json
import logging
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
