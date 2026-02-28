"""Semgrep registry delta import.

Imports python-pack rules from semgrep/semgrep-rules as lesson stubs.
Idempotent — checks enforcement_rules table for existing rule_id before insert.

Usage (CLI):
    lessons-db import semgrep           # full import
    lessons-db import semgrep --delta   # only new rules since last import
"""

import json
import logging
import sqlite3
import subprocess
from datetime import date

from lessons_db.db import insert_lesson, set_scan_state

_log = logging.getLogger(__name__)

SEVERITY_MAP = {"ERROR": 5, "WARNING": 3, "INFO": 1}
DEFAULT_SEVERITY = 2

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "security": ["hardcoded", "injection", "xss", "auth", "secret", "password", "crypto", "sql"],
    "performance": ["n+1", "loop", "cache", "slow", "inefficient"],
    "db-queries": ["sql", "query", "orm", "database"],
    "async": ["async", "await", "coroutine", "event loop"],
}


def semgrep_severity_to_int(severity: str) -> int:
    """Map Semgrep severity string to integer (1-5)."""
    return SEVERITY_MAP.get(severity.upper(), DEFAULT_SEVERITY)


def _infer_category(rule: dict) -> str:
    """Infer lesson category from rule id and message."""
    text = f"{rule.get('id', '')} {rule.get('message', '')}".lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return cat
    return "security"  # Semgrep rules are predominantly security


def parse_semgrep_rule(rule: dict) -> dict:
    """Extract lesson fields from a Semgrep rule dict."""
    rule_id = rule.get("id", "")
    message = rule.get("message", rule_id)
    severity = semgrep_severity_to_int(rule.get("severity", "WARNING"))
    category = _infer_category(rule)
    pattern = rule.get("pattern") or rule.get("pattern-regex", "")
    return {
        "title": message,
        "one_liner": f"Semgrep rule: {rule_id}",
        "category": category,
        "severity": severity,
        "rule_id": rule_id,
        "regex": pattern or None,
    }


def import_rule_as_lesson_stub(conn: sqlite3.Connection, rule: dict) -> int:
    """Import one Semgrep rule as a lesson stub. Idempotent by rule_id.

    Returns existing or new lesson id.
    """
    parsed = parse_semgrep_rule(rule)
    rule_id = parsed["rule_id"]

    # Check if already imported
    existing = conn.execute("SELECT lesson_id FROM enforcement_rules WHERE rule_id = ?", (rule_id,)).fetchone()
    if existing:
        return existing["lesson_id"]

    today = date.today().isoformat()
    lesson_id = insert_lesson(
        conn,
        {
            "title": parsed["title"],
            "one_liner": parsed["one_liner"],
            "category": parsed["category"],
            "severity": parsed["severity"],
            "tier": "observation",
            "source": "semgrep_registry",
            "created_date": today,
        },
    )

    conn.execute(
        """INSERT INTO enforcement_rules (lesson_id, rule_id, rule_type, created_date)
           VALUES (?, ?, 'semgrep', ?)""",
        (lesson_id, rule_id, today),
    )
    conn.commit()
    return lesson_id


def run_delta_import(conn: sqlite3.Connection, delta_only: bool = True) -> dict:
    """Run Semgrep registry import.

    Fetches all rules from the Semgrep python pack each run (the Semgrep CLI has
    no incremental API — delta behavior is achieved via idempotency: rules already
    in enforcement_rules are skipped). On first run ~300 rules are imported; on
    subsequent runs all are skipped in O(1) per rule via indexed rule_id lookup.

    delta_only=True: skip rules already in enforcement_rules table (default).
    delta_only=False: reimport all rules (useful for reprocessing).
    Returns {"imported": int, "skipped": int, "errors": int}.
    """
    imported = skipped = errors = 0

    try:
        result = subprocess.run(  # noqa: S603
            ["semgrep", "--config", "p/python", "--json", "--quiet", "--dry-run"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=60,
        )
        raw = result.stdout.strip()
        if not raw:
            _log.warning("semgrep returned no output")
            return {"imported": 0, "skipped": 0, "errors": 0}
        data = json.loads(raw)
        rules = data.get("rules", [])
    except Exception as exc:
        _log.error("semgrep registry fetch failed: %s", exc)
        return {"imported": 0, "skipped": 0, "errors": 1}

    for rule in rules:
        try:
            rule_id = rule.get("id", "")
            if delta_only:
                existing = conn.execute("SELECT 1 FROM enforcement_rules WHERE rule_id = ?", (rule_id,)).fetchone()
                if existing:
                    skipped += 1
                    continue
            import_rule_as_lesson_stub(conn, rule)
            imported += 1
        except Exception as exc:
            _log.warning("failed to import rule %s: %s", rule.get("id"), exc)
            errors += 1

    set_scan_state(conn, "last_semgrep_import", date.today().isoformat())
    return {"imported": imported, "skipped": skipped, "errors": errors}
