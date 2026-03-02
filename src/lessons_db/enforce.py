"""Enforcement escalation ladder — recurrence tracking and tier promotion."""

import logging
import sqlite3
from datetime import date

from lessons_db.db import get_lesson, update_lesson

_log = logging.getLogger(__name__)

# Escalation tiers indexed by recurrence count (after increment).
# recurrence 1 → documentation/emerging
# recurrence 2 → semgrep_warning/established
# recurrence 3 → semgrep_error/core
# recurrence 4+ → semgrep_autofix/core

BLOCKING_ENFORCEMENT: frozenset[str] = frozenset({"semgrep_error", "semgrep_autofix"})


def should_block(enforcement: str) -> bool:
    """Return True if this enforcement level should block a pre-edit action."""
    return enforcement in BLOCKING_ENFORCEMENT


_TIERS = {
    1: {"enforcement": "documentation", "confidence": "emerging"},
    2: {"enforcement": "semgrep_warning", "confidence": "established"},
    3: {"enforcement": "semgrep_error", "confidence": "core"},
}
_MAX_TIER = {"enforcement": "semgrep_autofix", "confidence": "core"}


def check_escalation(conn: sqlite3.Connection, lesson_id: int) -> dict:
    """Increment recurrence and escalate enforcement tier.

    Returns an action dict describing the new state and what should happen.
    """
    lesson = get_lesson(conn, lesson_id)
    if lesson is None:
        raise ValueError(f"Lesson {lesson_id} not found")

    new_count = lesson["recurrence_count"] + 1
    tier = _TIERS.get(new_count, _MAX_TIER)

    enforcement = tier["enforcement"]
    confidence = tier["confidence"]

    # Determine action flags based on tier
    generate_rule = new_count >= 2
    add_precommit = new_count == 3
    add_autofix = new_count >= 4

    update_lesson(
        conn,
        lesson_id,
        {
            "enforcement": enforcement,
            "confidence": confidence,
            "recurrence_count": new_count,
            "last_hit_date": date.today().isoformat(),
        },
    )

    return {
        "lesson_id": lesson_id,
        "level": enforcement,
        "recurrence_count": new_count,
        "generate_rule": generate_rule,
        "add_precommit": add_precommit,
        "add_autofix": add_autofix,
    }
