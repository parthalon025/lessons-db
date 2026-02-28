"""Learning pipeline: surfacing event recording and composite relevance scoring."""

import logging
import sqlite3
from datetime import UTC, datetime
from typing import Any

_log = logging.getLogger(__name__)

_VALID_OUTCOMES = ("heeded", "dismissed", "false_positive", "recurrence")


def record_surfacing(
    conn: sqlite3.Connection, lesson_id: int, hook_point: str, context: str = "", session_id: str | None = None
) -> int:
    """Record a surfacing event. Returns event ID for later outcome update."""
    cursor = conn.execute(
        "INSERT INTO surfacing_events "
        "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
        "VALUES (?, ?, ?, 'unknown', ?, ?)",
        [lesson_id, hook_point, context, datetime.now(UTC).isoformat(), session_id],
    )
    conn.commit()
    lastrowid: int = cursor.lastrowid  # type: ignore[assignment]
    return lastrowid


def record_outcome(conn: sqlite3.Connection, event_id: int, outcome: str) -> None:
    """Update outcome for a surfacing event. outcome must be one of _VALID_OUTCOMES."""
    if outcome not in _VALID_OUTCOMES:
        raise ValueError(f"Invalid outcome '{outcome}'. Must be one of: {', '.join(_VALID_OUTCOMES)}.")
    cursor = conn.execute(
        "UPDATE surfacing_events SET outcome = ? WHERE id = ?",
        [outcome, event_id],
    )
    conn.commit()
    if cursor.rowcount == 0:
        _log.warning("record_outcome: no event found with id=%d", event_id)
        raise ValueError(f"No surfacing event found with id={event_id}.")


def dismiss_latest(conn: sqlite3.Connection, lesson_id: int) -> bool:
    """Mark the most recent unknown surfacing event for lesson_id as false_positive.

    Returns True if an event was found and updated, False if none existed.
    """
    row = conn.execute(
        "SELECT id FROM surfacing_events " "WHERE lesson_id = ? AND outcome = 'unknown' " "ORDER BY id DESC LIMIT 1",
        [lesson_id],
    ).fetchone()
    if row is None:
        return False
    record_outcome(conn, row["id"], "false_positive")
    return True


def relevance_score(conn: sqlite3.Connection, lesson_id: int, context: str, semantic_sim: float) -> float:
    """Composite relevance score.

    score = 0.5 * semantic_sim
           + 0.3 * outcome_rate (heeded ratio in similar contexts)
           + 0.2 * recurrence_score (normalized near-miss + recurrence count)
    """
    outcome = _outcome_rate(conn, lesson_id, context)
    recurrence = _recurrence_score(conn, lesson_id)
    return round(0.5 * semantic_sim + 0.3 * outcome + 0.2 * recurrence, 4)


def surfacing_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Summary stats for the status command and efficiency tracking."""
    total = conn.execute("SELECT COUNT(*) FROM surfacing_events").fetchone()[0]
    heeded = conn.execute("SELECT COUNT(*) FROM surfacing_events WHERE outcome='heeded'").fetchone()[0]
    dismissed = conn.execute("SELECT COUNT(*) FROM surfacing_events WHERE outcome='dismissed'").fetchone()[0]
    avg_row = conn.execute(
        "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM surfacing_events GROUP BY session_id)"
    ).fetchone()[0]

    return {
        "total_surfacing_events": total,
        "heeded": heeded,
        "dismissed": dismissed,
        "unknown": total - heeded - dismissed,
        "heed_rate": round(heeded / total, 2) if total > 0 else None,
        "avg_per_session": round(avg_row or 0.0, 1),
    }


def _outcome_rate(conn: sqlite3.Connection, lesson_id: int, context: str) -> float:
    """Ratio of heeded outcomes for this lesson in similar contexts.
    Returns 0.5 (neutral) if no outcome data exists.

    Note: uses LIKE '%{context[:50]}%' matching. Short or common context strings
    (e.g. 'hub', 'db') may match unrelated events — scores in those cases are
    contaminated by false positives. Acceptable for local tooling; revisit if
    context strings become high-cardinality."""
    ctx_prefix = context[:50] if context else ""
    rows = conn.execute(
        "SELECT outcome FROM surfacing_events WHERE lesson_id = ? AND context LIKE ? AND outcome != 'unknown'",
        [lesson_id, f"%{ctx_prefix}%"],
    ).fetchall()
    if not rows:
        return 0.5
    heeded = sum(1 for r in rows if r["outcome"] == "heeded")
    return heeded / len(rows)


def _recurrence_score(conn: sqlite3.Connection, lesson_id: int) -> float:
    """Normalized recurrence + near-miss count. Caps at 1.0 (10+ events = max)."""
    row = conn.execute(
        "SELECT recurrence_count, "
        "(SELECT COUNT(*) FROM near_misses WHERE lesson_id = l.id) AS nm "
        "FROM lessons l WHERE l.id = ?",
        [lesson_id],
    ).fetchone()
    if not row:
        _log.warning("_recurrence_score: lesson %d not found", lesson_id)
        return 0.0
    raw = (row["recurrence_count"] or 0) + (row["nm"] or 0)
    return min(raw / 10.0, 1.0)
