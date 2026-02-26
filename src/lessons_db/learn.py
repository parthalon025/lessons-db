"""Learning pipeline: surfacing event recording and composite relevance scoring."""

from datetime import datetime


def record_surfacing(conn, lesson_id: int, hook_point: str,
                     context: str = "", session_id: str | None = None) -> int:
    """Record a surfacing event. Returns event ID for later outcome update."""
    cursor = conn.execute(
        "INSERT INTO surfacing_events "
        "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
        "VALUES (?, ?, ?, 'unknown', ?, ?)",
        [lesson_id, hook_point, context, datetime.now().isoformat(), session_id],
    )
    conn.commit()
    return cursor.lastrowid


def record_outcome(conn, event_id: int, outcome: str) -> None:
    """Update outcome for a surfacing event. outcome must be 'heeded' or 'dismissed'."""
    if outcome not in ("heeded", "dismissed"):
        raise ValueError(f"Invalid outcome '{outcome}'. Must be 'heeded' or 'dismissed'.")
    conn.execute(
        "UPDATE surfacing_events SET outcome = ? WHERE id = ?",
        [outcome, event_id],
    )
    conn.commit()


def relevance_score(conn, lesson_id: int, context: str,
                    semantic_sim: float) -> float:
    """Composite relevance score.

    score = 0.5 * semantic_sim
           + 0.3 * outcome_rate (heeded ratio in similar contexts)
           + 0.2 * recurrence_score (normalized near-miss + recurrence count)
    """
    outcome = _outcome_rate(conn, lesson_id, context)
    recurrence = _recurrence_score(conn, lesson_id)
    return round(0.5 * semantic_sim + 0.3 * outcome + 0.2 * recurrence, 4)


def surfacing_stats(conn) -> dict:
    """Summary stats for the status command and efficiency tracking."""
    total = conn.execute("SELECT COUNT(*) FROM surfacing_events").fetchone()[0]
    heeded = conn.execute(
        "SELECT COUNT(*) FROM surfacing_events WHERE outcome='heeded'"
    ).fetchone()[0]
    dismissed = conn.execute(
        "SELECT COUNT(*) FROM surfacing_events WHERE outcome='dismissed'"
    ).fetchone()[0]
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


def _outcome_rate(conn, lesson_id: int, context: str) -> float:
    """Ratio of heeded outcomes for this lesson in similar contexts.
    Returns 0.5 (neutral) if no outcome data exists."""
    ctx_prefix = context[:50] if context else ""
    rows = conn.execute(
        "SELECT outcome FROM surfacing_events "
        "WHERE lesson_id = ? AND context LIKE ? AND outcome != 'unknown'",
        [lesson_id, f"%{ctx_prefix}%"],
    ).fetchall()
    if not rows:
        return 0.5
    heeded = sum(1 for r in rows if r["outcome"] == "heeded")
    return heeded / len(rows)


def _recurrence_score(conn, lesson_id: int) -> float:
    """Normalized recurrence + near-miss count. Caps at 1.0 (10+ events = max)."""
    row = conn.execute(
        "SELECT recurrence_count, "
        "(SELECT COUNT(*) FROM near_misses WHERE lesson_id = l.id) AS nm "
        "FROM lessons l WHERE l.id = ?",
        [lesson_id],
    ).fetchone()
    if not row:
        return 0.0
    raw = (row["recurrence_count"] or 0) + (row["nm"] or 0)
    return min(raw / 10.0, 1.0)
