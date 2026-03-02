"""Learning pipeline: surfacing event recording and composite relevance scoring."""

import logging
import random
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

_log = logging.getLogger(__name__)

VALID_OUTCOMES = ("heeded", "dismissed", "false_positive", "recurrence", "exception_noted")


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
    """Update outcome for a surfacing event. outcome must be one of VALID_OUTCOMES."""
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"Invalid outcome '{outcome}'. Must be one of: {', '.join(VALID_OUTCOMES)}.")
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
        "unknown": conn.execute("SELECT COUNT(*) FROM surfacing_events WHERE outcome='unknown'").fetchone()[0],
        "heed_rate": round(heeded / total, 2) if total > 0 else None,
        "avg_per_session": round(avg_row or 0.0, 1),
    }


def _match_patterns(patterns: list[tuple[str, str]], diff_text: str) -> tuple[bool, str]:
    """Check if any (pattern, source) pair matches in diff_text.

    Returns (matched, source_label). Falls back to substring if regex invalid.
    """
    for pattern, source in patterns:
        try:
            if re.search(pattern, diff_text, re.MULTILINE):
                return True, source
        except re.error:
            if pattern in diff_text:
                return True, source
    return False, "no_match"


def _collect_patterns(
    row: sqlite3.Row,
    dp_by_lesson: dict[int, list[str]],
) -> list[tuple[str, str]]:
    """Gather detection patterns for a lesson from both lesson column and table."""
    patterns: list[tuple[str, str]] = []
    lesson_dp = row["detection_pattern"]
    if lesson_dp and lesson_dp.strip():
        patterns.append((lesson_dp.strip(), "lesson.detection_pattern"))
    for regex_str in dp_by_lesson.get(row["lesson_id"], []):
        patterns.append((regex_str, "detection_patterns_table"))
    return patterns


def evaluate_commit(
    conn: sqlite3.Connection,
    diff_text: str,
    hours: int = 24,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate recent surfacing events against a commit diff.

    For each unknown surfacing event within the lookback window, checks if the
    lesson's anti-pattern appears in the diff text.

    - If anti-pattern IS present in the diff: outcome = 'dismissed'
    - If anti-pattern is NOT present: outcome = 'heeded'
    - Lessons with no detection pattern are skipped.

    Returns a list of dicts: [{event_id, lesson_id, title, outcome, pattern_source}, ...]
    """
    cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

    rows = conn.execute(
        "SELECT se.id AS event_id, se.lesson_id, l.title, l.detection_pattern "
        "FROM surfacing_events se "
        "JOIN lessons l ON l.id = se.lesson_id "
        "WHERE se.outcome = 'unknown' AND se.timestamp >= ? "
        "ORDER BY se.timestamp DESC",
        [cutoff],
    ).fetchall()

    if not rows:
        return []

    dp_rows = conn.execute(
        "SELECT lesson_id, regex FROM detection_patterns " "WHERE pattern_type IN ('syntactic', 'regex')"
    ).fetchall()
    dp_by_lesson: dict[int, list[str]] = {}
    for dp in dp_rows:
        dp_by_lesson.setdefault(dp["lesson_id"], []).append(dp["regex"])

    results: list[dict[str, Any]] = []
    seen_lessons: dict[int, dict[str, str]] = {}

    for row in rows:
        event_id = row["event_id"]
        lesson_id = row["lesson_id"]

        if lesson_id in seen_lessons:
            prev = seen_lessons[lesson_id]
            if not dry_run:
                record_outcome(conn, event_id, prev["outcome"])
            results.append({**prev, "event_id": event_id, "title": row["title"]})
            continue

        patterns = _collect_patterns(row, dp_by_lesson)
        if not patterns:
            continue

        matched, pattern_source = _match_patterns(patterns, diff_text)
        outcome = "dismissed" if matched else "heeded"
        if not dry_run:
            record_outcome(conn, event_id, outcome)

        entry = {
            "event_id": event_id,
            "lesson_id": lesson_id,
            "title": row["title"],
            "outcome": outcome,
            "pattern_source": pattern_source,
        }
        seen_lessons[lesson_id] = {"outcome": outcome, "pattern_source": pattern_source, "lesson_id": lesson_id}
        results.append(entry)

    return results


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


def update_win_streak(conn: sqlite3.Connection, category: str, won: bool) -> dict:
    """Update win streak for category. Returns current streak info.

    If won: increment current_streak, update longest_streak if needed.
    If not won: reset current_streak to 0.
    Upserts into win_streaks table.
    """
    now = datetime.now(UTC).isoformat()
    row = conn.execute(
        "SELECT current_streak, longest_streak FROM win_streaks WHERE category = ?",
        [category],
    ).fetchone()

    if row is None:
        current = 1 if won else 0
        longest = current
        conn.execute(
            "INSERT INTO win_streaks (category, current_streak, longest_streak, last_updated) " "VALUES (?, ?, ?, ?)",
            [category, current, longest, now],
        )
    else:
        if won:
            current = row["current_streak"] + 1
            longest = max(row["longest_streak"], current)
        else:
            current = 0
            longest = row["longest_streak"]
        conn.execute(
            "UPDATE win_streaks SET current_streak = ?, longest_streak = ?, last_updated = ? " "WHERE category = ?",
            [current, longest, now, category],
        )
    conn.commit()
    return {"category": category, "current_streak": current, "longest_streak": longest}


def should_surface_positive(conn: sqlite3.Connection, category: str) -> tuple[bool, dict]:
    """30% probability gate for positive recognition (Skinner variable-ratio).

    Returns (should_surface, streak_info) where streak_info has
    current_streak, longest_streak, category.
    """
    row = conn.execute(
        "SELECT current_streak, longest_streak FROM win_streaks WHERE category = ?",
        [category],
    ).fetchone()

    if row is None:
        streak_info = {"category": category, "current_streak": 0, "longest_streak": 0}
    else:
        streak_info = {
            "category": category,
            "current_streak": row["current_streak"],
            "longest_streak": row["longest_streak"],
        }

    should = random.random() < 0.3  # noqa: S311 — non-crypto probability gate
    return should, streak_info


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


def find_exceptions(conn: sqlite3.Connection, lookback_sessions: int = 5) -> list[dict]:
    """Find anti-patterns absent from recent sessions that previously appeared (SFBT exception-finding).

    Returns list of dicts with: lesson_id, title, category, absent_sessions
    (consecutive sessions without this pattern).

    Logic:
    - Get distinct session_ids from surfacing_events ordered by timestamp DESC,
      limit to lookback_sessions.
    - For each negative lesson that has been surfaced at least once historically
      with outcome='dismissed':
      - Check if it appeared (outcome='dismissed') in any of the recent sessions.
      - If absent from ALL recent sessions, it's an "exception" (internalized pattern).
    - Return sorted by absent_sessions descending (most consistently absent first).
    """
    # Get the N most recent distinct session_ids
    recent_sessions = conn.execute(
        "SELECT session_id, MAX(timestamp) AS latest "
        "FROM surfacing_events "
        "WHERE session_id IS NOT NULL "
        "GROUP BY session_id "
        "ORDER BY latest DESC "
        "LIMIT ?",
        [lookback_sessions],
    ).fetchall()

    if not recent_sessions:
        return []

    recent_session_ids = [r["session_id"] for r in recent_sessions]
    num_recent = len(recent_session_ids)

    # Find all negative lessons that have ever been dismissed (i.e., anti-pattern recurred)
    historically_dismissed = conn.execute(
        "SELECT DISTINCT se.lesson_id, l.title, l.category "
        "FROM surfacing_events se "
        "JOIN lessons l ON l.id = se.lesson_id "
        "WHERE se.outcome = 'dismissed' AND l.polarity = 'negative'"
    ).fetchall()

    if not historically_dismissed:
        return []

    exceptions: list[dict] = []

    for row in historically_dismissed:
        lesson_id = row["lesson_id"]

        # Count how many of the recent sessions had this lesson dismissed
        placeholders = ", ".join("?" for _ in recent_session_ids)
        dismissed_in_recent = conn.execute(
            f"SELECT COUNT(DISTINCT session_id) FROM surfacing_events "
            f"WHERE lesson_id = ? AND outcome = 'dismissed' "
            f"AND session_id IN ({placeholders})",
            [lesson_id, *recent_session_ids],
        ).fetchone()[0]

        # If absent from ALL recent sessions, it's an exception
        if dismissed_in_recent == 0:
            exceptions.append(
                {
                    "lesson_id": lesson_id,
                    "title": row["title"],
                    "category": row["category"] or "uncategorized",
                    "absent_sessions": num_recent,
                }
            )

    # Sort by absent_sessions descending (all are equal here, but stable for future extension)
    exceptions.sort(key=lambda e: (-e["absent_sessions"], e["lesson_id"]))
    return exceptions
