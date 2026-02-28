"""Conflict detector — flag semantic near-duplicates with opposing polarity.

For each new lesson, find top-3 nearest neighbors in LanceDB.
If similarity > CONFLICT_THRESHOLD AND polarity differs → flag conflict.
Conflict blocks auto-approve and routes to capture_drafts with note.
"""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from lessons_db.vectors import semantic_search

_log = logging.getLogger(__name__)

CONFLICT_THRESHOLD = 0.85


@dataclass
class ConflictResult:
    has_conflict: bool
    conflicting_lesson_id: int | None
    similarity: float
    note: str


def _get_polarity(conn: sqlite3.Connection, lesson_id: int) -> str:
    try:
        row = conn.execute("SELECT polarity FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        return row["polarity"] if row and row["polarity"] else "negative"
    except Exception as exc:
        _log.warning("_get_polarity failed for lesson %d: %s", lesson_id, exc)
        return "negative"


def detect_conflicts(
    conn: sqlite3.Connection,
    lance_dir: Path,
    lesson_id: int,
    snippet: str,
) -> ConflictResult:
    """Check if snippet conflicts with existing lessons.

    Returns ConflictResult. has_conflict=True blocks auto-approve.
    """
    try:
        neighbors = semantic_search(snippet, lance_dir, top_k=3)
    except Exception as exc:
        _log.warning("conflict check skipped — vector search failed: %s", exc)
        return ConflictResult(
            has_conflict=False, conflicting_lesson_id=None, similarity=0.0, note="vector search unavailable"
        )

    new_polarity = _get_polarity(conn, lesson_id)

    for neighbor in neighbors:
        neighbor_id = neighbor.get("id")
        if neighbor_id is None:
            _log.warning("conflict_detector: neighbor missing 'id' field, skipping")
            continue
        distance = neighbor.get("_distance", 1.0)
        similarity = max(0.0, 1.0 - distance)

        if similarity < CONFLICT_THRESHOLD:
            continue
        if neighbor_id == lesson_id:
            continue

        neighbor_polarity = _get_polarity(conn, neighbor_id)
        if neighbor_polarity != new_polarity:
            note = (
                f"Conflict: lesson {lesson_id} ({new_polarity}) similarity "
                f"{similarity:.2f} with lesson {neighbor_id} ({neighbor_polarity})"
            )
            _log.warning(note)
            return ConflictResult(
                has_conflict=True,
                conflicting_lesson_id=neighbor_id,
                similarity=similarity,
                note=note,
            )

    return ConflictResult(has_conflict=False, conflicting_lesson_id=None, similarity=0.0, note="")
