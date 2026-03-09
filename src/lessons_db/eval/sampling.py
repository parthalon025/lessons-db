"""Test set selection: source lessons and transfer targets."""

import logging
import sqlite3

from lessons_db.eval.variants import VALID_GROUP_BY

_log = logging.getLogger(__name__)


def select_source_lessons(conn: sqlite3.Connection, per_cluster: int = 4, group_by: str = "category") -> list[dict]:
    """Select source lessons for evaluation.

    Finds all groups (by *group_by* column) with >= 3 single-loop lessons,
    then picks up to ``per_cluster`` lessons per group maximising category
    diversity.

    Args:
        group_by: Column to group lessons by. Must be ``"category"`` (default)
            or ``"cluster_seed"``.

    Returns a flat list of lesson dicts with keys:
        id, title, one_liner, description, cluster_seed, category
    """
    if group_by not in VALID_GROUP_BY:
        raise ValueError(f"group_by must be one of {VALID_GROUP_BY!r}, got {group_by!r}")

    # Find qualifying groups (>= 3 single-loop lessons)
    cluster_rows = conn.execute(
        f"""
        SELECT {group_by}, COUNT(*) AS cnt
        FROM lessons
        WHERE {group_by} IS NOT NULL
          AND (loop_level IS NULL OR loop_level = 'single')
        GROUP BY {group_by}
        HAVING cnt >= 3
        ORDER BY {group_by}
        """
    ).fetchall()

    results: list[dict] = []

    for crow in cluster_rows:
        group_value = crow[group_by]

        # Fetch all single-loop lessons in this group
        rows = conn.execute(
            f"""
            SELECT id, title, one_liner, description, cluster_seed, category
            FROM lessons
            WHERE {group_by} = ?
              AND (loop_level IS NULL OR loop_level = 'single')
            ORDER BY id
            """,
            (group_value,),
        ).fetchall()

        lessons = [dict(r) for r in rows]

        # Greedy category-diversity selection
        selected = _select_diverse(lessons, per_cluster)
        results.extend(selected)

    return results


def _select_diverse(lessons: list[dict], limit: int) -> list[dict]:
    """Greedy algorithm: first pick one from each unique category, then fill remaining slots."""
    if not lessons:
        return []

    selected: list[dict] = []
    used_ids: set[int] = set()

    # Pass 1: one per unique category
    seen_cats: set[str] = set()
    for lesson in lessons:
        cat = lesson.get("category")
        if cat not in seen_cats and len(selected) < limit:
            selected.append(lesson)
            used_ids.add(lesson["id"])
            seen_cats.add(cat)

    # Pass 2: fill remaining slots from unused lessons
    for lesson in lessons:
        if len(selected) >= limit:
            break
        if lesson["id"] not in used_ids:
            selected.append(lesson)
            used_ids.add(lesson["id"])

    return selected


def select_transfer_targets(
    conn: sqlite3.Connection,
    source_id: int,
    group_value: str,
    count_same: int = 2,
    count_diff: int = 2,
    group_by: str = "category",
) -> dict[str, list[dict]]:
    """Select transfer target lessons for a given source lesson.

    Args:
        group_value: The value of the *group_by* column for the source lesson.
        group_by: Column to group by (``"category"`` or ``"cluster_seed"``).

    Returns:
        {"same_cluster": [...], "diff_cluster": [...]}

    - same_cluster: other lessons from same group, excluding source,
      preferring different categories (sort: different category first).
    - diff_cluster: lessons from other groups, selected randomly.
    - All single-loop only.
    """
    if group_by not in VALID_GROUP_BY:
        raise ValueError(f"group_by must be one of {VALID_GROUP_BY!r}, got {group_by!r}")

    # Get source lesson's category for preference sorting
    source_row = conn.execute("SELECT category FROM lessons WHERE id = ?", (source_id,)).fetchone()
    if source_row is None:
        _log.warning("select_transfer_targets: source_id=%d not found", source_id)
    source_category = source_row["category"] if source_row else None

    # Same group, excluding source, single-loop only
    # Sort: different category first (0 before 1), then by id for stability
    same_rows = conn.execute(
        f"""
        SELECT id, title, one_liner, description, cluster_seed, category
        FROM lessons
        WHERE {group_by} = ?
          AND id != ?
          AND (loop_level IS NULL OR loop_level = 'single')
        ORDER BY
            CASE WHEN category = ? THEN 1 ELSE 0 END,
            id
        """,
        (group_value, source_id, source_category),
    ).fetchall()

    same_cluster = [dict(r) for r in same_rows[:count_same]]

    # Different group, single-loop, random selection
    diff_rows = conn.execute(
        f"""
        SELECT id, title, one_liner, description, cluster_seed, category
        FROM lessons
        WHERE {group_by} IS NOT NULL
          AND {group_by} != ?
          AND (loop_level IS NULL OR loop_level = 'single')
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (group_value, count_diff),
    ).fetchall()

    diff_cluster = [dict(r) for r in diff_rows]

    return {
        "same_cluster": same_cluster,
        "diff_cluster": diff_cluster,
    }
