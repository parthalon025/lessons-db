"""Test set selection: source lessons and transfer targets."""

import logging
import random
import sqlite3
from typing import Any

from lessons_db.eval.variants import VALID_GROUP_BY

_log = logging.getLogger(__name__)

# Lessons seen in this many or more eval runs are considered "overused" and
# deprioritised during selection (fresh lessons are preferred first).
SEEN_IN_EVAL_THRESHOLD = 4


def select_source_lessons(conn: sqlite3.Connection, per_cluster: int = 4, group_by: str = "category") -> list[dict]:
    """Select source lessons for evaluation.

    Finds all groups (by *group_by* column) with >= 3 single-loop lessons,
    then picks up to ``per_cluster`` lessons per group maximising category
    diversity.

    Lessons with ``seen_in_eval >= SEEN_IN_EVAL_THRESHOLD`` are deprioritised:
    fresh lessons fill slots first; overused ones are only drawn when the
    group has fewer fresh candidates than ``per_cluster``.

    Args:
        group_by: Column to group lessons by. Must be ``"category"`` (default)
            or ``"cluster_seed"``.

    Returns a flat list of lesson dicts with keys:
        id, title, one_liner, description, cluster_seed, category, seen_in_eval
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

        # Fetch all single-loop lessons in this group, fresh ones first
        rows = conn.execute(
            f"""
            SELECT id, title, one_liner, description, cluster_seed, category, seen_in_eval
            FROM lessons
            WHERE {group_by} = ?
              AND (loop_level IS NULL OR loop_level = 'single')
            ORDER BY seen_in_eval ASC, id ASC
            """,
            (group_value,),
        ).fetchall()

        lessons = [dict(r) for r in rows]

        # Two-pass: prefer fresh lessons; fall back to overused only if needed
        fresh = [l for l in lessons if l["seen_in_eval"] < SEEN_IN_EVAL_THRESHOLD]
        overused = [l for l in lessons if l["seen_in_eval"] >= SEEN_IN_EVAL_THRESHOLD]

        selected = _select_diverse(fresh, per_cluster)
        if len(selected) < per_cluster:
            remaining = per_cluster - len(selected)
            selected_ids = {l["id"] for l in selected}
            filler = [l for l in overused if l["id"] not in selected_ids]
            selected.extend(_select_diverse(filler, remaining))

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


def increment_eval_seen(conn: sqlite3.Connection, lesson_ids: list[int]) -> None:
    """Increment the seen_in_eval counter for each lesson in lesson_ids.

    Call this after a successful eval-generate run so that the selection
    function can deprioritise frequently sampled lessons on subsequent runs.
    Empty list is a no-op.
    """
    if not lesson_ids:
        return
    placeholders = ",".join("?" * len(lesson_ids))
    conn.execute(
        f"UPDATE lessons SET seen_in_eval = seen_in_eval + 1 WHERE id IN ({placeholders})",
        lesson_ids,
    )
    conn.commit()


def split_holdout(
    sources: list[dict[str, Any]],
    holdout_fraction: float = 0.3,
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split source lessons into a dev set and a held-out test set.

    The test set is used for final validation only — never for optimising
    variant prompts — to prevent Goodhart overfitting to the eval sample.

    Args:
        sources: Flat list of lesson dicts (as returned by select_source_lessons).
        holdout_fraction: Fraction to reserve for the test set (default 0.3).
        seed: Optional random seed for reproducibility.

    Returns:
        (dev_set, test_set) — disjoint lists, dev + test == sources.
    """
    if not sources:
        return [], []

    rng = random.Random(seed)  # noqa: S311 — not cryptographic, sampling only
    shuffled = list(sources)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return [], [shuffled[0]]

    test_size = max(1, round(len(shuffled) * holdout_fraction))
    test_set = shuffled[:test_size]
    dev_set = shuffled[test_size:]
    return dev_set, test_set
