"""Transfer-test evaluation pipeline: variant configs, test set selection, generation, judging."""

import logging
import sqlite3
from typing import Any

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant configurations (A-E)
# ---------------------------------------------------------------------------

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "A": {
        "prompt_id": "baseline-fewshot",
        "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
        "temperature": 0.7,
        "num_ctx": 4096,
        "chunked": False,
    },
    "B": {
        "prompt_id": "zero-shot-causal",
        "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
    },
    "C": {
        "prompt_id": "zero-shot-chunked",
        "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": True,
    },
    "D": {
        "prompt_id": "zero-shot-causal",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
    },
    "E": {
        "prompt_id": "zero-shot-chunked",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": True,
    },
}


# ---------------------------------------------------------------------------
# Test set selection
# ---------------------------------------------------------------------------


def select_source_lessons(conn: sqlite3.Connection, per_cluster: int = 4) -> list[dict]:
    """Select source lessons for evaluation.

    Finds all clusters with >= 3 single-loop lessons, then picks up to
    ``per_cluster`` lessons per cluster maximising category diversity.

    Returns a flat list of lesson dicts with keys:
        id, title, one_liner, description, cluster_seed, category
    """
    # Find qualifying clusters (>= 3 single-loop lessons)
    cluster_rows = conn.execute(
        """
        SELECT cluster_seed, COUNT(*) AS cnt
        FROM lessons
        WHERE cluster_seed IS NOT NULL
          AND (loop_level IS NULL OR loop_level = 'single')
        GROUP BY cluster_seed
        HAVING cnt >= 3
        ORDER BY cluster_seed
        """
    ).fetchall()

    results: list[dict] = []

    for crow in cluster_rows:
        seed = crow["cluster_seed"]

        # Fetch all single-loop lessons in this cluster
        rows = conn.execute(
            """
            SELECT id, title, one_liner, description, cluster_seed, category
            FROM lessons
            WHERE cluster_seed = ?
              AND (loop_level IS NULL OR loop_level = 'single')
            ORDER BY id
            """,
            (seed,),
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
    cluster_seed: str,
    count_same: int = 2,
    count_diff: int = 2,
) -> dict[str, list[dict]]:
    """Select transfer target lessons for a given source lesson.

    Returns:
        {"same_cluster": [...], "diff_cluster": [...]}

    - same_cluster: other lessons from same cluster, excluding source,
      preferring different categories (sort: different category first).
    - diff_cluster: lessons from other clusters, selected randomly.
    - All single-loop only.
    """
    # Get source lesson's category for preference sorting
    source_row = conn.execute("SELECT category FROM lessons WHERE id = ?", (source_id,)).fetchone()
    source_category = source_row["category"] if source_row else None

    # Same cluster, excluding source, single-loop only
    # Sort: different category first (0 before 1), then by id for stability
    same_rows = conn.execute(
        """
        SELECT id, title, one_liner, description, cluster_seed, category
        FROM lessons
        WHERE cluster_seed = ?
          AND id != ?
          AND (loop_level IS NULL OR loop_level = 'single')
        ORDER BY
            CASE WHEN category = ? THEN 1 ELSE 0 END,
            id
        """,
        (cluster_seed, source_id, source_category),
    ).fetchall()

    same_cluster = [dict(r) for r in same_rows[:count_same]]

    # Different cluster, single-loop, random selection
    diff_rows = conn.execute(
        """
        SELECT id, title, one_liner, description, cluster_seed, category
        FROM lessons
        WHERE cluster_seed IS NOT NULL
          AND cluster_seed != ?
          AND (loop_level IS NULL OR loop_level = 'single')
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (cluster_seed, count_diff),
    ).fetchall()

    diff_cluster = [dict(r) for r in diff_rows]

    return {
        "same_cluster": same_cluster,
        "diff_cluster": diff_cluster,
    }
