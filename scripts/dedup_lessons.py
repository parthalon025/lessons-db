#!/usr/bin/env python3
"""Deduplicate lessons in the DB using Jaccard similarity on one_liners.

Strategy:
  1. Compute all pairwise Jaccard similarities on one_liner tokens
  2. Build a graph of pairs with similarity >= threshold
  3. Find connected components (clusters of near-dupes)
  4. Within each cluster, keep the highest-quality lesson:
       - Prefer lesson with a detection_pattern
       - Then prefer source='migrated' (human-reviewed)
       - Then prefer lower id (older, more established)
  5. Delete the rest + their detection_patterns

Usage:
    python scripts/dedup_lessons.py              # dry-run
    python scripts/dedup_lessons.py --threshold 0.65   # looser matching
    python scripts/dedup_lessons.py --execute    # apply deletions
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lessons_db.config import SQLITE_PATH
from lessons_db.review import jaccard_similarity  # reuse the existing function

DEFAULT_THRESHOLD = 0.70


_PLACEHOLDER_RE = re.compile(r"^\[?TODO", re.IGNORECASE)


def load_lessons(conn: sqlite3.Connection) -> list[dict]:
    """Load lessons with non-empty, non-placeholder one_liners only."""
    rows = conn.execute(
        "SELECT id, one_liner, source FROM lessons WHERE one_liner IS NOT NULL AND one_liner != '' ORDER BY id"
    ).fetchall()
    # Filter out TODO/placeholder one_liners
    return [dict(r) for r in rows if not _PLACEHOLDER_RE.match(r["one_liner"])]


def lessons_with_patterns(conn: sqlite3.Connection) -> set[int]:
    """Set of lesson IDs that have at least one detection_pattern."""
    rows = conn.execute("SELECT DISTINCT lesson_id FROM detection_patterns").fetchall()
    return {r["lesson_id"] for r in rows}


def quality_rank(lesson: dict, has_pattern: bool) -> tuple[int, int, int]:
    """Return a sort key — lower is better (we want to keep the best).

    Priority: migrated (human-reviewed) > has detection_pattern > other source.
    Tiebreak: lower id (older, more established).
    """
    # source: 0 = migrated (best, human-reviewed), 1 = other
    source_score = 0 if lesson["source"] == "migrated" else 1
    # has_pattern: 0 = yes (better), 1 = no — secondary to source
    pattern_score = 0 if has_pattern else 1
    # id: lower = older/more established (better)
    return (source_score, pattern_score, lesson["id"])


def find_components(pairs: list[tuple[int, int]], all_ids: list[int]) -> list[set[int]]:
    """Union-find connected components from (a, b) similarity pairs."""
    parent = {i: i for i in all_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for a, b in pairs:
        union(a, b)

    components: dict[int, set[int]] = {}
    for i in all_ids:
        root = find(i)
        components.setdefault(root, set()).add(i)

    # Only return clusters with > 1 member
    return [c for c in components.values() if len(c) > 1]


def _rank_cluster(
    cluster: set[int],
    lesson_by_id: dict[int, dict],
    pattern_ids: set[int],
) -> tuple[int, list[int]]:
    """Return (keep_id, delete_ids) for a cluster, ranked by quality."""
    members = sorted(
        cluster,
        key=lambda lid: quality_rank(lesson_by_id[lid], lid in pattern_ids),
    )
    return members[0], members[1:]


def _apply_deletions(
    conn: sqlite3.Connection,
    clusters: list[set[int]],
    lesson_by_id: dict[int, dict],
    pattern_ids: set[int],
    to_delete: list[int],
) -> int:
    """Transplant detection_patterns then delete duplicates. Returns transplant count."""
    delete_to_keep: dict[int, int] = {}
    for cluster in clusters:
        keep_id, delete_ids = _rank_cluster(cluster, lesson_by_id, pattern_ids)
        for did in delete_ids:
            delete_to_keep[did] = keep_id

    transplanted = 0
    for del_id, keep_id in delete_to_keep.items():
        if del_id in pattern_ids:
            conn.execute(
                "UPDATE detection_patterns SET lesson_id = ? WHERE lesson_id = ?",
                [keep_id, del_id],
            )
            transplanted += 1

    for lid in to_delete:
        conn.execute("DELETE FROM detection_patterns WHERE lesson_id = ?", [lid])
        conn.execute("DELETE FROM lessons WHERE id = ?", [lid])
    conn.commit()
    return transplanted


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate lessons by Jaccard similarity")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Jaccard threshold for near-duplicate detection (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--execute", action="store_true", help="Apply deletions (default: dry-run)")
    args = parser.parse_args()

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    lessons = load_lessons(conn)
    pattern_ids = lessons_with_patterns(conn)
    n = len(lessons)

    print(f"Lessons to compare: {n}")
    print(f"Threshold: {args.threshold}")
    print("Computing pairwise similarities...")

    similar_pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = lessons[i], lessons[j]
            sim = jaccard_similarity(a["one_liner"] or "", b["one_liner"] or "")
            if sim >= args.threshold:
                similar_pairs.append((a["id"], b["id"]))

    print(f"Similar pairs found: {len(similar_pairs)}")
    if not similar_pairs:
        print("No near-duplicates found at this threshold.")
        return

    all_ids = [lsn["id"] for lsn in lessons]
    clusters = find_components(similar_pairs, all_ids)
    print(f"Duplicate clusters: {len(clusters)}")

    lesson_by_id = {lsn["id"]: lsn for lsn in lessons}
    to_delete: list[int] = []

    print("\n--- Duplicate clusters ---")
    for cluster in sorted(clusters, key=lambda c: min(c)):
        keep_id, delete_ids = _rank_cluster(cluster, lesson_by_id, pattern_ids)
        to_delete.extend(delete_ids)
        keep_lesson = lesson_by_id[keep_id]
        print(
            f"\n  KEEP  [{keep_id}] ({keep_lesson['source']}"
            f"{' +pattern' if keep_id in pattern_ids else ''}): "
            f"{keep_lesson['one_liner'][:70]}"
        )
        for did in delete_ids:
            dl = lesson_by_id[did]
            print(
                f"  DEL   [{did}] ({dl['source']}"
                f"{' +pattern' if did in pattern_ids else ''}): "
                f"{dl['one_liner'][:70]}"
            )

    print(f"\nSummary: {len(to_delete)} duplicates to delete across {len(clusters)} clusters")
    print(f"Lessons remaining after dedup: {n - len(to_delete)}")

    if not args.execute:
        print("\nRe-run with --execute to apply deletions.")
        return

    transplanted = _apply_deletions(conn, clusters, lesson_by_id, pattern_ids, to_delete)
    remaining = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    print(f"\nTransplanted detection patterns: {transplanted}")
    print(f"Deleted {len(to_delete)} lessons. Remaining: {remaining}")


if __name__ == "__main__":
    main()
