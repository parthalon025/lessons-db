"""Multi-strategy search: file path, content regex, semantic, combined."""

import logging
import re

from lessons_db.db import search_by_file as _db_search_by_file
from lessons_db.vectors import semantic_search

logger = logging.getLogger(__name__)


def search_for_file(conn, file_path: str) -> list[dict]:
    """Search lessons by affected file path. Delegates to db.search_by_file."""
    return _db_search_by_file(conn, file_path)


def search_by_content(conn, content: str, language: str = "any") -> list[dict]:
    """Match content against syntactic detection patterns.

    Loads all detection_patterns rows where pattern_type='syntactic'
    and language matches. Tests each regex against content.
    Returns list of dicts with: lesson_id, one_liner, matched_pattern, severity.
    """
    if language == "any":
        rows = conn.execute(
            """
            SELECT dp.lesson_id, dp.regex, l.one_liner, l.severity
            FROM detection_patterns dp
            JOIN lessons l ON dp.lesson_id = l.id
            WHERE dp.pattern_type = 'syntactic'
            """,
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT dp.lesson_id, dp.regex, l.one_liner, l.severity
            FROM detection_patterns dp
            JOIN lessons l ON dp.lesson_id = l.id
            WHERE dp.pattern_type = ?
              AND (dp.language = ? OR dp.language = 'any')
            """,
            ("syntactic", language),
        ).fetchall()

    matches = []
    for row in rows:
        try:
            if re.search(row["regex"], content):
                matches.append({
                    "lesson_id": row["lesson_id"],
                    "one_liner": row["one_liner"],
                    "matched_pattern": row["regex"],
                    "severity": row["severity"],
                })
        except re.error:
            logger.warning(
                "Invalid regex for lesson %d: %s", row["lesson_id"], row["regex"]
            )
    return matches


def search_semantic(lance_db, query: str, top_k: int = 3) -> list[dict]:
    """Semantic similarity search. Returns empty list if lance_db is None."""
    if lance_db is None:
        return []
    return semantic_search(lance_db, query, top_k)


def search_combined(
    conn,
    lance_db,
    file_path: str | None = None,
    content: str | None = None,
    query: str | None = None,
    language: str = "any",
    polarity: str | None = None,
) -> list[dict]:
    """Run all applicable strategies, deduplicate by lesson ID, sort by severity DESC.

    First hit wins for deduplication. Results are unified to have at minimum:
    id, one_liner, severity.

    polarity: optional filter — 'positive', 'negative', or None (no filter).
    """
    seen_ids: set[int] = set()
    results: list[dict] = []

    # Strategy 1: file path search
    if file_path:
        for hit in search_for_file(conn, file_path):
            lid = hit["id"]
            if lid not in seen_ids:
                seen_ids.add(lid)
                results.append(hit)

    # Strategy 2: content regex search
    if content:
        for hit in search_by_content(conn, content, language):
            lid = hit["lesson_id"]
            if lid not in seen_ids:
                seen_ids.add(lid)
                # Normalize key to 'id' for consistency
                results.append({
                    "id": lid,
                    "one_liner": hit["one_liner"],
                    "matched_pattern": hit["matched_pattern"],
                    "severity": hit["severity"],
                })

    # Strategy 3: semantic search
    if query:
        for hit in search_semantic(lance_db, query):
            lid = hit["lesson_id"]
            if lid not in seen_ids:
                seen_ids.add(lid)
                results.append({
                    "id": lid,
                    "one_liner": hit.get("text", ""),
                    "severity": 0,  # Semantic hits don't carry severity
                    "score": hit.get("score"),
                    "cluster": hit.get("cluster"),
                })

    # Apply polarity filter by joining with lessons table
    if polarity is not None and results:
        ids = [r["id"] for r in results]
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, polarity FROM lessons WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        allowed = {r["id"] for r in rows if r["polarity"] == polarity}
        results = [r for r in results if r["id"] in allowed]

    # Sort by severity descending
    results.sort(key=lambda r: r.get("severity", 0), reverse=True)
    return results
