"""Multi-strategy search: file path, content regex, semantic, combined."""

import logging
import re
import sqlite3
from typing import Any

import lancedb

from lessons_db.db import search_by_file as _db_search_by_file
from lessons_db.vectors import semantic_search

logger = logging.getLogger(__name__)


def search_for_file(conn: sqlite3.Connection, file_path: str) -> list[dict[str, Any]]:
    """Search lessons by affected file path. Delegates to db.search_by_file."""
    return _db_search_by_file(conn, file_path)


def search_by_content(conn: sqlite3.Connection, content: str, language: str = "any") -> list[dict[str, Any]]:
    """Match content against regex-based detection patterns.

    Loads all detection_patterns rows where pattern_type is 'regex' or
    'syntactic' (both are plain-regex patterns; 'structural' patterns are
    AST-based and excluded from text matching).  Filters by language when
    not 'any'.  Tests each regex against *content*.

    Returns list of dicts with: lesson_id, one_liner, matched_pattern, severity.
    """
    # Both 'regex' and 'syntactic' are plain-regex pattern types eligible for
    # text matching.  'structural' patterns require AST analysis and are excluded.
    _REGEX_PATTERN_TYPES = ("regex", "syntactic")

    if language == "any":
        placeholders = ",".join("?" for _ in _REGEX_PATTERN_TYPES)
        rows = conn.execute(
            f"""
            SELECT dp.lesson_id, dp.regex, l.one_liner, l.severity
            FROM detection_patterns dp
            JOIN lessons l ON dp.lesson_id = l.id
            WHERE dp.pattern_type IN ({placeholders})
            """,
            list(_REGEX_PATTERN_TYPES),
        ).fetchall()
    else:
        placeholders = ",".join("?" for _ in _REGEX_PATTERN_TYPES)
        rows = conn.execute(
            f"""
            SELECT dp.lesson_id, dp.regex, l.one_liner, l.severity
            FROM detection_patterns dp
            JOIN lessons l ON dp.lesson_id = l.id
            WHERE dp.pattern_type IN ({placeholders})
              AND (dp.language = ? OR dp.language = 'any')
            """,
            list(_REGEX_PATTERN_TYPES) + [language],
        ).fetchall()

    matches = []
    for row in rows:
        regex = row["regex"]
        lesson_id = row["lesson_id"]
        try:
            if re.search(regex, content):
                logger.debug(
                    "search_by_content: pattern matched — lesson_id=%d regex=%r",
                    lesson_id,
                    regex,
                )
                matches.append(
                    {
                        "lesson_id": lesson_id,
                        "one_liner": row["one_liner"],
                        "matched_pattern": regex,
                        "severity": row["severity"],
                    }
                )
        except re.error as exc:
            logger.warning(
                "search_by_content: invalid regex for lesson %d: %r — %s",
                lesson_id,
                regex,
                exc,
            )

    logger.debug(
        "search_by_content: tested %d patterns, %d matched",
        len(rows),
        len(matches),
    )
    return matches


def search_text_fallback(conn: sqlite3.Connection, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """SQLite LIKE fallback when semantic search is unavailable."""
    if not query:
        return []
    terms = query.strip().split()
    if not terms:
        return []
    # Build WHERE clause: all terms must appear in one_liner OR title OR description
    conditions = []
    params: list[Any] = []
    for term in terms:
        like = f"%{term}%"
        conditions.append(
            "(COALESCE(one_liner,'') || ' ' || COALESCE(title,'') || ' ' || COALESCE(description,'')) LIKE ?"
        )
        params.append(like)
    where = " AND ".join(conditions)
    rows = conn.execute(
        f"SELECT id, one_liner, severity, title FROM lessons WHERE {where} ORDER BY severity DESC LIMIT ?",
        params + [top_k],
    ).fetchall()
    return [{"id": r["id"], "one_liner": r["one_liner"] or r["title"], "severity": r["severity"] or 0} for r in rows]


def search_semantic(lance_db: lancedb.DBConnection | None, query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Semantic similarity search. Returns empty list if lance_db is None."""
    if lance_db is None:
        return []
    return semantic_search(lance_db, query, top_k)


def _merge_hits(
    hits: list[dict[str, Any]],
    id_key: str,
    seen_ids: set[Any],
    results: list[dict[str, Any]],
    transform: Any = None,
) -> None:
    """Deduplicate hits by lesson ID and append to results."""
    for hit in hits:
        lid = hit[id_key]
        if lid not in seen_ids:
            seen_ids.add(lid)
            results.append(transform(hit, lid) if transform else hit)


def _apply_composite_scores(conn: sqlite3.Connection, results: list[dict]) -> list[dict]:
    """Compute composite relevance score for each result and re-sort by it.

    Uses semantic similarity from result['score'] if present, else 0.0.
    Falls back to semantic_sim=0.0 if relevance_score() raises — logs the error
    so failures are visible (Cluster A: no silent swallowing).

    Adds 'composite_score' to each result dict.
    Returns results sorted by composite_score descending.
    """
    from lessons_db.learn import relevance_score

    for result in results:
        lesson_id = result.get("id")
        if lesson_id is None:
            result["composite_score"] = 0.0
            continue

        # Semantic similarity: use score from vector search if available, else 0.0
        semantic_sim: float = float(result.get("score") or 0.0)
        # Context: use matched_pattern or empty string as context signal
        context: str = result.get("matched_pattern") or ""

        try:
            score = relevance_score(conn, lesson_id, context, semantic_sim)
        except Exception as exc:
            logger.warning(
                "search_combined: relevance_score fallback for lesson %d — %s. Using semantic_sim=%.3f as score.",
                lesson_id,
                exc,
                semantic_sim,
            )
            score = semantic_sim

        result["composite_score"] = score

    results.sort(key=lambda r: r.get("composite_score", 0.0), reverse=True)
    return results


def search_combined(
    conn: sqlite3.Connection,
    lance_db: lancedb.DBConnection | None,
    file_path: str | None = None,
    content: str | None = None,
    query: str | None = None,
    language: str = "any",
    polarity: str | None = None,
) -> list[dict]:
    """Run all applicable strategies, deduplicate by lesson ID, rank by composite relevance.

    First hit wins for deduplication. Results are unified to have at minimum:
    id, one_liner, severity, composite_score.

    composite_score = 0.5 * semantic_sim + 0.3 * outcome_rate + 0.2 * recurrence_score.
    Falls back to semantic_sim for any lesson where relevance_score() raises.

    polarity: optional filter — 'positive', 'negative', or None (no filter).
    """
    seen_ids: set[int] = set()
    results: list[dict] = []

    # Strategy 1: file path search
    if file_path:
        _merge_hits(search_for_file(conn, file_path), "id", seen_ids, results)

    # Strategy 2: content regex search
    if content:
        _merge_hits(
            search_by_content(conn, content, language),
            "lesson_id",
            seen_ids,
            results,
            transform=lambda hit, lid: {
                "id": lid,
                "one_liner": hit["one_liner"],
                "matched_pattern": hit["matched_pattern"],
                "severity": hit["severity"],
            },
        )

    # Strategy 3: semantic search (with text fallback when LanceDB unavailable)
    if query:
        _merge_hits(
            search_semantic(lance_db, query),
            "lesson_id",
            seen_ids,
            results,
            transform=lambda hit, lid: {
                "id": lid,
                "one_liner": hit.get("text", ""),
                "severity": 0,
                "score": hit.get("score"),
                "cluster": hit.get("cluster"),
            },
        )
        if lance_db is None:
            _merge_hits(search_text_fallback(conn, query), "id", seen_ids, results)

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

    # Rank by composite relevance score (replaces severity-only sort)
    results = _apply_composite_scores(conn, results)
    return results
