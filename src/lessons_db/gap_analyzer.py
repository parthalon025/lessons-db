"""Gap analyzer — identify which lesson categories have thin coverage.

gap_score = category_severity * (1 / max(1, lesson_count)) * recency_factor

category_severity: security=5, performance=4, db-queries=4, async=4,
                   integration=3, testing=3, data-model=3, deployment=3,
                   ui=2, monitoring=2, registration=2, cold-start=2
recency_factor: 1.5 if category touched in last 7 days, else 1.0
"""

import logging
import sqlite3
from datetime import date, timedelta

_log = logging.getLogger(__name__)

CATEGORY_SEVERITY: dict[str, int] = {
    # Negative categories
    "security": 5,
    "performance": 4,
    "db-queries": 4,
    "async": 4,
    "integration": 3,
    "testing": 3,
    "data-model": 3,
    "deployment": 3,
    "ui": 2,
    "monitoring": 2,
    "registration": 2,
    "cold-start": 2,
    # Positive categories
    "architecture-pattern": 3,
    "workflow-optimization": 3,
    "debugging-strategy": 3,
    "testing-pattern": 3,
    "integration-approach": 3,
    "value-multiplier": 2,
    "planning-technique": 2,
    "tooling-innovation": 2,
}

RECENCY_DAYS = 7
RECENCY_BOOST = 1.5


def compute_gap_scores(conn: sqlite3.Connection) -> list[dict]:
    """Return ranked gap list with gap_score per category.

    Higher score = thinner coverage = more mining priority.
    """
    cutoff = (date.today() - timedelta(days=RECENCY_DAYS)).isoformat()

    # Count lessons per category
    rows = conn.execute("SELECT category, COUNT(*) as cnt FROM lessons GROUP BY category").fetchall()
    lesson_counts: dict[str, int] = {r["category"]: r["cnt"] for r in rows}

    # Recent categories (touched in last 7 days)
    recent_rows = conn.execute("SELECT DISTINCT category FROM lessons WHERE created_date >= ?", (cutoff,)).fetchall()
    recent_cats: set[str] = {r["category"] for r in recent_rows}

    scores = []
    for category, severity in CATEGORY_SEVERITY.items():
        count = lesson_counts.get(category, 0)
        recency = RECENCY_BOOST if category in recent_cats else 1.0
        gap_score = severity * (1.0 / max(1, count)) * recency
        scores.append(
            {
                "category": category,
                "severity": severity,
                "lesson_count": count,
                "recency_boosted": category in recent_cats,
                "gap_score": round(gap_score, 4),
            }
        )

    return sorted(scores, key=lambda x: x["gap_score"], reverse=True)


def get_gap_report(conn: sqlite3.Connection) -> list[dict]:
    """Return gap report as list of dicts sorted by gap_score descending."""
    return compute_gap_scores(conn)
