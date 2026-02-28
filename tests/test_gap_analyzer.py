"""Tests for gap_analyzer weighted gap scoring."""

import datetime

from lessons_db.db import init_db, insert_lesson
from lessons_db.gap_analyzer import compute_gap_scores, get_gap_report


def test_gap_scores_empty_db(db_path):
    conn = init_db(db_path)
    scores = compute_gap_scores(conn)
    assert isinstance(scores, list)
    assert len(scores) > 0
    # All required keys present
    first = scores[0]
    assert "category" in first
    assert "gap_score" in first
    assert "lesson_count" in first
    assert "severity" in first


def test_gap_scores_security_present(db_path):
    conn = init_db(db_path)
    scores = compute_gap_scores(conn)
    names = [s["category"] for s in scores]
    assert "security" in names


def test_gap_scores_drops_when_lessons_added(db_path):
    conn = init_db(db_path)
    # Get baseline security gap
    scores_before = compute_gap_scores(conn)
    sec_before = next(s for s in scores_before if s["category"] == "security")

    # Add 5 security lessons
    for i in range(5):
        insert_lesson(
            conn,
            {
                "title": f"sec lesson {i}",
                "one_liner": "avoid hardcoded secrets",
                "category": "security",
                "severity": 5,
                "tier": "lesson",
                "source": "manual",
                "created_date": "2026-01-01",
            },
        )

    scores_after = compute_gap_scores(conn)
    sec_after = next(s for s in scores_after if s["category"] == "security")
    assert sec_after["gap_score"] < sec_before["gap_score"]


def test_gap_scores_sorted_descending(db_path):
    conn = init_db(db_path)
    scores = compute_gap_scores(conn)
    gap_scores = [s["gap_score"] for s in scores]
    assert gap_scores == sorted(gap_scores, reverse=True)


def test_gap_report_returns_list(db_path):
    conn = init_db(db_path)
    report = get_gap_report(conn)
    assert isinstance(report, list)
    assert all("category" in r and "gap_score" in r for r in report)


def test_recency_boost_applied(db_path):
    conn = init_db(db_path)
    today = datetime.date.today().isoformat()
    # Add one recent performance lesson
    insert_lesson(
        conn,
        {
            "title": "perf lesson",
            "one_liner": "avoid n+1",
            "category": "performance",
            "severity": 4,
            "tier": "lesson",
            "source": "manual",
            "created_date": today,
        },
    )
    scores = compute_gap_scores(conn)
    perf = next(s for s in scores if s["category"] == "performance")
    assert perf["recency_boosted"] is True
