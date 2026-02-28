# tests/test_conflict_detector.py
from unittest.mock import patch

from lessons_db.conflict_detector import ConflictResult, detect_conflicts
from lessons_db.db import init_db, insert_lesson


def test_no_conflict_on_empty_db(db_path, lance_dir):
    conn = init_db(db_path)
    with patch("lessons_db.conflict_detector.semantic_search", return_value=[]):
        result = detect_conflicts(conn, lance_dir, lesson_id=1, snippet="time.sleep(1)")
    assert result.has_conflict is False


def test_conflict_detected_opposite_polarity(db_path, lance_dir):
    conn = init_db(db_path)
    lesson_id = insert_lesson(
        conn,
        {
            "title": "Use asyncio.sleep",
            "one_liner": "good",
            "tier": "tested",
            "source": "manual",
            "created_date": "2026-02-28",
        },
    )
    mock_neighbors = [{"id": lesson_id, "score": 0.92, "_distance": 0.05}]
    with patch("lessons_db.conflict_detector.semantic_search", return_value=mock_neighbors):
        with patch("lessons_db.conflict_detector._get_polarity", side_effect=["negative", "positive"]):
            result = detect_conflicts(conn, lance_dir, lesson_id=99, snippet="time.sleep(1)")
    assert result.has_conflict is True
    assert result.conflicting_lesson_id == lesson_id


def test_conflict_result_fields():
    r = ConflictResult(has_conflict=False, conflicting_lesson_id=None, similarity=0.0, note="")
    assert r.has_conflict is False
