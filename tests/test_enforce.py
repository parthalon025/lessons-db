"""Tests for enforcement escalation ladder."""

from datetime import date

from lessons_db.db import get_lesson, init_db, insert_lesson
from lessons_db.enforce import check_escalation


def _insert_lesson(conn, recurrence_count=0, enforcement="documentation"):
    """Helper: insert a lesson with specified recurrence state."""
    return insert_lesson(conn, {
        "title": "Test lesson",
        "one_liner": "A test lesson for escalation",
        "recurrence_count": recurrence_count,
        "enforcement": enforcement,
    })


class TestEnforcementEscalation:
    """4-tier enforcement escalation ladder."""

    def test_first_occurrence_stays_documentation(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, recurrence_count=0)

        action = check_escalation(conn, lid)

        assert action == {
            "lesson_id": lid,
            "level": "documentation",
            "recurrence_count": 1,
            "generate_rule": False,
            "add_precommit": False,
            "add_autofix": False,
        }
        row = get_lesson(conn, lid)
        assert row["enforcement"] == "documentation"
        assert row["confidence"] == "emerging"
        assert row["recurrence_count"] == 1
        assert row["last_hit_date"] == date.today().isoformat()

    def test_second_occurrence_escalates_to_warning(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, recurrence_count=1, enforcement="documentation")

        action = check_escalation(conn, lid)

        assert action == {
            "lesson_id": lid,
            "level": "semgrep_warning",
            "recurrence_count": 2,
            "generate_rule": True,
            "add_precommit": False,
            "add_autofix": False,
        }
        row = get_lesson(conn, lid)
        assert row["enforcement"] == "semgrep_warning"
        assert row["confidence"] == "established"
        assert row["recurrence_count"] == 2

    def test_third_occurrence_escalates_to_error(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, recurrence_count=2, enforcement="semgrep_warning")

        action = check_escalation(conn, lid)

        assert action == {
            "lesson_id": lid,
            "level": "semgrep_error",
            "recurrence_count": 3,
            "generate_rule": True,
            "add_precommit": True,
            "add_autofix": False,
        }
        row = get_lesson(conn, lid)
        assert row["enforcement"] == "semgrep_error"
        assert row["confidence"] == "core"
        assert row["recurrence_count"] == 3

    def test_fourth_occurrence_escalates_to_autofix(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, recurrence_count=3, enforcement="semgrep_error")

        action = check_escalation(conn, lid)

        assert action == {
            "lesson_id": lid,
            "level": "semgrep_autofix",
            "recurrence_count": 4,
            "generate_rule": True,
            "add_precommit": False,
            "add_autofix": True,
        }
        row = get_lesson(conn, lid)
        assert row["enforcement"] == "semgrep_autofix"
        assert row["confidence"] == "core"
        assert row["recurrence_count"] == 4

    def test_further_occurrences_stay_at_autofix(self, db_path):
        conn = init_db(db_path)
        lid = _insert_lesson(conn, recurrence_count=5, enforcement="semgrep_autofix")

        action = check_escalation(conn, lid)

        assert action == {
            "lesson_id": lid,
            "level": "semgrep_autofix",
            "recurrence_count": 6,
            "generate_rule": True,
            "add_precommit": False,
            "add_autofix": True,
        }
        row = get_lesson(conn, lid)
        assert row["enforcement"] == "semgrep_autofix"
        assert row["confidence"] == "core"
        assert row["recurrence_count"] == 6
