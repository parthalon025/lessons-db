"""Tests for semgrep_import — Semgrep registry delta import."""

from lessons_db.db import init_db
from lessons_db.semgrep_import import (
    import_rule_as_lesson_stub,
    parse_semgrep_rule,
    semgrep_severity_to_int,
)

SAMPLE_RULE = {
    "id": "python.security.audit.hardcoded-password",
    "message": "Hardcoded password detected",
    "severity": "ERROR",
    "languages": ["python"],
    "pattern": 'password = "..."',
}


def test_semgrep_severity_to_int():
    assert semgrep_severity_to_int("ERROR") == 5
    assert semgrep_severity_to_int("WARNING") == 3
    assert semgrep_severity_to_int("INFO") == 1
    assert semgrep_severity_to_int("UNKNOWN") == 2  # default


def test_parse_semgrep_rule_extracts_fields():
    result = parse_semgrep_rule(SAMPLE_RULE)
    assert result["title"] == "Hardcoded password detected"
    assert result["category"] == "security"
    assert result["severity"] == 5
    assert result["rule_id"] == "python.security.audit.hardcoded-password"


def test_import_rule_as_lesson_stub(db_path):
    conn = init_db(db_path)
    lesson_id = import_rule_as_lesson_stub(conn, SAMPLE_RULE)
    assert lesson_id > 0
    row = conn.execute("SELECT * FROM lessons WHERE id=?", (lesson_id,)).fetchone()
    assert row is not None
    assert row["source"] == "semgrep_registry"
    assert row["tier"] == "observation"


def test_import_idempotent(db_path):
    conn = init_db(db_path)
    id1 = import_rule_as_lesson_stub(conn, SAMPLE_RULE)
    id2 = import_rule_as_lesson_stub(conn, SAMPLE_RULE)
    assert id1 == id2
    count = conn.execute("SELECT COUNT(*) FROM lessons WHERE source='semgrep_registry'").fetchone()[0]
    assert count == 1


def test_enforcement_rule_recorded(db_path):
    conn = init_db(db_path)
    lesson_id = import_rule_as_lesson_stub(conn, SAMPLE_RULE)
    rule_row = conn.execute("SELECT * FROM enforcement_rules WHERE lesson_id=?", (lesson_id,)).fetchone()
    assert rule_row is not None
    assert rule_row["rule_id"] == "python.security.audit.hardcoded-password"
    assert rule_row["rule_type"] == "semgrep"
