"""Tests for positive entry promotion ladder."""

import pytest

from lessons_db.promote import record_reuse, list_templates, apply_template
from lessons_db.db import init_db, insert_lesson, get_lesson


@pytest.fixture
def conn_with_positive(db_path):
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Dual-axis pipeline testing",
        "one_liner": "Dual-axis testing catches integration bugs missed by unit tests",
        "polarity": "positive",
        "entry_type": "pattern",
        "tier": "noticed",
        "category": "testing-pattern",
        "created_date": "2026-02-26",
    })
    return conn, lid


class TestRecordReuse:
    def test_first_reuse_promotes_to_tested(self, conn_with_positive):
        conn, lid = conn_with_positive
        new_tier = record_reuse(conn, lid)
        assert new_tier == "tested"
        lesson = get_lesson(conn, lid)
        assert lesson["reuse_count"] == 1
        assert lesson["tier"] == "tested"

    def test_second_reuse_promotes_to_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)  # → tested
        new_tier = record_reuse(conn, lid)  # → proven
        assert new_tier == "proven"
        lesson = get_lesson(conn, lid)
        assert lesson["tier"] == "proven"

    def test_second_reuse_generates_template(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        templates = list_templates(conn)
        assert len(templates) == 1
        assert templates[0]["lesson_id"] == lid

    def test_third_reuse_promotes_to_standard(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        new_tier = record_reuse(conn, lid)
        assert new_tier == "standard"

    def test_increments_reuse_count(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        lesson = get_lesson(conn, lid)
        assert lesson["reuse_count"] == 3


class TestListTemplates:
    def test_empty_initially(self, db_path):
        conn = init_db(db_path)
        assert list_templates(conn) == []

    def test_returns_template_after_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        templates = list_templates(conn)
        assert len(templates) == 1
        assert "one_liner" in templates[0]
        assert templates[0]["tier"] == "proven"


class TestApplyTemplate:
    def test_returns_none_before_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        assert apply_template(conn, lid) is None

    def test_returns_content_after_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        content = apply_template(conn, lid)
        assert content is not None
        assert "Dual-axis" in content
