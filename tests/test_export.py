"""Tests for markdown export from DB records."""

import pytest

from lessons_db.export import format_lesson_markdown, format_status_line


class TestLessonExport:
    def test_formats_basic_lesson(self):
        lesson = {
            "id": 7,
            "title": "Bare except swallows failures",
            "one_liner": "Never use bare except:pass — log before returning fallback",
            "cluster": "A",
            "tier": "lesson",
            "category": "data-model",
            "severity": 5,
            "keywords": "except,pass,silent",
            "created_date": "2026-01-15",
            "enforcement": "semgrep_warning",
            "recurrence_count": 2,
        }
        md = format_lesson_markdown(lesson)
        assert "# Lesson #7" in md
        assert "Bare except swallows failures" in md
        assert "**Tier:** lesson" in md
        assert "**Cluster:** A" in md


class TestStatusLine:
    def test_formats_status_line(self):
        line = format_status_line(
            total_lessons=116,
            overdue_actions=3,
            open_findings=2,
        )
        assert "116" in line
        assert "3" in line
        assert "2" in line

    def test_clean_status(self):
        line = format_status_line(
            total_lessons=50,
            overdue_actions=0,
            open_findings=0,
        )
        assert "50" in line
