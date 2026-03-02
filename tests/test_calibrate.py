"""Tests for calibrate profile CLI command — strength profile from surfacing events."""

import pytest
from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db, insert_lesson
from lessons_db.learn import record_outcome, record_surfacing


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def db_with_lessons(db_path):
    """DB with lessons in different categories."""
    conn = init_db(db_path)

    # Category: async-discipline
    lid1 = insert_lesson(
        conn,
        {
            "title": "Async discipline lesson",
            "one_liner": "No async def without I/O",
            "category": "async-discipline",
            "created_date": "2026-01-01",
        },
    )
    # Category: error-handling
    lid2 = insert_lesson(
        conn,
        {
            "title": "Error handling lesson",
            "one_liner": "Log before swallowing exceptions",
            "category": "error-handling",
            "created_date": "2026-01-02",
        },
    )
    # Category: integration-boundaries
    lid3 = insert_lesson(
        conn,
        {
            "title": "Integration boundaries lesson",
            "one_liner": "Trace values end-to-end across layers",
            "category": "integration-boundaries",
            "created_date": "2026-01-03",
        },
    )
    # Category: testing (will have < 5 events)
    lid4 = insert_lesson(
        conn,
        {
            "title": "Testing lesson",
            "one_liner": "No hardcoded test counts",
            "category": "testing",
            "created_date": "2026-01-04",
        },
    )
    # Category: cold-start
    lid5 = insert_lesson(
        conn,
        {
            "title": "Cold start lesson",
            "one_liner": "Seed current state on restart",
            "category": "cold-start",
            "created_date": "2026-01-05",
        },
    )

    return (
        conn,
        db_path,
        {
            "async-discipline": lid1,
            "error-handling": lid2,
            "integration-boundaries": lid3,
            "testing": lid4,
            "cold-start": lid5,
        },
    )


def _seed_events(conn, lesson_id, heeded_count, dismissed_count):
    """Helper: create surfacing events with specified heeded/dismissed counts."""
    for _ in range(heeded_count):
        eid = record_surfacing(conn, lesson_id, "read", "test-context")
        record_outcome(conn, eid, "heeded")
    for _ in range(dismissed_count):
        eid = record_surfacing(conn, lesson_id, "read", "test-context")
        record_outcome(conn, eid, "dismissed")


class TestCalibrateHelp:
    def test_calibrate_help(self, runner):
        """calibrate --help works and mentions profile."""
        result = runner.invoke(main, ["calibrate", "--help"])
        assert result.exit_code == 0
        assert "profile" in result.output.lower()

    def test_calibrate_profile_help(self, runner):
        """calibrate profile --help shows strength profile description."""
        result = runner.invoke(main, ["calibrate", "profile", "--help"])
        assert result.exit_code == 0
        assert "strength" in result.output.lower()


class TestCalibrateEmptyDB:
    def test_empty_db_shows_more_data_needed(self, runner, db_path):
        """calibrate profile with no surfacing events shows a 'more data needed' message."""
        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile"])
        assert result.exit_code == 0
        assert "more data needed" in result.output.lower()


class TestCalibrateWithData:
    def test_identifies_strengths_and_growth_areas(self, runner, db_with_lessons):
        """calibrate profile correctly identifies top strengths and growth areas."""
        conn, db_path, lesson_ids = db_with_lessons

        # async-discipline: 9/10 heeded = 90%
        _seed_events(conn, lesson_ids["async-discipline"], heeded_count=9, dismissed_count=1)

        # error-handling: 8/10 heeded = 80%
        _seed_events(conn, lesson_ids["error-handling"], heeded_count=8, dismissed_count=2)

        # integration-boundaries: 3/10 heeded = 30%
        _seed_events(conn, lesson_ids["integration-boundaries"], heeded_count=3, dismissed_count=7)

        # cold-start: 5/10 heeded = 50%
        _seed_events(conn, lesson_ids["cold-start"], heeded_count=5, dismissed_count=5)

        # testing: only 3 events (below threshold)
        _seed_events(conn, lesson_ids["testing"], heeded_count=2, dismissed_count=1)

        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile"])
        assert result.exit_code == 0

        # Check structure
        assert "Strength Profile" in result.output
        assert "Strengths:" in result.output
        assert "Growth Areas:" in result.output

        # Strengths should list async-discipline first (90%)
        assert "async-discipline" in result.output
        assert "90% heeded" in result.output

        # Growth areas should list integration-boundaries (30%)
        assert "integration-boundaries" in result.output
        assert "30% heeded" in result.output

        # testing should appear in insufficient data
        assert "testing" in result.output
        assert "insufficient data" in result.output.lower()

    def test_minimum_event_threshold_excludes_low_count(self, runner, db_with_lessons):
        """Categories with fewer than min_events are excluded from strengths/growth."""
        conn, db_path, lesson_ids = db_with_lessons

        # Only give 4 events to async-discipline (below default threshold of 5)
        _seed_events(conn, lesson_ids["async-discipline"], heeded_count=4, dismissed_count=0)

        # Give 6 events to error-handling (above threshold)
        _seed_events(conn, lesson_ids["error-handling"], heeded_count=5, dismissed_count=1)

        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile"])
        assert result.exit_code == 0

        # error-handling should be in the profile
        assert "error-handling" in result.output

        # async-discipline should be in insufficient data
        lines = result.output.lower()
        assert "insufficient data" in lines
        assert "async-discipline" in lines

    def test_custom_min_events(self, runner, db_with_lessons):
        """--min-events flag adjusts the threshold."""
        conn, db_path, lesson_ids = db_with_lessons

        # Give 3 events to two categories
        _seed_events(conn, lesson_ids["async-discipline"], heeded_count=3, dismissed_count=0)
        _seed_events(conn, lesson_ids["error-handling"], heeded_count=1, dismissed_count=2)

        # Default threshold (5) should say more data needed
        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile"])
        assert result.exit_code == 0
        assert "more data needed" in result.output.lower()

        # With --min-events 3, both categories should qualify
        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile", "--min-events", "3"])
        assert result.exit_code == 0
        assert "Strength Profile" in result.output
        assert "async-discipline" in result.output
        assert "error-handling" in result.output

    def test_all_below_threshold_shows_message(self, runner, db_with_lessons):
        """When all categories have < min_events, shows appropriate message."""
        conn, db_path, lesson_ids = db_with_lessons

        # Give just 2 events to one category
        _seed_events(conn, lesson_ids["async-discipline"], heeded_count=1, dismissed_count=1)

        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile"])
        assert result.exit_code == 0
        assert "more data needed" in result.output.lower()
        assert "insufficient data" in result.output.lower()

    def test_strengths_ordered_by_rate_descending(self, runner, db_with_lessons):
        """Strengths are ordered by heeded rate, highest first."""
        conn, db_path, lesson_ids = db_with_lessons

        # Create clear ordering: async=90%, error=70%, cold-start=60%, integration=40%
        _seed_events(conn, lesson_ids["async-discipline"], heeded_count=9, dismissed_count=1)
        _seed_events(conn, lesson_ids["error-handling"], heeded_count=7, dismissed_count=3)
        _seed_events(conn, lesson_ids["cold-start"], heeded_count=6, dismissed_count=4)
        _seed_events(conn, lesson_ids["integration-boundaries"], heeded_count=4, dismissed_count=6)

        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile"])
        assert result.exit_code == 0

        output = result.output
        # Strengths section: async before error before cold-start
        strengths_start = output.index("Strengths:")
        growth_start = output.index("Growth Areas:")

        strengths_section = output[strengths_start:growth_start]
        assert strengths_section.index("async-discipline") < strengths_section.index("error-handling")
        assert strengths_section.index("error-handling") < strengths_section.index("cold-start")

        # Growth areas: integration first (lowest rate)
        growth_section = output[growth_start:]
        assert "integration-boundaries" in growth_section

    def test_uses_cluster_when_category_empty(self, runner, db_path):
        """Falls back to cluster field when category is empty."""
        conn = init_db(db_path)

        # Lesson with no category but has cluster
        lid = insert_lesson(
            conn,
            {
                "title": "Cluster-only lesson",
                "one_liner": "Test cluster fallback",
                "category": "",
                "cluster": "A",
                "created_date": "2026-01-01",
            },
        )

        _seed_events(conn, lid, heeded_count=8, dismissed_count=2)

        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile"])
        assert result.exit_code == 0
        # Should show cluster "A" since category is empty
        assert "A" in result.output

    def test_unknown_outcomes_excluded(self, runner, db_with_lessons):
        """Events with outcome='unknown' are not counted in the profile."""
        conn, db_path, lesson_ids = db_with_lessons

        # 5 heeded, 5 dismissed (10 total qualifying)
        _seed_events(conn, lesson_ids["async-discipline"], heeded_count=5, dismissed_count=5)

        # Add 10 unknown events (should be ignored)
        for _ in range(10):
            record_surfacing(conn, lesson_ids["async-discipline"], "read", "test")

        result = runner.invoke(main, ["--db", str(db_path), "calibrate", "profile"])
        assert result.exit_code == 0
        # Should show 50% (5/10), not counting the 10 unknowns
        assert "50% heeded (5/10)" in result.output
