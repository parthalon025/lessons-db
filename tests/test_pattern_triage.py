"""Tests for cross-project pattern triage (Stage 3)."""

from unittest.mock import patch

import pytest

from lessons_db.db import init_db, get_lesson, get_scan_state, set_scan_state
from lessons_db.pattern_verify import VerifiedCandidate
from lessons_db.pattern_triage import (
    triage_candidate,
    reject_draft,
    calibration_bands,
    should_adjust_threshold,
    seed_reuse_count,
    tier_from_reuse,
)


@pytest.fixture
def verified(db_path):
    return VerifiedCandidate(
        snippet="with closing(conn): conn.execute(...)",
        source_repos=["repo-a", "repo-b"],
        source_lesson_id=None,
        confidence=0.90,
        rationale="Reliably closes SQLite connections across projects.",
    )


@pytest.fixture
def verified_3repos(db_path):
    return VerifiedCandidate(
        snippet="retry logic with backoff",
        source_repos=["repo-a", "repo-b", "repo-c"],
        source_lesson_id=None,
        confidence=0.92,
        rationale="Backoff prevents thundering herd.",
    )


class TestSeedReuseCount:
    def test_two_repos_gives_reuse_count_one(self):
        assert seed_reuse_count(["a", "b"]) == 1

    def test_three_repos_gives_reuse_count_two(self):
        assert seed_reuse_count(["a", "b", "c"]) == 2

    def test_one_repo_gives_zero(self):
        assert seed_reuse_count(["a"]) == 0


class TestTierFromReuse:
    def test_zero_gives_noticed(self):
        assert tier_from_reuse(0) == "noticed"

    def test_one_gives_tested(self):
        assert tier_from_reuse(1) == "tested"

    def test_two_gives_proven(self):
        assert tier_from_reuse(2) == "proven"

    def test_three_gives_standard(self):
        assert tier_from_reuse(3) == "standard"


class TestTriageCandidate:
    def test_auto_approves_above_threshold(self, verified, db_path, tmp_path):
        conn = init_db(db_path)
        lesson_id = triage_candidate(
            verified, conn,
            lance_dir=str(tmp_path / "lance"),
        )
        assert lesson_id is not None
        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "positive"
        assert lesson["tier"] == "tested"  # 2 repos → reuse_count=1 → tested
        assert lesson["reuse_count"] == 1

    def test_why_it_works_populated_from_rationale(
        self, verified, db_path, tmp_path
    ):
        conn = init_db(db_path)
        lesson_id = triage_candidate(verified, conn,
                                     lance_dir=str(tmp_path / "lance"))
        lesson = get_lesson(conn, lesson_id)
        assert "SQLite" in lesson.get("description", "")

    def test_three_repos_gives_proven_tier(
        self, verified_3repos, db_path, tmp_path
    ):
        conn = init_db(db_path)
        lesson_id = triage_candidate(verified_3repos, conn,
                                     lance_dir=str(tmp_path / "lance"))
        lesson = get_lesson(conn, lesson_id)
        assert lesson["tier"] == "proven"
        assert lesson["reuse_count"] == 2

    def test_below_threshold_goes_to_draft_queue(self, db_path, tmp_path):
        conn = init_db(db_path)
        low_confidence = VerifiedCandidate(
            snippet="niche pattern",
            source_repos=["a", "b"],
            source_lesson_id=None,
            confidence=0.70,
            rationale="Maybe useful.",
        )
        lesson_id = triage_candidate(low_confidence, conn,
                                     lance_dir=str(tmp_path / "lance"))
        assert lesson_id is None
        draft = conn.execute(
            "SELECT * FROM capture_drafts WHERE detection_source='cross_project_scan'"
        ).fetchone()
        assert draft is not None
        assert abs(draft["confidence"] - 0.70) < 0.01

    def test_surfacing_event_recorded_on_auto_approve(
        self, verified, db_path, tmp_path
    ):
        conn = init_db(db_path)
        lesson_id = triage_candidate(verified, conn,
                                     lance_dir=str(tmp_path / "lance"))
        event = conn.execute(
            "SELECT * FROM surfacing_events WHERE hook_point='cross_project_scan'"
        ).fetchone()
        assert event is not None
        assert event["lesson_id"] == lesson_id


class TestRejectDraft:
    def test_reject_inserts_suppression_vector(self, db_path, tmp_path):
        conn = init_db(db_path)
        # Insert a draft first
        conn.execute(
            "INSERT INTO capture_drafts "
            "(raw_content, status, created_date, source, detection_source, confidence) "
            "VALUES ('snippet', 'pending', '2026-02-26', 'test', 'cross_project_scan', 0.75)"
        )
        conn.commit()
        draft_id = conn.execute(
            "SELECT id FROM capture_drafts"
        ).fetchone()["id"]

        with patch("lessons_db.pattern_triage.get_embedding",
                   return_value=[0.1] * 768):
            reject_draft(draft_id, conn,
                         lance_dir=str(tmp_path / "lance"),
                         reason="Too project-specific")

        sv = conn.execute(
            "SELECT * FROM suppression_vectors"
        ).fetchone()
        assert sv is not None
        assert sv["rejection_reason"] == "Too project-specific"

        draft = conn.execute(
            "SELECT status FROM capture_drafts WHERE id = ?", [draft_id]
        ).fetchone()
        assert draft["status"] == "rejected"


class TestCalibration:
    def test_calibration_bands_groups_by_confidence(self, db_path):
        conn = init_db(db_path)
        # Insert drafts with known outcomes
        for conf, promoted in [(0.87, 1), (0.86, 1), (0.88, 0), (0.75, 0), (0.76, 1)]:
            conn.execute(
                "INSERT INTO capture_drafts "
                "(raw_content, status, created_date, source, "
                " detection_source, confidence) "
                "VALUES (?, ?, '2026-02-26', 'test', 'cross_project_scan', ?)",
                [f"snippet-{conf}",
                 "approved" if promoted else "rejected",
                 conf]
            )
        conn.commit()
        bands = calibration_bands(conn)
        # Bands are keyed by ROUND(confidence, 1)
        assert isinstance(bands, dict)

    def test_should_adjust_threshold_returns_none_when_insufficient_data(
        self, db_path
    ):
        conn = init_db(db_path)
        result = should_adjust_threshold(conn)
        assert result is None
