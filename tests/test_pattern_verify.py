"""Tests for cross-project pattern verification (Stage 2)."""

from unittest.mock import MagicMock, patch

import pytest

from lessons_db.db import init_db
from lessons_db.pattern_extract import CandidatePattern
from lessons_db.pattern_verify import (
    VerifiedCandidate,
    is_suppressed,
    verify_candidate,
)


@pytest.fixture
def candidate():
    return CandidatePattern(
        snippet="with closing(conn):\n    conn.execute(...)",
        source_repos=["repo-a", "repo-b"],
        source_lesson_id=None,
    )


@pytest.fixture
def candidate_with_lesson():
    return CandidatePattern(
        snippet="with closing(conn):\n    conn.execute(...)",
        source_repos=["repo-a", "repo-b", "repo-c"],
        source_lesson_id=33,
    )


class TestIsSuppressed:
    def test_returns_false_when_no_suppression_vectors(self, db_path):
        conn = init_db(db_path)
        with patch("lessons_db.pattern_verify.get_embedding",
                   return_value=[0.1] * 768):
            result = is_suppressed("any snippet", conn,
                                   lance_dir=str(db_path.parent / "lance"))
        assert result is False

    def test_returns_true_when_similar_to_rejected(self, db_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO suppression_vectors "
            "(embedding_id, rejected_snippet, rejection_date) "
            "VALUES ('vec-1', 'with closing(conn): conn.execute(...)', '2026-02-26')"
        )
        conn.commit()
        with patch("lessons_db.pattern_verify.get_embedding",
                   return_value=[0.9] * 768), \
             patch("lessons_db.pattern_verify._suppression_similarity",
                   return_value=0.92):
            result = is_suppressed("with closing(conn):", conn,
                                   lance_dir=str(db_path.parent / "lance"))
        assert result is True


class TestVerifyCandidate:
    def test_returns_none_when_lancedb_dedup_matches(
        self, candidate, db_path, tmp_path
    ):
        conn = init_db(db_path)
        # score < 0.15 = very similar (LanceDB uses distance, lower = closer)
        with patch("lessons_db.pattern_verify.nearest_lessons",
                   return_value=[{"score": 0.05, "text": "existing"}]), \
             patch("lessons_db.pattern_verify.get_embedding",
                   return_value=[0.1] * 768):
            result = verify_candidate(candidate, conn,
                                      lance_dir=str(tmp_path / "lance"))
        assert result is None

    def test_returns_none_when_specificity_too_low(
        self, candidate, db_path, tmp_path
    ):
        conn = init_db(db_path)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "0.3"}
        with patch("lessons_db.pattern_verify.nearest_lessons",
                   return_value=[{"score": 0.9, "text": "far away"}]), \
             patch("lessons_db.pattern_verify.is_suppressed", return_value=False), \
             patch("lessons_db.pattern_verify.requests.post",
                   return_value=mock_resp):
            result = verify_candidate(candidate, conn,
                                      lance_dir=str(tmp_path / "lance"))
        assert result is None

    def test_returns_verified_candidate_with_confidence(
        self, candidate, db_path, tmp_path
    ):
        conn = init_db(db_path)
        # specificity=0.8, generality=0.9 → confidence = 0.8*0.4 + 0.9*0.6 = 0.86
        responses = iter([
            MagicMock(**{"json.return_value": {"response": "0.8"}}),
            MagicMock(**{"json.return_value": {
                "response": "0.9\nThis pattern solves resource cleanup."
            }}),
        ])
        with patch("lessons_db.pattern_verify.nearest_lessons",
                   return_value=[{"score": 0.9, "text": "far"}]), \
             patch("lessons_db.pattern_verify.is_suppressed", return_value=False), \
             patch("lessons_db.pattern_verify.requests.post",
                   side_effect=lambda *a, **kw: next(responses)):
            result = verify_candidate(candidate, conn,
                                      lance_dir=str(tmp_path / "lance"))
        assert isinstance(result, VerifiedCandidate)
        assert abs(result.confidence - 0.86) < 0.01
        assert result.rationale

    def test_anchors_to_source_lesson_when_set(
        self, candidate_with_lesson, db_path, tmp_path
    ):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO lessons (id, title, one_liner, tier, created_date) "
            "VALUES (33, 'sqlite', 'use closing()', 'lesson_learned', '2026-02-26')"
        )
        conn.commit()

        captured_prompts = []

        def capture_post(url, json=None, **kw):
            captured_prompts.append(json.get("prompt", ""))
            m = MagicMock()
            m.json.return_value = {"response": "0.8"}
            return m

        with patch("lessons_db.pattern_verify.nearest_lessons",
                   return_value=[{"score": 0.9, "text": "far"}]), \
             patch("lessons_db.pattern_verify.is_suppressed", return_value=False), \
             patch("lessons_db.pattern_verify.requests.post",
                   side_effect=capture_post):
            verify_candidate(candidate_with_lesson, conn,
                             lance_dir=str(tmp_path / "lance"))

        assert any("lesson #33" in p or "closing()" in p
                   for p in captured_prompts)

    def test_rationale_stored_in_verified_candidate(
        self, candidate, db_path, tmp_path
    ):
        conn = init_db(db_path)
        responses = iter([
            MagicMock(**{"json.return_value": {"response": "0.8"}}),
            MagicMock(**{"json.return_value": {
                "response": "0.9\nCleans up resources reliably across projects."
            }}),
        ])
        with patch("lessons_db.pattern_verify.nearest_lessons",
                   return_value=[{"score": 0.9, "text": "x"}]), \
             patch("lessons_db.pattern_verify.is_suppressed", return_value=False), \
             patch("lessons_db.pattern_verify.requests.post",
                   side_effect=lambda *a, **kw: next(responses)):
            result = verify_candidate(candidate, conn,
                                      lance_dir=str(tmp_path / "lance"))
        assert result.rationale

    def test_returns_none_when_suppressed(
        self, candidate, db_path, tmp_path
    ):
        conn = init_db(db_path)
        with patch("lessons_db.pattern_verify.nearest_lessons",
                   return_value=[{"score": 0.9, "text": "far"}]), \
             patch("lessons_db.pattern_verify.is_suppressed", return_value=True):
            result = verify_candidate(candidate, conn,
                                      lance_dir=str(tmp_path / "lance"))
        assert result is None

    def test_confidence_formula(self, candidate, db_path, tmp_path):
        conn = init_db(db_path)
        responses = iter([
            MagicMock(**{"json.return_value": {"response": "0.6"}}),
            MagicMock(**{"json.return_value": {"response": "1.0\nPerfect."}}),
        ])
        with patch("lessons_db.pattern_verify.nearest_lessons",
                   return_value=[{"score": 0.9, "text": "x"}]), \
             patch("lessons_db.pattern_verify.is_suppressed", return_value=False), \
             patch("lessons_db.pattern_verify.requests.post",
                   side_effect=lambda *a, **kw: next(responses)):
            result = verify_candidate(candidate, conn,
                                      lance_dir=str(tmp_path / "lance"))
        # 0.6*0.4 + 1.0*0.6 = 0.84
        assert abs(result.confidence - 0.84) < 0.01
