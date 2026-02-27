"""Tests for cross-project pattern extraction (Stage 1)."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lessons_db.db import init_db
from lessons_db.pattern_extract import (
    CandidatePattern,
    list_active_repos,
    build_semgrep_patterns,
    BOOTSTRAP_PATTERNS,
    _sliding_window,
    extract_python_candidates,
    extract_nonpython_candidates,
)


class TestListActiveRepos:
    def test_returns_repos_with_recent_commits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "lessons_db.pattern_extract.PROJECTS_DIR", tmp_path
        )
        repo = tmp_path / "my-repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        with patch("lessons_db.pattern_extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n")
            repos = list_active_repos("1970-01-01T00:00:00")

        assert repo in repos

    def test_skips_repos_with_no_recent_commits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "lessons_db.pattern_extract.PROJECTS_DIR", tmp_path
        )
        repo = tmp_path / "stale-repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        with patch("lessons_db.pattern_extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="")
            repos = list_active_repos("2099-01-01T00:00:00")

        assert repo not in repos


class TestBuildSemgrepPatterns:
    def test_db_seeded_patterns_from_corrective_action(self, db_path):
        conn = init_db(db_path)
        # Insert 10 rows to meet the threshold
        for i in range(10):
            conn.execute(
                "INSERT INTO lessons "
                "(title, one_liner, corrective_action, tier, created_date) "
                "VALUES (?, ?, ?, 'lesson_learned', '2026-02-26')",
                [f"Test {i}", f"one-liner {i}", "Wrap sqlite3.connect with contextlib.closing"]
            )
        conn.commit()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": "pattern: with closing($CONN): ..."
        }
        with patch("lessons_db.pattern_extract.requests.post",
                   return_value=mock_resp):
            patterns = build_semgrep_patterns(conn)

        assert any(p["source_lesson_id"] is not None for p in patterns)

    def test_falls_back_to_bootstrap_when_no_corrective_actions(self, db_path):
        conn = init_db(db_path)
        patterns = build_semgrep_patterns(conn)
        assert len(patterns) == len(BOOTSTRAP_PATTERNS)
        assert all(p["source_lesson_id"] is None for p in patterns)

    def test_falls_back_to_bootstrap_when_fewer_than_10_corrective_actions(self, db_path):
        conn = init_db(db_path)
        # Insert 5 lessons with corrective_action — below the 10-row threshold
        for i in range(5):
            conn.execute(
                "INSERT INTO lessons "
                "(title, one_liner, corrective_action, tier, created_date) "
                "VALUES (?, ?, ?, 'lesson_learned', '2026-02-26')",
                [f"Lesson {i}", f"one-liner {i}", f"corrective action {i}"]
            )
        conn.commit()
        # No Ollama mock needed — should return BOOTSTRAP_PATTERNS without calling Ollama
        patterns = build_semgrep_patterns(conn)
        assert len(patterns) == len(BOOTSTRAP_PATTERNS)
        assert all(p["source_lesson_id"] is None for p in patterns)


class TestSlidingWindow:
    def test_yields_15_line_windows(self):
        lines = [f"line {i}" for i in range(20)]
        windows = list(_sliding_window(lines, size=15))
        assert len(windows) == 6  # 20 - 15 + 1
        assert len(windows[0]) == 15

    def test_skips_all_blank_windows(self):
        lines = [""] * 20
        windows = list(_sliding_window(lines, size=15))
        assert windows == []


class TestExtractPythonCandidates:
    def test_requires_two_repos_minimum(self, tmp_path, db_path):
        conn = init_db(db_path)
        repo_a = tmp_path / "repo-a"
        repo_a.mkdir()

        semgrep_output = {
            "results": [
                {"path": str(repo_a / "mod.py"), "extra": {"lines": "with closing(conn):"}}
            ]
        }
        with patch("lessons_db.pattern_extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(semgrep_output)
            )
            candidates = extract_python_candidates(
                repos=[repo_a], patterns=BOOTSTRAP_PATTERNS, conn=conn
            )

        assert candidates == []

    def test_yields_candidate_when_two_repos_match(self, tmp_path, db_path):
        conn = init_db(db_path)
        repo_a = tmp_path / "repo-a"
        repo_b = tmp_path / "repo-b"
        repo_a.mkdir()
        repo_b.mkdir()

        semgrep_output = {
            "results": [
                {"path": str(repo_a / "a.py"), "extra": {"lines": "with closing(conn):"}},
                {"path": str(repo_b / "b.py"), "extra": {"lines": "with closing(conn):"}},
            ]
        }
        with patch("lessons_db.pattern_extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=json.dumps(semgrep_output)
            )
            candidates = extract_python_candidates(
                repos=[repo_a, repo_b], patterns=BOOTSTRAP_PATTERNS, conn=conn
            )

        assert len(candidates) >= 1
        assert isinstance(candidates[0], CandidatePattern)
        assert len(candidates[0].source_repos) >= 2


class TestExtractNonpythonCandidates:
    def test_embeds_blocks_and_clusters_by_similarity(self, tmp_path, db_path):
        conn = init_db(db_path)
        repo_a = tmp_path / "repo-a"
        repo_b = tmp_path / "repo-b"
        (repo_a / "scripts").mkdir(parents=True)
        (repo_b / "scripts").mkdir(parents=True)

        block = "\n".join([f"echo line {i}" for i in range(15)])
        (repo_a / "scripts" / "deploy.sh").write_text(block)
        (repo_b / "scripts" / "deploy.sh").write_text(block)

        same_vector = [0.1] * 768
        with patch("lessons_db.pattern_extract.get_embedding",
                   return_value=same_vector):
            candidates = extract_nonpython_candidates(
                repos=[repo_a, repo_b], conn=conn
            )

        assert len(candidates) >= 1
        assert len(candidates[0].source_repos) >= 2
