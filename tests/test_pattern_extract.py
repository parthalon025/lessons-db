"""Tests for cross-project pattern extraction (Stage 1)."""

import json
from unittest.mock import MagicMock, patch

from lessons_db.db import init_db
from lessons_db.pattern_extract import (
    _SKIP_DIRS,
    _SKIP_FILENAMES,
    BOOTSTRAP_PATTERNS,
    CandidatePattern,
    _sliding_window,
    build_semgrep_patterns,
    extract_nonpython_candidates,
    extract_python_candidates,
    list_active_repos,
)


class TestListActiveRepos:
    def test_returns_repos_with_recent_commits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.pattern_extract.PROJECTS_DIR", tmp_path)
        repo = tmp_path / "my-repo"
        repo.mkdir()
        (repo / ".git").mkdir()

        with patch("lessons_db.pattern_extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="abc123\n")
            repos = list_active_repos("1970-01-01T00:00:00")

        assert repo in repos

    def test_skips_repos_with_no_recent_commits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.pattern_extract.PROJECTS_DIR", tmp_path)
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
                [f"Test {i}", f"one-liner {i}", "Wrap sqlite3.connect with contextlib.closing"],
            )
        conn.commit()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "pattern: with closing($CONN): ..."}
        with patch("lessons_db.pattern_extract.requests.post", return_value=mock_resp):
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
                [f"Lesson {i}", f"one-liner {i}", f"corrective action {i}"],
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

        semgrep_output = {"results": [{"path": str(repo_a / "mod.py"), "extra": {"lines": "with closing(conn):"}}]}
        with patch("lessons_db.pattern_extract.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=json.dumps(semgrep_output))
            candidates = extract_python_candidates(repos=[repo_a], patterns=BOOTSTRAP_PATTERNS, conn=conn)

        assert candidates == []

    def test_yields_candidate_when_two_repos_match(self, tmp_path, db_path):
        conn = init_db(db_path)
        repo_a = tmp_path / "repo-a"
        repo_b = tmp_path / "repo-b"
        repo_a.mkdir()
        repo_b.mkdir()

        # Each call to subprocess.run returns results for that specific repo only.
        # This verifies the grouping logic keys by the repo argument, not semgrep paths.
        def semgrep_side_effect(cmd, **kwargs):
            target = cmd[-1]  # last arg is the target directory
            if "repo-a" in target:
                data = {"results": [{"path": str(repo_a / "a.py"), "extra": {"lines": "with closing(conn):"}}]}
            elif "repo-b" in target:
                data = {"results": [{"path": str(repo_b / "b.py"), "extra": {"lines": "with closing(conn):"}}]}
            else:
                data = {"results": []}
            return MagicMock(returncode=0, stdout=json.dumps(data))

        with patch("lessons_db.pattern_extract.subprocess.run", side_effect=semgrep_side_effect):
            candidates = extract_python_candidates(repos=[repo_a, repo_b], patterns=BOOTSTRAP_PATTERNS, conn=conn)

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
        with patch("lessons_db.pattern_extract.get_embedding", return_value=same_vector):
            candidates = extract_nonpython_candidates(repos=[repo_a, repo_b], conn=conn)

        assert len(candidates) >= 1
        assert len(candidates[0].source_repos) >= 2


class TestSkipFilenamesAndDirs:
    """Config/lock/linter noise files must be skipped before embedding."""

    def test_skip_filenames_set_is_nonempty(self):
        assert len(_SKIP_FILENAMES) > 0
        assert "package-lock.json" in _SKIP_FILENAMES
        assert "tsconfig.json" in _SKIP_FILENAMES

    def test_skip_dirs_set_is_nonempty(self):
        assert len(_SKIP_DIRS) > 0
        assert "node_modules" in _SKIP_DIRS
        assert ".venv" in _SKIP_DIRS

    def test_package_lock_json_is_not_embedded(self, tmp_path, db_path):
        """package-lock.json must be silently skipped — no embed call emitted."""
        conn = init_db(db_path)
        repo_a = tmp_path / "repo-a"
        repo_a.mkdir()
        # Write a package-lock.json with enough lines to produce windows
        block = "\n".join([f'"dep-{i}": {{"version": "1.0.{i}"}}' for i in range(20)])
        (repo_a / "package-lock.json").write_text(block)

        call_log: list[str] = []

        def tracking_embed(text: str):
            call_log.append(text[:40])
            return [0.1] * 768

        with patch("lessons_db.pattern_extract.get_embedding", side_effect=tracking_embed):
            extract_nonpython_candidates(repos=[repo_a], conn=conn)

        # No embed calls should have been triggered by package-lock.json
        assert len(call_log) == 0

    def test_node_modules_dir_is_not_embedded(self, tmp_path, db_path):
        """Files inside node_modules must be skipped — no embed call emitted."""
        conn = init_db(db_path)
        repo_a = tmp_path / "repo-a"
        nm_dir = repo_a / "node_modules" / "some-pkg"
        nm_dir.mkdir(parents=True)
        block = "\n".join([f"export function fn{i}() {{}}" for i in range(20)])
        (nm_dir / "index.js").write_text(block)

        call_log: list[str] = []

        def tracking_embed(text: str):
            call_log.append(text[:40])
            return [0.1] * 768

        with patch("lessons_db.pattern_extract.get_embedding", side_effect=tracking_embed):
            extract_nonpython_candidates(repos=[repo_a], conn=conn)

        assert len(call_log) == 0

    def test_non_skip_file_is_still_embedded(self, tmp_path, db_path):
        """A normal .sh file outside skip dirs must still be processed."""
        conn = init_db(db_path)
        repo_a = tmp_path / "repo-a"
        repo_a.mkdir()
        block = "\n".join([f"echo step {i}" for i in range(20)])
        (repo_a / "deploy.sh").write_text(block)

        call_log: list[str] = []

        def tracking_embed(text: str):
            call_log.append(text[:40])
            return [0.1] * 768

        with patch("lessons_db.pattern_extract.get_embedding", side_effect=tracking_embed):
            extract_nonpython_candidates(repos=[repo_a], conn=conn)

        assert len(call_log) > 0


class TestBuildSemgrepPatternsCache:
    """File-based cache for build_semgrep_patterns avoids redundant Ollama calls."""

    def _seed_lessons(self, conn, n=10):
        for i in range(n):
            conn.execute(
                "INSERT INTO lessons "
                "(title, one_liner, corrective_action, tier, created_date, polarity) "
                "VALUES (?, ?, ?, 'lesson_learned', '2026-03-04', 'negative')",
                [f"Test {i}", f"one-liner {i}", f"wrap calls in contextlib.closing #{i}"],
            )
        conn.commit()

    def test_cache_hit_skips_ollama(self, db_path, tmp_path, monkeypatch):
        """Second call with same lessons must not invoke Ollama at all."""
        cache_file = tmp_path / "semgrep-patterns-cache.json"
        monkeypatch.setattr("lessons_db.pattern_extract._SEMGREP_PATTERNS_CACHE_PATH", cache_file)

        conn = init_db(db_path)
        self._seed_lessons(conn)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "pattern: with closing($C): ..."}

        call_count = 0

        def counting_post(*a, **kw):
            nonlocal call_count
            call_count += 1
            return mock_resp

        with patch("lessons_db.pattern_extract.requests.post", side_effect=counting_post):
            patterns_first = build_semgrep_patterns(conn)

        first_call_count = call_count

        # Second call — cache should be populated, Ollama must not be called
        call_count = 0
        with patch("lessons_db.pattern_extract.requests.post", side_effect=counting_post):
            patterns_second = build_semgrep_patterns(conn)

        assert call_count == 0, f"Expected 0 Ollama calls on cache hit, got {call_count}"
        assert patterns_first == patterns_second

    def test_cache_miss_on_changed_lessons(self, db_path, tmp_path, monkeypatch):
        """Adding a new lesson invalidates the cache key → Ollama called again."""
        cache_file = tmp_path / "semgrep-patterns-cache.json"
        monkeypatch.setattr("lessons_db.pattern_extract._SEMGREP_PATTERNS_CACHE_PATH", cache_file)

        conn = init_db(db_path)
        self._seed_lessons(conn, n=10)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "pattern: with closing($C): ..."}

        with patch("lessons_db.pattern_extract.requests.post", return_value=mock_resp):
            build_semgrep_patterns(conn)

        # Add another lesson — cache key changes
        conn.execute(
            "INSERT INTO lessons "
            "(title, one_liner, corrective_action, tier, created_date, polarity) "
            "VALUES (?, ?, ?, 'lesson_learned', '2026-03-04', 'negative')",
            ["New lesson", "new one-liner", "always validate input"],
        )
        conn.commit()

        second_call_count = 0

        def counting_post(*a, **kw):
            nonlocal second_call_count
            second_call_count += 1
            return mock_resp

        with patch("lessons_db.pattern_extract.requests.post", side_effect=counting_post):
            build_semgrep_patterns(conn)

        assert second_call_count > 0, "Expected Ollama calls after cache invalidation"

    def test_cache_file_written_after_generation(self, db_path, tmp_path, monkeypatch):
        """Cache JSON file must be created after a successful Ollama generation run."""
        cache_file = tmp_path / "semgrep-patterns-cache.json"
        monkeypatch.setattr("lessons_db.pattern_extract._SEMGREP_PATTERNS_CACHE_PATH", cache_file)

        conn = init_db(db_path)
        self._seed_lessons(conn)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "pattern: with closing($C): ..."}

        with patch("lessons_db.pattern_extract.requests.post", return_value=mock_resp):
            build_semgrep_patterns(conn)

        assert cache_file.exists(), "Cache file was not created after generation"
        cached = json.loads(cache_file.read_text())
        assert "cache_key" in cached
        assert "patterns" in cached
        assert len(cached["patterns"]) > 0
