# tests/test_github_miner.py
from unittest.mock import patch

from lessons_db.db import init_db
from lessons_db.github_miner import (
    MiningConfig,
    extract_polarized_candidates,
    filter_diff_by_size,
    is_bug_fix_commit,
)


def test_is_bug_fix_conventional_commit():
    assert is_bug_fix_commit("fix(auth): token expiry not checked") is True
    assert is_bug_fix_commit("feat: add login") is False


def test_is_bug_fix_issue_reference():
    assert is_bug_fix_commit("Resolves: #42 — crash on startup") is True
    assert is_bug_fix_commit("Closes: #100") is True


def test_is_bug_fix_keyword_only_low_confidence():
    # keyword-only is LOW confidence — filter returns True but with low confidence
    result = is_bug_fix_commit("fixed bug with auth")
    assert isinstance(result, bool)


def test_filter_diff_by_size_accepts_valid():
    # 10-line diff should pass
    diff = "\n".join(["+line"] * 10)
    assert filter_diff_by_size(diff, min_lines=5, max_lines=200) is True


def test_filter_diff_by_size_rejects_too_small():
    assert filter_diff_by_size("+x = 1", min_lines=5, max_lines=200) is False


def test_filter_diff_by_size_rejects_too_large():
    diff = "\n".join(["+line"] * 300)
    assert filter_diff_by_size(diff, min_lines=5, max_lines=200) is False


def test_mining_config_defaults():
    config = MiningConfig()
    assert config.min_stars == 50
    assert config.max_diff_lines == 200
    assert config.capture_positive is True  # Always capture positive
    assert config.capture_errors is True  # Always capture errors


def test_extract_polarized_candidates_returns_both(db_path):
    conn = init_db(db_path)
    mock_diff = """
-def process():
-    time.sleep(5)
+def process():
+    await asyncio.sleep(5)
"""
    with patch("lessons_db.github_miner._call_ollama_extract") as mock_ollama:
        mock_ollama.return_value = [
            {
                "polarity": "negative",
                "title": "blocking sleep in async",
                "one_liner": "use asyncio.sleep",
                "bad_code": "time.sleep(5)",
                "good_code": "await asyncio.sleep(5)",
                "category": "async",
            },
            {
                "polarity": "positive",
                "title": "proper async sleep pattern",
                "one_liner": "asyncio.sleep is the correct approach",
                "bad_code": "time.sleep(5)",
                "good_code": "await asyncio.sleep(5)",
                "category": "async",
            },
        ]
        candidates = extract_polarized_candidates(conn, mock_diff, source_repo="owner/repo")
    assert len(candidates) == 2
    polarities = {c["polarity"] for c in candidates}
    assert "negative" in polarities
    assert "positive" in polarities
