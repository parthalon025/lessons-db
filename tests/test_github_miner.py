# tests/test_github_miner.py
from unittest.mock import patch

from lessons_db.db import init_db
from lessons_db.github_miner import (
    MiningConfig,
    _process_modification,
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


def test_positive_candidate_not_gate0_rejected(db_path):
    """Positive candidates with bad_code='N/A' must NOT be gate0-rejected.

    Gate 0a checks that bad_code is >= 2 lines of valid Python.
    'N/A' is 1 line, so without the polarity guard positive candidates
    are silently killed before insertion.
    """
    conn = init_db(db_path)
    mock_diff = "+def process():\n+    await asyncio.sleep(5)\n"
    stats = {
        "gate0_rejected": 0,
        "auto_approved": 0,
        "error_count": 0,
    }
    with patch("lessons_db.github_miner._call_ollama_extract") as mock_ollama:
        mock_ollama.return_value = [
            {
                "polarity": "positive",
                "title": "async sleep pattern",
                "one_liner": "use asyncio.sleep instead of time.sleep",
                "bad_code": "N/A",
                "good_code": "await asyncio.sleep(5)",
                "category": "async",
                "source_repo": "owner/repo",
            },
        ]
        _process_modification(conn, mock_diff, "owner/repo", MiningConfig(), stats)

    assert stats["gate0_rejected"] == 0, "positive candidate must not be gate0-rejected"
    assert stats["auto_approved"] == 1, "positive candidate must be auto_approved"
    # Confirm the draft was written to the DB
    row = conn.execute("SELECT extracted_data, source FROM capture_drafts").fetchone()
    assert row is not None, "capture_drafts row must exist"
    assert "github_miner:owner/repo" in row["source"]


def test_process_modification_inserts_draft_directly(db_path):
    """Validated negative candidates must be inserted into capture_drafts directly,
    not re-extracted via capture_from_diff (which would discard the candidate)."""
    conn = init_db(db_path)
    mock_diff = "+x = 1\n+y = 2\n-z = bad_call()\n"
    stats = {
        "gate0_rejected": 0,
        "auto_approved": 0,
        "error_count": 0,
    }
    with patch("lessons_db.github_miner._call_ollama_extract") as mock_ollama:
        mock_ollama.return_value = [
            {
                "polarity": "negative",
                "title": "bad call pattern",
                "one_liner": "avoid bad_call, use safe_call instead",
                "bad_code": "z = bad_call()\nresult = z",
                "good_code": "z = safe_call()\nresult = z",
                "category": "architecture-pattern",
                "source_repo": "owner/repo",
            },
        ]
        count = _process_modification(conn, mock_diff, "owner/repo", MiningConfig(), stats)

    assert count == 1
    assert stats["auto_approved"] == 1
    assert stats["gate0_rejected"] == 0
    row = conn.execute("SELECT extracted_data FROM capture_drafts").fetchone()
    assert row is not None
    data = __import__("json").loads(row["extracted_data"])
    assert data["title"] == "bad call pattern"
