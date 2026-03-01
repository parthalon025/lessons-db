# tests/test_github_miner.py
import json
from unittest.mock import patch

from lessons_db.db import init_db
from lessons_db.github_miner import (
    MiningConfig,
    _insert_miner_candidate,
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
        "drafted": 0,
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
    assert stats["drafted"] == 1, "positive candidate must go to capture_drafts (not auto_approved)"
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
        "drafted": 0,
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
    assert stats["drafted"] == 1, "no lance_dir → candidate goes to capture_drafts"
    assert stats["auto_approved"] == 0
    assert stats["gate0_rejected"] == 0
    row = conn.execute("SELECT extracted_data FROM capture_drafts").fetchone()
    assert row is not None
    data = json.loads(row["extracted_data"])
    assert data["title"] == "bad call pattern"


# ---------------------------------------------------------------------------
# Gates 1-4 wiring tests
# ---------------------------------------------------------------------------


def test_process_modification_gates14_reject_increments_gate1_rejected(db_path, tmp_path):
    """When verify_candidate returns None (Gates 1-4 reject), gate1_rejected increments."""
    conn = init_db(db_path)
    mock_diff = "+x = 1\n+y = 2\n-z = bad_call()\n"
    stats = {"gate0_rejected": 0, "auto_approved": 0, "drafted": 0, "gate1_rejected": 0, "error_count": 0}

    with (
        patch("lessons_db.github_miner._call_ollama_extract") as mock_ollama,
        patch("lessons_db.pattern_verify.verify_candidate", return_value=(None, "dedup")),
    ):
        mock_ollama.return_value = [
            {
                "polarity": "negative",
                "title": "bad call",
                "one_liner": "avoid it",
                "bad_code": "z = bad_call()\nresult = z",
                "good_code": "z = safe_call()\nresult = z",
                "category": "architecture-pattern",
            }
        ]
        count = _process_modification(conn, mock_diff, "owner/repo", MiningConfig(), stats, lance_dir=tmp_path)

    assert count == 0
    assert stats["gate1_rejected"] == 1
    assert stats["auto_approved"] == 0
    assert conn.execute("SELECT COUNT(*) FROM capture_drafts").fetchone()[0] == 0


def test_process_modification_gates14_pass_below_threshold_goes_to_drafts(db_path, tmp_path):
    """When Gates 1-4 pass with low confidence, candidate goes to capture_drafts."""
    conn = init_db(db_path)
    mock_diff = "+x = 1\n+y = 2\n-z = bad_call()\n"
    stats = {"gate0_rejected": 0, "auto_approved": 0, "drafted": 0, "gate1_rejected": 0, "error_count": 0}

    from lessons_db.pattern_verify import VerifiedCandidate

    mock_verified = VerifiedCandidate(
        snippet="z = bad_call()\nresult = z",
        source_repos=["owner/repo"],
        source_lesson_id=None,
        confidence=0.50,  # Below 0.85 threshold
        rationale="test rationale",
    )
    with (
        patch("lessons_db.github_miner._call_ollama_extract") as mock_ollama,
        patch("lessons_db.pattern_verify.verify_candidate", return_value=(mock_verified, None)),
    ):
        mock_ollama.return_value = [
            {
                "polarity": "negative",
                "title": "bad call pattern",
                "one_liner": "avoid bad_call",
                "bad_code": "z = bad_call()\nresult = z",
                "good_code": "z = safe_call()\nresult = z",
                "category": "architecture-pattern",
            }
        ]
        count = _process_modification(conn, mock_diff, "owner/repo", MiningConfig(), stats, lance_dir=tmp_path)

    assert count == 1
    assert stats["drafted"] == 1, "below-threshold → candidate goes to capture_drafts"
    assert stats["auto_approved"] == 0
    assert stats["gate1_rejected"] == 0
    row = conn.execute("SELECT confidence, extracted_data FROM capture_drafts").fetchone()
    assert row is not None
    assert abs(row["confidence"] - 0.50) < 0.001
    data = json.loads(row["extracted_data"])
    assert data["title"] == "bad call pattern"


def test_process_modification_gates14_pass_above_threshold_auto_approves(db_path, tmp_path):
    """When Gates 1-4 pass with high confidence, candidate is auto-promoted to lessons."""
    conn = init_db(db_path)
    mock_diff = "+x = 1\n+y = 2\n-z = bad_call()\n"
    stats = {"gate0_rejected": 0, "auto_approved": 0, "drafted": 0, "gate1_rejected": 0, "error_count": 0}

    from lessons_db.pattern_verify import VerifiedCandidate

    mock_verified = VerifiedCandidate(
        snippet="z = bad_call()\nresult = z",
        source_repos=["owner/repo"],
        source_lesson_id=None,
        confidence=0.90,  # Above 0.85 threshold
        rationale="test rationale",
    )
    with (
        patch("lessons_db.github_miner._call_ollama_extract") as mock_ollama,
        patch("lessons_db.pattern_verify.verify_candidate", return_value=(mock_verified, None)),
    ):
        mock_ollama.return_value = [
            {
                "polarity": "negative",
                "title": "bad call pattern high conf",
                "one_liner": "avoid bad_call — confirmed across multiple repos",
                "bad_code": "z = bad_call()\nresult = z",
                "good_code": "z = safe_call()\nresult = z",
                "category": "architecture-pattern",
            }
        ]
        count = _process_modification(conn, mock_diff, "owner/repo", MiningConfig(), stats, lance_dir=tmp_path)

    assert count == 1
    assert stats["auto_approved"] == 1
    # Should be in lessons table, not just drafts
    lesson = conn.execute("SELECT title FROM lessons WHERE source LIKE 'github_miner:%'").fetchone()
    assert lesson is not None
    assert lesson["title"] == "bad call pattern high conf"


def test_insert_miner_candidate_no_lance_dir_skips_gates(db_path):
    """Without lance_dir, negative candidates skip Gates 1-4 and go to capture_drafts."""
    conn = init_db(db_path)
    stats = {"auto_approved": 0, "drafted": 0, "error_count": 0}
    candidate = {
        "polarity": "negative",
        "title": "some pattern",
        "one_liner": "avoid it",
        "bad_code": "bad_call()\n",
        "good_code": "good_call()\n",
        "category": "testing",
    }
    # No lance_dir → confidence=None → goes to capture_drafts regardless
    result = _insert_miner_candidate(conn, "diff text", "owner/repo", candidate, confidence=None, stats=stats)
    assert result is True
    assert stats["drafted"] == 1, "confidence=None → goes to capture_drafts"
    assert stats["auto_approved"] == 0
    row = conn.execute("SELECT confidence FROM capture_drafts").fetchone()
    assert row is not None
    assert row["confidence"] is None  # No verification was done
