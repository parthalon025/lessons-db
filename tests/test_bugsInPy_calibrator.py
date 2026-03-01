# tests/test_bugsInPy_calibrator.py
"""Tests for the BugsInPy calibration module."""

from unittest.mock import patch

import pytest

from lessons_db.bugsInPy_calibrator import (
    CalibrationReport,
    _run_gate0,
    calibrate_pipeline,
    format_report,
    list_bugs,
)
from lessons_db.db import init_db


@pytest.fixture
def fake_bugsInPy(tmp_path):
    """Create a minimal BugsInPy-style directory structure for testing."""
    repo_dir = tmp_path / "BugsInPy"
    (repo_dir / ".git").mkdir(parents=True)

    # Two projects, two bugs each
    for project, bug_id, diff in [
        ("thefuck", "1", "+import asyncio\n-import time\n+asyncio.sleep(5)\n-time.sleep(5)\n"),
        ("thefuck", "2", "+x = safe_call()\n-x = bad_call()\n"),
        ("ansible", "1", "+result = None\n-result = bad_func()\n"),
        ("ansible", "2", "+pass\n"),  # too small — will fail size gate
    ]:
        bug_dir = repo_dir / "projects" / project / "bugs" / bug_id
        bug_dir.mkdir(parents=True)
        (bug_dir / "bug_patch.diff").write_text(diff)

    return repo_dir


def test_list_bugs_finds_diffs(fake_bugsInPy):
    bugs = list_bugs(fake_bugsInPy)
    assert len(bugs) == 4
    projects = {b.project for b in bugs}
    assert "thefuck" in projects
    assert "ansible" in projects


def test_list_bugs_empty_on_missing_dir(tmp_path):
    bugs = list_bugs(tmp_path / "nonexistent")
    assert bugs == []


def test_run_gate0_valid_negative():
    candidate = {
        "polarity": "negative",
        "title": "use asyncio.sleep",
        "one_liner": "avoid time.sleep in async code",
        "bad_code": "import time\ntime.sleep(5)",
        "good_code": "import asyncio\nawait asyncio.sleep(5)",
    }
    assert _run_gate0(candidate) is True


def test_run_gate0_rejects_missing_fields():
    candidate = {
        "polarity": "negative",
        "title": "",
        "one_liner": "",
        "bad_code": "bad()\n",
        "good_code": "good()\n",
    }
    assert _run_gate0(candidate) is False


def test_run_gate0_rejects_invalid_syntax():
    candidate = {
        "polarity": "negative",
        "title": "a",
        "one_liner": "b",
        "bad_code": "def broken(\n  pass",  # SyntaxError
        "good_code": "def good():\n    pass",
    }
    assert _run_gate0(candidate) is False


def test_run_gate0_positive_requires_title_oneliner_goodcode():
    candidate = {
        "polarity": "positive",
        "title": "good pattern",
        "one_liner": "use this approach",
        "bad_code": "N/A",
        "good_code": "await asyncio.sleep(5)",
    }
    assert _run_gate0(candidate) is True


def test_calibrate_pipeline_skip_extraction(fake_bugsInPy, db_path):
    """With skip_extraction=True, calibration runs offline (no Ollama)."""
    conn = init_db(db_path)
    # fake_bugsInPy has .git so ensure_bugsInPy returns it directly without cloning.
    report = calibrate_pipeline(
        conn=conn,
        lance_dir=None,
        sample_n=4,
        cache_dir=fake_bugsInPy,
        skip_extraction=True,
    )
    assert isinstance(report, CalibrationReport)


def test_calibrate_pipeline_with_mocked_extraction(fake_bugsInPy, db_path):
    """With mocked extraction, calibration counts gate0 passes correctly."""
    conn = init_db(db_path)

    with (
        patch("lessons_db.bugsInPy_calibrator.ensure_bugsInPy", return_value=fake_bugsInPy),
        patch("lessons_db.github_miner._call_ollama_extract") as mock_extract,
    ):
        mock_extract.return_value = [
            {
                "polarity": "negative",
                "title": "blocking sleep",
                "one_liner": "avoid time.sleep",
                "bad_code": "import time\ntime.sleep(5)",
                "good_code": "import asyncio\nawait asyncio.sleep(5)",
                "category": "async",
            }
        ]
        report = calibrate_pipeline(conn=conn, lance_dir=None, sample_n=4, skip_extraction=False)

    assert report.bugs_sampled == 4
    assert report.extraction_attempted >= 1
    assert report.extraction_success >= 1
    assert report.gate0_pass >= 1
    assert 0.0 <= report.pass_rate <= 1.0


def test_calibrate_pipeline_records_to_db(fake_bugsInPy, db_path):
    """Calibration results are persisted to calibration_runs table."""
    conn = init_db(db_path)

    with (
        patch("lessons_db.bugsInPy_calibrator.ensure_bugsInPy", return_value=fake_bugsInPy),
        patch("lessons_db.github_miner._call_ollama_extract", return_value=[]),
    ):
        calibrate_pipeline(conn=conn, lance_dir=None, sample_n=3)

    row = conn.execute("SELECT * FROM calibration_runs ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    assert row["dataset"] == "BugsInPy"
    assert row["bugs_sampled"] == 3


def test_format_report_shows_threshold_status():
    report = CalibrationReport(
        bugs_sampled=10,
        gate0_pass=8,
        pass_rate=0.80,
        threshold_met=True,
        notes="Pass rate 80% ≥ 70% — pipeline calibrated",
    )
    text = format_report(report)
    assert "80%" in text
    assert "calibrated" in text.lower()


def test_format_report_shows_failure_samples():
    report = CalibrationReport(
        bugs_sampled=10,
        gate0_pass=5,
        pass_rate=0.50,
        threshold_met=False,
        notes="Pass rate 50% < 70% — tune gate thresholds",
        results=[
            type(
                "R",
                (),
                {
                    "gate0_pass": False,
                    "size_pass": True,
                    "bug": type("B", (), {"project": "thefuck", "bug_id": "1"})(),
                    "candidates_extracted": 0,
                    "candidates_gate0": 0,
                },
            )()
        ],
    )
    text = format_report(report)
    assert "50%" in text
    assert "thefuck" in text
