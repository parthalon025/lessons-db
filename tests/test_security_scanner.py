# tests/test_security_scanner.py
import json
from unittest.mock import patch

from lessons_db.db import init_db
from lessons_db.security_scanner import (
    findings_to_lesson_candidates,
    parse_ruff_findings,
    run_pip_audit,
    run_ruff_security,
)

SAMPLE_RUFF_OUTPUT = json.dumps(
    {
        "results": [
            {
                "filename": "/path/to/file.py",
                "messages": [
                    {
                        "code": "S105",
                        "message": "Possible hardcoded password assigned to: 'password'",
                        "location": {"row": 10, "column": 5},
                    }
                ],
            }
        ]
    }
)

SAMPLE_RUFF_LIST_OUTPUT = json.dumps(
    [
        {
            "code": "S106",
            "message": "Possible hardcoded password",
            "filename": "/path/to/other.py",
            "location": {"row": 5, "column": 1},
        }
    ]
)


def test_parse_ruff_findings():
    findings = parse_ruff_findings(SAMPLE_RUFF_OUTPUT)
    assert len(findings) == 1
    assert findings[0]["code"] == "S105"
    assert findings[0]["file_path"] == "/path/to/file.py"
    assert findings[0]["line_number"] == 10


def test_parse_ruff_findings_list_format():
    findings = parse_ruff_findings(SAMPLE_RUFF_LIST_OUTPUT)
    assert len(findings) == 1
    assert findings[0]["code"] == "S106"
    assert findings[0]["file_path"] == "/path/to/other.py"
    assert findings[0]["line_number"] == 5


def test_findings_to_lesson_candidates(db_path):
    conn = init_db(db_path)
    findings = parse_ruff_findings(SAMPLE_RUFF_OUTPUT)
    # No enforcement_rules for S105 in a fresh DB → it is a candidate
    candidates = findings_to_lesson_candidates(conn, findings)
    assert len(candidates) == 1
    assert candidates[0]["code"] == "S105"


def test_run_ruff_security_nonexistent_path(tmp_path):
    # Should not raise — return empty result on missing path
    result = run_ruff_security(tmp_path / "nonexistent")
    assert result["findings"] == []
    assert result["errors"] == 0 or result["error"] is not None


def test_run_pip_audit_returns_dict():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = json.dumps({"dependencies": []})
        mock_run.return_value.returncode = 0
        result = run_pip_audit()
    assert "vulnerabilities" in result


def test_run_full_security_scan(db_path, tmp_path):
    conn = init_db(db_path)
    # Create a real path so the path-exists check passes
    scan_dir = tmp_path / "fake_project"
    scan_dir.mkdir()

    ruff_output = json.dumps(
        [
            {
                "code": "S105",
                "message": "Hardcoded password",
                "filename": str(scan_dir / "file.py"),
                "location": {"row": 1, "column": 0},
            }
        ]
    )
    pip_output = json.dumps({"dependencies": []})

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            type("R", (), {"stdout": ruff_output, "returncode": 1, "stderr": ""})(),
            type("R", (), {"stdout": pip_output, "returncode": 0, "stderr": ""})(),
        ]
        from lessons_db.security_scanner import run_full_security_scan

        summary = run_full_security_scan(conn, target=scan_dir)

    assert summary["ruff_findings"] == 1
    assert summary["vulnerabilities"] == 0
    assert summary["new_candidates"] == 1  # S105 not in enforcement_rules
    assert summary["errors"] == 0
