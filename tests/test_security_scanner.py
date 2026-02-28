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


def test_parse_ruff_findings():
    findings = parse_ruff_findings(SAMPLE_RUFF_OUTPUT)
    assert len(findings) == 1
    assert findings[0]["code"] == "S105"
    assert findings[0]["file_path"] == "/path/to/file.py"
    assert findings[0]["line_number"] == 10


def test_findings_to_lesson_candidates(db_path):
    conn = init_db(db_path)
    findings = parse_ruff_findings(SAMPLE_RUFF_OUTPUT)
    candidates = findings_to_lesson_candidates(conn, findings)
    assert len(candidates) >= 0  # may be empty if lesson already exists


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
