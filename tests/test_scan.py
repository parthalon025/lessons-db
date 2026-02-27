"""Tests for Semgrep scanning and SARIF parsing."""

import json
from unittest.mock import MagicMock, patch

from lessons_db.scan import parse_sarif, run_scan

SAMPLE_SARIF = {
    "version": "2.1.0",
    "runs": [
        {
            "results": [
                {
                    "ruleId": "lessons-db.python.bare-except-007",
                    "message": {"text": "Never use bare except:pass"},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": "src/bad.py"},
                                "region": {"startLine": 42},
                            }
                        }
                    ],
                    "properties": {"metadata": {"lesson_id": 7}},
                }
            ]
        }
    ],
}


class TestSARIFParsing:
    def test_parses_findings_from_sarif(self):
        findings = parse_sarif(SAMPLE_SARIF)
        assert len(findings) == 1
        assert findings[0]["file_path"] == "src/bad.py"
        assert findings[0]["line_number"] == 42
        assert findings[0]["rule_id"] == "lessons-db.python.bare-except-007"

    def test_handles_empty_sarif(self):
        empty = {"version": "2.1.0", "runs": [{"results": []}]}
        findings = parse_sarif(empty)
        assert findings == []


class TestRunScan:
    @patch("lessons_db.scan.subprocess.run")
    def test_run_scan_calls_semgrep(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"version": "2.1.0", "runs": [{"results": []}]}),
        )
        findings = run_scan(
            rules_dir=tmp_path / "rules",
            target_dir=tmp_path / "project",
            sarif_output=True,
        )
        assert findings == []
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "semgrep" in cmd[0]
        assert "--sarif" in cmd
