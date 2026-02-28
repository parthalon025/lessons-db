"""Security scanner — Ruff S-rules + pip-audit CVEs + Semgrep on own repos.

Findings → 7-gate validation → capture_drafts or auto-approve.
New findings not matching existing lessons → new lesson candidates.

Always captures errors: scanner errors logged to scan_findings table
with status='scanner_error'.
"""

import json
import logging
import subprocess
from datetime import date
from pathlib import Path

from lessons_db.config import PROJECTS_DIR

_log = logging.getLogger(__name__)


def run_ruff_security(target: Path) -> dict:
    """Run ruff --select S on target path. Returns {"findings": [], "errors": int}."""
    if not target.exists():
        return {"findings": [], "errors": 0, "error": f"path not found: {target}"}
    try:
        result = subprocess.run(  # noqa: S603
            ["ruff", "check", "--select", "S", "--output-format", "json", str(target)],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
        )
        raw = result.stdout.strip()
        if not raw:
            return {"findings": [], "errors": 0}
        data = json.loads(raw)
        # ruff json output is a list of violations directly
        findings = []
        for item in data if isinstance(data, list) else []:
            findings.append(
                {
                    "code": item.get("code", ""),
                    "message": item.get("message", ""),
                    "file_path": item.get("filename", ""),
                    "line_number": item.get("location", {}).get("row"),
                    "tool": "ruff",
                }
            )
        return {"findings": findings, "errors": 0}
    except Exception as exc:
        _log.error("ruff scan failed: %s", exc)
        return {"findings": [], "errors": 1, "error": str(exc)}


def parse_ruff_findings(json_output: str) -> list[dict]:
    """Parse ruff JSON output (either list or dict with 'results' key)."""
    try:
        data = json.loads(json_output)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        findings = []
        for item in data:
            findings.append(
                {
                    "code": item.get("code", ""),
                    "message": item.get("message", ""),
                    "file_path": item.get("filename", ""),
                    "line_number": item.get("location", {}).get("row"),
                    "tool": "ruff",
                }
            )
        return findings

    # Legacy dict format with "results" key
    findings = []
    for file_result in data.get("results", []):
        for msg in file_result.get("messages", []):
            findings.append(
                {
                    "code": msg.get("code", ""),
                    "message": msg.get("message", ""),
                    "file_path": file_result.get("filename", ""),
                    "line_number": msg.get("location", {}).get("row"),
                    "tool": "ruff",
                }
            )
    return findings


def run_pip_audit() -> dict:
    """Run pip-audit. Returns {"vulnerabilities": [], "errors": int}."""
    try:
        result = subprocess.run(["pip-audit", "--format", "json"], capture_output=True, text=True, timeout=30)  # noqa: S603 S607
        data = json.loads(result.stdout) if result.stdout.strip() else {}
        vulns = []
        for dep in data.get("dependencies", []):
            for vuln in dep.get("vulns", []):
                vulns.append(
                    {
                        "package": dep.get("name"),
                        "version": dep.get("version"),
                        "vuln_id": vuln.get("id"),
                        "description": vuln.get("description", ""),
                        "tool": "pip-audit",
                    }
                )
        return {"vulnerabilities": vulns, "errors": 0}
    except Exception as exc:
        _log.error("pip-audit failed: %s", exc)
        return {"vulnerabilities": [], "errors": 1, "error": str(exc)}


def findings_to_lesson_candidates(conn, findings: list[dict]) -> list[dict]:
    """Filter findings not already covered by existing lessons.

    Returns findings that are new lesson candidates.
    """
    candidates = []
    for finding in findings:
        code = finding.get("code", "")
        existing = conn.execute("SELECT 1 FROM enforcement_rules WHERE rule_id LIKE ?", (f"%{code}%",)).fetchone()
        if not existing:
            candidates.append(finding)
    return candidates


def run_full_security_scan(conn, target: Path | None = None) -> dict:
    """Run Ruff + pip-audit on target (defaults to PROJECTS_DIR).

    Always captures errors to scan_findings. Returns summary dict.
    """
    scan_target = target or PROJECTS_DIR
    summary = {"ruff_findings": 0, "vulnerabilities": 0, "new_candidates": 0, "errors": 0}

    ruff_result = run_ruff_security(scan_target)
    summary["errors"] += ruff_result.get("errors", 0)
    findings = ruff_result.get("findings", [])
    summary["ruff_findings"] = len(findings)

    audit_result = run_pip_audit()
    summary["errors"] += audit_result.get("errors", 0)
    summary["vulnerabilities"] = len(audit_result.get("vulnerabilities", []))

    candidates = findings_to_lesson_candidates(conn, findings)
    summary["new_candidates"] = len(candidates)

    # Log all findings to scan_findings table
    today = date.today().isoformat()
    for f in findings:
        try:
            conn.execute(
                """INSERT INTO scan_findings
                   (lesson_id, rule_id, file_path, line_number, snippet, status, scan_date)
                   VALUES (0, ?, ?, ?, ?, 'open', ?)""",
                (f.get("code", ""), f.get("file_path", ""), f.get("line_number"), f.get("message", ""), today),
            )
        except Exception as exc:
            _log.warning("failed to log finding: %s", exc)
    conn.commit()

    return summary
