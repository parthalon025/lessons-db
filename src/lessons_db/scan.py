"""Semgrep scanning and SARIF result parsing."""

import json
import logging
import shutil
import subprocess
from pathlib import Path

from lessons_db.config import SEMGREP_RULES_DIR

logger = logging.getLogger(__name__)

SEMGREP_BIN = shutil.which("semgrep") or str(Path.home() / ".local" / "bin" / "semgrep")


def parse_sarif(sarif: dict) -> list[dict]:
    """Parse SARIF JSON into a list of finding dicts."""
    findings = []
    for run in sarif.get("runs", []):
        for result in run.get("results", []):
            location = {}
            locations = result.get("locations", [])
            if locations:
                phys = locations[0].get("physicalLocation", {})
                location = {
                    "file_path": phys.get("artifactLocation", {}).get("uri", ""),
                    "line_number": phys.get("region", {}).get("startLine"),
                }

            findings.append(
                {
                    "rule_id": result.get("ruleId", ""),
                    "message": result.get("message", {}).get("text", ""),
                    "file_path": location.get("file_path", ""),
                    "line_number": location.get("line_number"),
                    "matched_content": result.get("message", {}).get("text", ""),
                }
            )

    return findings


def run_scan(
    rules_dir: Path | None = None,
    target_dir: Path | None = None,
    baseline_commit: str | None = None,
    sarif_output: bool = True,
) -> list[dict]:
    """Run Semgrep scan and return parsed findings."""
    rules = str(rules_dir or SEMGREP_RULES_DIR)
    target = str(target_dir or Path.home() / "Documents" / "projects")

    cmd = [SEMGREP_BIN, "--config", rules]

    if sarif_output:
        cmd.append("--sarif")

    if baseline_commit:
        cmd.extend(["--baseline-commit", baseline_commit])

    cmd.append(target)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Semgrep exits 0 (clean) or 1 (findings exist); other codes are errors
        if result.returncode not in (0, 1):
            logger.error("Semgrep exited %d. stderr: %s", result.returncode, result.stderr[:500])
            return []

        if sarif_output and result.stdout:
            try:
                sarif = json.loads(result.stdout)
                return parse_sarif(sarif)
            except json.JSONDecodeError:
                logger.error("Semgrep SARIF parse failed. stdout: %s", result.stdout[:200])
                return []

        return []

    except subprocess.TimeoutExpired:
        logger.error("Semgrep scan timed out after 300s")
        return []
    except Exception as e:
        logger.error("Semgrep scan failed: %s", e)
        return []
