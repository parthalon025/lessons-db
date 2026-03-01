"""BugsInPy calibration for the lesson extraction pipeline.

Uses the soarsmu/BugsInPy dataset (493 confirmed Python bugs across 17 projects)
as ground truth to measure how effectively the pipeline extracts lessons from
real bug-fix diffs.

Calibration flow:
  1. Clone soarsmu/BugsInPy (--depth 1, cached after first run).
  2. Scan projects/*/bugs/*/bug_patch.diff for valid diffs.
  3. For a configurable sample, run the extraction pipeline up to Gates 0a/0b
     and optionally through Gates 1-4 (requires Ollama).
  4. Compute pass rates at each gate.
  5. Record results in calibration_runs table.

Pass-rate interpretation:
  >= 0.70  Pipeline is calibrated — proceed to live GitHub mining.
  < 0.70   Gate thresholds need tuning before enabling live mining.

Usage:
  lessons-db calibrate bugsInPy [--sample N] [--cache-dir PATH] [--skip-extraction]
"""

import logging
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from lessons_db.db import insert_calibration_run
from lessons_db.github_miner import filter_diff_by_size

_log = logging.getLogger(__name__)

BUGSINPY_REPO = "https://github.com/soarsmu/BugsInPy"
DEFAULT_CACHE_DIR = Path.home() / ".local" / "share" / "lessons-db" / "bugsInPy"
DEFAULT_SAMPLE = 50


@dataclass
class BugEntry:
    project: str
    bug_id: str
    diff_path: Path
    diff_text: str = ""


@dataclass
class CalibrationResult:
    bug: BugEntry
    diff_lines: int = 0
    size_pass: bool = False
    extraction_attempted: bool = False
    extraction_success: bool = False
    gate0_pass: bool = False
    gate14_pass: bool = False
    candidates_extracted: int = 0
    candidates_gate0: int = 0
    candidates_gate14: int = 0
    error: str | None = None


@dataclass
class CalibrationReport:
    dataset: str = "BugsInPy"
    bugs_sampled: int = 0
    bugs_with_valid_diffs: int = 0
    extraction_attempted: int = 0
    extraction_success: int = 0
    gate0_pass: int = 0
    gate14_pass: int = 0
    pass_rate: float = 0.0
    threshold_met: bool = False
    results: list[CalibrationResult] = field(default_factory=list)
    notes: str = ""


def ensure_bugsInPy(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Clone or verify BugsInPy dataset, return path to cloned repo."""
    marker = cache_dir / ".git"
    if marker.exists():
        _log.info("BugsInPy already cached at %s", cache_dir)
        return cache_dir

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    _log.info("Cloning BugsInPy from %s …", BUGSINPY_REPO)
    result = subprocess.run(  # noqa: S603
        ["git", "clone", "--depth", "1", BUGSINPY_REPO, str(cache_dir)],  # noqa: S607
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    _log.info("BugsInPy cloned to %s", cache_dir)
    return cache_dir


def list_bugs(repo_dir: Path) -> list[BugEntry]:
    """Scan BugsInPy repo and return all bugs with valid diff files."""
    bugs: list[BugEntry] = []
    projects_dir = repo_dir / "projects"
    if not projects_dir.exists():
        _log.warning("BugsInPy projects/ directory not found at %s", projects_dir)
        return bugs

    for diff_path in sorted(projects_dir.glob("*/bugs/*/bug_patch.diff")):
        parts = diff_path.parts
        # Expect: ...projects/<project>/bugs/<bug_id>/bug_patch.diff
        try:
            bugs_idx = parts.index("bugs")
            project = parts[bugs_idx - 1]
            bug_id = parts[bugs_idx + 1]
        except (ValueError, IndexError):
            continue

        try:
            diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _log.debug("Could not read diff for %s/%s: %s", project, bug_id, exc)
            continue

        if diff_text.strip():
            bugs.append(BugEntry(project=project, bug_id=bug_id, diff_path=diff_path, diff_text=diff_text))

    _log.info("Found %d bugs with diff files in BugsInPy", len(bugs))
    return bugs


def _run_gate0(candidate: dict) -> bool:
    """Return True if candidate passes Gates 0a and 0b."""
    from lessons_db.pattern_validator import run_gate0

    return run_gate0(candidate)


def _run_gate14(candidate: dict, conn: sqlite3.Connection, lance_dir: str) -> bool:
    """Return True if negative candidate passes Gates 1-4 (verify_candidate)."""
    from lessons_db.pattern_extract import CandidatePattern
    from lessons_db.pattern_verify import verify_candidate

    cp = CandidatePattern(
        snippet=candidate.get("bad_code", ""),
        source_repos=["BugsInPy"],
        source_lesson_id=None,
        detection_method="calibration",
    )
    verified = verify_candidate(cp, conn, lance_dir)
    return verified is not None


def _process_bug(
    bug: BugEntry,
    report: CalibrationReport,
    conn: sqlite3.Connection,
    lance_dir: Path | str | None,
    skip_extraction: bool,
) -> CalibrationResult:
    """Process one bug entry and update report counters in place."""
    from lessons_db.github_miner import _call_ollama_extract

    res = CalibrationResult(bug=bug)
    res.diff_lines = sum(1 for line in bug.diff_text.splitlines() if line.startswith("+") or line.startswith("-"))
    res.size_pass = filter_diff_by_size(bug.diff_text, min_lines=2, max_lines=1000)
    if not res.size_pass:
        return res

    report.bugs_with_valid_diffs += 1

    if skip_extraction:
        res.gate0_pass = True
        report.gate0_pass += 1
        return res

    res.extraction_attempted = True
    report.extraction_attempted += 1
    try:
        candidates = _call_ollama_extract(bug.diff_text[:3000], f"BugsInPy:{bug.project}/{bug.bug_id}")
    except Exception as exc:
        res.error = f"extraction error: {exc}"
        _log.warning("extraction failed for %s/%s: %s", bug.project, bug.bug_id, exc)
        return res

    if not candidates:
        return res

    res.extraction_success = True
    res.candidates_extracted = len(candidates)
    report.extraction_success += 1
    _score_candidates(candidates, res, report, conn, lance_dir)
    return res


def _score_candidates(
    candidates: list[dict],
    res: CalibrationResult,
    report: CalibrationReport,
    conn: sqlite3.Connection,
    lance_dir: Path | str | None,
) -> None:
    """Run Gate 0 + optional Gates 1-4 on candidates; update res and report counts."""
    gate0_any = False
    gate14_any = False
    for c in candidates:
        if "polarity" not in c:
            c["polarity"] = "negative"
        if not _run_gate0(c):
            continue
        res.candidates_gate0 += 1
        gate0_any = True
        # Positive candidates have no bad_code — skip Gates 1-4.
        if lance_dir is not None and c.get("polarity") != "positive" and _run_gate14(c, conn, str(lance_dir)):
            res.candidates_gate14 += 1
            gate14_any = True

    res.gate0_pass = gate0_any
    res.gate14_pass = gate14_any
    if gate0_any:
        report.gate0_pass += 1
    if gate14_any:
        report.gate14_pass += 1


def _insert_run_safe(conn: sqlite3.Connection, report: CalibrationReport) -> None:
    """Insert calibration run record, suppressing DB errors so callers always return."""
    try:
        insert_calibration_run(
            conn,
            dataset=report.dataset,
            bugs_sampled=report.bugs_sampled,
            bugs_with_valid_diffs=report.bugs_with_valid_diffs,
            extraction_attempted=report.extraction_attempted,
            extraction_success=report.extraction_success,
            gate0_pass=report.gate0_pass,
            gate14_pass=report.gate14_pass,
            notes=report.notes,
        )
    except Exception as exc:
        _log.warning("calibration_runs insert failed: %s", exc)


def calibrate_pipeline(
    conn: sqlite3.Connection,
    lance_dir: Path | str | None = None,
    sample_n: int = DEFAULT_SAMPLE,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    skip_extraction: bool = False,
) -> CalibrationReport:
    """Run calibration against BugsInPy. Returns a CalibrationReport.

    Args:
        conn: SQLite connection (used for Gates 1-4 suppression lookup).
        lance_dir: LanceDB directory (required for Gates 1-4). If None, Gates
                   1-4 are skipped; only extraction + Gate 0 are measured.
        sample_n: Number of bugs to sample (default 50 to limit Ollama calls).
        cache_dir: Where to cache the cloned BugsInPy repo.
        skip_extraction: If True, skip Ollama extraction calls. Reports size
                         pass rate only (useful for offline testing).
    """
    report = CalibrationReport()

    try:
        repo_dir = ensure_bugsInPy(cache_dir)
    except RuntimeError as exc:
        report.notes = f"Clone failed: {exc}"
        _log.error("BugsInPy clone failed: %s", exc)
        _insert_run_safe(conn, report)
        return report

    all_bugs = list_bugs(repo_dir)
    if not all_bugs:
        report.notes = "No bugs found in BugsInPy repo"
        _insert_run_safe(conn, report)
        return report

    sample = all_bugs[:sample_n]
    report.bugs_sampled = len(sample)
    report.results = [_process_bug(bug, report, conn, lance_dir, skip_extraction) for bug in sample]

    report.pass_rate = report.gate0_pass / report.bugs_sampled if report.bugs_sampled else 0.0
    report.threshold_met = report.pass_rate >= 0.70
    report.notes = f"Pass rate {report.pass_rate:.0%} " + (
        "≥ 70% — pipeline calibrated" if report.threshold_met else "< 70% — tune gate thresholds before live mining"
    )

    _insert_run_safe(conn, report)
    return report


def format_report(report: CalibrationReport) -> str:
    """Format calibration report as human-readable text."""
    lines = [
        "=== BugsInPy Calibration Report ===",
        f"Dataset:           {report.dataset}",
        f"Bugs sampled:      {report.bugs_sampled}",
        f"Valid diffs:       {report.bugs_with_valid_diffs}",
        f"Extraction success:{report.extraction_success} / {report.extraction_attempted} attempted",
        f"Gate 0 pass:       {report.gate0_pass} ({report.pass_rate:.0%})",
        f"Gate 1-4 pass:     {report.gate14_pass}",
        "",
        f"{'✓' if report.threshold_met else '✗'} {report.notes}",
    ]
    if not report.threshold_met:
        lines.append("")
        lines.append("Failing bugs (sample):")
        fails = [r for r in report.results if not r.gate0_pass and r.size_pass]
        for r in fails[:5]:
            lines.append(
                f"  {r.bug.project}/{r.bug.bug_id}: extracted={r.candidates_extracted}, gate0={r.candidates_gate0}"
            )
    return "\n".join(lines)
