"""Prevention pipeline — detect, escalate, enforce, track, report.

This module owns the full enforcement cycle:
  recurrence event logged → velocity checked → escalation triggered
  → Semgrep rule generated & validated → blocking decision made → outcomes resolved

All callers (CLI, hooks, API) go through this module's public functions.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from lessons_db.config import RULES_DIR
from lessons_db.db import (
    add_to_fix_queue,
    get_fix_queue,
    get_lesson,
    get_open_findings,
    get_recurrence_velocity,
    get_velocity_warnings,
    insert_recurrence_event,
    update_fix_status,
)
from lessons_db.enforce import check_escalation, should_block
from lessons_db.learn import record_outcome
from lessons_db.rulegen import generate_rule, slug_from_title

_log = logging.getLogger(__name__)

VELOCITY_THRESHOLD = 2
VELOCITY_WINDOW_DAYS = 7


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class EnforcementDecision:
    """Result of assess_and_enforce() — one value per lesson hit."""

    lesson_id: int
    enforcement_level: str
    should_block: bool
    rule_generated: bool
    rule_path: Path | None
    recurrence_count: int
    velocity: int
    escalated: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rule_path"] = str(self.rule_path) if self.rule_path else None
        return d


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------


def generate_rule_for_lesson(
    conn: sqlite3.Connection,
    lesson_id: int,
    rules_dir: Path | None = None,
    severity: str = "WARNING",
    validate: bool = True,
) -> Path | None:
    """Generate and write a Semgrep rule YAML for one lesson.

    Returns the written path, or None if:
    - lesson not found
    - lesson has no detection_patterns
    - rule fails validation (when validate=True)

    Idempotent: overwrites existing file, upserts enforcement_rules row.
    """
    lesson = get_lesson(conn, lesson_id)
    if lesson is None:
        _log.warning("generate_rule_for_lesson: lesson %d not found", lesson_id)
        return None

    patterns = conn.execute("SELECT * FROM detection_patterns WHERE lesson_id = ?", (lesson_id,)).fetchall()
    if not patterns:
        _log.debug("generate_rule_for_lesson: lesson %d has no detection_patterns", lesson_id)
        return None

    out_dir = rules_dir or RULES_DIR
    language = patterns[0]["language"] or "any"
    lang_dir = out_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    slug = slug_from_title(lesson["title"])
    rule_id = f"lessons-db.{language}.{slug}-{lesson_id:03d}"
    rule_file = lang_dir / f"{slug}-{lesson_id:03d}.yaml"

    try:
        rule_yaml = generate_rule(dict(lesson), [dict(p) for p in patterns], severity=severity)
    except ValueError as e:
        _log.warning("generate_rule_for_lesson: lesson %d rule gen failed: %s", lesson_id, e)
        return None

    if validate and not _validate_rule_yaml(rule_yaml, rule_id):
        _log.warning(
            "generate_rule_for_lesson: lesson %d rule failed semgrep validation, skipping",
            lesson_id,
        )
        return None

    # Upsert enforcement_rules registry — commit BEFORE writing file to avoid TOCTOU.
    # A DB row without a file is recoverable (next call re-writes it).
    # A file without a DB row silently bypasses tracking.
    existing = conn.execute("SELECT id FROM enforcement_rules WHERE rule_id = ?", (rule_id,)).fetchone()
    if existing:
        conn.execute(
            "UPDATE enforcement_rules SET rule_content = ? WHERE rule_id = ?",
            (rule_yaml, rule_id),
        )
    else:
        conn.execute(
            "INSERT INTO enforcement_rules "
            "(lesson_id, rule_id, rule_type, rule_content, created_date) "
            "VALUES (?, ?, 'semgrep', ?, ?)",
            (lesson_id, rule_id, rule_yaml, date.today().isoformat()),
        )
    conn.commit()

    rule_file.write_text(rule_yaml, encoding="utf-8")
    _log.info("generate_rule_for_lesson: wrote %s", rule_file)
    return rule_file


def _validate_rule_yaml(rule_yaml: str, rule_id: str) -> bool:
    """Validate a Semgrep rule using `semgrep --validate`. Returns True if valid.

    Falls back to True (allow) when semgrep is not installed or validation errors.
    """
    semgrep = shutil.which("semgrep")
    if not semgrep:
        _log.debug("_validate_rule_yaml: semgrep not found, skipping validation")
        return True

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as fh:
            fh.write(rule_yaml)
            tmp_path = fh.name

        result = subprocess.run(  # noqa: S603 — inputs are hardcoded, not user-supplied
            [semgrep, "--validate", tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            _log.warning("_validate_rule_yaml: %s invalid — %s", rule_id, result.stderr[:300])
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        _log.debug("_validate_rule_yaml: exception validating %s: %s", rule_id, exc)
        return True  # best-effort: allow on error
    finally:
        if tmp_path:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


def bulk_generate_rules(
    conn: sqlite3.Connection,
    rules_dir: Path | None = None,
    only_enforcement: tuple[str, ...] | None = None,
    validate: bool = True,
) -> dict:
    """Generate Semgrep rules for all lessons that have detection_patterns.

    Args:
        only_enforcement: if provided, restrict to lessons at these enforcement levels.
        validate: run `semgrep --validate` on each generated rule.

    Returns:
        {generated, skipped_no_patterns, skipped_validation, paths}
    """
    if only_enforcement:
        placeholders = ",".join("?" * len(only_enforcement))
        rows = conn.execute(
            f"SELECT DISTINCT dp.lesson_id FROM detection_patterns dp "
            f"JOIN lessons l ON dp.lesson_id = l.id "
            f"WHERE l.enforcement IN ({placeholders})",
            list(only_enforcement),
        ).fetchall()
    else:
        rows = conn.execute("SELECT DISTINCT lesson_id FROM detection_patterns").fetchall()

    generated = 0
    skipped_no_patterns = 0
    skipped_validation = 0
    paths: list[str] = []

    for row in rows:
        lid = row[0]
        path = generate_rule_for_lesson(conn, lid, rules_dir, validate=validate)
        if path is not None:
            generated += 1
            paths.append(str(path))
        else:
            # Distinguish: no patterns vs. validation failure
            pcount = conn.execute("SELECT COUNT(*) FROM detection_patterns WHERE lesson_id = ?", (lid,)).fetchone()[0]
            if pcount == 0:
                skipped_no_patterns += 1
            else:
                skipped_validation += 1

    _log.info(
        "bulk_generate_rules: generated=%d skipped_no_patterns=%d skipped_validation=%d",
        generated,
        skipped_no_patterns,
        skipped_validation,
    )
    return {
        "generated": generated,
        "skipped_no_patterns": skipped_no_patterns,
        "skipped_validation": skipped_validation,
        "paths": paths,
    }


# ---------------------------------------------------------------------------
# Enforcement cycle
# ---------------------------------------------------------------------------


def assess_and_enforce(
    conn: sqlite3.Connection,
    lesson_id: int,
    hook_point: str,
    trigger_type: str,
    file_path: str | None = None,
    rules_dir: Path | None = None,
) -> EnforcementDecision:
    """Full enforcement cycle for a single lesson hit.

    1. Log recurrence event
    2. Check velocity (window=VELOCITY_WINDOW_DAYS)
    3. If velocity >= VELOCITY_THRESHOLD: call check_escalation()
    4. If action.generate_rule: generate + validate Semgrep rule
    5. Return EnforcementDecision with blocking decision

    Safe: returns a non-blocking decision if lesson is not found.
    """
    lesson = get_lesson(conn, lesson_id)
    if lesson is None:
        _log.error("assess_and_enforce: lesson %d not found", lesson_id)
        return EnforcementDecision(
            lesson_id=lesson_id,
            enforcement_level="documentation",
            should_block=False,
            rule_generated=False,
            rule_path=None,
            recurrence_count=0,
            velocity=0,
            escalated=False,
        )

    # Step 1: log the hit
    insert_recurrence_event(conn, lesson_id, hook_point, trigger_type, file_path)

    # Step 2: velocity in rolling window
    velocity = get_recurrence_velocity(conn, lesson_id, window_days=VELOCITY_WINDOW_DAYS)

    # Step 3: escalate only at the threshold crossing (edge-triggered, not level-triggered).
    # velocity >= VELOCITY_THRESHOLD would re-escalate on every subsequent hit, causing
    # repeated recurrence_count increments and tier promotions within the same window.
    rule_path: Path | None = None
    escalated = False
    action: dict = {
        "generate_rule": False,
        "level": lesson["enforcement"],
        "recurrence_count": lesson["recurrence_count"],
    }

    if velocity == VELOCITY_THRESHOLD:
        try:
            action = check_escalation(conn, lesson_id)
            escalated = True
        except ValueError as exc:
            _log.error("assess_and_enforce: escalation failed for lesson %d: %s", lesson_id, exc)

    # Step 4: generate rule when escalation flags it
    if action.get("generate_rule"):
        rule_path = generate_rule_for_lesson(conn, lesson_id, rules_dir)

    # Step 5: blocking decision. Use action dict directly — check_escalation already
    # returns the updated enforcement level and recurrence_count, avoiding a second DB read.
    return EnforcementDecision(
        lesson_id=lesson_id,
        enforcement_level=action["level"],
        should_block=should_block(action["level"]),
        rule_generated=rule_path is not None,
        rule_path=rule_path,
        recurrence_count=action["recurrence_count"],
        velocity=velocity,
        escalated=escalated,
    )


def check_content(
    conn: sqlite3.Connection,
    content: str,
    file_path: str | None = None,
    rules_dir: Path | None = None,
) -> dict:
    """Check content against detection patterns and run enforcement cycle.

    Returns:
        {block: bool, message: str, violations: [{lesson_id, enforcement, one_liner, decision}]}

    Used by the pre-edit hook and the `prevent check-content` CLI command.
    """
    from lessons_db.search import search_by_content

    matches = search_by_content(conn, content)
    if not matches:
        return {"block": False, "message": "", "violations": []}

    violations = []
    block = False
    block_messages: list[str] = []

    for match in matches:
        lesson_id = match["id"]
        decision = assess_and_enforce(
            conn,
            lesson_id,
            hook_point="edit",
            trigger_type="regex_match",
            file_path=file_path,
            rules_dir=rules_dir,
        )
        lesson = get_lesson(conn, lesson_id)
        violations.append(
            {
                "lesson_id": lesson_id,
                "enforcement": decision.enforcement_level,
                "one_liner": lesson["one_liner"] if lesson else "",
                "should_block": decision.should_block,
                "decision": decision.to_dict(),
            }
        )
        if decision.should_block:
            block = True
            block_messages.append(
                f"Lesson #{lesson_id} [{decision.enforcement_level}]: "
                f"{lesson['one_liner'] if lesson else 'No description'}"
            )

    message = ""
    if block:
        message = "BLOCKED by lessons-db prevention:\n" + "\n".join(f"  • {m}" for m in block_messages)

    return {"block": block, "message": message, "violations": violations}


# ---------------------------------------------------------------------------
# Outcome resolution
# ---------------------------------------------------------------------------


def resolve_outcomes(
    conn: sqlite3.Connection,
    max_age_hours: int = 24,
) -> dict:
    """Batch-resolve stale 'unknown' surfacing events using behavioral inference.

    Logic:
    - Events < 1 hour old: leave as unknown (session may still be active)
    - Events where lesson recurred (recurrence_events since surfacing timestamp): dismissed
    - Events where lesson did NOT recur: heeded

    Returns: {resolved, heeded, dismissed}
    """
    cutoff_recent = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    cutoff_lookback = (datetime.now(UTC) - timedelta(hours=max_age_hours)).isoformat()

    rows = conn.execute(
        "SELECT id, lesson_id, timestamp FROM surfacing_events "
        "WHERE outcome = 'unknown' AND timestamp < ? AND timestamp >= ?",
        [cutoff_recent, cutoff_lookback],
    ).fetchall()

    heeded = dismissed = 0

    for row in rows:
        recur_count = conn.execute(
            "SELECT COUNT(*) FROM recurrence_events " "WHERE lesson_id = ? AND timestamp > ?",
            [row["lesson_id"], row["timestamp"]],
        ).fetchone()[0]

        outcome = "dismissed" if recur_count > 0 else "heeded"
        try:
            record_outcome(conn, row["id"], outcome)
            if outcome == "heeded":
                heeded += 1
            else:
                dismissed += 1
        except ValueError as exc:
            _log.warning("resolve_outcomes: failed for event %d: %s", row["id"], exc)

    resolved = heeded + dismissed
    _log.info("resolve_outcomes: resolved=%d heeded=%d dismissed=%d", resolved, heeded, dismissed)
    return {"resolved": resolved, "heeded": heeded, "dismissed": dismissed}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def prevention_report(
    conn: sqlite3.Connection,
    window_days: int = 30,
) -> dict:
    """Comprehensive prevention effectiveness report.

    Covers: enforcement coverage, rule generation, surfacing outcomes,
    top recurring lessons, velocity alerts, false positive hotspots,
    hookify promotion candidates.
    """
    from lessons_db.learn import surfacing_stats

    # Enforcement coverage breakdown
    enforcement_counts = {
        row["enforcement"]: row["cnt"]
        for row in conn.execute("SELECT enforcement, COUNT(*) as cnt FROM lessons GROUP BY enforcement").fetchall()
    }

    # Rules generated
    rules_generated = conn.execute("SELECT COUNT(DISTINCT lesson_id) FROM enforcement_rules").fetchone()[0]

    # Surfacing stats
    surfacing = surfacing_stats(conn)

    # Top recurring lessons in window
    cutoff = (datetime.now(UTC).date() - timedelta(days=window_days)).isoformat()
    top_recurring = conn.execute(
        """
        SELECT re.lesson_id, l.title, l.enforcement, l.severity, COUNT(*) as hit_count
        FROM recurrence_events re
        JOIN lessons l ON re.lesson_id = l.id
        WHERE re.timestamp >= ?
        GROUP BY re.lesson_id
        ORDER BY hit_count DESC
        LIMIT 10
        """,
        [cutoff],
    ).fetchall()

    # Velocity alerts (2+ hits in 7 days)
    velocity_alerts = get_velocity_warnings(conn, window_days=7, threshold=VELOCITY_THRESHOLD)

    # False positive hotspots
    fp_hotspots = conn.execute(
        """
        SELECT lesson_id,
               COUNT(*) as fp_count,
               (SELECT COUNT(*) FROM surfacing_events s2
                WHERE s2.lesson_id = se.lesson_id) as total_surfacings
        FROM surfacing_events se
        WHERE outcome = 'false_positive'
        GROUP BY lesson_id
        ORDER BY fp_count DESC
        LIMIT 10
        """,
    ).fetchall()

    # Hookify promotion candidates: simple regex, high severity, low enforcement
    hookify_candidates = conn.execute(
        """
        SELECT l.id, l.title, l.enforcement, l.severity, dp.regex, dp.language
        FROM lessons l
        JOIN detection_patterns dp ON l.id = dp.lesson_id
        WHERE l.enforcement IN ('semgrep_warning', 'documentation')
          AND dp.pattern_type IN ('syntactic', 'regex')
          AND length(dp.regex) < 120
          AND l.severity >= 4
        ORDER BY l.severity DESC, l.recurrence_count DESC
        LIMIT 10
        """,
    ).fetchall()

    # Lessons with no detection patterns (documentation-only, no mechanical enforcement)
    no_patterns_count = conn.execute(
        """
        SELECT COUNT(*) FROM lessons
        WHERE id NOT IN (SELECT DISTINCT lesson_id FROM detection_patterns)
        """,
    ).fetchone()[0]

    return {
        "window_days": window_days,
        "enforcement_coverage": enforcement_counts,
        "total_lessons": sum(enforcement_counts.values()),
        "rules_generated": rules_generated,
        "lessons_without_patterns": no_patterns_count,
        "surfacing": surfacing,
        "top_recurring": [dict(r) for r in top_recurring],
        "velocity_alerts": velocity_alerts,
        "false_positive_hotspots": [dict(r) for r in fp_hotspots],
        "hookify_candidates": [dict(r) for r in hookify_candidates],
    }


# ---------------------------------------------------------------------------
# Fix queue population
# ---------------------------------------------------------------------------


def populate_fix_queue(
    conn: sqlite3.Connection,
    min_severity: int = 3,
) -> dict:
    """Populate the fix queue from open scan findings.

    Reads all open scan_findings, looks up their lesson's corrective_action
    (or falls back to one_liner), and adds each as a pending fix. Deduplicates
    via unique index — safe to call repeatedly.

    Returns: {added, skipped_duplicate, skipped_severity, skipped_no_lesson}
    """
    findings = get_open_findings(conn)
    added = skipped_duplicate = skipped_severity = skipped_no_lesson = 0

    for finding in findings:
        lesson_id = finding.get("lesson_id")
        if lesson_id is None:
            skipped_no_lesson += 1
            continue

        lesson = get_lesson(conn, lesson_id)
        if lesson is None:
            skipped_no_lesson += 1
            continue

        if (lesson["severity"] or 0) < min_severity:
            skipped_severity += 1
            continue

        suggested_fix = lesson.get("corrective_action") or lesson.get("one_liner")

        row_id = add_to_fix_queue(
            conn,
            lesson_id=lesson_id,
            file_path=finding["file_path"],
            line_number=finding.get("line_number"),
            snippet=finding.get("snippet"),
            suggested_fix=suggested_fix,
            scan_finding_id=finding["id"],
        )
        if row_id is None:
            skipped_duplicate += 1
        else:
            added += 1

    _log.info(
        "populate_fix_queue: added=%d skipped_duplicate=%d skipped_severity=%d skipped_no_lesson=%d",
        added,
        skipped_duplicate,
        skipped_severity,
        skipped_no_lesson,
    )
    return {
        "added": added,
        "skipped_duplicate": skipped_duplicate,
        "skipped_severity": skipped_severity,
        "skipped_no_lesson": skipped_no_lesson,
    }


# ---------------------------------------------------------------------------
# GitHub issue creation
# ---------------------------------------------------------------------------


def create_github_issues(
    conn: sqlite3.Connection,
    repo: str | None = None,
    min_severity: int = 4,
    dry_run: bool = False,
) -> dict:
    """Create GitHub issues for pending fix queue entries above min_severity.

    Uses `gh issue create` (must be authenticated). Stores the issue URL in
    fix_queue.github_issue_url and marks the entry status as 'issue_created'.
    Skips entries that already have a github_issue_url (idempotent).

    Args:
        repo: GitHub repo in owner/name format. Defaults to LESSONS_DB_ISSUES_REPO
              env var, then current git remote origin.
        min_severity: Only create issues for lessons at or above this severity.
        dry_run: Print what would be created without calling gh.

    Returns: {created, skipped_existing, skipped_severity, errors}
    """
    import os

    if not dry_run and not shutil.which("gh"):
        raise RuntimeError(
            "gh CLI not found — install GitHub CLI (https://cli.github.com) "
            "and run `gh auth login` before creating issues."
        )

    gh_repo = repo or os.environ.get("LESSONS_DB_ISSUES_REPO") or _detect_git_repo()

    pending = get_fix_queue(conn, status="pending")
    created = skipped_existing = skipped_severity = errors = 0

    for fix in pending:
        # Already has a GH issue
        if fix.get("github_issue_url"):
            skipped_existing += 1
            continue

        if (fix.get("severity") or 0) < min_severity:
            skipped_severity += 1
            continue

        title = f"[lessons-db] Fix #{fix['id']}: {fix['one_liner'] or fix['title']}"
        body = _format_issue_body(fix)

        if dry_run:
            _log.info("create_github_issues: DRY RUN — would create: %s", title)
            created += 1
            continue

        try:
            issue_url = _gh_create_issue(gh_repo, title, body)
            update_fix_status(conn, fix["id"], "issue_created", github_issue_url=issue_url)
            _log.info("create_github_issues: created %s", issue_url)
            created += 1
        except Exception as exc:  # noqa: BLE001
            _log.warning("create_github_issues: failed for fix %d: %s", fix["id"], exc)
            errors += 1

    _log.info(
        "create_github_issues: created=%d skipped_existing=%d skipped_severity=%d errors=%d",
        created,
        skipped_existing,
        skipped_severity,
        errors,
    )
    return {
        "created": created,
        "skipped_existing": skipped_existing,
        "skipped_severity": skipped_severity,
        "errors": errors,
    }


def _detect_git_repo() -> str | None:
    """Return 'owner/repo' from the current git remote origin, or None."""
    try:
        result = subprocess.run(  # noqa: S603 — hardcoded git command
            ["git", "remote", "get-url", "origin"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        url = result.stdout.strip()
        # Handle both HTTPS (https://github.com/owner/repo.git) and
        # SSH (git@github.com:owner/repo.git) formats
        if "github.com" in url:
            part = url.split("github.com")[-1].lstrip(":/")
            return part.removesuffix(".git")
        return None
    except Exception:  # noqa: BLE001
        return None


def _format_issue_body(fix: dict) -> str:
    """Format a GitHub issue body from a fix queue entry."""
    lines = [
        "## Lessons-DB Prevention Finding",
        "",
        f"**Lesson #{fix['lesson_id']}:** {fix['title']}",
        f"**Enforcement:** `{fix['enforcement']}`  |  **Severity:** {fix.get('severity', '?')}",
        "",
        "### Location",
        f"`{fix['file_path']}`" + (f":{fix['line_number']}" if fix.get("line_number") else ""),
        "",
    ]
    if fix.get("snippet"):
        lines += ["### Detected Pattern", "```", fix["snippet"], "```", ""]
    if fix.get("suggested_fix"):
        lines += ["### Suggested Fix", fix["suggested_fix"], ""]
    lines += [
        "---",
        "*Auto-generated by [lessons-db](https://github.com/parthalon025/lessons-db). " f"Fix queue ID: {fix['id']}*",
        "",
        f"To apply: `lessons-db fix done {fix['id']}` after fixing, " f"or `lessons-db fix skip {fix['id']}` to skip.",
    ]
    return "\n".join(lines)


def _gh_create_issue(repo: str | None, title: str, body: str) -> str:
    """Call `gh issue create` and return the issue URL. Raises on failure."""
    cmd = ["gh", "issue", "create", "--title", title, "--body", body, "--label", "lessons-db-finding"]
    if repo:
        cmd += ["--repo", repo]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)  # noqa: S603 — cmd built from hardcoded values
    if result.returncode != 0:
        raise RuntimeError(f"gh issue create failed: {result.stderr[:300]}")
    return result.stdout.strip()
