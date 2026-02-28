"""Click CLI for lessons-db: status, search, migrate."""

import json
import logging
from pathlib import Path

import click

from lessons_db import pattern_extract, pattern_triage, pattern_verify
from lessons_db.config import LANCE_DIR, LESSONS_SOURCE_DIR, RULES_DIR, SQLITE_PATH
from lessons_db.db import (
    get_near_miss_hotspots,
    get_open_findings,
    get_overdue_actions,
    get_scan_state,
    init_db,
    insert_corrective_action,
    insert_lesson,
    set_scan_state,
)
from lessons_db.migrate import parse_lesson_file
from lessons_db.search import search_combined

logger = logging.getLogger(__name__)


@click.group()
@click.option("--db", type=click.Path(), default=None, help="SQLite database path override.")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx, db, verbose):
    """lessons-learned prevention system — capture, search, and enforce coding lessons."""
    from lessons_db.logging_config import configure_logging

    configure_logging(level=logging.DEBUG if verbose else logging.WARNING)

    ctx.ensure_object(dict)
    db_path = Path(db) if db else SQLITE_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_db(db_path)
    ctx.obj["conn"] = conn
    ctx.obj["lance_dir"] = LANCE_DIR
    ctx.call_on_close(conn.close)


@main.command()
@click.pass_context
def status(ctx):
    """Show lesson counts, enforcement breakdown, overdue actions, findings, and hotspots."""
    conn = ctx.obj["conn"]

    # Total lessons
    total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    click.echo(f"Total lessons: {total}")

    # Enforcement breakdown
    rows = conn.execute(
        "SELECT enforcement, COUNT(*) as cnt FROM lessons GROUP BY enforcement ORDER BY cnt DESC"
    ).fetchall()
    if rows:
        click.echo("\nEnforcement breakdown:")
        for row in rows:
            click.echo(f"  {row['enforcement']}: {row['cnt']}")

    # Overdue corrective actions (top 5)
    overdue = get_overdue_actions(conn)[:5]
    if overdue:
        click.echo(f"\nOverdue corrective actions ({len(overdue)}):")
        for action in overdue:
            click.echo(f"  [{action['lesson_id']}] {action['action']} (due {action['due_date']})")

    # Open scan findings (top 5)
    findings = get_open_findings(conn)[:5]
    if findings:
        click.echo(f"\nOpen scan findings ({len(findings)}):")
        for f in findings:
            click.echo(f"  [{f['rule_id']}] {f['file_path']}:{f.get('line_number', '?')}")

    # Near-miss hotspots (top 5)
    hotspots = get_near_miss_hotspots(conn, limit=5)
    if hotspots:
        click.echo(f"\nNear-miss hotspots ({len(hotspots)}):")
        for h in hotspots:
            click.echo(f"  {h['file_path']}: {h['count']} events")


@main.command()
@click.argument("query")
@click.option("--file", "-f", default=None, help="File path to search for.")
@click.option("--content", "-c", default=None, help="Code content to match against patterns.")
@click.option("--top", "-k", default=5, type=int, help="Max results to return.")
@click.option(
    "--polarity",
    default=None,
    type=click.Choice(["positive", "negative"]),
    help="Filter by polarity: 'positive' for what works, 'negative' for anti-patterns.",
)
@click.pass_context
def search(ctx, query, file, content, top, polarity):
    """Search lessons by text, file path, or content pattern."""
    conn = ctx.obj["conn"]

    # Try to init LanceDB for semantic search (graceful failure)
    lance_db = None
    try:
        import lancedb

        if LANCE_DIR.exists():
            lance_db = lancedb.connect(str(LANCE_DIR))
    except Exception:
        logger.debug("LanceDB unavailable, skipping semantic search")

    results = search_combined(
        conn,
        lance_db,
        file_path=file,
        content=content,
        query=query,
        polarity=polarity,
    )[:top]

    if not results:
        click.echo("No results found.")
        return

    for r in results:
        rid = r.get("id", "?")
        one_liner = r.get("one_liner", "")
        source = r.get("matched_pattern") or r.get("cluster") or ""
        if source:
            click.echo(f"[#{rid}] {one_liner} (via {source})")
        else:
            click.echo(f"[#{rid}] {one_liner}")


@main.command()
@click.option(
    "--source", type=click.Path(exists=True), default=None, help="Source directory for lesson markdown files."
)
@click.option("--db", "db_override", type=click.Path(), default=None, help="Override DB path for migration.")
@click.option("--dry-run", is_flag=True, help="List files without inserting.")
@click.pass_context
def migrate(ctx, source, db_override, dry_run):
    """Migrate markdown lesson files into the database."""
    conn = ctx.obj["conn"]

    # If db_override provided, reinitialize connection
    if db_override:
        db_path = Path(db_override)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = init_db(db_path)

    source_dir = Path(source) if source else LESSONS_SOURCE_DIR
    md_files = sorted(f for f in source_dir.glob("[0-9][0-9][0-9][0-9]-*.md") if f.is_file())

    if dry_run:
        click.echo(f"Found {len(md_files)} lesson file(s):")
        for f in md_files:
            parsed = parse_lesson_file(f)
            num = parsed.get("lesson_number") or "?"
            title = parsed.get("title", f.name)
            click.echo(f"  #{num}: {title}")
        click.echo(f"\nTotal: {len(md_files)}")
        return

    migrated = 0
    skipped = 0
    errors = 0

    # Build set of markdown_paths already in DB to skip duplicates
    existing_paths = {
        row[0] for row in conn.execute("SELECT markdown_path FROM lessons WHERE markdown_path IS NOT NULL").fetchall()
    }

    for f in md_files:
        try:
            if str(f) in existing_paths:
                skipped += 1
                continue

            parsed = parse_lesson_file(f)
            lesson_data = {
                "title": parsed["title"],
                "one_liner": parsed.get("key_takeaway", ""),
                "description": parsed.get("description", ""),
                "cluster": parsed.get("cluster", ""),
                "tier": parsed.get("tier", "observation"),
                "category": parsed.get("category", ""),
                "scope": parsed.get("scope", ""),
                "keywords": parsed.get("keywords", ""),
                "created_date": parsed.get("date", ""),
                "source": "migrated",
                "markdown_path": str(f),
            }
            lesson_id = insert_lesson(conn, lesson_data)

            # Insert corrective actions
            for action in parsed.get("corrective_actions", []):
                insert_corrective_action(
                    conn,
                    {
                        "lesson_id": lesson_id,
                        "action": action.get("description", ""),
                        "status": action.get("status", "proposed"),
                    },
                )

            migrated += 1
        except Exception as exc:
            logger.error("Failed to migrate %s: %s", f.name, exc)
            errors += 1

    click.echo(f"Migrated: {migrated}, Skipped: {skipped}, Errors: {errors}")


@main.command()
@click.option("--seed-only", is_flag=True, help="Only backfill cluster_seed, skip embedding generation.")
@click.pass_context
def index(ctx, seed_only):
    """Backfill cluster_seed and generate LanceDB embeddings for all lessons.

    Run once after initial migrate, or after adding new lessons without embeddings.
    cluster_seed: copies cluster → cluster_seed for A-F historical labels.
    Embeddings: calls Ollama nomic-embed-text for each lesson's title + one_liner.
    """
    from lessons_db.vectors import init_lance, upsert_lesson

    conn = ctx.obj["conn"]

    # Step 1: backfill cluster_seed from cluster for lessons that have cluster but no seed
    updated = conn.execute(
        "UPDATE lessons SET cluster_seed = cluster WHERE cluster IS NOT NULL AND cluster != '' AND cluster_seed IS NULL"
    ).rowcount
    conn.commit()
    click.echo(f"cluster_seed backfill: {updated} rows updated")

    if seed_only:
        return

    # Step 2: generate embeddings for all lessons
    lance_db = init_lance(str(LANCE_DIR))
    LANCE_DIR.mkdir(parents=True, exist_ok=True)

    rows = conn.execute(
        "SELECT id, title, one_liner, keywords, cluster, tier, scope, enforcement, recurrence_count FROM lessons"
    ).fetchall()

    ok = 0
    failed = 0
    for row in rows:
        title = row["title"] or ""
        one_liner = row["one_liner"] or ""
        keywords = row["keywords"] or ""
        text = f"{title}. {one_liner}"
        if keywords:
            text += f". Keywords: {keywords}"

        data = {
            "lesson_id": row["id"],
            "text": text,
            "cluster": row["cluster"] or "",
            "tier": row["tier"] or "",
            "scope": row["scope"] or "",
            "enforcement": row["enforcement"] or "",
            "recurrence_count": row["recurrence_count"] or 0,
        }
        if upsert_lesson(lance_db, data):
            ok += 1
        else:
            failed += 1
            logger.warning("index: embedding failed for lesson #%d", row["id"])

        if (ok + failed) % 10 == 0:
            click.echo(f"  {ok + failed}/{len(rows)} indexed...", err=False)

    click.echo(f"Indexed: {ok}, Failed: {failed}")


@main.group()
def rule():
    """Generate and test Semgrep rules from lessons."""
    pass


@rule.command("generate")
@click.argument("lesson_id", type=int)
@click.option(
    "--rules-dir",
    type=click.Path(),
    default=None,
    help="Directory to write rules (default: ~/.local/share/lessons-db/rules/)",
)
@click.option(
    "--severity", default="WARNING", type=click.Choice(["WARNING", "ERROR", "INFO"]), help="Semgrep rule severity."
)
@click.pass_context
def rule_generate(ctx, lesson_id, rules_dir, severity):
    """Generate a Semgrep rule YAML file for a lesson."""
    from lessons_db.rulegen import generate_rule, slug_from_title

    conn = ctx.obj["conn"]
    lesson = conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
    if not lesson:
        logger.error("rule generate: lesson #%d not found", lesson_id)
        click.echo(f"Lesson #{lesson_id} not found.", err=True)
        ctx.exit(1)
        return

    patterns = conn.execute("SELECT * FROM detection_patterns WHERE lesson_id = ?", (lesson_id,)).fetchall()
    if not patterns:
        click.echo(f"No detection patterns for lesson #{lesson_id}. Add patterns via detection_patterns table first.")
        return

    out_dir = Path(rules_dir) if rules_dir else RULES_DIR
    language = patterns[0]["language"] or "any"
    lang_dir = out_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    slug = slug_from_title(lesson["title"])
    rule_file = lang_dir / f"{slug}-{lesson_id:03d}.yaml"

    rule_yaml = generate_rule(dict(lesson), [dict(p) for p in patterns], severity=severity)
    rule_file.write_text(rule_yaml, encoding="utf-8")
    click.echo(f"Generated: {rule_file}")


@rule.command("test")
@click.option(
    "--rules-dir",
    type=click.Path(),
    default=None,
    help="Directory containing rules (default: ~/.local/share/lessons-db/rules/)",
)
@click.pass_context
def rule_test(ctx, rules_dir):
    """Run semgrep --test against all generated rules."""
    import shutil
    import subprocess

    out_dir = Path(rules_dir) if rules_dir else RULES_DIR
    if not out_dir.exists() or not any(out_dir.rglob("*.yaml")):
        click.echo("No rules found. Run: lessons-db rule generate <id>")
        return

    semgrep = shutil.which("semgrep")
    if not semgrep:
        click.echo("semgrep not found on PATH.")
        return

    result = subprocess.run(
        [semgrep, "--test", str(out_dir)],
        capture_output=True,
        text=True,
    )
    click.echo(result.stdout or result.stderr)
    if result.returncode == 0:
        click.echo("All rules passed.")
    else:
        click.echo(f"Test failures (exit code {result.returncode}).")
        ctx.exit(result.returncode)


@main.command()
@click.option("--files", "-f", multiple=True, required=True, help="Files to check")
@click.option("--scope", "-s", default=None, help="Scope filter")
@click.option("--json", "json_output", is_flag=True, help="JSON output for script consumption")
@click.pass_context
def check(ctx, files, scope, json_output):
    """Check files against lesson detection patterns."""
    import json as json_mod

    from .check import check_files

    conn = ctx.obj["conn"]
    lance_dir = ctx.obj.get("lance_dir")

    violations = check_files(conn, lance_dir, list(files), scope=scope)

    if json_output:
        click.echo(json_mod.dumps(violations))
    else:
        for v in violations:
            click.echo(f"{v['file_path']}:{v['line_number']}: [lesson-{v['lesson_id']}] {v['one_liner']}")

    if violations:
        ctx.exit(1)


@main.command()
@click.option(
    "--rules-dir", type=click.Path(), default=None, help="Rules directory (default: ~/.local/share/lessons-db/rules/)"
)
@click.option(
    "--target", type=click.Path(), default=None, help="Target directory to scan (default: ~/Documents/projects/)"
)
@click.option("--baseline", default=None, help="Git commit hash for diff-aware scanning.")
@click.pass_context
def scan(ctx, rules_dir, target, baseline):
    """Run Semgrep scan against all lessons rules and record findings."""
    from lessons_db.db import insert_scan_finding
    from lessons_db.enforce import check_escalation
    from lessons_db.scan import run_scan

    rules = Path(rules_dir) if rules_dir else RULES_DIR
    if not rules.exists() or not any(rules.rglob("*.yaml")):
        click.echo("No rules found. Run: lessons-db rule generate <id>")
        return

    target_path = Path(target) if target else Path.home() / "Documents" / "projects"
    conn = ctx.obj["conn"]

    click.echo(f"Scanning {target_path} with rules from {rules}...")
    findings = run_scan(
        rules_dir=rules,
        target_dir=target_path,
        baseline_commit=baseline,
    )

    if not findings:
        click.echo("No findings.")
        return

    saved = 0
    for f in findings:
        rule_id = f.get("rule_id", "")
        click.echo(f"  [{rule_id}] {f.get('file_path')}:{f.get('line_number')}")

        # Parse lesson_id from rule_id suffix (format: lessons-db.<lang>.<slug>-NNN)
        lesson_id = None
        try:
            suffix = rule_id.rsplit("-", 1)[-1]
            lesson_id = int(suffix)
        except (ValueError, IndexError):
            pass

        if lesson_id is None:
            logger.warning("scan: could not parse lesson_id from rule_id %r, skipping DB insert", rule_id)
            click.echo(f"  [WARN] skipped {rule_id!r} — could not parse lesson_id", err=True)
            continue

        try:
            insert_scan_finding(
                conn,
                {
                    "lesson_id": lesson_id,
                    "rule_id": rule_id,
                    "file_path": f.get("file_path", ""),
                    "line_number": f.get("line_number"),
                    "snippet": f.get("message", ""),
                },
            )
            saved += 1
            action = check_escalation(conn, lesson_id)
            if action["recurrence_count"] >= 2:
                click.echo(
                    f"  [ESCALATED] lesson {lesson_id} → {action['level']}"
                    f" (recurrence #{action['recurrence_count']})"
                )
        except Exception as exc:
            logger.warning("scan: failed to insert finding %s: %s", rule_id, exc)

    click.echo(f"\nTotal findings: {len(findings)} found, {saved} saved to DB")


@main.command()
@click.argument("lesson_id", type=int)
@click.pass_context
def export(ctx, lesson_id):
    """Print a lesson as formatted markdown."""
    from lessons_db.db import get_lesson
    from lessons_db.export import format_lesson_markdown

    lesson = get_lesson(ctx.obj["conn"], lesson_id)
    if not lesson:
        click.echo(f"Lesson #{lesson_id} not found.")
        return
    click.echo(format_lesson_markdown(lesson))


@main.command()
@click.option(
    "--output", type=click.Path(), default=None, help="Output path (default: ~/Documents/docs/lessons/SUMMARY.md)"
)
@click.pass_context
def summary(ctx, output):
    """Auto-generate SUMMARY.md from DB records.

    Produces the Quick Reference table from live DB data.
    """
    conn = ctx.obj["conn"]

    rows = conn.execute("SELECT id, title, one_liner, cluster, tier, created_date FROM lessons ORDER BY id").fetchall()

    lines = [
        "# Lessons-Learned Summary",
        "",
        "> Auto-generated by `lessons-db summary`. Do not edit manually.",
        "",
        "## Quick Reference",
        "",
        "| # | Date | One-liner | Cluster | Tier |",
        "|---|------|-----------|---------|------|",
    ]
    for r in rows:
        date_short = (r["created_date"] or "")[-5:] or "?"
        cluster = r["cluster"] or "—"
        tier = r["tier"] or "observation"
        one_liner = (r["one_liner"] or r["title"] or "")[:80]
        lines.append(f"| {r['id']} | {date_short} | {one_liner} | {cluster} | {tier} |")

    lines.extend(["", f"**Total:** {len(rows)} lessons", ""])

    out_path = Path(output) if output else LESSONS_SOURCE_DIR / "SUMMARY.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        logger.error("summary: failed to write %s: %s", out_path, exc)
        click.echo(f"Error: could not write {out_path}: {exc}", err=True)
        raise SystemExit(1)
    click.echo(f"Written: {out_path} ({len(rows)} lessons)")


@main.group()
def capture():
    """Capture new lessons and manage draft queue."""
    pass


@capture.command("drafts")
@click.pass_context
def capture_drafts_cmd(ctx):
    """List pending auto-captured drafts awaiting approval."""
    from lessons_db.capture import list_drafts

    conn = ctx.obj["conn"]
    drafts = list_drafts(conn)
    if not drafts:
        click.echo("No pending drafts.")
        return
    for d in drafts:
        click.echo(f"[{d['id']}] {d['source']} | {d['created_date']}")
        try:
            data = json.loads(d["extracted_data"])
            click.echo(f"    {data.get('one_liner', '(no one-liner)')}")
        except Exception:
            click.echo("    (unparseable data)")


@capture.command("approve")
@click.argument("draft_id", type=int)
@click.pass_context
def capture_approve(ctx, draft_id):
    """Promote a pending draft to a live positive lesson."""
    from lessons_db.capture import promote_draft

    conn = ctx.obj["conn"]
    lesson_id = promote_draft(conn, draft_id)
    if lesson_id:
        click.echo(f"✓ Draft {draft_id} promoted → lesson #{lesson_id}")
    else:
        click.echo(f"✗ Draft {draft_id} not found or already processed.")


@capture.command("positive")
@click.pass_context
def capture_positive_cmd(ctx):
    """Interactively capture a positive knowledge entry (what worked well).

    Prompts for: what worked, why it worked, what category it belongs to.
    Scores the one-liner quality via Ollama before saving.
    Entry must score >= 3/5 to pass the quality gate.
    """
    from lessons_db.capture import capture_positive_manual

    conn = ctx.obj["conn"]

    one_liner = click.prompt("What worked well (one-liner, be specific)")
    why = click.prompt("Why did it work / what problem does it solve")
    category = click.prompt("Category", default="architecture-pattern")

    lesson_id = capture_positive_manual(conn, one_liner, why, category)
    if lesson_id is not None:
        click.echo(f"\n✓ Captured positive entry #{lesson_id}: {one_liner}")
    else:
        click.echo("\n✗ Capture aborted (quality gate failed — one-liner scored < 3/5).")


@capture.command("design-doc")
@click.argument("doc_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def capture_design_doc_cmd(ctx, doc_path):
    """Extract positive patterns from a design doc → draft queue.

    Sends the document to Ollama for extraction. Drafts require review:
    run 'lessons-db capture drafts' to inspect, then 'capture approve <id>'.
    """
    from lessons_db.capture import capture_from_design_doc

    conn = ctx.obj["conn"]
    drafts = capture_from_design_doc(doc_path, conn)
    if drafts:
        click.echo(f"Queued {len(drafts)} positive pattern draft(s) for review.")
        click.echo("Run: lessons-db capture drafts")
    else:
        click.echo("No positive patterns extracted.")


@capture.command("transcript")
@click.argument("transcript_file", type=click.Path(exists=True))
@click.option("--positive", is_flag=True, help="Extract positive patterns (what worked well) instead of failures.")
@click.pass_context
def capture_transcript(ctx, transcript_file, positive):
    """Analyze a saved session transcript for new lessons (writes to draft queue).

    Use --positive with a reasoning model (e.g. deepseek-r1) to extract effective
    approaches and good patterns instead of bugs and anti-patterns.
    """
    from lessons_db.capture import capture_from_transcript

    text = Path(transcript_file).read_text(encoding="utf-8", errors="replace")
    conn = ctx.obj["conn"]
    polarity = "positive" if positive else "negative"
    try:
        drafts = capture_from_transcript(text, conn, polarity=polarity)
    except Exception as exc:
        logger.error("capture transcript: %s", exc)
        click.echo(f"Capture failed: {exc}", err=True)
        ctx.exit(1)
        return
    if drafts:
        click.echo(f"Created {len(drafts)} draft(s). Review with: lessons-db capture drafts")
    else:
        click.echo("No lessons extracted.")


@capture.command("diff")
@click.argument("diff_file", type=click.Path(exists=True), required=False)
@click.pass_context
def capture_diff(ctx, diff_file):
    """Analyze a git diff for new lessons (writes to draft queue).

    If DIFF_FILE not provided, reads from stdin.
    """
    import sys

    from lessons_db.capture import capture_from_diff

    if diff_file:
        text = Path(diff_file).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    conn = ctx.obj["conn"]
    try:
        drafts = capture_from_diff(text, conn)
    except Exception as exc:
        logger.error("capture diff: %s", exc)
        click.echo(f"Capture failed: {exc}", err=True)
        ctx.exit(1)
        return
    if drafts:
        click.echo(f"Created {len(drafts)} draft(s). Review with: lessons-db capture drafts")
    else:
        click.echo("No lessons extracted.")


@capture.command("review")
@click.option("--dry-run", is_flag=True, help="Run filter only, skip Claude API call, print summary.")
@click.pass_context
def capture_review(ctx, dry_run):
    """Run automated triage: noise filter + Claude review → promote/dismiss drafts.

    Processes pending drafts. Writes decisions to ~/.local/share/lessons-db/triage-YYYY-MM-DD.jsonl.
    """
    import os

    from lessons_db.config import OPENAI_API_KEY, TRIAGE_LOG_DIR
    from lessons_db.review import claude_review_batch, execute_verdicts, filter_noise

    conn = ctx.obj["conn"]

    # Load pending drafts
    drafts = [
        dict(r)
        for r in conn.execute(
            "SELECT id, extracted_data, source FROM capture_drafts WHERE status = 'pending'"
        ).fetchall()
    ]

    if not drafts:
        click.echo("No pending drafts to review.")
        return

    # Load existing lesson one_liners for dedup
    existing = [r[0] for r in conn.execute("SELECT one_liner FROM lessons WHERE one_liner IS NOT NULL").fetchall()]

    # Phase 1: noise filter
    kept, dismissed_noise = filter_noise(drafts, existing_one_liners=existing)
    click.echo(f"Filter: {len(drafts)} drafts → {len(kept)} kept, {len(dismissed_noise)} noise-dismissed")

    if dry_run:
        click.echo("[dry-run] Skipping Claude review. Kept drafts:")
        for d in kept[:10]:
            data = json.loads(d.get("extracted_data") or "{}")
            click.echo(f"  [{d['id']}] {data.get('one_liner', '')[:80]}")
        return

    # Mark noise-dismissed drafts
    for d in dismissed_noise:
        conn.execute("UPDATE capture_drafts SET status='dismissed' WHERE id=?", [d["id"]])
    conn.commit()

    if not kept:
        click.echo("All drafts dismissed by noise filter.")
        return

    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        click.echo("Error: OPENAI_API_KEY not set. Export it or add to ~/.env.", err=True)
        ctx.exit(1)
        return

    # Phase 2: OpenAI review
    click.echo(f"Sending {len(kept)} drafts to OpenAI for review...")
    verdicts = claude_review_batch(kept, existing_titles=existing, api_key=api_key)

    # Phase 3: execute
    summary = execute_verdicts(conn, verdicts, log_dir=TRIAGE_LOG_DIR)
    click.echo(
        f"Done: {summary['promoted']} promoted, {summary['dismissed']} dismissed"
        + (f", {summary['errors']} errors" if summary.get("errors") else "")
        + "."
    )
    import datetime

    click.echo(f"Log: {TRIAGE_LOG_DIR}/triage-{datetime.date.today().isoformat()}.jsonl")


@capture.command("triage")
@click.option("--review-log", is_flag=True, help="Show triage decisions from the log.")
@click.option("--date", "log_date", default=None, help="Date to show log for (YYYY-MM-DD). Defaults to today.")
@click.option("--override", "override_id", type=int, default=None, help="Re-promote a dismissed draft by ID.")
@click.pass_context
def capture_triage(ctx, review_log, log_date, override_id):
    """Audit triage decisions or override a specific dismissed draft."""
    from lessons_db.capture import promote_draft
    from lessons_db.config import TRIAGE_LOG_DIR

    conn = ctx.obj["conn"]

    if override_id:
        # Re-promote: only reset dismissed drafts — guard against duplicating approved ones
        cursor = conn.execute(
            "UPDATE capture_drafts SET status='pending' WHERE id=? AND status='dismissed'",
            [override_id],
        )
        conn.commit()
        if cursor.rowcount == 0:
            # Check whether it exists at all vs already approved
            row = conn.execute("SELECT status FROM capture_drafts WHERE id=?", [override_id]).fetchone()
            if row is None:
                click.echo(f"Draft {override_id} not found.")
            else:
                click.echo(
                    f"Draft {override_id} has status '{row['status']}' — only dismissed drafts can be overridden."
                )
            return
        lesson_id = promote_draft(conn, override_id)
        if lesson_id:
            click.echo(f"Draft {override_id} re-promoted → lesson #{lesson_id}")
        else:
            click.echo(f"Draft {override_id}: promote failed unexpectedly.")
        return

    if review_log:
        import datetime

        target_date = log_date or datetime.date.today().isoformat()
        log_path = TRIAGE_LOG_DIR / f"triage-{target_date}.jsonl"
        if not log_path.exists():
            click.echo(f"No triage log for {target_date}.")
            return

        promoted = []
        dismissed = []
        errors = []
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                click.echo(f"  [warning] Skipping malformed line: {line[:60]!r}", err=True)
                continue
            if entry["verdict"] == "PROMOTE":
                promoted.append(entry)
            elif entry["verdict"] == "PROMOTE_FAILED":
                errors.append(entry)
            else:
                dismissed.append(entry)

        click.echo(f"\n=== Triage log: {target_date} ===")
        click.echo(f"Promoted: {len(promoted)} | Dismissed: {len(dismissed)} | Errors: {len(errors)}\n")

        if promoted:
            click.echo("PROMOTED:")
            for e in promoted:
                click.echo(f"  [draft {e['draft_id']} → lesson {e['lesson_id']}] {e.get('one_liner', '')}")
                click.echo(f"    Reason: {e['reason']}")

        if errors:
            click.echo("\nPROMOTE FAILURES:")
            for e in errors:
                click.echo(f"  [draft {e['draft_id']}] {e['reason']}")
            click.echo("  To re-promote: lessons-db capture triage --override <draft_id>")

        if dismissed:
            click.echo(f"\nDISMISSED (first 20 of {len(dismissed)}):")
            for e in dismissed[:20]:
                click.echo(f"  [draft {e['draft_id']}] {e.get('one_liner', '') or '(empty)'}")
                click.echo(f"    Reason: {e['reason']}")
            if len(dismissed) > 20:
                click.echo(f"  ... and {len(dismissed) - 20} more")

        click.echo("\nTo override a dismissal: lessons-db capture triage --override <draft_id>")
        return

    click.echo("Usage: lessons-db capture triage [--review-log [--date DATE] | --override ID]")


@main.group()
def cluster():
    """Adaptive cluster discovery and management."""
    pass


@cluster.command("show")
@click.pass_context
def cluster_show(ctx):
    """Show current cluster assignments for all lessons."""
    conn = ctx.obj["conn"]
    rows = conn.execute("SELECT cluster, COUNT(*) as n FROM lessons GROUP BY cluster ORDER BY n DESC").fetchall()
    for r in rows:
        label = r["cluster"] or "(unassigned)"
        click.echo(f"  {label}: {r['n']} lessons")


@cluster.command("history")
@click.pass_context
def cluster_history(ctx):
    """Show history of past clustering runs."""
    from lessons_db.cluster import get_cluster_history

    runs = get_cluster_history(ctx.obj["conn"])
    if not runs:
        click.echo("No clustering runs yet. Run: lessons-db cluster discover")
        return
    for run in runs:
        click.echo(f"[{run['run_date']}] {run['proposal_count']} proposals, {run['confirmed_count']} confirmed")


@cluster.command("discover")
@click.option("--min-size", default=5, type=int, help="Minimum cluster size for HDBSCAN.")
@click.pass_context
def cluster_discover(ctx, min_size):
    """Run HDBSCAN on embeddings and propose new cluster assignments."""
    from lessons_db.cluster import apply_cluster_proposals, discover_clusters

    conn = ctx.obj["conn"]
    click.echo("Running HDBSCAN clustering...")
    try:
        proposals = discover_clusters(conn, min_cluster_size=min_size)
    except RuntimeError as e:
        click.echo(str(e))
        return
    if not proposals:
        click.echo("Not enough data to cluster (need at least min_size * 2 entries with embeddings).")
        return
    confirmed = {}
    for p in proposals:
        seed_info = f" (overlaps seed {p['overlaps_seed']})" if p["overlaps_seed"] else ""
        click.echo(f"\nCluster {p['cluster_id']}: {len(p['lesson_ids'])} lessons{seed_info}")
        click.echo(f"  Suggested name: {p['suggested_name']}")
        click.echo(f"  Key terms: {', '.join(p['representative_terms'])}")
        name = click.prompt(
            "  Accept name? (Enter to accept, or type a new name, or 's' to skip)", default=p["suggested_name"]
        )
        if name.lower() != "s":
            confirmed[p["cluster_id"]] = name
    if confirmed:
        count = apply_cluster_proposals(conn, proposals, confirmed)
        click.echo(f"\n✓ Updated {count} lesson cluster assignments.")
    else:
        click.echo("No clusters confirmed.")


@main.group()
def learn():
    """Learning pipeline: record surfacing events and view statistics."""
    pass


@learn.command("record")
@click.option("--lesson-id", required=True, type=int)
@click.option(
    "--hook", "hook_point", required=True, type=click.Choice(["read", "edit", "plan", "bash", "session_start"])
)
@click.option("--context", "hook_context", default="", help="File path, query, or error text.")
@click.pass_context
def learn_record(click_ctx, lesson_id, hook_point, hook_context):
    """Record that a lesson was surfaced at a hook point."""
    from lessons_db.learn import record_surfacing

    conn = click_ctx.obj["conn"]
    event_id = record_surfacing(conn, lesson_id, hook_point, hook_context)
    click.echo(f"Recorded surfacing event {event_id}")


@main.group()
def stats():
    """Surfacing and efficiency statistics."""
    pass


@stats.command("surfacing")
@click.pass_context
def stats_surfacing(ctx):
    """Show outcome rates and surfacing event counts."""
    from lessons_db.learn import surfacing_stats

    s = surfacing_stats(ctx.obj["conn"])
    click.echo(f"Total surfacing events : {s['total_surfacing_events']}")
    click.echo(f"  Heeded               : {s['heeded']}")
    click.echo(f"  Dismissed            : {s['dismissed']}")
    click.echo(f"  Unknown              : {s['unknown']}")
    if s["heed_rate"] is not None:
        click.echo(f"  Heed rate            : {s['heed_rate']:.0%}")
    click.echo(f"Avg per session        : {s['avg_per_session']}")


@main.group()
def template():
    """View and apply templates from proven positive patterns."""
    pass


@template.command("list")
@click.pass_context
def template_list(ctx):
    """List all generated templates."""
    from lessons_db.promote import list_templates

    templates = list_templates(ctx.obj["conn"])
    if not templates:
        click.echo("No templates yet. Positive entries reach 'proven' tier after 2 reuses.")
        return
    for t in templates:
        click.echo(f"[#{t['lesson_id']}] ({t['tier']}) {t['one_liner']}")
        click.echo(f"      type={t['template_type']} | created={t['created_date']}")


@template.command("show")
@click.argument("lesson_id", type=int)
@click.pass_context
def template_show(ctx, lesson_id):
    """Show template content for a lesson."""
    from lessons_db.promote import apply_template

    content = apply_template(ctx.obj["conn"], lesson_id)
    if content:
        click.echo(content)
    else:
        click.echo(f"No template for lesson #{lesson_id}. Entry must reach 'proven' tier first.")


@main.group()
def pattern():
    """Cross-project pattern detection and review."""


@pattern.command("scan")
@click.pass_context
def pattern_scan(ctx):
    """Run cross-project pattern scan (all 3 stages)."""
    conn = ctx.obj["conn"]
    lance_dir = str(ctx.obj["lance_dir"])

    since = get_scan_state(conn, "last_scan_timestamp") or "1970-01-01T00:00:00"
    click.echo(f"Scanning repos with commits since {since}...")

    repos = pattern_extract.list_active_repos(since)
    if not repos:
        click.echo("No repos with recent commits. Nothing to scan.")
    else:
        click.echo(f"Active repos: {[r.name for r in repos]}")
        patterns = pattern_extract.build_semgrep_patterns(conn)
        try:
            candidates = pattern_extract.extract_python_candidates(
                repos, patterns, conn
            ) + pattern_extract.extract_nonpython_candidates(repos, conn)
            click.echo(f"Found {len(candidates)} raw candidates.")

            auto_approved = 0
            queued = 0
            for cand in candidates:
                verified = pattern_verify.verify_candidate(cand, conn, lance_dir)
                if verified is None:
                    continue
                triage_result = pattern_triage.triage_candidate(verified, conn, lance_dir)
                if triage_result is not None:
                    auto_approved += 1
                else:
                    queued += 1

            click.echo(f"Done: {auto_approved} auto-captured, {queued} queued for review.")
        except Exception as exc:
            click.echo(f"Scan error: {exc}", err=True)

    # Always update scan timestamp
    from datetime import datetime

    set_scan_state(conn, "last_scan_timestamp", datetime.now().isoformat(timespec="seconds"))


@pattern.command("review")
@click.pass_context
def pattern_review(ctx):
    """Batch review pending cross-project pattern drafts."""
    conn = ctx.obj["conn"]
    lance_dir = str(ctx.obj["lance_dir"])

    rows = conn.execute("""
        SELECT id, raw_content, extracted_data, confidence
        FROM capture_drafts
        WHERE status = 'pending' AND detection_source = 'cross_project_scan'
        ORDER BY confidence DESC
    """).fetchall()

    if not rows:
        click.echo("No pending pattern drafts.")
        return

    for row in rows:
        click.echo("\n" + "─" * 60)
        conf = row["confidence"]
        conf_str = f"{conf:.2f}" if conf is not None else "N/A"
        click.echo(f"Draft #{row['id']} | confidence: {conf_str}")
        click.echo(f"Snippet:\n{row['raw_content'][:300]}")
        if row["extracted_data"]:
            click.echo(f"Rationale: {row['extracted_data'][:200]}")

        action = click.prompt("[a]pprove / [r]eject / [s]kip", default="s").strip().lower()

        if action == "a":
            conn.execute("UPDATE capture_drafts SET status='approved' WHERE id=?", [row["id"]])
            conn.commit()
            click.echo("Approved.")
        elif action == "r":
            reason = click.prompt("Rejection reason (optional)", default="")
            pattern_triage.reject_draft(row["id"], conn, lance_dir=lance_dir, reason=reason or None)
            click.echo("Rejected and suppression vector stored.")

    click.echo("\nReview complete.")


@pattern.command("status")
@click.pass_context
def pattern_status(ctx):
    """Show pattern scan counts and threshold."""
    conn = ctx.obj["conn"]

    auto = conn.execute(
        "SELECT COUNT(*) FROM lessons WHERE source='cross_project_scan' AND polarity='positive'"
    ).fetchone()[0]
    pending = conn.execute(
        "SELECT COUNT(*) FROM capture_drafts WHERE detection_source='cross_project_scan' AND status='pending'"
    ).fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM suppression_vectors").fetchone()[0]
    threshold = get_scan_state(conn, "auto_approve_threshold") or "0.85"
    last_scan = get_scan_state(conn, "last_scan_timestamp") or "never"

    click.echo(
        f"{auto} auto-captured | {pending} pending review | "
        f"{rejected} suppressed | threshold: {threshold} | last scan: {last_scan}"
    )


@pattern.command("calibrate")
@click.option("--apply", is_flag=True, help="Propose and apply threshold adjustment.")
@click.pass_context
def pattern_calibrate(ctx, apply):
    """Show promotion stats by confidence band. Use --apply to adjust threshold."""
    conn = ctx.obj["conn"]

    bands = pattern_triage.calibration_bands(conn)
    if not bands:
        click.echo("No outcome data yet. Run the scanner and review drafts first.")
        if apply:
            click.echo("\nInsufficient data for threshold adjustment (need 20+ outcomes across bands).")
        return

    click.echo(f"{'Band':>6}  {'Total':>5}  {'Approved':>8}  {'Rate':>6}")
    for band, data in sorted(bands.items()):
        click.echo(f"{band:>6.1f}  {data['total']:>5}  {data['approved']:>8}  {data['promotion_rate']:>6.0%}")

    if apply:
        suggestion = pattern_triage.should_adjust_threshold(conn)
        if suggestion is None:
            click.echo("\nInsufficient data for threshold adjustment (need 20+ outcomes across bands).")
        else:
            click.echo(f"\n{suggestion['rationale']}")
            if click.confirm(
                f"Adjust threshold from {suggestion['current_threshold']:.2f} "
                f"to {suggestion['proposed_threshold']:.2f}?"
            ):
                set_scan_state(conn, "auto_approve_threshold", str(suggestion["proposed_threshold"]))
                click.echo(f"Threshold updated to {suggestion['proposed_threshold']:.2f}.")


@main.command()
@click.option("--tail", "-n", default=50, type=int, help="Number of lines from end to show.")
@click.option(
    "--level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Filter by log level."
)
def logs(tail, level):
    """Show recent log entries from ~/.local/share/lessons-db/lessons-db.log."""
    from lessons_db.logging_config import LOG_FILE

    if not LOG_FILE.exists():
        click.echo("No log file yet. Run some commands first.")
        return
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    if level:
        lines = [l for l in lines if f" {level} " in l]
    for line in lines[-tail:]:
        click.echo(line)
