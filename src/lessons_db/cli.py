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
from lessons_db.learn import VALID_OUTCOMES
from lessons_db.migrate import import_lesson_file, parse_lesson_file
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


@main.group("import")
@click.pass_context
def import_group(ctx):
    """Import lesson files or external rule sets into the database."""


@import_group.command("file")
@click.argument("path", type=click.Path(exists=True))
@click.pass_context
def import_cmd(ctx, path):
    """Import YAML-frontmatter lesson file(s) into the database.

    PATH can be a single .md file or a directory. When a directory is given,
    all files matching [0-9][0-9][0-9][0-9]-*.md are imported.

    Duplicates (matched by markdown_path or title) are skipped gracefully.
    """
    conn = ctx.obj["conn"]
    target = Path(path)

    if target.is_file():
        candidates = [target]
    elif target.is_dir():
        candidates = sorted(target.rglob("[0-9][0-9][0-9][0-9]-*.md"))
    else:
        click.echo(f"import: {path!r} is neither a file nor a directory", err=True)
        ctx.exit(1)
        return

    if not candidates:
        click.echo("import: no lesson files found")
        return

    imported = 0
    skipped = 0
    errors = 0

    for f in candidates:
        try:
            result = import_lesson_file(conn, f)
            if result is None:
                skipped += 1
                click.echo(f"  skipped (duplicate): {f.name}")
            else:
                imported += 1
                click.echo(f"  imported: {f.name} → DB id {result}")
        except ValueError as exc:
            # Non-YAML-frontmatter file — skip with message
            logger.debug("import: skipping %s — %s", f.name, exc)
            skipped += 1
            click.echo(f"  skipped (no frontmatter): {f.name}")
        except Exception as exc:
            logger.error("import: failed for %s: %s", f.name, exc)
            errors += 1
            click.echo(f"  error: {f.name} — {exc}", err=True)

    click.echo(f"\nImported: {imported}, Skipped: {skipped}, Errors: {errors}")
    if errors:
        ctx.exit(1)


@import_group.command("semgrep")
@click.option("--delta", is_flag=True, help="Only import new/changed rules.")
@click.pass_context
def import_semgrep(ctx, delta):
    """Import Semgrep registry Python rules as lesson stubs."""
    from lessons_db.semgrep_import import run_delta_import

    conn = ctx.obj["conn"]
    result = run_delta_import(conn, delta_only=delta)
    click.echo(
        f"Semgrep import: {result['imported']} imported, " f"{result['skipped']} skipped, {result['errors']} errors"
    )


@main.command()
@click.option("--seed-only", is_flag=True, help="Only backfill cluster_seed, skip embedding generation.")
@click.option("--reindex-all", is_flag=True, help="Re-embed all lessons, even those already in LanceDB.")
@click.pass_context
def index(ctx, seed_only, reindex_all):
    """Backfill cluster_seed and generate LanceDB embeddings for unindexed lessons.

    By default only indexes lessons not yet present in LanceDB (incremental).
    Use --reindex-all to rebuild embeddings for every lesson.
    cluster_seed: copies cluster → cluster_seed for A-F historical labels.
    Embeddings: calls Ollama nomic-embed-text for each lesson's title + one_liner + description + category + false_assumption.
    """
    from lessons_db.vectors import TABLE_NAME, init_lance, upsert_lesson

    conn = ctx.obj["conn"]

    # Step 1: backfill cluster_seed from cluster for lessons that have cluster but no seed
    updated = conn.execute(
        "UPDATE lessons SET cluster_seed = cluster WHERE cluster IS NOT NULL AND cluster != '' AND cluster_seed IS NULL"
    ).rowcount
    conn.commit()
    click.echo(f"cluster_seed backfill: {updated} rows updated")

    if seed_only:
        return

    # Step 2: incremental embedding — skip lessons already in LanceDB
    LANCE_DIR.mkdir(parents=True, exist_ok=True)
    lance_db = init_lance(str(LANCE_DIR))

    # Determine which lesson_ids are already indexed
    already_indexed: set[int] = set()
    if not reindex_all and TABLE_NAME in lance_db.list_tables().tables:
        table = lance_db.open_table(TABLE_NAME)
        already_indexed = {int(v) for v in table.to_arrow()["lesson_id"].to_pylist()}

    rows = conn.execute(
        "SELECT id, title, one_liner, keywords, description, category, false_assumption,"
        " cluster, tier, scope, enforcement, recurrence_count FROM lessons"
    ).fetchall()

    to_index = [r for r in rows if r["id"] not in already_indexed]
    skipped = len(rows) - len(to_index)

    if skipped:
        click.echo(f"Skipping {skipped} already-indexed lessons (use --reindex-all to force).")

    ok = 0
    failed = 0
    for row in to_index:
        title = row["title"] or ""
        one_liner = row["one_liner"] or ""
        keywords = row["keywords"] or ""
        description = row["description"] or ""
        category = row["category"] or ""
        false_assumption = row["false_assumption"] or ""

        # Build rich embedding text: title + one_liner + description (truncated) + category + false_assumption
        parts = [f"{title}. {one_liner}"]
        if description:
            parts.append(description[:500])
        if category:
            parts.append(f"Category: {category}")
        if false_assumption:
            parts.append(f"False assumption: {false_assumption}")
        if keywords:
            parts.append(f"Keywords: {keywords}")
        text = " ".join(parts)

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
            click.echo(f"  {ok + failed}/{len(to_index)} indexed...", err=False)

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


@main.group()
@click.pass_context
def scan(ctx):
    """Scan repositories for rule violations, security issues, and anti-patterns."""


@scan.command("run")
@click.option(
    "--rules-dir", type=click.Path(), default=None, help="Rules directory (default: ~/.local/share/lessons-db/rules/)"
)
@click.option(
    "--target", type=click.Path(), default=None, help="Target directory to scan (default: ~/Documents/projects/)"
)
@click.option("--baseline", default=None, help="Git commit hash for diff-aware scanning.")
@click.option(
    "--populate-fixes/--no-populate-fixes", default=True, help="Auto-populate fix queue after scan (default: on)."
)
@click.pass_context
def scan_run(ctx, rules_dir, target, baseline, populate_fixes):
    """Run Semgrep scan against all lessons rules and record findings."""
    from lessons_db.db import insert_scan_finding
    from lessons_db.prevention import assess_and_enforce, populate_fix_queue
    from lessons_db.scan import run_scan

    rules = Path(rules_dir) if rules_dir else RULES_DIR
    if not rules.exists() or not any(rules.rglob("*.yaml")):
        click.echo("No rules found. Run: lessons-db prevent bulk-generate")
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
    blocked = 0
    escalated = 0

    for f in findings:
        rule_id = f.get("rule_id", "")
        file_path = f.get("file_path", "")

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
                    "file_path": file_path,
                    "line_number": f.get("line_number"),
                    "snippet": f.get("message", ""),
                },
            )
            saved += 1

            # Full enforcement cycle: log recurrence event, velocity check, escalate if needed
            decision = assess_and_enforce(
                conn,
                lesson_id,
                hook_point="scan",
                trigger_type="semgrep",
                file_path=file_path,
                rules_dir=rules,
            )

            status = f"  [{rule_id}] {file_path}:{f.get('line_number')}"
            if decision.escalated:
                status += f"  → ESCALATED to {decision.enforcement_level}"
                escalated += 1
            if decision.should_block:
                status += "  [BLOCKING]"
                blocked += 1
            click.echo(status)

        except Exception as exc:
            logger.warning("scan: failed to insert finding %s: %s", rule_id, exc)

    click.echo(f"\nTotal findings: {len(findings)} found, {saved} saved" f" | escalated={escalated} blocking={blocked}")

    if populate_fixes and saved > 0:
        fix_result = populate_fix_queue(conn)
        click.echo(f"Fix queue: +{fix_result['added']} added" f" ({fix_result['skipped_duplicate']} already queued)")


@scan.command("security")
@click.option("--target", type=click.Path(), default=None, help="Target directory (default: ~/Documents/projects/).")
@click.pass_context
def scan_security(ctx, target):
    """Run Ruff S-rules + pip-audit on target directory."""
    from pathlib import Path as _Path

    from lessons_db.security_scanner import run_full_security_scan

    conn = ctx.obj["conn"]
    target_path = _Path(target) if target else None
    summary = run_full_security_scan(conn, target_path)
    click.echo("Security scan complete:")
    for k, v in summary.items():
        click.echo(f"  {k}: {v}")


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


@capture.command("detect-wins")
@click.option("--lookback", type=int, default=4, help="Hours to look back for surfacing events.")
@click.pass_context
def capture_detect_wins_cmd(ctx, lookback):
    """Detect positive session wins from surfacing event patterns.

    Checks for heeded lessons, clean sessions (no anti-pattern hits),
    and positive pattern reuse. Routes detected wins through the draft
    capture pipeline for sustain-oriented knowledge retention.
    """
    from lessons_db.capture import detect_wins

    conn = ctx.obj["conn"]
    wins = detect_wins(conn, lookback_hours=lookback)

    if not wins:
        click.echo("No wins detected this session.")
        return

    click.echo(f"Detected {len(wins)} win(s):")
    for win in wins:
        click.echo(f"  [{win['win_type']}] {win['detail']}")

    # Route wins through draft capture pipeline
    for win in wins:
        try:
            conn.execute(
                "INSERT INTO capture_drafts "
                "(raw_content, extracted_data, status, created_date, source) "
                "VALUES (?, ?, 'pending', date('now'), 'auto_win_detection')",
                [
                    win["detail"],
                    json.dumps(
                        {
                            "one_liner": win["detail"],
                            "win_type": win["win_type"],
                            "lesson_ids": win.get("lesson_ids", []),
                        }
                    ),
                ],
            )
        except Exception as exc:
            logger.warning("detect-wins: draft insert failed: %s", exc)
    conn.commit()
    click.echo(f"Queued {len(wins)} win draft(s). Review with: lessons-db capture drafts")


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
    "--hook",
    "hook_point",
    required=True,
    type=click.Choice(
        [
            "read",
            "edit",
            "plan",
            "bash",
            "session_start",
            "session_start_fsrs",
            "session_start_exception",
            "commit",
            "stop",
        ]
    ),
)
@click.option("--context", "hook_context", default="", help="File path, query, or error text.")
@click.option(
    "--outcome",
    default=None,
    type=click.Choice(list(VALID_OUTCOMES)),
    help="Outcome to record immediately (optional, default: unknown).",
)
@click.pass_context
def learn_record(click_ctx, lesson_id, hook_point, hook_context, outcome):
    """Record that a lesson was surfaced at a hook point."""
    from lessons_db.learn import record_outcome, record_surfacing

    conn = click_ctx.obj["conn"]
    event_id = record_surfacing(conn, lesson_id, hook_point, hook_context)
    if outcome is not None:
        record_outcome(conn, event_id, outcome)
    click.echo(f"Recorded surfacing event {event_id}")


@learn.command("evaluate-commit")
@click.option("--hours", default=24, type=int, help="Lookback window in hours (default: 24).")
@click.option("--dry-run", is_flag=True, help="Preview outcomes without updating the database.")
@click.option(
    "--diff-text",
    default=None,
    help="Provide diff text directly (bypasses git). For testing or piped input.",
)
@click.pass_context
def learn_evaluate_commit(click_ctx, hours, dry_run, diff_text):
    """Evaluate whether recently-surfaced lessons were heeded or dismissed.

    Reads surfacing events with outcome='unknown' from the last N hours,
    gets the latest git diff (HEAD~1..HEAD), and checks if each lesson's
    anti-pattern appears in the diff.

    If anti-pattern present: outcome = 'dismissed' (recurrence).
    If anti-pattern absent: outcome = 'heeded'.
    """
    import subprocess

    from lessons_db.learn import evaluate_commit

    conn = click_ctx.obj["conn"]

    if diff_text is None:
        # Get the latest commit diff
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD~1..HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                click.echo(f"git diff failed: {result.stderr.strip()}", err=True)
                click_ctx.exit(1)
                return
            diff_text = result.stdout
        except FileNotFoundError:
            click.echo("git not found on PATH.", err=True)
            click_ctx.exit(1)
            return
        except subprocess.TimeoutExpired:
            click.echo("git diff timed out.", err=True)
            click_ctx.exit(1)
            return

    if not diff_text.strip():
        click.echo("Empty diff — nothing to evaluate.")
        return

    results = evaluate_commit(conn, diff_text, hours=hours, dry_run=dry_run)

    if not results:
        click.echo("No unknown surfacing events with detection patterns in window.")
        return

    prefix = "[DRY RUN] " if dry_run else ""
    click.echo(f"{prefix}Evaluated {len(results)} surfacing event(s):")
    for r in results:
        marker = "X" if r["outcome"] == "dismissed" else "."
        click.echo(
            f"  [{marker}] event={r['event_id']} lesson=#{r['lesson_id']} "
            f"outcome={r['outcome']} ({r['pattern_source']})"
        )

    heeded = sum(1 for r in results if r["outcome"] == "heeded")
    dismissed = sum(1 for r in results if r["outcome"] == "dismissed")
    click.echo(f"\nSummary: {heeded} heeded, {dismissed} dismissed")


@learn.command("dismiss")
@click.argument("lesson_id", type=int)
@click.pass_context
def learn_dismiss(click_ctx, lesson_id):
    """Mark most recent surfacing of LESSON_ID as a false positive."""
    from lessons_db.learn import dismiss_latest

    conn = click_ctx.obj["conn"]
    updated = dismiss_latest(conn, lesson_id)
    if updated:
        click.echo(f"Marked latest surfacing of lesson #{lesson_id} as false_positive.")
    else:
        click.echo(f"No surfacing events found for lesson #{lesson_id}.")


@learn.command("list")
@click.option(
    "--since",
    default="24h",
    help="Time window: '1h', '6h', '24h', '7d'. Default: 24h.",
)
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "ids"]),
    help="Output format. 'ids' prints one lesson_id per line for scripting.",
)
@click.option(
    "--outcome",
    "outcome_filter",
    default=None,
    type=click.Choice(["unknown", "heeded", "dismissed", "false_positive", "recurrence"]),
    help="Filter by outcome value. Omit to return all outcomes.",
)
@click.pass_context
def learn_list(click_ctx, since, output_format, outcome_filter):
    """List recent surfacing events."""
    import re
    from datetime import UTC, datetime, timedelta

    conn = click_ctx.obj["conn"]

    # Parse window: e.g. "2h" -> 7200, "7d" -> 604800
    match = re.fullmatch(r"(\d+)([hd])", since)
    if not match:
        raise click.BadParameter(
            f"Invalid window '{since}'. Use e.g. '1h', '24h', '7d'.",
            param_hint="--since",
        )
    n, unit = int(match.group(1)), match.group(2)
    seconds = n * 3600 if unit == "h" else n * 86400
    cutoff = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()

    where_parts = ["se.timestamp >= ?"]
    params: list = [cutoff]
    if outcome_filter is not None:
        where_parts.append("se.outcome = ?")
        params.append(outcome_filter)

    rows = conn.execute(
        "SELECT se.id, se.lesson_id, l.title, se.hook_point, se.outcome, se.timestamp "
        "FROM surfacing_events se JOIN lessons l ON l.id = se.lesson_id "
        f"WHERE {' AND '.join(where_parts)} ORDER BY se.timestamp DESC",
        params,
    ).fetchall()

    if not rows:
        click.echo("No surfacing events in window.")
        return

    if output_format == "ids":
        seen = set()
        for row in rows:
            if row["lesson_id"] not in seen:
                click.echo(row["lesson_id"])
                seen.add(row["lesson_id"])
    else:
        click.echo(f"{'ID':>6}  {'LID':>5}  {'Hook':<14}  {'Outcome':<14}  Title")
        click.echo("-" * 72)
        for row in rows:
            title = row["title"][:35]
            click.echo(
                f"{row['id']:>6}  {row['lesson_id']:>5}  {row['hook_point']:<14}  " f"{row['outcome']:<14}  {title}"
            )


@learn.command("find-exceptions")
@click.option("--lookback", default=5, type=int, help="Number of recent sessions to check (default: 5).")
@click.pass_context
def learn_find_exceptions(click_ctx, lookback):
    """Find anti-patterns absent from recent sessions (SFBT exception-finding).

    Identifies negative lessons that previously recurred but have been absent
    from recent sessions — evidence of internalized learning.
    """
    from lessons_db.learn import find_exceptions

    conn = click_ctx.obj["conn"]
    exceptions = find_exceptions(conn, lookback_sessions=lookback)

    if not exceptions:
        click.echo("No exceptions found — no previously-dismissed anti-patterns absent from recent sessions.")
        return

    click.echo(f"Found {len(exceptions)} internalized pattern(s):")
    for exc in exceptions:
        click.echo(
            f"  {exc['absent_sessions']}-session streak: zero {exc['category']} "
            f"issues — [#{exc['lesson_id']}] {exc['title']}"
        )


@main.group()
def reuse():
    """Positive reuse tracking — record pattern reuse to advance promotion tier."""
    pass


@reuse.command("record")
@click.argument("lesson_id", type=int)
@click.pass_context
def reuse_record(ctx, lesson_id):
    """Record a positive pattern reuse for LESSON_ID.

    Increments reuse_count and promotes the lesson through tiers:
    noticed -> tested (1) -> proven (2, template generated) -> standard (3).

    Also records a surfacing event with outcome='reused' for the learning pipeline.
    """
    from lessons_db.promote import record_reuse

    conn = ctx.obj["conn"]
    try:
        new_tier = record_reuse(conn, lesson_id)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        ctx.exit(1)
        return

    # Record a surfacing event so the learning pipeline tracks positive reuse
    from lessons_db.learn import record_surfacing

    event_id = record_surfacing(conn, lesson_id, "edit", "positive_reuse")
    # Mark outcome as 'heeded' — reuse of a positive pattern is always a good outcome
    from lessons_db.learn import record_outcome

    record_outcome(conn, event_id, "heeded")

    lesson = conn.execute("SELECT one_liner FROM lessons WHERE id = ?", [lesson_id]).fetchone()
    one_liner = lesson["one_liner"] if lesson else ""
    click.echo(f"Recorded reuse for lesson #{lesson_id} — tier: {new_tier}")
    if one_liner:
        click.echo(f"  {one_liner}")


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


@stats.command("efficiency")
@click.pass_context
def stats_efficiency(ctx):
    """Show wasted surfacings and enforcement candidates.

    Wasted surfacings: lessons surfaced at least once but never heeded.
    Enforcement candidates: lessons with high recurrence and low heed rate (<30%).
    """
    conn = ctx.obj["conn"]

    # --- Wasted surfacings: surfaced > 0 AND heeded = 0 ---
    wasted_rows = conn.execute(
        """
        SELECT l.id, l.one_liner, l.title,
               COUNT(se.id) AS surfaced,
               SUM(CASE WHEN se.outcome = 'heeded' THEN 1 ELSE 0 END) AS heeded
        FROM lessons l
        JOIN surfacing_events se ON se.lesson_id = l.id
        GROUP BY l.id
        HAVING surfaced > 0 AND heeded = 0
        ORDER BY surfaced DESC
        """
    ).fetchall()

    # --- Enforcement candidates: high recurrence, low heed rate ---
    # Threshold: recurrence_count > 3 AND heed_rate < 0.3 (must have at least 1 surfacing)
    candidate_rows = conn.execute(
        """
        SELECT l.id, l.one_liner, l.title, l.recurrence_count,
               COUNT(se.id) AS surfaced,
               SUM(CASE WHEN se.outcome = 'heeded' THEN 1 ELSE 0 END) AS heeded
        FROM lessons l
        JOIN surfacing_events se ON se.lesson_id = l.id
        GROUP BY l.id
        HAVING surfaced > 0
           AND l.recurrence_count > 3
           AND (CAST(heeded AS REAL) / surfaced) < 0.3
        ORDER BY l.recurrence_count DESC
        """
    ).fetchall()

    # --- Average outcome rate across lessons with at least 1 surfacing ---
    avg_row = conn.execute(
        """
        SELECT AVG(CAST(heeded AS REAL) / surfaced) AS avg_outcome_rate
        FROM (
            SELECT l.id,
                   SUM(CASE WHEN se.outcome = 'heeded' THEN 1 ELSE 0 END) AS heeded,
                   COUNT(se.id) AS surfaced
            FROM lessons l
            JOIN surfacing_events se ON se.lesson_id = l.id
            GROUP BY l.id
            HAVING surfaced > 0
        )
        """
    ).fetchone()
    avg_outcome_rate = avg_row[0] if avg_row and avg_row[0] is not None else None

    # --- Output ---
    click.echo("Efficiency Report")
    click.echo("─" * 45)

    if wasted_rows:
        click.echo("\nWasted surfacings (surfaced, never heeded):")
        for r in wasted_rows:
            label = r["one_liner"] or r["title"] or f"lesson #{r['id']}"
            click.echo(f"  #{r['id']} {label[:50]:<50}  surfaced: {r['surfaced']}   heeded: 0")
    else:
        click.echo("\nWasted surfacings: none")

    if candidate_rows:
        click.echo("\nEnforcement candidates (high recurrence, low heed rate):")
        for r in candidate_rows:
            label = r["one_liner"] or r["title"] or f"lesson #{r['id']}"
            heed_rate = r["heeded"] / r["surfaced"] if r["surfaced"] > 0 else 0.0
            click.echo(
                f"  #{r['id']} {label[:50]:<50}  recurrence: {r['recurrence_count']}  " f"heed_rate: {heed_rate:.2f}"
            )
    else:
        click.echo("\nEnforcement candidates: none")

    if avg_outcome_rate is not None:
        click.echo(f"\nAverage outcome rate: {avg_outcome_rate:.2f}")
    else:
        click.echo("\nAverage outcome rate: N/A (no surfacing data)")


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


@main.command("kpi")
@click.pass_context
def kpi_dashboard(click_ctx):
    """Show learning KPI dashboard with profile, stability, streaks, and ZPD."""
    from datetime import UTC, datetime, timedelta

    from lessons_db.fsrs import get_fading_level

    conn = click_ctx.obj["conn"]

    def q(sql, params=None):
        row = conn.execute(sql, params or []).fetchone()
        return row[0] if row else 0

    total_lessons = q("SELECT COUNT(*) FROM lessons WHERE polarity='negative'")
    total_surfacings = q("SELECT COUNT(*) FROM surfacing_events")
    decided = q("SELECT COUNT(*) FROM surfacing_events WHERE outcome != 'unknown'")
    heeded = q("SELECT COUNT(*) FROM surfacing_events WHERE outcome = 'heeded'")
    recurrences = q("SELECT COUNT(*) FROM surfacing_events WHERE outcome = 'recurrence'")
    false_positives = q("SELECT COUNT(*) FROM surfacing_events WHERE outcome = 'false_positive'")
    heed_recur = heeded + recurrences
    cutoff_90d = (datetime.now(UTC) - timedelta(seconds=86400 * 90)).isoformat()
    dead_lessons = q(
        "SELECT COUNT(*) FROM lessons l "
        "WHERE l.polarity = 'negative' AND NOT EXISTS ("
        "  SELECT 1 FROM surfacing_events se "
        "  WHERE se.lesson_id = l.id AND se.timestamp >= ?"
        ")",
        [cutoff_90d],
    )
    growth_7d = q("SELECT COUNT(*) FROM lessons WHERE created_date >= date('now','-7 days')")

    actionable = decided - false_positives  # exclude noise from heed_rate denominator
    heed_rate = round(heeded / actionable * 100, 1) if actionable > 0 else None
    recurrence_rate = round(recurrences / heed_recur * 100, 1) if heed_recur > 0 else None
    fp_rate = round(false_positives / decided * 100, 1) if decided > 0 else None
    dead_pct = round(dead_lessons / total_lessons * 100, 1) if total_lessons > 0 else 0

    def fmt(val, target_ok, unit="%"):
        if val is None:
            return "  n/a    (no data yet)"
        ok = "+" if target_ok(val) else "-"
        return f"  {val}{unit}  {ok}"

    click.echo("")
    click.echo("=== Learning KPI Dashboard ===")
    click.echo("")
    click.echo(f"  Total lessons          : {total_lessons}")
    click.echo(f"  Total surfacings       : {total_surfacings}  ({decided} with outcome)")
    click.echo("")
    click.echo("  Outcome KPIs (need outcome data to populate):")
    click.echo(f"  Recurrence Rate        :{fmt(recurrence_rate, lambda v: v < 5)}")
    click.echo(f"  Heed Rate              :{fmt(heed_rate,        lambda v: v > 50)}")
    click.echo(f"  False Positive Rate    :{fmt(fp_rate,          lambda v: v < 15)}")
    click.echo("")
    click.echo("  System Health:")
    click.echo(f"  Dead Lessons (90d)     :  {dead_lessons} ({dead_pct}%)  " f"{'+'  if dead_pct < 10 else '-'}")
    click.echo(f"  DB Growth (7d)         :  +{growth_7d} lessons")
    click.echo("")

    # --- Section: Heeded Rate by Category ---
    click.echo("Heeded Rate by Category:")
    cat_rows = conn.execute(
        "SELECT COALESCE(NULLIF(l.cluster, ''), '(unclustered)') AS cluster_label, "
        "  SUM(CASE WHEN se.outcome = 'heeded' THEN 1 ELSE 0 END) AS heeded_count, "
        "  COUNT(*) AS total_count "
        "FROM surfacing_events se "
        "JOIN lessons l ON se.lesson_id = l.id "
        "WHERE se.outcome != 'unknown' "
        "GROUP BY cluster_label "
        "ORDER BY total_count DESC"
    ).fetchall()
    if cat_rows:
        for row in cat_rows:
            cluster = row["cluster_label"]
            h = row["heeded_count"]
            t = row["total_count"]
            pct = round(h / t * 100) if t > 0 else 0
            click.echo(f"  {cluster}: {pct}% ({h}/{t})")
    else:
        click.echo("  (no surfacing data)")
    click.echo("")

    # --- Section: Stability Distribution ---
    click.echo("Stability Distribution:")
    stability_rows = conn.execute("SELECT stability FROM lessons WHERE stability IS NOT NULL").fetchall()
    level_counts = {"full": 0, "brief": 0, "silent": 0, "enforced": 0}
    for row in stability_rows:
        level = get_fading_level(row["stability"])
        level_counts[level] += 1
    click.echo(
        f"  full: {level_counts['full']}  "
        f"brief: {level_counts['brief']}  "
        f"silent: {level_counts['silent']}  "
        f"enforced: {level_counts['enforced']}"
    )
    click.echo("")

    # --- Section: Positive/Negative Ratio ---
    click.echo("Positive/Negative Ratio:")
    pos_count = q("SELECT COUNT(*) FROM lessons WHERE polarity = 'positive'")
    neg_count = q("SELECT COUNT(*) FROM lessons WHERE polarity = 'negative'")
    click.echo(f"  positive: {pos_count}  negative: {neg_count}")
    if pos_count + neg_count > 0:
        ratio = round(pos_count / (pos_count + neg_count) * 100, 1)
        click.echo(f"  positive ratio: {ratio}%")
    click.echo("")

    # --- Section: Win Streaks ---
    click.echo("Win Streaks:")
    streak_rows = conn.execute(
        "SELECT category, current_streak, longest_streak "
        "FROM win_streaks "
        "ORDER BY longest_streak DESC, current_streak DESC "
        "LIMIT 10"
    ).fetchall()
    if streak_rows:
        for row in streak_rows:
            click.echo(f"  {row['category']}: " f"current={row['current_streak']} " f"longest={row['longest_streak']}")
    else:
        click.echo("  (no win streak data)")
    click.echo("")

    # --- Section: Learning Velocity ---
    click.echo("Learning Velocity (30d):")
    cutoff_30d = (datetime.now(UTC) - timedelta(days=30)).date().isoformat()
    velocity_rows = conn.execute(
        "SELECT id, title, stability, last_review_date "
        "FROM lessons "
        "WHERE last_review_date IS NOT NULL "
        "  AND last_review_date >= ? "
        "  AND stability IS NOT NULL "
        "ORDER BY last_review_date DESC",
        [cutoff_30d],
    ).fetchall()
    if velocity_rows:
        click.echo(f"  {len(velocity_rows)} lesson(s) reviewed in last 30 days:")
        for row in velocity_rows[:10]:
            level = get_fading_level(row["stability"])
            click.echo(f"    #{row['id']} {row['title'][:40]} -> {level} (S={row['stability']:.1f})")
        if len(velocity_rows) > 10:
            click.echo(f"    ... and {len(velocity_rows) - 10} more")
    else:
        click.echo("  (no reviews in last 30 days)")
    click.echo("")

    # --- Section: ZPD Identification (Vygotsky) ---
    click.echo("ZPD Identification (50-80% heeded rate):")
    zpd_rows = conn.execute(
        "SELECT l.id, l.title, l.cluster, "
        "  SUM(CASE WHEN se.outcome = 'heeded' THEN 1 ELSE 0 END) AS heeded_count, "
        "  COUNT(*) AS total_count "
        "FROM surfacing_events se "
        "JOIN lessons l ON se.lesson_id = l.id "
        "WHERE se.outcome != 'unknown' "
        "GROUP BY l.id "
        "HAVING total_count >= 2 "
        "ORDER BY total_count DESC"
    ).fetchall()
    zpd_found = []
    for row in zpd_rows:
        rate = row["heeded_count"] / row["total_count"]
        if 0.50 <= rate <= 0.80:
            zpd_found.append(row)
    if zpd_found:
        for row in zpd_found[:10]:
            h = row["heeded_count"]
            t = row["total_count"]
            pct = round(h / t * 100)
            cluster = row["cluster"] or ""
            label = f" [{cluster}]" if cluster else ""
            click.echo(f"  #{row['id']} {row['title'][:40]}{label}: {pct}% ({h}/{t})")
        if len(zpd_found) > 10:
            click.echo(f"  ... and {len(zpd_found) - 10} more")
    else:
        click.echo("  (no lessons in ZPD range, or insufficient surfacing data)")
    click.echo("")


@main.command("enrich")
@click.option("--id", "lesson_id", type=int, default=None, help="Enrich a single lesson by ID.")
@click.option("--batch", type=int, default=None, help="Process at most N un-enriched lessons.")
@click.option("--dry-run", is_flag=True, help="Generate output but do not write to DB.")
@click.option("--model", default=None, help="Override ANALYSIS_MODEL for this run.")
@click.option(
    "--ollama-url",
    default=None,
    help=(
        "Ollama base URL. Defaults to queue proxy (7683) for standalone runs. "
        "Pass http://127.0.0.1:11434 when running as a queue subprocess to avoid self-deadlock."
    ),
)
@click.pass_context
def enrich_cmd(ctx, lesson_id, batch, dry_run, model, ollama_url):
    """Backfill false_assumption, detection_pattern, invariant via Ollama.

    Standalone: routes each call through the queue proxy (serialized).
    As a queue job: pass --ollama-url http://127.0.0.1:11434 to call Ollama directly.
    """
    from lessons_db.config import ANALYSIS_MODEL, OLLAMA_QUEUE_URL
    from lessons_db.enrich import backfill_lessons, enrich_lesson

    conn = ctx.obj["conn"]
    resolved_model = model or ANALYSIS_MODEL
    resolved_url = ollama_url or OLLAMA_QUEUE_URL

    if dry_run:
        click.echo(f"[dry-run] model={resolved_model} url={resolved_url}")

    if lesson_id is not None:
        row = conn.execute(
            "SELECT id, title, one_liner, description FROM lessons WHERE id = ?", (lesson_id,)
        ).fetchone()
        if not row:
            click.echo(f"Lesson #{lesson_id} not found.", err=True)
            raise SystemExit(1)
        result = enrich_lesson(
            conn=conn,
            lesson_id=row["id"],
            title=row["title"] or "",
            description=row["description"] or "",
            one_liner=row["one_liner"] or "",
            model=resolved_model,
            ollama_url=resolved_url,
            dry_run=dry_run,
        )
        if result:
            suffix = " (dry-run, not written)" if dry_run else "→ written"
            click.echo(f"  enriched: #{lesson_id} {(row['title'] or '')[:60]} {suffix}")
            if dry_run:
                click.echo(f"    false_assumption  : {result['false_assumption']}")
                click.echo(f"    detection_pattern : {result['detection_pattern']}")
                click.echo(f"    invariant         : {result['invariant']}")
        else:
            click.echo(f"  error: #{lesson_id} — enrichment failed (check logs for details)", err=True)
        return

    # Batch / full backfill
    enriched, skipped, errors = backfill_lessons(
        conn=conn,
        model=resolved_model,
        ollama_url=resolved_url,
        batch=batch,
        dry_run=dry_run,
    )
    click.echo(f"Enriched: {enriched}, Skipped: {skipped}, Errors: {errors}")


@main.group()
@click.pass_context
def mine(ctx):
    """Mine external repositories for lesson patterns."""


@mine.command("github")
@click.option("--topic", multiple=True, help="GitHub topics to target.")
@click.option("--limit", default=500, help="Max commits per repo.")
@click.pass_context
def mine_github(ctx, topic, limit):
    """Mine GitHub repos for anti-patterns and positive novel methods."""
    from lessons_db.github_miner import MiningConfig, mine_repos_for_gaps

    conn = ctx.obj["conn"]
    lance_dir = ctx.obj["lance_dir"]
    default_topics = MiningConfig().topics
    config = MiningConfig(
        max_commits_per_repo=limit,
        topics=list(topic) if topic else default_topics,
    )
    stats = mine_repos_for_gaps(conn, lance_dir, config=config)
    click.echo("Mining complete:")
    for k, v in stats.items():
        click.echo(f"  {k}: {v}")


@main.group("calibrate")
@click.pass_context
def calibrate(ctx):
    """Calibrate the lesson extraction pipeline and view strength profiles."""


@calibrate.command("bugsInPy")
@click.option("--sample", default=50, help="Number of bugs to sample (default 50).")
@click.option("--cache-dir", default=None, help="Directory to cache the BugsInPy clone.")
@click.option(
    "--skip-extraction",
    is_flag=True,
    default=False,
    help="Skip Ollama extraction — measure size gate pass rate only.",
)
@click.pass_context
def calibrate_bugsInPy(ctx, sample, cache_dir, skip_extraction):
    """Calibrate pipeline against the BugsInPy dataset (493 confirmed Python bugs).

    Clones soarsmu/BugsInPy (cached after first run), samples N bugs, runs
    extraction and gate checks, and prints a pass-rate report.

    Pass rate >= 70%: pipeline is calibrated for live mining.
    Pass rate < 70%: tune gate thresholds before enabling live mining.
    """
    from pathlib import Path

    from lessons_db.bugsInPy_calibrator import DEFAULT_CACHE_DIR, calibrate_pipeline, format_report

    conn = ctx.obj["conn"]
    lance_dir = ctx.obj.get("lance_dir")
    resolved_cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    click.echo(f"Running BugsInPy calibration (sample={sample}, skip_extraction={skip_extraction}) …")
    report = calibrate_pipeline(
        conn=conn,
        lance_dir=lance_dir,
        sample_n=sample,
        cache_dir=resolved_cache,
        skip_extraction=skip_extraction,
    )
    click.echo(format_report(report))


@calibrate.command("profile")
@click.option(
    "--min-events",
    default=5,
    type=int,
    help="Minimum surfacing events per category to include (default: 5).",
)
@click.pass_context
def calibrate_profile(ctx, min_events):
    """Show calibration feedback: strength profile based on heeded/dismissed ratios per category.

    Groups surfacing events by lesson category to identify strengths (highest
    heeded rate) and growth areas (lowest heeded rate). Categories with fewer
    than --min-events events are excluded and listed separately.
    """
    conn = ctx.obj["conn"]

    # Query per-category heeded/total counts (exclude unknown outcomes)
    rows = conn.execute(
        "SELECT COALESCE(NULLIF(l.category, ''), l.cluster, 'uncategorized') AS cat, "
        "  SUM(CASE WHEN se.outcome = 'heeded' THEN 1 ELSE 0 END) AS heeded, "
        "  COUNT(*) AS total "
        "FROM surfacing_events se "
        "JOIN lessons l ON se.lesson_id = l.id "
        "WHERE se.outcome IN ('heeded', 'dismissed') "
        "GROUP BY cat "
        "ORDER BY total DESC"
    ).fetchall()

    if not rows:
        click.echo("No surfacing outcome data yet. More data needed to generate a strength profile.")
        click.echo("Record surfacing events with: lessons-db learn record")
        return

    qualified = []
    insufficient = []

    for row in rows:
        cat = row["cat"]
        total = row["total"]
        heeded = row["heeded"]
        if total >= min_events:
            rate = heeded / total
            qualified.append({"category": cat, "heeded": heeded, "total": total, "rate": rate})
        else:
            insufficient.append(cat)

    if not qualified:
        click.echo(f"No categories have {min_events}+ events yet. More data needed to generate a strength profile.")
        if insufficient:
            click.echo(f"\nCategories with insufficient data (<{min_events} events): {', '.join(sorted(insufficient))}")
        return

    # Sort by heeded rate descending for strengths, ascending for growth areas
    by_rate_desc = sorted(qualified, key=lambda x: x["rate"], reverse=True)
    by_rate_asc = sorted(qualified, key=lambda x: x["rate"])

    strengths = by_rate_desc[:3]
    growth_areas = by_rate_asc[:3]

    click.echo("=== Strength Profile ===")
    click.echo("")
    click.echo("Strengths:")
    for i, entry in enumerate(strengths, 1):
        pct = round(entry["rate"] * 100)
        click.echo(f"  {i}. {entry['category']}: {pct}% heeded ({entry['heeded']}/{entry['total']})")

    click.echo("")
    click.echo("Growth Areas:")
    for i, entry in enumerate(growth_areas, 1):
        pct = round(entry["rate"] * 100)
        click.echo(f"  {i}. {entry['category']}: {pct}% heeded ({entry['heeded']}/{entry['total']})")

    if insufficient:
        click.echo(f"\nCategories with insufficient data (<{min_events} events): {', '.join(sorted(insufficient))}")


@main.command("gaps")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
@click.pass_context
def gaps(ctx, as_json):
    """Show weighted gap report — categories with thin lesson coverage."""
    import json as json_mod

    from lessons_db.gap_analyzer import get_gap_report

    conn = ctx.obj["conn"]
    report = get_gap_report(conn)
    if as_json:
        click.echo(json_mod.dumps(report, indent=2))
    else:
        click.echo("Category gap scores (higher = more gaps):")
        for entry in report[:10]:
            bar = "█" * min(20, int(entry["gap_score"] * 4))
            click.echo(f"  {entry['category']:25s} {entry['gap_score']:6.3f} {bar}")


@main.command("mining-history")
@click.option("--limit", default=10, help="Number of recent runs to show.")
@click.pass_context
def mining_history(ctx, limit):
    """Show recent mining run history."""
    conn = ctx.obj["conn"]
    rows = conn.execute("SELECT * FROM mining_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    if not rows:
        click.echo("No mining runs yet.")
        return
    for row in rows:
        click.echo(
            f"  [{row['run_date']}] repos={row['repos_searched']} "
            f"commits={row['commits_analyzed']} approved={row['auto_approved']} "
            f"errors={row['error_count']}"
        )


# ---------------------------------------------------------------------------
# fix — actionable fix queue for Claude and GitHub Issues
# ---------------------------------------------------------------------------


@main.group()
def fix():
    """Manage the fix queue — actionable items for Claude or GitHub Issues."""
    pass


@fix.command("next")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON for scripting.")
@click.pass_context
def fix_next(ctx, json_output):
    """Print the highest-priority pending fix in a Claude-actionable format."""
    import json as _json

    from lessons_db.db import get_next_fix

    conn = ctx.obj["conn"]
    fix_item = get_next_fix(conn)
    if fix_item is None:
        click.echo("Fix queue is empty — no pending fixes.")
        return

    if json_output:
        click.echo(_json.dumps(fix_item, indent=2, default=str))
        return

    click.echo(f"\nFix #{fix_item['id']} — Lesson #{fix_item['lesson_id']}: {fix_item['title']}")
    click.echo(f"Enforcement: {fix_item['enforcement']}  |  Severity: {fix_item.get('severity', '?')}")
    click.echo(
        f"\nFile: {fix_item['file_path']}" + (f":{fix_item['line_number']}" if fix_item.get("line_number") else "")
    )
    if fix_item.get("snippet"):
        click.echo(f"\nDetected pattern:\n  {fix_item['snippet']}")
    if fix_item.get("suggested_fix"):
        click.echo(f"\nSuggested fix:\n  {fix_item['suggested_fix']}")
    click.echo("\nAfter fixing, run:")
    click.echo(f"  lessons-db fix done {fix_item['id']}")
    click.echo(f"  lessons-db fix skip {fix_item['id']}   (to skip)")


@fix.command("list")
@click.option(
    "--status",
    default="pending",
    type=click.Choice(["pending", "applied", "skipped", "issue_created", "wont_fix"]),
    help="Filter by status.",
)
@click.option("--limit", default=20, help="Max rows to show.")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.pass_context
def fix_list(ctx, status, limit, json_output):
    """List fix queue entries."""
    import json as _json

    from lessons_db.db import get_fix_queue

    conn = ctx.obj["conn"]
    items = get_fix_queue(conn, status=status, limit=limit)
    if not items:
        click.echo(f"No {status} fixes in queue.")
        return

    if json_output:
        click.echo(_json.dumps(items, indent=2, default=str))
        return

    click.echo(f"\n{'ID':>4}  {'Sev':>3}  {'Lesson':>6}  {'File':40s}  {'Status':14s}")
    click.echo("-" * 80)
    for item in items:
        path = item["file_path"]
        if len(path) > 38:
            path = "…" + path[-37:]
        click.echo(
            f"{item['id']:>4}  {item.get('severity', '?'):>3}  "
            f"#{item['lesson_id']:>5}  {path:40s}  {item['status']:14s}"
        )


@fix.command("done")
@click.argument("fix_id", type=int)
@click.pass_context
def fix_done(ctx, fix_id):
    """Mark a fix as applied."""
    from lessons_db.db import update_fix_status

    conn = ctx.obj["conn"]
    row = conn.execute("SELECT id FROM fix_queue WHERE id=?", (fix_id,)).fetchone()
    if not row:
        click.echo(f"Fix #{fix_id} not found.", err=True)
        ctx.exit(1)
        return
    update_fix_status(conn, fix_id, "applied")
    click.echo(f"Fix #{fix_id} marked as applied.")


@fix.command("skip")
@click.argument("fix_id", type=int)
@click.pass_context
def fix_skip(ctx, fix_id):
    """Mark a fix as skipped (won't fix)."""
    from lessons_db.db import update_fix_status

    conn = ctx.obj["conn"]
    row = conn.execute("SELECT id FROM fix_queue WHERE id=?", (fix_id,)).fetchone()
    if not row:
        click.echo(f"Fix #{fix_id} not found.", err=True)
        ctx.exit(1)
        return
    update_fix_status(conn, fix_id, "skipped")
    click.echo(f"Fix #{fix_id} marked as skipped.")


@fix.command("populate")
@click.option(
    "--min-severity",
    default=3,
    type=int,
    help="Minimum lesson severity to include (default: 3).",
)
@click.pass_context
def fix_populate(ctx, min_severity):
    """Populate fix queue from open scan findings."""
    from lessons_db.prevention import populate_fix_queue

    conn = ctx.obj["conn"]
    result = populate_fix_queue(conn, min_severity=min_severity)
    click.echo(
        f"Populated: added={result['added']}  "
        f"skipped_duplicate={result['skipped_duplicate']}  "
        f"skipped_severity={result['skipped_severity']}  "
        f"skipped_no_lesson={result['skipped_no_lesson']}"
    )


@fix.command("issues")
@click.option("--repo", default=None, help="GitHub repo (owner/name). Defaults to current origin.")
@click.option(
    "--min-severity",
    default=4,
    type=int,
    help="Minimum severity to create an issue for (default: 4).",
)
@click.option("--dry-run", is_flag=True, help="Show what would be created without calling gh.")
@click.pass_context
def fix_issues(ctx, repo, min_severity, dry_run):
    """Create GitHub issues for pending high-severity fixes."""
    from lessons_db.prevention import create_github_issues

    conn = ctx.obj["conn"]
    result = create_github_issues(conn, repo=repo, min_severity=min_severity, dry_run=dry_run)
    prefix = "[DRY RUN] " if dry_run else ""
    click.echo(
        f"{prefix}Issues: created={result['created']}  "
        f"skipped_existing={result['skipped_existing']}  "
        f"skipped_severity={result['skipped_severity']}  "
        f"errors={result['errors']}"
    )


# ---------------------------------------------------------------------------
# prevent — enforcement cycle, rule generation, content checks
# ---------------------------------------------------------------------------


@main.group()
def prevent():
    """Run the prevention pipeline — enforce, generate rules, check content."""
    pass


@prevent.command("check-content")
@click.option("--content", "-c", default=None, help="Content string to check.")
@click.option("--file", "-f", "file_path", default=None, type=click.Path(), help="Read content from file.")
@click.option(
    "--context-path",
    default=None,
    type=click.Path(),
    help="File path for metadata context (used when content comes from --file).",
)
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.pass_context
def prevent_check_content(ctx, content, file_path, context_path, json_output):
    """Check content against detection patterns and run enforcement cycle."""
    import json as _json

    from lessons_db.prevention import check_content

    conn = ctx.obj["conn"]
    if file_path and content is None:
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    if not content:
        click.echo("Provide --content TEXT or --file PATH.", err=True)
        ctx.exit(1)
        return

    # context_path overrides file_path for recurrence metadata (e.g. when content
    # is a temp file but the original path is what matters for tracking).
    effective_path = context_path or file_path
    result = check_content(conn, content, file_path=effective_path)

    if json_output:
        click.echo(_json.dumps(result, indent=2, default=str))
    elif result["block"]:
        click.echo(result["message"], err=True)
        ctx.exit(2)
        return
    elif result["violations"]:
        for v in result["violations"]:
            click.echo(f"  ⚠  Lesson #{v['lesson_id']} [{v['enforcement']}]: {v['one_liner']}")
    else:
        click.echo("OK — no pattern matches.")


@prevent.command("resolve-outcomes")
@click.option("--max-age-hours", default=24, type=int, help="Lookback window in hours (default: 24).")
@click.pass_context
def prevent_resolve_outcomes(ctx, max_age_hours):
    """Batch-resolve stale 'unknown' surfacing events via behavioral inference."""
    from lessons_db.prevention import resolve_outcomes

    conn = ctx.obj["conn"]
    result = resolve_outcomes(conn, max_age_hours=max_age_hours)
    click.echo(f"Resolved: {result['resolved']}  heeded={result['heeded']}  dismissed={result['dismissed']}")


@prevent.command("bulk-generate")
@click.option(
    "--enforcement",
    multiple=True,
    default=None,
    help="Only generate for lessons at this enforcement level (repeatable).",
)
@click.option(
    "--rules-dir",
    type=click.Path(),
    default=None,
    help="Output directory (default: ~/.local/share/lessons-db/rules/)",
)
@click.option("--no-validate", is_flag=True, help="Skip semgrep --validate.")
@click.pass_context
def prevent_bulk_generate(ctx, enforcement, rules_dir, no_validate):
    """Generate Semgrep rules for all lessons that have detection patterns."""
    from lessons_db.prevention import bulk_generate_rules

    conn = ctx.obj["conn"]
    out_dir = Path(rules_dir) if rules_dir else None
    result = bulk_generate_rules(
        conn,
        rules_dir=out_dir,
        only_enforcement=tuple(enforcement) if enforcement else None,
        validate=not no_validate,
    )
    click.echo(
        f"Generated: {result['generated']}  "
        f"skipped_no_patterns={result['skipped_no_patterns']}  "
        f"skipped_validation={result['skipped_validation']}"
    )
    for p in result["paths"]:
        click.echo(f"  {p}")


@prevent.command("report")
@click.option("--window-days", default=30, type=int, help="Lookback window in days (default: 30).")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON.")
@click.pass_context
def prevent_report(ctx, window_days, json_output):
    """Comprehensive prevention effectiveness report."""
    import json as _json

    from lessons_db.prevention import prevention_report

    conn = ctx.obj["conn"]
    report = prevention_report(conn, window_days=window_days)

    if json_output:
        click.echo(_json.dumps(report, indent=2, default=str))
        return

    click.echo(f"\n── Prevention Report (last {window_days} days) ──────────────────────")
    click.echo(f"  Total lessons:        {report['total_lessons']}")
    click.echo(f"  Rules generated:      {report['rules_generated']}")
    click.echo(f"  Without patterns:     {report['lessons_without_patterns']}")
    click.echo("\n  Enforcement coverage:")
    for level, count in sorted(report["enforcement_coverage"].items()):
        click.echo(f"    {level:25s} {count}")
    if report["velocity_alerts"]:
        click.echo(f"\n  Velocity alerts ({len(report['velocity_alerts'])} lessons hitting 2+/7d):")
        for a in report["velocity_alerts"][:5]:
            click.echo(f"    #{a['lesson_id']:4d}  {a['hit_count']:2d}x  {a['title'][:50]}")
    if report["top_recurring"]:
        click.echo(f"\n  Top recurring lessons (last {window_days}d):")
        for r in report["top_recurring"][:5]:
            click.echo(f"    #{r['lesson_id']:4d}  {r['hit_count']:2d}x  {r['title'][:50]}")
    if report["hookify_candidates"]:
        click.echo("\n  Hookify candidates (promote to blocking):")
        for h in report["hookify_candidates"][:5]:
            click.echo(f"    #{h['id']:4d}  sev={h['severity']}  {h['title'][:45]}")


# ---------------------------------------------------------------------------
# meta — batch metadata enrichment commands
# ---------------------------------------------------------------------------


@main.group()
def meta():
    """Batch metadata enrichment commands (LLM-powered)."""
    pass


def _warm_model(queue_url: str, model_name: str) -> bool:
    """Send a trivial prompt to ensure the model is loaded in Ollama's memory.

    Returns True if the model responded, False on error. This prevents
    cold-load timeouts on the first real request — large models (9GB+)
    can take >120s to load from disk, exceeding the queue proxy timeout.
    """
    import json as _json
    import urllib.error
    import urllib.request

    click.echo(f"  Warming model {model_name}...", nl=False)
    payload = _json.dumps({"model": model_name, "prompt": "hi", "stream": False}).encode("utf-8")
    try:
        req = urllib.request.Request(  # noqa: S310
            f"{queue_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
            resp.read()
        click.echo(" ready.")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        click.echo(f" failed ({exc})")
        return False


@meta.command("extract-principles")
@click.option("--batch-size", default=10, type=int, help="Number of lessons to process per batch (default: 10).")
@click.option("--dry-run", is_flag=True, help="Preview extracted principles without updating the database.")
@click.option(
    "--model",
    default=None,
    help="Ollama model to use (default: deepseek-r1:8b-0528-qwen3-q4_K_M for reasoning).",
)
@click.pass_context
def meta_extract_principles(ctx, batch_size, dry_run, model):
    """Extract domain-independent principles from lessons via LLM.

    Uses a reasoning model (deepseek-r1) to abstract concrete coding lessons
    into universal principles that transfer across technologies and projects.

    Example: "Subscriber lifecycle management" -> "Resources acquired in
    callbacks must be explicitly released in a symmetric teardown path"
    """
    import json as _json
    import re as _re
    import urllib.error
    import urllib.request

    from lessons_db.config import OLLAMA_QUEUE_URL

    # Reasoning model for abstraction — deepseek-r1 chain-of-thought
    # produces better principles than general-purpose models
    META_REASONING_MODEL = "deepseek-r1:8b-0528-qwen3-q4_K_M"

    conn = ctx.obj["conn"]
    effective_model = model or META_REASONING_MODEL

    # Find lessons without a principle
    rows = conn.execute(
        "SELECT id, title, one_liner, description FROM lessons " "WHERE principle IS NULL " "ORDER BY id " "LIMIT ?",
        (batch_size,),
    ).fetchall()

    if not rows:
        click.echo("No lessons without principles found.")
        return

    click.echo(f"Processing {len(rows)} lessons (model: {effective_model})...")
    if not dry_run:
        _warm_model(OLLAMA_QUEUE_URL, effective_model)

    updated = 0
    errors = 0
    for row in rows:
        lesson_id = row["id"]
        one_liner = row["one_liner"] or ""
        description = row["description"] or ""
        title = row["title"] or ""

        # Build context from available fields
        context_parts = []
        if title:
            context_parts.append(f"Title: {title}")
        if one_liner:
            context_parts.append(f"One-liner: {one_liner}")
        if description:
            context_parts.append(f"Description: {description[:500]}")

        if not context_parts:
            click.echo(f"  #{lesson_id}: SKIP (no title/one_liner/description)")
            continue

        lesson_context = "\n".join(context_parts)

        prompt = (
            "You are extracting a transferable principle from a specific coding lesson.\n\n"
            "A GOOD principle:\n"
            "- Names the structural pattern, not the technology (e.g., 'acquired/release symmetry' not 'close the file')\n"
            "- Is falsifiable — someone could violate it\n"
            "- Applies to at least 3 different domains (not just the one described)\n"
            "- Is one sentence, 10-25 words\n\n"
            "Examples of good principles:\n"
            "- 'Resources acquired in callbacks must be released in a symmetric teardown path.'\n"
            "- 'When two representations of the same data exist, one must be designated authoritative.'\n"
            "- 'Silent fallbacks that return default values mask upstream failures indefinitely.'\n"
            "- 'Integration boundaries require end-to-end value tracing, not per-layer unit tests.'\n\n"
            "Examples of BAD principles (too vague or just restating the lesson):\n"
            "- 'Always handle errors properly.' (not falsifiable, no structure)\n"
            "- 'Log errors before discarding them.' (restates the fix, not the principle)\n"
            "- 'Be careful with async code.' (too vague)\n\n"
            f"Lesson:\n{lesson_context}\n\n"
            "Return ONLY the principle statement. One sentence. No quotes, no explanation."
        )

        payload = _json.dumps(
            {
                "model": effective_model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")

        try:
            req = urllib.request.Request(  # noqa: S310 — localhost Ollama queue
                f"{OLLAMA_QUEUE_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
                result = _json.loads(resp.read().decode("utf-8"))
            principle = result.get("response", "").strip()
            # Strip <think>...</think> blocks from reasoning models
            principle = _re.sub(r"<think>.*?</think>", "", principle, flags=_re.DOTALL).strip()
            # Clean up any remaining artifacts
            principle = principle.strip("\"'").strip()
            if not principle:
                click.echo(f"  #{lesson_id}: SKIP (empty LLM response)")
                errors += 1
                continue
            # Truncate if model was too verbose (keep first sentence)
            if ". " in principle and len(principle) > 200:
                principle = principle[: principle.index(". ") + 1]
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, _json.JSONDecodeError) as exc:
            click.echo(f"  #{lesson_id}: ERROR ({exc})", err=True)
            errors += 1
            continue

        if dry_run:
            click.echo(f"  #{lesson_id}: {principle}")
        else:
            conn.execute(
                "UPDATE lessons SET principle = ? WHERE id = ?",
                (principle, lesson_id),
            )
            conn.commit()
            click.echo(f"  #{lesson_id}: {principle}")
            updated += 1

    click.echo(
        f"\nDone. {'Would update' if dry_run else 'Updated'}: {updated if not dry_run else len(rows) - errors}  "
        f"Errors: {errors}"
    )


@meta.command("generate-meta-lessons")
@click.option(
    "--min-cluster-size",
    default=3,
    type=int,
    help="Minimum lessons sharing a cluster_seed to trigger meta-lesson generation (default: 3).",
)
@click.option("--dry-run", is_flag=True, help="Preview clusters and prompts without writing to the database.")
@click.option(
    "--model",
    default=None,
    help="Ollama model to use (default: deepseek-r1:8b-0528-qwen3-q4_K_M for reasoning).",
)
@click.pass_context
def meta_generate_meta_lessons(ctx, min_cluster_size, dry_run, model):
    """Generate double-loop meta-lessons from clusters of related lessons.

    Uses a reasoning model (deepseek-r1) to analyze WHY patterns recur —
    identifying governing variables (Argyris double-loop learning), not
    just repeating the fix.

    Example: 5 async lessons -> "Governing variable: no symmetric
    acquire/release protocol enforced at the framework level"
    """
    import json as _json
    import re as _re
    import urllib.error
    import urllib.request

    from lessons_db.config import OLLAMA_QUEUE_URL
    from lessons_db.db import insert_lesson

    # deepseek-r1 chain-of-thought reasoning compensates for smaller size —
    # it thinks through the double-loop analysis in <think> tags before answering.
    # 5.2GB loads within the queue's 120s proxy timeout even on cold start.
    META_REASONING_MODEL = "deepseek-r1:8b-0528-qwen3-q4_K_M"

    conn = ctx.obj["conn"]
    effective_model = model or META_REASONING_MODEL

    # Step 1: Find clusters of lessons sharing the same cluster_seed
    clusters = find_meta_lesson_clusters(conn, min_cluster_size)

    if not clusters:
        click.echo(f"No clusters with >= {min_cluster_size} lessons found.")
        return

    click.echo(f"Found {len(clusters)} cluster(s) (model: {effective_model})...")
    if not dry_run:
        _warm_model(OLLAMA_QUEUE_URL, effective_model)

    generated = 0
    skipped = 0
    errors = 0

    for seed, lessons in clusters.items():
        # Check if a double-loop meta-lesson already exists for this cluster_seed
        existing = conn.execute(
            "SELECT id FROM lessons WHERE cluster_seed = ? AND loop_level = 'double' LIMIT 1",
            (seed,),
        ).fetchone()
        if existing:
            click.echo(f"  cluster '{seed}' ({len(lessons)} lessons): SKIP (meta-lesson #{existing['id']} exists)")
            skipped += 1
            continue

        # Build context from all lessons in the cluster
        lesson_summaries = []
        for lesson in lessons:
            parts = []
            if lesson["title"]:
                parts.append(lesson["title"])
            if lesson["one_liner"]:
                parts.append(lesson["one_liner"])
            lesson_summaries.append(f"  - #{lesson['id']}: {' | '.join(parts)}")

        cluster_context = "\n".join(lesson_summaries)

        prompt = (
            "You are performing double-loop learning analysis (Argyris) on a cluster of "
            "recurring coding lessons.\n\n"
            "SINGLE-LOOP (wrong): 'Fix: add error logging.' — restates the solution.\n"
            "DOUBLE-LOOP (correct): 'Governing variable: the team assumes stdlib error "
            "propagation is sufficient, so no explicit logging protocol exists at the "
            "framework level.' — identifies the hidden assumption that causes recurrence.\n\n"
            "Your task: identify the GOVERNING VARIABLE — the unquestioned assumption, "
            "mental model, or missing protocol that allows this class of bug to keep "
            "appearing despite individual fixes.\n\n"
            f"Cluster: {seed}\n"
            f"Lessons ({len(lessons)}):\n{cluster_context}\n\n"
            "Return a JSON object with exactly these fields:\n"
            "{\n"
            '  "title": "Governing Variable: <specific name>",\n'
            '  "one_liner": "The assumption that <X> causes <Y> to recur because <Z>.",\n'
            '  "description": "2-3 sentences: (1) the hidden assumption, (2) why individual '
            "fixes don't prevent recurrence, (3) what systemic change would.\"\n"
            "}\n\n"
            "BAD one_liner examples (too vague, rejected):\n"
            '- "Silent errors persist because of unlogged failures." (restates symptom)\n'
            '- "Standardize conflict resolution." (action, not governing variable)\n\n'
            "GOOD one_liner examples:\n"
            '- "The assumption that each layer owns its own error handling causes '
            "cross-boundary failures to vanish — no layer claims responsibility for "
            'errors that originate elsewhere."\n'
            '- "The absence of a symmetric acquire/release protocol means resource '
            "cleanup depends on developer memory rather than framework enforcement, "
            'guaranteeing eventual leaks."\n\n'
            "Return ONLY the JSON object."
        )

        if dry_run:
            click.echo(f"\n  cluster '{seed}' ({len(lessons)} lessons):")
            for s in lesson_summaries:
                click.echo(f"  {s}")
            click.echo(f"  [would generate meta-lesson via {effective_model}]")
            generated += 1
            continue

        payload = _json.dumps(
            {
                "model": effective_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            }
        ).encode("utf-8")

        try:
            req = urllib.request.Request(  # noqa: S310 — localhost Ollama queue
                f"{OLLAMA_QUEUE_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=300) as resp:  # noqa: S310
                result = _json.loads(resp.read().decode("utf-8"))

            raw_response = result.get("response", "").strip()
            # Strip <think>...</think> blocks (deepseek-r1 style)
            raw_response = _re.sub(r"<think>.*?</think>", "", raw_response, flags=_re.DOTALL).strip()

            meta_data = _json.loads(raw_response)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            click.echo(f"  cluster '{seed}': ERROR (network: {exc})", err=True)
            errors += 1
            continue
        except (_json.JSONDecodeError, KeyError) as exc:
            click.echo(f"  cluster '{seed}': ERROR (parse: {exc})", err=True)
            errors += 1
            continue

        title = meta_data.get("title", f"Meta: {seed}")
        one_liner = meta_data.get("one_liner", "")
        description = meta_data.get("description", "")

        if not one_liner:
            click.echo(f"  cluster '{seed}': SKIP (empty one_liner from LLM)")
            errors += 1
            continue

        # Use the first lesson in the cluster as the parent
        parent_id = lessons[0]["id"]
        lesson_id = insert_lesson(
            conn,
            {
                "title": title,
                "one_liner": one_liner,
                "description": description,
                "cluster_seed": seed,
                "loop_level": "double",
                "parent_lesson_id": parent_id,
                "source": "auto_meta",
                "entry_type": "lesson",
                "tier": "insight",
            },
        )

        click.echo(f"  cluster '{seed}' ({len(lessons)} lessons) -> meta-lesson #{lesson_id}: {one_liner[:60]}")
        generated += 1

    click.echo(
        f"\nDone. {'Would generate' if dry_run else 'Generated'}: {generated}  " f"Skipped: {skipped}  Errors: {errors}"
    )


@meta.command("eval-generate")
@click.option("--variants", default="A,B,C,D,E", help="Comma-separated variant IDs (default: A,B,C,D,E).")
@click.option("--per-cluster", default=4, type=int, help="Source lessons per cluster (default: 4).")
@click.option(
    "--output", type=click.Path(), default=None, help="Output JSON path (default: auto-timestamped in EVAL_DIR)."
)
@click.option("--resume", is_flag=True, help="Skip already-completed (variant, lesson_id) pairs.")
@click.option("--priority", type=int, default=None, help="Queue priority (1=highest). Unset uses queue default.")
@click.pass_context
def meta_eval_generate(ctx, variants, per_cluster, output, resume, priority):
    """Generate principles across prompt variants for transfer-test evaluation.

    Runs each variant (prompt x model x settings) across a fixed set of source
    lessons. Results saved to a JSON file for later judging with eval-judge.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    from lessons_db.config import EVAL_DIR, OLLAMA_QUEUE_URL
    from lessons_db.eval import VARIANT_CONFIGS, run_eval_generate, select_source_lessons

    conn = ctx.obj["conn"]
    variant_list = [v.strip() for v in variants.split(",")]

    # Validate variant IDs
    for v in variant_list:
        if v not in VARIANT_CONFIGS:
            click.echo(f"Unknown variant '{v}'. Valid: {', '.join(VARIANT_CONFIGS.keys())}", err=True)
            ctx.exit(1)
            return

    # Check source lessons exist
    sources = select_source_lessons(conn, per_cluster=per_cluster)
    if not sources:
        click.echo("No source lessons found (need clusters with >= 3 lessons).")
        return

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        output_path = EVAL_DIR / f"results-{ts}.json"

    click.echo(f"Eval-generate: {len(variant_list)} variants x {len(sources)} lessons")
    click.echo(f"Output: {output_path}")

    # Warm models (deduplicate)
    models_to_warm = {VARIANT_CONFIGS[v]["model"] for v in variant_list}
    for model_name in models_to_warm:
        _warm_model(OLLAMA_QUEUE_URL, model_name)

    def _progress(variant_id, lesson_id, success):
        status = "OK" if success else "FAIL"
        click.echo(f"  [{variant_id}] lesson #{lesson_id}: {status}")

    result = run_eval_generate(
        conn=conn,
        queue_url=OLLAMA_QUEUE_URL,
        variants=variant_list,
        per_cluster=per_cluster,
        output_path=output_path,
        resume=resume,
        progress_callback=_progress,
        priority=priority,
    )

    total = len(result["results"])
    errors = sum(1 for r in result["results"] if r.get("error"))
    click.echo(f"\nDone. Total: {total}  Errors: {errors}")
    click.echo(f"Results: {output_path}")


@meta.command("eval-judge")
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default=None, help="Output report path (default: auto in EVAL_DIR).")
@click.option("--openai", "use_openai", is_flag=True, help="Use OpenAI GPT-4o-mini as judge (requires OPENAI_API_KEY).")
@click.option("--judge-model", default=None, help="Judge model name (Ollama model or OpenAI model with --openai).")
@click.option("--priority", type=int, default=None, help="Queue priority (1=highest). Unset uses queue default.")
@click.pass_context
def meta_eval_judge(ctx, results_file, output, use_openai, judge_model, priority):
    """Score generated principles against transfer test targets.

    Reads a results JSON from eval-generate, constructs transfer tests
    (same-cluster true positives + different-cluster true negatives),
    scores each pair, and produces a markdown report with F1 metrics.
    """
    from datetime import UTC, datetime
    from pathlib import Path

    from lessons_db.config import EVAL_DIR, OLLAMA_QUEUE_URL, OPENAI_API_KEY
    from lessons_db.eval import DEFAULT_JUDGE_MODEL, run_eval_judge

    conn = ctx.obj["conn"]
    results_path = Path(results_file)

    # Determine output path
    if output:
        report_path = Path(output)
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        report_path = EVAL_DIR / f"report-{datetime.now(UTC).strftime('%Y-%m-%d')}.md"

    # Configure judge backend
    if use_openai:
        if not OPENAI_API_KEY:
            click.echo("OPENAI_API_KEY not set. Set it in ~/.env or environment.", err=True)
            ctx.exit(1)
            return
        backend = "openai"
        model = judge_model or "gpt-4o-mini"
        click.echo(f"Judge: OpenAI {model}")
    else:
        backend = "ollama"
        model = judge_model or DEFAULT_JUDGE_MODEL
        click.echo(f"Judge: Ollama {model}")
        _warm_model(OLLAMA_QUEUE_URL, model)

    def _progress(variant, target_id, label, scores):
        s = scores
        click.echo(
            f"  [{variant}] target #{target_id} ({label}): "
            f"T={s['transfer']} P={s['precision']} A={s['actionability']}"
        )

    scored_pairs, metrics = run_eval_judge(
        results_path=results_path,
        conn=conn,
        report_path=report_path,
        backend=backend,
        ollama_url=OLLAMA_QUEUE_URL,
        ollama_model=model if backend == "ollama" else "",
        openai_api_key=OPENAI_API_KEY if backend == "openai" else "",
        openai_model=model if backend == "openai" else "",
        progress_callback=_progress,
        priority=priority,
    )

    click.echo(f"\nScored {len(scored_pairs)} pairs across {len(metrics)} variants.")
    if metrics:
        winner = max(metrics.keys(), key=lambda v: metrics[v]["f1"])
        wm = metrics[winner]
        click.echo(f"Winner: Variant {winner} (F1={wm['f1']:.2f})")
    click.echo(f"Report: {report_path}")


def find_meta_lesson_clusters(
    conn,
    min_cluster_size: int = 3,
) -> dict[str, list[dict]]:
    """Find clusters of lessons sharing the same cluster_seed with at least min_cluster_size members.

    Returns a dict mapping cluster_seed -> list of lesson dicts (id, title, one_liner).
    Only includes single-loop lessons (excludes existing meta-lessons).
    """
    # Find cluster_seeds with enough lessons
    rows = conn.execute(
        "SELECT cluster_seed, COUNT(*) as cnt "
        "FROM lessons "
        "WHERE cluster_seed IS NOT NULL AND cluster_seed != '' "
        "  AND (loop_level IS NULL OR loop_level = 'single') "
        "GROUP BY cluster_seed "
        "HAVING COUNT(*) >= ? "
        "ORDER BY cnt DESC",
        (min_cluster_size,),
    ).fetchall()

    clusters: dict[str, list[dict]] = {}
    for row in rows:
        seed = row["cluster_seed"]
        lessons = conn.execute(
            "SELECT id, title, one_liner FROM lessons "
            "WHERE cluster_seed = ? AND (loop_level IS NULL OR loop_level = 'single') "
            "ORDER BY id",
            (seed,),
        ).fetchall()
        clusters[seed] = [dict(l) for l in lessons]

    return clusters


# ---------------------------------------------------------------------------
# FSRS spaced-repetition commands
# ---------------------------------------------------------------------------


@main.group()
@click.pass_context
def fsrs(ctx):
    """FSRS spaced-repetition scheduling commands."""


@fsrs.command("init")
@click.pass_context
def fsrs_init(ctx):
    """Backfill all existing lessons with FSRS-6 default parameters.

    Sets stability=1.0, difficulty=5.0, retrievability=1.0 for any lesson
    that has not yet been initialized. Safe to run multiple times (idempotent).
    """
    from lessons_db.fsrs import backfill_fsrs_defaults

    conn = ctx.obj["conn"]
    count = backfill_fsrs_defaults(conn)
    total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    click.echo(f"FSRS init complete: {count} lessons backfilled ({total} total).")


@fsrs.command("due")
@click.option("--threshold", type=float, default=0.9, help="Retrievability threshold (default 0.9).")
@click.pass_context
def fsrs_due(ctx, threshold):
    """List lessons whose retrievability is below threshold (most forgotten first).

    Shows lesson_id, title, stability, retrievability, and days since last review.
    Results are sorted by retrievability ascending — the lessons you've forgotten
    most appear first.
    """
    from lessons_db.fsrs import ensure_fsrs_columns, get_due_lessons, get_fading_level, interleave_due_lessons

    conn = ctx.obj["conn"]
    ensure_fsrs_columns(conn)
    due = get_due_lessons(conn, threshold=threshold)
    if not due:
        click.echo("No lessons due for review.")
        return

    interleaved = interleave_due_lessons(due)
    click.echo(f"Due lessons (R < {threshold}): {len(interleaved)}\n")
    for lesson in interleaved:
        fading = get_fading_level(lesson["stability"])
        click.echo(
            f"  [#{lesson['id']}] {lesson.get('title', '(untitled)')}"
            f"  S={lesson['stability']:.2f}  R={lesson['retrievability']:.3f}"
            f"  days={lesson['days_since_review']}  level={fading}"
        )


@fsrs.command("stats")
@click.pass_context
def fsrs_stats(ctx):
    """Show FSRS stability distribution and review forecast.

    Stability distribution: count of lessons at each fading level
    (full/brief/silent/enforced).

    Review forecast: how many lessons will be due in 1, 3, 7, 14, and 30 days
    assuming no new reviews occur.
    """
    from lessons_db.fsrs import compute_retrievability, ensure_fsrs_columns, get_fading_level

    conn = ctx.obj["conn"]
    ensure_fsrs_columns(conn)

    # --- Stability distribution ---
    rows = conn.execute("SELECT stability FROM lessons WHERE stability IS NOT NULL").fetchall()

    level_counts: dict[str, int] = {"full": 0, "brief": 0, "silent": 0, "enforced": 0}
    for row in rows:
        level = get_fading_level(row["stability"])
        level_counts[level] += 1

    click.echo("Stability distribution:")
    for level in ("full", "brief", "silent", "enforced"):
        click.echo(f"  {level:10s}: {level_counts[level]}")

    # --- Review forecast ---
    reviewed = conn.execute(
        """
        SELECT id, stability, last_review_date
        FROM lessons
        WHERE last_review_date IS NOT NULL
          AND stability IS NOT NULL
          AND stability > 0
        """
    ).fetchall()

    from datetime import date

    today = date.today()
    forecast_days = [1, 3, 7, 14, 30]
    click.echo("\nReview forecast (lessons due at R < 0.9):")

    for future_days in forecast_days:
        count = 0
        for row in reviewed:
            review_date = date.fromisoformat(row["last_review_date"])
            days_elapsed = (today - review_date).days + future_days
            r = compute_retrievability(row["stability"], float(days_elapsed))
            if r < 0.9:
                count += 1
        click.echo(f"  in {future_days:2d} day(s): {count}")


# ---------------------------------------------------------------------------
# Transfer — cross-project analogical matching
# ---------------------------------------------------------------------------


@main.group()
@click.pass_context
def transfer(ctx):
    """Cross-project analogical matching — find lessons that transfer across scopes."""


@transfer.command("find")
@click.argument("context")
@click.option("--limit", "-n", default=5, type=int, help="Max results to return.")
@click.option("--min-score", default=0.3, type=float, help="Minimum similarity score threshold.")
@click.pass_context
def transfer_find(ctx, context, limit, min_score):
    """Search for transferable lessons by principle similarity across ALL scopes.

    CONTEXT is the situation or problem you're facing. Results are drawn from
    every scope in the database, ignoring the current project's scope filter,
    so you can discover lessons learned elsewhere that apply here.

    Searches the 'principle' field first. Falls back to one_liner + description
    when no principle is populated.
    """
    conn = ctx.obj["conn"]

    # Try to init LanceDB for semantic search (graceful failure)
    lance_db = None
    try:
        import lancedb

        if LANCE_DIR.exists():
            lance_db = lancedb.connect(str(LANCE_DIR))
    except Exception:
        logger.debug("LanceDB unavailable, skipping semantic search for transfer find")

    # Use search_combined without scope filter (cross-project by design)
    results = search_combined(
        conn,
        lance_db,
        query=context,
    )

    if not results:
        click.echo("No transferable lessons found.")
        return

    # Enrich results with principle, scope, and description from DB
    enriched = []
    for r in results:
        rid = r.get("id")
        if rid is None:
            continue
        row = conn.execute(
            "SELECT id, principle, one_liner, description, scope FROM lessons WHERE id = ?",
            (rid,),
        ).fetchone()
        if row is None:
            continue

        score = r.get("composite_score") or r.get("score") or 0.0
        if score < min_score:
            continue

        enriched.append(
            {
                "id": row["id"],
                "principle": row["principle"],
                "one_liner": row["one_liner"],
                "description": row["description"],
                "scope": row["scope"] or "unscoped",
                "score": score,
            }
        )

    # Sort by score descending, take top N
    enriched.sort(key=lambda x: x["score"], reverse=True)
    enriched = enriched[:limit]

    if not enriched:
        click.echo("No transferable lessons above min-score threshold.")
        return

    for item in enriched:
        # Prefer principle if populated, otherwise fall back to one_liner
        display_text = item["principle"] if item["principle"] else item["one_liner"]
        scope = item["scope"]
        click.echo(f"From [{scope}]: {display_text} — {item['one_liner']}")
        if item["principle"] and item["description"]:
            # Show a truncated description for extra context
            desc_preview = (item["description"] or "")[:120]
            if len(item["description"] or "") > 120:
                desc_preview += "..."
            click.echo(f"  ({desc_preview})")
        click.echo(f"  [#{item['id']}] score={item['score']:.3f}")
