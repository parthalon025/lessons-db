"""Click CLI for lessons-db: status, search, migrate."""

import json
import logging
from pathlib import Path

import click

from lessons_db.config import SQLITE_PATH, LANCE_DIR, LESSONS_SOURCE_DIR, RULES_DIR
from lessons_db.db import (
    init_db,
    get_overdue_actions,
    get_open_findings,
    get_near_miss_hotspots,
    insert_lesson,
    insert_corrective_action,
)
from lessons_db.search import search_combined
from lessons_db.migrate import parse_lesson_file

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
@click.pass_context
def search(ctx, query, file, content, top):
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
@click.option("--source", type=click.Path(exists=True), default=None, help="Source directory for lesson markdown files.")
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
    errors = 0
    for f in md_files:
        try:
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
                insert_corrective_action(conn, {
                    "lesson_id": lesson_id,
                    "action": action.get("description", ""),
                    "status": action.get("status", "proposed"),
                })

            migrated += 1
        except Exception as exc:
            logger.error("Failed to migrate %s: %s", f.name, exc)
            errors += 1

    click.echo(f"Migrated: {migrated}, Errors: {errors}")


@main.command()
@click.option("--seed-only", is_flag=True, help="Only backfill cluster_seed, skip embedding generation.")
@click.pass_context
def index(ctx, seed_only):
    """Backfill cluster_seed and generate LanceDB embeddings for all lessons.

    Run once after initial migrate, or after adding new lessons without embeddings.
    cluster_seed: copies cluster → cluster_seed for A-F historical labels.
    Embeddings: calls Ollama nomic-embed-text for each lesson's title + one_liner.
    """
    import lancedb
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
@click.option("--rules-dir", type=click.Path(), default=None,
              help="Directory to write rules (default: ~/.local/share/lessons-db/rules/)")
@click.option("--severity", default="WARNING",
              type=click.Choice(["WARNING", "ERROR", "INFO"]),
              help="Semgrep rule severity.")
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

    patterns = conn.execute(
        "SELECT * FROM detection_patterns WHERE lesson_id = ?", (lesson_id,)
    ).fetchall()
    if not patterns:
        click.echo(f"No detection patterns for lesson #{lesson_id}. "
                   "Add patterns via detection_patterns table first.")
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
@click.option("--rules-dir", type=click.Path(), default=None,
              help="Directory containing rules (default: ~/.local/share/lessons-db/rules/)")
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
        capture_output=True, text=True,
    )
    click.echo(result.stdout or result.stderr)
    if result.returncode == 0:
        click.echo("All rules passed.")
    else:
        click.echo(f"Test failures (exit code {result.returncode}).")
        ctx.exit(result.returncode)


@main.command()
@click.option("--rules-dir", type=click.Path(), default=None,
              help="Rules directory (default: ~/.local/share/lessons-db/rules/)")
@click.option("--target", type=click.Path(), default=None,
              help="Target directory to scan (default: ~/Documents/projects/)")
@click.option("--baseline", default=None,
              help="Git commit hash for diff-aware scanning.")
@click.pass_context
def scan(ctx, rules_dir, target, baseline):
    """Run Semgrep scan against all lessons rules and record findings."""
    from lessons_db.scan import run_scan
    from lessons_db.db import insert_scan_finding

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

    for f in findings:
        rule_id = f.get("rule_id", "")
        click.echo(f"  [{rule_id}] {f.get('file_path')}:{f.get('line_number')}")
        try:
            insert_scan_finding(conn, {
                "lesson_id": 0,
                "rule_id": rule_id,
                "file_path": f.get("file_path", ""),
                "line_number": f.get("line_number"),
                "snippet": f.get("message", ""),
            })
        except Exception as exc:
            logger.warning("scan: failed to insert finding %s: %s", rule_id, exc)

    click.echo(f"\nTotal findings: {len(findings)} (saved to DB)")


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


@main.group()
def cluster():
    """Adaptive cluster discovery and management."""
    pass


@cluster.command("show")
@click.pass_context
def cluster_show(ctx):
    """Show current cluster assignments for all lessons."""
    conn = ctx.obj["conn"]
    rows = conn.execute(
        "SELECT cluster, COUNT(*) as n FROM lessons GROUP BY cluster ORDER BY n DESC"
    ).fetchall()
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
        click.echo(f"[{run['run_date']}] {run['proposal_count']} proposals, "
                   f"{run['confirmed_count']} confirmed")


@cluster.command("discover")
@click.option("--min-size", default=5, type=int, help="Minimum cluster size for HDBSCAN.")
@click.pass_context
def cluster_discover(ctx, min_size):
    """Run HDBSCAN on embeddings and propose new cluster assignments."""
    from lessons_db.cluster import discover_clusters, apply_cluster_proposals
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
        name = click.prompt("  Accept name? (Enter to accept, or type a new name, or 's' to skip)",
                            default=p["suggested_name"])
        if name.lower() != "s":
            confirmed[p["cluster_id"]] = name
    if confirmed:
        count = apply_cluster_proposals(conn, proposals, confirmed)
        click.echo(f"\n✓ Updated {count} lesson cluster assignments.")
    else:
        click.echo("No clusters confirmed.")


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


@main.command()
@click.option("--tail", "-n", default=50, type=int, help="Number of lines from end to show.")
@click.option("--level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Filter by log level.")
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
