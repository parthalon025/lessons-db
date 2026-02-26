"""Click CLI for lessons-db: status, search, migrate."""

import json
import logging
from pathlib import Path

import click

from lessons_db.config import SQLITE_PATH, LANCE_DIR, LESSONS_SOURCE_DIR
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
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

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
