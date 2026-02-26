"""Click CLI for lessons-db: status, search, migrate."""

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
    ctx.obj["conn"] = init_db(db_path)


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
    md_files = sorted(f for f in source_dir.glob("2026-*.md") if f.is_file())

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
