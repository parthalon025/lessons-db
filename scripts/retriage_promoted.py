#!/usr/bin/env python3
"""Re-triage auto-promoted lessons from today's backfill using the improved prompt.

Usage:
    python scripts/retriage_promoted.py --dry-run    # preview only (default)
    python scripts/retriage_promoted.py --execute    # actually delete low-quality lessons
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# Ensure the venv package is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lessons_db.config import OPENAI_API_KEY, SQLITE_PATH
from lessons_db.review import claude_review_batch

RETRIAGE_DATE = "2026-02-27"
RETRIAGE_SOURCES = ("auto_transcript", "auto_transcript_positive", "auto_diff")


def load_lessons(conn: sqlite3.Connection) -> list[dict]:
    """Load all auto-promoted lessons from the backfill date."""
    rows = conn.execute(
        """
        SELECT id, one_liner, source, created_date
        FROM lessons
        WHERE source IN ({}) AND created_date = ?
        ORDER BY id
        """.format(",".join("?" * len(RETRIAGE_SOURCES))),
        [*RETRIAGE_SOURCES, RETRIAGE_DATE],
    ).fetchall()
    return [dict(r) for r in rows]


def load_migrated_titles(conn: sqlite3.Connection) -> list[str]:
    """Load one_liners from original migrated lessons — the quality baseline."""
    rows = conn.execute("SELECT one_liner FROM lessons WHERE source = 'migrated' ORDER BY id").fetchall()
    return [r["one_liner"] for r in rows]


def lessons_to_mock_drafts(lessons: list[dict]) -> list[dict]:
    """Convert lesson rows to the draft format expected by claude_review_batch."""
    import json

    return [
        {
            "id": lesson["id"],
            "extracted_data": json.dumps({"one_liner": lesson["one_liner"]}),
            "source": lesson["source"],
        }
        for lesson in lessons
    ]


def delete_lesson(conn: sqlite3.Connection, lesson_id: int) -> None:
    """Delete a lesson and its detection patterns."""
    conn.execute("DELETE FROM detection_patterns WHERE lesson_id = ?", [lesson_id])
    conn.execute("DELETE FROM lessons WHERE id = ?", [lesson_id])


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-triage auto-promoted lessons")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True, help="Preview only (default)")
    group.add_argument("--execute", action="store_true", help="Actually delete low-quality lessons")
    args = parser.parse_args()

    execute = args.execute

    api_key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    lessons = load_lessons(conn)
    print(f"Loaded {len(lessons)} auto-promoted lessons from {RETRIAGE_DATE}")

    migrated_titles = load_migrated_titles(conn)
    print(f"Using {len(migrated_titles)} migrated lessons as quality baseline\n")

    if not lessons:
        print("Nothing to re-triage.")
        return

    drafts = lessons_to_mock_drafts(lessons)
    print(f"Sending {len(drafts)} lessons to OpenAI for re-triage...")
    verdicts = claude_review_batch(drafts, existing_titles=migrated_titles, api_key=api_key)

    keep = [v for v in verdicts if v["verdict"] == "PROMOTE"]
    delete = [v for v in verdicts if v["verdict"] in ("DISMISS", "ERROR")]
    errors = [v for v in verdicts if v["verdict"] == "ERROR"]

    print("\nResults:")
    print(f"  KEEP  (PROMOTE, confidence >= 4): {len(keep)}")
    print(f"  DELETE (DISMISS + ERROR):          {len(delete)}  ({len(errors)} errors)")
    print()

    if not execute:
        print("--- DRY RUN: showing first 20 to-delete ---")
        for v in delete[:20]:
            lesson = next((l for l in lessons if l["id"] == v["id"]), None)
            one_liner = lesson["one_liner"] if lesson else "?"
            print(f"  [{v['id']}] confidence={v.get('confidence', 0)} | {one_liner[:70]}")
            print(f"         reason: {v['reason'][:80]}")
        if len(delete) > 20:
            print(f"  ... and {len(delete) - 20} more")
        print()
        print("Re-run with --execute to apply deletions.")
        return

    # Execute: delete low-quality lessons
    print(f"Deleting {len(delete)} lessons + their detection patterns...")
    deleted = 0
    for v in delete:
        delete_lesson(conn, v["id"])
        deleted += 1
    conn.commit()

    remaining = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    patterns_remaining = conn.execute("SELECT COUNT(*) FROM detection_patterns").fetchone()[0]

    print("\nDone.")
    print(f"  Deleted: {deleted} lessons")
    print(f"  Remaining lessons: {remaining}")
    print(f"  Remaining detection patterns: {patterns_remaining}")


if __name__ == "__main__":
    main()
