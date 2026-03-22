#!/usr/bin/env python3
"""Infer and apply scope tags to lessons that have no scope in the DB.

Uses the same heuristic signal set as scope-infer.sh but operates directly
on lesson rows (one_liner + keywords + category fields) rather than markdown files.

Usage:
    python scripts/retag_scope.py            # dry-run (default)
    python scripts/retag_scope.py --execute  # write scope to DB
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lessons_db.config import SQLITE_PATH

# ---------------------------------------------------------------------------
# Heuristic rules — ordered by specificity (first match wins)
# ---------------------------------------------------------------------------

_SCOPE_RULES: list[tuple[str, list[str]]] = [
    # Domain-specific — narrow signals first
    (
        "domain:ha-aria",
        [
            r"home.assistant",
            r"\bha\b",
            r"\bhass\b",
            r"ha-aria",
            r"entity.*area",
            r"automation.*trigger",
            r"lovelace",
            r"ha_token",
        ],
    ),
    (
        "domain:telegram",
        [
            r"telegram",
            r"bot.*poll",
            r"getupdates",
            r"chat_id",
            r"telegram-brief",
            r"telegram-capture",
        ],
    ),
    (
        "domain:notion",
        [
            r"\bnotion\b",
            r"notion.*sync",
            r"notion.*database",
            r"notion-tools",
            r"notion_api",
        ],
    ),
    (
        "domain:ollama",
        [
            r"\bollama\b",
            r"ollama.*queue",
            r"local.*llm",
            r"ollama-queue",
            r"qwen",
            r"nomic-embed",
        ],
    ),
    (
        "domain:lessons-db",
        [
            r"lessons.db",
            r"lessons-db",
            r"capture.*draft",
            r"draft.*triage",
            r"detection.pattern",
            r"semgrep.*rule",
        ],
    ),
    # Framework/tool
    (
        "framework:systemd",
        [
            r"systemd",
            r"systemctl",
            r"\.service\b",
            r"\.timer\b",
            r"journalctl",
            r"\benvfile\b",
            r"execstart",
        ],
    ),
    (
        "framework:pytest",
        [
            r"\bpytest\b",
            r"\bconftest\b",
            r"\bfixture\b",
            r"\bparametrize\b",
            r"assert.*len\s*==",
            r"test.*function",
        ],
    ),
    (
        "framework:preact",
        [
            r"\bpreact\b",
            r"\bjsx\b",
            r"esbuild.*jsx",
            r"jsx.*factory",
        ],
    ),
    # Project-specific
    (
        "project:autonomous-coding-toolkit",
        [
            r"run-plan",
            r"quality.gate",
            r"lesson-check",
            r"mab-run",
            r"batch.*audit",
            r"ralph.*loop",
            r"headless.*mode",
        ],
    ),
    # Language — broader signals
    (
        "language:python",
        [
            r"\bpython\b",
            r"\basync def\b",
            r"\bawait\b",
            r"\bsqlite3\b",
            r"\brequests\b",
            r"\bpyproject\b",
            r"\bpip install\b",
            r"except.*error",
            r"import.*module",
        ],
    ),
    (
        "language:bash",
        [
            r"\bbash\b",
            r"\bshell\b",
            r"\.sh\b",
            r"\bcron\b",
            r"\bset -e\b",
            r"\$\{",
            r"export.*=",
        ],
    ),
    (
        "language:javascript",
        [
            r"\bjavascript\b",
            r"\btypescript\b",
            r"\bnpm\b",
            r"\bnode\b",
            r"const\s+\w+\s*=",
            r"\.tsx?\b",
        ],
    ),
]

_COMPILED_RULES: list[tuple[str, list[re.Pattern[str]]]] = [
    (scope, [re.compile(p, re.IGNORECASE) for p in patterns]) for scope, patterns in _SCOPE_RULES
]


def infer_scope(one_liner: str, keywords: str, category: str) -> str:
    """Return the best scope tag for a lesson, or 'universal' if no signals match."""
    text = " ".join(filter(None, [one_liner, keywords, category])).lower()
    for scope, patterns in _COMPILED_RULES:
        if any(p.search(text) for p in patterns):
            return scope
    return "universal"


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer scope tags for unscoped lessons")
    parser.add_argument("--execute", action="store_true", help="Write inferred scopes to DB (default: dry-run)")
    args = parser.parse_args()

    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, one_liner, keywords, category FROM lessons WHERE scope IS NULL OR scope = '' ORDER BY id"
    ).fetchall()

    print(f"Unscoped lessons: {len(rows)}")
    if not rows:
        print("Nothing to do.")
        return

    scope_counts: dict[str, int] = {}
    updates: list[tuple[str, int]] = []

    for row in rows:
        scope = infer_scope(
            row["one_liner"] or "",
            row["keywords"] or "",
            row["category"] or "",
        )
        scope_counts[scope] = scope_counts.get(scope, 0) + 1
        updates.append((scope, row["id"]))

    print("\nInferred scope distribution:")
    for scope, count in sorted(scope_counts.items(), key=lambda x: -x[1]):
        print(f"  {scope}: {count}")

    if not args.execute:
        print("\n--- DRY RUN: sample assignments ---")
        for scope, lesson_id in updates[:20]:
            one_liner = next(r["one_liner"] for r in rows if r["id"] == lesson_id)
            print(f"  [{lesson_id}] {scope:35s} | {one_liner[:60]}")
        if len(updates) > 20:
            print(f"  ... and {len(updates) - 20} more")
        print("\nRe-run with --execute to apply.")
        return

    # Apply
    for scope, lesson_id in updates:
        conn.execute("UPDATE lessons SET scope = ? WHERE id = ?", [scope, lesson_id])
    conn.commit()

    print(f"\nApplied scope tags to {len(updates)} lessons.")
    total_scoped = conn.execute("SELECT COUNT(*) FROM lessons WHERE scope IS NOT NULL AND scope != ''").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    print(f"Coverage: {total_scoped}/{total} lessons now have scope ({100 * total_scoped // total}%)")


if __name__ == "__main__":
    main()
