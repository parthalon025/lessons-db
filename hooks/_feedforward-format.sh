#!/usr/bin/env bash
# Shared feedforward formatting for lesson surfacing hooks.
# Source this file from any hook that surfaces lessons.
#
# Usage:
#   source "$(dirname "${BASH_SOURCE[0]}")/_feedforward-format.sh"
#   feedforward_format "$RESULTS"
#
# Output format (Goldsmith feedforward methodology):
#   Negative lessons: SUGGESTION: In this context, use [corrective_action] — it prevents [one_liner]
#   Positive lessons: PROVEN PATTERN: [one_liner] — successfully used in [scope] ([reuse_count] times)
#
# Falls back to "Consider: [one_liner]" when corrective_action is NULL/empty.

feedforward_format() {
    local results="$1"
    if [[ -z "$results" ]]; then
        return
    fi

    # Parse each [#ID] line and reformat using DB metadata
    python3 - "$results" <<'PYEOF'
import sqlite3
import sys
import pathlib
import re

results_text = sys.argv[1]
db_path = pathlib.Path.home() / ".local/share/lessons-db/lessons.db"

if not db_path.exists():
    # No DB — pass through original output
    print(results_text)
    sys.exit(0)

conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row

for line in results_text.splitlines():
    m = re.match(r'^\[#(\d+)\]\s*(.*)', line)
    if not m:
        # Non-lesson lines (headers, blank lines) — pass through
        print(line)
        continue

    lesson_id = int(m.group(1))
    one_liner = m.group(2).strip()
    # Strip trailing "(via ...)" from one_liner if present
    via_match = re.match(r'^(.*?)\s*\(via\s+.*\)$', one_liner)
    if via_match:
        one_liner = via_match.group(1).strip()

    row = conn.execute(
        "SELECT polarity, corrective_action, scope, reuse_count FROM lessons WHERE id = ?",
        (lesson_id,),
    ).fetchone()

    if row is None:
        # Lesson not found — fall back to original format
        print(line)
        continue

    polarity = row["polarity"] or "negative"
    corrective_action = (row["corrective_action"] or "").strip()
    scope = (row["scope"] or "").strip()
    reuse_count = row["reuse_count"] or 0

    if polarity == "positive":
        scope_part = f" in {scope}" if scope else ""
        count_part = f" ({reuse_count} times)" if reuse_count > 0 else ""
        print(f"PROVEN PATTERN: [#{lesson_id}] {one_liner} — successfully used{scope_part}{count_part}")
    else:
        # Negative lesson — feedforward framing
        if corrective_action:
            print(f"SUGGESTION: [#{lesson_id}] In this context, use {corrective_action} — it prevents {one_liner}")
        else:
            print(f"SUGGESTION: [#{lesson_id}] Consider: {one_liner}")

conn.close()
PYEOF
}
