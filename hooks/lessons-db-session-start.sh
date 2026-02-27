#!/usr/bin/env bash
# SessionStart hook: show lessons-db status line and positive pattern count
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")

if [[ -z "$LESSONS_DB" ]]; then
    exit 0  # lessons-db not installed yet
fi

"$LESSONS_DB" status 2>/dev/null || true

# Show count of positive patterns as signal of accumulated knowledge.
# Direct DB query — avoids LanceDB/CLI dependency and works even when search is unavailable.
POS_COUNT=$(python3 - <<'PYEOF' 2>/dev/null || echo "0"
import sqlite3, os, pathlib
db = pathlib.Path.home() / ".local/share/lessons-db/lessons.db"
if db.exists():
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM lessons WHERE polarity='positive'").fetchone()[0]
    conn.close()
    print(n)
else:
    print(0)
PYEOF
)

if [[ "${POS_COUNT:-0}" -gt 0 ]]; then
    echo "Positive patterns available: ${POS_COUNT}"
fi

# Pattern scan counts (cross-project pattern detection v3)
PATTERN_AUTO=$(lessons-db pattern status 2>/dev/null | grep -oP '\d+(?= auto-captured)' || echo "0")
PATTERN_PENDING=$(lessons-db pattern status 2>/dev/null | grep -oP '\d+(?= pending)' || echo "0")

if [ "${PATTERN_AUTO:-0}" -gt 0 ] || [ "${PATTERN_PENDING:-0}" -gt 0 ]; then
    echo "${PATTERN_AUTO} patterns auto-captured | ${PATTERN_PENDING} awaiting review"
fi
