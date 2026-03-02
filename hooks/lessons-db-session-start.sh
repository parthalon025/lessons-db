#!/usr/bin/env bash
# SessionStart hook: show lessons-db status line, resolve stale outcomes, show fix queue
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")

if [[ -z "$LESSONS_DB" ]]; then
    exit 0  # lessons-db not installed yet
fi

# Source shared feedforward formatting
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hooks/_feedforward-format.sh
source "${HOOK_DIR}/_feedforward-format.sh"

"$LESSONS_DB" status 2>/dev/null || true

# Resolve stale unknown surfacing outcomes from prior session (behavioral inference)
RESOLVE_RESULT=$("$LESSONS_DB" prevent resolve-outcomes 2>/dev/null || echo "")
if [[ -n "$RESOLVE_RESULT" && "$RESOLVE_RESULT" != "Resolved: 0  heeded=0  dismissed=0" ]]; then
    echo "$RESOLVE_RESULT"
fi

# Show pending fix queue count (actionable items for this session)
FIX_COUNT=$(python3 - <<'PYEOF' 2>/dev/null || echo "0"
import sqlite3, pathlib
db = pathlib.Path.home() / ".local/share/lessons-db/lessons.db"
if db.exists():
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM fix_queue WHERE status='pending'").fetchone()[0]
    conn.close()
    print(n)
else:
    print(0)
PYEOF
)
if [[ "${FIX_COUNT:-0}" -gt 0 ]]; then
    echo "Fix queue: ${FIX_COUNT} pending — run: lessons-db fix next"
fi

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
PATTERN_STATUS=$(lessons-db pattern status 2>/dev/null || echo "")
PATTERN_AUTO=$(echo "$PATTERN_STATUS" | grep -oP '\d+(?= auto-captured)' || echo "0")
PATTERN_PENDING=$(echo "$PATTERN_STATUS" | grep -oP '\d+(?= pending)' || echo "0")

if [ "${PATTERN_AUTO:-0}" -gt 0 ] || [ "${PATTERN_PENDING:-0}" -gt 0 ]; then
    echo "${PATTERN_AUTO} patterns auto-captured | ${PATTERN_PENDING} awaiting review"
fi

# Show overnight promoted lessons from today's triage log
TRIAGE_LOG="$HOME/.local/share/lessons-db/triage-$(date +%Y-%m-%d).jsonl"
if [[ -f "$TRIAGE_LOG" ]]; then
    PROMOTED_TONIGHT=$(grep -c '"verdict": "PROMOTE",' "$TRIAGE_LOG" 2>/dev/null || echo "0")
    if [[ "${PROMOTED_TONIGHT:-0}" -gt 0 ]]; then
        echo "${PROMOTED_TONIGHT} lessons auto-promoted overnight — run: lessons-db capture triage --review-log"
    fi
fi

# Pending triage count (drafts not yet reviewed by Claude)
PENDING_COUNT=$(python3 - <<'PYEOF' 2>/dev/null || echo "0"
import sqlite3, pathlib
db = pathlib.Path.home() / ".local/share/lessons-db/lessons.db"
if db.exists():
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM capture_drafts WHERE status='pending'").fetchone()[0]
    conn.close()
    print(n)
else:
    print(0)
PYEOF
)
if [[ "${PENDING_COUNT:-0}" -gt 50 ]]; then
    echo "Draft backlog: ${PENDING_COUNT} pending — run: lessons-db capture review --dry-run"
fi

# Surface top-3 contextually relevant lessons for current project.
# Mirrors the enter-plan hook — session start should show the same signal.
# Output format: SUGGESTION (negative) or PROVEN PATTERN (positive) via feedforward_format.
PROJ_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJ_NAME=$(basename "$PROJ_DIR")
CONTEXT_RESULTS=$("$LESSONS_DB" search "planning ${PROJ_NAME}" --top 3 2>/dev/null || echo "")
if [[ -n "$CONTEXT_RESULTS" && "$CONTEXT_RESULTS" != "No results found." ]]; then
    echo ""
    echo "Top lessons for ${PROJ_NAME}:"
    feedforward_format "$CONTEXT_RESULTS"

    # Record surfacing events for learning pipeline
    while IFS= read -r line; do
        if [[ "$line" =~ \[#([0-9]+)\] ]]; then
            "$LESSONS_DB" learn record \
                --lesson-id "${BASH_REMATCH[1]}" \
                --hook "session_start" \
                --context "$PROJ_NAME" \
                2>>/tmp/lessons-db-errors.log || true
        fi
    done <<< "$CONTEXT_RESULTS"
fi
