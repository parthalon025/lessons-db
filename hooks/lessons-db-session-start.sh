#!/usr/bin/env bash
# SessionStart hook: show lessons-db status line, resolve stale outcomes, show fix queue
set -euo pipefail

# Surface CLAUDE.md quality gate results from the previous session (written by validate-on-clear.sh)
VALIDATE_RESULTS="$HOME/.claude/.validate-results"
if [[ -f "$VALIDATE_RESULTS" ]]; then
    echo "CLAUDE.md QUALITY GATE (from last session):"
    echo ""
    cat "$VALIDATE_RESULTS"
    echo ""
    echo "Fix issues above or run: bash ~/Documents/scripts/claude-md-validate.sh --verbose"
    echo "---"
    rm -f "$VALIDATE_RESULTS"
fi

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

# SFBT exception reporting: surface internalized anti-patterns (variable-ratio, 30%)
# Uses variable-ratio probability gate (30%, Skinner schedule).
# Write to temp file to avoid $() heredoc paren-counting issues in bash.
EXCEPTION_TMP=$(mktemp /tmp/lessons-exceptions.XXXXXX 2>/dev/null || echo "")
if [[ -n "$EXCEPTION_TMP" ]]; then
    python3 - "$EXCEPTION_TMP" <<'PYEOF' 2>/dev/null || true
import sqlite3, random, pathlib, sys

outfile = sys.argv[1]
db = pathlib.Path.home() / ".local/share/lessons-db/lessons.db"
if not db.exists():
    sys.exit(0)

# Variable-ratio scheduling: 30% probability gate
random.seed()
if random.random() >= 0.3:
    sys.exit(0)

conn = sqlite3.connect(str(db))
conn.row_factory = sqlite3.Row

# Get 5 most recent distinct session_ids
recent = conn.execute(
    "SELECT session_id, MAX(timestamp) AS latest "
    "FROM surfacing_events WHERE session_id IS NOT NULL "
    "GROUP BY session_id ORDER BY latest DESC LIMIT 5"
).fetchall()

if not recent:
    conn.close()
    sys.exit(0)

session_ids = [r["session_id"] for r in recent]
num_recent = len(session_ids)
placeholders = ", ".join(["?"] * num_recent)

# Find negative lessons dismissed historically but absent from recent sessions
historically = conn.execute(
    "SELECT DISTINCT se.lesson_id, l.title, l.category "
    "FROM surfacing_events se JOIN lessons l ON l.id = se.lesson_id "
    "WHERE se.outcome = 'dismissed' AND l.polarity = 'negative'"
).fetchall()

found = []
for row in historically:
    lid = row["lesson_id"]
    cnt = conn.execute(
        "SELECT COUNT(DISTINCT session_id) FROM surfacing_events "
        "WHERE lesson_id = ? AND outcome = 'dismissed' AND session_id IN (%s)" % placeholders,
        [lid] + session_ids
    ).fetchone()[0]
    if cnt == 0:
        cat = row["category"] or "uncategorized"
        found.append([lid, row["title"], cat, num_recent])

if found:
    with open(outfile, "w") as f:
        for item in sorted(found, key=lambda e: [-e[3], e[0]]):
            lid, title, cat, n = item
            f.write("%d-session streak: zero %s issues — [#%d] %s\n" % (n, cat, lid, title))
            # Record surfacing event with outcome=exception_noted
            conn.execute(
                "INSERT INTO surfacing_events "
                "(lesson_id, hook_point, context, outcome, timestamp) "
                "VALUES (?, 'session_start_exception', ?, 'exception_noted', datetime('now'))",
                [lid, "absent_%d_sessions" % n]
            )
    conn.commit()

conn.close()
PYEOF

    if [[ -s "$EXCEPTION_TMP" ]]; then
        echo ""
        echo "Internalized patterns (SFBT exceptions):"
        cat "$EXCEPTION_TMP"
    fi
    rm -f "$EXCEPTION_TMP"
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

# FSRS spaced-repetition review: surface due lessons with fading-level verbosity.
# Interleaved by cluster (Bjork desirable difficulties) and filtered by fading level.
FSRS_DUE=$("$LESSONS_DB" fsrs due --threshold 0.9 2>/dev/null || echo "")
if [[ -n "$FSRS_DUE" && "$FSRS_DUE" != "No lessons due for review." ]]; then
    # Extract due count from first line
    DUE_COUNT=$(echo "$FSRS_DUE" | head -1 | grep -oP '\d+' || echo "0")
    if [[ "${DUE_COUNT:-0}" -gt 0 ]]; then
        echo ""
        echo "FSRS review due: ${DUE_COUNT} lessons"

        # Show lessons based on fading level:
        #   full   -> full line (title + stability + retrievability)
        #   brief  -> one-liner only
        #   silent -> suppressed (Semgrep-only)
        #   enforced -> suppressed
        while IFS= read -r line; do
            if [[ "$line" =~ \[#([0-9]+)\] ]]; then
                LESSON_ID="${BASH_REMATCH[1]}"
                if [[ "$line" =~ level=full ]]; then
                    # Full presentation: show entire line
                    echo "$line"
                    # Record surfacing event
                    "$LESSONS_DB" learn record \
                        --lesson-id "$LESSON_ID" \
                        --hook "session_start_fsrs" \
                        --context "fsrs_review" \
                        2>>/tmp/lessons-db-errors.log || true
                elif [[ "$line" =~ level=brief ]]; then
                    # Brief: just the title
                    TITLE=$(echo "$line" | sed 's/.*\] //' | sed 's/  S=.*//')
                    echo "  [#${LESSON_ID}] ${TITLE}"
                    "$LESSONS_DB" learn record \
                        --lesson-id "$LESSON_ID" \
                        --hook "session_start_fsrs" \
                        --context "fsrs_review" \
                        2>>/tmp/lessons-db-errors.log || true
                fi
                # silent and enforced: suppressed — no output, no surfacing event
            fi
        done <<< "$FSRS_DUE"
    fi
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

exit 0
