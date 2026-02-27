#!/usr/bin/env bash
# lessons-db Stop hook — auto-capture lessons from session diff
# Runs at Claude Code session end. Exits 0 always — never blocks session.
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")
if [[ -z "$LESSONS_DB" ]]; then
    exit 0
fi

# Get the working directory from CLAUDE_PROJECT_DIR env var (set by Claude Code)
PROJ="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$PROJ" 2>/dev/null || exit 0

# Check we're in a git repo
git rev-parse --git-dir &>/dev/null || exit 0

# Get diff stat to estimate size (staged + unstaged vs HEAD)
DIFF_STAT=$(git diff HEAD --stat 2>/dev/null || echo "")
STAT_LINES=$(echo "$DIFF_STAT" | wc -l)

# Only proceed if the stat output is non-trivial (>5 lines of stat = meaningful change set)
if [[ "$STAT_LINES" -lt 5 ]]; then
    exit 0
fi

# Get the full diff for analysis
FULL_DIFF=$(git diff HEAD 2>/dev/null || echo "")

if [[ -z "$FULL_DIFF" ]]; then
    exit 0
fi

# Count actual changed lines (+ and - lines) to enforce the 50-line threshold
CHANGED_LINES=$(echo "$FULL_DIFF" | grep -cE '^[+-]' 2>/dev/null || echo 0)
if [[ "$CHANGED_LINES" -lt 50 ]]; then
    exit 0
fi

# Write diff to temp file to avoid argument length limits
TMPFILE=$(mktemp /tmp/lessons-db-diff-XXXXXX.txt)
echo "$FULL_DIFF" > "$TMPFILE"

# Capture lessons from diff — non-blocking, session must not fail if this errors
"$LESSONS_DB" capture diff "$TMPFILE" 2>/tmp/lessons-db-stop-hook.log || true

rm -f "$TMPFILE"

# Check for newly-created design docs this session and capture positive patterns
# Use -mmin -120 to find docs created/modified in the last 2 hours (session scope)
if [[ -d "${PROJ}/docs/plans" ]]; then
    while IFS= read -r -d '' doc_file; do
        "$LESSONS_DB" capture design-doc "$doc_file" \
            2>>/tmp/lessons-db-stop-positive.log || true
    done < <(find "${PROJ}/docs/plans" -name "*.md" -mmin -120 -print0 2>/dev/null)
fi

exit 0
