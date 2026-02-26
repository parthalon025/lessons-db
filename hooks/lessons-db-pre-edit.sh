#!/usr/bin/env bash
# PreToolUse:Edit hook: check new content against detection patterns
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")

if [[ -z "$LESSONS_DB" ]]; then
    exit 0
fi

# Extract new_string from the Edit tool input
NEW_CONTENT=$(cat | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('new_string',''))" 2>/dev/null || echo "")

if [[ -z "$NEW_CONTENT" ]]; then
    exit 0
fi

# Check content against detection patterns
RESULTS=$("$LESSONS_DB" search "" --content "$NEW_CONTENT" 2>/dev/null || true)

if [[ -n "$RESULTS" && "$RESULTS" != "No results found." ]]; then
    echo "$RESULTS"
fi
