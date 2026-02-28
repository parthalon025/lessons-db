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

    # Record each surfaced lesson for the learning pipeline
    CONTEXT=$(echo "$NEW_CONTENT" | head -c 100)
    while IFS= read -r line; do
        if [[ "$line" =~ ^\[#([0-9]+)\] ]]; then
            LESSON_ID="${BASH_REMATCH[1]}"
            "$LESSONS_DB" learn record \
                --lesson-id "$LESSON_ID" \
                --hook "edit" \
                --context "$CONTEXT" \
                2>>/tmp/lessons-db-errors.log || true
        fi
    done <<< "$RESULTS"
fi
