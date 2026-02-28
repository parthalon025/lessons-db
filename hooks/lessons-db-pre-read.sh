#!/usr/bin/env bash
# PreToolUse:Read hook: surface lessons for the file being read (~30 tokens/hit)
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")

if [[ -z "$LESSONS_DB" ]]; then
    exit 0
fi

# The file path comes from the tool input (Claude Code passes it as JSON on stdin)
FILE_PATH=$(cat | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('file_path',''))" 2>/dev/null || echo "")

if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

# Search for lessons affecting this file
RESULTS=$("$LESSONS_DB" search "" --file "$FILE_PATH" 2>/dev/null || true)

if [[ -n "$RESULTS" && "$RESULTS" != "No results found." ]]; then
    echo "$RESULTS"

    # Record each surfaced lesson for the learning pipeline
    while IFS= read -r line; do
        if [[ "$line" =~ ^\[#([0-9]+)\] ]]; then
            LESSON_ID="${BASH_REMATCH[1]}"
            "$LESSONS_DB" learn record \
                --lesson-id "$LESSON_ID" \
                --hook "read" \
                --context "$FILE_PATH" \
                2>>/tmp/lessons-db-errors.log || true
        fi
    done <<< "$RESULTS"
fi
