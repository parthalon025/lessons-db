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
fi
