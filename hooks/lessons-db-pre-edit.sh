#!/usr/bin/env bash
# PreToolUse:Edit|Write|MultiEdit hook — check new content against prevention pipeline.
# Exits 2 (block) when content matches a semgrep_error/semgrep_autofix lesson.
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")

if [[ -z "$LESSONS_DB" ]]; then
    exit 0
fi

# Read stdin once into a variable for both new_string and file_path extraction
INPUT=$(cat)

NEW_CONTENT=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('new_string', d.get('content', '')))
except Exception:
    print('')
" 2>/dev/null || echo "")

FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('file_path', d.get('path', '')))
except Exception:
    print('')
" 2>/dev/null || echo "")

if [[ -z "$NEW_CONTENT" ]]; then
    exit 0
fi

# Write content to a temp file to avoid ARG_MAX (~2MB) limit when passing content
# as a shell argument. Large generated files or minified assets would silently fail
# the argument-based approach, causing the fallback to always allow.
TMPFILE=$(mktemp /tmp/lessons-db-content.XXXXXX)
trap 'rm -f "$TMPFILE"' EXIT
printf '%s' "$NEW_CONTENT" > "$TMPFILE"

# Run the full prevention check (logs recurrence event, checks velocity, runs enforcement cycle)
# --file reads content from TMPFILE; FILE_PATH is passed separately for context metadata.
RESULT=$("$LESSONS_DB" prevent check-content \
    --file "$TMPFILE" \
    ${FILE_PATH:+--context-path "$FILE_PATH"} \
    --json 2>/dev/null || echo '{"block":false,"violations":[]}')

BLOCK=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('block',False))" 2>/dev/null || echo "False")

if [[ "$BLOCK" == "True" ]]; then
    # Extract human-readable message from JSON and emit to stderr
    MESSAGE=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('message','BLOCKED by lessons-db prevention'))" 2>/dev/null || echo "BLOCKED by lessons-db prevention")
    echo "$MESSAGE" >&2
    exit 2
fi

# Non-blocking: surface advisory violations to stderr (keeps tool output clean)
VIOLATIONS=$(echo "$RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for v in d.get('violations', []):
    print(f\"[#{v['lesson_id']}] [{v['enforcement']}] {v.get('one_liner','')}\")
" 2>/dev/null || echo "")

if [[ -n "$VIOLATIONS" ]]; then
    echo "$VIOLATIONS" >&2

    # Record each surfaced lesson for the learning pipeline
    while IFS= read -r line; do
        if [[ "$line" =~ ^\[#([0-9]+)\] ]]; then
            LESSON_ID="${BASH_REMATCH[1]}"
            "$LESSONS_DB" learn record \
                --lesson-id "$LESSON_ID" \
                --hook "edit" \
                --context "${FILE_PATH:-}" \
                2>>/tmp/lessons-db-errors.log || true
        fi
    done <<< "$VIOLATIONS"
fi
