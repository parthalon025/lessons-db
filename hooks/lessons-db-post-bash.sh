#!/usr/bin/env bash
# lessons-db PostToolUse hook — surface lessons on test failures
# Triggered after every Bash tool call. Reads JSON from stdin with keys:
#   tool_name, tool_input (object with .command), tool_result (string output)
# Exit 0 always — advisory only, never blocks.
set -uo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")
if [[ -z "${LESSONS_DB}" ]]; then
    exit 0
fi

# Read full JSON from stdin
INPUT=$(cat)

# Extract tool_result (the bash output) — field is tool_result, not tool_response
# tool_result may be a string or a dict with an 'output' key
RESPONSE=$(echo "${INPUT}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    resp = data.get('tool_result', '')
    if isinstance(resp, dict):
        print(resp.get('output', ''))
    else:
        print(resp)
except Exception:
    pass
" 2>/dev/null || echo "")

# Extract the command that was run
COMMAND=$(echo "${INPUT}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('tool_input', {}).get('command', ''))
except Exception:
    pass
" 2>/dev/null || echo "")

# If user ran a dismiss command, confirm it was recorded
if echo "${COMMAND}" | grep -qE 'lessons-db.*(dismiss)'; then
    echo "[lessons-db] False positive recorded."
    exit 0
fi

# Check if the output looks like a test failure
# Match pytest, npm test, go test, cargo test patterns
if ! echo "${RESPONSE}" | grep -qE 'FAILED|ERRORS|AssertionError|Error:|failed [0-9]+ test|[0-9]+ error'; then
    exit 0
fi

# Extract the most informative failure lines (first FAILED or Error lines)
FAILURE_LINE=$(echo "${RESPONSE}" | grep -E 'FAILED|AssertionError|Error:' | head -3 | tr '\n' ' ')

if [[ -z "${FAILURE_LINE}" ]]; then
    exit 0
fi

# Truncate to reasonable search query length
QUERY=$(echo "${FAILURE_LINE}" | cut -c1-150)

# Search for matching lessons — use --top/-k flag (not --limit)
RESULTS=$("${LESSONS_DB}" search "${QUERY}" --top 3 2>/dev/null || echo "")

if [[ -n "${RESULTS}" ]]; then
    echo ""
    echo "## Lessons-DB: Relevant lessons for this failure"
    echo "\`\`\`"
    echo "${RESULTS}"
    echo "\`\`\`"
    echo ""

    # Record each surfaced lesson for the learning pipeline (Lesson #65: wire call sites)
    while IFS= read -r line; do
        if [[ "${line}" =~ ^\[#([0-9]+)\] ]]; then
            LESSON_ID="${BASH_REMATCH[1]}"
            "${LESSONS_DB}" learn record \
                --lesson-id "${LESSON_ID}" \
                --hook "bash" \
                --context "${QUERY}" \
                2>>/tmp/lessons-db-errors.log || true
        fi
    done <<< "${RESULTS}"
fi

exit 0
