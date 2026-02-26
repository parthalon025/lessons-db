#!/usr/bin/env bash
# lessons-db PostToolUse hook — surface lessons on test failures
# Triggered after every Bash tool call. Reads JSON from stdin with keys:
#   tool_name, tool_input (object with .command), tool_result (string output)
# Exit 0 always — advisory only, never blocks.
set -uo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")
if [[ -z "$LESSONS_DB" ]]; then
    exit 0
fi

# Read full JSON from stdin
INPUT=$(cat)

# Extract tool_result (the bash output) — field is tool_result, not tool_response
# tool_result may be a string or a dict with an 'output' key
RESPONSE=$(echo "$INPUT" | python3 -c "
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

# Check if the output looks like a test failure
# Match pytest, npm test, go test, cargo test patterns
if ! echo "$RESPONSE" | grep -qE 'FAILED|ERRORS|AssertionError|Error:|failed [0-9]+ test|[0-9]+ error'; then
    exit 0
fi

# Extract the most informative failure lines (first FAILED or Error lines)
FAILURE_LINE=$(echo "$RESPONSE" | grep -E 'FAILED|AssertionError|Error:' | head -3 | tr '\n' ' ')

if [[ -z "$FAILURE_LINE" ]]; then
    exit 0
fi

# Truncate to reasonable search query length
QUERY=$(echo "$FAILURE_LINE" | cut -c1-150)

# Search for matching lessons — use --top/-k flag (not --limit)
RESULTS=$("$LESSONS_DB" search "$QUERY" --top 3 2>/dev/null || echo "")

if [[ -n "$RESULTS" ]]; then
    echo ""
    echo "## Lessons-DB: Relevant lessons for this failure"
    echo "\`\`\`"
    echo "$RESULTS"
    echo "\`\`\`"
    echo ""
fi

exit 0
