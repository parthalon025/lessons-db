#!/usr/bin/env bash
# lessons-db PreToolUse hook — semantic lesson search before EnterPlanMode
set -uo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")
if [[ -z "$LESSONS_DB" ]]; then
    exit 0
fi

# Read tool input from stdin (JSON) — EnterPlanMode has no arguments, discard it
INPUT=$(cat)

# Derive context from the project directory
PROJ_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJ_NAME=$(basename "$PROJ_DIR")

# Run 2 focused semantic searches using the correct --top flag
RESULTS1=$("$LESSONS_DB" search "planning ${PROJ_NAME}" --top 3 2>/dev/null || echo "")
RESULTS2=$("$LESSONS_DB" search "integration boundary silent failure" --top 3 2>/dev/null || echo "")

# If we got results, output them as a formatted message Claude will see
if [[ -n "$RESULTS1" || -n "$RESULTS2" ]]; then
    echo "## Relevant Lessons (from lessons-db) — Check Before Planning"
    echo ""
    if [[ -n "$RESULTS1" ]]; then
        echo "### Query: planning ${PROJ_NAME}"
        echo "$RESULTS1"
        echo ""
    fi
    if [[ -n "$RESULTS2" ]]; then
        echo "### Query: integration boundary silent failure"
        echo "$RESULTS2"
        echo ""
    fi
fi

exit 0
