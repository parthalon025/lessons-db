#!/usr/bin/env bash
# lessons-db PreToolUse hook — semantic lesson search before EnterPlanMode
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")
if [[ -z "$LESSONS_DB" ]]; then
    exit 0
fi

# Source shared feedforward formatting
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hooks/_feedforward-format.sh
source "${HOOK_DIR}/_feedforward-format.sh"

# Read tool input from stdin (JSON) — EnterPlanMode has no arguments, discard it
# shellcheck disable=SC2034  # _INPUT drains stdin; hook framework requires consuming it
_INPUT=$(cat)

# Derive context from the project directory
PROJ_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
PROJ_NAME=$(basename "$PROJ_DIR")

# Run 2 focused semantic searches using the correct --top flag
RESULTS1=$("$LESSONS_DB" search "planning ${PROJ_NAME}" --top 3 2>/dev/null || echo "")
RESULTS2=$("$LESSONS_DB" search "integration boundary silent failure" --top 3 2>/dev/null || echo "")

# If we got results, output them with feedforward framing
if [[ -n "$RESULTS1" || -n "$RESULTS2" ]]; then
    echo "## Consider these proven patterns for ${PROJ_NAME}:"
    echo ""
    if [[ -n "$RESULTS1" ]]; then
        echo "### Query: planning ${PROJ_NAME}"
        feedforward_format "$RESULTS1"
        echo ""
    fi
    if [[ -n "$RESULTS2" ]]; then
        echo "### Query: integration boundary silent failure"
        feedforward_format "$RESULTS2"
        echo ""
    fi
fi

# Surface top-3 positive entries by semantic similarity
RESULTS_POS=$("$LESSONS_DB" search "planning ${PROJ_NAME}" \
    --top 3 --polarity positive 2>/dev/null || echo "")

if [[ -n "$RESULTS_POS" ]]; then
    echo "### Positive Patterns (what has worked well)"
    feedforward_format "$RESULTS_POS"
    echo ""
fi

# Record each surfaced lesson for the learning pipeline (Lesson #65: wire call sites)
# Extract IDs from search output format: [#NNN] one_liner (raw, pre-feedforward)
# Includes RESULTS_POS so positive pattern surfacing events are also tracked.
for RESULT_BLOCK in "$RESULTS1" "$RESULTS2" "$RESULTS_POS"; do
    while IFS= read -r line; do
        if [[ "$line" =~ \[#([0-9]+)\] ]]; then
            LESSON_ID="${BASH_REMATCH[1]}"
            "$LESSONS_DB" learn record \
                --lesson-id "$LESSON_ID" \
                --hook "plan" \
                --context "$PROJ_NAME" \
                2>>/tmp/lessons-db-errors.log || true
        fi
    done <<< "$RESULT_BLOCK"
done

exit 0
