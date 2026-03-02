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

# Count negative lessons surfaced for 3:1 ratio balancing
count_lessons() {
    local text="$1"
    local count=0
    while IFS= read -r line; do
        if [[ "$line" =~ \[#([0-9]+)\] ]]; then
            count=$((count + 1))
        fi
    done <<< "$text"
    echo "$count"
}

NEGATIVE_COUNT=0
if [[ -n "$RESULTS1" ]]; then
    NEGATIVE_COUNT=$((NEGATIVE_COUNT + $(count_lessons "$RESULTS1")))
fi
if [[ -n "$RESULTS2" ]]; then
    NEGATIVE_COUNT=$((NEGATIVE_COUNT + $(count_lessons "$RESULTS2")))
fi

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

# Balance negative/positive surfacing at 3:1 ratio
# ceil(negative_count / 3), minimum 1
if [[ "$NEGATIVE_COUNT" -gt 0 ]]; then
    POSITIVE_LIMIT=$(( (NEGATIVE_COUNT + 2) / 3 ))  # integer ceiling division
else
    POSITIVE_LIMIT=1
fi
# Enforce minimum of 1 positive pattern
if [[ "$POSITIVE_LIMIT" -lt 1 ]]; then
    POSITIVE_LIMIT=1
fi

# Surface positive patterns (Bright Spots) for current project scope
# Uses --polarity positive to find what has worked well
RESULTS_POS=$("$LESSONS_DB" search "planning ${PROJ_NAME}" \
    --top "$POSITIVE_LIMIT" --polarity positive 2>/dev/null || echo "")

if [[ -n "$RESULTS_POS" ]]; then
    echo "### Bright Spots — Positive Patterns (what has worked well)"
    feedforward_format "$RESULTS_POS"
    echo ""
fi

# Pro-mortem prompt: shift from "what could go wrong" to "what patterns would make this succeed"
echo "---"
echo "**Pro-mortem: What patterns would make this feature succeed?**"
echo "Review the PROVEN PATTERNs above and identify which positive patterns to sustain and amplify."
echo ""

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
