#!/usr/bin/env bash
# SessionStart hook: show lessons-db status line and positive pattern count
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")

if [[ -z "$LESSONS_DB" ]]; then
    exit 0  # lessons-db not installed yet
fi

"$LESSONS_DB" status 2>/dev/null || true

# Show count of positive patterns (polarity=positive) as signal of accumulated knowledge
POS_COUNT=$("$LESSONS_DB" search "" --polarity positive --top 100 2>/dev/null \
    | grep -c '^\[#' || echo "0")

if [[ "$POS_COUNT" -gt 0 ]]; then
    echo "Positive patterns available: ${POS_COUNT}"
fi
