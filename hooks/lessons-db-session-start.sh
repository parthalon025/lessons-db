#!/usr/bin/env bash
# SessionStart hook: show lessons-db status line (~20 tokens)
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")

if [[ -z "$LESSONS_DB" ]]; then
    exit 0  # lessons-db not installed yet
fi

"$LESSONS_DB" status 2>/dev/null || true
