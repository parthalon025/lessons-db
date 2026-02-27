#!/usr/bin/env bash
GENERATE="$(command -v generate-embeddings 2>/dev/null || echo "$HOME/.local/bin/generate-embeddings")"
# shellcheck disable=SC2015  # exec short-circuits || branch on success; pattern is safe
[ -x "$GENERATE" ] && exec "$GENERATE" "$@" || { echo "generate-embeddings not found"; exit 1; }
