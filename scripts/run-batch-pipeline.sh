#!/usr/bin/env bash
# Run all batch-capture passes sequentially — one model at a time.
# Uses kill -0 polling (not `wait`) so chaining works across subshells.
#
# Passes (in order):
#   1. qwen2.5:7b        — negative, all sessions (fast baseline)
#   2. qwen2.5-coder:14b — negative, all sessions (code-aware)
#   3. deepseek-r1:*     — positive, all sessions (reasoning model)
#   4. deepseek-r1:*     — negative, all sessions (reasoning model)
#
# Usage: bash scripts/run-batch-pipeline.sh [--since YYYY-MM-DD]
# Log:   tail -f /tmp/lessons-batch-capture.log

set -uo pipefail

SCRIPT="$(dirname "$0")/batch-capture-transcripts.sh"
LOG="/tmp/lessons-batch-capture.log"
DEEPSEEK="deepseek-r1:8b-0528-qwen3-q4_K_M"
SINCE_ARG="${1:-}"

run_pass() {
    local label="$1"
    local model="$2"
    local mode="$3"   # "" or "--positive"

    {
        echo "[$(date)] === Starting: $label ==="

        # Build args array — avoids passing empty strings as positional args
        local args=()
        [[ -n "$mode" ]] && args+=("$mode")
        [[ -n "$SINCE_ARG" ]] && args+=("$SINCE_ARG")

        LESSONS_DB_OLLAMA_ANALYSIS_MODEL="$model" \
            bash "$SCRIPT" "${args[@]}" 2>&1

        echo "[$(date)] === Done: $label ==="
        echo ""
    } >> "$LOG"
}

echo "[$(date)] Pipeline starting — 4 passes" >> "$LOG"

run_pass "Pass 1: qwen2.5:7b negative"        "qwen2.5:7b"            ""
run_pass "Pass 2: qwen2.5-coder:14b negative" "qwen2.5-coder:14b"     ""
run_pass "Pass 3: deepseek-r1 positive"        "$DEEPSEEK"             "--positive"
run_pass "Pass 4: deepseek-r1 negative"        "$DEEPSEEK"             ""

echo "[$(date)] All passes complete." >> "$LOG"
echo "Review drafts: lessons-db capture drafts" >> "$LOG"
