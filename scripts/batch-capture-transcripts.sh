#!/usr/bin/env bash
# Batch-capture lessons from past Claude Code session transcripts.
# Reads ~/.claude/projects/**/*.jsonl (top-level sessions only, not subagents).
# Extracts user/assistant text, passes to `lessons-db capture transcript`.
#
# Usage:
#   ./scripts/batch-capture-transcripts.sh [--dry-run] [--since YYYY-MM-DD] [--positive]
#
# Options:
#   --dry-run     Print what would be processed without calling lessons-db
#   --since DATE  Only process sessions modified after DATE (default: all)
#   --positive    Extract positive patterns (what worked well) — use with deepseek-r1:8b-0528-qwen3-q4_K_M
#
# Examples:
#   # Default: extract failure lessons with current ANALYSIS_MODEL
#   ./scripts/batch-capture-transcripts.sh
#
#   # Positive sweep with deepseek-r1 (qwen3-based checkpoint — best for positive patterns)
#   LESSONS_DB_OLLAMA_ANALYSIS_MODEL=deepseek-r1:8b-0528-qwen3-q4_K_M ./scripts/batch-capture-transcripts.sh --positive
#
# Output: progress to stderr, capture results to stdout.
# Estimated time: ~30-60s per session (Ollama inference). Run in tmux.

set -uo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")
if [[ -z "$LESSONS_DB" ]]; then
    echo "ERROR: lessons-db not found on PATH" >&2
    exit 1
fi

DRY_RUN=false
SINCE=""
POSITIVE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true ;;
        --positive) POSITIVE=true ;;
        --since)    SINCE="$2"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
    shift
done

PROJ_DIR="$HOME/.claude/projects"
TMPFILE=$(mktemp /tmp/lessons-db-transcript-XXXXXX.txt)
trap 'rm -f "$TMPFILE"' EXIT

# Extract text from a session JSONL — user and assistant turns only
extract_transcript() {
    local path="$1"
    python3 - "$path" <<'PYEOF'
import sys, json

path = sys.argv[1]
texts = []
try:
    with open(path, errors='replace') as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get('type') in ('user', 'assistant'):
                    msg = d.get('message', {})
                    for block in msg.get('content', []):
                        if isinstance(block, dict) and block.get('type') == 'text':
                            t = block.get('text', '').strip()
                            if t and len(t) > 20:
                                role = d['type'].upper()
                                texts.append(f'[{role}] {t}')
            except Exception:
                pass
except Exception as e:
    print(f"ERROR reading {path}: {e}", file=sys.stderr)

print('\n'.join(texts))
PYEOF
}

# Collect sessions — top-level only (maxdepth 2 skips subagent dirs)
mapfile -t SESSIONS < <(find "$PROJ_DIR" -maxdepth 2 -name "*.jsonl" | sort)

if [[ -n "$SINCE" ]]; then
    mapfile -t SESSIONS < <(find "$PROJ_DIR" -maxdepth 2 -name "*.jsonl" -newer <(touch -d "$SINCE" /tmp/since-ref && echo /tmp/since-ref) | sort)
fi

TOTAL=${#SESSIONS[@]}
echo "Sessions to process: $TOTAL" >&2
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN — will not call lessons-db" >&2
fi

OK=0
SKIPPED=0
FAILED=0
CAPTURED=0

for i in "${!SESSIONS[@]}"; do
    SESSION="${SESSIONS[$i]}"
    NUM=$((i + 1))
    NAME=$(basename "$(dirname "$SESSION")")/$(basename "$SESSION")

    echo -n "[$NUM/$TOTAL] $NAME ... " >&2

    TEXT=$(extract_transcript "$SESSION")
    CHARS=${#TEXT}

    if [[ "$CHARS" -lt 100 ]]; then
        echo "skip (${CHARS} chars)" >&2
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "would capture (${CHARS} chars)" >&2
        OK=$((OK + 1))
        continue
    fi

    echo "$TEXT" > "$TMPFILE"

    POSITIVE_FLAG=""
    [[ "$POSITIVE" == "true" ]] && POSITIVE_FLAG="--positive"

    OUTPUT=$("$LESSONS_DB" capture transcript $POSITIVE_FLAG "$TMPFILE" 2>&1)
    EXIT=$?

    if [[ $EXIT -eq 0 ]]; then
        COUNT=$(echo "$OUTPUT" | grep -oE '[0-9]+' | head -1 || echo "0")
        echo "ok — ${COUNT} draft(s) captured (${CHARS} chars)" >&2
        CAPTURED=$((CAPTURED + COUNT))
        OK=$((OK + 1))
    else
        echo "FAILED: $OUTPUT" >&2
        FAILED=$((FAILED + 1))
    fi

    # Brief pause to avoid hammering Ollama
    sleep 1
done

echo "" >&2
echo "Done: $OK processed, $SKIPPED skipped, $FAILED failed" >&2
echo "Total drafts captured: $CAPTURED" >&2
echo "Review with: lessons-db capture drafts" >&2
