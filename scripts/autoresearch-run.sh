#!/usr/bin/env bash
# autoresearch-run.sh — run one eval experiment, decide keep/discard, always learn
#
# Usage: ./scripts/autoresearch-run.sh <VARIANT_ID> [--per-cluster N]
#   VARIANT_ID    — e.g. F, G, X01
#   --per-cluster — lessons per cluster (default: 2 for fast signal)
#
# Best F1 is read from best.json — no need to pass it manually.
# Exits 0 if kept (new best), 1 if discarded, 2 if crashed.
# Learning step ALWAYS runs regardless of outcome.

set -euo pipefail

VARIANT="${1:?Usage: $0 <VARIANT_ID> [--per-cluster N]}"
PER_CLUSTER=2  # fast default — enough for directional signal

# Parse optional --per-cluster flag
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --per-cluster) PER_CLUSTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_TSV="$PROJECT_DIR/results.tsv"
BEST_JSON="$HOME/.local/share/lessons-db/eval/best.json"
GEN_OUT="/tmp/ar-${VARIANT}.json"
REPORT_OUT="/tmp/ar-${VARIANT}-report.md"
LOG_OUT="/tmp/ar-${VARIANT}.log"

cd "$PROJECT_DIR"
source .venv/bin/activate

# Read current best F1 from best.json (written by learn.py after each judge run)
_read_best_f1() {
    if [[ -f "$BEST_JSON" ]]; then
        python3 -c "import json; d=json.load(open('$BEST_JSON')); print(d.get('f1', 0.0))" 2>/dev/null || echo "0.0"
    else
        echo "0.0"
    fi
}

# Append a crash learning note via learn.py (reuses the same code path)
_learn_crash() {
    local variant="$1" reason="$2"
    python3 -c "
from pathlib import Path
from lessons_db.eval.learn import append_to_program_md
insight = {
    'date': '$(date +%Y-%m-%d)',
    'variant': '$variant',
    'summary': 'CRASH',
    'diagnosis': '$reason',
    'recommendation': 'fix config before retrying',
}
p = Path('$PROJECT_DIR/program.md')
if append_to_program_md([insight], p):
    print('program.md updated with crash note')
" 2>/dev/null || echo "Warning: crash learning failed (non-fatal)"
}

BEST_F1_BEFORE="$(_read_best_f1)"
echo "=== autoresearch: variant $VARIANT (per-cluster=$PER_CLUSTER, best so far: $BEST_F1_BEFORE) ==="
echo "Started: $(date)"

# --- 1. Generate ---
echo "[1/3] eval-generate --variants $VARIANT --per-cluster $PER_CLUSTER..."
if ! lessons-db meta eval-generate \
    --variants "$VARIANT" \
    --per-cluster "$PER_CLUSTER" \
    --output "$GEN_OUT" > "$LOG_OUT" 2>&1; then
    echo "ERROR: eval-generate failed. Tail of log:"
    tail -20 "$LOG_OUT"
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'none')"
    printf "%s\t%s\t0.000\t0.000\t0.000\tcrash\tgenerate failed\n" \
        "$COMMIT" "$VARIANT" >> "$RESULTS_TSV"
    _learn_crash "$VARIANT" "eval-generate failed — see $LOG_OUT"
    exit 2
fi

if [[ ! -s "$GEN_OUT" ]]; then
    echo "ERROR: results file is empty: $GEN_OUT"
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'none')"
    printf "%s\t%s\t0.000\t0.000\t0.000\tcrash\tempty results JSON\n" \
        "$COMMIT" "$VARIANT" >> "$RESULTS_TSV"
    _learn_crash "$VARIANT" "eval-generate produced empty JSON"
    exit 2
fi

# --- 2. Judge (learning step runs automatically inside eval-judge) ---
echo "[2/3] eval-judge (learning auto-runs)..."
if ! lessons-db meta eval-judge "$GEN_OUT" --output "$REPORT_OUT" >> "$LOG_OUT" 2>&1; then
    echo "ERROR: eval-judge failed. Tail of log:"
    tail -20 "$LOG_OUT"
    COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'none')"
    printf "%s\t%s\t0.000\t0.000\t0.000\tcrash\tjudge failed\n" \
        "$COMMIT" "$VARIANT" >> "$RESULTS_TSV"
    _learn_crash "$VARIANT" "eval-judge failed — see $LOG_OUT"
    exit 2
fi

# --- 3. Read result from best.json (written by learn.py during judge) ---
echo "[3/3] reading result..."
BEST_F1_AFTER="$(_read_best_f1)"

# Extract this variant's metrics from the report for TSV logging
VARIANT_ROW="$(grep "^| $VARIANT " "$REPORT_OUT" 2>/dev/null | head -1)"
if [[ -n "$VARIANT_ROW" ]]; then
    RECALL="$(echo "$VARIANT_ROW" | awk -F'|' '{gsub(/ /,"",$3); print $3}')"
    PRECISION="$(echo "$VARIANT_ROW" | awk -F'|' '{gsub(/ /,"",$4); print $4}')"
    F1="$(echo "$VARIANT_ROW" | awk -F'|' '{gsub(/ /,"",$5); print $5}')"
else
    RECALL="0.000"; PRECISION="0.000"; F1="0.000"
fi

COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'none')"
GIT_MSG="$(git log -1 --pretty=%s 2>/dev/null | head -c 60 || echo 'no commit')"

# Keep if this variant's F1 exceeds the prior best (compare numbers, not names)
IMPROVED="$(python3 -c "print('yes' if float('$F1') > float('$BEST_F1_BEFORE') else 'no')" 2>/dev/null || echo 'no')"

if [[ "$IMPROVED" == "yes" ]]; then
    echo "KEEP: $VARIANT is new best (F1=$F1, was $BEST_F1_BEFORE)"
    printf "%s\t%s\t%s\t%s\t%s\tkeep\t%s\n" \
        "$COMMIT" "$VARIANT" "$F1" "$PRECISION" "$RECALL" "$GIT_MSG" >> "$RESULTS_TSV"
    exit 0
else
    echo "DISCARD: $VARIANT did not beat best (F1=$F1 vs best=$BEST_F1_AFTER)"
    printf "%s\t%s\t%s\t%s\t%s\tdiscard\t%s\n" \
        "$COMMIT" "$VARIANT" "$F1" "$PRECISION" "$RECALL" "$GIT_MSG" >> "$RESULTS_TSV"
    # Revert variants.py for experimental (X*) variants only — surgical restore, not reset
    if [[ "$VARIANT" == X* ]]; then
        echo "Reverting variants.py injection for $VARIANT..."
        git restore src/lessons_db/eval/variants.py
    fi
    exit 1
fi
