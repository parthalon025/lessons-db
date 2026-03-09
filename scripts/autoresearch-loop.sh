#!/usr/bin/env bash
# autoresearch-loop.sh — autonomous improvement loop
#
# Usage: ./scripts/autoresearch-loop.sh [--max-runs N] [--per-cluster N]
#
# Runs the propose → generate → judge → learn cycle until:
#   - Max runs reached (default: 10)
#   - No improvement for 3 consecutive runs
#   - Proposal system exhausts strategies
#
# Each iteration:
#   1. lessons-db meta eval-propose → proposed_variant.json
#   2. Inject config into variants.py (temporary X-variant)
#   3. Run autoresearch-run.sh with the proposed variant
#   4. Record result, loop

set -euo pipefail

MAX_RUNS=10
PER_CLUSTER=2
STALE_LIMIT=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-runs) MAX_RUNS="$2"; shift 2 ;;
        --per-cluster) PER_CLUSTER="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PROPOSAL_JSON="$HOME/.local/share/lessons-db/eval/proposed_variant.json"

cd "$PROJECT_DIR"
source .venv/bin/activate

stale_count=0
run=0

for run in $(seq 1 "$MAX_RUNS"); do
    echo ""
    echo "=== autoresearch-loop: run $run/$MAX_RUNS (stale=$stale_count/$STALE_LIMIT) ==="

    # 1. Propose next variant
    echo "[1/4] Proposing next variant..."
    if ! lessons-db meta eval-propose 2>/dev/null; then
        echo "Proposal failed — running best variant for stability data"
        VARIANT="$(python3 -c "import json; print(json.load(open('$HOME/.local/share/lessons-db/eval/best.json')).get('variant', 'A'))")"
    else
        VARIANT="$(python3 -c "import json; print(json.load(open('$PROPOSAL_JSON'))['variant_id'])")"

        # 2. Inject proposed config into variants.py
        echo "[2/4] Injecting $VARIANT into variants.py..."
        python3 -c "
import json
proposal = json.load(open('$PROPOSAL_JSON'))
vid = proposal['variant_id']
config = proposal['config']

vfile = '$PROJECT_DIR/src/lessons_db/eval/variants.py'
with open(vfile) as f:
    content = f.read()

# Only inject if not already present
if '\"$VARIANT\"' not in content:
    # Find the closing brace of VARIANT_CONFIGS
    marker = '}  # end VARIANT_CONFIGS'
    if marker not in content:
        # Fallback: insert before last closing brace
        idx = content.rindex('}')
        insert = f'    \"{vid}\": {json.dumps(config, indent=8)},\n'
        content = content[:idx] + insert + content[idx:]
    else:
        content = content.replace(marker, f'    \"{vid}\": {json.dumps(config, indent=8)},\n' + marker)
    with open(vfile, 'w') as f:
        f.write(content)
    print(f'Injected {vid} into variants.py')
else:
    print(f'{vid} already in variants.py')
"
    fi

    # 3. Run experiment
    echo "[3/4] Running experiment: $VARIANT --per-cluster $PER_CLUSTER..."
    EXIT_CODE=0
    "$SCRIPT_DIR/autoresearch-run.sh" "$VARIANT" --per-cluster "$PER_CLUSTER" || EXIT_CODE=$?

    # 4. Interpret result
    case $EXIT_CODE in
        0)
            echo "IMPROVED — $VARIANT is new best!"
            stale_count=0
            ;;
        1)
            echo "No improvement from $VARIANT"
            stale_count=$((stale_count + 1))
            ;;
        2)
            echo "CRASH — $VARIANT failed"
            stale_count=$((stale_count + 1))
            ;;
    esac

    if [[ $stale_count -ge $STALE_LIMIT ]]; then
        echo "Stopping: $STALE_LIMIT consecutive runs without improvement."
        break
    fi
done

echo ""
echo "=== autoresearch-loop complete: $run runs ==="
echo "Results: $PROJECT_DIR/results.tsv"
