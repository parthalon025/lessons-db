#!/bin/bash
# Run eval-generate with resume — fills open slots until all (variant, lesson) pairs complete.
# Uses a fixed output path so --resume can find the previous run's progress.
# Registered as recurring job "lessons-db-eval-generate" (every 30m, priority 5)

set -euo pipefail

LESSONS_DB="$HOME/Documents/projects/lessons-db/.venv/bin/lessons-db"
PYTHON="$HOME/Documents/projects/lessons-db/.venv/bin/python3"
DB_PATH="$HOME/.local/share/lessons-db/lessons.db"
OUTPUT="$HOME/.local/share/lessons-db/eval/current-run.json"
LOG="$HOME/.local/share/lessons-db/meta-eval-generate.log"

mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Short-circuit: if output exists, count completed pairs and skip if all done
if [ -f "$OUTPUT" ]; then
    completed=$(python3 -c "
import json, sys
with open('$OUTPUT') as f:
    d = json.load(f)
results = [r for r in d.get('results', []) if not r.get('error') and r.get('principle')]
print(len(results))
" 2>/dev/null || echo 0)

    # Count expected pairs: SUM(MIN(cluster_size, 4)) * 5 variants
    max_expected=$("$PYTHON" -c "
import sqlite3
c = sqlite3.connect('$DB_PATH')
rows = c.execute('''
    SELECT MIN(COUNT(*), 4) as cnt FROM lessons
    WHERE cluster_seed IS NOT NULL AND (loop_level IS NULL OR loop_level = \"single\")
    GROUP BY cluster_seed HAVING COUNT(*) >= 3
''').fetchall()
print(sum(r[0] for r in rows) * 5)
" 2>/dev/null || echo 9999)

    if [ "$completed" -ge "$max_expected" ] 2>/dev/null; then
        log "All $completed pairs complete — nothing to do"
        exit 2
    fi
    log "=== eval-generate start: $completed/$max_expected pairs complete, resuming ==="
else
    log "=== eval-generate start: fresh run ==="
fi

"$LESSONS_DB" meta eval-generate \
    --variants A,B,C,D,E \
    --per-cluster 4 \
    --resume \
    --output "$OUTPUT" \
    2>&1 | tee -a "$LOG"

log "=== eval-generate complete ==="
