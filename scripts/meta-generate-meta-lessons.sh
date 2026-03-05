#!/bin/bash
# Generate double-loop meta-lessons from lesson clusters via LLM — registered with ollama-queue
# Runs daily at the best open slot. Skips clusters that already have a loop_level='double' lesson.
# Registered as recurring job "lessons-db-generate-meta" (daily, priority 7)

set -euo pipefail

LESSONS_DB="$HOME/Documents/projects/lessons-db/.venv/bin/lessons-db"
PYTHON="$HOME/Documents/projects/lessons-db/.venv/bin/python3"
DB_PATH="$HOME/.local/share/lessons-db/lessons.db"
LOG="$HOME/.local/share/lessons-db/meta-generate-meta-lessons.log"

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Short-circuit: check if any clusters lack a double-loop meta-lesson
pending=$("$PYTHON" -c "
import sqlite3
c = sqlite3.connect('$DB_PATH')
rows = c.execute('''
    SELECT COUNT(DISTINCT cluster_seed)
    FROM lessons
    WHERE cluster_seed IS NOT NULL
      AND cluster_seed NOT IN (
          SELECT DISTINCT cluster_seed FROM lessons
          WHERE loop_level = \"double\" AND cluster_seed IS NOT NULL
      )
    GROUP BY cluster_seed HAVING COUNT(*) >= 3
''').fetchall()
print(len(rows))
" 2>/dev/null || echo 999)

if [ -z "$pending" ] || [ "$pending" -eq 0 ] 2>/dev/null; then
    log "All clusters have meta-lessons — nothing to do"
    exit 0
fi

log "=== generate-meta-lessons start ==="
"$LESSONS_DB" meta generate-meta-lessons 2>&1 | tee -a "$LOG"
log "=== generate-meta-lessons complete ==="
