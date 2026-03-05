#!/bin/bash
# Extract transferable principles from lessons via LLM — registered with ollama-queue
# Runs every 30 min, processes 50 lessons per batch until all lessons have principles.
# Checks remaining count first to avoid loading the model when there's nothing to do.
# Registered as recurring job "lessons-db-extract-principles" (every 30m, priority 7)

set -euo pipefail

LESSONS_DB="$HOME/Documents/projects/lessons-db/.venv/bin/lessons-db"
PYTHON="$HOME/Documents/projects/lessons-db/.venv/bin/python3"
DB_PATH="$HOME/.local/share/lessons-db/lessons.db"
LOG="$HOME/.local/share/lessons-db/meta-extract-principles.log"
BATCH_SIZE=50

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Short-circuit: query DB directly before loading the model
remaining=$("$PYTHON" -c "import sqlite3; c=sqlite3.connect('$DB_PATH'); print(c.execute('SELECT COUNT(*) FROM lessons WHERE principle IS NULL').fetchone()[0])" 2>/dev/null || echo 999)

if [ "$remaining" -eq 0 ]; then
    log "All lessons have principles — nothing to do"
    exit 0
fi

log "=== extract-principles start: $remaining lessons remaining ==="
"$LESSONS_DB" meta extract-principles --batch-size "$BATCH_SIZE" 2>&1 | tee -a "$LOG"
log "=== extract-principles complete ==="
