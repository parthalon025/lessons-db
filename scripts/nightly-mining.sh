#!/bin/bash
# Nightly lessons-db mining pipeline — registered with ollama-queue
# Steps: semgrep delta import → gap analysis → security scan → github mine → mining history
# Registered as recurring job "lessons-db-mining" (03:30 daily, cron: "30 3 * * *")

set -euo pipefail

LESSONS_DB="$HOME/Documents/projects/lessons-db/.venv/bin/lessons-db"
LOG="$HOME/.local/share/lessons-db/nightly-mining.log"

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "=== nightly mining start ==="

log "Step 1: semgrep delta import"
"$LESSONS_DB" import semgrep --delta 2>&1 | tee -a "$LOG" || log "WARN: semgrep import failed"

log "Step 2: gap analysis"
"$LESSONS_DB" gaps 2>&1 | tee -a "$LOG" || log "WARN: gap analysis failed"

log "Step 3: security scan"
"$LESSONS_DB" scan security 2>&1 | tee -a "$LOG" || log "WARN: security scan failed"

log "Step 4: github mining"
"$LESSONS_DB" mine github 2>&1 | tee -a "$LOG" || log "WARN: github mining failed"

log "Step 5: mining history"
"$LESSONS_DB" mining-history --limit 3 2>&1 | tee -a "$LOG" || log "WARN: mining history failed"

log "=== nightly mining complete ==="
