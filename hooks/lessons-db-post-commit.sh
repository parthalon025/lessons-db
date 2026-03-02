#!/usr/bin/env bash
# lessons-db-post-commit.sh — Post-commit hook for evaluating surfaced lessons.
#
# After a commit, checks if recently-surfaced anti-patterns appear in the diff.
# Updates surfacing event outcomes: 'heeded' if the anti-pattern was avoided,
# 'dismissed' if the anti-pattern recurred in the committed code.
#
# Usage (as a git post-commit hook):
#   cp hooks/lessons-db-post-commit.sh .git/hooks/post-commit
#   # or symlink:
#   ln -sf ../../hooks/lessons-db-post-commit.sh .git/hooks/post-commit
#
# Usage (standalone):
#   bash hooks/lessons-db-post-commit.sh [--hours N] [--dry-run]

set -euo pipefail

# Resolve lessons-db CLI — prefer venv, fall back to PATH
LESSONS_DB="${LESSONS_DB_CLI:-}"
if [ -z "$LESSONS_DB" ]; then
    # Try the project venv first
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    if [ -x "$PROJECT_ROOT/.venv/bin/lessons-db" ]; then
        LESSONS_DB="$PROJECT_ROOT/.venv/bin/lessons-db"
    elif command -v lessons-db &>/dev/null; then
        LESSONS_DB="lessons-db"
    else
        # Silent exit if lessons-db is not installed — don't block commits
        exit 0
    fi
fi

# Parse arguments
HOURS=24
DRY_RUN=""
for arg in "$@"; do
    case "$arg" in
        --hours)
            shift
            HOURS="${1:-24}"
            shift
            ;;
        --hours=*)
            HOURS="${arg#*=}"
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            ;;
    esac
done

# Run the evaluate-commit command
# Use git diff to get the committed changes and pass via --diff-text
# to avoid the hook running in a potentially detached state
DIFF_TEXT="$(git diff HEAD~1..HEAD 2>/dev/null || true)"

if [ -z "$DIFF_TEXT" ]; then
    # No diff available (initial commit or error) — skip silently
    exit 0
fi

exec "$LESSONS_DB" learn evaluate-commit --hours "$HOURS" $DRY_RUN --diff-text "$DIFF_TEXT"
