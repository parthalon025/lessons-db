# lessons-db

Automated lessons-learned prevention system. Captures coding mistakes, surfaces them at decision points, and generates Semgrep rules that prevent recurrence.

**Repo:** https://github.com/parthalon025/lessons-db (private)

## Structure

```
src/lessons_db/
  __init__.py
  cli.py              # Click CLI: capture, search, scan, rule, index, export, summary, status, migrate
  config.py            # Paths, Ollama URLs (queue/embed/analysis), env var overrides
  db.py                # SQLite schema, migrations, CRUD
  vectors.py           # LanceDB + Ollama embedding via ollama-queue
  capture.py           # Auto-capture from transcript/diff/test
  search.py            # Semantic search + file-path lookup + content match
  enforce.py           # Escalation ladder, recurrence tracking
  rulegen.py           # Generate Semgrep rules from lessons
  scan.py              # Trigger + parse Semgrep scans (SARIF)
  migrate.py           # Parse 122 markdown lessons → DB + generate rules
  export.py            # Generate markdown from DB records
rules/                 # Community Semgrep rules (lesson-derived)
  python/
  testing/
  patterns/
hooks/                 # Hook scripts (also deployed to ~/.claude/hooks/)
scripts/               # batch-capture-transcripts.sh, run-batch-pipeline.sh
tests/
```

## How to Run

```bash
cd ~/Documents/projects/lessons-db
source .venv/bin/activate

# Run tests
pytest --timeout=120 -x -q

# Core CLI
lessons-db status
lessons-db search "subscriber lifecycle"
lessons-db index                          # backfill cluster_seed + generate embeddings
lessons-db index --seed-only              # cluster_seed only, skip embeddings
lessons-db export <id>                    # export lesson as markdown
lessons-db summary                        # auto-generate SUMMARY.md from DB
lessons-db summary --output PATH

# Capture
lessons-db capture transcript <file>      # extract lessons from session transcript
lessons-db capture transcript <file> --positive
lessons-db capture diff                   # extract from git diff (stdin)
lessons-db capture diff <file>

# Rules + scanning
lessons-db rule generate <id>
lessons-db rule test
lessons-db scan

# Batch scripts
scripts/batch-capture-transcripts.sh [--dry-run] [--since DATE] [--positive]
scripts/run-batch-pipeline.sh [--since DATE]
```

## Deployment

- **Symlink:** `~/.local/bin/lessons-db` → `.venv/bin/lessons-db`
- **Data:** `~/.local/share/lessons-db/` (lessons.db + lance/ + rules/)
- **Hooks:** `hooks/` in repo; deployed to `~/.claude/hooks/`
  - `lessons-db-session-start.sh` — SessionStart
  - `lessons-db-pre-read.sh` — PreToolUse:Read
  - `lessons-db-pre-edit.sh` — PreToolUse:Edit|Write|MultiEdit
  - `lessons-db-enter-plan.sh` — PreToolUse:EnterPlanMode (semantic search)
  - `lessons-db-post-bash.sh` — PostToolUse:Bash (test failure diagnostics)
  - `lessons-db-stop.sh` — Stop (auto-capture from diff)
- **Nightly timer:** `~/.config/systemd/user/lessons-db-nightly.timer` — 03:30 daily

## Key Decisions

- **SQLite** (stdlib) for structured queries — no external DB dependency
- **LanceDB** for semantic vector search — embedded, no server
- **Semgrep** for pattern detection — reused, not rebuilt
- **Ollama** via ollama-queue for generation tasks; direct for embeddings (nomic-embed-text, 768 dims) and analysis (default: qwen3:8b)
- **Click CLI** with subcommands matching design doc

## Scope Tags
language:python, domain:lessons-db

## Design Doc

`~/Documents/docs/plans/2026-02-26-lessons-db-design.md`
