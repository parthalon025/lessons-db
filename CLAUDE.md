# lessons-db

Automated lessons-learned prevention system. Captures coding mistakes, surfaces them at decision points, and generates Semgrep rules that prevent recurrence.

**Repo:** https://github.com/parthalon025/lessons-db (private)

## Structure

```
src/lessons_db/
  __init__.py
  cli.py              # Click CLI: capture, search, scan, rule, status, migrate
  config.py            # Paths, defaults, Ollama queue URL
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
tests/
```

## How to Run

```bash
cd ~/Documents/projects/lessons-db
source .venv/bin/activate

# Run tests
pytest --timeout=120 -x -q

# CLI
lessons-db status
lessons-db search "subscriber lifecycle"
lessons-db scan --all
lessons-db rule test
```

## Deployment

- **Symlink:** `~/.local/bin/lessons-db` → `.venv/bin/lessons-db`
- **Data:** `~/.local/share/lessons-db/` (lessons.db + lance/ + rules/)
- **Hooks:** Shell hooks in `~/.claude/hooks/` call `lessons-db` CLI

## Key Decisions

- **SQLite** (stdlib) for structured queries — no external DB dependency
- **LanceDB** for semantic vector search — embedded, no server
- **Semgrep** for pattern detection — reused, not rebuilt
- **Ollama** via ollama-queue for embeddings (nomic-embed-text, 768 dims) and analysis (qwen2.5:7b)
- **Click CLI** with subcommands matching design doc

## Scope Tags
language:python, domain:lessons-db

## Design Doc

`~/Documents/docs/plans/2026-02-26-lessons-db-design.md`
