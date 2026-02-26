# lessons-db

Automated lessons-learned prevention system. Captures coding mistakes into a SQLite + LanceDB database, surfaces them semantically before planning, and generates Semgrep rules to prevent recurrence.

## How It Works

1. **Capture** — import existing lesson markdown files or let the Stop hook auto-extract from session diffs
2. **Search** — semantic vector search (LanceDB + Ollama embeddings) surfaces relevant lessons before you code
3. **Enforce** — generate Semgrep rules from lessons; scan your codebase; block anti-patterns in CI

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai) running locally (`ollama pull nomic-embed-text`)
- [Claude Code](https://claude.ai/claude-code) (for hooks)
- Semgrep (optional, for rule generation): `pip install semgrep`

## Installation

```bash
git clone https://github.com/parthalon025/lessons-db
cd lessons-db
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Add the `lessons-db` binary to your PATH:
```bash
ln -sf "$(pwd)/.venv/bin/lessons-db" ~/.local/bin/lessons-db
```

Verify:
```bash
lessons-db --help
```

## First Run

```bash
# Initialize the database
lessons-db status

# Import existing lesson markdown files (optional)
lessons-db migrate --source /path/to/your/lessons/

# Generate embeddings (requires Ollama with nomic-embed-text)
lessons-db index

# Verify semantic search works
lessons-db search "exception swallowed silently"
```

## Claude Code Hooks

lessons-db integrates with Claude Code via hooks. Scripts live in `hooks/`; deploy by copying or symlinking to `~/.claude/hooks/`.

| Hook file | Event | Purpose |
|-----------|-------|---------|
| `lessons-db-session-start.sh` | SessionStart | Surface top lessons at session open |
| `lessons-db-pre-read.sh` | PreToolUse:Read | Check lessons relevant to file being read |
| `lessons-db-pre-edit.sh` | PreToolUse:Edit\|Write\|MultiEdit | Warn before editing if lessons apply |
| `lessons-db-enter-plan.sh` | PreToolUse:EnterPlanMode | Semantic search at planning boundary |
| `lessons-db-post-bash.sh` | PostToolUse:Bash | Diagnose test failures against lessons |
| `lessons-db-stop.sh` | Stop | Auto-capture new lessons from session diff |

Register hooks in `~/.claude/settings.json` under the matching event keys.

## Commands

```bash
# Database
lessons-db status
lessons-db migrate --source /path/to/lessons/
lessons-db index                          # backfill cluster_seed + generate embeddings
lessons-db index --seed-only              # cluster_seed only, skip embeddings

# Search
lessons-db search "exception swallowed silently"

# Capture
lessons-db capture transcript <file>      # extract lessons from session transcript
lessons-db capture transcript <file> --positive
lessons-db capture diff                   # extract from git diff (stdin)
lessons-db capture diff <file>

# Export
lessons-db export <id>                    # export lesson as markdown
lessons-db summary                        # auto-generate SUMMARY.md from DB
lessons-db summary --output PATH

# Rules + scanning
lessons-db rule generate <id>
lessons-db rule test
lessons-db scan

# Batch scripts
scripts/batch-capture-transcripts.sh [--dry-run] [--since DATE] [--positive]
scripts/run-batch-pipeline.sh [--since DATE]
```

## Configuration

Override defaults with environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LESSONS_DB_DATA_DIR` | `~/.local/share/lessons-db/` | SQLite DB + LanceDB + rules storage |
| `LESSONS_DB_SOURCE_DIR` | `~/Documents/docs/lessons/` | Source markdown lesson files |
| `LESSONS_DB_OLLAMA_EMBED_URL` | `http://127.0.0.1:11434` | Ollama direct API (embeddings) |
| `LESSONS_DB_OLLAMA_ANALYSIS_URL` | `http://127.0.0.1:11434` | Ollama direct API (analysis/capture) |
| `LESSONS_DB_OLLAMA_QUEUE_URL` | `http://127.0.0.1:7683` | Ollama queue API (generation tasks) |
| `LESSONS_DB_OLLAMA_ANALYSIS_MODEL` | `qwen3:8b` | Model for analysis and capture |

## Adaptive Clustering (optional)

For HDBSCAN-based adaptive cluster discovery:
```bash
pip install -e ".[clustering]"
lessons-db cluster discover
```

## License

MIT
