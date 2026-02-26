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

lessons-db integrates with Claude Code via hooks that surface relevant lessons automatically.

Register in `~/.claude/settings.json` under the relevant events. See `docs/hooks-setup.md` for the exact JSON to add.

## Configuration

Override defaults with environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LESSONS_DB_DATA_DIR` | `~/.local/share/lessons-db/` | SQLite DB + LanceDB + rules storage |
| `LESSONS_DB_SOURCE_DIR` | `~/Documents/docs/lessons/` | Source markdown lesson files |
| `LESSONS_DB_OLLAMA_EMBED_URL` | `http://127.0.0.1:11434` | Ollama direct API (embeddings) |
| `LESSONS_DB_OLLAMA_QUEUE_URL` | `http://127.0.0.1:7683` | Ollama queue API (analysis) |

## Adaptive Clustering (optional)

For HDBSCAN-based adaptive cluster discovery:
```bash
pip install -e ".[clustering]"
lessons-db cluster discover
```

## License

MIT
