# AGENTS.md — lessons-db

Instructions for AI agents and Claude Code operating in this repository.

## Quick Start
```bash
pip install -e '.[dev]'
pytest
```

## Architecture
Lessons-learned prevention system with semantic search and anti-pattern scanning

Key directories:
- `src/` or `{{PACKAGE}}/` — source code
- `tests/` — test suite
- `docs/plans/` — implementation plans and tech specs
- `tasks/` — PRD, risk log, pipeline status

## Commands Agents Must Know
- Run tests: `pytest`
- Lint: `make lint`
- Format: `make format`
- Check lessons: `lessons-db scan --target . --baseline HEAD`

## What NOT to Do
- Never commit `.env` or secrets
- Never skip tests before committing
- Never claim done without running `/verify`
- Never start implementing without checking `/check-lessons` first

## Pipeline Status
See `tasks/pipeline-status.md` for current project phase.
