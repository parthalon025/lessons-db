.PHONY: lint lint-py lint-sh lint-audit lint-types format install uninstall status

all: lint

lint: lint-py lint-sh lint-audit

lint-py:
	ruff check .

lint-sh:
	@shellcheck scripts/generate-embeddings.sh scripts/run-batch-pipeline.sh scripts/batch-capture-transcripts.sh hooks/*.sh 2>&1

lint-audit:
	pip-audit --progress-spinner off -q 2>/dev/null || true

lint-types:
	.venv/bin/mypy src/lessons_db/

format:
	ruff format .
	ruff check --fix .

install:
	@bash install.sh

uninstall:
	@bash install.sh --uninstall

status:
	@bash install.sh --status
