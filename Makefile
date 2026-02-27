.PHONY: lint format

all: lint

lint:
	ruff check .
	pip-audit --progress-spinner off -q 2>/dev/null || true

format:
	ruff format .
	ruff check --fix .
