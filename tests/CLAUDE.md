# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Test Suite

Pytest suite with 30+ test files covering the full pipeline from DB CRUD to eval metrics.

## Running Tests

```bash
# Always use parallel — suite is ~5min single-threaded
pytest --timeout=120 -x -q -n 6        # standard
pytest --timeout=120 -x -q -n auto     # light parallel
pytest --timeout=120 -x -q -n 0        # debug (single thread, full output)

# Single file
pytest tests/test_eval.py --timeout=120 -x -q -n 0

# Single test class or function
pytest tests/test_eval.py::TestComputeMetrics --timeout=120 -x -q
pytest tests/test_eval.py::TestComputeMetrics::test_perfect_recall -x -q
```

## Key Fixtures (`conftest.py`)

| Fixture | Scope | What it provides |
|---------|-------|-----------------|
| `tmp_data_dir` | function | Temp `DATA_DIR` + `LANCE_DIR` + `EVAL_DIR` — fully isolated per test |
| `db_path` | function | Path to fresh SQLite DB inside `tmp_data_dir` |
| `app` | function | FastAPI `TestClient` with routes mounted against `tmp_data_dir` |
| `sample_lessons` | function | 3 pre-inserted lessons (negative, positive, meta) with FSRS defaults |

All tests using DB or vectors get isolated state — no shared side effects.

## Mocking Patterns

**Ollama calls:** Mock at the usage-site module, not the package root.

```python
# generate.py uses call_ollama
with mock.patch("lessons_db.eval.generate.call_ollama") as m:
    m.return_value = {"response": '{"principle": "..."}', "thinking": ""}

# judge.py uses call_judge
with mock.patch("lessons_db.eval.judge.call_judge") as m:
    m.return_value = '{"score": 3}'
```

**Never** patch `lessons_db.eval.call_ollama` — the re-export in `__init__.py` is not what the module binds to.

## Test Organization by Pipeline Stage

| File | Stage | Key assertions |
|------|-------|----------------|
| `test_db.py` | Schema + CRUD | All columns exist, FK constraints, FSRS defaults |
| `test_api.py` | FastAPI routes | Status codes, response shape, draft promotion flow |
| `test_eval.py` | Eval pipeline | Variant configs, prompt builders, parser edge cases, F1/AUC math |
| `test_eval_learn.py` | Bayesian learning | Posterior computation, holdout split invariants |
| `test_eval_optimize.py` | APO | Variant selection, cost constraints |
| `test_eval_analysis.py` | Statistics | Bootstrap CI coverage, failure case extraction |
| `test_capture.py` | Capture flow | JSON extraction from free-text, quality scoring |
| `test_fsrs.py` | Spaced repetition | Stability/retrievability math, due scheduling |
| `test_cli.py` | CLI commands | Exit codes, output format for each subcommand |
| `test_github_miner.py` | Mining pipeline | Semgrep integration, clone/scan flow |

## Adding Tests for New Features

- New DB columns: add assertion in `TestSchemaCreation` in `test_db.py`
- New eval variant: update `TestVariantConfigs` to expect the new variant letter with required fields
- New API route: add to `test_api.py`; use the `app` fixture
- New Ollama-calling code: always mock the local binding, not the package import
