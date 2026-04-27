# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Eval Package — Transfer-Test Evaluation Pipeline

Answers: **which prompt + model combination produces the most transferable principles?**

Two-stage pipeline: **generate** principles per variant → **judge** their cross-cluster transfer quality → produce F1/AUC report.

## Module Responsibilities

| Module | Role |
|--------|------|
| `variants.py` | `VARIANT_CONFIGS` dict (A–H, M) — each entry has `prompt_id`, `model`, `temperature`, `num_ctx`, `chunked` |
| `sampling.py` | `select_source_lessons()` + `select_transfer_targets()` — test set construction with cluster stratification |
| `prompts.py` | All prompt builders: `build_generation_prompt()`, `build_judge_prompt()`, `build_paired_judge_prompt()`, `build_mechanism_extraction_prompt()` |
| `client.py` | `call_ollama()` + `call_judge()` — HTTP to ollama-queue; `_clean_principle()` strips CoT preamble |
| `generate.py` | `run_eval_generate()` — loops over variants × source lessons, writes `results.json` |
| `judge.py` | `run_eval_judge()` + `run_paired_tournament()` — scores principles, produces `report.md` |
| `signals.py` | Bayesian signal extractors: `compute_embedding_signal()`, `compute_scope_signal()`, `compute_mechanism_signal()` |
| `sampling.py` | `select_source_lessons()` + `select_transfer_targets()` — test set selection |
| `reports.py` | `render_report()` + `render_v2_report()` — markdown report renderers |
| `runs.py` | `record_eval_run()` + `get_eval_history()` — log aggregate metrics to `eval_runs` table |
| `optimize.py` | `run_apo_batch()` — Adaptive Prompt Optimization: selects next variant based on score trends |
| `analysis.py` | `bootstrap_f1_ci()`, `compute_stability()`, `extract_failure_cases()`, `propose_next_variant()` |
| `learn.py` | `compute_posterior()`, `split_holdout()` — Bayesian principle learning from eval history |
| `__init__.py` | Re-exports all public symbols for backward-compatible `from lessons_db.eval import X` |

## Architectural Rules

**Patch at the usage site, not `__init__`.** `call_judge` is imported into `judge.py` as a local binding. Tests must patch `"lessons_db.eval.judge.call_judge"`. Patching `"lessons_db.eval.call_judge"` (the re-export) does nothing. Same for `call_ollama` in `generate.py` → patch `"lessons_db.eval.generate.call_ollama"`.

**All Ollama calls go through ollama-queue** at `OLLAMA_QUEUE_URL` (`http://127.0.0.1:7683`). Never call `localhost:11434` directly.

**qwen3/qwen3.5 + `format: "json"` = empty response.** Thinking models put output in `thinking`, not `response`. Omit `format: "json"`, append `/no_think`, use `_extract_json()` to parse JSON from free-text.

## Data Flow

```
select_source_lessons()
  → build_generation_prompt(variant, lesson)
  → call_ollama() → principle string
  → write results.json

results.json
  → select_transfer_targets(lesson)
  → build_judge_prompt(principle, target)
  → call_judge() → score 1-5
  → compute_metrics() → recall, precision, F1, AUC
  → render_report() → report.md
```

## Metrics Interpretation

- **F1 ↑** — the number to optimize
- **Recall high, precision low** — principle too broad; make prompt more specific
- **Precision high, recall low** — principle too narrow; make prompt more general
- **Mean AUC** — most trustworthy; rank-based, immune to judge score inflation

## Cluster Ground Truth

Lessons in the same cluster share a root cause. A good principle:
- Transfers within cluster → recall TP (score ≥ 3 on same-cluster pair)
- Doesn't false-positive outside cluster → precision (score ≤ 2 on diff-cluster pair)

## Adding a New Variant

1. Add entry to `VARIANT_CONFIGS` in `variants.py` — must include `prompt_id`, `model`, `temperature`, `num_ctx`, `chunked`
2. If `prompt_id` is new, add builder function in `prompts.py`
3. Update `TestVariantConfigs` in `tests/test_eval.py` to expect the new variant
4. Run against variant A baseline before investing compute: `lessons-db meta eval-generate --variants A,NEWID --per-cluster 2`
