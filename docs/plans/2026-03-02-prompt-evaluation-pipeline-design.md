# Transfer-Test Evaluation Pipeline for Principle Extraction

**Date:** 2026-03-02
**Status:** Approved
**Goal:** Systematically evaluate and optimize prompt × model × settings combinations for `extract-principles` and `generate-meta-lessons` by measuring actual transfer quality, not text aesthetics.

## Problem

Manual A/B testing of prompts is slow (10+ min/round), subjective (eyeballing quality), confounded (502 errors, multiple variables changed per round), and measures the wrong thing (text quality vs functional utility). The pipeline needs an automated, reproducible evaluation that tests whether generated principles actually help an LLM recognize the same structural pattern in different technology contexts.

## Architecture

Two-stage CLI pipeline under `meta`:

```
lessons-db meta eval-generate --variants A,B,C,D,E --sample-size 20 [--resume]
lessons-db meta eval-judge <results-file> --output report.md
```

### Stage 1: eval-generate

Runs N variants across a fixed set of source lessons. Each variant combines a prompt template, model, and generation settings. Results saved to a timestamped JSON file at `~/.local/share/lessons-db/eval/`.

**results.json schema:**
```json
{
  "meta": {
    "generated_at": "ISO-8601",
    "variants": ["A","B","C","D","E"],
    "sample_size": 20,
    "source_lessons": [68, 92, 115, ...]
  },
  "results": [
    {
      "variant": "A",
      "lesson_id": 68,
      "lesson_title": "|| true on git apply...",
      "cluster_seed": "A",
      "principle": "Silent success wrappers on data-critical operations...",
      "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
      "prompt_id": "baseline-fewshot",
      "settings": {"temperature": 0.7, "num_ctx": 4096},
      "generation_time_s": 45.2,
      "error": null
    }
  ]
}
```

**Resume support:** `--resume` skips (variant, lesson_id) pairs already present in the output file. Handles 502 transient failures.

### Stage 2: eval-judge

Reads the results JSON. For each generated principle, constructs transfer test cases using the lesson database as ground truth:

- **2 same-cluster targets** (different category) → true positives (should match)
- **2 different-cluster targets** → true negatives (should NOT match)

The judge is Opus (the user's current session model), scoring each (principle, target) pair on a 3-criterion rubric. Results appended to a markdown report.

## Variant Design (ABCDE)

| ID | Prompt Style | Model | Temperature | Context | Rationale |
|----|-------------|-------|-------------|---------|-----------|
| A  | Current few-shot (4 examples) | deepseek-r1:8b | 0.7 | 4096 | Baseline — current production prompt |
| B  | Zero-shot, causal framing | deepseek-r1:8b | 0.6 | 8192 | Research: R1 is better zero-shot, temp 0.6 |
| C  | Zero-shot + chunked (3-4 siblings) | deepseek-r1:8b | 0.6 | 8192 | Recall via context batching |
| D  | Zero-shot, causal framing | qwen3:14b | 0.6 | 8192 | Larger non-reasoning model |
| E  | Zero-shot + chunked | qwen3:14b | 0.6 | 8192 | Best of C + D |

### Chunking Strategy (Variants C, E)

Instead of extracting a principle from a single lesson, show 3-4 lessons from the same cluster:

```
These lessons all share the same structural failure pattern across different technologies:

1. Title: ... One-liner: ...
2. Title: ... One-liner: ...
3. Title: ... One-liner: ...

What is the ONE structural principle that explains ALL of these?
Causal form: '<pattern> causes <consequence> when <condition>'
One sentence, 10-25 words. No technology names.
```

Rationale: seeing multiple instances of the same structural pattern may help the model abstract past individual technologies. This tests recall through batching.

## Test Set Construction

**Source lessons:** 4 per cluster × 5 clusters = 20 source lessons.
Selection criteria: maximize category diversity within each cluster.

**Transfer targets per principle:**
- 2 same-cluster lessons (different category/technology) → true positive
- 2 different-cluster lessons → true negative

**Total judge decisions:** 20 principles × 4 targets × 5 variants = **400 scored pairs**.

## Judging Rubric

Scored by Opus (frontier model) as pipeline monitor. Each (principle, target_lesson) pair scored 1-5 on three criteria:

| Criterion | 1 (fail) | 3 (partial) | 5 (pass) |
|-----------|----------|-------------|----------|
| Transfer recognition | Principle doesn't help recognize pattern in target | Vague connection | Clear structural match |
| Precision | Would false-positive on unrelated lessons | Somewhat specific | Only matches structurally similar |
| Actionability | LLM couldn't act on it | Could act with additional context | LLM could immediately prevent this bug class |

## Aggregate Metrics

Per variant:
- **Transfer recall**: % of same-cluster targets scored ≥ 3 on transfer recognition
- **Precision**: % of different-cluster targets scored ≤ 2 on transfer recognition
- **Mean actionability**: average actionability across all targets
- **F1**: harmonic mean of recall and precision

## Output Report

Markdown at `~/.local/share/lessons-db/eval/report-YYYY-MM-DD.md`:
- Summary table (variant × metric)
- Winner identification with prompt text and settings
- Per-cluster breakdown (which clusters are easiest/hardest to abstract)
- Failure analysis (examples where best variant still failed)
- Recommendations for production prompt update

## Research Findings Applied

Key findings from prompt engineering research for reasoning models (2024-2026):

1. **Zero-shot > few-shot for R1** — few-shot examples degrade reasoning and cause copying
2. **No explicit CoT** — conflicts with native `<think>` tag reasoning
3. **Temperature 0.6** — recommended sweet spot for R1
4. **Context 8192+** — default 4096 is too small for reasoning chains
5. **Reasoning models struggle with structured JSON** — separate reasoning from formatting
6. **Self-verification bottlenecked by verifier quality** — hence Opus as external judge
7. **Small models copy format from examples** — observed in rounds 2-4 of manual testing

## Constraints

- All generation runs through ollama-queue (port 7683) per user requirement
- `_warm_model()` called before each variant's batch to prevent cold-load timeouts
- 502 errors handled via `--resume` (skip completed pairs)
- Judge scoring done by Opus inline (no additional API calls needed)

## Success Criteria

- Pipeline identifies a variant with F1 > 0.75 (recall + precision balanced)
- Winning variant demonstrably outperforms baseline (A) on all three metrics
- Results are reproducible (same variant produces similar scores on re-run)
