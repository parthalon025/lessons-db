# Eval Judge Binary Pivot — Analysis & Plan

## Problem

The rubric-based judge (1-5 transfer score) produces precision collapse regardless of:
- Judge model (deepseek-r1:8b, GPT-4o-mini)
- Generation approach (zero-shot, contrastive)
- Threshold (≥3, ≥4, ≥5)

Models give Transfer ≥ 3 to 90%+ of diff-cluster targets. The 1-5 scale is fundamentally too easy to inflate.

## 10-Method Experiment (2026-03-08)

| Method | P | R | F1 | Notes |
|--------|---|---|-----|-------|
| gemma3:12b + binary | 0.750 | 1.000 | **0.857** | Best F1 |
| GPT-4o-mini + binary | 0.722 | 0.867 | 0.788 | #2 |
| GPT-4o-mini rubric t≥4 | 0.609 | 0.933 | 0.737 | |
| GPT-4o-mini rubric t≥5 | 0.609 | 0.933 | 0.737 | |
| GPT-4o-mini rubric t≥3 | 0.500 | 1.000 | 0.667 | |
| deepseek-r1:8b + binary | 0.667 | 0.667 | 0.667 | |
| gemma3:12b + mechanism-naming | 1.000 | 0.500 | 0.667 | Perfect P |
| deepseek-r1:8b rubric t≥3 | 0.550 | 1.000 | 0.710 | 21 samples |
| GPT-4o-mini + mechanism-naming | 1.000 | 0.133 | 0.235 | Too strict |
| Embedding cosine similarity | AUC=0.707 | | | No LLM |
| GPT-4o-mini paired comparison | Accuracy: 100% | | | 5 samples |

## Key Findings

1. **Binary > Rubric**: YES/NO eliminates scale inflation (F1 0.857 vs 0.47)
2. **gemma3:12b > deepseek-r1:8b for judging**: Better calibration on discrimination tasks
3. **Contrastive generation hurt with rubric judge**: Longer principles → more false positive surface area
4. **Mechanism-naming has perfect precision**: P=1.0 but R=0.5 — too conservative as sole judge
5. **Embeddings show signal** (AUC=0.707): Could be used as pre-filter or tiebreaker

## Implementation Plan

### Phase 1: Binary judge integration
1. Add `--binary` flag to `eval-judge` CLI
2. Use `build_binary_judge_prompt()` + `parse_binary_judge()` (already in eval.py)
3. Update `compute_metrics()` to handle binary (matched=True/False instead of transfer≥3)
4. Default judge model: gemma3:12b
5. Update `render_report()` to show binary results

### Phase 2: Re-judge existing data
6. Re-run judge on contrastive F/G results with binary/gemma3:12b
7. Re-run judge on baseline A-E results with binary/gemma3:12b
8. Compare: does contrastive generation help with a competent judge?

### Phase 3: Integrate into ollama-queue
9. Mirror binary judge changes to eval_engine.py
10. Update UI to show binary judge option in settings
