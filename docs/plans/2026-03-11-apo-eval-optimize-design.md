# APO: Automatic Prompt Optimization (`eval-optimize`)

> **Goal:** Automatically generate, evaluate, and promote improved instruction texts for the principle-extraction prompt, replacing manual variant design with a data-driven optimization loop.

**Approach:** OPRO-style optimization (DeepMind, ICLR 2024) adapted for local LLM constraints. Three selectable strategies: feedback-driven (default, works with 14B), pure OPRO (requires 32B+), and API-backed OPRO (Claude/GPT-4o-mini).

**Metric:** F1 score on the transfer test. Current best: D at F1=0.47 (recall 0.79, precision 0.33). Target: F1 > 0.47 without collapsing recall.

**Research basis:**
- [OPRO: Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409) — DeepMind, ICLR 2024
- [Revisiting OPRO: Limitations of Small-Scale LLMs](https://arxiv.org/abs/2405.10276) — ACL Findings 2024: OPRO fails below 13B parameters
- [Evidently AI: Automated Prompt Optimization](https://www.evidentlyai.com/blog/automated-prompt-optimization) — Feedback-driven strategy (error-grounded refinement)
- Gap analysis: `research/2026-03-11-llm-eval-best-practices-gap-analysis.md`, Finding 4

---

## Architecture

```
eval-optimize --strategy feedback|opro|opro-api
       │
       ├─ 1. Load history ──── eval_runs + prompt_variants (DB)
       │                       + get_instruction_text() (hand-authored baselines)
       │
       ├─ 2. Build optimizer prompt ──── strategy-specific
       │     feedback:  show false positives + current instruction
       │     opro:      show top-3 prompts + F1 scores (ascending)
       │     opro-api:  same as opro, but calls Claude/GPT-4o-mini
       │
       ├─ 3. Parse candidates ──── JSON array of {instruction, hypothesis}
       │
       ├─ 4. Register variants ──── prompt_variants table (text + config)
       │
       ├─ 5. Eval cycle ──── run_eval_generate (dev set, --holdout 0.3)
       │                     run_eval_judge → eval_runs (auto-recorded)
       │
       ├─ 6. Holdout validation ──── winners re-judged on test set
       │
       └─ 7. Update program.md ──── one-line insight per variant
```

---

## Data Model

### New table: `prompt_variants`

Stores APO-generated instruction texts and their config. Hand-authored variants (A-H, M) stay in `variants.py` unchanged.

```sql
CREATE TABLE IF NOT EXISTS prompt_variants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id      TEXT NOT NULL UNIQUE,       -- e.g. "X01"
    instruction_text TEXT NOT NULL,             -- optimizer-generated preamble
    config_json     TEXT NOT NULL,              -- model, temperature, num_ctx, etc.
    parent_variant  TEXT,                       -- which variant it was derived from
    strategy        TEXT NOT NULL,              -- "feedback", "opro", "opro-api"
    optimizer_model TEXT,                       -- which LLM generated this
    hypothesis      TEXT,                       -- optimizer's stated reasoning
    created_at      TEXT NOT NULL               -- ISO timestamp
);
CREATE INDEX IF NOT EXISTS idx_prompt_variants_variant
    ON prompt_variants(variant_id);
```

**Key decisions:**
- `variant_id` has a UNIQUE constraint to prevent X-ID collisions
- `config_json` stores the full variant config (model, temperature, num_ctx, chunked, contrastive, etc.) so `variants.py` is never modified at runtime
- Hand-authored variants are NOT duplicated here — they stay in code

### Variant config merge at runtime

`run_eval_generate` merges configs from two sources:

```python
def load_all_variant_configs(conn) -> dict[str, dict]:
    """Merge hand-authored VARIANT_CONFIGS with DB-stored APO variants."""
    merged = dict(VARIANT_CONFIGS)  # code-defined A-H, M
    rows = conn.execute(
        "SELECT variant_id, config_json, instruction_text FROM prompt_variants"
    ).fetchall()
    for row in rows:
        config = json.loads(row["config_json"])
        config["_instruction_text"] = row["instruction_text"]
        config["_apo_generated"] = True
        merged[row["variant_id"]] = config
    return merged
```

The `_instruction_text` key signals `build_generation_prompt` to use the stored text instead of dispatching on flags.

---

## Instruction Text Extraction (Blocker Fix)

Hand-authored variants have instruction text embedded in Python functions. The optimizer needs plain-text versions to build its meta-prompt.

New function in `prompts.py`:

```python
def get_instruction_text(variant_id: str) -> str:
    """Return the instruction preamble for a hand-authored variant.

    This is the text BEFORE the lesson content — the part the optimizer
    can modify. Used by eval-optimize to seed the optimization history.
    """
    templates = {
        "A": (
            "You are extracting a transferable principle from a specific coding lesson.\n\n"
            "A GOOD principle:\n"
            "- Names the structural pattern, not the technology\n"
            "- Is falsifiable — someone could violate it\n"
            "- Applies to at least 3 different domains\n"
            "- Is one sentence, 10-25 words\n\n"
            "Examples of good principles:\n"
            "- 'Resources acquired in callbacks must be released in a symmetric teardown path.'\n"
            "- 'When two representations of the same data exist, one must be designated authoritative.'\n"
            "- 'Silent fallbacks that return default values mask upstream failures indefinitely.'\n"
            "- 'Integration boundaries require end-to-end value tracing, not per-layer unit tests.'\n\n"
        ),
        # B, D: zero-shot causal
        # F, G: contrastive
        # H: multi-stage (two separate instruction texts)
        # M: mechanism extraction
    }
    # ... dispatch per variant, return the instruction-only portion
```

This is a one-time extraction. Each hand-authored variant's instruction text is defined once and returned as a string.

---

## `build_generation_prompt` Change

Minimal change — one new parameter, one new code path:

```python
def build_generation_prompt(
    variant_id: str,
    lesson: dict[str, Any],
    siblings: list[dict[str, Any]] | None = None,
    diff_cluster_items: list[dict[str, Any]] | None = None,
    prompt_overrides: dict[str, str] | None = None,  # NEW
) -> str:
    # APO-generated variants: use stored instruction text
    if prompt_overrides and variant_id in prompt_overrides:
        return _build_apo_prompt(prompt_overrides[variant_id], lesson)

    # Existing dispatch unchanged
    config = VARIANT_CONFIGS[variant_id]
    if config.get("contrastive") and siblings and diff_cluster_items:
        return _build_contrastive_prompt(lesson, siblings, diff_cluster_items)
    # ... rest unchanged
```

`_build_apo_prompt` assembles:
```
{instruction_text}

Lesson:
Title: {title}
One-liner: {one_liner}
Description: {description}

Return ONLY the principle statement. One sentence. No quotes, no explanation.
```

The suffix ("Return ONLY...") is fixed — the optimizer controls only the instruction preamble, not the output format.

---

## Strategy Implementations

### Feedback Strategy (default)

**Optimizer model:** Local, qwen3:14b (or best available ≥14B)

**Why it works with small models:** The task is concrete analysis ("why did this principle wrongly match this lesson?"), not abstract meta-optimization. 14B models handle analytical reasoning well.

**Optimizer prompt:**
```
You are improving a principle-extraction prompt for a lessons-learned system.

Current instruction (F1={best_f1}):
---
{current_instruction_text}
---

This instruction produces principles that are too broad. Here are the worst
false positives — cases where a principle wrongly matched an unrelated lesson:

{for each of top-5 false positives:}
  Principle: "{principle}"
  Wrongly matched lesson: "{target_title}" (cluster: {target_cluster})
  Source cluster: {source_cluster}

Analyze what about the current instruction causes these false matches.
Then generate {N} improved instructions that would prevent them.

Return JSON array:
[{"instruction": "...", "hypothesis": "why this should reduce false positives"}]
```

**Data source:** Most recent `*.scored.json` from `EVAL_DIR`, filtered to:
- Best variant by F1
- False positives: `is_same_cluster=False AND scores.matched=True`
- Sorted by confidence (if available) or random sample of top-5

### OPRO Strategy

**Optimizer model:** Local, requires ≥32B. Auto-detected from `ollama list`.

**Why it needs large models:** The ["Revisiting OPRO" paper](https://arxiv.org/abs/2405.10276) shows models below 13B "merely repeat generic instructions." 32B is the practical minimum for meaningful prompt innovation.

**Optimizer prompt:**
```
You are optimizing a prompt instruction for a principle-extraction system.
Below are past instructions sorted by F1 score (higher = better).

{for each of top-3 variants, ascending by F1:}
[Score: {f1}] "{instruction_text}"

The main failure mode: high recall (>0.9) but low precision (0.07-0.17).
Principles match too broadly across unrelated bug categories.

Generate {N} new instructions that should score higher. Each must:
- Be a complete instruction (not a diff/edit)
- Target precision improvement specifically
- Be 50-200 words

Return JSON array:
[{"instruction": "...", "hypothesis": "why this should score higher"}]
```

### OPRO-API Strategy

Same prompt as OPRO, but routed through:
- `--openai` flag → GPT-4o-mini via OpenAI API
- `--anthropic` flag → Claude via Anthropic API (future)

One API call per cycle. Generator models stay local.

---

## Holdout Integration

APO optimization runs on the **dev set** only. Winners are validated on the **holdout set** before promotion.

```
eval-optimize iteration:
  1. eval-generate --variants X01,X02,X03 --holdout 0.3  → dev results
  2. eval-judge dev_results.json                          → dev F1 per variant
  3. IF any dev F1 > best_dev_f1:
       eval-generate --variants X_best --holdout 0.0      → holdout-only
         (using holdout_ids from step 1's meta)
       eval-judge holdout_results.json                    → holdout F1
       IF holdout F1 also improves:
         → promote as new best
       ELSE:
         → log "Goodhart: dev improved but holdout didn't" to program.md
```

This prevents the exact overfitting scenario holdout was built for.

---

## Optimizer Model Auto-Detection

```python
def _select_optimizer_model(conn, strategy: str) -> str:
    """Select the best available optimizer model for the given strategy."""
    if strategy == "opro-api":
        return "api"  # handled by --openai/--anthropic flags

    # Query ollama for installed models
    available = _get_installed_models()  # calls ollama-queue /api/tags

    if strategy == "opro":
        # OPRO needs ≥32B (research threshold)
        large = [m for m in available if _param_count(m) >= 32e9]
        if not large:
            raise click.UsageError(
                "OPRO strategy requires a 32B+ model. "
                "Install one (e.g. 'ollama pull qwen3:32b') or use --strategy feedback"
            )
        return large[0]["name"]

    # feedback strategy: best available ≥14B, fallback to largest
    medium = [m for m in available if _param_count(m) >= 14e9]
    if medium:
        return medium[0]["name"]
    return available[0]["name"] if available else "qwen3:14b"
```

When a larger model is installed later, `eval-optimize` auto-detects it and prints:
```
Detected qwen3:32b (32B params) — eligible for OPRO strategy.
Currently using: feedback (default). To upgrade: --strategy opro
```

---

## CLI Design

```
lessons-db meta eval-optimize [OPTIONS]

Options:
  --strategy [feedback|opro|opro-api]  Optimization strategy (default: feedback)
  --candidates N                       Number of prompt candidates per iteration (default: 3)
  --max-iterations N                   Maximum optimization iterations (default: 3)
  --holdout FLOAT                      Holdout fraction for Goodhart prevention (default: 0.3)
  --per-cluster N                      Lessons per cluster for eval (default: 4)
  --parent VARIANT                     Variant to optimize from (default: auto-detect best)
  --prune-below FLOAT                  Remove variants below this F1 from future consideration
  --dry-run                            Show what would be generated without running eval
  --openai                             Use OpenAI API for optimizer (opro-api strategy)
  --priority N                         Queue priority for eval jobs
```

**Help text for each strategy:**
```
Strategy choices:
  feedback   (default) Analyze your worst false positives and ask the optimizer
             "what about this instruction caused the error?" Works with local
             14B+ models. Best starting point — grounded in concrete failures.

  opro       OPRO pattern (DeepMind ICLR 2024): show the optimizer your top-3
             prompts + F1 scores, ask for better ones. Requires 32B+ local
             model — smaller models repeat generic instructions (ACL 2024).

  opro-api   Same as opro but uses an API model (GPT-4o-mini or Claude) as
             the optimizer. Most reliable, ~$0.01/iteration. Generator models
             stay local. Requires --openai or --anthropic flag.
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Optimizer returns unparseable JSON | Log warning, retry once with "Return valid JSON" appended. If still fails, skip this iteration. |
| Candidate generates all errors (principle=None for every lesson) | Log as `crash` in program.md, mark variant as failed in `prompt_variants` (add `status` column?), continue to next candidate. |
| X-ID collision (UNIQUE constraint violation) | Increment X-ID counter and retry. |
| No scored_pairs exist (feedback strategy) | Fall back to OPRO strategy with hand-authored variant scores. If no eval_runs either, error: "Run eval-judge first." |
| Holdout F1 doesn't improve despite dev improvement | Log "Goodhart detected" to program.md. Don't promote. Continue to next iteration. |
| Optimizer model not installed | Clear error message: "Strategy X requires model Y. Install with `ollama pull Y` or use --strategy Z." |

---

## Files Changed

| File | Change |
|------|--------|
| `db.py` | Add `prompt_variants` table + index to SCHEMA_SQL; idempotent migration |
| `eval/optimize.py` | **New** — 3 strategies, optimizer prompts, candidate parser, variant registration, model auto-detection, holdout validation, program.md updater |
| `eval/prompts.py` | Add `prompt_overrides` param to `build_generation_prompt`; add `_build_apo_prompt`; add `get_instruction_text()` |
| `eval/generate.py` | Import `load_all_variant_configs`; load overrides from DB; pass `prompt_overrides` to `build_generation_prompt` |
| `eval/__init__.py` | Re-export new symbols |
| `cli.py` | Add `eval-optimize` command with all options |
| `scripts/autoresearch-loop.sh` | Add APO interleave logic (`APO_EVERY=3`) |
| `tests/test_eval_optimize.py` | **New** — optimizer prompt construction, candidate parsing, variant registration, strategy selection, holdout validation, error handling |
| `program.md` | Document `eval-optimize` in "Unexplored design space" section |

---

## Testing Strategy

| Test class | What it verifies |
|-----------|-----------------|
| `TestPromptVariantsTable` | Schema exists, UNIQUE constraint, insert/query round-trip |
| `TestLoadAllVariantConfigs` | Merges VARIANT_CONFIGS + DB entries, `_instruction_text` key present for APO variants |
| `TestGetInstructionText` | Returns correct preamble for each hand-authored variant (A, B, F, H, M) |
| `TestBuildApoPrompt` | Assembles instruction + lesson + suffix correctly |
| `TestFeedbackStrategy` | Constructs optimizer prompt with false positives from scored_pairs |
| `TestOproStrategy` | Constructs meta-prompt with score-sorted history |
| `TestParseCandidates` | Valid JSON → list of candidates; invalid JSON → empty list + warning |
| `TestVariantRegistration` | Writes to prompt_variants + generates valid X-ID; dedup on collision |
| `TestHoldoutValidation` | Dev improvement + holdout improvement → promote; dev only → reject |
| `TestSelectOptimizerModel` | Auto-detects largest model; errors if OPRO without 32B+ |
| `TestEvalOptimizeEndToEnd` | Mocked LLM; full loop: load → optimize → register → eval → record |

---

## Scope Guard

**In scope:**
- `prompt_variants` table, 3 optimizer strategies, CLI command, holdout validation, program.md updates

**Out of scope (future work):**
- Evolutionary crossover between prompts (Approach C — punt until 20+ variants exist)
- Per-principle refinement (Self-Refine — orthogonal, separate feature)
- Anthropic API backend (add when SDK is wired)
- Auto-promotion of APO winners to production variant (needs human review gate first)
