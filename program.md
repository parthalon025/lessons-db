# autoresearch — lessons-db eval

Autonomous overnight prompt engineering for the lessons-db transfer-test eval pipeline.
Adapted from Karpathy's autoresearch pattern: modify one file, fixed metric, loop forever.

## What this is

The eval pipeline tests which prompt + model combination extracts the most *transferable*
principles from lessons. A principle is transferable if it helps detect the same bug pattern
in a new codebase (same cluster) without false-positiving on unrelated bugs (diff cluster).

**The metric: F1 score. Higher is better.**

Current baseline (2026-03-08, OpenAI judge):

| Variant | Recall | Precision | F1 |
|---------|--------|-----------|-----|
| A       | 0.93   | 0.17      | 0.28 |
| B       | 0.82   | 0.31      | 0.45 |
| C       | 1.00   | 0.04      | 0.08 |
| D       | 0.79   | 0.33      | **0.47** ← current best |
| E       | 1.00   | 0.01      | 0.03 |

**Diagnostic:** Recall is already saturated. The problem is precision — principles are too
broad and false-positive on unrelated clusters. Target: increase precision without
collapsing recall. F1 > 0.47 is a win.

Variants F–M exist in variants.py but have NOT been benchmarked yet. Start there.

## Setup

1. Read this file fully.
2. Read `src/lessons_db/eval/variants.py` to understand the existing variant design space.
3. Read `~/.local/share/lessons-db/eval/report-baseline-openai-2026-03-08.md` for full baseline.
4. Check `results.tsv` in this directory for experiment history (create if it doesn't exist).
5. Create `results.tsv` with the header if missing:
   ```
   commit	variant	f1	precision	recall	status	description
   ```
6. Create a branch: `git checkout -b autoresearch/<date>` from main. Do not work on main.
7. Confirm setup and begin.

## The file you modify

**`src/lessons_db/eval/variants.py`** — add new experimental variants to `VARIANT_CONFIGS`.

Rules:
- NEVER rename or delete existing variants A–M. They are baselines.
- New variants get sequential IDs: X01, X02, X03, ...
- Use only existing `prompt_id` values — the field is a log label, not a dispatch key.
  Valid: "baseline-fewshot", "zero-shot-causal", "zero-shot-chunked", "contrastive",
         "contrastive-multistage", "mechanism"
- Behavior is driven entirely by boolean flags: `chunked`, `contrastive`, `multi_stage`,
  `mechanism`. Mix freely.
- Required fields: `prompt_id`, `model`, `temperature`, `num_ctx`, `chunked`

## Available models (Ollama)

Check `ollama list` to see what's installed. Common options:
- `deepseek-r1:8b` — strong reasoner, used in A/B/C/F/H
- `deepseek-r1:8b-0528-qwen3-q4_K_M` — newer deepseek-r1 build, NOT YET TESTED
- `qwen3:14b` — larger, used in D/E/G; best F1 so far
- `qwen3:32b` — 2× the size of current best, NOT YET TESTED (slower, higher quality expected)
- `qwen3:8b` — medium, used in M

## Unexplored design space

Priority order (most likely to improve precision):

1. **Variants F, G, H, M** — already configured, just never benchmarked. Run these first.
2. **Lower temperature** — try 0.2 or 0.3 on winning configs (tighter → more specific)
3. **Mechanism + contrastive combo** — not yet tried; root-cause + boundary conditions
4. **Mechanism + multi_stage combo** — not yet tried
5. **Different models** — try any other installed models on the best prompt strategies
6. **Larger context** — try 16384 or 32768 if the model supports it
7. **group-by cluster_seed** — alternative grouping; add `--group-by cluster_seed` flag
8. **Contrastive + chunked + larger model** — no variant covers this combination yet
9. **APO: `eval-optimize`** — automatic prompt optimization. Three strategies:
   - `feedback` (default): shows false positives to optimizer, asks for instruction fixes
   - `opro`: OPRO meta-prompt with score-sorted history (requires 32B+ model)
   - `opro-api`: same as opro via API (Claude/GPT-4o-mini)
   Usage: `lessons-db meta eval-optimize --strategy feedback --candidates 3`

## Running one experiment

Each experiment runs a SINGLE new variant. Output goes to /tmp to avoid polluting EVAL_DIR.

```bash
# 1. Add the new variant to variants.py (edit VARIANT_CONFIGS dict)
# 2. Commit:
git add src/lessons_db/eval/variants.py
git commit -m "autoresearch: add variant X01 — <one-line hypothesis>"

# 3. Generate principles for just this variant:
lessons-db meta eval-generate --variants X01 --per-cluster 4 --output /tmp/ar-X01.json

# 4. Judge the results:
lessons-db meta eval-judge /tmp/ar-X01.json --output /tmp/ar-X01-report.md

# 5. Extract the metric:
grep "^| X01 " /tmp/ar-X01-report.md

# Full summary line is also at:
grep "F1:" /tmp/ar-X01-report.md | head -1
```

If the generate step prints errors for every lesson, the variant config is broken —
check the flag combination and fix before judging. Treat as a crash: log status=crash
and git reset HEAD~1.

If the judge step produces an empty report (no variant rows), the results.json had no
successful generations — treat as crash.

## Keep or discard

- **Keep** (advance branch): F1 > 0.47, OR equal F1 but fewer config lines (simpler)
- **Discard** (revert): F1 ≤ 0.47 with no simplicity benefit
  ```bash
  git reset HEAD~1   # revert the variants.py commit
  ```

Update `current best` in your working memory when you advance.

## Always learn — update program.md after every run

**Every experiment teaches something, even failures.** After logging to results.tsv,
append a one-line insight to the "Learned so far" section below BEFORE starting the
next hypothesis. This is how the research org improves:

- Keep: "X01 — contrastive + temp 0.3 pushed precision to 0.40. Try even lower temp."
- Discard: "X02 — mechanism + multi_stage crashed on chunked input. Avoid that combo."
- Discard: "X03 — qwen3:32b same F1 as 14b but 3× slower. Skip 32b for now."

These notes inform every subsequent hypothesis. Never skip this step.

## Learned so far
- 2026-03-12: [A] below best F1=0.000 (-0.470 vs best 0.470). principle too broad — high recall, low precision (false positives across clusters). Next: add contrastive scope constraints; try contrastive=True or lower temperature.
- 2026-03-11: [A] below best F1=0.000 (-0.470 vs best 0.470). principle too broad — high recall, low precision (false positives across clusters). Next: add contrastive scope constraints; try contrastive=True or lower temperature.
- 2026-03-09: [A] below best F1=0.000 (-0.470 vs best 0.470). principle too broad — high recall, low precision (false positives across clusters). Next: add contrastive scope constraints; try contrastive=True or lower temperature.

*(This section is written by the agent — one line per completed experiment)*

## Logging results

After every run, append to `results.tsv` (untracked — do NOT git add it):

```
<short-commit>	<variant-id>	<f1>	<precision>	<recall>	<keep|discard|crash>	<hypothesis>
```

Example:
```
a1b2c3d	X01	0.51	0.40	0.72	keep	contrastive + qwen3:14b + temp 0.3
b2c3d4e	X02	0.44	0.29	0.75	discard	mechanism + multi_stage deepseek-r1:8b
```

## Evaluation timing

Each experiment takes approximately:
- Generate (per-cluster=4, 1 variant): 10–40 min depending on model and queue load
- Judge (deepseek-r1:8b default): 5–20 min

You can run ~3–6 experiments overnight. Plan accordingly.

## NEVER STOP

Once the experiment loop starts, do NOT pause to ask if you should continue. The human
is asleep. You are autonomous. If you run out of obvious ideas:

- Re-read variants.py and the baseline report for new angles
- Try combining near-miss configs (e.g., take X03 which had high precision but low
  recall, and add chunked=False to broaden it)
- Try the same winning config with a different model
- Try temperature sweeps on the current best variant

Loop until you are manually stopped. Every kept commit is a win.
