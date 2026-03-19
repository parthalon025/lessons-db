# lessons-db

Automated lessons-learned system with human-like learning. Captures mistakes and successes, surfaces them via FSRS spaced repetition, tracks outcomes, and adapts presentation through 8 learning science mechanisms.

**Repo:** https://github.com/parthalon025/lessons-db

## Structure

```
src/lessons_db/
  __init__.py
  cli.py              # Click CLI: capture, search, scan, rule, index, export, summary, status, migrate, fsrs, learn, kpi, calibrate, transfer, reuse, meta
  config.py            # Paths, Ollama URLs (queue/embed/analysis), env var overrides (incl. REPO_CACHE_DIR)
  db.py                # SQLite schema, migrations, CRUD (incl. win_streaks, surfacing_events)
  fsrs.py              # FSRS-6 spaced repetition: retrievability, stability, difficulty, adaptive fading
  learn.py             # Surfacing events, outcome recording, feedback loop, exception-finding (SFBT)
  prevention.py        # Velocity detection, fix queue, content checking
  github_miner.py      # GitHub mining pipeline: discover_repos, mine_repos_for_gaps, MiningConfig
  vectors.py           # LanceDB + Ollama embedding via ollama-queue
  eval/                # Transfer-test evaluation package (split from monolith)
    __init__.py        # Re-exports all public symbols for backward compatibility
    variants.py        # Variant configs (A-H, M), retry constants, group_by values
    sampling.py        # Test set selection: source lessons + transfer targets
    prompts.py         # All prompt builders: generation, judge, mechanism, simulation
    client.py          # Ollama queue + OpenAI HTTP integration, _clean_principle
    signals.py         # Bayesian signal extractors + fusion (paired, embedding, scope, mechanism)
    generate.py        # Generation orchestrator: produce principles for (variant, lesson) pairs
    judge.py           # Judge orchestrator: scoring, metrics, paired tournament
    reports.py         # Report renderers: V1/V2 markdown, diagnostics, simulation lift
  eval_diagnostics.py  # Confusion matrix + variant comparison diagnostics
  capture.py           # Auto-capture from transcript/diff/test + win detection
  search.py            # Semantic search + file-path lookup + content match
  enforce.py           # Escalation ladder, recurrence tracking
  rulegen.py           # Generate Semgrep rules from lessons
  scan.py              # Trigger + parse Semgrep scans (SARIF)
  migrate.py           # Parse 122 markdown lessons → DB + generate rules
  export.py            # Generate markdown from DB records
graphrag/              # GraphRAG config (git-tracked); artifacts at ~/.local/share/lessons-db/.graphrag/ (gitignored)
  settings.yml         # v3 completion_models → ollama-queue proxy (localhost:7683)
  prompts/             # entity_extraction, community_report, summarize_descriptions
rules/                 # Community Semgrep rules (lesson-derived)
  python/
  testing/
  patterns/
hooks/                 # Hook scripts (also deployed to ~/.claude/hooks/)
scripts/               # batch-capture-transcripts.sh, run-batch-pipeline.sh
spa/                   # SPA dashboard (Preact + esbuild + superhot-ui) — see Web UI section
  src/                 # Source: pages/, components/, stores/, hooks/, api.js, polling.js
  dist/                # Build output (gitignored — must npm run build)
tests/
```

## How to Run

```bash
cd ~/Documents/projects/lessons-db
source .venv/bin/activate

# Run tests (always use parallel — suite takes ~5min single-threaded)
pytest --timeout=120 -x -q -n 6       # standard parallel run
pytest --timeout=120 -x -q -n auto    # light parallel
pytest --timeout=120 -x -q -n 0      # debug (single thread)

# Core CLI
lessons-db status
lessons-db search "subscriber lifecycle"
lessons-db index                          # backfill cluster_seed + generate embeddings
lessons-db index --seed-only              # cluster_seed only, skip embeddings
lessons-db export <id>                    # export lesson as markdown
lessons-db summary                        # auto-generate SUMMARY.md from DB
lessons-db summary --output PATH

# Capture
lessons-db capture transcript <file>      # extract lessons from session transcript
lessons-db capture transcript <file> --positive
lessons-db capture diff                   # extract from git diff (stdin)
lessons-db capture diff <file>

# Hybrid search (BM25 + RRF)
lessons-db hybrid-search "query"              # BM25Okapi fused via RRF (k=60), top 5
lessons-db hybrid-search "query" --top 10    # return top 10
lessons-db hybrid-search "query" --json      # machine-readable output

# GraphRAG index
lessons-db graph-build                        # export lessons → submit index job to ollama-queue
lessons-db graph-build --local               # run graphrag directly (blocking)
lessons-db graph-build --status              # show artifact count from last build
lessons-db graph-search "query"              # global mode (community synthesis)
lessons-db graph-search "query" --mode local # local mode (entity-focused)

# Rules + scanning
lessons-db rule generate <id>
lessons-db rule test
lessons-db scan

# FSRS spaced repetition
lessons-db fsrs init                      # backfill FSRS defaults on all lessons
lessons-db fsrs due                       # list lessons due for review (R < 0.9)
lessons-db fsrs due --threshold 0.8       # custom threshold
lessons-db fsrs stats                     # stability distribution + review forecast

# Learning & feedback
lessons-db learn evaluate-commit          # evaluate recent surfacing events against commit diff
lessons-db learn find-exceptions          # SFBT: find internalized patterns (absent anti-patterns)
lessons-db learn record ID --hook pre_edit  # record a surfacing event

# Metacognition
lessons-db kpi                            # learning KPI dashboard (heeded rates, ZPD, streaks)
lessons-db calibrate profile              # per-category strength/growth areas

# Positive patterns
lessons-db reuse record ID                # record positive pattern reuse
lessons-db capture detect-wins            # detect wins from recent session
lessons-db transfer find "context"        # cross-project analogical matching

# Meta-learning (Ollama)
lessons-db meta extract-principles        # batch-extract transferable principles
lessons-db meta generate-meta-lessons     # generate double-loop meta-lessons from clusters

# Evaluation pipeline (transfer-test)
lessons-db meta eval-generate --variants A,B,C,D,E --per-cluster 4   # generate principles across variants
lessons-db meta eval-generate --variants A --per-cluster 1 --resume  # resume after transient errors
lessons-db meta eval-generate --variants A --priority 1              # high priority (preempts scheduled jobs)
lessons-db meta eval-judge results.json                              # score principles, produce F1 report
lessons-db meta eval-judge results.json --openai --judge-model gpt-4o-mini  # use OpenAI as judge
lessons-db meta eval-judge results.json --priority 1                 # high priority judge calls

# Batch scripts
scripts/batch-capture-transcripts.sh [--dry-run] [--since DATE] [--positive]
scripts/run-batch-pipeline.sh [--since DATE]
```

## Web UI Dashboard

SPA served by FastAPI at `http://localhost:7685/ui/`. Source at `spa/`.

**Tech stack:** Preact 10 + `@preact/signals` + esbuild + `superhot-ui` (file: dep from `../../superhot-ui`)
**Routing:** Signal-based via `currentRoute` signal in `AppLayout.jsx` — no router library
**Pages:** Dashboard, Lessons, Triage, Eval, Admin
**API client:** `spa/src/api.js` — all calls relative to `/api/` (same origin)
**State:** Signal stores in `spa/src/stores/` (stats, health, lessons, pipelines, prevention, triage)
**Polling:** `spa/src/polling.js` — shared interval orchestrator

**Build:**
```bash
cd ~/Documents/projects/lessons-db/spa && npm run build   # produces dist/
npm run dev                                                 # watch mode
```

**Gotchas:**
- `spa/dist/` is gitignored — must `npm run build` after cloning
- `superhot-ui` is a `file:` dep — if `../../superhot-ui` path is missing the build fails
- FastAPI only mounts `/ui/` if `spa/dist/` exists at startup — build before starting the server

## API Endpoints

Exposed via FastAPI at `localhost:7685` (proxied by project-hub Express at `/hub/api/lessons/*`):

- **GET /api/lessons** — paginated lesson list. Query params: `q`, `category`, `tier`, `polarity`
- **GET /api/lessons/stats** — counts by category, polarity, tier
- **GET /api/lessons/categories** — distinct category values
- **GET /api/lessons/{id}** — single lesson record
- **GET /api/gaps** — identified gap records
- **GET /api/calibration/history** — paginated `calibration_runs` (default: last 20, max 100)
- **POST /api/calibration/run** — queues `calibrate_pipeline()`. Query params: `sample_n` (default 50), `skip_extraction` (default false). Response: `{status: "queued"}`
- **GET /api/mining/history** — paginated `mining_runs` with per-gate rejection counts
- **GET /api/mining/repos** — mined repo list
- **POST /api/mining/run** — queues GitHub mining task
- **GET /api/security/findings** — open scan findings
- **POST /api/security/scan** — trigger Semgrep security scan
- **GET /api/scan/summary** — decision-context dashboard: promotion rate, drafts captured, sessions processed, scan age, embed failure rate, FSRS review backlog
- **GET /api/capture-drafts** — lessons pending promotion
- **GET /api/fix-queue** — fix queue items
- **GET /api/fix-queue/next** — next actionable fix
- **POST /api/fix-queue/populate** — populate from scan findings
- **POST /api/fix-queue/issues** — create GH issues for fix-queue items
- **GET /api/prevention/report** — velocity + recurrence summary
- **GET /api/prevention/recurrence** — recurrence tracking records
- **POST /api/prevention/resolve-outcomes** — mark outcomes resolved
- **POST /api/prevention/bulk-generate** — bulk-generate Semgrep rules
- **POST /api/prevention/check-content** — check content against lessons
- **GET /eval/health** — eval service health
- **POST /eval/prime** — seed eval queue
- **GET /eval/items** — items awaiting eval (requires cluster assignment)
- **GET /eval/clusters** — cluster list with lesson counts
- **POST /eval/results** — submit eval results
- **POST /eval/production-variant** — set production variant

## Deployment

- **Symlink:** `~/.local/bin/lessons-db` → `.venv/bin/lessons-db`
- **Data:** `~/.local/share/lessons-db/` (lessons.db + lance/ + rules/)
- **Hooks:** `hooks/` in repo; deployed to `~/.claude/hooks/`
  - `lessons-db-session-start.sh` — SessionStart (FSRS due lessons + exception reporting + feedforward)
  - `lessons-db-pre-read.sh` — PreToolUse:Read (feedforward formatting)
  - `lessons-db-pre-edit.sh` — PreToolUse:Edit|Write|MultiEdit (positive reuse detection + surfacing)
  - `lessons-db-enter-plan.sh` — PreToolUse:EnterPlanMode (semantic search + pro-mortem + bright spots)
  - `lessons-db-post-bash.sh` — PostToolUse:Bash (test failure diagnostics + feedforward)
  - `lessons-db-post-commit.sh` — post-commit (evaluate-commit outcome recording)
  - `lessons-db-stop.sh` — Stop (auto-capture + win detection + AAR sustain prompt)
  - `_feedforward-format.sh` — shared helper (SUGGESTION/PROVEN PATTERN formatting)
- **Nightly timer:** `~/.config/systemd/user/lessons-db-nightly.timer` — 03:30 daily

## Key Decisions

- **SQLite** (stdlib) for structured queries — no external DB dependency
- **LanceDB** for semantic vector search — embedded, no server
- **Semgrep** for pattern detection — reused, not rebuilt
- **Ollama** via ollama-queue for all tasks including embeddings (nomic-embed-text, 768 dims) and analysis (default: qwen3.5:9b); embed calls route through queue at port 7683 (PROXY_WAIT_TIMEOUT=300s)
- **FSRS-6** implemented directly (~500 lines, no external deps) — power-law forgetting curve R=(1+F*t/S)^DECAY with 19 optimized parameters
- **Click CLI** with subcommands matching design doc

## Gotchas

- **All Ollama calls go through ollama-queue** — never call `localhost:11434` directly. All eval functions (`call_judge`, `run_eval_judge`, `run_paired_tournament`) default `ollama_url` to `OLLAMA_QUEUE_URL` (`http://127.0.0.1:7683`). Override via `LESSONS_DB_OLLAMA_QUEUE_URL` env var.
- **Patch at the usage site, not the definition** — `call_judge` is imported into `judge.py` as a local binding. Tests must patch `"lessons_db.eval.judge.call_judge"`, not `"lessons_db.eval.call_judge"`. Similarly, `call_ollama` in `generate.py` must be patched as `"lessons_db.eval.generate.call_ollama"`. Patching the `__init__` re-export does nothing.
- **qwen3/qwen3.5 + `format: "json"` = empty `response`** — thinking models put their output in `thinking`, not `response`, when `format: "json"` is set. Workaround: omit `format: "json"`, append `/no_think` to the prompt, and use `_extract_json()` in `capture.py` to parse JSON from free-text response.
- **`capture transcript` timeout with CPU models** — qwen3.5:9b is CPU-bound and can time out on large transcripts (504 from queue after 300s). Use `LESSONS_DB_OLLAMA_ANALYSIS_MODEL=qwen3.5:4b` (GPU-native) or override `PROXY_WAIT_TIMEOUT` in ollama-queue for large-context jobs.

## Eval Pipeline — Variants & Tests

### What the eval pipeline does (plain language)

Every lesson in the DB encodes a mistake or pattern. The goal is to generate a short *principle* — a generalisable rule — that another AI can use to recognise the same type of problem in a completely different codebase. The eval pipeline answers: **which prompt + model combination produces the most useful principles?**

It works in two stages:

1. **Generate** (`eval-generate`) — For each lesson, ask each variant (A-M) to produce a principle. Output: `results.json` (one row per variant × lesson).
2. **Judge** (`eval-judge`) — An independent judge model reads each principle and rates: "Does this principle apply to a *related* lesson (same cluster)? Does it wrongly apply to an *unrelated* lesson?" Output: F1 score per variant + `report.md`.

The cluster system is the ground truth: lessons in the same cluster share a root cause. A good principle transfers *within* a cluster (recall) and doesn't false-positive outside it (precision).

**Recommended run order:**
```
# 1. Generate principles for all variants (takes ~20-60 min depending on model load)
lessons-db meta eval-generate --variants A,B,C,D,E,F,G,H,M --per-cluster 4

# 2. Score with the default judge (deepseek-r1:8b)
lessons-db meta eval-judge results.json

# 3. If you want a second opinion, rerun with OpenAI
lessons-db meta eval-judge results.json --openai --judge-model gpt-4o-mini

# 4. Check the report
cat ~/.local/share/lessons-db/eval/report.md
```
Run A first alone to establish a baseline before investing compute in all variants.

**Reading the results:**
- **F1 ↑** = better overall. That's the number to optimise.
- **Recall high, precision low** = principle is too broad — it matches everything, including unrelated lessons. Make the prompt more specific.
- **Precision high, recall low** = principle is too narrow — it misses related lessons. Make the prompt more general.
- **Mean AUC** (from rank metrics) is the most trustworthy number — it's rank-based and immune to a judge that inflates all scores equally.

### Variant Matrix

Each variant is one experiment: swap one thing (prompt style, model size, or generation strategy) and see if F1 improves. The letters are just IDs — no ranking implied.

| ID | Why it exists | What it changes vs control | What it produces |
|----|--------------|---------------------------|-----------------|
| **A** *(control)* | Baseline — measures the floor. All other variants are compared to this. | Few-shot examples in prompt; 4k context window | Principles anchored to example format; fast to run |
| **B** | Does "explain *why* this failed" beat "match this format"? | Removes few-shot examples; adds causal framing; 8k context | Principles stated as root causes, not just rules |
| **C** | Does splitting a long lesson into small chunks avoid losing context halfway through? | Same prompt as B but each lesson is split into ≈512-token chunks first | One principle per chunk; more focused, but may miss cross-cutting patterns |
| **D** | Does a bigger model do better with the same prompt? | Swaps deepseek-r1:8b → qwen3:14b (model ablation vs B) | Same causal principles, but from a 14B model |
| **E** | Does chunking help a bigger model too? | qwen3:14b + chunked input (combination of C and D) | Chunk-level principles from the larger model |
| **F** | Does asking "when does this NOT apply?" sharpen the principle? | Adds a contrastive instruction ("state the boundary conditions") | Principles with explicit scope limits — fewer false positives |
| **G** | Can a bigger model follow contrastive instructions better? | qwen3:14b + contrastive prompt (model ablation vs F) | More precisely scoped principles from a 14B model |
| **H** | Does a two-pass pipeline (observe → distill) produce the most transferable output? | Two LLM calls: pass 1 extracts the abstract pattern, pass 2 distills the principle | The most deliberate output; slowest (2× calls); best for hard-to-generalise lessons |
| **M** | Does capturing *why* something fails (mechanism) transfer better than a surface rule? | Separate mechanism-extraction prompt; chunked; qwen3.5:9b; asks for root-cause chain | Causal mechanism triplets (trigger → failure → consequence) rather than rules |

### Test Classes in `tests/test_eval.py`

Tests are grouped by pipeline stage. Run them to verify the pipeline without needing a live Ollama connection (all external calls are mocked).

**Config & structure tests** — verify the pipeline is wired up correctly before running anything:

| Class | Why it exists | What it verifies |
|-------|--------------|-----------------|
| `TestEvalConfig` | Catch misconfigured data paths early | `EVAL_DIR` points to `DATA_DIR / "eval"` |
| `TestVariantConfigs` | Prevent silent variant misconfiguration | 9 variants exist, all have required fields (`prompt_id`, `model`, `temperature`, `num_ctx`, `chunked`), A is the control |

**Sampling tests** — verify that test set selection is unbiased and respects limits:

| Class | Why it exists | What it verifies |
|-------|--------------|-----------------|
| `TestSelectSourceLessons` | Source lessons must be diverse and bounded | Per-cluster limits respected; category diversity maximised; double-loop meta-lessons excluded |
| `TestSelectTransferTargets` | Transfer targets must be correctly split | Returns `same_cluster` and `diff_cluster` lists; source lesson excluded from targets |

**Prompt builder tests** — verify prompts are assembled correctly before spending LLM time:

| Class | Why it exists | What it verifies |
|-------|--------------|-----------------|
| `TestBuildGenerationPrompt` | Wrong prompt = wrong principle | Prompt contains lesson content; chunked variants split input into multiple chunks |
| `TestBuildJudgePrompt` | Judge must see both principle and target | Prompt contains principle text and target lesson |
| `TestBuildBinaryJudgePrompt` | YES/NO judge needs clean format | Prompt asks for YES/NO; includes both principle and target |
| `TestBuildPairedJudgePrompt` | A/B position must be randomised to avoid position bias | A/B assignment is random; `same_is_a` flag correctly tracks which position is ground truth |

**Response parser tests** — verify the pipeline doesn't silently drop scores when the LLM returns messy output:

| Class | Why it exists | What it verifies |
|-------|--------------|-----------------|
| `TestParseJudgement` | LLMs return JSON in various formats | Valid JSON extracts scores 1-5; missing keys or bad JSON returns `None` (not a crash) |
| `TestParseBinaryJudge` | YES/NO can appear in reasoning traces | Strips `<think>` blocks; `YES`/`NO` at start takes priority; ambiguous short responses handled |
| `TestParsePairedJudge` | A/B/NEITHER can be buried in long responses | Strips thinking tags; single-letter fallback for short responses; `NEITHER` recognised anywhere |

**Metrics tests** — verify that scores aggregate into the right F1/AUC numbers:

| Class | Why it exists | What it verifies |
|-------|--------------|-----------------|
| `TestComputeMetrics` | Rubric scoring (1-5) must produce correct recall/precision/F1 | Same-cluster pairs with score ≥ 3 count as recall; diff-cluster pairs with score ≤ 2 count as precision |
| `TestComputeMetricsBinary` | Binary scoring (YES/NO) uses TP/FP/FN/TN framing | `matched=True` on same-cluster = TP; `matched=True` on diff-cluster = FP; standard F1 formula |
| `TestComputeRankMetrics` | Prevents "judge inflation" from inflating all scores equally | AUC per principle via Mann-Whitney U; `discriminating_frac` = fraction of principles with AUC > 0.5 |

**End-to-end orchestration tests** — verify the full pipeline runs without a live model:

| Class | Why it exists | What it verifies |
|-------|--------------|-----------------|
| `TestRunEvalGenerate` | Generation loop must write a valid results JSON | Mocks `call_ollama`; confirms `results.json` written, contains one entry per variant × source lesson |
| `TestRunEvalJudge` | Judge loop must write scored pairs and a report | Mocks `call_judge`; confirms scored pairs JSON + `report.md` written; metrics aggregated |
| `TestRunPairedTournament` | Tournament win_rate computation must be correct | Mocks judge + `build_paired_judge_prompt`; same-group target wins counted correctly |
| `TestComputeTournamentMetrics` | Tournament aggregation must handle edge cases | Aggregates per-principle win_rates into mean, discriminating_frac, win/loss/neither totals |

**Utility tests:**

| Class | Why it exists | What it verifies |
|-------|--------------|-----------------|
| `TestCleanPrinciple` | deepseek-r1 emits reasoning traces before the principle | Strips CoT preamble, `**Principle:**` markers, parenthetical `*(This principle applies...)*` suffixes |
| `TestBayesianSignals` | Multi-signal fusion (paired + embedding + scope + mechanism) | Each extractor returns a probability in [0,1]; posterior fusion combines them correctly |

## Learning System (8 Mechanisms)

| Mechanism | Implementation | Key Component |
|-----------|---------------|--------------|
| M1 Encoding via Effort | Bjork interleaving in session-start | `interleave_due_lessons()` |
| M2 Spaced Repetition | FSRS-6 power-law forgetting curve | `compute_retrievability()`, `record_review()` |
| M3 Feedback Loop | Post-commit outcome recording | `evaluate_commit()`, post-commit hook |
| M4 Transfer | Principle extraction + cross-project matching | `transfer find` CLI |
| M5 Metacognition | KPI dashboard + calibration profiles | `kpi`, `calibrate profile` |
| M6 Amplification | Positive pattern detection + reuse recording | `detect_wins()`, `reuse record` |
| M7 Reinforcement | Variable-ratio 30% probability gate | `should_surface_positive()`, win_streaks |
| M8 Evolution | Adaptive fading: full→brief→silent→enforced | `get_fading_level()` |

**Adaptive fading thresholds** (Kalyuga expertise reversal):
- S < 2.0 → `full` (complete lesson text + code example)
- 2.0 ≤ S < 10.0 → `brief` (one-liner reminder)
- 10.0 ≤ S < 50.0 → `silent` (Semgrep rule only)
- S ≥ 50.0 → `enforced` (automated, never shown)

**Polarity differentiation**: Positive lessons start with S=3.0 (identity consolidation), negative with S=1.0. Positive surfacing uses 30% variable-ratio gate (Skinner) to prevent habituation.

## Scope Tags
language:python, domain:lessons-db

## Design Doc

See `docs/` for implementation notes and design decisions.

## Code Quality
- Lint: `make lint`
- Format: `make format`

## Quality Gates
- Before committing: `/verify`
- Before PRs: `lessons-db scan --target . --baseline HEAD`

## Lessons
- Check before planning: `/check-lessons`
- Capture after bugs: `/capture-lesson`
- Lessons: `lessons-db search` to query, `lessons-db capture` to add. DB is authoritative — never write lesson .md files directly.

## Local AI Review
- Code review: `ollama-code-review .`

## Semantic Search
- Generate: `bash scripts/generate-embeddings.sh`
- Storage: `.embeddings/` (gitignored)
