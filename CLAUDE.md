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

## API Endpoints

Exposed via FastAPI at `localhost:7685` (proxied by project-hub Express at `/hub/api/lessons/*`):

- **GET /api/calibration/history** — paginated `calibration_runs` (default: last 20, max 100). Response: `[{id, run_date, dataset, bugs_sampled, pass_rate, gate14_pass, notes, ...}]`
- **POST /api/calibration/run** — queues `calibrate_pipeline()` as background task. Query params: `sample_n` (default 50), `skip_extraction` (default false). Response: `{status: "queued"}`.
- **GET /api/mining/history** — paginated `mining_runs`. Response: `[{id, run_date, repos_searched, commits_analyzed, candidates_extracted, diff_size_rejected, gate0_rejected, gate1_rejected, gate2_rejected, gate3_rejected, gate4_rejected, auto_approved, drafted, conflicts_flagged, error_count, duration_seconds}]`
- **POST /api/mining/run** — queues GitHub mining task
- **GET /api/security/findings** — open scan findings
- **POST /api/security/scan** — trigger Semgrep security scan
- **GET /api/scan/summary** — decision-context dashboard: promotion rate, drafts captured last run, sessions processed, scan age, embed failure rate, FSRS review backlog

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
- **Ollama** via ollama-queue for all tasks including embeddings (nomic-embed-text, 768 dims) and analysis (default: qwen3:8b); embed calls route through queue at port 7683 (PROXY_WAIT_TIMEOUT=300s)
- **FSRS-6** implemented directly (~500 lines, no external deps) — power-law forgetting curve R=(1+F*t/S)^DECAY with 19 optimized parameters
- **Click CLI** with subcommands matching design doc

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
