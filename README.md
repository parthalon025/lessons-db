# lessons-db

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org)
[![Tests](https://img.shields.io/badge/tests-753%20passing-brightgreen)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Security](https://github.com/parthalon025/lessons-db/actions/workflows/security.yml/badge.svg)](https://github.com/parthalon025/lessons-db/actions/workflows/security.yml)
[![CodeQL](https://github.com/parthalon025/lessons-db/actions/workflows/codeql.yml/badge.svg)](https://github.com/parthalon025/lessons-db/actions/workflows/codeql.yml)

**A lessons-learned system with spaced repetition and eval pipeline for AI-assisted development.**

lessons-db captures bugs, near-misses, and positive patterns from your development sessions, then surfaces them at the right moment using FSRS-6 spaced repetition — the same algorithm powering Anki. As you internalize a lesson, its presentation fades automatically: full text first, then a one-liner, then a silent Semgrep rule running in CI, then nothing. The eval pipeline tests prompt variants against real lesson quality using LLM judges and F1-gated auto-promotion, so the system improves itself over time.

Built to integrate with [Claude Code](https://claude.ai/claude-code) via session hooks, but usable standalone.

---

## Who This Is For

- **AI-assisted developers** using Claude Code, Cursor, or similar tools who repeat the same mistakes across sessions because context resets every conversation
- **Teams** who do retrospectives or post-mortems but find the lessons never actually change behavior
- **Researchers** interested in applying spaced repetition (FSRS-6) and reinforcement learning to developer skill acquisition
- **Anyone** building on Claude Code who wants a production example of session hooks, semantic code search, and an eval pipeline for prompt variants

**Standalone or integrated:** lessons-db works as a standalone CLI tool or as a Claude Code plugin via session hooks. The Claude Code integration adds automatic capture and surfacing, but every feature is accessible without it.

---

## Why This Exists

Every project accumulates hard-won lessons from bugs and near-misses. The problem: you forget them. A new session starts, context resets, and you repeat the same mistake.

The second problem: when systems try to prevent this, they spam you with reminders until you ignore them. lessons-db solves both:

1. **It uses cognitive science** — FSRS-6 spaced repetition, adaptive fading based on expertise reversal research, variable-ratio reinforcement for positive patterns, and SFBT exception-finding to stop surfacing what you've already internalized.
2. **It improves itself** — an eval pipeline measures whether generated principles actually transfer across contexts, scores prompt variants against a held-out test set, and auto-promotes winning configurations when F1 clears a threshold.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Claude Code Session Hooks                                      │
│  SessionStart · PreEdit · EnterPlanMode · PostBash · Stop       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Capture Pipeline                                               │
│  transcript → LLM extraction → OIL tier classification         │
│  diff → pattern detection · win detection · velocity check      │
└──────┬──────────────────────────────────┬───────────────────────┘
       │                                  │
       ▼                                  ▼
┌──────────────────┐             ┌────────────────────┐
│  SQLite DB       │             │  LanceDB            │
│  lessons         │◄───────────►│  nomic-embed-text  │
│  surfacing_events│             │  768-dim vectors    │
│  win_streaks     │             │  semantic search    │
│  eval_runs       │             └────────────────────┘
│  fsrs_state      │
└──────┬───────────┘
       │
       ├──────────────────────────────────┐
       │                                  │
       ▼                                  ▼
┌──────────────────┐             ┌────────────────────────────────┐
│  FSRS-6 Engine   │             │  Semgrep Rules                 │
│  Spaced repetition│            │  lesson → AST pattern          │
│  Adaptive fading │             │  CI enforcement                │
│  Review scheduler│             │  silent → enforced escalation  │
└──────────────────┘             └────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Eval Pipeline                                                  │
│  eval-generate (ABCDE variants × source lessons)                │
│  eval-judge (transfer test · F1 scoring · auto-promote)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## OIL Maturity Model

Every captured entry progresses through four maturity tiers:

| Tier | Meaning | Behavior |
|------|---------|----------|
| `observation` | Raw event — something happened | Stored, not yet surfaced |
| `insight` | Pattern recognized — this is why | Surfaced in planning hooks |
| `lesson` | Actionable rule — do/don't X | FSRS scheduled, Semgrep candidate |
| `lesson_learned` | Internalized — no recurrence in N commits | Fading to silent/enforced |

Lessons are classified across **6 root-cause clusters** (A–F):

- **Cluster A** — Silent Failures (external errors logged nowhere)
- **Cluster B** — Integration Boundaries (passes unit tests; breaks at the seam)
- **Cluster C** — Cold-Start (works steady-state, fails on restart)
- **Cluster D** — Specification Drift (agent builds the wrong thing correctly)
- **Cluster E** — Context & Retrieval (information available but misscoped)
- **Cluster F** — Planning & Control Flow (wrong decomposition contaminates downstream)

And **10 categories**: data-model, registration, cold-start, integration, deployment, monitoring, ui, testing, performance, security.

---

## Human Learning System (8 Mechanisms)

| Mechanism | Research Basis | Implementation |
|-----------|---------------|----------------|
| **Spaced repetition** | Ebbinghaus forgetting curve; FSRS-6 outperforms SM-2 | `fsrs.py` — power-law R=(1+F·t/S)^DECAY, 19 optimized params |
| **Adaptive fading** | Kalyuga (2007): expertise reversal effect | `get_fading_level()` — full→brief→silent→enforced by stability S |
| **Feedback loop** | Outcome recording closes the learning loop | Post-commit hook → `evaluate_commit()` |
| **Transfer** | Double-loop learning (Argyris 1977) | `meta extract-principles`, cross-project analogical matching |
| **Metacognition** | Zone of proximal development tracking | `kpi` dashboard — heeded rates, ZPD progress, review backlog |
| **Win amplification** | Safety-II: learn from what goes right (Hollnagel) | `capture detect-wins`, `reuse record` |
| **Variable reinforcement** | Skinner variable-ratio schedule | 30% probability gate on positive surfacing (`should_surface_positive()`) |
| **Exception finding** | Solution-Focused Brief Therapy (SFBT) | `learn find-exceptions` — marks internalized patterns for fading |

### Adaptive Fading Thresholds

Stability S (days) determines what the learner sees:

```
S <  2.0  →  full      — complete lesson text + code example
S <  10.0 →  brief     — one-liner reminder only
S <  50.0 →  silent    — Semgrep rule runs in CI; nothing shown
S >= 50.0 →  enforced  — fully automated; never surfaced again
```

### Velocity Detection (Burnout Prevention)

`prevention.py` tracks lesson-surfacing velocity. When lessons arrive faster than they can be absorbed, the system detects overload and reduces surfacing frequency — preventing the habituation that makes reminder systems fail.

### Polarity Differentiation

Positive lessons start with stability S=3.0 (identity consolidation phase); negative lessons start at S=1.0 (higher urgency). Positive surfacing uses the 30% variable-ratio gate; negative surfacing is deterministic. This prevents positive reinforcement from becoming background noise.

---

## Eval Pipeline

The eval pipeline answers: *which prompt × model × settings combination produces principles that actually transfer to new contexts?*

### How It Works

**Stage 1 — eval-generate:** Runs N variants (A/B/C/D/E, differing in prompt style, model, temperature, and context window) across a fixed set of source lessons. Each variant generates a principle from a lesson. Results are saved to a timestamped JSON file.

**Stage 2 — eval-judge:** For each generated principle, constructs transfer test cases using the lesson DB as ground truth:
- 2 same-cluster targets (different category) → true positives — the principle should match
- 2 different-cluster targets → true negatives — the principle should NOT match

The judge scores each (principle, target) pair on a 3-criterion rubric using a local LLM (`deepseek-r1:8b` by default) or GPT-4o-mini via `--openai`. Outputs a full F1 report.

**Auto-promotion:** Three-gate logic runs after every eval run:
1. F1 ≥ threshold
2. F1 > current production + minimum improvement margin
3. Error budget not exceeded

If all three pass, the winning variant is promoted to production automatically. Manual promotion available via `meta eval-promote`.

### Variant Design (ABCDE)

| ID | Prompt Style | Model | Rationale |
|----|-------------|-------|-----------|
| A | Few-shot (4 examples) | deepseek-r1:8b | Baseline — current production prompt |
| B | Zero-shot, causal framing | deepseek-r1:8b | R1 performs better zero-shot |
| C | Zero-shot + chunked siblings | deepseek-r1:8b | Context batching improves recall |
| D | Zero-shot, causal framing | qwen3:14b | Larger non-reasoning model |
| E | Zero-shot + chunked | qwen3:14b | Best of C and D |

---

## Install

```bash
git clone https://github.com/parthalon025/lessons-db
cd lessons-db
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Link to PATH:
```bash
ln -sf "$(pwd)/.venv/bin/lessons-db" ~/.local/bin/lessons-db
```

Verify:
```bash
lessons-db --help
```

**Prerequisites:**
- Python 3.12+
- [Ollama](https://ollama.ai) running locally: `ollama pull nomic-embed-text`
- [ollama-queue](https://github.com/parthalon025/ollama-queue) at port 7683 — all embed (`nomic-embed-text`) and analysis (`qwen3:8b`) calls route through the queue to prevent model thrashing. Override with `LESSONS_DB_OLLAMA_EMBED_URL` / `LESSONS_DB_OLLAMA_ANALYSIS_URL` to point directly at Ollama port 11434 if needed.
- [Semgrep](https://semgrep.dev) (optional, for rule generation): `pip install semgrep`

---

## Key CLI Commands

### Initialize

```bash
# Check status and DB health
lessons-db status

# Migrate existing lesson markdown files into the DB
lessons-db migrate --source /path/to/your/lessons/

# Generate semantic embeddings (requires Ollama + nomic-embed-text)
lessons-db index
```

### Capture

```bash
# Extract lessons from a Claude Code session transcript
lessons-db capture transcript session.md

# Extract positive patterns (wins) from a transcript
lessons-db capture transcript session.md --positive

# Extract lessons from a git diff (pipe or file)
lessons-db capture diff
lessons-db capture diff my.diff

# Detect wins from the most recent session
lessons-db capture detect-wins
```

### Search

```bash
# Semantic search (LanceDB vector similarity)
lessons-db search "subscriber lifecycle cleanup"
lessons-db search "async without await"
lessons-db search "exception swallowed silently" --polarity negative
```

### Hybrid Search

BM25 (keyword) + Reciprocal Rank Fusion (k=60). Falls back to BM25-only when embeddings are not available. Requires the `rank-bm25` dependency (installed automatically with `pip install -e .`).

```bash
# Hybrid BM25+RRF search (returns top 5 by default)
lessons-db hybrid-search "subscriber lifecycle cleanup"

# Return top 10 results
lessons-db hybrid-search "async without await" --top 10

# Machine-readable JSON output
lessons-db hybrid-search "exception swallowed silently" --json
```

### GraphRAG Index

Exports all lessons to markdown and indexes them with Microsoft GraphRAG. Requires `graphrag` installed at `~/.local/venvs/notion-rag/bin/graphrag`. Artifacts are stored at `~/.local/share/lessons-db/.graphrag/output/` (gitignored).

```bash
# Build the index via ollama-queue (default — non-blocking, queued as priority-3 job)
lessons-db graph-build

# Build locally (blocking — runs graphrag directly)
lessons-db graph-build --local

# Show artifact count from the last build
lessons-db graph-build --status
```

```bash
# Query the index — global mode (community synthesis, broad patterns)
lessons-db graph-search "what causes silent failures at integration boundaries"

# Query the index — local mode (entity-focused, specific lessons)
lessons-db graph-search "async discipline" --mode local
```

`graph-search` exits with a clear error if artifacts are missing, with a hint to run `graph-build --local` first.

### Spaced Repetition (FSRS)

```bash
# List lessons due for review (retrievability R < 0.9)
lessons-db fsrs due
lessons-db fsrs due --threshold 0.8

# View stability distribution and upcoming review forecast
lessons-db fsrs stats

# Backfill FSRS defaults on all existing lessons
lessons-db fsrs init
```

### Learning Feedback

```bash
# Record that a lesson was applied at a hook event
lessons-db learn record 42 --hook pre_edit

# Evaluate recent commits against surfaced lessons (did lessons influence code?)
lessons-db learn evaluate-commit

# SFBT exception-finding: identify fully internalized patterns
lessons-db learn find-exceptions
```

### KPI Dashboard and Calibration

```bash
# Full learning KPI dashboard: heeded rates, ZPD progress, win streaks
lessons-db kpi

# Per-category strength and growth area breakdown
lessons-db calibrate profile
```

### Positive Patterns

```bash
# Record reuse of a proven positive pattern
lessons-db reuse record 42

# Find cross-project analogical matches
lessons-db transfer find "caching pattern"
```

### Semgrep Rules

```bash
# Generate a Semgrep rule from a lesson
lessons-db rule generate 42

# Test all generated rules against their example code
lessons-db rule test

# Scan the current repo against all rules
lessons-db scan
```

### Meta-Learning and Eval Pipeline

```bash
# Extract transferable principles from lesson clusters
lessons-db meta extract-principles

# Generate double-loop meta-lessons from cluster analysis
lessons-db meta generate-meta-lessons

# Run prompt variant evaluation (A/B/C/D/E × source lessons)
lessons-db meta eval-generate --variants A,B,C,D,E --per-cluster 4

# Resume after transient failures (skips already-completed pairs)
lessons-db meta eval-generate --variants A --per-cluster 1 --resume

# Score generated principles and produce F1 report
lessons-db meta eval-judge results.json

# Use OpenAI as judge instead of local model
lessons-db meta eval-judge results.json --openai --judge-model gpt-4o-mini
```

---

## Claude Code Hooks

Copy or symlink scripts from `hooks/` to `~/.claude/hooks/`, then register in `~/.claude/settings.json`:

| Hook File | Event | Purpose |
|-----------|-------|---------|
| `lessons-db-session-start.sh` | SessionStart | FSRS due lessons + exception reporting + feedforward |
| `lessons-db-pre-read.sh` | PreToolUse:Read | Feedforward formatting for files being read |
| `lessons-db-pre-edit.sh` | PreToolUse:Edit\|Write | Positive reuse detection + relevant lesson surfacing |
| `lessons-db-enter-plan.sh` | PreToolUse:EnterPlanMode | Semantic search + pro-mortem + bright spots |
| `lessons-db-post-bash.sh` | PostToolUse:Bash | Test failure diagnostics + feedforward |
| `lessons-db-post-commit.sh` | post-commit | Evaluate-commit outcome recording |
| `lessons-db-stop.sh` | Stop | Auto-capture + win detection + AAR prompt |

Session-start injects FSRS-due lessons before the model begins planning. Pre-edit surfaces relevant lessons immediately before any file write. Stop captures new lessons and detects wins automatically at the end of every session.

---

## API

FastAPI server at `localhost:7685`. Used by project dashboards for status widgets.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/calibration/history` | GET | Paginated calibration run history |
| `/api/calibration/run` | POST | Queue a calibration pipeline run |
| `/api/mining/history` | GET | Paginated GitHub mining run history |
| `/api/mining/run` | POST | Queue a GitHub mining task |
| `/api/security/findings` | GET | Open Semgrep scan findings |
| `/api/security/scan` | POST | Trigger a Semgrep security scan |
| `/api/scan/summary` | GET | Decision-context dashboard (6 metrics: promotion rate, drafts captured, sessions processed, scan age, embed failure rate, FSRS review backlog) |
| `/eval/prime` | POST | Backfill `cluster_seed` for eval pipeline readiness |

---

## Performance

Pattern-scan and embedding calls are optimized to avoid redundant Ollama round-trips:

- **File filter** — `_SKIP_FILENAMES` and `_SKIP_DIRS` drop lock files, config files, and dependency directories (`node_modules`, `.venv`) before any embedding occurs.
- **Suppression embedding cache** — module-level dict keyed by SHA256 of snippet content; avoids re-embedding up to 500 suppression snippets per candidate evaluation call.
- **Semgrep patterns cache** — SHA256-keyed JSON at `~/.local/share/lessons-db/semgrep-patterns-cache.json`; a cache hit skips ~378 Ollama calls on warm runs.
- **Per-repo block cap** — `MAX_BLOCKS_PER_REPO=20` in the outer loop of `extract_nonpython_candidates()` prevents unbounded embedding across large repos.
- **Draft cap** — transcript capture truncates LLM output at 50 lessons per session.
- **Embed timeout** — 300s, matching `PROXY_WAIT_TIMEOUT` in ollama-queue so long jobs don't time out prematurely.

---

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `LESSONS_DB_DATA_DIR` | `~/.local/share/lessons-db/` | SQLite DB + LanceDB + rules |
| `LESSONS_DB_OLLAMA_EMBED_URL` | `http://127.0.0.1:7683` | Embedding API (routes through ollama-queue) |
| `LESSONS_DB_OLLAMA_ANALYSIS_URL` | `http://127.0.0.1:7683` | Analysis/capture API |
| `LESSONS_DB_OLLAMA_QUEUE_URL` | `http://127.0.0.1:7683` | Queue API for generation tasks |
| `LESSONS_DB_OLLAMA_ANALYSIS_MODEL` | `qwen3:8b` | Model for analysis and capture |
| `LESSONS_DB_REPO_CACHE_DIR` | `~/.local/share/lessons-db/repo-cache` | Local git clone cache for GitHub mining |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Storage | SQLite (stdlib) — no external DB dependency |
| Vector search | LanceDB — embedded, no server required |
| Embeddings | `nomic-embed-text` via Ollama (768 dimensions) |
| Keyword search | `rank-bm25` (BM25Okapi) — fused with semantic via RRF |
| Pattern detection | Semgrep — reused, not rebuilt |
| Spaced repetition | FSRS-6 — implemented directly (~500 lines, no external deps) |
| Analysis / capture | Ollama (`qwen3:8b` default) via ollama-queue |
| Graph indexing | Microsoft GraphRAG (`~/.local/venvs/notion-rag/`) — artifacts at `~/.local/share/lessons-db/.graphrag/` |
| CLI | Click with subcommands |
| API | FastAPI + uvicorn |

---

## Optional Extras

```bash
# HDBSCAN adaptive cluster discovery
pip install -e ".[clustering]"
lessons-db cluster discover

# GitHub mining pipeline (mutation testing integration)
pip install -e ".[mining]"
```

---

## Tests

```bash
source .venv/bin/activate
pytest --timeout=120 -x -q -n 6   # parallel (recommended, ~1 min)
pytest --timeout=120 -x -q -n 0   # single-threaded (debug)
```

753 tests across all modules. Run the parallel suite before committing.

---

## Project Structure

```
src/lessons_db/
  cli.py          # Click CLI: all subcommands
  config.py       # Paths, Ollama URLs, env var overrides
  db.py           # SQLite schema, migrations, CRUD
  fsrs.py         # FSRS-6: retrievability, stability, adaptive fading
  learn.py        # Surfacing events, outcome recording, SFBT exception-finding
  prevention.py   # Velocity detection, fix queue, content checking
  eval.py         # eval-generate (ABCDE variants) + eval-judge (F1 report)
  capture.py      # Auto-capture from transcript/diff + win detection
  vectors.py      # LanceDB + Ollama embedding via ollama-queue
  search.py       # Semantic search + file-path + content match
  rulegen.py      # Generate Semgrep rules from lessons
  scan.py         # Trigger + parse Semgrep scans (SARIF)
  enforce.py      # Escalation ladder, recurrence tracking
  github_miner.py # GitHub mining: discover_repos, mine_repos_for_gaps
  migrate.py      # Parse legacy markdown lessons → DB + generate rules
  export.py       # Generate markdown from DB records
graphrag/         # GraphRAG configuration (git-tracked)
  settings.yml    # v3 completion_models schema → ollama-queue proxy at localhost:7683
  prompts/        # entity_extraction, community_report, summarize_descriptions
rules/            # Community Semgrep rules (lesson-derived)
  python/
  testing/
  patterns/
hooks/            # Claude Code hook scripts
scripts/          # batch-capture-transcripts.sh, run-batch-pipeline.sh
tests/
```

---

## Research Grounding

The learning mechanisms are grounded in peer-reviewed research:

**Spaced repetition and forgetting**
- [Ebbinghaus Forgetting Curve Replication — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/)
- [FSRS Algorithm (open-spaced-repetition)](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)
- [Why FSRS Outperforms SM-2 (Denicola, 2025)](https://domenic.me/fsrs/)

**Desirable difficulties and encoding via effort**
- [Bjork Lab: Desirable Difficulties (UCLA)](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/RBjork_inpress.pdf)

**Expertise reversal (adaptive fading)**
- [Kalyuga (2007): Expertise Reversal Effect](https://www.uky.edu/~gmswan3/EDC608/Kalyuga2007_Article_ExpertiseReversalEffectAndItsI.pdf)

**Learning from success, not just failure**
- [Learning from Success or Failure — Frontiers in Psychology](https://www.frontiersin.org/articles/10.3389/fpsyg.2020.01627/full)
- [Not Learning From Failure — Eskreis-Winkler & Fishbach (2019)](https://journals.sagepub.com/doi/abs/10.1177/0956797619881133)
- [Safety-II: Learning from What Goes Right — Hollnagel](https://erikhollnagel.com/ideas/safety-i%20and%20safety-ii.html)

**Exception finding (SFBT)**
- [Solution-Focused Brief Therapy — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10098109/)

**Variable-ratio reinforcement**
- [Schedules of Reinforcement — Simply Psychology](https://www.simplypsychology.org/schedules-of-reinforcement.html)

**Double-loop learning (meta-lessons)**
- [Argyris (1977): Double Loop Learning in Organizations — HBR](https://theisrm.org/documents/Argyris%20(1977)%20Double%20Loop%20Learning%20in%20Organizations.pdf)

---

## License

MIT
