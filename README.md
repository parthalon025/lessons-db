# lessons-db

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Security](https://github.com/parthalon025/lessons-db/actions/workflows/security.yml/badge.svg)](https://github.com/parthalon025/lessons-db/actions/workflows/security.yml)
[![CodeQL](https://github.com/parthalon025/lessons-db/actions/workflows/codeql.yml/badge.svg)](https://github.com/parthalon025/lessons-db/actions/workflows/codeql.yml)

Lessons-learned system that prevents you from repeating the same coding mistakes. Captures bugs into a database, surfaces them before you code using spaced repetition, and generates Semgrep rules to catch anti-patterns in CI.

## Why

Every project accumulates hard-won lessons from bugs and near-misses. The problem: you forget them. A new session starts, the context resets, and you repeat the same mistake.

lessons-db treats those lessons like flashcards. It stores them in SQLite + LanceDB with semantic embeddings, surfaces the relevant ones before you plan using FSRS-6 spaced repetition (the same algorithm used by Anki), and gets out of the way as your internalization improves — showing full lessons at first, then one-liners, then nothing (just a Semgrep rule running silently in CI). It also captures wins and positive patterns, not just failures.

Built to integrate with [Claude Code](https://claude.ai/claude-code) via session hooks, but usable standalone.

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
- [Semgrep](https://semgrep.dev) (optional, for rule generation): `pip install semgrep`

## Use

### First run

```bash
# Initialize and check status
lessons-db status

# If you have existing lesson markdown files:
lessons-db migrate --source /path/to/your/lessons/

# Generate embeddings (requires Ollama with nomic-embed-text)
lessons-db index

# Verify search works
lessons-db search "exception swallowed silently"
```

### Search and capture

```bash
# Semantic search
lessons-db search "subscriber lifecycle cleanup"
lessons-db search "async without await"

# Capture a new lesson
lessons-db capture transcript session.md          # extract from Claude Code transcript
lessons-db capture transcript session.md --positive   # extract wins
lessons-db capture diff                           # extract from git diff (stdin)

# Export a lesson to markdown
lessons-db export 42
```

### Spaced repetition

```bash
# List lessons due for review (FSRS retrievability < 0.9)
lessons-db fsrs due

# View stability distribution + upcoming review forecast
lessons-db fsrs stats

# Initialize FSRS parameters on existing lessons
lessons-db fsrs init
```

### Learning feedback

```bash
# Record that a lesson was applied (or dismissed)
lessons-db learn record 42 --hook pre_edit

# Find lessons you've fully internalized (no anti-patterns in recent code)
lessons-db learn find-exceptions

# Evaluate recent commits against surfaced lessons
lessons-db learn evaluate-commit
```

### KPI and calibration

```bash
# Dashboard: heeded rates, ZPD progress, win streaks
lessons-db kpi

# Per-category strength/growth breakdown
lessons-db calibrate profile
```

### Positive patterns

```bash
lessons-db capture detect-wins              # detect wins from recent session
lessons-db reuse record 42                  # record that you reused a good pattern
lessons-db transfer find "caching pattern"  # cross-project analogical matching
```

### Semgrep rules

```bash
lessons-db rule generate 42     # generate Semgrep rule from lesson
lessons-db rule test             # test all generated rules
lessons-db scan                  # scan current repo against all rules
```

### Meta-learning (requires Ollama)

```bash
lessons-db meta extract-principles           # batch-extract transferable principles
lessons-db meta generate-meta-lessons        # generate double-loop meta-lessons from clusters
lessons-db meta eval-generate --variants A,B,C,D,E   # generate eval variants
lessons-db meta eval-judge results.json      # score principles, produce F1 report
```

## How It Works

Lessons go through 8 learning science mechanisms:

| Mechanism | What It Does |
|-----------|-------------|
| **Spaced repetition** | FSRS-6 algorithm schedules each lesson based on your retrieval history |
| **Adaptive fading** | Lessons transition: full text → one-liner → silent (Semgrep only) → enforced as you internalize them |
| **Feedback loop** | Post-commit hook evaluates whether surfaced lessons influenced your code |
| **Transfer** | Principle extraction finds patterns that apply across projects |
| **Metacognition** | KPI dashboard shows heeded rates, review backlog, ZPD progress |
| **Win amplification** | Positive patterns get captured and reinforced, not just failures |
| **Variable reinforcement** | 30% random gate on positive surfacing (Skinner) prevents habituation |
| **Exception finding** | SFBT-style: identifies patterns you've already internalized so they fade |

**Adaptive fading thresholds** (stability S in days):
- `S < 2.0` → full lesson text + code example
- `2.0 ≤ S < 10.0` → one-liner reminder
- `10.0 ≤ S < 50.0` → silent (Semgrep rule only)
- `S ≥ 50.0` → enforced (automated, never shown)

## Claude Code Hooks

Copy or symlink scripts from `hooks/` to `~/.claude/hooks/`, then register in `~/.claude/settings.json`:

| Hook file | Event | Purpose |
|-----------|-------|---------|
| `lessons-db-session-start.sh` | SessionStart | FSRS due lessons + exception reporting + feedforward |
| `lessons-db-pre-read.sh` | PreToolUse:Read | Feedforward formatting for files being read |
| `lessons-db-pre-edit.sh` | PreToolUse:Edit\|Write | Positive reuse detection + relevant lesson surfacing |
| `lessons-db-enter-plan.sh` | PreToolUse:EnterPlanMode | Semantic search + pro-mortem + bright spots |
| `lessons-db-post-bash.sh` | PostToolUse:Bash | Test failure diagnostics + feedforward |
| `lessons-db-post-commit.sh` | post-commit | Evaluate-commit outcome recording |
| `lessons-db-stop.sh` | Stop | Auto-capture + win detection + AAR prompt |

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `LESSONS_DB_DATA_DIR` | `~/.local/share/lessons-db/` | SQLite DB + LanceDB + rules |
| `LESSONS_DB_OLLAMA_EMBED_URL` | `http://127.0.0.1:7683` | Ollama API for embeddings (routes through ollama-queue) |
| `LESSONS_DB_OLLAMA_ANALYSIS_URL` | `http://127.0.0.1:7683` | Ollama API for analysis/capture |
| `LESSONS_DB_OLLAMA_QUEUE_URL` | `http://127.0.0.1:7683` | ollama-queue API for generation tasks |
| `LESSONS_DB_OLLAMA_ANALYSIS_MODEL` | `qwen3:8b` | Model for analysis and capture |
| `LESSONS_DB_REPO_CACHE_DIR` | `~/.local/share/lessons-db/repo-cache` | Local git clone cache for GitHub mining |

## Optional Extras

```bash
# HDBSCAN adaptive cluster discovery
pip install -e ".[clustering]"
lessons-db cluster discover

# Mutation testing integration
pip install -e ".[mining]"
```

## Requirements

```
Python 3.12+
click>=8.1.0
lancedb>=0.20.0
pyarrow>=15.0.0
requests>=2.31.0
openai>=1.0.0
pyyaml>=6.0
pydriller>=2.6
fastapi>=0.115.0
uvicorn>=0.32.0
```

Dev/test: `pip install -r requirements-dev.txt`

## Tests

```bash
source .venv/bin/activate
pytest --timeout=120 -x -q -n 6   # parallel (recommended, ~1 min)
pytest --timeout=120 -x -q -n 0   # single-threaded (debug)
```

## Research

The learning mechanisms are grounded in peer-reviewed research. Key sources by mechanism:

**Spaced repetition & forgetting**
- [Ebbinghaus Forgetting Curve Replication — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4492928/)
- [FSRS Algorithm (open-spaced-repetition)](https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm)
- [Why FSRS Outperforms SM-2 (Denicola, 2025)](https://domenic.me/fsrs/)

**Desirable difficulties & encoding via effort**
- [Bjork Lab: Desirable Difficulties (UCLA)](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/07/RBjork_inpress.pdf)
- [Desirable Difficulties in Theory and Practice (ResearchGate)](https://www.researchgate.net/publication/347931447_Desirable_Difficulties_in_Theory_and_Practice)

**Expertise reversal (adaptive fading)**
- [Kalyuga (2007): Expertise Reversal Effect](https://www.uky.edu/~gmswan3/EDC608/Kalyuga2007_Article_ExpertiseReversalEffectAndItsI.pdf)

**Learning from success, not just failure**
- [Learning from Success or Failure? — Frontiers in Psychology](https://www.frontiersin.org/articles/10.3389/fpsyg.2020.01627/full)
- [Not Learning From Failure — Eskreis-Winkler & Fishbach (2019)](https://journals.sagepub.com/doi/abs/10.1177/0956797619881133)
- [Safety-II: Learning from What Goes Right — Hollnagel](https://erikhollnagel.com/ideas/safety-i%20and%20safety-ii.html)

**Exception finding (SFBT)**
- [Solution-Focused Brief Therapy — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10098109/)

**Feedforward (positive reinforcement framing)**
- [Feedforward Instead of Feedback — Marshall Goldsmith](https://www.marshallgoldsmith.com/post/try-feedforward-instead-of-feedback)

**Variable-ratio reinforcement (positive pattern gate)**
- [Schedules of Reinforcement — Simply Psychology](https://www.simplypsychology.org/schedules-of-reinforcement.html)

**Pre-mortem (prospective hindsight at planning boundary)**
- [The Pro-Mortem Method — Psychology Today](https://www.psychologytoday.com/us/blog/seeing-what-others-dont/201510/the-pro-mortem-method)

**Double-loop learning (meta-lessons)**
- [Argyris (1977): Double Loop Learning in Organizations — HBR](https://theisrm.org/documents/Argyris%20(1977)%20Double%20Loop%20Learning%20in%20Organizations.pdf)

## License

MIT
