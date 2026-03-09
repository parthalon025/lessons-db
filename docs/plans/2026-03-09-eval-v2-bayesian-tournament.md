# Eval V2: Bayesian Tournament Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the rubric-based eval judge with a multi-signal Bayesian fusion pipeline that uses paired tournament comparisons, mechanism extraction, and simulation validation — achieving >0.85 AUC with zero human-in-the-loop.

**Architecture:** The current eval pipeline generates principles from lessons then judges them via 1-5 rubric or binary YES/NO. It fails because (1) cluster_seed ground truth is noisy (73% of lessons in one mega-cluster), (2) absolute judgment inflates scores, and (3) a single judge signal can't discriminate. The V2 pipeline fixes ground truth by using `category` instead of `cluster_seed`, replaces absolute judgment with paired tournament comparisons, and fuses 4 independent signals via naïve Bayes log-likelihood ratios — the same math used in the ollama-queue stall detector and ARIA occupancy system.

**Tech Stack:** Python 3.12+, SQLite, Click CLI, Ollama via ollama-queue proxy, Preact SPA (ollama-queue dashboard)

**Cross-Repo:** Changes span two repos:
- `lessons-db` — CLI eval commands, eval.py engine, tests
- `ollama-queue` — eval_engine.py, DB schema, API endpoints, SPA dashboard

---

## Phase 0: Fix Ground Truth (lessons-db only)

The single highest-leverage change. "Must-Do Coding Rules" cluster_seed contains 625/854 lessons across 15 unrelated categories. Switch eval ground truth from `cluster_seed` to `category`.

### Task 1: Add `--group-by` flag to eval-generate

**Files:**
- Modify: `src/lessons_db/eval.py` — `select_source_lessons()`, `select_transfer_targets()`
- Modify: `src/lessons_db/cli.py` — `meta_eval_generate`, `meta_eval_judge`
- Test: `tests/test_eval.py`

**Step 1: Write failing test for category-based source selection**

```python
class TestSelectSourceLessonsByCategory:
    """select_source_lessons groups by category when group_by='category'."""

    def test_returns_lessons_from_distinct_categories(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)  # existing helper seeds cluster_seed
        # Override: set categories on seeded lessons
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.commit()
        result = select_source_lessons(conn, per_cluster=2, group_by="category")
        categories = {r["category"] for r in result}
        assert "error-handling" in categories
        assert "testing" in categories
        conn.close()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval.py::TestSelectSourceLessonsByCategory -xvs`
Expected: FAIL — `select_source_lessons()` doesn't accept `group_by`

**Step 3: Implement group_by parameter**

In `eval.py`, modify `select_source_lessons()`:
- Add `group_by: str = "cluster_seed"` parameter
- Replace hardcoded `cluster_seed` grouping with `group_by` column
- Validate `group_by in ("cluster_seed", "category")`

In `select_transfer_targets()`:
- Add `group_by: str = "cluster_seed"` parameter
- Same-group targets use `WHERE {group_by} = ?`
- Diff-group targets use `WHERE {group_by} != ?`

**Step 4: Run tests**

Run: `pytest tests/test_eval.py -x -q -n 6`
Expected: All pass

**Step 5: Wire CLI flag**

In `cli.py`, add to both `meta_eval_generate` and `meta_eval_judge`:
```python
@click.option("--group-by", type=click.Choice(["cluster_seed", "category"]),
              default="category", help="Grouping field for same/diff targets (default: category).")
```

Pass through to `run_eval_generate()` and `run_eval_judge()`.

**Step 6: Commit**

```bash
git add src/lessons_db/eval.py src/lessons_db/cli.py tests/test_eval.py
git commit -m "feat(eval): add --group-by flag, default to category instead of cluster_seed"
```

### Task 2: Validate category ground truth

**Files:**
- Create: `scripts/eval-category-audit.py` (one-time diagnostic)

**Step 1: Write audit script**

Script that:
1. For each category with >=5 lessons, sample 3 pairs of lessons
2. Compute embedding cosine similarity within-category vs across-category
3. Report: mean within-category similarity, mean across-category similarity, separation ratio
4. Flag categories where within < across (bad category)

**Step 2: Run it**

```bash
python3 scripts/eval-category-audit.py
```

Expected: within-category similarity > across-category for most categories.

**Step 3: Commit**

```bash
git add scripts/eval-category-audit.py
git commit -m "chore(eval): add category ground truth audit script"
```

---

## Phase 1: Paired Tournament Judge (lessons-db)

Replace absolute YES/NO with paired comparison. Given a principle, present (same-category target, diff-category target) and ask which one the principle applies to more specifically.

### Task 3: Build paired tournament prompt

**Files:**
- Modify: `src/lessons_db/eval.py`
- Test: `tests/test_eval.py`

**Step 1: Write failing test**

```python
class TestBuildPairedPrompt:
    def test_contains_both_targets(self):
        same = {"title": "Resource cleanup", "one_liner": "Close connections", "description": "..."}
        diff = {"title": "CSS specificity", "one_liner": "Use BEM", "description": "..."}
        prompt = build_paired_judge_prompt("Always close resources", same, diff)
        assert "TARGET A" in prompt
        assert "TARGET B" in prompt
        assert "Resource cleanup" in prompt
        assert "CSS specificity" in prompt

    def test_asks_for_a_or_b(self):
        same = {"title": "T1", "one_liner": "O1", "description": "D1"}
        diff = {"title": "T2", "one_liner": "O2", "description": "D2"}
        prompt = build_paired_judge_prompt("Principle", same, diff)
        assert "'A'" in prompt or "A" in prompt
        assert "'B'" in prompt or "B" in prompt

    def test_randomizes_position(self):
        """Over many calls, same-cluster target should appear in both A and B positions."""
        same = {"title": "Same", "one_liner": "S", "description": "S"}
        diff = {"title": "Diff", "one_liner": "D", "description": "D"}
        positions = set()
        for seed in range(20):
            prompt = build_paired_judge_prompt("P", same, diff, position_seed=seed)
            if "Same" in prompt.split("TARGET B")[0]:
                positions.add("A")
            else:
                positions.add("B")
        assert len(positions) == 2  # both positions used
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval.py::TestBuildPairedPrompt -xvs`

**Step 3: Implement**

```python
def build_paired_judge_prompt(
    principle: str,
    same_target: dict[str, Any],
    diff_target: dict[str, Any],
    position_seed: int | None = None,
) -> str:
    """Paired comparison prompt — which target does the principle apply to more?

    Randomizes A/B position to eliminate position bias.
    Returns prompt and whether same_target is in position A.
    """
    principle = _clean_principle(principle)
    import hashlib
    if position_seed is None:
        position_seed = int(hashlib.md5(principle.encode()).hexdigest()[:8], 16)
    swap = position_seed % 2 == 0

    target_a = diff_target if swap else same_target
    target_b = same_target if swap else diff_target

    def _fmt(t):
        title = t.get("title") or ""
        one_liner = t.get("one_liner") or ""
        desc = (t.get("description") or "")[:200]
        return f"Title: {title}\nOne-liner: {one_liner}\nDescription: {desc}"

    prompt = (
        f'PRINCIPLE: "{principle}"\n\n'
        f"TARGET A:\n{_fmt(target_a)}\n\n"
        f"TARGET B:\n{_fmt(target_b)}\n\n"
        "Which target does this principle apply to MORE specifically?\n"
        "Consider the STRUCTURAL failure mechanism, not surface-level topic similarity.\n\n"
        "Rules:\n"
        "- Pick the target where the principle identifies the EXACT same bug class.\n"
        "- If neither applies well, answer NEITHER.\n\n"
        "Answer ONLY: A, B, or NEITHER"
    )
    same_is_a = not swap
    return prompt, same_is_a


def parse_paired_judge(response: str) -> str | None:
    """Parse A/B/NEITHER from paired comparison response."""
    if not response:
        return None
    text = response.strip().upper()
    text = _re.sub(r"<THINK>.*?</THINK>", "", text, flags=_re.DOTALL).strip()
    if text.startswith("A"):
        return "A"
    if text.startswith("B"):
        return "B"
    if "NEITHER" in text:
        return "NEITHER"
    for ch in ["A", "B"]:
        if ch in text and len(text) < 30:
            return ch
    return None
```

**Step 4: Run tests**

Run: `pytest tests/test_eval.py -x -q -n 6`

**Step 5: Commit**

```bash
git add src/lessons_db/eval.py tests/test_eval.py
git commit -m "feat(eval): add paired tournament judge prompt and parser"
```

### Task 4: Build tournament runner

**Files:**
- Modify: `src/lessons_db/eval.py` — add `run_paired_tournament()`
- Test: `tests/test_eval.py`

**Step 1: Write failing test**

```python
class TestRunPairedTournament:
    def test_returns_win_rate(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        # Set categories
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.commit()

        results_data = {
            "meta": {"variants": ["A"]},
            "results": [{
                "variant": "A", "lesson_id": ids["A"][0],
                "cluster_seed": "A", "category": "error-handling",
                "principle": "Always close resources in finally blocks.",
                "error": None,
            }],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))

        def mock_judge(prompt, **kwargs):
            return "A"  # always picks A

        with patch("lessons_db.eval.call_judge", side_effect=mock_judge):
            tournament_results = run_paired_tournament(
                results_path=results_path, conn=conn,
                backend="ollama", group_by="category",
                pairs_per_principle=4,
            )

        assert len(tournament_results) > 0
        for r in tournament_results:
            assert "win_rate" in r
            assert "comparisons" in r
            assert 0.0 <= r["win_rate"] <= 1.0
        conn.close()
```

**Step 2: Run test to verify it fails**

**Step 3: Implement `run_paired_tournament()`**

For each principle in the results JSON:
1. Get N same-category targets and N diff-category targets
2. Create N paired comparisons (one same + one diff per pair)
3. Call judge with paired prompt, randomizing A/B position
4. Track: did the judge pick the same-category target?
5. Win rate = correct picks / total comparisons
6. Return list of `{variant, principle, win_rate, comparisons, wins}`

**Step 4: Run tests, commit**

### Task 5: Add tournament metrics (AUC from win rates)

**Files:**
- Modify: `src/lessons_db/eval.py` — add `compute_tournament_metrics()`
- Test: `tests/test_eval.py`

**Step 1: Write failing test**

```python
class TestComputeTournamentMetrics:
    def test_perfect_discrimination(self):
        results = [
            {"variant": "A", "win_rate": 1.0, "comparisons": 4, "wins": 4},
            {"variant": "A", "win_rate": 1.0, "comparisons": 4, "wins": 4},
        ]
        metrics = compute_tournament_metrics(results)
        assert metrics["A"]["mean_win_rate"] == 1.0
        assert metrics["A"]["auc"] == 1.0

    def test_random_discrimination(self):
        results = [
            {"variant": "A", "win_rate": 0.5, "comparisons": 4, "wins": 2},
        ]
        metrics = compute_tournament_metrics(results)
        assert metrics["A"]["mean_win_rate"] == 0.5
```

**Step 2-5: Implement, test, commit**

Metrics per variant:
- `mean_win_rate`: average win rate across principles (≈ AUC)
- `discriminating_frac`: fraction of principles with win_rate > 0.5
- `principle_count`: number of principles evaluated
- `comparison_count`: total comparisons made

### Task 6: Wire `eval-tournament` CLI command

**Files:**
- Modify: `src/lessons_db/cli.py`
- Test: `tests/test_eval.py` (CLI integration test optional)

**Step 1: Add CLI command**

```python
@meta.command("eval-tournament")
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default=None)
@click.option("--judge-model", default=None, help="Default: gemma3:12b")
@click.option("--group-by", type=click.Choice(["cluster_seed", "category"]),
              default="category")
@click.option("--pairs-per-principle", type=int, default=4,
              help="Number of paired comparisons per principle.")
@click.option("--priority", type=int, default=None)
@click.pass_context
def meta_eval_tournament(ctx, results_file, output, judge_model, group_by,
                         pairs_per_principle, priority):
    """Run paired tournament evaluation on generated principles."""
```

**Step 2: Commit**

```bash
git add src/lessons_db/cli.py
git commit -m "feat(eval): add eval-tournament CLI command with paired comparisons"
```

---

## Phase 2: Mechanism Extraction (lessons-db)

Replace single-sentence principles with mechanism triplets (trigger, target, fix) extracted from cross-lesson comparison.

### Task 7: Build mechanism extraction prompt

**Files:**
- Modify: `src/lessons_db/eval.py`
- Test: `tests/test_eval.py`

**Step 1: Write failing test**

```python
class TestBuildMechanismPrompt:
    def test_contains_both_lessons(self):
        lesson_a = {"title": "Resource cleanup", "one_liner": "Close DB connections",
                     "description": "Database connections left open in error paths"}
        lesson_b = {"title": "File handle leak", "one_liner": "Close file handles",
                     "description": "File handles not closed when exception thrown"}
        prompt = build_mechanism_extraction_prompt(lesson_a, lesson_b)
        assert "Resource cleanup" in prompt
        assert "File handle leak" in prompt

    def test_requests_triplet_format(self):
        a = {"title": "A", "one_liner": "A", "description": "A"}
        b = {"title": "B", "one_liner": "B", "description": "B"}
        prompt = build_mechanism_extraction_prompt(a, b)
        assert "trigger" in prompt.lower()
        assert "target" in prompt.lower()
        assert "fix" in prompt.lower()
```

**Step 2-3: Implement**

```python
def build_mechanism_extraction_prompt(lesson_a: dict, lesson_b: dict) -> str:
    """Extract shared failure mechanism as a triplet from two lessons."""
    def _fmt(l):
        return (f"Title: {l.get('title','')}\n"
                f"One-liner: {l.get('one_liner','')}\n"
                f"Description: {(l.get('description','') or '')[:300]}")

    return (
        "You are analyzing two software engineering lessons that share a failure pattern.\n\n"
        f"LESSON A:\n{_fmt(lesson_a)}\n\n"
        f"LESSON B:\n{_fmt(lesson_b)}\n\n"
        "Extract the SPECIFIC structural mechanism these two lessons share.\n\n"
        "Format your answer as exactly three lines:\n"
        "TRIGGER: [what condition causes the bug, 3-10 words]\n"
        "TARGET: [what component/resource breaks, 3-10 words]\n"
        "FIX: [what structural change prevents it, 3-10 words]\n\n"
        "Rules:\n"
        "- Be SPECIFIC — 'error handling' is too vague. "
        "'Uncaught exception in cleanup path' is specific.\n"
        "- Name the MECHANISM, not the topic. Two lessons about 'testing' may have "
        "completely different mechanisms.\n"
        "- If these lessons do NOT share a specific mechanism, answer: NONE"
    )


def parse_mechanism_triplet(response: str) -> dict[str, str] | None:
    """Parse TRIGGER/TARGET/FIX triplet from mechanism extraction response."""
    if not response:
        return None
    text = _re.sub(r"<THINK>.*?</THINK>", "", response, flags=_re.DOTALL).strip()
    if "NONE" in text.upper() and len(text) < 50:
        return None
    trigger = _re.search(r"TRIGGER:\s*(.+)", text)
    target = _re.search(r"TARGET:\s*(.+)", text)
    fix = _re.search(r"FIX:\s*(.+)", text)
    if not trigger or not target or not fix:
        return None
    return {
        "trigger": trigger.group(1).strip()[:100],
        "target": target.group(1).strip()[:100],
        "fix": fix.group(1).strip()[:100],
    }
```

**Step 4-5: Test, commit**

### Task 8: Build mechanism-based generation variant

**Files:**
- Modify: `src/lessons_db/eval.py` — add to VARIANT_CONFIGS, update `_generate_for_lesson()`
- Test: `tests/test_eval.py`

Add variant "M" (mechanism) that:
1. Samples 2 same-category siblings of the source lesson
2. Calls mechanism extraction prompt with (source, sibling) pairs
3. Returns the mechanism triplet as the "principle" (formatted as single string)

**Step 1-5: Test, implement, commit**

---

## Phase 3: Bayesian Signal Fusion (lessons-db)

Combine 4 independent signals into a posterior P(transfers) per (principle, target) pair.

### Task 9: Implement signal extractors

**Files:**
- Modify: `src/lessons_db/eval.py`
- Test: `tests/test_eval.py`

**Step 1: Write failing tests**

```python
class TestSignalExtractors:
    def test_paired_signal(self):
        """Paired comparison outcome → log-likelihood ratio."""
        assert compute_paired_signal(winner="same") > 0
        assert compute_paired_signal(winner="diff") < 0
        assert compute_paired_signal(winner="neither") == 0

    def test_embedding_signal(self):
        """Cosine similarity → log-likelihood ratio."""
        assert compute_embedding_signal(0.9) > 0  # high sim → positive
        assert compute_embedding_signal(0.2) < 0  # low sim → negative

    def test_scope_signal(self):
        """Scope tag overlap → log-likelihood ratio."""
        assert compute_scope_signal({"python", "async"}, {"python", "async"}) > 0
        assert compute_scope_signal({"python"}, {"javascript"}) < 0
```

**Step 2-3: Implement**

```python
# Log-likelihood ratios calibrated from the 10-method experiment
def compute_paired_signal(winner: str) -> float:
    """Convert paired comparison outcome to log-LR."""
    return {"same": 2.5, "diff": -2.5, "neither": 0.0}.get(winner, 0.0)

def compute_embedding_signal(cosine_sim: float) -> float:
    """Convert cosine similarity to log-LR. Calibrated from AUC=0.707 baseline."""
    if cosine_sim >= 0.7:
        return 1.5
    elif cosine_sim >= 0.5:
        return 0.5
    elif cosine_sim >= 0.3:
        return -0.5
    else:
        return -1.5

def compute_scope_signal(principle_scopes: set, target_scopes: set) -> float:
    """Convert scope tag overlap to log-LR."""
    if not principle_scopes or not target_scopes:
        return 0.0  # no signal
    overlap = len(principle_scopes & target_scopes) / len(principle_scopes | target_scopes)
    if overlap >= 0.5:
        return 1.0
    elif overlap > 0:
        return 0.3
    else:
        return -0.5

def compute_mechanism_signal(mechanism_match: bool | None) -> float:
    """Convert mechanism-naming match to log-LR. P=1.0, R=0.5 calibrated."""
    if mechanism_match is True:
        return 2.0  # strong positive (P=1.0)
    elif mechanism_match is False:
        return -1.5
    else:
        return 0.0  # no signal
```

### Task 10: Implement Bayesian fusion

**Files:**
- Modify: `src/lessons_db/eval.py`
- Test: `tests/test_eval.py`

**Step 1: Write failing test**

```python
class TestBayesianFusion:
    def test_all_positive_signals(self):
        """All positive signals → high posterior."""
        posterior = compute_transfer_posterior(
            paired_signal=2.5,
            embedding_signal=1.5,
            scope_signal=1.0,
            mechanism_signal=2.0,
        )
        assert posterior > 0.9

    def test_all_negative_signals(self):
        posterior = compute_transfer_posterior(
            paired_signal=-2.5,
            embedding_signal=-1.5,
            scope_signal=-0.5,
            mechanism_signal=-1.5,
        )
        assert posterior < 0.1

    def test_mixed_signals(self):
        """Conflicting signals → moderate posterior."""
        posterior = compute_transfer_posterior(
            paired_signal=2.5,
            embedding_signal=-1.5,
            scope_signal=0.0,
            mechanism_signal=0.0,
        )
        assert 0.3 < posterior < 0.7

    def test_prior_is_skeptical(self):
        """With no signals, posterior equals prior (0.25)."""
        posterior = compute_transfer_posterior(0.0, 0.0, 0.0, 0.0)
        assert abs(posterior - 0.25) < 0.01
```

**Step 2-3: Implement**

```python
import math

# Prior: P(transfers) = 0.25 — most principles DON'T transfer
_PRIOR_LOG_ODDS = math.log(0.25 / 0.75)  # ≈ -1.10

def compute_transfer_posterior(
    paired_signal: float,
    embedding_signal: float,
    scope_signal: float,
    mechanism_signal: float,
) -> float:
    """Compute P(transfers | signals) via naïve Bayes log-odds fusion.

    Each signal is a log-likelihood ratio from an independent evidence group.
    Combines via addition in log-odds space, then sigmoid to probability.
    Same math as ollama-queue stall detector and ARIA occupancy fusion.
    """
    log_odds = _PRIOR_LOG_ODDS + paired_signal + embedding_signal + scope_signal + mechanism_signal
    return 1.0 / (1.0 + math.exp(-log_odds))
```

**Step 4-5: Test, commit**

### Task 11: Compute Bayesian metrics from fusion posteriors

**Files:**
- Modify: `src/lessons_db/eval.py`
- Test: `tests/test_eval.py`

**Step 1: Write failing test**

```python
class TestComputeBayesianMetrics:
    def test_separation_metric(self):
        """Good fusion → high separation between same/diff posteriors."""
        scored = [
            {"variant": "A", "is_same_cluster": True, "posterior": 0.9},
            {"variant": "A", "is_same_cluster": True, "posterior": 0.85},
            {"variant": "A", "is_same_cluster": False, "posterior": 0.1},
            {"variant": "A", "is_same_cluster": False, "posterior": 0.15},
        ]
        metrics = compute_bayesian_metrics(scored)
        assert metrics["A"]["same_mean_posterior"] > 0.8
        assert metrics["A"]["diff_mean_posterior"] < 0.2
        assert metrics["A"]["auc"] > 0.9
```

**Step 2-3: Implement**

Per-variant metrics:
- `same_mean_posterior`: mean posterior for same-category pairs
- `diff_mean_posterior`: mean posterior for diff-category pairs
- `separation`: same_mean - diff_mean (higher = better)
- `auc`: AUC via Mann-Whitney U on posteriors (target > 0.85)
- `calibration_error`: |mean_posterior - actual_fraction| (Brier-like)

**Step 4-5: Test, commit**

---

## Phase 4: Reference Model Validation (lessons-db)

Borrow ARIA's reference model pattern: keep a frozen baseline to distinguish data drift from mechanism improvement.

### Task 12: Implement reference comparison

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `src/lessons_db/cli.py`
- Test: `tests/test_eval.py`

**Step 1: Write failing test**

```python
class TestReferenceComparison:
    def test_diagnoses_improvement(self):
        reference_metrics = {"A": {"auc": 0.70}}
        new_metrics = {"A": {"auc": 0.85}}
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics)
        assert diagnosis["A"]["status"] == "improved"

    def test_diagnoses_regression(self):
        reference_metrics = {"A": {"auc": 0.80}}
        new_metrics = {"A": {"auc": 0.60}}
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics)
        assert diagnosis["A"]["status"] == "regressed"

    def test_diagnoses_drift(self):
        """Both degrade → data drift, not mechanism failure."""
        reference_metrics = {"A": {"auc": 0.80}}
        new_metrics = {"A": {"auc": 0.60}}
        reference_rerun = {"A": {"auc": 0.62}}  # reference also degraded
        diagnosis = diagnose_vs_reference(reference_metrics, new_metrics,
                                          reference_rerun=reference_rerun)
        assert diagnosis["A"]["status"] == "data_drift"
```

**Step 2-3: Implement `diagnose_vs_reference()`, add `--reference` flag to CLI**

**Step 4-5: Test, commit**

---

## Phase 5: Simulation Validation (lessons-db)

Test whether principles actually prevent bugs — the ground-truth metric.

### Task 13: Build simulation prompt

**Files:**
- Modify: `src/lessons_db/eval.py`
- Test: `tests/test_eval.py`

**Step 1: Write failing test**

```python
class TestBuildSimulationPrompt:
    def test_with_principle_contains_rule(self):
        scenario = "Writing a database query handler that catches exceptions"
        principle = "Uncaught exception in cleanup path causes resource leak"
        prompt = build_simulation_prompt(scenario, principle)
        assert "CODING RULE" in prompt
        assert principle in prompt

    def test_without_principle_has_no_rule(self):
        scenario = "Writing a database query handler"
        prompt = build_simulation_prompt(scenario, principle=None)
        assert "CODING RULE" not in prompt
```

**Step 2-3: Implement**

```python
def build_simulation_prompt(scenario: str, principle: str | None = None) -> str:
    """Build a bug-catching simulation prompt.

    With principle: LLM has the rule and should catch the bug.
    Without principle (control): LLM has no rule.
    Lift = with_catch_rate - without_catch_rate.
    """
    rule_section = ""
    if principle:
        rule_section = (
            "\n## CODING RULE (always check for this)\n"
            f"{principle}\n\n"
        )
    return (
        "You are reviewing code for a potential bug.\n"
        f"{rule_section}"
        f"## SCENARIO\n{scenario}\n\n"
        "Does this code have a bug related to resource management, "
        "error handling, or structural correctness?\n\n"
        "Answer: BUG FOUND: [description] or NO BUG FOUND"
    )


def parse_simulation_result(response: str) -> bool:
    """Parse whether the LLM found a bug."""
    if not response:
        return False
    text = response.strip().upper()
    return "BUG FOUND" in text and "NO BUG FOUND" not in text
```

**Step 4-5: Test, commit**

### Task 14: Build simulation runner and lift metric

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `src/lessons_db/cli.py`
- Test: `tests/test_eval.py`

For each principle:
1. Extract the scenario from the source lesson's description
2. Run with-principle simulation (should catch bug)
3. Run without-principle simulation (control)
4. Lift = with_catch_rate - without_catch_rate across N trials

Add `eval-simulate` CLI command.

**Step 1-5: Implement, test, commit**

---

## Phase 6: Unified Report (lessons-db)

### Task 15: Render unified V2 report

**Files:**
- Modify: `src/lessons_db/eval.py` — `render_v2_report()`
- Test: `tests/test_eval.py`

Report sections:
1. **Tournament Results** — per-variant win rate (AUC) table
2. **Bayesian Fusion** — per-variant posterior separation, AUC, calibration
3. **Reference Comparison** — improvement/regression/drift diagnosis
4. **Simulation Lift** — per-variant lift scores
5. **Failure Analysis** — false negatives (same-category, low posterior) and false positives (diff-category, high posterior) with mechanism triplets
6. **Signal Diagnostics** — per-signal contribution breakdown (which signals disagree?)

**Step 1-5: Test, implement, commit**

---

## Phase 7: Mirror to ollama-queue (cross-repo)

### Task 16: Add binary + paired columns to eval_results schema

**Files:**
- Modify: `ollama_queue/db.py` — eval_results table
- Create: `scripts/migrate_eval_v2.py` — schema migration
- Test: `tests/test_db.py`

Add columns to `eval_results`:
```sql
ALTER TABLE eval_results ADD COLUMN score_paired_winner TEXT;  -- 'same', 'diff', 'neither'
ALTER TABLE eval_results ADD COLUMN score_mechanism_match INTEGER;  -- 0/1/NULL
ALTER TABLE eval_results ADD COLUMN score_embedding_sim REAL;
ALTER TABLE eval_results ADD COLUMN score_posterior REAL;
ALTER TABLE eval_results ADD COLUMN mechanism_trigger TEXT;
ALTER TABLE eval_results ADD COLUMN mechanism_target TEXT;
ALTER TABLE eval_results ADD COLUMN mechanism_fix TEXT;
```

Add to `eval_runs`:
```sql
ALTER TABLE eval_runs ADD COLUMN judge_mode TEXT DEFAULT 'rubric'
    CHECK (judge_mode IN ('rubric', 'binary', 'tournament', 'bayesian'));
ALTER TABLE eval_runs ADD COLUMN analysis_md TEXT;  -- already exists, verify
```

**Step 1-5: Migration script, test, commit**

### Task 17: Port Bayesian fusion to eval_engine.py

**Files:**
- Modify: `ollama_queue/eval_engine.py` — add paired/mechanism/Bayesian functions
- Test: `tests/test_eval_engine.py`

Port from lessons-db eval.py:
- `build_paired_judge_prompt()` / `parse_paired_judge()`
- `build_mechanism_extraction_prompt()` / `parse_mechanism_triplet()`
- Signal extractors + `compute_transfer_posterior()`
- `compute_tournament_metrics()` / `compute_bayesian_metrics()`

Add `judge_mode` parameter to `run_eval_judge_phase()` and `_judge_one_target()`.

**Step 1-5: Test, implement, commit**

### Task 18: Add judge_mode to eval API endpoints

**Files:**
- Modify: `ollama_queue/api.py`
- Test: `tests/test_api_eval_runs.py`

Update endpoints:
- `POST /api/eval/runs` — accept `judge_mode` in body (default: "bayesian")
- `GET /api/eval/runs/{id}` — return `judge_mode` in response
- `GET /api/eval/runs/{id}/progress` — show signal breakdown in progress data

**Step 1-5: Test, implement, commit**

---

## Phase 8: Dashboard UI/UX Changes (ollama-queue SPA)

### Task 19: Update RunTriggerPanel — judge mode selector

**Files:**
- Modify: `ollama_queue/dashboard/spa/src/components/eval/RunTriggerPanel.jsx`
- Modify: `ollama_queue/dashboard/spa/src/store.js`

**Current state:** RunTriggerPanel has readiness check, variant picker, and "Start Run" button.

**Change:** Add judge mode dropdown:
```jsx
// What it shows: Which judge strategy will be used for the eval run
// Decision it drives: User picks between rubric (legacy), binary, tournament, or bayesian (recommended)
<select value={judgeMode.value} onChange={e => judgeMode.value = e.target.value}>
  <option value="bayesian">Bayesian Fusion (recommended)</option>
  <option value="tournament">Paired Tournament</option>
  <option value="binary">Binary YES/NO</option>
  <option value="rubric">Rubric 1-5 (legacy)</option>
</select>
```

Add `judgeMode` signal to store.js.

**Step 1: Implement, Step 2: Build (`npm run build`), Step 3: Commit**

### Task 20: Update RunRow — show Bayesian metrics

**Files:**
- Modify: `ollama_queue/dashboard/spa/src/components/eval/RunRow.jsx`

**Current state:** RunRow L1 shows: variant, F1, recall, precision, actionability in a table row. L2 expands to show analysis panel.

**Change for Bayesian runs:** L1 shows AUC and posterior separation instead of F1:

```jsx
// What it shows: Summary metrics for a completed eval run
// Decision it drives: User sees at a glance whether this run discriminates well
{run.judge_mode === 'bayesian' ? (
  <>
    <td>{m.auc?.toFixed(2)}</td>
    <td>{m.same_mean_posterior?.toFixed(2)}</td>
    <td>{m.diff_mean_posterior?.toFixed(2)}</td>
    <td>{(m.same_mean_posterior - m.diff_mean_posterior)?.toFixed(2)}</td>
  </>
) : (
  <>
    <td>{m.recall?.toFixed(2)}</td>
    <td>{m.precision?.toFixed(2)}</td>
    <td>{m.f1?.toFixed(2)}</td>
    <td>{m.actionability?.toFixed(2)}</td>
  </>
)}
```

L2 additions for Bayesian:
- **Signal breakdown bar** — horizontal stacked bar showing contribution of each signal group (paired, embedding, scope, mechanism) to the posterior. Color-coded: green=positive, red=negative.
- **Posterior distribution** — histogram of same-category vs diff-category posteriors side by side. Good run = two well-separated humps.

**Step 1: Implement, Step 2: Build, Step 3: Commit**

### Task 21: Update F1LineChart → AUC trend chart

**Files:**
- Modify: `ollama_queue/dashboard/spa/src/components/eval/F1LineChart.jsx`

**Current state:** uPlot line chart showing F1 over time per variant.

**Change:** Detect judge_mode from run data. For Bayesian runs, plot AUC and posterior separation instead of F1. Add a mode toggle in the chart header.

```jsx
// What it shows: Eval quality trend over time — F1 (legacy) or AUC (bayesian)
// Decision it drives: User sees whether eval quality is improving or degrading across runs
const metric = runs.some(r => r.judge_mode === 'bayesian') ? 'auc' : 'f1';
```

**Step 1: Implement, Step 2: Build, Step 3: Commit**

### Task 22: Add SignalQualityPanel to EvalSettings

**Files:**
- Modify: `ollama_queue/dashboard/spa/src/components/eval/SignalQualityPanel.jsx` (exists but may need updates)
- Modify: `ollama_queue/dashboard/spa/src/views/EvalSettings.jsx`

**Add settings:**
- Judge mode default (dropdown: bayesian/tournament/binary/rubric)
- Group-by default (dropdown: category/cluster_seed)
- Pairs per principle (number input, default 4)
- Prior probability (number input, default 0.25)
- Signal weights panel showing the 4 log-LR values (read-only, informational)

```jsx
// What it shows: Configuration for the Bayesian fusion eval pipeline
// Decision it drives: User can tune the eval parameters without editing code
```

**Step 1: Implement, Step 2: Build, Step 3: Commit**

### Task 23: Add posterior distribution visualization to RunRow L2

**Files:**
- Modify: `ollama_queue/dashboard/spa/src/components/eval/RunRow.jsx`
- May need: new `PosteriorHistogram.jsx` component

When a Bayesian run is expanded (L2), show:
1. **Posterior histogram** — two overlapping distributions (same-category in green, diff-category in red). 10 bins from 0.0 to 1.0. Good run = green peaks right, red peaks left.
2. **Signal agreement matrix** — for each scored pair, show which signals agreed/disagreed. Heatmap with signals as columns, pairs as rows.
3. **Reference comparison badge** — if a reference run exists, show "improved/regressed/stable" badge with delta.

Use uPlot bars for the histogram (same library already in use).

**Step 1: Implement, Step 2: Build, Step 3: Commit**

### Task 24: Update auto-promote gates for Bayesian

**Files:**
- Modify: `ollama_queue/eval_engine.py` — `check_auto_promote()`
- Modify: `ollama_queue/dashboard/spa/src/components/eval/GeneralSettings.jsx`
- Test: `tests/test_eval_engine.py`

**Current:** Auto-promote checks `F1 >= threshold` and `F1 > production + min_improvement`.

**Change:** For Bayesian runs, check `AUC >= threshold` (default 0.85) and `posterior_separation > min_separation` (default 0.4). Add settings:
- `eval.auc_threshold` (float, default 0.85)
- `eval.min_posterior_separation` (float, default 0.4)

**Step 1-5: Test, implement, commit**

---

## Phase 9: Integration Testing

### Task 25: End-to-end pipeline test (lessons-db)

**Files:**
- Test: `tests/test_eval.py`

**Step 1: Write end-to-end test**

```python
class TestEvalV2EndToEnd:
    def test_full_bayesian_pipeline(self, db_path, tmp_path):
        """Generate → Tournament → Bayesian fusion → Report."""
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        conn.execute("UPDATE lessons SET category = 'error-handling' WHERE cluster_seed = 'A'")
        conn.execute("UPDATE lessons SET category = 'testing' WHERE cluster_seed = 'B'")
        conn.commit()

        # ... mock judge calls, run full pipeline, verify:
        # 1. Tournament results have win_rates
        # 2. Bayesian metrics have AUC > 0.5
        # 3. Report contains all sections
        # 4. Scored pairs have posterior values
```

**Step 2-5: Implement, run, commit**

### Task 26: Vertical integration test (ollama-queue)

**Files:**
- Test: `tests/test_eval_engine.py`

Test: `POST /api/eval/runs` with `judge_mode=bayesian` → generates → judges via tournament → computes Bayesian metrics → stores posteriors → renders report with signal breakdown.

**Step 1-5: Implement, run, commit**

### Task 27: Run real evaluation and compare

**Step 1:** Run V2 pipeline on full dataset with category grouping:
```bash
lessons-db meta eval-generate --variants A,D --per-cluster 4 --group-by category
lessons-db meta eval-tournament results.json --group-by category
```

**Step 2:** Compare V2 AUC against V1 best F1:
- V1 best: F1=0.72 (contrastive G, binary, cluster_seed)
- V2 target: AUC > 0.85 (tournament, category)

**Step 3:** If AUC < 0.85, diagnose with reference model:
- Is it data drift? (Re-run reference variant)
- Is it signal calibration? (Check per-signal contribution)
- Is it category quality? (Run audit script from Task 2)

**Step 4:** Commit results and update plan doc with findings.

---

## Batch Sequencing

| Batch | Tasks | Dependencies | Estimated pairs |
|-------|-------|-------------|-----------------|
| 1 | 1-2 | None | Ground truth fix |
| 2 | 3-6 | Batch 1 | Tournament judge |
| 3 | 7-8 | Batch 1 | Mechanism extraction |
| 4 | 9-11 | Batches 2-3 | Bayesian fusion |
| 5 | 12 | Batch 4 | Reference model |
| 6 | 13-14 | Batch 4 | Simulation |
| 7 | 15 | Batches 4-6 | Unified report |
| 8 | 16-18 | Batch 7 | ollama-queue backend |
| 9 | 19-24 | Batch 8 | Dashboard UI |
| 10 | 25-27 | Batch 9 | Integration + real eval |

**Critical path:** Batches 1 → 2 → 4 → 7 → 8 → 9. Batches 2 and 3 can run in parallel. Batches 5 and 6 can run in parallel with batch 7.
