# APO `eval-optimize` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an `eval-optimize` command that automatically generates improved instruction texts via 3 selectable strategies (feedback, OPRO, OPRO-API), evaluates them against the transfer-test pipeline, and validates winners on a holdout set.

**Architecture:** APO-generated configs live in a new `prompt_variants` DB table (never `variants.py`). `run_eval_generate` merges code-defined `VARIANT_CONFIGS` with DB-loaded APO variants at startup. Three optimizer strategies share the same downstream pipeline (generate → judge → eval_runs). A new `eval/optimize.py` module contains all optimizer logic.

**Tech Stack:** Python stdlib only. SQLite for storage. Existing `call_ollama`/`call_judge` for LLM calls. No new dependencies.

**Design doc:** `docs/plans/2026-03-11-apo-eval-optimize-design.md`

---

## Data Structures Reference

**prompt_variants table:**
```sql
CREATE TABLE IF NOT EXISTS prompt_variants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    variant_id      TEXT NOT NULL UNIQUE,
    instruction_text TEXT NOT NULL,
    config_json     TEXT NOT NULL,
    parent_variant  TEXT,
    strategy        TEXT NOT NULL,
    optimizer_model TEXT,
    hypothesis      TEXT,
    created_at      TEXT NOT NULL
);
```

**Merged variant config (runtime):**
```python
{
    "X01": {
        "prompt_id": "apo-generated",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
        "_instruction_text": "You are extracting...",  # from DB
        "_apo_generated": True,                         # marker
    }
}
```

**Optimizer candidate (parsed from LLM response):**
```python
{"instruction": "...", "hypothesis": "..."}
```

---

## Batch 1: Data Model — `prompt_variants` Table

### Task 1.1: Write failing tests for prompt_variants schema

**Files:**
- Create: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

```python
"""Tests for APO eval-optimize: prompt_variants table, optimizer strategies, variant registration."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lessons_db.db import init_db


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


class TestPromptVariantsTable:
    """prompt_variants table: schema, UNIQUE constraint, round-trip."""

    def test_table_exists(self, db_path):
        """init_db creates the prompt_variants table."""
        conn = init_db(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "prompt_variants" in tables

    def test_unique_variant_id(self, db_path):
        """Duplicate variant_id raises IntegrityError."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("X01", "test", "{}", "feedback", "2026-01-01"),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO prompt_variants
                   (variant_id, instruction_text, config_json, strategy, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("X01", "dupe", "{}", "feedback", "2026-01-01"),
            )

    def test_insert_and_query_round_trip(self, db_path):
        """Insert a prompt variant and read it back with all fields."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, parent_variant,
                strategy, optimizer_model, hypothesis, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("X01", "Extract principle", '{"model":"qwen3:14b"}',
             "D", "feedback", "qwen3:14b", "reduce false positives",
             "2026-03-11T00:00:00"),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM prompt_variants WHERE variant_id = ?", ("X01",)
        ).fetchone()
        assert row["variant_id"] == "X01"
        assert row["instruction_text"] == "Extract principle"
        assert row["parent_variant"] == "D"
        assert row["strategy"] == "feedback"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_optimize.py::TestPromptVariantsTable -v -x --timeout=30`
Expected: FAIL — `prompt_variants` table does not exist yet.

### Task 1.2: Create prompt_variants table in db.py

**Files:**
- Modify: `src/lessons_db/db.py:418-459` (inside the v11 eval tables executescript block)

**Step 3: Add the table to SCHEMA_SQL executescript**

Find the block starting with `# v11 eval contract tables` and append the new table inside the same `conn.executescript("""...""")` call, AFTER the `eval_runs` index:

```python
        CREATE TABLE IF NOT EXISTS prompt_variants (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            variant_id      TEXT NOT NULL UNIQUE,
            instruction_text TEXT NOT NULL,
            config_json     TEXT NOT NULL,
            parent_variant  TEXT,
            strategy        TEXT NOT NULL,
            optimizer_model TEXT,
            hypothesis      TEXT,
            created_at      TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_prompt_variants_variant
            ON prompt_variants(variant_id);
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_optimize.py::TestPromptVariantsTable -v -x --timeout=30`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add tests/test_eval_optimize.py src/lessons_db/db.py
git commit -m "feat(eval): add prompt_variants table for APO-generated instruction texts"
```

---

### Task 1.3: Write failing tests for load_all_variant_configs

**Files:**
- Modify: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

Append to `tests/test_eval_optimize.py`:

```python
from lessons_db.eval.optimize import load_all_variant_configs


class TestLoadAllVariantConfigs:
    """Merges VARIANT_CONFIGS (code) with prompt_variants (DB)."""

    def test_returns_hand_authored_variants(self, db_path):
        """With empty DB, returns only VARIANT_CONFIGS."""
        conn = init_db(db_path)
        merged = load_all_variant_configs(conn)
        assert "A" in merged
        assert "D" in merged
        assert "_apo_generated" not in merged["A"]

    def test_includes_db_variants(self, db_path):
        """DB-stored variants appear in merged result."""
        conn = init_db(db_path)
        config = {"model": "qwen3:14b", "temperature": 0.6, "num_ctx": 8192,
                  "chunked": False, "prompt_id": "apo-generated"}
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("X01", "Test instruction", json.dumps(config), "feedback", "2026-03-11"),
        )
        conn.commit()
        merged = load_all_variant_configs(conn)
        assert "X01" in merged
        assert merged["X01"]["_instruction_text"] == "Test instruction"
        assert merged["X01"]["_apo_generated"] is True
        assert merged["X01"]["model"] == "qwen3:14b"

    def test_db_does_not_override_hand_authored(self, db_path):
        """DB variant with same ID as hand-authored does NOT replace it."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("A", "Override attempt", '{"model":"fake"}', "feedback", "2026-03-11"),
        )
        conn.commit()
        merged = load_all_variant_configs(conn)
        # Hand-authored "A" should be preserved, DB row ignored
        assert merged["A"]["model"] == "deepseek-r1:8b"
        assert "_apo_generated" not in merged["A"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_optimize.py::TestLoadAllVariantConfigs -v -x --timeout=30`
Expected: FAIL — `ImportError: cannot import name 'load_all_variant_configs' from 'lessons_db.eval.optimize'`

### Task 1.4: Implement load_all_variant_configs

**Files:**
- Create: `src/lessons_db/eval/optimize.py`

**Step 3: Write minimal implementation**

```python
"""Automatic Prompt Optimization (APO) for the eval pipeline.

Three strategies:
  feedback   — analyze false positives, ask optimizer to fix instruction flaws
  opro       — OPRO meta-prompt (DeepMind ICLR 2024), requires 32B+ local model
  opro-api   — OPRO via API (Claude/GPT-4o-mini), most reliable
"""

from __future__ import annotations

import json as _json
import logging
import sqlite3
from typing import Any

from lessons_db.eval.variants import VARIANT_CONFIGS

_log = logging.getLogger(__name__)


def load_all_variant_configs(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Merge hand-authored VARIANT_CONFIGS with DB-stored APO variants.

    Hand-authored variants (A-H, M) always take precedence — a DB row
    with variant_id='A' is silently skipped to prevent config corruption.
    """
    merged: dict[str, dict[str, Any]] = dict(VARIANT_CONFIGS)
    rows = conn.execute(
        "SELECT variant_id, config_json, instruction_text FROM prompt_variants"
    ).fetchall()
    for row in rows:
        vid = row["variant_id"]
        if vid in VARIANT_CONFIGS:
            _log.warning("Skipping DB variant %s — hand-authored variant exists", vid)
            continue
        config = _json.loads(row["config_json"])
        config["_instruction_text"] = row["instruction_text"]
        config["_apo_generated"] = True
        merged[vid] = config
    return merged
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_optimize.py::TestLoadAllVariantConfigs -v -x --timeout=30`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add src/lessons_db/eval/optimize.py tests/test_eval_optimize.py
git commit -m "feat(eval): load_all_variant_configs merges code + DB variants"
```

---

## Batch 2: Instruction Text Extraction + APO Prompt Builder

### Task 2.1: Write failing tests for get_instruction_text

**Files:**
- Modify: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

```python
from lessons_db.eval.prompts import get_instruction_text


class TestGetInstructionText:
    """Extract instruction preamble from hand-authored prompt builders."""

    def test_variant_a_returns_fewshot_preamble(self):
        """Variant A instruction contains the few-shot examples."""
        text = get_instruction_text("A")
        assert "transferable principle" in text
        assert "Resources acquired in callbacks" in text
        assert "Lesson:" not in text  # must NOT contain lesson placeholder

    def test_variant_b_returns_causal_preamble(self):
        """Variant B instruction contains causal framing."""
        text = get_instruction_text("B")
        assert "causal statement" in text.lower() or "structural principle" in text.lower()
        assert "Lesson:" not in text

    def test_variant_f_returns_contrastive_preamble(self):
        """Variant F instruction references contrastive discrimination."""
        text = get_instruction_text("F")
        assert "SAME PATTERN" in text or "DISTINGUISH" in text or "contrastive" in text.lower()

    def test_unknown_variant_raises(self):
        """Unknown variant ID raises KeyError."""
        with pytest.raises(KeyError):
            get_instruction_text("UNKNOWN")

    def test_all_hand_authored_variants_covered(self):
        """Every variant in VARIANT_CONFIGS has an instruction text."""
        from lessons_db.eval.variants import VARIANT_CONFIGS as VC
        for vid in VC:
            text = get_instruction_text(vid)
            assert isinstance(text, str)
            assert len(text) > 20, f"Variant {vid} instruction too short"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_optimize.py::TestGetInstructionText -v -x --timeout=30`
Expected: FAIL — `ImportError: cannot import name 'get_instruction_text'`

### Task 2.2: Implement get_instruction_text in prompts.py

**Files:**
- Modify: `src/lessons_db/eval/prompts.py` (add function after the generation prompt section, before judge prompts)

**Step 3: Write implementation**

Add this function after the `_build_self_critique_prompt` function (around line 189) and before the `# Judge prompts` section:

```python
def get_instruction_text(variant_id: str) -> str:
    """Return the instruction preamble for a hand-authored variant.

    This is the portion BEFORE the lesson content injection — the part
    that APO can modify. Used by eval-optimize to seed optimizer history.

    Raises KeyError if variant_id is not a hand-authored variant.
    """
    _INSTRUCTION_TEXTS = {
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
            "- 'Integration boundaries require end-to-end value tracing, not per-layer unit tests.'"
        ),
        "B": (
            "Extract the structural principle from this coding lesson as a causal statement.\n\n"
            "Format: '<pattern> causes <consequence> when <condition>'\n\n"
            "Requirements:\n"
            "- One sentence, 10-25 words\n"
            "- No technology names, no fixes, no tool references\n"
            "- Name the structural pattern, not the specific bug"
        ),
        "C": (
            "These lessons all share the same structural failure pattern "
            "across different technologies.\n\n"
            "What is the ONE structural principle that explains ALL of these?\n\n"
            "Causal form: '<pattern> causes <consequence> when <condition>'\n"
            "One sentence, 10-25 words. No technology names."
        ),
        "F": (
            "Extract ONE structural principle that:\n"
            "- Is TRUE for ALL lessons in the SAME PATTERN group\n"
            "- Is FALSE or IRRELEVANT for the DIFFERENT PATTERNS group\n"
            "- Names the structural pattern, not the technology\n\n"
            "The principle must be specific enough to DISTINGUISH this failure type "
            "from the others listed above.\n\n"
            "Causal form: '<pattern> causes <consequence> when <condition>'\n"
            "One sentence, 10-25 words. No technology names."
        ),
        "H": (
            "Extract the abstract failure pattern from this lesson. "
            "Then distill it into a single transferable principle.\n\n"
            "Two-pass process:\n"
            "1. What is the abstract pattern? (not the specific bug)\n"
            "2. State the principle in causal form.\n\n"
            "Causal form: '<pattern> causes <consequence> when <condition>'\n"
            "One sentence, 10-25 words. No technology names."
        ),
        "M": (
            "Extract the SPECIFIC structural mechanism from this lesson.\n\n"
            "Format:\n"
            "TRIGGER: [what condition causes the bug, 3-10 words]\n"
            "TARGET: [what component/resource breaks, 3-10 words]\n"
            "FIX: [what structural change prevents it, 3-10 words]\n\n"
            "Be SPECIFIC — 'error handling' is too vague. "
            "'Uncaught exception in cleanup path' is specific."
        ),
    }
    # D shares B's instruction, E shares C's, G shares F's
    _INSTRUCTION_TEXTS["D"] = _INSTRUCTION_TEXTS["B"]
    _INSTRUCTION_TEXTS["E"] = _INSTRUCTION_TEXTS["C"]
    _INSTRUCTION_TEXTS["G"] = _INSTRUCTION_TEXTS["F"]

    if variant_id not in _INSTRUCTION_TEXTS:
        raise KeyError(f"No instruction text for variant {variant_id!r}")
    return _INSTRUCTION_TEXTS[variant_id]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_optimize.py::TestGetInstructionText -v -x --timeout=30`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/lessons_db/eval/prompts.py tests/test_eval_optimize.py
git commit -m "feat(eval): get_instruction_text extracts preambles from hand-authored variants"
```

---

### Task 2.3: Write failing tests for _build_apo_prompt

**Files:**
- Modify: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

```python
from lessons_db.eval.prompts import build_generation_prompt


class TestBuildApoPrompt:
    """APO-generated prompts use stored instruction text."""

    def test_override_replaces_default_prompt(self):
        """When prompt_overrides has the variant, uses stored instruction."""
        lesson = {"title": "Test Lesson", "one_liner": "Fix the bug",
                  "description": "Description here"}
        overrides = {"X01": "Custom instruction for extracting principles."}
        result = build_generation_prompt("X01", lesson, prompt_overrides=overrides)
        assert "Custom instruction for extracting principles." in result
        assert "Test Lesson" in result
        assert "Return ONLY" in result

    def test_no_override_falls_through(self):
        """Without override, variant A uses the existing fewshot prompt."""
        lesson = {"title": "Test", "one_liner": "Bug", "description": "Desc"}
        result = build_generation_prompt("A", lesson, prompt_overrides={})
        assert "transferable principle" in result

    def test_override_dict_none_falls_through(self):
        """prompt_overrides=None uses existing dispatch."""
        lesson = {"title": "Test", "one_liner": "Bug", "description": "Desc"}
        result = build_generation_prompt("A", lesson, prompt_overrides=None)
        assert "transferable principle" in result

    def test_apo_prompt_contains_lesson_context(self):
        """APO prompt injects title, one_liner, description."""
        lesson = {"title": "My Title", "one_liner": "My liner",
                  "description": "My desc"}
        overrides = {"X01": "Instruction preamble here."}
        result = build_generation_prompt("X01", lesson, prompt_overrides=overrides)
        assert "My Title" in result
        assert "My liner" in result
        assert "My desc" in result

    def test_apo_prompt_has_fixed_suffix(self):
        """APO prompt always ends with 'Return ONLY the principle statement.'"""
        lesson = {"title": "T", "one_liner": "O", "description": "D"}
        overrides = {"X01": "Any instruction."}
        result = build_generation_prompt("X01", lesson, prompt_overrides=overrides)
        assert "Return ONLY the principle statement" in result
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_optimize.py::TestBuildApoPrompt -v -x --timeout=30`
Expected: FAIL — `build_generation_prompt() got an unexpected keyword argument 'prompt_overrides'`

### Task 2.4: Implement prompt_overrides in build_generation_prompt

**Files:**
- Modify: `src/lessons_db/eval/prompts.py:25-50`

**Step 3: Write implementation**

Modify `build_generation_prompt` signature to add `prompt_overrides` parameter. Add `_build_apo_prompt` helper. The APO path must come BEFORE the existing `config = VARIANT_CONFIGS[variant_id]` line (since X-variants won't be in VARIANT_CONFIGS):

```python
def build_generation_prompt(
    variant_id: str,
    lesson: dict[str, Any],
    siblings: list[dict[str, Any]] | None = None,
    diff_cluster_items: list[dict[str, Any]] | None = None,
    prompt_overrides: dict[str, str] | None = None,
) -> str:
    """Build the principle-extraction prompt for a given variant.

    Variants A use few-shot examples. B/D use zero-shot causal framing.
    C/E use chunked (multiple sibling lessons from same cluster).
    F/G use contrastive (same-cluster + diff-cluster for specificity).
    When prompt_overrides contains variant_id, uses stored APO instruction text.
    """
    # APO-generated variants: use stored instruction text
    if prompt_overrides and variant_id in prompt_overrides:
        return _build_apo_prompt(prompt_overrides[variant_id], lesson)

    title = lesson.get("title") or ""
    one_liner = lesson.get("one_liner") or ""
    description = (lesson.get("description") or "")[:500]

    config = VARIANT_CONFIGS[variant_id]

    if config.get("contrastive") and siblings and diff_cluster_items:
        return _build_contrastive_prompt(lesson, siblings, diff_cluster_items)
    elif config["chunked"] and siblings:
        return _build_chunked_prompt(lesson, siblings)
    elif config["prompt_id"] == "baseline-fewshot":
        return _build_fewshot_prompt(title, one_liner, description)
    else:
        return _build_zero_shot_prompt(title, one_liner, description)


def _build_apo_prompt(instruction_text: str, lesson: dict[str, Any]) -> str:
    """Build prompt from APO-generated instruction text + lesson context."""
    context_parts = []
    title = lesson.get("title") or ""
    one_liner = lesson.get("one_liner") or ""
    description = (lesson.get("description") or "")[:500]
    if title:
        context_parts.append(f"Title: {title}")
    if one_liner:
        context_parts.append(f"One-liner: {one_liner}")
    if description:
        context_parts.append(f"Description: {description}")
    lesson_context = "\n".join(context_parts)

    return (
        f"{instruction_text}\n\n"
        f"Lesson:\n{lesson_context}\n\n"
        "Return ONLY the principle statement. One sentence. No quotes, no explanation."
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_optimize.py::TestBuildApoPrompt -v -x --timeout=30`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/lessons_db/eval/prompts.py tests/test_eval_optimize.py
git commit -m "feat(eval): build_generation_prompt supports prompt_overrides for APO variants"
```

---

## Batch 3: Optimizer Strategies + Candidate Parsing

### Task 3.1: Write failing tests for candidate parsing

**Files:**
- Modify: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

```python
from lessons_db.eval.optimize import parse_optimizer_candidates


class TestParseCandidates:
    """Parse optimizer LLM response into instruction candidates."""

    def test_valid_json_array(self):
        """Standard JSON array of candidates."""
        response = '[{"instruction": "Do X", "hypothesis": "because Y"}]'
        result = parse_optimizer_candidates(response)
        assert len(result) == 1
        assert result[0]["instruction"] == "Do X"
        assert result[0]["hypothesis"] == "because Y"

    def test_multiple_candidates(self):
        """Multiple candidates parsed correctly."""
        response = json.dumps([
            {"instruction": "First", "hypothesis": "H1"},
            {"instruction": "Second", "hypothesis": "H2"},
            {"instruction": "Third", "hypothesis": "H3"},
        ])
        result = parse_optimizer_candidates(response)
        assert len(result) == 3

    def test_json_embedded_in_text(self):
        """JSON array embedded in surrounding text/reasoning."""
        response = 'Here are my suggestions:\n[{"instruction": "X", "hypothesis": "Y"}]\nDone.'
        result = parse_optimizer_candidates(response)
        assert len(result) == 1
        assert result[0]["instruction"] == "X"

    def test_invalid_json_returns_empty(self):
        """Unparseable response returns empty list."""
        result = parse_optimizer_candidates("This is not JSON at all")
        assert result == []

    def test_missing_instruction_key_skipped(self):
        """Candidates without 'instruction' key are filtered out."""
        response = json.dumps([
            {"instruction": "Valid", "hypothesis": "H"},
            {"text": "Missing instruction key", "hypothesis": "H2"},
        ])
        result = parse_optimizer_candidates(response)
        assert len(result) == 1
        assert result[0]["instruction"] == "Valid"

    def test_think_tags_stripped(self):
        """<think> blocks stripped before parsing."""
        response = '<think>reasoning</think>[{"instruction": "X", "hypothesis": "Y"}]'
        result = parse_optimizer_candidates(response)
        assert len(result) == 1

    def test_empty_response_returns_empty(self):
        """None or empty string returns empty list."""
        assert parse_optimizer_candidates("") == []
        assert parse_optimizer_candidates(None) == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_optimize.py::TestParseCandidates -v -x --timeout=30`
Expected: FAIL — `ImportError: cannot import name 'parse_optimizer_candidates'`

### Task 3.2: Implement parse_optimizer_candidates

**Files:**
- Modify: `src/lessons_db/eval/optimize.py`

**Step 3: Write implementation**

Add to `optimize.py`:

```python
import re as _re


def parse_optimizer_candidates(response: str | None) -> list[dict[str, str]]:
    """Parse optimizer LLM response into instruction candidates.

    Expects a JSON array of objects with 'instruction' and 'hypothesis' keys.
    Strips <think> blocks, extracts first JSON array from surrounding text.
    Returns empty list on parse failure.
    """
    if not response:
        return []

    # Strip think blocks
    text = _re.sub(r"<think>.*?</think>", "", response, flags=_re.DOTALL | _re.IGNORECASE).strip()

    # Find JSON array in response
    match = _re.search(r"\[.*\]", text, flags=_re.DOTALL)
    if not match:
        _log.warning("No JSON array found in optimizer response")
        return []

    try:
        data = _json.loads(match.group())
    except _json.JSONDecodeError:
        _log.warning("Failed to parse optimizer response as JSON")
        return []

    if not isinstance(data, list):
        return []

    # Filter to valid candidates
    return [c for c in data if isinstance(c, dict) and "instruction" in c]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_optimize.py::TestParseCandidates -v -x --timeout=30`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add src/lessons_db/eval/optimize.py tests/test_eval_optimize.py
git commit -m "feat(eval): parse_optimizer_candidates extracts instruction candidates from LLM"
```

---

### Task 3.3: Write failing tests for optimizer prompt builders

**Files:**
- Modify: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

```python
from lessons_db.eval.optimize import build_feedback_prompt, build_opro_prompt


class TestFeedbackStrategy:
    """Feedback strategy: shows false positives to optimizer."""

    def test_includes_current_instruction(self):
        """Prompt contains the current best instruction text."""
        prompt = build_feedback_prompt(
            instruction_text="Current instruction here",
            f1=0.47,
            false_positives=[
                {"principle": "P1", "target_title": "T1",
                 "target_cluster_seed": "X", "cluster_seed": "Y"},
            ],
            n_candidates=3,
        )
        assert "Current instruction here" in prompt
        assert "0.47" in prompt

    def test_includes_false_positives(self):
        """Prompt lists false positive examples."""
        fps = [
            {"principle": "Too broad principle", "target_title": "Unrelated lesson",
             "target_cluster_seed": "cluster_B", "cluster_seed": "cluster_A"},
        ]
        prompt = build_feedback_prompt(
            instruction_text="Instr", f1=0.3, false_positives=fps, n_candidates=3,
        )
        assert "Too broad principle" in prompt
        assert "Unrelated lesson" in prompt

    def test_requests_n_candidates(self):
        """Prompt asks for the specified number of candidates."""
        prompt = build_feedback_prompt(
            instruction_text="I", f1=0.3, false_positives=[], n_candidates=5,
        )
        assert "5" in prompt

    def test_requests_json_format(self):
        """Prompt asks for JSON array output."""
        prompt = build_feedback_prompt(
            instruction_text="I", f1=0.3, false_positives=[], n_candidates=3,
        )
        assert "JSON" in prompt


class TestOproStrategy:
    """OPRO strategy: shows sorted prompt-score pairs."""

    def test_includes_prompts_sorted_ascending(self):
        """Past prompts appear sorted by F1 ascending (worst first)."""
        history = [
            {"instruction_text": "Best prompt", "f1": 0.47},
            {"instruction_text": "Worst prompt", "f1": 0.10},
            {"instruction_text": "Middle prompt", "f1": 0.28},
        ]
        prompt = build_opro_prompt(history=history, n_candidates=3)
        # Worst should appear before best in the prompt
        worst_pos = prompt.index("Worst prompt")
        best_pos = prompt.index("Best prompt")
        assert worst_pos < best_pos

    def test_includes_f1_scores(self):
        """Each prompt is labeled with its F1 score."""
        history = [{"instruction_text": "P1", "f1": 0.47}]
        prompt = build_opro_prompt(history=history, n_candidates=3)
        assert "0.47" in prompt

    def test_requests_json_format(self):
        """Prompt asks for JSON array output."""
        prompt = build_opro_prompt(
            history=[{"instruction_text": "P", "f1": 0.5}], n_candidates=3,
        )
        assert "JSON" in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_optimize.py::TestFeedbackStrategy tests/test_eval_optimize.py::TestOproStrategy -v -x --timeout=30`
Expected: FAIL — `ImportError`

### Task 3.4: Implement build_feedback_prompt and build_opro_prompt

**Files:**
- Modify: `src/lessons_db/eval/optimize.py`

**Step 3: Write implementation**

```python
def build_feedback_prompt(
    instruction_text: str,
    f1: float,
    false_positives: list[dict[str, str]],
    n_candidates: int = 3,
) -> str:
    """Build optimizer prompt for feedback strategy.

    Shows the current instruction + its worst false positives and asks
    the optimizer to fix the instruction to prevent them.
    """
    fp_lines = []
    for i, fp in enumerate(false_positives[:5], 1):
        fp_lines.append(
            f"  {i}. Principle: \"{fp.get('principle', '')}\"\n"
            f"     Wrongly matched: \"{fp.get('target_title', '')}\" "
            f"(cluster: {fp.get('target_cluster_seed', '?')})\n"
            f"     Source cluster: {fp.get('cluster_seed', '?')}"
        )
    fp_block = "\n".join(fp_lines) if fp_lines else "  (no false positives available)"

    return (
        "You are improving a principle-extraction prompt for a lessons-learned system.\n\n"
        f"Current instruction (F1={f1:.3f}):\n"
        "---\n"
        f"{instruction_text}\n"
        "---\n\n"
        "This instruction produces principles that are too broad. Here are the worst\n"
        "false positives — cases where a principle wrongly matched an unrelated lesson:\n\n"
        f"{fp_block}\n\n"
        "Analyze what about the current instruction causes these false matches.\n"
        f"Then generate {n_candidates} improved instructions that would prevent them.\n\n"
        "Each instruction must:\n"
        "- Be a complete replacement (not a diff/edit)\n"
        "- Be 50-200 words\n"
        "- Target precision improvement specifically\n\n"
        "Return JSON array:\n"
        '[{"instruction": "...", "hypothesis": "why this should reduce false positives"}]'
    )


def build_opro_prompt(
    history: list[dict[str, Any]],
    n_candidates: int = 3,
) -> str:
    """Build OPRO-style meta-prompt with past prompts sorted by score.

    Follows DeepMind OPRO (ICLR 2024): solution-score pairs sorted
    ascending so the best prompt appears last (recency bias favors it).
    """
    sorted_history = sorted(history, key=lambda h: h.get("f1", 0.0))
    prompt_lines = []
    for entry in sorted_history:
        f1 = entry.get("f1", 0.0)
        text = entry.get("instruction_text", "")
        prompt_lines.append(f'[Score: {f1:.3f}] "{text}"')
    history_block = "\n\n".join(prompt_lines)

    return (
        "You are optimizing a prompt instruction for a principle-extraction system.\n"
        "Below are past instructions sorted by F1 score (higher = better).\n\n"
        f"{history_block}\n\n"
        "The main failure mode: high recall (>0.9) but low precision (0.07-0.17).\n"
        "Principles match too broadly across unrelated bug categories.\n\n"
        f"Generate {n_candidates} new instructions that should score higher. Each must:\n"
        "- Be a complete instruction (not a diff/edit)\n"
        "- Target precision improvement specifically\n"
        "- Be 50-200 words\n\n"
        "Return JSON array:\n"
        '[{"instruction": "...", "hypothesis": "why this should score higher"}]'
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_optimize.py::TestFeedbackStrategy tests/test_eval_optimize.py::TestOproStrategy -v -x --timeout=30`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add src/lessons_db/eval/optimize.py tests/test_eval_optimize.py
git commit -m "feat(eval): feedback + OPRO optimizer prompt builders"
```

---

## Batch 4: Variant Registration + X-ID Generation

### Task 4.1: Write failing tests for register_apo_variant

**Files:**
- Modify: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

```python
from lessons_db.eval.optimize import register_apo_variant, next_x_id


class TestVariantRegistration:
    """Register APO-generated variants in the DB."""

    def test_next_x_id_starts_at_x01(self, db_path):
        """First X-ID is X01."""
        conn = init_db(db_path)
        assert next_x_id(conn) == "X01"

    def test_next_x_id_increments(self, db_path):
        """After X01 exists, next is X02."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("X01", "t", "{}", "feedback", "2026-01-01"),
        )
        conn.commit()
        assert next_x_id(conn) == "X02"

    def test_next_x_id_skips_gaps(self, db_path):
        """If X01 and X03 exist, next is X02 (fills gaps)."""
        conn = init_db(db_path)
        for xid in ["X01", "X03"]:
            conn.execute(
                """INSERT INTO prompt_variants
                   (variant_id, instruction_text, config_json, strategy, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (xid, "t", "{}", "feedback", "2026-01-01"),
            )
        conn.commit()
        assert next_x_id(conn) == "X02"

    def test_register_writes_to_db(self, db_path):
        """register_apo_variant writes a row and returns the variant_id."""
        conn = init_db(db_path)
        vid = register_apo_variant(
            conn,
            instruction_text="New instruction",
            parent_variant="D",
            strategy="feedback",
            optimizer_model="qwen3:14b",
            hypothesis="reduce false positives",
            config_overrides={"model": "qwen3:14b"},
        )
        assert vid.startswith("X")
        row = conn.execute(
            "SELECT * FROM prompt_variants WHERE variant_id = ?", (vid,)
        ).fetchone()
        assert row is not None
        assert row["instruction_text"] == "New instruction"
        assert row["parent_variant"] == "D"

    def test_register_uses_parent_config(self, db_path):
        """config_json inherits from parent variant config."""
        conn = init_db(db_path)
        vid = register_apo_variant(
            conn,
            instruction_text="Instr",
            parent_variant="D",
            strategy="feedback",
        )
        row = conn.execute(
            "SELECT config_json FROM prompt_variants WHERE variant_id = ?", (vid,)
        ).fetchone()
        config = json.loads(row["config_json"])
        assert config["model"] == "qwen3:14b"  # inherited from D
        assert config["prompt_id"] == "apo-generated"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_optimize.py::TestVariantRegistration -v -x --timeout=30`
Expected: FAIL — `ImportError`

### Task 4.2: Implement next_x_id and register_apo_variant

**Files:**
- Modify: `src/lessons_db/eval/optimize.py`

**Step 3: Write implementation**

```python
from datetime import UTC, datetime


def next_x_id(conn: sqlite3.Connection) -> str:
    """Generate the next available X-ID (X01, X02, ...) checking both code and DB."""
    existing_db = {
        r[0] for r in conn.execute("SELECT variant_id FROM prompt_variants").fetchall()
    }
    existing = existing_db | set(VARIANT_CONFIGS.keys())
    for i in range(1, 100):
        candidate = f"X{i:02d}"
        if candidate not in existing:
            return candidate
    raise RuntimeError("X-ID space exhausted (X01-X99 all taken)")


def register_apo_variant(
    conn: sqlite3.Connection,
    instruction_text: str,
    parent_variant: str,
    strategy: str,
    optimizer_model: str | None = None,
    hypothesis: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> str:
    """Register an APO-generated variant in the DB. Returns the new variant_id.

    Config is inherited from parent_variant (must be in VARIANT_CONFIGS or DB),
    with prompt_id set to 'apo-generated'. config_overrides can override
    specific fields (e.g. temperature).
    """
    # Build config from parent
    if parent_variant in VARIANT_CONFIGS:
        config = dict(VARIANT_CONFIGS[parent_variant])
    else:
        row = conn.execute(
            "SELECT config_json FROM prompt_variants WHERE variant_id = ?",
            (parent_variant,),
        ).fetchone()
        if row:
            config = _json.loads(row["config_json"])
        else:
            config = dict(VARIANT_CONFIGS.get("D", {}))  # fallback to D

    config["prompt_id"] = "apo-generated"
    if config_overrides:
        config.update(config_overrides)

    variant_id = next_x_id(conn)
    conn.execute(
        """INSERT INTO prompt_variants
           (variant_id, instruction_text, config_json, parent_variant,
            strategy, optimizer_model, hypothesis, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (variant_id, instruction_text, _json.dumps(config), parent_variant,
         strategy, optimizer_model, hypothesis, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return variant_id
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_optimize.py::TestVariantRegistration -v -x --timeout=30`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/lessons_db/eval/optimize.py tests/test_eval_optimize.py
git commit -m "feat(eval): variant registration — next_x_id + register_apo_variant"
```

---

## Batch 5: Generate/Judge Integration — Prompt Overrides Flow

### Task 5.1: Write failing test for prompt overrides in run_eval_generate

**Files:**
- Modify: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

```python
from lessons_db.eval.generate import run_eval_generate
from lessons_db.eval.optimize import register_apo_variant, load_all_variant_configs


class TestGenerateWithApoVariants:
    """run_eval_generate uses prompt_overrides for APO variants."""

    def test_apo_variant_generates_with_stored_instruction(self, db_path, tmp_path):
        """APO variant X01 uses instruction from prompt_variants, not VARIANT_CONFIGS."""
        from tests.test_eval import _seed_clusters

        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        # Register an APO variant
        vid = register_apo_variant(
            conn,
            instruction_text="CUSTOM APO INSTRUCTION for testing.",
            parent_variant="D",
            strategy="feedback",
        )

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "APO Principle."}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=[vid],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        # Verify the request body contained our custom instruction
        call_args = mock_url.call_args
        request_body = json.loads(call_args[0][0].data.decode("utf-8"))
        assert "CUSTOM APO INSTRUCTION" in request_body["prompt"]

        # Verify results were written
        data = json.loads(output_path.read_text())
        assert any(r["variant"] == vid for r in data["results"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_optimize.py::TestGenerateWithApoVariants -v -x --timeout=30`
Expected: FAIL — `run_eval_generate` doesn't know about prompt_overrides or APO variants.

### Task 5.2: Wire prompt overrides into run_eval_generate

**Files:**
- Modify: `src/lessons_db/eval/generate.py` (import `load_all_variant_configs`, load overrides, pass to `build_generation_prompt`)

**Step 3: Write implementation**

In `generate.py`, make these changes:

1. Add import: `from lessons_db.eval.optimize import load_all_variant_configs`

2. In `run_eval_generate`, after the source lesson selection and before the generation loop, load the merged configs and extract overrides:

```python
    # Load merged variant configs (code + DB APO variants)
    all_configs = load_all_variant_configs(conn)

    # Build prompt overrides for APO-generated variants
    prompt_overrides = {
        vid: cfg["_instruction_text"]
        for vid, cfg in all_configs.items()
        if cfg.get("_apo_generated")
    }
```

3. In the generation loop, pass `prompt_overrides` to `_generate_for_lesson` calls. In `_generate_for_lesson`, pass it through to `build_generation_prompt`.

4. In `_generate_for_lesson` (around line 151), add `prompt_overrides` parameter and pass it to `build_generation_prompt`:

```python
def _generate_for_lesson(
    ...
    prompt_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
```

And where `build_generation_prompt` is called, add `prompt_overrides=prompt_overrides`.

5. Where variant config is looked up (`VARIANT_CONFIGS[variant_id]`), use `all_configs` instead so APO variant configs are found.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_optimize.py::TestGenerateWithApoVariants -v -x --timeout=30`
Expected: PASS

Then run full eval test suite to check no regressions:

Run: `pytest tests/test_eval.py --timeout=120 -x -q -n 6`
Expected: All pass (231+)

**Step 5: Commit**

```bash
git add src/lessons_db/eval/generate.py tests/test_eval_optimize.py
git commit -m "feat(eval): run_eval_generate loads prompt_overrides for APO variants"
```

---

## Batch 6: CLI Command + __init__.py Exports

### Task 6.1: Add eval-optimize CLI command

**Files:**
- Modify: `src/lessons_db/cli.py` (add `eval-optimize` command after `eval-propose`)
- Modify: `src/lessons_db/eval/__init__.py` (re-export new symbols)

**Step 1: Add exports to `__init__.py`**

Add re-exports for all public symbols from `optimize.py`:

```python
# --- optimize (APO) ---
from lessons_db.eval.optimize import (
    build_feedback_prompt as build_feedback_prompt,
)
from lessons_db.eval.optimize import (
    build_opro_prompt as build_opro_prompt,
)
from lessons_db.eval.optimize import (
    load_all_variant_configs as load_all_variant_configs,
)
from lessons_db.eval.optimize import (
    next_x_id as next_x_id,
)
from lessons_db.eval.optimize import (
    parse_optimizer_candidates as parse_optimizer_candidates,
)
from lessons_db.eval.optimize import (
    register_apo_variant as register_apo_variant,
)
```

**Step 2: Add CLI command**

In `cli.py`, after the `eval-propose` command, add:

```python
@meta.command("eval-optimize")
@click.option("--strategy", type=click.Choice(["feedback", "opro", "opro-api"]),
              default="feedback",
              help="Optimization strategy: feedback (default, 14B+), opro (32B+), opro-api (API).")
@click.option("--candidates", type=int, default=3,
              help="Number of prompt candidates per iteration (default: 3).")
@click.option("--max-iterations", type=int, default=3,
              help="Maximum optimization iterations (default: 3).")
@click.option("--holdout", type=float, default=0.3,
              help="Holdout fraction for Goodhart prevention (default: 0.3).")
@click.option("--per-cluster", type=int, default=4,
              help="Lessons per cluster for eval (default: 4).")
@click.option("--parent", type=str, default=None,
              help="Variant to optimize from (default: auto-detect best).")
@click.option("--dry-run", is_flag=True, help="Show what would be generated without running eval.")
@click.option("--openai", is_flag=True, help="Use OpenAI API for optimizer (opro-api strategy).")
@click.option("--priority", type=int, default=None, help="Queue priority for eval jobs.")
@click.pass_context
def meta_eval_optimize(ctx, strategy, candidates, max_iterations, holdout,
                       per_cluster, parent, dry_run, openai, priority):
    """Automatic Prompt Optimization — generate improved instruction texts.

    Three strategies:

    \b
    feedback   (default) Analyze false positives and ask the optimizer to fix
               instruction flaws. Works with local 14B+ models.
    opro       OPRO pattern (DeepMind ICLR 2024): show top-3 prompts + F1
               scores, ask for better ones. Requires 32B+ local model.
    opro-api   Same as opro but uses an API model. Most reliable (~$0.01/iter).
               Requires --openai flag.
    """
    from lessons_db.eval.client import call_ollama
    from lessons_db.eval.generate import run_eval_generate
    from lessons_db.eval.judge import run_eval_judge
    from lessons_db.eval.optimize import (
        build_feedback_prompt,
        build_opro_prompt,
        load_all_variant_configs,
        parse_optimizer_candidates,
        register_apo_variant,
    )
    from lessons_db.eval.prompts import get_instruction_text
    from lessons_db.eval.runs import get_eval_history

    conn = ctx.obj["conn"]

    # Determine parent variant (best F1 from eval_runs, or fallback)
    if parent is None:
        history = get_eval_history(conn, limit=50)
        if history:
            best = max(history, key=lambda r: r.get("f1") or 0.0)
            parent = best["variant"]
            best_f1 = best["f1"]
        else:
            parent = "D"
            best_f1 = 0.0
        click.echo(f"Auto-selected parent variant: {parent} (F1={best_f1:.3f})")
    else:
        hist = get_eval_history(conn, variant=parent, limit=1)
        best_f1 = hist[0]["f1"] if hist else 0.0

    # Get parent instruction text
    try:
        parent_instruction = get_instruction_text(parent)
    except KeyError:
        # APO variant — load from DB
        row = conn.execute(
            "SELECT instruction_text FROM prompt_variants WHERE variant_id = ?",
            (parent,),
        ).fetchone()
        parent_instruction = row["instruction_text"] if row else "Extract the principle."

    click.echo(f"Strategy: {strategy} | Candidates: {candidates} | Max iterations: {max_iterations}")

    for iteration in range(1, max_iterations + 1):
        click.echo(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Build optimizer prompt based on strategy
        if strategy == "feedback":
            # Load most recent scored pairs for false positives
            from lessons_db.config import DATA_DIR
            eval_dir = DATA_DIR / "eval"
            scored_files = sorted(eval_dir.glob("*.scored.json"), reverse=True)
            false_positives = []
            if scored_files:
                import json as json_mod
                scored_data = json_mod.loads(scored_files[0].read_text())
                false_positives = [
                    p for p in scored_data
                    if not p.get("is_same_cluster") and p.get("scores", {}).get("matched")
                ][:5]
            optimizer_prompt = build_feedback_prompt(
                instruction_text=parent_instruction,
                f1=best_f1,
                false_positives=false_positives,
                n_candidates=candidates,
            )
        else:
            # OPRO or OPRO-API — build history
            hist = get_eval_history(conn, limit=10)
            opro_history = []
            for h in hist:
                vid = h["variant"]
                try:
                    instr = get_instruction_text(vid)
                except KeyError:
                    row = conn.execute(
                        "SELECT instruction_text FROM prompt_variants WHERE variant_id = ?",
                        (vid,),
                    ).fetchone()
                    instr = row["instruction_text"] if row else ""
                if instr:
                    opro_history.append({"instruction_text": instr, "f1": h.get("f1", 0.0)})
            # Deduplicate by instruction text, keep best F1
            seen = {}
            for entry in opro_history:
                key = entry["instruction_text"][:100]
                if key not in seen or entry["f1"] > seen[key]["f1"]:
                    seen[key] = entry
            opro_history = list(seen.values())[:5]
            optimizer_prompt = build_opro_prompt(history=opro_history, n_candidates=candidates)

        if dry_run:
            click.echo("--- Optimizer prompt ---")
            click.echo(optimizer_prompt[:500] + "..." if len(optimizer_prompt) > 500 else optimizer_prompt)
            click.echo("--- (dry run — not calling LLM) ---")
            return

        # Call optimizer LLM
        click.echo("Calling optimizer LLM...")
        from lessons_db.config import OLLAMA_QUEUE_URL
        optimizer_model = "qwen3:14b"  # default for feedback
        optimizer_response = call_ollama(
            queue_url=OLLAMA_QUEUE_URL,
            model=optimizer_model,
            prompt=optimizer_prompt,
            settings={"temperature": 0.8, "num_ctx": 8192},
            priority=priority,
            source="eval-optimize",
        )

        if not optimizer_response:
            click.echo("Optimizer returned no response. Skipping iteration.", err=True)
            continue

        # Parse candidates
        parsed = parse_optimizer_candidates(optimizer_response)
        if not parsed:
            click.echo("Failed to parse optimizer candidates. Skipping iteration.", err=True)
            continue

        click.echo(f"Parsed {len(parsed)} candidates")

        # Register variants
        new_variant_ids = []
        for candidate in parsed[:candidates]:
            vid = register_apo_variant(
                conn,
                instruction_text=candidate["instruction"],
                parent_variant=parent,
                strategy=strategy,
                optimizer_model=optimizer_model,
                hypothesis=candidate.get("hypothesis"),
            )
            new_variant_ids.append(vid)
            click.echo(f"  Registered {vid}: {candidate.get('hypothesis', '')[:60]}")

        # Run eval cycle
        from datetime import UTC
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        output_path = eval_dir / f"apo-{ts}.json"
        report_path = eval_dir / f"apo-{ts}-report.md"

        click.echo(f"Running eval-generate for {new_variant_ids}...")
        run_eval_generate(
            conn=conn,
            queue_url=OLLAMA_QUEUE_URL,
            variants=new_variant_ids,
            per_cluster=per_cluster,
            output_path=output_path,
            resume=False,
            priority=priority,
            holdout_fraction=holdout,
        )

        click.echo("Running eval-judge...")
        _scored, metrics = run_eval_judge(
            results_path=output_path,
            conn=conn,
            report_path=report_path,
            priority=priority,
        )

        # Report results
        for vid, m in metrics.items():
            f1 = m.get("f1", 0.0)
            delta = f1 - best_f1
            symbol = "↑" if delta > 0 else "↓" if delta < 0 else "="
            click.echo(f"  {vid}: F1={f1:.3f} ({symbol}{abs(delta):.3f} vs parent)")
            if f1 > best_f1:
                best_f1 = f1
                parent = vid
                # Load the new best instruction for next iteration
                row = conn.execute(
                    "SELECT instruction_text FROM prompt_variants WHERE variant_id = ?",
                    (vid,),
                ).fetchone()
                if row:
                    parent_instruction = row["instruction_text"]
                click.echo(f"  New best: {vid} (F1={f1:.3f})")

    click.echo(f"\nDone. Best variant: {parent} (F1={best_f1:.3f})")
```

**Step 3: Run full test suite**

Run: `pytest tests/test_eval.py tests/test_eval_optimize.py --timeout=120 -x -q -n 6`
Expected: All pass

**Step 4: Commit**

```bash
git add src/lessons_db/cli.py src/lessons_db/eval/__init__.py
git commit -m "feat(eval): eval-optimize CLI command with 3 strategies"
```

---

## Batch 7: End-to-End Test + program.md Update

### Task 7.1: Write end-to-end test for eval-optimize

**Files:**
- Modify: `tests/test_eval_optimize.py`

**Step 1: Write the failing test**

```python
class TestEvalOptimizeEndToEnd:
    """Full loop: load history → optimize → register → eval → record."""

    def test_feedback_loop_registers_and_evaluates(self, db_path, tmp_path):
        """Feedback strategy: produces candidates, registers them, runs eval."""
        from tests.test_eval import _seed_clusters

        conn = init_db(db_path)
        _seed_clusters(conn)

        # Mock the optimizer LLM to return candidates
        optimizer_response = json.dumps([
            {"instruction": "Improved instruction 1", "hypothesis": "H1"},
        ])
        # Mock the generator LLM
        generator_response = json.dumps({"response": "Generated principle."})

        mock_resp = MagicMock()
        mock_resp.read.return_value = generator_response.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        call_count = {"n": 0}
        def mock_urlopen(req, *args, **kwargs):
            call_count["n"] += 1
            resp = MagicMock()
            # First call is optimizer, rest are generator/judge
            if call_count["n"] == 1:
                resp.read.return_value = optimizer_response.encode("utf-8")
            else:
                resp.read.return_value = generator_response.encode("utf-8")
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            from click.testing import CliRunner
            from lessons_db.cli import cli

            runner = CliRunner()
            result = runner.invoke(cli, [
                "--db", str(db_path),
                "meta", "eval-optimize",
                "--strategy", "feedback",
                "--candidates", "1",
                "--max-iterations", "1",
                "--per-cluster", "1",
            ])

        # Should have registered at least one X-variant
        rows = conn.execute("SELECT variant_id FROM prompt_variants").fetchall()
        assert len(rows) >= 1
        assert rows[0]["variant_id"].startswith("X")
```

**Step 2: Run test**

Run: `pytest tests/test_eval_optimize.py::TestEvalOptimizeEndToEnd -v -x --timeout=120`

**Step 3: Fix any issues until it passes**

**Step 4: Commit**

```bash
git add tests/test_eval_optimize.py
git commit -m "test(eval): end-to-end test for eval-optimize feedback loop"
```

---

### Task 7.2: Update program.md with eval-optimize documentation

**Files:**
- Modify: `program.md`

**Step 1: Add eval-optimize to the "Unexplored design space" section**

Add after item 8:

```markdown
9. **APO: `eval-optimize`** — automatic prompt optimization. Three strategies:
   - `feedback` (default): shows false positives to optimizer, asks for instruction fixes
   - `opro`: OPRO meta-prompt with score-sorted history (requires 32B+ model)
   - `opro-api`: same as opro via API (Claude/GPT-4o-mini)
   Usage: `lessons-db meta eval-optimize --strategy feedback --candidates 3`
```

**Step 2: Commit**

```bash
git add program.md
git commit -m "docs: add eval-optimize to program.md"
```

---

### Task 7.3: Run full test suite and verify

**Step 1: Run all tests**

Run: `pytest tests/ --timeout=120 -x -q -n 6`
Expected: All pass (240+ tests)

**Step 2: Run linter**

Run: `cd ~/Documents/projects/lessons-db && source .venv/bin/activate && make lint`
Expected: No errors

**Step 3: Final commit if any lint fixes needed**

```bash
git add -u
git commit -m "chore: lint fixes for eval-optimize"
```

---

## Summary

| Batch | Tasks | What it delivers |
|-------|-------|-----------------|
| 1 | 1.1–1.4 | `prompt_variants` table + `load_all_variant_configs` |
| 2 | 2.1–2.4 | `get_instruction_text` + `_build_apo_prompt` + `prompt_overrides` |
| 3 | 3.1–3.4 | `parse_optimizer_candidates` + `build_feedback_prompt` + `build_opro_prompt` |
| 4 | 4.1–4.2 | `next_x_id` + `register_apo_variant` |
| 5 | 5.1–5.2 | `run_eval_generate` wired with prompt overrides |
| 6 | 6.1 | `eval-optimize` CLI command + `__init__.py` exports |
| 7 | 7.1–7.3 | End-to-end test + program.md + full verification |
