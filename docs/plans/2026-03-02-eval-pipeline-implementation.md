# Transfer-Test Evaluation Pipeline — Implementation Plan

**Status:** Complete (PRs #11, #12, #13 merged to main)

**Goal:** Build two CLI subcommands (`meta eval-generate` and `meta eval-judge`) that systematically evaluate prompt × model × settings variants for principle extraction by measuring transfer quality.

**Architecture:** New `src/lessons_db/eval.py` module contains all logic (variant configs, test set selection, generation, judging, metrics). CLI in `cli.py` provides thin Click wrappers. Judge scoring uses Ollama by default (deepseek-r1:8b) or OpenAI GPT-4o-mini via `--openai` flag.

**Tech Stack:** Python 3.14, Click CLI, SQLite (lessons.db), urllib (HTTP), JSON

**Design Doc:** `docs/plans/2026-03-02-prompt-evaluation-pipeline-design.md`

**Post-implementation enhancements (PRs #12, #13):**
- Default judge upgraded from qwen2.5:7b to deepseek-r1:8b (stronger discrimination)
- Retry with exponential backoff on 502/503 transient errors
- Variants sorted by model to minimize Ollama model swaps
- `--priority` flag threads queue priority via ollama-queue proxy (depends on ollama-queue PR #10)

---

## Batch 1: Foundation (Tasks 1–3)

### Task 1: Add EVAL_DIR to config.py

**Files:**
- Modify: `src/lessons_db/config.py:15` (after RULES_DIR)
- Test: `tests/test_eval.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_eval.py
"""Tests for the transfer-test evaluation pipeline."""

from lessons_db.config import EVAL_DIR


class TestEvalConfig:
    """EVAL_DIR must be defined and point inside DATA_DIR."""

    def test_eval_dir_exists_in_config(self):
        from lessons_db.config import DATA_DIR
        assert EVAL_DIR == DATA_DIR / "eval"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval.py::TestEvalConfig::test_eval_dir_exists_in_config -v`
Expected: FAIL with `ImportError: cannot import name 'EVAL_DIR'`

**Step 3: Add EVAL_DIR to config.py**

In `src/lessons_db/config.py`, after line 15 (`RULES_DIR = DATA_DIR / "rules"`), add:

```python
EVAL_DIR = DATA_DIR / "eval"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval.py::TestEvalConfig -v`
Expected: PASS

---

### Task 2: Create eval.py with variant configs and test set selection

**Files:**
- Create: `src/lessons_db/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write failing tests for variant configs**

Append to `tests/test_eval.py`:

```python
from lessons_db.eval import VARIANT_CONFIGS, select_source_lessons


class TestVariantConfigs:
    """VARIANT_CONFIGS must define all 5 variants per design doc."""

    def test_has_five_variants(self):
        assert set(VARIANT_CONFIGS.keys()) == {"A", "B", "C", "D", "E"}

    def test_each_variant_has_required_fields(self):
        required = {"prompt_id", "model", "temperature", "num_ctx", "chunked"}
        for vid, cfg in VARIANT_CONFIGS.items():
            assert required.issubset(cfg.keys()), f"Variant {vid} missing: {required - cfg.keys()}"

    def test_baseline_variant_a(self):
        a = VARIANT_CONFIGS["A"]
        assert a["prompt_id"] == "baseline-fewshot"
        assert a["model"] == "deepseek-r1:8b-0528-qwen3-q4_K_M"
        assert a["temperature"] == 0.7
        assert a["num_ctx"] == 4096
        assert a["chunked"] is False

    def test_chunked_variants_c_e(self):
        assert VARIANT_CONFIGS["C"]["chunked"] is True
        assert VARIANT_CONFIGS["E"]["chunked"] is True
        assert VARIANT_CONFIGS["A"]["chunked"] is False
        assert VARIANT_CONFIGS["B"]["chunked"] is False
        assert VARIANT_CONFIGS["D"]["chunked"] is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py::TestVariantConfigs -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lessons_db.eval'`

**Step 3: Create eval.py with variant configs**

Create `src/lessons_db/eval.py`:

```python
"""Transfer-test evaluation pipeline for principle extraction quality."""

import json as _json
import logging
import re as _re
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Variant definitions (design doc §Variant Design)
# ---------------------------------------------------------------------------

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "A": {
        "prompt_id": "baseline-fewshot",
        "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
        "temperature": 0.7,
        "num_ctx": 4096,
        "chunked": False,
    },
    "B": {
        "prompt_id": "zero-shot-causal",
        "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
    },
    "C": {
        "prompt_id": "zero-shot-chunked",
        "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": True,
    },
    "D": {
        "prompt_id": "zero-shot-causal",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
    },
    "E": {
        "prompt_id": "zero-shot-chunked",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": True,
    },
}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestVariantConfigs -v`
Expected: PASS

**Step 5: Write failing tests for select_source_lessons**

Append to `tests/test_eval.py`:

```python
from lessons_db.db import init_db, insert_lesson


def _seed_clusters(conn, clusters=None):
    """Seed a test DB with lessons in known clusters.

    Default: 5 clusters (A-E) with varying sizes and categories.
    Returns dict of cluster_seed -> list of lesson IDs.
    """
    if clusters is None:
        clusters = {
            "A": [
                ("Silent failure 0", "integration"),
                ("Silent failure 1", "testing"),
                ("Silent failure 2", "monitoring"),
                ("Silent failure 3", "error-handling"),
                ("Silent failure 4", "caching"),
            ],
            "B": [
                ("Boundary issue 0", "integration"),
                ("Boundary issue 1", "data-model"),
                ("Boundary issue 2", "testing"),
                ("Boundary issue 3", "deployment"),
                ("Boundary issue 4", "integration"),
                ("Boundary issue 5", "data-model"),
            ],
            "D": [
                ("Spec drift 0", "integration"),
                ("Spec drift 1", "specification-drift"),
                ("Spec drift 2", "specification-drift"),
                ("Spec drift 3", "integration"),
            ],
            "E": [
                ("Context issue 0", "context-retrieval"),
                ("Context issue 1", "context-retrieval"),
                ("Context issue 2", "context-retrieval"),
                ("Context issue 3", "context-retrieval"),
            ],
            "F": [
                ("Plan issue 0", "planning-control-flow"),
                ("Plan issue 1", "planning-control-flow"),
                ("Plan issue 2", "data-model"),
                ("Plan issue 3", "frontend"),
            ],
        }
    ids_by_cluster: dict[str, list[int]] = {}
    for seed, lessons in clusters.items():
        ids = []
        for title, cat in lessons:
            lid = insert_lesson(conn, {
                "title": title,
                "one_liner": f"One-liner for {title}",
                "description": f"Description for {title}",
                "cluster_seed": seed,
                "category": cat,
            })
            ids.append(lid)
        ids_by_cluster[seed] = ids
    return ids_by_cluster


class TestSelectSourceLessons:
    """select_source_lessons picks N lessons per cluster, maximizing category diversity."""

    def test_returns_correct_count_per_cluster(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        sources = select_source_lessons(conn, per_cluster=2)
        # Should have lessons from all 5 clusters
        clusters_seen = {s["cluster_seed"] for s in sources}
        assert len(clusters_seen) == 5
        # 2 per cluster × 5 clusters = 10 total
        assert len(sources) == 10
        conn.close()

    def test_respects_per_cluster_limit(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        sources = select_source_lessons(conn, per_cluster=4)
        # Cluster E has exactly 4, so all 4 used. Others capped at 4.
        per_cluster = {}
        for s in sources:
            per_cluster.setdefault(s["cluster_seed"], []).append(s)
        for seed, lessons in per_cluster.items():
            assert len(lessons) <= 4
        conn.close()

    def test_maximizes_category_diversity(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        sources = select_source_lessons(conn, per_cluster=3)
        # Cluster B has 3 distinct categories in first 4 entries
        # Selecting 3 should pick from 3 different categories
        b_sources = [s for s in sources if s["cluster_seed"] == "B"]
        b_cats = {s["category"] for s in b_sources}
        assert len(b_cats) >= 2  # at least 2 distinct categories
        conn.close()

    def test_empty_db_returns_empty(self, db_path):
        conn = init_db(db_path)
        sources = select_source_lessons(conn, per_cluster=4)
        assert sources == []
        conn.close()

    def test_only_selects_single_loop_lessons(self, db_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        # Add a double-loop meta-lesson to cluster A
        insert_lesson(conn, {
            "title": "Meta: cluster A",
            "one_liner": "Meta lesson",
            "cluster_seed": "A",
            "loop_level": "double",
        })
        sources = select_source_lessons(conn, per_cluster=10)
        for s in sources:
            assert s.get("loop_level", "single") != "double"
        conn.close()
```

**Step 6: Implement select_source_lessons**

Add to `src/lessons_db/eval.py`:

```python
def select_source_lessons(
    conn: sqlite3.Connection,
    per_cluster: int = 4,
) -> list[dict[str, Any]]:
    """Select source lessons for evaluation, maximizing category diversity per cluster.

    Returns a flat list of lesson dicts with keys: id, title, one_liner,
    description, cluster_seed, category.
    """
    # Find all clusters with at least 3 single-loop lessons
    cluster_rows = conn.execute(
        "SELECT cluster_seed, COUNT(*) as cnt "
        "FROM lessons "
        "WHERE cluster_seed IS NOT NULL AND cluster_seed != '' "
        "  AND (loop_level IS NULL OR loop_level = 'single') "
        "GROUP BY cluster_seed "
        "HAVING cnt >= 3 "
        "ORDER BY cluster_seed",
    ).fetchall()

    sources: list[dict[str, Any]] = []
    for crow in cluster_rows:
        seed = crow["cluster_seed"]
        # Get all single-loop lessons in this cluster
        lessons = conn.execute(
            "SELECT id, title, one_liner, description, cluster_seed, category "
            "FROM lessons "
            "WHERE cluster_seed = ? AND (loop_level IS NULL OR loop_level = 'single') "
            "ORDER BY id",
            (seed,),
        ).fetchall()
        lessons = [dict(r) for r in lessons]

        if len(lessons) <= per_cluster:
            sources.extend(lessons)
        else:
            # Greedy category-diversity selection
            selected: list[dict[str, Any]] = []
            seen_cats: set[str] = set()
            remaining = list(lessons)

            # First pass: pick one from each unique category
            for lesson in list(remaining):
                cat = lesson.get("category") or "uncategorized"
                if cat not in seen_cats and len(selected) < per_cluster:
                    selected.append(lesson)
                    seen_cats.add(cat)
                    remaining.remove(lesson)

            # Second pass: fill remaining slots
            while len(selected) < per_cluster and remaining:
                selected.append(remaining.pop(0))

            sources.extend(selected)

    return sources
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestSelectSourceLessons -v`
Expected: PASS

---

### Task 3: Implement select_transfer_targets

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write failing tests**

Append to `tests/test_eval.py`:

```python
from lessons_db.eval import select_transfer_targets


class TestSelectTransferTargets:
    """select_transfer_targets returns same-cluster (TP) and different-cluster (TN) targets."""

    def test_returns_correct_structure(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        targets = select_transfer_targets(conn, source_id, "A")
        assert "same_cluster" in targets
        assert "diff_cluster" in targets
        conn.close()

    def test_same_cluster_count(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        targets = select_transfer_targets(conn, source_id, "A", count_same=2, count_diff=2)
        assert len(targets["same_cluster"]) == 2
        conn.close()

    def test_diff_cluster_count(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        targets = select_transfer_targets(conn, source_id, "A", count_same=2, count_diff=2)
        assert len(targets["diff_cluster"]) == 2
        conn.close()

    def test_same_cluster_excludes_source(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        targets = select_transfer_targets(conn, source_id, "A")
        same_ids = {t["id"] for t in targets["same_cluster"]}
        assert source_id not in same_ids
        conn.close()

    def test_diff_cluster_from_other_clusters(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        source_id = ids["A"][0]
        targets = select_transfer_targets(conn, source_id, "A")
        for t in targets["diff_cluster"]:
            assert t["cluster_seed"] != "A"
        conn.close()

    def test_prefers_different_category_in_same_cluster(self, db_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        # Source is A[0] with category "integration"
        source_id = ids["A"][0]
        targets = select_transfer_targets(conn, source_id, "A", count_same=2)
        # Should prefer lessons from different categories when available
        source_cat = "integration"
        diff_cat_count = sum(1 for t in targets["same_cluster"] if t["category"] != source_cat)
        assert diff_cat_count >= 1  # at least 1 from a different category
        conn.close()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py::TestSelectTransferTargets -v`
Expected: FAIL with `ImportError`

**Step 3: Implement select_transfer_targets**

Add to `src/lessons_db/eval.py`:

```python
def select_transfer_targets(
    conn: sqlite3.Connection,
    source_id: int,
    cluster_seed: str,
    count_same: int = 2,
    count_diff: int = 2,
) -> dict[str, list[dict[str, Any]]]:
    """Select transfer test targets for a source lesson.

    Returns {"same_cluster": [...], "diff_cluster": [...]} where each entry
    is a lesson dict with id, title, one_liner, description, cluster_seed, category.

    same_cluster: other lessons from the same cluster (true positives).
    diff_cluster: lessons from different clusters (true negatives).
    Prefers different categories within same_cluster for harder transfer tests.
    """
    # Get source lesson's category
    source_row = conn.execute(
        "SELECT category FROM lessons WHERE id = ?", (source_id,)
    ).fetchone()
    source_cat = source_row["category"] if source_row else None

    # Same-cluster targets: exclude source, prefer different categories
    same_all = conn.execute(
        "SELECT id, title, one_liner, description, cluster_seed, category "
        "FROM lessons "
        "WHERE cluster_seed = ? AND id != ? "
        "  AND (loop_level IS NULL OR loop_level = 'single') "
        "ORDER BY id",
        (cluster_seed, source_id),
    ).fetchall()
    same_all = [dict(r) for r in same_all]

    # Sort: different category first (harder transfer test)
    same_all.sort(key=lambda x: (x.get("category") == source_cat, x["id"]))
    same_cluster = same_all[:count_same]

    # Different-cluster targets: pick from other clusters
    diff_all = conn.execute(
        "SELECT id, title, one_liner, description, cluster_seed, category "
        "FROM lessons "
        "WHERE cluster_seed IS NOT NULL AND cluster_seed != '' "
        "  AND cluster_seed != ? "
        "  AND (loop_level IS NULL OR loop_level = 'single') "
        "ORDER BY RANDOM() "
        "LIMIT ?",
        (cluster_seed, count_diff),
    ).fetchall()
    diff_cluster = [dict(r) for r in diff_all]

    return {"same_cluster": same_cluster, "diff_cluster": diff_cluster}
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestSelectTransferTargets -v`
Expected: PASS

**Step 5: Commit Batch 1**

```bash
git add src/lessons_db/config.py src/lessons_db/eval.py tests/test_eval.py
git commit -m "feat(eval): add EVAL_DIR config, variant definitions, test set selection"
```

---

## Batch 2: Generation Pipeline (Tasks 4–6)

### Task 4: Implement generation prompt construction

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write failing tests**

Append to `tests/test_eval.py`:

```python
from lessons_db.eval import build_generation_prompt


class TestBuildGenerationPrompt:
    """build_generation_prompt produces variant-specific prompts."""

    def _lesson(self, **overrides):
        base = {
            "id": 1,
            "title": "Test lesson",
            "one_liner": "Test one-liner",
            "description": "Test description of the lesson",
            "cluster_seed": "A",
            "category": "testing",
        }
        base.update(overrides)
        return base

    def test_variant_a_includes_examples(self):
        prompt = build_generation_prompt("A", self._lesson())
        assert "Examples of good principles" in prompt
        assert "Test lesson" in prompt

    def test_variant_b_is_zero_shot(self):
        prompt = build_generation_prompt("B", self._lesson())
        assert "Examples of good principles" not in prompt
        assert "causal" in prompt.lower() or "causes" in prompt.lower()

    def test_variant_c_requires_siblings(self):
        siblings = [self._lesson(id=2, title="Sibling 1"), self._lesson(id=3, title="Sibling 2")]
        prompt = build_generation_prompt("C", self._lesson(), siblings=siblings)
        assert "Sibling 1" in prompt
        assert "Sibling 2" in prompt

    def test_variant_c_without_siblings_falls_back(self):
        prompt = build_generation_prompt("C", self._lesson(), siblings=None)
        # Should fall back to variant B's zero-shot prompt
        assert "Test lesson" in prompt

    def test_variant_d_same_prompt_as_b(self):
        prompt_b = build_generation_prompt("B", self._lesson())
        prompt_d = build_generation_prompt("D", self._lesson())
        # Same prompt template, different model (handled at call level)
        assert prompt_b == prompt_d

    def test_variant_e_same_prompt_as_c(self):
        siblings = [self._lesson(id=2)]
        prompt_c = build_generation_prompt("C", self._lesson(), siblings=siblings)
        prompt_e = build_generation_prompt("E", self._lesson(), siblings=siblings)
        assert prompt_c == prompt_e

    def test_truncates_long_descriptions(self):
        long_desc = "x" * 1000
        prompt = build_generation_prompt("A", self._lesson(description=long_desc))
        assert "x" * 501 not in prompt  # description truncated at 500
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py::TestBuildGenerationPrompt -v`
Expected: FAIL with `ImportError`

**Step 3: Implement build_generation_prompt**

Add to `src/lessons_db/eval.py`:

```python
def build_generation_prompt(
    variant_id: str,
    lesson: dict[str, Any],
    siblings: list[dict[str, Any]] | None = None,
) -> str:
    """Build the principle-extraction prompt for a given variant.

    Variants A use few-shot examples. B/D use zero-shot causal framing.
    C/E use chunked (multiple sibling lessons from same cluster).
    """
    title = lesson.get("title") or ""
    one_liner = lesson.get("one_liner") or ""
    description = (lesson.get("description") or "")[:500]

    config = VARIANT_CONFIGS[variant_id]

    if config["chunked"] and siblings:
        return _build_chunked_prompt(lesson, siblings)
    elif config["prompt_id"] == "baseline-fewshot":
        return _build_fewshot_prompt(title, one_liner, description)
    else:
        return _build_zero_shot_prompt(title, one_liner, description)


def _build_fewshot_prompt(title: str, one_liner: str, description: str) -> str:
    """Variant A: current production prompt with few-shot examples."""
    context_parts = []
    if title:
        context_parts.append(f"Title: {title}")
    if one_liner:
        context_parts.append(f"One-liner: {one_liner}")
    if description:
        context_parts.append(f"Description: {description}")
    lesson_context = "\n".join(context_parts)

    return (
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
        f"Lesson:\n{lesson_context}\n\n"
        "Return ONLY the principle statement. One sentence. No quotes, no explanation."
    )


def _build_zero_shot_prompt(title: str, one_liner: str, description: str) -> str:
    """Variants B/D: zero-shot causal framing."""
    context_parts = []
    if title:
        context_parts.append(f"Title: {title}")
    if one_liner:
        context_parts.append(f"One-liner: {one_liner}")
    if description:
        context_parts.append(f"Description: {description}")
    lesson_context = "\n".join(context_parts)

    return (
        "Extract the structural principle from this coding lesson as a causal statement.\n\n"
        "Format: '<pattern> causes <consequence> when <condition>'\n\n"
        "Requirements:\n"
        "- One sentence, 10-25 words\n"
        "- No technology names, no fixes, no tool references\n"
        "- Name the structural pattern, not the specific bug\n\n"
        f"Lesson:\n{lesson_context}\n\n"
        "Return ONLY the causal principle. No quotes, no explanation."
    )


def _build_chunked_prompt(
    primary: dict[str, Any],
    siblings: list[dict[str, Any]],
) -> str:
    """Variants C/E: show multiple sibling lessons to aid abstraction."""
    lines = []
    all_lessons = [primary] + siblings
    for i, lesson in enumerate(all_lessons, 1):
        t = lesson.get("title") or ""
        o = lesson.get("one_liner") or ""
        lines.append(f"{i}. Title: {t}\n   One-liner: {o}")
    lesson_block = "\n".join(lines)

    return (
        "These lessons all share the same structural failure pattern "
        "across different technologies:\n\n"
        f"{lesson_block}\n\n"
        "What is the ONE structural principle that explains ALL of these?\n\n"
        "Causal form: '<pattern> causes <consequence> when <condition>'\n"
        "One sentence, 10-25 words. No technology names."
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestBuildGenerationPrompt -v`
Expected: PASS

---

### Task 5: Implement Ollama HTTP caller and eval-generate orchestrator

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write failing tests for call_ollama**

Append to `tests/test_eval.py`:

```python
from unittest.mock import MagicMock, patch

from lessons_db.eval import call_ollama


class TestCallOllama:
    """call_ollama sends HTTP request and returns cleaned response."""

    def test_returns_cleaned_response(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": "  Test principle.  "}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = call_ollama("http://localhost:7683", "test-model", "prompt", {})
        assert result == "Test principle."

    def test_strips_think_tags(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": "<think>reasoning here</think>Clean principle."}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = call_ollama("http://localhost:7683", "test-model", "prompt", {})
        assert result == "Clean principle."

    def test_returns_none_on_http_error(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            "http://localhost", 502, "Bad Gateway", {}, None
        )):
            result = call_ollama("http://localhost:7683", "test-model", "prompt", {})
        assert result is None

    def test_sends_correct_payload(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "ok"}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            call_ollama(
                "http://localhost:7683", "my-model", "my prompt",
                {"temperature": 0.6, "num_ctx": 8192},
            )
        req = mock_url.call_args[0][0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "my-model"
        assert payload["prompt"] == "my prompt"
        assert payload["options"]["temperature"] == 0.6
        assert payload["options"]["num_ctx"] == 8192
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py::TestCallOllama -v`
Expected: FAIL with `ImportError`

**Step 3: Implement call_ollama**

Add to `src/lessons_db/eval.py`:

```python
def call_ollama(
    queue_url: str,
    model: str,
    prompt: str,
    settings: dict[str, Any],
    timeout: int = 300,
) -> str | None:
    """Call Ollama via queue and return cleaned response text.

    Returns None on any error (network, timeout, parse).
    Strips <think>...</think> reasoning blocks from response.
    """
    options = {}
    if "temperature" in settings:
        options["temperature"] = settings["temperature"]
    if "num_ctx" in settings:
        options["num_ctx"] = settings["num_ctx"]

    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        **({"options": options} if options else {}),
    }).encode("utf-8")

    try:
        req = urllib.request.Request(  # noqa: S310
            f"{queue_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            result = _json.loads(resp.read().decode("utf-8"))
        text = result.get("response", "").strip()
        # Strip reasoning blocks
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
        text = text.strip("\"'").strip()
        return text if text else None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, _json.JSONDecodeError) as exc:
        _log.warning("call_ollama error: %s", exc)
        return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestCallOllama -v`
Expected: PASS

**Step 5: Write failing tests for run_eval_generate**

Append to `tests/test_eval.py`:

```python
import json
from pathlib import Path

from lessons_db.eval import run_eval_generate


class TestRunEvalGenerate:
    """run_eval_generate orchestrates variant × lesson generation."""

    def test_generates_results_json(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": "Test principle from model."}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["A"],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "meta" in data
        assert "results" in data
        assert len(data["results"]) > 0
        assert data["results"][0]["variant"] == "A"
        assert data["results"][0]["principle"] is not None
        conn.close()

    def test_resume_skips_existing(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        # Pre-seed a partial results file
        existing = {
            "meta": {"variants": ["A"], "per_cluster": 1},
            "results": [{
                "variant": "A",
                "lesson_id": ids["A"][0],
                "principle": "Already done",
                "error": None,
            }],
        }
        output_path.write_text(json.dumps(existing))

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": "New principle."}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["A"],
                per_cluster=1,
                output_path=output_path,
                resume=True,
            )

        data = json.loads(output_path.read_text())
        # The pre-existing result should still be there
        a_results = [r for r in data["results"] if r["variant"] == "A" and r["lesson_id"] == ids["A"][0]]
        assert len(a_results) == 1
        assert a_results[0]["principle"] == "Already done"
        conn.close()

    def test_records_errors(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            "http://localhost", 502, "Bad Gateway", {}, None
        )):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["A"],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        data = json.loads(output_path.read_text())
        error_results = [r for r in data["results"] if r["error"] is not None]
        assert len(error_results) > 0
        conn.close()

    def test_includes_metadata(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": "Principle."}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=["A", "B"],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        data = json.loads(output_path.read_text())
        assert data["meta"]["variants"] == ["A", "B"]
        assert "generated_at" in data["meta"]
        assert "source_lessons" in data["meta"]
        conn.close()
```

**Step 6: Implement run_eval_generate**

Add to `src/lessons_db/eval.py`:

```python
def run_eval_generate(
    conn: sqlite3.Connection,
    queue_url: str,
    variants: list[str],
    per_cluster: int,
    output_path: Path,
    resume: bool,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run eval-generate: produce principles for all (variant, lesson) pairs.

    Saves results incrementally to output_path as JSON.
    Returns the full results dict.
    """
    # Load existing results if resuming
    existing_results: list[dict[str, Any]] = []
    completed_pairs: set[tuple[str, int]] = set()
    if resume and output_path.exists():
        existing = _json.loads(output_path.read_text())
        existing_results = existing.get("results", [])
        for r in existing_results:
            if r.get("error") is None and r.get("principle"):
                completed_pairs.add((r["variant"], r["lesson_id"]))

    # Select source lessons
    sources = select_source_lessons(conn, per_cluster=per_cluster)
    source_ids = [s["id"] for s in sources]

    # Build results structure
    results: list[dict[str, Any]] = list(existing_results)

    for variant_id in variants:
        config = VARIANT_CONFIGS[variant_id]
        model = config["model"]
        settings = {"temperature": config["temperature"], "num_ctx": config["num_ctx"]}

        # Get siblings for chunked variants
        siblings_by_cluster: dict[str, list[dict[str, Any]]] = {}
        if config["chunked"]:
            for src in sources:
                seed = src["cluster_seed"]
                if seed not in siblings_by_cluster:
                    sibs = conn.execute(
                        "SELECT id, title, one_liner, description, cluster_seed, category "
                        "FROM lessons "
                        "WHERE cluster_seed = ? AND (loop_level IS NULL OR loop_level = 'single') "
                        "ORDER BY id",
                        (seed,),
                    ).fetchall()
                    siblings_by_cluster[seed] = [dict(r) for r in sibs]

        for lesson in sources:
            lesson_id = lesson["id"]
            if (variant_id, lesson_id) in completed_pairs:
                continue

            # Build prompt
            siblings = None
            if config["chunked"]:
                all_sibs = siblings_by_cluster.get(lesson["cluster_seed"], [])
                siblings = [s for s in all_sibs if s["id"] != lesson_id][:3]

            prompt = build_generation_prompt(variant_id, lesson, siblings=siblings)

            # Generate
            t0 = time.monotonic()
            principle = call_ollama(queue_url, model, prompt, settings)
            elapsed = round(time.monotonic() - t0, 1)

            entry = {
                "variant": variant_id,
                "lesson_id": lesson_id,
                "lesson_title": lesson.get("title", ""),
                "cluster_seed": lesson.get("cluster_seed", ""),
                "principle": principle,
                "model": model,
                "prompt_id": config["prompt_id"],
                "settings": settings,
                "generation_time_s": elapsed,
                "error": None if principle else "generation_failed",
            }
            results.append(entry)

            if progress_callback:
                progress_callback(variant_id, lesson_id, principle is not None)

    # Build output
    output = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "variants": variants,
            "per_cluster": per_cluster,
            "source_lessons": source_ids,
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json.dumps(output, indent=2))
    return output
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestRunEvalGenerate -v`
Expected: PASS

---

### Task 6: Add eval-generate CLI command

**Files:**
- Modify: `src/lessons_db/cli.py` (after line 2761, before `find_meta_lesson_clusters`)
- Create: `tests/test_eval_cli.py`

**Step 1: Write failing CLI tests**

Create `tests/test_eval_cli.py`:

```python
"""Tests for eval-generate and eval-judge CLI commands."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db, insert_lesson


def _seed_eval_db(conn):
    """Create a minimal test DB with 2 clusters for eval testing."""
    for i, cat in enumerate(["integration", "testing", "monitoring"]):
        insert_lesson(conn, {
            "title": f"Cluster A lesson {i}",
            "one_liner": f"A one-liner {i}",
            "description": f"A description {i}",
            "cluster_seed": "A",
            "category": cat,
        })
    for i, cat in enumerate(["data-model", "deployment", "integration"]):
        insert_lesson(conn, {
            "title": f"Cluster B lesson {i}",
            "one_liner": f"B one-liner {i}",
            "description": f"B description {i}",
            "cluster_seed": "B",
            "category": cat,
        })


class TestEvalGenerateHelp:
    """eval-generate --help must work."""

    def test_help_exits_zero(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "eval-generate", "--help"])
        assert result.exit_code == 0
        assert "eval" in result.output.lower()

    def test_lists_in_meta_help(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "--help"])
        assert "eval-generate" in result.output


class TestEvalGenerateCommand:
    """eval-generate runs generation and produces results JSON."""

    def test_produces_results_file(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_eval_db(conn)
        conn.close()

        output_file = tmp_path / "results.json"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": "Test principle generated."}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(main, [
                "--db", str(db_path),
                "meta", "eval-generate",
                "--variants", "A",
                "--per-cluster", "1",
                "--output", str(output_file),
            ])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert len(data["results"]) > 0

    def test_no_clusters_reports_empty(self, db_path, tmp_path):
        init_db(db_path)
        output_file = tmp_path / "results.json"

        runner = CliRunner()
        result = runner.invoke(main, [
            "--db", str(db_path),
            "meta", "eval-generate",
            "--variants", "A",
            "--output", str(output_file),
        ])

        assert result.exit_code == 0
        assert "No source lessons" in result.output

    def test_resume_flag(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_eval_db(conn)
        conn.close()

        output_file = tmp_path / "results.json"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": "Principle."}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(main, [
                "--db", str(db_path),
                "meta", "eval-generate",
                "--variants", "A",
                "--per-cluster", "1",
                "--output", str(output_file),
                "--resume",
            ])
        assert result.exit_code == 0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_cli.py::TestEvalGenerateHelp -v`
Expected: FAIL (command doesn't exist yet)

**Step 3: Add eval-generate CLI command to cli.py**

In `src/lessons_db/cli.py`, after the `meta_generate_meta_lessons` function (around line 2761), add:

```python
@meta.command("eval-generate")
@click.option("--variants", default="A,B,C,D,E", help="Comma-separated variant IDs (default: A,B,C,D,E).")
@click.option("--per-cluster", default=4, type=int, help="Source lessons per cluster (default: 4).")
@click.option("--output", type=click.Path(), default=None, help="Output JSON path (default: auto-timestamped in EVAL_DIR).")
@click.option("--resume", is_flag=True, help="Skip already-completed (variant, lesson_id) pairs.")
@click.pass_context
def meta_eval_generate(ctx, variants, per_cluster, output, resume):
    """Generate principles across prompt variants for transfer-test evaluation.

    Runs each variant (prompt × model × settings) across a fixed set of source
    lessons. Results saved to a JSON file for later judging with eval-judge.
    """
    from lessons_db.config import EVAL_DIR, OLLAMA_QUEUE_URL
    from lessons_db.eval import VARIANT_CONFIGS, run_eval_generate, select_source_lessons

    conn = ctx.obj["conn"]
    variant_list = [v.strip() for v in variants.split(",")]

    # Validate variant IDs
    for v in variant_list:
        if v not in VARIANT_CONFIGS:
            click.echo(f"Unknown variant '{v}'. Valid: {', '.join(VARIANT_CONFIGS.keys())}", err=True)
            ctx.exit(1)
            return

    # Check source lessons exist
    sources = select_source_lessons(conn, per_cluster=per_cluster)
    if not sources:
        click.echo("No source lessons found (need clusters with >= 3 lessons).")
        return

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        output_path = EVAL_DIR / f"results-{ts}.json"

    click.echo(f"Eval-generate: {len(variant_list)} variants × {len(sources)} lessons")
    click.echo(f"Output: {output_path}")

    # Warm models (deduplicate)
    models_to_warm = {VARIANT_CONFIGS[v]["model"] for v in variant_list}
    for model_name in models_to_warm:
        _warm_model(OLLAMA_QUEUE_URL, model_name)

    def _progress(variant_id, lesson_id, success):
        status = "OK" if success else "FAIL"
        click.echo(f"  [{variant_id}] lesson #{lesson_id}: {status}")

    result = run_eval_generate(
        conn=conn,
        queue_url=OLLAMA_QUEUE_URL,
        variants=variant_list,
        per_cluster=per_cluster,
        output_path=output_path,
        resume=resume,
        progress_callback=_progress,
    )

    total = len(result["results"])
    errors = sum(1 for r in result["results"] if r.get("error"))
    click.echo(f"\nDone. Total: {total}  Errors: {errors}")
    click.echo(f"Results: {output_path}")
```

Also add `from datetime import UTC, datetime` to the imports at the top of `cli.py` if not already present — check the existing imports first; the file already imports `datetime` indirectly. Add it near the existing imports.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_cli.py -v`
Expected: PASS

**Step 5: Commit Batch 2**

```bash
git add src/lessons_db/eval.py src/lessons_db/cli.py tests/test_eval.py tests/test_eval_cli.py
git commit -m "feat(eval): add eval-generate command with variant prompt construction"
```

---

## Batch 3: Judge Pipeline (Tasks 7–10)

### Task 7: Implement judge prompt and score parsing

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write failing tests**

Append to `tests/test_eval.py`:

```python
from lessons_db.eval import build_judge_prompt, parse_judge_scores


class TestBuildJudgePrompt:
    """build_judge_prompt creates the rubric-based scoring prompt."""

    def test_contains_principle(self):
        prompt = build_judge_prompt("Silent fallbacks mask failures.", {
            "title": "Git apply silent failure",
            "one_liner": "|| true discards errors",
            "description": "The git apply command...",
        })
        assert "Silent fallbacks mask failures." in prompt

    def test_contains_target_lesson(self):
        prompt = build_judge_prompt("Test principle.", {
            "title": "My Target Lesson",
            "one_liner": "Target one-liner",
            "description": "Target description",
        })
        assert "My Target Lesson" in prompt

    def test_contains_scoring_criteria(self):
        prompt = build_judge_prompt("Principle.", {"title": "T", "one_liner": "O", "description": "D"})
        assert "transfer" in prompt.lower()
        assert "precision" in prompt.lower()
        assert "actionability" in prompt.lower()

    def test_requests_json_output(self):
        prompt = build_judge_prompt("Principle.", {"title": "T", "one_liner": "O", "description": "D"})
        assert "JSON" in prompt or "json" in prompt


class TestParseJudgeScores:
    """parse_judge_scores extracts 3 integer scores from judge response."""

    def test_parses_valid_json(self):
        response = '{"transfer": 4, "precision": 3, "actionability": 5}'
        scores = parse_judge_scores(response)
        assert scores == {"transfer": 4, "precision": 3, "actionability": 5}

    def test_parses_json_with_surrounding_text(self):
        response = 'Here are the scores:\n{"transfer": 2, "precision": 1, "actionability": 3}\nDone.'
        scores = parse_judge_scores(response)
        assert scores == {"transfer": 2, "precision": 1, "actionability": 3}

    def test_returns_none_on_invalid(self):
        scores = parse_judge_scores("I cannot score this.")
        assert scores is None

    def test_returns_none_on_missing_keys(self):
        scores = parse_judge_scores('{"transfer": 4}')
        assert scores is None

    def test_clamps_scores_to_1_5(self):
        response = '{"transfer": 0, "precision": 7, "actionability": 3}'
        scores = parse_judge_scores(response)
        assert scores["transfer"] == 1
        assert scores["precision"] == 5
        assert scores["actionability"] == 3
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py::TestBuildJudgePrompt tests/test_eval.py::TestParseJudgeScores -v`
Expected: FAIL with `ImportError`

**Step 3: Implement build_judge_prompt and parse_judge_scores**

Add to `src/lessons_db/eval.py`:

```python
def build_judge_prompt(principle: str, target: dict[str, Any]) -> str:
    """Build the rubric-based scoring prompt for a (principle, target) pair.

    The judge evaluates whether the principle helps recognize the same
    structural pattern in the target lesson.
    """
    title = target.get("title") or ""
    one_liner = target.get("one_liner") or ""
    description = (target.get("description") or "")[:300]

    return (
        "You are evaluating whether a structural principle helps recognize "
        "a pattern in a target lesson.\n\n"
        f'PRINCIPLE: "{principle}"\n\n'
        f"TARGET LESSON:\n"
        f"Title: {title}\n"
        f"One-liner: {one_liner}\n"
        f"Description: {description}\n\n"
        "Score this (principle, target) pair on three criteria, each 1-5:\n\n"
        "1. **Transfer Recognition** — does the principle help identify the "
        "structural pattern in the target?\n"
        "   1=No connection  3=Vague connection  5=Clear structural match\n\n"
        "2. **Precision** — would this principle false-positive on unrelated lessons?\n"
        "   1=Would match anything  3=Somewhat specific  5=Only matches structurally similar\n\n"
        "3. **Actionability** — could an LLM use this principle to prevent this bug class?\n"
        "   1=Too abstract to act on  3=Useful with context  5=Immediately actionable\n\n"
        'Return ONLY a JSON object: {"transfer": N, "precision": N, "actionability": N}\n'
        "No explanation."
    )


def parse_judge_scores(response: str) -> dict[str, int] | None:
    """Extract transfer/precision/actionability scores from judge response.

    Returns dict with keys transfer, precision, actionability (ints 1-5),
    or None if parsing fails.
    """
    # Try to find JSON in response
    match = _re.search(r"\{[^}]+\}", response)
    if not match:
        return None

    try:
        data = _json.loads(match.group())
    except _json.JSONDecodeError:
        return None

    required = {"transfer", "precision", "actionability"}
    if not required.issubset(data.keys()):
        return None

    # Clamp to 1-5 range
    scores = {}
    for key in required:
        val = int(data[key])
        scores[key] = max(1, min(5, val))
    return scores
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestBuildJudgePrompt tests/test_eval.py::TestParseJudgeScores -v`
Expected: PASS

---

### Task 8: Implement judge model callers (Ollama + OpenAI)

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write failing tests**

Append to `tests/test_eval.py`:

```python
from lessons_db.eval import call_judge


class TestCallJudge:
    """call_judge routes to Ollama or OpenAI based on backend parameter."""

    def test_ollama_backend(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": '{"transfer": 4, "precision": 3, "actionability": 5}'}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = call_judge(
                prompt="test prompt",
                backend="ollama",
                ollama_url="http://localhost:7683",
                ollama_model="qwen2.5:7b",
            )
        assert result is not None
        assert "transfer" in result

    def test_openai_backend(self):
        mock_resp = MagicMock()
        openai_response = {
            "choices": [{
                "message": {
                    "content": '{"transfer": 3, "precision": 2, "actionability": 4}'
                }
            }]
        }
        mock_resp.read.return_value = json.dumps(openai_response).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = call_judge(
                prompt="test prompt",
                backend="openai",
                openai_api_key="test-key",
                openai_model="gpt-4o-mini",
            )
        assert result is not None

    def test_returns_none_on_error(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail")):
            result = call_judge(
                prompt="test prompt",
                backend="ollama",
                ollama_url="http://localhost:7683",
                ollama_model="qwen2.5:7b",
            )
        assert result is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py::TestCallJudge -v`
Expected: FAIL with `ImportError`

**Step 3: Implement call_judge**

Add to `src/lessons_db/eval.py`:

```python
def call_judge(
    prompt: str,
    backend: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
) -> str | None:
    """Call the judge model and return raw response text.

    Routes to Ollama or OpenAI based on backend parameter.
    Returns None on any error.
    """
    if backend == "openai":
        return _call_openai(openai_api_key, openai_model, prompt)
    return call_ollama(ollama_url, ollama_model, prompt, {})


def _call_openai(api_key: str, model: str, prompt: str) -> str | None:
    """Call OpenAI Chat Completions API. Returns response text or None."""
    payload = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.1,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(  # noqa: S310
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            result = _json.loads(resp.read().decode("utf-8"))
        return result["choices"][0]["message"]["content"].strip()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            _json.JSONDecodeError, KeyError, IndexError) as exc:
        _log.warning("_call_openai error: %s", exc)
        return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestCallJudge -v`
Expected: PASS

---

### Task 9: Implement metrics computation and report rendering

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write failing tests**

Append to `tests/test_eval.py`:

```python
from lessons_db.eval import compute_metrics, render_report


class TestComputeMetrics:
    """compute_metrics calculates F1, recall, precision per variant."""

    def _make_scored_pair(self, variant, is_same_cluster, transfer=3, precision=3, actionability=3):
        return {
            "variant": variant,
            "is_same_cluster": is_same_cluster,
            "scores": {"transfer": transfer, "precision": precision, "actionability": actionability},
        }

    def test_perfect_scores(self):
        pairs = [
            # Same-cluster (TP): high transfer = good recall
            self._make_scored_pair("A", True, transfer=5, precision=5, actionability=5),
            self._make_scored_pair("A", True, transfer=5, precision=5, actionability=5),
            # Diff-cluster (TN): low transfer = good precision
            self._make_scored_pair("A", False, transfer=1, precision=5, actionability=5),
            self._make_scored_pair("A", False, transfer=1, precision=5, actionability=5),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["recall"] == 1.0
        assert metrics["A"]["precision"] == 1.0
        assert metrics["A"]["f1"] == 1.0

    def test_zero_recall(self):
        pairs = [
            self._make_scored_pair("A", True, transfer=1),
            self._make_scored_pair("A", True, transfer=2),
            self._make_scored_pair("A", False, transfer=1),
            self._make_scored_pair("A", False, transfer=1),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["recall"] == 0.0

    def test_zero_precision(self):
        pairs = [
            self._make_scored_pair("A", True, transfer=5),
            self._make_scored_pair("A", True, transfer=5),
            self._make_scored_pair("A", False, transfer=5),
            self._make_scored_pair("A", False, transfer=4),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["precision"] == 0.0

    def test_multiple_variants(self):
        pairs = [
            self._make_scored_pair("A", True, transfer=5),
            self._make_scored_pair("A", False, transfer=1),
            self._make_scored_pair("B", True, transfer=3),
            self._make_scored_pair("B", False, transfer=3),
        ]
        metrics = compute_metrics(pairs)
        assert "A" in metrics
        assert "B" in metrics

    def test_mean_actionability(self):
        pairs = [
            self._make_scored_pair("A", True, actionability=4),
            self._make_scored_pair("A", True, actionability=2),
            self._make_scored_pair("A", False, actionability=3),
            self._make_scored_pair("A", False, actionability=5),
        ]
        metrics = compute_metrics(pairs)
        assert metrics["A"]["mean_actionability"] == 3.5


class TestRenderReport:
    """render_report produces valid markdown."""

    def test_contains_summary_table(self):
        metrics = {
            "A": {"recall": 0.8, "precision": 0.7, "f1": 0.75, "mean_actionability": 3.5},
            "B": {"recall": 0.9, "precision": 0.6, "f1": 0.72, "mean_actionability": 4.0},
        }
        report = render_report(metrics, [], {"A": {}, "B": {}})
        assert "| Variant" in report
        assert "0.80" in report or "0.8" in report

    def test_identifies_winner(self):
        metrics = {
            "A": {"recall": 0.5, "precision": 0.5, "f1": 0.50, "mean_actionability": 3.0},
            "B": {"recall": 0.9, "precision": 0.9, "f1": 0.90, "mean_actionability": 4.5},
        }
        report = render_report(metrics, [], {"A": {}, "B": {}})
        assert "B" in report  # Winner mentioned
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py::TestComputeMetrics tests/test_eval.py::TestRenderReport -v`
Expected: FAIL with `ImportError`

**Step 3: Implement compute_metrics and render_report**

Add to `src/lessons_db/eval.py`:

```python
def compute_metrics(scored_pairs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute per-variant aggregate metrics from scored pairs.

    Each pair has: variant, is_same_cluster, scores (dict with transfer/precision/actionability).

    Returns dict[variant_id -> {recall, precision, f1, mean_actionability}].
    - recall: fraction of same-cluster targets with transfer >= 3
    - precision: fraction of diff-cluster targets with transfer <= 2
    - f1: harmonic mean of recall and precision
    - mean_actionability: average actionability across all targets
    """
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for pair in scored_pairs:
        by_variant.setdefault(pair["variant"], []).append(pair)

    metrics: dict[str, dict[str, float]] = {}
    for variant, pairs in by_variant.items():
        same = [p for p in pairs if p["is_same_cluster"]]
        diff = [p for p in pairs if not p["is_same_cluster"]]

        recall = (
            sum(1 for p in same if p["scores"]["transfer"] >= 3) / len(same)
            if same else 0.0
        )
        precision = (
            sum(1 for p in diff if p["scores"]["transfer"] <= 2) / len(diff)
            if diff else 0.0
        )
        f1 = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) > 0 else 0.0
        )
        all_act = [p["scores"]["actionability"] for p in pairs]
        mean_act = sum(all_act) / len(all_act) if all_act else 0.0

        metrics[variant] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "mean_actionability": round(mean_act, 4),
        }

    return metrics


def render_report(
    metrics: dict[str, dict[str, float]],
    scored_pairs: list[dict[str, Any]],
    variant_configs: dict[str, Any],
) -> str:
    """Render evaluation results as a markdown report."""
    lines: list[str] = []
    lines.append("# Transfer-Test Evaluation Report\n")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Variant | Recall | Precision | F1 | Actionability |")
    lines.append("|---------|--------|-----------|-----|---------------|")
    for vid in sorted(metrics.keys()):
        m = metrics[vid]
        lines.append(
            f"| {vid} | {m['recall']:.2f} | {m['precision']:.2f} "
            f"| {m['f1']:.2f} | {m['mean_actionability']:.2f} |"
        )

    # Winner
    lines.append("\n## Winner\n")
    if metrics:
        winner = max(metrics.keys(), key=lambda v: metrics[v]["f1"])
        wm = metrics[winner]
        lines.append(
            f"**Variant {winner}** — F1: {wm['f1']:.2f} "
            f"(Recall: {wm['recall']:.2f}, Precision: {wm['precision']:.2f}, "
            f"Actionability: {wm['mean_actionability']:.2f})"
        )
        cfg = variant_configs.get(winner, {})
        if cfg:
            lines.append(f"\nModel: `{cfg.get('model', 'N/A')}`")
            lines.append(f"Prompt: `{cfg.get('prompt_id', 'N/A')}`")
            lines.append(f"Settings: temperature={cfg.get('temperature', 'N/A')}, "
                         f"num_ctx={cfg.get('num_ctx', 'N/A')}")

    # Per-cluster breakdown (if scored_pairs available)
    if scored_pairs:
        lines.append("\n## Per-Cluster Breakdown\n")
        clusters = sorted({p.get("cluster_seed", "") for p in scored_pairs})
        for cluster in clusters:
            if not cluster:
                continue
            cluster_pairs = [p for p in scored_pairs if p.get("cluster_seed") == cluster]
            if not cluster_pairs:
                continue
            avg_transfer = sum(
                p["scores"]["transfer"] for p in cluster_pairs
            ) / len(cluster_pairs)
            lines.append(f"- **Cluster {cluster}**: avg transfer = {avg_transfer:.1f} "
                         f"({len(cluster_pairs)} pairs)")

    # Failure analysis
    if scored_pairs:
        lines.append("\n## Failure Analysis\n")
        failures = [
            p for p in scored_pairs
            if p.get("is_same_cluster") and p["scores"]["transfer"] < 3
        ]
        if failures:
            lines.append(f"{len(failures)} same-cluster pairs scored below threshold:\n")
            for f in failures[:5]:
                lines.append(
                    f"- [{f.get('variant', '?')}] Principle: \"{f.get('principle', '?')[:60]}...\" "
                    f"→ Target: \"{f.get('target_title', '?')[:40]}\" (transfer={f['scores']['transfer']})"
                )
        else:
            lines.append("No same-cluster failures (all scored >= 3 on transfer).")

    return "\n".join(lines) + "\n"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestComputeMetrics tests/test_eval.py::TestRenderReport -v`
Expected: PASS

---

### Task 10: Implement run_eval_judge orchestrator and add eval-judge CLI command

**Files:**
- Modify: `src/lessons_db/eval.py`
- Modify: `src/lessons_db/cli.py`
- Modify: `tests/test_eval.py`
- Modify: `tests/test_eval_cli.py`

**Step 1: Write failing tests for run_eval_judge**

Append to `tests/test_eval.py`:

```python
from lessons_db.eval import run_eval_judge


class TestRunEvalJudge:
    """run_eval_judge orchestrates scoring of generated principles."""

    def test_produces_scored_pairs_and_metrics(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        # Create a minimal results file
        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [ids["A"][0]]},
            "results": [{
                "variant": "A",
                "lesson_id": ids["A"][0],
                "lesson_title": "Silent failure 0",
                "cluster_seed": "A",
                "principle": "Silent fallbacks mask upstream failures.",
                "model": "test-model",
                "prompt_id": "baseline-fewshot",
                "settings": {},
                "generation_time_s": 1.0,
                "error": None,
            }],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))
        report_path = tmp_path / "report.md"

        # Mock the judge to return scores
        def mock_judge(prompt, **kwargs):
            return '{"transfer": 4, "precision": 3, "actionability": 5}'

        with patch("lessons_db.eval.call_judge", side_effect=mock_judge):
            scored_pairs, metrics = run_eval_judge(
                results_path=results_path,
                conn=conn,
                report_path=report_path,
                backend="ollama",
            )

        assert len(scored_pairs) > 0
        assert "A" in metrics
        assert report_path.exists()
        report_text = report_path.read_text()
        assert "Variant" in report_text
        conn.close()

    def test_skips_error_results(self, db_path, tmp_path):
        conn = init_db(db_path)
        ids = _seed_clusters(conn)

        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [ids["A"][0]]},
            "results": [{
                "variant": "A",
                "lesson_id": ids["A"][0],
                "lesson_title": "Error lesson",
                "cluster_seed": "A",
                "principle": None,
                "error": "generation_failed",
            }],
        }
        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps(results_data))
        report_path = tmp_path / "report.md"

        scored_pairs, metrics = run_eval_judge(
            results_path=results_path,
            conn=conn,
            report_path=report_path,
            backend="ollama",
        )

        assert len(scored_pairs) == 0
        conn.close()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval.py::TestRunEvalJudge -v`
Expected: FAIL with `ImportError`

**Step 3: Implement run_eval_judge**

Add to `src/lessons_db/eval.py`:

```python
def run_eval_judge(
    results_path: Path,
    conn: sqlite3.Connection,
    report_path: Path,
    backend: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    progress_callback: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """Run eval-judge: score generated principles against transfer targets.

    Reads results JSON, constructs transfer test cases, scores each pair,
    computes metrics, and writes a markdown report.

    Returns (scored_pairs, metrics_by_variant).
    """
    results_data = _json.loads(results_path.read_text())
    results = results_data.get("results", [])

    scored_pairs: list[dict[str, Any]] = []

    for entry in results:
        principle = entry.get("principle")
        if not principle or entry.get("error"):
            continue

        variant = entry["variant"]
        lesson_id = entry["lesson_id"]
        cluster_seed = entry.get("cluster_seed", "")

        # Get transfer targets
        targets = select_transfer_targets(conn, lesson_id, cluster_seed)

        # Score each target
        for is_same, target_list in [
            (True, targets["same_cluster"]),
            (False, targets["diff_cluster"]),
        ]:
            for target in target_list:
                prompt = build_judge_prompt(principle, target)
                response = call_judge(
                    prompt=prompt,
                    backend=backend,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    openai_api_key=openai_api_key,
                    openai_model=openai_model,
                )

                scores = parse_judge_scores(response) if response else None
                if scores is None:
                    scores = {"transfer": 1, "precision": 1, "actionability": 1}

                pair = {
                    "variant": variant,
                    "source_lesson_id": lesson_id,
                    "principle": principle,
                    "target_id": target["id"],
                    "target_title": target.get("title", ""),
                    "cluster_seed": cluster_seed,
                    "is_same_cluster": is_same,
                    "scores": scores,
                }
                scored_pairs.append(pair)

                if progress_callback:
                    label = "TP" if is_same else "TN"
                    progress_callback(variant, target["id"], label, scores)

    # Compute metrics
    metrics = compute_metrics(scored_pairs)

    # Render and save report
    report = render_report(metrics, scored_pairs, VARIANT_CONFIGS)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    return scored_pairs, metrics
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval.py::TestRunEvalJudge -v`
Expected: PASS

**Step 5: Write failing CLI test for eval-judge**

Append to `tests/test_eval_cli.py`:

```python
class TestEvalJudgeHelp:
    """eval-judge --help must work."""

    def test_help_exits_zero(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "eval-judge", "--help"])
        assert result.exit_code == 0
        assert "judge" in result.output.lower() or "score" in result.output.lower()

    def test_lists_in_meta_help(self, db_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "meta", "--help"])
        assert "eval-judge" in result.output


class TestEvalJudgeCommand:
    """eval-judge reads results and produces a report."""

    def test_produces_report(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_eval_db(conn)
        conn.close()

        # Create a results file
        results_data = {
            "meta": {"variants": ["A"], "per_cluster": 1, "source_lessons": [1]},
            "results": [{
                "variant": "A",
                "lesson_id": 1,
                "lesson_title": "Cluster A lesson 0",
                "cluster_seed": "A",
                "principle": "Silent fallbacks mask upstream failures.",
                "model": "test-model",
                "prompt_id": "baseline-fewshot",
                "settings": {},
                "generation_time_s": 1.0,
                "error": None,
            }],
        }
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(results_data))
        report_file = tmp_path / "report.md"

        # Mock judge
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"response": '{"transfer": 4, "precision": 3, "actionability": 5}'}
        ).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = runner.invoke(main, [
                "--db", str(db_path),
                "meta", "eval-judge",
                str(results_file),
                "--output", str(report_file),
            ])

        assert result.exit_code == 0, f"Failed: {result.output}"
        assert report_file.exists()
        report_text = report_file.read_text()
        assert "Variant" in report_text

    def test_missing_results_file(self, db_path, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "--db", str(db_path),
            "meta", "eval-judge",
            str(tmp_path / "nonexistent.json"),
            "--output", str(tmp_path / "report.md"),
        ])
        assert result.exit_code != 0 or "not found" in result.output.lower() or "Error" in result.output
```

**Step 6: Add eval-judge CLI command to cli.py**

In `src/lessons_db/cli.py`, after the `meta_eval_generate` function, add:

```python
@meta.command("eval-judge")
@click.argument("results_file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), default=None, help="Output report path (default: auto in EVAL_DIR).")
@click.option("--openai", "use_openai", is_flag=True, help="Use OpenAI GPT-4o-mini as judge (requires OPENAI_API_KEY).")
@click.option("--judge-model", default=None, help="Judge model name (Ollama model or OpenAI model with --openai).")
@click.pass_context
def meta_eval_judge(ctx, results_file, output, use_openai, judge_model):
    """Score generated principles against transfer test targets.

    Reads a results JSON from eval-generate, constructs transfer tests
    (same-cluster true positives + different-cluster true negatives),
    scores each pair, and produces a markdown report with F1 metrics.
    """
    from lessons_db.config import EVAL_DIR, OLLAMA_QUEUE_URL, OPENAI_API_KEY
    from lessons_db.eval import VARIANT_CONFIGS, run_eval_judge

    conn = ctx.obj["conn"]
    results_path = Path(results_file)

    # Determine output path
    if output:
        report_path = Path(output)
    else:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        report_path = EVAL_DIR / f"report-{datetime.now(UTC).strftime('%Y-%m-%d')}.md"

    # Configure judge backend
    if use_openai:
        if not OPENAI_API_KEY:
            click.echo("OPENAI_API_KEY not set. Set it in ~/.env or environment.", err=True)
            ctx.exit(1)
            return
        backend = "openai"
        model = judge_model or "gpt-4o-mini"
        click.echo(f"Judge: OpenAI {model}")
    else:
        backend = "ollama"
        model = judge_model or "qwen2.5:7b"
        click.echo(f"Judge: Ollama {model}")
        _warm_model(OLLAMA_QUEUE_URL, model)

    def _progress(variant, target_id, label, scores):
        s = scores
        click.echo(f"  [{variant}] target #{target_id} ({label}): "
                    f"T={s['transfer']} P={s['precision']} A={s['actionability']}")

    scored_pairs, metrics = run_eval_judge(
        results_path=results_path,
        conn=conn,
        report_path=report_path,
        backend=backend,
        ollama_url=OLLAMA_QUEUE_URL,
        ollama_model=model if backend == "ollama" else "",
        openai_api_key=OPENAI_API_KEY if backend == "openai" else "",
        openai_model=model if backend == "openai" else "",
        progress_callback=_progress,
    )

    # Summary
    click.echo(f"\nScored {len(scored_pairs)} pairs across {len(metrics)} variants.")
    if metrics:
        winner = max(metrics.keys(), key=lambda v: metrics[v]["f1"])
        wm = metrics[winner]
        click.echo(f"Winner: Variant {winner} (F1={wm['f1']:.2f})")
    click.echo(f"Report: {report_path}")
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_eval_cli.py -v`
Expected: PASS

**Step 8: Commit Batch 3**

```bash
git add src/lessons_db/eval.py src/lessons_db/cli.py tests/test_eval.py tests/test_eval_cli.py
git commit -m "feat(eval): add eval-judge command with transfer-test scoring and metrics"
```

---

## Batch 4: Integration + Quality Gate (Task 11)

### Task 11: Integration test and full suite

**Files:**
- Modify: `tests/test_eval_cli.py`

**Step 1: Write end-to-end integration test**

Append to `tests/test_eval_cli.py`:

```python
class TestEvalPipelineIntegration:
    """End-to-end: eval-generate → eval-judge → report."""

    def test_full_pipeline(self, db_path, tmp_path):
        conn = init_db(db_path)
        _seed_eval_db(conn)
        conn.close()

        results_file = tmp_path / "results.json"
        report_file = tmp_path / "report.md"

        # Mock Ollama for generation
        gen_resp = MagicMock()
        gen_resp.read.return_value = json.dumps(
            {"response": "Pattern masking causes delayed detection when errors are silently swallowed."}
        ).encode("utf-8")
        gen_resp.__enter__ = lambda s: s
        gen_resp.__exit__ = MagicMock(return_value=False)

        runner = CliRunner()

        # Stage 1: eval-generate
        with patch("urllib.request.urlopen", return_value=gen_resp):
            result = runner.invoke(main, [
                "--db", str(db_path),
                "meta", "eval-generate",
                "--variants", "A",
                "--per-cluster", "1",
                "--output", str(results_file),
            ])
        assert result.exit_code == 0, f"eval-generate failed: {result.output}"
        assert results_file.exists()

        # Stage 2: eval-judge (with mocked judge)
        judge_resp = MagicMock()
        judge_resp.read.return_value = json.dumps(
            {"response": '{"transfer": 4, "precision": 3, "actionability": 4}'}
        ).encode("utf-8")
        judge_resp.__enter__ = lambda s: s
        judge_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=judge_resp):
            result = runner.invoke(main, [
                "--db", str(db_path),
                "meta", "eval-judge",
                str(results_file),
                "--output", str(report_file),
            ])
        assert result.exit_code == 0, f"eval-judge failed: {result.output}"
        assert report_file.exists()

        # Verify report content
        report = report_file.read_text()
        assert "# Transfer-Test Evaluation Report" in report
        assert "| Variant" in report
        assert "Winner" in report
        assert "Variant A" in report or "| A |" in report
```

**Step 2: Run integration test**

Run: `pytest tests/test_eval_cli.py::TestEvalPipelineIntegration -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `pytest --timeout=120 -x -q`
Expected: All existing tests PASS + all new eval tests PASS

**Step 4: Lint and format**

Run: `make lint && make format` (or `ruff check src/ tests/ && ruff format src/ tests/`)

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(eval): transfer-test evaluation pipeline — eval-generate + eval-judge

Two-stage CLI pipeline for systematically evaluating prompt × model × settings
variants. Stage 1 generates principles via Ollama, Stage 2 scores transfer
quality against same-cluster/different-cluster targets. Supports resume,
Ollama and OpenAI judge backends, and produces markdown reports with F1 metrics."
```

---

## Notes for Implementation

1. **Import `datetime` in cli.py**: The eval-generate command uses `datetime.now(UTC)`. Check that `from datetime import UTC, datetime` is already imported or add it to the existing imports.

2. **Random seed in tests**: `select_transfer_targets` uses `ORDER BY RANDOM()` for diff-cluster targets. Tests that check exact counts may need to account for this. The test asserts `len(targets["diff_cluster"]) == 2` which is deterministic (LIMIT handles it).

3. **OPENAI_API_KEY config**: Already exists in `config.py:113`. The eval-judge CLI imports it from there.

4. **Error handling pattern**: Follows the existing `extract-principles` pattern — errors are logged but don't abort the batch. Individual pair failures produce default scores of 1/1/1.

5. **`_warm_model` reuse**: The eval-generate command deduplicates models to warm (variants A/B/C share deepseek-r1, D/E share qwen3:14b).

6. **Resume semantics**: Only skips pairs where `error is None AND principle is not None`. Failed pairs are re-attempted on resume.
