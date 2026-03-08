"""Transfer-test evaluation pipeline: variant configs, test set selection, generation, judging."""

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

DEFAULT_JUDGE_MODEL = "deepseek-r1:8b-0528-qwen3-q4_K_M"

_RETRYABLE_CODES = {502, 503}
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0


# ---------------------------------------------------------------------------
# Variant configurations (A-E)
# Intentionally hardcoded: these are experiment parameters, not deployment config.
# The eval pipeline tests specific prompt × model × settings combinations.
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
    "F": {
        "prompt_id": "contrastive",
        "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
        "contrastive": True,
    },
    "G": {
        "prompt_id": "contrastive",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
        "contrastive": True,
    },
    "H": {
        "prompt_id": "contrastive-multistage",
        "model": "deepseek-r1:8b-0528-qwen3-q4_K_M",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
        "contrastive": True,
        "multi_stage": True,
    },
}


# ---------------------------------------------------------------------------
# Test set selection
# ---------------------------------------------------------------------------


def select_source_lessons(conn: sqlite3.Connection, per_cluster: int = 4) -> list[dict]:
    """Select source lessons for evaluation.

    Finds all clusters with >= 3 single-loop lessons, then picks up to
    ``per_cluster`` lessons per cluster maximising category diversity.

    Returns a flat list of lesson dicts with keys:
        id, title, one_liner, description, cluster_seed, category
    """
    # Find qualifying clusters (>= 3 single-loop lessons)
    cluster_rows = conn.execute(
        """
        SELECT cluster_seed, COUNT(*) AS cnt
        FROM lessons
        WHERE cluster_seed IS NOT NULL
          AND (loop_level IS NULL OR loop_level = 'single')
        GROUP BY cluster_seed
        HAVING cnt >= 3
        ORDER BY cluster_seed
        """
    ).fetchall()

    results: list[dict] = []

    for crow in cluster_rows:
        seed = crow["cluster_seed"]

        # Fetch all single-loop lessons in this cluster
        rows = conn.execute(
            """
            SELECT id, title, one_liner, description, cluster_seed, category
            FROM lessons
            WHERE cluster_seed = ?
              AND (loop_level IS NULL OR loop_level = 'single')
            ORDER BY id
            """,
            (seed,),
        ).fetchall()

        lessons = [dict(r) for r in rows]

        # Greedy category-diversity selection
        selected = _select_diverse(lessons, per_cluster)
        results.extend(selected)

    return results


def _select_diverse(lessons: list[dict], limit: int) -> list[dict]:
    """Greedy algorithm: first pick one from each unique category, then fill remaining slots."""
    if not lessons:
        return []

    selected: list[dict] = []
    used_ids: set[int] = set()

    # Pass 1: one per unique category
    seen_cats: set[str] = set()
    for lesson in lessons:
        cat = lesson.get("category")
        if cat not in seen_cats and len(selected) < limit:
            selected.append(lesson)
            used_ids.add(lesson["id"])
            seen_cats.add(cat)

    # Pass 2: fill remaining slots from unused lessons
    for lesson in lessons:
        if len(selected) >= limit:
            break
        if lesson["id"] not in used_ids:
            selected.append(lesson)
            used_ids.add(lesson["id"])

    return selected


def select_transfer_targets(
    conn: sqlite3.Connection,
    source_id: int,
    cluster_seed: str,
    count_same: int = 2,
    count_diff: int = 2,
) -> dict[str, list[dict]]:
    """Select transfer target lessons for a given source lesson.

    Returns:
        {"same_cluster": [...], "diff_cluster": [...]}

    - same_cluster: other lessons from same cluster, excluding source,
      preferring different categories (sort: different category first).
    - diff_cluster: lessons from other clusters, selected randomly.
    - All single-loop only.
    """
    # Get source lesson's category for preference sorting
    source_row = conn.execute("SELECT category FROM lessons WHERE id = ?", (source_id,)).fetchone()
    if source_row is None:
        _log.warning("select_transfer_targets: source_id=%d not found", source_id)
    source_category = source_row["category"] if source_row else None

    # Same cluster, excluding source, single-loop only
    # Sort: different category first (0 before 1), then by id for stability
    same_rows = conn.execute(
        """
        SELECT id, title, one_liner, description, cluster_seed, category
        FROM lessons
        WHERE cluster_seed = ?
          AND id != ?
          AND (loop_level IS NULL OR loop_level = 'single')
        ORDER BY
            CASE WHEN category = ? THEN 1 ELSE 0 END,
            id
        """,
        (cluster_seed, source_id, source_category),
    ).fetchall()

    same_cluster = [dict(r) for r in same_rows[:count_same]]

    # Different cluster, single-loop, random selection
    diff_rows = conn.execute(
        """
        SELECT id, title, one_liner, description, cluster_seed, category
        FROM lessons
        WHERE cluster_seed IS NOT NULL
          AND cluster_seed != ?
          AND (loop_level IS NULL OR loop_level = 'single')
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (cluster_seed, count_diff),
    ).fetchall()

    diff_cluster = [dict(r) for r in diff_rows]

    return {
        "same_cluster": same_cluster,
        "diff_cluster": diff_cluster,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_generation_prompt(
    variant_id: str,
    lesson: dict[str, Any],
    siblings: list[dict[str, Any]] | None = None,
    diff_cluster_items: list[dict[str, Any]] | None = None,
) -> str:
    """Build the principle-extraction prompt for a given variant.

    Variants A use few-shot examples. B/D use zero-shot causal framing.
    C/E use chunked (multiple sibling lessons from same cluster).
    F/G use contrastive (same-cluster + diff-cluster for specificity).
    """
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


def _build_contrastive_prompt(
    primary: dict[str, Any],
    same_cluster_items: list[dict[str, Any]],
    diff_cluster_items: list[dict[str, Any]],
) -> str:
    """Variants F/G: show same-cluster AND diff-cluster items to force specificity."""
    same_lines = []
    all_same = [primary, *same_cluster_items]
    for i, item in enumerate(all_same, 1):
        t = item.get("title") or ""
        o = item.get("one_liner") or ""
        same_lines.append(f"  {i}. {t} — {o}")
    same_block = "\n".join(same_lines)

    diff_lines = []
    for i, item in enumerate(diff_cluster_items, 1):
        t = item.get("title") or ""
        o = item.get("one_liner") or ""
        diff_lines.append(f"  {i}. {t} — {o}")
    diff_block = "\n".join(diff_lines)

    return (
        "SAME PATTERN (these lessons share the same structural failure):\n"
        f"{same_block}\n\n"
        "DIFFERENT PATTERNS (these are UNRELATED failure types):\n"
        f"{diff_block}\n\n"
        "Extract ONE structural principle that:\n"
        "- Is TRUE for ALL lessons in the SAME PATTERN group\n"
        "- Is FALSE or IRRELEVANT for the DIFFERENT PATTERNS group\n"
        "- Names the structural pattern, not the technology\n\n"
        "The principle must be specific enough to DISTINGUISH this failure type "
        "from the others listed above.\n\n"
        "Causal form: '<pattern> causes <consequence> when <condition>'\n"
        "One sentence, 10-25 words. No technology names."
    )


def _build_self_critique_prompt(
    principle: str,
    diff_cluster_items: list[dict],
) -> str:
    """Build a self-critique prompt that tests if a principle is too general."""
    diff_lines = []
    for i, item in enumerate(diff_cluster_items, 1):
        t = item.get("title") or ""
        o = item.get("one_liner") or ""
        diff_lines.append(f"  {i}. {t} — {o}")
    diff_block = "\n".join(diff_lines)

    return (
        f'You previously extracted this principle: "{principle}"\n\n'
        "Here are UNRELATED lessons from different failure categories:\n"
        f"{diff_block}\n\n"
        "Question: Does this principle also apply to ANY of the "
        "unrelated lessons above?\n\n"
        "If YES — the principle is too general. Rewrite it to be more "
        "specific, so it ONLY matches the original failure type and NOT "
        "the unrelated ones.\n"
        "If NO — the principle is specific enough. Return it unchanged.\n\n"
        "Return ONLY the (possibly refined) principle. One sentence. "
        "Causal form: '<pattern> causes <consequence> when <condition>'\n"
        "No explanation."
    )


# ---------------------------------------------------------------------------
# Ollama integration
# ---------------------------------------------------------------------------


def call_ollama(
    queue_url: str,
    model: str,
    prompt: str,
    settings: dict[str, Any],
    timeout: int = 300,
    priority: int | None = None,
    source: str | None = None,
) -> str | None:
    """Call Ollama via queue and return cleaned response text.

    Retries up to _MAX_RETRIES times on 502/503 (model swap transients).
    Returns None on any error (network, timeout, parse).
    Strips <think>...</think> reasoning blocks from response.

    When priority/source are set, passes _priority/_source/_timeout to
    ollama-queue's proxy endpoint for job tracking and prioritization.
    """
    options = {}
    if "temperature" in settings:
        options["temperature"] = settings["temperature"]
    if "num_ctx" in settings:
        options["num_ctx"] = settings["num_ctx"]

    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        **({"options": options} if options else {}),
    }
    if priority is not None:
        body["_priority"] = priority
    if source is not None:
        body["_source"] = source
    if priority is not None or source is not None:
        body["_timeout"] = timeout

    payload = _json.dumps(body).encode("utf-8")

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
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
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in _RETRYABLE_CODES and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                _log.warning("call_ollama %d retry in %.0fs: %s", exc.code, delay, exc)
                time.sleep(delay)
                continue
            _log.warning("call_ollama error: %s", exc)
            return None
        except (urllib.error.URLError, OSError, _json.JSONDecodeError) as exc:
            _log.warning("call_ollama error: %s", exc)
            return None

    _log.warning("call_ollama exhausted retries: %s", last_exc)
    return None


# ---------------------------------------------------------------------------
# Generation orchestrator
# ---------------------------------------------------------------------------


def _load_resume_state(output_path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    """Load existing results and extract completed (variant, lesson_id) pairs."""
    existing = _json.loads(output_path.read_text())
    existing_results = existing.get("results", [])
    completed_pairs: set[tuple[str, int]] = set()
    for r in existing_results:
        if r.get("error") is None and r.get("principle"):
            completed_pairs.add((r["variant"], r["lesson_id"]))
    return existing_results, completed_pairs


def _load_siblings_by_cluster(
    conn: sqlite3.Connection,
    sources: list[dict],
) -> dict[str, list[dict[str, Any]]]:
    """Pre-fetch sibling lessons grouped by cluster_seed for chunked variants."""
    siblings_by_cluster: dict[str, list[dict[str, Any]]] = {}
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
    return siblings_by_cluster


def _generate_for_lesson(
    variant_id: str,
    config: dict[str, Any],
    lesson: dict[str, Any],
    queue_url: str,
    siblings_by_cluster: dict[str, list[dict[str, Any]]],
    priority: int | None = None,
) -> dict[str, Any]:
    """Generate a principle for a single (variant, lesson) pair."""
    lesson_id = lesson["id"]
    model = config["model"]
    settings = {"temperature": config["temperature"], "num_ctx": config["num_ctx"]}

    siblings = None
    diff_cluster_items = None
    seed = lesson.get("cluster_seed", "")

    if config["chunked"] or config.get("contrastive"):
        all_sibs = siblings_by_cluster.get(seed, [])
        siblings = [s for s in all_sibs if s["id"] != lesson_id][:3]

    if config.get("contrastive"):
        all_diff = []
        for other_seed, other_items in sorted(siblings_by_cluster.items()):
            if other_seed != seed:
                all_diff.extend(other_items[:2])
        diff_cluster_items = all_diff[:4]

    prompt = build_generation_prompt(variant_id, lesson, siblings=siblings, diff_cluster_items=diff_cluster_items)

    t0 = time.monotonic()
    principle = call_ollama(queue_url, model, prompt, settings, priority=priority, source="eval-generate")
    elapsed = round(time.monotonic() - t0, 1)

    # Self-critique: refine if principle is too general
    if principle and config.get("multi_stage") and diff_cluster_items:
        critique_prompt = _build_self_critique_prompt(principle, diff_cluster_items)
        refined = call_ollama(
            queue_url,
            model,
            critique_prompt,
            settings,
            priority=priority,
            source="eval-generate-critique",
        )
        if refined and len(refined.strip()) > 10:
            principle = refined.strip()

    return {
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


def _save_results(
    output_path: Path,
    variants: list[str],
    per_cluster: int,
    source_ids: list[int],
    results: list[dict[str, Any]],
) -> None:
    """Write results JSON to disk (called incrementally after each generation)."""
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


def run_eval_generate(
    conn: sqlite3.Connection,
    queue_url: str,
    variants: list[str],
    per_cluster: int,
    output_path: Path,
    resume: bool,
    progress_callback: Any = None,
    priority: int | None = None,
) -> dict[str, Any]:
    """Run eval-generate: produce principles for all (variant, lesson) pairs.

    Saves results incrementally to output_path as JSON.
    When priority is set, passes it to ollama-queue for job prioritization.
    """
    # Load existing results if resuming
    existing_results: list[dict[str, Any]] = []
    completed_pairs: set[tuple[str, int]] = set()
    if resume and output_path.exists():
        existing_results, completed_pairs = _load_resume_state(output_path)

    # Select source lessons
    sources = select_source_lessons(conn, per_cluster=per_cluster)
    source_ids = [s["id"] for s in sources]

    # Build results structure
    results: list[dict[str, Any]] = list(existing_results)

    # Sort variants by model to minimize Ollama model swaps
    sorted_variants = sorted(variants, key=lambda v: VARIANT_CONFIGS[v]["model"])

    for variant_id in sorted_variants:
        config = VARIANT_CONFIGS[variant_id]

        # Pre-fetch siblings for chunked and contrastive variants
        siblings_by_cluster: dict[str, list[dict[str, Any]]] = {}
        if config["chunked"] or config.get("contrastive"):
            siblings_by_cluster = _load_siblings_by_cluster(conn, sources)

        for lesson in sources:
            if (variant_id, lesson["id"]) in completed_pairs:
                continue

            entry = _generate_for_lesson(variant_id, config, lesson, queue_url, siblings_by_cluster, priority=priority)
            results.append(entry)

            if progress_callback:
                progress_callback(variant_id, lesson["id"], entry["principle"] is not None)

            # Incremental save after each generation to survive crashes
            _save_results(output_path, variants, per_cluster, source_ids, results)

    # Final save with updated timestamp
    _save_results(output_path, variants, per_cluster, source_ids, results)
    return {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "variants": variants,
            "per_cluster": per_cluster,
            "source_lessons": source_ids,
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Judge prompt + scoring
# ---------------------------------------------------------------------------


def build_judge_prompt(principle: str, target: dict[str, Any]) -> str:
    """Build rubric-based scoring prompt with calibration anchors.

    Includes concrete scored examples so the judge's internal scale is
    anchored, reducing score inflation on cross-cluster pairs.
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
        "Score this (principle, target) pair on three criteria, each 1-5.\n\n"
        "## Scoring Guide with Examples\n\n"
        "**Transfer Recognition** — does the principle structurally match the target?\n"
        "  1 = No structural connection. E.g. principle about resource cleanup → target about naming conventions → 1\n"
        "  3 = Vague thematic overlap but different mechanism. "
        "E.g. error handling principle → logging gaps target → 3\n"
        "  5 = Same structural pattern, different technology. "
        "E.g. resource cleanup principle → unclosed DB connections → 5\n\n"
        "**Precision** — would this principle false-positive on unrelated lessons?\n"
        "  1 = So general it matches everything (e.g. 'always test your code')\n"
        "  3 = Matches a broad category but not everything\n"
        "  5 = Only matches lessons with the same specific structural failure\n\n"
        "**Actionability** — could an LLM use this to prevent this class of bug?\n"
        "  1 = Too abstract to act on (e.g. 'be careful with state')\n"
        "  3 = Useful with additional context\n"
        "  5 = Specific enough to implement a check or review step\n\n"
        "IMPORTANT: Be skeptical. Most principles do NOT transfer to unrelated lessons. "
        "Default to low transfer scores unless there is a clear structural match.\n\n"
        'Return ONLY a JSON object: {"transfer": N, "precision": N, "actionability": N}\n'
        "No explanation."
    )


def parse_judge_scores(response: str) -> dict[str, int] | None:
    """Extract transfer/precision/actionability scores from judge response.

    Returns dict with keys transfer, precision, actionability (ints 1-5),
    or None if parsing fails.
    """
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

    scores = {}
    for key in required:
        val = int(data[key])
        scores[key] = max(1, min(5, val))
    return scores


# ---------------------------------------------------------------------------
# Judge backend (Ollama + OpenAI)
# ---------------------------------------------------------------------------


def call_judge(
    prompt: str,
    backend: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
    openai_api_key: str = "",
    openai_model: str = "gpt-4o-mini",
    priority: int | None = None,
) -> str | None:
    """Call the judge model and return raw response text.

    Routes to Ollama or OpenAI based on backend parameter.
    Returns None on any error.
    """
    if backend == "openai":
        return _call_openai(openai_api_key, openai_model, prompt)
    return call_ollama(ollama_url, ollama_model, prompt, {}, priority=priority, source="eval-judge")


def _call_openai(api_key: str, model: str, prompt: str) -> str | None:
    """Call OpenAI Chat Completions API. Returns response text or None."""
    payload = _json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.1,
        }
    ).encode("utf-8")

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
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, _json.JSONDecodeError, KeyError, IndexError) as exc:
        _log.warning("_call_openai error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Metrics + reporting
# ---------------------------------------------------------------------------


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

        recall = sum(1 for p in same if p["scores"]["transfer"] >= 3) / len(same) if same else 0.0
        precision = sum(1 for p in diff if p["scores"]["transfer"] <= 2) / len(diff) if diff else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        all_act = [p["scores"]["actionability"] for p in pairs]
        mean_act = sum(all_act) / len(all_act) if all_act else 0.0

        metrics[variant] = {
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "mean_actionability": round(mean_act, 4),
        }

    return metrics


def _render_pair_sections(scored_pairs: list[dict[str, Any]], lines: list[str]) -> None:
    """Append per-cluster breakdown and failure analysis to report lines."""
    if not scored_pairs:
        return

    # Per-cluster breakdown
    lines.append("\n## Per-Cluster Breakdown\n")
    clusters = sorted({p.get("cluster_seed", "") for p in scored_pairs})
    for cluster in clusters:
        if not cluster:
            continue
        cluster_pairs = [p for p in scored_pairs if p.get("cluster_seed") == cluster]
        if not cluster_pairs:
            continue
        avg_transfer = sum(p["scores"]["transfer"] for p in cluster_pairs) / len(cluster_pairs)
        lines.append(f"- **Cluster {cluster}**: avg transfer = {avg_transfer:.1f} " f"({len(cluster_pairs)} pairs)")

    # Failure analysis
    lines.append("\n## Failure Analysis\n")
    failures = [p for p in scored_pairs if p.get("is_same_cluster") and p["scores"]["transfer"] < 3]
    if failures:
        lines.append(f"{len(failures)} same-cluster pairs scored below threshold:\n")
        for f in failures[:5]:
            lines.append(
                f"- [{f.get('variant', '?')}] Principle: \"{f.get('principle', '?')[:60]}...\" "
                f"-> Target: \"{f.get('target_title', '?')[:40]}\" (transfer={f['scores']['transfer']})"
            )
    else:
        lines.append("No same-cluster failures (all scored >= 3 on transfer).")


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
            f"| {vid} | {m['recall']:.2f} | {m['precision']:.2f} " f"| {m['f1']:.2f} | {m['mean_actionability']:.2f} |"
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
            lines.append(
                f"Settings: temperature={cfg.get('temperature', 'N/A')}, " f"num_ctx={cfg.get('num_ctx', 'N/A')}"
            )

    _render_pair_sections(scored_pairs, lines)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Judge orchestrator
# ---------------------------------------------------------------------------


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
    priority: int | None = None,
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

        targets = select_transfer_targets(conn, lesson_id, cluster_seed)

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
                    priority=priority,
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
                    "target_cluster_seed": target.get("cluster_seed", ""),
                    "is_same_cluster": is_same,
                    "scores": scores,
                }
                scored_pairs.append(pair)

                if progress_callback:
                    label = "TP" if is_same else "TN"
                    progress_callback(variant, target["id"], label, scores)

    metrics = compute_metrics(scored_pairs)

    report = render_report(metrics, scored_pairs, VARIANT_CONFIGS)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    # Save scored pairs for diagnostic tools (confusion matrix, etc.)
    scored_path = report_path.with_suffix(".scored.json")
    scored_path.write_text(_json.dumps(scored_pairs, indent=2))

    return scored_pairs, metrics
