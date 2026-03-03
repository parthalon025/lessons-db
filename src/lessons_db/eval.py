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


# ---------------------------------------------------------------------------
# Ollama integration
# ---------------------------------------------------------------------------


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

    payload = _json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **({"options": options} if options else {}),
        }
    ).encode("utf-8")

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
) -> dict[str, Any]:
    """Generate a principle for a single (variant, lesson) pair."""
    lesson_id = lesson["id"]
    model = config["model"]
    settings = {"temperature": config["temperature"], "num_ctx": config["num_ctx"]}

    siblings = None
    if config["chunked"]:
        all_sibs = siblings_by_cluster.get(lesson["cluster_seed"], [])
        siblings = [s for s in all_sibs if s["id"] != lesson_id][:3]

    prompt = build_generation_prompt(variant_id, lesson, siblings=siblings)

    t0 = time.monotonic()
    principle = call_ollama(queue_url, model, prompt, settings)
    elapsed = round(time.monotonic() - t0, 1)

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
) -> dict[str, Any]:
    """Run eval-generate: produce principles for all (variant, lesson) pairs.

    Saves results incrementally to output_path as JSON.
    Returns the full results dict.
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

    for variant_id in variants:
        config = VARIANT_CONFIGS[variant_id]

        # Pre-fetch siblings for chunked variants
        siblings_by_cluster: dict[str, list[dict[str, Any]]] = {}
        if config["chunked"]:
            siblings_by_cluster = _load_siblings_by_cluster(conn, sources)

        for lesson in sources:
            if (variant_id, lesson["id"]) in completed_pairs:
                continue

            entry = _generate_for_lesson(variant_id, config, lesson, queue_url, siblings_by_cluster)
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

    metrics = compute_metrics(scored_pairs)

    report = render_report(metrics, scored_pairs, VARIANT_CONFIGS)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)

    return scored_pairs, metrics
