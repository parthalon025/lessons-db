"""Generation orchestrator: produce principles for (variant, lesson) pairs."""

import json as _json
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lessons_db.eval.client import _clean_principle, call_ollama
from lessons_db.eval.prompts import (
    _build_self_critique_prompt,
    build_generation_prompt,
    build_mechanism_extraction_prompt,
)
from lessons_db.eval.sampling import increment_eval_seen, select_source_lessons, split_holdout
from lessons_db.eval.signals import parse_mechanism_triplet
from lessons_db.eval.variants import VALID_GROUP_BY, VARIANT_CONFIGS

_log = logging.getLogger(__name__)


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
    group_by: str = "category",
) -> dict[str, list[dict[str, Any]]]:
    """Pre-fetch sibling lessons grouped by *group_by* column for chunked variants."""
    if group_by not in VALID_GROUP_BY:
        raise ValueError(f"group_by must be one of {VALID_GROUP_BY!r}, got {group_by!r}")

    siblings_by_group: dict[str, list[dict[str, Any]]] = {}
    for src in sources:
        group_value = src.get(group_by, "")
        if group_value not in siblings_by_group:
            sibs = conn.execute(
                "SELECT id, title, one_liner, description, cluster_seed, category "
                "FROM lessons "
                f"WHERE {group_by} = ? AND (loop_level IS NULL OR loop_level = 'single') "
                "ORDER BY id",
                (group_value,),
            ).fetchall()
            siblings_by_group[group_value] = [dict(r) for r in sibs]
    return siblings_by_group


def _generate_mechanism(
    variant_id: str,
    config: dict[str, Any],
    lesson: dict[str, Any],
    queue_url: str,
    siblings_by_cluster: dict[str, list[dict[str, Any]]],
    group_value: str,
    priority: int | None = None,
) -> dict[str, Any]:
    """Generate a principle via mechanism extraction from sibling lesson pairs."""
    lesson_id = lesson["id"]
    model = config["model"]
    settings = {"temperature": config["temperature"], "num_ctx": config["num_ctx"]}

    base_entry = {
        "variant": variant_id,
        "lesson_id": lesson_id,
        "lesson_title": lesson.get("title", ""),
        "cluster_seed": lesson.get("cluster_seed", ""),
        "category": lesson.get("category", ""),
        "model": model,
        "prompt_id": config["prompt_id"],
        "settings": settings,
    }

    all_sibs = siblings_by_cluster.get(group_value, [])
    sibs = [s for s in all_sibs if s["id"] != lesson_id][:2]
    if not sibs:
        return {
            **base_entry,
            "principle": None,
            "generation_time_s": 0.0,
            "error": "no_siblings_for_mechanism",
        }

    t0 = time.monotonic()
    triplets = []
    for sib in sibs:
        mech_prompt = build_mechanism_extraction_prompt(lesson, sib)
        response = call_ollama(
            queue_url,
            model,
            mech_prompt,
            settings,
            priority=priority,
            source="eval-generate-mechanism",
        )
        triplet = parse_mechanism_triplet(response)
        if triplet:
            triplets.append(triplet)

    elapsed = round(time.monotonic() - t0, 1)
    if not triplets:
        return {
            **base_entry,
            "principle": None,
            "generation_time_s": elapsed,
            "error": "mechanism_extraction_failed",
        }

    best = triplets[0]
    principle = f"TRIGGER: {best['trigger']} | TARGET: {best['target']} | FIX: {best['fix']}"
    return {
        **base_entry,
        "principle": principle,
        "generation_time_s": elapsed,
        "error": None,
    }


def _generate_for_lesson(
    variant_id: str,
    config: dict[str, Any],
    lesson: dict[str, Any],
    queue_url: str,
    siblings_by_cluster: dict[str, list[dict[str, Any]]],
    priority: int | None = None,
    group_by: str = "category",
) -> dict[str, Any]:
    """Generate a principle for a single (variant, lesson) pair."""
    lesson_id = lesson["id"]
    model = config["model"]
    settings = {"temperature": config["temperature"], "num_ctx": config["num_ctx"]}

    siblings = None
    diff_cluster_items = None
    group_value = lesson.get(group_by, "")

    # Mechanism extraction: extract shared failure mechanism triplet from sibling pairs
    if config.get("mechanism"):
        return _generate_mechanism(
            variant_id,
            config,
            lesson,
            queue_url,
            siblings_by_cluster,
            group_value,
            priority,
        )

    if config["chunked"] or config.get("contrastive"):
        all_sibs = siblings_by_cluster.get(group_value, [])
        siblings = [s for s in all_sibs if s["id"] != lesson_id][:3]

    if config.get("contrastive"):
        all_diff = []
        for other_key, other_items in sorted(siblings_by_cluster.items()):
            if other_key != group_value:
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

    # Clean CoT artifacts before storing
    if principle:
        principle = _clean_principle(principle)

    return {
        "variant": variant_id,
        "lesson_id": lesson_id,
        "lesson_title": lesson.get("title", ""),
        "cluster_seed": lesson.get("cluster_seed", ""),
        "category": lesson.get("category", ""),
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
    group_by: str = "category",
    holdout_ids: list[int] | None = None,
) -> None:
    """Write results JSON to disk (called incrementally after each generation)."""
    meta: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "variants": variants,
        "per_cluster": per_cluster,
        "group_by": group_by,
        "source_lessons": source_ids,
    }
    if holdout_ids:
        meta["holdout_lessons"] = holdout_ids
    output = {"meta": meta, "results": results}
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
    group_by: str = "category",
    holdout_fraction: float | None = None,
) -> dict[str, Any]:
    """Run eval-generate: produce principles for all (variant, lesson) pairs.

    Saves results incrementally to output_path as JSON.
    When priority is set, passes it to ollama-queue for job prioritization.
    When holdout_fraction is set, reserves that fraction as a held-out test set
    (not used for generation) to prevent Goodhart overfitting.
    """
    # Load existing results if resuming
    existing_results: list[dict[str, Any]] = []
    completed_pairs: set[tuple[str, int]] = set()
    if resume and output_path.exists():
        existing_results, completed_pairs = _load_resume_state(output_path)

    # Select source lessons
    all_sources = select_source_lessons(conn, per_cluster=per_cluster, group_by=group_by)

    # Optionally split into dev (used for generation) and held-out test set
    holdout_ids: list[int] = []
    if holdout_fraction is not None:
        sources, holdout = split_holdout(all_sources, holdout_fraction=holdout_fraction)
        holdout_ids = [s["id"] for s in holdout]
        _log.info("Holdout split: %d dev, %d test", len(sources), len(holdout))
    else:
        sources = all_sources

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
            siblings_by_cluster = _load_siblings_by_cluster(conn, sources, group_by=group_by)

        for lesson in sources:
            if (variant_id, lesson["id"]) in completed_pairs:
                continue

            entry = _generate_for_lesson(
                variant_id,
                config,
                lesson,
                queue_url,
                siblings_by_cluster,
                priority=priority,
                group_by=group_by,
            )
            results.append(entry)

            if progress_callback:
                progress_callback(variant_id, lesson["id"], entry["principle"] is not None)

            # Incremental save after each generation to survive crashes
            _save_results(
                output_path, variants, per_cluster, source_ids, results, group_by=group_by, holdout_ids=holdout_ids
            )

    # Final save with updated timestamp
    _save_results(output_path, variants, per_cluster, source_ids, results, group_by=group_by, holdout_ids=holdout_ids)

    # Increment rotation counter so future runs deprioritise these lessons
    increment_eval_seen(conn, source_ids)

    meta: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "variants": variants,
        "per_cluster": per_cluster,
        "group_by": group_by,
        "source_lessons": source_ids,
    }
    if holdout_ids:
        meta["holdout_lessons"] = holdout_ids
    return {"meta": meta, "results": results}
