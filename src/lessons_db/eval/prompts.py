"""Prompt construction for generation, judging, mechanism extraction, and simulation."""

import hashlib
import re as _re
from typing import Any

from lessons_db.eval.variants import VARIANT_CONFIGS


def _clean_principle_for_prompt(text: str) -> str:
    """Lightweight principle cleaning for prompt embedding.

    Imported lazily to avoid circular imports during transitional extraction.
    """
    from lessons_db.eval import _clean_principle

    return _clean_principle(text)


# ---------------------------------------------------------------------------
# Generation prompts
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
# Judge prompts
# ---------------------------------------------------------------------------


def build_judge_prompt(principle: str, target: dict[str, Any]) -> str:
    """Build rubric-based scoring prompt with calibration anchors.

    Cleans CoT artifacts from the principle before embedding in the prompt.
    Includes concrete scored examples so the judge's internal scale is
    anchored, reducing score inflation on cross-cluster pairs.
    """
    principle = _clean_principle_for_prompt(principle)
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


def build_binary_judge_prompt(principle: str, target: dict[str, Any]) -> str:
    """Binary discrimination prompt — forces YES/NO instead of 1-5 scale.

    Designed to be harder for models to hedge.  Asks whether the principle
    describes the SPECIFIC mechanism in the target, not just a general theme.
    """
    principle = _clean_principle_for_prompt(principle)
    title = target.get("title") or ""
    one_liner = target.get("one_liner") or ""
    description = (target.get("description") or "")[:300]

    return (
        "You are a strict evaluator. Answer ONLY 'YES' or 'NO'.\n\n"
        f'PRINCIPLE: "{principle}"\n\n'
        f"TARGET LESSON:\n"
        f"Title: {title}\n"
        f"One-liner: {one_liner}\n"
        f"Description: {description}\n\n"
        "Question: Does this principle describe the SPECIFIC failure "
        "mechanism in this target lesson?\n\n"
        "Rules:\n"
        "- YES means: the principle identifies the EXACT structural pattern "
        "that caused this specific bug. If you removed the principle, this "
        "bug class would not be caught.\n"
        "- NO means: the principle is about a DIFFERENT failure mechanism, "
        "or is so general it would match any lesson about software bugs.\n"
        "- Two lessons both involving 'errors' is NOT enough for YES. "
        "The mechanism must be the same (e.g. both about resource cleanup, "
        "or both about race conditions, or both about missing validation).\n\n"
        "Answer: YES or NO"
    )


def build_paired_judge_prompt(
    principle: str,
    same_target: dict[str, Any],
    diff_target: dict[str, Any],
    position_seed: int | None = None,
) -> tuple[str, bool]:
    """Paired comparison prompt — which target does the principle apply to more?

    Randomizes A/B position to eliminate position bias.
    Returns (prompt_text, same_is_a) where same_is_a indicates if the same-group
    target was placed in position A.
    """
    # Strip think tags before cleaning (case-insensitive — models emit both cases)
    principle = _re.sub(r"<think>.*?</think>", "", principle, flags=_re.DOTALL | _re.IGNORECASE).strip()
    principle = _clean_principle_for_prompt(principle)

    if position_seed is None:
        position_seed = int(hashlib.md5(principle.encode(), usedforsecurity=False).hexdigest()[:8], 16)
    swap = position_seed % 2 == 0

    target_a = diff_target if swap else same_target
    target_b = same_target if swap else diff_target

    def _fmt(t: dict[str, Any]) -> str:
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


# ---------------------------------------------------------------------------
# Mechanism extraction prompt
# ---------------------------------------------------------------------------


def build_mechanism_extraction_prompt(lesson_a: dict, lesson_b: dict) -> str:
    """Extract shared failure mechanism as a triplet from two lessons."""

    def _fmt(l: dict) -> str:
        return (
            f"Title: {l.get('title', '')}\n"
            f"One-liner: {l.get('one_liner', '')}\n"
            f"Description: {(l.get('description', '') or '')[:300]}"
        )

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


# ---------------------------------------------------------------------------
# Simulation prompt
# ---------------------------------------------------------------------------


def build_simulation_prompt(scenario: str, principle: str | None = None) -> str:
    """Build a bug-catching simulation prompt.

    With principle: LLM has the rule and should catch the bug.
    Without principle (control): LLM has no rule.
    Lift = with_catch_rate - without_catch_rate.
    """
    rule_section = ""
    if principle:
        rule_section = "\n## CODING RULE (always check for this)\n" f"{principle}\n\n"
    return (
        "You are reviewing code for a potential bug.\n"
        f"{rule_section}"
        f"## SCENARIO\n{scenario}\n\n"
        "Does this code have a bug related to resource management, "
        "error handling, or structural correctness?\n\n"
        "Answer: BUG FOUND: [description] or NO BUG FOUND"
    )
