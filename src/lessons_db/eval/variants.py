"""Variant configurations and shared constants for the eval pipeline.

Eval pipeline overview
----------------------
The pipeline measures how well different prompt/model combinations extract
*transferable* principles from lessons.  A principle is transferable if it
helps a model correctly apply a lesson to a *new* situation (same cluster,
different lesson) and does *not* incorrectly apply it to an unrelated lesson
(different cluster).

Ground truth is derived from cluster membership (cluster_seed or category).
For each generated principle the judge scores pairs:

  same-cluster pair  → high transfer score expected  (recall signal)
  diff-cluster pair  → low transfer score expected   (precision signal)

F1 = harmonic mean of recall and precision over all pairs.
"""

import logging
from typing import Any

_log = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "deepseek-r1:8b"
DEFAULT_BINARY_JUDGE_MODEL = "gemma3:12b"

_RETRYABLE_CODES = {502, 503}
_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0

# Valid group_by values for eval ground truth grouping.
# Used in SQL via f-string interpolation — validated before any query.
VALID_GROUP_BY = ("category", "cluster_seed")


# ---------------------------------------------------------------------------
# Variant configurations (A, B, C, D, E, F, G, H, M)
#
# Each variant is one cell in a factorial experiment:
#   prompt_id  — which generation prompt strategy to use
#   model      — which local Ollama model runs the generation
#   chunked    — whether lessons are split into ~512-token chunks before
#                generation (trades coherence for context window efficiency)
#   contrastive — prompt explicitly asks the model to contrast what the
#                 principle does/doesn't apply to (sharper precision)
#   multi_stage — two-pass generation: first extract pattern, then distill
#                 principle (more deliberate, slower)
#   mechanism   — prompt asks for root-cause mechanism, not just a rule
#                 (captures *why* the lesson applies, not just *what* to do)
#
# Intentionally hardcoded: these are experiment parameters, not runtime
# config.  Change them only when adding a new experimental hypothesis.
# ---------------------------------------------------------------------------

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    # A — Baseline (few-shot, deepseek-r1:8b, small context)
    #   Hypothesis: few-shot examples anchor the model on the output format.
    #   Uses 4096-token context (fits 1-2 lessons comfortably).
    #   Serves as the control against which all other variants are judged.
    "A": {
        "prompt_id": "baseline-fewshot",
        "model": "deepseek-r1:8b",
        "temperature": 0.7,
        "num_ctx": 4096,
        "chunked": False,
    },
    # B — Zero-shot causal, deepseek-r1:8b, full context
    #   Hypothesis: removing few-shot examples forces the model to reason
    #   causally ("why did this fail?") rather than pattern-match the format.
    #   Double the context window vs A.
    "B": {
        "prompt_id": "zero-shot-causal",
        "model": "deepseek-r1:8b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
    },
    # C — Zero-shot chunked, deepseek-r1:8b
    #   Hypothesis: chunking each lesson independently (≈512 tokens/chunk)
    #   prevents long-context degradation and isolates each failure pattern.
    #   Tests whether per-chunk focus beats whole-lesson coherence (B).
    "C": {
        "prompt_id": "zero-shot-chunked",
        "model": "deepseek-r1:8b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": True,
    },
    # D — Zero-shot causal, qwen3:14b (model swap vs B)
    #   Hypothesis: a larger instruction-tuned model (14B vs 8B) extracts
    #   more precise principles from the same zero-shot-causal prompt.
    #   B vs D isolates model capability; C vs E adds chunking on top.
    "D": {
        "prompt_id": "zero-shot-causal",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
    },
    # E — Zero-shot chunked, qwen3:14b (chunking + larger model)
    #   Hypothesis: qwen3:14b + chunking combines the best of D and C.
    #   Chunked isolation may help the 14B model stay on-topic per chunk.
    "E": {
        "prompt_id": "zero-shot-chunked",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": True,
    },
    # F — Contrastive, deepseek-r1:8b
    #   Hypothesis: explicitly prompting the model to state *when* the
    #   principle does NOT apply sharpens specificity and improves precision
    #   (fewer false positives at the judge step).
    "F": {
        "prompt_id": "contrastive",
        "model": "deepseek-r1:8b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
        "contrastive": True,
    },
    # G — Contrastive, qwen3:14b (model swap for contrastive prompt)
    #   Hypothesis: the 14B model follows contrastive instructions more
    #   faithfully, producing even sharper precision than F.
    "G": {
        "prompt_id": "contrastive",
        "model": "qwen3:14b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
        "contrastive": True,
    },
    # H — Contrastive + multi-stage, deepseek-r1:8b
    #   Hypothesis: a two-pass pipeline (pass 1: extract abstract pattern;
    #   pass 2: distill into transferable principle) produces the most
    #   generalisable output, at the cost of 2× LLM calls.
    "H": {
        "prompt_id": "contrastive-multistage",
        "model": "deepseek-r1:8b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
        "contrastive": True,
        "multi_stage": True,
    },
    # M — Mechanism extraction, qwen3.5:9b
    #   Hypothesis: capturing *why* a pattern occurs (root-cause mechanism)
    #   transfers better than a surface-level rule, because the mechanism
    #   applies wherever the same causal chain is present.
    #   Uses chunked input to keep each mechanism focused on one failure mode.
    "M": {
        "prompt_id": "mechanism",
        "model": "qwen3.5:9b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": True,
        "mechanism": True,
    },
}
