"""Variant configurations and shared constants for the eval pipeline."""

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
# Variant configurations (A-E)
# Intentionally hardcoded: these are experiment parameters, not deployment config.
# The eval pipeline tests specific prompt × model × settings combinations.
# ---------------------------------------------------------------------------

VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "A": {
        "prompt_id": "baseline-fewshot",
        "model": "deepseek-r1:8b",
        "temperature": 0.7,
        "num_ctx": 4096,
        "chunked": False,
    },
    "B": {
        "prompt_id": "zero-shot-causal",
        "model": "deepseek-r1:8b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
    },
    "C": {
        "prompt_id": "zero-shot-chunked",
        "model": "deepseek-r1:8b",
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
        "model": "deepseek-r1:8b",
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
        "model": "deepseek-r1:8b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": False,
        "contrastive": True,
        "multi_stage": True,
    },
    "M": {
        "prompt_id": "mechanism",
        "model": "qwen3:8b",
        "temperature": 0.6,
        "num_ctx": 8192,
        "chunked": True,
        "mechanism": True,
    },
}
