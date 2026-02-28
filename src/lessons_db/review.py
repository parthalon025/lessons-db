"""Draft triage pipeline: noise filter + Claude batch review + verdict execution."""

from __future__ import annotations

import json
import logging
import re

_log = logging.getLogger(__name__)

# Noise patterns — one_liners matching these are auto-dismissed
_NOISE_PATTERNS = [
    re.compile(r"no\s+(coding\s+)?mistakes?\s+were\s+(found|discovered)", re.IGNORECASE),
    re.compile(r"no\s+bugs?\s+(were\s+)?(found|discovered)", re.IGNORECASE),
    re.compile(r"repeated\s+content", re.IGNORECASE),
    re.compile(r"no\s+anti.?patterns?", re.IGNORECASE),
    re.compile(r"transcript\s+(does\s+not|doesn'?t)\s+include", re.IGNORECASE),
    re.compile(r"same\s+questions?\s+were\s+presented\s+twice", re.IGNORECASE),
]

_MIN_ONE_LINER_LEN = 20
_JACCARD_THRESHOLD = 0.35


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _extract_one_liner(draft: dict) -> str:
    """Extract one_liner string from a draft dict."""
    try:
        data = json.loads(draft.get("extracted_data") or "{}")
        return data.get("one_liner", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return ""


def filter_noise(
    drafts: list[dict],
    existing_one_liners: list[str],
) -> tuple[list[dict], list[dict]]:
    """Split drafts into (kept, dismissed) using dedup + regex noise filter.

    Checks in order:
    1. Empty one_liner
    2. Jaccard similarity > threshold vs any existing lesson or prior batch entry
    3. One_liner length < minimum
    4. Matches a noise regex pattern
    """
    kept: list[dict] = []
    dismissed: list[dict] = []
    seen_one_liners: list[str] = list(existing_one_liners)

    for draft in drafts:
        one_liner = _extract_one_liner(draft)

        if not one_liner:
            draft["_dismiss_reason"] = "empty one_liner"
            dismissed.append(draft)
            continue

        similar = any(jaccard_similarity(one_liner, seen) >= _JACCARD_THRESHOLD for seen in seen_one_liners)
        if similar:
            draft["_dismiss_reason"] = "duplicate (Jaccard)"
            dismissed.append(draft)
            continue

        if len(one_liner) < _MIN_ONE_LINER_LEN:
            draft["_dismiss_reason"] = f"too short ({len(one_liner)} chars)"
            dismissed.append(draft)
            continue

        if any(p.search(one_liner) for p in _NOISE_PATTERNS):
            draft["_dismiss_reason"] = "noise pattern match"
            dismissed.append(draft)
            continue

        seen_one_liners.append(one_liner)
        kept.append(draft)

    return kept, dismissed
