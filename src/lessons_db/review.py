"""Draft triage pipeline: noise filter + Claude batch review + verdict execution."""

from __future__ import annotations

import json
import logging
import re

_log = logging.getLogger(__name__)

# Noise patterns — one_liners matching these are auto-dismissed
_NOISE_PATTERNS = [
    re.compile(r"no\s+(coding\s+)?mistakes?\s+were\s+(found|discovered)", re.IGNORECASE),
    re.compile(r"no\s+bugs?\s+", re.IGNORECASE),
    re.compile(r"repeated\s+content", re.IGNORECASE),
    re.compile(r"no\s+anti.?patterns?", re.IGNORECASE),
    re.compile(r"transcript\s+(does\s+not|doesn'?t)\s+include", re.IGNORECASE),
    re.compile(r"same\s+questions?\s+were\s+presented\s+twice", re.IGNORECASE),
]

_MIN_ONE_LINER_LEN = 20
_JACCARD_THRESHOLD = 0.85


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings.

    Tokens are lowercased and stripped of non-alphanumeric characters
    so that 'close()' and 'close' are treated as the same token.
    """

    def tokenize(s: str) -> set[str]:
        return {cleaned for t in s.lower().split() if (cleaned := re.sub(r"[^a-z0-9]", "", t))}

    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
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
            draft = {**draft, "_dismiss_reason": "empty one_liner"}
            dismissed.append(draft)
            continue

        similar = any(jaccard_similarity(one_liner, seen) >= _JACCARD_THRESHOLD for seen in seen_one_liners)
        if similar:
            draft = {**draft, "_dismiss_reason": "duplicate (Jaccard)"}
            dismissed.append(draft)
            continue

        if len(one_liner) < _MIN_ONE_LINER_LEN:
            draft = {**draft, "_dismiss_reason": f"too short ({len(one_liner)} chars)"}
            dismissed.append(draft)
            continue

        if any(p.search(one_liner) for p in _NOISE_PATTERNS):
            draft = {**draft, "_dismiss_reason": "noise pattern match"}
            dismissed.append(draft)
            continue

        seen_one_liners.append(one_liner)
        kept.append(draft)

    return kept, dismissed


_REVIEW_PROMPT_TEMPLATE = """\
You are reviewing draft lessons for a coding lessons-learned system.
For each draft, decide PROMOTE or DISMISS.

Existing lessons (do not promote duplicates of these):
{existing_titles}

Drafts to review:
{draft_lines}

Criteria for PROMOTE:
- Specific: names a concrete pattern, not a general principle
- Actionable: clear do/don't a developer can follow
- Prevents recurrence: would catch this mistake if checked automatically
- Novel: not already in the existing lessons list above

Return ONLY valid JSON, no other text:
{{
  "reviews": [
    {{
      "id": <integer draft id>,
      "verdict": "PROMOTE" or "DISMISS",
      "reason": "<one sentence>",
      "improved_one_liner": "<cleaned wording if PROMOTE, else empty string>",
      "detection_pattern": "<Python regex string for code matching if PROMOTE, else empty string>",
      "semgrep_rule": "<YAML Semgrep rule text if syntactic pattern possible, else empty string>"
    }}
  ]
}}"""

_BATCH_SIZE = 20


def claude_review_batch(
    drafts: list[dict],
    existing_titles: list[str],
    api_key: str,
) -> list[dict]:
    """Send drafts to Claude haiku for PROMOTE/DISMISS review.

    Processes in batches of _BATCH_SIZE. On API error, marks all drafts
    in that batch as DISMISS with reason='error: <msg>'.

    Args:
        drafts: List of draft dicts with 'id' and 'extracted_data' fields.
        existing_titles: One-liner strings of already-promoted lessons (duplicate guard).
        api_key: Anthropic API key.

    Returns:
        List of verdict dicts with keys: id, verdict, reason, improved_one_liner,
        detection_pattern, semgrep_rule.
    """
    import anthropic  # lazy import — only needed when calling Claude

    from lessons_db.config import CLAUDE_REVIEW_MODEL

    client = anthropic.Anthropic(api_key=api_key)
    all_verdicts: list[dict] = []

    for i in range(0, len(drafts), _BATCH_SIZE):
        batch = drafts[i : i + _BATCH_SIZE]
        draft_lines = "\n".join(f"[{d['id']}] {_extract_one_liner(d)}" for d in batch)
        titles_block = "\n".join(f"- {t}" for t in existing_titles[:150])
        prompt = _REVIEW_PROMPT_TEMPLATE.format(
            existing_titles=titles_block or "(none yet)",
            draft_lines=draft_lines,
        )

        try:
            msg = client.messages.create(
                model=CLAUDE_REVIEW_MODEL,
                max_tokens=8192,
                messages=[{"role": "user", "content": prompt}],
            )
            if msg.stop_reason == "max_tokens":
                raise ValueError("Response truncated by max_tokens limit")
            raw = msg.content[0].text.strip()
            # Extract JSON object — tolerates preamble/postamble from model
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON object in response: {raw[:200]}")
            data = json.loads(json_match.group())
            all_verdicts.extend(data.get("reviews", []))
        except Exception as exc:
            _log.warning("claude_review_batch: batch %d failed: %s", i // _BATCH_SIZE, exc)
            for d in batch:
                all_verdicts.append(
                    {
                        "id": d["id"],
                        "verdict": "ERROR",
                        "reason": f"error: {exc}",
                        "improved_one_liner": "",
                        "detection_pattern": "",
                        "semgrep_rule": "",
                    }
                )

    return all_verdicts
