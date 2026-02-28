"""Draft triage pipeline: noise filter + OpenAI batch review + verdict execution."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from lessons_db.capture import promote_draft
from lessons_db.db import insert_detection_pattern

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for OpenAI structured output
# ---------------------------------------------------------------------------


class DraftReview(BaseModel):
    id: int
    verdict: Literal["PROMOTE", "DISMISS"]
    confidence: int  # 1-5; only auto-promote if >= _CONFIDENCE_THRESHOLD
    reason: str
    improved_one_liner: str
    detection_pattern: str
    semgrep_rule: str


class ReviewBatch(BaseModel):
    reviews: list[DraftReview]


_CONFIDENCE_THRESHOLD = 4  # raise to filter out marginal promotions

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
You are a strict reviewer for a coding lessons-learned system.
For each draft, decide PROMOTE or DISMISS and assign a confidence score 1-5.

PROMOTE only if ALL of the following are true:
1. Names a specific API call, method name, config key, error message, or code construct — NOT a general principle
2. The mistake would NOT be caught by a basic code review
3. A developer could write a regex or Semgrep rule to detect it today
4. Not already covered by an existing lesson (see list below)

DISMISS if any of:
- Vague principle ("always use good practices", "ensure security", "be careful")
- Already covered by an existing lesson
- Not a coding mistake/anti-pattern (general advice, tool tips, workflow suggestions)
- Would be caught immediately by any code review

Confidence scale:
5 = Concrete, specific, novel — definitely promote
4 = Specific enough, worth promoting
3 = Borderline — default to DISMISS
1-2 = Clear DISMISS

Examples:

DISMISS (confidence 2): "Skill invocation should be conditional and context-dependent"
→ General workflow advice, not a code pattern, no regex possible

DISMISS (confidence 1): "Settings can be scoped per-project or globally for better control"
→ Documentation tip, not a coding mistake, too vague to detect

DISMISS (confidence 3): "Avoid using broad permissions unless necessary"
→ Principle without a specific code pattern to match

PROMOTE (confidence 5): "sqlite3 connection used as context manager does not close — call .close() explicitly"
→ Specific: sqlite3 module, context manager, .close() method. Detectable: regex for `with sqlite3.connect`

PROMOTE (confidence 5): "asyncio.create_task without storing the return value silently drops exceptions"
→ Specific: asyncio.create_task, return value. Detectable: regex for `create_task(` without assignment

PROMOTE (confidence 4): "Hardcoded test counts break when fixtures change — use len(results) not assert len == 5"
→ Specific: test assertions with literal counts. Detectable: regex for `assert len(` with integer literal

Existing lessons (do not promote duplicates):
{existing_titles}

Drafts to review:
{draft_lines}"""

_BATCH_SIZE = 20


_RETRY_BATCH_SIZE = 5  # sub-batch size when retrying a failed batch


def _call_openai_batch(
    client: object,
    model: str,
    drafts: list[dict],
    existing_titles: list[str],
) -> list[dict]:
    """Call OpenAI structured output for one batch. Returns list of verdict dicts."""

    draft_lines = "\n".join(f"[{d['id']}] {_extract_one_liner(d)}" for d in drafts)
    titles_block = "\n".join(f"- {t}" for t in existing_titles[:150])
    prompt = _REVIEW_PROMPT_TEMPLATE.format(
        existing_titles=titles_block or "(none yet)",
        draft_lines=draft_lines,
    )
    response = client.beta.chat.completions.parse(  # type: ignore[attr-defined]
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
        response_format=ReviewBatch,
    )
    if response.choices[0].finish_reason == "length":
        raise ValueError("Response truncated by max_tokens limit")
    parsed: ReviewBatch = response.choices[0].message.parsed
    verdicts = []
    for r in parsed.reviews:
        # Apply confidence gate: downgrade low-confidence PROMOTE to DISMISS
        verdict = r.verdict
        if verdict == "PROMOTE" and r.confidence < _CONFIDENCE_THRESHOLD:
            verdict = "DISMISS"
            _log.debug(
                "_call_openai_batch: draft %d demoted PROMOTE→DISMISS (confidence %d < %d)",
                r.id,
                r.confidence,
                _CONFIDENCE_THRESHOLD,
            )
        verdicts.append(
            {
                "id": r.id,
                "verdict": verdict,
                "confidence": r.confidence,
                "reason": r.reason,
                "improved_one_liner": r.improved_one_liner,
                "detection_pattern": r.detection_pattern,
                "semgrep_rule": r.semgrep_rule,
            }
        )
    return verdicts


def claude_review_batch(
    drafts: list[dict],
    existing_titles: list[str],
    api_key: str,
) -> list[dict]:
    """Send drafts to OpenAI for PROMOTE/DISMISS review with confidence gate.

    Uses structured outputs (Pydantic schema) to eliminate JSON parse failures.
    On batch failure, retries at sub-batch size _RETRY_BATCH_SIZE. On retry
    failure, marks drafts as ERROR.

    Args:
        drafts: List of draft dicts with 'id' and 'extracted_data' fields.
        existing_titles: One-liner strings of already-promoted lessons (duplicate guard).
        api_key: OpenAI API key.

    Returns:
        List of verdict dicts with keys: id, verdict, confidence, reason,
        improved_one_liner, detection_pattern, semgrep_rule.
    """
    import openai  # lazy import — only needed when calling the reviewer

    from lessons_db.config import OPENAI_REVIEW_MODEL

    client = openai.OpenAI(api_key=api_key)
    all_verdicts: list[dict] = []

    for i in range(0, len(drafts), _BATCH_SIZE):
        batch = drafts[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE

        try:
            all_verdicts.extend(_call_openai_batch(client, OPENAI_REVIEW_MODEL, batch, existing_titles))
        except Exception as exc:
            _log.warning("claude_review_batch: batch %d failed (%s) — retrying in sub-batches", batch_num, exc)
            # Retry at smaller sub-batch size to reduce JSON complexity
            any_retry_failed = False
            for j in range(0, len(batch), _RETRY_BATCH_SIZE):
                sub = batch[j : j + _RETRY_BATCH_SIZE]
                try:
                    all_verdicts.extend(_call_openai_batch(client, OPENAI_REVIEW_MODEL, sub, existing_titles))
                except Exception as sub_exc:
                    _log.warning(
                        "claude_review_batch: sub-batch %d.%d failed: %s",
                        batch_num,
                        j // _RETRY_BATCH_SIZE,
                        sub_exc,
                    )
                    any_retry_failed = True
                    for d in sub:
                        all_verdicts.append(
                            {
                                "id": d["id"],
                                "verdict": "ERROR",
                                "confidence": 0,
                                "reason": f"error: {sub_exc}",
                                "improved_one_liner": "",
                                "detection_pattern": "",
                                "semgrep_rule": "",
                            }
                        )
            if not any_retry_failed:
                _log.info("claude_review_batch: batch %d recovered via sub-batches", batch_num)

    return all_verdicts


def _apply_promote(conn: sqlite3.Connection, v: dict, today: str) -> dict | None:
    """Promote a single draft verdict. Returns log entry dict or None if promote_draft fails."""
    draft_id = v["id"]

    # Inject improved_one_liner into extracted_data before promoting
    row = conn.execute("SELECT extracted_data FROM capture_drafts WHERE id = ?", [draft_id]).fetchone()
    if not row:
        _log.warning("_apply_promote: draft %d not found in capture_drafts — skipping", draft_id)
        return None
    data = json.loads(row["extracted_data"] or "{}")
    if v.get("improved_one_liner"):
        data["improved_one_liner"] = v["improved_one_liner"]
    conn.execute(
        "UPDATE capture_drafts SET extracted_data = ? WHERE id = ?",
        [json.dumps(data), draft_id],
    )
    # No commit here — promote_draft commits its own transaction which includes
    # the extracted_data UPDATE above, keeping both changes in one atomic write.

    lesson_id = promote_draft(conn, draft_id)
    if not lesson_id:
        _log.warning("_apply_promote: promote_draft returned None for draft %d", draft_id)
        return None

    # Insert detection pattern if provided
    pattern = v.get("detection_pattern", "").strip()
    if pattern:
        try:
            insert_detection_pattern(
                conn,
                {
                    "lesson_id": lesson_id,
                    "pattern_type": "regex",
                    "regex": pattern,
                    "description": v.get("reason", ""),
                    "language": "any",
                },
            )
        except Exception as exc:
            _log.warning("_apply_promote: pattern insert failed for lesson %d: %s", lesson_id, exc)

    # Write Semgrep rule if provided
    rule_yaml = v.get("semgrep_rule", "").strip()
    if rule_yaml:
        _write_semgrep_rule(lesson_id, rule_yaml)

    return {
        "date": today,
        "draft_id": draft_id,
        "lesson_id": lesson_id,
        "verdict": "PROMOTE",
        "reason": v.get("reason", ""),
        "one_liner": v.get("improved_one_liner", ""),
    }


def execute_verdicts(
    conn: sqlite3.Connection,
    verdicts: list[dict],
    log_dir: Path,
) -> dict:
    """Apply PROMOTE/DISMISS verdicts. Returns summary dict."""
    promoted = 0
    dismissed = 0
    errors = 0
    log_entries: list[dict] = []
    today = date.today().isoformat()

    for v in verdicts:
        draft_id = v["id"]
        verdict = v.get("verdict", "DISMISS")

        if verdict == "PROMOTE":
            entry = _apply_promote(conn, v, today)
            if entry:
                promoted += 1
                log_entries.append(entry)
            else:
                errors += 1
                # Mark the draft so the nightly loop doesn't retry it forever.
                conn.execute(
                    "UPDATE capture_drafts SET status='promote_failed' WHERE id=?",
                    [draft_id],
                )
                conn.commit()
                log_entries.append(
                    {
                        "date": today,
                        "draft_id": draft_id,
                        "lesson_id": None,
                        "verdict": "PROMOTE_FAILED",
                        "reason": "promote_draft returned None",
                        "one_liner": "",
                    }
                )
        else:
            conn.execute(
                "UPDATE capture_drafts SET status = 'dismissed' WHERE id = ?",
                [draft_id],
            )
            conn.commit()
            dismissed += 1
            log_entries.append(
                {
                    "date": today,
                    "draft_id": draft_id,
                    "lesson_id": None,
                    "verdict": "DISMISS",
                    "reason": v.get("reason", ""),
                    "one_liner": "",
                }
            )

    # Write JSONL log — single write minimises crash window for append mode.
    log_dir.mkdir(parents=True, exist_ok=True)
    if log_entries:
        log_path = log_dir / f"triage-{today}.jsonl"
        content = "\n".join(json.dumps(e) for e in log_entries) + "\n"
        with log_path.open("a") as f:
            f.write(content)

    _log.info("execute_verdicts: promoted=%d dismissed=%d errors=%d", promoted, dismissed, errors)
    return {"promoted": promoted, "dismissed": dismissed, "errors": errors}


def _write_semgrep_rule(lesson_id: int, rule_yaml: str) -> None:
    """Write a Semgrep rule YAML to the rules directory."""
    try:
        from lessons_db.config import DATA_DIR

        rules_dir = DATA_DIR / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        rule_path = rules_dir / f"lesson-{lesson_id}.yaml"
        rule_path.write_text(rule_yaml)
        _log.debug("_write_semgrep_rule: wrote %s", rule_path)
    except Exception as exc:
        _log.warning("_write_semgrep_rule: failed for lesson %d: %s", lesson_id, exc)
