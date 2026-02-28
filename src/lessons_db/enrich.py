"""Ollama-powered enrichment: backfill false_assumption, detection_pattern, invariant."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time

import requests

from lessons_db.config import ANALYSIS_MODEL, OLLAMA_QUEUE_URL

_log = logging.getLogger(__name__)

# When run directly (not as a queue job) → route through queue proxy for serialization.
# When run AS a queue job (--ollama-url passed explicitly) → direct Ollama at 11434.
# This avoids self-deadlock: queue job → proxy → "busy" → timeout.
DEFAULT_OLLAMA_URL = OLLAMA_QUEUE_URL

_PROMPT_TEMPLATE = """\
You are analyzing a software engineering lesson to extract three structured fields.

LESSON TITLE: {title}
LESSON SUMMARY: {one_liner}
LESSON DESCRIPTION:
{description}

Extract exactly three fields based ONLY on what is described above.
Do not invent API names, method calls, or concepts not present in the text above.

"false_assumption": One sentence beginning with "AI assumes" — the specific incorrect \
belief an AI code generator holds when producing code with this bug. Must reference \
the exact API, method, or behavior named in the lesson. Be precise, not generic.

"detection_pattern": 1–5 lines of code showing the anti-pattern being made. Format as:
<language>
<code>
Use only real APIs explicitly mentioned above. Default to Python unless the lesson \
explicitly describes another language.

"invariant": One positive enforcement rule (one sentence). Begin with Always / Ensure \
/ Set / Pass / Return. No negations ("don't", "never", "avoid").

Return ONLY a JSON object with exactly these three keys. No explanation, no markdown.\
"""

_REQUIRED_KEYS = {"false_assumption", "detection_pattern", "invariant"}


def enrich_lesson(
    conn: sqlite3.Connection,
    lesson_id: int,
    title: str,
    description: str,
    one_liner: str,
    model: str,
    ollama_url: str,
    dry_run: bool = False,
) -> dict | None:
    """Call Ollama to generate the three why-capture fields for one lesson.

    When run standalone (not as a queue job), pass ollama_url=OLLAMA_QUEUE_URL
    so each call is serialized through the proxy. When run as a queue subprocess,
    pass ollama_url=OLLAMA_ANALYSIS_URL (direct) to avoid self-deadlock.

    Returns the generated dict on success, None on failure.
    Does not write to DB when dry_run=True.
    """
    from lessons_db.db import update_lesson

    prompt = _PROMPT_TEMPLATE.format(
        title=title,
        one_liner=one_liner or "(none)",
        description=(description or "")[:2000],
    )

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                # format:json ensures clean JSON from instruction-following models
                # (qwen2.5:7b, gemma3, etc.). For thinking models (qwen3:*),
                # format:json suppresses the reasoning block and produces empty {}.
                # The <think> stripping below handles thinking model preamble if
                # format:json is not set. Default model qwen2.5:7b works correctly.
                "format": "json",
            },
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        _log.warning("enrich_lesson: HTTP error for lesson %d: %s", lesson_id, exc)
        return None

    raw = resp.json()
    content = raw.get("response", "")
    if not content:
        # Ollama occasionally returns empty response on rapid sequential calls.
        # Retry once with a brief backoff before giving up.
        _log.debug("enrich_lesson: empty response for lesson %d, retrying in 3s", lesson_id)
        time.sleep(3)
        try:
            resp2 = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
                timeout=120,
            )
            resp2.raise_for_status()
            content = resp2.json().get("response", "")
        except requests.RequestException:
            pass
    if not content:
        _log.warning("enrich_lesson: empty content after retry for lesson %d", lesson_id)
        return None

    # qwen3 and other thinking models emit <think>...</think> before JSON output;
    # strip those blocks before json.loads().
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    # Handle responses wrapped in markdown fences
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        _log.warning("enrich_lesson: JSON parse failed for lesson %d: %s", lesson_id, exc)
        return None

    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        _log.warning(
            "enrich_lesson: missing keys %s for lesson %d (got: %s)",
            missing,
            lesson_id,
            list(data.keys()),
        )
        return None

    result = {
        "false_assumption": str(data["false_assumption"]).strip(),
        "detection_pattern": str(data["detection_pattern"]).strip(),
        "invariant": str(data["invariant"]).strip(),
    }

    if not dry_run:
        update_lesson(conn, lesson_id, result)

    return result


def backfill_lessons(
    conn: sqlite3.Connection,
    model: str | None = None,
    ollama_url: str | None = None,
    batch: int | None = None,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    """Backfill why-capture fields for all lessons missing false_assumption.

    Returns (enriched_count, skipped_count, error_count).
    skipped = lessons already enriched (had false_assumption before this run).
    """
    resolved_model = model or ANALYSIS_MODEL
    resolved_url = ollama_url or DEFAULT_OLLAMA_URL

    # Count already-enriched before we start (for accurate skipped reporting)
    already_done = conn.execute(
        "SELECT COUNT(*) FROM lessons WHERE false_assumption IS NOT NULL AND false_assumption != ''"
    ).fetchone()[0]

    sql = """
        SELECT id, title, one_liner, description
        FROM lessons
        WHERE false_assumption IS NULL OR false_assumption = ''
        ORDER BY id
    """
    if batch is not None:
        sql += f" LIMIT {int(batch)}"

    rows = conn.execute(sql).fetchall()

    enriched = 0
    errors = 0

    for i, row in enumerate(rows):
        if i > 0:
            # Brief pause between sequential calls; prevents Ollama from
            # returning empty responses under rapid-fire requests.
            time.sleep(1)
        result = enrich_lesson(
            conn=conn,
            lesson_id=row["id"],
            title=row["title"] or "",
            description=row["description"] or "",
            one_liner=row["one_liner"] or "",
            model=resolved_model,
            ollama_url=resolved_url,
            dry_run=dry_run,
        )
        if result is None:
            errors += 1
        else:
            enriched += 1

    return enriched, already_done, errors
