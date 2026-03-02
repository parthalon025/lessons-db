"""Positive knowledge capture — manual interactive and auto-detect from artifacts."""

import json
import logging
import re
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import requests

from lessons_db.config import (
    ANALYSIS_MODEL,
    OLLAMA_ANALYSIS_URL,
    QUALITY_MIN_SCORE,
)
from lessons_db.db import insert_lesson

_log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think(text: str) -> str:
    """Strip deepseek-r1 <think>...</think> blocks before JSON parsing."""
    return _THINK_RE.sub("", text).strip()


def score_one_liner(text: str) -> int:
    """Ask Ollama to score one-liner specificity 1-5. Returns 3 on any error."""
    try:
        r = requests.post(
            f"{OLLAMA_ANALYSIS_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    f"Score this knowledge entry one-liner for specificity "
                    f"(1=vague, 5=precise and actionable):\n'{text}'\n\n"
                    "Respond with only a single integer 1-5."
                ),
                "stream": False,
            },
            timeout=30,
        )
        score = int(_strip_think(r.json().get("response", "3")))
        return max(1, min(5, score))
    except Exception as e:
        _log.warning("score_one_liner failed: %s", e)
        return 3


def capture_from_design_doc(doc_path: Path, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Extract positive patterns from a design doc. Sends to capture_drafts (quarantine).

    Returns list of extracted entry dicts. Does NOT create live lessons."""
    content = doc_path.read_text()[:4000]

    try:
        r = requests.post(
            f"{OLLAMA_ANALYSIS_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    "Extract positive knowledge patterns (what worked well, effective approaches) "
                    "from this document. Return JSON with this exact structure: "
                    '{"entries": [{"one_liner": "...", "why": "...", "category": "..."}]}\n\n'
                    f"Document:\n{content}"
                ),
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        data = json.loads(_strip_think(r.json().get("response", "{}")))
        entries = data.get("entries", [])
    except Exception as e:
        _log.warning("capture_from_design_doc Ollama call failed: %s", e)
        return []

    if not entries:
        return []

    try:
        for entry in entries:
            conn.execute(
                "INSERT INTO capture_drafts "
                "(raw_content, extracted_data, status, created_date, source) "
                "VALUES (?, ?, 'pending', ?, 'auto_design_doc')",
                [content[:500], json.dumps(entry), date.today().isoformat()],
            )
        conn.commit()
    except Exception as exc:
        _log.warning("capture_from_design_doc: DB insert failed: %s", exc)
        conn.rollback()
        return []
    _log.debug("capture_from_design_doc: created %d drafts from %s", len(entries), doc_path.name)
    return cast(list[dict[str, Any]], entries)


_POLARITY_BY_SOURCE = {
    "auto_transcript": ("negative", "lesson"),
    "auto_transcript_positive": ("positive", "pattern"),
    "auto_diff": ("negative", "lesson"),
    "auto_design_doc": ("positive", "pattern"),
}


def promote_draft(conn: sqlite3.Connection, draft_id: int) -> int | None:
    """Promote a pending draft to a live lesson. Returns lesson_id."""
    row = conn.execute(
        "SELECT extracted_data, source FROM capture_drafts WHERE id = ? AND status = 'pending'",
        [draft_id],
    ).fetchone()
    if not row:
        return None

    data = json.loads(row["extracted_data"])
    source = row["source"] or "auto_transcript"
    polarity, entry_type = _POLARITY_BY_SOURCE.get(source, ("negative", "lesson"))

    lesson_id = insert_lesson(
        conn,
        {
            "title": data.get("improved_one_liner") or data.get("one_liner", "Untitled"),
            "one_liner": data.get("improved_one_liner") or data.get("one_liner", ""),
            "description": data.get("why", ""),
            "polarity": polarity,
            "entry_type": entry_type,
            "category": data.get("category", "architecture-pattern"),
            "tier": data.get("tier", "noticed"),
            "source": source,
            "created_date": date.today().isoformat(),
        },
    )
    conn.execute(
        "UPDATE capture_drafts SET status = 'approved' WHERE id = ?",
        [draft_id],
    )
    conn.commit()
    return lesson_id


def list_drafts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all pending capture drafts."""
    rows = conn.execute(
        "SELECT id, extracted_data, status, created_date, source "
        "FROM capture_drafts WHERE status = 'pending' ORDER BY created_date DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def capture_from_transcript(
    transcript: str, conn: sqlite3.Connection, polarity: str = "negative"
) -> list[dict[str, Any]]:
    """Extract lessons from a session transcript. Drafts go to capture_drafts.

    polarity="negative" (default) — extracts bugs, anti-patterns, mistakes.
    polarity="positive" — extracts effective approaches and good patterns.

    Returns list of extracted lesson dicts. Returns [] on failure or empty transcript."""
    if not transcript or len(transcript.strip()) < 100:
        return []

    excerpt = transcript[-6000:]  # last 6000 chars — most recent context

    if polarity == "positive":
        prompt = (
            "Analyze this Claude Code session transcript. "
            "Extract effective approaches, good patterns, and techniques that worked well. "
            "Focus on what was done RIGHT — design decisions, testing strategies, debugging approaches, "
            "architectural choices that paid off. "
            "Return JSON: "
            '{"lessons": [{"one_liner": "...", "cluster": "A-F or empty", "tier": "observation|insight|lesson|lesson_learned"}]}\n\n'
            f"Transcript excerpt:\n{excerpt}"
        )
        source = "auto_transcript_positive"
    else:
        prompt = (
            "Analyze this Claude Code session transcript. "
            "Extract any coding mistakes, bugs, or anti-patterns that were discovered and fixed. "
            "Return JSON: "
            '{"lessons": [{"one_liner": "...", "cluster": "A-F or empty", "tier": "observation|insight|lesson|lesson_learned"}]}\n\n'
            f"Transcript excerpt:\n{excerpt}"
        )
        source = "auto_transcript"

    try:
        r = requests.post(
            f"{OLLAMA_ANALYSIS_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        r.raise_for_status()
        data = json.loads(_strip_think(r.json().get("response", "{}")))
        lessons = data.get("lessons", [])
    except Exception as e:
        _log.warning("capture_from_transcript Ollama call failed: %s", e)
        raise

    if not lessons:
        return []

    inserted = []
    try:
        for entry in lessons:
            one_liner = entry.get("one_liner", "")
            score = score_one_liner(one_liner)
            if score < QUALITY_MIN_SCORE:
                _log.debug(
                    "capture_from_transcript: skipping low-quality draft (score %d): %s",
                    score,
                    one_liner[:80],
                )
                continue
            conn.execute(
                "INSERT INTO capture_drafts "
                "(raw_content, extracted_data, status, created_date, source) "
                "VALUES (?, ?, 'pending', ?, ?)",
                [excerpt[:500], json.dumps(entry), date.today().isoformat(), source],
            )
            inserted.append(entry)
        conn.commit()
    except Exception as e:
        _log.warning("capture_from_transcript DB insert failed: %s", e)
        conn.rollback()
        raise

    _log.debug("capture_from_transcript: created %d drafts (%d filtered)", len(inserted), len(lessons) - len(inserted))
    return cast(list[dict[str, Any]], inserted)


def capture_from_diff(diff_text: str, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Extract negative lessons from a git diff. Drafts go to capture_drafts.

    Returns list of extracted lesson dicts. Returns [] on empty diff."""
    if not diff_text or len(diff_text.strip()) < 20:
        return []

    excerpt = diff_text[:4000]

    try:
        r = requests.post(
            f"{OLLAMA_ANALYSIS_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    "Analyze this git diff. Look for anti-patterns in REMOVED lines (prefixed with -) "
                    "that were fixed in ADDED lines (prefixed with +). "
                    "Extract any coding lessons. "
                    "Return JSON: "
                    '{"lessons": [{"one_liner": "...", "cluster": "A-F or empty", "tier": "observation|insight|lesson|lesson_learned"}]}\n\n'
                    f"Diff:\n{excerpt}"
                ),
                "stream": False,
                "format": "json",
            },
            timeout=120,
        )
        r.raise_for_status()
        data = json.loads(_strip_think(r.json().get("response", "{}")))
        lessons = data.get("lessons", [])
    except Exception as e:
        _log.warning("capture_from_diff Ollama call failed: %s", e)
        raise

    if not lessons:
        return []

    inserted = []
    try:
        for entry in lessons:
            one_liner = entry.get("one_liner", "")
            score = score_one_liner(one_liner)
            if score < QUALITY_MIN_SCORE:
                _log.debug(
                    "capture_from_diff: skipping low-quality draft (score %d): %s",
                    score,
                    one_liner[:80],
                )
                continue
            conn.execute(
                "INSERT INTO capture_drafts "
                "(raw_content, extracted_data, status, created_date, source) "
                "VALUES (?, ?, 'pending', ?, 'auto_diff')",
                [excerpt[:500], json.dumps(entry), date.today().isoformat()],
            )
            inserted.append(entry)
        conn.commit()
    except Exception as e:
        _log.warning("capture_from_diff DB insert failed: %s", e)
        conn.rollback()
        raise

    _log.debug("capture_from_diff: created %d drafts (%d filtered)", len(inserted), len(lessons) - len(inserted))
    return cast(list[dict[str, Any]], inserted)


def capture_positive_manual(
    conn: sqlite3.Connection,
    one_liner: str,
    why: str,
    category: str,
    when_to_apply: str = "",
    when_not_to_apply: str = "",
) -> int | None:
    """Capture a positive knowledge entry manually. Runs quality gate first.

    Returns lesson_id if quality passes, None if rejected."""
    score = score_one_liner(one_liner)
    if score < QUALITY_MIN_SCORE:
        _log.warning("capture_positive_manual: score %d below threshold %d", score, QUALITY_MIN_SCORE)
        return None

    return insert_lesson(
        conn,
        {
            "title": one_liner,
            "one_liner": one_liner,
            "description": why,
            "polarity": "positive",
            "entry_type": "pattern",
            "category": category,
            "tier": "noticed",
            "source": "manual",
            "created_date": date.today().isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Win detection — aggressive positive capture from session signals
# ---------------------------------------------------------------------------

#: Minimum fraction of heeded outcomes to declare a "heeded lessons" win.
_HEED_THRESHOLD = 0.7

#: Minimum number of surfacing events to evaluate heed rate.
_MIN_EVENTS_FOR_HEED = 1


def detect_wins(
    conn: sqlite3.Connection,
    lookback_hours: int = 4,
) -> list[dict[str, Any]]:
    """Detect positive session signals worth capturing.

    Checks three independent heuristics over the lookback window:

    1. **All-heeded** — surfacing events in the window had a heed rate >= 70%.
       Indicates the developer internalized the surfaced lessons.
    2. **No anti-pattern hits** — session had surfacing events but zero
       ``dismissed`` or ``recurrence`` outcomes (clean session).
    3. **Positive pattern reused** — a positive-polarity lesson was surfaced
       and heeded, meaning an established good pattern was actively applied.

    Returns a list of win dicts, each with ``win_type``, ``detail``, and
    optional ``lesson_ids``.  Empty list means no wins detected.
    """
    cutoff = (datetime.now(UTC) - timedelta(hours=lookback_hours)).isoformat()

    events = conn.execute(
        "SELECT se.id, se.lesson_id, se.outcome, l.polarity "
        "FROM surfacing_events se "
        "JOIN lessons l ON l.id = se.lesson_id "
        "WHERE se.timestamp >= ? "
        "ORDER BY se.timestamp DESC",
        [cutoff],
    ).fetchall()

    if not events:
        return []

    wins: list[dict[str, Any]] = []

    # Partition outcomes
    total = len(events)
    heeded = [e for e in events if e["outcome"] == "heeded"]
    negative = [e for e in events if e["outcome"] in ("dismissed", "recurrence")]
    positive_heeded = [e for e in events if e["polarity"] == "positive" and e["outcome"] == "heeded"]

    # --- Win 1: High heed rate ---
    if total >= _MIN_EVENTS_FOR_HEED:
        heed_rate = len(heeded) / total
        if heed_rate >= _HEED_THRESHOLD:
            wins.append(
                {
                    "win_type": "all_heeded",
                    "detail": (f"{len(heeded)}/{total} surfaced lessons heeded " f"({heed_rate:.0%} heed rate)"),
                    "lesson_ids": sorted({e["lesson_id"] for e in heeded}),
                }
            )

    # --- Win 2: Clean session (no anti-pattern hits) ---
    if total >= _MIN_EVENTS_FOR_HEED and len(negative) == 0:
        wins.append(
            {
                "win_type": "no_anti_pattern_hits",
                "detail": (f"{total} lessons surfaced, zero anti-pattern violations"),
                "lesson_ids": sorted({e["lesson_id"] for e in events}),
            }
        )

    # --- Win 3: Positive pattern reused ---
    if positive_heeded:
        wins.append(
            {
                "win_type": "positive_pattern_reused",
                "detail": (f"{len(positive_heeded)} positive pattern(s) actively applied"),
                "lesson_ids": sorted({e["lesson_id"] for e in positive_heeded}),
            }
        )

    return wins
