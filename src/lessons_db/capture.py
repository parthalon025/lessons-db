"""Positive knowledge capture — manual interactive and auto-detect from artifacts."""

import json
from datetime import date
from pathlib import Path

import requests

from lessons_db.config import (
    ANALYSIS_MODEL,
    OLLAMA_QUEUE_URL,
    QUALITY_MIN_SCORE,
)
from lessons_db.db import init_db, insert_lesson


def score_one_liner(text: str) -> int:
    """Ask Ollama to score one-liner specificity 1-5. Returns 3 on any error."""
    try:
        r = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
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
        score = int(r.json().get("response", "3").strip())
        return max(1, min(5, score))
    except Exception:
        return 3


def capture_from_design_doc(doc_path: Path,
                             conn) -> list[dict]:
    """Extract positive patterns from a design doc. Sends to capture_drafts (quarantine).

    Returns list of extracted entry dicts. Does NOT create live lessons."""
    content = doc_path.read_text()[:4000]

    try:
        r = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
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
            timeout=60,
        )
        data = json.loads(r.json().get("response", "{}"))
        entries = data.get("entries", [])
    except Exception:
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
    except Exception:
        conn.rollback()
        return []
    return entries


def promote_draft(conn, draft_id: int) -> int | None:
    """Promote a pending draft to a live positive lesson. Returns lesson_id."""
    row = conn.execute(
        "SELECT extracted_data FROM capture_drafts WHERE id = ? AND status = 'pending'",
        [draft_id],
    ).fetchone()
    if not row:
        return None

    data = json.loads(row["extracted_data"])
    lesson_id = insert_lesson(conn, {
        "title": data.get("one_liner", "Untitled pattern"),
        "one_liner": data.get("one_liner", ""),
        "description": data.get("why", ""),
        "polarity": "positive",
        "entry_type": "pattern",
        "category": data.get("category", "architecture-pattern"),
        "tier": "noticed",
        "source": "auto_design_doc",
        "created_date": date.today().isoformat(),
    })
    conn.execute(
        "UPDATE capture_drafts SET status = 'approved' WHERE id = ?",
        [draft_id],
    )
    conn.commit()
    return lesson_id


def list_drafts(conn) -> list[dict]:
    """Return all pending capture drafts."""
    rows = conn.execute(
        "SELECT id, extracted_data, status, created_date, source "
        "FROM capture_drafts WHERE status = 'pending' ORDER BY created_date DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def capture_positive_manual(conn, one_liner: str, why: str, category: str,
                              when_to_apply: str = "", when_not_to_apply: str = "") -> int | None:
    """Capture a positive knowledge entry manually. Runs quality gate first.

    Returns lesson_id if quality passes, None if rejected."""
    score = score_one_liner(one_liner)
    if score < QUALITY_MIN_SCORE:
        return None

    return insert_lesson(conn, {
        "title": one_liner,
        "one_liner": one_liner,
        "description": why,
        "polarity": "positive",
        "entry_type": "pattern",
        "category": category,
        "tier": "noticed",
        "source": "manual",
        "created_date": date.today().isoformat(),
    })
