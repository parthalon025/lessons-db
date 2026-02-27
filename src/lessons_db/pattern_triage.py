"""Stage 3: Triage verified candidates — auto-approve or draft queue.

Auto-approve confidence >= threshold:
  - Insert as positive lesson with seeded reuse_count and tier
  - Record surfacing_event
  - Upsert into LanceDB

Below threshold:
  - Insert into capture_drafts (detection_source='cross_project_scan', confidence stored)

On rejection via reject_draft():
  - Embed snippet → suppression_vectors (prevents future false positives)
  - Mark draft as rejected
"""

import logging
from datetime import date

from lessons_db.db import (
    get_scan_state,
    insert_lesson,
)
from lessons_db.pattern_verify import VerifiedCandidate
from lessons_db.vectors import get_embedding, init_lance, upsert_lesson

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.85


def seed_reuse_count(source_repos: list[str]) -> int:
    """Reuse_count seeded from discovery count.

    Cross-project detection is retroactive — the pattern was already applied
    N times before capture. len(repos) - 1 reflects real evidence.
    """
    return max(0, len(source_repos) - 1)


def tier_from_reuse(reuse_count: int) -> str:
    """Derive positive entry tier from seeded reuse_count."""
    if reuse_count >= 3:
        return "standard"
    if reuse_count >= 2:
        return "proven"
    if reuse_count >= 1:
        return "tested"
    return "noticed"


def _get_threshold(conn) -> float:
    """Read auto_approve_threshold from scan_state. Default 0.85."""
    val = get_scan_state(conn, "auto_approve_threshold")
    try:
        return float(val) if val else DEFAULT_THRESHOLD
    except ValueError:
        return DEFAULT_THRESHOLD


def triage_candidate(
    candidate: VerifiedCandidate,
    conn,
    lance_dir: str | None = None,
) -> int | None:
    """Triage a verified candidate. Returns lesson_id if auto-approved, else None."""
    threshold = _get_threshold(conn)
    today = date.today().isoformat()

    if candidate.confidence >= threshold:
        reuse = seed_reuse_count(candidate.source_repos)
        tier = tier_from_reuse(reuse)

        try:
            lesson_id = insert_lesson(conn, {
                "title": candidate.snippet[:80],
                "one_liner": candidate.snippet[:120],
                "description": candidate.rationale,
                "polarity": "positive",
                "entry_type": "pattern",
                "tier": tier,
                "reuse_count": reuse,
                "category": "architecture-pattern",
                "created_date": today,
                "source": "cross_project_scan",
            })
        except Exception as e:
            logger.error("triage: insert_lesson failed: %s", e)
            return None

        # Record surfacing event
        conn.execute(
            "INSERT INTO surfacing_events "
            "(lesson_id, hook_point, context, outcome, timestamp) "
            "VALUES (?, 'cross_project_scan', ?, 'heeded', datetime('now'))",
            [lesson_id, ",".join(candidate.source_repos)]
        )
        conn.commit()

        # Upsert into LanceDB
        if lance_dir:
            try:
                lance_db = init_lance(lance_dir)
                upsert_lesson(lance_db, {
                    "lesson_id": lesson_id,
                    "text": candidate.snippet,
                    "tier": tier,
                    "cluster": "",
                    "scope": "",
                    "enforcement": "documentation",
                    "recurrence_count": 0,
                })
            except Exception as e:
                logger.warning("triage: LanceDB upsert failed for lesson %d: %s", lesson_id, e)

        logger.info(
            "triage: auto-approved lesson %d (tier=%s, reuse=%d, conf=%.2f)",
            lesson_id, tier, reuse, candidate.confidence
        )
        return lesson_id

    else:
        # Below threshold → draft queue
        conn.execute(
            "INSERT INTO capture_drafts "
            "(raw_content, extracted_data, status, created_date, source, "
            " detection_source, confidence) "
            "VALUES (?, ?, 'pending', ?, 'cross_project_scan', 'cross_project_scan', ?)",
            [
                candidate.snippet,
                candidate.rationale,
                today,
                candidate.confidence,
            ]
        )
        conn.commit()
        logger.debug("triage: queued draft (conf=%.2f)", candidate.confidence)
        return None


def reject_draft(
    draft_id: int,
    conn,
    lance_dir: str | None = None,
    reason: str | None = None,
) -> None:
    """Reject a pending draft and store as suppression vector.

    Rejected snippets suppress future candidates with >0.85 similarity,
    preventing the scanner from repeating the same false positive nightly.
    """
    row = conn.execute(
        "SELECT raw_content FROM capture_drafts WHERE id = ?", [draft_id]
    ).fetchone()
    if not row:
        logger.warning("reject_draft: draft %d not found", draft_id)
        return

    snippet = row["raw_content"]
    embedding_id = f"suppression-{draft_id}"

    conn.execute(
        "INSERT INTO suppression_vectors "
        "(embedding_id, rejected_snippet, rejection_date, rejection_reason) "
        "VALUES (?, ?, date('now'), ?)",
        [embedding_id, snippet, reason]
    )

    conn.execute(
        "UPDATE capture_drafts SET status = 'rejected' WHERE id = ?",
        [draft_id]
    )
    conn.commit()

    logger.info("reject_draft: draft %d rejected, suppression vector stored", draft_id)


def calibration_bands(conn) -> dict[float, dict]:
    """Return promotion stats grouped by ROUND(confidence, 1) band."""
    rows = conn.execute("""
        SELECT
            ROUND(confidence, 1) as band,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved
        FROM capture_drafts
        WHERE detection_source = 'cross_project_scan'
          AND confidence IS NOT NULL
          AND status IN ('approved', 'rejected')
        GROUP BY band
        ORDER BY band
    """).fetchall()

    return {
        row["band"]: {
            "total": row["total"],
            "approved": row["approved"],
            "promotion_rate": row["approved"] / row["total"] if row["total"] else 0.0,
        }
        for row in rows
    }


def should_adjust_threshold(conn) -> dict | None:
    """Propose threshold adjustment if 20+ outcomes exist and data supports it.

    Returns dict with proposed_threshold and rationale, or None if insufficient data.
    """
    bands = calibration_bands(conn)
    total_outcomes = sum(b["total"] for b in bands.values())

    if total_outcomes < 20:
        return None

    current = _get_threshold(conn)
    current_band = round(current, 1)

    if current_band not in bands:
        return None

    current_rate = bands[current_band]["promotion_rate"]
    candidate_band = round(current - 0.05, 2)
    candidate_band_key = round(candidate_band, 1)

    if candidate_band_key not in bands:
        return None

    candidate_rate = bands[candidate_band_key]["promotion_rate"]

    if candidate_rate >= current_rate:
        return {
            "current_threshold": current,
            "proposed_threshold": candidate_band,
            "current_rate": current_rate,
            "proposed_rate": candidate_rate,
            "rationale": (
                f"confidence {candidate_band_key:.1f}-{current_band:.1f}: "
                f"{candidate_rate:.0%} promoted (≥ current {current_rate:.0%}). "
                f"Lowering threshold to {candidate_band:.2f} increases recall."
            ),
        }

    return None
