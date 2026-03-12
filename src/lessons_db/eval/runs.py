"""Eval run history: record aggregate metrics per judge run, query trends."""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime

_log = logging.getLogger(__name__)


def record_eval_run(
    conn: sqlite3.Connection,
    variant: str,
    f1: float,
    recall: float,
    precision: float,
    *,
    auc: float | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    prompt_id: str | None = None,
    results_file: str | None = None,
    notes: str | None = None,
) -> int:
    """Insert one row into eval_runs and return the new row id.

    Called by run_eval_judge after scoring each variant so that future APO
    runs can inspect the score trend when proposing prompt improvements.
    """
    run_date = datetime.now(UTC).isoformat()
    cur = conn.execute(
        """
        INSERT INTO eval_runs
            (run_date, variant, f1, recall, precision, auc,
             model, judge_model, prompt_id, results_file, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (run_date, variant, f1, recall, precision, auc, model, judge_model, prompt_id, results_file, notes),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_eval_history(
    conn: sqlite3.Connection,
    variant: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Return eval run rows sorted newest-first.

    Args:
        variant: If set, filter to this variant only.
        limit: Maximum rows to return (default 20).

    Returns:
        List of dicts with keys matching eval_runs columns.
    """
    if variant is not None:
        rows = conn.execute(
            """
            SELECT id, run_date, variant, f1, recall, precision, auc,
                   model, judge_model, prompt_id, results_file, notes
            FROM eval_runs
            WHERE variant = ?
            ORDER BY run_date DESC
            LIMIT ?
            """,
            (variant, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, run_date, variant, f1, recall, precision, auc,
                   model, judge_model, prompt_id, results_file, notes
            FROM eval_runs
            ORDER BY run_date DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
