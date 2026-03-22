"""FastAPI server for lessons-db — port 7685.

Eliminates subprocess cold start (~300ms) per Express API call.
All routes are thin wrappers over existing DB/analyzer functions.
"""

import logging
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from lessons_db.config import LANCE_DIR, SQLITE_PATH
from lessons_db.db import get_scan_state, init_db
from lessons_db.gap_analyzer import get_gap_report

_log = logging.getLogger(__name__)


class StatusUpdate(BaseModel):
    status: str


class FixStatusUpdate(BaseModel):
    status: str
    github_issue_url: str | None = None


def create_app(  # noqa: C901, PLR0915
    db_path: Path | None = None,
    lance_dir: Path | None = None,
) -> FastAPI:
    """Factory for testability."""
    _db_path = db_path or SQLITE_PATH
    _lance_dir = lance_dir or LANCE_DIR

    app = FastAPI(title="lessons-db API", version="1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:7684", "http://127.0.0.1:7684"],
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["*"],
    )

    def get_conn() -> sqlite3.Connection:
        return init_db(_db_path)

    _SORT_MAP = {
        "id_asc": "id ASC",
        "severity": "tier ASC, id DESC",
        "recurrence_count": "recurrence_count DESC, id DESC",
    }
    _SORT_DEFAULT = "id DESC"

    @app.get("/api/lessons")
    def list_lessons(
        q: str | None = None,
        category: str | None = None,
        tier: str | None = None,
        polarity: str | None = None,
        sort: str | None = None,
        limit: int = Query(50, le=200),
        offset: int = 0,
    ) -> dict:
        conn = get_conn()
        try:
            clauses = []
            params: list[Any] = []
            if q:
                clauses.append("(title LIKE ? OR one_liner LIKE ?)")
                params += [f"%{q}%", f"%{q}%"]
            if category:
                clauses.append("category = ?")
                params.append(category)
            if tier:
                clauses.append("tier = ?")
                params.append(tier)
            if polarity:
                clauses.append("polarity = ?")
                params.append(polarity)
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            order = _SORT_MAP.get(sort or "", _SORT_DEFAULT)
            rows = conn.execute(
                f"SELECT * FROM lessons {where} ORDER BY {order} LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            total = conn.execute(f"SELECT COUNT(*) FROM lessons {where}", params).fetchone()[0]
            return {"lessons": [dict(r) for r in rows], "total": total, "offset": offset}
        finally:
            conn.close()

    @app.get("/api/lessons/stats")
    def lessons_stats() -> dict:
        conn = get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
            positive = conn.execute("SELECT COUNT(*) FROM lessons WHERE polarity='positive'").fetchone()[0]
            negative = conn.execute("SELECT COUNT(*) FROM lessons WHERE polarity='negative'").fetchone()[0]
            tier_rows = conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM lessons GROUP BY tier ORDER BY cnt DESC LIMIT 5"
            ).fetchall()
            drafts = conn.execute("SELECT COUNT(*) FROM capture_drafts WHERE status='pending'").fetchone()[0]
            return {
                "total": total,
                "positive": positive,
                "negative": negative,
                "pending_drafts": drafts,
                "top_tiers": [{"tier": r["tier"], "count": r["cnt"]} for r in tier_rows],
            }
        finally:
            conn.close()

    @app.get("/api/lessons/categories")
    def lessons_categories() -> list:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT category FROM lessons WHERE category IS NOT NULL AND category != '' ORDER BY category"
            ).fetchall()
            return [r["category"] for r in rows]
        finally:
            conn.close()

    @app.get("/api/lessons/{lesson_id}")
    def get_lesson(lesson_id: int) -> dict:
        conn = get_conn()
        try:
            row = conn.execute("SELECT * FROM lessons WHERE id=?", (lesson_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="lesson not found")
            return dict(row)
        finally:
            conn.close()

    @app.get("/api/gaps")
    def list_gaps() -> list:
        conn = get_conn()
        try:
            return get_gap_report(conn)
        finally:
            conn.close()

    @app.get("/api/mining/history")
    def mining_history(limit: int = Query(20, le=500)) -> list:
        conn = get_conn()
        try:
            rows = conn.execute("SELECT * FROM mining_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @app.get("/api/mining/repos")
    def mining_repos(limit: int = 50) -> list:
        conn = get_conn()
        try:
            rows = conn.execute("SELECT * FROM mined_repos ORDER BY last_mined_date DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @app.get("/api/security/findings")
    def security_findings(status: str = "open", limit: int = Query(50, le=500)) -> list:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM scan_findings WHERE status=? ORDER BY scan_date DESC LIMIT ?",
                (status, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @app.get("/api/calibration/history")
    def calibration_history(limit: int = Query(20, le=100)) -> list:
        conn = get_conn()
        try:
            rows = conn.execute("SELECT * FROM calibration_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @app.post("/api/calibration/run")
    def trigger_calibration_run(
        background_tasks: BackgroundTasks,
        sample_n: int = 50,
        skip_extraction: bool = False,
    ) -> dict:
        from lessons_db.bugsInPy_calibrator import calibrate_pipeline

        def _run() -> None:
            conn = get_conn()
            try:
                calibrate_pipeline(conn, lance_dir=_lance_dir, sample_n=sample_n, skip_extraction=skip_extraction)
            finally:
                conn.close()

        background_tasks.add_task(_run)
        return {"status": "queued", "message": "Calibration started — check /api/calibration/history for results"}

    @app.post("/api/mining/run")
    def trigger_mining_run(background_tasks: BackgroundTasks, topics: list[str] | None = None) -> dict:
        from lessons_db.github_miner import MiningConfig, mine_repos_for_gaps

        config = MiningConfig(topics=topics or MiningConfig().topics)

        def _run() -> None:
            conn = get_conn()
            try:
                mine_repos_for_gaps(conn, _lance_dir, config=config)
            finally:
                conn.close()

        background_tasks.add_task(_run)
        return {"status": "queued", "message": "Mining started — check /api/mining/history for results"}

    @app.post("/api/security/scan")
    def trigger_security_scan(target: str | None = None) -> dict:
        from lessons_db.config import PROJECTS_DIR
        from lessons_db.security_scanner import run_full_security_scan

        target_path: Path | None = None
        if target:
            # Restrict scan target to PROJECTS_DIR — prevents path traversal
            # from any process that can reach localhost:7685.
            candidate = Path(target).resolve()
            try:
                candidate.relative_to(PROJECTS_DIR.resolve())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="target must be within projects directory") from exc
            target_path = candidate

        conn = get_conn()
        try:
            return run_full_security_scan(conn, target_path)
        finally:
            conn.close()

    @app.patch("/api/security/findings/{finding_id}")
    def update_finding(finding_id: int, body: StatusUpdate) -> dict:
        conn = get_conn()
        try:
            row = conn.execute("SELECT id FROM scan_findings WHERE id=?", (finding_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="finding not found")
            conn.execute(
                "UPDATE scan_findings SET status=? WHERE id=?",
                (body.status, finding_id),
            )
            conn.commit()
            return {"id": finding_id, "status": body.status}
        finally:
            conn.close()

    @app.get("/api/capture-drafts")
    def list_capture_drafts(status: str = "pending", limit: int = Query(50, le=200)) -> list:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM capture_drafts WHERE status=? ORDER BY created_date DESC LIMIT ?",
                (status, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    @app.patch("/api/capture-drafts/{draft_id}")
    def update_capture_draft(draft_id: int, body: StatusUpdate) -> dict:
        conn = get_conn()
        try:
            row = conn.execute("SELECT id FROM capture_drafts WHERE id=?", (draft_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="draft not found")
            conn.execute(
                "UPDATE capture_drafts SET status=? WHERE id=?",
                (body.status, draft_id),
            )
            conn.commit()
            return {"id": draft_id, "status": body.status}
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Fix queue
    # -----------------------------------------------------------------------

    @app.get("/api/fix-queue")
    def list_fix_queue(
        status: str = "pending",
        limit: int = Query(50, le=200),
    ) -> list:
        """List fix queue entries by status."""
        from lessons_db.db import get_fix_queue

        conn = get_conn()
        try:
            return get_fix_queue(conn, status=status, limit=limit)
        finally:
            conn.close()

    @app.get("/api/fix-queue/next")
    def get_next_fix_item() -> dict:
        """Return the highest-priority pending fix, or 404 if queue is empty."""
        from lessons_db.db import get_next_fix

        conn = get_conn()
        try:
            fix = get_next_fix(conn)
            if fix is None:
                raise HTTPException(status_code=404, detail="fix queue is empty")
            return fix
        finally:
            conn.close()

    @app.patch("/api/fix-queue/{fix_id}")
    def update_fix_queue_item(fix_id: int, body: FixStatusUpdate) -> dict:
        """Update fix status (applied, skipped, wont_fix, etc.)."""
        from lessons_db.db import update_fix_status

        valid = {"applied", "skipped", "wont_fix", "issue_created", "pending", "in_progress"}
        if body.status not in valid:
            raise HTTPException(status_code=422, detail=f"status must be one of {sorted(valid)}")

        conn = get_conn()
        try:
            row = conn.execute("SELECT id FROM fix_queue WHERE id=?", (fix_id,)).fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="fix not found")
            update_fix_status(conn, fix_id, body.status, github_issue_url=body.github_issue_url)
            return {"id": fix_id, "status": body.status}
        finally:
            conn.close()

    @app.post("/api/fix-queue/populate")
    def populate_fix_queue_endpoint(
        background_tasks: BackgroundTasks,
        min_severity: int = Query(3, ge=1, le=5),
    ) -> dict:
        """Populate fix queue from open scan findings (background)."""
        from lessons_db.prevention import populate_fix_queue

        def _run():
            conn = get_conn()
            try:
                populate_fix_queue(conn, min_severity=min_severity)
            finally:
                conn.close()

        background_tasks.add_task(_run)
        return {"status": "queued"}

    @app.post("/api/fix-queue/issues")
    def create_github_issues_endpoint(
        background_tasks: BackgroundTasks,
        repo: str | None = Query(None),
        min_severity: int = Query(4, ge=1, le=5),
        dry_run: bool = Query(False),
    ) -> dict:
        """Create GitHub issues for pending high-severity fixes (background)."""
        from lessons_db.prevention import create_github_issues

        def _run():
            conn = get_conn()
            try:
                create_github_issues(conn, repo=repo, min_severity=min_severity, dry_run=dry_run)
            finally:
                conn.close()

        background_tasks.add_task(_run)
        return {"status": "queued", "dry_run": dry_run}

    # -----------------------------------------------------------------------
    # Prevention pipeline
    # -----------------------------------------------------------------------

    @app.get("/api/prevention/report")
    def prevention_report_endpoint(window_days: int = Query(30, ge=1, le=365)) -> dict:
        """Comprehensive prevention effectiveness report."""
        from lessons_db.prevention import prevention_report

        conn = get_conn()
        try:
            return prevention_report(conn, window_days=window_days)
        finally:
            conn.close()

    @app.get("/api/prevention/recurrence")
    def prevention_recurrence_endpoint(
        window_days: int = Query(7, ge=1, le=90),
        threshold: int = Query(2, ge=1),
        limit: int = Query(20, le=100),
    ) -> list:
        """Lessons with high recurrence velocity (potential hotspots)."""
        from lessons_db.db import get_velocity_warnings

        conn = get_conn()
        try:
            return get_velocity_warnings(conn, window_days=window_days, threshold=threshold)[:limit]
        finally:
            conn.close()

    @app.post("/api/prevention/resolve-outcomes")
    def resolve_outcomes_endpoint(
        background_tasks: BackgroundTasks,
        max_age_hours: int = Query(24, ge=1),
    ) -> dict:
        """Batch-resolve stale unknown surfacing events (background)."""
        from lessons_db.prevention import resolve_outcomes

        def _run():
            conn = get_conn()
            try:
                resolve_outcomes(conn, max_age_hours=max_age_hours)
            finally:
                conn.close()

        background_tasks.add_task(_run)
        return {"status": "queued"}

    @app.post("/api/prevention/bulk-generate")
    def bulk_generate_rules_endpoint(
        background_tasks: BackgroundTasks,
        validate: bool = Query(True),
    ) -> dict:
        """Generate Semgrep rules for all lessons with detection patterns (background)."""
        from lessons_db.prevention import bulk_generate_rules

        def _run():
            conn = get_conn()
            try:
                bulk_generate_rules(conn, validate=validate)
            finally:
                conn.close()

        background_tasks.add_task(_run)
        return {"status": "queued"}

    @app.post("/api/prevention/check-content")
    def check_content_endpoint(body: dict) -> dict:
        """Check content string against detection patterns.

        Body: {content: str, file_path?: str}
        Returns: {block: bool, message: str, violations: [...]}
        """
        from lessons_db.prevention import check_content

        content = body.get("content", "")
        file_path = body.get("file_path")
        if not content:
            raise HTTPException(status_code=422, detail="content is required")

        conn = get_conn()
        try:
            return check_content(conn, content, file_path=file_path)
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Scan health dashboard
    # -----------------------------------------------------------------------

    @app.get("/api/scan/summary")
    def scan_summary() -> dict:  # noqa: C901, PLR0912, PLR0915
        """Decision-context dashboard for scan pipeline health.

        Returns each metric with both the raw value and a plain-English
        decision_context explaining what the number means, whether it's
        good or bad, and what you should do about it.
        """
        from datetime import UTC, datetime, timedelta

        conn = get_conn()
        try:
            result: dict[str, Any] = {}
            seven_days_ago = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")

            # ------------------------------------------------------------------
            # 1. promotion_rate — last 7 days
            # ------------------------------------------------------------------
            try:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'promoted') AS promoted,
                        COUNT(*) FILTER (WHERE status = 'dismissed') AS dismissed
                    FROM capture_drafts
                    WHERE created_date >= ?
                    """,
                    (seven_days_ago,),
                ).fetchone()
                promoted = row["promoted"] if row else 0
                dismissed = row["dismissed"] if row else 0
                total_decided = promoted + dismissed
                rate = promoted / total_decided if total_decided > 0 else None

                if rate is None:
                    status_val = "ok"
                    label = f"No decided drafts in last 7 days (promoted={promoted}, dismissed={dismissed})"
                elif rate >= 0.05:
                    status_val = "ok"
                    label = f"{promoted} of {total_decided} candidates promoted ({rate:.1%})"
                elif rate >= 0.02:
                    status_val = "warn"
                    label = f"{promoted} of {total_decided} candidates promoted ({rate:.1%})"
                else:
                    status_val = "alert"
                    label = f"{promoted} of {total_decided} candidates promoted ({rate:.1%})"

                result["promotion_rate"] = {
                    "value": round(rate, 4) if rate is not None else None,
                    "label": label,
                    "decision_context": (
                        "This is your signal-to-noise ratio for the capture pipeline. "
                        "Below 5% means the triage criteria are too loose — consider tightening "
                        "the QUALITY_MIN_SCORE threshold in config.py. "
                        "Above 20% means the filter is too aggressive and valuable patterns "
                        "may be dismissed prematurely. "
                        "null means no drafts reached a final status this week — check whether "
                        "the nightly pipeline is running."
                    ),
                    "status": status_val,
                }
            except Exception as exc:
                _log.warning("scan_summary: promotion_rate query failed: %s", exc)
                result["promotion_rate"] = {
                    "value": None,
                    "label": "Query failed",
                    "decision_context": (
                        "Failed to compute promotion rate. "
                        "Diagnose with: sqlite3 ~/.local/share/lessons-db/lessons.db "
                        '"SELECT status, COUNT(*) FROM capture_drafts GROUP BY status"'
                    ),
                    "status": "alert",
                }

            # ------------------------------------------------------------------
            # 2. drafts_captured_last_run — most recent nightly run
            # ------------------------------------------------------------------
            try:
                # scan_state stores metadata about the most recent nightly run
                drafted_raw = get_scan_state(conn, "last_run_drafted")
                sessions_raw = get_scan_state(conn, "last_run_sessions_processed")

                drafted_count = int(drafted_raw) if drafted_raw is not None else None
                sessions_count = int(sessions_raw) if sessions_raw is not None else None

                if drafted_count is None:
                    result["drafts_captured_last_run"] = {
                        "value": None,
                        "label": "No nightly run record found",
                        "decision_context": (
                            "The nightly pipeline has not written run metadata to scan_state yet. "
                            "This is normal on first install. After the first nightly run at 03:30, "
                            "this will show how many lesson candidates were captured. "
                            "Check service status: journalctl --user -u lessons-db-nightly.service --since today"
                        ),
                        "status": "warn",
                    }
                else:
                    status_val = "ok" if drafted_count > 0 else "warn"
                    result["drafts_captured_last_run"] = {
                        "value": drafted_count,
                        "label": f"{drafted_count} drafts captured in last run",
                        "decision_context": (
                            "Zero drafts on a run with sessions processed usually means the Ollama "
                            "extraction model returned no candidates — check "
                            "journalctl --user -u lessons-db-nightly.service for extraction errors. "
                            "A healthy run captures 3-15 drafts per batch."
                        ),
                        "status": status_val,
                    }
            except Exception as exc:
                _log.warning("scan_summary: drafts_captured_last_run query failed: %s", exc)
                result["drafts_captured_last_run"] = {
                    "value": None,
                    "label": "Query failed",
                    "decision_context": (
                        "Failed to retrieve last-run draft count from scan_state. "
                        "Check the scan_state table for key 'last_run_drafted'."
                    ),
                    "status": "alert",
                }

            # ------------------------------------------------------------------
            # 3. sessions_processed_last_run
            # ------------------------------------------------------------------
            try:
                sessions_raw = get_scan_state(conn, "last_run_sessions_processed")
                sessions_count = int(sessions_raw) if sessions_raw is not None else None

                if sessions_count is None:
                    result["sessions_processed_last_run"] = {
                        "value": None,
                        "label": "No nightly run record found",
                        "decision_context": (
                            "No session count recorded yet. The nightly pipeline writes this after "
                            "each batch-capture run. If the service has run and this is still null, "
                            "check whether batch-capture-transcripts.sh writes to scan_state."
                        ),
                        "status": "warn",
                    }
                else:
                    status_val = "ok" if sessions_count > 0 else "warn"
                    result["sessions_processed_last_run"] = {
                        "value": sessions_count,
                        "label": f"{sessions_count} sessions processed in last run",
                        "decision_context": (
                            "Zero sessions processed means either no new transcripts existed since "
                            "the last run, or the --since DATE filter excluded all files. "
                            "If this is consistently zero, check that Claude session transcripts "
                            "are accumulating in the expected directory."
                        ),
                        "status": status_val,
                    }
            except Exception as exc:
                _log.warning("scan_summary: sessions_processed_last_run query failed: %s", exc)
                result["sessions_processed_last_run"] = {
                    "value": None,
                    "label": "Query failed",
                    "decision_context": (
                        "Failed to retrieve session count from scan_state. "
                        "Check the scan_state table for key 'last_run_sessions_processed'."
                    ),
                    "status": "alert",
                }

            # ------------------------------------------------------------------
            # 4. last_scan_age_hours — pattern-scan freshness
            # ------------------------------------------------------------------
            try:
                ts_str = get_scan_state(conn, "last_scan_timestamp")

                if ts_str is None or ts_str == "1970-01-01T00:00:00":
                    age_hours = None
                    status_val = "warn"
                    label = "Pattern scan has never run"
                else:
                    # Parse ISO timestamp — try both with and without timezone
                    try:
                        ts = datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=UTC)
                        now = datetime.now(UTC)
                        age_hours = round((now - ts).total_seconds() / 3600, 1)
                    except ValueError:
                        age_hours = None

                    if age_hours is None:
                        status_val = "warn"
                        label = f"Could not parse timestamp: {ts_str}"
                    elif age_hours < 25:
                        status_val = "ok"
                        label = f"Last scan: {age_hours} hours ago"
                    elif age_hours < 48:
                        status_val = "warn"
                        label = f"Last scan: {age_hours} hours ago"
                    else:
                        status_val = "alert"
                        label = f"Last scan: {age_hours} hours ago"

                result["last_scan_age_hours"] = {
                    "value": age_hours,
                    "label": label,
                    "decision_context": (
                        "Pattern scan runs nightly at 03:00 via lessons-db-pattern-scan.service. "
                        "If this exceeds 25 hours, the timer has likely failed. "
                        "Diagnose with: journalctl --user -u lessons-db-pattern-scan.service --since today "
                        "and: systemctl --user status lessons-db-pattern-scan.timer"
                    ),
                    "status": status_val,
                }
            except Exception as exc:
                _log.warning("scan_summary: last_scan_age_hours query failed: %s", exc)
                result["last_scan_age_hours"] = {
                    "value": None,
                    "label": "Query failed",
                    "decision_context": (
                        "Failed to read last_scan_timestamp from scan_state. "
                        "Run: lessons-db scan to trigger a manual pattern scan."
                    ),
                    "status": "alert",
                }

            # ------------------------------------------------------------------
            # 5. embed_failure_rate — cross_project_scan drafts, last 7 days
            # ------------------------------------------------------------------
            try:
                # Embed failures are proxied by capture_drafts with
                # detection_source='cross_project_scan' that stayed 'pending'
                # past the triage window (older than 24h and not decided).
                # If the detection_source column doesn't exist, skip gracefully.
                cols = [r[1] for r in conn.execute("PRAGMA table_info('capture_drafts')").fetchall()]
                if "detection_source" not in cols:
                    result["embed_failure_rate"] = {
                        "value": None,
                        "label": "detection_source column not present",
                        "decision_context": (
                            "Embed failure tracking requires the detection_source column on "
                            "capture_drafts. Run: lessons-db migrate to apply schema migrations."
                        ),
                        "status": "warn",
                    }
                else:
                    row = conn.execute(
                        """
                        SELECT
                            COUNT(*) FILTER (WHERE detection_source = 'cross_project_scan') AS scan_count,
                            COUNT(*) AS total_count
                        FROM capture_drafts
                        WHERE created_date >= ?
                        """,
                        (seven_days_ago,),
                    ).fetchone()
                    scan_count = row["scan_count"] if row else 0

                    # Stale pending scan drafts = proxies for embed failures
                    stale_row = conn.execute(
                        """
                        SELECT COUNT(*) AS stale
                        FROM capture_drafts
                        WHERE detection_source = 'cross_project_scan'
                          AND status = 'pending'
                          AND created_date < date('now', '-1 day')
                        """,
                    ).fetchone()
                    stale = stale_row["stale"] if stale_row else 0

                    rate = stale / scan_count if scan_count > 0 else None

                    if rate is None:
                        status_val = "ok"
                        label = "No cross-project scan drafts in last 7 days"
                    elif rate < 0.1:
                        status_val = "ok"
                        label = f"{stale} of {scan_count} scan drafts stale/unresolved ({rate:.1%})"
                    elif rate < 0.3:
                        status_val = "warn"
                        label = f"{stale} of {scan_count} scan drafts stale/unresolved ({rate:.1%})"
                    else:
                        status_val = "alert"
                        label = f"{stale} of {scan_count} scan drafts stale/unresolved ({rate:.1%})"

                    result["embed_failure_rate"] = {
                        "value": round(rate, 4) if rate is not None else None,
                        "label": label,
                        "decision_context": (
                            "Stale cross-project scan drafts indicate embedding or triage failures — "
                            "the pattern scanner found candidates but they were never triaged. "
                            "Above 10%: check the Ollama embed service (lessons-db index --seed-only). "
                            "Above 30%: the triage pipeline is likely broken — "
                            "check lessons-db-nightly.service logs."
                        ),
                        "status": status_val,
                    }
            except Exception as exc:
                _log.warning("scan_summary: embed_failure_rate query failed: %s", exc)
                result["embed_failure_rate"] = {
                    "value": None,
                    "label": "Query failed",
                    "decision_context": (
                        "Failed to compute embed failure rate. Check capture_drafts table and detection_source column."
                    ),
                    "status": "alert",
                }

            # ------------------------------------------------------------------
            # 6. lessons_due_for_review — FSRS retrievability < 0.9
            # ------------------------------------------------------------------
            try:
                # Check if retrievability column exists (added in v8)
                cols = [r[1] for r in conn.execute("PRAGMA table_info('lessons')").fetchall()]
                if "retrievability" not in cols:
                    result["lessons_due_for_review"] = {
                        "value": None,
                        "label": "FSRS columns not present",
                        "decision_context": (
                            "FSRS spaced repetition columns are not yet initialized. "
                            "Run: lessons-db fsrs init  to backfill defaults on all lessons."
                        ),
                        "status": "warn",
                    }
                else:
                    due_count = conn.execute("SELECT COUNT(*) FROM lessons WHERE retrievability < 0.9").fetchone()[0]
                    total_lessons = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]

                    if due_count < 10:
                        status_val = "ok"
                    elif due_count < 30:
                        status_val = "warn"
                    else:
                        status_val = "alert"

                    result["lessons_due_for_review"] = {
                        "value": due_count,
                        "label": f"{due_count} of {total_lessons} lessons due for review (R < 0.9)",
                        "decision_context": (
                            "Retrievability < 0.9 means FSRS predicts a >10% chance you've forgotten "
                            "the lesson since last review. "
                            "Under 10 due: healthy — run 'lessons-db fsrs due' to review them. "
                            "10-30 due: review backlog building — consider a dedicated review session. "
                            "Over 30 due: review is significantly overdue; "
                            "run 'lessons-db fsrs due --threshold 0.7' to prioritize the most critical."
                        ),
                        "status": status_val,
                    }
            except Exception as exc:
                _log.warning("scan_summary: lessons_due_for_review query failed: %s", exc)
                result["lessons_due_for_review"] = {
                    "value": None,
                    "label": "Query failed",
                    "decision_context": (
                        "Failed to query FSRS retrievability. "
                        "Run: lessons-db fsrs init  to ensure FSRS columns are populated."
                    ),
                    "status": "alert",
                }

            return result

        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Eval data source contract
    # -----------------------------------------------------------------------

    class EvalResultItem(BaseModel):
        source_item_id: str
        target_item_id: str
        variant: str
        principle: str | None = None
        is_same_cluster: int | None = None
        score_transfer: int | None = None
        score_precision: int | None = None
        score_action: int | None = None

    class EvalResultsBody(BaseModel):
        run_id: str
        source: str
        results: list[EvalResultItem]

    class EvalProductionVariantBody(BaseModel):
        variant_id: str
        model: str
        prompt_template_id: str
        temperature: float
        num_ctx: int

    def _check_eval_auth(authorization: str | None) -> None:
        """Verify Bearer token if eval.data_source_token is configured."""
        token = os.environ.get("LESSONS_DB_EVAL_TOKEN", "").strip()
        if not token:
            return  # no token configured — auth disabled
        if authorization is None:
            raise HTTPException(status_code=401, detail="Authorization header required")
        parts = authorization.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer" or parts[1] != token:
            raise HTTPException(status_code=401, detail="Invalid token")

    @app.get("/eval/health")
    def eval_health(authorization: str | None = Header(default=None)) -> dict:
        """Health check — returns item count and cluster count (clusters with >= 3 items)."""
        _check_eval_auth(authorization)
        conn = get_conn()
        try:
            item_count = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
            cluster_count = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT cluster_seed
                    FROM lessons
                    WHERE cluster_seed IS NOT NULL AND cluster_seed != ''
                    GROUP BY cluster_seed
                    HAVING COUNT(*) >= 3
                )
                """
            ).fetchone()[0]
            return {"ok": True, "item_count": item_count, "cluster_count": cluster_count}
        finally:
            conn.close()

    @app.post("/eval/prime")
    def eval_prime(authorization: str | None = Header(default=None)) -> dict:
        """Backfill cluster_seed from cluster field for lessons that are missing it.

        Mirrors `lessons-db index --seed-only`. After this runs, lessons with a
        cluster assignment become eligible to appear in /eval/items (which requires
        cluster_seed to be set). Safe to call repeatedly — only affects NULL rows.
        """
        _check_eval_auth(authorization)
        conn = get_conn()
        try:
            cur = conn.execute(
                "UPDATE lessons SET cluster_seed = cluster "
                "WHERE cluster IS NOT NULL AND cluster != '' AND cluster_seed IS NULL"
            )
            updated = cur.rowcount
            conn.commit()
            item_count = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
            cluster_count = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT cluster_seed
                    FROM lessons
                    WHERE cluster_seed IS NOT NULL AND cluster_seed != ''
                    GROUP BY cluster_seed
                    HAVING COUNT(*) >= 3
                )
                """
            ).fetchone()[0]
            return {"ok": True, "updated": updated, "item_count": item_count, "cluster_count": cluster_count}
        finally:
            conn.close()

    @app.get("/eval/items")
    def eval_items(
        cluster_id: str | None = None,
        limit: int = Query(100, ge=1, le=1000),
        authorization: str | None = Header(default=None),
    ) -> list:
        """Return lessons from clusters with >= 3 items, optionally filtered by cluster_id."""
        _check_eval_auth(authorization)
        conn = get_conn()
        try:
            # Build subquery: clusters with >= 3 items
            params: list[Any] = []
            if cluster_id is not None:
                where = """
                    WHERE cluster_seed IS NOT NULL AND cluster_seed != ''
                      AND cluster_seed IN (
                          SELECT cluster_seed FROM lessons
                          WHERE cluster_seed IS NOT NULL AND cluster_seed != ''
                          GROUP BY cluster_seed HAVING COUNT(*) >= 3
                      )
                      AND cluster_seed = ?
                """
                params.append(cluster_id)
            else:
                where = """
                    WHERE cluster_seed IS NOT NULL AND cluster_seed != ''
                      AND cluster_seed IN (
                          SELECT cluster_seed FROM lessons
                          WHERE cluster_seed IS NOT NULL AND cluster_seed != ''
                          GROUP BY cluster_seed HAVING COUNT(*) >= 3
                      )
                """
            rows = conn.execute(
                f"SELECT id, title, one_liner, description, cluster_seed, category FROM lessons {where} LIMIT ?",
                params + [limit],
            ).fetchall()
            return [
                {
                    "id": str(r["id"]),
                    "title": r["title"],
                    "one_liner": r["one_liner"] if r["one_liner"] else r["title"],
                    "description": r["description"],
                    "cluster_id": r["cluster_seed"],
                    "category": r["category"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    @app.get("/eval/clusters")
    def eval_clusters(authorization: str | None = Header(default=None)) -> list:
        """Return clusters with >= 3 items."""
        _check_eval_auth(authorization)
        conn = get_conn()
        try:
            rows = conn.execute(
                """
                SELECT cluster_seed, COUNT(*) as item_count,
                       (SELECT category FROM lessons l2
                        WHERE l2.cluster_seed = l.cluster_seed
                          AND l2.category IS NOT NULL
                        ORDER BY l2.id ASC LIMIT 1) as first_category
                FROM lessons l
                WHERE cluster_seed IS NOT NULL AND cluster_seed != ''
                GROUP BY cluster_seed
                HAVING COUNT(*) >= 3
                ORDER BY item_count DESC
                """
            ).fetchall()
            return [
                {
                    "id": str(r["cluster_seed"]),
                    "label": r["first_category"] if r["first_category"] else r["cluster_seed"],
                    "item_count": r["item_count"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    @app.post("/eval/results")
    def eval_results(
        body: EvalResultsBody,
        authorization: str | None = Header(default=None),
    ) -> dict:
        """Upsert eval results. Idempotent — INSERT OR IGNORE preserves original created_at on retry."""
        _check_eval_auth(authorization)
        conn = get_conn()
        try:
            now = datetime.now(UTC).isoformat()
            accepted = 0
            for item in body.results:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO eval_results
                        (run_id, source_item_id, target_item_id, variant, principle,
                         is_same_cluster, score_transfer, score_precision, score_action, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        body.run_id,
                        item.source_item_id,
                        item.target_item_id,
                        item.variant,
                        item.principle,
                        item.is_same_cluster,
                        item.score_transfer,
                        item.score_precision,
                        item.score_action,
                        now,
                    ],
                )
                accepted += 1
            conn.commit()
            return {"accepted": accepted}
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"eval_results DB error: {exc}") from exc
        finally:
            conn.close()

    @app.post("/eval/production-variant")
    def eval_production_variant(
        body: EvalProductionVariantBody,
        authorization: str | None = Header(default=None),
    ) -> dict:
        """Store (or replace) the production variant. Idempotent — keyed by fixed id 'production'."""
        _check_eval_auth(authorization)
        conn = get_conn()
        try:
            now = datetime.now(UTC).isoformat()
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_production_variant
                    (id, variant_id, model, prompt_template_id, temperature, num_ctx, updated_at)
                VALUES ('production', ?, ?, ?, ?, ?, ?)
                """,
                [
                    body.variant_id,
                    body.model,
                    body.prompt_template_id,
                    body.temperature,
                    body.num_ctx,
                    now,
                ],
            )
            conn.commit()
            return {"accepted": True}
        finally:
            conn.close()

    # --- Static files for SPA dashboard ---
    spa_dir = Path(__file__).resolve().parent.parent.parent / "spa" / "dist"
    if spa_dir.exists():
        _no_store = {"Cache-Control": "no-store"}

        @app.get("/ui/{path:path}")
        async def spa_static(path: str):
            """Serve SPA — static files or fallback to index.html for client-side routing."""
            if path and "\x00" in path:
                return HTMLResponse("Not found", status_code=404)
            real = (spa_dir / path).resolve() if path else None
            if real and real.is_file() and real.is_relative_to(spa_dir.resolve()):
                return FileResponse(real, headers=_no_store)
            index = spa_dir / "index.html"
            return (
                FileResponse(index, headers=_no_store)
                if index.is_file()
                else HTMLResponse("Not found", status_code=404)
            )

    return app


# Entry point for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lessons_db.api:app", host="127.0.0.1", port=7685, reload=False)
