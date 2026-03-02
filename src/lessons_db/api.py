"""FastAPI server for lessons-db — port 7685.

Eliminates subprocess cold start (~300ms) per Express API call.
All routes are thin wrappers over existing DB/analyzer functions.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lessons_db.config import LANCE_DIR, SQLITE_PATH
from lessons_db.db import init_db
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

    return app


# Entry point for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lessons_db.api:app", host="127.0.0.1", port=7685, reload=False)
