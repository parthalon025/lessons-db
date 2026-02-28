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

    @app.get("/api/lessons")
    def list_lessons(
        q: str | None = None,
        category: str | None = None,
        tier: str | None = None,
        polarity: str | None = None,
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
            rows = conn.execute(
                f"SELECT * FROM lessons {where} ORDER BY id DESC LIMIT ? OFFSET ?",
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
        from lessons_db.security_scanner import run_full_security_scan

        conn = get_conn()
        try:
            target_path = Path(target) if target else None
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

    return app


# Entry point for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lessons_db.api:app", host="127.0.0.1", port=7685, reload=False)
