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

from lessons_db.config import LANCE_DIR, SQLITE_PATH
from lessons_db.db import init_db
from lessons_db.gap_analyzer import get_gap_report

_log = logging.getLogger(__name__)


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
        allow_methods=["GET", "POST"],
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

    return app


# Entry point for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lessons_db.api:app", host="127.0.0.1", port=7685, reload=False)
