# tests/test_api.py
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(db_path, tmp_path):
    from lessons_db.api import create_app

    app = create_app(db_path=db_path, lance_dir=tmp_path / "lance")
    return TestClient(app)


def test_get_lessons_empty(client):
    resp = client.get("/api/lessons")
    assert resp.status_code == 200
    data = resp.json()
    assert "lessons" in data
    assert isinstance(data["lessons"], list)


def test_get_gaps(client):
    resp = client.get("/api/gaps")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert all("category" in item for item in data)


def test_get_mining_history(client):
    resp = client.get("/api/mining/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_security_findings(client):
    resp = client.get("/api/security/findings")
    assert resp.status_code == 200


def test_lesson_not_found(client):
    resp = client.get("/api/lessons/99999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Additional endpoint coverage (Task 15)
# ---------------------------------------------------------------------------


def test_post_mining_run_queues(client):
    """POST /api/mining/run returns queued immediately — does not block."""
    from unittest.mock import patch

    # mine_repos_for_gaps is imported locally inside the background task
    with patch("lessons_db.github_miner.mine_repos_for_gaps"):
        resp = client.post("/api/mining/run")
    assert resp.status_code in (200, 202)
    data = resp.json()
    assert data.get("status") == "queued"


def test_get_mining_repos_empty(client):
    resp = client.get("/api/mining/repos")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_capture_drafts_empty(client):
    resp = client.get("/api/capture-drafts")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_patch_capture_draft_approve(client, db_path):
    """PATCH /api/capture-drafts/{id} updates status."""
    from lessons_db.db import init_db

    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
        "VALUES (?, ?, 'pending', date('now'), 'test')",
        ["raw diff", json.dumps({"title": "test"})],
    )
    conn.commit()
    draft_id = conn.execute("SELECT id FROM capture_drafts").fetchone()["id"]

    resp = client.patch(f"/api/capture-drafts/{draft_id}", json={"status": "approved"})
    assert resp.status_code == 200

    updated = conn.execute("SELECT status FROM capture_drafts WHERE id=?", [draft_id]).fetchone()
    assert updated["status"] == "approved"


def test_patch_capture_draft_not_found(client):
    resp = client.patch("/api/capture-drafts/99999", json={"status": "approved"})
    assert resp.status_code == 404


def test_get_lessons_stats(client):
    resp = client.get("/api/lessons/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data


def test_get_lessons_categories(client):
    resp = client.get("/api/lessons/categories")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
