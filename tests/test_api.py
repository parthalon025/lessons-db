# tests/test_api.py
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
