"""Tests for eval data source contract endpoints (/eval/*)."""

import pytest
from fastapi.testclient import TestClient

from lessons_db.db import init_db, insert_lesson

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(db_path, tmp_path):
    from lessons_db.api import create_app

    app = create_app(db_path=db_path, lance_dir=tmp_path / "lance")
    return TestClient(app)


@pytest.fixture
def seeded_db(db_path):
    """DB pre-seeded with enough lessons to form clusters."""
    conn = init_db(db_path)
    # Cluster "A": 4 lessons (>= 3, qualifies)
    for i in range(4):
        insert_lesson(
            conn,
            {
                "title": f"Cluster A lesson {i}",
                "one_liner": f"one-liner A{i}",
                "description": f"description A{i}",
                "cluster_seed": "A",
                "category": "integration",
                "created_date": "2026-01-01",
            },
        )
    # Cluster "B": 3 lessons (exactly 3, qualifies)
    for i in range(3):
        insert_lesson(
            conn,
            {
                "title": f"Cluster B lesson {i}",
                "one_liner": f"one-liner B{i}",
                "description": f"description B{i}",
                "cluster_seed": "B",
                "category": "testing",
                "created_date": "2026-01-01",
            },
        )
    # Cluster "C": 2 lessons (< 3, does NOT qualify)
    for i in range(2):
        insert_lesson(
            conn,
            {
                "title": f"Cluster C lesson {i}",
                "one_liner": f"one-liner C{i}",
                "description": f"description C{i}",
                "cluster_seed": "C",
                "category": "security",
                "created_date": "2026-01-01",
            },
        )
    # 1 lesson with no cluster_seed
    insert_lesson(
        conn,
        {
            "title": "No cluster lesson",
            "created_date": "2026-01-01",
        },
    )
    conn.close()
    return db_path


@pytest.fixture
def seeded_client(seeded_db, tmp_path):
    from lessons_db.api import create_app

    app = create_app(db_path=seeded_db, lance_dir=tmp_path / "lance")
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /eval/health
# ---------------------------------------------------------------------------


def test_eval_health_ok(seeded_client):
    resp = seeded_client.get("/eval/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["item_count"] > 0
    assert data["cluster_count"] >= 0


def test_eval_health_item_count(seeded_client, seeded_db):
    conn = init_db(seeded_db)
    total = conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
    conn.close()

    resp = seeded_client.get("/eval/health")
    data = resp.json()
    assert data["item_count"] == total


def test_eval_health_cluster_count(seeded_client):
    """Only clusters with >= 3 items are counted."""
    resp = seeded_client.get("/eval/health")
    data = resp.json()
    # Cluster A (4) and B (3) qualify; C (2) does not
    assert data["cluster_count"] == 2


def test_eval_health_empty_db(client):
    resp = client.get("/eval/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["item_count"] == 0
    assert data["cluster_count"] == 0


# ---------------------------------------------------------------------------
# GET /eval/items
# ---------------------------------------------------------------------------


def test_eval_items_returns_list(seeded_client):
    resp = seeded_client.get("/eval/items")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_eval_items_required_fields(seeded_client):
    resp = seeded_client.get("/eval/items")
    items = resp.json()
    assert len(items) > 0
    for item in items:
        assert "id" in item
        assert "title" in item
        assert "one_liner" in item
        assert "description" in item
        assert "cluster_id" in item
        assert "category" in item


def test_eval_items_id_is_string(seeded_client):
    resp = seeded_client.get("/eval/items")
    items = resp.json()
    for item in items:
        assert isinstance(item["id"], str)


def test_eval_items_only_qualified_clusters(seeded_client):
    """Only lessons from clusters with >= 3 items are returned."""
    resp = seeded_client.get("/eval/items")
    items = resp.json()
    cluster_ids = {item["cluster_id"] for item in items}
    # Cluster C (2 items) must not appear; unclustered lesson must not appear
    assert "C" not in cluster_ids
    assert None not in cluster_ids
    # Cluster A and B should appear
    assert "A" in cluster_ids
    assert "B" in cluster_ids


def test_eval_items_filter_by_cluster_id(seeded_client):
    resp = seeded_client.get("/eval/items?cluster_id=A")
    items = resp.json()
    assert len(items) == 4
    for item in items:
        assert item["cluster_id"] == "A"


def test_eval_items_filter_nonexistent_cluster(seeded_client):
    resp = seeded_client.get("/eval/items?cluster_id=ZZZZ")
    assert resp.status_code == 200
    assert resp.json() == []


def test_eval_items_filter_small_cluster_excluded(seeded_client):
    """Filtering by cluster C (2 items) returns empty — doesn't meet threshold."""
    resp = seeded_client.get("/eval/items?cluster_id=C")
    assert resp.status_code == 200
    assert resp.json() == []


def test_eval_items_one_liner_fallback(db_path, tmp_path):
    """If one_liner is null, one_liner field returns the lesson title."""
    conn = init_db(db_path)
    for i in range(3):
        insert_lesson(
            conn,
            {
                "title": f"No one-liner lesson {i}",
                "one_liner": None,
                "cluster_seed": "X",
                "created_date": "2026-01-01",
            },
        )
    conn.close()

    from lessons_db.api import create_app

    app = create_app(db_path=db_path, lance_dir=tmp_path / "lance")
    c = TestClient(app)
    items = c.get("/eval/items?cluster_id=X").json()
    assert len(items) == 3
    for item in items:
        assert item["one_liner"] == item["title"]


def test_eval_items_limit_param(seeded_client):
    resp = seeded_client.get("/eval/items?limit=3")
    assert resp.status_code == 200
    assert len(resp.json()) <= 3


# ---------------------------------------------------------------------------
# GET /eval/clusters
# ---------------------------------------------------------------------------


def test_eval_clusters_returns_list(seeded_client):
    resp = seeded_client.get("/eval/clusters")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_eval_clusters_required_fields(seeded_client):
    resp = seeded_client.get("/eval/clusters")
    clusters = resp.json()
    for c in clusters:
        assert "id" in c
        assert "label" in c
        assert "item_count" in c


def test_eval_clusters_only_qualified(seeded_client):
    """Cluster C (2 items) must not appear."""
    resp = seeded_client.get("/eval/clusters")
    clusters = resp.json()
    ids = [c["id"] for c in clusters]
    assert "C" not in ids
    assert "A" in ids
    assert "B" in ids


def test_eval_clusters_item_counts(seeded_client):
    resp = seeded_client.get("/eval/clusters")
    clusters = resp.json()
    by_id = {c["id"]: c for c in clusters}
    assert by_id["A"]["item_count"] == 4
    assert by_id["B"]["item_count"] == 3


def test_eval_clusters_empty_db(client):
    resp = client.get("/eval/clusters")
    assert resp.status_code == 200
    assert resp.json() == []


def test_eval_clusters_label(seeded_client):
    """Label is the first category in the cluster."""
    resp = seeded_client.get("/eval/clusters")
    clusters = resp.json()
    by_id = {c["id"]: c for c in clusters}
    assert by_id["A"]["label"] == "integration"
    assert by_id["B"]["label"] == "testing"


# ---------------------------------------------------------------------------
# POST /eval/results
# ---------------------------------------------------------------------------

SAMPLE_RESULT = {
    "source_item_id": "1",
    "target_item_id": "2",
    "variant": "A",
    "principle": "test principle",
    "is_same_cluster": 1,
    "score_transfer": 3,
    "score_precision": 4,
    "score_action": 2,
}


def test_eval_results_insert(client):
    body = {
        "run_id": "run-001",
        "source": "ollama-queue",
        "results": [SAMPLE_RESULT],
    }
    resp = client.post("/eval/results", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["accepted"] == 1


def test_eval_results_multiple(client):
    results = [{**SAMPLE_RESULT, "target_item_id": str(i), "variant": "A"} for i in range(5)]
    body = {"run_id": "run-002", "source": "ollama-queue", "results": results}
    resp = client.post("/eval/results", json=body)
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 5


def test_eval_results_idempotent(client):
    """Posting same results twice should not create duplicates."""
    body = {
        "run_id": "run-003",
        "source": "ollama-queue",
        "results": [SAMPLE_RESULT],
    }
    resp1 = client.post("/eval/results", json=body)
    resp2 = client.post("/eval/results", json=body)
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["accepted"] == 1
    assert resp2.json()["accepted"] == 1

    # Verify only one row in DB (INSERT OR REPLACE)
    from lessons_db.db import init_db

    conn = init_db(client.app.state._db_path if hasattr(client.app.state, "_db_path") else None)


def test_eval_results_idempotent_db_check(db_path, tmp_path):
    """Direct DB check: two identical POSTs produce only one row."""
    from lessons_db.api import create_app

    app = create_app(db_path=db_path, lance_dir=tmp_path / "lance")
    c = TestClient(app)
    body = {
        "run_id": "run-idem",
        "source": "ollama-queue",
        "results": [SAMPLE_RESULT],
    }
    c.post("/eval/results", json=body)
    c.post("/eval/results", json=body)

    conn = init_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM eval_results WHERE run_id='run-idem'").fetchone()[0]
    conn.close()
    assert count == 1


def test_eval_results_empty_results(client):
    body = {"run_id": "run-empty", "source": "ollama-queue", "results": []}
    resp = client.post("/eval/results", json=body)
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 0


# ---------------------------------------------------------------------------
# POST /eval/production-variant
# ---------------------------------------------------------------------------

SAMPLE_VARIANT = {
    "variant_id": "v-abc123",
    "model": "qwen2.5:7b",
    "prompt_template_id": "tmpl-001",
    "temperature": 0.7,
    "num_ctx": 4096,
}


def test_eval_production_variant_accepted(client):
    resp = client.post("/eval/production-variant", json=SAMPLE_VARIANT)
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True


def test_eval_production_variant_idempotent(db_path, tmp_path):
    """Two POSTs with different variant_id should only keep the latest (1 row)."""
    from lessons_db.api import create_app

    app = create_app(db_path=db_path, lance_dir=tmp_path / "lance")
    c = TestClient(app)

    c.post("/eval/production-variant", json=SAMPLE_VARIANT)
    updated = {**SAMPLE_VARIANT, "variant_id": "v-xyz999"}
    c.post("/eval/production-variant", json=updated)

    conn = init_db(db_path)
    rows = conn.execute("SELECT * FROM eval_production_variant WHERE id='production'").fetchall()
    conn.close()
    assert len(rows) == 1
    assert dict(rows[0])["variant_id"] == "v-xyz999"


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_eval_auth_no_token_configured(monkeypatch, client):
    """When LESSONS_DB_EVAL_TOKEN is not set, all requests pass without auth."""
    monkeypatch.delenv("LESSONS_DB_EVAL_TOKEN", raising=False)
    resp = client.get("/eval/health")
    assert resp.status_code == 200


def test_eval_auth_token_configured_no_header(monkeypatch, client):
    """When token is set, missing Authorization header → 401."""
    monkeypatch.setenv("LESSONS_DB_EVAL_TOKEN", "secret-token")
    resp = client.get("/eval/health")
    assert resp.status_code == 401


def test_eval_auth_token_configured_wrong_token(monkeypatch, client):
    """Wrong token → 401."""
    monkeypatch.setenv("LESSONS_DB_EVAL_TOKEN", "secret-token")
    resp = client.get("/eval/health", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 401


def test_eval_auth_token_configured_correct_token(monkeypatch, client):
    """Correct token → 200."""
    monkeypatch.setenv("LESSONS_DB_EVAL_TOKEN", "secret-token")
    resp = client.get("/eval/health", headers={"Authorization": "Bearer secret-token"})
    assert resp.status_code == 200


def test_eval_auth_applies_to_all_eval_endpoints(monkeypatch, client):
    """Auth guard covers all 5 eval endpoints."""
    monkeypatch.setenv("LESSONS_DB_EVAL_TOKEN", "tok")
    endpoints = [
        ("GET", "/eval/health"),
        ("GET", "/eval/items"),
        ("GET", "/eval/clusters"),
    ]
    for method, path in endpoints:
        resp = client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} should return 401, got {resp.status_code}"

    # POST endpoints
    post_cases = [
        ("/eval/results", {"run_id": "x", "source": "y", "results": []}),
        ("/eval/production-variant", SAMPLE_VARIANT),
    ]
    for path, body in post_cases:
        resp = client.post(path, json=body)
        assert resp.status_code == 401, f"POST {path} should return 401, got {resp.status_code}"
