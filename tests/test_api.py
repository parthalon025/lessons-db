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


# ---------------------------------------------------------------------------
# Calibration endpoints (Task 15 — BugsInPy calibration feature)
# ---------------------------------------------------------------------------


def test_get_calibration_history_empty(client):
    resp = client.get("/api/calibration/history")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_calibration_history_with_data(client, db_path):
    from lessons_db.db import init_db

    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO calibration_runs (run_date, dataset, bugs_sampled, bugs_with_valid_diffs, "
        "extraction_attempted, extraction_success, gate0_pass, gate14_pass, pass_rate, notes) "
        "VALUES (date('now'), 'BugsInPy', 50, 45, 40, 35, 32, 10, 0.64, 'test run')"
    )
    conn.commit()

    resp = client.get("/api/calibration/history")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["dataset"] == "BugsInPy"
    assert data[0]["bugs_sampled"] == 50
    assert data[0]["pass_rate"] == pytest.approx(0.64)


def test_post_calibration_run_queues(client):
    """POST /api/calibration/run returns queued immediately — does not block."""
    from unittest.mock import patch

    with patch("lessons_db.bugsInPy_calibrator.calibrate_pipeline"):
        resp = client.post("/api/calibration/run")
    assert resp.status_code in (200, 202)
    data = resp.json()
    assert data.get("status") == "queued"


# ---------------------------------------------------------------------------
# GET /api/scan/summary — decision-context dashboard
# ---------------------------------------------------------------------------

SCAN_SUMMARY_METRICS = [
    "promotion_rate",
    "drafts_captured_last_run",
    "sessions_processed_last_run",
    "last_scan_age_hours",
    "embed_failure_rate",
    "lessons_due_for_review",
]
METRIC_KEYS = {"value", "label", "decision_context", "status"}
VALID_STATUSES = {"ok", "warn", "alert"}


def test_scan_summary_returns_200(client):
    """GET /api/scan/summary returns 200 on an empty DB."""
    resp = client.get("/api/scan/summary")
    assert resp.status_code == 200


def test_scan_summary_has_all_metrics(client):
    """Response contains every required top-level metric key."""
    resp = client.get("/api/scan/summary")
    data = resp.json()
    for metric in SCAN_SUMMARY_METRICS:
        assert metric in data, f"missing metric: {metric}"


def test_scan_summary_metric_structure(client):
    """Every metric has value, label, decision_context, and status fields."""
    resp = client.get("/api/scan/summary")
    data = resp.json()
    for metric in SCAN_SUMMARY_METRICS:
        obj = data[metric]
        assert set(obj.keys()) == METRIC_KEYS, f"{metric} has wrong keys: {set(obj.keys())}"
        assert obj["status"] in VALID_STATUSES, f"{metric}.status invalid: {obj['status']}"
        assert isinstance(obj["label"], str) and obj["label"], f"{metric}.label must be non-empty string"
        assert (
            isinstance(obj["decision_context"], str) and obj["decision_context"]
        ), f"{metric}.decision_context must be non-empty string"


def test_scan_summary_promotion_rate_empty_db(client):
    """With no decided drafts, promotion_rate.value is None and status is ok."""
    resp = client.get("/api/scan/summary")
    data = resp.json()
    pr = data["promotion_rate"]
    assert pr["value"] is None
    assert pr["status"] == "ok"


def test_scan_summary_promotion_rate_with_data(client, db_path):
    """promotion_rate reflects actual promoted/dismissed counts."""
    import json

    from lessons_db.db import init_db

    conn = init_db(db_path)
    # Insert 2 promoted, 18 dismissed in last 7 days -> rate = 0.10 -> ok
    for i in range(2):
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'promoted', date('now'), 'test')",
            [f"raw{i}", json.dumps({})],
        )
    for i in range(18):
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'dismissed', date('now'), 'test')",
            [f"raw_d{i}", json.dumps({})],
        )
    conn.commit()
    conn.close()

    resp = client.get("/api/scan/summary")
    data = resp.json()
    pr = data["promotion_rate"]
    assert pr["value"] == pytest.approx(0.10, abs=0.01)
    assert pr["status"] == "ok"


def test_scan_summary_promotion_rate_alert(client, db_path):
    """promotion_rate < 0.02 triggers alert status."""
    import json

    from lessons_db.db import init_db

    conn = init_db(db_path)
    # 0 promoted, 100 dismissed -> rate = 0.0 -> alert
    for i in range(100):
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'dismissed', date('now'), 'test')",
            [f"raw_d{i}", json.dumps({})],
        )
    conn.commit()
    conn.close()

    resp = client.get("/api/scan/summary")
    data = resp.json()
    pr = data["promotion_rate"]
    assert pr["value"] == pytest.approx(0.0)
    assert pr["status"] == "alert"


def test_scan_summary_promotion_rate_warn(client, db_path):
    """promotion_rate in [0.02, 0.05) triggers warn status."""
    import json

    from lessons_db.db import init_db

    conn = init_db(db_path)
    # 3 promoted, 97 dismissed -> rate = 0.03 -> warn
    for i in range(3):
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'promoted', date('now'), 'test')",
            [f"raw_p{i}", json.dumps({})],
        )
    for i in range(97):
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'dismissed', date('now'), 'test')",
            [f"raw_d{i}", json.dumps({})],
        )
    conn.commit()
    conn.close()

    resp = client.get("/api/scan/summary")
    data = resp.json()
    pr = data["promotion_rate"]
    assert pr["status"] == "warn"


def test_scan_summary_last_scan_age_fresh(client, db_path):
    """last_scan_age_hours is ok when timestamp is recent."""
    from datetime import UTC, datetime

    from lessons_db.db import init_db, set_scan_state

    conn = init_db(db_path)
    # Set timestamp to 2 hours ago
    recent = (datetime.now(UTC).replace(microsecond=0)).isoformat()
    set_scan_state(conn, "last_scan_timestamp", recent)
    conn.close()

    resp = client.get("/api/scan/summary")
    data = resp.json()
    age = data["last_scan_age_hours"]
    assert age["status"] == "ok"
    assert age["value"] is not None
    assert age["value"] < 25


def test_scan_summary_last_scan_age_never_run(client):
    """Default epoch timestamp (never run) gives warn status."""
    # Default DB has '1970-01-01T00:00:00' seeded by _seed_scan_state
    resp = client.get("/api/scan/summary")
    data = resp.json()
    age = data["last_scan_age_hours"]
    # Never-run epoch -> None value, warn status
    assert age["value"] is None
    assert age["status"] == "warn"


def test_scan_summary_last_scan_age_stale(client, db_path):
    """Timestamp older than 25 hours gives warn or alert."""
    from datetime import UTC, datetime, timedelta

    from lessons_db.db import init_db, set_scan_state

    conn = init_db(db_path)
    stale = (datetime.now(UTC) - timedelta(hours=30)).replace(microsecond=0).isoformat()
    set_scan_state(conn, "last_scan_timestamp", stale)
    conn.close()

    resp = client.get("/api/scan/summary")
    data = resp.json()
    age = data["last_scan_age_hours"]
    assert age["status"] in ("warn", "alert")
    assert age["value"] is not None
    assert age["value"] > 25


def test_scan_summary_lessons_due_empty_db(client):
    """With no lessons, lessons_due_for_review.value is 0 and status is ok."""
    resp = client.get("/api/scan/summary")
    data = resp.json()
    due = data["lessons_due_for_review"]
    assert due["value"] == 0
    assert due["status"] == "ok"


def test_scan_summary_lessons_due_with_overdue(client, db_path):
    """Lessons with retrievability < 0.9 are counted correctly."""
    from lessons_db.db import init_db, insert_lesson

    conn = init_db(db_path)
    # Insert 35 lessons with low retrievability -> alert threshold (>30)
    for i in range(35):
        lid = insert_lesson(conn, {"title": f"Lesson {i}", "created_date": "2026-01-01"})
        conn.execute("UPDATE lessons SET retrievability = 0.5 WHERE id = ?", [lid])
    conn.commit()
    conn.close()

    resp = client.get("/api/scan/summary")
    data = resp.json()
    due = data["lessons_due_for_review"]
    assert due["value"] == 35
    assert due["status"] == "alert"


def test_scan_summary_lessons_due_warn_threshold(client, db_path):
    """15 lessons with low retrievability -> warn status."""
    from lessons_db.db import init_db, insert_lesson

    conn = init_db(db_path)
    for i in range(15):
        lid = insert_lesson(conn, {"title": f"Lesson {i}", "created_date": "2026-01-01"})
        conn.execute("UPDATE lessons SET retrievability = 0.7 WHERE id = ?", [lid])
    conn.commit()
    conn.close()

    resp = client.get("/api/scan/summary")
    data = resp.json()
    due = data["lessons_due_for_review"]
    assert due["value"] == 15
    assert due["status"] == "warn"


def test_scan_summary_nightly_run_metadata(client, db_path):
    """scan_state keys for last_run_drafted and sessions_processed are reflected."""
    from lessons_db.db import init_db, set_scan_state

    conn = init_db(db_path)
    set_scan_state(conn, "last_run_drafted", "7")
    set_scan_state(conn, "last_run_sessions_processed", "12")
    conn.close()

    resp = client.get("/api/scan/summary")
    data = resp.json()
    assert data["drafts_captured_last_run"]["value"] == 7
    assert data["drafts_captured_last_run"]["status"] == "ok"
    assert data["sessions_processed_last_run"]["value"] == 12
    assert data["sessions_processed_last_run"]["status"] == "ok"


def test_scan_summary_nightly_run_missing_keys(client):
    """When scan_state lacks run metadata, value is None and status is warn."""
    resp = client.get("/api/scan/summary")
    data = resp.json()
    assert data["drafts_captured_last_run"]["value"] is None
    assert data["drafts_captured_last_run"]["status"] == "warn"
    assert data["sessions_processed_last_run"]["value"] is None
    assert data["sessions_processed_last_run"]["status"] == "warn"
