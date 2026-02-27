"""Tests for scan_state table and capture_drafts v3 columns."""

import pytest
from lessons_db.db import init_db, get_scan_state, set_scan_state


class TestScanState:
    def test_get_default_threshold(self, db_path):
        conn = init_db(db_path)
        val = get_scan_state(conn, "auto_approve_threshold")
        assert val == "0.85"

    def test_set_and_get_threshold(self, db_path):
        conn = init_db(db_path)
        set_scan_state(conn, "auto_approve_threshold", "0.80")
        assert get_scan_state(conn, "auto_approve_threshold") == "0.80"

    def test_capture_drafts_has_detection_source(self, db_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO capture_drafts "
            "(raw_content, status, created_date, source, detection_source, confidence) "
            "VALUES (?, 'pending', '2026-02-26', 'test', 'cross_project_scan', 0.88)",
            ["snippet"]
        )
        conn.commit()
        row = conn.execute(
            "SELECT detection_source, confidence FROM capture_drafts"
        ).fetchone()
        assert row["detection_source"] == "cross_project_scan"
        assert abs(row["confidence"] - 0.88) < 0.001
