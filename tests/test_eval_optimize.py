"""Tests for APO eval-optimize: prompt_variants table, optimizer strategies, variant registration."""

import json
import sqlite3

import pytest

from lessons_db.db import init_db
from lessons_db.eval.optimize import load_all_variant_configs, parse_optimizer_candidates


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


class TestPromptVariantsTable:
    """prompt_variants table: schema, UNIQUE constraint, round-trip."""

    def test_table_exists(self, db_path):
        """init_db creates the prompt_variants table."""
        conn = init_db(db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "prompt_variants" in tables

    def test_unique_variant_id(self, db_path):
        """Duplicate variant_id raises IntegrityError."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("X01", "test", "{}", "feedback", "2026-01-01"),
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO prompt_variants
                   (variant_id, instruction_text, config_json, strategy, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("X01", "dupe", "{}", "feedback", "2026-01-01"),
            )

    def test_insert_and_query_round_trip(self, db_path):
        """Insert a prompt variant and read it back with all fields."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, parent_variant,
                strategy, optimizer_model, hypothesis, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "X01",
                "Extract principle",
                '{"model":"qwen3:14b"}',
                "D",
                "feedback",
                "qwen3:14b",
                "reduce false positives",
                "2026-03-11T00:00:00",
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM prompt_variants WHERE variant_id = ?", ("X01",)).fetchone()
        assert row["variant_id"] == "X01"
        assert row["instruction_text"] == "Extract principle"
        assert row["parent_variant"] == "D"
        assert row["strategy"] == "feedback"


class TestLoadAllVariantConfigs:
    """Merges VARIANT_CONFIGS (code) with prompt_variants (DB)."""

    def test_returns_hand_authored_variants(self, db_path):
        """With empty DB, returns only VARIANT_CONFIGS."""
        conn = init_db(db_path)
        merged = load_all_variant_configs(conn)
        assert "A" in merged
        assert "D" in merged
        assert "_apo_generated" not in merged["A"]

    def test_includes_db_variants(self, db_path):
        """DB-stored variants appear in merged result."""
        conn = init_db(db_path)
        config = {
            "model": "qwen3:14b",
            "temperature": 0.6,
            "num_ctx": 8192,
            "chunked": False,
            "prompt_id": "apo-generated",
        }
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("X01", "Test instruction", json.dumps(config), "feedback", "2026-03-11"),
        )
        conn.commit()
        merged = load_all_variant_configs(conn)
        assert "X01" in merged
        assert merged["X01"]["_instruction_text"] == "Test instruction"
        assert merged["X01"]["_apo_generated"] is True
        assert merged["X01"]["model"] == "qwen3:14b"

    def test_db_does_not_override_hand_authored(self, db_path):
        """DB variant with same ID as hand-authored does NOT replace it."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("A", "Override attempt", '{"model":"fake"}', "feedback", "2026-03-11"),
        )
        conn.commit()
        merged = load_all_variant_configs(conn)
        # Hand-authored "A" should be preserved, DB row ignored
        assert merged["A"]["model"] == "deepseek-r1:8b"
        assert "_apo_generated" not in merged["A"]


class TestParseCandidates:
    """Parse optimizer LLM response into instruction candidates."""

    def test_valid_json_array(self):
        """Standard JSON array of candidates."""
        response = '[{"instruction": "Do X", "hypothesis": "because Y"}]'
        result = parse_optimizer_candidates(response)
        assert len(result) == 1
        assert result[0]["instruction"] == "Do X"
        assert result[0]["hypothesis"] == "because Y"

    def test_multiple_candidates(self):
        """Multiple candidates parsed correctly."""
        response = json.dumps(
            [
                {"instruction": "First", "hypothesis": "H1"},
                {"instruction": "Second", "hypothesis": "H2"},
                {"instruction": "Third", "hypothesis": "H3"},
            ]
        )
        result = parse_optimizer_candidates(response)
        assert len(result) == 3

    def test_json_embedded_in_text(self):
        """JSON array embedded in surrounding text/reasoning."""
        response = 'Here are my suggestions:\n[{"instruction": "X", "hypothesis": "Y"}]\nDone.'
        result = parse_optimizer_candidates(response)
        assert len(result) == 1
        assert result[0]["instruction"] == "X"

    def test_invalid_json_returns_empty(self):
        """Unparseable response returns empty list."""
        result = parse_optimizer_candidates("This is not JSON at all")
        assert result == []

    def test_missing_instruction_key_skipped(self):
        """Candidates without 'instruction' key are filtered out."""
        response = json.dumps(
            [
                {"instruction": "Valid", "hypothesis": "H"},
                {"text": "Missing instruction key", "hypothesis": "H2"},
            ]
        )
        result = parse_optimizer_candidates(response)
        assert len(result) == 1
        assert result[0]["instruction"] == "Valid"

    def test_think_tags_stripped(self):
        """<think> blocks stripped before parsing."""
        response = '<think>reasoning</think>[{"instruction": "X", "hypothesis": "Y"}]'
        result = parse_optimizer_candidates(response)
        assert len(result) == 1

    def test_empty_response_returns_empty(self):
        """None or empty string returns empty list."""
        assert parse_optimizer_candidates("") == []
        assert parse_optimizer_candidates(None) == []
