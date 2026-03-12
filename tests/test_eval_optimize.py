"""Tests for APO eval-optimize: prompt_variants table, optimizer strategies, variant registration."""

import sqlite3

import pytest

from lessons_db.db import init_db


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
