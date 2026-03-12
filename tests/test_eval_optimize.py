"""Tests for APO eval-optimize: prompt_variants table, optimizer strategies, variant registration."""

import json
import sqlite3

import pytest

from lessons_db.db import init_db
from lessons_db.eval.optimize import load_all_variant_configs


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


class TestGetInstructionText:
    """Extract instruction preamble from hand-authored prompt builders."""

    def test_variant_a_returns_fewshot_preamble(self):
        """Variant A instruction contains the few-shot examples."""
        from lessons_db.eval.prompts import get_instruction_text

        text = get_instruction_text("A")
        assert "transferable principle" in text
        assert "Resources acquired in callbacks" in text
        assert "Lesson:" not in text  # must NOT contain lesson placeholder

    def test_variant_b_returns_causal_preamble(self):
        """Variant B instruction contains causal framing."""
        from lessons_db.eval.prompts import get_instruction_text

        text = get_instruction_text("B")
        assert "causal statement" in text.lower() or "structural principle" in text.lower()
        assert "Lesson:" not in text

    def test_variant_f_returns_contrastive_preamble(self):
        """Variant F instruction references contrastive discrimination."""
        from lessons_db.eval.prompts import get_instruction_text

        text = get_instruction_text("F")
        assert "SAME PATTERN" in text or "DISTINGUISH" in text or "contrastive" in text.lower()

    def test_unknown_variant_raises(self):
        """Unknown variant ID raises KeyError."""
        from lessons_db.eval.prompts import get_instruction_text

        with pytest.raises(KeyError):
            get_instruction_text("UNKNOWN")

    def test_all_hand_authored_variants_covered(self):
        """Every variant in VARIANT_CONFIGS has an instruction text."""
        from lessons_db.eval.prompts import get_instruction_text
        from lessons_db.eval.variants import VARIANT_CONFIGS as VC

        for vid in VC:
            text = get_instruction_text(vid)
            assert isinstance(text, str)
            assert len(text) > 20, f"Variant {vid} instruction too short"
