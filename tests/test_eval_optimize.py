"""Tests for APO eval-optimize: prompt_variants table, optimizer strategies, variant registration."""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from lessons_db.db import init_db
from lessons_db.eval.generate import run_eval_generate
from lessons_db.eval.optimize import (
    build_feedback_prompt,
    build_opro_prompt,
    load_all_variant_configs,
    next_x_id,
    parse_optimizer_candidates,
    register_apo_variant,
)


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

    def test_corrupt_config_json_skipped(self, db_path):
        """DB variant with corrupt JSON is skipped, not crash the whole load."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("X01", "Good instruction", "NOT VALID JSON{{{", "feedback", "2026-03-11"),
        )
        conn.commit()
        merged = load_all_variant_configs(conn)
        # Corrupt row should be skipped, hand-authored variants still present
        assert "X01" not in merged
        assert "A" in merged

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


class TestBuildApoPrompt:
    """APO-generated prompts use stored instruction text."""

    def test_override_replaces_default_prompt(self):
        """When prompt_overrides has the variant, uses stored instruction."""
        from lessons_db.eval.prompts import build_generation_prompt

        lesson = {"title": "Test Lesson", "one_liner": "Fix the bug", "description": "Description here"}
        overrides = {"X01": "Custom instruction for extracting principles."}
        result = build_generation_prompt("X01", lesson, prompt_overrides=overrides)
        assert "Custom instruction for extracting principles." in result
        assert "Test Lesson" in result
        assert "Return ONLY" in result

    def test_no_override_falls_through(self):
        """Without override, variant A uses the existing fewshot prompt."""
        from lessons_db.eval.prompts import build_generation_prompt

        lesson = {"title": "Test", "one_liner": "Bug", "description": "Desc"}
        result = build_generation_prompt("A", lesson, prompt_overrides={})
        assert "transferable principle" in result

    def test_override_dict_none_falls_through(self):
        """prompt_overrides=None uses existing dispatch."""
        from lessons_db.eval.prompts import build_generation_prompt

        lesson = {"title": "Test", "one_liner": "Bug", "description": "Desc"}
        result = build_generation_prompt("A", lesson, prompt_overrides=None)
        assert "transferable principle" in result

    def test_apo_prompt_contains_lesson_context(self):
        """APO prompt injects title, one_liner, description."""
        from lessons_db.eval.prompts import build_generation_prompt

        lesson = {"title": "My Title", "one_liner": "My liner", "description": "My desc"}
        overrides = {"X01": "Instruction preamble here."}
        result = build_generation_prompt("X01", lesson, prompt_overrides=overrides)
        assert "My Title" in result
        assert "My liner" in result
        assert "My desc" in result

    def test_apo_prompt_has_fixed_suffix(self):
        """APO prompt always ends with 'Return ONLY the principle statement.'"""
        from lessons_db.eval.prompts import build_generation_prompt

        lesson = {"title": "T", "one_liner": "O", "description": "D"}
        overrides = {"X01": "Any instruction."}
        result = build_generation_prompt("X01", lesson, prompt_overrides=overrides)
        assert "Return ONLY the principle statement" in result


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


class TestFeedbackStrategy:
    """Feedback strategy: shows false positives to optimizer."""

    def test_includes_current_instruction(self):
        """Prompt contains the current best instruction text."""
        prompt = build_feedback_prompt(
            instruction_text="Current instruction here",
            f1=0.47,
            false_positives=[
                {
                    "principle": "P1",
                    "target_title": "T1",
                    "target_cluster_seed": "X",
                    "cluster_seed": "Y",
                },
            ],
            n_candidates=3,
        )
        assert "Current instruction here" in prompt
        assert "0.47" in prompt

    def test_includes_false_positives(self):
        """Prompt lists false positive examples."""
        fps = [
            {
                "principle": "Too broad principle",
                "target_title": "Unrelated lesson",
                "target_cluster_seed": "cluster_B",
                "cluster_seed": "cluster_A",
            },
        ]
        prompt = build_feedback_prompt(
            instruction_text="Instr",
            f1=0.3,
            false_positives=fps,
            n_candidates=3,
        )
        assert "Too broad principle" in prompt
        assert "Unrelated lesson" in prompt

    def test_requests_n_candidates(self):
        """Prompt asks for the specified number of candidates."""
        prompt = build_feedback_prompt(
            instruction_text="I",
            f1=0.3,
            false_positives=[],
            n_candidates=5,
        )
        assert "5" in prompt

    def test_requests_json_format(self):
        """Prompt asks for JSON array output."""
        prompt = build_feedback_prompt(
            instruction_text="I",
            f1=0.3,
            false_positives=[],
            n_candidates=3,
        )
        assert "JSON" in prompt


class TestOproStrategy:
    """OPRO strategy: shows sorted prompt-score pairs."""

    def test_includes_prompts_sorted_ascending(self):
        """Past prompts appear sorted by F1 ascending (worst first)."""
        history = [
            {"instruction_text": "Best prompt", "f1": 0.47},
            {"instruction_text": "Worst prompt", "f1": 0.10},
            {"instruction_text": "Middle prompt", "f1": 0.28},
        ]
        prompt = build_opro_prompt(history=history, n_candidates=3)
        # Worst should appear before best in the prompt
        worst_pos = prompt.index("Worst prompt")
        best_pos = prompt.index("Best prompt")
        assert worst_pos < best_pos

    def test_includes_f1_scores(self):
        """Each prompt is labeled with its F1 score."""
        history = [{"instruction_text": "P1", "f1": 0.47}]
        prompt = build_opro_prompt(history=history, n_candidates=3)
        assert "0.47" in prompt

    def test_requests_json_format(self):
        """Prompt asks for JSON array output."""
        prompt = build_opro_prompt(
            history=[{"instruction_text": "P", "f1": 0.5}],
            n_candidates=3,
        )
        assert "JSON" in prompt


class TestVariantRegistration:
    """Register APO-generated variants in the DB."""

    def test_next_x_id_starts_at_x01(self, db_path):
        """First X-ID is X01."""
        conn = init_db(db_path)
        assert next_x_id(conn) == "X01"

    def test_next_x_id_increments(self, db_path):
        """After X01 exists, next is X02."""
        conn = init_db(db_path)
        conn.execute(
            """INSERT INTO prompt_variants
               (variant_id, instruction_text, config_json, strategy, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("X01", "t", "{}", "feedback", "2026-01-01"),
        )
        conn.commit()
        assert next_x_id(conn) == "X02"

    def test_next_x_id_skips_gaps(self, db_path):
        """If X01 and X03 exist, next is X02 (fills gaps)."""
        conn = init_db(db_path)
        for xid in ["X01", "X03"]:
            conn.execute(
                """INSERT INTO prompt_variants
                   (variant_id, instruction_text, config_json, strategy, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (xid, "t", "{}", "feedback", "2026-01-01"),
            )
        conn.commit()
        assert next_x_id(conn) == "X02"

    def test_register_writes_to_db(self, db_path):
        """register_apo_variant writes a row and returns the variant_id."""
        conn = init_db(db_path)
        vid = register_apo_variant(
            conn,
            instruction_text="New instruction",
            parent_variant="D",
            strategy="feedback",
            optimizer_model="qwen3:14b",
            hypothesis="reduce false positives",
            config_overrides={"model": "qwen3:14b"},
        )
        assert vid.startswith("X")
        row = conn.execute("SELECT * FROM prompt_variants WHERE variant_id = ?", (vid,)).fetchone()
        assert row is not None
        assert row["instruction_text"] == "New instruction"
        assert row["parent_variant"] == "D"

    def test_register_uses_parent_config(self, db_path):
        """config_json inherits from parent variant config."""
        conn = init_db(db_path)
        vid = register_apo_variant(
            conn,
            instruction_text="Instr",
            parent_variant="D",
            strategy="feedback",
        )
        row = conn.execute("SELECT config_json FROM prompt_variants WHERE variant_id = ?", (vid,)).fetchone()
        config = json.loads(row["config_json"])
        assert config["model"] == "qwen3:14b"  # inherited from D
        assert config["prompt_id"] == "apo-generated"

    def test_register_chain_inherits_from_db_parent(self, db_path):
        """X02 as child of X01 inherits X01's DB config, not variant D's."""
        conn = init_db(db_path)
        # Register X01 as child of D with a custom override
        x01 = register_apo_variant(
            conn,
            instruction_text="First gen instruction",
            parent_variant="D",
            strategy="feedback",
            config_overrides={"temperature": 0.2},
        )
        # Register X02 as child of X01 (DB-stored parent)
        x02 = register_apo_variant(
            conn,
            instruction_text="Second gen instruction",
            parent_variant=x01,
            strategy="feedback",
        )
        row = conn.execute("SELECT config_json FROM prompt_variants WHERE variant_id = ?", (x02,)).fetchone()
        config = json.loads(row["config_json"])
        # Should inherit X01's temperature override, not D's default
        assert config["temperature"] == 0.2
        assert config["prompt_id"] == "apo-generated"

    def test_register_unknown_parent_raises(self, db_path):
        """Unknown parent variant raises ValueError, not silent fallback."""
        conn = init_db(db_path)
        with pytest.raises(ValueError, match="not found"):
            register_apo_variant(
                conn,
                instruction_text="Should fail",
                parent_variant="NONEXISTENT",
                strategy="feedback",
            )


class TestGenerateWithApoVariants:
    """run_eval_generate uses prompt_overrides for APO variants."""

    def test_apo_variant_generates_with_stored_instruction(self, db_path, tmp_path):
        """APO variant X01 uses instruction from prompt_variants, not VARIANT_CONFIGS."""
        from tests.test_eval import _seed_clusters

        conn = init_db(db_path)
        _seed_clusters(conn)
        output_path = tmp_path / "results.json"

        # Register an APO variant
        vid = register_apo_variant(
            conn,
            instruction_text="CUSTOM APO INSTRUCTION for testing.",
            parent_variant="D",
            strategy="feedback",
        )

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "APO Principle."}).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            run_eval_generate(
                conn=conn,
                queue_url="http://localhost:7683",
                variants=[vid],
                per_cluster=1,
                output_path=output_path,
                resume=False,
            )

        # Verify the request body contained our custom instruction
        call_args = mock_url.call_args
        request_body = json.loads(call_args[0][0].data.decode("utf-8"))
        assert "CUSTOM APO INSTRUCTION" in request_body["prompt"]

        # Verify results were written
        data = json.loads(output_path.read_text())
        assert any(r["variant"] == vid for r in data["results"])


class TestEvalOptimizeEndToEnd:
    """Full loop: load history -> optimize -> register -> eval -> record."""

    def test_feedback_loop_registers_and_evaluates(self, db_path, tmp_path):
        """Feedback strategy: produces candidates, registers them, runs eval."""
        from click.testing import CliRunner

        from lessons_db.cli import main
        from tests.test_eval import _seed_clusters

        conn = init_db(db_path)
        _seed_clusters(conn)

        # Mock LLM responses at urllib level.
        # call_ollama wraps urllib.request.urlopen and parses {"response": "..."}.
        # Call order: 1=optimizer, then N=generator calls, then M=judge calls.
        optimizer_candidates = json.dumps(
            [
                {"instruction": "Improved instruction 1", "hypothesis": "H1"},
            ]
        )
        # Optimizer response — call_ollama expects {"response": "<text>"}
        optimizer_body = json.dumps({"response": optimizer_candidates})
        # Generator response
        generator_body = json.dumps({"response": "Generated principle."})
        # Judge response — parse_judge_scores expects {"transfer": N, ...} inside text
        judge_body = json.dumps({"response": '{"transfer": 3, "precision": 3, "actionability": 3}'})

        call_count = {"n": 0}

        def mock_urlopen(req, *args, **kwargs):
            call_count["n"] += 1
            resp = MagicMock()
            if call_count["n"] == 1:
                # First call is the optimizer
                resp.read.return_value = optimizer_body.encode("utf-8")
            elif "judge" not in (req.data.decode("utf-8") if hasattr(req, "data") and req.data else ""):
                # Generator calls — check source field to distinguish
                data = json.loads(req.data.decode("utf-8")) if req.data else {}
                if data.get("_source") == "eval-judge":
                    resp.read.return_value = judge_body.encode("utf-8")
                else:
                    resp.read.return_value = generator_body.encode("utf-8")
            else:
                resp.read.return_value = judge_body.encode("utf-8")
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--db",
                    str(db_path),
                    "meta",
                    "eval-optimize",
                    "--strategy",
                    "feedback",
                    "--candidates",
                    "1",
                    "--max-iterations",
                    "1",
                    "--per-cluster",
                    "1",
                ],
            )

        # Debug output on failure
        if result.exit_code != 0:
            raise AssertionError(
                f"CLI exited with code {result.exit_code}\n"
                f"Output: {result.output}\n"
                f"Exception: {result.exception}"
            ) from result.exception

        # Should have registered at least one X-variant
        rows = conn.execute("SELECT variant_id FROM prompt_variants").fetchall()
        assert len(rows) >= 1
        assert rows[0]["variant_id"].startswith("X")
