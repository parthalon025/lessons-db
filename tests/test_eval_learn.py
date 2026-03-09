"""Tests for eval/learn.py — always-on post-judge learning step."""

import json
from pathlib import Path

import pytest

from lessons_db.eval.learn import (
    _config_diff_vs_control,
    _diagnose,
    append_to_program_md,
    derive_insights,
    run_eval_learn,
    save_learnings,
)

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_VARIANT_CONFIGS = {
    "A": {"model": "deepseek-r1:8b", "temperature": 0.7, "num_ctx": 4096, "chunked": False},
    "B": {"model": "deepseek-r1:8b", "temperature": 0.6, "num_ctx": 8192, "chunked": False},
    "D": {"model": "qwen3:14b", "temperature": 0.6, "num_ctx": 8192, "chunked": False},
    "F": {"model": "deepseek-r1:8b", "temperature": 0.6, "num_ctx": 8192, "chunked": False, "contrastive": True},
}


# ---------------------------------------------------------------------------
# _diagnose
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_too_broad_high_recall_low_precision(self):
        diag, rec = _diagnose(recall=0.90, precision=0.15)
        assert "too broad" in diag
        assert "contrastive" in rec

    def test_too_narrow_low_recall_high_precision(self):
        diag, rec = _diagnose(recall=0.30, precision=0.70)
        assert "too narrow" in diag
        assert "context window" in rec or "chunking" in rec

    def test_both_low_ineffective(self):
        diag, rec = _diagnose(recall=0.20, precision=0.10)
        assert "ineffective" in diag
        assert "model" in rec or "prompt" in rec

    def test_both_high_balanced(self):
        diag, rec = _diagnose(recall=0.75, precision=0.55)
        assert "balanced" in diag
        assert "promote" in rec

    def test_boundary_high_recall(self):
        # recall=0.65 hits _HIGH_RECALL, precision=0.25 hits _LOW_PRECISION → too broad
        diag, _ = _diagnose(recall=0.65, precision=0.25)
        assert "too broad" in diag

    def test_returns_two_strings(self):
        result = _diagnose(0.5, 0.5)
        assert isinstance(result, tuple) and len(result) == 2
        assert all(isinstance(s, str) for s in result)


# ---------------------------------------------------------------------------
# _config_diff_vs_control
# ---------------------------------------------------------------------------


class TestConfigDiff:
    def test_control_returns_baseline(self):
        diff = _config_diff_vs_control("A", _VARIANT_CONFIGS)
        assert diff == "baseline (control)"

    def test_model_change_detected(self):
        diff = _config_diff_vs_control("D", _VARIANT_CONFIGS)
        assert "qwen3:14b" in diff

    def test_contrastive_flag_detected(self):
        diff = _config_diff_vs_control("F", _VARIANT_CONFIGS)
        assert "contrastive=True" in diff

    def test_unknown_variant_graceful(self):
        diff = _config_diff_vs_control("ZZ", _VARIANT_CONFIGS)
        assert "unknown" in diff

    def test_same_config_returns_same_as_control(self):
        # B differs only in temperature and num_ctx from A
        diff = _config_diff_vs_control("B", _VARIANT_CONFIGS)
        assert "num_ctx" in diff or "temperature" in diff or "temp" in diff


# ---------------------------------------------------------------------------
# derive_insights
# ---------------------------------------------------------------------------


class TestDeriveInsights:
    def test_returns_one_insight_per_variant(self):
        metrics = {
            "A": {"f1": 0.28, "recall": 0.93, "precision": 0.17},
            "D": {"f1": 0.47, "recall": 0.79, "precision": 0.33},
        }
        insights = derive_insights(metrics, _VARIANT_CONFIGS)
        assert len(insights) == 2

    def test_win_status_for_new_best(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"D": {"f1": 0.55, "recall": 0.70, "precision": 0.45}}
        insights = derive_insights(metrics, _VARIANT_CONFIGS)
        assert insights[0]["status"] == "win"
        assert "NEW BEST" in insights[0]["summary"]

    def test_miss_status_below_threshold(self, tmp_path, monkeypatch):
        # Pre-seed best.json with a high F1 so D is a miss
        best_json = tmp_path / "best.json"
        best_json.write_text(json.dumps({"variant": "X", "f1": 0.80, "recall": 0.8, "precision": 0.8}))
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", best_json)
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"D": {"f1": 0.47, "recall": 0.79, "precision": 0.33}}
        insights = derive_insights(metrics, _VARIANT_CONFIGS)
        assert insights[0]["status"] == "miss"

    def test_near_miss_within_threshold(self, tmp_path, monkeypatch):
        best_json = tmp_path / "best.json"
        best_json.write_text(json.dumps({"variant": "X", "f1": 0.50, "recall": 0.7, "precision": 0.4}))
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", best_json)
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"D": {"f1": 0.48, "recall": 0.79, "precision": 0.33}}  # delta = -0.02
        insights = derive_insights(metrics, _VARIANT_CONFIGS)
        assert insights[0]["status"] == "near-miss"

    def test_always_runs_even_with_zero_f1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"A": {"f1": 0.0, "recall": 0.0, "precision": 0.0}}
        insights = derive_insights(metrics, _VARIANT_CONFIGS)
        assert len(insights) == 1
        assert insights[0]["f1"] == 0.0

    def test_sorted_descending_by_f1(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {
            "A": {"f1": 0.28, "recall": 0.93, "precision": 0.17},
            "D": {"f1": 0.47, "recall": 0.79, "precision": 0.33},
            "F": {"f1": 0.35, "recall": 0.60, "precision": 0.26},
        }
        insights = derive_insights(metrics, _VARIANT_CONFIGS)
        f1s = [i["f1"] for i in insights]
        assert f1s == sorted(f1s, reverse=True)

    def test_rolling_best_updates_within_run(self, tmp_path, monkeypatch):
        """F (0.55) processes first (sorted by F1 desc), wins. D (0.47) is then a miss vs F's bar."""
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {
            "D": {"f1": 0.47, "recall": 0.79, "precision": 0.33},
            "F": {"f1": 0.55, "recall": 0.70, "precision": 0.45},
        }
        insights = derive_insights(metrics, _VARIANT_CONFIGS)
        statuses = {i["variant"]: i["status"] for i in insights}
        # F processes first (highest F1), wins over prior best=0.0
        # D then compared against F's new bar of 0.55 — is a miss
        assert statuses["F"] == "win"
        assert statuses["D"] == "miss"


# ---------------------------------------------------------------------------
# save_learnings
# ---------------------------------------------------------------------------


class TestSaveLearnings:
    def test_appends_to_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        insights = [{"variant": "D", "f1": 0.47, "status": "win", "summary": "test"}]
        save_learnings(insights)
        lines = (tmp_path / "learnings.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["variant"] == "D"

    def test_multiple_appends_accumulate(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        save_learnings([{"variant": "A", "f1": 0.28, "status": "miss"}])
        save_learnings([{"variant": "D", "f1": 0.47, "status": "win"}])
        lines = (tmp_path / "learnings.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# append_to_program_md
# ---------------------------------------------------------------------------


class TestAppendToProgramMd:
    def _make_program_md(self, tmp_path: Path, with_italic: bool = True) -> Path:
        p = tmp_path / "program.md"
        italic = "*(This section is written by the agent)*\n" if with_italic else ""
        p.write_text(f"# autoresearch\n\nSome content.\n\n## Learned so far\n\n{italic}")
        return p

    def test_inserts_after_section_header(self, tmp_path):
        p = self._make_program_md(tmp_path)
        insights = [
            {"date": "2026-03-09", "variant": "F", "summary": "test win", "diagnosis": "diag", "recommendation": "rec"}
        ]
        result = append_to_program_md(insights, p)
        assert result is True
        content = p.read_text()
        assert "- 2026-03-09: [F] test win" in content

    def test_section_not_found_returns_false(self, tmp_path):
        p = tmp_path / "program.md"
        p.write_text("# No section here\n")
        result = append_to_program_md(
            [{"date": "2026-03-09", "variant": "A", "summary": "x", "diagnosis": "d", "recommendation": "r"}], p
        )
        assert result is False

    def test_file_not_found_returns_false(self, tmp_path):
        result = append_to_program_md([], tmp_path / "nonexistent.md")
        assert result is False

    def test_new_entries_appear_before_existing(self, tmp_path):
        p = tmp_path / "program.md"
        p.write_text("## Learned so far\n\n- 2026-03-08: [A] old entry\n")
        insights = [
            {"date": "2026-03-09", "variant": "F", "summary": "new entry", "diagnosis": "d", "recommendation": "r"}
        ]
        append_to_program_md(insights, p)
        lines = p.read_text().splitlines()
        f_idx = next(i for i, l in enumerate(lines) if "[F]" in l)
        a_idx = next(i for i, l in enumerate(lines) if "[A]" in l)
        assert f_idx < a_idx  # newest first


# ---------------------------------------------------------------------------
# run_eval_learn (integration)
# ---------------------------------------------------------------------------


class TestRunEvalLearn:
    def test_returns_insights_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"D": {"f1": 0.47, "recall": 0.79, "precision": 0.33}}
        insights = run_eval_learn(metrics, _VARIANT_CONFIGS)
        assert isinstance(insights, list)
        assert len(insights) == 1

    def test_crash_variant_still_runs(self, tmp_path, monkeypatch):
        """Zero-F1 result (crash-adjacent) still produces an insight."""
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"X01": {"f1": 0.0, "recall": 0.0, "precision": 0.0}}
        insights = run_eval_learn(metrics, _VARIANT_CONFIGS)
        assert len(insights) == 1
        assert insights[0]["f1"] == 0.0

    def test_updates_program_md_when_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        program_md = tmp_path / "program.md"
        program_md.write_text("# test\n\n## Learned so far\n\n")
        metrics = {"D": {"f1": 0.47, "recall": 0.79, "precision": 0.33}}
        run_eval_learn(metrics, _VARIANT_CONFIGS, program_md_path=program_md)
        content = program_md.read_text()
        assert "[D]" in content

    def test_survives_missing_program_md(self, tmp_path, monkeypatch):
        """No crash when program.md does not exist."""
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"D": {"f1": 0.47, "recall": 0.79, "precision": 0.33}}
        # Should not raise
        insights = run_eval_learn(metrics, _VARIANT_CONFIGS, program_md_path=tmp_path / "nonexistent.md")
        assert len(insights) == 1

    def test_learnings_persisted_to_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        learnings = tmp_path / "learnings.jsonl"
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", learnings)
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"F": {"f1": 0.51, "recall": 0.65, "precision": 0.42}}
        run_eval_learn(metrics, _VARIANT_CONFIGS)
        assert learnings.exists()
        record = json.loads(learnings.read_text().strip())
        assert record["variant"] == "F"
        assert record["f1"] == pytest.approx(0.51)
