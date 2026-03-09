"""Tests for eval/learn.py — always-on post-judge learning step."""

import json
from pathlib import Path

import pytest

from lessons_db.eval.learn import (
    _config_diff_vs_control,
    _diagnose,
    _normalize_cfg_value,
    append_to_program_md,
    compute_ablations,
    compute_dimension_impacts,
    compute_variant_trends,
    derive_insights,
    format_ablation_summary,
    load_learnings,
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

    def test_middle_zone_moderate(self):
        # recall=0.55 is between LOW (0.45) and HIGH (0.65) → moderate, not balanced
        diag, rec = _diagnose(recall=0.55, precision=0.30)
        assert "moderate" in diag
        assert "ablation" in rec

    def test_one_metric_below_high_is_moderate(self):
        # recall is above HIGH but precision below HIGH → moderate
        diag, _ = _diagnose(recall=0.70, precision=0.35)
        assert "moderate" in diag

    def test_both_above_high_is_balanced(self):
        # Both must be >= HIGH to qualify as balanced
        diag, _ = _diagnose(recall=0.65, precision=0.40)
        assert "balanced" in diag

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
    def test_returns_insights_and_ablations(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"D": {"f1": 0.47, "recall": 0.79, "precision": 0.33}}
        insights, ablations, analysis = run_eval_learn(metrics, _VARIANT_CONFIGS)
        assert isinstance(insights, list)
        assert len(insights) == 1
        # Single variant → no ablation pairs
        assert ablations == []
        # No scored_pairs → empty analysis
        assert analysis == {}

    def test_crash_variant_still_runs(self, tmp_path, monkeypatch):
        """Zero-F1 result (crash-adjacent) still produces an insight."""
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"X01": {"f1": 0.0, "recall": 0.0, "precision": 0.0}}
        insights, _ablations, _analysis = run_eval_learn(metrics, _VARIANT_CONFIGS)
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
        insights, _ablations, _analysis = run_eval_learn(
            metrics, _VARIANT_CONFIGS, program_md_path=tmp_path / "nonexistent.md"
        )
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

    def test_multi_variant_returns_ablations(self, tmp_path, monkeypatch):
        """Variants differing by one config dimension produce ablation pairs."""
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        # B and F differ only on contrastive flag → valid ablation pair
        metrics = {
            "B": {"f1": 0.40, "recall": 0.70, "precision": 0.28},
            "F": {"f1": 0.52, "recall": 0.65, "precision": 0.42},
        }
        insights, ablations, _analysis = run_eval_learn(metrics, _VARIANT_CONFIGS)
        assert len(insights) == 2
        assert len(ablations) > 0
        assert all("dimension" in ab for ab in ablations)

    def test_analysis_included_when_scored_pairs_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        metrics = {"A": {"f1": 0.28, "recall": 0.93, "precision": 0.17}}
        scored_pairs = [
            {
                "variant": "A",
                "source_lesson_id": 1,
                "is_same_cluster": True,
                "scores": {"matched": True},
                "principle": "p",
                "target_id": 10,
            },
            {
                "variant": "A",
                "source_lesson_id": 1,
                "is_same_cluster": False,
                "scores": {"matched": True},
                "principle": "p",
                "target_id": 20,
            },
        ]
        insights, ablations, analysis = run_eval_learn(metrics, _VARIANT_CONFIGS, scored_pairs=scored_pairs)
        assert len(insights) == 1
        assert "per_lesson" in analysis
        assert "failure_cases" in analysis
        assert "confidence_intervals" in analysis


# ---------------------------------------------------------------------------
# compute_ablations
# ---------------------------------------------------------------------------


class TestComputeAblations:
    def test_finds_single_dimension_diff(self):
        """B and F differ only by contrastive flag → one ablation on that dimension."""
        metrics = {
            "B": {"f1": 0.40, "recall": 0.70, "precision": 0.28},
            "F": {"f1": 0.52, "recall": 0.65, "precision": 0.42},
        }
        ablations = compute_ablations(metrics, _VARIANT_CONFIGS)
        dims = [ab["dimension"] for ab in ablations]
        assert "contrastive" in dims

    def test_ablation_delta_is_correct(self):
        """delta_f1 = f1_b - f1_a (order determined by enumeration)."""
        metrics = {
            "B": {"f1": 0.40, "recall": 0.70, "precision": 0.28},
            "F": {"f1": 0.52, "recall": 0.65, "precision": 0.42},
        }
        ablations = compute_ablations(metrics, _VARIANT_CONFIGS)
        contrastive_ab = next(ab for ab in ablations if ab["dimension"] == "contrastive")
        assert abs(contrastive_ab["delta_f1"]) == pytest.approx(0.12, abs=0.001)

    def test_sorted_by_absolute_delta_descending(self):
        """Ablations are sorted by most impactful first."""
        metrics = {
            "A": {"f1": 0.28, "recall": 0.93, "precision": 0.17},
            "B": {"f1": 0.40, "recall": 0.70, "precision": 0.28},
            "D": {"f1": 0.47, "recall": 0.79, "precision": 0.33},
            "F": {"f1": 0.52, "recall": 0.65, "precision": 0.42},
        }
        ablations = compute_ablations(metrics, _VARIANT_CONFIGS)
        deltas = [abs(ab["delta_f1"]) for ab in ablations]
        assert deltas == sorted(deltas, reverse=True)

    def test_empty_metrics_returns_empty(self):
        assert compute_ablations({}, _VARIANT_CONFIGS) == []

    def test_single_variant_returns_empty(self):
        metrics = {"A": {"f1": 0.28, "recall": 0.93, "precision": 0.17}}
        assert compute_ablations(metrics, _VARIANT_CONFIGS) == []

    def test_unknown_variant_skipped(self):
        """Variant not in VARIANT_CONFIGS is silently skipped."""
        metrics = {
            "A": {"f1": 0.28, "recall": 0.93, "precision": 0.17},
            "ZZ": {"f1": 0.50, "recall": 0.60, "precision": 0.40},
        }
        ablations = compute_ablations(metrics, _VARIANT_CONFIGS)
        variants_in_ablations = {ab["variant_a"] for ab in ablations} | {ab["variant_b"] for ab in ablations}
        assert "ZZ" not in variants_in_ablations


# ---------------------------------------------------------------------------
# format_ablation_summary
# ---------------------------------------------------------------------------


class TestFormatAblationSummary:
    def test_formats_top_n(self):
        ablations = [
            {
                "dimension": "contrastive",
                "from": False,
                "to": True,
                "delta_f1": 0.12,
                "variant_a": "B",
                "variant_b": "F",
                "f1_a": 0.40,
                "f1_b": 0.52,
            },
            {
                "dimension": "model",
                "from": "deepseek-r1:8b",
                "to": "qwen3:14b",
                "delta_f1": 0.07,
                "variant_a": "B",
                "variant_b": "D",
                "f1_a": 0.40,
                "f1_b": 0.47,
            },
        ]
        lines = format_ablation_summary(ablations, top_n=1)
        assert len(lines) == 1
        assert "contrastive" in lines[0]
        assert "+0.120" in lines[0]

    def test_negative_delta_no_plus(self):
        ablations = [
            {
                "dimension": "temperature",
                "from": 0.7,
                "to": 0.6,
                "delta_f1": -0.05,
                "variant_a": "A",
                "variant_b": "B",
                "f1_a": 0.50,
                "f1_b": 0.45,
            },
        ]
        lines = format_ablation_summary(ablations)
        assert len(lines) == 1
        assert "+0" not in lines[0]  # negative delta should not have +
        assert "-0.050" in lines[0]

    def test_empty_ablations(self):
        assert format_ablation_summary([]) == []


# ---------------------------------------------------------------------------
# _normalize_cfg_value
# ---------------------------------------------------------------------------


class TestNormalizeCfgValue:
    def test_boolean_flag_none_becomes_false(self):
        assert _normalize_cfg_value("contrastive", None) is False

    def test_boolean_flag_true_stays_true(self):
        assert _normalize_cfg_value("contrastive", True) is True

    def test_boolean_flag_false_stays_false(self):
        assert _normalize_cfg_value("chunked", False) is False

    def test_non_boolean_key_passes_through(self):
        assert _normalize_cfg_value("model", "deepseek-r1:8b") == "deepseek-r1:8b"

    def test_non_boolean_key_none_stays_none(self):
        assert _normalize_cfg_value("temperature", None) is None


# ---------------------------------------------------------------------------
# save_learnings with ablations
# ---------------------------------------------------------------------------


class TestSaveLearningsWithAblations:
    def test_ablation_entry_appended(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        insights = [{"variant": "B", "date": "2026-03-09", "f1": 0.40}]
        ablations = [{"dimension": "contrastive", "from": False, "to": True, "delta_f1": 0.12}]
        save_learnings(insights, ablations)
        lines = (tmp_path / "learnings.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2  # 1 insight + 1 ablation entry
        ab_record = json.loads(lines[1])
        assert ab_record["type"] == "ablations"
        assert len(ab_record["ablations"]) == 1

    def test_no_ablation_entry_when_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "learnings.jsonl")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        save_learnings([{"variant": "A", "date": "2026-03-09"}])
        lines = (tmp_path / "learnings.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# append_to_program_md with ablations
# ---------------------------------------------------------------------------


class TestAppendToProgramMdWithAblations:
    def test_ablation_line_included(self, tmp_path):
        p = tmp_path / "program.md"
        p.write_text("## Learned so far\n\n")
        insights = [{"date": "2026-03-09", "variant": "F", "summary": "win", "diagnosis": "d", "recommendation": "r"}]
        ablations = [
            {
                "dimension": "contrastive",
                "from": False,
                "to": True,
                "delta_f1": 0.12,
                "variant_a": "B",
                "variant_b": "F",
                "f1_a": 0.40,
                "f1_b": 0.52,
            },
        ]
        append_to_program_md(insights, p, ablations)
        content = p.read_text()
        assert "[ABLATION]" in content
        assert "contrastive" in content

    def test_no_ablation_line_when_empty(self, tmp_path):
        p = tmp_path / "program.md"
        p.write_text("## Learned so far\n\n")
        insights = [{"date": "2026-03-09", "variant": "A", "summary": "x", "diagnosis": "d", "recommendation": "r"}]
        append_to_program_md(insights, p, ablations=[])
        content = p.read_text()
        assert "[ABLATION]" not in content


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


class TestLoadLearnings:
    def test_loads_entries(self, tmp_path, monkeypatch):
        f = tmp_path / "learnings.jsonl"
        f.write_text('{"variant": "A", "f1": 0.28}\n{"variant": "B", "f1": 0.40}\n')
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", f)
        entries = load_learnings()
        assert len(entries) == 2

    def test_skips_malformed_lines(self, tmp_path, monkeypatch):
        f = tmp_path / "learnings.jsonl"
        f.write_text('{"variant": "A"}\nthis is not json\n{"variant": "B"}\n')
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", f)
        entries = load_learnings()
        assert len(entries) == 2

    def test_returns_empty_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.LEARNINGS_FILE", tmp_path / "nope.jsonl")
        assert load_learnings() == []


class TestComputeVariantTrends:
    def test_groups_by_variant(self):
        entries = [
            {"variant": "A", "date": "2026-03-08", "f1": 0.28, "recall": 0.93, "precision": 0.17, "status": "miss"},
            {"variant": "A", "date": "2026-03-09", "f1": 0.30, "recall": 0.90, "precision": 0.20, "status": "miss"},
            {"variant": "B", "date": "2026-03-09", "f1": 0.40, "recall": 0.70, "precision": 0.28, "status": "win"},
        ]
        trends = compute_variant_trends(entries)
        assert len(trends["A"]) == 2
        assert len(trends["B"]) == 1

    def test_excludes_ablation_entries(self):
        entries = [
            {"variant": "A", "date": "2026-03-09", "f1": 0.28},
            {"type": "ablations", "date": "2026-03-09", "ablations": []},
        ]
        trends = compute_variant_trends(entries)
        assert len(trends) == 1
        assert "A" in trends


class TestComputeDimensionImpacts:
    def test_extracts_deltas(self):
        entries = [
            {
                "type": "ablations",
                "date": "2026-03-09",
                "ablations": [
                    {"dimension": "contrastive", "delta_f1": 0.12},
                    {"dimension": "model", "delta_f1": -0.03},
                ],
            },
            {
                "type": "ablations",
                "date": "2026-03-10",
                "ablations": [
                    {"dimension": "contrastive", "delta_f1": 0.08},
                ],
            },
        ]
        impacts = compute_dimension_impacts(entries)
        assert len(impacts["contrastive"]) == 2
        assert len(impacts["model"]) == 1

    def test_ignores_non_ablation_entries(self):
        entries = [
            {"variant": "A", "f1": 0.28},
        ]
        assert compute_dimension_impacts(entries) == {}


# ---------------------------------------------------------------------------
# save_best with config
# ---------------------------------------------------------------------------


class TestSaveBestWithConfig:
    def test_config_embedded_in_best_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        from lessons_db.eval.learn import save_best

        config = {"model": "deepseek-r1:8b", "temperature": 0.6, "contrastive": True}
        save_best("F", {"f1": 0.52, "recall": 0.65, "precision": 0.42}, config)
        record = json.loads((tmp_path / "best.json").read_text())
        assert record["config"]["model"] == "deepseek-r1:8b"
        assert record["config"]["contrastive"] is True

    def test_config_omitted_when_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.eval.learn.BEST_JSON", tmp_path / "best.json")
        monkeypatch.setattr("lessons_db.eval.learn._EVAL_DIR", tmp_path)
        from lessons_db.eval.learn import save_best

        save_best("A", {"f1": 0.28, "recall": 0.93, "precision": 0.17})
        record = json.loads((tmp_path / "best.json").read_text())
        assert "config" not in record
