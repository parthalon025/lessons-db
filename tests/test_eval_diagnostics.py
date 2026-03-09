"""Tests for eval diagnostic utilities."""

from lessons_db.eval_diagnostics import (
    build_confusion_matrix,
    compute_roc_curve,
    render_confusion_report,
    render_roc_report,
)


class TestBuildConfusionMatrix:
    def test_basic_two_clusters(self):
        """Two clusters, each with same/diff pairs — verify matrix shape and values."""
        scored_pairs = [
            {
                "cluster_seed": "A",
                "is_same_cluster": True,
                "scores": {"transfer": 4, "precision": 3, "actionability": 3},
            },
            {
                "cluster_seed": "A",
                "is_same_cluster": False,
                "scores": {"transfer": 1, "precision": 4, "actionability": 2},
                "target_cluster_seed": "B",
            },
            {
                "cluster_seed": "B",
                "is_same_cluster": True,
                "scores": {"transfer": 5, "precision": 4, "actionability": 4},
            },
            {
                "cluster_seed": "B",
                "is_same_cluster": False,
                "scores": {"transfer": 4, "precision": 2, "actionability": 3},
                "target_cluster_seed": "A",
            },
        ]

        matrix = build_confusion_matrix(scored_pairs)

        assert ("A", "A") in matrix
        assert ("A", "B") in matrix
        assert ("B", "A") in matrix
        assert ("B", "B") in matrix
        assert matrix[("A", "A")]["avg_transfer"] == 4.0
        assert matrix[("A", "B")]["avg_transfer"] == 1.0
        assert matrix[("B", "A")]["avg_transfer"] == 4.0
        assert matrix[("B", "B")]["avg_transfer"] == 5.0

    def test_empty_returns_empty(self):
        assert build_confusion_matrix([]) == {}

    def test_skips_pairs_without_scores(self):
        scored_pairs = [
            {"cluster_seed": "A", "is_same_cluster": True},
            {
                "cluster_seed": "A",
                "is_same_cluster": True,
                "scores": {"transfer": 3, "precision": 2, "actionability": 2},
            },
        ]
        matrix = build_confusion_matrix(scored_pairs)
        assert matrix[("A", "A")]["count"] == 1

    def test_multiple_pairs_averaged(self):
        scored_pairs = [
            {
                "cluster_seed": "A",
                "is_same_cluster": True,
                "scores": {"transfer": 2},
            },
            {
                "cluster_seed": "A",
                "is_same_cluster": True,
                "scores": {"transfer": 4},
            },
        ]
        matrix = build_confusion_matrix(scored_pairs)
        assert matrix[("A", "A")]["avg_transfer"] == 3.0
        assert matrix[("A", "A")]["count"] == 2

    def test_unknown_target_cluster_defaults(self):
        scored_pairs = [
            {
                "cluster_seed": "A",
                "is_same_cluster": False,
                "scores": {"transfer": 2},
            },
        ]
        matrix = build_confusion_matrix(scored_pairs)
        assert ("A", "unknown") in matrix


class TestRenderConfusionReport:
    def test_empty_matrix(self):
        assert render_confusion_report({}) == "No data.\n"

    def test_contains_header_and_table(self):
        matrix = {
            ("A", "A"): {"avg_transfer": 4.0, "count": 2},
            ("A", "B"): {"avg_transfer": 1.0, "count": 1},
        }
        report = render_confusion_report(matrix)
        assert "Cluster Confusion Matrix" in report
        assert "**A**" in report
        assert "4.0" in report

    def test_flags_high_cross_cluster(self):
        matrix = {
            ("A", "A"): {"avg_transfer": 4.0, "count": 2},
            ("A", "B"): {"avg_transfer": 3.5, "count": 1},
        }
        report = render_confusion_report(matrix)
        assert "Flagged Pairs" in report
        assert "A -> B" in report
        assert "3.5" in report

    def test_no_flags_when_all_low(self):
        matrix = {
            ("A", "A"): {"avg_transfer": 4.0, "count": 2},
            ("A", "B"): {"avg_transfer": 1.0, "count": 1},
        }
        report = render_confusion_report(matrix)
        assert "No cross-cluster pairs" in report


class TestComputeRocCurve:
    def test_basic_roc(self):
        scored_pairs = [
            {"is_same_cluster": True, "scores": {"transfer": 5}},
            {"is_same_cluster": True, "scores": {"transfer": 3}},
            {"is_same_cluster": False, "scores": {"transfer": 2}},
            {"is_same_cluster": False, "scores": {"transfer": 4}},
        ]
        curve = compute_roc_curve(scored_pairs)
        assert len(curve) == 5
        # At threshold 1: all same-cluster match → recall=1.0
        assert curve[1]["recall"] == 1.0
        # At threshold 5: only transfer=5 matches → recall=0.5
        assert curve[5]["recall"] == 0.5
        # Higher threshold = lower recall
        assert curve[5]["recall"] < curve[3]["recall"]

    def test_empty_returns_zeros(self):
        curve = compute_roc_curve([])
        assert len(curve) == 5
        assert curve[1]["f1"] == 0.0

    def test_perfect_separation(self):
        """Same=5, diff=1 → at threshold 5, recall=1.0 and precision=1.0."""
        scored_pairs = [
            {"is_same_cluster": True, "scores": {"transfer": 5}},
            {"is_same_cluster": False, "scores": {"transfer": 1}},
        ]
        curve = compute_roc_curve(scored_pairs)
        # At threshold 5: same (5>=5)=True → recall=1.0; diff (1<5)=True → precision=1.0
        assert curve[5]["recall"] == 1.0
        assert curve[5]["precision"] == 1.0
        assert curve[5]["f1"] == 1.0

    def test_custom_thresholds(self):
        scored_pairs = [
            {"is_same_cluster": True, "scores": {"transfer": 3}},
            {"is_same_cluster": False, "scores": {"transfer": 2}},
        ]
        curve = compute_roc_curve(scored_pairs, thresholds=[2, 4])
        assert set(curve.keys()) == {2, 4}

    def test_skips_pairs_without_scores(self):
        scored_pairs = [
            {"is_same_cluster": True, "scores": {"transfer": 4}},
            {"is_same_cluster": True},  # no scores — should be skipped
            {"is_same_cluster": False, "scores": {"transfer": 2}},
        ]
        curve = compute_roc_curve(scored_pairs)
        # Only 1 same-cluster pair counted
        assert curve[4]["recall"] == 1.0  # 1/1 same pair >= 4
        assert curve[5]["recall"] == 0.0  # 0/1 same pair >= 5


class TestRenderRocReport:
    def test_empty_curve(self):
        assert render_roc_report({}) == "No data.\n"

    def test_marks_best_threshold(self):
        curve = {
            3: {"recall": 0.8, "precision": 0.7, "f1": 0.74},
            4: {"recall": 0.5, "precision": 0.9, "f1": 0.64},
        }
        report = render_roc_report(curve)
        assert "<-- best" in report
        assert "0.74" in report
