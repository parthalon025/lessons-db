"""Tests for eval diagnostic utilities."""

from lessons_db.eval_diagnostics import build_confusion_matrix, render_confusion_report


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
