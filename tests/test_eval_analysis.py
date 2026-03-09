"""Tests for eval/analysis.py — per-lesson breakdown and failure case extraction."""

from lessons_db.eval.analysis import compute_per_lesson_breakdown, extract_failure_cases

# ---------------------------------------------------------------------------
# Helpers — scored pair builders
# ---------------------------------------------------------------------------


def _binary_pair(
    variant: str,
    source_lesson_id: int,
    principle: str,
    target_id: int,
    target_title: str,
    is_same_cluster: bool,
    matched: bool,
    *,
    cluster_seed: str = "cluster-A",
    target_cluster_seed: str | None = None,
) -> dict:
    return {
        "variant": variant,
        "source_lesson_id": source_lesson_id,
        "principle": principle,
        "target_id": target_id,
        "target_title": target_title,
        "is_same_cluster": is_same_cluster,
        "cluster_seed": cluster_seed,
        "target_cluster_seed": target_cluster_seed or ("cluster-A" if is_same_cluster else "cluster-B"),
        "scores": {"matched": matched},
    }


def _rubric_pair(
    variant: str,
    source_lesson_id: int,
    principle: str,
    target_id: int,
    target_title: str,
    is_same_cluster: bool,
    transfer: int,
    *,
    precision: int = 3,
    actionability: int = 3,
    cluster_seed: str = "cluster-A",
    target_cluster_seed: str | None = None,
) -> dict:
    return {
        "variant": variant,
        "source_lesson_id": source_lesson_id,
        "principle": principle,
        "target_id": target_id,
        "target_title": target_title,
        "is_same_cluster": is_same_cluster,
        "cluster_seed": cluster_seed,
        "target_cluster_seed": target_cluster_seed or ("cluster-A" if is_same_cluster else "cluster-B"),
        "scores": {"transfer": transfer, "precision": precision, "actionability": actionability},
    }


# ---------------------------------------------------------------------------
# TestPerLessonBreakdown
# ---------------------------------------------------------------------------


class TestPerLessonBreakdown:
    def test_groups_by_source_lesson(self):
        """3 pairs across 2 lessons produce 2 breakdown entries."""
        pairs = [
            _binary_pair("A", 1, "principle-1", 10, "target-10", True, True),
            _binary_pair("A", 1, "principle-1", 11, "target-11", False, False),
            _binary_pair("A", 2, "principle-2", 12, "target-12", True, True),
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 2
        lesson_ids = {r["source_lesson_id"] for r in result}
        assert lesson_ids == {1, 2}

    def test_computes_per_lesson_f1(self):
        """Perfect pair: TP on same-cluster, TN on diff-cluster -> F1=1.0."""
        pairs = [
            _binary_pair("A", 1, "principle-1", 10, "target-10", True, True),  # TP
            _binary_pair("A", 1, "principle-1", 11, "target-11", False, False),  # TN
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 1
        entry = result[0]
        assert entry["f1"] == 1.0
        assert entry["tp"] == 1
        assert entry["fn"] == 0
        assert entry["fp"] == 0
        assert entry["total_pairs"] == 2

    def test_sorted_worst_first(self):
        """Lesson with F1=0.0 appears before lesson with F1=1.0."""
        pairs = [
            # Lesson 1: perfect (TP same, TN diff) -> F1=1.0
            _binary_pair("A", 1, "good-principle", 10, "t-10", True, True),
            _binary_pair("A", 1, "good-principle", 11, "t-11", False, False),
            # Lesson 2: total miss (FN same, FP diff) -> F1=0.0
            _binary_pair("A", 2, "bad-principle", 12, "t-12", True, False),
            _binary_pair("A", 2, "bad-principle", 13, "t-13", False, True),
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 2
        assert result[0]["f1"] == 0.0
        assert result[0]["source_lesson_id"] == 2
        assert result[1]["f1"] == 1.0
        assert result[1]["source_lesson_id"] == 1

    def test_includes_variant(self):
        """Result includes the variant field."""
        pairs = [
            _binary_pair("F", 5, "principle-F", 20, "t-20", True, True),
        ]
        result = compute_per_lesson_breakdown(pairs)
        assert len(result) == 1
        assert result[0]["variant"] == "F"

    def test_empty_pairs(self):
        """Empty input returns empty list."""
        assert compute_per_lesson_breakdown([]) == []


# ---------------------------------------------------------------------------
# TestExtractFailureCases
# ---------------------------------------------------------------------------


class TestExtractFailureCases:
    def test_finds_false_positives(self):
        """Diff-cluster pair with matched=True is a false positive."""
        pairs = [
            _binary_pair(
                "A", 1, "p1", 10, "t-10", False, True, cluster_seed="cluster-A", target_cluster_seed="cluster-B"
            ),
        ]
        result = extract_failure_cases(pairs)
        assert len(result) == 1
        assert result[0]["failure_type"] == "false_positive"
        assert result[0]["target_id"] == 10

    def test_finds_false_negatives(self):
        """Same-cluster pair with matched=False is a false negative."""
        pairs = [
            _binary_pair("A", 1, "p1", 10, "t-10", True, False),
        ]
        result = extract_failure_cases(pairs)
        assert len(result) == 1
        assert result[0]["failure_type"] == "false_negative"

    def test_correct_pairs_excluded(self):
        """TP (same+matched) and TN (diff+not matched) are not failures."""
        pairs = [
            _binary_pair("A", 1, "p1", 10, "t-10", True, True),  # TP
            _binary_pair("A", 1, "p1", 11, "t-11", False, False),  # TN
        ]
        result = extract_failure_cases(pairs)
        assert result == []

    def test_rubric_mode_threshold(self):
        """In rubric mode, transfer >= 3 on diff-cluster is a false positive."""
        pairs = [
            _rubric_pair("B", 3, "p3", 30, "t-30", False, transfer=4),  # FP (diff + transfer>=3)
            _rubric_pair("B", 3, "p3", 31, "t-31", True, transfer=2),  # FN (same + transfer<3)
            _rubric_pair("B", 3, "p3", 32, "t-32", True, transfer=3),  # TP (same + transfer>=3)
            _rubric_pair("B", 3, "p3", 33, "t-33", False, transfer=1),  # TN (diff + transfer<3)
        ]
        result = extract_failure_cases(pairs)
        assert len(result) == 2
        types = {r["failure_type"] for r in result}
        assert types == {"false_positive", "false_negative"}
        fp = next(r for r in result if r["failure_type"] == "false_positive")
        assert fp["target_id"] == 30
        fn = next(r for r in result if r["failure_type"] == "false_negative")
        assert fn["target_id"] == 31

    def test_empty_pairs(self):
        """Empty input returns empty list."""
        assert extract_failure_cases([]) == []
