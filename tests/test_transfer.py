"""Tests for transfer (cross-project analogical matching) CLI commands."""

from unittest.mock import patch

from click.testing import CliRunner

from lessons_db.cli import main
from lessons_db.db import init_db, insert_lesson


class TestTransferFindHelp:
    """transfer find --help renders correctly."""

    def test_transfer_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["transfer", "--help"])
        assert result.exit_code == 0
        assert "cross-project" in result.output.lower() or "analogical" in result.output.lower()

    def test_transfer_find_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["transfer", "find", "--help"])
        assert result.exit_code == 0
        assert "--limit" in result.output
        assert "--min-score" in result.output
        assert "CONTEXT" in result.output


class TestTransferFindWithResults:
    """transfer find returns cross-project results with proper framing.

    Patches LANCE_DIR to a non-existent path so the production LanceDB index
    is not consulted and the text fallback search is used instead.
    """

    def test_find_returns_results_with_scope_framing(self, db_path, tmp_path):
        """Results are framed as 'From [scope]: ...'."""
        conn = init_db(db_path)
        insert_lesson(
            conn,
            {
                "title": "subscriber lifecycle callback",
                "one_liner": "Store callback ref on self and unsubscribe in shutdown",
                "description": "Event subscribers that don't store refs leak memory",
                "scope": "language:python, domain:ha-aria",
                "principle": "Resources acquired in init must be released in shutdown",
                "severity": 4,
            },
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        # Patch LANCE_DIR so production LanceDB is not used (forces text fallback)
        with patch("lessons_db.cli.LANCE_DIR", tmp_path / "nonexistent-lance"):
            result = runner.invoke(
                main,
                ["--db", str(db_path), "transfer", "find", "callback shutdown", "--min-score", "0"],
            )
        assert result.exit_code == 0
        assert "From [" in result.output
        assert "language:python" in result.output

    def test_find_falls_back_to_one_liner_without_principle(self, db_path, tmp_path):
        """When principle is empty, one_liner is used as the display text."""
        conn = init_db(db_path)
        insert_lesson(
            conn,
            {
                "title": "bare except swallowing",
                "one_liner": "Never use bare except without logging",
                "scope": "language:python",
                "severity": 5,
            },
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        with patch("lessons_db.cli.LANCE_DIR", tmp_path / "nonexistent-lance"):
            result = runner.invoke(
                main,
                ["--db", str(db_path), "transfer", "find", "bare except", "--min-score", "0"],
            )
        assert result.exit_code == 0
        assert "Never use bare except without logging" in result.output

    def test_find_shows_unscoped_for_null_scope(self, db_path, tmp_path):
        """Lessons without scope display as 'unscoped'."""
        conn = init_db(db_path)
        insert_lesson(
            conn,
            {
                "title": "schema changes consumers",
                "one_liner": "Schema changes must update all consumers in same PR",
                "severity": 5,
            },
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        with patch("lessons_db.cli.LANCE_DIR", tmp_path / "nonexistent-lance"):
            result = runner.invoke(
                main,
                ["--db", str(db_path), "transfer", "find", "schema consumers", "--min-score", "0"],
            )
        assert result.exit_code == 0
        assert "unscoped" in result.output

    def test_find_respects_limit(self, db_path, tmp_path):
        """--limit controls max number of results."""
        conn = init_db(db_path)
        for i in range(10):
            insert_lesson(
                conn,
                {
                    "title": f"Lesson {i} exception handling",
                    "one_liner": f"Pattern {i} for exception handling",
                    "scope": f"domain:project-{i}",
                    "severity": 3,
                },
            )
        conn.commit()
        conn.close()

        runner = CliRunner()
        with patch("lessons_db.cli.LANCE_DIR", tmp_path / "nonexistent-lance"):
            result = runner.invoke(
                main,
                ["--db", str(db_path), "transfer", "find", "exception", "--limit", "2", "--min-score", "0"],
            )
        assert result.exit_code == 0
        from_count = result.output.count("From [")
        assert from_count <= 2

    def test_find_respects_min_score(self, db_path, tmp_path):
        """--min-score filters out low-scoring results."""
        conn = init_db(db_path)
        insert_lesson(
            conn,
            {
                "title": "unrelated lesson",
                "one_liner": "Something about cooking recipes",
                "severity": 1,
            },
        )
        conn.commit()
        conn.close()

        runner = CliRunner()
        with patch("lessons_db.cli.LANCE_DIR", tmp_path / "nonexistent-lance"):
            result = runner.invoke(
                main,
                ["--db", str(db_path), "transfer", "find", "exception handling", "--min-score", "0.99"],
            )
        assert result.exit_code == 0
        assert "No transferable lessons" in result.output or "From [" not in result.output

    def test_find_empty_db(self, db_path, tmp_path):
        """transfer find on empty DB returns no results gracefully."""
        runner = CliRunner()
        with patch("lessons_db.cli.LANCE_DIR", tmp_path / "nonexistent-lance"):
            result = runner.invoke(
                main,
                ["--db", str(db_path), "transfer", "find", "anything"],
            )
        assert result.exit_code == 0
        assert "no" in result.output.lower() or "No" in result.output


class TestTransferFindWithMockedSearch:
    """transfer find with mocked search_combined for deterministic testing."""

    @patch("lessons_db.cli.search_combined")
    def test_find_uses_search_combined(self, mock_search, db_path):
        """transfer find calls search_combined with the context as query."""
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Test lesson",
                "one_liner": "Always validate inputs",
                "principle": "Validate at trust boundaries",
                "scope": "domain:api-layer",
                "severity": 3,
            },
        )
        conn.commit()

        mock_search.return_value = [
            {"id": lid, "one_liner": "Always validate inputs", "composite_score": 0.85},
        ]

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "transfer", "find", "input validation patterns"],
        )
        assert result.exit_code == 0
        # Verify search_combined was called
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs[1]["query"] == "input validation patterns"

        # Verify cross-project framing
        assert "From [domain:api-layer]" in result.output
        assert "Validate at trust boundaries" in result.output

    @patch("lessons_db.cli.search_combined")
    def test_find_with_principle_shows_principle_and_one_liner(self, mock_search, db_path):
        """When principle is populated, output shows both principle and one_liner."""
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "Subscriber lifecycle",
                "one_liner": "Store callback ref on self",
                "principle": "Resources acquired in init must be released in shutdown",
                "description": "Event subscribers that don't store refs leak memory over time",
                "scope": "language:python, domain:ha-aria",
                "severity": 4,
            },
        )
        conn.commit()

        mock_search.return_value = [
            {"id": lid, "one_liner": "Store callback ref on self", "composite_score": 0.75},
        ]

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "transfer", "find", "resource cleanup"],
        )
        assert result.exit_code == 0
        assert "Resources acquired in init must be released in shutdown" in result.output
        assert "Store callback ref on self" in result.output
        assert "score=0.750" in result.output

    @patch("lessons_db.cli.search_combined")
    def test_find_no_results_from_search(self, mock_search, db_path):
        """When search_combined returns empty, shows 'no transferable lessons'."""
        mock_search.return_value = []

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "transfer", "find", "nonexistent topic"],
        )
        assert result.exit_code == 0
        assert "No transferable lessons found." in result.output
