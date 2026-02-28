"""Tests for the check command and check_files function."""

import json

import pytest
from click.testing import CliRunner

from lessons_db.check import _scope_matches, check_files
from lessons_db.cli import main
from lessons_db.db import init_db


@pytest.fixture
def db_with_pattern(db_path):
    """DB with one syntactic detection pattern."""
    conn = init_db(db_path)
    conn.execute(
        "INSERT INTO lessons (id, title, one_liner, severity, scope, created_date) "
        "VALUES (1, 'No bare except', 'Always log before returning fallback', 'ERROR', 'universal', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO detection_patterns (lesson_id, pattern_type, regex, description, language) "
        "VALUES (1, 'syntactic', 'except\\s*:', 'Catches bare except blocks', 'python')"
    )
    conn.commit()
    return conn


class TestCheckFiles:
    def test_empty_detection_patterns_returns_clean(self, db_path):
        conn = init_db(db_path)
        result = check_files(conn, None, ["nonexistent.py"])
        assert result == []
        conn.close()

    def test_syntactic_match_returns_violation(self, db_with_pattern, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("try:\n    pass\nexcept:\n    pass\n")
        result = check_files(db_with_pattern, None, [str(bad_file)])
        assert len(result) == 1
        assert result[0]["lesson_id"] == 1
        assert result[0]["line_number"] == 3
        assert result[0]["source"] == "syntactic"
        db_with_pattern.close()

    def test_no_match_returns_clean(self, db_with_pattern, tmp_path):
        good_file = tmp_path / "good.py"
        good_file.write_text("try:\n    pass\nexcept ValueError:\n    pass\n")
        result = check_files(db_with_pattern, None, [str(good_file)])
        assert result == []
        db_with_pattern.close()

    def test_scope_filter_excludes_unmatched(self, db_path, tmp_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO lessons (id, title, one_liner, severity, scope, created_date) "
            "VALUES (1, 'Python only', 'Python specific rule', 'ERROR', 'language:python', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO detection_patterns (lesson_id, pattern_type, regex, description, language) "
            "VALUES (1, 'syntactic', 'echo', 'test', 'any')"
        )
        conn.commit()
        f = tmp_path / "script.sh"
        f.write_text("echo hello\n")
        result = check_files(conn, None, [str(f)], scope="language:bash")
        assert result == []
        conn.close()

    def test_multiple_files(self, db_with_pattern, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("except:\n    pass\n")
        good = tmp_path / "good.py"
        good.write_text("except ValueError:\n    pass\n")
        result = check_files(db_with_pattern, None, [str(bad), str(good)])
        assert len(result) == 1
        assert result[0]["file_path"] == str(bad)
        db_with_pattern.close()

    def test_unreadable_file_skipped(self, db_with_pattern):
        result = check_files(db_with_pattern, None, ["/nonexistent/file.py"])
        assert result == []
        db_with_pattern.close()

    def test_lance_unavailable_still_works(self, db_with_pattern, tmp_path):
        bad = tmp_path / "bad.py"
        bad.write_text("except:\n    pass\n")
        result = check_files(db_with_pattern, None, [str(bad)])
        assert len(result) == 1
        db_with_pattern.close()


class TestCheckCLI:
    def test_json_output(self, db_path, tmp_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO lessons (id, title, one_liner, severity, scope, created_date) "
            "VALUES (1, 'Test', 'Test rule', 'ERROR', 'universal', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO detection_patterns (lesson_id, pattern_type, regex, description, language) "
            "VALUES (1, 'syntactic', 'bad_pattern', 'test', 'any')"
        )
        conn.commit()
        conn.close()

        bad = tmp_path / "bad.txt"
        bad.write_text("this has bad_pattern in it\n")

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "check", "-f", str(bad), "--json"])
        assert result.exit_code == 1
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["lesson_id"] == 1

    def test_clean_exit_zero(self, db_path, tmp_path):
        conn = init_db(db_path)
        conn.close()
        good = tmp_path / "good.txt"
        good.write_text("nothing wrong here\n")
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "check", "-f", str(good)])
        assert result.exit_code == 0

    def test_human_output_format(self, db_path, tmp_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO lessons (id, title, one_liner, severity, scope, created_date) "
            "VALUES (1, 'Test', 'Test rule', 'ERROR', 'universal', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO detection_patterns (lesson_id, pattern_type, regex, description, language) "
            "VALUES (1, 'syntactic', 'bad_pattern', 'test', 'any')"
        )
        conn.commit()
        conn.close()

        bad = tmp_path / "bad.txt"
        bad.write_text("this has bad_pattern in it\n")

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(db_path), "check", "-f", str(bad)])
        assert "[lesson-1]" in result.output
        assert "Test rule" in result.output


class TestScopeMatches:
    def test_universal_matches_everything(self):
        assert _scope_matches("universal", "language:python") is True

    def test_matching_tags(self):
        assert _scope_matches("language:python, domain:web", "language:python") is True

    def test_no_match(self):
        assert _scope_matches("language:python", "language:bash") is False

    def test_case_insensitive(self):
        assert _scope_matches("Language:Python", "language:python") is True
