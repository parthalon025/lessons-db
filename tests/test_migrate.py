"""Tests for markdown lesson file parser."""

from pathlib import Path

from lessons_db.migrate import import_lesson_file, parse_lesson_file

SAMPLE_LESSON = """\
# Lesson #88: `contextlib.suppress` in Finally Blocks Is a Silent Failure Trap

**Date:** 2026-02-25
**System:** ha-aria (UniFi integration — WebSocket Protect pipeline)
**Tier:** lesson
**Category:** silent-failures
**Cluster:** A (Silent Failures)
**Scope:** `language:python, domain:ha-aria`
**Keywords:** contextlib.suppress, finally, cleanup, silent failure
**Related:** #7 (silent except: return []), #37 (subscriber lifecycle cleanup)

---

## Observation (What Happened)

In `UniFiModule._disconnect_websocket()`, the finally block used `contextlib.suppress(Exception)`.

## Analysis (Root Cause — 5 Whys)

**Why #1:** Disconnect errors were never visible in logs.
**Why #2:** `contextlib.suppress(Exception)` ate all exceptions silently.
**Why #3:** `contextlib.suppress` reads as intentional defensive programming.

## Corrective Actions

| # | Action | Status |
|---|--------|--------|
| 1 | Replaced `contextlib.suppress` with try/except + logger.debug | implemented |
| 2 | Applied same pattern to all cleanup paths | implemented |

## Key Takeaway

`contextlib.suppress(Exception)` in cleanup paths is `except: pass` with better marketing.
"""


def _write_lesson(tmp_path: Path, content: str = SAMPLE_LESSON) -> Path:
    p = tmp_path / "lesson.md"
    p.write_text(content)
    return p


def test_extracts_title(tmp_path: Path) -> None:
    result = parse_lesson_file(_write_lesson(tmp_path))
    assert result["title"] == "`contextlib.suppress` in Finally Blocks Is a Silent Failure Trap"


def test_extracts_lesson_number(tmp_path: Path) -> None:
    result = parse_lesson_file(_write_lesson(tmp_path))
    assert result["lesson_number"] == 88

    # No number variant
    no_num = tmp_path / "no_num.md"
    no_num.write_text("# Lesson: No Number Here\n\n**Date:** 2026-01-01\n**Tier:** lesson\n**Cluster:** X\n")
    result2 = parse_lesson_file(no_num)
    assert result2["lesson_number"] is None


def test_extracts_metadata(tmp_path: Path) -> None:
    result = parse_lesson_file(_write_lesson(tmp_path))
    assert result["date"] == "2026-02-25"
    assert result["tier"] == "lesson"
    assert result["cluster"] == "A"
    assert result["category"] == "silent-failures"


def test_extracts_scope(tmp_path: Path) -> None:
    result = parse_lesson_file(_write_lesson(tmp_path))
    assert result["scope"] == "language:python, domain:ha-aria"


def test_extracts_keywords(tmp_path: Path) -> None:
    result = parse_lesson_file(_write_lesson(tmp_path))
    assert "contextlib.suppress" in result["keywords"]


def test_extracts_key_takeaway(tmp_path: Path) -> None:
    result = parse_lesson_file(_write_lesson(tmp_path))
    assert "except: pass" in result["key_takeaway"]


def test_extracts_corrective_actions(tmp_path: Path) -> None:
    result = parse_lesson_file(_write_lesson(tmp_path))
    actions = result["corrective_actions"]
    assert len(actions) == 2
    assert actions[0]["status"] == "implemented"
    assert "contextlib.suppress" in actions[0]["description"]


def test_extracts_related_lessons(tmp_path: Path) -> None:
    result = parse_lesson_file(_write_lesson(tmp_path))
    assert result["related"] == [7, 37]


def test_handles_missing_fields_gracefully(tmp_path: Path) -> None:
    minimal = tmp_path / "minimal.md"
    minimal.write_text("# Lesson: Minimal Example\n\n**Date:** 2026-01-01\n**Tier:** lesson\n**Cluster:** Z\n")
    result = parse_lesson_file(minimal)
    assert result["title"] == "Minimal Example"
    assert result["lesson_number"] is None
    assert result["date"] == "2026-01-01"
    assert result["scope"] == ""
    assert result["keywords"] == ""
    assert result["related"] == []
    assert result["corrective_actions"] == []
    assert result["key_takeaway"] == ""
    assert result["description"] == ""


# ---------------------------------------------------------------------------
# YAML frontmatter format (0001-*.md style lessons)
# ---------------------------------------------------------------------------

YAML_LESSON = """\
---
id: 1
title: "Bare exception swallowing hides failures"
severity: blocker
languages: [python]
scope: [language:python]
category: silent-failures
pattern:
  type: syntactic
  regex: "^\\\\s*except\\\\s*:"
  description: "bare except clause without logging"
fix: "Always log the exception before returning a fallback"
positive_alternative: "Use 'except Exception as e: logger.error(...)'"
---

## Observation
Bare `except:` clauses silently swallow all exceptions.

## Insight
The root cause is defensive exception handling with no logging.

## Lesson
Never use bare `except:` — always catch a specific exception class and log.
"""

YAML_LESSON_CLUSTER_B = """\
---
id: 60
title: "Integration boundary bug hides under passing unit tests"
severity: should-fix
languages: [python, shell]
scope: [language:python, language:bash]
category: integration-boundaries
pattern:
  type: syntactic
  regex: 'mock\\.patch'
  description: "over-mocked integration test"
fix: "Trace one value end-to-end instead of mocking at boundaries"
---

## Observation
Each layer's tests pass individually but the integration fails.

## Lesson
Always run a vertical trace test after unit tests pass.
"""


def _write_yaml_lesson(tmp_path: Path, content: str = YAML_LESSON, name: str = "0001-bare-except.md") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


class TestImportLessonFile:
    """Tests for import_lesson_file() — YAML frontmatter format."""

    def test_returns_inserted_id(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = _write_yaml_lesson(tmp_path)
        lesson_id = import_lesson_file(conn, path)
        assert isinstance(lesson_id, int)
        assert lesson_id > 0

    def test_inserts_title_and_one_liner(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = _write_yaml_lesson(tmp_path)
        lesson_id = import_lesson_file(conn, path)
        row = conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        assert row["title"] == "Bare exception swallowing hides failures"
        assert "log" in row["one_liner"].lower()

    def test_inserts_cluster_from_category(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = _write_yaml_lesson(tmp_path)
        lesson_id = import_lesson_file(conn, path)
        row = conn.execute("SELECT cluster FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        # silent-failures → cluster A
        assert row["cluster"] == "A"

    def test_inserts_scope(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = _write_yaml_lesson(tmp_path)
        lesson_id = import_lesson_file(conn, path)
        row = conn.execute("SELECT scope FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        assert "language:python" in row["scope"]

    def test_inserts_detection_pattern(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = _write_yaml_lesson(tmp_path)
        lesson_id = import_lesson_file(conn, path)
        patterns = conn.execute("SELECT * FROM detection_patterns WHERE lesson_id = ?", (lesson_id,)).fetchall()
        assert len(patterns) == 1
        assert patterns[0]["pattern_type"] == "syntactic"
        assert "except" in patterns[0]["regex"]

    def test_inserts_markdown_path(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = _write_yaml_lesson(tmp_path)
        lesson_id = import_lesson_file(conn, path)
        row = conn.execute("SELECT markdown_path FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        assert row["markdown_path"] == str(path)

    def test_duplicate_by_path_returns_none(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = _write_yaml_lesson(tmp_path)
        first = import_lesson_file(conn, path)
        second = import_lesson_file(conn, path)
        assert first is not None
        assert second is None  # duplicate — skipped

    def test_duplicate_by_title_returns_none(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path1 = _write_yaml_lesson(tmp_path, name="0001-a.md")
        path2 = _write_yaml_lesson(tmp_path, name="0001-b.md")  # same title, different path
        import_lesson_file(conn, path1)
        second = import_lesson_file(conn, path2)
        assert second is None

    def test_cluster_b_category_mapping(self, tmp_path: Path) -> None:
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = _write_yaml_lesson(tmp_path, content=YAML_LESSON_CLUSTER_B, name="0060-boundary.md")
        lesson_id = import_lesson_file(conn, path)
        row = conn.execute("SELECT cluster FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        assert row["cluster"] == "B"

    def test_imports_heading_bold_format(self, tmp_path: Path) -> None:
        """import_lesson_file accepts heading+bold format as a fallback."""
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        path = tmp_path / "2026-02-28-old-format.md"
        path.write_text(
            "# Lesson: Old-style lesson\n\n"
            "**Date:** 2026-02-28\n"
            "**System:** lessons-db\n"
            "**Tier:** lesson\n"
            "**Category:** integration\n"
            "**Cluster:** B (Integration Boundary)\n"
            "**Keywords:** test, integration\n"
            "\n"
            "## Observation (What Happened)\n"
            "Something failed at the seam.\n"
            "\n"
            "## Key Takeaway\n"
            "Always verify at the boundary.\n"
        )
        lesson_id = import_lesson_file(conn, path)
        assert isinstance(lesson_id, int)
        assert lesson_id > 0
        row = conn.execute("SELECT title, cluster, one_liner FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
        assert row["title"] == "Old-style lesson"
        assert row["cluster"] == "B"
        assert "boundary" in row["one_liner"].lower()

    def test_heading_bold_duplicate_by_title_returns_none(self, tmp_path: Path) -> None:
        """import_lesson_file skips duplicate heading+bold lessons by title."""
        from lessons_db.db import init_db

        conn = init_db(tmp_path / "test.db")
        content = (
            "# Lesson: Duplicate heading bold\n\n"
            "**Date:** 2026-02-28\n"
            "**Tier:** lesson\n"
            "**Category:** integration\n"
            "**Cluster:** B\n"
            "\n"
            "## Key Takeaway\n"
            "Only imported once.\n"
        )
        path1 = tmp_path / "file-a.md"
        path1.write_text(content)
        path2 = tmp_path / "file-b.md"
        path2.write_text(content)

        first = import_lesson_file(conn, path1)
        second = import_lesson_file(conn, path2)
        assert first is not None
        assert second is None  # duplicate title skipped


class TestImportCLICommand:
    """Tests for `lessons-db import` CLI command."""

    def test_import_single_file(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from lessons_db.cli import main

        path = _write_yaml_lesson(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "import", "file", str(path)])
        assert result.exit_code == 0, result.output
        assert "imported" in result.output.lower()

    def test_import_directory(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from lessons_db.cli import main

        lesson_dir = tmp_path / "lessons"
        lesson_dir.mkdir()
        _write_yaml_lesson(lesson_dir, name="0001-bare-except.md")
        _write_yaml_lesson(lesson_dir, content=YAML_LESSON_CLUSTER_B, name="0060-boundary.md")

        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "import", "file", str(lesson_dir)])
        assert result.exit_code == 0, result.output
        assert "2" in result.output  # 2 imported

    def test_import_skips_duplicates(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from lessons_db.cli import main

        path = _write_yaml_lesson(tmp_path)
        runner = CliRunner()
        db = str(tmp_path / "test.db")
        runner.invoke(main, ["--db", db, "import", "file", str(path)])
        result = runner.invoke(main, ["--db", db, "import", "file", str(path)])
        assert result.exit_code == 0
        assert "skipped" in result.output.lower()

    def test_import_missing_path_exits_nonzero(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from lessons_db.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main, ["--db", str(tmp_path / "test.db"), "import", "file", str(tmp_path / "nonexistent.md")]
        )
        assert result.exit_code != 0
