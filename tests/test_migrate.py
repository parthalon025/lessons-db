"""Tests for markdown lesson file parser."""

from pathlib import Path

from lessons_db.migrate import parse_lesson_file

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
