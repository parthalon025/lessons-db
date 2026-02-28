"""Tests for Gates 0a (syntax) and 0b (regex self-consistency)."""

from lessons_db.pattern_validator import validate_regex_self_consistency, validate_syntax

GOOD_BAD = "def foo():\n    time.sleep(1)"
GOOD_GOOD = "async def foo():\n    await asyncio.sleep(1)"
VALID_REGEX = r"time\.sleep\("


def test_syntax_passes_valid_lesson():
    result = validate_syntax(
        title="Blocking sleep in async",
        one_liner="Use asyncio.sleep not time.sleep",
        bad_code=GOOD_BAD,
        good_code=GOOD_GOOD,
    )
    assert result["passed"] is True


def test_syntax_rejects_missing_title():
    result = validate_syntax(title="", one_liner="x", bad_code="x=1\ny=2", good_code="x=2\ny=3")
    assert result["passed"] is False
    assert "title" in result["reason"].lower()


def test_syntax_rejects_invalid_python():
    result = validate_syntax(title="Test", one_liner="x", bad_code="def foo(:\n    pass", good_code="x=1\ny=2")
    assert result["passed"] is False
    assert "syntax" in result["reason"].lower()


def test_syntax_rejects_too_short():
    result = validate_syntax(title="T", one_liner="x", bad_code="x", good_code="y")
    assert result["passed"] is False


def test_syntax_accepts_regex_field():
    result = validate_syntax(
        title="T",
        one_liner="x",
        bad_code=GOOD_BAD,
        good_code=GOOD_GOOD,
        regex=VALID_REGEX,
    )
    assert result["passed"] is True


def test_syntax_rejects_bad_regex():
    result = validate_syntax(
        title="T",
        one_liner="x",
        bad_code=GOOD_BAD,
        good_code=GOOD_GOOD,
        regex=r"(unclosed[",
    )
    assert result["passed"] is False
    assert "regex" in result["reason"].lower()


def test_regex_self_consistency_passes():
    result = validate_regex_self_consistency(VALID_REGEX, GOOD_BAD, GOOD_GOOD)
    assert result["passed"] is True


def test_regex_self_consistency_fails_if_matches_good():
    # regex that matches both bad and good (too broad)
    result = validate_regex_self_consistency(r"def foo", GOOD_BAD, GOOD_GOOD)
    assert result["passed"] is False
    assert "good_code" in result["reason"].lower()


def test_regex_self_consistency_fails_if_no_match_bad():
    result = validate_regex_self_consistency(r"requests\.get\(", GOOD_BAD, GOOD_GOOD)
    assert result["passed"] is False
    assert "bad_code" in result["reason"].lower()


def test_regex_skipped_when_no_regex():
    result = validate_regex_self_consistency(None, GOOD_BAD, GOOD_GOOD)
    assert result["passed"] is True
    assert result.get("skipped") is True
