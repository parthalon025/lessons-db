"""Tests for Semgrep rule generation."""

import yaml

from lessons_db.rulegen import generate_rule, generate_test_file, slug_from_title


def test_simple_title():
    assert slug_from_title("Bare except swallows failures") == "bare-except-swallows-failures"


def test_strips_special_chars():
    assert slug_from_title("`contextlib.suppress` trap") == "contextlib-suppress-trap"


def test_generates_valid_yaml():
    lesson = {
        "id": 7,
        "title": "Bare except swallows failures",
        "one_liner": "Never use bare except — log before returning fallback",
        "cluster": "silent-failures",
        "confidence": "high",
    }
    patterns = [
        {
            "pattern_type": "regex",
            "regex": r"except\s*:",
            "description": "Bare except clause",
            "language": "python",
        }
    ]
    output = generate_rule(lesson, patterns)
    parsed = yaml.safe_load(output)

    assert "rules" in parsed
    rule = parsed["rules"][0]
    assert rule["id"] == "lessons-db.python.bare-except-swallows-failures-007"
    assert rule["severity"] == "WARNING"
    assert rule["languages"] == ["python"]
    assert rule["metadata"]["lesson_id"] == 7
    assert rule["metadata"]["cluster"] == "silent-failures"
    assert rule["metadata"]["confidence"] == "high"
    assert "Lesson #7" in rule["message"]
    assert "pattern-regex" in rule


def test_generates_with_autofix():
    lesson = {
        "id": 88,
        "title": "`contextlib.suppress` trap",
        "one_liner": "contextlib.suppress hides exceptions silently",
        "cluster": "silent-failures",
        "confidence": "medium",
    }
    patterns = [
        {
            "pattern_type": "structural",
            "regex": "",
            "description": "contextlib.suppress usage",
            "language": "python",
            "semgrep_pattern": "with contextlib.suppress($EXC): ...",
            "semgrep_pattern_not": "with contextlib.suppress($EXC): ... \nlogger.debug(...)",
            "fix": "try:\n    ...\nexcept $EXC as e:\n    logger.debug(e)",
        }
    ]
    output = generate_rule(lesson, patterns)
    parsed = yaml.safe_load(output)

    rule = parsed["rules"][0]
    assert "fix" in rule
    assert "patterns" in rule
    assert any("pattern" in p for p in rule["patterns"])
    assert any("pattern-not-inside" in p for p in rule["patterns"])


def test_generates_test_file():
    rule_id = "lessons-db.python.bare-except-007"
    true_positive = "try:\n    x()\nexcept:\n    pass"
    true_negative = "try:\n    x()\nexcept ValueError:\n    pass"
    output = generate_test_file(rule_id, true_positive, true_negative)

    assert f"# ruleid: {rule_id}" in output
    assert f"# ok: {rule_id}" in output
    assert true_positive in output
    assert true_negative in output
