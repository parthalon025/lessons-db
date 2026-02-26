"""Generate Semgrep rules from lessons-db entries."""

from __future__ import annotations

import re

import yaml


def slug_from_title(title: str) -> str:
    """Convert title to URL-safe slug.

    Lowercase, strip backticks/quotes, replace non-alphanumeric with hyphens,
    collapse consecutive hyphens, strip leading/trailing hyphens.
    """
    slug = title.lower()
    slug = slug.replace("`", "").replace("'", "").replace('"', "")
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug


def generate_rule(
    lesson: dict,
    patterns: list[dict],
    severity: str = "WARNING",
) -> str:
    """Generate Semgrep rule YAML string from a lesson and its detection patterns.

    Args:
        lesson: Dict with keys: id, title, one_liner, cluster, confidence.
        patterns: List of pattern dicts with keys: pattern_type, regex,
            description, language. Optional: semgrep_pattern,
            semgrep_pattern_not, fix.
        severity: One of WARNING, ERROR, INFO.

    Returns:
        YAML string with top-level ``rules:`` key.
    """
    slug = slug_from_title(lesson["title"])
    lesson_id = lesson["id"]
    language = patterns[0]["language"]

    rule: dict = {
        "id": f"lessons-db.{language}.{slug}-{lesson_id:03d}",
        "message": f"{lesson['one_liner']} (Lesson #{lesson_id})",
        "severity": severity,
        "languages": [language],
        "metadata": {
            "lesson_id": lesson_id,
            "cluster": lesson["cluster"],
            "confidence": lesson["confidence"],
        },
    }

    # Build pattern section based on type and count
    regex_patterns = [p for p in patterns if p["pattern_type"] == "regex"]
    structural_patterns = [p for p in patterns if p["pattern_type"] == "structural"]

    if structural_patterns:
        sp = structural_patterns[0]
        pattern_list: list[dict] = [{"pattern": sp["semgrep_pattern"]}]
        if sp.get("semgrep_pattern_not"):
            pattern_list.append({"pattern-not-inside": sp["semgrep_pattern_not"]})
        rule["patterns"] = pattern_list
        if sp.get("fix"):
            rule["fix"] = sp["fix"]
    elif len(regex_patterns) == 1:
        rule["pattern-regex"] = regex_patterns[0]["regex"]
    else:
        rule["pattern-either"] = [
            {"pattern-regex": p["regex"]} for p in regex_patterns
        ]

    return yaml.dump(
        {"rules": [rule]},
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


def generate_test_file(
    rule_id: str,
    true_positive: str,
    true_negative: str,
) -> str:
    """Generate a Semgrep test file with ruleid/ok annotations.

    Args:
        rule_id: Full rule ID (e.g. ``lessons-db.python.bare-except-007``).
        true_positive: Code that should trigger the rule.
        true_negative: Code that should not trigger the rule.

    Returns:
        Python source string with annotation comments.
    """
    lines = [
        f"# ruleid: {rule_id}",
        true_positive,
        "",
        f"# ok: {rule_id}",
        true_negative,
        "",
    ]
    return "\n".join(lines)
