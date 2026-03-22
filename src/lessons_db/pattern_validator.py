"""Gates 0a and 0b: syntax check + regex self-consistency.

Run BEFORE existing pattern_verify gates to reject ~60-70% of
LLM-generated noise before any Ollama calls.

Gate 0a (validate_syntax):
  - Required fields: title, one_liner, bad_code, good_code
  - bad_code and good_code parse as valid Python (ast.parse)
  - Regex compiles without error (if present)
  - Code length 2-500 lines

Gate 0b (validate_regex_self_consistency):
  - regex matches bad_code (must match)
  - regex does NOT match good_code (must not match)
  - No catastrophic backtracking (timeout > 1s = reject)
"""

import ast
import logging
import re
import threading

_log = logging.getLogger(__name__)

MIN_CODE_LINES = 2
MAX_CODE_LINES = 500
REGEX_TIMEOUT_SECONDS = 1


def validate_syntax(
    title: str,
    one_liner: str,
    bad_code: str,
    good_code: str,
    regex: str | None = None,
) -> dict:
    """Gate 0a: structural and syntax validation.

    Returns {"passed": bool, "reason": str | None}.
    """
    # Required fields
    for field_name, value in [
        ("title", title),
        ("one_liner", one_liner),
        ("bad_code", bad_code),
        ("good_code", good_code),
    ]:
        if not value or not value.strip():
            return {"passed": False, "reason": f"missing required field: {field_name}"}

    # Code length
    bad_lines = len(bad_code.splitlines())
    if not (MIN_CODE_LINES <= bad_lines <= MAX_CODE_LINES):
        return {
            "passed": False,
            "reason": (f"bad_code length {bad_lines} not in [{MIN_CODE_LINES}, {MAX_CODE_LINES}]"),
        }

    # Python syntax
    for label, code in [("bad_code", bad_code), ("good_code", good_code)]:
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return {"passed": False, "reason": f"syntax error in {label}: {exc}"}

    # Regex compiles
    if regex:
        try:
            re.compile(regex)
        except re.error as exc:
            return {"passed": False, "reason": f"regex compile error: {exc}"}

    return {"passed": True, "reason": None}


def _match_with_timeout(
    pattern: str,
    text: str,
    timeout: int = REGEX_TIMEOUT_SECONDS,
) -> bool | None:
    """Return True if pattern matches text. Returns None on timeout."""
    result: list[bool | None] = [None]

    def _run() -> None:
        flags = re.MULTILINE | re.DOTALL
        result[0] = bool(re.search(pattern, text, flags))

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None  # timeout — catastrophic backtracking suspected
    return result[0]


def validate_regex_self_consistency(
    regex: str | None,
    bad_code: str,
    good_code: str,
) -> dict:
    """Gate 0b: regex must match bad_code and NOT match good_code.

    Returns {"passed": bool, "reason": str | None, "skipped": bool}.
    """
    if not regex or not regex.strip():
        return {"passed": True, "reason": None, "skipped": True}

    # Test against bad_code — must match
    bad_match = _match_with_timeout(regex, bad_code)
    if bad_match is None:
        return {
            "passed": False,
            "reason": "regex timeout on bad_code — catastrophic backtracking suspected",
            "skipped": False,
        }
    if not bad_match:
        return {
            "passed": False,
            "reason": "regex did not match bad_code — pattern is incorrect",
            "skipped": False,
        }

    # Test against good_code — must NOT match
    good_match = _match_with_timeout(regex, good_code)
    if good_match is None:
        return {
            "passed": False,
            "reason": "regex timeout on good_code — catastrophic backtracking suspected",
            "skipped": False,
        }
    if good_match:
        return {
            "passed": False,
            "reason": "regex matched good_code — pattern is too broad (false positive)",
            "skipped": False,
        }

    return {"passed": True, "reason": None, "skipped": False}


def run_gate0(candidate: dict) -> bool:
    """Run Gates 0a and 0b on a candidate dict. Returns True if both pass.

    Shared by the GitHub miner and the BugsInPy calibrator so the gate logic
    lives in one place.
    """
    polarity = candidate.get("polarity", "negative")

    if polarity != "positive":
        result = validate_syntax(
            title=candidate.get("title", ""),
            one_liner=candidate.get("one_liner", ""),
            bad_code=candidate.get("bad_code", ""),
            good_code=candidate.get("good_code", ""),
            regex=candidate.get("regex"),
        )
        if not result["passed"]:
            return False
    elif not all(
        [
            candidate.get("title", "").strip(),
            candidate.get("one_liner", "").strip(),
            candidate.get("good_code", "").strip(),
        ]
    ):
        return False

    if candidate.get("regex"):
        reg = validate_regex_self_consistency(
            candidate["regex"],
            candidate.get("bad_code", ""),
            candidate.get("good_code", ""),
        )
        if not reg["passed"]:
            return False

    return True
