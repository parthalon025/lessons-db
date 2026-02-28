"""File checking against lesson detection patterns."""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


_LANG_EXT_MAP = {
    "python": ["py"],
    "javascript": ["js", "jsx", "mjs"],
    "typescript": ["ts", "tsx"],
    "bash": ["sh", "bash"],
    "go": ["go"],
    "rust": ["rs"],
}


def _get_patterns(conn):
    """Load all syntactic detection patterns joined with lesson metadata."""
    cursor = conn.execute(
        "SELECT dp.lesson_id, dp.regex, dp.description, dp.language, "
        "l.title, l.one_liner, l.scope "
        "FROM detection_patterns dp "
        "JOIN lessons l ON dp.lesson_id = l.id "
        "WHERE dp.pattern_type IN ('syntactic', 'regex') "
        "AND dp.regex != ''"
    )
    return cursor.fetchall()


def _pattern_applies(row, fpath, scope):
    """Check if a detection pattern applies to the given file and scope."""
    _lesson_id, _regex, _desc, lang, _title, _one_liner, lesson_scope = row

    if scope and lesson_scope and not _scope_matches(lesson_scope, scope):
        return False

    if lang and lang != "any":
        ext = Path(fpath).suffix.lstrip(".")
        allowed_exts = _LANG_EXT_MAP.get(lang, [lang])
        if ext not in allowed_exts:
            return False

    return True


def check_files(
    conn,
    lance_dir,
    file_paths: list[str],
    scope: str | None = None,
) -> list[dict]:
    """Check file contents against detection patterns and semantic search.

    Returns list of violation dicts:
        {lesson_id, title, one_liner, file_path, line_number, pattern, source}
    source is 'syntactic' or 'semantic'.
    """
    violations = []
    seen = set()
    rows = _get_patterns(conn)

    for fpath in file_paths:
        try:
            content = Path(fpath).read_text(errors="replace")
        except (OSError, PermissionError) as e:
            logger.warning("Skipping unreadable file %s: %s", fpath, e)
            continue

        lines = content.splitlines()

        for row in rows:
            if not _pattern_applies(row, fpath, scope):
                continue

            lesson_id, regex, _desc, _lang, title, one_liner, _lesson_scope = row

            try:
                pattern = re.compile(regex)
            except re.error:
                logger.warning("Invalid regex in lesson %d: %s", lesson_id, regex)
                continue

            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    key = (lesson_id, fpath)
                    if key not in seen:
                        seen.add(key)
                        violations.append(
                            {
                                "lesson_id": lesson_id,
                                "title": title or "",
                                "one_liner": one_liner or title or "",
                                "file_path": fpath,
                                "line_number": i,
                                "pattern": regex,
                                "source": "syntactic",
                            }
                        )
                    break  # One hit per lesson per file is enough

    # Sort by file_path, then line_number
    violations.sort(key=lambda v: (v["file_path"], v["line_number"]))
    return violations


def _scope_matches(lesson_scope: str, target_scope: str) -> bool:
    """Check if lesson scope intersects with target scope.

    Scope is comma-separated tags. Match if any tag overlaps,
    or if lesson scope contains 'universal'.
    """
    lesson_tags = {t.strip().lower() for t in lesson_scope.split(",")}
    if "universal" in lesson_tags:
        return True
    target_tags = {t.strip().lower() for t in target_scope.split(",")}
    return bool(lesson_tags & target_tags)
