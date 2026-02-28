"""Parse markdown lesson files into structured dicts for DB migration."""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import date
from pathlib import Path

import yaml

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cluster mapping: category name → cluster letter
# ---------------------------------------------------------------------------
_CATEGORY_TO_CLUSTER: dict[str, str] = {
    "silent-failures": "A",
    "silent_failures": "A",
    "integration-boundaries": "B",
    "integration_boundaries": "B",
    "cold-start": "C",
    "cold_start": "C",
    "specification-drift": "D",
    "specification_drift": "D",
    "context-retrieval": "E",
    "context_retrieval": "E",
    "planning-control-flow": "F",
    "planning_control_flow": "F",
}


def extract_lesson_number(title_line: str) -> int | None:
    """Extract lesson number from '# Lesson #88: ...' → 88. None if absent."""
    m = re.match(r"^#\s+Lesson\s+#(\d+):", title_line)
    return int(m.group(1)) if m else None


def _extract_metadata(lines: list[str], key: str) -> str:
    """Extract value from '**Key:** value' line."""
    prefix = f"**{key}:**"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return ""


def _extract_section(lines: list[str], heading_prefix: str) -> str:
    """Extract text under a ## heading (matching prefix) until next ## or EOF."""
    collecting = False
    parts: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if collecting:
                break
            if line[3:].strip().startswith(heading_prefix):
                collecting = True
                continue
        elif collecting:
            parts.append(line)
    return "\n".join(parts).strip()


def _parse_corrective_actions(lines: list[str]) -> list[dict]:
    """Parse markdown table under ## Corrective Actions."""
    section = _extract_section(lines, "Corrective Actions")
    if not section:
        return []

    actions: list[dict] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        # Filter empty strings from leading/trailing pipes
        cells = [c for c in cells if c]
        if len(cells) < 3:
            continue
        # Skip header row and separator row
        if cells[0] == "#" or set(cells[0]) <= {"-"}:
            continue
        actions.append(
            {
                "description": cells[1],
                "status": cells[2].lower(),
            }
        )
    return actions


def _extract_related(lines: list[str]) -> list[int]:
    """Extract lesson numbers from '**Related:** #7 (desc), #37 (desc)'."""
    raw = _extract_metadata(lines, "Related")
    if not raw:
        return []
    return [int(m) for m in re.findall(r"#(\d+)", raw)]


def parse_lesson_file(path: Path) -> dict:
    """Parse a markdown lesson file into a structured dict."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Title (line 1)
    title_line = lines[0] if lines else ""
    lesson_number = extract_lesson_number(title_line)

    # Strip '# Lesson #NN: ' or '# Lesson: ' prefix
    title = title_line
    m = re.match(r"^#\s+Lesson\s+#\d+:\s*", title_line)
    if m:
        title = title_line[m.end() :]
    else:
        m2 = re.match(r"^#\s+Lesson:\s*", title_line)
        title = title_line[m2.end() :] if m2 else re.sub(r"^#\s+", "", title_line)

    # Metadata
    cluster_raw = _extract_metadata(lines, "Cluster")
    cluster = cluster_raw.split()[0] if cluster_raw else ""

    scope_raw = _extract_metadata(lines, "Scope")
    scope = scope_raw.strip("`")

    # Description = Observation + Analysis
    obs = _extract_section(lines, "Observation")
    analysis = _extract_section(lines, "Analysis")
    desc_parts = [p for p in (obs, analysis) if p]
    description = "\n\n".join(desc_parts)

    result = {
        "title": title,
        "lesson_number": lesson_number,
        "date": _extract_metadata(lines, "Date"),
        "system": _extract_metadata(lines, "System"),
        "tier": _extract_metadata(lines, "Tier").lower(),
        "category": _extract_metadata(lines, "Category").lower(),
        "cluster": cluster,
        "scope": scope,
        "keywords": _extract_metadata(lines, "Keywords"),
        "related": _extract_related(lines),
        "key_takeaway": _extract_section(lines, "Key Takeaway"),
        "corrective_actions": _parse_corrective_actions(lines),
        "description": description,
        "markdown_path": str(path),
    }
    if not result["title"] or not result["date"]:
        _log.warning("parse_lesson_file: missing required field in %s", path.name)
    _log.debug("parse_lesson_file: parsed %s", path.name)
    return result


# ---------------------------------------------------------------------------
# YAML frontmatter format (0001-*.md through 0091-*.md)
# ---------------------------------------------------------------------------


def _parse_yaml_frontmatter(path: Path) -> dict:
    """Extract and parse YAML frontmatter between leading --- delimiters.

    Raises ValueError if no frontmatter is found.
    """
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        raise ValueError(f"YAML frontmatter not found in {path.name}")
    fm = yaml.safe_load(m.group(1)) or {}

    # Parse body sections after frontmatter
    body = text[m.end() :].strip()
    body_lines = body.splitlines()

    def _body_section(heading: str) -> str:
        collecting = False
        parts: list[str] = []
        for line in body_lines:
            if line.startswith("## "):
                if collecting:
                    break
                if line[3:].strip() == heading:
                    collecting = True
                    continue
            elif collecting:
                parts.append(line)
        return "\n".join(parts).strip()

    fm["_observation"] = _body_section("Observation")
    fm["_insight"] = _body_section("Insight")
    fm["_lesson"] = _body_section("Lesson")
    return fm


def _insert_yaml_detection_pattern(conn: sqlite3.Connection, fm: dict, lesson_id: int) -> None:
    """Insert detection pattern from YAML frontmatter if present."""
    from lessons_db.db import insert_detection_pattern

    pattern_block = fm.get("pattern", {})
    if not (isinstance(pattern_block, dict) and pattern_block.get("type") == "syntactic"):
        return
    regex = pattern_block.get("regex", "")
    if not regex:
        return
    languages_raw = fm.get("languages", ["any"])
    language = languages_raw[0] if isinstance(languages_raw, list) and languages_raw else str(languages_raw)
    insert_detection_pattern(
        conn,
        {
            "lesson_id": lesson_id,
            "pattern_type": "syntactic",
            "regex": regex,
            "description": pattern_block.get("description", ""),
            "language": language,
        },
    )


def import_lesson_file(conn: sqlite3.Connection, path: Path) -> int | None:
    """Import a lesson file into the DB.

    Accepts two formats:
    - YAML frontmatter (--- ... --- delimiters): 0001-*.md through 0091-*.md
    - Heading+bold (**Key:** value): Documents workspace lessons

    Returns the new lesson DB id, or None if the lesson was already present
    (duplicate detected by markdown_path or title).
    """
    from lessons_db.db import insert_lesson

    # Try YAML frontmatter first; fall back to heading+bold format
    try:
        fm = _parse_yaml_frontmatter(path)
        use_yaml = True
    except ValueError:
        fm = None
        use_yaml = False

    if use_yaml:
        title: str = fm.get("title", path.stem)
    else:
        parsed = parse_lesson_file(path)
        title = parsed["title"] or path.stem

    path_str = str(path)

    # Duplicate check — by path first, then title (shared for both formats)
    existing_path = conn.execute("SELECT id FROM lessons WHERE markdown_path = ?", (path_str,)).fetchone()
    if existing_path:
        _log.debug("import_lesson_file: skipping %s (path already in DB)", path.name)
        return None

    existing_title = conn.execute("SELECT id FROM lessons WHERE lower(title) = lower(?)", (title,)).fetchone()
    if existing_title:
        _log.debug("import_lesson_file: skipping %s (title '%s' already in DB)", path.name, title)
        return None

    if use_yaml:
        # --- YAML path (unchanged) ---
        category: str = fm.get("category", "")
        cluster = _CATEGORY_TO_CLUSTER.get(category, "")

        # Scope: frontmatter stores as list ["language:python", ...] or string
        scope_raw = fm.get("scope", [])
        scope = ", ".join(scope_raw) if isinstance(scope_raw, list) else str(scope_raw)

        # one_liner: use fix field, or first 120 chars of lesson body
        fix = fm.get("fix", "")
        lesson_body = fm.get("_lesson", "")
        one_liner = fix or (lesson_body[:120] if lesson_body else "")

        # Description from body sections
        desc_parts = [p for p in (fm["_observation"], fm["_insight"], fm["_lesson"]) if p]
        description = "\n\n".join(desc_parts)

        lesson_data = {
            "title": title,
            "one_liner": one_liner,
            "description": description,
            "cluster": cluster,
            "tier": "lesson",
            "category": category,
            "scope": scope,
            "created_date": date.today().isoformat(),
            "source": "imported",
            "markdown_path": path_str,
        }
    else:
        # --- Heading+bold path ---
        category = parsed["category"]
        # Use explicit cluster letter if present (e.g. "B" from "**Cluster:** B (Integration Boundary)")
        # Fall back to category → letter mapping
        cluster = parsed["cluster"] or _CATEGORY_TO_CLUSTER.get(category, "")
        scope = parsed["scope"]
        one_liner = (parsed["key_takeaway"] or "")[:120]
        description = parsed["description"]
        lesson_data = {
            "title": title,
            "one_liner": one_liner,
            "description": description,
            "cluster": cluster,
            "tier": parsed["tier"] or "lesson",
            "category": category,
            "scope": scope,
            "created_date": parsed["date"] or date.today().isoformat(),
            "source": "imported",
            "markdown_path": path_str,
        }

    lesson_id = insert_lesson(conn, lesson_data)

    # Detection pattern — YAML frontmatter only (heading+bold format has no pattern: field)
    if use_yaml:
        _insert_yaml_detection_pattern(conn, fm, lesson_id)

    _log.info("import_lesson_file: inserted lesson id=%d from %s", lesson_id, path.name)
    return lesson_id
