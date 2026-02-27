"""Parse markdown lesson files into structured dicts for DB migration."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)


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
