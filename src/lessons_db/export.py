"""Generate markdown from DB records."""


def format_lesson_markdown(lesson: dict) -> str:
    """Format a lesson dict as markdown matching FRAMEWORK.md template."""
    lines = [
        f"# Lesson #{lesson['id']}: {lesson['title']}",
        "",
        f"**Date:** {lesson.get('created_date', '')}",
        f"**Tier:** {lesson.get('tier', 'observation')}",
        f"**Category:** {lesson.get('category', '')}",
        f"**Cluster:** {lesson.get('cluster', '')}",
        f"**Keywords:** {lesson.get('keywords', '')}",
        f"**Enforcement:** {lesson.get('enforcement', 'documentation')}",
        f"**Recurrences:** {lesson.get('recurrence_count', 0)}",
        "",
        "## Key Takeaway",
        "",
        lesson.get("one_liner", ""),
        "",
    ]

    if lesson.get("description"):
        lines.extend([
            "## Description",
            "",
            lesson["description"],
            "",
        ])

    return "\n".join(lines)


def format_status_line(
    total_lessons: int,
    overdue_actions: int,
    open_findings: int,
) -> str:
    """Format a one-line status for SessionStart hook."""
    parts = [f"lessons-db: {total_lessons} lessons"]
    if overdue_actions:
        parts.append(f"{overdue_actions} overdue actions")
    if open_findings:
        parts.append(f"{open_findings} open findings")
    return ", ".join(parts)
