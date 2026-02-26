"""Positive knowledge promotion ladder.

reuse_count >= 1 → tested
reuse_count >= 2 → proven  (template generated)
reuse_count >= 3 → standard
"""

from datetime import date

from lessons_db.config import (
    PROMOTION_STANDARD_THRESHOLD,
    PROMOTION_TEMPLATE_THRESHOLD,
    PROMOTION_TESTED_THRESHOLD,
)


def record_reuse(conn, lesson_id: int) -> str:
    """Increment reuse_count and promote tier if threshold reached.

    Returns the new tier name."""
    row = conn.execute(
        "SELECT reuse_count, tier, one_liner, description FROM lessons WHERE id = ?",
        [lesson_id],
    ).fetchone()
    if not row:
        raise ValueError(f"Lesson {lesson_id} not found")

    reuse_count = (row["reuse_count"] or 0) + 1
    tier = row["tier"]
    one_liner = row["one_liner"] or ""
    description = row["description"] or ""

    if reuse_count >= PROMOTION_STANDARD_THRESHOLD:
        tier = "standard"
    elif reuse_count >= PROMOTION_TEMPLATE_THRESHOLD:
        tier = "proven"
        _generate_template(conn, lesson_id, one_liner, description)
    elif reuse_count >= PROMOTION_TESTED_THRESHOLD and tier == "noticed":
        tier = "tested"

    conn.execute(
        "UPDATE lessons SET reuse_count = ?, tier = ? WHERE id = ?",
        [reuse_count, tier, lesson_id],
    )
    conn.commit()
    return tier


def list_templates(conn) -> list[dict]:
    """Return all generated templates with associated lesson data."""
    rows = conn.execute(
        """SELECT t.id, t.lesson_id, t.template_type, t.content, t.created_date,
                  l.one_liner, l.tier, l.category
           FROM templates t JOIN lessons l ON t.lesson_id = l.id
           ORDER BY t.created_date DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def apply_template(conn, lesson_id: int) -> str | None:
    """Return template content for a lesson, or None if not yet generated."""
    row = conn.execute(
        "SELECT content FROM templates WHERE lesson_id = ? ORDER BY id DESC LIMIT 1",
        [lesson_id],
    ).fetchone()
    return row["content"] if row else None


def _generate_template(conn, lesson_id: int, one_liner: str,
                        description: str) -> None:
    """Auto-generate a scaffold template from a proven positive entry."""
    template_type = "approach"
    lower = one_liner.lower()
    if any(w in lower for w in ("test", "verify", "check", "assert", "coverage")):
        template_type = "checklist"
    elif any(w in lower for w in ("scaffold", "init", "create", "bootstrap", "setup")):
        template_type = "scaffold"
    elif any(w in lower for w in ("snippet", "pattern", "code", "implementation")):
        template_type = "snippet"

    content = (
        f"## Pattern: {one_liner}\n\n"
        f"{description}\n\n"
        "### When to apply\n\n"
        "_[Fill in: context, preconditions, trigger signals]_\n\n"
        "### When NOT to apply\n\n"
        "_[Fill in: constraints, anti-patterns, edge cases]_\n\n"
        "### Steps\n\n"
        "_[Fill in: step-by-step implementation guide]_\n"
    )

    conn.execute(
        "INSERT INTO templates (lesson_id, template_type, content, created_date) "
        "VALUES (?, ?, ?, ?)",
        [lesson_id, template_type, content, date.today().isoformat()],
    )
