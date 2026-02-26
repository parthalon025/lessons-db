"""SQLite schema, migrations, and CRUD for lessons-db."""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    one_liner TEXT,
    description TEXT,
    cluster TEXT,
    tier TEXT NOT NULL DEFAULT 'observation',
    category TEXT,
    severity INTEGER NOT NULL DEFAULT 3,
    confidence TEXT NOT NULL DEFAULT 'emerging',
    scope TEXT,
    keywords TEXT,
    enforcement TEXT NOT NULL DEFAULT 'documentation',
    recurrence_count INTEGER NOT NULL DEFAULT 0,
    last_hit_date TEXT,
    created_date TEXT NOT NULL,
    source TEXT,
    parent_lesson_id INTEGER,
    markdown_path TEXT,
    FOREIGN KEY (parent_lesson_id) REFERENCES lessons(id)
);

CREATE TABLE IF NOT EXISTS corrective_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed',
    due_date TEXT,
    completed_date TEXT,
    created_date TEXT NOT NULL,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE TABLE IF NOT EXISTS affected_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    project TEXT,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE TABLE IF NOT EXISTS enforcement_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    rule_type TEXT NOT NULL DEFAULT 'semgrep',
    rule_content TEXT,
    created_date TEXT NOT NULL,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE TABLE IF NOT EXISTS detection_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    pattern_type TEXT NOT NULL,
    regex TEXT NOT NULL,
    description TEXT,
    language TEXT NOT NULL DEFAULT 'any',
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE TABLE IF NOT EXISTS near_misses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    event_type TEXT NOT NULL,
    rule_id TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE TABLE IF NOT EXISTS scan_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    snippet TEXT,
    status TEXT NOT NULL DEFAULT 'open',
    scan_date TEXT NOT NULL,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE INDEX IF NOT EXISTS idx_affected_files_path ON affected_files(file_path);
CREATE INDEX IF NOT EXISTS idx_affected_files_project ON affected_files(project);
CREATE INDEX IF NOT EXISTS idx_lessons_enforcement ON lessons(enforcement);
CREATE INDEX IF NOT EXISTS idx_corrective_status ON corrective_actions(status);
CREATE INDEX IF NOT EXISTS idx_near_misses_lesson ON near_misses(lesson_id);
CREATE INDEX IF NOT EXISTS idx_detection_patterns_type ON detection_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_scan_findings_status ON scan_findings(status);

-- Extension tables (created fresh; new columns added via _add_extension_columns)

CREATE TABLE IF NOT EXISTS surfacing_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    hook_point TEXT NOT NULL,
    context TEXT,
    outcome TEXT NOT NULL DEFAULT 'unknown',
    timestamp TEXT NOT NULL,
    session_id TEXT,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE TABLE IF NOT EXISTS templates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    template_type TEXT NOT NULL,
    content TEXT NOT NULL,
    applicable_contexts TEXT,
    created_date TEXT NOT NULL,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE TABLE IF NOT EXISTS capture_drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_content TEXT NOT NULL,
    extracted_data TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    created_date TEXT NOT NULL,
    source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    proposal_count INTEGER,
    confirmed_count INTEGER,
    result_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_surfacing_lesson ON surfacing_events(lesson_id);
CREATE INDEX IF NOT EXISTS idx_surfacing_outcome ON surfacing_events(outcome);
CREATE INDEX IF NOT EXISTS idx_templates_lesson ON templates(lesson_id);
"""


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create schema and return connection with Row factory."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    _add_extension_columns(conn)
    return conn


def _add_extension_columns(conn: sqlite3.Connection) -> None:
    """Add v2 extension columns to lessons table (idempotent).

    SQLite has no ALTER TABLE ADD COLUMN IF NOT EXISTS — catch OperationalError instead."""
    new_columns = [
        ("entry_type",   "TEXT NOT NULL DEFAULT 'lesson'"),
        ("polarity",     "TEXT NOT NULL DEFAULT 'negative'"),
        ("cluster_seed", "TEXT"),
        ("reuse_count",  "INTEGER NOT NULL DEFAULT 0"),
    ]
    for col_name, col_def in new_columns:
        try:
            conn.execute(f"ALTER TABLE lessons ADD COLUMN {col_name} {col_def}")
            conn.commit()
        except Exception:
            pass  # Column already exists — safe to ignore


LESSON_COLUMNS = {
    "title", "one_liner", "description", "cluster", "cluster_seed",
    "tier", "entry_type", "polarity", "category", "severity", "confidence",
    "scope", "keywords", "enforcement", "recurrence_count", "reuse_count",
    "last_hit_date", "created_date", "source", "parent_lesson_id", "markdown_path",
}


def insert_lesson(conn: sqlite3.Connection, data: dict) -> int:
    """Insert a lesson with defaults. Returns the new row id."""
    defaults = {
        "title": None,
        "one_liner": None,
        "description": None,
        "cluster": None,
        "tier": "observation",
        "category": None,
        "severity": 3,
        "confidence": "emerging",
        "scope": None,
        "keywords": None,
        "enforcement": "documentation",
        "recurrence_count": 0,
        "last_hit_date": None,
        "created_date": date.today().isoformat(),
        "source": None,
        "parent_lesson_id": None,
        "markdown_path": None,
    }
    row = {**defaults, **data}
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    cursor = conn.execute(
        f"INSERT INTO lessons ({col_names}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()
    return cursor.lastrowid


def get_lesson(conn: sqlite3.Connection, lesson_id: int) -> dict | None:
    """Return a lesson as dict, or None if not found."""
    row = conn.execute(
        "SELECT * FROM lessons WHERE id = ?", (lesson_id,)
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def update_lesson(conn: sqlite3.Connection, lesson_id: int, data: dict) -> None:
    """Update arbitrary fields on a lesson."""
    if not data:
        return
    invalid = set(data.keys()) - LESSON_COLUMNS
    if invalid:
        raise ValueError(f"Invalid column names: {invalid}")
    set_clause = ", ".join(f"{k} = ?" for k in data)
    values = list(data.values()) + [lesson_id]
    conn.execute(
        f"UPDATE lessons SET {set_clause} WHERE id = ?",
        values,
    )
    conn.commit()


def search_by_file(conn: sqlite3.Connection, file_path: str) -> list[dict]:
    """Search lessons by affected file path (LIKE match)."""
    rows = conn.execute(
        """
        SELECT l.id, l.one_liner, l.cluster, l.enforcement, l.severity
        FROM lessons l
        JOIN affected_files af ON l.id = af.lesson_id
        WHERE af.file_path LIKE ?
        """,
        (f"%{file_path}%",),
    ).fetchall()
    return [dict(r) for r in rows]


def search_by_enforcement(conn: sqlite3.Connection, enforcement: str) -> list[dict]:
    """Filter lessons by enforcement level."""
    rows = conn.execute(
        "SELECT * FROM lessons WHERE enforcement = ?",
        (enforcement,),
    ).fetchall()
    return [dict(r) for r in rows]


def insert_corrective_action(conn: sqlite3.Connection, data: dict) -> int:
    """Insert a corrective action. Auto-sets due_date to +7 days if missing."""
    defaults = {
        "lesson_id": None,
        "action": None,
        "status": "proposed",
        "due_date": (date.today() + timedelta(days=7)).isoformat(),
        "completed_date": None,
        "created_date": date.today().isoformat(),
    }
    row = {**defaults, **data}
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    cursor = conn.execute(
        f"INSERT INTO corrective_actions ({col_names}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()
    return cursor.lastrowid


def get_overdue_actions(conn: sqlite3.Connection) -> list[dict]:
    """Return corrective actions that are overdue (proposed/in_progress, past due)."""
    today = date.today().isoformat()
    rows = conn.execute(
        """
        SELECT * FROM corrective_actions
        WHERE status IN ('proposed', 'in_progress')
          AND due_date < ?
        """,
        (today,),
    ).fetchall()
    return [dict(r) for r in rows]


def insert_near_miss(conn: sqlite3.Connection, data: dict) -> int:
    """Log a hookify block/warn event as a near miss."""
    defaults = {
        "lesson_id": None,
        "file_path": None,
        "event_type": None,
        "rule_id": None,
        "timestamp": date.today().isoformat(),
    }
    row = {**defaults, **data}
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    cursor = conn.execute(
        f"INSERT INTO near_misses ({col_names}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()
    return cursor.lastrowid


def get_near_miss_hotspots(conn: sqlite3.Connection, limit: int = 10) -> list[dict]:
    """Top files by near-miss count."""
    rows = conn.execute(
        """
        SELECT file_path, COUNT(*) as count
        FROM near_misses
        GROUP BY file_path
        ORDER BY count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def insert_scan_finding(conn: sqlite3.Connection, data: dict) -> int:
    """Insert a scan result."""
    defaults = {
        "lesson_id": None,
        "rule_id": None,
        "file_path": None,
        "line_number": None,
        "snippet": None,
        "status": "open",
        "scan_date": date.today().isoformat(),
    }
    row = {**defaults, **data}
    cols = list(row.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    cursor = conn.execute(
        f"INSERT INTO scan_findings ({col_names}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )
    conn.commit()
    return cursor.lastrowid


def get_open_findings(conn: sqlite3.Connection) -> list[dict]:
    """Return open scan findings with lesson title and one_liner."""
    rows = conn.execute(
        """
        SELECT sf.*, l.title, l.one_liner
        FROM scan_findings sf
        JOIN lessons l ON sf.lesson_id = l.id
        WHERE sf.status = 'open'
        """,
    ).fetchall()
    return [dict(r) for r in rows]
