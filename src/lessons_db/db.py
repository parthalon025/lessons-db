"""SQLite schema, migrations, and CRUD for lessons-db."""

import json
import logging
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_log = logging.getLogger(__name__)

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
    lesson_id INTEGER,
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

CREATE INDEX IF NOT EXISTS idx_surfacing_lesson_ctx ON surfacing_events(lesson_id, context);
CREATE INDEX IF NOT EXISTS idx_surfacing_outcome ON surfacing_events(lesson_id, outcome);
CREATE INDEX IF NOT EXISTS idx_templates_lesson ON templates(lesson_id);

CREATE TABLE IF NOT EXISTS suppression_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding_id TEXT NOT NULL,
    rejected_snippet TEXT NOT NULL,
    rejection_date TEXT NOT NULL,
    rejection_reason TEXT
);

CREATE TABLE IF NOT EXISTS scan_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mined_repos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_full_name TEXT NOT NULL UNIQUE,
    last_mined_date TEXT,
    commit_count INTEGER DEFAULT 0,
    lessons_extracted INTEGER DEFAULT 0,
    quality_score REAL,
    topics TEXT
);

CREATE TABLE IF NOT EXISTS mining_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    repos_searched INTEGER DEFAULT 0,
    commits_analyzed INTEGER DEFAULT 0,
    candidates_extracted INTEGER DEFAULT 0,
    diff_size_rejected INTEGER DEFAULT 0,
    gate0_rejected INTEGER DEFAULT 0,
    gate1_rejected INTEGER DEFAULT 0,
    gate2_rejected INTEGER DEFAULT 0,
    gate3_rejected INTEGER DEFAULT 0,
    gate4_rejected INTEGER DEFAULT 0,
    auto_approved INTEGER DEFAULT 0,
    drafted INTEGER DEFAULT 0,
    conflicts_flagged INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    duration_seconds REAL
);

CREATE INDEX IF NOT EXISTS idx_mined_repos_name ON mined_repos(repo_full_name);
CREATE INDEX IF NOT EXISTS idx_mining_runs_date ON mining_runs(run_date);

CREATE TABLE IF NOT EXISTS calibration_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    dataset TEXT NOT NULL DEFAULT 'BugsInPy',
    bugs_sampled INTEGER DEFAULT 0,
    bugs_with_valid_diffs INTEGER DEFAULT 0,
    extraction_attempted INTEGER DEFAULT 0,
    extraction_success INTEGER DEFAULT 0,
    gate0_pass INTEGER DEFAULT 0,
    gate14_pass INTEGER DEFAULT 0,
    pass_rate REAL,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_calibration_runs_date ON calibration_runs(run_date);

CREATE TABLE IF NOT EXISTS recurrence_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    hook_point TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    file_path TEXT,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id)
);

CREATE INDEX IF NOT EXISTS idx_recurrence_lesson_ts ON recurrence_events(lesson_id, timestamp);

CREATE TABLE IF NOT EXISTS fix_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lesson_id INTEGER NOT NULL,
    scan_finding_id INTEGER,
    file_path TEXT NOT NULL,
    line_number INTEGER,
    snippet TEXT,
    suggested_fix TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    github_issue_url TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id),
    FOREIGN KEY (scan_finding_id) REFERENCES scan_findings(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_fix_queue_dedup
    ON fix_queue(lesson_id, file_path, COALESCE(line_number, -1));
CREATE INDEX IF NOT EXISTS idx_fix_queue_status ON fix_queue(status);
"""


def _migrate_scan_findings_lesson_id_nullable(conn: sqlite3.Connection) -> None:
    """Make scan_findings.lesson_id nullable if it was created NOT NULL."""
    rows = conn.execute("PRAGMA table_info('scan_findings')").fetchall()
    col = next((r for r in rows if r["name"] == "lesson_id"), None)
    if col is None or col["notnull"] == 0:
        return  # already nullable or column doesn't exist — nothing to do

    # Recreate table with nullable lesson_id
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scan_findings_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lesson_id INTEGER REFERENCES lessons(id),
            rule_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_number INTEGER,
            snippet TEXT,
            status TEXT NOT NULL DEFAULT 'open',
            scan_date TEXT NOT NULL
        );
        INSERT INTO scan_findings_v2
            SELECT id, lesson_id, rule_id, file_path, line_number, snippet, status, scan_date
            FROM scan_findings;
        DROP TABLE scan_findings;
        ALTER TABLE scan_findings_v2 RENAME TO scan_findings;
    """)
    _log.info("_migrate_scan_findings_lesson_id_nullable: migrated scan_findings to nullable lesson_id")


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create schema and return connection with Row factory."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA_SQL)
    _add_extension_columns(conn)
    _seed_scan_state(conn)
    _migrate_scan_findings_lesson_id_nullable(conn)
    _log.debug("init_db: opened %s", db_path)
    return conn


def _add_extension_columns(conn: sqlite3.Connection) -> None:  # noqa: PLR0912
    """Add v2 extension columns to lessons table (idempotent).

    SQLite has no ALTER TABLE ADD COLUMN IF NOT EXISTS — catch OperationalError instead."""
    new_columns = [
        ("entry_type", "TEXT NOT NULL DEFAULT 'lesson'"),
        ("polarity", "TEXT NOT NULL DEFAULT 'negative'"),
        ("cluster_seed", "TEXT"),
        ("reuse_count", "INTEGER NOT NULL DEFAULT 0"),
    ]
    for col_name, col_def in new_columns:
        try:
            conn.execute(f"ALTER TABLE lessons ADD COLUMN {col_name} {col_def}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise

    # v3 cross-project detection columns on capture_drafts
    draft_columns = [
        ("detection_source", "TEXT NOT NULL DEFAULT 'stop_hook'"),
        ("confidence", "REAL"),
    ]
    for col_name, col_def in draft_columns:
        try:
            conn.execute(f"ALTER TABLE capture_drafts ADD COLUMN {col_name} {col_def}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise

    # v3 corrective_action shortcut column on lessons (denormalized from corrective_actions table)
    try:
        conn.execute("ALTER TABLE lessons ADD COLUMN corrective_action TEXT")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise

    # v4 why-capture columns (AI-framed cognitive gap fields)
    why_columns = [
        ("false_assumption", "TEXT"),
        ("detection_pattern", "TEXT"),
        ("invariant", "TEXT"),
    ]
    for col_name, col_def in why_columns:
        try:
            conn.execute(f"ALTER TABLE lessons ADD COLUMN {col_name} {col_def}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise

    # v6 recurrence_events table — now in SCHEMA_SQL; CREATE IF NOT EXISTS for
    # existing DBs (idempotent).
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS recurrence_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lesson_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            hook_point TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            file_path TEXT,
            FOREIGN KEY (lesson_id) REFERENCES lessons(id)
        );
        CREATE INDEX IF NOT EXISTS idx_recurrence_lesson_ts
            ON recurrence_events(lesson_id, timestamp);
    """)

    # v7 fix_queue table — actionable work items for Claude or GitHub issues.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS fix_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lesson_id INTEGER NOT NULL,
            scan_finding_id INTEGER,
            file_path TEXT NOT NULL,
            line_number INTEGER,
            snippet TEXT,
            suggested_fix TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            github_issue_url TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (lesson_id) REFERENCES lessons(id),
            FOREIGN KEY (scan_finding_id) REFERENCES scan_findings(id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fix_queue_dedup
            ON fix_queue(lesson_id, file_path, COALESCE(line_number, -1));
        CREATE INDEX IF NOT EXISTS idx_fix_queue_status ON fix_queue(status);
    """)

    # v5 per-gate visibility columns — now in SCHEMA_SQL; keep ALTER TABLE for
    # existing DBs upgraded from the previous schema (idempotent, duplicate-safe).
    for col_name, col_def in [
        ("candidates_extracted", "INTEGER DEFAULT 0"),
        ("diff_size_rejected", "INTEGER DEFAULT 0"),
        ("gate2_rejected", "INTEGER DEFAULT 0"),
        ("gate3_rejected", "INTEGER DEFAULT 0"),
        ("gate4_rejected", "INTEGER DEFAULT 0"),
    ]:
        try:
            conn.execute(f"ALTER TABLE mining_runs ADD COLUMN {col_name} {col_def}")
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e):
                raise


def _seed_scan_state(conn: sqlite3.Connection) -> None:
    """Seed scan_state defaults (idempotent via INSERT OR IGNORE)."""
    defaults = [
        ("last_scan_timestamp", "1970-01-01T00:00:00"),
        ("auto_approve_threshold", "0.85"),
    ]
    for key, value in defaults:
        conn.execute("INSERT OR IGNORE INTO scan_state (key, value) VALUES (?, ?)", [key, value])
    conn.commit()


def get_scan_state(conn: sqlite3.Connection, key: str) -> str | None:
    """Get a value from scan_state by key. Returns None if key missing."""
    row = conn.execute("SELECT value FROM scan_state WHERE key = ?", [key]).fetchone()
    return row["value"] if row else None


def set_scan_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert a key-value pair in scan_state."""
    conn.execute("INSERT OR REPLACE INTO scan_state (key, value) VALUES (?, ?)", [key, value])
    conn.commit()


LESSON_COLUMNS = {
    "title",
    "one_liner",
    "description",
    "cluster",
    "cluster_seed",
    "tier",
    "entry_type",
    "polarity",
    "category",
    "severity",
    "confidence",
    "scope",
    "keywords",
    "enforcement",
    "recurrence_count",
    "reuse_count",
    "last_hit_date",
    "created_date",
    "source",
    "parent_lesson_id",
    "markdown_path",
    "corrective_action",
    "false_assumption",
    "detection_pattern",
    "invariant",
}


def insert_lesson(conn: sqlite3.Connection, data: dict) -> int:
    """Insert a lesson with defaults. Returns the new row id."""
    invalid = set(data.keys()) - LESSON_COLUMNS
    if invalid:
        raise ValueError(f"Invalid column names: {invalid}")
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
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def get_lesson(conn: sqlite3.Connection, lesson_id: int) -> dict | None:
    """Return a lesson as dict, or None if not found."""
    row = conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
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


def insert_detection_pattern(conn: sqlite3.Connection, data: dict) -> int:
    """Insert a detection pattern for a lesson. Returns new row id."""
    defaults = {"description": None, "language": "any"}
    row = {**defaults, **data}
    cols = list(row.keys())
    cursor = conn.execute(
        f"INSERT INTO detection_patterns ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [row[c] for c in cols],
    )
    conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


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
    assert cursor.lastrowid is not None
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
    assert cursor.lastrowid is not None
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
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def get_open_findings(conn: sqlite3.Connection) -> list[dict]:
    """Return open scan findings with lesson title and one_liner."""
    rows = conn.execute(
        """
        SELECT sf.*, l.title, l.one_liner
        FROM scan_findings sf
        LEFT JOIN lessons l ON sf.lesson_id = l.id
        WHERE sf.status = 'open'
        """,
    ).fetchall()
    return [dict(r) for r in rows]


def insert_mined_repo(
    conn: sqlite3.Connection,
    repo_full_name: str,
    topics: list[str] | None = None,
) -> int:
    """Insert or return existing mined_repos entry (idempotent by repo_full_name)."""
    existing = conn.execute("SELECT id FROM mined_repos WHERE repo_full_name = ?", (repo_full_name,)).fetchone()
    if existing:
        return existing["id"]
    cursor = conn.execute(
        "INSERT INTO mined_repos (repo_full_name, topics) VALUES (?, ?)",
        (repo_full_name, json.dumps(topics or [])),
    )
    conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def get_mined_repo(conn: sqlite3.Connection, repo_full_name: str) -> dict | None:
    row = conn.execute("SELECT * FROM mined_repos WHERE repo_full_name = ?", (repo_full_name,)).fetchone()
    return dict(row) if row else None


def update_mined_repo(
    conn: sqlite3.Connection,
    repo_full_name: str,
    commit_count: int = 0,
    lessons_extracted: int = 0,
    quality_score: float | None = None,
) -> None:
    conn.execute(
        """UPDATE mined_repos
           SET last_mined_date = date('now'),
               commit_count = commit_count + ?,
               lessons_extracted = lessons_extracted + ?,
               quality_score = COALESCE(?, quality_score)
           WHERE repo_full_name = ?""",
        (commit_count, lessons_extracted, quality_score, repo_full_name),
    )
    conn.commit()


def insert_mining_run(
    conn: sqlite3.Connection,
    repos_searched: int = 0,
    commits_analyzed: int = 0,
    candidates_extracted: int = 0,
    diff_size_rejected: int = 0,
    gate0_rejected: int = 0,
    gate1_rejected: int = 0,
    gate2_rejected: int = 0,
    gate3_rejected: int = 0,
    gate4_rejected: int = 0,
    auto_approved: int = 0,
    drafted: int = 0,
    conflicts_flagged: int = 0,
    error_count: int = 0,
    duration_seconds: float | None = None,
) -> int:
    cursor = conn.execute(
        """INSERT INTO mining_runs
           (run_date, repos_searched, commits_analyzed, candidates_extracted,
            diff_size_rejected, gate0_rejected,
            gate1_rejected, gate2_rejected, gate3_rejected, gate4_rejected,
            auto_approved, drafted, conflicts_flagged, error_count, duration_seconds)
           VALUES (date('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            repos_searched,
            commits_analyzed,
            candidates_extracted,
            diff_size_rejected,
            gate0_rejected,
            gate1_rejected,
            gate2_rejected,
            gate3_rejected,
            gate4_rejected,
            auto_approved,
            drafted,
            conflicts_flagged,
            error_count,
            duration_seconds,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_capture_draft(
    conn: sqlite3.Connection,
    raw_content: str,
    extracted_data: str,
    source: str,
    confidence: float | None = None,
) -> int:
    """Insert a pending capture draft. Returns the new row id."""
    cursor = conn.execute(
        "INSERT INTO capture_drafts "
        "(raw_content, extracted_data, status, created_date, source, confidence) "
        "VALUES (?, ?, 'pending', date('now'), ?, ?)",
        [raw_content, extracted_data, source, confidence],
    )
    conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def insert_calibration_run(
    conn: sqlite3.Connection,
    dataset: str = "BugsInPy",
    bugs_sampled: int = 0,
    bugs_with_valid_diffs: int = 0,
    extraction_attempted: int = 0,
    extraction_success: int = 0,
    gate0_pass: int = 0,
    gate14_pass: int = 0,
    notes: str | None = None,
) -> int:
    pass_rate = gate0_pass / bugs_sampled if bugs_sampled else 0.0
    cursor = conn.execute(
        """INSERT INTO calibration_runs
           (run_date, dataset, bugs_sampled, bugs_with_valid_diffs,
            extraction_attempted, extraction_success, gate0_pass, gate14_pass,
            pass_rate, notes)
           VALUES (date('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            dataset,
            bugs_sampled,
            bugs_with_valid_diffs,
            extraction_attempted,
            extraction_success,
            gate0_pass,
            gate14_pass,
            pass_rate,
            notes,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def insert_recurrence_event(
    conn: sqlite3.Connection,
    lesson_id: int,
    hook_point: str,
    trigger_type: str,
    file_path: str | None = None,
) -> int:
    """Log a recurrence event for a lesson. Returns new row id."""
    from datetime import UTC, datetime

    cursor = conn.execute(
        "INSERT INTO recurrence_events (lesson_id, timestamp, hook_point, trigger_type, file_path) "
        "VALUES (?, ?, ?, ?, ?)",
        [lesson_id, datetime.now(UTC).isoformat(), hook_point, trigger_type, file_path],
    )
    conn.commit()
    assert cursor.lastrowid is not None
    return cursor.lastrowid


def get_recurrence_velocity(
    conn: sqlite3.Connection,
    lesson_id: int,
    window_days: int = 7,
) -> int:
    """Count recurrence events for a lesson in the past window_days."""
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
    row = conn.execute(
        "SELECT COUNT(*) FROM recurrence_events WHERE lesson_id = ? AND timestamp >= ?",
        [lesson_id, cutoff],
    ).fetchone()
    return row[0] if row else 0


def get_velocity_warnings(
    conn: sqlite3.Connection,
    window_days: int = 7,
    threshold: int = 2,
) -> list[dict]:
    """Return all lessons with recurrence velocity >= threshold in the past window_days."""
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=window_days)).isoformat()
    rows = conn.execute(
        """
        SELECT re.lesson_id, l.title, l.enforcement, l.severity, COUNT(*) as hit_count
        FROM recurrence_events re
        JOIN lessons l ON re.lesson_id = l.id
        WHERE re.timestamp >= ?
        GROUP BY re.lesson_id
        HAVING COUNT(*) >= ?
        ORDER BY hit_count DESC
        """,
        [cutoff, threshold],
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Fix queue
# ---------------------------------------------------------------------------


def add_to_fix_queue(
    conn: sqlite3.Connection,
    lesson_id: int,
    file_path: str,
    line_number: int | None = None,
    snippet: str | None = None,
    suggested_fix: str | None = None,
    scan_finding_id: int | None = None,
) -> int | None:
    """Add a fixable item to the fix queue. Idempotent via unique index on
    (lesson_id, file_path, line_number). Returns new row id, or None if duplicate.
    """
    try:
        cursor = conn.execute(
            "INSERT INTO fix_queue "
            "(lesson_id, scan_finding_id, file_path, line_number, snippet, suggested_fix, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                lesson_id,
                scan_finding_id,
                file_path,
                line_number,
                snippet,
                suggested_fix,
                datetime.now(UTC).isoformat(),
            ],
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None  # duplicate — already queued


def get_next_fix(conn: sqlite3.Connection) -> dict | None:
    """Return the highest-priority pending fix: highest severity lesson first,
    then earliest created_at. Returns None if queue is empty.
    """
    result = get_fix_queue(conn, status="pending", limit=1)
    return result[0] if result else None


def get_fix_queue(
    conn: sqlite3.Connection,
    status: str = "pending",
    limit: int = 50,
) -> list[dict]:
    """Return fix queue entries filtered by status."""
    rows = conn.execute(
        """
        SELECT fq.*, l.title, l.one_liner, l.severity, l.enforcement
        FROM fix_queue fq
        JOIN lessons l ON fq.lesson_id = l.id
        WHERE fq.status = ?
        ORDER BY l.severity DESC, fq.created_at ASC
        LIMIT ?
        """,
        [status, limit],
    ).fetchall()
    return [dict(r) for r in rows]


def update_fix_status(
    conn: sqlite3.Connection,
    fix_id: int,
    status: str,
    github_issue_url: str | None = None,
) -> None:
    """Update the status (and optionally github_issue_url) of a fix queue entry."""
    conn.execute(
        "UPDATE fix_queue SET status = ?, github_issue_url = COALESCE(?, github_issue_url) WHERE id = ?",
        [status, github_issue_url, fix_id],
    )
    conn.commit()
