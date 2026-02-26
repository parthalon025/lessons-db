# Lessons-DB Extension Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend lessons-db with positive knowledge capture, adaptive HDBSCAN clustering, a learning pipeline that improves retrieval precision over time, and a promotion ladder that escalates proven patterns into templates.

**Architecture:** Four new modules (capture, cluster, learn, promote) layered on top of existing SQLite + LanceDB infrastructure. Schema extended with 4 new columns and 2 new tables via `ALTER TABLE ADD COLUMN`. All new code follows TDD: test first, minimal implementation, green, commit. CLI extended with new subcommand groups.

**Tech Stack:** Python 3.14, SQLite3 (stdlib), LanceDB 0.29, Click 8, Requests (Ollama queue at port 7683). Clustering requires `umap-learn` + `hdbscan` as optional dependencies.

**Design doc:** `~/Documents/docs/plans/2026-02-26-lessons-db-extension-design.md`
**Base plan (Tasks 1-13, complete):** `~/Documents/docs/plans/2026-02-26-lessons-db-implementation.md`
**Working dir:** `~/Documents/projects/lessons-db/`
**Run tests with:** `.venv/bin/python -m pytest --timeout=120 -x -q`

---

## Task 14: Schema Extension

**Files:**
- Modify: `src/lessons_db/config.py`
- Modify: `src/lessons_db/db.py`
- Modify: `tests/test_db.py`

**What:** Add `entry_type`, `polarity`, `cluster_seed`, `reuse_count` columns to `lessons`. Add `surfacing_events` and `templates` tables. Extend `init_db` to create new tables. Update `LESSON_COLUMNS` so `insert_lesson` accepts the new fields.

**Key fact:** `cluster` is already `TEXT` (nullable) in the existing schema — no table rebuild needed. Pure `ALTER TABLE ADD COLUMN` statements added to `init_db`.

---

### Step 1: Write failing tests for new columns

Add to `tests/test_db.py`:

```python
class TestSchemaExtension:
    """Tests for v2 schema extension columns and tables."""

    def test_lessons_has_entry_type_column(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "entry_type" in cols

    def test_lessons_has_polarity_column(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "polarity" in cols

    def test_lessons_has_cluster_seed_column(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "cluster_seed" in cols

    def test_lessons_has_reuse_count_column(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "reuse_count" in cols

    def test_surfacing_events_table_exists(self, db_path):
        conn = init_db(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "surfacing_events" in tables

    def test_templates_table_exists(self, db_path):
        conn = init_db(db_path)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert "templates" in tables

    def test_insert_lesson_with_polarity(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(conn, {
            "title": "Test positive entry",
            "one_liner": "Dual-axis testing catches integration bugs",
            "created_date": "2026-02-26",
            "polarity": "positive",
            "entry_type": "pattern",
        })
        row = get_lesson(conn, lid)
        assert row["polarity"] == "positive"
        assert row["entry_type"] == "pattern"

    def test_reuse_count_defaults_to_zero(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(conn, {
            "title": "T", "one_liner": "X", "created_date": "2026-02-26",
        })
        row = get_lesson(conn, lid)
        assert row["reuse_count"] == 0
```

### Step 2: Run tests to confirm they fail

```bash
cd ~/Documents/projects/lessons-db
.venv/bin/python -m pytest tests/test_db.py::TestSchemaExtension -v
```

Expected: 8 FAILED (columns and tables don't exist yet).

### Step 3: Extend config.py

Replace the entire contents of `src/lessons_db/config.py`:

```python
"""Central configuration for lessons-db."""

from pathlib import Path

# Data directory
DATA_DIR = Path.home() / ".local" / "share" / "lessons-db"
SQLITE_PATH = DATA_DIR / "lessons.db"
LANCE_DIR = DATA_DIR / "lance"
RULES_DIR = DATA_DIR / "rules"

# Source lesson files (for migration)
LESSONS_SOURCE_DIR = Path.home() / "Documents" / "docs" / "lessons"

# Ollama queue API (embeddings + analysis)
OLLAMA_QUEUE_URL = "http://127.0.0.1:7683"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIMS = 768
ANALYSIS_MODEL = "qwen2.5:7b"

# Thresholds
DEDUP_THRESHOLD = 0.85
QUALITY_MIN_SCORE = 3
NEAR_MISS_TOP_N = 10

# Semgrep
SEMGREP_RULES_DIR = DATA_DIR / "rules"

# Positive promotion thresholds
PROMOTION_TESTED_THRESHOLD = 1    # reuse_count >= 1 → tested
PROMOTION_TEMPLATE_THRESHOLD = 2  # reuse_count >= 2 → proven, template generated
PROMOTION_STANDARD_THRESHOLD = 3  # reuse_count >= 3 → standard

# Valid enums (negative OIL ladder)
VALID_TIERS_NEGATIVE = ("observation", "insight", "lesson", "lesson_learned")

# Valid enums (positive ladder)
VALID_TIERS_POSITIVE = ("noticed", "tested", "proven", "standard")

# Combined
VALID_TIERS = VALID_TIERS_NEGATIVE + VALID_TIERS_POSITIVE

VALID_CATEGORIES_NEGATIVE = (
    "data-model", "registration", "cold-start", "integration",
    "deployment", "monitoring", "ui", "testing", "performance", "security",
)
VALID_CATEGORIES_POSITIVE = (
    "architecture-pattern", "planning-technique", "workflow-optimization",
    "value-multiplier", "debugging-strategy", "testing-pattern",
    "integration-approach", "tooling-innovation",
)
VALID_CATEGORIES = VALID_CATEGORIES_NEGATIVE + VALID_CATEGORIES_POSITIVE

VALID_CLUSTERS = ("A", "B", "C", "D", "E", "F")  # Historical seeds only
VALID_POLARITIES = ("negative", "positive")
VALID_ENTRY_TYPES = ("lesson", "insight", "pattern", "innovation")
VALID_ENFORCEMENT = (
    "documentation", "semgrep_warning", "semgrep_error", "semgrep_autofix",
)
VALID_SOURCES = (
    "manual", "auto_diff", "auto_transcript", "auto_test",
    "community", "migrated", "auto_design_doc", "auto_plan",
)
```

### Step 4: Extend db.py — add new columns and tables

At the bottom of `SCHEMA_SQL` (before the closing `"""`), add the extension DDL. Find the line:

```python
CREATE INDEX IF NOT EXISTS idx_scan_findings_status ON scan_findings(status);
"""
```

Replace it with:

```python
CREATE INDEX IF NOT EXISTS idx_scan_findings_status ON scan_findings(status);

-- Extension: new columns added via ALTER TABLE in init_db (SQLite ADD COLUMN)
-- Tables below are created fresh via CREATE TABLE IF NOT EXISTS

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
```

Then update `init_db` to add the new columns to any existing DB (idempotent):

```python
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
    """Add v2 extension columns to lessons table (idempotent via IF NOT EXISTS workaround).

    SQLite has no ALTER TABLE ADD COLUMN IF NOT EXISTS — we catch OperationalError instead."""
    new_columns = [
        ("entry_type", "TEXT NOT NULL DEFAULT 'lesson'"),
        ("polarity",   "TEXT NOT NULL DEFAULT 'negative'"),
        ("cluster_seed", "TEXT"),
        ("reuse_count",  "INTEGER NOT NULL DEFAULT 0"),
    ]
    for col_name, col_def in new_columns:
        try:
            conn.execute(f"ALTER TABLE lessons ADD COLUMN {col_name} {col_def}")
            conn.commit()
        except Exception:
            pass  # Column already exists — safe to ignore
```

Also update `LESSON_COLUMNS` to include the new fields:

```python
LESSON_COLUMNS = {
    "title", "one_liner", "description", "cluster", "cluster_seed",
    "tier", "entry_type", "polarity", "category", "severity", "confidence",
    "scope", "keywords", "enforcement", "recurrence_count", "reuse_count",
    "last_hit_date", "created_date", "source", "parent_lesson_id", "markdown_path",
}
```

### Step 5: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/test_db.py -v
```

Expected: All 21 tests PASS (13 original + 8 new).

### Step 6: Run full suite to confirm no regressions

```bash
.venv/bin/python -m pytest --timeout=120 -x -q
```

Expected: 52 passed (all original tests still green).

### Step 7: Commit

```bash
git add src/lessons_db/config.py src/lessons_db/db.py tests/test_db.py
git commit -m "feat: extend schema with polarity, entry_type, cluster_seed, reuse_count, surfacing_events, templates"
```

---

## Task 15: Positive Knowledge Capture

**Files:**
- Create: `src/lessons_db/capture.py`
- Create: `tests/test_capture.py`
- Modify: `src/lessons_db/cli.py` (add `capture` command group)

**What:** Two capture paths: (1) manual interactive CLI with quality scoring, (2) auto-detect from session artifacts (design docs) via Ollama. Auto-captured entries go to `capture_drafts` (quarantine) — not live until approved. Quality gate: Ollama scores one-liner specificity 1-5, reject < 3.

**Note on Ollama calls:** All Ollama-calling code uses `requests` with a timeout. Tests mock `requests.post` to avoid live network calls.

---

### Step 1: Write failing tests

Create `tests/test_capture.py`:

```python
"""Tests for positive knowledge capture."""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from lessons_db.capture import (
    capture_positive_manual,
    capture_from_design_doc,
    score_one_liner,
    promote_draft,
    list_drafts,
)
from lessons_db.db import init_db, get_lesson


class TestScoreOneLiner:
    """Ollama-based quality scoring."""

    def test_score_parses_integer_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "4"}
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            score = score_one_liner("Store subscriber refs on self for lifecycle cleanup")
        assert score == 4

    def test_score_returns_default_on_network_error(self):
        with patch("lessons_db.capture.requests.post", side_effect=Exception("timeout")):
            score = score_one_liner("anything")
        assert score == 3

    def test_score_returns_default_on_bad_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "not-a-number"}
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            score = score_one_liner("something")
        assert score == 3


class TestCaptureFromDesignDoc:
    """Auto-capture drafts from design doc content."""

    def test_creates_draft_in_db(self, db_path, tmp_path):
        doc = tmp_path / "design.md"
        doc.write_text("## Decision\nDual-axis testing outperforms single-axis in integration scenarios.")
        conn = init_db(db_path)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": json.dumps({
                "entries": [{"one_liner": "Dual-axis testing catches integration bugs", "why": "Tests both horizontal and vertical", "category": "testing-pattern"}]
            })
        }
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            drafts = capture_from_design_doc(doc, conn)

        assert len(drafts) == 1
        rows = conn.execute("SELECT * FROM capture_drafts WHERE status='pending'").fetchall()
        assert len(rows) == 1

    def test_returns_empty_on_ollama_failure(self, db_path, tmp_path):
        doc = tmp_path / "design.md"
        doc.write_text("Some content")
        conn = init_db(db_path)
        with patch("lessons_db.capture.requests.post", side_effect=Exception("timeout")):
            drafts = capture_from_design_doc(doc, conn)
        assert drafts == []

    def test_draft_has_pending_status(self, db_path, tmp_path):
        doc = tmp_path / "design.md"
        doc.write_text("Decision: use Thompson Sampling for routing")
        conn = init_db(db_path)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "response": json.dumps({"entries": [{"one_liner": "Thompson Sampling beats round-robin", "why": "Adapts to observed performance", "category": "architecture-pattern"}]})
        }
        with patch("lessons_db.capture.requests.post", return_value=mock_resp):
            capture_from_design_doc(doc, conn)
        row = conn.execute("SELECT status FROM capture_drafts LIMIT 1").fetchone()
        assert row["status"] == "pending"


class TestPromoteDraft:
    """Promoting a draft to a live lesson."""

    def test_promote_inserts_lesson(self, db_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', ?, 'auto_design_doc')",
            ["raw", json.dumps({"one_liner": "Test pattern", "why": "Because", "category": "testing-pattern"}), date.today().isoformat()]
        )
        conn.commit()
        draft_id = conn.execute("SELECT id FROM capture_drafts LIMIT 1").fetchone()["id"]

        lesson_id = promote_draft(conn, draft_id)
        assert lesson_id is not None
        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "positive"
        assert lesson["tier"] == "noticed"

    def test_promote_marks_draft_approved(self, db_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', ?, 'auto_design_doc')",
            ["raw", json.dumps({"one_liner": "X", "why": "Y", "category": "architecture-pattern"}), date.today().isoformat()]
        )
        conn.commit()
        draft_id = conn.execute("SELECT id FROM capture_drafts LIMIT 1").fetchone()["id"]
        promote_draft(conn, draft_id)
        status = conn.execute("SELECT status FROM capture_drafts WHERE id=?", [draft_id]).fetchone()["status"]
        assert status == "approved"


class TestListDrafts:
    def test_list_returns_pending_drafts(self, db_path):
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES ('raw', '{}', 'pending', '2026-02-26', 'auto_design_doc')"
        )
        conn.commit()
        drafts = list_drafts(conn)
        assert len(drafts) == 1
        assert drafts[0]["status"] == "pending"
```

### Step 2: Run tests to confirm they fail

```bash
.venv/bin/python -m pytest tests/test_capture.py -v
```

Expected: ImportError — `capture` module doesn't exist yet.

### Step 3: Implement capture.py

Create `src/lessons_db/capture.py`:

```python
"""Positive knowledge capture — manual interactive and auto-detect from artifacts."""

import json
from datetime import date
from pathlib import Path

import requests

from lessons_db.config import (
    ANALYSIS_MODEL,
    OLLAMA_QUEUE_URL,
    QUALITY_MIN_SCORE,
)
from lessons_db.db import init_db, insert_lesson


def score_one_liner(text: str) -> int:
    """Ask Ollama to score one-liner specificity 1-5. Returns 3 on any error."""
    try:
        r = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    f"Score this knowledge entry one-liner for specificity "
                    f"(1=vague, 5=precise and actionable):\n'{text}'\n\n"
                    "Respond with only a single integer 1-5."
                ),
                "stream": False,
            },
            timeout=30,
        )
        return int(r.json().get("response", "3").strip())
    except Exception:
        return 3


def capture_from_design_doc(doc_path: Path,
                             conn) -> list[dict]:
    """Extract positive patterns from a design doc. Sends to capture_drafts (quarantine).

    Returns list of extracted entry dicts. Does NOT create live lessons."""
    content = doc_path.read_text()[:4000]

    try:
        r = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    "Extract positive knowledge patterns (what worked well, effective approaches) "
                    "from this document. Return JSON with this exact structure: "
                    '{"entries": [{"one_liner": "...", "why": "...", "category": "..."}]}\n\n'
                    f"Document:\n{content}"
                ),
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )
        data = json.loads(r.json().get("response", "{}"))
        entries = data.get("entries", [])
    except Exception:
        return []

    if not entries:
        return []

    for entry in entries:
        conn.execute(
            "INSERT INTO capture_drafts "
            "(raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', ?, 'auto_design_doc')",
            [content[:500], json.dumps(entry), date.today().isoformat()],
        )
    conn.commit()
    return entries


def promote_draft(conn, draft_id: int) -> int | None:
    """Promote a pending draft to a live positive lesson. Returns lesson_id."""
    row = conn.execute(
        "SELECT extracted_data FROM capture_drafts WHERE id = ? AND status = 'pending'",
        [draft_id],
    ).fetchone()
    if not row:
        return None

    data = json.loads(row["extracted_data"])
    lesson_id = insert_lesson(conn, {
        "title": data.get("one_liner", "Untitled pattern"),
        "one_liner": data.get("one_liner", ""),
        "description": data.get("why", ""),
        "polarity": "positive",
        "entry_type": "pattern",
        "category": data.get("category", "architecture-pattern"),
        "tier": "noticed",
        "source": "auto_design_doc",
        "created_date": date.today().isoformat(),
    })
    conn.execute(
        "UPDATE capture_drafts SET status = 'approved' WHERE id = ?",
        [draft_id],
    )
    conn.commit()
    return lesson_id


def list_drafts(conn) -> list[dict]:
    """Return all pending capture drafts."""
    rows = conn.execute(
        "SELECT id, extracted_data, status, created_date, source "
        "FROM capture_drafts WHERE status = 'pending' ORDER BY created_date DESC"
    ).fetchall()
    return [dict(r) for r in rows]
```

### Step 4: Run tests to verify they pass

```bash
.venv/bin/python -m pytest tests/test_capture.py -v
```

Expected: All 9 tests PASS.

### Step 5: Add `capture` CLI command to cli.py

Add after the existing `migrate` command (before end of file):

```python
@main.group()
def capture():
    """Capture new lessons and manage draft queue."""
    pass


@capture.command("drafts")
@click.pass_context
def capture_drafts_cmd(ctx):
    """List pending auto-captured drafts awaiting approval."""
    from lessons_db.capture import list_drafts
    conn = ctx.obj["conn"]
    drafts = list_drafts(conn)
    if not drafts:
        click.echo("No pending drafts.")
        return
    for d in drafts:
        click.echo(f"[{d['id']}] {d['source']} | {d['created_date']}")
        try:
            data = json.loads(d["extracted_data"])
            click.echo(f"    {data.get('one_liner', '(no one-liner)')}")
        except Exception:
            pass


@capture.command("approve")
@click.argument("draft_id", type=int)
@click.pass_context
def capture_approve(ctx, draft_id):
    """Promote a pending draft to a live positive lesson."""
    from lessons_db.capture import promote_draft
    conn = ctx.obj["conn"]
    lesson_id = promote_draft(conn, draft_id)
    if lesson_id:
        click.echo(f"✓ Draft {draft_id} promoted → lesson #{lesson_id}")
    else:
        click.echo(f"✗ Draft {draft_id} not found or already processed.")
```

Also add `import json` at top of cli.py if not already present.

### Step 6: Run full suite

```bash
.venv/bin/python -m pytest --timeout=120 -x -q
```

Expected: 52+ tests passed (all original + new capture tests).

### Step 7: Commit

```bash
git add src/lessons_db/capture.py src/lessons_db/cli.py tests/test_capture.py
git commit -m "feat: positive knowledge capture — auto-detect from design docs, draft queue, CLI"
```

---

## Task 16: Adaptive Clustering (HDBSCAN)

**Files:**
- Create: `src/lessons_db/cluster.py`
- Create: `tests/test_cluster.py`
- Modify: `pyproject.toml` (add optional `[clustering]` dependency group)
- Modify: `src/lessons_db/cli.py` (add `cluster` command group)

**What:** HDBSCAN-based pipeline to discover natural cluster groupings from LanceDB embeddings. No predefined cluster count. Outliers stay unassigned (label -1). Results are proposals — user confirms via CLI before cluster labels are written to DB. A-F seed labels preserved as historical context.

**Test approach:** Tests mock numpy/umap/hdbscan so clustering deps aren't required to run the test suite. Only integration test requires the real packages.

---

### Step 1: Write failing tests

Create `tests/test_cluster.py`:

```python
"""Tests for adaptive HDBSCAN clustering pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest

from lessons_db.cluster import (
    extract_representative_terms,
    apply_cluster_proposals,
    get_cluster_history,
    find_seed_overlap,
)
from lessons_db.db import init_db, insert_lesson


@pytest.fixture
def conn_with_lessons(db_path):
    """DB with 6 sample lessons across two topic areas."""
    conn = init_db(db_path)
    for i in range(3):
        insert_lesson(conn, {
            "title": f"Subscriber lesson {i}",
            "one_liner": f"Store subscriber refs on self for cleanup {i}",
            "cluster_seed": "A",
            "keywords": "subscriber, lifecycle, cleanup",
            "created_date": "2026-02-26",
        })
    for i in range(3):
        insert_lesson(conn, {
            "title": f"Planning lesson {i}",
            "one_liner": f"Plan quality exceeds execution quality {i}",
            "cluster_seed": "F",
            "keywords": "planning, quality, execution",
            "created_date": "2026-02-26",
        })
    return conn


class TestExtractRepresentativeTerms:
    def test_returns_top_terms_from_one_liners(self, conn_with_lessons):
        lesson_ids = conn_with_lessons.execute(
            "SELECT id FROM lessons WHERE cluster_seed='A'"
        ).fetchall()
        ids = [r["id"] for r in lesson_ids]
        terms = extract_representative_terms(conn_with_lessons, ids)
        assert isinstance(terms, list)
        assert len(terms) > 0
        assert all(isinstance(t, str) for t in terms)

    def test_filters_stopwords(self, conn_with_lessons):
        ids = [r["id"] for r in conn_with_lessons.execute("SELECT id FROM lessons").fetchall()]
        terms = extract_representative_terms(conn_with_lessons, ids)
        stopwords = {"the", "a", "an", "and", "or", "in", "on", "of", "to", "is"}
        assert not any(t in stopwords for t in terms)


class TestFindSeedOverlap:
    def test_finds_majority_seed(self, conn_with_lessons):
        ids = [r["id"] for r in conn_with_lessons.execute(
            "SELECT id FROM lessons WHERE cluster_seed='A'"
        ).fetchall()]
        seed = find_seed_overlap(conn_with_lessons, ids)
        assert seed == "A"

    def test_returns_none_for_mixed_cluster(self, conn_with_lessons):
        # Mix of A and F — no majority
        ids = [r["id"] for r in conn_with_lessons.execute("SELECT id FROM lessons").fetchall()]
        seed = find_seed_overlap(conn_with_lessons, ids)
        assert seed is None  # 50/50 split — below 60% threshold


class TestApplyClusterProposals:
    def test_writes_cluster_label_to_db(self, conn_with_lessons):
        ids = [r["id"] for r in conn_with_lessons.execute(
            "SELECT id FROM lessons WHERE cluster_seed='A'"
        ).fetchall()]
        proposals = [{"cluster_id": 0, "lesson_ids": ids, "suggested_name": "Subscriber Lifecycle"}]
        confirmed = {0: "Subscriber Lifecycle"}
        count = apply_cluster_proposals(conn_with_lessons, proposals, confirmed)
        assert count == len(ids)
        for lid in ids:
            row = conn_with_lessons.execute(
                "SELECT cluster FROM lessons WHERE id=?", [lid]
            ).fetchone()
            assert row["cluster"] == "Subscriber Lifecycle"

    def test_skips_unconfirmed_proposals(self, conn_with_lessons):
        ids = [r["id"] for r in conn_with_lessons.execute(
            "SELECT id FROM lessons WHERE cluster_seed='F'"
        ).fetchall()]
        proposals = [{"cluster_id": 1, "lesson_ids": ids, "suggested_name": "Planning Quality"}]
        confirmed = {}  # Nothing confirmed
        count = apply_cluster_proposals(conn_with_lessons, proposals, confirmed)
        assert count == 0

    def test_records_cluster_run(self, conn_with_lessons):
        proposals = [{"cluster_id": 0, "lesson_ids": [1], "suggested_name": "Test Cluster"}]
        apply_cluster_proposals(conn_with_lessons, proposals, {0: "Test Cluster"})
        runs = get_cluster_history(conn_with_lessons)
        assert len(runs) == 1
        assert runs[0]["proposal_count"] == 1


class TestGetClusterHistory:
    def test_returns_empty_initially(self, db_path):
        conn = init_db(db_path)
        assert get_cluster_history(conn) == []
```

### Step 2: Run tests to confirm they fail

```bash
.venv/bin/python -m pytest tests/test_cluster.py -v
```

Expected: ImportError — `cluster` module doesn't exist.

### Step 3: Implement cluster.py

Create `src/lessons_db/cluster.py`:

```python
"""Adaptive clustering pipeline using HDBSCAN on LanceDB embeddings.

The discover_clusters() function requires optional dependencies:
    pip install 'lessons-db[clustering]'

All other functions (extract_representative_terms, apply_cluster_proposals,
find_seed_overlap, get_cluster_history) have no extra dependencies.
"""

import json
from collections import Counter
from datetime import date
from typing import Optional

from lessons_db.config import LANCE_DIR, OLLAMA_QUEUE_URL, ANALYSIS_MODEL

# Words to ignore when extracting representative terms
_STOPWORDS = {
    "the", "a", "an", "and", "or", "in", "on", "of", "to", "is", "are",
    "was", "be", "with", "for", "it", "this", "that", "not", "no",
    "from", "by", "at", "as", "but", "if", "so", "do", "use",
}


def extract_representative_terms(conn, lesson_ids: list[int],
                                  top_n: int = 5) -> list[str]:
    """Extract the most frequent non-stopword terms from one-liners + keywords."""
    if not lesson_ids:
        return []
    placeholders = ",".join("?" * len(lesson_ids))
    rows = conn.execute(
        f"SELECT one_liner, keywords FROM lessons WHERE id IN ({placeholders})",
        lesson_ids,
    ).fetchall()
    words = []
    for row in rows:
        text = (row["one_liner"] or "") + " " + (row["keywords"] or "")
        words.extend(text.lower().split())
    counter = Counter(w.strip(".,;:()[]") for w in words if w not in _STOPWORDS and len(w) > 2)
    return [w for w, _ in counter.most_common(top_n * 2) if w not in _STOPWORDS][:top_n]


def find_seed_overlap(conn, lesson_ids: list[int],
                      threshold: float = 0.6) -> Optional[str]:
    """Return the dominant cluster_seed if >= threshold fraction of lessons share it."""
    if not lesson_ids:
        return None
    placeholders = ",".join("?" * len(lesson_ids))
    rows = conn.execute(
        f"SELECT cluster_seed FROM lessons WHERE id IN ({placeholders}) AND cluster_seed IS NOT NULL",
        lesson_ids,
    ).fetchall()
    if not rows:
        return None
    counter = Counter(r["cluster_seed"] for r in rows)
    top_seed, top_count = counter.most_common(1)[0]
    if top_count / len(lesson_ids) >= threshold:
        return top_seed
    return None


def apply_cluster_proposals(conn, proposals: list[dict],
                              confirmed: dict[int, str]) -> int:
    """Write confirmed cluster names to lessons.cluster. Records the run.

    proposals: list of {"cluster_id": int, "lesson_ids": [...], "suggested_name": str}
    confirmed: {cluster_id: final_name} — only these get written
    Returns count of updated lesson rows."""
    updated = 0
    for proposal in proposals:
        cid = proposal["cluster_id"]
        if cid not in confirmed:
            continue
        name = confirmed[cid]
        for lid in proposal["lesson_ids"]:
            conn.execute(
                "UPDATE lessons SET cluster = ? WHERE id = ?",
                [name, lid],
            )
            updated += 1

    conn.execute(
        "INSERT INTO cluster_runs (run_date, proposal_count, confirmed_count, result_json) "
        "VALUES (?, ?, ?, ?)",
        [
            date.today().isoformat(),
            len(proposals),
            len(confirmed),
            json.dumps([
                {"id": p["cluster_id"],
                 "name": confirmed.get(p["cluster_id"]),
                 "size": len(p["lesson_ids"])}
                for p in proposals
            ]),
        ],
    )
    conn.commit()
    return updated


def get_cluster_history(conn) -> list[dict]:
    """Return all past clustering runs in descending date order."""
    rows = conn.execute(
        "SELECT id, run_date, proposal_count, confirmed_count, result_json "
        "FROM cluster_runs ORDER BY run_date DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def generate_cluster_name(terms: list[str]) -> str:
    """Ask Ollama to generate a human-readable cluster name from terms.
    Falls back to joining the top 2 terms if Ollama unavailable."""
    import requests
    try:
        r = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    f"Generate a 3-5 word cluster name for a group of software engineering "
                    f"lessons with these key terms: {', '.join(terms)}. "
                    "Respond with only the cluster name, no explanation."
                ),
                "stream": False,
            },
            timeout=30,
        )
        return r.json().get("response", "").strip() or f"{terms[0].title()} Patterns"
    except Exception:
        return " ".join(t.title() for t in terms[:2]) + " Patterns"


def discover_clusters(conn, min_cluster_size: int = 5) -> list[dict]:
    """Run HDBSCAN on LanceDB embeddings and return cluster proposals.

    Requires: pip install 'lessons-db[clustering]'

    Returns list of proposal dicts:
      {"cluster_id": int, "lesson_ids": [...], "suggested_name": str,
       "representative_terms": [...], "overlaps_seed": str | None}
    """
    try:
        import umap
        import hdbscan
        import numpy as np
        import lancedb
    except ImportError as e:
        raise RuntimeError(
            f"Clustering dependencies not installed ({e}). Run:\n"
            "pip install 'lessons-db[clustering]'"
        ) from e

    db = lancedb.connect(str(LANCE_DIR))
    try:
        table = db.open_table("lessons")
    except Exception:
        return []

    rows = table.to_pandas()
    if len(rows) < min_cluster_size * 2:
        return []

    vectors = np.vstack(rows["vector"].values)
    lesson_ids = rows["lesson_id"].tolist()

    reducer = umap.UMAP(n_components=min(5, len(rows) - 1), random_state=42)
    reduced = reducer.fit_transform(vectors)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(reduced)

    clusters: dict[int, list[int]] = {}
    for lid, label in zip(lesson_ids, labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(lid)

    proposals = []
    for label, ids in clusters.items():
        terms = extract_representative_terms(conn, ids)
        name = generate_cluster_name(terms)
        seed = find_seed_overlap(conn, ids)
        proposals.append({
            "cluster_id": label,
            "lesson_ids": ids,
            "suggested_name": name,
            "representative_terms": terms,
            "overlaps_seed": seed,
        })
    return proposals
```

### Step 4: Add optional clustering deps to pyproject.toml

Find:
```toml
[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-xdist>=3.5.0", "pytest-timeout>=2.4.0"]
```

Replace with:
```toml
[project.optional-dependencies]
clustering = ["umap-learn>=0.5.0", "hdbscan>=0.8.0", "scikit-learn>=1.0.0"]
dev = ["pytest>=8.0.0", "pytest-xdist>=3.5.0", "pytest-timeout>=2.4.0"]
```

### Step 5: Add `cluster` CLI commands to cli.py

```python
@main.group()
def cluster():
    """Adaptive cluster discovery and management."""
    pass


@cluster.command("show")
@click.pass_context
def cluster_show(ctx):
    """Show current cluster assignments for all lessons."""
    conn = ctx.obj["conn"]
    rows = conn.execute(
        "SELECT cluster, COUNT(*) as n FROM lessons GROUP BY cluster ORDER BY n DESC"
    ).fetchall()
    for r in rows:
        label = r["cluster"] or "(unassigned)"
        click.echo(f"  {label}: {r['n']} lessons")


@cluster.command("history")
@click.pass_context
def cluster_history(ctx):
    """Show history of past clustering runs."""
    from lessons_db.cluster import get_cluster_history
    runs = get_cluster_history(ctx.obj["conn"])
    if not runs:
        click.echo("No clustering runs yet. Run: lessons-db cluster discover")
        return
    for run in runs:
        click.echo(f"[{run['run_date']}] {run['proposal_count']} proposals, "
                   f"{run['confirmed_count']} confirmed")


@cluster.command("discover")
@click.option("--min-size", default=5, type=int, help="Minimum cluster size for HDBSCAN.")
@click.pass_context
def cluster_discover(ctx, min_size):
    """Run HDBSCAN on embeddings and propose new cluster assignments."""
    from lessons_db.cluster import discover_clusters, apply_cluster_proposals
    conn = ctx.obj["conn"]
    click.echo("Running HDBSCAN clustering...")
    try:
        proposals = discover_clusters(conn, min_cluster_size=min_size)
    except RuntimeError as e:
        click.echo(str(e))
        return
    if not proposals:
        click.echo("Not enough data to cluster (need at least min_size * 2 entries with embeddings).")
        return
    confirmed = {}
    for p in proposals:
        seed_info = f" (overlaps seed {p['overlaps_seed']})" if p["overlaps_seed"] else ""
        click.echo(f"\nCluster {p['cluster_id']}: {len(p['lesson_ids'])} lessons{seed_info}")
        click.echo(f"  Suggested name: {p['suggested_name']}")
        click.echo(f"  Key terms: {', '.join(p['representative_terms'])}")
        name = click.prompt("  Accept name? (Enter to accept, or type a new name, or 's' to skip)",
                            default=p["suggested_name"])
        if name.lower() != "s":
            confirmed[p["cluster_id"]] = name
    if confirmed:
        count = apply_cluster_proposals(conn, proposals, confirmed)
        click.echo(f"\n✓ Updated {count} lesson cluster assignments.")
    else:
        click.echo("No clusters confirmed.")
```

### Step 6: Run tests

```bash
.venv/bin/python -m pytest tests/test_cluster.py -v
```

Expected: All 9 tests PASS.

### Step 7: Run full suite

```bash
.venv/bin/python -m pytest --timeout=120 -x -q
```

Expected: All tests PASS.

### Step 8: Commit

```bash
git add src/lessons_db/cluster.py src/lessons_db/cli.py tests/test_cluster.py pyproject.toml
git commit -m "feat: adaptive HDBSCAN clustering — discover clusters from embeddings, CLI propose/confirm flow"
```

---

## Task 17: Learning Pipeline (Outcome Tracking + Relevance Scoring)

**Files:**
- Create: `src/lessons_db/learn.py`
- Create: `tests/test_learn.py`
- Modify: `src/lessons_db/cli.py` (add `stats` command group)

**What:** Record every time a lesson surfaces (hook point + context). Later update the outcome (heeded/dismissed) based on inferred user behavior. Composite relevance score = 50% semantic similarity + 30% outcome rate + 20% recurrence score. CLI shows surfacing stats and efficiency trend.

---

### Step 1: Write failing tests

Create `tests/test_learn.py`:

```python
"""Tests for learning pipeline — outcome tracking and relevance scoring."""

import pytest
from datetime import datetime

from lessons_db.learn import (
    record_surfacing,
    record_outcome,
    relevance_score,
    surfacing_stats,
)
from lessons_db.db import init_db, insert_lesson


@pytest.fixture
def conn_with_lesson(db_path):
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Test lesson",
        "one_liner": "Always log before swallowing exceptions",
        "created_date": "2026-02-26",
    })
    return conn, lid


class TestRecordSurfacing:
    def test_creates_surfacing_event(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="read", context="src/hub.py")
        assert event_id is not None
        row = conn.execute(
            "SELECT * FROM surfacing_events WHERE id=?", [event_id]
        ).fetchone()
        assert row["lesson_id"] == lid
        assert row["hook_point"] == "read"
        assert row["outcome"] == "unknown"

    def test_stores_context(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="plan", context="authentication refactor")
        row = conn.execute("SELECT context FROM surfacing_events WHERE id=?", [event_id]).fetchone()
        assert "authentication" in row["context"]


class TestRecordOutcome:
    def test_updates_outcome_to_heeded(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="read", context="hub.py")
        record_outcome(conn, event_id, "heeded")
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id=?", [event_id]).fetchone()
        assert row["outcome"] == "heeded"

    def test_updates_outcome_to_dismissed(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="read", context="hub.py")
        record_outcome(conn, event_id, "dismissed")
        row = conn.execute("SELECT outcome FROM surfacing_events WHERE id=?", [event_id]).fetchone()
        assert row["outcome"] == "dismissed"

    def test_rejects_invalid_outcome(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        event_id = record_surfacing(conn, lid, hook_point="read", context="hub.py")
        with pytest.raises(ValueError):
            record_outcome(conn, event_id, "ignored")


class TestRelevanceScore:
    def test_cold_start_returns_half_semantic(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        # No history → outcome_rate=0.5, recurrence=0
        score = relevance_score(conn, lid, context="hub.py", semantic_sim=0.8)
        # 0.5*0.8 + 0.3*0.5 + 0.2*0.0 = 0.4 + 0.15 = 0.55
        assert abs(score - 0.55) < 0.01

    def test_heeded_history_boosts_score(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        for _ in range(3):
            eid = record_surfacing(conn, lid, "read", "hub.py")
            record_outcome(conn, eid, "heeded")
        score_with_history = relevance_score(conn, lid, context="hub.py", semantic_sim=0.8)
        score_cold = relevance_score(conn, lid, context="other.py", semantic_sim=0.8)
        assert score_with_history > score_cold

    def test_dismissed_history_lowers_score(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        for _ in range(3):
            eid = record_surfacing(conn, lid, "read", "hub.py")
            record_outcome(conn, eid, "dismissed")
        score = relevance_score(conn, lid, context="hub.py", semantic_sim=0.8)
        # outcome_rate=0.0 → 0.5*0.8 + 0.3*0.0 + 0.2*0 = 0.4
        assert score < 0.55

    def test_recurrence_boosts_score(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        # Add 10 near-misses to push recurrence to max
        for _ in range(10):
            conn.execute(
                "INSERT INTO near_misses (lesson_id, file_path, event_type, timestamp) "
                "VALUES (?, 'hub.py', 'hookify_warn', '2026-02-26T10:00:00')",
                [lid]
            )
        conn.commit()
        score = relevance_score(conn, lid, context="other.py", semantic_sim=0.5)
        # 0.5*0.5 + 0.3*0.5 + 0.2*1.0 = 0.25 + 0.15 + 0.20 = 0.60
        assert score > 0.55


class TestSurfacingStats:
    def test_returns_zero_counts_when_empty(self, db_path):
        conn = init_db(db_path)
        stats = surfacing_stats(conn)
        assert stats["total_surfacing_events"] == 0
        assert stats["heed_rate"] is None

    def test_counts_heeded_and_dismissed(self, conn_with_lesson):
        conn, lid = conn_with_lesson
        e1 = record_surfacing(conn, lid, "read", "a.py")
        record_outcome(conn, e1, "heeded")
        e2 = record_surfacing(conn, lid, "read", "b.py")
        record_outcome(conn, e2, "dismissed")
        record_surfacing(conn, lid, "plan", "c")  # unknown
        stats = surfacing_stats(conn)
        assert stats["total_surfacing_events"] == 3
        assert stats["heeded"] == 1
        assert stats["dismissed"] == 1
        assert stats["unknown"] == 1
        assert stats["heed_rate"] == 0.33
```

### Step 2: Run tests to confirm they fail

```bash
.venv/bin/python -m pytest tests/test_learn.py -v
```

Expected: ImportError — `learn` module doesn't exist.

### Step 3: Implement learn.py

Create `src/lessons_db/learn.py`:

```python
"""Learning pipeline: surfacing event recording and composite relevance scoring."""

from datetime import datetime


def record_surfacing(conn, lesson_id: int, hook_point: str,
                     context: str = "", session_id: str | None = None) -> int:
    """Record a surfacing event. Returns event ID for later outcome update."""
    cursor = conn.execute(
        "INSERT INTO surfacing_events "
        "(lesson_id, hook_point, context, outcome, timestamp, session_id) "
        "VALUES (?, ?, ?, 'unknown', ?, ?)",
        [lesson_id, hook_point, context, datetime.now().isoformat(), session_id],
    )
    conn.commit()
    return cursor.lastrowid


def record_outcome(conn, event_id: int, outcome: str) -> None:
    """Update outcome for a surfacing event. outcome must be 'heeded' or 'dismissed'."""
    if outcome not in ("heeded", "dismissed"):
        raise ValueError(f"Invalid outcome '{outcome}'. Must be 'heeded' or 'dismissed'.")
    conn.execute(
        "UPDATE surfacing_events SET outcome = ? WHERE id = ?",
        [outcome, event_id],
    )
    conn.commit()


def relevance_score(conn, lesson_id: int, context: str,
                    semantic_sim: float) -> float:
    """Composite relevance score.

    score = 0.5 * semantic_sim
           + 0.3 * outcome_rate (heeded ratio in similar contexts)
           + 0.2 * recurrence_score (normalized near-miss + recurrence count)
    """
    outcome = _outcome_rate(conn, lesson_id, context)
    recurrence = _recurrence_score(conn, lesson_id)
    return round(0.5 * semantic_sim + 0.3 * outcome + 0.2 * recurrence, 4)


def surfacing_stats(conn) -> dict:
    """Summary stats for the status command and efficiency tracking."""
    total = conn.execute("SELECT COUNT(*) FROM surfacing_events").fetchone()[0]
    heeded = conn.execute(
        "SELECT COUNT(*) FROM surfacing_events WHERE outcome='heeded'"
    ).fetchone()[0]
    dismissed = conn.execute(
        "SELECT COUNT(*) FROM surfacing_events WHERE outcome='dismissed'"
    ).fetchone()[0]
    avg_row = conn.execute(
        "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM surfacing_events GROUP BY session_id)"
    ).fetchone()[0]

    return {
        "total_surfacing_events": total,
        "heeded": heeded,
        "dismissed": dismissed,
        "unknown": total - heeded - dismissed,
        "heed_rate": round(heeded / total, 2) if total > 0 else None,
        "avg_per_session": round(avg_row or 0.0, 1),
    }


def _outcome_rate(conn, lesson_id: int, context: str) -> float:
    """Ratio of heeded outcomes for this lesson in similar contexts.
    Returns 0.5 (neutral) if no outcome data exists."""
    ctx_prefix = context[:50] if context else ""
    rows = conn.execute(
        "SELECT outcome FROM surfacing_events "
        "WHERE lesson_id = ? AND context LIKE ? AND outcome != 'unknown'",
        [lesson_id, f"%{ctx_prefix}%"],
    ).fetchall()
    if not rows:
        return 0.5
    heeded = sum(1 for r in rows if r["outcome"] == "heeded")
    return heeded / len(rows)


def _recurrence_score(conn, lesson_id: int) -> float:
    """Normalized recurrence + near-miss count. Caps at 1.0 (10+ events = max)."""
    row = conn.execute(
        "SELECT recurrence_count, "
        "(SELECT COUNT(*) FROM near_misses WHERE lesson_id = l.id) AS nm "
        "FROM lessons l WHERE l.id = ?",
        [lesson_id],
    ).fetchone()
    if not row:
        return 0.0
    raw = (row["recurrence_count"] or 0) + (row["nm"] or 0)
    return min(raw / 10.0, 1.0)
```

### Step 4: Add `stats` CLI commands

```python
@main.group()
def stats():
    """Surfacing and efficiency statistics."""
    pass


@stats.command("surfacing")
@click.pass_context
def stats_surfacing(ctx):
    """Show outcome rates and surfacing event counts."""
    from lessons_db.learn import surfacing_stats
    s = surfacing_stats(ctx.obj["conn"])
    click.echo(f"Total surfacing events : {s['total_surfacing_events']}")
    click.echo(f"  Heeded               : {s['heeded']}")
    click.echo(f"  Dismissed            : {s['dismissed']}")
    click.echo(f"  Unknown              : {s['unknown']}")
    if s["heed_rate"] is not None:
        click.echo(f"  Heed rate            : {s['heed_rate']:.0%}")
    click.echo(f"Avg per session        : {s['avg_per_session']}")
```

### Step 5: Run tests

```bash
.venv/bin/python -m pytest tests/test_learn.py -v
```

Expected: All 11 tests PASS.

### Step 6: Run full suite

```bash
.venv/bin/python -m pytest --timeout=120 -x -q
```

Expected: All tests PASS.

### Step 7: Commit

```bash
git add src/lessons_db/learn.py src/lessons_db/cli.py tests/test_learn.py
git commit -m "feat: learning pipeline — surfacing event tracking, outcome recording, composite relevance scoring"
```

---

## Task 18: Promotion Ladder (Positive Entries)

**Files:**
- Create: `src/lessons_db/promote.py`
- Create: `tests/test_promote.py`
- Modify: `src/lessons_db/cli.py` (add `template` command group)

**What:** When a positive entry's `reuse_count` reaches thresholds, it graduates through noticed→tested→proven→standard. At `proven` (reuse_count >= 2), a template is auto-generated in the `templates` table. At `standard` (reuse_count >= 3), a notification is emitted. CLI lists and shows templates.

---

### Step 1: Write failing tests

Create `tests/test_promote.py`:

```python
"""Tests for positive entry promotion ladder."""

import pytest

from lessons_db.promote import record_reuse, list_templates, apply_template
from lessons_db.db import init_db, insert_lesson, get_lesson


@pytest.fixture
def conn_with_positive(db_path):
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Dual-axis pipeline testing",
        "one_liner": "Dual-axis testing catches integration bugs missed by unit tests",
        "polarity": "positive",
        "entry_type": "pattern",
        "tier": "noticed",
        "category": "testing-pattern",
        "created_date": "2026-02-26",
    })
    return conn, lid


class TestRecordReuse:
    def test_first_reuse_promotes_to_tested(self, conn_with_positive):
        conn, lid = conn_with_positive
        new_tier = record_reuse(conn, lid)
        assert new_tier == "tested"
        lesson = get_lesson(conn, lid)
        assert lesson["reuse_count"] == 1
        assert lesson["tier"] == "tested"

    def test_second_reuse_promotes_to_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)  # → tested
        new_tier = record_reuse(conn, lid)  # → proven
        assert new_tier == "proven"
        lesson = get_lesson(conn, lid)
        assert lesson["tier"] == "proven"

    def test_second_reuse_generates_template(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        templates = list_templates(conn)
        assert len(templates) == 1
        assert templates[0]["lesson_id"] == lid

    def test_third_reuse_promotes_to_standard(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        new_tier = record_reuse(conn, lid)
        assert new_tier == "standard"

    def test_increments_reuse_count(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        lesson = get_lesson(conn, lid)
        assert lesson["reuse_count"] == 3


class TestListTemplates:
    def test_empty_initially(self, db_path):
        conn = init_db(db_path)
        assert list_templates(conn) == []

    def test_returns_template_after_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        templates = list_templates(conn)
        assert len(templates) == 1
        assert "one_liner" in templates[0]
        assert templates[0]["tier"] == "proven"


class TestApplyTemplate:
    def test_returns_none_before_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        assert apply_template(conn, lid) is None

    def test_returns_content_after_proven(self, conn_with_positive):
        conn, lid = conn_with_positive
        record_reuse(conn, lid)
        record_reuse(conn, lid)
        content = apply_template(conn, lid)
        assert content is not None
        assert "Dual-axis" in content
```

### Step 2: Run tests to confirm they fail

```bash
.venv/bin/python -m pytest tests/test_promote.py -v
```

Expected: ImportError — `promote` module doesn't exist.

### Step 3: Implement promote.py

Create `src/lessons_db/promote.py`:

```python
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

    reuse_count = row["reuse_count"] + 1
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
```

### Step 4: Add `template` CLI commands

```python
@main.group()
def template():
    """View and apply templates from proven positive patterns."""
    pass


@template.command("list")
@click.pass_context
def template_list(ctx):
    """List all generated templates."""
    from lessons_db.promote import list_templates
    templates = list_templates(ctx.obj["conn"])
    if not templates:
        click.echo("No templates yet. Positive entries reach 'proven' tier after 2 reuses.")
        return
    for t in templates:
        click.echo(f"[#{t['lesson_id']}] ({t['tier']}) {t['one_liner']}")
        click.echo(f"      type={t['template_type']} | created={t['created_date']}")


@template.command("show")
@click.argument("lesson_id", type=int)
@click.pass_context
def template_show(ctx, lesson_id):
    """Show template content for a lesson."""
    from lessons_db.promote import apply_template
    content = apply_template(ctx.obj["conn"], lesson_id)
    if content:
        click.echo(content)
    else:
        click.echo(f"No template for lesson #{lesson_id}. Entry must reach 'proven' tier first.")
```

### Step 5: Run tests

```bash
.venv/bin/python -m pytest tests/test_promote.py -v
```

Expected: All 10 tests PASS.

### Step 6: Run full suite

```bash
.venv/bin/python -m pytest --timeout=120 -x -q
```

Expected: 85+ tests PASS (all 52 original + ~33 new).

### Step 7: Commit

```bash
git add src/lessons_db/promote.py src/lessons_db/cli.py tests/test_promote.py
git commit -m "feat: promotion ladder — noticed→tested→proven→standard, template generation at proven tier"
```

---

## Final Verification

After all 4 tasks are complete:

### Run full suite one more time

```bash
.venv/bin/python -m pytest --timeout=120 -q
```

Expected: ~85 tests PASS, 0 failures.

### Smoke-test the CLI

```bash
# Use the installed venv CLI
source .venv/bin/activate

lessons-db status
lessons-db capture drafts
lessons-db cluster show
lessons-db cluster history
lessons-db stats surfacing
lessons-db template list
```

Expected: Each command runs without error. "No data yet" messages are fine.

### Final commit if anything needed tidying

```bash
git add -u
git commit -m "chore: final cleanup after extension implementation"
```

---

## Task Summary

| Task | Module | Tests | Key Command |
|------|--------|-------|-------------|
| 14 | Schema extension | 8 | `init_db()` adds columns + tables |
| 15 | capture.py | 9 | `lessons-db capture drafts/approve` |
| 16 | cluster.py | 9 | `lessons-db cluster discover/show/history` |
| 17 | learn.py | 11 | `lessons-db stats surfacing` |
| 18 | promote.py | 10 | `lessons-db template list/show` |

**Total extension tests: ~47**
**Grand total with v1: ~99 tests**

**Critical path:** 14 → 15 → 18 (schema before capture; capture before promotion)

**Parallel-safe after Task 14:** Tasks 16 and 17 have no dependency on each other.
