# Draft Triage Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automated nightly pipeline that triages the 1,906-draft backlog and all future drafts using Claude haiku as reviewer — producing promoted lessons with detection patterns and Semgrep rules, zero human effort required.

**Architecture:** New `review.py` module handles dedup + noise filter + Claude batch review. `promote_draft` bug fixed for negative polarity. Nightly service extended to call `lessons-db capture review`. SessionStart hook surfaces overnight promotions.

**Tech Stack:** Python 3.12, anthropic SDK (claude-haiku-4-5-20251001), Click CLI, SQLite, existing db.py/capture.py/rulegen.py patterns.

**Design doc:** `docs/plans/2026-02-27-draft-triage-design.md`

---

### Task 1: Add anthropic dependency + config

**Files:**
- Modify: `pyproject.toml` (line 15–21)
- Modify: `src/lessons_db/config.py`

**Step 1: Add anthropic to pyproject.toml**

In `pyproject.toml`, add `"anthropic>=0.40.0",` to the `dependencies` list after `requests`:

```toml
dependencies = [
    "click>=8.1.0",
    "lancedb>=0.20.0",
    "pyarrow>=15.0.0",
    "requests>=2.31.0",
    "anthropic>=0.40.0",
    "pyyaml>=6.0",
]
```

**Step 2: Install it**

```bash
cd ~/Documents/projects/lessons-db
.venv/bin/python -m pip install anthropic>=0.40.0
```

Expected: `Successfully installed anthropic-...`

**Step 3: Add config constant to config.py**

Read `src/lessons_db/config.py` first, then add after the existing constants:

```python
import os

# Claude API config (for draft triage reviewer)
CLAUDE_REVIEW_MODEL = os.environ.get("LESSONS_DB_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Triage log directory
TRIAGE_LOG_DIR = DATA_DIR  # reuse existing ~/.local/share/lessons-db/
```

**Step 4: Verify import works**

```bash
cd ~/Documents/projects/lessons-db
.venv/bin/python -c "import anthropic; from lessons_db.config import CLAUDE_REVIEW_MODEL; print(CLAUDE_REVIEW_MODEL)"
```

Expected: `claude-haiku-4-5-20251001`

**Step 5: Commit**

```bash
git add pyproject.toml src/lessons_db/config.py
git commit -m "feat: add anthropic SDK dependency and review model config"
```

---

### Task 2: Fix promote_draft polarity bug

**Files:**
- Modify: `src/lessons_db/capture.py` (lines 101–130)
- Modify: `tests/test_capture.py`

**Step 1: Write failing test**

Add to `tests/test_capture.py` in the existing `TestPromoteDraft` class (or create it):

```python
class TestPromoteDraftPolarity:
    def test_auto_transcript_promotes_as_negative(self, db_path):
        conn = init_db(db_path)
        entry = {"one_liner": "Never swallow exceptions silently", "cluster": "A", "tier": "lesson"}
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', '2026-02-27', 'auto_transcript')",
            ["raw", json.dumps(entry)],
        )
        conn.commit()
        draft_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        lesson_id = promote_draft(conn, draft_id)

        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "negative"
        assert lesson["source"] == "auto_transcript"

    def test_auto_transcript_positive_promotes_as_positive(self, db_path):
        conn = init_db(db_path)
        entry = {"one_liner": "Dual-axis testing catches integration bugs", "cluster": "", "tier": "lesson"}
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES (?, ?, 'pending', '2026-02-27', 'auto_transcript_positive')",
            ["raw", json.dumps(entry)],
        )
        conn.commit()
        draft_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        lesson_id = promote_draft(conn, draft_id)

        lesson = get_lesson(conn, lesson_id)
        assert lesson["polarity"] == "positive"
```

**Step 2: Run to verify failure**

```bash
cd ~/Documents/projects/lessons-db
pytest tests/test_capture.py::TestPromoteDraftPolarity -v
```

Expected: FAIL — `assert 'positive' == 'negative'`

**Step 3: Fix promote_draft in capture.py**

Replace the `promote_draft` function body (lines ~101–130). The fix: read source from draft row, infer polarity and entry_type:

```python
_POLARITY_BY_SOURCE = {
    "auto_transcript": ("negative", "lesson"),
    "auto_transcript_positive": ("positive", "pattern"),
    "auto_diff": ("negative", "lesson"),
    "auto_design_doc": ("positive", "pattern"),
}

def promote_draft(conn: sqlite3.Connection, draft_id: int) -> int | None:
    """Promote a pending draft to a live lesson. Returns lesson_id."""
    row = conn.execute(
        "SELECT extracted_data, source FROM capture_drafts WHERE id = ? AND status = 'pending'",
        [draft_id],
    ).fetchone()
    if not row:
        return None

    data = json.loads(row["extracted_data"])
    source = row["source"] or "auto_transcript"
    polarity, entry_type = _POLARITY_BY_SOURCE.get(source, ("negative", "lesson"))

    lesson_id = insert_lesson(
        conn,
        {
            "title": data.get("improved_one_liner") or data.get("one_liner", "Untitled"),
            "one_liner": data.get("improved_one_liner") or data.get("one_liner", ""),
            "description": data.get("why", ""),
            "polarity": polarity,
            "entry_type": entry_type,
            "category": data.get("category", "architecture-pattern"),
            "tier": data.get("tier", "noticed"),
            "source": source,
            "created_date": date.today().isoformat(),
        },
    )
    conn.execute(
        "UPDATE capture_drafts SET status = 'approved' WHERE id = ?",
        [draft_id],
    )
    conn.commit()
    return lesson_id
```

**Step 4: Run tests**

```bash
pytest tests/test_capture.py -v
```

Expected: all pass including the two new polarity tests.

**Step 5: Commit**

```bash
git add src/lessons_db/capture.py tests/test_capture.py
git commit -m "fix: promote_draft infers polarity from draft source (was hardcoded positive)"
```

---

### Task 3: Add insert_detection_pattern to db.py

**Files:**
- Modify: `src/lessons_db/db.py`
- Modify: `tests/test_db.py`

**Step 1: Write failing test**

Add to `tests/test_db.py`:

```python
from lessons_db.db import insert_detection_pattern

class TestInsertDetectionPattern:
    def test_inserts_pattern_for_lesson(self, db_path):
        conn = init_db(db_path)
        lesson_id = insert_lesson(conn, {"one_liner": "Never swallow exceptions", "tier": "lesson"})

        pattern_id = insert_detection_pattern(conn, {
            "lesson_id": lesson_id,
            "pattern_type": "regex",
            "regex": r"except\s*:",
            "description": "Bare except clause — always log before swallowing",
            "language": "python",
        })

        row = conn.execute(
            "SELECT * FROM detection_patterns WHERE id = ?", [pattern_id]
        ).fetchone()
        assert row["lesson_id"] == lesson_id
        assert row["regex"] == r"except\s*:"
        assert row["language"] == "python"
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_db.py::TestInsertDetectionPattern -v
```

Expected: FAIL — `ImportError: cannot import name 'insert_detection_pattern'`

**Step 3: Add insert_detection_pattern to db.py**

After the `insert_lesson` function, add:

```python
def insert_detection_pattern(conn: sqlite3.Connection, data: dict) -> int:
    """Insert a detection pattern for a lesson. Returns new row id."""
    required = {"lesson_id", "pattern_type", "regex"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    defaults = {"description": None, "language": "any"}
    row = {**defaults, **data}
    cols = list(row.keys())
    cursor = conn.execute(
        f"INSERT INTO detection_patterns ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [row[c] for c in cols],
    )
    conn.commit()
    return cursor.lastrowid
```

**Step 4: Run tests**

```bash
pytest tests/test_db.py -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/lessons_db/db.py tests/test_db.py
git commit -m "feat: add insert_detection_pattern to db.py"
```

---

### Task 4: Create review.py — noise filter

**Files:**
- Create: `src/lessons_db/review.py`
- Create: `tests/test_review.py`

**Step 1: Write failing tests**

Create `tests/test_review.py`:

```python
"""Tests for draft triage review pipeline."""
import json
import pytest
from lessons_db.db import init_db, insert_lesson
from lessons_db.review import filter_noise, jaccard_similarity


class TestJaccardSimilarity:
    def test_identical_strings_score_one(self):
        assert jaccard_similarity("never swallow exceptions silently", "never swallow exceptions silently") == 1.0

    def test_completely_different_strings_score_zero(self):
        assert jaccard_similarity("apple orange", "banana grape") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        score = jaccard_similarity("never swallow exceptions", "always log exceptions first")
        assert 0.0 < score < 1.0


class TestFilterNoise:
    def _draft(self, one_liner, source="auto_transcript"):
        return {"id": 1, "extracted_data": json.dumps({"one_liner": one_liner}), "source": source}

    def test_dismisses_no_mistakes_pattern(self):
        drafts = [self._draft("No coding mistakes were discovered in this session.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(kept) == 0
        assert len(dismissed) == 1

    def test_dismisses_repeated_content_pattern(self):
        drafts = [self._draft("Repeated content in the transcript was found.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_dismisses_too_short_one_liner(self):
        drafts = [self._draft("Write tests.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_keeps_good_one_liner(self):
        drafts = [self._draft("Never call close() on sqlite3 connections inside a context manager — use closing().")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(kept) == 1

    def test_dismisses_near_duplicate_of_existing_lesson(self):
        existing = ["Never call close() on sqlite3 connections inside a context manager"]
        drafts = [self._draft("Never call close on sqlite3 connections inside context manager")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=existing)
        assert len(dismissed) == 1

    def test_dismisses_duplicate_within_batch(self):
        drafts = [
            {"id": 1, "extracted_data": json.dumps({"one_liner": "Always log before swallowing exceptions"}), "source": "auto_transcript"},
            {"id": 2, "extracted_data": json.dumps({"one_liner": "Always log before swallowing exceptions silently"}), "source": "auto_transcript"},
        ]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(kept) == 1
        assert len(dismissed) == 1
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_review.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'lessons_db.review'`

**Step 3: Create src/lessons_db/review.py with noise filter**

```python
"""Draft triage pipeline: noise filter + Claude batch review + verdict execution."""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

_log = logging.getLogger(__name__)

# Noise patterns — one_liners matching these are auto-dismissed
_NOISE_PATTERNS = [
    re.compile(r"no\s+(coding\s+)?mistakes?\s+were\s+(found|discovered)", re.IGNORECASE),
    re.compile(r"no\s+bugs?\s+", re.IGNORECASE),
    re.compile(r"repeated\s+content", re.IGNORECASE),
    re.compile(r"no\s+anti.?patterns?", re.IGNORECASE),
    re.compile(r"transcript\s+(does\s+not|doesn't)\s+include", re.IGNORECASE),
    re.compile(r"same\s+questions?\s+were\s+presented\s+twice", re.IGNORECASE),
]

_MIN_ONE_LINER_LEN = 20
_JACCARD_THRESHOLD = 0.85


def jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _extract_one_liner(draft: dict) -> str:
    """Extract one_liner string from draft dict."""
    try:
        data = json.loads(draft.get("extracted_data") or "{}")
        return data.get("one_liner", "").strip()
    except (json.JSONDecodeError, AttributeError):
        return ""


def filter_noise(
    drafts: list[dict],
    existing_one_liners: list[str],
) -> tuple[list[dict], list[dict]]:
    """Split drafts into (kept, dismissed) using dedup + regex noise filter.

    Dismissal reasons (in order):
    1. Jaccard similarity > threshold vs any existing lesson one_liner
    2. Jaccard similarity > threshold vs any earlier draft in this batch
    3. One_liner length < minimum
    4. Matches a noise regex pattern
    """
    kept: list[dict] = []
    dismissed: list[dict] = []
    seen_one_liners: list[str] = list(existing_one_liners)

    for draft in drafts:
        one_liner = _extract_one_liner(draft)

        if not one_liner:
            draft["_dismiss_reason"] = "empty one_liner"
            dismissed.append(draft)
            continue

        # Check similarity vs all seen (existing + prior batch)
        similar = any(
            jaccard_similarity(one_liner, seen) >= _JACCARD_THRESHOLD
            for seen in seen_one_liners
        )
        if similar:
            draft["_dismiss_reason"] = "duplicate (Jaccard)"
            dismissed.append(draft)
            continue

        if len(one_liner) < _MIN_ONE_LINER_LEN:
            draft["_dismiss_reason"] = f"too short ({len(one_liner)} chars)"
            dismissed.append(draft)
            continue

        if any(p.search(one_liner) for p in _NOISE_PATTERNS):
            draft["_dismiss_reason"] = "noise pattern match"
            dismissed.append(draft)
            continue

        seen_one_liners.append(one_liner)
        kept.append(draft)

    return kept, dismissed
```

**Step 4: Run tests**

```bash
pytest tests/test_review.py::TestJaccardSimilarity tests/test_review.py::TestFilterNoise -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/lessons_db/review.py tests/test_review.py
git commit -m "feat: add review.py noise filter with Jaccard dedup and regex patterns"
```

---

### Task 5: Create review.py — Claude batch reviewer

**Files:**
- Modify: `src/lessons_db/review.py`
- Modify: `tests/test_review.py`

**Step 1: Write failing tests**

Add to `tests/test_review.py`:

```python
from unittest.mock import MagicMock, patch
from lessons_db.review import claude_review_batch


class TestClaudeReviewBatch:
    def _draft(self, id_, one_liner):
        return {"id": id_, "extracted_data": json.dumps({"one_liner": one_liner}), "source": "auto_transcript"}

    def test_returns_promote_verdict_for_specific_lesson(self):
        drafts = [self._draft(42, "Never use bare except: without logging the error first")]
        mock_response = {
            "reviews": [{
                "id": 42,
                "verdict": "PROMOTE",
                "reason": "Specific, actionable, prevents silent failures",
                "improved_one_liner": "Never use bare `except:` — always log before swallowing",
                "detection_pattern": r"except\s*:",
                "semgrep_rule": "",
            }]
        }
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(mock_response))]

        with patch("lessons_db.review.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_msg
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "PROMOTE"
        assert verdicts[0]["id"] == 42

    def test_returns_dismiss_verdict_for_vague_lesson(self):
        drafts = [self._draft(99, "Write cleaner code and test more thoroughly")]
        mock_response = {
            "reviews": [{"id": 99, "verdict": "DISMISS", "reason": "Too vague",
                         "improved_one_liner": "", "detection_pattern": "", "semgrep_rule": ""}]
        }
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=json.dumps(mock_response))]

        with patch("lessons_db.review.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = mock_msg
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        assert verdicts[0]["verdict"] == "DISMISS"

    def test_handles_api_error_gracefully(self):
        drafts = [self._draft(7, "Always log exceptions")]
        with patch("lessons_db.review.anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.side_effect = Exception("API timeout")
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        # On error, drafts are skipped (not crashed), returned as DISMISS with error reason
        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "DISMISS"
        assert "error" in verdicts[0]["reason"].lower()
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_review.py::TestClaudeReviewBatch -v
```

Expected: FAIL — `ImportError: cannot import name 'claude_review_batch'`

**Step 3: Add claude_review_batch to review.py**

Append to `src/lessons_db/review.py`:

```python
import anthropic


_REVIEW_PROMPT_TEMPLATE = """\
You are reviewing draft lessons for a coding lessons-learned system.
For each draft, decide PROMOTE or DISMISS.

Existing lessons (do not promote duplicates of these):
{existing_titles}

Drafts to review:
{draft_lines}

Criteria for PROMOTE:
- Specific: names a concrete pattern, not a general principle
- Actionable: clear do/don't a developer can follow
- Prevents recurrence: would catch this mistake if checked automatically
- Novel: not already in the existing lessons list above

Return ONLY valid JSON, no other text:
{{
  "reviews": [
    {{
      "id": <integer draft id>,
      "verdict": "PROMOTE" or "DISMISS",
      "reason": "<one sentence>",
      "improved_one_liner": "<cleaned wording if PROMOTE, else empty string>",
      "detection_pattern": "<Python regex string for code matching if PROMOTE, else empty string>",
      "semgrep_rule": "<YAML Semgrep rule text if syntactic pattern possible, else empty string>"
    }}
  ]
}}"""

_BATCH_SIZE = 20


def claude_review_batch(
    drafts: list[dict],
    existing_titles: list[str],
    api_key: str,
) -> list[dict]:
    """Send drafts to Claude haiku for PROMOTE/DISMISS review.

    Processes in batches of 20. Returns list of verdict dicts.
    On API error, marks all drafts in that batch as DISMISS with reason='error: <msg>'.
    """
    from lessons_db.config import CLAUDE_REVIEW_MODEL

    client = anthropic.Anthropic(api_key=api_key)
    all_verdicts: list[dict] = []

    for i in range(0, len(drafts), _BATCH_SIZE):
        batch = drafts[i : i + _BATCH_SIZE]
        draft_lines = "\n".join(
            f"[{d['id']}] {_extract_one_liner(d)}" for d in batch
        )
        titles_block = "\n".join(f"- {t}" for t in existing_titles[:150])  # cap to avoid token overflow
        prompt = _REVIEW_PROMPT_TEMPLATE.format(
            existing_titles=titles_block or "(none yet)",
            draft_lines=draft_lines,
        )

        try:
            msg = client.messages.create(
                model=CLAUDE_REVIEW_MODEL,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text.strip()
            data = json.loads(raw)
            all_verdicts.extend(data.get("reviews", []))
        except Exception as exc:
            _log.warning("claude_review_batch: batch %d failed: %s", i // _BATCH_SIZE, exc)
            for d in batch:
                all_verdicts.append({
                    "id": d["id"],
                    "verdict": "DISMISS",
                    "reason": f"error: {exc}",
                    "improved_one_liner": "",
                    "detection_pattern": "",
                    "semgrep_rule": "",
                })

    return all_verdicts
```

**Step 4: Run tests**

```bash
pytest tests/test_review.py -v
```

Expected: all pass.

**Step 5: Commit**

```bash
git add src/lessons_db/review.py tests/test_review.py
git commit -m "feat: add claude_review_batch to review.py — haiku evaluates PROMOTE/DISMISS"
```

---

### Task 6: Create review.py — execute_verdicts + triage log

**Files:**
- Modify: `src/lessons_db/review.py`
- Modify: `tests/test_review.py`

**Step 1: Write failing tests**

Add to `tests/test_review.py`:

```python
import sqlite3
from pathlib import Path
from lessons_db.db import init_db, get_lesson
from lessons_db.review import execute_verdicts


class TestExecuteVerdicts:
    def _insert_draft(self, conn, one_liner, source="auto_transcript"):
        data = {"one_liner": one_liner, "improved_one_liner": one_liner + " (improved)"}
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES ('raw', ?, 'pending', '2026-02-27', ?)",
            [json.dumps(data), source],
        )
        conn.commit()
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def test_promote_verdict_inserts_lesson(self, db_path, tmp_path):
        conn = init_db(db_path)
        draft_id = self._insert_draft(conn, "Never swallow exceptions silently")
        verdicts = [{
            "id": draft_id,
            "verdict": "PROMOTE",
            "reason": "Specific and actionable",
            "improved_one_liner": "Never swallow exceptions — log first",
            "detection_pattern": r"except\s*:",
            "semgrep_rule": "",
        }]

        result = execute_verdicts(conn, verdicts, log_dir=tmp_path)

        assert result["promoted"] == 1
        assert result["dismissed"] == 0
        lesson = conn.execute(
            "SELECT * FROM lessons WHERE one_liner LIKE '%swallow%'"
        ).fetchone()
        assert lesson is not None
        assert lesson["polarity"] == "negative"

    def test_promote_verdict_inserts_detection_pattern(self, db_path, tmp_path):
        conn = init_db(db_path)
        draft_id = self._insert_draft(conn, "Never swallow exceptions silently")
        verdicts = [{
            "id": draft_id, "verdict": "PROMOTE", "reason": "Good",
            "improved_one_liner": "Never swallow exceptions",
            "detection_pattern": r"except\s*:", "semgrep_rule": "",
        }]

        execute_verdicts(conn, verdicts, log_dir=tmp_path)

        pattern = conn.execute("SELECT * FROM detection_patterns").fetchone()
        assert pattern is not None
        assert pattern["regex"] == r"except\s*:"

    def test_dismiss_verdict_marks_draft_dismissed(self, db_path, tmp_path):
        conn = init_db(db_path)
        draft_id = self._insert_draft(conn, "Write better code generally")
        verdicts = [{
            "id": draft_id, "verdict": "DISMISS", "reason": "Too vague",
            "improved_one_liner": "", "detection_pattern": "", "semgrep_rule": "",
        }]

        result = execute_verdicts(conn, verdicts, log_dir=tmp_path)

        assert result["dismissed"] == 1
        row = conn.execute(
            "SELECT status FROM capture_drafts WHERE id = ?", [draft_id]
        ).fetchone()
        assert row["status"] == "dismissed"

    def test_writes_triage_jsonl_log(self, db_path, tmp_path):
        conn = init_db(db_path)
        draft_id = self._insert_draft(conn, "Write better code generally")
        verdicts = [{
            "id": draft_id, "verdict": "DISMISS", "reason": "Too vague",
            "improved_one_liner": "", "detection_pattern": "", "semgrep_rule": "",
        }]

        execute_verdicts(conn, verdicts, log_dir=tmp_path)

        log_files = list(tmp_path.glob("triage-*.jsonl"))
        assert len(log_files) == 1
        lines = log_files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["verdict"] == "DISMISS"
        assert entry["draft_id"] == draft_id
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_review.py::TestExecuteVerdicts -v
```

Expected: FAIL — `ImportError: cannot import name 'execute_verdicts'`

**Step 3: Add execute_verdicts to review.py**

Append to `src/lessons_db/review.py`:

```python
import sqlite3
from lessons_db.capture import promote_draft
from lessons_db.db import insert_detection_pattern


def execute_verdicts(
    conn: sqlite3.Connection,
    verdicts: list[dict],
    log_dir: Path,
) -> dict:
    """Apply PROMOTE/DISMISS verdicts. Returns summary dict."""
    promoted = 0
    dismissed = 0
    log_entries: list[dict] = []
    today = date.today().isoformat()

    for v in verdicts:
        draft_id = v["id"]
        verdict = v.get("verdict", "DISMISS")

        if verdict == "PROMOTE":
            # Inject improved_one_liner into extracted_data before promoting
            row = conn.execute(
                "SELECT extracted_data FROM capture_drafts WHERE id = ?", [draft_id]
            ).fetchone()
            if row:
                data = json.loads(row["extracted_data"] or "{}")
                if v.get("improved_one_liner"):
                    data["improved_one_liner"] = v["improved_one_liner"]
                conn.execute(
                    "UPDATE capture_drafts SET extracted_data = ? WHERE id = ?",
                    [json.dumps(data), draft_id],
                )
                conn.commit()

            lesson_id = promote_draft(conn, draft_id)
            if lesson_id:
                # Insert detection pattern if provided
                pattern = v.get("detection_pattern", "").strip()
                if pattern:
                    try:
                        insert_detection_pattern(conn, {
                            "lesson_id": lesson_id,
                            "pattern_type": "regex",
                            "regex": pattern,
                            "description": v.get("reason", ""),
                            "language": "any",
                        })
                    except Exception as exc:
                        _log.warning("execute_verdicts: pattern insert failed for lesson %d: %s", lesson_id, exc)

                # Write Semgrep rule if provided
                rule_yaml = v.get("semgrep_rule", "").strip()
                if rule_yaml:
                    _write_semgrep_rule(lesson_id, rule_yaml)

                promoted += 1
                log_entries.append({
                    "date": today, "draft_id": draft_id, "lesson_id": lesson_id,
                    "verdict": "PROMOTE", "reason": v.get("reason", ""),
                    "one_liner": v.get("improved_one_liner", ""),
                })
        else:
            conn.execute(
                "UPDATE capture_drafts SET status = 'dismissed' WHERE id = ?",
                [draft_id],
            )
            conn.commit()
            dismissed += 1
            log_entries.append({
                "date": today, "draft_id": draft_id, "lesson_id": None,
                "verdict": "DISMISS", "reason": v.get("reason", ""),
                "one_liner": "",
            })

    # Write JSONL log
    if log_entries:
        log_path = log_dir / f"triage-{today}.jsonl"
        with log_path.open("a") as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + "\n")

    _log.info("execute_verdicts: promoted=%d dismissed=%d", promoted, dismissed)
    return {"promoted": promoted, "dismissed": dismissed}


def _write_semgrep_rule(lesson_id: int, rule_yaml: str) -> None:
    """Write a Semgrep rule YAML to the rules directory."""
    try:
        from lessons_db.config import DATA_DIR
        rules_dir = DATA_DIR / "rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        rule_path = rules_dir / f"lesson-{lesson_id}.yaml"
        rule_path.write_text(rule_yaml)
        _log.debug("_write_semgrep_rule: wrote %s", rule_path)
    except Exception as exc:
        _log.warning("_write_semgrep_rule: failed for lesson %d: %s", lesson_id, exc)
```

**Step 4: Run tests**

```bash
pytest tests/test_review.py -v
```

Expected: all pass.

**Step 5: Run full suite to check for regressions**

```bash
pytest --timeout=120 -x -q
```

Expected: all pass (or only pre-existing failures).

**Step 6: Commit**

```bash
git add src/lessons_db/review.py tests/test_review.py
git commit -m "feat: add execute_verdicts — promote/dismiss with detection pattern and triage log"
```

---

### Task 7: Add `capture review` CLI command

**Files:**
- Modify: `src/lessons_db/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing test**

Add to `tests/test_cli.py` (find existing `TestCapture` class or add alongside it):

```python
from unittest.mock import patch, MagicMock

class TestCaptureReview:
    def test_review_command_runs_pipeline(self, cli_runner, db_path):
        from lessons_db.cli import main
        # Insert a pending draft
        from lessons_db.db import init_db
        conn = init_db(db_path)
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES ('raw', '{\"one_liner\": \"Never swallow exceptions without logging\"}', 'pending', '2026-02-27', 'auto_transcript')"
        )
        conn.commit()

        with patch("lessons_db.review.claude_review_batch") as mock_review:
            mock_review.return_value = []  # all dismissed by noise filter or empty
            result = cli_runner.invoke(main, ["--db", str(db_path), "capture", "review", "--dry-run"])

        assert result.exit_code == 0
```

Check `tests/test_cli.py` for the `cli_runner` fixture name — it may be called `runner`. Adjust accordingly.

**Step 2: Run to verify failure**

```bash
pytest tests/test_cli.py::TestCaptureReview -v
```

Expected: FAIL — `No such command 'review'`

**Step 3: Add `capture review` command to cli.py**

In `src/lessons_db/cli.py`, add after the existing `capture_diff` command (~line 620):

```python
@capture.command("review")
@click.option("--backfill", is_flag=True, help="Process all pending drafts (not just recent).")
@click.option("--dry-run", is_flag=True, help="Run filter only, skip Claude API call, print summary.")
@click.pass_context
def capture_review(ctx, backfill, dry_run):
    """Run automated triage: noise filter + Claude review → promote/dismiss drafts.

    Processes drafts created since the last review run (or all if --backfill).
    Writes decision log to ~/.local/share/lessons-db/triage-YYYY-MM-DD.jsonl.
    """
    import os
    from lessons_db.config import ANTHROPIC_API_KEY, TRIAGE_LOG_DIR
    from lessons_db.review import filter_noise, claude_review_batch, execute_verdicts

    conn = ctx.obj["conn"]

    # Load pending drafts
    query = "SELECT id, extracted_data, source FROM capture_drafts WHERE status = 'pending'"
    drafts = [dict(r) for r in conn.execute(query).fetchall()]

    if not drafts:
        click.echo("No pending drafts to review.")
        return

    # Load existing lesson one_liners for dedup
    existing = [
        r[0] for r in conn.execute("SELECT one_liner FROM lessons WHERE one_liner IS NOT NULL").fetchall()
    ]
    existing_titles = [
        r[0] for r in conn.execute("SELECT one_liner FROM lessons WHERE one_liner IS NOT NULL").fetchall()
    ]

    # Phase 1: noise filter
    kept, dismissed_noise = filter_noise(drafts, existing_one_liners=existing)
    click.echo(f"Filter: {len(drafts)} drafts → {len(kept)} kept, {len(dismissed_noise)} noise-dismissed")

    if dry_run:
        click.echo("[dry-run] Skipping Claude review. Kept drafts:")
        for d in kept[:10]:
            import json as _json
            data = _json.loads(d.get("extracted_data") or "{}")
            click.echo(f"  [{d['id']}] {data.get('one_liner','')[:80]}")
        return

    if not kept:
        click.echo("All drafts dismissed by noise filter. Nothing to send to Claude.")
        # Still mark noise-dismissed drafts
        for d in dismissed_noise:
            conn.execute("UPDATE capture_drafts SET status='dismissed' WHERE id=?", [d["id"]])
        conn.commit()
        return

    # Mark noise-dismissed
    for d in dismissed_noise:
        conn.execute("UPDATE capture_drafts SET status='dismissed' WHERE id=?", [d["id"]])
    conn.commit()

    api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        click.echo("Error: ANTHROPIC_API_KEY not set. Export it or add to ~/.env.", err=True)
        ctx.exit(1)
        return

    # Phase 2: Claude review
    click.echo(f"Sending {len(kept)} drafts to Claude for review...")
    verdicts = claude_review_batch(kept, existing_titles=existing_titles, api_key=api_key)

    # Phase 3: execute
    summary = execute_verdicts(conn, verdicts, log_dir=TRIAGE_LOG_DIR)
    click.echo(f"Done: {summary['promoted']} promoted, {summary['dismissed']} dismissed.")
    click.echo(f"Log: {TRIAGE_LOG_DIR}/triage-{__import__('datetime').date.today().isoformat()}.jsonl")
```

**Step 4: Run tests**

```bash
pytest tests/test_cli.py -v -k "review"
```

Expected: pass.

**Step 5: Smoke test manually**

```bash
lessons-db capture review --dry-run
```

Expected: shows filter counts, exits cleanly.

**Step 6: Commit**

```bash
git add src/lessons_db/cli.py tests/test_cli.py
git commit -m "feat: add 'capture review' CLI command — automated nightly triage pipeline"
```

---

### Task 8: Add `capture triage --review-log` CLI command

**Files:**
- Modify: `src/lessons_db/cli.py`

**Step 1: Add triage command to cli.py**

After the `capture_review` command, add:

```python
@capture.command("triage")
@click.option("--review-log", is_flag=True, help="Show triage decisions from the log.")
@click.option("--date", "log_date", default=None, help="Date to show log for (YYYY-MM-DD). Defaults to today.")
@click.option("--override", "override_id", type=int, default=None,
              help="Re-promote a dismissed draft by ID.")
@click.pass_context
def capture_triage(ctx, review_log, log_date, override_id):
    """Audit triage decisions or override a specific draft."""
    import json as _json
    from pathlib import Path
    from lessons_db.config import TRIAGE_LOG_DIR
    from lessons_db.capture import promote_draft

    conn = ctx.obj["conn"]

    if override_id:
        # Re-promote: reset status to pending, then promote
        conn.execute(
            "UPDATE capture_drafts SET status='pending' WHERE id=?", [override_id]
        )
        conn.commit()
        lesson_id = promote_draft(conn, override_id)
        if lesson_id:
            click.echo(f"✓ Draft {override_id} re-promoted → lesson #{lesson_id}")
        else:
            click.echo(f"✗ Draft {override_id} not found.")
        return

    if review_log:
        target_date = log_date or __import__("datetime").date.today().isoformat()
        log_path = TRIAGE_LOG_DIR / f"triage-{target_date}.jsonl"
        if not log_path.exists():
            click.echo(f"No triage log for {target_date}.")
            return

        promoted = []
        dismissed = []
        for line in log_path.read_text().splitlines():
            entry = _json.loads(line)
            if entry["verdict"] == "PROMOTE":
                promoted.append(entry)
            else:
                dismissed.append(entry)

        click.echo(f"\n=== Triage log: {target_date} ===")
        click.echo(f"Promoted: {len(promoted)} | Dismissed: {len(dismissed)}\n")

        if promoted:
            click.echo("PROMOTED:")
            for e in promoted:
                click.echo(f"  [draft {e['draft_id']} → lesson {e['lesson_id']}] {e['one_liner']}")
                click.echo(f"    Reason: {e['reason']}")

        if dismissed:
            click.echo("\nDISMISSED (sample — first 20):")
            for e in dismissed[:20]:
                click.echo(f"  [draft {e['draft_id']}] {e.get('one_liner','(empty)')}")
                click.echo(f"    Reason: {e['reason']}")
            if len(dismissed) > 20:
                click.echo(f"  ... and {len(dismissed) - 20} more")

        click.echo(f"\nTo override a dismissal: lessons-db capture triage --override <draft_id>")
        return

    click.echo("Usage: lessons-db capture triage --review-log | --override <id>")
```

**Step 2: Smoke test**

```bash
lessons-db capture triage --review-log
```

Expected: "No triage log for YYYY-MM-DD." (none run yet — that's correct)

**Step 3: Commit**

```bash
git add src/lessons_db/cli.py
git commit -m "feat: add 'capture triage --review-log' audit command and --override correction"
```

---

### Task 9: Update nightly service to run capture review

**Files:**
- Modify: `~/.config/systemd/user/lessons-db-nightly.service`

**Step 1: Read current service file**

```bash
cat ~/.config/systemd/user/lessons-db-nightly.service
```

**Step 2: Update ExecStart to chain capture review after transcript batch**

Replace the `ExecStart=` line with:

```ini
ExecStart=/bin/bash -c '\
  source %h/Documents/projects/lessons-db/.venv/bin/activate && \
  bash %h/Documents/projects/lessons-db/scripts/batch-capture-transcripts.sh \
    --since "$(date -d yesterday +%%Y-%%m-%%d)" && \
  source %h/.env 2>/dev/null || true && \
  lessons-db capture review'
```

**Step 3: Reload and verify**

```bash
systemctl --user daemon-reload
systemctl --user status lessons-db-nightly.service
```

Expected: service listed, no errors.

**Step 4: Test trigger (optional — runs the full pipeline)**

```bash
systemctl --user start lessons-db-nightly.service
journalctl --user -u lessons-db-nightly.service -n 30
```

**Step 5: Commit the service file to the repo for reference**

```bash
cp ~/.config/systemd/user/lessons-db-nightly.service \
   ~/Documents/projects/lessons-db/hooks/lessons-db-nightly.service
git add hooks/lessons-db-nightly.service
git commit -m "chore: update nightly service to run 'capture review' after transcript batch"
```

---

### Task 10: Update SessionStart hook to show overnight promotions

**Files:**
- Modify: `~/.claude/hooks/lessons-db-session-start.sh`
- Modify: `~/Documents/projects/lessons-db/hooks/lessons-db-session-start.sh` (repo copy)

**Step 1: Read current hook**

```bash
cat ~/.claude/hooks/lessons-db-session-start.sh
```

**Step 2: Add overnight promotion count after the existing status output**

Add before the final `exit 0` (or at the end of the file):

```bash
# Show overnight promoted lessons from today's triage log
TRIAGE_LOG="$HOME/.local/share/lessons-db/triage-$(date +%Y-%m-%d).jsonl"
if [[ -f "$TRIAGE_LOG" ]]; then
    PROMOTED_TONIGHT=$(grep -c '"verdict": "PROMOTE"' "$TRIAGE_LOG" 2>/dev/null || echo "0")
    if [[ "${PROMOTED_TONIGHT:-0}" -gt 0 ]]; then
        echo "${PROMOTED_TONIGHT} lessons auto-promoted overnight — run: lessons-db capture triage --review-log"
    fi
fi

# Pending triage count (scored but not yet reviewed by Claude)
PENDING_COUNT=$(python3 - <<'PYEOF' 2>/dev/null || echo "0"
import sqlite3, pathlib
db = pathlib.Path.home() / ".local/share/lessons-db/lessons.db"
if db.exists():
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM capture_drafts WHERE status='pending'").fetchone()[0]
    conn.close()
    print(n)
else:
    print(0)
PYEOF
)
if [[ "${PENDING_COUNT:-0}" -gt 50 ]]; then
    echo "Draft backlog: ${PENDING_COUNT} pending — run: lessons-db capture review --dry-run"
fi
```

**Step 3: Copy to repo and deploy**

```bash
cp ~/.claude/hooks/lessons-db-session-start.sh \
   ~/Documents/projects/lessons-db/hooks/lessons-db-session-start.sh
git add hooks/lessons-db-session-start.sh
git commit -m "feat: session start shows overnight promotions and draft backlog count"
```

---

### Task 11: Backfill — clear the 1,906-draft backlog

**Step 1: Dry run first**

```bash
lessons-db capture review --backfill --dry-run
```

Expected: shows filter counts. Check that noise filter dismisses the majority.

**Step 2: Run the full backfill**

```bash
lessons-db capture review --backfill
```

This will:
- Filter 1,906 drafts (dedup + noise → expect ~90% dismissed)
- Send survivors to Claude haiku in batches of 20
- Promote high-signal lessons with detection patterns
- Write `~/.local/share/lessons-db/triage-YYYY-MM-DD.jsonl`

Monitor progress in a second terminal:

```bash
tail -f ~/.local/share/lessons-db/triage-$(date +%Y-%m-%d).jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    e = json.loads(line)
    print(f\"[{e['verdict']}] {e.get('one_liner','')[:70]}\")"
```

**Step 3: Check results**

```bash
lessons-db status
lessons-db capture triage --review-log
```

Expected: `status` now shows detection patterns > 0.

**Step 4: Run full test suite**

```bash
cd ~/Documents/projects/lessons-db
pytest --timeout=120 -x -q
```

Expected: all pass.

**Step 5: Final commit**

```bash
git add -u
git commit -m "chore: backfill complete — automated triage pipeline operational"
```

---

## Summary of Changes

| File | Change |
|------|--------|
| `pyproject.toml` | Add `anthropic>=0.40.0` |
| `src/lessons_db/config.py` | Add `CLAUDE_REVIEW_MODEL`, `ANTHROPIC_API_KEY` |
| `src/lessons_db/capture.py` | Fix `promote_draft` polarity inference |
| `src/lessons_db/db.py` | Add `insert_detection_pattern` |
| `src/lessons_db/review.py` | New: `filter_noise`, `claude_review_batch`, `execute_verdicts` |
| `src/lessons_db/cli.py` | Add `capture review`, `capture triage` commands |
| `hooks/lessons-db-nightly.service` | Chain `capture review` after transcript batch |
| `hooks/lessons-db-session-start.sh` | Add overnight promotion + backlog count |
| `tests/test_capture.py` | Add polarity tests |
| `tests/test_db.py` | Add detection pattern insert tests |
| `tests/test_review.py` | New: full coverage of review.py |

**Success condition:** `lessons-db status` shows detection_patterns > 0 after backfill run.
