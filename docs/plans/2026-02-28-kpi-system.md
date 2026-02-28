# Lessons-DB KPI System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Close the two measurement gaps — recurrence rate and false positive rate — and surface all KPIs in a single `lessons-db kpi` command.

**Architecture:** Extend the `surfacing_events.outcome` enum with `'false_positive'` and `'recurrence'` outcomes. Add `learn dismiss` (false positive signal), `learn list` (query recent events for the post-commit evaluator), and a `lessons-db kpi` dashboard command. Wire a post-commit evaluator hook into ACT that checks recently surfaced lessons against each commit's diff and records heeded/recurrence outcomes automatically.

**Tech Stack:** Python 3.12, Click CLI, SQLite, lessons-db existing DB layer (`src/lessons_db/learn.py`, `src/lessons_db/db.py`, `src/lessons_db/cli.py`), bash hooks in `~/.claude/hooks/` and `projects/autonomous-coding-toolkit/hooks/`

**Test suite:** `pytest --timeout=120 -x -q --override-ini="addopts="` (xdist disabled for isolation)

---

## Task 1: Extend `record_outcome()` to accept new outcome types

The current `record_outcome()` only allows `'heeded'` or `'dismissed'`. We need `'false_positive'` and `'recurrence'`.

**Files:**
- Modify: `src/lessons_db/learn.py:26-36`
- Modify: `tests/test_learn.py`

**Step 1: Write failing tests**

Add to `tests/test_learn.py`:

```python
def test_record_outcome_false_positive(tmp_path):
    """record_outcome accepts false_positive outcome."""
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_surfacing, record_outcome
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Test", "one_liner": "test", "cluster": "A",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    eid = record_surfacing(conn, lid, "plan", "ctx")
    record_outcome(conn, eid, "false_positive")
    row = conn.execute(
        "SELECT outcome FROM surfacing_events WHERE id = ?", [eid]
    ).fetchone()
    assert row["outcome"] == "false_positive"


def test_record_outcome_recurrence(tmp_path):
    """record_outcome accepts recurrence outcome."""
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_surfacing, record_outcome
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Test2", "one_liner": "test2", "cluster": "A",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    eid = record_surfacing(conn, lid, "bash", "ctx")
    record_outcome(conn, eid, "recurrence")
    row = conn.execute(
        "SELECT outcome FROM surfacing_events WHERE id = ?", [eid]
    ).fetchone()
    assert row["outcome"] == "recurrence"


def test_record_outcome_rejects_invalid(tmp_path):
    """record_outcome raises ValueError on unknown outcome."""
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_surfacing, record_outcome
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Test3", "one_liner": "test3", "cluster": "A",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    eid = record_surfacing(conn, lid, "plan", "ctx")
    with pytest.raises(ValueError, match="Invalid outcome"):
        record_outcome(conn, eid, "wrong")
```

**Step 2: Run to confirm FAIL**

```bash
cd /home/justin/Documents/projects/lessons-db
.venv/bin/python -m pytest tests/test_learn.py::test_record_outcome_false_positive -v --override-ini="addopts="
```
Expected: FAIL — `ValueError: Invalid outcome 'false_positive'`

**Step 3: Implement**

In `src/lessons_db/learn.py`, change:
```python
# Before
if outcome not in ("heeded", "dismissed"):
    raise ValueError(f"Invalid outcome '{outcome}'. Must be 'heeded' or 'dismissed'.")

# After
_VALID_OUTCOMES = ("heeded", "dismissed", "false_positive", "recurrence")
if outcome not in _VALID_OUTCOMES:
    raise ValueError(f"Invalid outcome '{outcome}'. Must be one of: {', '.join(_VALID_OUTCOMES)}.")
```

**Step 4: Run tests to confirm PASS**

```bash
.venv/bin/python -m pytest tests/test_learn.py -v --override-ini="addopts="
```
Expected: all pass including 3 new tests

**Step 5: Commit**

```bash
git add src/lessons_db/learn.py tests/test_learn.py
git commit -m "feat: extend record_outcome to accept false_positive and recurrence"
```

---

## Task 2: Add `--outcome` option to `learn record` CLI

Currently `learn record` only records a surfacing event (always `outcome='unknown'`). Add optional `--outcome` to set it immediately.

**Files:**
- Modify: `src/lessons_db/cli.py` (around line 958–972)
- Modify: `tests/test_cli.py`

**Step 1: Write failing test**

Add to `tests/test_cli.py`:

```python
def test_learn_record_with_outcome_heeded(tmp_path):
    """learn record --outcome heeded records heeded surfacing event."""
    from lessons_db.db import init_db, insert_lesson
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Test", "one_liner": "t", "cluster": "A",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--db", str(db_path), "learn", "record",
         "--lesson-id", "1", "--hook", "plan", "--context", "ctx",
         "--outcome", "heeded"],
    )
    assert result.exit_code == 0
    row = conn.execute(
        "SELECT outcome FROM surfacing_events WHERE lesson_id = 1"
    ).fetchone()
    assert row["outcome"] == "heeded"


def test_learn_record_with_outcome_false_positive(tmp_path):
    """learn record --outcome false_positive records dismissal."""
    from lessons_db.db import init_db, insert_lesson
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Test2", "one_liner": "t", "cluster": "B",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--db", str(db_path), "learn", "record",
         "--lesson-id", "1", "--hook", "edit", "--context", "ctx",
         "--outcome", "false_positive"],
    )
    assert result.exit_code == 0
    row = conn.execute(
        "SELECT outcome FROM surfacing_events WHERE lesson_id = 1"
    ).fetchone()
    assert row["outcome"] == "false_positive"
```

**Step 2: Run to confirm FAIL**

```bash
.venv/bin/python -m pytest tests/test_cli.py::test_learn_record_with_outcome_heeded -v --override-ini="addopts="
```
Expected: FAIL — unrecognized option `--outcome`

**Step 3: Implement**

In `cli.py`, update the `learn_record` command:

```python
@learn.command("record")
@click.option("--lesson-id", required=True, type=int)
@click.option(
    "--hook", "hook_point", required=True,
    type=click.Choice(["read", "edit", "plan", "bash", "session_start", "commit"])
)
@click.option("--context", "hook_context", default="", help="File path, query, or error text.")
@click.option(
    "--outcome", default=None,
    type=click.Choice(["heeded", "dismissed", "false_positive", "recurrence"]),
    help="Outcome to record immediately (optional, default: unknown).",
)
@click.pass_context
def learn_record(click_ctx, lesson_id, hook_point, hook_context, outcome):
    """Record that a lesson was surfaced at a hook point."""
    from lessons_db.learn import record_surfacing, record_outcome

    conn = click_ctx.obj["conn"]
    event_id = record_surfacing(conn, lesson_id, hook_point, hook_context)
    if outcome is not None:
        record_outcome(conn, event_id, outcome)
    click.echo(f"Recorded surfacing event {event_id}")
```

Note: also add `"commit"` to the hook_point Choice list (used by post-commit evaluator).

**Step 4: Run tests to confirm PASS**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v --override-ini="addopts=" -k "learn_record"
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/lessons_db/cli.py tests/test_cli.py
git commit -m "feat: add --outcome option to learn record CLI"
```

---

## Task 3: Add `learn dismiss <lesson_id>` convenience command

Used by hooks and users to mark the most recent surfacing event for a lesson as a false positive.

**Files:**
- Modify: `src/lessons_db/cli.py`
- Modify: `src/lessons_db/learn.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing tests**

Add to `tests/test_cli.py`:

```python
def test_learn_dismiss_marks_latest_event_false_positive(tmp_path):
    """learn dismiss marks most recent unknown surfacing as false_positive."""
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_surfacing
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Dismiss test", "one_liner": "t", "cluster": "A",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    # Record two events; dismiss should only affect the latest
    record_surfacing(conn, 1, "plan", "ctx-old")
    record_surfacing(conn, 1, "edit", "ctx-new")
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "learn", "dismiss", "1"])
    assert result.exit_code == 0
    rows = conn.execute(
        "SELECT outcome FROM surfacing_events WHERE lesson_id = 1 ORDER BY id"
    ).fetchall()
    assert rows[0]["outcome"] == "unknown"       # first event unchanged
    assert rows[1]["outcome"] == "false_positive" # latest event dismissed


def test_learn_dismiss_no_events(tmp_path):
    """learn dismiss exits cleanly when no surfacing events exist."""
    from lessons_db.db import init_db, insert_lesson
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "No events", "one_liner": "t", "cluster": "A",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "learn", "dismiss", "1"])
    assert result.exit_code == 0
    assert "no surfacing events" in result.output.lower()
```

**Step 2: Run to confirm FAIL**

```bash
.venv/bin/python -m pytest tests/test_cli.py::test_learn_dismiss_marks_latest_event_false_positive -v --override-ini="addopts="
```
Expected: FAIL — no command `dismiss`

**Step 3: Implement**

Add to `src/lessons_db/learn.py`:

```python
def dismiss_latest(conn: sqlite3.Connection, lesson_id: int) -> bool:
    """Mark the most recent unknown surfacing event for lesson_id as false_positive.
    Returns True if an event was found and updated, False if none existed.
    """
    row = conn.execute(
        "SELECT id FROM surfacing_events "
        "WHERE lesson_id = ? AND outcome = 'unknown' "
        "ORDER BY id DESC LIMIT 1",
        [lesson_id],
    ).fetchone()
    if row is None:
        return False
    record_outcome(conn, row["id"], "false_positive")
    return True
```

Add to `cli.py` inside the `learn` group:

```python
@learn.command("dismiss")
@click.argument("lesson_id", type=int)
@click.pass_context
def learn_dismiss(click_ctx, lesson_id):
    """Mark most recent surfacing of LESSON_ID as a false positive."""
    from lessons_db.learn import dismiss_latest

    conn = click_ctx.obj["conn"]
    updated = dismiss_latest(conn, lesson_id)
    if updated:
        click.echo(f"Marked latest surfacing of lesson #{lesson_id} as false_positive.")
    else:
        click.echo(f"No surfacing events found for lesson #{lesson_id}.")
```

**Step 4: Run tests to confirm PASS**

```bash
.venv/bin/python -m pytest tests/test_cli.py -v --override-ini="addopts=" -k "dismiss"
```
Expected: all pass

**Step 5: Commit**

```bash
git add src/lessons_db/learn.py src/lessons_db/cli.py tests/test_cli.py
git commit -m "feat: add learn dismiss command for false positive recording"
```

---

## Task 4: Add `learn list --since <window>` command

The post-commit evaluator needs to query "which lessons were surfaced in the last N hours." Add a `learn list` command.

**Files:**
- Modify: `src/lessons_db/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing test**

```python
def test_learn_list_since_filters_by_time(tmp_path):
    """learn list --since 1h returns only recent surfacing events."""
    import time
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_surfacing
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Recent", "one_liner": "t", "cluster": "A",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    insert_lesson(conn, {
        "title": "Old", "one_liner": "t", "cluster": "B",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    # Insert old event by manipulating timestamp directly
    record_surfacing(conn, 1, "plan", "recent-ctx")
    conn.execute(
        "INSERT INTO surfacing_events (lesson_id, hook_point, context, outcome, timestamp) "
        "VALUES (2, 'plan', 'old-ctx', 'unknown', ?)",
        [int(time.time()) - 7200],  # 2 hours ago
    )
    conn.commit()

    runner = CliRunner()
    result = runner.invoke(
        main, ["--db", str(db_path), "learn", "list", "--since", "1h", "--format", "ids"]
    )
    assert result.exit_code == 0
    assert "1" in result.output     # recent lesson
    assert "2" not in result.output  # old lesson filtered out
```

**Step 2: Run to confirm FAIL**

```bash
.venv/bin/python -m pytest tests/test_cli.py::test_learn_list_since_filters_by_time -v --override-ini="addopts="
```

**Step 3: Implement**

Add to `cli.py` inside the `learn` group:

```python
@learn.command("list")
@click.option(
    "--since", default="24h",
    help="Time window: '1h', '6h', '24h', '7d'. Default: 24h.",
)
@click.option(
    "--format", "output_format", default="table",
    type=click.Choice(["table", "ids"]),
    help="Output format. 'ids' prints one lesson_id per line for scripting.",
)
@click.pass_context
def learn_list(click_ctx, since, output_format):
    """List recent surfacing events."""
    import time
    import re

    conn = click_ctx.obj["conn"]

    # Parse window: e.g. "2h" -> 7200, "7d" -> 604800
    match = re.fullmatch(r"(\d+)([hd])", since)
    if not match:
        raise click.BadParameter(f"Invalid window '{since}'. Use e.g. '1h', '24h', '7d'.")
    n, unit = int(match.group(1)), match.group(2)
    seconds = n * 3600 if unit == "h" else n * 86400
    cutoff = int(time.time()) - seconds

    rows = conn.execute(
        "SELECT se.id, se.lesson_id, l.title, se.hook_point, se.outcome, se.timestamp "
        "FROM surfacing_events se JOIN lessons l ON l.id = se.lesson_id "
        "WHERE se.timestamp >= ? ORDER BY se.timestamp DESC",
        [cutoff],
    ).fetchall()

    if not rows:
        click.echo("No surfacing events in window.")
        return

    if output_format == "ids":
        seen = set()
        for row in rows:
            if row["lesson_id"] not in seen:
                click.echo(row["lesson_id"])
                seen.add(row["lesson_id"])
    else:
        click.echo(f"{'ID':>6}  {'LID':>5}  {'Hook':<14}  {'Outcome':<14}  Title")
        click.echo("-" * 72)
        for row in rows:
            title = row["title"][:35]
            click.echo(
                f"{row['id']:>6}  {row['lesson_id']:>5}  {row['hook_point']:<14}  "
                f"{row['outcome']:<14}  {title}"
            )
```

**Step 4: Run tests to confirm PASS**

```bash
.venv/bin/python -m pytest tests/test_cli.py::test_learn_list_since_filters_by_time -v --override-ini="addopts="
```

**Step 5: Commit**

```bash
git add src/lessons_db/cli.py tests/test_cli.py
git commit -m "feat: add learn list --since command for post-commit evaluator"
```

---

## Task 5: Add `lessons-db kpi` dashboard command

Single command that prints all KPIs. Queries are pure SQL against existing tables — no new schema needed.

**Files:**
- Modify: `src/lessons_db/cli.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing test**

```python
def test_kpi_command_runs_and_shows_metrics(tmp_path):
    """kpi command outputs all expected metric labels."""
    from lessons_db.db import init_db, insert_lesson
    from lessons_db.learn import record_surfacing, record_outcome
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Test KPI", "one_liner": "t", "cluster": "A",
        "tier": "lesson", "created_date": "2026-01-01",
    })
    eid1 = record_surfacing(conn, lid, "plan", "ctx")
    record_outcome(conn, eid1, "heeded")
    eid2 = record_surfacing(conn, lid, "edit", "ctx")
    record_outcome(conn, eid2, "recurrence")
    eid3 = record_surfacing(conn, lid, "bash", "ctx")
    record_outcome(conn, eid3, "false_positive")

    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "kpi"])
    assert result.exit_code == 0
    assert "Recurrence Rate" in result.output
    assert "Heed Rate" in result.output
    assert "False Positive Rate" in result.output
    assert "Dead Lessons" in result.output
    assert "DB Growth" in result.output
```

**Step 2: Run to confirm FAIL**

```bash
.venv/bin/python -m pytest tests/test_cli.py::test_kpi_command_runs_and_shows_metrics -v --override-ini="addopts="
```

**Step 3: Implement**

Add as a top-level command (not under `stats` or `learn`) in `cli.py`:

```python
@main.command("kpi")
@click.pass_context
def kpi_dashboard(click_ctx):
    """Show all KPI metrics for the lesson learning system."""
    import time

    conn = click_ctx.obj["conn"]

    def q(sql, params=None):
        row = conn.execute(sql, params or []).fetchone()
        return row[0] if row else 0

    total_lessons = q("SELECT COUNT(*) FROM lessons WHERE polarity='negative'")
    total_surfacings = q("SELECT COUNT(*) FROM surfacing_events")
    decided = q(
        "SELECT COUNT(*) FROM surfacing_events WHERE outcome != 'unknown'"
    )
    heeded = q(
        "SELECT COUNT(*) FROM surfacing_events WHERE outcome = 'heeded'"
    )
    recurrences = q(
        "SELECT COUNT(*) FROM surfacing_events WHERE outcome = 'recurrence'"
    )
    false_positives = q(
        "SELECT COUNT(*) FROM surfacing_events WHERE outcome = 'false_positive'"
    )
    heed_recur = heeded + recurrences
    cutoff_90d = int(time.time()) - 86400 * 90
    cutoff_7d = int(time.time()) - 86400 * 7
    dead_lessons = q(
        "SELECT COUNT(*) FROM lessons l WHERE NOT EXISTS ("
        "  SELECT 1 FROM surfacing_events se "
        "  WHERE se.lesson_id = l.id AND se.timestamp >= ?"
        ")",
        [cutoff_90d],
    )
    growth_7d = q(
        "SELECT COUNT(*) FROM lessons WHERE created_date >= date('now','-7 days')"
    )
    heed_rate = round(heeded / decided * 100, 1) if decided > 0 else None
    recurrence_rate = round(recurrences / heed_recur * 100, 1) if heed_recur > 0 else None
    fp_rate = round(false_positives / decided * 100, 1) if decided > 0 else None
    dead_pct = round(dead_lessons / total_lessons * 100, 1) if total_lessons > 0 else 0

    def fmt(val, target_ok, unit="%"):
        if val is None:
            return "  n/a    (no data yet)"
        ok = "✓" if target_ok(val) else "✗"
        return f"  {val}{unit}  {ok}"

    click.echo("")
    click.echo("  Lessons-DB KPI Dashboard")
    click.echo("  " + "─" * 44)
    click.echo(f"  Total lessons          : {total_lessons}")
    click.echo(f"  Total surfacings       : {total_surfacings}  ({decided} with outcome)")
    click.echo("")
    click.echo("  Outcome KPIs (need outcome data to populate):")
    click.echo(f"  Recurrence Rate        :{fmt(recurrence_rate, lambda v: v < 5)}")
    click.echo(f"  Heed Rate              :{fmt(heed_rate,        lambda v: v > 50)}")
    click.echo(f"  False Positive Rate    :{fmt(fp_rate,          lambda v: v < 15)}")
    click.echo("")
    click.echo("  System Health:")
    click.echo(f"  Dead Lessons (90d)     :  {dead_lessons} ({dead_pct}%)  {'✓' if dead_pct < 10 else '✗'}")
    click.echo(f"  DB Growth (7d)         :  +{growth_7d} lessons")
    click.echo("")
```

**Step 4: Run tests to confirm PASS**

```bash
.venv/bin/python -m pytest tests/test_cli.py::test_kpi_command_runs_and_shows_metrics -v --override-ini="addopts="
```

**Step 5: Run full suite**

```bash
.venv/bin/python -m pytest --timeout=120 -x -q --override-ini="addopts="
```
Expected: all existing tests + 5 new tests pass

**Step 6: Commit**

```bash
git add src/lessons_db/cli.py tests/test_cli.py
git commit -m "feat: add kpi dashboard command"
```

---

## Task 6: Post-commit evaluator hook (ACT)

Wire a post-commit hook into ACT that checks recently-surfaced lessons against the current diff and records heeded/recurrence outcomes.

**Files:**
- Create: `/home/justin/Documents/projects/autonomous-coding-toolkit/hooks/post-commit-evaluator.sh`
- Modify: `/home/justin/Documents/projects/autonomous-coding-toolkit/hooks/post-commit`
- Modify: `/home/justin/Documents/projects/autonomous-coding-toolkit/install.sh`

**Step 1: Write the evaluator script**

Create `hooks/post-commit-evaluator.sh`:

```bash
#!/usr/bin/env bash
# Post-commit evaluator: records heeded/recurrence outcomes for recently surfaced lessons.
# Called from hooks/post-commit after the lesson auto-import step.
set -euo pipefail

LESSONS_DB=$(command -v lessons-db 2>/dev/null || echo "")
if [[ -z "$LESSONS_DB" ]]; then
    exit 0
fi

# Get lessons surfaced in the last 6 hours (covering current session)
SURFACED_IDS=$("$LESSONS_DB" learn list --since 6h --format ids 2>/dev/null || echo "")
if [[ -z "$SURFACED_IDS" ]]; then
    exit 0
fi

# Get the diff of the current commit (content lines only, no metadata)
DIFF=$(git diff HEAD~1 2>/dev/null || echo "")
if [[ -z "$DIFF" ]]; then
    exit 0
fi

# Write diff to temp file for lessons-db check
DIFF_TMP=$(mktemp)
trap 'rm -f "$DIFF_TMP"' EXIT
echo "$DIFF" > "$DIFF_TMP"

while IFS= read -r lesson_id; do
    [[ -z "$lesson_id" ]] && continue

    # Check if this lesson's patterns appear in the diff
    VIOLATIONS=$("$LESSONS_DB" check --lesson-id "$lesson_id" --file "$DIFF_TMP" 2>/dev/null || echo "")

    if [[ -n "$VIOLATIONS" ]]; then
        # Pattern found in diff — lesson was NOT applied (recurrence)
        "$LESSONS_DB" learn record \
            --lesson-id "$lesson_id" \
            --hook "commit" \
            --context "$(git log -1 --format='%s' 2>/dev/null || echo 'commit')" \
            --outcome "recurrence" \
            2>>/tmp/lessons-db-errors.log || true
    else
        # Pattern absent from diff — lesson was applied (heeded)
        "$LESSONS_DB" learn record \
            --lesson-id "$lesson_id" \
            --hook "commit" \
            --context "$(git log -1 --format='%s' 2>/dev/null || echo 'commit')" \
            --outcome "heeded" \
            2>>/tmp/lessons-db-errors.log || true
    fi
done <<< "$SURFACED_IDS"

exit 0
```

**Step 2: Update `hooks/post-commit` to call the evaluator**

At the end of the existing `hooks/post-commit`, add:

```bash
# Run post-commit evaluator to record heeded/recurrence outcomes
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HOOK_DIR/post-commit-evaluator.sh" ]]; then
    bash "$HOOK_DIR/post-commit-evaluator.sh" || true
fi
```

**Step 3: Update `install.sh`**

Add alongside the existing post-commit hook install line:

```bash
cp hooks/post-commit-evaluator.sh .git/hooks/post-commit-evaluator.sh
chmod +x .git/hooks/post-commit-evaluator.sh
```

**Step 4: Verify syntax**

```bash
bash -n hooks/post-commit-evaluator.sh && echo "OK"
bash -n hooks/post-commit && echo "OK"
shellcheck hooks/post-commit-evaluator.sh
```
Expected: no errors

**Step 5: Commit**

```bash
cd /home/justin/Documents/projects/autonomous-coding-toolkit
git add hooks/post-commit hooks/post-commit-evaluator.sh install.sh
git commit -m "feat: add post-commit evaluator for heeded/recurrence outcome recording"
```

---

## Task 7: Wire `learn dismiss` into global Claude hooks

The pre-edit hook already fires when code is written. When a user wants to dismiss a false positive from the terminal, they should be able to call `lessons-db learn dismiss <id>`. Add a global hook that detects this pattern and records the outcome.

**Files:**
- Modify: `~/.claude/hooks/lessons-db-post-bash.sh`

**Step 1: Read current post-bash hook**

```bash
cat ~/.claude/hooks/lessons-db-post-bash.sh
```

**Step 2: Add dismissal detection**

Find the section that reads bash output and add detection for explicit dismiss commands. If the user runs `lessons-db learn dismiss N` or `learn dismiss N`, the hook records it. (The CLI command itself already handles this — no hook change needed if the user calls the CLI directly.)

If the file doesn't already handle this, add to the end of the post-bash hook:

```bash
# If user explicitly ran a dismiss command, confirm it was recorded
if echo "${BASH_OUTPUT:-}" | grep -q "false_positive"; then
    echo "[lessons-db] False positive recorded."
fi
```

**Step 3: Verify syntax**

```bash
bash -n ~/.claude/hooks/lessons-db-post-bash.sh && echo "OK"
```

**Step 4: No commit needed** — hooks/ is not a tracked repo. Changes take effect immediately.

---

## Task 8: Final integration test + push

**Step 1: Run full lessons-db test suite**

```bash
cd /home/justin/Documents/projects/lessons-db
.venv/bin/python -m pytest --timeout=120 -x -q --override-ini="addopts="
```
Expected: all tests pass (268 existing + ~12 new = ~280 total)

**Step 2: Smoke-test the KPI command against the live DB**

```bash
lessons-db kpi
```
Expected: dashboard prints without error. Recurrence Rate will show "n/a" until the post-commit evaluator runs, which is correct.

**Step 3: Smoke-test learn dismiss**

```bash
lessons-db learn list --since 24h | head -5
# Pick any lesson_id from output
lessons-db learn dismiss <id>
```
Expected: "Marked latest surfacing of lesson #N as false_positive."

**Step 4: Push both repos**

```bash
cd /home/justin/Documents/projects/lessons-db && git push origin main
cd /home/justin/Documents/projects/autonomous-coding-toolkit && git push origin main
```

---

## Quality Gate

```
lessons-db test suite: all pass
lessons-db kpi: runs without error
lessons-db learn list --since 1h: returns list
lessons-db learn dismiss <id>: updates outcome
post-commit-evaluator.sh: bash -n clean, shellcheck clean
```
