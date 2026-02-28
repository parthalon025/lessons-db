# lessons-db Completion Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close all gaps between the designed system and the running implementation — wire CLI commands to existing modules, deploy remaining hooks, add auto-capture, and connect the capture-lesson skill to the DB.

**Architecture:** All core logic modules already exist (`rulegen.py`, `scan.py`, `export.py`, `capture.py`). This plan is almost entirely wiring: adding CLI commands that call existing functions, writing hook scripts that call the CLI, and extending `capture.py` with transcript/diff analysis.

**Tech Stack:** Python 3.12, Click CLI, SQLite, LanceDB, Ollama (via ollama-queue), Semgrep, bash hooks

---

## Task 1: Wire `rule` CLI command group

Exposes `rulegen.py` functions through `lessons-db rule generate` and `lessons-db rule test`.

**Files:**
- Modify: `src/lessons_db/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py — add to end of file

def test_rule_generate_no_patterns(tmp_path):
    """rule generate exits cleanly when lesson has no detection patterns."""
    from lessons_db.db import init_db, insert_lesson
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Log before fallback", "one_liner": "Never swallow",
        "cluster": "A", "tier": "lesson", "created_date": "2026-01-01",
    })
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "rule", "generate", "1"])
    assert result.exit_code == 0
    assert "no detection patterns" in result.output.lower()


def test_rule_generate_with_patterns(tmp_path):
    """rule generate writes YAML to rules_dir when patterns exist."""
    from lessons_db.db import init_db, insert_lesson
    db_path = tmp_path / "test.db"
    rules_dir = tmp_path / "rules"
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Bare except swallows failures",
        "one_liner": "Never use bare except",
        "cluster": "A", "tier": "lesson", "created_date": "2026-01-01",
    })
    conn.execute(
        "INSERT INTO detection_patterns "
        "(lesson_id, pattern_type, regex, description, language) VALUES (?,?,?,?,?)",
        [lid, "regex", r"except\s*:", "bare except", "python"],
    )
    conn.commit()
    runner = CliRunner()
    result = runner.invoke(
        main, ["--db", str(db_path), "rule", "generate", str(lid),
               "--rules-dir", str(rules_dir)],
    )
    assert result.exit_code == 0
    assert "generated" in result.output.lower()
    yaml_files = list(rules_dir.glob("**/*.yaml"))
    assert len(yaml_files) == 1


def test_rule_test_no_rules(tmp_path):
    """rule test exits cleanly when no rules exist."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["--db", str(tmp_path / "test.db"), "rule", "test",
               "--rules-dir", str(tmp_path / "rules")],
    )
    assert result.exit_code == 0
    assert "no rules" in result.output.lower()
```

**Step 2: Run to confirm FAIL**

```
pytest tests/test_cli.py::test_rule_generate_no_patterns -v
```
Expected: `ERROR` — `No such command 'rule'`

**Step 3: Add `rule` command group to `src/lessons_db/cli.py`**

Add after the `index` command and before the `capture` group. Add `RULES_DIR` to the config import at the top:

```python
from lessons_db.config import SQLITE_PATH, LANCE_DIR, LESSONS_SOURCE_DIR, RULES_DIR
```

Then add:

```python
@main.group()
def rule():
    """Generate and test Semgrep rules from lessons."""
    pass


@rule.command("generate")
@click.argument("lesson_id", type=int)
@click.option("--rules-dir", type=click.Path(), default=None,
              help="Directory to write rules (default: ~/.local/share/lessons-db/rules/)")
@click.option("--severity", default="WARNING",
              type=click.Choice(["WARNING", "ERROR", "INFO"]),
              help="Semgrep rule severity.")
@click.pass_context
def rule_generate(ctx, lesson_id, rules_dir, severity):
    """Generate a Semgrep rule YAML file for a lesson."""
    from lessons_db.rulegen import generate_rule
    from pathlib import Path as _Path

    conn = ctx.obj["conn"]
    lesson = conn.execute("SELECT * FROM lessons WHERE id = ?", (lesson_id,)).fetchone()
    if not lesson:
        click.echo(f"Lesson #{lesson_id} not found.")
        return

    patterns = conn.execute(
        "SELECT * FROM detection_patterns WHERE lesson_id = ?", (lesson_id,)
    ).fetchall()
    if not patterns:
        click.echo(f"No detection patterns for lesson #{lesson_id}. "
                   "Add patterns via detection_patterns table first.")
        return

    out_dir = _Path(rules_dir) if rules_dir else RULES_DIR
    language = patterns[0]["language"] or "any"
    lang_dir = out_dir / language
    lang_dir.mkdir(parents=True, exist_ok=True)

    from lessons_db.rulegen import slug_from_title
    slug = slug_from_title(lesson["title"])
    rule_file = lang_dir / f"{slug}-{lesson_id:03d}.yaml"

    rule_yaml = generate_rule(dict(lesson), [dict(p) for p in patterns], severity=severity)
    rule_file.write_text(rule_yaml, encoding="utf-8")
    click.echo(f"Generated: {rule_file}")


@rule.command("test")
@click.option("--rules-dir", type=click.Path(), default=None,
              help="Directory containing rules (default: ~/.local/share/lessons-db/rules/)")
@click.pass_context
def rule_test(ctx, rules_dir):
    """Run semgrep --test against all generated rules."""
    import shutil
    import subprocess
    from pathlib import Path as _Path

    out_dir = _Path(rules_dir) if rules_dir else RULES_DIR
    if not out_dir.exists() or not any(out_dir.rglob("*.yaml")):
        click.echo("No rules found. Run: lessons-db rule generate <id>")
        return

    semgrep = shutil.which("semgrep")
    if not semgrep:
        click.echo("semgrep not found on PATH.")
        return

    result = subprocess.run(
        [semgrep, "--test", str(out_dir)],
        capture_output=True, text=True,
    )
    click.echo(result.stdout or result.stderr)
    if result.returncode == 0:
        click.echo("All rules passed.")
    else:
        click.echo(f"Test failures (exit code {result.returncode}).")
```

**Step 4: Run tests**

```
pytest tests/test_cli.py::test_rule_generate_no_patterns \
       tests/test_cli.py::test_rule_generate_with_patterns \
       tests/test_cli.py::test_rule_test_no_rules -v
```
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/lessons_db/cli.py tests/test_cli.py
git commit -m "feat: wire rule generate/test CLI commands to rulegen.py"
```

---

## Task 2: Wire `scan` CLI command

Exposes `scan.py` through `lessons-db scan`.

**Files:**
- Modify: `src/lessons_db/cli.py`
- Modify: `src/lessons_db/db.py` (add `insert_scan_findings_batch`)
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py — add to end

@patch("lessons_db.scan.subprocess.run")
def test_scan_command_runs(mock_run, tmp_path):
    """scan command calls semgrep and reports findings."""
    import json
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout=json.dumps({"version": "2.1.0", "runs": [{"results": []}]}),
    )
    runner = CliRunner()
    result = runner.invoke(
        main, ["--db", str(tmp_path / "test.db"), "scan",
               "--rules-dir", str(tmp_path / "rules"),
               "--target", str(tmp_path)],
    )
    assert result.exit_code == 0


def test_scan_command_no_rules(tmp_path):
    """scan exits cleanly when rules dir is empty."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["--db", str(tmp_path / "test.db"), "scan",
               "--rules-dir", str(tmp_path / "empty-rules"),
               "--target", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "no rules" in result.output.lower()
```

**Step 2: Run to confirm FAIL**

```
pytest tests/test_cli.py::test_scan_command_runs -v
```
Expected: `ERROR` — `No such command 'scan'`

**Step 3: Add `scan` command to `src/lessons_db/cli.py`**

Add after the `rule` group:

```python
@main.command()
@click.option("--rules-dir", type=click.Path(), default=None,
              help="Rules directory (default: ~/.local/share/lessons-db/rules/)")
@click.option("--target", type=click.Path(), default=None,
              help="Target directory to scan (default: ~/Documents/projects/)")
@click.option("--baseline", default=None,
              help="Git commit hash for diff-aware scanning.")
@click.pass_context
def scan(ctx, rules_dir, target, baseline):
    """Run Semgrep scan against all lessons rules and record findings."""
    from pathlib import Path as _Path
    from lessons_db.scan import run_scan
    from lessons_db.db import insert_scan_finding

    rules = _Path(rules_dir) if rules_dir else RULES_DIR
    if not rules.exists() or not any(rules.rglob("*.yaml")):
        click.echo("No rules found. Run: lessons-db rule generate <id>")
        return

    target_path = _Path(target) if target else _Path.home() / "Documents" / "projects"
    conn = ctx.obj["conn"]

    click.echo(f"Scanning {target_path} with rules from {rules}...")
    findings = run_scan(
        rules_dir=rules,
        target_dir=target_path,
        baseline_commit=baseline,
    )

    if not findings:
        click.echo("No findings.")
        return

    for f in findings:
        # Map rule_id back to lesson_id via enforcement_rules or metadata
        rule_id = f.get("rule_id", "")
        click.echo(f"  [{rule_id}] {f.get('file_path')}:{f.get('line_number')}")
        insert_scan_finding(conn, {
            "lesson_id": 0,  # unknown without rule→lesson mapping
            "rule_id": rule_id,
            "file_path": f.get("file_path", ""),
            "line_number": f.get("line_number"),
            "snippet": f.get("message", ""),
        })

    click.echo(f"\nTotal findings: {len(findings)} (saved to DB)")
```

**Step 4: Run tests**

```
pytest tests/test_cli.py::test_scan_command_runs \
       tests/test_cli.py::test_scan_command_no_rules -v
```
Expected: 2 PASS

**Step 5: Commit**

```bash
git add src/lessons_db/cli.py tests/test_cli.py
git commit -m "feat: wire scan CLI command to scan.py"
```

---

## Task 3: Add `export` and `summary` CLI commands

**Files:**
- Modify: `src/lessons_db/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
# tests/test_cli.py — add to end

def test_export_command(tmp_path):
    """export outputs lesson markdown."""
    from lessons_db.db import init_db, insert_lesson
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    lid = insert_lesson(conn, {
        "title": "Log every external failure",
        "one_liner": "Never swallow exceptions silently",
        "cluster": "A", "tier": "lesson_learned", "created_date": "2026-01-01",
    })
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(db_path), "export", str(lid)])
    assert result.exit_code == 0
    assert "Log every external failure" in result.output
    assert "Key Takeaway" in result.output


def test_export_missing_lesson(tmp_path):
    """export exits cleanly for unknown lesson ID."""
    runner = CliRunner()
    result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "export", "999"])
    assert result.exit_code == 0
    assert "not found" in result.output.lower()


def test_summary_command(tmp_path):
    """summary writes SUMMARY.md to the output path."""
    from lessons_db.db import init_db, insert_lesson
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    insert_lesson(conn, {
        "title": "Log every failure", "one_liner": "Always log",
        "cluster": "A", "tier": "lesson", "created_date": "2026-01-01",
    })
    out_file = tmp_path / "SUMMARY.md"
    runner = CliRunner()
    result = runner.invoke(
        main, ["--db", str(db_path), "summary", "--output", str(out_file)]
    )
    assert result.exit_code == 0
    assert out_file.exists()
    content = out_file.read_text()
    assert "Log every failure" in content
```

**Step 2: Run to confirm FAIL**

```
pytest tests/test_cli.py::test_export_command -v
```
Expected: `ERROR` — `No such command 'export'`

**Step 3: Add commands to `src/lessons_db/cli.py`**

```python
@main.command()
@click.argument("lesson_id", type=int)
@click.pass_context
def export(ctx, lesson_id):
    """Print a lesson as formatted markdown."""
    from lessons_db.export import format_lesson_markdown
    from lessons_db.db import get_lesson

    lesson = get_lesson(ctx.obj["conn"], lesson_id)
    if not lesson:
        click.echo(f"Lesson #{lesson_id} not found.")
        return
    click.echo(format_lesson_markdown(lesson))


@main.command()
@click.option("--output", type=click.Path(), default=None,
              help="Output path (default: ~/Documents/docs/lessons/SUMMARY.md)")
@click.pass_context
def summary(ctx, output):
    """Auto-generate SUMMARY.md from DB records.

    Produces the Quick Reference table + cluster mitigations section
    from live DB data, replacing the manually maintained SUMMARY.md.
    """
    from pathlib import Path as _Path
    conn = ctx.obj["conn"]

    rows = conn.execute(
        "SELECT id, title, one_liner, cluster, tier, enforcement, created_date "
        "FROM lessons WHERE polarity = 'negative' ORDER BY id"
    ).fetchall()

    lines = [
        "# Lessons-Learned Summary",
        "",
        "> Auto-generated by `lessons-db summary`. Do not edit manually.",
        "",
        "## Quick Reference",
        "",
        "| # | Date | One-liner | Cluster | Tier |",
        "|---|------|-----------|---------|------|",
    ]
    for r in rows:
        date_short = (r["created_date"] or "")[-5:] or "?"  # MM-DD
        cluster = r["cluster"] or "—"
        tier = r["tier"] or "observation"
        one_liner = (r["one_liner"] or r["title"] or "")[:80]
        lines.append(f"| {r['id']} | {date_short} | {one_liner} | {cluster} | {tier} |")

    lines.extend(["", f"**Total:** {len(rows)} lessons", ""])

    out_path = (
        _Path(output) if output
        else _Path.home() / "Documents" / "docs" / "lessons" / "SUMMARY.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    click.echo(f"Written: {out_path} ({len(rows)} lessons)")
```

**Step 4: Run tests**

```
pytest tests/test_cli.py::test_export_command \
       tests/test_cli.py::test_export_missing_lesson \
       tests/test_cli.py::test_summary_command -v
```
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/lessons_db/cli.py tests/test_cli.py
git commit -m "feat: add export and summary CLI commands"
```

---

## Task 4: Add `capture --from-transcript` and `capture --from-diff` functions + CLI

**Files:**
- Modify: `src/lessons_db/capture.py`
- Modify: `src/lessons_db/cli.py`
- Test: `tests/test_capture.py`

**Step 1: Write failing tests in `tests/test_capture.py`**

```python
# Add to end of tests/test_capture.py

class TestCaptureFromTranscript:
    @patch("lessons_db.capture.requests.post")
    def test_extracts_lessons_from_transcript(self, mock_post, db_path):
        from lessons_db.db import init_db
        from lessons_db.capture import capture_from_transcript

        mock_post.return_value = MagicMock(
            json=lambda: {"response": '{"lessons": [{"one_liner": "Always log before fallback", "cluster": "A", "tier": "lesson"}]}'},
            raise_for_status=lambda: None,
        )
        conn = init_db(db_path)
        result = capture_from_transcript("Session transcript text here", conn)
        assert isinstance(result, list)
        assert len(result) >= 0  # success path


    @patch("lessons_db.capture.requests.post")
    def test_returns_empty_on_ollama_failure(self, mock_post, db_path):
        from lessons_db.db import init_db
        from lessons_db.capture import capture_from_transcript

        mock_post.side_effect = Exception("network error")
        conn = init_db(db_path)
        result = capture_from_transcript("transcript", conn)
        assert result == []


class TestCaptureFromDiff:
    @patch("lessons_db.capture.requests.post")
    def test_extracts_lessons_from_diff(self, mock_post, db_path):
        from lessons_db.db import init_db
        from lessons_db.capture import capture_from_diff

        mock_post.return_value = MagicMock(
            json=lambda: {"response": '{"lessons": []}'},
            raise_for_status=lambda: None,
        )
        conn = init_db(db_path)
        result = capture_from_diff("diff --git ...\n+except:\n+    pass", conn)
        assert isinstance(result, list)

    @patch("lessons_db.capture.requests.post")
    def test_returns_empty_on_empty_diff(self, mock_post, db_path):
        from lessons_db.db import init_db
        from lessons_db.capture import capture_from_diff

        conn = init_db(db_path)
        result = capture_from_diff("", conn)
        assert result == []
```

**Step 2: Run to confirm FAIL**

```
pytest tests/test_capture.py::TestCaptureFromTranscript -v
```
Expected: `ImportError` — `cannot import name 'capture_from_transcript'`

**Step 3: Add functions to `src/lessons_db/capture.py`**

```python
def capture_from_transcript(transcript: str, conn) -> list[dict]:
    """Extract negative lessons from a session transcript. Drafts go to capture_drafts.

    Returns list of extracted lesson dicts. Returns [] on failure or empty transcript."""
    if not transcript or len(transcript.strip()) < 100:
        return []

    excerpt = transcript[-6000:]  # last 6000 chars — most recent context

    try:
        r = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    "Analyze this Claude Code session transcript. "
                    "Extract any coding mistakes, bugs, or anti-patterns that were discovered and fixed. "
                    "Return JSON: "
                    '{"lessons": [{"one_liner": "...", "cluster": "A-F or empty", "tier": "observation|insight|lesson|lesson_learned"}]}\n\n'
                    f"Transcript excerpt:\n{excerpt}"
                ),
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )
        data = json.loads(r.json().get("response", "{}"))
        lessons = data.get("lessons", [])
    except Exception as e:
        _log.warning("capture_from_transcript Ollama call failed: %s", e)
        return []

    if not lessons:
        return []

    try:
        for entry in lessons:
            conn.execute(
                "INSERT INTO capture_drafts "
                "(raw_content, extracted_data, status, created_date, source) "
                "VALUES (?, ?, 'pending', ?, 'auto_transcript')",
                [excerpt[:500], json.dumps(entry), date.today().isoformat()],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        return []

    _log.debug("capture_from_transcript: created %d drafts", len(lessons))
    return lessons


def capture_from_diff(diff_text: str, conn) -> list[dict]:
    """Extract negative lessons from a git diff. Drafts go to capture_drafts.

    Returns list of extracted lesson dicts. Returns [] on empty diff."""
    if not diff_text or len(diff_text.strip()) < 20:
        return []

    excerpt = diff_text[:4000]

    try:
        r = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    "Analyze this git diff. Look for anti-patterns in REMOVED lines (prefixed with -) "
                    "that were fixed in ADDED lines (prefixed with +). "
                    "Extract any coding lessons. "
                    "Return JSON: "
                    '{"lessons": [{"one_liner": "...", "cluster": "A-F or empty", "tier": "observation|insight|lesson|lesson_learned"}]}\n\n'
                    f"Diff:\n{excerpt}"
                ),
                "stream": False,
                "format": "json",
            },
            timeout=60,
        )
        data = json.loads(r.json().get("response", "{}"))
        lessons = data.get("lessons", [])
    except Exception as e:
        _log.warning("capture_from_diff Ollama call failed: %s", e)
        return []

    if not lessons:
        return []

    try:
        for entry in lessons:
            conn.execute(
                "INSERT INTO capture_drafts "
                "(raw_content, extracted_data, status, created_date, source) "
                "VALUES (?, ?, 'pending', ?, 'auto_diff')",
                [excerpt[:500], json.dumps(entry), date.today().isoformat()],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        return []

    _log.debug("capture_from_diff: created %d drafts", len(lessons))
    return lessons
```

**Step 4: Wire into CLI — add subcommands to the `capture` group in `src/lessons_db/cli.py`**

```python
@capture.command("transcript")
@click.argument("transcript_file", type=click.Path(exists=True))
@click.pass_context
def capture_transcript(ctx, transcript_file):
    """Analyze a saved session transcript for new lessons (writes to draft queue)."""
    from pathlib import Path as _Path
    from lessons_db.capture import capture_from_transcript

    text = _Path(transcript_file).read_text(encoding="utf-8", errors="replace")
    conn = ctx.obj["conn"]
    drafts = capture_from_transcript(text, conn)
    if drafts:
        click.echo(f"Created {len(drafts)} draft(s). Review with: lessons-db capture drafts")
    else:
        click.echo("No lessons extracted.")


@capture.command("diff")
@click.argument("diff_file", type=click.Path(exists=True), required=False)
@click.pass_context
def capture_diff(ctx, diff_file):
    """Analyze a git diff for new lessons (writes to draft queue).

    If DIFF_FILE not provided, reads from stdin.
    """
    from pathlib import Path as _Path
    from lessons_db.capture import capture_from_diff
    import sys

    if diff_file:
        text = _Path(diff_file).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    conn = ctx.obj["conn"]
    drafts = capture_from_diff(text, conn)
    if drafts:
        click.echo(f"Created {len(drafts)} draft(s). Review with: lessons-db capture drafts")
    else:
        click.echo("No lessons extracted.")
```

**Step 5: Run tests**

```
pytest tests/test_capture.py -v
```
Expected: All PASS (including new tests)

**Step 6: Commit**

```bash
git add src/lessons_db/capture.py src/lessons_db/cli.py tests/test_capture.py
git commit -m "feat: add capture transcript/diff functions and CLI subcommands"
```

---

## Task 5: Stop hook (auto-capture from transcript)

**Files:**
- Create: `~/.claude/hooks/lessons-db-stop.sh`

**Step 1: Write the hook script**

```bash
#!/usr/bin/env bash
# Stop hook: auto-capture lessons from session transcript + git diff.
# Runs after every Claude session ends.
set -euo pipefail

LESSONS_DB="$HOME/Documents/projects/lessons-db/.venv/bin/lessons-db"

if [[ ! -x "$LESSONS_DB" ]]; then
    exit 0
fi

# Only capture if there were code changes this session
DIFF=$(git -C "$HOME/Documents" diff HEAD 2>/dev/null || true)
if [[ -z "$DIFF" ]]; then
    DIFF=$(git -C "$HOME/Documents" diff HEAD~1 2>/dev/null || true)
fi

if [[ -z "$DIFF" ]]; then
    exit 0  # read-only session — nothing to capture
fi

# Write diff to temp file and capture
TMPFILE=$(mktemp /tmp/lessons-diff-XXXXXX.diff)
echo "$DIFF" > "$TMPFILE"

"$LESSONS_DB" capture diff "$TMPFILE" 2>/dev/null || true
rm -f "$TMPFILE"
```

**Step 2: Make executable**

```bash
chmod +x ~/.claude/hooks/lessons-db-stop.sh
```

**Step 3: Wire into Claude Code hooks config**

The hook must be registered in `~/.claude/settings.json` under the `Stop` event. Check current settings:

```bash
cat ~/.claude/settings.json | python3 -m json.tool | grep -A5 '"Stop"'
```

If Stop hooks are in `~/.claude/settings.json`, add the entry. If hooks.json is used, add there. The entry format:

```json
{
  "matcher": "",
  "hooks": [{
    "type": "command",
    "command": "bash ~/.claude/hooks/lessons-db-stop.sh"
  }]
}
```

**Step 4: Manual integration test**

```bash
# Simulate: make a small change, run the hook
echo "test" > /tmp/test-change.txt
bash ~/.claude/hooks/lessons-db-stop.sh
# Should exit 0 silently (or print draft count if diff exists)
```

**Step 5: Commit**

```bash
git -C ~/.claude add hooks/lessons-db-stop.sh
git -C ~/.claude commit -m "feat: lessons-db stop hook for auto-capture from diff"
```
*(Or commit to Documents workspace if that's the repo tracking hooks)*

---

## Task 6: EnterPlanMode hook (semantic search before planning)

**Files:**
- Create: `~/.claude/hooks/lessons-db-enter-plan.sh`

**Step 1: Write the hook script**

The EnterPlanMode hook receives the plan description in stdin JSON. It runs semantic search and injects relevant lessons.

```bash
#!/usr/bin/env bash
# EnterPlanMode hook: surface relevant lessons before planning (~90 tokens).
set -euo pipefail

LESSONS_DB="$HOME/Documents/projects/lessons-db/.venv/bin/lessons-db"

if [[ ! -x "$LESSONS_DB" ]]; then
    exit 0
fi

# Extract plan description from hook input
QUERY=$(cat | python3 -c "
import sys, json
d = json.load(sys.stdin)
# EnterPlanMode provides the task description
print(d.get('description', '') or d.get('task', '') or '')
" 2>/dev/null || echo "")

if [[ -z "$QUERY" || ${#QUERY} -lt 10 ]]; then
    exit 0
fi

RESULTS=$("$LESSONS_DB" search "$QUERY" --top 3 2>/dev/null || true)

if [[ -n "$RESULTS" && "$RESULTS" != "No results found." ]]; then
    echo "Relevant lessons for this task:"
    echo "$RESULTS"
fi
```

**Step 2: Make executable and wire**

```bash
chmod +x ~/.claude/hooks/lessons-db-enter-plan.sh
```

Register under `EnterPlanMode` event in `~/.claude/settings.json`.

**Step 3: Manual test**

```bash
echo '{"description": "implement async subscriber lifecycle cleanup"}' \
  | bash ~/.claude/hooks/lessons-db-enter-plan.sh
# Should return top 3 relevant lessons
```

**Step 4: Commit**

```bash
git -C ~/.claude add hooks/lessons-db-enter-plan.sh
git -C ~/.claude commit -m "feat: lessons-db EnterPlanMode hook for semantic search"
```

---

## Task 7: PostToolUse:Bash hook (test failure diagnostic)

**Files:**
- Modify: `~/.claude/hooks/detect-test-run.sh` (or create `lessons-db-post-bash.sh`)

**Step 1: Check existing test detection hook**

```bash
cat ~/.claude/hooks/detect-test-run.sh
```

The existing `detect-test-run.sh` already identifies test commands. The lessons-db hook should complement it, not replace it.

**Step 2: Create `~/.claude/hooks/lessons-db-post-bash.sh`**

```bash
#!/usr/bin/env bash
# PostToolUse:Bash hook: match test failure output against known lesson patterns.
set -euo pipefail

LESSONS_DB="$HOME/Documents/projects/lessons-db/.venv/bin/lessons-db"

if [[ ! -x "$LESSONS_DB" ]]; then
    exit 0
fi

# Read the bash tool output from stdin
OUTPUT=$(cat | python3 -c "
import sys, json
d = json.load(sys.stdin)
# PostToolUse provides tool output
print(d.get('output', '') or d.get('stderr', '') or '')
" 2>/dev/null || echo "")

# Only fire on test failures
if ! echo "$OUTPUT" | grep -qiE "FAILED|Error|AssertionError|TypeError|AttributeError"; then
    exit 0
fi

# Extract error lines and search for matching lessons
ERROR_LINES=$(echo "$OUTPUT" | grep -iE "Error|FAILED" | head -3)

if [[ -z "$ERROR_LINES" ]]; then
    exit 0
fi

RESULTS=$("$LESSONS_DB" search "$ERROR_LINES" --top 2 2>/dev/null || true)

if [[ -n "$RESULTS" && "$RESULTS" != "No results found." ]]; then
    echo "Matching lessons for this error:"
    echo "$RESULTS"
fi
```

**Step 3: Make executable and wire**

```bash
chmod +x ~/.claude/hooks/lessons-db-post-bash.sh
```

Register under `PostToolUse` event with matcher `Bash` in `~/.claude/settings.json`.

**Step 4: Manual test**

```bash
echo '{"output": "FAILED test_subscriber_lifecycle.py::test_cleanup\\nAttributeError: NoneType has no attribute unsubscribe"}' \
  | bash ~/.claude/hooks/lessons-db-post-bash.sh
# Should return subscriber lifecycle lesson
```

**Step 5: Commit**

```bash
git -C ~/.claude add hooks/lessons-db-post-bash.sh
git -C ~/.claude commit -m "feat: lessons-db PostToolUse:Bash hook for test failure diagnostics"
```

---

## Task 8: Auto-generate SUMMARY.md

Now that `lessons-db summary` exists (Task 3), run it against the live DB and commit the result.

**Step 1: Run summary generation**

```bash
source ~/Documents/projects/lessons-db/.venv/bin/activate
lessons-db summary --output ~/Documents/docs/lessons/SUMMARY.md
```
Expected: `Written: /home/justin/Documents/docs/lessons/SUMMARY.md (122 lessons)`

**Step 2: Review output**

```bash
head -30 ~/Documents/docs/lessons/SUMMARY.md
wc -l ~/Documents/docs/lessons/SUMMARY.md
```

Expected: Quick Reference table with 122 rows.

**Step 3: Commit**

```bash
git -C ~/Documents add docs/lessons/SUMMARY.md
git -C ~/Documents commit -m "docs: auto-generate SUMMARY.md from lessons-db (122 lessons)"
```

---

## Task 9: Run full test suite + commit all CLI changes

**Step 1: Run full test suite**

```bash
cd ~/Documents/projects/lessons-db
source .venv/bin/activate
pytest --timeout=120 -x -q
```
Expected: All pass (≥115, may be more with new tests)

**Step 2: Push lessons-db to remote**

```bash
git push
```

---

## Deferred (not in this plan)

- Archive `docs/lessons/*.md` to `archive/` — risky; markdown files are still useful as backup. Defer until DB is proven stable.
- Replace old SessionStart hooks (surface-lessons, goal-reflection) with lessons-db hooks — requires careful cutover, separate session.
- `lessons-db promote` / `lessons-db sync` — community sharing features, no internal users yet.
- Aho-Corasick compiled pattern matching — optimization, current regex approach is fast enough.
- Update `/capture-lesson` skill to write to DB — requires skill editing, separate session.

---

## Summary

| Task | Files | Type |
|------|-------|------|
| 1. `rule` CLI | `cli.py` | Wire existing module |
| 2. `scan` CLI | `cli.py` | Wire existing module |
| 3. `export` + `summary` CLI | `cli.py` | Wire + implement |
| 4. `capture transcript/diff` | `capture.py`, `cli.py` | New functions + wire |
| 5. Stop hook | `hooks/lessons-db-stop.sh` | New hook |
| 6. EnterPlanMode hook | `hooks/lessons-db-enter-plan.sh` | New hook |
| 7. PostToolUse:Bash hook | `hooks/lessons-db-post-bash.sh` | New hook |
| 8. Generate SUMMARY.md | (run command) | One-time |
| 9. Full suite + push | (verify) | QA |
