# Draft Triage Pipeline Design

**Date:** 2026-02-27
**Status:** Approved
**Problem:** 1,906 pending drafts, 0 promoted, 0 surfacing events — system captures but never improves

---

## Problem Statement

The lessons-db capture pipeline generates drafts aggressively but has no automated path from draft → enforced lesson. The human review step (`capture approve`) was never operationalized. Root causes:

1. No automated triage — every draft requires manual review
2. `promote_draft` hardcodes `polarity: "positive"` — negative lessons (bugs/anti-patterns) are miscategorized on promotion
3. After promotion, lessons stay at `enforcement: documentation` — no detection patterns, no Semgrep rules
4. `score_one_liner()` exists but is never called outside tests
5. No habit-forcing trigger to remind about triage backlog

---

## Design: Claude-Reviewed Automated Pipeline

### Architecture

```
[Stop hook]
  capture_from_diff() → raw drafts in queue

[Nightly, 03:30 — extends lessons-db-nightly.service]
  Phase 1 — Filter (regex, no LLM, <100ms)
    Deduplicate: Jaccard similarity > 0.85 on one_liner
      vs existing lessons + other pending drafts → auto-dismiss
    Noise filter: regex patterns → auto-dismiss
      ("no.*mistake", "no.*bug", "no coding", "repeated content", len < 20)

  Phase 2 — Claude Review (claude-haiku-4-5, batches of 20)
    Input per batch:
      - Draft IDs + one_liners
      - Existing lesson titles (duplicate guard)
      - Evaluation criteria
    Output per draft (structured JSON):
      - verdict: PROMOTE | DISMISS
      - reason: one sentence
      - improved_one_liner: cleaned-up wording
      - detection_pattern: regex string for pre-edit hook matching
      - semgrep_rule: YAML rule text (best effort, may be empty string)

  Phase 3 — Execute verdicts
    PROMOTE:
      insert_lesson() — polarity inferred from draft source
        auto_transcript → negative, auto_transcript_positive → positive
      insert detection_pattern into detection_patterns table
      write semgrep_rule to rules/ if non-empty
      log decision to triage JSONL
    DISMISS:
      status = 'dismissed', store Claude's reason
      log decision to triage JSONL

  Write: ~/.local/share/lessons-db/triage-YYYY-MM-DD.jsonl

[Session Start hook]
  Surface: "N lessons auto-promoted overnight" when N > 0

[Audit — on demand]
  lessons-db capture triage --review-log [--date DATE]
    Read-only: shows promoted + dismissed with Claude's reasons
  lessons-db capture triage --override <id>
    Human correction: re-promote a dismissed draft or remove a bad promotion
```

---

## New Components

| Component | File | Description |
|-----------|------|-------------|
| `capture review` CLI | `cli.py` | Runs Phase 1-3; callable manually or from nightly |
| `capture triage` CLI | `cli.py` | `--review-log` audit, `--override` correction |
| `promote_draft` fix | `capture.py` | Infer polarity from draft source; fix hardcoded values |
| Claude reviewer | `review.py` (new) | Batch Claude haiku calls, structured JSON output |
| Noise filter | `review.py` | Jaccard dedup + regex noise patterns |
| Detection pattern insert | `db.py` | Insert Claude's regex into `detection_patterns` table |
| Semgrep rule writer | `rulegen.py` | Write Claude's YAML to `rules/` directory |
| Nightly service update | `lessons-db-nightly.service` | Add `lessons-db capture review` |
| SessionStart hook update | `lessons-db-session-start.sh` | Show overnight promotion count |
| Triage JSONL log | `~/.local/share/lessons-db/` | `triage-YYYY-MM-DD.jsonl` audit trail |

---

## Claude Reviewer Design

### Model
`claude-haiku-4-5-20251001` — sufficient for structured JSON evaluation, cost-effective for batch use (~$0.01 per 200 drafts).

### Prompt structure (per batch of 20)

```
You are reviewing draft lessons from a coding lessons-learned system.
For each draft, decide: PROMOTE (real, actionable, specific anti-pattern worth enforcing)
or DISMISS (vague, trivial, already covered, or noise).

Existing lessons (do not promote duplicates):
<titles of 122 existing lessons>

Drafts to review:
[1] <one_liner>
[2] <one_liner>
...

Criteria for PROMOTE:
- Specific: names a concrete pattern, not a general principle
- Actionable: clear do/don't that a developer can follow
- Prevents recurrence: would catch this bug/mistake if checked automatically
- Novel: not already in the existing lessons list

Return JSON:
{
  "reviews": [
    {
      "id": <draft_id>,
      "verdict": "PROMOTE" | "DISMISS",
      "reason": "<one sentence>",
      "improved_one_liner": "<cleaned wording if PROMOTE, else empty>",
      "detection_pattern": "<regex string for code matching if PROMOTE, else empty>",
      "semgrep_rule": "<YAML rule text if PROMOTE and pattern is syntactic, else empty>"
    }
  ]
}
```

### Polarity inference from source

| Draft source | Polarity on promotion |
|---|---|
| `auto_transcript` | `negative` |
| `auto_transcript_positive` | `positive` |
| `auto_diff` | `negative` |

---

## Backfill Strategy

The existing 1,906 pending drafts have `confidence = NULL`. Initial run:

```bash
lessons-db capture review --backfill
```

Processes all pending drafts through the full Phase 1-3 pipeline. Expected outcome:
- ~90% dismissed (duplicates, noise, "no mistakes found" entries)
- ~100-200 survivors after dedup/noise filter
- ~50-100 promoted after Claude review

Subsequent nightly runs process only drafts created since last run timestamp.

---

## Dependencies

- `anthropic` Python SDK — add to `pyproject.toml` dependencies
- `ANTHROPIC_API_KEY` — sourced from `~/.env` (present for Claude Code)
- No new infrastructure — reuses existing nightly timer, detection_patterns table, rules/ directory

---

## Success Criteria

1. `lessons-db status` shows > 0 detection patterns after first nightly run
2. Pre-edit hook surfaces at least one lesson match per week (surfacing_events > 0)
3. Triage JSONL log written after each nightly run
4. `lessons-db capture triage --review-log` shows auditable decision trail
5. Backfill clears the 1,906 pending backlog in one run

---

## Out of Scope

- `learn record` wiring (surfacing event tracking) — separate task
- MAB feedback loop on promoted lessons — future phase
- Cross-project pattern detection integration — already designed separately
