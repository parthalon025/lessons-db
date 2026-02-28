# Knowledge Capture System — Brainstorming Context

**Date:** 2026-02-26
**Status:** Brainstorming started, explore phase complete, questions not yet asked
**Resume with:** `/brainstorming` — extend lessons-db into broader knowledge capture

## User Intent

Extend the lessons-db concept beyond anti-patterns/mistakes into a **general knowledge capture system** that works across all `~/Documents/projects/` repos. Captures:

- **Positive insights** — what worked well, effective patterns
- **Planning innovations** — novel approaches to planning that should be remembered
- **Decision patterns** — frameworks and decision methods that proved effective
- **Value multipliers** — techniques with outsized ROI
- **Workflow discoveries** — process improvements across all projects

## Current Foundation (lessons-db)

- 122 lessons migrated into SQLite, 296 corrective actions
- OIL taxonomy: observation → insight → lesson → lesson_learned
- 6 clusters (A-F): Silent Failures, Integration Boundary, Cold-Start, Spec Drift, Context & Retrieval, Planning & Control Flow
- 10 categories: data-model, registration, cold-start, integration, deployment, monitoring, ui, testing, performance, security
- Scope tags: `domain:X`, `language:X`, `framework:X`, `universal`
- Enforcement escalation: documentation → semgrep_warning → semgrep_error → semgrep_autofix
- Design doc: `docs/plans/2026-02-26-lessons-db-design.md`

## Key Design Questions (not yet asked)

1. **Taxonomy extension** — Does the OIL tier model (observation → insight → lesson → lesson_learned) work for positive knowledge? Or does positive knowledge need different maturity stages (e.g., noticed → tested → proven → standard)?
2. **Storage** — Same DB with a `polarity` field (negative/positive), or separate tables/DB?
3. **Surfacing** — How should positive insights surface? Same hooks? Different triggers? (e.g., when planning, surface "approaches that worked for similar problems")
4. **Categories** — The current 10 categories are failure-mode-oriented. What categories describe positive knowledge? (architecture-pattern, planning-technique, workflow-optimization, value-multiplier, etc.)
5. **Enforcement analogy** — Lessons escalate enforcement on recurrence. What's the positive analog? (template → recommended-practice → standard → automated?)
6. **Cross-project scope** — Current scope tags are per-project. Broader knowledge may be domain-agnostic. How to handle?
7. **Capture trigger** — Lessons capture from bugs/test failures. What triggers positive capture? End of successful project? After a planning session that went well? Manual only?

## Approaches to Explore

1. **Extend lessons-db** — Add `entry_type` column (lesson/insight/pattern/innovation), reuse all infrastructure
2. **Separate knowledge-db** — New project, shares LanceDB for cross-search, different schema for positive knowledge
3. **Unified knowledge graph** — Single system where lessons (negative) and insights (positive) are nodes with relationships

## Pipeline Audit Findings (from this session)

- Search has no SQLite LIKE fallback — filed as GitHub issue #1
- `affected_files` and `detection_patterns` tables are empty after migration
- LanceDB embeddings not yet generated
- All 52 tests pass, 122 lessons migrated
