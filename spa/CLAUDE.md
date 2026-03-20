<!-- superhot-ui:0.4.0 -->
## superhot-ui Design System

**Current integration: CSS tokens + facility state. Narrator, audio, and Preact components not yet adopted.**

This project consumes superhot-ui for all visual effects and UI components.
**Read these docs before implementing any UI in this project.**

### Reference Docs

- Integration: `node_modules/superhot-ui/docs/consumer-guide.md`
- Design philosophy: `node_modules/superhot-ui/docs/design-philosophy.md`
- Experience design: `node_modules/superhot-ui/docs/experience-design.md`
- Component docs: `node_modules/superhot-ui/docs/components/`

### Palette

Six palette colors — four original SUPERHOT, two Portal:

| Token                  | Role                                       |
| ---------------------- | ------------------------------------------ |
| `--sh-bright` (white)  | Primary text, active elements              |
| `--sh-void` (black)    | Backgrounds, negative space                |
| `--sh-threat` (red)    | Danger, failure, critical state            |
| `--sh-phosphor` (cyan) | Health, connection, interactive elements   |
| `--sh-portal-blue`     | Portal connection lines, inactive antlines |
| `--sh-portal-orange`   | Active antlines, flow indicators           |

Facility state shifts Portal tokens — in `breach` mode, all six converge toward threat red.

### Key Rules

**Typography — terminal voice, not web UI**

- Labels: UPPERCASE, `--tracking-widest`, `--text-muted`
- Values: numeric or terse — never prose
- Status text: lowercase monospace codes (`healthy`, `error`, `warning`)
- System messages: UPPERCASE authoritative (`SYSTEM DEGRADED` not "Some services are down")
- Empty states: never apologetic (`NO ACTIVE JOBS` not "No jobs found")

**Navigation — places, not functions**

- `TOPOLOGY` not "Graph", `SYSTEMS` not "Services", `QUEUE` not "Job Queue"
- Active route: phosphor left border — never filled highlight or background
- No hover tooltips — if a label needs one, it's too long

**Facility state — the atmosphere master switch**

- `setFacilityState('normal')` — Portal blue, calm (all systems healthy)
- `setFacilityState('alert')` — SUPERHOT red bleeds into Portal blue (degraded)
- `setFacilityState('breach')` — Full threat mode, all Portal tokens become red (critical)
- Wire to health detection: breach when critical, alert when degraded, normal when healthy
- One call on `<html>` — every component inherits the shift via CSS custom properties
- This is the CSS atmosphere layer. The JS escalation APIs (`orchestrateEscalation`, `EscalationTimer`) animate specific surfaces over time. Use both together.

**Interface health states — 7 coordinated surfaces**

1. Facility state — `setFacilityState()` shifts all Portal colors globally (the umbrella)
2. Graph node — `ShThreatPulse persistent`, color shifts to `--sh-threat`
3. Stat card — `status="error"` + `ShThreatPulse`
4. Sidebar — indicator dot pulses threat red
5. Incident HUD — `ShIncidentHUD` banner with severity + message
6. Toast — `ShToast type="error" duration={0}` (persistent)
7. Layout root (critical only) — `ShMantra` watermarks entire interface

- Rule: health state changes the whole interface simultaneously — never card by card

**Time-freeze discipline — data age is visible**

- Every data surface that can go stale wraps in `<ShFrozen timestamp={...}>` — no exceptions
- Fresh: 0–5min (full color), Cooling: 5–30min (desaturated), Frozen: 30–60min (grey), Stale: 60+min (ghost + NO DATA mantra)
- Header "last updated": `applyFreshness(el, timestamp)`

**Narrator — personality-driven UI text**

- Set once at init: `ShNarrator.personality = 'glados'` + `ShAudio.narratorPersonality = 'glados'` (keep in sync)
- `narrate('error')` for personality-appropriate phrases, `narrate('loading')` for loading text, etc.
- 5 personalities: `glados`, `cave`, `wheatley`, `turret`, `superhot`
- 12 categories: `toast`, `error`, `loading`, `status`, `empty`, `greeting`, `farewell`, `countdown`, `warning`, `destructive`, `success`, `search`
- Narrator does NOT auto-change based on facility state — switch personality explicitly if needed
- Use `ShAnnouncement` component for typed-out narrative banners with personality styling
- Personality is context, not preference: GLaDOS for main monitoring, Cave for onboarding/about, Wheatley for user-facing recovery, Turret for terse alerts, SUPERHOT for breach mantras
- During `breach`, use only GLaDOS (coldest), Turret, or SUPERHOT — other personalities fight the severity

**Portal components**

- `ShAntline` — animated connecting line between elements. `active` toggles orange/blue. Colors shift with facility state.
- `ShTestChamber` — panel assembly container. Children with `.sh-panel` class animate in staggered. `compromised={true}` for failure state.
- `ShAnnouncement` — narrative banner with typing effect. Set `personality` + `message` + optional `source` label.

**Emotional arc — every session follows this loop**
Tension → Pause → Plan → Execute → Catharsis

- Page load: `ShTestChamber` panels assemble + `ShSkeleton` everywhere + `ShAnnouncement` greeting (Tension → Pause)
- Data stale: `ShFrozen` activates, mantra appears (Tension)
- Service failure: `setFacilityState('alert'|'breach')` + escalation + narrator error phrases (Tension)
- Command palette opens: glitch burst — "I'm taking action" (Plan)
- Recovery: `setFacilityState('normal')` + `recoverySequence()` + narrator success phrase (Catharsis)
- Job complete: `playSfx('complete')` (personality-remapped) if audio enabled (Catharsis)

**Interaction feedback — no silent actions**

| Action               | Response                                                            |
| -------------------- | ------------------------------------------------------------------- |
| Open command palette | `ShCommandPalette` opens with glitch burst                          |
| Dismiss alert        | `shatterElement` on the element                                     |
| Job submitted        | `ShToast type="info"` + `playSfx('complete')`                       |
| Job failed           | `ShToast type="error" duration={0}` + `playSfx('error')`            |
| Navigate route       | `@starting-style` entrance on incoming content                      |
| Data refreshed       | `ShGlitch` micro-burst on "last updated" timestamp                  |
| Service recovers     | `setFacilityState('normal')` + `ShGlitch` burst, then phosphor calm |

**Audio — all SFX through one system**

- Original: `complete`, `error`, `dlq`, `pause`
- Portal: `portal-chime`, `portal-fail`, `turret-deploy`, `turret-shutdown`, `facility-hum`, `panel-slide`
- Tension: `setTensionDrone(level)` / `stopTensionDrone()` (auto-wired into `orchestrateEscalation`)
- Personality remapping: `complete` → `portal-chime` (GLaDOS/Cave/Wheatley) or `turret-deploy` (Turret)
- `ShAudio.enabled` defaults to `false` — always set from user preference, never automatically

**Component rules**

- Every effect communicates exactly one signal — not "status plus action"
- New UI patterns go in superhot-ui first — never build a DS component directly in this project
- Components are diegetic — no decorative animation (no bounce, wiggle, idle flip)
- Layout components (grids, sidebars, nav) live in the consuming project, not superhot-ui

### Gotchas

- Never use `h` or `Fragment` as callback param names in JSX — esbuild injects `h` as JSX factory, shadowing it causes silent render crashes
- `shatterElement` requires `position: relative` on the parent element
- `--sh-fill` in `.sh-vram-bar` is a unitless number 0–100, not a percentage string (`72` not `"72%"`)
- `@import 'superhot-ui/css'` fails silently inside inline `<style>` blocks (no base URL) — inject as separate `<style>` elements
- esbuild + `file:` dep: pin all Preact imports in `esbuild.config.mjs` to avoid dual-Preact crash (see consumer-guide.md)
- `ShAnnouncement` respects `prefers-reduced-motion` — shows full text immediately, no typing effect
- `ShTestChamber` plays `panel-slide` SFX on mount — only audible when `ShAudio.enabled = true`
- `setFacilityState()` is a no-op during SSR (checks `typeof document`) — safe to call unconditionally
- `ShNarrator.personality` and `ShAudio.narratorPersonality` must be set together — mismatched values cause narrator text and SFX to use different personalities
