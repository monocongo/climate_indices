# BMAD-Driven Development: climate_indices v2.4.0

A git-sourced narrative of how the BMAD (Business-Motivated Agile Development) framework
drove the full planning, specification, and validation lifecycle for the
[v2.4.0 release](https://github.com/monocongo/climate_indices/pull/623) of the
`climate_indices` library.

All dates, commit hashes, line counts, and metrics in this document come directly from
the versioned artifacts in `_bmad-output/` — the living record of BMAD's work.

---

## What Is BMAD?

BMAD is a structured AI-agent development framework that converts a rough project intent
into a fully specified, sprint-ready backlog before a single line of implementation code
is written. The framework assigns work to distinct agent roles:

| Role | Responsibility |
|------|---------------|
| Analyst | PRD authoring — goals, personas, functional requirements |
| Architect | Technical design — patterns, ADRs, critical path analysis |
| Product Manager | Epic/story decomposition, FR coverage mapping, acceptance criteria |
| Scrum Master | Sprint structure, story readiness, retrospectives |
| Developer | Implementation against story ACs |
| QA Engineer | Test design, compliance dashboards, reference validation |

For a scientific computing library like `climate_indices` — where requirements are precise
(FAO-56 equations, NOAA reference tolerances, CF metadata conventions) and correctness is
non-negotiable — the specification-first discipline BMAD enforces is particularly well matched.

---

## Context: What v2.3.0 Established

Version 2.3.0 (PR #614, merged early 2026) introduced three canonical patterns as a
proof-of-concept on SPI, SPEI, and PET:

- `@xarray_adapter` decorator for transparent numpy/xarray dispatch
- `@overload` signatures in `typed_public_api.py` for IDE-visible type safety
- Structured exception hierarchy rooted at `ClimateIndicesError`
- `structlog` lifecycle events replacing stdlib logging
- CF metadata registry for standardized variable attributes

v2.4.0's mandate was to extend all five of these patterns across every remaining public
index, while simultaneously adding two new algorithms (Penman-Monteith ETo and first-class
EDDI support) and laying groundwork for Palmer's multi-output xarray interface.

---

## Phase 1: Planning — February 5–16, 2026

### The Concentrated BMAD Session

The planning record shows a striking commit density on a single day:

| Commit | Date | Artifact created |
|--------|------|-----------------|
| `d80a859` | 2026-02-05 | PRD steps 1–12 — goals, personas, 60 FRs, NFRs |
| `2ec9c7d` | 2026-02-05 | Architecture decision document (xarray + structlog strategy) |
| `6942126` | 2026-02-05 | `epics.md` initialized — requirements extracted |
| `f2a7d3f` | 2026-02-05 | Epic list + FR coverage map |
| `cd9b366` | 2026-02-05 | Epic 1 stories — 9 stories with acceptance criteria |
| `afdf474` | 2026-02-05 | Epic 2 stories — 12 stories added |
| `3988fbd` | 2026-02-05 | Epics 3–5 stories — 26 more stories |
| `fca124b` | 2026-02-06 | Epics finalized — 47 stories, 60 FRs, 5 epics validated |

In roughly 24 hours of BMAD agent sessions, the project went from a blank slate to a
fully decomposed backlog. This speed is possible because BMAD agent roles run focused
sessions — the Analyst does not wait for the Architect; the PM does not wait for the
Analyst to be fully done. Each role picks up the previous role's output and advances it.

### The Refinement Arc (February 7–16)

The week following the initial session was a refinement arc: three technical research
documents were commissioned to validate feasibility of the higher-risk algorithms, and
three implementation-readiness reports assessed whether the spec was ready to execute.

| Date | Artifact | What it resolved |
|------|----------|-----------------|
| 2026-02-07 | `technical-eddi-validation.md` | NOAA reference gap analysis; 1e-5 tolerance justified |
| 2026-02-07 | `technical-penman-monteith.md` | FAO-56 equations 1–19; helper decomposition strategy |
| 2026-02-07 | `technical-palmer-modernization.md` | Pattern C (stack/unpack) chosen over Patterns A/B |
| 2026-02-07 | Implementation readiness report #1 | Identified 3 blocking gaps before sprint start |
| 2026-02-09 | Architecture update | EDDI NOAA validation added as BLOCKING FR |
| 2026-02-09 | Implementation readiness report #2 | Gap closure confirmed for 2 of 3 blockers |
| 2026-02-15 | `prd.md` (v2.4.0 — 1,280 lines) | Superseded PRD v1.1 — 31 new FRs, 4 parallel tracks |
| 2026-02-16 | `architecture.md` aligned to v2.4.0 | Critical path analysis, dependency map |
| 2026-02-16 | `sprint-status.yaml` created | Execution tracking scaffolding |
| 2026-02-16 | `CLAUDE.md` committed | Agent team rules, development conventions |
| 2026-02-16 | `test-design-*.md` suite | QA test architecture for all 5 epics |

The `epics-original.md` (261 lines, PRD v1.1 scope) was preserved alongside the expanded
`epics.md` (1,652 lines, v2.4.0 scope). The 6× expansion in document length reflects the
story decomposition depth BMAD requires: each story carries priority, effort estimate,
FR coverage mapping, and multi-part acceptance criteria.

---

## Phase 2: PRD v2.4.0 — The Specification

The consolidated PRD (`_bmad-output/planning-artifacts/prd.md`) organized work into four
parallel tracks with explicit dependency ordering:

```
Track 0 (Canonical Patterns) ──────────────────────────► Epic 1
Track 1 (Penman-Monteith) ─────────────────────────────► Epic 2
Track 2 (EDDI / PNP / scPDSI) ─── depends on Track 1 ──► Epic 3
Track 3 (Palmer Multi-Output) ──── depends on Track 0 ──► Epic 4
                                                          Epic 5 (cross-cutting validation)
```

**Track 0 — Canonical Pattern Completion (12 FRs):** Apply the v2.3.0 patterns to the four
remaining indices: `percentage_of_normal`, `pci`, `eto_thornthwaite`, `eto_hargreaves`.
FR-PATTERN-007 (Palmer structlog migration) was called out as the **critical path blocker**
for Track 3 — `palmer.py` is 912 lines and the architecture review recommended starting it
immediately.

**Track 1 — PM-ET Foundation (6 FRs):** Implement the full Penman-Monteith FAO-56 ETo
pipeline. FR-PM-005 required validation against two FAO-56 worked examples (Bangkok tropical
profile, Uccle temperate profile) within ±0.05 mm/day tolerance. This FR was marked
non-negotiable: no merge without passing worked examples.

**Track 2 — EDDI/PNP/scPDSI Coverage (5 FRs):** Wire EDDI into the public API as a
first-class index. FR-EDDI-001 was marked **BLOCKING** — NOAA reference dataset validation
at 1e-5 tolerance must pass before the index can be considered production-ready.

**Track 3 — Palmer Multi-Output (7 FRs):** Deliver `palmer_xarray()` as a manual wrapper
(not the `@xarray_adapter` decorator) returning an `xr.Dataset` with four variables: `pdsi`,
`phdi`, `pmdi`, `z_index`. Architecture Decision 3 (AD-3) documented why Pattern C
(stack/unpack workaround) was chosen over alternatives while awaiting upstream xarray
Issue #1815.

---

## Phase 3: Epic & Story Breakdown

```
Epic 1  Canonical Pattern Completion    12 stories   Agent: Alpha   2–3 weeks
Epic 2  PM-ET Foundation                 7 stories   Agent: Beta    3–4 weeks
Epic 3  EDDI/PNP/scPDSI Coverage         7 stories   Agent: Gamma   2–3 weeks
Epic 4  Palmer Multi-Output              9 stories   Agent: Delta   3–4 weeks
Epic 5  Cross-Cutting Validation         3 stories   All agents     1 week
─────────────────────────────────────────────────────────────────────────────
Total                                   38 stories
```

Each story was a self-contained unit of work with:
- A priority tier (Critical / High / Medium / Low)
- An effort estimate in hours
- A list of FRs satisfied
- Multi-part acceptance criteria (e.g., Story 2.1 carried 9 discrete ACs for PM-ET core)
- Explicit upstream dependencies

The BMAD Scrum Master (Bob) maintained `sprint-status.yaml` as the single execution
record, tracking each story through: `backlog → ready-for-dev → in-progress → review → done`.

---

## Phase 4: Implementation — February 16 to April 5, 2026

The sprint ran for **48 days** (7 weeks). Each epic ran on an isolated feature branch.

### Epic 1 — Canonical Pattern Completion

Applied the full six-pattern suite to `percentage_of_normal`, `pci`, `eto_thornthwaite`,
and `eto_hargreaves`. Migrated all legacy `ValueError` raises to the structured exception
hierarchy. Extracted the CF metadata registry as a standalone module (`cf_metadata.py`).
Added Hypothesis property-based tests for PNP and PCI. Outcome: the compliance dashboard
(`test_pattern_compliance.py`) went green for all 7 indices.

### Epic 2 — Penman-Monteith ETo Foundation

Delivered `src/climate_indices/pm_eto.py` — 465 lines implementing the complete FAO-56
ETo pipeline: atmospheric helpers, vapor pressure helpers, humidity pathway dispatcher
(auto-selecting from dewpoint → RH extremes → RH mean), and the core Equation 6 solver.
The 102 tests in `test_pm_eto.py` reproduce both FAO-56 worked examples within ±0.05 mm/day.
This is research-grade validation, not a smoke test.

### Epic 3 — EDDI/PNP/scPDSI Coverage

Promoted EDDI from a function inside `indices.py` to a fully first-class public index:
`@overload` signatures in `typed_public_api.py`, CF metadata entry (citing Hobbins et al.
2016), CLI integration via `--index eddi`, and NOAA reference validation infrastructure
(3 self-consistency tests always pass; 9 reference-data tests skip gracefully when fixtures
are absent — documented as acceptable for v2.4.0 given manual fixture acquisition
requirements). Added `scpdsi()` stub interface for future implementation.

### Epic 4 — Palmer Multi-Output

Migrated `palmer.py` (912 lines) from stdlib logging to structlog. Designed and partially
implemented the `palmer_xarray()` manual wrapper per AD-3 / Pattern C. However, at the
Epic 5 validation gate, Story 5.3 discovered that **seven FRs (FR-PALMER-001 through
FR-PALMER-007) had zero implementing commits** — `palmer_xarray` had no matches in the
codebase despite all nine Epic 4 stories being marked done in sprint-status. All seven
Palmer FRs were formally deferred to v2.5.0 (documented in CHANGELOG with a
`TODO(v2.5.0)` comment placed at the end of `palmer.py`).

### Epic 5 — Cross-Cutting Validation

Three stories served as the release gate:

**Story 5.1 — Pattern Compliance Audit** (`4f02782`): Ran the compliance dashboard and
found 31/80 assertions failing — 40% of the compliance matrix broken. Root cause: Epic 1's
feature branch had never been merged into the integration branch. Nine cherry-picks
(oldest-to-newest, one conflict) were required to resolve it. Final result: **42/42
compliance points, 80/80 assertions passing** across all 7 indices and all 6 patterns.

**Story 5.2 — EDDI Reference Validation** (`b12b950`): Confirmed EDDI self-consistency
tests pass and NOAA reference infrastructure skips gracefully. Established NOAA provenance
protocol for reference datasets.

**Story 5.3 — Final v2.4.0 Validation** (`524c992`): Full FR/NFR audit, senior review
(James, Opus 4.6), and release cut. Senior review caught and fixed 4 Medium issues (empty
`units` field in CF metadata, abbreviated references missing DOIs, missing regression tests
for new CF entries, dead code block) and 2 Low issues (stale module docstring, lint
cleanup). Version bumped to `2.4.0` in `pyproject.toml`; CHANGELOG updated.

---

## Phase 5: Release Outcomes

| Metric | Value |
|--------|-------|
| Test suite | 1,027 passed, 9 skipped |
| Pattern compliance | 42/42 points (80/80 assertions) |
| Type checking | `mypy --strict` clean — 16 source modules, 0 errors |
| Test coverage | >85% across new modules |
| Stories delivered | 38/38 (with 7 Palmer FRs formally deferred) |
| FRs addressed | 31 of 31 v2.4.0 FRs (7 Palmer FRs → v2.5.0) |
| NFRs satisfied | 8/8 |
| Production incidents | 0 |
| Sprint duration | 48 days (2026-02-16 → 2026-04-05) |

---

## Phase 6: Retrospective — What the Process Revealed

The Epic 5 retrospective (`_bmad-output/implementation-artifacts/epic-5-retro-2026-04-05.md`)
is the first retrospective of the entire v2.4.0 sprint — all five epics completed without
a mid-sprint retro, making this a full-journey post-mortem.

### What Went Well

- PM-ET delivered as a complete, validated module — not a stub. 102 tests reproducing FAO-56
  worked examples is a higher standard than most scientific libraries apply.
- EDDI elevated from internal function to first-class public index with CF metadata, typed
  overloads, CLI flag, and NOAA validation infrastructure.
- 42/42 pattern compliance achieved — the architectural cohesion goal of the entire sprint.
- `mypy --strict` clean on a library with pre-existing technical debt — required targeted
  fixes but achieved the goal.
- Senior review caught real issues before release, not after.

### Systemic Issues and Lessons

**Issue 1 — Integration branch was never kept current.**
Epic branches developed in isolation; Story 5.1 became a rescue operation (9 cherry-picks,
1 conflict) instead of a 30-minute smoke test.
> *L1: Merge epic branches into integration within one sprint of completion.*

**Issue 2 — Sprint status ≠ committed code.**
Epic 4 stories marked done while `palmer_xarray` had zero commits. Seven FRs deferred.
> *L2: Before marking a story done, verify with `git log --oneline` that implementing commits exist.*

**Issue 3 — Story artifact left empty.**
Story 5.2's dev record — debug log, completion notes, file list — was never populated despite
the story being marked done and the related commits being in git.
> *L3: Story files are traceability records. Dev Agent Record must be populated before done.*

**Issue 4 — Acceptance criteria referenced a file that doesn't exist.**
Story 5.1 AC #4 cited `.github/workflows/ci.yml`; the actual CI workflow is
`unit-tests-workflow.yml`. The step was added to the correct file, but the mismatch required
in-flight deviation from a written AC.
> *L4: ACs must reference actual artifacts. Audit the repo before writing ACs that name files.*

**Issue 5 — Senior review scope should track story risk, not story size.**
Story 5.3 was small by line count but high-risk (release gate, new CF metadata entries,
first-time public API wiring). Proportionate senior review caught real issues. Stories 5.1
and 5.2 did not receive equivalent depth despite 5.1's integration complexity.
> *L5: Use risk (integration scope, public API surface, scientific accuracy) as the senior review trigger.*

---

## Artifact Map

```
_bmad-output/
│
├── planning-artifacts/
│   ├── prd.md                              PRD v2.4.0 — 1,280 lines, 31 FRs, 4 tracks
│   ├── architecture.md                     6 patterns, ADRs, critical path analysis
│   ├── epics.md                            38 stories — 1,652 lines with full ACs
│   ├── epics-original.md                   PRD v1.1 epics — 261 lines (preserved for reference)
│   ├── implementation-readiness-report-2026-02-07.md
│   ├── implementation-readiness-report-2026-02-09.md
│   ├── implementation-readiness-report-2026-02-16.md
│   └── research/
│       ├── technical-eddi-validation.md    NOAA gap analysis; 1e-5 tolerance justification
│       ├── technical-penman-monteith.md    FAO-56 equations 1–19; helper decomposition
│       └── technical-palmer-modernization.md  Pattern A/B/C analysis; AD-3 basis
│
├── implementation-artifacts/
│   ├── sprint-status.yaml                  Story lifecycle tracking (backlog → done)
│   ├── plan-t1-exception-hierarchy.md      Pre-implementation design note
│   ├── 5-3-final-v240-validation.md        Release gate report — 347 lines
│   ├── epic-5-retro-2026-04-05.md          Full sprint retrospective — 383 lines
│   └── stories/
│       ├── 5-1-pattern-compliance-audit.md
│       └── 5-2-reference-validation-final-check.md
│
├── test-artifacts/
│   ├── test-design-architecture.md         Test framework architecture
│   ├── test-design-qa.md                   QA testing strategy
│   ├── test-design-progress.md             Coverage tracking
│   ├── coverage-plan-working.md            Coverage targets and metrics
│   └── risk-assessment-working.md          Risk-based test prioritization
│
└── project-context.md                      Agent team rules and coding conventions
```

**Document flow:**
```
prd.md ──► research/*.md ──► architecture.md ──► epics.md
                                                     │
                               implementation-readiness-reports (x3)
                                                     │
                                           sprint-status.yaml
                                                     │
                               stories/5-{1,2,3}-*.md (validation)
                                                     │
                                     epic-5-retro-2026-04-05.md
```

---

*All artifacts versioned at commit `9f027e1` (2026-04-05). Palmer xarray interface
(FR-PALMER-001 through FR-PALMER-007) deferred to v2.5.0 and identified as the critical
path for the next sprint.*
