# climate_indices + BMAD: A 20-Minute Demo Journey

**Presenter Guide for Demonstrating BMAD Methodology Across Multiple Development Iterations**

---

## Timing Reference

| Segment | Topic | Duration |
|---------|-------|----------|
| 1 | What This Is | 90 seconds |
| 2 | The Before State | 2 minutes |
| 3 | The Spec Chain | 2 minutes |
| 4 | The xarray Adapter — 6 Layers | 4 minutes |
| 5 | Live Code | 2 minutes |
| 6 | The EDDI Retrospective | 4 minutes |
| 7 | The Parallel Agent Teams | 3 minutes |
| 8 | Palmer Roadmap | 1.5 minutes |
| 9 | Three Lessons + Close | 1 minute |
| **Total** | | **20 minutes** |

---

## Segment 1: What This Is (90 seconds)

### Repository Context

This is `climate_indices` — a Python library for climate drought index computation used by climate researchers worldwide. The library implements scientifically rigorous calculations for:

- **SPI** (Standardized Precipitation Index)
- **SPEI** (Standardized Precipitation Evapotranspiration Index)
- **PET** (Potential Evapotranspiration — Thornthwaite, Hargreaves, Penman-Monteith)
- **Palmer Drought Indices** (PDSI, PHDI, PMDI, Z-Index)
- **EDDI** (Evaporative Demand Drought Index)

**Technology Stack:** NumPy, xarray/dask, scipy, structlog

**Repository:** https://github.com/monocongo/climate_indices

This demo walks through multiple BMAD iterations across February 2026, each more ambitious than the last, showing how structured AI development methodology scales from single issues to parallel agent teams implementing 38 stories across 5 epics.

### What is BMAD?

BMAD (Build More Architect Dreams) is a structured AI development methodology with four phases:

| Phase | Name | What Happens |
|-------|------|-------------|
| 1 | Analysis | Brainstorming, research, product brief or PRFAQ *(optional)* |
| 2 | Planning | Create requirements (PRD or spec) |
| 3 | Solutioning | Design architecture *(BMad Method/Enterprise only)* |
| 4 | Implementation | Build epic by epic, story by story |

The v6 improvement introduced a critical sequencing change:

> "Epics and stories are now created *after* architecture. This produces better quality stories because architecture decisions (database, API patterns, tech stack) directly affect how work should be broken down."

**Three Planning Tracks:**
- **Quick Flow:** 1-15 stories (lightweight projects)
- **BMad Method:** 10-50+ stories (this project)
- **Enterprise:** 30+ stories with additional governance

**This project used the BMad Method track** with **38 stories across 5 epics**.

---

## Segment 2: The Before State (2 minutes)

*Presenter note: Establish the pain points BMAD solved.*

The legacy codebase before BMAD had four major anti-patterns:

### 1. Exception Anti-Pattern: Tuple-of-None Returns

Functions returned `(None, None, None, None)` tuples on failure instead of raising exceptions. The Palmer `pdsi()` function returns a 5-tuple — when validation failed, it silently returned None in positions meant for drought indices. Users got cryptic downstream `TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'` errors with no context about what went wrong.

**Impact:** Debugging required tracing back through the call stack to find the root cause. No actionable error messages.

### 2. No Structured Logging

The codebase used stdlib `logging` with prose messages:

```python
logger.info("Starting calculation")
# ... computation ...
logger.info("Calculation complete")
```

These messages were impossible to parse programmatically for operational monitoring or debugging automation. No structured context about inputs, timings, or failure modes.

### 3. xarray Boilerplate Tax

Every user who wanted to use xarray with the library had to write ~30 lines of boilerplate:

```python
# Extract coordinates
time = precip_da.coords["time"]
lat = precip_da.coords["lat"]
lon = precip_da.coords["lon"]

# Extract numpy array
precip_values = precip_da.values

# Call library function (numpy only)
spi_values = spi(precip_values, scale=6, ...)

# Re-wrap with coordinates
spi_da = xr.DataArray(
    spi_values,
    coords={"time": time, "lat": lat, "lon": lon},
    dims=["time", "lat", "lon"],
)
```

The library only accepted raw NumPy arrays. xarray is the standard for climate data, so this was a major usability barrier.

### 4. Build Bloat

The distribution wheel was **37 MB** — it included test fixtures and non-essential files. After modernization (poetry → hatch + uv), the wheel dropped to **207 KB** (a 180x reduction).

*Note: This wheel size reduction is from the v2.1.1/v2.2.0 modernization era. The exact figures come from project history; hedge if asked for source.*

---

## Segment 3: The Spec Chain (2 minutes)

*Presenter note: Walk through the BMAD planning artifact chain. Each quote is from ACTUAL project files.*

BMAD produces a chain of documents that flow from high-level requirements down to implementation tasks. Here's how that played out for v2.4.0.

### The PRD

> Source: `_bmad-output/planning-artifacts/prd.md`

From the Executive Summary:

> "Version 2.4.0 focuses on **canonical pattern completion**, **scientific algorithm completeness**, and **advanced xarray capabilities**, informed by three comprehensive technical research documents and codebase inventory analysis completed in February 2026."

The PRD defines **31 functional requirements** across 4 parallel tracks:

- **Track 0: Canonical Pattern Completion** (12 FRs) — apply v2.3.0 patterns to ALL remaining indices
- **Track 1: PM-ET Foundation** (6 FRs) — physics-based evapotranspiration
- **Track 2: EDDI/PNP/scPDSI** (6 FRs) — index coverage expansion
- **Track 3: Palmer Multi-Output** (7 FRs) — advanced xarray with Dataset return

Each track has explicit dependencies. Track 0 and Track 1 run in parallel. Tracks 2 and 3 can only start after both Track 0 and Track 1 complete. This dependency design is what enables the parallel agent execution shown later in the demo.

### The Architecture Document

> Source: `_bmad-output/planning-artifacts/architecture.md`

From the complexity assessment:

> "Complexity level: **Medium-High** — Physics-based algorithms (PM FAO56: 19 coupled equations with non-linear vapor pressure), Multi-output xarray patterns (Palmer: 4 variables + params dict, no Python library precedent)"

The architecture document makes explicit decisions that become the "shared context" for parallel agents:

**Decision 1:** NOAA Provenance Protocol (JSON-based metadata for reference datasets)

**Decision 2:** Palmer xarray wrapper stays IN `palmer.py` (not separate module)

**Decision 3:** Property-based testing with Hypothesis (50-60 hours per index)

**Decision 4:** Per-module incremental exception migration

**Decision 5:** CF Metadata Registry centralization

The architecture also defines **24 naming patterns** and explicit **module boundaries**. This shared understanding prevents design conflicts when multiple agents implement different parts of the system.

### The Epics Document

> Source: `_bmad-output/planning-artifacts/epics.md`

From the dependency specification:

> "Palmer structlog migration (Story 1.6) MUST complete before Palmer xarray work (Story 4.1+)."

The epics document breaks down the architecture into **38 stories across 5 epics** with explicit agent assignments:

| Agent | Track | Epic | Story Count |
|-------|-------|------|-------------|
| Alpha | 0 (Patterns) | Epic 1: Canonical Pattern Completion | 12 stories |
| Beta | 1 (PM-ET) | Epic 2: PM-ET Foundation | 7 stories |
| Gamma | 2 (EDDI) | Epic 3: EDDI/PNP/scPDSI Coverage | 7 stories |
| Delta | 3 (Palmer) | Epic 4: Palmer Multi-Output | 9 stories |
| Multiple | Integration | Epic 5: Cross-Cutting Validation | 3 stories |

**Critical path identified:**

```
1.1 → 1.2 → 1.6 → 4.1 → 4.2-4.9  (~5-6 weeks)
```

Story 1.6 (Palmer structlog migration) is a **gate**. It MUST complete before Story 4.1 (Palmer xarray handoff validation) can begin. This explicit dependency appears in the agent orchestration screenshot later.

The epics document transforms the architecture's conceptual decisions into concrete implementation tasks with acceptance criteria, effort estimates, and dependency graphs.

---

## Segment 4: The xarray Adapter — 6 Layers (4 minutes)

*Presenter note: This is the technical heart of the demo. Walk through the code architecture.*

The xarray adapter is the innovation that eliminated the 30-line boilerplate tax. Let's walk through its 6-layer processing pipeline.

### Layer 1: Detect

> Source: `src/climate_indices/xarray_adapter.py` (line 141)

```python
class InputType(Enum):
    NUMPY = auto()
    XARRAY = auto()
```

The `detect_input_type()` function (line 157) classifies inputs:

- Returns `InputType.NUMPY` for `np.ndarray`
- Returns `InputType.XARRAY` for `xr.DataArray`
- Raises `InputTypeError` for unsupported types

**If NUMPY is detected, the decorator passes through to the wrapped function unchanged.** No overhead for users who don't use xarray.

### Layer 2: Resolve & Align

> Source: `src/climate_indices/xarray_adapter.py` (lines 668, 723)

For multi-input functions like SPEI (which needs both precipitation AND temperature for the water balance), two helper functions kick in:

- `_resolve_secondary_inputs()` (line 668) — Resolves named parameters to their actual DataArray objects
- `_align_inputs()` (line 723) — Aligns time coordinates across inputs, warning if data is dropped

**Example:** If precipitation covers 1980-2020 but temperature covers 1985-2020, alignment produces a warning and uses the intersection (1985-2020).

### Layer 3: Extract

The adapter pulls NumPy arrays from DataArrays while preserving coordinate metadata for later re-wrapping. This is a pure data transformation — no computation happens here.

### Layer 4: Infer

> Source: `src/climate_indices/xarray_adapter.py` (line 944)

The `_infer_temporal_parameters()` function automatically detects:

- `data_start_year` — from the time coordinate
- `periodicity` — monthly vs daily (inferred from time step frequency)
- `calibration_period` — optionally inferred from coordinate attributes

**This is the magic.** Users don't have to specify these parameters manually when using xarray inputs. The adapter reads them from the CF-compliant time coordinate.

### Layer 5: Compute

The adapter calls the original NumPy function. The core scientific algorithms in `indices.py` and `compute.py` remain **completely untouched**. This is a critical design principle — the adapter is a wrapper, not a replacement.

### Layer 6: Rewrap & Log

> Source: `src/climate_indices/xarray_adapter.py` (line 1134)

The `_build_output_dataarray()` function constructs the output DataArray with:

- Original coordinates (time, lat, lon) preserved
- **CF metadata** from the `CF_METADATA` registry (line 79)
- **Provenance history** attribute documenting the computation

Then the adapter logs completion via structlog:

```python
log.info("calculation_completed", duration_ms=elapsed)
```

### The CF_METADATA Registry

> Source: `src/climate_indices/xarray_adapter.py` (line 79)

The registry uses `CFAttributes` TypedDict (line 67) and currently has entries for:

- `spi` — Standardized Precipitation Index
- `spei` — Standardized Precipitation Evapotranspiration Index
- `pet_thornthwaite` — Thornthwaite PET
- `pet_hargreaves` — Hargreaves PET

Each entry provides:

```python
{
    "long_name": "Human-readable name",
    "units": "mm day-1" or "dimensionless",
    "references": "Citation with DOI",
    "standard_name": "CF convention name (optional)",
}
```

### Three Execution Paths

The adapter chooses an execution path based on input characteristics:

1. **Dask-backed arrays** → `apply_ufunc` with chunked execution (parallelized)
2. **Multi-dimensional in-memory** → `apply_ufunc` without Dask (vectorized)
3. **1D in-memory** → Direct function call (fast path, no overhead)

This tiered approach optimizes for both small interactive computations and large distributed processing.

---

## Segment 5: Live Code (2 minutes)

*Presenter note: Show concrete code artifacts. Each is a grep or cat command the presenter runs live.*

### Exceptions Hierarchy

> Source: `src/climate_indices/exceptions.py`

**The root:**

```python
class ClimateIndicesError(Exception):  # line 38
    """Base exception for all climate_indices library errors.
    
    Catch this exception to handle any error raised by the library.
    """
    pass
```

**11 exception classes:**

- `DistributionFittingError` — Base for statistical fitting failures
- `InsufficientDataError` — Not enough data for calibration
- `PearsonFittingError` — Pearson Type III fitting failure
- `ConvergenceError` — Iterative algorithm failed to converge
- `DimensionMismatchError` — Array dimensions incompatible
- `CoordinateValidationError` — Invalid or missing coordinates
- `InputTypeError` — Wrong input type
- `InvalidArgumentError` — Parameter out of valid range
- `PeriodicityError` — Invalid periodicity (monthly/daily)
- `DataShapeError` — Array shape doesn't match expected structure

**6 warning classes:**

- `ClimateIndicesWarning` — Base for all library warnings
- `MissingDataWarning` — Excessive missing data in calibration
- `ShortCalibrationWarning` — Calibration period too short
- `GoodnessOfFitWarning` — Distribution fit quality poor
- `InputAlignmentWarning` — Data dropped during alignment
- `BetaFeatureWarning` — Feature marked as experimental

**Key design:** Each exception supports keyword-only context attributes for actionable error messages:

```python
raise InvalidArgumentError(
    "Scale must be between 1 and 72 months",
    argument_name="scale",
    argument_value=str(scale),
    valid_values="1-72",
)
```

### Typed Public API

> Source: `src/climate_indices/typed_public_api.py`

**SPI example (lines 49-75):**

```python
@overload
def spi(
    values: npt.NDArray[np.float64],
    scale: int,
    distribution: compute.Distribution,
    ...
) -> npt.NDArray[np.float64]:
    """NumPy path — returns NumPy array."""
    ...

@overload
def spi(
    values: xr.DataArray,
    scale: int,
    distribution: compute.Distribution,
    ...
) -> xr.DataArray:
    """xarray path — returns xarray DataArray."""
    ...
```

The implementation dispatches based on `isinstance()` check:

```python
if isinstance(values, xr.DataArray):
    # xarray path
    return _spi_xarray(values, scale, distribution, ...)
else:
    # numpy path
    return indices.spi(values, scale, distribution, ...)
```

This pattern appears for `spi()` (lines 49-75) and `spei()` (lines 137-165). Type checkers like mypy can infer the return type based on the input type.

### Structured Logging

> Source: `src/climate_indices/indices.py`

**All calculation functions use the lifecycle event pattern:**

```python
log = get_logger(__name__)
log = log.bind(
    calculation="spi",
    scale=scale,
    distribution=distribution.value,
    data_shape=values.shape,
)
log.info("calculation_started")

# ... computation ...

log.info("calculation_completed", duration_ms=elapsed)
```

**Examples in the code:**
- Line 200 — SPI calculation started
- Line 395 — SPEI calculation started
- Line 559 — PET Thornthwaite calculation started
- Line 670 — PET Hargreaves calculation started
- Line 761 — PCI calculation started

**Error events with context:**

```python
log.error(
    "calculation_failed",
    error=str(e),
    distribution=distribution.value,
)
```

This structured approach enables operational monitoring, automated alerting, and post-mortem analysis. All logs can be aggregated, parsed, and queried programmatically.

---

## Segment 6: The EDDI Retrospective (4 minutes)

*Presenter note: This is the emotional center of the demo. Read directly from committed retrospective insights.*

This section comes from the EDDI implementation effort — a real 5-year-old dormant issue that BMAD methodology tackled in 3 days.

### The Setup

**Issue #414** — "Implement EDDI" — opened **2021-01-31** by @monocongo. Two-line description:

> "Add EDDI (Evaporative Demand Drought Index) support to the library."

That's it. No algorithm details, no reference data, no validation criteria. The issue sat dormant for approximately **5 years**.

### Phase 1: Discovery (Feb 3)

First BMAD session ran `/bmad-bmm-generate-project-context` to understand the brownfield codebase. This generated `project-context.md` — the equivalent of **2-3 hours of manual exploration compressed into ~5 minutes**.

The project context document revealed:
- Existing patterns for SPI/SPEI (standardized indices)
- xarray adapter infrastructure from v2.3.0
- structlog logging conventions
- Exception hierarchy rooted at `ClimateIndicesError`

This automated discovery established the implementation patterns to follow.

### Phase 2: Requirements (Feb 5 morning)

PM agent "John" (via `/bmad-agent-bmm-pm`) created a PRD. The PM agent:

- Scoped MVP to EDDI only
- Deferred Palmer Z-Index to backlog (related but separate algorithm)
- Set success criteria: **"match NOAA reference"**
- Identified input requirements: PET (not precipitation), scale parameter, calibration period

This requirements work took ~90 minutes and produced a 20-page PRD with user journeys, acceptance criteria, and technical constraints.

### Phase 3: The Wrong Turn (Feb 5 afternoon)

**This is where the interesting lesson appears.**

PM agent "John" EXCEEDED its role boundary and began implementing. It looked at SPI/SPEI code and pattern-matched:

- SPI uses gamma distribution fitting
- SPEI uses Pearson Type III distribution fitting
- Therefore, EDDI must also use parametric distribution fitting

**This is wrong.** EDDI is **non-parametric** — it uses empirical ranking against climatology. There's no distribution fitting involved.

The agent implemented a parametric approach because it looked at similar code in the repository instead of reading the NOAA Fortran reference.

From the retrospective:

> "When implementing algorithms from scientific literature, **read the reference code first**, even if it's in a different language (Fortran, C, R). Pattern-matching against similar functions in the codebase is unreliable when algorithms differ fundamentally (parametric vs. non-parametric)."

### Phase 4: Course Correction (Feb 5 late afternoon)

The user read the NOAA Fortran reference (`calc_eddi.f90`) and identified the correct algorithm:

1. **Empirical ranking** against calibration climatology (not distribution fitting)
2. **Tukey plotting positions**: P = (rank - 0.33) / (N + 0.33)
3. **Hastings polynomial approximation** for inverse normal (NOT scipy.stats.norm.ppf)

Re-implementation used the Hastings coefficients exactly as in the NOAA Fortran:

```python
c0 = 2.515517
c1 = 0.802853
c2 = 0.010328
```

Matching the reference implementation bit-for-bit was critical for validation.

**Cost of wrong turn:** ~3-4 hours of wasted implementation effort.

### Phase 5: Quality (Feb 5 evening)

The original branch had **845 files changed (226,035 insertions)** due to BMAD exploration artifacts, PRD drafts, architecture documents, and intermediate experiments.

**Solution:** Clean branch strategy — cherry-pick only production commits.

From the retrospective metrics table:

| Metric | Value |
|--------|-------|
| Issue age | ~5 years (2021-01-31 → 2026-02-05) |
| Active development time | 3 days (Feb 3–5, 2026) |
| Original branch files changed | 845 files (226,035 insertions) |
| Clean branch files changed | 3 files (718 insertions, 1 deletion) |
| **Reduction ratio** | **282:1** (files), 314:1 (insertions) |
| Final test count | 15 (9 original + 6 from Sourcery review) |
| CI checks | 12 (all passing) |
| Time lost to wrong turn | ~3–4 hours |

**282:1 reduction.** This is the real ratio of AI-assisted development. Massive exploration churn compressed into clean production commits.

### The Lesson on Agent Roles

From the retrospective:

> "BMAD agents have **role-specific expertise**. A PM agent is trained on requirements gathering, not algorithm implementation. When an agent exceeds its role, quality degrades."

This lesson directly informed the v2.4.0 team structure:

- **Alpha** — Pattern completion specialist
- **Beta** — PM-ET algorithm specialist
- **Gamma** — EDDI/reference validation specialist
- **Delta** — Palmer multi-output specialist
- **Omega** — Integration orchestrator

Each agent stays in its lane. Role boundaries are enforced in the team structure.

---

## Segment 7: The Parallel Agent Teams (3 minutes)

*Presenter note: This is the single most visually compelling moment in the demo. Display the screenshot: `~/screenshots/Screenshot 2026-02-17 at 12.10.32 AM.png`*

*If screenshot not available, describe verbally based on the structure below.*

The screenshot shows a `team-lead` orchestrator on branch `feature/v2.4.0-planning` managing **7 named specialist subagents**, each working in an isolated git worktree.

### What the Screenshot Shows

**Status block (top panel):**

- ✅ **Epic 2: 100% COMPLETE** (7/7 stories) — PM-ET Foundation finished
- ✅ **T1, T20 complete and pushed** — Exception hierarchy (1.1) and EDDI merge conflicts (3.1) done
- 🔄 **T6 (Palmer structlog) — CRITICAL PATH in progress** — Story 1.6, **blocks all Palmer xarray work**
- 🔄 **T21 (NOAA Provenance) in progress** — Story 3.2, establishing reference dataset metadata protocol
- **Overall:** 8/35 stories (23%) complete

**Agent panel (bottom):**

- `@canonical-dev-2` (145.0k tokens): Editing `.worktrees/epic-1-canonical/tests/test_exceptions.py` — implementing **T6 (Palmer structlog migration)**
- `@eddi-dev-2` (146.1k tokens): Writing `.worktrees/epic-3-eddi/tests/fixture/palmer/provenance.json` — implementing **T21 (NOAA Provenance Protocol)**
- `@palmer-dev` (87.4k tokens, **idle**): Consumed tokens doing preparatory work but now **BLOCKED** — waiting for T6 to complete before Epic 4 can begin
- `@pm-et-dev` / `@pm-et-dev-2` (0 tokens, idle): Epic 2 already 100% complete, these agents were never activated
- `team-lead` (291 tokens): Lightweight orchestrator — dispatching, monitoring, enforcing gates

**Token budget (footer):**

101,394/200,000 used (98,606 remaining) — this is the team-lead's context window tracking. At 60% utilization, orchestration has headroom for the remaining 27 stories.

### Three Key Patterns to Explain

#### Pattern 1: Architecture as Shared Context

From BMAD documentation:

> "When multiple AI agents implement different parts of a system, they can make conflicting technical decisions. Architecture documentation prevents this by establishing shared standards."

The architecture document (with its 24 naming patterns, module boundaries, and 5 architectural decisions) is the **shared context** that allows 7 agents to work in parallel without producing conflicting implementations.

**The flow:**

```
PRD: "What to build"
     ↓
Architecture: "How to build it"  ← SHARED CONTEXT
     ↓
Agent A reads architecture → implements Epic 1 (canonical patterns)
Agent B reads architecture → implements Epic 2 (PM-ET)
Agent C reads architecture → implements Epic 3 (EDDI)
     ↓
Result: Consistent implementation (no merge conflicts)
```

Without the architecture document, each agent would make independent decisions about:
- Exception naming conventions
- CF metadata structure
- Logging patterns
- Module boundaries

The architecture document aligns these decisions **before** implementation begins.

#### Pattern 2: Epic Gates and Dependency Design

The critical path from `epics.md`:

```
1.1 → 1.2 → 1.6 → 4.1 → 4.2-4.9
```

**T6 (Story 1.6: Palmer structlog migration) is on the CRITICAL PATH** because Story 4.1 ("Palmer xarray Handoff Validation") requires:

- Story 1.6 complete: Palmer uses structlog (no stdlib logging)
- Story 2.7 complete: Baseline performance measurement exists

Since Epic 2 was already 100% complete (including 2.7), only T6 remained as the gate blocker.

The team-lead message **"Epic 4 gate opens when T6 completes"** is a LIVE enforcement of this dependency design. `@palmer-dev` sits idle with 87.4k tokens consumed (it did preparatory work) but **cannot proceed**.

Meanwhile, **T21 (NOAA Provenance) runs concurrently** because it has NO dependency on T6 — it's in Epic 3, a completely independent track. This parallelism was designed into the story structure.

**This is not accidental.** The epics document explicitly designed these dependencies:

- Track 0 + Track 1 can run **in parallel** (no shared dependencies)
- Track 2 + Track 3 can run **in parallel after** Track 0 and Track 1 complete
- Story 1.6 (Palmer structlog) **partially blocks** Track 3 (Palmer xarray work)

The team-lead enforces these gates automatically.

#### Pattern 3: Token Budget as Orchestration Cost

**Token budget:** 101,394/200,000 used (98,606 remaining)

This is the team-lead's context window. It tells us:

- The orchestrator is **lightweight** (291 tokens of its own)
- Most tokens consumed by agent dispatching, status updates, and gate checking
- At **60% context utilization**, orchestration has headroom for the remaining 27 stories

The team-lead doesn't do implementation work — it dispatches tasks, monitors progress, enforces gates, and synthesizes status. The heavy lifting happens in the subagent contexts.

### Critical Distinction: This is NOT "Party Mode"

BMAD "Party Mode" is a discussion feature where multiple BMAD agents debate in one conversation:

> "Run `bmad-party-mode` and you've got your whole AI team in one room - PM, Architect, Dev, UX Designer, whoever you need."

What the screenshot shows is fundamentally different: **parallel agent subagents** in Claude Code's experimental agent teams feature, each running in an **isolated git worktree**, implementing **different stories concurrently**.

**Key differences:**

| Party Mode | Parallel Subagents (Screenshot) |
|------------|--------------------------------|
| Single conversation | Multiple isolated contexts |
| Discussion/planning focus | Implementation focus |
| No code changes | Active development in worktrees |
| All agents "present" always | Agents activated on-demand |

### The Worktree Structure

```
.worktrees/
├── epic-1-canonical/  → branch: feature/epic-1-canonical-patterns (19 commits)
├── epic-2-pm-et/      → branch: feature/epic-2-pm-et-foundation (9 commits)
├── epic-3-eddi/       → branch: feature/epic-3-eddi-pnp-scpdsi (15 commits)
└── epic-4-palmer/     → branch: feature/epic-4-palmer-multi-output (22 commits)
```

Each worktree is an isolated working directory pointing to a different branch. Agents work independently, then merge back to the main branch via pull requests. This prevents file conflicts and enables true parallelism.

### Model Note

The entire parallel execution runs on **Sonnet 4.5** (visible in screenshot footer), not Opus. This means the parallel agent strategy is **cost-effective** — using the faster, cheaper model for implementation while the BMAD planning artifacts (created by humans + more capable models) provide the guardrails.

---

## Segment 8: Palmer Roadmap (1.5 minutes)

*Presenter note: Contrast disciplined vs undisciplined exploration.*

There's a branch in the repository that illustrates the "before BMAD" exploration pattern.

### The `boy_scouting` Branch

**PR #596**, opened **Dec 30, 2025**, has **13 commits** of Palmer exploration work:

- Multi-source PET (allowing users to provide PET from any method)
- Zarr support (cloud-optimized storage format)
- numba JIT compilation (performance optimization)
- xarray accessor pattern (alternative to decorator approach)

This is **real, substantial work** — but it's undisciplined. The branch was opened before BMAD was applied to the project. It explores multiple ideas simultaneously without clear success criteria or integration plan.

### The v2.4.0 Plan: Structured Ambition

Epic 4 (Palmer Multi-Output) takes the best ideas from `boy_scouting` and structures them into **9 stories with explicit dependencies:**

- **Story 4.3:** `palmer_xarray()` manual wrapper foundation
- **Story 4.4:** AWC spatial parameter handling (novel validation pattern)
- **Story 4.5:** Multi-output Dataset construction (4 variables with independent CF metadata)
- **Story 4.7:** typed_public_api @overload signatures (numpy→tuple, xarray→Dataset)
- **Story 4.9:** Performance validation (≥80% of multiprocessing baseline)

The architecture document (Decision 2) already resolves the key question:

> "Palmer xarray wrapper stays IN `palmer.py` (not separate module)"

This decision was made in the architecture phase based on module size analysis:

- `palmer.py` is currently 912 lines
- Adding ~150 lines for xarray wrapper → 1,062 lines total
- Module complexity threshold: 1,400 lines
- Conclusion: Keep wrapper in same module for cohesion

### The Contrast

**boy_scouting pattern:** Exploration without structure. Multiple ideas in parallel, no clear integration path, no success criteria.

**Epic 4 pattern:** The same ambition with BMAD's spec-first discipline. Architecture decisions made upfront, stories with acceptance criteria, performance targets defined, dependencies explicit.

The v2.4.0 plan takes ~10 weeks to implement, but it has:
- Clear gates and dependencies
- Performance targets (≥80% baseline)
- Validation criteria (1e-8 equivalence)
- Integration strategy (manual wrapper stays in palmer.py)

This is the difference between exploration and implementation.

---

## Segment 9: Three Lessons + Close (1 minute)

*Presenter note: Distill insights and provide closing context.*

### Lesson 1: The Spec is Ground Truth

The EDDI wrong turn happened because the PM agent pattern-matched instead of reading the reference. The agent looked at similar code (SPI/SPEI parametric approaches) and assumed EDDI followed the same pattern.

**Wrong.** EDDI is non-parametric.

The v2.4.0 architecture document prevents this at scale — **7 agents, one shared context, consistent output**. Each agent reads the same architecture before starting. The result: no design conflicts, no conflicting patterns.

The lesson: **Read the reference first.** For scientific algorithms, the reference implementation (even in Fortran) is more reliable than pattern-matching against similar code.

### Lesson 2: Role Separation Matters

PM agents should stop after PRD delivery. Developer agents should implement. When boundaries blur, quality degrades.

The EDDI PM agent exceeded its role and implemented the wrong algorithm. The v2.4.0 team structure (Alpha, Beta, Gamma, Delta, Omega) enforces this with **named specialist agents**:

- Alpha owns pattern completion (exceptions, logging, type safety)
- Beta owns PM-ET (physics-based algorithms)
- Gamma owns EDDI (reference validation)
- Delta owns Palmer (multi-output xarray)
- Omega orchestrates integration

Each agent stays in its lane. The team-lead enforces gates and dependencies.

### Lesson 3: 282:1 is the Real Ratio

**845 files of exploration → 3 files of production code.**

AI-assisted development generates **massive churn**. The BMAD artifact chain (PRD → Architecture → Epics → Stories), the clean branch strategy, and the worktree isolation pattern are all responses to this reality.

The structure BMAD provides:
- Planning artifacts are SEPARATE from code (in `_bmad-output/`)
- Production commits are cherry-picked from exploration branches
- Each epic gets its own worktree (isolated, then merged)

This keeps the main branch clean while allowing unrestricted exploration in feature branches.

### Close

From a **5-year-old dormant issue** (EDDI #414 in 2021) to **parallel agent teams implementing 38 stories** (v2.4.0 in 2026) — this is what structured AI development looks like on a real scientific computing project.

The progression:
1. Manual exploration (boy_scouting branch)
2. Single-issue BMAD (EDDI in 3 days)
3. Multi-epic orchestration (v2.4.0 with 38 stories)

BMAD scaled from fixing a single issue to orchestrating 7 agents implementing 38 stories across 5 epics. The methodology adapted, but the principles remained constant:

- **Spec before code** (PRD → Architecture → Epics)
- **Shared context prevents conflicts** (24 naming patterns, 5 architectural decisions)
- **Role separation enforces quality** (PM doesn't implement, developers don't do requirements)
- **Clean branches from messy exploration** (282:1 reduction)

That's the journey. Questions?

---

*Demo complete. 20 minutes.*
