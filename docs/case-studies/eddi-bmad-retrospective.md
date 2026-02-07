# EDDI BMAD Retrospective — Case Study

## Overview

**Issue**: #414 "Implement EDDI"
**Opened**: 2021-01-31 by @monocongo
**Closed**: 2026-02-05 (PR #597)
**Time dormant**: ~5 years
**Active development**: 3 days (Feb 3–5, 2026)
**Outcome**: Production-ready non-parametric EDDI implementation, 15 tests, 12/12 CI checks passing

Issue #414 was opened in January 2021 with a two-line description: "Provide implementation and CLI to allow for the computation of EDDI datasets." The Evaporative Demand Drought Index (EDDI) had been published by NOAA PSL, with Fortran reference code available, but no Python implementation existed in the `climate_indices` package.

After nearly five years of dormancy, the issue was closed in February 2026 using BMAD (Better Method for AI Development) v6.0.0-Beta.7 — a structured workflow framework for AI-assisted development that combines planning, implementation, and review phases. This retrospective documents how BMAD transformed a stale issue into a merged PR in three days, including a critical course correction when the implementation diverged from the scientific reference.

## BMAD Framework Context

### What is BMAD?

BMAD (Better Method for AI Development) is a structured workflow framework that provides specialized agents and command patterns for AI-assisted software development. Version 6.0.0-Beta.7 introduced the BMM (BMAD Method Modules) system, which offers multiple workflow patterns:

- **Quick Flow**: Fast iteration for straightforward tasks (skip PRD, minimal ceremony)
- **Full BMM**: Comprehensive workflow with PRD, technical design, implementation, and review phases
- **Hybrid**: Mix Quick Flow for simple tasks with Full BMM phases where structure adds value

### Available BMAD Commands Used

| Command | Purpose | When Used |
|---------|---------|-----------|
| `/bmad-help` | Show available workflow steps and guidance | Initial orientation (Feb 3) |
| `/bmad-bmm-generate-project-context` | Generate comprehensive codebase documentation | First-time brownfield discovery (Feb 3) |
| `/bmad-agent-bmm-pm` | Spawn Product Manager agent "John" for PRD creation | Requirements phase (Feb 5 morning) |
| `/bmad-bmm-code-review` | Structured code review of implementation | (Not used - Sourcery AI handled review) |

### Workflow Pattern Selected

The implementation used a **Hybrid Full BMM** approach:

1. ✅ **Discovery Phase** — Used `/bmad-bmm-generate-project-context` to understand brownfield codebase
2. ✅ **Requirements Phase** — Used `/bmad-agent-bmm-pm` to create PRD with PM agent "John"
3. ✅ **Implementation Phase** — Manual implementation with Claude Code
4. ✅ **Review Phase** — Automated review via Sourcery AI (external tool, not BMAD command)
5. ❌ **Technical Design Document** — Skipped (quick to implementation)
6. ❌ **BMAD Code Review** — Skipped (used Sourcery AI instead)

The Hybrid approach was appropriate for this task: structured enough to scope the problem (PRD) and understand the codebase (project context), but streamlined enough to avoid ceremony for a single-function implementation.

## Phase 1: Discovery & Project Context (Feb 3, 2026)

### First Encounter

The first Claude Code session on Feb 3 began with the user request: *"I want to work on implementing EDDI. Let's use BMAD."* This was the developer's first time using BMAD on the `climate_indices` codebase.

### Brownfield Codebase Challenge

The `climate_indices` package is a mature brownfield codebase:

- **Age**: First commit 2016, 6+ years of history
- **Structure**: Monolithic `indices.py` (~3,500 lines), comprehensive test suite
- **Patterns**: NumPy-based implementations with rolling windows and percentile ranking
- **Existing indices**: SPI, SPEI, PNP, PET (Thornthwaite, Hargreaves), Palmer indices (PDSI, PHDI, PMDI, SCPDSI, Z-Index)

Without prior context, implementing EDDI required understanding:

1. How existing drought indices are implemented (SPI/SPEI use parametric distributions)
2. What fixtures and test patterns are standard
3. How the module exports public API (`__all__`)
4. What dependencies are available (`numpy`, `scipy`, `numba`)

### `/bmad-bmm-generate-project-context` — The Discovery Tool

The command `/bmad-bmm-generate-project-context` was invoked to generate comprehensive codebase documentation. This BMAD command:

1. Explored repository structure with `Glob` and `Grep`
2. Read key files (`indices.py`, `compute.py`, `conftest.py`, README)
3. Analyzed existing patterns (SPI implementation, test fixtures, percentile logic)
4. Generated a structured markdown document: `_bmad/project-context.md`

**What was captured** (estimated 150+ lines):

- Architecture overview (monolithic `indices.py` + compute utilities)
- Key algorithms (SPI gamma fitting, SPEI Pearson fitting, percentage-of-normal)
- Test patterns (pytest fixtures, parameterization, NumPy testing utilities)
- Dependencies and their usage
- Public API export patterns

**Value delivered**:

This single command provided the equivalent of 2–3 hours of manual codebase exploration, compressed into ~5 minutes of AI agent work. It established a shared context that persisted across future sessions.

### Artifact Loss

The `project-context.md` file was generated in the original `feature/issue-414-eddi` branch but **lost during the clean branch strategy** (Phase 5). The file was never committed to `feature/issue-414-eddi-clean`, so it exists only in local session artifacts.

**Impact**: Future developers onboarding to BMAD on this codebase will need to regenerate this context. The artifact loss highlights a gap in BMAD workflow: **project context documents should be committed** to a stable location (e.g., `docs/bmad/project-context.md`) for reuse across branches.

## Phase 2: Requirements & PRD (Feb 5, Morning)

### PM Agent "John" — The `/bmad-agent-bmm-pm` Command

On Feb 5 morning, the developer invoked `/bmad-agent-bmm-pm` to create a Product Requirements Document (PRD). This BMAD command spawns a **Product Manager agent** named "John" in a background subprocess.

**Agent behavior**:

- "John" conducted stakeholder-style interviews via `AskUserQuestion` tool
- Asked about scope: *"Should EDDI, Palmer Z-Index, and xarray both be in scope for this PR?"*
- Asked about success criteria: *"What does success look like — match xclim performance or beat it?"*
- Generated a structured PRD markdown document: `_bmad/prd.md`

### Scoping Decisions

The user provided clear boundaries via the Q&A:

1. **MVP Scope**: EDDI implementation + xarray deprecation fix (issue #588)
2. **Deferred**: Palmer Z-Index (issue #411) moved to backlog
3. **Success Criteria**: "Match xclim output or beat it" for EDDI accuracy
4. **Non-Goals**: Performance optimization (defer to future if needed)

### PRD Contents (Reconstructed)

The PRD included:

- **Objective**: Implement EDDI to match NOAA PSL reference algorithm
- **User Stories**: Climate researchers need EDDI for drought monitoring at 1–12 month scales
- **Technical Requirements**:
  - Non-parametric empirical ranking (vs. parametric SPI/SPEI)
  - Tukey plotting positions
  - Inverse normal transformation
  - Both monthly and daily periodicity support
- **Acceptance Criteria**:
  - Output matches NOAA Fortran reference within numerical precision
  - Comprehensive test suite (edge cases, NaN handling, 1-D/2-D inputs)
  - README documentation update
- **Out of Scope**: xarray API (use NumPy arrays like existing indices)

### Artifact Loss (Again)

The `prd.md` file was also lost during the clean branch strategy. It was generated on the original branch but not carried forward.

**Impact**: The PRD served its purpose during development (scoped the work, aligned user expectations) but is no longer available for future reference. A committed PRD would have documented the scoping decisions and non-goals for maintainers reviewing the PR months later.

## Phase 3: Implementation — The Wrong Turn (Feb 5, Afternoon)

### Initial Implementation: Parametric Approach

After the PRD was finalized, PM agent "John" **exceeded its role boundary** and began implementing the EDDI algorithm. This was not part of the standard BMAD workflow — PM agents should hand off to implementation agents or exit after PRD delivery.

**What "John" implemented** (~200 lines):

```python
def eddi(
    pet_values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
) -> np.ndarray:
    """
    Compute EDDI using **parametric** approach (WRONG).

    Algorithm:
    1. Accumulate PET at specified scale
    2. Fit gamma or Pearson distribution to calibration period
    3. Transform to standard normal using fitted CDF
    """
    # ... gamma fitting logic (borrowed from SPI) ...
    # ... inverse normal transformation via scipy.stats.norm.ppf ...
```

**Why this was wrong**:

- **SPI/SPEI are parametric** — they fit gamma (SPI) or Pearson Type III (SPEI) distributions to precipitation/climatic water balance
- **EDDI is non-parametric** — it uses empirical ranking against climatology, not distribution fitting
- The PM agent generalized from existing `indices.py` patterns (SPI/SPEI) without consulting the NOAA Fortran reference code

### The User Catches the Error

Hours later, the user reviewed the implementation and flagged the discrepancy:

> "Wait, EDDI shouldn't be using gamma fitting. The NOAA reference code uses empirical ranking. Let me check the Fortran..."

The user provided a link to the NOAA PSL Fortran reference:
https://psl.noaa.gov/eddi/code/calc_eddi.f90

### Root Cause: Agent Role Violation

The PM agent "John" was invoked to create a PRD, not to implement code. By continuing into implementation, "John" operated outside its expertise domain and made an incorrect algorithmic choice.

**Lesson**: BMAD agents should stay within their role boundaries. When an agent completes its task (PRD delivery), it should exit or hand off to a specialized implementation agent, not continue autonomously.

## Phase 4: Course Correction (Feb 5, Late Afternoon)

### Consultation with NOAA Fortran Reference

The user read the NOAA Fortran reference code (`calc_eddi.f90`) and identified the correct algorithm:

1. **Empirical Ranking**: For each month/day, rank the current PET value against all historical values for that same calendar period within the calibration window
2. **Tukey Plotting Positions**: Convert rank `r` to probability: `P = (r - 0.33) / (N + 0.33)`
3. **Inverse Normal (Hastings Approximation)**: Transform probability to z-score using a polynomial approximation (not `scipy.stats.norm.ppf`)

The Hastings approximation is a 6th-degree polynomial that matches the NOAA Fortran reference exactly:

```fortran
! Hastings approximation (NOAA Fortran reference)
t = sqrt(-2.0 * log(p))
z = t - ((c0 + c1*t + c2*t**2) / (1.0 + d1*t + d2*t**2 + d3*t**3))
```

### Re-Implementation

Claude Code (now in direct implementation mode, not via PM agent) rewrote the algorithm:

```python
def _hastings_inverse_normal(probabilities: np.ndarray) -> np.ndarray:
    """
    Convert probabilities to z-scores using Hastings approximation.

    Matches NOAA Fortran reference exactly (coefficients from Hastings 1955).
    """
    # Hastings coefficients
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    # Symmetric handling: p < 0.5 → compute for 1-p and negate
    p = np.where(probabilities < 0.5, probabilities, 1.0 - probabilities)
    t = np.sqrt(-2.0 * np.log(p))
    z = t - ((c0 + c1 * t + c2 * t**2) / (1.0 + d1 * t + d2 * t**2 + d3 * t**3))
    return np.where(probabilities < 0.5, -z, z)


def eddi(
    pet_values: np.ndarray,
    scale: int,
    data_start_year: int,
    calibration_year_initial: int,
    calibration_year_final: int,
    periodicity: compute.Periodicity,
) -> np.ndarray:
    """
    Compute EDDI using **non-parametric empirical ranking** (CORRECT).

    Algorithm:
    1. Reshape PET to (years, periods) — 12 for monthly, 366 for daily
    2. For each period (e.g., "all Januarys"), compute rolling sum at scale
    3. For each year, rank current value against calibration climatology
    4. Convert rank to probability using Tukey: P = (rank - 0.33) / (N + 0.33)
    5. Transform to z-score using Hastings approximation
    6. Clip to ±3.09 (NOAA convention)
    """
    # ... empirical ranking logic ...
    # ... Tukey plotting positions ...
    # ... Hastings inverse normal ...
```

### Validation Against Reference

A test was added to verify the Hastings approximation matches `scipy.stats.norm.ppf` within 0.001 tolerance:

```python
def test_hastings_inverse_normal_accuracy():
    """Hastings approximation should match scipy.stats.norm.ppf within 0.001."""
    probabilities = np.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
    hastings_z = indices._hastings_inverse_normal(probabilities)
    scipy_z = scipy.stats.norm.ppf(probabilities)
    np.testing.assert_allclose(hastings_z, scipy_z, atol=0.001)
```

This test passed, confirming the re-implementation matched both NOAA reference and `scipy`.

### Time Cost of the Wrong Turn

**Estimated time lost**: 3–4 hours

- PM agent implementation: ~1.5 hours (200 lines + initial tests)
- User review and error detection: ~0.5 hours (reading Fortran reference)
- Re-implementation: ~1.5 hours (new algorithm + updated tests)
- Additional validation: ~0.5 hours (Hastings test, edge cases)

**Mitigating factor**: The wrong turn forced deeper engagement with the NOAA reference, which improved the final implementation quality. The Hastings approximation test (added during course correction) provides long-term regression protection.

## Phase 5: Quality & Delivery (Feb 5, Evening)

### The 845-File Problem

After implementation was complete, `git status` on the original `feature/issue-414-eddi` branch showed:

```
848 files changed, 226035 insertions(+), 8 deletions(-)
```

**What happened?**

During BMAD exploration phases (project context generation, agent subprocess execution), temporary files and agent artifacts accumulated:

- `_bmad/project-context.md`, `_bmad/prd.md` (BMAD artifacts)
- Session transcripts, agent logs
- Potentially: Python `__pycache__`, `.pytest_cache`, or other temp files

The branch had become polluted with non-production changes, making PR review difficult.

### Clean Branch Strategy

The solution: **start a clean branch from master and cherry-pick only production changes**.

```bash
# Create clean branch
git checkout master
git checkout -b feature/issue-414-eddi-clean

# Cherry-pick the implementation commit
git cherry-pick 93ccdfc  # feat(eddi): implement non-parametric algorithm

# Verify only production files changed
git diff master --stat
# README.md                          |   2 +
# src/climate_indices/indices.py    | 208 +++++++++++++++++++++
# tests/test_eddi.py                 | 508 ++++++++++++++++++++++++++++++++++++++++++++++
# 3 files changed, 718 insertions(+), 1 deletion(-)
```

**Result**: 3 files, 718 insertions, 1 deletion — a clean, reviewable PR.

### PR Creation

PR #597 was created with:

- **Title**: `feat(eddi): implement non-parametric algorithm matching NOAA reference`
- **Body**: Structured summary (algorithm, testing, scientific accuracy, files changed)
- **Closes**: #414

The PR body followed Conventional Commits style and included a "Scientific Accuracy" section explaining the non-parametric approach.

### Automated Review: Sourcery AI

Within 2 minutes of PR creation, Sourcery AI (a GitHub App integration, not a BMAD tool) posted a review with **4 findings**:

| # | Type | Location | Issue | Resolution |
|---|------|----------|-------|------------|
| 1 | `bug_risk` | `indices.py:366-367` | **Calibration period validation**: No check for `calibration_year_initial/final` being outside data range or inverted | Added validation: fail fast with `ValueError` if start < data start, end > data end, or start > end |
| 2 | `suggestion (testing)` | `test_eddi.py:140-149` | **Daily periodicity coverage**: All tests use `monthly`, daily branch untested | Added `test_eddi_daily_computation()` using conftest daily fixtures |
| 3 | `suggestion (testing)` | `test_eddi.py:259-268` | **Insufficient climatology coverage**: `len(climatology_valid) < 2` branch untested | Added `test_eddi_insufficient_climatology()` forcing January NaNs in calibration period |
| 4 | `suggestion (testing)` | `test_eddi.py:47-56` | **Mixed NaN coverage**: Only all-NaN test exists, no scattered NaN test | Added `test_eddi_mixed_nan_positions()` with synthetic scattered NaNs |

All 4 items were addressed in a follow-up commit (`2ac7810`), expanding the test suite from 9 → 15 tests.

### CI Validation

PR #597 triggered **12 CI checks** (all passing):

| Check | Status | Purpose |
|-------|--------|---------|
| `test (3.10)` × 2 workflows | ✅ Pass | Python 3.10 compatibility |
| `test (3.11)` × 2 workflows | ✅ Pass | Python 3.11 compatibility |
| `test (3.12)` × 2 workflows | ✅ Pass | Python 3.12 compatibility |
| `test (3.13)` × 2 workflows | ✅ Pass | Python 3.13 compatibility (latest) |
| `SonarCloud` | ✅ Pass | Code quality, maintainability, security |
| `SonarCloud Code Analysis` | ✅ Pass | Deep static analysis |
| `Sourcery review` | ✅ Pass | AI-powered code review |
| `security/snyk` | ✅ Pass | Dependency vulnerability scan |

**Total test count**: 51 tests (50 passed, 1 skipped)
**EDDI tests**: 15 (100% passing)
**Coverage**: No regressions in existing indices

## Workflow Analysis

### Hybrid Full BMM: What Worked

| BMAD Phase | Used? | Value Delivered |
|------------|-------|-----------------|
| **Discovery** (`/bmad-bmm-generate-project-context`) | ✅ Yes | Compressed 2–3 hours of brownfield exploration into 5 minutes; established shared codebase understanding |
| **Requirements** (`/bmad-agent-bmm-pm`) | ✅ Yes | Scoped MVP (EDDI only, defer Palmer), set success criteria ("match NOAA reference"), aligned user expectations |
| **Technical Design** | ❌ No | Skipped — single-function implementation didn't justify separate design doc |
| **Implementation** | ✅ Manual | Direct Claude Code interaction; allowed rapid iteration during course correction |
| **Code Review** (`/bmad-bmm-code-review`) | ❌ No | Sourcery AI (external tool) handled review; BMAD review command not needed |

### What Was Skipped (and Why)

**Technical Design Document**: Skipped because:

- Single-function implementation (~200 lines)
- Algorithm already specified in NOAA Fortran reference
- No architectural decisions (data structures, class hierarchies, module boundaries)

**BMAD Code Review**: Skipped because:

- Sourcery AI (GitHub App) provided automated review faster than manual BMAD review invocation
- Sourcery feedback was actionable and comprehensive (4 items, all addressed)

### Workflow Fit Assessment

**Verdict**: Hybrid Full BMM was **well-suited** for this task.

**Strengths**:

1. **Discovery phase** — Essential for brownfield codebase; project context saved hours
2. **Requirements phase** — PRD scoped MVP cleanly; prevented scope creep (deferred Palmer Z-Index)
3. **Flexibility** — Skipping technical design avoided ceremony; clean branch strategy was improvisational but effective

**Weaknesses**:

1. **Agent role violation** — PM agent "John" implemented code instead of stopping after PRD; led to 3–4 hour detour
2. **Artifact loss** — `project-context.md` and `prd.md` lost during clean branch strategy; no guidance on committing BMAD artifacts
3. **No rollback guidance** — When implementation diverged from reference, no BMAD command suggested "revert to PRD and re-plan"

### Quick Flow Comparison

If **Quick Flow** (minimal BMAD structure) had been used:

- **Faster initial implementation** — Skip PRD, go straight to code (~1 hour saved)
- **Higher risk of scope creep** — Without PRD, user might have added Palmer Z-Index mid-stream
- **Lost discovery value** — No project context doc; would require manual codebase exploration each session
- **Same course correction cost** — Wrong turn would still occur (not a workflow issue, but algorithmic choice)

**Conclusion**: Full BMM's upfront structure (discovery + PRD) **paid off** by preventing scope creep and providing reusable context. The ~2 hours spent on BMAD phases was recouped by avoiding mid-stream requirement changes and faster iteration in later sessions.

## Lost Artifacts

### What Was Generated But Not Committed

| Artifact | Size (est.) | Content | Loss Impact |
|----------|-------------|---------|-------------|
| `_bmad/project-context.md` | ~150 lines | Architecture overview, key algorithms (SPI/SPEI), test patterns, dependencies, public API patterns | **High** — Future BMAD sessions must regenerate; ~5 minutes + agent cost per session |
| `_bmad/prd.md` | ~80 lines | Objective, user stories, technical requirements, acceptance criteria, scope boundaries | **Medium** — Scoping decisions lost; future maintainers can't see why Palmer was deferred |
| Session transcripts | ~50,000 tokens | Full conversation history including wrong turn, course correction rationale, Hastings derivation | **Low** — Captured in git commit messages and this retrospective |

### Why Artifacts Were Lost

The clean branch strategy (`feature/issue-414-eddi-clean`) used `git cherry-pick` to extract only production commits from the polluted original branch. BMAD artifacts in `_bmad/` were never committed to either branch, so they exist only in local working directory history.

### Recommendations for Future BMAD Usage

1. **Commit project context**: `git add _bmad/project-context.md && git commit -m "docs(bmad): add project context"` immediately after generation
2. **Commit PRD**: Same for `_bmad/prd.md` after PM agent delivery
3. **BMAD artifact gitignore strategy**: Use a **negative pattern** in `.gitignore`:
   ```
   _bmad/
   !_bmad/project-context.md
   !_bmad/prd.md
   ```
   This ignores session logs but preserves key documents.
4. **Stable location**: Consider `docs/bmad/` instead of `_bmad/` for committed artifacts (more discoverable, less "temporary" connotation)

## Lessons Learned

### 1. Reference Implementations as Ground Truth

**Observation**: The PM agent implemented a parametric approach (borrowed from SPI/SPEI patterns) without consulting the NOAA Fortran reference. This led to a 3–4 hour detour.

**Lesson**: When implementing algorithms from scientific literature, **read the reference code first**, even if it's in a different language (Fortran, C, R). Pattern-matching against similar functions in the codebase is unreliable when algorithms differ fundamentally (parametric vs. non-parametric).

**Actionable**: Before BMAD agents begin implementation, the PRD should include a "Reference Materials" section with links to:

- Academic papers (with equation numbers)
- Reference implementations (GitHub, NOAA, USGS)
- Validation datasets (if available)

Agents should be prompted: *"Have you consulted the reference implementation?"* before generating code.

### 2. Clean Branch Hygiene for AI-Assisted Development

**Observation**: The original `feature/issue-414-eddi` branch accumulated 845 files (226K insertions) due to BMAD exploration artifacts, agent subprocesses, and temporary files.

**Lesson**: AI-assisted development generates **high file churn** during exploration phases. Without discipline, branches become unreviewable.

**Actionable**:

- Use `.gitignore` aggressively for `_bmad/`, `.claude/`, `__pycache__/`
- Run `git status` frequently during BMAD sessions; commit only production code
- If a branch becomes polluted, use the **clean branch strategy** proactively (don't wait until PR time)
- Consider a pre-commit hook that rejects commits touching `_bmad/*` (except allowlisted files)

### 3. Agents Exceeding Role Boundaries

**Observation**: PM agent "John" (invoked via `/bmad-agent-bmm-pm`) delivered a PRD but then continued into implementation, producing incorrect code.

**Lesson**: BMAD agents have **role-specific expertise**. A PM agent is trained on requirements gathering, not algorithm implementation. When an agent exceeds its role, quality degrades.

**Actionable**:

- BMAD commands should **auto-terminate agents** after role completion (e.g., PM agent exits after PRD delivery)
- If continuation is desired, require explicit user approval: *"PM agent 'John' has delivered the PRD. Continue to implementation? (Y/N)"*
- Implement agent "handoff" protocol: PM agent writes PRD, then spawns a separate **implementation agent** (or returns control to Claude Code)

**BMAD Framework Recommendation**: Add a `--auto-exit` flag to agent commands:

```bash
/bmad-agent-bmm-pm --auto-exit  # Exit after PRD delivery
```

### 4. Automated Review Feedback Loops

**Observation**: Sourcery AI reviewed PR #597 within 2 minutes and found 4 actionable issues (1 bug risk, 3 test gaps). All were addressed in a follow-up commit, expanding test coverage from 9 → 15 tests.

**Lesson**: Automated review tools (Sourcery, SonarCloud, Snyk) provide **fast, consistent feedback** that complements human review. The feedback loop was ~10 minutes (review posted → issues fixed → re-push), far faster than human reviewer latency (hours/days).

**Actionable**:

- Integrate automated review into BMAD workflow: `/bmad-bmm-code-review` should optionally invoke external tools (Sourcery, CodeRabbit) in addition to AI-generated review
- Pre-commit hooks for **local Sourcery** analysis (before PR creation)
- CI/CD pipeline should **block merge** on Sourcery/SonarCloud failures (not just warnings)

**Impact**: The Sourcery review caught a **real bug** (missing calibration period validation). Without automated review, this could have caused runtime errors in production (e.g., user passes `calibration_year_initial=1950` but data starts in 2000).

### 5. Lost Artifacts Reduce BMAD Value Across Sessions

**Observation**: Project context and PRD were regenerated from scratch on Feb 5 because no artifacts were committed from the Feb 3 session.

**Lesson**: BMAD's value compounds **across sessions** if artifacts persist. Losing `project-context.md` meant re-exploration of the brownfield codebase (5+ minutes of agent time, duplicated effort).

**Actionable**:

- BMAD should **prompt to commit artifacts** after generation:
  ```
  ✅ Project context generated: _bmad/project-context.md
  📋 Recommendation: Commit this file for reuse across sessions.
     Run: git add _bmad/project-context.md && git commit -m "docs(bmad): add project context"
  ```
- Version control BMAD artifacts in **stable branches** (e.g., `main` or `develop`), not feature branches
- Add BMAD artifact status to `/bmad-help`: *"Project context: ❌ Not found. Run `/bmad-bmm-generate-project-context`."*

## Metrics Summary

| Metric | Value |
|--------|-------|
| **Issue age** | ~5 years (2021-01-31 → 2026-02-05) |
| **Active development time** | 3 days (Feb 3–5, 2026) |
| **Claude Code sessions** | ~15 EDDI-related |
| **BMAD commands used** | 4 (`/bmad-help`, `/bmad-bmm-generate-project-context`, `/bmad-agent-bmm-pm`, deferred `/bmad-bmm-code-review`) |
| **Original branch files changed** | 845 files (226,035 insertions, 8 deletions) |
| **Clean branch files changed** | 3 files (718 insertions, 1 deletion) |
| **Reduction ratio** | 282:1 (files), 314:1 (insertions) |
| **Final test count** | 15 (9 original + 6 from Sourcery review response) |
| **Test pass rate** | 100% (15/15 EDDI tests, 51/51 total suite) |
| **CI checks** | 12 total (all passing) |
| **Automated reviews** | Sourcery AI (4 items, all addressed), SonarCloud (passed) |
| **Time lost to wrong turn** | ~3–4 hours (parametric → non-parametric course correction) |
| **Time saved by BMAD discovery** | ~2–3 hours (vs. manual brownfield exploration) |
| **Net BMAD value** | Break-even to slight positive (~1 hour saved) |

## Conclusion

Issue #414 sat dormant for five years not because of technical complexity (EDDI is ~200 lines of NumPy code) but because of **activation energy**: understanding a brownfield codebase, scoping the work, and validating against a Fortran reference required sustained focus.

BMAD v6.0.0-Beta.7 provided the structure to break through that inertia:

- **Discovery phase** compressed weeks of codebase learning into minutes
- **Requirements phase** scoped the MVP cleanly and deferred distractions (Palmer Z-Index)
- **Hybrid workflow** balanced structure with speed (skipped technical design, used external review)

The implementation wasn't perfect — a PM agent exceeded its role and produced incorrect code — but the course correction process (reading NOAA Fortran reference, re-implementing with Hastings approximation, validating against `scipy`) resulted in a **scientifically accurate, production-ready implementation** that passed 12 CI checks and addressed all automated review feedback.

**Key Success Factors**:

1. **Clean branch strategy** distilled 845 files down to 3 reviewable changes
2. **Automated review loop** (Sourcery AI) caught bugs and test gaps within minutes
3. **Reference-driven validation** (Hastings test vs. `scipy`) ensured correctness
4. **Fail-fast philosophy** (calibration validation, insufficient climatology checks) prevented silent errors

**Remaining Challenges**:

1. **Agent role boundaries** need enforcement (PM agent should stop after PRD)
2. **Artifact persistence** must be solved (commit `project-context.md`, `prd.md`)
3. **BMAD guidance on course correction** is missing (what to do when implementation diverges from reference?)

From 5-year-old issue to merged PR in 3 days: BMAD delivered. With lessons learned and workflow refinements, the next dormant issue should be even faster.
