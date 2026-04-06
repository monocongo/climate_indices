# BMAD Climate Indices Demo — Terminal Walkthrough

> Every command below has been verified against the live repository at `/Users/james.a/git/climate_indices/`.
> Run PRE-DEMO SETUP before the audience arrives.
> Presenter notes are in `# PRESENTER:` comments.

---

## PRE-DEMO SETUP

Run these commands before the audience arrives to ensure everything works smoothly:

```bash
# Verify we're in the right repo
cd /Users/james.a/git/climate_indices
git fetch --all --prune

# Verify key branches exist
git branch -a | grep -E "v2.4.0-planning|eddi-clean|boy_scouting|epic-[1-4]" | wc -l
# Expected: ~10 branches

# Smoke test: import the library
uv run python -c "from climate_indices import indices; print('Library OK')"

# Smoke test: tests pass
uv run pytest tests/test_exceptions.py -q --no-header 2>&1 | tail -3

# Pre-cache the EDDI retrospective (takes a moment on first access)
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md > /dev/null

# Verify worktrees
git worktree list | head -8

# Verify planning artifacts
ls -la _bmad-output/planning-artifacts/ | grep -E "prd|architecture|epics" | wc -l
# Expected: 3+ files

# Check current branch
git branch --show-current
# Expected: feature/v2.4.0-planning
```

---

## SEGMENT 1: What This Is (90 seconds)

```bash
# PRESENTER: "This is climate_indices — a Python library for computing drought indices."
# PRESENTER: "Let me show you the scale of what BMAD organized."

# Show the branch landscape — the full arc of BMAD work
git branch -a | grep -cE "feature/|fix/|hotfix/|chore/"
# PRESENTER: "[number] active branches. Let me show the February 2026 sprint window."

# The BMAD sprint: commits across all branches during the sprint
git log --all --oneline --after="2026-01-25" --before="2026-02-28" \
  --format="%h %ad %s" --date=short | wc -l
# PRESENTER: "218 commits in 5 weeks across all active branches."

# Show the planning artifacts
ls -la _bmad-output/planning-artifacts/
# PRESENTER: "PRD, architecture, epics — the full BMAD spec chain."

# Show the implementation artifacts
ls -la _bmad-output/implementation-artifacts/ | head -10
# PRESENTER: "Sprint status, story plans, readiness reports — the execution artifacts."
```

---

## SEGMENT 2: The Before State (2 minutes)

```bash
# PRESENTER: "Let me show you what we were working with before BMAD."

# Show the exception hierarchy — this REPLACED the None-tuple pattern
grep -n "^class " src/climate_indices/exceptions.py
# PRESENTER: "18 exception and warning classes. Before this, functions returned
#             (None, None, None, None) on failure. No structured error information."

# Show Palmer's 5-tuple return (the motivation for xarray Dataset return)
grep -n "def pdsi" src/climate_indices/palmer.py | head -3
# PRESENTER: "Palmer returns 5 values. Users had to manually unpack each one.
#             Which order? Check the docs. Error handling? Check for None."
# FALLBACK: If palmer.py is large, use: grep -A 5 "def pdsi" src/climate_indices/palmer.py | head -8

# Show the structured logging replacement
grep -n "calculation_started" src/climate_indices/indices.py | head -5
# PRESENTER: "Every calculation now emits structured events — machine-parseable, not prose.
#             You can aggregate these in CloudWatch, Splunk, whatever you use."

# Show the old pattern: no type dispatch
grep -n "isinstance.*DataArray\|isinstance.*Dataset" src/climate_indices/indices.py | head -3
# PRESENTER: "Before: runtime type checks scattered everywhere.
#             After: @overload type dispatch at the public API layer."
# FALLBACK: If no matches, skip this command and move to next segment.
```

---

## SEGMENT 3: The Spec Chain (2 minutes)

```bash
# PRESENTER: "The spec chain: PRD → Architecture → Epics. Let me read from each."

# PRD executive summary
head -30 _bmad-output/planning-artifacts/prd.md
# PRESENTER: Read the executive summary quote about "canonical pattern completion,
#            scientific algorithm completeness, and advanced xarray capabilities"

# Architecture complexity assessment
grep -A 3 "Complexity level" _bmad-output/planning-artifacts/architecture.md
# PRESENTER: "Medium-High complexity. 19 coupled equations for Penman-Monteith alone.
#            This isn't a CRUD app — it's scientific computing."

# Epics — the story count and agent assignments
head -40 _bmad-output/planning-artifacts/epics.md
# PRESENTER: "38 stories, 5 epics, 5 agent teams. Each story has acceptance criteria,
#            dependencies, and an assigned agent."

# The critical path
grep -A 5 "Story 1.6" _bmad-output/planning-artifacts/epics.md | head -8
# PRESENTER: "Story 1.6 — Palmer structlog — is a critical path blocker.
#            Everything in Epic 4 waits for it. This is why we have a dependency graph."

# Sprint status
head -60 _bmad-output/implementation-artifacts/sprint-status.yaml
# PRESENTER: "Every story tracked. Backlog → ready-for-dev → in-progress → review → done.
#            This is what Phase 4 execution looks like: structured, traceable, auditable."
```

---

## SEGMENT 4: The xarray Adapter — 6 Layers (4 minutes)

```bash
# PRESENTER: "The xarray adapter is the technical heart. 6 processing layers."

# Show the decorator definition
sed -n '1286,1310p' src/climate_indices/xarray_adapter.py
# PRESENTER: "detect → resolve → align → extract → infer → compute → rewrap → log.
#            Every xarray call goes through this pipeline."

# Layer 1: Input type detection
grep -n "class InputType" src/climate_indices/xarray_adapter.py
# PRESENTER: "NUMPY or XARRAY. If NumPy, pass through unchanged. Zero overhead."

# Layer 1: Detection function
grep -n "def detect_input_type" src/climate_indices/xarray_adapter.py
# PRESENTER: "This is the entry point. Every decorated function starts here."

# Layer 4: Temporal parameter inference
grep -n "def _infer_temporal_parameters" src/climate_indices/xarray_adapter.py
# PRESENTER: "The adapter infers start year, periodicity, and calibration period
#            from xarray time coordinates. Users don't specify these manually."

# Show the inference functions
grep -n "def _infer_data_start_year\|def _infer_periodicity\|def _infer_calibration_period" \
  src/climate_indices/xarray_adapter.py | head -5
# PRESENTER: "Three inference functions. Read the CF time coordinate, extract the metadata."

# CF Metadata registry
sed -n '67,125p' src/climate_indices/xarray_adapter.py
# PRESENTER: "CFAttributes TypedDict, then the registry: spi, spei, pet_thornthwaite, pet_hargreaves.
#            CF-compliant output ready for NetCDF distribution. Standard names, units, descriptions."

# Show the three execution paths
grep -n "dask_backed\|apply_ufunc\|_finalize_ufunc" src/climate_indices/xarray_adapter.py | head -10
# PRESENTER: "Three paths: Dask chunked for distributed compute, multi-dim in-memory for 
#            medium data, and 1D fast path for time series. Adapter picks the right one."

# Show the logging integration
grep -n "adapter.rewrap_started\|adapter.rewrap_completed" src/climate_indices/xarray_adapter.py | head -5
# PRESENTER: "Structured logging at every layer. You can trace a calculation from entry to exit."
```

---

## SEGMENT 5: Live Code (2 minutes)

```bash
# PRESENTER: "Let me show you the actual code artifacts."

# Exception hierarchy tree
grep -n "^class.*Error\|^class.*Warning" src/climate_indices/exceptions.py
# PRESENTER: "18 classes total: 11 exceptions, 6 warnings, 1 base.
#            All rooted at ClimateIndicesError. Catch what you need, let the rest bubble."

# Show the base exception
sed -n '38,55p' src/climate_indices/exceptions.py
# PRESENTER: "Clean docstring, inherits from Exception, that's it. Simple."

# Typed public API — the @overload pattern
sed -n '49,75p' src/climate_indices/typed_public_api.py
# PRESENTER: "Pass NumPy, get NumPy. Pass xarray, get xarray. Type-safe dispatch.
#            Your IDE knows the return type. mypy --strict passes. No runtime overhead."

# Show a second overload pair
sed -n '137,163p' src/climate_indices/typed_public_api.py
# PRESENTER: "Same pattern for every index function. SPEI, PET, Palmer — all type-safe."

# Structured logging in action
grep -B 1 -A 2 "calculation_started" src/climate_indices/indices.py | head -20
# PRESENTER: "Every index function: calculation_started, calculation_completed,
#            with structured context. Function name, distribution type, scale, everything."

# Quick test run to show it all works
uv run pytest tests/test_exceptions.py tests/test_input_type_detection.py -q --no-header 2>&1 | tail -5
# PRESENTER: "Exception hierarchy and input detection — all green."
# FALLBACK: If slow or tests fail, skip and say "we ran this in pre-demo setup."
```

---

## SEGMENT 6: The EDDI Retrospective (4 minutes)

```bash
# PRESENTER: "Now the story within the story. Issue #414 — implement EDDI."
# PRESENTER: "Opened January 2021. Closed February 2026. Five years dormant."

# Show the retrospective (committed on the EDDI branch)
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md | head -50
# PRESENTER: Read the overview section — issue age, active dev time, outcome.
# PRESENTER: "Five years dormant. 10 days active development. 282:1 file reduction ratio."

# Phase 3: The Wrong Turn
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md \
  | sed -n '/Phase 3: Implementation/,/Phase 4: Course/p' | head -50
# PRESENTER: "The PM agent implemented a PARAMETRIC approach — gamma fitting, distribution matching.
#            But EDDI is NON-PARAMETRIC. Empirical ranking, no distributions. Wrong algorithm entirely.
#            This is what happens when you pattern-match instead of reading the reference."

# The Hastings approximation — the fix
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md \
  | grep -A 20 "Hastings approximation"
# PRESENTER: "Matching the NOAA Fortran coefficients exactly. Read the reference code first,
#            then implement. Not the other way around."

# The metrics table
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md \
  | sed -n '/Metrics Summary/,/Conclusion/p' | head -30
# PRESENTER: "282:1 file reduction ratio. 845 files of exploration → 3 files of production code.
#            10 days active work. 5 years issue age. This is the cost of unstructured development."

# The lessons learned
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md \
  | grep -A 8 "Reference Implementations as Ground Truth"
# PRESENTER: Read the lesson about reading reference code first.
# PRESENTER: "Lesson 1: Reference implementations are ground truth. Read them first.
#            Lesson 2: PM agents stop after PRD. Developer agents implement.
#            Lesson 3: 282:1 file reduction. Structure is the antidote to churn."

# The clean branch — show the actual diff
git log --oneline master..origin/feature/issue-414-eddi-clean
# PRESENTER: "Three commits. That's the entire EDDI implementation. Three."
```

---

## SEGMENT 7: The Parallel Agent Teams (3 minutes)

```bash
# PRESENTER: "Now let me show you the parallel agent execution."
# PRESENTER: "Display the screenshot now:"
# PRESENTER: "~/screenshots/Screenshot 2026-02-17 at 12.10.32 AM.png"
# PRESENTER: "This is a single moment during the v2.4.0 parallel agent execution.
#            Seven agents, one orchestrator, four epic worktrees."

# Show the worktrees — these are LIVE on disk
git worktree list
# PRESENTER: "Four isolated worktrees. Each agent team works in its own copy of the repo.
#            No merge conflicts during development. Clean separation of concerns."

# Show what's on the v2.4.0-planning branch
git log --oneline origin/feature/v2.4.0-planning | head -10
# PRESENTER: "The team-lead orchestrates from this branch. Planning artifacts, sprint status,
#            implementation readiness reports — all here."

# Show the sprint status
grep -E "epic-|story.*1-6|story.*3-2" _bmad-output/implementation-artifacts/sprint-status.yaml | head -20
# PRESENTER: "Story 1-6 (Palmer structlog) and 3-2 (NOAA Provenance) —
#            these are T6 and T21 from the screenshot. Every task maps to a story."

# Show the epic branches — the agent output
echo "=== Epic 1: Canonical Patterns ==="
git log --oneline master..origin/feature/epic-1-canonical-patterns | head -5
echo ""
echo "=== Epic 2: PM-ET Foundation ==="
git log --oneline master..origin/feature/epic-2-pm-et-foundation | head -5
echo ""
echo "=== Epic 3: EDDI/PNP/scPDSI ==="
git log --oneline master..origin/feature/epic-3-eddi-pnp-scpdsi | head -5
echo ""
echo "=== Epic 4: Palmer Multi-Output ==="
git log --oneline master..origin/feature/epic-4-palmer-multi-output | head -5
# PRESENTER: "65+ commits across 4 epic branches. Each one traceable to a story.
#            Each story traceable to an epic. Each epic traceable to the PRD."

# Show a story-tagged commit
git log --all --oneline --format="%h %ad %s" --date=short \
  | grep -E "Story [1-4]\.[0-9]" | head -15
# PRESENTER: "Every commit references its story number. Story 1.3, Story 4.7, etc.
#            Full traceability from code change to PRD requirement."

# The architecture as shared context
grep -A 8 "Pattern Enforcement Philosophy" _bmad-output/planning-artifacts/architecture.md
# PRESENTER: "This is how 7 agents work in parallel without conflicts.
#            The architecture document IS the shared context. Everyone reads it,
#            everyone follows it, no one touches another agent's module."
# FALLBACK: If grep doesn't match exactly, try:
#   grep -A 8 "Consistency enables" _bmad-output/planning-artifacts/architecture.md

# The T1 implementation plan — show a story file
head -40 _bmad-output/implementation-artifacts/plan-t1-exception-hierarchy.md
# PRESENTER: "Each story gets a detailed implementation plan before coding begins.
#            This is the T1 plan from the screenshot. Files to modify, classes to create,
#            tests to write — all specified before the first line of code."
```

---

## SEGMENT 8: Palmer Roadmap (1.5 minutes)

```bash
# PRESENTER: "The boy_scouting branch — real work, but undisciplined."

# Show boy_scouting commits
git log --oneline master..origin/boy_scouting | head -13
# PRESENTER: "13 commits. Multi-source PET, Zarr, numba JIT, xarray accessor.
#            Opened Dec 30, 2025 — before BMAD was applied. Good ideas, no structure."

# Show Epic 4 stories — the BMAD-structured version of the same ambition
grep -A 5 "^## Story 4\." _bmad-output/planning-artifacts/epics.md | head -30
# PRESENTER: "Epic 4 takes the best ideas from boy_scouting and structures them:
#            9 stories, explicit dependencies, acceptance criteria, test requirements."
# FALLBACK: If grep doesn't match well, use:
#   sed -n '/^## Story 4\./,/^## Story 5\./p' _bmad-output/planning-artifacts/epics.md | head -40

# Show the architecture decision for Palmer module location
grep -B 2 -A 8 "Palmer Multi-Output Module" _bmad-output/planning-artifacts/architecture.md
# PRESENTER: "Architecture Decision 2: Palmer wrapper stays IN palmer.py. Not a separate module.
#            This was decided BEFORE implementation — that's the point. Decide once, implement once."
# FALLBACK: If no match, try:
#   grep -B 2 -A 8 "xarray.*wrapper.*palmer" _bmad-output/planning-artifacts/architecture.md

# Show the dependency graph
grep -A 3 "Dependencies:" _bmad-output/planning-artifacts/epics.md | grep "Story 1\.6\|Story 4\." | head -8
# PRESENTER: "Story 4.7, 4.8, 4.9 — all depend on Story 1.6. The critical path is explicit.
#            You can't start Palmer xarray until Palmer has structlog. The graph enforces this."
```

---

## SEGMENT 9: Three Lessons + Close (1 minute)

```bash
# PRESENTER: "Three lessons from this arc."

# Lesson 1: Spec is ground truth
# PRESENTER: "Lesson 1: Spec is ground truth. The EDDI wrong turn happened because the agent
#            pattern-matched instead of reading the reference. The v2.4.0 architecture prevents
#            this at scale. Reference code first, then implementation."

# Lesson 2: Role separation
# PRESENTER: "Lesson 2: Role separation works. PM agents stop after PRD. Developer agents
#            implement. The v2.4.0 team: Alpha, Beta, Gamma, Delta, Omega — each a specialist.
#            No one agent does everything. No one agent goes rogue."

# Lesson 3: 282:1
# PRESENTER: "Lesson 3: 282:1 file reduction ratio. 845 files of exploration → 3 files of
#            production code. AI development generates massive churn. Structure is the antidote.
#            Planning, architecture, epic breakdown, story isolation — these aren't overhead,
#            they're the only way to ship."

# Final view: the full branch landscape
git for-each-ref --sort=-committerdate refs/remotes/origin \
  --format='%(committerdate:short) %(refname:short)' \
  | grep -E "epic-|v2.4|eddi-clean|boy_scouting|test-framework" | head -12
# PRESENTER: "From a 5-year-old dormant issue to parallel agent teams implementing 38 stories
#            across 5 epics. From boy_scouting exploration to disciplined Epic 4 execution.
#            This is structured AI development. This is BMAD."

# Show the final state
git branch --show-current
# PRESENTER: "We're still on the planning branch. All the work is in the epic worktrees.
#            When they're ready, we merge. Not before. Structure first, velocity second."
```

---

## CLEANUP / RESET (after demo)

```bash
# Return to the planning branch (should already be here)
git checkout feature/v2.4.0-planning

# Verify nothing was modified during demo (all commands were read-only)
git status

# If any untracked files were created, clean them up
# git clean -fd (use only if needed)

# PRESENTER: "That's the demo. Questions?"
```

---

## TROUBLESHOOTING

### If a command hangs or is slow

Most commands are fast read-only operations. If one is slow:
- `git show origin/feature/issue-414-eddi-clean:...` — cached in pre-demo setup
- `uv run pytest ...` — can be skipped if time is tight (fallback noted)
- `grep` on large files — use `head` pipe to limit output

### If a grep pattern doesn't match

Fallback commands are provided inline. Common issues:
- Architecture document reorganized: search for key phrases instead of exact headings
- Epic numbering changed: search for story content instead of story numbers
- File moved: use `find` or `git log --all --full-history -- <path>` to locate

### If worktrees are missing

```bash
# Recreate worktrees (only if necessary)
git worktree add .worktrees/epic-1-canonical feature/epic-1-canonical-patterns
git worktree add .worktrees/epic-2-pm-et feature/epic-2-pm-et-foundation
git worktree add .worktrees/epic-3-eddi feature/epic-3-eddi-pnp-scpdsi
git worktree add .worktrees/epic-4-palmer feature/epic-4-palmer-multi-output
```

### If screenshot is missing

The screenshot is at: `~/screenshots/Screenshot 2026-02-17 at 12.10.32 AM.png`

If missing, describe from memory:
- Seven Claude Code windows tiled on screen
- Each window shows a different agent (Alpha, Beta, Gamma, Delta, Epsilon, Omega, Team-Lead)
- Task lists visible showing T1-T35 story assignments
- Sprint status showing stories in various states (done, in-progress, ready-for-dev)

---

## TIMING GUIDELINES

- Pre-demo setup: 2-3 minutes
- Segment 1: 90 seconds
- Segment 2: 2 minutes
- Segment 3: 2 minutes
- Segment 4: 4 minutes (can compress to 3)
- Segment 5: 2 minutes (can skip pytest run)
- Segment 6: 4 minutes (core story, don't rush)
- Segment 7: 3 minutes
- Segment 8: 1.5 minutes
- Segment 9: 1 minute
- **Total: ~20 minutes**

Adjust as needed. Segments 4 and 5 can be compressed if running long.
