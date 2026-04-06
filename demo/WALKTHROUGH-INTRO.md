# BMAD Workflow Demo — Terminal Walkthrough (10 Minutes)

> Companion to `DEMO-INTRO.md`. Every command below has been verified against the live repository.
> Run PRE-DEMO SETUP before the audience arrives.
> Presenter notes are in `# PRESENTER:` comments.

---

## PRE-DEMO SETUP

```bash
# Verify we're in the right repo
cd /Users/james.a/git/climate_indices
git fetch --all --prune

# Verify key branches exist
git branch -a | grep -E "v2.4.0-planning|eddi-clean|epic-[1-4]" | wc -l
# Expected: ~8-10 branches

# Verify planning artifacts
ls _bmad-output/planning-artifacts/ | grep -E "prd|architecture|epics"
# Expected: 3 files

# Verify sprint status
head -5 _bmad-output/implementation-artifacts/sprint-status.yaml
# Expected: generated date and project name

# Verify worktrees
git worktree list | head -6

# Check current branch
git branch --show-current
# Expected: feature/v2.4.0-planning
```

---

## SEGMENT 1: The Problem BMAD Solves (90 seconds)

```bash
# PRESENTER: "Don't open with the repo. Open with the problem."
# PRESENTER: "AI agents making contradictory decisions. No persistent reasoning.
#             Context windows collapsing. No concurrency."
# PRESENTER: "BMAD solves this with planning artifacts as shared context."
# PRESENTER: "Let me show you the artifacts."

# Show the planning artifact chain
ls -la _bmad-output/planning-artifacts/
# PRESENTER: "PRD, architecture, epics — three documents that constrain
#            everything every agent does. These live in the repo.
#            The git history is the audit trail."
```

---

## SEGMENT 2: The Four Phases, Concretely (2 minutes)

```bash
# PRESENTER: "Four phases. Discovery, Planning, Solutioning, Building."

# Phase 1 — Discovery: show the codebase scale
find src/climate_indices -name "*.py" | wc -l
# PRESENTER: "[N] Python modules. The discovery agent read all of them
#            and produced a project brief in minutes, not hours."

# Phase 2 — Planning: PRD executive summary
head -30 _bmad-output/planning-artifacts/prd.md
# PRESENTER: "31 functional requirements across 4 parallel tracks.
#            This document is what every agent reads before writing code."

# Phase 2 — Architecture: complexity assessment
grep -A 3 "Complexity level" _bmad-output/planning-artifacts/architecture.md
# PRESENTER: "The architecture document makes 5 explicit decisions —
#            naming conventions, module boundaries, testing strategy.
#            This is the shared context that prevents design conflicts."

# Phase 3 — Solutioning: show one story with clear before/after
grep -A 25 "^## Story 1.1:" _bmad-output/planning-artifacts/epics.md | head -30
# PRESENTER: "Story 1.1 — Exception Hierarchy Foundation.
#            Before: functions returned silent None values on failure.
#            After: 18 typed exception classes with actionable error messages.
#            The acceptance criteria tell the agent exactly what 'done' means."

# Show the acceptance criteria specifically
grep -A 15 "### Acceptance Criteria" _bmad-output/planning-artifacts/epics.md | head -18
# PRESENTER: "Checkboxes. Specific classes to create. Specific tests to write.
#            An implementing agent reads this and knows precisely what to build."

# Phase 3 — Critical path
grep -A 3 "Critical Path" _bmad-output/planning-artifacts/epics.md | head -5
# PRESENTER: "Critical path: Story 1.1 → 1.2 → 1.6 → 4.1 → 4.2 through 4.9.
#            The dependency graph is explicit. Not in anyone's head — in the document."

# Phase 4 — Building: show commit trail
git log --all --oneline --format="%h %ad %s" --date=short \
  | grep -iE "Story [1-4]\.[0-9]|feat|test" | head -12
# PRESENTER: "Each commit names its story. Full traceability from code change to requirement."
```

---

## SEGMENT 3: The Scale (90 seconds)

```bash
# PRESENTER: "Three BMAD iterations, approximately 3 weeks in February 2026."

# Total story count
grep -c "^  [0-9]" _bmad-output/implementation-artifacts/sprint-status.yaml
# PRESENTER: "38 stories across 5 epics. Each tracked."

# Sprint status — show the tracking structure
head -50 _bmad-output/implementation-artifacts/sprint-status.yaml
# PRESENTER: "Every story has a status. Backlog, ready-for-dev, in-progress, review, done.
#            This is what the orchestrator reads. This is what makes the screenshot possible."

# Commit count across the sprint window
git log --all --oneline --after="2026-01-25" --before="2026-02-28" \
  --format="%h %ad %s" --date=short | wc -l
# PRESENTER: "[N] commits in 5 weeks across all branches."

# Show the distribution size win (universally understandable)
# PRESENTER: "37 MB wheel down to 207 KB. Someone was shipping test fixtures
#            in the package. 180x reduction. No domain knowledge required to appreciate that."
```

---

## SEGMENT 4: The Parallel Agent Teams (3 minutes)

```bash
# PRESENTER: "Display the screenshot now."
# PRESENTER: "~/screenshots/Screenshot 2026-02-17 at 12.10.32 AM.png"
# PRESENTER: "This is a single moment during parallel agent execution.
#            Multiple agents, one orchestrator, isolated worktrees."

# Show the worktrees — these are LIVE on disk
git worktree list
# PRESENTER: "Each agent team works in its own isolated copy of the repo.
#            No merge conflicts during development. Clean separation."

# Show what T6 is — the critical path story
grep -A 20 "^## Story 1.6:" _bmad-output/planning-artifacts/epics.md | head -22
# PRESENTER: "T6 — Story 1.6, Palmer structured logging migration.
#            CRITICAL PATH. Blocks all Palmer advanced features.
#            This is what @canonical-dev-2 is implementing right now in the screenshot."

# Show what T21 is — running concurrently
grep -A 20 "^## Story 3.2:" _bmad-output/planning-artifacts/epics.md | head -22
# PRESENTER: "T21 — Story 3.2, NOAA Provenance Protocol.
#            Different track, different agent, different worktree, no shared dependencies.
#            This is what @eddi-dev-2 is implementing in parallel."

# Show why they're parallelizable — different tracks
echo "=== T6 is in Epic 1 (Track 0: Pattern Completion) ==="
grep "1-6-palmer-structlog" _bmad-output/implementation-artifacts/sprint-status.yaml
echo ""
echo "=== T21 is in Epic 3 (Track 2: Index Coverage) ==="
grep "3-2-noaa-provenance" _bmad-output/implementation-artifacts/sprint-status.yaml
# PRESENTER: "Different tracks, different files, different agents.
#            This parallelism was designed in the stories before any code was written."

# Show the epic branches — the agent output
echo "=== Epic 1: Canonical Patterns ==="
git log --oneline master..origin/feature/epic-1-canonical-patterns 2>/dev/null | head -5
echo ""
echo "=== Epic 3: EDDI/Coverage ==="
git log --oneline master..origin/feature/epic-3-eddi-pnp-scpdsi 2>/dev/null | head -5
echo ""
echo "=== Epic 4: Palmer (blocked until T6 completes) ==="
git log --oneline master..origin/feature/epic-4-palmer-multi-output 2>/dev/null | head -5
# PRESENTER: "Commits across 3 epic branches, running concurrently.
#            Each traceable to a story. Each story traceable to the PRD."

# Show the critical path enforcement
grep "Critical Path" _bmad-output/planning-artifacts/epics.md
# PRESENTER: "1.1 → 1.2 → 1.6 → 4.1 → 4.2-4.9. That's why @palmer-dev sits idle
#            with 87k tokens consumed. The gate hasn't opened yet.
#            This isn't a suggestion — it's enforced by the dependency design."
```

---

## SEGMENT 5: The Wrong Turn (90 seconds)

```bash
# PRESENTER: "Now the honest story. A 5-year-old issue. Two-line description."
# PRESENTER: "BMAD compressed discovery to minutes. PM agent scoped the MVP."
# PRESENTER: "Then the wrong turn."

# Show the retrospective overview
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md \
  | head -20
# PRESENTER: "5 years dormant. 3 days active development. 282:1 file reduction."

# Show the wrong turn
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md \
  | sed -n '/Phase 3: Implementation/,/Phase 4: Course/p' | head -25
# PRESENTER: "The agent pattern-matched to the nearest similar function in the codebase.
#            Three existing functions use statistical curve fitting.
#            The new feature requires empirical ranking. Wrong approach entirely."

# Show the lesson — verbatim from the committed retrospective
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md \
  | grep -A 5 "Reference Implementations as Ground Truth"
# PRESENTER: "Read the reference code first. Even if it's in Fortran.
#            Pattern-matching against the codebase is unreliable
#            when algorithms differ fundamentally."

# Show the metrics
git show origin/feature/issue-414-eddi-clean:docs/case-studies/eddi-bmad-retrospective.md \
  | sed -n '/Metrics Summary/,/Conclusion/p' | head -20
# PRESENTER: "845 files of exploration. 3 files of production code.
#            282:1 reduction. That's the real ratio of AI-assisted development.
#            Structure is the antidote to churn."
```

---

## SEGMENT 6: Lessons and Roadmap (1 minute)

```bash
# PRESENTER: "Three lessons."

# PRESENTER: "One: the spec is ground truth. Not the codebase, not conversation history.
#            The PRD and the architecture document. When those are precise, agents produce
#            correct code. When they're vague, agents improvise — and improvisations conflict."

# PRESENTER: "Two: stories designed for parallelism enable parallel execution.
#            T6 and T21 ran in parallel because someone designed them to.
#            Vague stories produce serial work. Precise dependency maps unlock concurrent teams."

# PRESENTER: "Three: the audit trail is the repo."

# Show the artifact chain one last time
ls _bmad-output/planning-artifacts/ _bmad-output/implementation-artifacts/ 2>/dev/null
# PRESENTER: "Every planning artifact committed. Every story references a commit.
#            Every retrospective in docs/. Legible to future humans and future agents."

git branch --show-current
# PRESENTER: "Still on the planning branch. All work is in the epic worktrees.
#            When it's ready, we merge. Structure first, velocity second."
```

---

## CLEANUP / RESET (after demo)

```bash
# All commands were read-only — nothing to clean up
git status
# Should show no changes from demo
```

---

## TROUBLESHOOTING

### If a branch reference doesn't resolve

```bash
# Fetch all remotes first
git fetch --all --prune
# Then retry the command
```

### If the retrospective branch is missing

The EDDI retrospective lives on `origin/feature/issue-414-eddi-clean`. If the remote reference is gone:

```bash
# Check if it exists locally
git branch -a | grep eddi-clean
# If not, the retrospective segment can be delivered from memory using the quotes in DEMO-INTRO.md
```

### If worktrees are missing

```bash
# Recreate if needed
git worktree add .worktrees/epic-1-canonical feature/epic-1-canonical-patterns
git worktree add .worktrees/epic-3-eddi feature/epic-3-eddi-pnp-scpdsi
git worktree add .worktrees/epic-4-palmer feature/epic-4-palmer-multi-output
```

### Screenshot location

`~/screenshots/Screenshot 2026-02-17 at 12.10.32 AM.png`

If missing, describe from DEMO-INTRO.md Segment 4 notes — the narrative carries without the visual.

---

## TIMING GUIDELINES

- Pre-demo setup: 2 minutes
- Segment 1: 90 seconds (mostly talking, 1 command)
- Segment 2: 2 minutes (5-6 commands, keep moving)
- Segment 3: 90 seconds (3 commands + narration)
- Segment 4: 3 minutes (core — take your time here)
- Segment 5: 90 seconds (3 commands from retrospective)
- Segment 6: 1 minute (mostly talking, 1-2 commands)
- **Total: 10 minutes**

If running long, compress Segment 2 (skip the `grep "Complexity level"` command) and Segment 3 (skip commit count). Do NOT cut from Segments 4 or 5.
