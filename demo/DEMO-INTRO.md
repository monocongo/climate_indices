# BMAD Workflow Demo: 10-Minute Presenter Narrative

**Companion to the full 20-minute demo (`DEMO.md`). This version requires zero domain knowledge from the audience.**

---

## Timing Reference

| Segment | Topic | Duration |
|---------|-------|----------|
| 1 | The Problem BMAD Solves | 90 seconds |
| 2 | The Four Phases, Concretely | 2 minutes |
| 3 | The Scale: What Three Iterations Produced | 90 seconds |
| 4 | The Parallel Agent Teams | 3 minutes |
| 5 | The Honest Story: The Wrong Turn | 90 seconds |
| 6 | Lessons and Roadmap | 1 minute |
| **Total** | | **10 minutes** |

---

## Segment 1: The Problem BMAD Solves (90 seconds)

*Do NOT open with the repo. Open with the problem.*

The problem is not "AI writes bad code." The problem is what happens when you point an AI coding agent at a real codebase and say "improve it." You get:

- **Contradictory decisions across sessions.** Session A chooses pattern X. Session B doesn't know about Session A. Session B chooses pattern Y. Now the codebase has both, and neither agent documented why.

- **No persistent record of why.** Conversation history evaporates. The reasoning behind a design decision disappears the moment the session ends.

- **Context window collapse.** As the codebase grows, agents can't hold enough of it in working memory to make coherent decisions. Local optimizations that contradict the global design.

- **No concurrency.** One agent, one session, one task at a time. For a project with 38 implementation tasks, that's serial execution — weeks instead of days.

BMAD solves this with **structured planning artifacts that become the shared context** for every agent, every session, every parallel worker. The artifacts live in the repo. The git history is the audit trail. No agent needs to remember anything — it reads the spec.

---

## Segment 2: The Four Phases, Concretely (2 minutes)

*Walk the four BMAD phases using this repo as the example. Keep domain details thin — the phases are what matter.*

The project is `climate_indices` — a 7-year-old, 385-star Python library used in production at NOAA for drought monitoring. The domain doesn't matter for this demo. What matters is the workflow.

**Phase 1 — Discover.** An analyst agent reads the entire codebase and produces a project brief. For a brownfield codebase with 14 modules and 9,800 lines of source code, this compressed hours of manual exploration into minutes.

**Phase 2 — Plan.** A PM agent produces a Product Requirements Document. An architect agent produces an architecture document. These two documents constrain everything that follows. The PRD defined 31 functional requirements across 4 parallel tracks. The architecture document made 5 explicit decisions — naming conventions, module boundaries, testing strategy — that every implementing agent would follow.

**Phase 3 — Solutionize.** A PO agent produces epics and stories *after* architecture — a key improvement in BMAD v6. Each story has explicit acceptance criteria. Here's what one looks like:

> *Show Story 1.1 from the terminal walkthrough.*

Before: functions returned silent `None` values on failure. After: 18 typed exception classes with actionable error messages. The acceptance criteria specify exactly what classes to create, what context attributes to include, what tests to write. An implementing agent reads this story and knows precisely what "done" means.

**Phase 4 — Build.** A dev agent implements one story per session. The session reads the story file and the architecture document. Nothing else. Each commit message names the story it implements. Full traceability from code change to requirement.

---

## Segment 3: The Scale: What Three Iterations Produced (90 seconds)

*Introduce the repo through outcomes only. No code. Numbers that require zero domain knowledge.*

Three BMAD iterations over approximately 3 weeks in February 2026. Here's what that produced:

- **38 stories across 5 epics**, each with acceptance criteria, dependency maps, and agent assignments
- **Test count: 250 to 703** — nearly tripled the test suite
- **A 5-year-old dormant issue closed in 3 days** — opened January 2021, merged February 2026
- **Distribution size: 37 MB down to 207 KB** — someone had been shipping test fixtures in the package wheel. A 180x reduction.
- **Zero breaking changes** across all of it. Every existing user's code still works.

> *Show `sprint-status.yaml` from the terminal. Every story tracked: backlog, ready-for-dev, in-progress, review, done. This is what structured execution looks like — inspectable, auditable, universally readable.*

---

## Segment 4: The Parallel Agent Teams (3 minutes)

*This is the strongest moment in the demo. Display the screenshot: `~/screenshots/Screenshot 2026-02-17 at 12.10.32 AM.png`*

This screenshot shows something that doesn't exist anywhere else in AI development discourse: **multiple named specialist agents working in parallel on the same codebase, orchestrated by a team lead.**

Let me walk through what you're seeing.

**The orchestrator** — `team-lead` — sits at the top. It dispatches tasks, monitors progress, enforces dependency gates, and tracks token budget. It has consumed only 291 tokens of its own context. It doesn't write code. It manages.

**Named specialist agents:**

- `@canonical-dev-2` (145k tokens) — editing `tests/test_exceptions.py`. It's implementing **T6: Story 1.6, Palmer structlog Migration**. This is on the critical path. Every Palmer advanced feature waits for this.

- `@eddi-dev-2` (146k tokens) — writing `tests/fixture/palmer/provenance.json`. It's implementing **T21: Story 3.2, NOAA Provenance Protocol Establishment** — a metadata standard for reference datasets.

- `@palmer-dev` (87k tokens, **idle**) — consumed tokens doing preparatory work but is now **blocked**. It cannot start Epic 4 until T6 completes. The team-lead enforces this: *"Epic 4 gate opens when T6 completes."*

**Why T6 and T21 can run in parallel:** They were *designed* to have no shared dependencies. T6 is in Epic 1 (Track 0 — pattern completion). T21 is in Epic 3 (Track 2 — index coverage). Different tracks, different files, different agents. This parallelism was specified in the stories before a single line of code was written.

**Why `@palmer-dev` is blocked:** The critical path is `1.1 → 1.2 → 1.6 → 4.1 → 4.2-4.9`. Story 1.6 (Palmer structured logging) must complete before Story 4.1 (Palmer advanced feature handoff) can begin. This isn't a prompt instruction — it's enforced by the dependency design in the epics document.

> *Show `sprint-status.yaml` in the terminal — the orchestrator's state file.*
> *Show the Story 1.6 file — what T6 is actually implementing.*
> *Show the Story 3.2 file — what T21 is running concurrently.*

**Git worktrees** — each agent team works in an isolated filesystem copy of the repo. No merge conflicts during implementation. The code paths don't touch until PR time:

```
.worktrees/epic-1-canonical/   → @canonical-dev-2
.worktrees/epic-3-eddi/        → @eddi-dev-2
.worktrees/epic-4-palmer/      → @palmer-dev (blocked, waiting)
```

**Token budget:** 101,394 / 200,000 used. The team-lead is making resource allocation decisions at 60% utilization — headroom for the remaining stories.

The point to land: **the stories were designed for parallelism before a single line of code was written.** That design decision — which stories can run concurrently, which are gated — lives in the BMAD artifacts, not in any agent's judgment during implementation.

---

## Segment 5: The Honest Story: The Wrong Turn (90 seconds)

*Use the retrospective but strip out all domain terminology. The lesson is universal.*

A GitHub issue sat open for 5 years. Two-line description: "add support for this computation." BMAD compressed the exploration phase to minutes. A PM agent scoped the MVP. The success criterion was explicit: *"match the government reference output."*

Then the wrong turn. The dev agent read the existing codebase and pattern-matched to the nearest similar function. It implemented the new feature using the same statistical approach as three existing features. That approach was fundamentally wrong for this feature — the existing functions use parametric curve fitting; the new one requires non-parametric empirical ranking.

From the committed retrospective:

> *"Pattern-matching against similar functions in the codebase is unreliable when algorithms differ fundamentally."*

Course correction: read the actual reference implementation — a Fortran program from NOAA. Rewrite. Validate. Three days total. 845 files of exploration distilled into 3 files of production code — a 282:1 reduction ratio.

The lesson is not about this specific algorithm. The lesson is: **the spec is ground truth, not the codebase.** When the existing code and the spec diverge, the agent chose the code. That's the failure mode. Write specs that agents can't ignore.

---

## Segment 6: Lessons and Roadmap (1 minute)

Three lessons:

1. **The spec is ground truth.** Not the codebase, not the conversation history, not the most recent agent's reasoning. The PRD and the architecture document. When those are precise, agents produce correct code. When they're vague, agents improvise — and improvisations conflict.

2. **Stories designed for parallelism enable parallel execution.** The agent teams feature is a multiplier on story quality. Vague stories produce serial work because dependencies are implicit. Precise, dependency-mapped stories unlock concurrent teams. T6 and T21 ran in parallel because someone designed them to.

3. **The audit trail is the repo.** Every planning artifact is committed. Every story references a commit. Every retrospective is in `docs/`. The work is legible to future humans and future agents alike.

Roadmap: the Palmer advanced features and remaining index coverage land next under the same approach — and the new index gets advanced data format support nearly for free because the adapter pattern was specified in the architecture before any story was written.

---

*Demo complete. 10 minutes.*
