# climate_indices — Development Context

## Project overview
Python library for climate drought index computation (SPI, SPEI, PET, and more).
Scientific computing stack: NumPy, xarray/dask, scipy, structlog.
Repository: https://github.com/monocongo/climate_indices

## Current state
v2.3.0 (PR #614) merged — introduced xarray adapters, exception hierarchy,
structlog logging for SPI/SPEI/PET as a proof of concept. Three new workstreams
are now planned via BMAD v6 to extend this work across the full library.

## BMAD planning artifacts
All planning is complete (Phases 1–3). Read these before starting any work:
- PRD: _bmad-output/planning-artifacts/prd.md
- Architecture: _bmad-output/planning-artifacts/architecture.md
- Epics & Stories: _bmad-output/planning-artifacts/epics.md
- Sprint Status: _bmad-output/implementation-artifacts/sprint-status.yaml

## Coding conventions (from v2.3.0 — follow these exactly)
- Exception hierarchy rooted at ClimateIndicesError (see src/climate_indices/exceptions.py)
- structlog for all logging — never use stdlib logging
- xarray support via @overload + adapter pattern (see src/climate_indices/xarray_adapter.py)
- Type hints on all public functions
- Google-style docstrings with description, Args, and Returns sections
- Tests in tests/ using pytest, reference data in tests/fixtures/
- Conventional commits: feat:, fix:, test:, docs:

## BMAD Phase 4 — Story implementation cycle
For EACH story in your assigned epic, follow this cycle in order:

  1. CS (Create Story): run /bmad-bmm-create-story
     Prepares the next story from the sprint plan for development.

  2. VS (Validate Story — optional but recommended): run /bmad-bmm-create-story
     in validate mode to confirm the story is ready for development.

  3. DS (Dev Story): run /bmad-bmm-dev-story
     Execute the story implementation tasks and write tests.

  4. CR (Code Review): run /bmad-bmm-code-review
     Self-review the implemented code. If issues found, loop back to DS.

  5. Commit with a conventional commit message, then start over at CS
     for the next story.

  The cycle is: CS → VS → DS → CR → commit → next CS → ...

## Rules for Agent Team members
- Do NOT modify files outside your assigned workstream/epic without lead approval
- Run pytest on your changes before marking any task complete
- If you need a file owned by another teammate, message the lead — do not edit it yourself
- Commit after each completed story, not after each individual file change
- If you get stuck on an algorithm, message the lead with specifics
- Use /bmad-bmm-sprint-status to check overall sprint progress at any time

