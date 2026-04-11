# climate_indices — Development Context

## Project overview
Python library for climate drought index computation (SPI, SPEI, PET, and more).
Scientific computing stack: NumPy, xarray/dask, scipy, structlog.
Repository: https://github.com/monocongo/climate_indices

## Current state
v2.4.0 — all planned epics complete. xarray adapters, exception hierarchy,
structlog logging, PM-ET, EDDI/PNP/scPDSI coverage, and Palmer multi-output
are fully implemented and tested.

## BMAD development tooling

The BMAD spec-driven workflow (`_bmad/`) is **not included in the repository**
because not all contributors need it. Install it locally if you plan to use
the spec-driven approach for new features:

```
# Install BMAD framework locally (Claude Code plugin)
/plugin install bmad@claude-plugins-official
```

When BMAD is used for a feature, commit its output artifacts under `_bmad-output/`
so the planning record is preserved in version history. The `_bmad/` framework
directory itself remains local-only and gitignored.

## Coding conventions (from v2.3.0 — follow these exactly)
- Exception hierarchy rooted at ClimateIndicesError (see src/climate_indices/exceptions.py)
- structlog for all logging — never use stdlib logging
- xarray support via @overload + adapter pattern (see src/climate_indices/xarray_adapter.py)
- Type hints on all public functions
- Google-style docstrings with description, Args, and Returns sections
- Tests in tests/ using pytest, reference data in tests/fixtures/
- Conventional commits: feat:, fix:, test:, docs:

## BMAD story implementation cycle (when BMAD is in use)

For each story, follow this cycle:

  1. CS (Create Story): `/bmad-bmm-create-story`
  2. VS (Validate Story — optional): confirm story is ready
  3. DS (Dev Story): `/bmad-bmm-dev-story` — implement and write tests
  4. CR (Code Review): `/bmad-bmm-code-review` — loop back to DS if issues found
  5. Commit with a conventional commit message, then start the next CS

  Cycle: CS → VS → DS → CR → commit → next CS → …

  Output artifacts (stories, sprint status, planning docs) go in `_bmad-output/` and are committed.

