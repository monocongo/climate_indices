# Agent Instructions

`climate_indices` is a Python library for climate drought indices, including
SPI, SPEI, PET, and the Palmer family. This is the portable project guidance
for all coding agents; tool-specific files must point here rather than copy it.

## Read for the task

- [Agent task map](docs/agent/README.md)
- [Core-library vocabulary](src/climate_indices/CONTEXT.md) before using domain
  terms; [CONTEXT-MAP.md](CONTEXT-MAP.md) also covers the planned Explorer.
- [Contributing workflow](CONTRIBUTING.md) for branches, PRs, and code style.
- [Validation scopes](VALIDATION.md) for scientific-validation work.
- [ADRs](docs/adr/) for non-obvious, hard-to-reverse decisions.

Preserve public behavior and follow the responsible module's established
patterns. Scope new conventions to new code; do not migrate unrelated legacy
code, planning artifacts, notebooks, or generated files.

## Validate source or test changes

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

For packaging, release, or workflow changes, also run:

```bash
uv run pytest tests/test_release_integrity.py
```

If you edit any document listed in `SUMMARY_FILES` or `FULL_FILES` in
`scripts/generate_llms_txt.py` — `README.md` and `VALIDATION.md` among them —
regenerate the derived bundles and commit them alongside the change:

```bash
uv run scripts/generate_llms_txt.py
```

Skipping this fails `tests/test_review_scripts.py` in every CI job.

## Releases

Releases are maintainer-owned and tag-based from `main`. Never create or push a
release tag without approval; use [the release runbook](docs/release-process.md).
