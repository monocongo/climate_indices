# Agent Instructions

This file is the canonical agent-facing workflow reference for
`climate_indices`. Tool-specific files such as `CLAUDE.md` should point here
rather than duplicating release or branching policy.

## Project context

`climate_indices` is a Python scientific computing library for climate drought
index computation, including SPI, SPEI, PET, Palmer indices, and related APIs.
The source tree uses a `src/` layout, pytest, Ruff, mypy, Hatchling, and `uv`.

## Trunk workflow

- `main` is trunk and should always be releasable.
- Start work from updated `main`.
- Use short-lived branches named `feature/<topic>`, `fix/<topic>`,
  `docs/<topic>`, `chore/<topic>`, or `hotfix/<topic>`.
- Open PRs into `main`.
- Merge only after CI passes.
- Avoid long-lived `release/*` branches except approved maintenance work.

## Validation

Run the normal validation gate for source or test changes:

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

## Coding conventions

- Follow existing module boundaries and local patterns.
- Keep public APIs typed and documented.
- Use Google-style docstrings for public functions.
- Use `structlog` for project logging; do not introduce stdlib logging.
- Keep tests in `tests/` and reference fixtures in `tests/fixture/`.
- Do not modify unrelated planning artifacts, notebooks, or generated files
  unless the task explicitly requires it.

## Release policy

Releases are maintainer-owned and tag-based from `main`.

- Tag format: `vX.Y.Z`
- Package version: `X.Y.Z`
- GitHub Release: `vX.Y.Z`
- PyPI version: `X.Y.Z`

Pushing a valid release tag triggers the release workflow and PyPI publishing
through Trusted Publishing/OIDC. Do not create or push release tags without
maintainer approval. Use `docs/release-process.md` as the release runbook.
