# Agent Instructions

This file is the canonical agent-facing workflow reference for
`climate_indices`. Tool-specific files such as `CLAUDE.md` should point here
rather than duplicating release or branching policy.

## Project context

`climate_indices` is a Python scientific computing library for climate drought
index computation, including SPI, SPEI, PET, Palmer indices, and related APIs.
The source tree uses a `src/` layout, pytest, Ruff, mypy, Hatchling, and `uv`.

Domain vocabulary (SPI, SPEI, Timescale, Calibration Period, Palmer family,
etc.) is defined once in [`src/climate_indices/CONTEXT.md`](src/climate_indices/CONTEXT.md).
Start there before using project-specific terms in code, docs, or commits —
it is opinionated about canonical names vs. terms to avoid. See
[`CONTEXT-MAP.md`](CONTEXT-MAP.md) if you also need the planned Explorer
context. Architecturally significant decisions (hard to reverse, non-obvious)
are recorded as ADRs in [`docs/adr/`](docs/adr/).

## Trunk workflow

- `main` is trunk and should always be releasable.
- Start work from updated `main`.
- Use short-lived branches named `feature/<topic>`, `fix/<topic>`,
  `docs/<topic>`, `chore/<topic>`, or `hotfix/<topic>`.
- Open PRs into `main`.
- Merge only after CI passes.
- Avoid long-lived `release/*` branches except approved maintenance work.

## Git safety (non-negotiable)

- Never commit directly on `main` or push directly to `origin/main`.
- Interpret “commit and push” as: create or update an appropriately named
  short-lived branch, commit there, push that branch to `origin`, then open a
  PR into `main`.
- If work begins on `main`, create the branch before making the task commit.
- If work begins on `main` with uncommitted changes, inspect them first.
  Stop and ask for confirmation before committing or publishing pre-existing
  or unrelated changes.
- Before every push, inspect the destination ref and commits to be published.
  Stop and ask for confirmation if the branch includes pre-existing or
  unrelated commits.
- A direct `origin/main` push requires explicit maintainer approval that names
  `origin/main` after the agent has explained that it bypasses the PR workflow.
- Never force-push or rewrite `main` without explicit maintainer approval.

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
