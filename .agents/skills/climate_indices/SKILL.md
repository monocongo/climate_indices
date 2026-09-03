---
name: climate-indices-conventions
description: "Repository conventions for the climate_indices Python scientific-computing library — enforces coding style, module boundaries, test placement, and Conventional Commit format. Use when changing climate_indices, writing its tests or docs, preparing commits, or running ruff/mypy validation and pytest tests."
---

# climate_indices Conventions

> Generated from [monocongo/climate_indices](https://github.com/monocongo/climate_indices) on 2026-08-08, then reviewed against this repository.

Use this skill when changing `climate_indices`, writing its tests or documentation,
or preparing commits.

## Project Layout and Module Boundaries

`climate_indices` is a Python library with a `src/` layout. Its production code
is the `src/climate_indices/` package; tests are direct pytest modules in
`tests/`, with reference data in `tests/fixture/`.

The package uses responsibility-focused modules:

| Area | Modules |
| --- | --- |
| Command-line interfaces | `__main__.py`, `__spi__.py` |
| Public APIs | `indices.py` (NumPy), `typed_public_api.py`, `xarray_adapter.py` |
| Climate-index computation | `compute.py`, `palmer.py` |
| PET and statistics | `eto.py`, `pm_eto.py`, `lmoments.py` |
| Supporting services | `cf_metadata_registry.py`, `exceptions.py`, `logging_config.py`, `performance.py`, `utils.py` |

Keep changes within the established module boundary. Read
`src/climate_indices/CONTEXT.md` before using project domain terminology.

## Python Style

| Element | Convention |
| --- | --- |
| Modules and files | `snake_case.py` |
| Functions and variables | `snake_case` |
| Classes | `PascalCase` |
| Constants | `SCREAMING_SNAKE_CASE` |
| Tests | `tests/test_<subject>.py` |

Follow the existing package-qualified import style:

```python
from climate_indices import compute
from climate_indices.exceptions import InvalidArgumentError
```

For an explicitly curated module API, use Python's `__all__`. Package-level
re-exports belong in `src/climate_indices/__init__.py`.

Public APIs must be typed and documented with Google-style docstrings. Use
`structlog` through `climate_indices.logging_config`; do not introduce stdlib
`logging`.

## Tests and Documentation

- Put pytest tests directly under `tests/` as `test_*.py`.
- Put shared fixtures in `tests/conftest.py` and stable reference data in
  `tests/fixture/`.
- Add or update tests for behavior changes and bug fixes.
- Update documentation for public API, behavior, or workflow changes.

A typical feature touches paths such as:

```text
src/climate_indices/<module>.py
tests/test_<module>.py
tests/fixture/<reference-data>    # when needed
docs/<document>                   # when public behavior changes
```

## Commit Conventions

Use Conventional Commit messages. Common types are:

- `feat`
- `fix`
- `docs`
- `test`
- `refactor`
- `chore`
- `ci`
- `build`
- `style`

Use a scope when it adds useful context, for example:

```text
feat: add rolling-window extreme Z-index sum with wet-side anomaly filter
fix(review): align integrity checks with repository standards
chore(deps): bump actions/checkout from 7.0.0 to 7.0.1
```

Keep the subject concise, descriptive, and focused on one coherent change.

## Validation

For source or test changes, run:

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

## Feature Workflow

1. Identify the responsible module and existing tests.
2. Implement the smallest coherent change in `src/climate_indices/`.
3. Add or update `tests/test_*.py` coverage and fixtures as needed.
4. Update user-facing documentation when behavior changes.
5. Run the relevant validation commands before review.
