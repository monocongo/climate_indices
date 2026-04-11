# Project Context — climate_indices

> Place this file at `_bmad-output/project-context.md` before running any BMAD
> workflow. Every agent in every fresh chat will inherit these conventions
> automatically.

---

## Project Identity

- **Package name:** `climate_indices`
- **Repository:** https://github.com/monocongo/climate_indices
- **PyPI:** https://pypi.org/project/climate-indices/
- **License:** BSD 3-Clause
- **Primary maintainer:** James Adams (@monocongo)
- **Current stable release:** 2.x
- **Active development target:** v2.5

---

## Tech Stack & Tooling

| Tool          | Purpose                                      | Notes                              |
|---------------|----------------------------------------------|------------------------------------|
| `uv`          | Package and virtual environment management   | Use for all installs and script runs (`uv run`, `uv sync`) |
| `ruff`        | Linting and formatting                       | Single tool replacing flake8/isort/black |
| `structlog`   | Structured logging                           | See logging conventions below      |
| `pytest`      | Test runner                                  | See testing conventions below      |
| `xarray`      | Primary array/dataset abstraction for users  | See xarray conventions below       |
| `numpy`       | Internal array operations                    |                                    |
| `cartopy`     | Geospatial plotting                          | Used in gallery scripts and notebooks |
| `matplotlib`  | Plotting                                     |                                    |
| `nbconvert`   | Notebook CI execution                        | `nbconvert --execute`              |
| `gh` CLI      | GitHub issue and PR management               | Must be authenticated before use   |
| `sqlalchemy`  | Database interaction where applicable        |                                    |

---

## Coding Conventions

### General

- All public functions must have complete type hints on all arguments and return
  values.
- Use Google-style docstrings on all functions with `Args` and `Returns`
  sections.
- Docstring descriptions and inline comments are capitalized only if they form a
  complete sentence.
- Maximum line length follows `ruff` project configuration (do not hardcode a
  value here; defer to `pyproject.toml`).
- Prefer explicit over implicit; avoid clever one-liners that reduce readability.

### Docstring Format

```python
def compute_spi(
    values: np.ndarray,
    scale: int,
    periodicity: str,
) -> np.ndarray:
    """Compute the Standardized Precipitation Index.

    Args:
        values: array of precipitation values, chronologically ordered.
        scale: number of time steps over which to compute accumulation.
        periodicity: temporal periodicity of the input data, e.g. 'monthly'.

    Returns:
        array of SPI values with the same shape as the input.
    """
```

### Imports

- Standard library → third-party → internal; separated by blank lines.
- No wildcard imports.

### Error Handling

- Raise specific, descriptive exception types (not bare `Exception`).
- Functions should raise rather than return `None` on failure.

---

## Structured Logging Conventions

- Use `structlog` exclusively. Do not use `loguru`, the stdlib `logging` module
  directly, or `print` statements for diagnostic output in library code.
- Bind contextual fields to the logger rather than interpolating them into
  message strings.
- Canonical context fields for this package:

| Field         | Type    | Example value         |
|---------------|---------|-----------------------|
| `index`       | `str`   | `"spi"`, `"pdsi"`    |
| `timescale`   | `int`   | `3`, `6`, `12`        |
| `periodicity` | `str`   | `"monthly"`           |
| `input_shape` | `tuple` | `(480, 120)`          |
| `data_var`    | `str`   | `"precip"`            |

**Correct pattern:**

```python
import structlog

log = structlog.get_logger()

log.info(
    "computing index",
    index="spi",
    timescale=scale,
    input_shape=values.shape,
)
```

**Incorrect — do not do this:**

```python
import logging
logging.info(f"Computing SPI at scale {scale} with shape {values.shape}")
```

---

## Testing Conventions

### Pytest Markers

| Marker        | Meaning                                                            |
|---------------|--------------------------------------------------------------------|
| `unit`        | Fast, no I/O, no network, no large fixtures                        |
| `integration` | May touch filesystem or external data; slower                      |
| `validation`  | Compares outputs against authoritative reference datasets; may be skipped pending fixture availability |

Run unit tests only:
```bash
uv run pytest -m "unit"
```

Run everything except validation (safe for CI on PRs):
```bash
uv run pytest -m "not validation"
```

Run validation suite separately:
```bash
uv run pytest -m "validation"
```

### Fixture Locations

| Type                          | Location                                 |
|-------------------------------|------------------------------------------|
| General test fixtures         | `tests/fixtures/`                        |
| EDDI literature examples      | `tests/fixtures/eddi_literature/`        |
| Palmer literature examples    | `tests/fixtures/palmer_literature/`      |
| Notebook sample data          | `notebooks/data/`                        |

### Fixture Size Limits

- Committed fixtures: prefer NetCDF or CSV, under 5 MB each.
- Larger reference datasets: downloaded on-demand in CI; never committed to the
  repo directly.
- For fixtures pending receipt from external sources (e.g., NOAA CPC), stub the
  test with:

```python
@pytest.mark.skip(reason="awaiting NOAA CPC fixtures — see VALIDATION.md")
@pytest.mark.validation
def test_eddi_noaa_cpc():
    ...
```

### Numerical Tolerance

- Default tolerance for validation tests: `atol=1e-3` unless the reference
  dataset or literature specifies otherwise.
- Document the chosen tolerance and its justification in the test module
  docstring, with a citation.

---

## xarray Conventions

- Public API functions should accept `xarray.DataArray` and `xarray.Dataset`
  where applicable, in addition to `numpy.ndarray`.
- Preserve coordinate metadata and attributes on outputs.
- Follow CF conventions for output variable attributes (`units`, `long_name`,
  `standard_name`, `valid_min`, `valid_max`).
- Use `xr.apply_ufunc` with `dask='parallelized'` for Dask-chunked array
  support where the underlying computation is element-wise or reducible.
- The xarray compatibility matrix lives at `docs/xarray_compatibility.md` and
  must be kept current when adding or modifying public functions.

---

## Branching Strategy

### Topology

```
main
└── release/v2.5                    ← integration target for all v2.5 work
    └── feature/e{epic}-{slug}      ← one short-lived branch per story
```

- `release/v2.5` is branched from `main` before any story work begins.
- It is merged back to `main` as a single PR when the milestone is complete and
  CI is green. That final PR is the canonical v2.5 release artifact.
- `feature/e{epic}-{slug}` branches are short-lived, one per story, merged via
  PR into `release/v2.5`, then deleted.

### Branch Naming

Story branches follow the pattern `feature/e{epic_number}-{short-slug}`,
mirroring the `epic:*` label taxonomy:

- Epic 1 (validation): `feature/e1-{slug}`
- Epic 2 (xarray): `feature/e2-{slug}`
- Epic 3 (docs): `feature/e3-{slug}`
- Infrastructure: `feature/s0-{slug}`

### Worktrees

Use one git worktree per active Claude Code session to allow parallel story
work without context bleed. Keep no more than 3–4 worktrees active at once.

```bash
git worktree add ../climate-indices-{slug} -b {branch_name} release/v2.5
```

### Branch Protection Rules

- **`main`:** require PR, require CI green, require at least one review,
  restrict direct push to maintainers.
- **`release/v2.5`:** require CI green on PRs; allow maintainer direct push for
  trivial fixes.
- **`feature/*`:** no protection; delete automatically on merge.

### Pending-Data Branches

Stories blocked on external data (e.g., NOAA CPC fixtures) should have their
branch opened immediately with stub test files committed, then parked. Do not
open a PR until the data arrives. Apply `status:pending-data` to the issue.

---

## GitHub Tracking Conventions

- **`_bmad-output/sprint-status.yaml` is the authoritative source of truth**
  for all epics, stories, and status.
- GitHub Issues are the public-facing tracking surface. They are generated
  programmatically from `sprint-status.yaml` via
  `scripts/create_github_issues.py` and are **not created or edited manually
  in bulk**.
- One-off issues for bugs or unplanned work outside v2.5 epics are still
  created manually in GitHub as normal.

### Required Fields in Each `sprint-status.yaml` Story Entry

| Field           | Description                                                        |
|-----------------|--------------------------------------------------------------------|
| `epic_label`    | one of `epic:validation`, `epic:xarray`, `epic:docs`              |
| `type_label`    | one of `type:literature`, `type:testing`, `type:notebook`, `type:infrastructure` |
| `branch_name`   | e.g. `feature/e1-eddi-literature`                                  |
| `blocked_on`    | optional; triggers `status:pending-data` label if set              |
| `github_issue`  | empty at creation; populated by `create_github_issues.py`          |

### Status Label Transitions

Status labels are applied manually by the contributor via `gh issue edit`
during the story workflow — they are not automated.

| Label                  | Applied when                                    |
|------------------------|-------------------------------------------------|
| `status:in-progress`   | Worktree opened, work begun                     |
| `status:in-review`     | PR opened against `release/v2.5`                |
| `status:pending-data`  | Story blocked on external data                  |
| `status:blocked`       | Blocked on a dependency other than data         |

### PR Convention

- PR title: `{conventional-commit-type}: {story title} (#{github_issue})`
- PR body must include `Closes #{github_issue}` to auto-close on merge.
- PRs target `release/v2.5`, not `main`.

### `create_github_issues.py` Script

This script is Story 0 of the build cycle — it is implemented before any epic
story begins and must be merged to `release/v2.5` before issue generation runs.
It is idempotent (skips stories where `github_issue` is already set) and
supports `--dry-run` and `--story {slug}` flags.

---

## CI Platform

- **GitHub Actions**
- Three test suites run as separate jobs:
  1. `unit` — runs on every PR to `release/v2.5` and `main`
  2. `integration` — runs on every PR to `release/v2.5` and `main`
  3. `validation` — runs as a separate workflow; skipped tests are reported but
     do not fail the job
- Notebooks are executed via `nbconvert --execute` in a dedicated CI job; any
  unhandled cell exception fails the job.
- `ruff check` and `ruff format --check` run on every PR.

---

## Key Reference Documents (created during v2.5)

| Document                          | Purpose                                               |
|-----------------------------------|-------------------------------------------------------|
| `VALIDATION.md`                   | Per-index validation status, tolerance criteria, known discrepancies |
| `docs/algorithm_refs/eddi.md`     | Citable algorithm spec for EDDI derived from literature |
| `docs/algorithm_refs/palmer.md`   | Citable algorithm spec for Palmer indices derived from literature |
| `docs/xarray_compatibility.md`    | xarray compatibility matrix for all public functions  |
| `llms-full.txt`                   | Machine-optimized single-file context document for AI tools |
| `llms.txt`                        | Companion summary; registered at site root            |
