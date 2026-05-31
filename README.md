![Banner Image](https://raw.githubusercontent.com/monocongo/climate_indices/main/assets/Global_Monthly_SPI.jpg)

# climate_indices

[//]: # ([![Coverage Status]&#40;https://coveralls.io/repos/github/monocongo/climate_indices/badge.svg?branch=main&#41;]&#40;https://coveralls.io/github/monocongo/climate_indices?branch=main&#41;)
[//]: # ([![Codacy Status]&#40;https://api.codacy.com/project/badge/Grade/48563cbc37504fc6aa72100370e71f58&#41;]&#40;https://www.codacy.com/app/monocongo/climate_indices?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=monocongo/climate_indices&amp;utm_campaign=Badge_Grade&#41;)
[![Actions Status](https://github.com/monocongo/climate_indices/workflows/tests/badge.svg)](https://github.com/monocongo/climate_indices/actions)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/climate-indices)

#### Python library of indices useful for climate monitoring

This project contains Python implementations of various climate index algorithms which provide
a geographical and temporal picture of the severity and duration of precipitation and temperature
anomalies useful for climate monitoring and research.

The following indices are provided:

- [SPI](https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi),
  Standardized Precipitation Index, utilizing both gamma and Pearson Type III distributions
- [SPEI](https://www.researchgate.net/publication/252361460_The_Standardized_Precipitation-Evapotranspiration_Index_SPEI_a_multiscalar_drought_index),
  Standardized Precipitation Evapotranspiration Index, utilizing both gamma and Pearson Type III distributions
- [PET](https://www.ncdc.noaa.gov/monitoring-references/dyk/potential-evapotranspiration), Potential Evapotranspiration, utilizing either [Thornthwaite](http://dx.doi.org/10.2307/21073)
  or [Hargreaves](http://dx.doi.org/10.13031/2013.26773) equations
- [PNP](http://www.droughtmanagement.info/percent-of-normal-precipitation/),
  Percentage of Normal Precipitation
- [PCI](https://www.tandfonline.com/doi/abs/10.1111/J.0033-0124.1980.00300.X), Precipitation Concentration Index
- [EDDI](https://psl.noaa.gov/eddi/), Evaporative Demand Drought Index
- [Palmer indices](https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf),
  including PDSI, PHDI, PMDI, and Z-Index

This Python implementation of the above climate index algorithms is being developed
with the following goals in mind:

- to provide an open source software package to compute a suite of
  climate indices commonly used for climate monitoring, with well
  documented code that is faithful to the relevant literature and
  which produces scientifically verifiable results
- to provide a central, open location for participation and collaboration
  for researchers, developers, and users of climate indices
- to facilitate standardization and consensus on best-of-breed
  climate index algorithms and corresponding compliant implementations in Python
- to provide transparency into the operational code used for climate
  monitoring activities at NCEI/NOAA, and consequent reproducibility
  of published datasets computed from this package
- to incorporate modern software engineering principles and scientific programming
  best practices


This is a developmental/forked version of code that was originally developed by NIDIS/NCEI/NOAA. 
See [drought.gov](https://www.drought.gov/drought/python-climate-indices).

- [__Documentation__](https://climate-indices.readthedocs.io/en/latest/)
- [__License__](https://github.com/monocongo/climate_indices/blob/main/LICENSE)
- [__Disclaimer__](https://github.com/monocongo/climate_indices/blob/main/DISCLAIMER)

## Developer Workflow

This project uses trunk-based development. `main` is the trunk and should always
be releasable.

1. Start from current trunk:
   `git switch main && git pull --ff-only origin main`
2. Create a short-lived branch:
   `git switch -c feature/<short-topic>`
3. Make focused changes with tests.
4. Run validation:
   `uv run ruff check src/ tests/`
   `uv run ruff format --check src/ tests/`
   `uv run mypy src/`
   `uv run pytest`
5. Open a PR into `main`.
6. Merge only after CI passes.

Use `feature/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<topic>`, or
`hotfix/<topic>` branch names. Release branches are avoided; use maintenance
branches only for approved older-version support.

## Release Recipe

Releases are tag-based. The Git tag, package version, GitHub Release, and PyPI
version must match.

- Git tag: `v1.2.3`
- Package version: `1.2.3`
- GitHub Release: `v1.2.3`
- PyPI release: `1.2.3`

1. Prepare and merge a release PR that updates `pyproject.toml`, `CHANGELOG.md`,
   and release notes/docs.
2. Confirm `main` is green.
3. Create an annotated tag from `main`.
4. Push the tag. The release workflow builds, validates, publishes to PyPI, and
   creates the GitHub Release.

Tag creation and publishing require maintainer approval. See
[`docs/release-process.md`](docs/release-process.md) for the full checklist.

### Maintainer Quick Commands

Read-only preflight:

```bash
git status --short
git branch --show-current
git log --oneline --decorate -5
uv run pytest tests/test_release_integrity.py
```

Safe PR branch setup:

```bash
git switch main
git pull --ff-only origin main
git switch -c chore/issue-667-release-docs
```

Approval-required release tag commands:

```bash
git switch main
git pull --ff-only origin main
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

## Supported Python Versions

| Python Version | Status | Notes |
|:--------------:|:------:|:------|
| 3.10 | Supported | Minimum supported version |
| 3.11 | Supported | |
| 3.12 | Supported | |
| 3.13 | Supported | |
| 3.14 | Supported | Latest supported version |

All versions are tested on Linux (ubuntu-latest). Python 3.10 and 3.14 are additionally
tested on macOS. Both latest and minimum declared dependency versions are tested in CI.

### Version Support Policy

This project provides **12 months notice** before dropping support for a Python version.
When a version approaches end-of-life, removal will be announced via the CHANGELOG and a
GitHub issue, and implemented no sooner than 12 months after announcement with a version bump.

Python 3.9 support was dropped in v2.2.0 (August 2025) due to `scipy>=1.15.3` requiring 3.10+.

### API Stability

| API Surface | Status | Guarantee |
|:------------|:------:|:----------|
| NumPy array functions (`indices.spi`, `indices.spei`, `indices.pet`) | **Stable** | No breaking changes in minor versions |
| xarray DataArray functions (`spi()`, `spei()`, `pet_thornthwaite()`, `pet_hargreaves()`) | **Beta** | No breaking changes in patch versions |

**Stable API**: The NumPy-based computation functions follow strict semantic versioning.

**Beta API**: The xarray adapter layer provides automatic parameter inference, coordinate
preservation, CF metadata, and Dask support. While beta, computation results are **identical**
to the stable NumPy API — only the interface surface (parameter names, metadata attributes,
coordinate handling) may evolve. Beta features are tagged with ``BetaFeatureWarning`` and
marked in docstrings.

See `docs/xarray_compatibility.md` for the v2.5 compatibility matrix, including
Dask chunking constraints, metadata behavior, and the current Palmer xarray
workflow.

### Validation Notes

The v2.5 validation status is tracked in `VALIDATION.md`. EDDI has executable
NOAA PSL reference tests, but those tests skip unless the external
`tests/fixture/noaa-eddi-{1,3,6}month/` datasets are present. Palmer tests cover
the committed regression fixtures for PDSI, PHDI, PMDI, and Z-Index; those
fixtures are not treated as independent authoritative reference outputs because
their provenance identifies them as generated by this library.

## Migration Guide for v2.2.0

**Breaking Change: Exception-Based Error Handling**

Version 2.2.0 introduces a significant architectural improvement in error handling. The library now uses exception-based error handling instead of returning `None` tuples for error conditions.

### What Changed

**Before (v2.1.x and earlier):**
```python
# Old behavior - functions returned None tuples on failure
result = some_internal_function(data)
if result == (None, None, None, None):
    # Handle error case
    pass
```

**After (v2.2.0+):**
```python
# New behavior - functions raise specific exceptions
try:
    result = some_internal_function(data)
except climate_indices.compute.InsufficientDataError as e:
    # Handle insufficient data case
    print(f"Not enough data: {e.non_zero_count} values found, {e.required_count} required")
except climate_indices.compute.PearsonFittingError as e:
    # Handle fitting failure case
    print(f"Fitting failed: {e}")
```

### New Exception Hierarchy

- `DistributionFittingError` (base class)
  - `InsufficientDataError` - raised when there are too few non-zero values for statistical fitting
  - `PearsonFittingError` - raised when L-moments calculation fails for Pearson Type III distribution

### Impact on Users

- **Direct API users**: No changes needed - the public SPI/SPEI functions handle exceptions internally
- **Library integrators**: If you were checking for `None` return values from internal functions, update to use try/catch blocks
- **Benefits**: More informative error messages, better debugging, and automatic fallback from Pearson to Gamma distribution when appropriate

### Code Quality Improvements

Version 2.2.0 also addresses floating point comparison issues (`python:S1244`) throughout the codebase:

**Floating Point Comparisons:**
```python
# ❌ OLD: Direct equality checks (unreliable)
if values == 0.0:
    handle_zero_case()

# ✅ NEW: Safe comparison using numpy.isclose()
if np.isclose(values, 0.0, atol=1e-8):
    handle_zero_case()
```

**Benefits:**
- Eliminates floating point precision issues in statistical parameter validation
- Improves test reliability and numerical robustness
- Follows scientific computing best practices for floating point arithmetic
- See `docs/floating_point_best_practices.md` for comprehensive guidelines

#### Citation
You can cite `climate_indices` in your projects and research papers via the BibTeX 
entry below.
```
@misc {climate_indices,
    author = "James Adams",
    title  = "climate_indices, an open source Python library providing reference implementations of commonly used climate indices",
    url    = "https://github.com/monocongo/climate_indices",
    month  = "may",
    year   = "2017--"
}
```
