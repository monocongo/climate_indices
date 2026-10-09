![Banner Image](https://raw.githubusercontent.com/monocongo/climate_indices/main/assets/multi-index-compare.png)

# climate_indices

[//]: # ([![Coverage Status]&#40;https://coveralls.io/repos/github/monocongo/climate_indices/badge.svg?branch=main&#41;]&#40;https://coveralls.io/github/monocongo/climate_indices?branch=main&#41;)
[//]: # ([![Codacy Status]&#40;https://api.codacy.com/project/badge/Grade/48563cbc37504fc6aa72100370e71f58&#41;]&#40;https://www.codacy.com/app/monocongo/climate_indices?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=monocongo/climate_indices&amp;utm_campaign=Badge_Grade&#41;)
[![Actions Status](https://github.com/monocongo/climate_indices/workflows/tests/badge.svg)](https://github.com/monocongo/climate_indices/actions)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python | 3.10-3.14](https://img.shields.io/badge/Python-3.10--3.14-blue?logo=python)](#supported-python-versions)

#### Python library of climate indices for drought, wildfire, and flood monitoring

`climate_indices` provides reference implementations of climate index algorithms that give a
geographical and temporal picture of the severity and duration of precipitation, temperature,
evaporative-demand, fire-weather, and wet-extreme anomalies for climate monitoring and research.
It began as a drought library; it now covers drought, wildfire, and flood potential, and an
energy family is in development.

### Indices

**Drought and moisture balance**

- [SPI](https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi),
  Standardized Precipitation Index, utilizing gamma and Pearson Type III distributions, with
  selectable placement of zero accumulations (`zero_handling`)
- [SPEI](https://www.researchgate.net/publication/252361460_The_Standardized_Precipitation-Evapotranspiration_Index_SPEI_a_multiscalar_drought_index),
  Standardized Precipitation Evapotranspiration Index, utilizing gamma, Pearson Type III, and generalized-logistic (log-logistic) distributions
- Standardized index (`indices.standardized_index()`), the SPI fitting pipeline for any
  non-negative series, such as runoff or streamflow for
  [SRI/SSI work](docs/standardized-hydrologic-indices.md)
- [PNP](http://www.droughtmanagement.info/percent-of-normal-precipitation/),
  Percentage of Normal Precipitation
- [PCI](https://www.tandfonline.com/doi/abs/10.1111/J.0033-0124.1980.00300.X), Precipitation Concentration Index
- [EDDI](https://psl.noaa.gov/eddi/), Evaporative Demand Drought Index
- [Palmer indices](https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf),
  including PDSI, PHDI, PMDI, Z-Index, and [scPDSI](https://doi.org/10.1175/1520-0442(2004)017%3C2335:ASPDSI%3E2.0.CO;2)

**Evapotranspiration**

- [PET](https://www.ncdc.noaa.gov/monitoring-references/dyk/potential-evapotranspiration), Potential Evapotranspiration, utilizing the [Thornthwaite](http://dx.doi.org/10.2307/21073),
  [Hargreaves](http://dx.doi.org/10.13031/2013.26773), or [FAO-56 Penman-Monteith](https://www.fao.org/4/x0490e/x0490e00.htm) equations

**[Wildfire](https://climate-indices.readthedocs.io/en/latest/wildfire_applications.html)** (`climate_indices.fire`)

- KBDI, the Keetch-Byram Drought Index
- CFFWIS, the Canadian Forest Fire Weather Index System: the FFMC, DMC, and DC moisture
  codes (with Drought Code overwintering) and the ISI, BUI, FWI, and DSR behavior indices
- Fosberg Fire Weather Index
- Hot-Dry-Windy Index
- Haines Index

**[Flood potential](https://climate-indices.readthedocs.io/en/latest/flood_applications.html)** (`climate_indices.flood`)

- Effective Precipitation (PE), the Effective Drought Index (EDI, distinct from EDDI), the
  Flood Index (I_F), and the Antecedent Precipitation Index (API): daily accumulated-wetness
  measures that indicate flood potential, not flooding
- `flood_events()`, which extracts events with onset, duration, peak, and severity from a
  daily flood index

**Energy (in development)**

- Degree days, standardized energy production and demand indices, and related
  energy-meteorology indices are planned under
  [#1157](https://github.com/monocongo/climate_indices/issues/1157); none ships in the
  package yet.

**Supporting tools**

- Run theory (`climate_indices.runs`): runs above or below a threshold with duration,
  magnitude, intensity, and peak, per series or per grid cell
- Distribution fit diagnostics (`fit_diagnostics()`): the fitted parameters, probability of
  zero, and Kolmogorov-Smirnov statistic per calendar step and cell
- Probability-scale output (`output_scale`): standardized indices as normal scores, fitted
  cumulative probabilities, or a bounded `[-1, 1]` scale

### Highlights

- **Rust backend.** An optional compiled extension (`climate_indices._native`)
  runs Rust ports of the numerical kernels behind SPI, SPEI, the standardized index, EDDI,
  PET, PNP, PCI, the Palmer family, the KBDI and CFFWIS moisture-code recurrences, and the
  flood family. It preserves the public API; a cross-backend parity registry checks every
  ported kernel against the Python reference at `rtol = atol = 1e-10` with matching NaN
  positions, not bit-for-bit equality; see the [parity contract](docs/architecture.md#parity-tolerance).
  Speedups vary by kernel: the flood family
  measured 1.2x to 6.7x faster than its Python path and Thornthwaite PET 1.4x, while
  Hargreaves measured at parity and Penman-Monteith slightly slower (0.89x) at the benchmark
  size; see [the measurements](benchmarks/README.md).
- **Vectorized gridded computation.** SPI, SPEI, EDDI, PNP, PET, and PDSI fit or rank a
  whole `(time, *cells)` block in one call instead of looping over cells. On a 38x87-cell,
  40-year reference grid that is 4.4x faster for SPI, 3.6x for SPEI, 53x for Thornthwaite
  PET, 342x for EDDI, and 111x for gridded PDSI. On fully populated monthly grids, gamma
  SPI/SPEI, EDDI, and PNP are bit-identical to per-cell computation; Thornthwaite PET and
  successful Pearson fits agree within `1e-12`. Pearson SPI's block-wide gamma fallback is
  not per-cell equivalent and can depend on chunk layout; daily and missing-data bounds
  are [documented separately](docs/performance.md#numerical-equivalence).
- **xarray, Dask, and Zarr.** Supported xarray entry points accept `xarray.DataArray` input,
  preserve coordinates, attach CF metadata, and stay lazy on Dask-backed arrays for
  chunk-by-chunk Zarr output. Temporal inference and core-dimension chunk requirements
  vary by function; `indices.standardized_index()` and scPDSI remain NumPy-only.
  See the [compatibility matrix](docs/xarray_compatibility.md),
  [writing NetCDF and Zarr outputs](docs/writing-outputs.md), the
  [Zarr/Dask notebook](notebooks/zarr_dask_spi_spei.ipynb), and
  [chunking and scheduler guidance](docs/performance.md).
- **Validation.** Documented evidence distinguishes external checks against NOAA, NRCan,
  SPEIbase, and literature reference data from regression-only coverage; see
  [Validation](#validation).
- **Command-line interface.** `climate_indices --index
  {spi,spei,pnp,scaled,pet,palmers,kbdi,pe,edi,flood_index,api,all}` computes indices from
  NetCDF inputs.

### Project goals

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
- [Climate Indices for Wildfire Applications](https://climate-indices.readthedocs.io/en/latest/wildfire_applications.html)
- [Climate Indices for Flood and Wet-Extreme Applications](https://climate-indices.readthedocs.io/en/latest/flood_applications.html)
- [__Validation status__](VALIDATION.md)
- [__Changelog__](CHANGELOG.md)
- [__License__](https://github.com/monocongo/climate_indices/blob/main/LICENSE)
- [__Disclaimer__](https://github.com/monocongo/climate_indices/blob/main/DISCLAIMER)

## Installation

This README describes 3.0.0, which is not yet published on PyPI. The released 2.x
package does not include the `fire`, `flood`, or `runs` modules or the Rust backend.
For 3.0.0 features, install from a [development checkout](docs/development-guide.md#installation).

Install the released 2.x package from PyPI:

```bash
pip install "climate-indices>=2.3"
```

`uv` users can run `uv pip install "climate-indices>=2.3"`. The 2.3.0 floor matches the
[quickstart](https://github.com/monocongo/climate_indices/blob/main/docs/quickstart.md),
which uses the xarray API added in 2.3.0; see
[Supported Python Versions](#supported-python-versions) for interpreter support.

Starting with 3.0.0, wheels for Linux (x86-64 and aarch64, manylinux_2_28, so glibc 2.28 or
newer), macOS (Apple silicon and Intel), and Windows (x86-64) include the optional Rust
acceleration backend; every other platform, including a Linux system with older glibc,
installs the pure-Python wheel. Both carry the same version and the same public
API; results agree within the [cross-backend parity tolerance](docs/architecture.md#parity-tolerance),
not necessarily bit for bit. Calling code need not depend on which one pip selected.
To see which one you have:

```bash
python -c "import climate_indices._native"   # succeeds when the extension is installed
```

A missing extension is not an error: `climate_indices` imports, and every computation
runs the Python implementation. Installing from source needs no Rust toolchain either —
`pip install --no-binary climate_indices climate_indices` installs from the sdist, which
builds the pure-Python wheel. Developers who
want the extension from a checkout build it with `uv run maturin develop --release`; see
[the Rust backend section of the architecture notes](docs/architecture.md#optional-rust-backend).

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
   `uv run mypy src/ tests/test_type_checking.py`
   `uv run pytest`
5. Open a PR into `main`.
6. The maintainer merges the PR after review and passing CI; agents never merge.

Use `feature/<topic>`, `fix/<topic>`, `docs/<topic>`, `chore/<topic>`,
`perf/<topic>`, `refactor/<topic>`, `test/<topic>`, `ci/<topic>`, or
`hotfix/<topic>` branch names. Release branches are avoided; use maintenance
branches only for approved older-version support.

The methodology behind this workflow — parallel agent sessions, worktree
isolation, and the evidence each change must carry — is documented in
[AI-assisted development](docs/ai-assisted-development.md).

## Release Recipe

Releases are tag-based. The Git tag, package version, GitHub Release, and PyPI
version must match.

- Git tag: `v1.2.3`
- Package version: `1.2.3`
- GitHub Release: `v1.2.3`
- PyPI release: `1.2.3`

1. Prepare a release PR that updates `pyproject.toml`, `CHANGELOG.md`,
   and release notes/docs; the maintainer merges it after review and passing
   CI.
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

All supported versions are tested on Linux (ubuntu-latest) for every pull request and push
to `main`. Python 3.14 is additionally tested on macOS on every event, and 3.10 on macOS on
pushes to `main`, merge groups, the weekly schedule, and manual dispatch. Both latest and
minimum declared dependency versions are tested in CI.

### Version Support Policy

This project provides **12 months notice** before dropping support for a Python version.
When a version approaches end-of-life, removal will be announced via the CHANGELOG and a
GitHub issue, and implemented no sooner than 12 months after announcement with a version bump.

Python 3.9 support was dropped in v2.2.0 (August 2025) due to `scipy>=1.15.3` requiring 3.10+.

### API Stability

| API Surface | Status | Guarantee |
|:------------|:------:|:----------|
| NumPy array functions (`indices.spi`, `indices.spei`, `indices.pet`, and the `flood` indices) | **Stable** | No breaking changes in minor versions |
| xarray DataArray entry points (any index function accepting `xr.DataArray`) | **Beta** | No breaking changes in patch versions |

**Stable API**: The NumPy-based computation functions follow strict semantic versioning.

**Beta API**: The xarray adapter layer provides automatic parameter inference, coordinate
preservation, CF metadata, and Dask support. While beta, computation results follow the
stable NumPy API's documented [numerical-equivalence bounds](docs/performance.md#numerical-equivalence)
— only the interface surface (parameter names, metadata attributes,
coordinate handling) may evolve. Beta features are marked in docstrings, with
``BetaFeatureWarning`` as their public warning category. The adapter stays Beta in 3.0.0
and is promoted no earlier than 3.1.0, once the 3.0.0 calendar alignment has a release
of soak time ([ADR-0012](docs/adr/0012-xarray-api-stays-beta-through-3.0.0.md)).

See `docs/xarray_compatibility.md` for the 3.0.0 compatibility matrix, including
Dask chunking constraints, metadata behavior, and the Palmer xarray adapter.
See `docs/performance.md` for a runnable parallel SPI/SPEI example and
the measured speedups behind the chunk and scheduler guidance.

## Validation

[`VALIDATION.md`](VALIDATION.md) classifies the evidence for the indices in its per-index
inventory, separating independent external validation from regression coverage and recording
their known gaps. Fosberg Fire Weather Index and Haines Index are not yet classified there.
`uv run pytest -m validation` runs the marked reference-fixture and specification checks.
FAO-56 worked-example checks run in the core suite via `uv run pytest tests/test_pm_eto.py`.
The evidence includes:

- **EDDI**: NOAA PSL monthly reference ET/EDDI for 1-, 3-, and 6-month timescales
  (1979–2023); maximum observed error `2.43e-6`.
- **SPI**: characterized against NOAA NCEI's operational climate-divisional SPI for all 344
  divisions and 7 timescales; zero placement compared with the R `SEI` and `SCI` packages.
- **SPEI**: reproduces SPEIbase v2.11 from its own CRU TS inputs (correlation
  0.9996–0.99999) and matches the R `SPEI` package's log-logistic fit within `1e-6`.
- **Standard Palmer** (PDSI, PHDI, PMDI, Z-Index): qualified external validation against
  NOAA NCEI nClimDiv for all 344 divisions, 1895–2022 (median absolute difference 0.0127).
- **scPDSI**: cross-validated against Wells-lineage reference fixtures (`atol=5e-5`).
- **CFFWIS**: all seven components match the NRCan `cffdrs` reference within `1e-9`.
- **KBDI**: reproduces the Keetch & Byram (1968) worked example.
- **HDW**: reproduces event timing and tracks the daily series in the Srock et al. (2018)
  Cedar Fire case; published magnitudes are characterization only, not numerical validation.
- **PET**: selected FAO-56 helper examples and the daily end-to-end Example 18, plus
  Thornthwaite and Hargreaves literature examples; monthly Example 17 is not encoded.
- **Flood family**: specification-level and regression coverage only; no external numeric
  oracle has been adopted yet.

The Rust backend has its own gate: the parity registry in `tests/parity_registry.py` runs
each ported entry point through the extension and through the Python implementation on
property-generated inputs, and fails if a kernel is added without an entry.

## Upgrading to 3.0.0

3.0.0 changes some computed values and exception types without a deprecation period, most
notably corrected daily calendar alignment in the xarray adapter, the PCI February fix, and
the gamma probability of zero now taken from the Calibration Period. It also removes the `spi`
console script in favor of `climate_indices --index spi`. Each change, how to detect it, and
what to change is in [`docs/deprecations/api-changes.md`](docs/deprecations/api-changes.md);
the full list is in the [CHANGELOG](CHANGELOG.md). Upgrading from 2.1.x or earlier also
crosses the 2.2.0 switch from `None` return tuples to exceptions.

## Citation
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
