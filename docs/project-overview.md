# Project Overview

## Summary

**climate_indices** is a Python library that implements published climate and hydrologic indices. It covers standardized drought indices, the Palmer indices, evapotranspiration estimates, fire-weather indices, flood-potential indices, energy-meteorology indices, power-system reliability indices, and compound-hazard statistics.

It computes indices from data you supply, for NumPy arrays and `xarray.DataArray` objects, and it can process large grids lazily with Dask. Index computation makes no network request; the optional `aws` available-water-capacity ingest fetches third-party soil rasters, so treat those tiles as untrusted input. It does not quality-control data, and it does not produce forecasts.

The code is a fork of work originally developed by NIDIS/NCEI/NOAA for drought monitoring (see [Licensing](licensing.md)). This repository is a scientific product, not an official communication of NOAA; see the `DISCLAIMER` file in the repository root.

### Project status

- **Maturity**: the PyPI classifier is "Development Status :: 5 - Production/Stable". The API stability tiers (Stable and Beta) are listed in the repository `README.md` and in the [compatibility matrix](xarray_compatibility.md).
- **Releases**: see [PyPI](https://pypi.org/project/climate-indices/) for published Python releases and the repository `CHANGELOG.md` for what each one changed.
- **Python support**: `requires-python = ">=3.11,<3.16"` on `main`. The classifiers on a published release show the versions it supports.
- **License**: BSD-3-Clause for the project. The Rust routines include code derived from other projects, including code under the Boost Software License 1.0, so the distribution metadata reads `BSD-3-Clause AND BSL-1.0`. See [Licensing](licensing.md).
- **Documentation**: [Read the Docs](https://climate-indices.readthedocs.io/)
- **Repository**: [GitHub](https://github.com/monocongo/climate_indices)

## What the library computes

| Family | Indices | Scope note |
|:--|:--|:--|
| Standardized drought indices | SPI, SPEI, SPAI (Beta), the generic `standardized_index()`, PNP, PCI, EDDI | SPI, SPEI, and SPAI fit a distribution to a calibration period and convert each value to a standard-normal score. EDDI ranks evaporative demand instead of fitting a distribution. |
| Palmer indices | PDSI, PHDI, PMDI, Z-Index, and the self-calibrating PDSI (scPDSI, Wells et al. 2004) | scPDSI is available through the NumPy API and the CLI. |
| Evapotranspiration | Thornthwaite, Hargreaves-Samani, and FAO-56 Penman-Monteith (`climate_indices.pet_penman_monteith`) | Hargreaves-Samani and FAO-56 Penman-Monteith estimate reference evapotranspiration (ETo). Lower-level Penman-Monteith helpers in `drought/pm_eto.py` are internal; the fire-weather indices use them. |
| Fire weather (`climate_indices.fire`) | KBDI, the Canadian Forest Fire Weather Index System (FFMC, DMC, DC, ISI, BUI, FWI, DSR), Fosberg FFWI, Hot-Dry-Windy (HDW), Haines | These are fuel-dryness and fire-weather indices. They are not fire-behavior or fire-spread predictions. |
| Flood potential (`climate_indices.flood`) | Effective Precipitation (PE), Effective Drought Index (EDI), Flood Index, Antecedent Precipitation Index (API), flood-event extraction | These are accumulated-wetness measures that indicate flood potential. They do not model flooding. EDI (flood) and EDDI (drought) are different indices. |
| Energy (`climate_indices.energy`) | Heating, cooling, and energy degree days; hub-height wind speed; wind and PV capacity factors; standardized energy indices (SREPI, demand index, SRLI); energy-drought events | These describe the climate-driven part of supply and demand, not actual generation or load. |
| Reliability (`climate_indices.reliability`) | SAIFI, SAIDI, CAIDI, MAIFI, MAIFI_E, ASAI, outage counts, Major Event Day filtering | These are IEEE Std 1366 distribution reliability indices computed from an interruption log. They are not resilience metrics. |
| Compound hazards (`climate_indices.compound`) | Joint counts, likelihood multiplication factor, conditional probabilities, event matching, block bootstrap | Counts and probabilities describe co-occurrence. They make no causal claim. Block-bootstrap p-values are supported only within the scope that `VALIDATION.md` in the repository root characterizes. |
| Supporting tools | Run theory (`climate_indices.runs`), distribution fit diagnostics (`fit_diagnostics()`), probability-scale output (`output_scale`) | Run theory extracts events above or below a threshold, with duration, magnitude, intensity, and peak. |

Application guides are available for selected families: [wildfire](wildfire_applications.md), [flood](flood_applications.md), [energy](energy_applications.md), [reliability](reliability_applications.md), and [compound hazards](compound-hazards.md).

## Inputs, outputs, and options

- **Array types**: NumPy arrays, `xarray.DataArray`, and Dask-backed DataArrays. Most DataArray index routes keep coordinates, attach CF metadata, and stay lazy on Dask input. Some routes are eager, such as `energy.energy_drought_events()` on a one-dimensional series and the compound event catalogs, which reject Dask input. Some functions need a single chunk along `time`. The [compatibility matrix](xarray_compatibility.md) lists the constraints per function. The reliability indices take an interruption log instead of an array.
- **Periodicity**: monthly and daily series for most indices. Among the standardized indices, only the empirical routes (`standardized_index()`, `energy.srepi()`, `energy.standardized_demand_index()`, `energy.srli()`) accept hourly series. Cadence-independent elementwise routes, such as `fire.haines_index()`, also accept them. Weekly series are unsupported. See [Input Data Requirements](data_requirements.md).
- **Time scales**: for the standardized indices, monthly data accept scales of 1 to 72 months, and daily data accept scales of 1 to 2,196 days.
- **Distributions**: SPI uses gamma or Pearson Type III. SPEI also accepts log-logistic (generalized logistic) and generalized extreme value (GEV). `standardized_index()` also accepts empirical and beta. SPAI uses a t location-scale, Pearson Type III, or normal fit to the pooled precipitation anomalies. See [Choosing Parameters](choosing-parameters.md) and [Choosing a Distribution and Checking the Fit](checking-distribution-fit.md).
- **Calibration period**: the period used to fit the distribution. On the xarray routes it defaults to the full record. The NumPy functions, such as `drought.indices.spi()`, require both calibration years. A period shorter than 30 years triggers `ShortCalibrationWarning`; a single eager series that contains missing values also raises `InsufficientDataError` when its calibration window holds fewer than 30 years of non-NaN data. Gridded and Dask-backed inputs bypass that preflight.
- **Output scale**: `output_scale` returns standard-normal scores (the default), fitted cumulative probabilities, or a bounded `[-1, 1]` scale. `zero_handling` selects how a zero accumulation is placed within the probability of zero.
- **Command line**: a single `climate_indices` command computes indices from NetCDF files. See the [CLI reference](reference.md#command-line-interface).

## Scientific basis and evidence

Each index implements a published method: a journal article or an agency technical report. Examples are McKee et al. (1993) for SPI, Vicente-Serrano et al. (2010) for SPEI, Thornthwaite (1948) and Hargreaves and Samani (1985) for evapotranspiration, Palmer (1965) and Wells et al. (2004) for the Palmer indices, and Allen et al. (1998), FAO-56, for Penman-Monteith. The [algorithm descriptions](algorithms.md) and the [algorithm references](algorithm-reference.md) cite the sources and record where this implementation makes a documented choice.

Evidence differs by index, and the repository records the difference rather than asserting validation everywhere. `VALIDATION.md` in the repository root classifies each index as externally validated, reproduced from a published example, characterized on synthetic data, or covered by regression and specification tests only. For example, EDDI is compared with NOAA PSL reference values, SPEI with SPEIbase and the R `SPEI` package, and the standard Palmer indices with NOAA NCEI nClimDiv. The flood family and SPAI have no matched observed-input reference yet. On synthetic inputs, the tests compare the Antecedent Precipitation Index (API) recurrence with the R package `ahrapi`.

The optional Rust backend has a separate check. A parity registry runs each ported kernel through the Rust extension and the Python implementation on generated inputs and compares the results within a stated tolerance. See [Architecture](architecture.md#parity-tolerance).

## Goals

- Provide open-source implementations of climate indices commonly used for climate monitoring and research, with code documented against the relevant literature.
- Make the evidence for each implementation visible, including its known gaps.
- Keep the algorithms reproducible, so that results computed from this package can be repeated by others.
- Provide an open place for researchers, developers, and users of climate indices to collaborate and compare implementations.
- Follow current scientific-software practice: typed public APIs, automated tests, and documented decisions ([architecture decision records](adr/index.md)).

## Typical uses

The library supplies indices of the kind used as inputs to:

- drought monitoring and historical drought analysis from user-supplied precipitation, temperature, or evapotranspiration data
- agricultural and water-resource studies that use standardized drought indices as one input
- daily flood-potential screening and fire-weather assessment
- event analysis, such as drought, wet, and fire-weather spells, using run theory
- climate-risk studies of energy supply and demand, and of power-system reliability
- studies of how hazards co-occur

Operational decisions need more than an index value. Check the evidence in `VALIDATION.md` and the input requirements before relying on a result.

## Project classification

### Repository structure

- **Type**: single repository containing a Python package and two Rust crates
- **Primary language**: Python 3.11 or later, with an optional Rust backend
- **Build system**: Hatchling (PEP 517) for the Python package; maturin builds the Rust extension in the published wheels
- **Dependency management**: uv (`uv.lock`)

### Package layout

- `climate_indices.drought` holds the NumPy implementations (`drought.indices`, `drought.palmer`, `drought.eto`, `drought.pm_eto`, `drought.self_calibration`) and re-exports the typed drought functions that are also available at the package root. Shared `compute`, `vocabulary`, and `lmoments` modules stay at the root. See [ADR-0027](adr/0027-drought-namespace-and-module-migration.md) for the module moves and the temporary public-module aliases.
- `climate_indices.fire`, `.flood`, `.energy`, `.reliability`, and `.compound` are namespaced subpackages. Fire-weather indices are exposed only under `climate_indices.fire` ([ADR-0005](adr/0005-fire-module-api.md)), and flood-potential indices only under `climate_indices.flood` ([ADR-0013](adr/0013-flood-module-api-and-naming.md)).
- `crates/climate-py` holds the PyO3 bindings that build the `climate_indices._native` extension. `crates/climate-indices` is the standalone pure-Rust crate of numerical kernels.

### Architecture pattern

The library is layered:

```
┌─────────────────────────────────────┐
│   CLI Layer                         │  ← __main__.py, _cli.py, _cli_output.py,
│                                     │     _cli_transport.py
├─────────────────────────────────────┤
│   Public API Layer                  │  ← typed_public_api.py, xarray_adapter.py,
│                                     │     cf_metadata_registry.py, drought/, fire/,
│                                     │     flood/, energy/, reliability/, compound/
├─────────────────────────────────────┤
│   Computation Layer                 │  ← drought/indices.py, compute.py, drought/palmer.py,
│                                     │     drought/self_calibration.py, drought/eto.py,
│                                     │     drought/pm_eto.py, runs.py, validation.py
├─────────────────────────────────────┤
│   Math/Statistics Layer             │  ← lmoments.py, drought/_palmer_duration.py,
│                                     │     drought/_palmer_pdi.py, drought/_palmer_wells.py,
│                                     │     optional Rust kernels via _native_backend.py
├─────────────────────────────────────┤
│   Infrastructure Layer              │  ← utils.py, logging_config.py,
│                                     │     exceptions.py, performance.py, provenance.py
└─────────────────────────────────────┘
```

See [Architecture](architecture.md) for the detailed design.

### Entry points

The installed script names `climate_indices` and `process_climate_indices` both run `__main__.main`, the only CLI entry point. The `--index` choices are `spi`, `spei`, `pnp`, `scaled`, `pet`, `palmers`, `kbdi`, `pe`, `edi`, `flood_index`, `api`, `hdd`, `cdd`, `edd`, `wind_profile`, `srepi`, `demand_index`, `srli`, and `all`. The former specialized `spi` CLI (parameter save and load) was retired in 3.0.0. To reuse a NumPy fit, call `fit_diagnostics()` once at the same `scale` and calibration period you will standardize with. Pass its `.parameters` dict as `fitting_params` and its `.distribution` as `distribution`. A Pearson Type III request that fell back to gamma returns gamma parameters and reports `Distribution.gamma`, so pass that distribution; the same parameters under Pearson raise `ValueError`. The `.parameters` dict includes `prob_zero`, so the zero mass is reused too. The xarray route returns an `xr.Dataset`, not a `FitDiagnostics`. Pass its parameter variables as a `fitting_params` dict of NumPy arrays, and select the family that `distribution_used` reports. `compute.gamma_parameters()` and `compute.pearson_parameters()` return plain tuples of arrays, not a `fitting_params` dict, and `gamma_parameters()` omits `prob_zero`. Parameters fitted at one scale silently mis-standardize a call at another; see [pre-computing the fit](workflow-examples.md#pre-compute-spi-distribution-fitting-variables).

### Technology stack

| Category | Technology |
|----------|-----------|
| **Core language** | Python 3.11+ |
| **Scientific computing** | NumPy, SciPy, xarray, Dask, pandas, cftime, h5netcdf |
| **Optional acceleration** | Rust (PyO3 bindings, built with maturin) |
| **Logging** | structlog (structured logging) |
| **Testing** | pytest, hypothesis (property-based), pytest-benchmark |
| **Type checking** | mypy (strict) |
| **Linting and formatting** | ruff |
| **Build** | Hatchling (PEP 517) |
| **CI/CD** | GitHub Actions (`unit-tests-workflow.yml`, `release.yml`, `benchmarks.yml`, `bandit.yml`, `crate-release.yml`) |
| **Containerization** | Docker (Python 3.14-slim base) |
| **Documentation** | Sphinx with Read the Docs hosting |

## Quick reference

### Installation

```bash
# from PyPI
pip install climate-indices

# from source (development mode with uv)
git clone https://github.com/monocongo/climate_indices.git
cd climate_indices
uv sync --group dev
```

### Core API usage

#### xarray API

```python
import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import spi
from climate_indices.vocabulary import Distribution

# 40 years of synthetic monthly precipitation; replace with xr.open_dataarray("precip.nc")
time = pd.date_range("1980-01-01", periods=480, freq="MS")
precip = xr.DataArray(np.random.default_rng(0).gamma(2.0, 40.0, 480), coords={"time": time}, dims="time")

# 6-month SPI; the start year and periodicity are inferred from the time coordinate
spi_6 = spi(
    precip,
    scale=6,
    distribution=Distribution.gamma,
    calibration_year_initial=1981,
    calibration_year_final=2010,
)
```

#### NumPy API

> Note: the calibration-window boundary parameters are `calibration_year_initial`/
> `calibration_year_final` across the Python API. `percentage_of_normal` still
> accepts the older `calibration_start_year`/`calibration_end_year` keywords as
> deprecated aliases (removed in 4.0.0), and the CLI accepts
> `--calibration_year_initial`/`--calibration_year_final` as aliases for its
> `--calibration_start_year`/`--calibration_end_year` flags.

```python
import numpy as np

from climate_indices.drought import indices
from climate_indices.vocabulary import Distribution, Periodicity

# 40 years of synthetic monthly precipitation, starting in January 1980
precip_mm = np.random.default_rng(0).gamma(2.0, 40.0, 480)

spi_6 = indices.spi(
    precip_mm,
    scale=6,
    distribution=Distribution.gamma,
    data_start_year=1980,
    calibration_year_initial=1981,
    calibration_year_final=2010,
    periodicity=Periodicity.monthly,
)
```

### CLI usage

```bash
# compute SPI at 1-, 3-, 6-, and 12-month scales from a gridded NetCDF file
climate_indices \
    --index spi \
    --periodicity monthly \
    --scales 1 3 6 12 \
    --netcdf_precip precip.nc \
    --var_name_precip prcp \
    --calibration_start_year 1981 \
    --calibration_end_year 2010 \
    --output_file_base results/spi
```

### Development commands

```bash
# run tests
uv run pytest

# run tests with coverage
uv run pytest --cov=src --cov-report=term

# type checking (Python 3.12, the stack the CI lint job uses)
uv run --python 3.12 mypy src/ tests/test_type_checking.py

# linting and formatting
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# run benchmarks (deselected by default)
uv run pytest -m benchmark
```

See [CONTRIBUTING.md](https://github.com/monocongo/climate_indices/blob/main/CONTRIBUTING.md) for the full list of checks a pull request must pass.

## Contributor and AI-agent reference

### Recommended starting points

1. **Computation**: start with `docs/architecture.md`, then `src/climate_indices/compute.py` and `src/climate_indices/drought/indices.py`.
2. **Public API**: read `src/climate_indices/typed_public_api.py` (strict typing) and `src/climate_indices/xarray_adapter.py`.
3. **CLI**: read `src/climate_indices/__main__.py` (the index pipeline registry and `--index` handling) and `src/climate_indices/_cli.py` (shared argument registration).
4. **Palmer calculations**: read `src/climate_indices/drought/palmer.py`, `drought/_palmer_pdi.py`, `drought/_palmer_wells.py`, `drought/_palmer_duration.py`, and `drought/self_calibration.py`.
5. **Fire-weather indices**: read `src/climate_indices/fire/__init__.py` and the subsystem design in `docs/design/fire-subsystem.md`.
6. **Flood-potential indices**: read `src/climate_indices/flood/__init__.py` (public dispatch) and the kernels in `flood/_pe.py`, `flood/_edi.py`, `flood/_if.py`, and `flood/_antecedent.py`.
7. **Run theory**: read `src/climate_indices/runs.py`.
8. **Testing patterns**: read `tests/conftest.py` (fixtures), `tests/test_xarray_adapter.py` (xarray API), and `tests/test_property_based.py` (invariants).
9. **Error handling**: read `src/climate_indices/exceptions.py` (the exception hierarchy and its attributes).

### Architectural invariants

- **Time-dimension chunking**: SPI/SPEI and the stateful fire adapters reject split Dask time chunks. PET adapters rechunk split time internally. See the [chunking guidance](./xarray_compatibility.md#chunking-guidance-for-gridded-indices).
- **Calibration period**: the recommended minimum is 30 years. A shorter period triggers `ShortCalibrationWarning`.
- **Pearson Type III fitting**: a calendar step needs at least four non-zero calibration values (`compute.MIN_NON_ZERO_VALUES_FOR_PEARSON`). The low-level fit raises `InsufficientDataError` below that count. The public index functions handle that case as their docstrings describe.
- **Coordinate validation**: time coordinates must be monotonically increasing. xarray inputs are validated automatically.
- **API stability**: the NumPy and DataArray index APIs are Stable and the compound Dataset routes and SPAI are Beta. Avoid breaking the Stable APIs in minor releases.
- **Namespaces**: fire-weather indices live only under `climate_indices.fire` ([ADR-0005](adr/0005-fire-module-api.md)). Flood-potential indices live only under `climate_indices.flood` ([ADR-0013](adr/0013-flood-module-api-and-naming.md)). Neither is a top-level package function.

### Code patterns to follow

1. **Type annotations**: public functions need full type hints. mypy runs in strict mode (`pyproject.toml`).
2. **Docstrings**: Google-style docstrings for public functions.
3. **Error handling**: raise the specific exception types from `exceptions.py`, with their context attributes.
4. **Logging**: use structlog with key-value pairs, and avoid string interpolation in log messages.
5. **Testing**: property-based tests for mathematical invariants, regression tests against reference datasets, and benchmark tests for performance.

### Dependencies and constraints

- **Core dependencies**: cftime>=1.6.4.post1, dask>=2025.7.0, h5netcdf>=1.6.3, numpy>=1.24.0, pandas>=2.0.0, scipy>=1.15.3, structlog>=24.1.0, xarray>=2025.6.1
- **Python version**: Python 3.11-3.15 (`requires-python = ">=3.11,<3.16"`)
- **Line length**: 120 characters (ruff configuration)
- **Import order**: standard library, third party, then local (enforced by ruff)
- **Test markers**: `@pytest.mark.benchmark` for performance tests, `@pytest.mark.slow` for long-running tests, `@pytest.mark.validation` for external or release-gate validation tests

### Common pitfalls

1. Do not use wildcard imports (`from module import *`); ruff rejects them.
2. Dask time chunks: SPI/SPEI and the stateful fire adapters reject split time chunks, and PET adapters accept them and rechunk internally.
3. Do not break the Stable NumPy and DataArray APIs in a minor release.
4. Run the checks in the "Local validation" section of `CONTRIBUTING.md` before opening a pull request. CI runs the same checks.
5. If you edit a document that `scripts/generate_llms_txt.py` bundles (this page and `README.md` among them), regenerate `llms.txt` and `llms-full.txt` with `uv run scripts/generate_llms_txt.py`.

### Key files for modification scenarios

| Task | Primary files | Test files |
|------|--------------|------------|
| Add new index | `compute.py`, `drought/indices.py`, `typed_public_api.py`, `xarray_adapter.py` | `test_compute.py`, `test_indices.py`, `test_typed_public_api.py`, `test_xarray_adapter.py` |
| Add new distribution | `compute.py`, `drought/indices.py`, `lmoments.py` | `test_compute.py`, `test_property_based.py` |
| Modify Palmer indices | `drought/palmer.py`, `drought/_palmer_pdi.py`, `drought/_palmer_wells.py`, `drought/_palmer_duration.py`, `drought/self_calibration.py` | `test_palmer.py`, `test_scpdsi.py`, `test_self_calibration.py` |
| Add a fire-weather index | `fire/` subpackage | `test_fire*.py`, reference tests |
| Add a flood-potential index | `flood/` subpackage, `typed_public_api.py` | `test_flood*.py`, `test_main_flood.py` |
| Modify run extraction | `runs.py` | `test_runs.py` |
| Fix CLI bug | `__main__.py`, `_cli.py` | `test_cli_common.py`, `test_main_*.py` |
| Add validation | `validation.py`, `xarray_adapter.py`, `exceptions.py` | `test_input_validation.py`, `test_exceptions.py` |
| Update CF metadata | `cf_metadata_registry.py` | `test_cf_metadata.py`, `test_metadata_validation.py` |
| Performance optimization | `compute.py`, `performance.py`, chunking strategies | `test_benchmark_*.py` |

---

**Next steps**: see [architecture.md](./architecture.md) for the technical architecture, [development-guide.md](./development-guide.md) for setup instructions, and [reference.md](./reference.md) for API reference details.
