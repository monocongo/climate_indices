# Agentic Implementation Plan: `climate_indices` Modernization

**Date:** December 25, 2025
**Status:** Draft
**Target:** v3.0.0
**Context:** Based on `docs/2025_modernization_roadmap.md`

This document operationalizes the modernization roadmap into atomic, agent-executable tasks. It is ordered logically to build a stable foundation before attempting complex refactoring.

## Assumptions & Constraints
- **Scope guardrails:** No new architectures/frameworks; keep changes incremental and reversible.
- **Baseline behavior:** Existing units, missing-data rules, and distribution parameterizations are preserved unless explicitly stated.
- **Minimum versions:** Python, NumPy, SciPy, xarray, dask versions must remain compatible with current `pyproject.toml`.

---

## Phase 1: Foundation & Quality of Life
**Goal:** Establish a safety net of dependencies, linting, and typing before touching core logic.

### Task 1.1: Dependency Management Update
**Objective:** Add necessary modernization libraries (`numba`, `pre-commit`, `mypy`) to the project configuration.

*   **Prompt to Self:** "Update `pyproject.toml` to include `numba` in the main dependencies. Add `pre-commit` and `mypy` to the `dev` dependency group. Lock the dependencies using `uv lock`."
*   **Target Files:** `pyproject.toml`, `uv.lock`
*   **Dependencies:** None.
*   **Verification:** `uv sync && uv run mypy --version && uv run python -c "import numba; print(numba.__version__)"`
*   **Acceptance Criteria:** Lockfile updated; versions remain compatible with current runtime targets.
*   **Rollback:** `git checkout pyproject.toml uv.lock`

### Task 1.2: Pre-commit Configuration
**Objective:** Enforce code standards automatically to prevent style regressions during refactoring.

*   **Prompt to Self:** "Create a `.pre-commit-config.yaml` file. Configure hooks for `ruff` (linting/formatting), `mypy` (typing), and `check-yaml`. Run `pre-commit install` and `pre-commit run --all-files` to baseline the repo."
*   **Target Files:** `.pre-commit-config.yaml`, (potentially all `.py` files if auto-fixes apply).
*   **Dependencies:** Task 1.1 (pre-commit installed).
*   **Verification:** `pre-commit run --all-files` returns "Passed".
*   **Acceptance Criteria:** Hooks run without modifying files beyond formatting and lint fixes.
*   **Rollback:** `rm .pre-commit-config.yaml` and `git checkout .`

### Task 1.3: Strict Typing on Core Modules
**Objective:** Add type hints to `src/climate_indices/compute.py` to ensure refactoring doesn't break interfaces.

*   **Prompt to Self:** "Apply strict type hints to `src/climate_indices/compute.py`. Ensure all function signatures specify input/output types (e.g., `np.ndarray`, `float`, `int`). Use `numpy.typing.NDArray` where applicable. Fix any resulting mypy errors."
*   **Target Files:** `src/climate_indices/compute.py`
*   **Dependencies:** Task 1.1 (mypy installed).
*   **Verification:** `uv run mypy src/climate_indices/compute.py`
*   **Acceptance Criteria:** Public function signatures unchanged; mypy passes in strict mode for this file.
*   **Rollback:** `git checkout src/climate_indices/compute.py`

### Testing Strategy (Applies to all phases)
**Objective:** Define a minimal, reliable test matrix to prevent silent scientific regressions.

*   **Unit tests:** Target edge cases (all-zero series, constant series, NaNs, negative values where invalid).
*   **Regression/golden tests:** Use existing `.npy` fixtures as golden outputs; add small SPI fixtures only if needed.
*   **Determinism:** Tests must not depend on random seeds or system-specific BLAS behavior without pinning.
*   **SPI windowing:** Verify rolling window alignment (e.g., scale=6) matches current behavior for start/end padding.
*   **NaN propagation:** Confirm NaNs in inputs propagate only to corresponding outputs; no new NaNs for valid windows.
*   **Numerical tolerances:** Match current tests (e.g., SPI/SPEI outputs `atol=0.001`; Pearson Type III params `atol=0.01`).
*   **Window alignment example:** For `sum_to_scale(values, 6)`, first 5 outputs are NaN and later values match current `tests/test_compute.py` expectations.
*   **Fixture coverage (examples):** SPI: `tests/fixture/spi_01_gamma.npy`, `tests/fixture/spi_06_gamma.npy`, `tests/fixture/spi_06_pearson3.npy`; SPEI: `tests/fixture/spei_06_gamma.npy`, `tests/fixture/spei_06_pearson3.npy`; Gamma transforms: `tests/fixture/transformed_gamma_monthly.npy`, `tests/fixture/transformed_gamma_daily.npy`; Pearson III: `tests/fixture/pearson3_monthly.npy`, `tests/fixture/pearson3_monthly_full.npy`; Palmer: `tests/fixture/palmer/*/pdsi.npy`, `tests/fixture/palmer/*/phdi.npy`, `tests/fixture/palmer/*/pmdi.npy`, `tests/fixture/palmer/*/zindex.npy`.
*   **Fixture-to-test references:** SPI fixtures → `tests/test_indices.py` (`test_spi`); SPEI fixtures → `tests/test_indices.py` (`test_spei`); Gamma transform fixtures → `tests/test_compute.py` (`test_transform_fitted_gamma`); Gamma parameter fixtures → `tests/test_compute.py` (`test_gamma_parameters`); Pearson III fixtures → `tests/test_compute.py` (`test_transform_fitted_pearson`); Palmer fixtures → `tests/test_palmer.py` (`test_pdsi`, `atol=5e-5`).

---

## Phase 2: Core Optimization (Numba)
**Goal:** Accelerate numerical kernels to enable faster testing and future scaling.

### Task 2.1: Optimize Pearson Type III Logic
**Objective:** Speed up the `pearson_parameters` and fitting functions using Numba to remove slow Python loops.

*   **Prompt to Self:** "Refactor `src/climate_indices/compute.py`. Import `numba`. Decorate `_pearson_fit` and any helper calculation functions with `@numba.jit(nopython=True, cache=True)`. Ensure the functions utilize NumPy vectorization compatible with Numba."
*   **Target Files:** `src/climate_indices/compute.py`
*   **Dependencies:** Task 1.1 (numba installed).
*   **Verification:** `pytest tests/test_compute.py` (Functional correctness must be maintained).
*   **Acceptance Criteria:** Results match pre-Numba outputs within `np.allclose` tolerance; NaNs preserved at matching indices; SPI windowing alignment unchanged.
*   **Rollback:** `git checkout src/climate_indices/compute.py`

### Task 2.2: Optimize Palmer Water Balance
**Objective:** Accelerate the iterative water balance calculations in `palmer.py`, which are currently very slow due to dependency chains.

*   **Prompt to Self:** "Refactor `src/climate_indices/palmer.py`. Identify the main time-step loop in `_calc_water_balances`. Extract the loop body into a standalone, Numba-compiled function (`@jit(nopython=True)`). Replace the original loop with a call to this compiled function."
*   **Target Files:** `src/climate_indices/palmer.py`
*   **Dependencies:** Task 1.1.
*   **Verification:** `pytest tests/test_palmer.py`
*   **Acceptance Criteria:** Outputs match baseline within tolerance; no new NaN/inf introduced in valid inputs.
*   **Rollback:** `git checkout src/climate_indices/palmer.py`

---

## Phase 3: API Modernization (Xarray)
**Goal:** Create a user-friendly, metadata-aware API.

### Task 3.1: Define Xarray Accessor Structure
**Objective:** Scaffold the accessor to allow `ds.indices.spi()` syntax.

*   **Prompt to Self:** "Create a new file `src/climate_indices/accessors.py`. Define a class `IndicesAccessor` decorated with `@xr.register_dataarray_accessor('indices')`. Implement a skeleton `__init__` method that stores the Xarray object."
*   **Target Files:** `src/climate_indices/accessors.py`, `src/climate_indices/__init__.py` (to expose it).
*   **Dependencies:** Phase 1.
*   **Verification:** `python -c "import xarray as xr; import climate_indices; ds = xr.DataArray([1,2]); assert hasattr(ds, 'indices')"`
*   **Acceptance Criteria:** Accessor registers on `DataArray`; no side effects on import.
*   **Rollback:** `rm src/climate_indices/accessors.py`

### Task 3.2: Implement SPI Accessor Method
**Objective:** Connect the accessor to the existing SPI logic, handling metadata wrapping.

*   **Prompt to Self:** "In `src/climate_indices/accessors.py`, implement the `spi` method. It should accept `scale`, `distribution`, etc. It must extract the raw data, call `indices.spi` (or the `compute` module directly), and wrap the result back into an `xarray.DataArray` with appropriate coordinates."
*   **Target Files:** `src/climate_indices/accessors.py`
*   **Dependencies:** Task 3.1.
*   **Verification:** Create a new test `tests/test_accessors.py` that loads a dummy NetCDF, calls `.indices.spi()`, and asserts the result is a DataArray with correct dims.
*   **Acceptance Criteria:** Coordinate/dimension names preserved; `attrs` include `long_name` and `units` where applicable; NaN locations match underlying compute results.
*   **Rollback:** `git checkout src/climate_indices/accessors.py`

---

## Phase 4: Scaling (Dask)
**Goal:** Enable out-of-core processing and remove complex manual parallelism.

### Task 4.1: Refactor SPI to `apply_ufunc`
**Objective:** Rewrite the core SPI glue code to use `xarray.apply_ufunc`, enabling Dask automatically.

*   **Prompt to Self:** "Refactor `src/climate_indices/indices.py`. Create a new function `spi_xarray` (or update `spi` if feasible without breaking changes) that uses `xarray.apply_ufunc`. It should utilize `dask='parallelized'` to map the `compute.transform_fitted_gamma` function across Dask chunks."
*   **Target Files:** `src/climate_indices/indices.py`
*   **Dependencies:** Phase 3.
*   **Verification:** `pytest tests/test_indices.py` and manually verify a Dask array input returns a lazy Dask array output.
*   **Acceptance Criteria:** Dask inputs remain lazy; chunking strategy documented in docstring; SPI scale windowing and NaN handling unchanged.
*   **Rollback:** `git checkout src/climate_indices/indices.py`

### Task 4.2: Deprecate Multiprocessing CLI
**Objective:** Simplify the CLI by removing the manual shared-memory multiprocessing in favor of Dask.

*   **Prompt to Self:** "Refactor `src/climate_indices/__spi__.py`. Remove the `multiprocessing` imports and the `_global_shared_arrays` logic. Rewrite `_compute_write_index` to use the new Xarray/Dask pipeline developed in Task 4.1. Use `dask.distributed.Client` if 'all' cores are requested."
*   **Target Files:** `src/climate_indices/__spi__.py`
*   **Dependencies:** Task 4.1.
*   **Verification:** Run the CLI against the sample data: `climate_indices --netcdf_precip ...` and ensure it completes.
*   **Acceptance Criteria:** Outputs match baseline for sample data; `--all` cores produces expected behavior.
*   **Rollback:** `git checkout src/climate_indices/__spi__.py`
