---
stepsCompleted: [1, 2, 3, 4, 5]
lastStep: 5
status: complete
---

# Test Design for QA: climate_indices v2.4.0

**Purpose:** Test execution recipe with implementation details, code examples, and effort estimates for QA team.

**Date:** 2026-02-16
**Author:** James
**Status:** QA Review Pending
**Project:** climate_indices (Python Scientific Library)
**PRD Reference:** `_bmad-output/planning-artifacts/prd.md` (v2.4.0)
**Architecture Reference:** `_bmad-output/planning-artifacts/architecture.md` (v2.4.0)
**Architecture Test Design:** `test-design-architecture.md` (this directory)

---

## Executive Summary

**Risk Summary:**
- **High-priority risks (≥6):** 5 risks requiring immediate mitigation (R-001 to R-005)
- **Sprint 0 blockers:** 3 items (NOAA dataset, Palmer baseline, Track 0 baseline infrastructure)
- See `test-design-architecture.md` for full risk assessment and mitigation plans

**Coverage Summary:**
- **Total scenarios:** 44 tests across 4 tracks (P0: 9, P1: 21, P2: 9, P3: 5)
- **Estimated effort:** ~100-175 hours (~2.5-4.5 weeks for 1 QA, ~1.5-2.5 weeks for 2 QAs)
- **Quality gates:** P0=100%, P1≥95%, P2≥85%, coverage≥80%, Palmer xarray ≥80% baseline performance

---

## Dependencies & Test Blockers

### Sprint 0 Blockers (MUST RESOLVE BEFORE TESTING STARTS)

**Backend/Architecture Dependencies:**

1. **NOAA EDDI Reference Dataset** (R-004)
   - **What:** Download NOAA PSL EDDI CONUS archive subset
   - **Where:** `tests/data/reference/eddi/noaa_conus_2020_6mo.nc`
   - **Provenance:** source, URL, download_date, SHA256 checksum, subset_description
   - **Owner:** QA/Dev (Track 2 implementer)
   - **Timeline:** Week 1 of Sprint 0
   - **Impact:** Blocks T-007 (P0 EDDI validation)

2. **Palmer Performance Baseline** (R-005)
   - **What:** Establish multiprocessing CLI speed on synthetic 360×180×240 grid
   - **Where:** `tests/benchmark/test_palmer_baseline.py`
   - **Owner:** Dev (Track 3 implementer)
   - **Timeline:** Week 1 of Sprint 0
   - **Impact:** Blocks T-009 (P0 Palmer performance validation)

3. **Track 0 Baseline Capture Infrastructure** (R-001)
   - **What:** Directory structure + protocol for before/after equivalence snapshots
   - **Where:** `tests/baseline/` (e.g., `baseline_pnp.npy`, `baseline_pci.npy`)
   - **Owner:** Dev (Track 0 implementer)
   - **Timeline:** Week 1 of Sprint 0
   - **Impact:** Blocks T-001, T-002 (P0 equivalence validation)

**QA Infrastructure Setup:**

```python
# tests/conftest.py additions for v2.4.0

import json
import hashlib
import numpy as np
import xarray as xr
import pytest

# --- Sprint 0: NOAA EDDI Reference Dataset Fixture ---
@pytest.fixture(scope="session")
def noaa_eddi_reference_da() -> xr.DataArray:
    """Load NOAA PSL EDDI CONUS reference dataset (cached in Sprint 0)."""
    ds = xr.open_dataset("tests/data/reference/eddi/noaa_conus_2020_6mo.nc")

    # Validate provenance metadata
    required_attrs = ["source", "url", "download_date", "sha256_checksum"]
    for attr in required_attrs:
        assert attr in ds.attrs, f"Missing provenance attribute: {attr}"

    # Validate SHA256 checksum
    with open("tests/data/reference/eddi/noaa_conus_2020_6mo.nc", "rb") as f:
        computed_checksum = hashlib.sha256(f.read()).hexdigest()
    assert computed_checksum == ds.attrs["sha256_checksum"], "Dataset checksum mismatch"

    return ds["eddi_6mo"]

# --- Track 0: Baseline Capture Protocol ---
def save_baseline(test_name: str, result: np.ndarray) -> None:
    """Save baseline output for equivalence validation."""
    baseline_path = f"tests/baseline/baseline_{test_name}.npy"
    np.save(baseline_path, result)

def load_baseline(test_name: str) -> np.ndarray:
    """Load baseline output for equivalence validation."""
    baseline_path = f"tests/baseline/baseline_{test_name}.npy"
    return np.load(baseline_path)

# --- Track 3: Palmer Synthetic Grid Fixture ---
@pytest.fixture(scope="session")
def palmer_synthetic_grid_360x180x240() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic 360×180×240 grid for Palmer performance testing."""
    rng = np.random.default_rng(999)
    # 10 years monthly (240 timesteps), 180 lats, 360 lons (1-degree global resolution)
    precip = rng.gamma(shape=2.0, scale=50.0, size=(240, 180, 360))
    pet = rng.gamma(shape=2.5, scale=60.0, size=(240, 180, 360))
    return precip, pet

```

---

## Risk Assessment (Brief — See Architecture Doc for Full Details)

**High-Priority Risks (≥6):**

| Risk ID | Category | Description | QA Test Coverage |
|---------|----------|-------------|------------------|
| R-001 | SCI | Numerical drift during Track 0 refactoring | T-001, T-002: Equivalence tests with 1e-8 tolerance + baseline capture |
| R-002 | SCI | FAO56 constant precision errors | T-003: Helper unit tests with reference values ±0.01 kPa |
| R-003 | SCI | SVP non-linearity error | T-004: Unit test verifies `_mean_svp() != _svp_from_t(Tmean)` |
| R-004 | DATA | NOAA reference dataset unavailable | T-007: EDDI NOAA reference validation with provenance tracking |
| R-005 | PERF | Palmer xarray performance regression | T-009: Benchmark vs baseline (target ≥80%) |

**Medium-Priority Risks (3-5):**

| Risk ID | Category | Description | QA Test Coverage |
|---------|----------|-------------|------------------|
| R-006 | TECH | Humidity pathway logic bug | T-018, T-019, T-020: Pathway selection integration + priority order + error handling |
| R-007 | DATA | EDDI tolerance insufficient | T-022: EDDI xarray adapter with 1e-5 tolerance + docstring rationale |
| R-008 | TECH | params_dict serialization failure | T-028: JSON round-trip test + NaN/Inf edge cases |
| R-009 | TECH | AWC time dimension bypass | T-027: AWC validation test (raise error if time dim present) |
| R-010 | TECH | Stack/unpack pattern fragility | T-025: Palmer manual wrapper integration test |
| R-011 | TECH | Type safety violations | T-010, T-011, T-030: @overload dispatch tests + mypy --strict |
| R-012 | TECH | structlog lifecycle inconsistency | T-012, T-013: Lifecycle event validation tests |

---

## Test Coverage Plan

**Note:** P0/P1/P2/P3 classifications represent **priority and risk level**, NOT execution timing. See Execution Strategy section for when tests run.

### P0: Critical Core Functionality

**Criteria:** Blocks core scientific correctness, high risk (≥6), no workaround. **Must pass 100% before release.**

| Test ID | Requirement | Test Level | Risk Link | Test File | Test Function |
|---------|-------------|------------|-----------|-----------|---------------|
| **T-001** | FR-PATTERN-001, NFR-PATTERN-EQUIV | Integration | R-001 | `test_xarray_equivalence.py` | `test_percentage_of_normal_xarray_equivalence()` |
| **T-002** | FR-PATTERN-002, NFR-PATTERN-EQUIV | Integration | R-001 | `test_xarray_equivalence.py` | `test_pci_xarray_equivalence()` |
| **T-003** | FR-PM-002 | Unit | R-002 | `test_eto.py` | `test_fao56_constants_precision()` |
| **T-004** | FR-PM-003 | Unit | R-003 | `test_eto.py` | `test_svp_nonlinearity()` |
| **T-005** | FR-PM-005 | Integration | R-002 | `test_eto.py` | `test_fao56_example_17_bangkok()` |
| **T-006** | FR-PM-005 | Integration | R-002 | `test_eto.py` | `test_fao56_example_18_uccle()` |
| **T-007** | FR-EDDI-001, NFR-EDDI-VAL | Integration | R-004 | `test_reference_validation.py` (new) | `test_eddi_noaa_reference()` |
| **T-008** | FR-PALMER-007, NFR-PALMER-PERF | Integration | R-005 | `test_palmer_xarray.py` (new) | `test_palmer_xarray_equivalence()` |
| **T-009** | NFR-PALMER-PERF | Benchmark | R-005 | `test_palmer_xarray.py` (new) | `test_palmer_xarray_performance()` |

**P0 Implementation Example (T-003: FAO56 Constants Precision):**

```python
# tests/test_eto.py additions for Track 1

import pytest
import numpy as np
from climate_indices.eto import (
    _atm_pressure,
    _psy_const,
    _svp_from_t,
)

class TestFAO56HelperPrecision:
    """Validate FAO56 helper functions with known reference values."""

    def test_svp_from_t_reference_value(self):
        """Test SVP at 21.5°C matches FAO56 reference (2.564 kPa)."""
        temp_celsius = 21.5
        expected_svp_kpa = 2.564
        tolerance_kpa = 0.01

        actual_svp = _svp_from_t(np.array([temp_celsius]))[0]

        assert np.isclose(actual_svp, expected_svp_kpa, atol=tolerance_kpa), \
            f"SVP(21.5°C) should be 2.564 ±0.01 kPa, got {actual_svp:.4f} kPa"

    def test_atm_pressure_reference_value(self):
        """Test atmospheric pressure at 100m altitude matches FAO56 (100.1 kPa)."""
        altitude_m = 100.0
        expected_pressure_kpa = 100.1
        tolerance_kpa = 0.01

        actual_pressure = _atm_pressure(altitude_m)

        assert np.isclose(actual_pressure, expected_pressure_kpa, atol=tolerance_kpa), \
            f"Pressure(100m) should be 100.1 ±0.01 kPa, got {actual_pressure:.4f} kPa"

    def test_psy_const_reference_value(self):
        """Test psychrometric constant at 100.1 kPa matches FAO56 (0.06667 kPa/°C)."""
        pressure_kpa = 100.1
        expected_gamma = 0.06667
        tolerance = 0.00001

        actual_gamma = _psy_const(pressure_kpa)

        assert np.isclose(actual_gamma, expected_gamma, atol=tolerance), \
            f"Gamma(100.1 kPa) should be 0.06667 ±0.00001, got {actual_gamma:.5f}"
```

**P0 Implementation Example (T-004: SVP Non-Linearity):**

```python
# tests/test_eto.py additions for Track 1

def test_mean_svp_nonlinearity():
    """Validate SVP averaging is non-linear (common FAO56 error detection)."""
    tmin_celsius = 10.0
    tmax_celsius = 30.0

    # Correct approach (FAO56 Eq. 12): average SVP at extremes
    mean_svp_correct = _mean_svp(np.array([tmin_celsius]), np.array([tmax_celsius]))[0]

    # Common error: SVP at mean temperature
    tmean_celsius = (tmin_celsius + tmax_celsius) / 2.0
    mean_svp_wrong = _svp_from_t(np.array([tmean_celsius]))[0]

    # These should NOT be equal (SVP is non-linear)
    assert not np.isclose(mean_svp_correct, mean_svp_wrong, rtol=0.01), \
        f"SVP averaging is non-linear: _mean_svp({tmin_celsius}, {tmax_celsius}) = {mean_svp_correct:.4f} " \
        f"!= _svp_from_t({tmean_celsius}) = {mean_svp_wrong:.4f}"
```

**P0 Implementation Example (T-007: EDDI NOAA Reference Validation):**

```python
# tests/test_reference_validation.py (NEW MODULE for Track 2)

import pytest
import numpy as np
import xarray as xr
from climate_indices import eddi

class TestEDDINOAAReference:
    """Validate EDDI against NOAA PSL reference dataset."""

    def test_eddi_noaa_reference(self, noaa_eddi_reference_da):
        """Test EDDI output matches NOAA PSL CONUS archive within 1e-5.

        Tolerance: 1e-5 (looser than equivalence tests)
        Rationale: Non-parametric empirical ranking has different floating-point
                   accumulation characteristics than parametric distribution fitting.
                   1e-5 is acceptable per Architecture v2.4.0 Decision 12.
        """
        # Load reference data
        noaa_reference = noaa_eddi_reference_da

        # Extract PET input (assumed to exist in reference dataset)
        pet_da = noaa_reference.pet  # or reconstruct from reference metadata

        # Compute EDDI with climate_indices library
        computed_eddi = eddi(
            pet_da,
            scale=6,
            data_start_year=2020,
            calibration_year_start=2010,
            calibration_year_end=2019,
            periodicity="monthly",
        )

        # Validate against NOAA reference
        np.testing.assert_allclose(
            computed_eddi.values,
            noaa_reference.values,
            rtol=1e-5,
            atol=1e-5,
            err_msg="EDDI does not match NOAA reference within 1e-5 tolerance"
        )

    def test_eddi_reference_provenance(self, noaa_eddi_reference_da):
        """Validate NOAA reference dataset has required provenance metadata."""
        required_attrs = ["source", "url", "download_date", "sha256_checksum", "subset_description"]

        for attr in required_attrs:
            assert attr in noaa_eddi_reference_da.attrs, \
                f"Reference dataset missing provenance attribute: {attr}"

        # Validate provenance values are non-empty
        for attr in required_attrs:
            assert len(str(noaa_eddi_reference_da.attrs[attr])) > 0, \
                f"Provenance attribute '{attr}' is empty"
```

---

### P1: Important Features

**Criteria:** Important features, medium risk (3-4), ≥95% pass rate target.

*See `coverage-plan-working.md` for full P1 test list (21 tests: T-010 through T-030)*

**P1 Implementation Example (T-018: PM-ET Core Calculation):**

```python
# tests/test_eto.py additions for Track 1

class TestPenmanMonteithCore:
    """Test PM-ET core calculation with all humidity pathways."""

    @pytest.mark.parametrize("humidity_input,expected_pathway", [
        ({"dewpoint_celsius": 15.0}, "dewpoint"),
        ({"rh_max": 80.0, "rh_min": 40.0}, "rh_extremes"),
        ({"rh_mean": 60.0}, "rh_mean"),
    ])
    def test_humidity_pathway_selection(
        self,
        humidity_input,
        expected_pathway,
        sample_monthly_temp_da,
    ):
        """Test auto-dispatcher selects correct humidity pathway."""
        # Minimal inputs for PM-ET
        tmin_da = sample_monthly_temp_da - 5.0
        tmax_da = sample_monthly_temp_da + 5.0
        tmean_da = sample_monthly_temp_da
        wind_2m_da = xr.DataArray([2.0] * len(sample_monthly_temp_da), dims=["time"])
        net_radiation_da = xr.DataArray([15.0] * len(sample_monthly_temp_da), dims=["time"])

        # Call PM-ET with specific humidity input
        result = eto_penman_monteith(
            tmin_celsius=tmin_da,
            tmax_celsius=tmax_da,
            tmean_celsius=tmean_da,
            wind_speed_2m=wind_2m_da,
            net_radiation=net_radiation_da,
            latitude_degrees=40.0,
            altitude_meters=1500,
            **humidity_input,  # dewpoint OR rh_max/rh_min OR rh_mean
        )

        # Validate result is reasonable
        assert np.all(result.values > 0), "PM-ET must be positive"
        assert np.all(result.values < 20), "PM-ET should be <20 mm/day for typical conditions"

        # TODO: Add logging capture to verify expected_pathway selected
        # (requires structlog test utilities)
```

**P1 Implementation Example (T-028: params_dict Serialization):**

```python
# tests/test_palmer_xarray.py (NEW MODULE for Track 3)

import json
import numpy as np
import xarray as xr
from climate_indices import pdsi

class TestPalmerParamsDictSerialization:
    """Test Palmer params_dict JSON serialization and dual access."""

    def test_params_dict_json_round_trip(
        self,
        sample_monthly_precip_da,
        sample_monthly_pet_da,
    ):
        """Test params_dict JSON serialization preserves structure."""
        # Compute Palmer indices (returns Dataset with params_dict in attrs)
        ds_palmer = pdsi(
            sample_monthly_precip_da,
            sample_monthly_pet_da,
            awc=2.5,
        )

        # Extract params_dict from JSON attribute
        params_json = ds_palmer.attrs["palmer_params"]
        params_dict = json.loads(params_json)

        # Validate structure
        required_keys = ["alpha", "beta", "gamma", "delta"]
        for key in required_keys:
            assert key in params_dict, f"params_dict missing key: {key}"
            assert isinstance(params_dict[key], (int, float)), \
                f"params_dict['{key}'] should be numeric, got {type(params_dict[key])}"

        # Validate round-trip equivalence
        params_json_again = json.dumps(params_dict)
        params_dict_again = json.loads(params_json_again)
        assert params_dict == params_dict_again, "JSON round-trip failed"

    def test_params_dict_dual_access(
        self,
        sample_monthly_precip_da,
        sample_monthly_pet_da,
    ):
        """Test params_dict accessible via both JSON and individual attrs."""
        ds_palmer = pdsi(
            sample_monthly_precip_da,
            sample_monthly_pet_da,
            awc=2.5,
        )

        # Access via JSON
        params_dict = json.loads(ds_palmer.attrs["palmer_params"])
        alpha_from_json = params_dict["alpha"]

        # Access via individual attr
        alpha_from_attr = ds_palmer.attrs["palmer_alpha"]

        # Should be identical
        assert np.isclose(alpha_from_json, alpha_from_attr, rtol=1e-10), \
            f"Dual access mismatch: JSON alpha={alpha_from_json}, attr alpha={alpha_from_attr}"

    def test_params_dict_edge_cases(
        self,
        sample_monthly_precip_da,
        sample_monthly_pet_da,
    ):
        """Test params_dict handles edge cases (NaN, Inf, large floats)."""
        # Create edge case input that might produce NaN/Inf params
        precip_edge = sample_monthly_precip_da.copy()
        precip_edge[:] = 0.0  # All-zero precipitation (extreme case)

        # Compute Palmer (may produce NaN/Inf params in edge case)
        ds_palmer = pdsi(
            precip_edge,
            sample_monthly_pet_da,
            awc=2.5,
        )

        # JSON serialization should not crash (use allow_nan=True if needed)
        try:
            params_json = ds_palmer.attrs["palmer_params"]
            params_dict = json.loads(params_json)
        except json.JSONDecodeError as e:
            pytest.fail(f"JSON serialization failed on edge case: {e}")
```

---

### P2: Secondary Features

**Criteria:** Secondary features, low risk (1-2), ≥85% pass rate target.

*See `coverage-plan-working.md` for full P2 test list (9 tests: T-031 through T-039)*

---

### P3: Nice-to-Have

**Criteria:** Exploratory testing, deferred features, informational value.

*See `coverage-plan-working.md` for full P3 test list (5 tests: T-040 through T-044)*

---

## Execution Strategy (Organized by Infrastructure Overhead)

**Philosophy:** "Run everything in PRs unless expensive/long-running." Maximize fast feedback.

### Every PR (Full Suite — <5 Minutes)

**pytest with pytest-xdist parallel execution:**

```bash
# Run all tests except benchmarks and slow tests (8 workers parallel)
uv run pytest tests/ -v -m "not benchmark and not slow" -n auto

# Type checking (mypy --strict on typed_public_api.py)
uv run mypy src/climate_indices/typed_public_api.py --strict

# Linting and formatting
ruff check src/ tests/
ruff format --check src/ tests/
```

**What runs:**
- All P0, P1, P2 unit tests
- All P0, P1, P2 integration tests (except benchmarks)
- All property-based tests (hypothesis with default example counts)
- Type checking, linting, formatting

**Rationale:**
- pytest-xdist enables 8-worker parallel execution (typical CI)
- Fast fixtures (session-scoped .npy files loaded once)
- In-memory computation (no I/O bottlenecks)
- Target: <5 min total runtime

---

### Nightly (Performance + Extended Validation — ~30-60 Minutes)

**pytest-benchmark + extended hypothesis:**

```bash
# Run all benchmark tests
uv run pytest tests/ -v -m benchmark --benchmark-only

# Run property tests with extended example counts (10,000+ examples)
uv run pytest tests/test_property_based.py --hypothesis-profile=extensive

# Memory profiling (if pytest-memray integrated)
uv run pytest tests/ --memray --most-allocations=20
```

**What runs:**
- All pytest-benchmark tests (T-009, P2 performance tests)
- Extended Palmer 344-dataset validation (T-040, when implemented)
- Hypothesis property tests with extensive examples
- Memory profiling

**Rationale:**
- Benchmarks require stable hardware (no CI concurrency)
- Extended validation datasets too large for frequent runs
- Performance regression tracking needs baseline stability

---

### Weekly (Long-Running + Manual — ~Hours)

**Manual validation + chaos testing:**

```bash
# Dask multi-node testing (when implemented)
uv run pytest tests/test_dask_distributed.py -v --slow

# Manual EDDI reference validation (extended coverage)
uv run pytest tests/test_reference_validation.py -v --extended
```

**What runs:**
- Dask multi-node testing (T-042, when implemented)
- Chaos testing (infrastructure failures)
- Manual validation against NOAA EDDI reference datasets

**Rationale:**
- Long-running tests block development if run frequently
- Manual validation requires human judgment

---

## QA Effort Estimate

**Interval-Based Estimates (QA Effort ONLY):**

| Activity                        | Estimated Effort  | Notes                                                                                           |
| ------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------- |
| **P0 Test Implementation**      | ~25-40 hours      | 9 tests: equivalence, FAO56 examples, NOAA reference, Palmer baseline                          |
| **P1 Test Implementation**      | ~35-60 hours      | 21 tests: typed_public_api, structlog lifecycle, property tests, xarray adapters, Palmer multi-output |
| **P2 Test Implementation**      | ~15-25 hours      | 9 tests: edge cases, CLI integration, NetCDF round-trip, pattern compliance                    |
| **P3 Exploration**              | ~5-10 hours       | 5 tests: extended validation, future enhancements                                              |
| **Fixture Setup**               | ~10-20 hours      | NOAA dataset download + provenance, Palmer baseline, Track 0 baselines                         |
| **CI Configuration**            | ~5-10 hours       | pytest-benchmark setup, nightly/weekly job configuration                                        |
| **Documentation**               | ~5-10 hours       | Test design review, knowledge transfer                                                          |
| **TOTAL QA EFFORT**             | **~100-175 hours** | **~2.5-4.5 weeks for 1 QA, ~1.5-2.5 weeks for 2 QAs**                                          |

**Timeline Estimate:**
- **Sprint 0 (Fixtures):** 1 week (NOAA dataset, Palmer baseline, Track 0 infrastructure)
- **Track 0 Tests:** 1.5-2.5 weeks (equivalence, type safety, property tests)
- **Track 1 Tests:** 1.5-2 weeks (PM-ET helpers, FAO56 validation, xarray adapter)
- **Track 2 Tests:** 1-1.5 weeks (EDDI reference, PNP/scPDSI adapters)
- **Track 3 Tests:** 2-3 weeks (Palmer xarray, multi-output, performance benchmarks)

**Critical Path:** Track 0 (Palmer structlog) + Track 1 (PM-ET) → Track 3 (Palmer xarray)

---

## Sprint Planning Handoff (Optional)

**Implementation Tasks for Sprint 0:**

| Task                                       | Owner          | Target Sprint | Estimated Effort |
| ------------------------------------------ | -------------- | ------------- | ---------------- |
| Download NOAA EDDI reference dataset       | QA/Dev (Track 2) | Sprint 0      | ~2-4 hours       |
| Validate SHA256 + document provenance      | QA/Dev (Track 2) | Sprint 0      | ~1-2 hours       |
| Establish Palmer multiprocessing baseline  | Dev (Track 3)  | Sprint 0      | ~3-5 hours       |
| Create Track 0 baseline infrastructure     | Dev (Track 0)  | Sprint 0      | ~2-4 hours       |
| Configure nightly CI job for benchmarks    | DevOps         | Sprint 0      | ~2-4 hours       |

---

## Appendix A: Code Examples & Tagging

### pytest Tagging Strategy

**pyproject.toml additions:**

```toml
[tool.pytest.ini_options]
markers = [
    "benchmark: performance benchmark tests (deselect with '-m \"not benchmark\"')",
    "slow: slow tests that may be skipped in CI (deselect with '-m \"not slow\"')",
    "track0: Track 0 (Canonical Pattern Completion) tests",
    "track1: Track 1 (PM-ET FAO56) tests",
    "track2: Track 2 (EDDI/PNP/scPDSI) tests",
    "track3: Track 3 (Palmer Multi-Output xarray) tests",
    "p0: P0 priority tests (must pass 100%)",
    "p1: P1 priority tests (target ≥95%)",
    "p2: P2 priority tests (target ≥85%)",
    "p3: P3 priority tests (exploratory)",
]
```

**Test Tagging Example:**

```python
import pytest

@pytest.mark.p0
@pytest.mark.track1
def test_fao56_constants_precision():
    """P0 Track 1: Validate FAO56 helper constants."""
    ...

@pytest.mark.p1
@pytest.mark.track0
def test_pnp_xarray_equivalence():
    """P1 Track 0: PNP numpy vs xarray equivalence."""
    ...

@pytest.mark.benchmark
@pytest.mark.p0
@pytest.mark.track3
def test_palmer_xarray_performance(benchmark):
    """P0 Track 3 Benchmark: Palmer xarray vs baseline."""
    ...
```

### pytest-benchmark Usage

**Benchmark Template:**

```python
import pytest

@pytest.mark.benchmark
def test_palmer_xarray_performance(
    benchmark,
    palmer_synthetic_grid_360x180x240,
):
    """Benchmark Palmer xarray path vs multiprocessing baseline (≥80% target)."""
    precip, pet = palmer_synthetic_grid_360x180x240
    precip_da = xr.DataArray(precip, dims=["time", "lat", "lon"])
    pet_da = xr.DataArray(pet, dims=["time", "lat", "lon"])

    # Benchmark Palmer xarray
    result = benchmark(palmer_xarray, precip_da, pet_da, awc=2.5)

    # Validate result structure
    assert isinstance(result, xr.Dataset)
    assert "pdsi" in result.data_vars

    # Performance comparison logged by pytest-benchmark
    # Target: ≥80% of baseline (baseline measured in Sprint 0)
```

---

## Appendix B: Knowledge Base References

**Test Design Principles:**
- Test pyramid: More unit tests, fewer integration tests, minimal E2E
- Equivalence testing: Validate xarray vs numpy paths within 1e-8
- Property-based testing: Use hypothesis for mathematical invariants
- Baseline capture: Save .npy snapshots before refactoring for equivalence validation

**Scientific Testing Patterns:**
- Numerical precision: 1e-8 tolerance for float64 (scientific correctness)
- Reference dataset validation: External ground truth (NOAA EDDI)
- Performance benchmarking: Measure against established baselines
- Fixture isolation: Session-scoped for expensive operations, reset utilities for state

**Python Testing Standards:**
- pytest framework with fixtures, parametrization, markers
- hypothesis for property-based testing (boundedness, NaN propagation)
- pytest-benchmark for performance regression tracking
- mypy --strict for type safety validation

---

**Document Complete:** 44 test scenarios with pytest implementation examples, ~100-175 hours QA effort, Sprint 0 blockers identified, execution strategy defined

**Next Step:** Review with QA team, allocate resources per estimates, schedule Sprint 0 fixture setup, proceed to test implementation
