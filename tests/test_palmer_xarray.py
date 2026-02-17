"""Tests for Palmer xarray adapter (palmer_xarray wrapper).

Story 4.3: Foundation smoke tests verifying the manual wrapper produces
correct Dataset structure with preserved coordinates and provenance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices.palmer import palmer_xarray

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_N_MONTHS = 240  # 20 years
_DATA_START_YEAR = 2000
_AWC = 5.0  # inches, typical loamy soil


def _make_time_coord(
    start: str = "2000-01",
    periods: int = _N_MONTHS,
) -> pd.DatetimeIndex:
    """Create a monthly time coordinate."""
    return pd.date_range(start, periods=periods, freq="MS")


def _make_palmer_dataarrays(
    n_months: int = _N_MONTHS,
    start: str = "2000-01",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Generate realistic precipitation and PET DataArrays for Palmer.

    Values are in inches (as required by palmer.pdsi).

    Args:
        n_months: Number of monthly values.
        start: Start date string for time coordinate.

    Returns:
        Tuple of (precipitation, PET) DataArrays.
    """
    time_coord = _make_time_coord(start=start, periods=n_months)
    months = np.arange(n_months) % 12

    # seasonal precipitation pattern ~2-4 in/month
    precip_seasonal = 2.5 + 1.5 * np.sin(2 * np.pi * months / 12)
    precip = precip_seasonal + _RNG.normal(0, 0.5, n_months)
    precip = np.clip(precip, 0.0, None)

    # seasonal PET pattern ~1-5 in/month
    pet_seasonal = 3.0 + 2.0 * np.sin(2 * np.pi * (months - 3) / 12)
    pet = pet_seasonal + _RNG.normal(0, 0.3, n_months)
    pet = np.clip(pet, 0.1, None)

    precip_da = xr.DataArray(
        precip,
        coords={"time": time_coord},
        dims=["time"],
        name="precip",
    )
    pet_da = xr.DataArray(
        pet,
        coords={"time": time_coord},
        dims=["time"],
        name="pet",
    )
    return precip_da, pet_da


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestPalmerXarraySmoke:
    """Basic smoke tests that the wrapper runs and returns correct structure."""

    def test_returns_dataset(self) -> None:
        """palmer_xarray returns an xr.Dataset."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        assert isinstance(ds, xr.Dataset)

    def test_dataset_has_four_variables(self) -> None:
        """Dataset contains exactly pdsi, phdi, pmdi, z_index."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        expected_vars = {"pdsi", "phdi", "pmdi", "z_index"}
        assert set(ds.data_vars) == expected_vars

    def test_output_shape_matches_input(self) -> None:
        """Each output variable has the same shape as the input time series."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        for var_name in ("pdsi", "phdi", "pmdi", "z_index"):
            assert ds[var_name].shape == precip_da.shape, (
                f"{var_name} shape {ds[var_name].shape} != input shape {precip_da.shape}"
            )

    def test_coordinates_preserved(self) -> None:
        """Time coordinates are preserved from input to output."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        xr.testing.assert_equal(ds.coords["time"], precip_da.coords["time"])

    def test_version_in_attrs(self) -> None:
        """Dataset attrs include climate_indices_version."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        assert "climate_indices_version" in ds.attrs

    def test_history_in_attrs(self) -> None:
        """Dataset attrs include history provenance entry."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        assert "history" in ds.attrs
        assert "Palmer" in ds.attrs["history"]

    def test_variable_attrs_have_long_name(self) -> None:
        """Each variable has a long_name attribute."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        for var_name in ("pdsi", "phdi", "pmdi", "z_index"):
            assert "long_name" in ds[var_name].attrs, f"{var_name} missing long_name"

    def test_variable_attrs_have_references(self) -> None:
        """Each variable has a references attribute from the CF metadata registry."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        for var_name in ("pdsi", "phdi", "pmdi", "z_index"):
            assert "references" in ds[var_name].attrs, f"{var_name} missing references"
            assert len(ds[var_name].attrs["references"]) > 0, f"{var_name} references is empty"

    def test_calibration_params_in_attrs(self) -> None:
        """Dataset attrs include Palmer calibration parameters."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        for param_name in ("palmer_alpha", "palmer_beta", "palmer_gamma", "palmer_delta"):
            assert param_name in ds.attrs, f"Missing {param_name} in Dataset attrs"


# ---------------------------------------------------------------------------
# Parameter inference tests
# ---------------------------------------------------------------------------


class TestPalmerXarrayInference:
    """Tests that temporal parameters are correctly inferred from coordinates."""

    def test_infers_all_temporal_params(self) -> None:
        """All temporal params inferred when not provided."""
        precip_da, pet_da = _make_palmer_dataarrays()
        # no explicit data_start_year, calibration years
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 4

    def test_explicit_params_override_inference(self) -> None:
        """Explicit temporal parameters take precedence over inference."""
        precip_da, pet_da = _make_palmer_dataarrays()
        ds = palmer_xarray(
            precip_da,
            pet_da,
            awc=_AWC,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2019,
        )
        assert isinstance(ds, xr.Dataset)
        assert len(ds.data_vars) == 4


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestPalmerXarrayValidation:
    """Tests for input validation errors."""

    def test_missing_time_dimension_raises(self) -> None:
        """Raises CoordinateValidationError if time dim missing."""
        precip = xr.DataArray(
            np.random.uniform(0, 5, 240),
            dims=["x"],
        )
        pet = xr.DataArray(
            np.random.uniform(1, 4, 240),
            dims=["x"],
        )
        from climate_indices.exceptions import CoordinateValidationError

        with pytest.raises(CoordinateValidationError, match="time.*not found"):
            palmer_xarray(precip, pet, awc=_AWC)

    def test_non_monotonic_time_raises(self) -> None:
        """Raises CoordinateValidationError if time is not monotonic."""
        time_coord = _make_time_coord()
        # shuffle to make non-monotonic
        shuffled = np.array(time_coord)
        shuffled[0], shuffled[1] = shuffled[1], shuffled[0]

        precip = xr.DataArray(
            np.random.uniform(0, 5, _N_MONTHS),
            coords={"time": shuffled},
            dims=["time"],
        )
        pet = xr.DataArray(
            np.random.uniform(1, 4, _N_MONTHS),
            coords={"time": shuffled},
            dims=["time"],
        )
        from climate_indices.exceptions import CoordinateValidationError

        with pytest.raises(CoordinateValidationError, match="monotonic"):
            palmer_xarray(precip, pet, awc=_AWC)


# ---------------------------------------------------------------------------
# Input alignment tests
# ---------------------------------------------------------------------------


class TestPalmerXarrayAlignment:
    """Tests for input alignment behavior."""

    def test_different_time_ranges_aligned(self) -> None:
        """Precip and PET with different time ranges are inner-joined."""
        # precip: 2000-2019 (240 months)
        precip_da, _ = _make_palmer_dataarrays(n_months=240, start="2000-01")
        # PET: 2005-2024 (240 months)
        _, pet_da = _make_palmer_dataarrays(n_months=240, start="2005-01")

        # overlap: 2005-2019 = 180 months
        with pytest.warns(match="alignment"):
            ds = palmer_xarray(precip_da, pet_da, awc=_AWC)

        # output should have 180 time steps (the intersection)
        assert ds.dims["time"] == 180

    def test_no_overlap_raises(self) -> None:
        """Raises CoordinateValidationError when no time overlap."""
        precip_da, _ = _make_palmer_dataarrays(n_months=120, start="1990-01")
        _, pet_da = _make_palmer_dataarrays(n_months=120, start="2010-01")

        from climate_indices.exceptions import CoordinateValidationError

        with pytest.raises(CoordinateValidationError, match="No overlapping"):
            palmer_xarray(precip_da, pet_da, awc=_AWC)
