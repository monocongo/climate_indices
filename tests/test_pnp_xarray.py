"""Tests for PNP (Percentage of Normal) xarray adapter integration.

Validates xarray DataArray support for percentage_of_normal() via the
typed_public_api, including numerical equivalence with the NumPy path,
CF metadata application, and coordinate preservation.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from climate_indices import percentage_of_normal
from climate_indices.compute import Periodicity
from climate_indices.indices import percentage_of_normal as numpy_pnp


@pytest.fixture(scope="module")
def pnp_xarray_result(sample_monthly_precip_da: xr.DataArray) -> xr.DataArray:
    """Cached PNP-6 result for 1D monthly precipitation DataArray."""
    return percentage_of_normal(
        values=sample_monthly_precip_da,
        scale=6,
    )


@pytest.fixture(scope="module")
def pnp_numpy_result(sample_monthly_precip_da: xr.DataArray) -> np.ndarray:
    """PNP-6 result computed via the NumPy path for equivalence comparison."""
    # extract start year from the DataArray time coordinate
    start_year = int(sample_monthly_precip_da.time.dt.year.values[0])
    end_year = int(sample_monthly_precip_da.time.dt.year.values[-1])
    return numpy_pnp(
        values=sample_monthly_precip_da.values,
        scale=6,
        data_start_year=start_year,
        calibration_start_year=start_year,
        calibration_end_year=end_year,
        periodicity=Periodicity.monthly,
    )


class TestPNPXarrayEquivalence:
    """Validate numerical equivalence between NumPy and xarray paths."""

    def test_values_match_numpy(
        self,
        pnp_xarray_result: xr.DataArray,
        pnp_numpy_result: np.ndarray,
    ) -> None:
        """xarray PNP values should match NumPy PNP values within 1e-8."""
        np.testing.assert_allclose(
            pnp_xarray_result.values,
            pnp_numpy_result,
            atol=1e-8,
            equal_nan=True,
            err_msg="PNP xarray vs NumPy equivalence failed",
        )

    def test_output_shape_matches_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        pnp_xarray_result: xr.DataArray,
    ) -> None:
        """Output shape should match input shape."""
        assert pnp_xarray_result.shape == sample_monthly_precip_da.shape

    def test_output_is_dataarray(self, pnp_xarray_result: xr.DataArray) -> None:
        """xarray input should produce xarray output."""
        assert isinstance(pnp_xarray_result, xr.DataArray)


class TestPNPXarrayCFMetadata:
    """Validate CF Convention metadata on PNP xarray output."""

    def test_has_long_name(self, pnp_xarray_result: xr.DataArray) -> None:
        assert "long_name" in pnp_xarray_result.attrs
        assert pnp_xarray_result.attrs["long_name"] == "Percent of Normal Precipitation"

    def test_has_units(self, pnp_xarray_result: xr.DataArray) -> None:
        assert "units" in pnp_xarray_result.attrs
        assert pnp_xarray_result.attrs["units"] == "%"

    def test_has_references(self, pnp_xarray_result: xr.DataArray) -> None:
        assert "references" in pnp_xarray_result.attrs
        assert "Willeke" in pnp_xarray_result.attrs["references"]

    def test_has_version(self, pnp_xarray_result: xr.DataArray) -> None:
        assert "climate_indices_version" in pnp_xarray_result.attrs

    def test_has_history(self, pnp_xarray_result: xr.DataArray) -> None:
        assert "history" in pnp_xarray_result.attrs
        assert "PNP" in pnp_xarray_result.attrs["history"]

    def test_has_scale_in_attrs(self, pnp_xarray_result: xr.DataArray) -> None:
        assert "scale" in pnp_xarray_result.attrs
        assert pnp_xarray_result.attrs["scale"] == 6


class TestPNPCoordinatePreservation:
    """Validate coordinate preservation through PNP xarray computation."""

    def test_time_coords_match_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        pnp_xarray_result: xr.DataArray,
    ) -> None:
        """Time coordinates should exactly match input."""
        xr.testing.assert_equal(
            pnp_xarray_result.coords["time"],
            sample_monthly_precip_da.coords["time"],
        )

    def test_dims_match_input(
        self,
        sample_monthly_precip_da: xr.DataArray,
        pnp_xarray_result: xr.DataArray,
    ) -> None:
        """Dimension names and order should be preserved."""
        assert pnp_xarray_result.dims == sample_monthly_precip_da.dims

    def test_coord_keys_exact(
        self,
        sample_monthly_precip_da: xr.DataArray,
        pnp_xarray_result: xr.DataArray,
    ) -> None:
        """No extra or missing coordinate keys."""
        assert set(pnp_xarray_result.coords.keys()) == set(sample_monthly_precip_da.coords.keys())


class TestPNPNumpyPassthrough:
    """Validate NumPy path is unchanged when called through typed API."""

    def test_numpy_returns_ndarray(self) -> None:
        """NumPy input should return ndarray, not DataArray."""
        rng = np.random.default_rng(123)
        values = rng.gamma(shape=2.0, scale=50.0, size=480)
        result = percentage_of_normal(
            values=values,
            scale=6,
            data_start_year=1980,
            calibration_start_year=1980,
            calibration_end_year=2019,
            periodicity=Periodicity.monthly,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == values.shape
