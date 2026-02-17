"""Tests for EDDI xarray adapter integration.

Validates that the xarray-adapted EDDI produces numerically identical results
to the NumPy path, and that CF metadata is correctly applied to the output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, indices
from climate_indices.typed_public_api import eddi as eddi_typed
from climate_indices.xarray_adapter import CF_METADATA


class TestEddiXarrayEquivalence:
    """Verify EDDI xarray results match NumPy reference computations."""

    @pytest.fixture()
    def synthetic_pet_monthly(self) -> dict[str, Any]:
        """30 years of monthly synthetic PET with calibration parameters."""
        rng = np.random.default_rng(456)
        n_years = 30
        data_start_year = 1990
        months = np.tile(np.arange(12), n_years)
        seasonal = 50.0 + 30.0 * np.sin(2.0 * np.pi * months / 12.0)
        noise = rng.normal(0, 5.0, size=n_years * 12)
        pet = np.maximum(seasonal + noise, 0.0)
        return {
            "pet": pet,
            "data_start_year": data_start_year,
            "calibration_year_initial": data_start_year,
            "calibration_year_final": data_start_year + n_years - 1,
            "n_years": n_years,
        }

    @pytest.mark.parametrize("scale", [1, 3, 6])
    def test_eddi_1d_equivalence(
        self,
        synthetic_pet_monthly: dict[str, Any],
        scale: int,
    ) -> None:
        """1D xarray EDDI should match NumPy EDDI within 1e-8."""
        params = synthetic_pet_monthly
        pet = params["pet"]

        # compute via NumPy path
        numpy_result = indices.eddi(
            pet_values=pet,
            scale=scale,
            data_start_year=params["data_start_year"],
            calibration_year_initial=params["calibration_year_initial"],
            calibration_year_final=params["calibration_year_final"],
            periodicity=compute.Periodicity.monthly,
        )

        # wrap as DataArray and compute via xarray typed API
        time = pd.date_range(
            f"{params['data_start_year']}-01-01",
            periods=len(pet),
            freq="MS",
        )
        da = xr.DataArray(
            pet,
            coords={"time": time},
            dims=["time"],
            attrs={"units": "mm"},
        )

        xarray_result = eddi_typed(
            pet_values=da,
            scale=scale,
            calibration_year_initial=params["calibration_year_initial"],
            calibration_year_final=params["calibration_year_final"],
        )

        # verify type and shape
        assert isinstance(xarray_result, xr.DataArray)
        assert xarray_result.shape == numpy_result.shape

        # verify numerical equivalence
        np.testing.assert_allclose(
            xarray_result.values,
            numpy_result,
            atol=1e-8,
            rtol=1e-8,
            equal_nan=True,
            err_msg=f"EDDI scale={scale} differs between NumPy and xarray paths",
        )

    def test_eddi_xarray_preserves_coordinates(
        self,
        synthetic_pet_monthly: dict[str, Any],
    ) -> None:
        """xarray output should preserve the input time coordinate."""
        params = synthetic_pet_monthly
        time = pd.date_range(
            f"{params['data_start_year']}-01-01",
            periods=len(params["pet"]),
            freq="MS",
        )
        da = xr.DataArray(
            params["pet"],
            coords={"time": time},
            dims=["time"],
        )

        result = eddi_typed(
            pet_values=da,
            scale=1,
            calibration_year_initial=params["calibration_year_initial"],
            calibration_year_final=params["calibration_year_final"],
        )

        assert isinstance(result, xr.DataArray)
        assert "time" in result.dims
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(result.coords["time"].values),
            time,
        )

    def test_eddi_xarray_infers_parameters(
        self,
        synthetic_pet_monthly: dict[str, Any],
    ) -> None:
        """xarray adapter should infer data_start_year and periodicity from coords."""
        params = synthetic_pet_monthly
        time = pd.date_range(
            f"{params['data_start_year']}-01-01",
            periods=len(params["pet"]),
            freq="MS",
        )
        da = xr.DataArray(
            params["pet"],
            coords={"time": time},
            dims=["time"],
        )

        # compute with explicit params for reference
        explicit_result = eddi_typed(
            pet_values=da,
            scale=1,
            data_start_year=params["data_start_year"],
            calibration_year_initial=params["calibration_year_initial"],
            calibration_year_final=params["calibration_year_final"],
            periodicity=compute.Periodicity.monthly,
        )

        # compute with inferred params (omit data_start_year and periodicity)
        inferred_result = eddi_typed(
            pet_values=da,
            scale=1,
            calibration_year_initial=params["calibration_year_initial"],
            calibration_year_final=params["calibration_year_final"],
        )

        np.testing.assert_allclose(
            inferred_result.values,
            explicit_result.values,
            atol=1e-10,
            equal_nan=True,
            err_msg="EDDI with inferred params should match explicit params",
        )


class TestEddiCFMetadata:
    """Verify EDDI CF metadata is correctly applied."""

    def test_eddi_cf_metadata_exists(self) -> None:
        """EDDI should have a CF metadata entry in the registry."""
        assert "eddi" in CF_METADATA
        meta = CF_METADATA["eddi"]
        assert meta["long_name"] == "Evaporative Demand Drought Index"
        assert meta["units"] == ""
        assert meta["standard_name"] == "atmosphere_water_vapor_evaporative_demand_anomaly"
        assert "Hobbins" in meta["references"]

    def test_eddi_xarray_output_has_cf_attrs(self) -> None:
        """xarray EDDI output should have CF metadata attributes."""
        rng = np.random.default_rng(789)
        pet = rng.uniform(20.0, 80.0, size=360)
        time = pd.date_range("1990-01-01", periods=360, freq="MS")
        da = xr.DataArray(pet, coords={"time": time}, dims=["time"])

        result = eddi_typed(
            pet_values=da,
            scale=1,
            calibration_year_initial=1990,
            calibration_year_final=2019,
        )

        assert isinstance(result, xr.DataArray)
        # check CF metadata attributes
        assert result.attrs["long_name"] == "Evaporative Demand Drought Index"
        assert result.attrs["units"] == ""
        assert "Hobbins" in result.attrs["references"]

    def test_eddi_xarray_output_has_calculation_metadata(self) -> None:
        """xarray EDDI output should include calculation metadata."""
        rng = np.random.default_rng(101)
        pet = rng.uniform(20.0, 80.0, size=360)
        time = pd.date_range("1990-01-01", periods=360, freq="MS")
        da = xr.DataArray(pet, coords={"time": time}, dims=["time"])

        result = eddi_typed(
            pet_values=da,
            scale=3,
            calibration_year_initial=1990,
            calibration_year_final=2019,
        )

        assert isinstance(result, xr.DataArray)
        assert result.attrs["scale"] == 3
        assert result.attrs["calibration_year_initial"] == 1990
        assert result.attrs["calibration_year_final"] == 2019

    def test_eddi_xarray_output_has_history(self) -> None:
        """xarray EDDI output should have a history attribute."""
        rng = np.random.default_rng(202)
        pet = rng.uniform(20.0, 80.0, size=360)
        time = pd.date_range("1990-01-01", periods=360, freq="MS")
        da = xr.DataArray(pet, coords={"time": time}, dims=["time"])

        result = eddi_typed(
            pet_values=da,
            scale=1,
            calibration_year_initial=1990,
            calibration_year_final=2019,
        )

        assert isinstance(result, xr.DataArray)
        assert "history" in result.attrs
        assert "EDDI" in result.attrs["history"]
