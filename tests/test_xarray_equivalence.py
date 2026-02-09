"""Test equivalence between NumPy and xarray computation paths.

Ensures that wrapping data in xarray DataArrays produces numerically identical
results to the original NumPy-based computations (within floating-point tolerance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import indices, spei, spi
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution


class TestSPIXarrayEquivalence:
    """Verify SPI xarray results match NumPy reference computations."""

    @pytest.mark.parametrize("scale", [1, 3, 6, 12])
    @pytest.mark.parametrize(
        "distribution",
        [Distribution.gamma, Distribution.pearson],
    )
    def test_spi_1d_equivalence(
        self,
        precips_mm_monthly: np.ndarray,
        calibration_year_start_monthly: int,
        calibration_year_end_monthly: int,
        data_year_start_monthly: int,
        scale: int,
        distribution: Distribution,
    ):
        """1D xarray SPI should match NumPy SPI computation.

        Tests across multiple scales (1, 3, 6, 12 months) and distributions
        (gamma, pearson) to ensure consistent numerical results.
        """
        # compute via NumPy path
        numpy_result = indices.spi(
            values=precips_mm_monthly,
            scale=scale,
            distribution=distribution,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=Periodicity.monthly,
        )

        # wrap as DataArray and compute via xarray path
        # flatten the 2D (years, months) array to 1D time series
        precip_flat = precips_mm_monthly.flatten()
        time = pd.date_range(
            f"{data_year_start_monthly}-01-01",
            periods=len(precip_flat),
            freq="MS",
        )
        da = xr.DataArray(
            precip_flat,
            coords={"time": time},
            dims=["time"],
            attrs={"units": "mm"},
        )

        xarray_result = spi(
            values=da,
            scale=scale,
            distribution=distribution,
            # use explicit params to match NumPy path
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
        )

        # verify equivalence
        assert isinstance(xarray_result, xr.DataArray)

        # tolerance depends on distribution
        # gamma uses more stable numerical methods
        atol = 1e-8 if distribution == Distribution.gamma else 1e-5

        # numpy_result might be 2D, flatten it for comparison
        numpy_result_flat = numpy_result.flatten() if numpy_result.ndim > 1 else numpy_result

        np.testing.assert_allclose(
            xarray_result.values,
            numpy_result_flat,
            atol=atol,
            rtol=1e-7,
            equal_nan=True,
            err_msg=f"SPI scale={scale} distribution={distribution} differs between NumPy and xarray paths",
        )

    def test_spi_with_reference_values(
        self,
        precips_mm_monthly: np.ndarray,
        spi_6_month_gamma: np.ndarray,
        data_year_start_monthly: int,
        data_year_end_monthly: int,
    ):
        """xarray SPI should match pre-computed reference values.

        Uses the existing .npy fixture as ground truth to verify both
        NumPy and xarray paths produce expected results.
        """
        # wrap as DataArray - flatten 2D array
        precip_flat = precips_mm_monthly.flatten()
        time = pd.date_range(
            f"{data_year_start_monthly}-01-01",
            periods=len(precip_flat),
            freq="MS",
        )
        da = xr.DataArray(
            precip_flat,
            coords={"time": time},
            dims=["time"],
        )

        # compute via xarray with same calibration as reference
        # reference fixture was generated with full-range calibration
        result = spi(
            values=da,
            scale=6,
            distribution=Distribution.gamma,
            calibration_year_initial=data_year_start_monthly,
            calibration_year_final=data_year_end_monthly,
        )

        # compare against reference - flatten reference if needed
        spi_ref_flat = spi_6_month_gamma.flatten() if spi_6_month_gamma.ndim > 1 else spi_6_month_gamma

        np.testing.assert_allclose(
            result.values,
            spi_ref_flat,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
            err_msg="xarray SPI-6 gamma differs from reference fixture",
        )

    def test_spi_3d_gridded_equivalence(
        self,
        gridded_monthly_precip_3d: xr.DataArray,
        calibration_year_start_monthly: int,
        calibration_year_end_monthly: int,
    ):
        """3D gridded xarray should produce equivalent results to point-by-point NumPy.

        Verifies that spatial broadcasting works correctly and produces the same
        values as iterating over spatial points with NumPy computations.
        """
        # compute via xarray on full 3D grid
        xarray_result = spi(
            values=gridded_monthly_precip_3d,
            scale=6,
            distribution=Distribution.gamma,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
        )

        # manually compute numpy result for one grid point
        sample_lat_idx = 2
        sample_lon_idx = 3
        point_data = gridded_monthly_precip_3d.values[:, sample_lat_idx, sample_lon_idx]

        # Reshape point data to (years, 12) for NumPy path
        num_years = len(point_data) // 12
        point_data_2d = point_data.reshape((num_years, 12))

        # extract data_start_year from the fixture's time coordinate
        fixture_start_year = gridded_monthly_precip_3d.time.dt.year.values[0]

        numpy_result_point = indices.spi(
            values=point_data_2d,
            scale=6,
            distribution=Distribution.gamma,
            data_start_year=fixture_start_year,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=Periodicity.monthly,
        )

        # flatten numpy result for comparison with xarray 1D time series at grid point
        numpy_result_flat = numpy_result_point.flatten() if numpy_result_point.ndim > 1 else numpy_result_point

        # verify grid point matches
        np.testing.assert_allclose(
            xarray_result.values[:, sample_lat_idx, sample_lon_idx],
            numpy_result_flat,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
            err_msg="3D xarray grid point differs from equivalent NumPy computation",
        )


class TestSPEIXarrayEquivalence:
    """Verify SPEI xarray results match NumPy reference computations."""

    @pytest.mark.parametrize("scale", [3, 6, 12])
    @pytest.mark.parametrize(
        "distribution",
        [Distribution.gamma, Distribution.pearson],
    )
    def test_spei_1d_equivalence(
        self,
        precips_mm_monthly: np.ndarray,
        pet_thornthwaite_mm: np.ndarray,
        calibration_year_start_monthly: int,
        calibration_year_end_monthly: int,
        data_year_start_monthly: int,
        scale: int,
        distribution: Distribution,
    ):
        """1D xarray SPEI should match NumPy SPEI computation."""
        # compute via NumPy path
        numpy_result = indices.spei(
            precips_mm=precips_mm_monthly,
            pet_mm=pet_thornthwaite_mm,
            scale=scale,
            distribution=distribution,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=Periodicity.monthly,
        )

        # wrap as DataArrays and compute via xarray path - flatten 2D arrays
        precip_flat = precips_mm_monthly.flatten()
        pet_flat = pet_thornthwaite_mm.flatten()
        time = pd.date_range(
            f"{data_year_start_monthly}-01-01",
            periods=len(precip_flat),
            freq="MS",
        )
        precip_da = xr.DataArray(
            precip_flat,
            coords={"time": time},
            dims=["time"],
        )
        pet_da = xr.DataArray(
            pet_flat,
            coords={"time": time},
            dims=["time"],
        )

        xarray_result = spei(
            precips_mm=precip_da,
            pet_mm=pet_da,
            scale=scale,
            distribution=distribution,
            # use explicit params to match NumPy path
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
        )

        # verify equivalence
        atol = 1e-8 if distribution == Distribution.gamma else 1e-5

        # numpy_result might be 2D, flatten it for comparison
        numpy_result_flat = numpy_result.flatten() if numpy_result.ndim > 1 else numpy_result

        np.testing.assert_allclose(
            xarray_result.values,
            numpy_result_flat,
            atol=atol,
            rtol=1e-7,
            equal_nan=True,
            err_msg=f"SPEI scale={scale} distribution={distribution} differs between NumPy and xarray paths",
        )

    def test_spei_with_reference_values(
        self,
        precips_mm_monthly: np.ndarray,
        pet_thornthwaite_mm: np.ndarray,
        spei_6_month_gamma: np.ndarray,
        data_year_start_monthly: int,
        data_year_end_monthly: int,
    ):
        """xarray SPEI should match pre-computed reference values."""
        # wrap as DataArrays - flatten 2D arrays
        precip_flat = precips_mm_monthly.flatten()
        pet_flat = pet_thornthwaite_mm.flatten()
        time = pd.date_range(
            f"{data_year_start_monthly}-01-01",
            periods=len(precip_flat),
            freq="MS",
        )
        precip_da = xr.DataArray(precip_flat, coords={"time": time}, dims=["time"])
        pet_da = xr.DataArray(pet_flat, coords={"time": time}, dims=["time"])

        # compute via xarray with same calibration as reference
        # reference fixture was generated with full-range calibration
        result = spei(
            precips_mm=precip_da,
            pet_mm=pet_da,
            scale=6,
            distribution=Distribution.gamma,
            calibration_year_initial=data_year_start_monthly,
            calibration_year_final=data_year_end_monthly,
        )

        # compare against reference - flatten reference if needed
        spei_ref_flat = spei_6_month_gamma.flatten() if spei_6_month_gamma.ndim > 1 else spei_6_month_gamma

        # use looser tolerance for SPEI (matches backward compat test tolerance)
        # SPEI involves more computation steps than SPI and accumulates small numerical errors
        np.testing.assert_allclose(
            result.values,
            spei_ref_flat,
            atol=1e-5,
            rtol=1e-7,
            equal_nan=True,
            err_msg="xarray SPEI-6 gamma differs from reference fixture",
        )


class TestPETXarrayEquivalence:
    """Verify PET xarray results match NumPy reference computations.

    These tests are stubbed pending Epic 3 implementation of PET xarray wrappers.
    """

    @pytest.mark.skip(reason="Awaiting Epic 3 PET xarray wrappers")
    def test_pet_thornthwaite_equivalence(self):
        """Thornthwaite PET xarray should match NumPy computation."""
        pass

    @pytest.mark.skip(reason="Awaiting Epic 3 PET xarray wrappers")
    def test_pet_hargreaves_equivalence(self):
        """Hargreaves PET xarray should match NumPy computation."""
        pass
