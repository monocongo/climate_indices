"""Test equivalence between NumPy and xarray computation paths.

Ensures that wrapping data in xarray DataArrays produces numerically identical
results to the original NumPy-based computations (within floating-point tolerance).
"""

from __future__ import annotations

import cftime
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import eddi, indices, percentage_of_normal, spei, spi, utils
from climate_indices.compute import Periodicity
from climate_indices.exceptions import CoordinateValidationError, InputTypeError
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


class TestSPIXarrayGregorianDailyCalendar:
    """Verify daily xarray SPI honors Gregorian calendar positions."""

    def test_daily_spi_matches_cli_style_calendar_conversion(self) -> None:
        """Daily SPI preserves Gregorian dates after 366-day computation."""
        start_year = 1999
        end_year = 2030
        time = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        values = np.stack(
            [
                np.arange(1, len(time) + 1, dtype=float),
                np.arange(1, len(time) + 1, dtype=float) * 1.5,
            ]
        )
        precip = xr.DataArray(
            values,
            coords={"location": ["west", "east"], "time": time},
            dims=["location", "time"],
        )

        expected = np.stack(
            [
                utils.transform_to_gregorian(
                    indices.spi(
                        values=utils.transform_to_366day(location_values, start_year, end_year - start_year + 1),
                        scale=3,
                        distribution=Distribution.gamma,
                        data_start_year=start_year,
                        calibration_year_initial=start_year,
                        calibration_year_final=end_year,
                        periodicity=Periodicity.daily,
                    ),
                    start_year,
                )
                for location_values in values
            ]
        )

        result = spi(
            values=precip,
            scale=3,
            distribution=Distribution.gamma,
            calibration_year_initial=start_year,
            calibration_year_final=end_year,
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == precip.dims
        xr.testing.assert_equal(result.coords["time"], precip.coords["time"])
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_daily_pnp_matches_cli_style_calendar_conversion(self) -> None:
        """Daily PNP uses calendar-aligned precipitation positions through the shared seam."""
        start_year = 1999
        end_year = 2030
        time = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        precip_values = np.arange(1, len(time) + 1, dtype=float)
        precip = xr.DataArray(precip_values, coords={"time": time}, dims=["time"])
        expected = utils.transform_to_gregorian(
            indices.percentage_of_normal(
                values=utils.transform_to_366day(precip_values, start_year, end_year - start_year + 1),
                scale=3,
                data_start_year=start_year,
                calibration_start_year=start_year,
                calibration_end_year=end_year,
                periodicity=Periodicity.daily,
            ),
            start_year,
        )

        result = percentage_of_normal(
            values=precip,
            scale=3,
            calibration_start_year=start_year,
            calibration_end_year=end_year,
        )

        assert isinstance(result, xr.DataArray)
        xr.testing.assert_equal(result.coords["time"], precip.coords["time"])
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_daily_eddi_matches_cli_style_calendar_conversion(self) -> None:
        """Daily EDDI uses calendar-aligned PET positions through the shared seam."""
        start_year = 1999
        end_year = 2030
        time = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        pet_values = np.arange(1, len(time) + 1, dtype=float)
        pet = xr.DataArray(pet_values, coords={"time": time}, dims=["time"])
        expected = utils.transform_to_gregorian(
            indices.eddi(
                pet_values=utils.transform_to_366day(pet_values, start_year, end_year - start_year + 1),
                scale=3,
                data_start_year=start_year,
                calibration_year_initial=start_year,
                calibration_year_final=end_year,
                periodicity=Periodicity.daily,
            ),
            start_year,
        )

        result = eddi(
            pet_values=pet,
            scale=3,
            calibration_year_initial=start_year,
            calibration_year_final=end_year,
        )

        assert isinstance(result, xr.DataArray)
        xr.testing.assert_equal(result.coords["time"], pet.coords["time"])
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_daily_spi_preserves_synthetic_february_29_missing_data_behavior(self) -> None:
        """A missing February neighbor propagates through the synthetic day and Timescale."""
        start_year = 1999
        end_year = 2030
        time = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        values = np.arange(1, len(time) + 1, dtype=float)
        february_28_index = time.get_loc(pd.Timestamp("1999-02-28"))
        march_1_index = time.get_loc(pd.Timestamp("1999-03-01"))
        values[february_28_index] = np.nan
        precip = xr.DataArray(values, coords={"time": time}, dims=["time"])
        expected = utils.transform_to_gregorian(
            indices.spi(
                values=utils.transform_to_366day(values, start_year, end_year - start_year + 1),
                scale=3,
                distribution=Distribution.gamma,
                data_start_year=start_year,
                calibration_year_initial=start_year,
                calibration_year_final=end_year,
                periodicity=Periodicity.daily,
            ),
            start_year,
        )

        result = spi(
            values=precip,
            scale=3,
            distribution=Distribution.gamma,
            calibration_year_initial=start_year,
            calibration_year_final=end_year,
        )

        assert isinstance(result, xr.DataArray)
        assert np.isnan(expected[february_28_index])
        assert np.isnan(expected[march_1_index])
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_daily_spei_with_dask_pet_matches_calendar_converted_reference(self) -> None:
        """A Dask-backed PET input keeps SPEI lazy and calendar-aligned."""
        start_year = 1999
        end_year = 2030
        time = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        precip_values = np.arange(100, len(time) + 100, dtype=float)
        pet_values = np.arange(1, len(time) + 1, dtype=float) / 10
        precip = xr.DataArray(precip_values, coords={"time": time}, dims=["time"])
        pet = xr.DataArray(pet_values, coords={"time": time}, dims=["time"]).chunk({"time": -1})
        total_years = end_year - start_year + 1
        expected = utils.transform_to_gregorian(
            indices.spei(
                precips_mm=utils.transform_to_366day(precip_values, start_year, total_years),
                pet_mm=utils.transform_to_366day(pet_values, start_year, total_years),
                scale=3,
                distribution=Distribution.gamma,
                periodicity=Periodicity.daily,
                data_start_year=start_year,
                calibration_year_initial=start_year,
                calibration_year_final=end_year,
            ),
            start_year,
        )

        result = spei(
            precips_mm=precip,
            pet_mm=pet,
            scale=3,
            distribution=Distribution.gamma,
            calibration_year_initial=start_year,
            calibration_year_final=end_year,
        )

        assert isinstance(result, xr.DataArray)
        assert result.chunks is not None
        np.testing.assert_allclose(result.compute().values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_daily_spei_rejects_cftime_secondary_before_alignment(self) -> None:
        """Daily SPEI rejects cftime PET instead of aligning incompatible calendars."""
        primary_time = pd.date_range("2000-01-01", periods=3, freq="D")
        precip = xr.DataArray([1.0, 2.0, 3.0], coords={"time": primary_time}, dims=["time"])
        pet = xr.DataArray(
            [0.1, 0.2, 0.3],
            coords={
                "time": [
                    cftime.DatetimeNoLeap(2000, 1, 1),
                    cftime.DatetimeNoLeap(2000, 1, 2),
                    cftime.DatetimeNoLeap(2000, 1, 3),
                ]
            },
            dims=["time"],
        )

        with pytest.raises(CoordinateValidationError, match="cftime calendars are not supported"):
            spei(
                precips_mm=precip,
                pet_mm=pet,
                scale=1,
                distribution=Distribution.gamma,
            )

    def test_daily_spei_rejects_secondary_with_unsupported_calendar(self) -> None:
        """Daily SPEI validates the calendar declaration on both xarray inputs."""
        time = pd.date_range("2000-01-01", periods=3, freq="D")
        precip = xr.DataArray([1.0, 2.0, 3.0], coords={"time": time}, dims=["time"])
        pet = xr.DataArray([0.1, 0.2, 0.3], coords={"time": time}, dims=["time"])
        pet.coords["time"].attrs["calendar"] = "noleap"

        with pytest.raises(CoordinateValidationError, match="Unsupported calendar"):
            spei(
                precips_mm=precip,
                pet_mm=pet,
                scale=1,
                distribution=Distribution.gamma,
            )

    def test_daily_spei_rejects_multi_chunked_dask_secondary(self) -> None:
        """Daily SPEI enforces the one-time-chunk invariant on PET as well as precipitation."""
        time = pd.date_range("2000-01-01", periods=3, freq="D")
        precip = xr.DataArray([1.0, 2.0, 3.0], coords={"time": time}, dims=["time"])
        pet = xr.DataArray([0.1, 0.2, 0.3], coords={"time": time}, dims=["time"]).chunk({"time": 1})

        with pytest.raises(CoordinateValidationError, match="split across 3 chunks"):
            spei(
                precips_mm=precip,
                pet_mm=pet,
                scale=1,
                distribution=Distribution.gamma,
            )

    def test_daily_spei_rejects_numpy_pet_secondary(self) -> None:
        """Daily SPEI requires a coordinate-bearing PET input for calendar adaptation."""
        time = pd.date_range("2000-01-01", periods=3, freq="D")
        precip = xr.DataArray([1.0, 2.0, 3.0], coords={"time": time}, dims=["time"])

        with pytest.raises(InputTypeError, match="requires 'pet_mm' to be an xarray.DataArray"):
            spei(
                precips_mm=precip,
                pet_mm=np.array([0.1, 0.2, 0.3]),
                scale=1,
                distribution=Distribution.gamma,
            )

    def test_daily_spei_matches_cli_style_calendar_conversion(self) -> None:
        """Daily SPEI adapts both xarray time series at the shared calendar seam."""
        start_year = 1999
        end_year = 2030
        time = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        precip_values = np.arange(100, len(time) + 100, dtype=float)
        pet_values = np.arange(1, len(time) + 1, dtype=float) / 10
        precip = xr.DataArray(precip_values, coords={"time": time}, dims=["time"])
        pet = xr.DataArray(pet_values, coords={"time": time}, dims=["time"])
        total_years = end_year - start_year + 1
        expected = utils.transform_to_gregorian(
            indices.spei(
                precips_mm=utils.transform_to_366day(precip_values, start_year, total_years),
                pet_mm=utils.transform_to_366day(pet_values, start_year, total_years),
                scale=3,
                distribution=Distribution.gamma,
                periodicity=Periodicity.daily,
                data_start_year=start_year,
                calibration_year_initial=start_year,
                calibration_year_final=end_year,
            ),
            start_year,
        )

        result = spei(
            precips_mm=precip,
            pet_mm=pet,
            scale=3,
            distribution=Distribution.gamma,
            calibration_year_initial=start_year,
            calibration_year_final=end_year,
        )

        assert isinstance(result, xr.DataArray)
        xr.testing.assert_equal(result.coords["time"], precip.coords["time"])
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_daily_spi_supports_partial_final_year(self) -> None:
        """Daily SPI restores only the observed days of a partial final year."""
        start_year = 1999
        end_year = 2030
        time = pd.date_range(f"{start_year}-01-01", f"{end_year}-06-30", freq="D")
        values = np.arange(1, len(time) + 1, dtype=float)
        precip = xr.DataArray(values, coords={"time": time}, dims=["time"])
        all_leap_values = utils.transform_to_366day(values, start_year, end_year - start_year + 1)
        all_leap_spi = indices.spi(
            values=all_leap_values,
            scale=3,
            distribution=Distribution.gamma,
            data_start_year=start_year,
            calibration_year_initial=start_year,
            calibration_year_final=end_year,
            periodicity=Periodicity.daily,
        )
        expected = utils.transform_to_gregorian(all_leap_spi, start_year)[: len(values)]

        result = spi(
            values=precip,
            scale=3,
            distribution=Distribution.gamma,
            calibration_year_initial=start_year,
            calibration_year_final=end_year,
        )

        assert isinstance(result, xr.DataArray)
        xr.testing.assert_equal(result.coords["time"], precip.coords["time"])
        np.testing.assert_allclose(result.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)

    def test_spi_rejects_cftime_calendar(self) -> None:
        """SPI explicitly rejects cftime calendars in the initial calendar slice."""
        precip = xr.DataArray(
            [1.0, 2.0, 3.0],
            coords={
                "time": [
                    cftime.DatetimeNoLeap(2000, 1, 1),
                    cftime.DatetimeNoLeap(2000, 1, 2),
                    cftime.DatetimeNoLeap(2000, 1, 3),
                ]
            },
            dims=["time"],
        )

        with pytest.raises(CoordinateValidationError, match="cftime calendars are not supported"):
            spi(values=precip, scale=1, distribution=Distribution.gamma)

    def test_spi_rejects_daily_input_that_does_not_begin_on_january_first(self) -> None:
        """Daily SPI fails rather than treating a partial first year as calendar-aligned."""
        precip = xr.DataArray(
            [1.0, 2.0, 3.0],
            coords={"time": pd.date_range("2000-01-02", periods=3, freq="D")},
            dims=["time"],
        )

        with pytest.raises(CoordinateValidationError, match="begin on January 1"):
            spi(values=precip, scale=1, distribution=Distribution.gamma)

    def test_spi_rejects_monthly_input_that_does_not_begin_in_january(self) -> None:
        """Monthly SPI fails rather than reinterpreting a non-January origin."""
        precip = xr.DataArray(
            [1.0, 2.0, 3.0],
            coords={"time": pd.date_range("2000-03-01", periods=3, freq="MS")},
            dims=["time"],
        )

        with pytest.raises(CoordinateValidationError, match="begin in January"):
            spi(values=precip, scale=1, distribution=Distribution.gamma)

    def test_daily_spi_dask_matches_cli_style_calendar_conversion(self) -> None:
        """Dask-backed daily SPI stays lazy and preserves Gregorian dates."""
        start_year = 1999
        end_year = 2030
        time = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
        values = np.arange(1, len(time) + 1, dtype=float)
        precip = xr.DataArray(values, coords={"time": time}, dims=["time"]).chunk({"time": -1})
        expected = utils.transform_to_gregorian(
            indices.spi(
                values=utils.transform_to_366day(values, start_year, end_year - start_year + 1),
                scale=3,
                distribution=Distribution.gamma,
                data_start_year=start_year,
                calibration_year_initial=start_year,
                calibration_year_final=end_year,
                periodicity=Periodicity.daily,
            ),
            start_year,
        )

        result = spi(
            values=precip,
            scale=3,
            distribution=Distribution.gamma,
            calibration_year_initial=start_year,
            calibration_year_final=end_year,
        )

        assert isinstance(result, xr.DataArray)
        assert result.chunks is not None
        computed = result.compute()
        xr.testing.assert_equal(computed.coords["time"], precip.coords["time"])
        np.testing.assert_allclose(computed.values, expected, atol=1e-8, rtol=1e-7, equal_nan=True)


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
    """Verify PET xarray results match NumPy reference computations."""

    def test_pet_thornthwaite_equivalence(
        self,
        bench_monthly_temp_np: np.ndarray,
        bench_monthly_temp_da: xr.DataArray,
    ):
        """Thornthwaite PET xarray should match NumPy computation.

        Verifies that wrapping monthly temperature data in xarray DataArray
        produces numerically identical PET results to the NumPy path.
        """
        # import here to avoid circular dependency
        from climate_indices.xarray_adapter import pet_thornthwaite

        # latitude for testing - typical mid-latitude location
        latitude = 40.0

        # compute via NumPy path
        numpy_result = indices.pet(
            temperature_celsius=bench_monthly_temp_np,
            latitude_degrees=latitude,
            data_start_year=1980,
        )

        # compute via xarray path (fixture already has proper time coords)
        xarray_result = pet_thornthwaite(
            temperature=bench_monthly_temp_da,
            latitude=latitude,
        )

        # verify equivalence
        assert isinstance(xarray_result, xr.DataArray)

        np.testing.assert_allclose(
            xarray_result.values,
            numpy_result,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
            err_msg="PET Thornthwaite differs between NumPy and xarray paths",
        )

    def test_pet_hargreaves_equivalence(
        self,
        bench_daily_tmin_np: np.ndarray,
        bench_daily_tmax_np: np.ndarray,
        bench_daily_tmin_da: xr.DataArray,
        bench_daily_tmax_da: xr.DataArray,
    ):
        """Hargreaves PET xarray should match CLI-style calendar conversion.

        The NumPy core groups daily values into 366 positional slots per year, so
        the xarray path converts Gregorian input to the all-leap calendar before
        computing and restores Gregorian positions afterward. The reference here is
        therefore the explicit CLI-style conversion, not the raw positional NumPy
        result, which would leave every day after February 28 of a non-leap year
        shifted.
        """
        # import here to avoid circular dependency
        from climate_indices.eto import eto_hargreaves
        from climate_indices.xarray_adapter import pet_hargreaves

        # latitude for testing - typical mid-latitude location
        latitude = 40.0

        # the fixture coordinate starts on January 1 and ends mid-year, so the
        # reference spans whole calendar years and is trimmed to observed days
        time_coord = bench_daily_tmin_da.coords["time"]
        start_year = pd.Timestamp(time_coord.values[0]).year
        end_year = pd.Timestamp(time_coord.values[-1]).year
        total_years = end_year - start_year + 1

        # utils.transform_to_366day requires whole calendar years, so pad the partial
        # final year with NaN; Hargreaves is a per-day formula with no cross-year
        # fitting, so the padded positions stay NaN and are trimmed off below
        full_length = len(pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D"))
        padding = np.full(full_length - len(bench_daily_tmin_np), np.nan)
        all_leap_tmin = utils.transform_to_366day(np.append(bench_daily_tmin_np, padding), start_year, total_years)
        all_leap_tmax = utils.transform_to_366day(np.append(bench_daily_tmax_np, padding), start_year, total_years)
        all_leap_tmean = (all_leap_tmin + all_leap_tmax) / 2.0
        expected = utils.transform_to_gregorian(
            eto_hargreaves(
                daily_tmin_celsius=all_leap_tmin,
                daily_tmax_celsius=all_leap_tmax,
                daily_tmean_celsius=all_leap_tmean,
                latitude_degrees=latitude,
            ),
            start_year,
        )[: len(bench_daily_tmin_np)]

        # compute via xarray path (pet_hargreaves derives tmean automatically)
        xarray_result = pet_hargreaves(
            daily_tmin_celsius=bench_daily_tmin_da,
            daily_tmax_celsius=bench_daily_tmax_da,
            latitude=latitude,
        )

        # verify equivalence
        assert isinstance(xarray_result, xr.DataArray)
        xr.testing.assert_equal(xarray_result.coords["time"], time_coord)

        np.testing.assert_allclose(
            xarray_result.values,
            expected,
            atol=1e-8,
            rtol=1e-7,
            equal_nan=True,
            err_msg="PET Hargreaves differs from CLI-style calendar conversion",
        )

    def test_pet_hargreaves_differs_from_raw_positional_result(
        self,
        bench_daily_tmin_np: np.ndarray,
        bench_daily_tmax_np: np.ndarray,
        bench_daily_tmin_da: xr.DataArray,
        bench_daily_tmax_da: xr.DataArray,
    ):
        """Hargreaves PET no longer reuses drifted positional NumPy output."""
        from climate_indices.eto import eto_hargreaves
        from climate_indices.xarray_adapter import pet_hargreaves

        latitude = 40.0
        tmean_np = (bench_daily_tmin_np + bench_daily_tmax_np) / 2.0
        drifted = eto_hargreaves(
            daily_tmin_celsius=bench_daily_tmin_np,
            daily_tmax_celsius=bench_daily_tmax_np,
            daily_tmean_celsius=tmean_np,
            latitude_degrees=latitude,
        )

        xarray_result = pet_hargreaves(
            daily_tmin_celsius=bench_daily_tmin_da,
            daily_tmax_celsius=bench_daily_tmax_da,
            latitude=latitude,
        )

        assert isinstance(xarray_result, xr.DataArray)
        # the first non-leap year matches up to February 28 (positions 0-58) and
        # diverges from March 1 onward, once the synthetic February 29 shifts the core
        np.testing.assert_allclose(xarray_result.values[:59], drifted[:59], atol=1e-8, rtol=1e-7)
        assert not np.allclose(xarray_result.values[59:], drifted[59:], atol=1e-8, rtol=1e-7)


class TestPETXarrayCalendarSemantics:
    """Verify the PET adapters enforce the shared xarray calendar contract."""

    @staticmethod
    def _daily_temps(start: str, periods: int) -> tuple[xr.DataArray, xr.DataArray]:
        """Build aligned daily tmin/tmax DataArrays over a Gregorian coordinate."""
        time = pd.date_range(start, periods=periods, freq="D")
        tmin = xr.DataArray(np.full(periods, 5.0), coords={"time": time}, dims=["time"])
        tmax = xr.DataArray(np.full(periods, 20.0), coords={"time": time}, dims=["time"])
        return tmin, tmax

    def test_hargreaves_rejects_cftime_calendar(self) -> None:
        """Daily PET rejects cftime calendars rather than drifting positions."""
        from climate_indices.xarray_adapter import pet_hargreaves

        time = [cftime.DatetimeNoLeap(2000, 1, day) for day in range(1, 4)]
        tmin = xr.DataArray([5.0, 5.0, 5.0], coords={"time": time}, dims=["time"])
        tmax = xr.DataArray([20.0, 20.0, 20.0], coords={"time": time}, dims=["time"])

        with pytest.raises(CoordinateValidationError, match="cftime calendars are not supported"):
            pet_hargreaves(daily_tmin_celsius=tmin, daily_tmax_celsius=tmax, latitude=40.0)

    def test_hargreaves_rejects_daily_input_that_does_not_begin_on_january_first(self) -> None:
        """Daily PET fails rather than treating a partial first year as calendar-aligned."""
        from climate_indices.xarray_adapter import pet_hargreaves

        tmin, tmax = self._daily_temps("2000-01-02", 400)

        with pytest.raises(CoordinateValidationError, match="begin on January 1"):
            pet_hargreaves(daily_tmin_celsius=tmin, daily_tmax_celsius=tmax, latitude=40.0)

    def test_thornthwaite_rejects_monthly_input_that_does_not_begin_in_january(self) -> None:
        """Monthly PET fails rather than reinterpreting a non-January origin."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        time = pd.date_range("2000-03-01", periods=24, freq="MS")
        temperature = xr.DataArray(np.full(24, 15.0), coords={"time": time}, dims=["time"])

        with pytest.raises(CoordinateValidationError, match="begin in January"):
            pet_thornthwaite(temperature=temperature, latitude=40.0)

    def test_thornthwaite_rejects_daily_coordinate(self) -> None:
        """Monthly PET rejects a daily coordinate instead of misreading it as months."""
        from climate_indices.xarray_adapter import pet_thornthwaite

        time = pd.date_range("2000-01-01", periods=400, freq="D")
        temperature = xr.DataArray(np.full(400, 15.0), coords={"time": time}, dims=["time"])

        with pytest.raises(CoordinateValidationError, match="does not match the requested monthly"):
            pet_thornthwaite(temperature=temperature, latitude=40.0)
