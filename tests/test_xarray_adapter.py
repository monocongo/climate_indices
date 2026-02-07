"""Tests for xarray adapter decorator infrastructure.

This module tests Story 2.2 functionality:
- Parameter inference from time coordinates
- Extract → infer → compute → rewrap → log adapter contract
- CF metadata application
- NumPy passthrough behavior
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, indices
from climate_indices.exceptions import CoordinateValidationError
from climate_indices.xarray_adapter import (
    _infer_calibration_period,
    _infer_data_start_year,
    _infer_periodicity,
    xarray_adapter,
)


# fixtures for test data
@pytest.fixture
def sample_monthly_precip_da() -> xr.DataArray:
    """Create a 1D monthly precipitation DataArray (40 years, 1980-2019)."""
    # 40 years * 12 months = 480 values
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    # generate random precipitation values
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
        },
    )


@pytest.fixture
def sample_daily_precip_da() -> xr.DataArray:
    """Create a 1D daily precipitation DataArray (5 years, 2015-2019)."""
    # 5 years of daily data
    time = pd.date_range("2015-01-01", "2019-12-31", freq="D")
    # generate random precipitation values
    rng = np.random.default_rng(123)
    values = rng.gamma(shape=2.0, scale=5.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Daily Precipitation",
        },
    )


class TestXarrayAdapterNumpyPassthrough:
    """Test that numpy inputs pass through unchanged."""

    def test_numpy_array_passthrough(self, sample_monthly_precip_da):
        """Decorator with numpy input returns numpy result unmodified."""

        @xarray_adapter()
        def simple_function(values: np.ndarray, scale: int) -> np.ndarray:
            return values * scale

        numpy_input = np.array([1.0, 2.0, 3.0])
        result = simple_function(numpy_input, scale=2)

        assert isinstance(result, np.ndarray)
        assert not isinstance(result, xr.DataArray)
        np.testing.assert_array_equal(result, numpy_input * 2)

    def test_no_inference_on_numpy_input(self):
        """No parameter inference attempted on numpy input."""

        @xarray_adapter(infer_params=True)
        def needs_params(
            values: np.ndarray,
            data_start_year: int,
            periodicity: compute.Periodicity,
        ) -> np.ndarray:
            # if inference ran on numpy input, these params would be missing
            return values

        numpy_input = np.array([1.0, 2.0, 3.0])
        result = needs_params(
            numpy_input,
            data_start_year=2000,
            periodicity=compute.Periodicity.monthly,
        )

        assert isinstance(result, np.ndarray)


class TestXarrayAdapterExtractRewrap:
    """Test extraction of values and rewrapping with coordinates."""

    def test_dataarray_input_returns_dataarray(self, sample_monthly_precip_da):
        """1D DataArray input produces DataArray result with same coords/dims."""

        @xarray_adapter()
        def simple_scale(values: np.ndarray, scale: int) -> np.ndarray:
            return values * scale

        result = simple_scale(
            sample_monthly_precip_da,
            scale=2,
            data_start_year=1980,
            periodicity=compute.Periodicity.monthly,
        )

        assert isinstance(result, xr.DataArray)
        assert result.dims == sample_monthly_precip_da.dims
        assert result.coords.keys() == sample_monthly_precip_da.coords.keys()
        xr.testing.assert_equal(result.coords["time"], sample_monthly_precip_da.coords["time"])

    def test_result_values_match_numpy_function(self, sample_monthly_precip_da):
        """Result values match calling numpy function directly."""

        @xarray_adapter()
        def simple_scale(values: np.ndarray, scale: int) -> np.ndarray:
            return values * scale

        result_da = simple_scale(
            sample_monthly_precip_da,
            scale=3,
            data_start_year=1980,
            periodicity=compute.Periodicity.monthly,
        )

        # manually compute what the result should be
        expected_values = sample_monthly_precip_da.values * 3

        np.testing.assert_array_equal(result_da.values, expected_values)

    def test_input_attributes_preserved(self, sample_monthly_precip_da):
        """Input attributes are preserved in output."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        assert result.attrs["units"] == "mm"
        assert result.attrs["long_name"] == "Monthly Precipitation"


class TestXarrayAdapterCFMetadata:
    """Test CF metadata application."""

    def test_cf_metadata_applied(self, sample_monthly_precip_da):
        """CF metadata dict applied to output attributes."""
        cf_meta = {
            "standard_name": "standardized_precipitation_index",
            "units": "1",
        }

        @xarray_adapter(cf_metadata=cf_meta)
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        assert result.attrs["standard_name"] == "standardized_precipitation_index"
        assert result.attrs["units"] == "1"

    def test_cf_metadata_overrides_conflicts(self, sample_monthly_precip_da):
        """CF metadata overrides conflicting input attributes."""
        # input has units="mm"
        cf_meta = {"units": "dimensionless"}

        @xarray_adapter(cf_metadata=cf_meta)
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        # CF metadata should win
        assert result.attrs["units"] == "dimensionless"

    def test_no_cf_metadata_preserves_input_attrs(self, sample_monthly_precip_da):
        """No CF metadata (None) preserves only input attrs."""

        @xarray_adapter(cf_metadata=None)
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        assert result.attrs == sample_monthly_precip_da.attrs


class TestXarrayAdapterParameterInference:
    """Test automatic parameter inference from time coordinates."""

    def test_infer_data_start_year(self, sample_monthly_precip_da):
        """data_start_year inferred from time coordinate."""

        @xarray_adapter(infer_params=True)
        def needs_start_year(
            values: np.ndarray,
            data_start_year: int,
        ) -> np.ndarray:
            # if inference worked, data_start_year should be 1980
            assert data_start_year == 1980
            return values

        result = needs_start_year(sample_monthly_precip_da)
        assert isinstance(result, xr.DataArray)

    def test_infer_periodicity_monthly(self, sample_monthly_precip_da):
        """periodicity inferred as monthly from monthly time series."""

        @xarray_adapter(infer_params=True)
        def needs_periodicity(
            values: np.ndarray,
            periodicity: compute.Periodicity,
        ) -> np.ndarray:
            assert periodicity == compute.Periodicity.monthly
            return values

        result = needs_periodicity(sample_monthly_precip_da)
        assert isinstance(result, xr.DataArray)

    def test_infer_periodicity_daily(self, sample_daily_precip_da):
        """periodicity inferred as daily from daily time series."""

        @xarray_adapter(infer_params=True)
        def needs_periodicity(
            values: np.ndarray,
            periodicity: compute.Periodicity,
        ) -> np.ndarray:
            assert periodicity == compute.Periodicity.daily
            return values

        result = needs_periodicity(sample_daily_precip_da)
        assert isinstance(result, xr.DataArray)

    def test_infer_calibration_period_defaults_to_full_range(self, sample_monthly_precip_da):
        """Calibration period defaults to full time range."""

        @xarray_adapter(infer_params=True)
        def needs_calibration(
            values: np.ndarray,
            calibration_year_initial: int,
            calibration_year_final: int,
        ) -> np.ndarray:
            # sample data is 1980-2019
            assert calibration_year_initial == 1980
            assert calibration_year_final == 2019
            return values

        result = needs_calibration(sample_monthly_precip_da)
        assert isinstance(result, xr.DataArray)

    def test_explicit_params_override_inferred(self, sample_monthly_precip_da):
        """Explicit parameters override inferred values."""

        @xarray_adapter(infer_params=True)
        def needs_start_year(
            values: np.ndarray,
            data_start_year: int,
        ) -> np.ndarray:
            # explicit value should be used, not inferred 1980
            assert data_start_year == 1990
            return values

        # explicitly provide 1990 even though time coord starts at 1980
        result = needs_start_year(sample_monthly_precip_da, data_start_year=1990)
        assert isinstance(result, xr.DataArray)

    def test_infer_params_false_disables_inference(self, sample_monthly_precip_da):
        """infer_params=False disables all inference."""

        @xarray_adapter(infer_params=False)
        def needs_params(
            values: np.ndarray,
            data_start_year: int,
            periodicity: compute.Periodicity,
        ) -> np.ndarray:
            return values

        # must provide all params explicitly when inference disabled
        result = needs_params(
            sample_monthly_precip_da,
            data_start_year=1980,
            periodicity=compute.Periodicity.monthly,
        )
        assert isinstance(result, xr.DataArray)


class TestXarrayAdapterInferenceHelpers:
    """Test the private inference helper functions."""

    def test_infer_data_start_year_returns_correct_year(self):
        """_infer_data_start_year extracts year from first timestamp."""
        time = pd.date_range("1985-03-01", periods=12, freq="MS")
        time_da = xr.DataArray(time, dims=["time"])

        year = _infer_data_start_year(time_da)
        assert year == 1985

    def test_infer_data_start_year_empty_raises(self):
        """_infer_data_start_year raises CoordinateValidationError for empty coord."""
        time_da = xr.DataArray([], dims=["time"])

        with pytest.raises(CoordinateValidationError) as exc_info:
            _infer_data_start_year(time_da)

        assert "empty" in str(exc_info.value).lower()

    def test_infer_periodicity_monthly(self):
        """_infer_periodicity maps MS to Periodicity.monthly."""
        time = pd.date_range("2000-01-01", periods=24, freq="MS")
        time_da = xr.DataArray(time, dims=["time"])

        periodicity = _infer_periodicity(time_da)
        assert periodicity == compute.Periodicity.monthly

    def test_infer_periodicity_daily(self):
        """_infer_periodicity maps D to Periodicity.daily."""
        time = pd.date_range("2000-01-01", periods=100, freq="D")
        time_da = xr.DataArray(time, dims=["time"])

        periodicity = _infer_periodicity(time_da)
        assert periodicity == compute.Periodicity.daily

    def test_infer_periodicity_unsupported_raises(self):
        """_infer_periodicity raises CoordinateValidationError for unsupported freq."""
        # hourly frequency not supported
        time = pd.date_range("2000-01-01", periods=100, freq="h")
        time_da = xr.DataArray(time, dims=["time"])

        with pytest.raises(CoordinateValidationError) as exc_info:
            _infer_periodicity(time_da)

        assert "unsupported frequency" in str(exc_info.value).lower()

    def test_infer_periodicity_irregular_raises(self):
        """_infer_periodicity raises CoordinateValidationError when freq cannot be inferred."""
        # irregular spacing
        time = pd.to_datetime(["2000-01-01", "2000-01-15", "2000-02-01"])
        time_da = xr.DataArray(time, dims=["time"])

        with pytest.raises(CoordinateValidationError) as exc_info:
            _infer_periodicity(time_da)

        assert "could not infer frequency" in str(exc_info.value).lower()

    def test_infer_calibration_period_returns_endpoints(self):
        """_infer_calibration_period returns (first_year, last_year)."""
        time = pd.date_range("1990-01-01", "1999-12-01", freq="MS")
        time_da = xr.DataArray(time, dims=["time"])

        cal_start, cal_end = _infer_calibration_period(time_da)
        assert cal_start == 1990
        assert cal_end == 1999


class TestXarrayAdapterLogging:
    """Test logging events from the decorator."""

    def test_logs_completion_event(self, sample_monthly_precip_da, caplog):
        """Logs xarray_adapter_completed event on success."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        with caplog.at_level("INFO"):
            _ = identity(sample_monthly_precip_da)

        # check that completion event was logged
        assert any("xarray_adapter_completed" in record.message for record in caplog.records)

    def test_logs_include_shapes(self, sample_monthly_precip_da, caplog):
        """Logs include input/output shape information."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        with caplog.at_level("INFO"):
            _ = identity(sample_monthly_precip_da)

        # verify shape info in logs
        log_messages = [record.message for record in caplog.records]
        assert any("input_shape" in msg for msg in log_messages)


class TestXarrayAdapterIntegration:
    """Integration tests with real SPI function."""

    def test_works_with_actual_spi_function(self, sample_monthly_precip_da):
        """Decorator works with the actual indices.spi() function."""
        # wrap the real SPI function
        wrapped_spi = xarray_adapter(
            cf_metadata={
                "standard_name": "standardized_precipitation_index",
                "units": "1",
            }
        )(indices.spi)

        # call with xarray DataArray - params will be inferred
        result = wrapped_spi(
            sample_monthly_precip_da,
            scale=3,
            distribution=indices.Distribution.gamma,
        )

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape
        assert result.attrs["standard_name"] == "standardized_precipitation_index"
        assert result.attrs["units"] == "1"

    def test_numpy_input_with_real_spi(self, sample_monthly_precip_da):
        """NumPy passthrough works with real SPI function."""
        wrapped_spi = xarray_adapter()(indices.spi)

        # call with numpy array
        numpy_values = sample_monthly_precip_da.values
        result = wrapped_spi(
            numpy_values,
            scale=3,
            distribution=indices.Distribution.gamma,
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )

        # should return numpy array, not DataArray
        assert isinstance(result, np.ndarray)
        assert not isinstance(result, xr.DataArray)
