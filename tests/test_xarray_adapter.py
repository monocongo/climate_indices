"""Tests for xarray adapter decorator infrastructure.

This module tests Story 2.2 functionality:
- Parameter inference from time coordinates
- Extract → infer → compute → rewrap → log adapter contract
- CF metadata application
- NumPy passthrough behavior
"""

from __future__ import annotations

import datetime
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, indices
from climate_indices.exceptions import CoordinateValidationError
from climate_indices.xarray_adapter import (
    CF_METADATA,
    _append_history,
    _build_history_entry,
    _build_output_dataarray,
    _infer_calibration_period,
    _infer_data_start_year,
    _infer_periodicity,
    _serialize_attr_value,
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


@pytest.fixture
def coord_rich_1d_da() -> xr.DataArray:
    """Create 1D DataArray with rich coordinate metadata.

    Features:
    - time dimension coord with multiple attrs (axis, calendar, long_name, bounds, standard_name)
    - month non-dimension auxiliary coord
    - station_id scalar coord with attrs
    """
    # 36 months (3 years)
    time = pd.date_range("2020-01-01", "2022-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    # auxiliary coordinate: month numbers
    month = xr.DataArray(
        [t.month for t in time],
        dims=["time"],
        attrs={"long_name": "month of year", "units": "1"},
    )

    # scalar coordinate with attrs
    station_id = xr.DataArray(
        "GHCN-12345",
        attrs={"long_name": "station identifier", "cf_role": "timeseries_id"},
    )

    da = xr.DataArray(
        values,
        coords={
            "time": time,
            "month": month,
            "station_id": station_id,
        },
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation",
        },
        name="precip",
    )

    # add rich attrs to time coord
    da.coords["time"].attrs = {
        "axis": "T",
        "calendar": "standard",
        "long_name": "time",
        "bounds": "time_bounds",
        "standard_name": "time",
    }

    return da


@pytest.fixture
def multi_coord_da() -> xr.DataArray:
    """Create 2D DataArray with multiple dimension coords and rich metadata.

    Features:
    - time and lat dimension coords, each with rich attrs
    - month auxiliary coord (time-dependent)
    - ensemble scalar coord with attrs
    """
    time = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    lat = np.array([30.0, 35.0, 40.0, 45.0])
    rng = np.random.default_rng(99)
    values = rng.gamma(shape=2.0, scale=50.0, size=(len(time), len(lat)))

    # auxiliary coordinate
    month = xr.DataArray(
        [t.month for t in time],
        dims=["time"],
        attrs={"long_name": "month of year"},
    )

    # scalar coordinate
    ensemble = xr.DataArray(
        "ens01",
        attrs={"long_name": "ensemble member", "type": "hindcast"},
    )

    da = xr.DataArray(
        values,
        coords={
            "time": time,
            "lat": lat,
            "month": month,
            "ensemble": ensemble,
        },
        dims=["time", "lat"],
        attrs={
            "units": "mm",
            "long_name": "Gridded Precipitation",
        },
        name="precip_grid",
    )

    # add rich attrs to dimension coords
    da.coords["time"].attrs = {
        "axis": "T",
        "standard_name": "time",
        "long_name": "time",
    }
    da.coords["lat"].attrs = {
        "axis": "Y",
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
    }

    return da


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


class TestCFMetadataRegistry:
    """Test CF metadata registry structure and SPI entry."""

    def test_spi_entry_exists(self):
        """CF_METADATA contains 'spi' key."""
        assert "spi" in CF_METADATA

    def test_spi_long_name(self):
        """SPI long_name is 'Standardized Precipitation Index'."""
        assert CF_METADATA["spi"]["long_name"] == "Standardized Precipitation Index"

    def test_spi_units(self):
        """SPI units are 'dimensionless'."""
        assert CF_METADATA["spi"]["units"] == "dimensionless"

    def test_spi_references_contains_mckee(self):
        """SPI references contain McKee et al. (1993) citation."""
        references = CF_METADATA["spi"]["references"]
        assert "McKee" in references
        assert "1993" in references

    def test_spi_has_no_standard_name(self):
        """SPI entry has no standard_name key (no official CF standard name)."""
        assert "standard_name" not in CF_METADATA["spi"]

    def test_all_entries_have_required_keys(self):
        """All entries have required keys: long_name, units, references."""
        required_keys = {"long_name", "units", "references"}
        for index_name, metadata in CF_METADATA.items():
            actual_keys = set(metadata.keys())
            # check that all required keys are present
            assert required_keys.issubset(actual_keys), (
                f"Entry '{index_name}' missing required keys: {required_keys - actual_keys}"
            )

    def test_all_values_are_non_empty_strings(self):
        """All metadata values are non-empty strings."""
        for index_name, metadata in CF_METADATA.items():
            for key, value in metadata.items():
                assert isinstance(value, str), f"Entry '{index_name}', key '{key}' is not a string"
                assert value.strip(), f"Entry '{index_name}', key '{key}' is empty or whitespace"


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
        """No CF metadata (None) preserves input attrs plus version."""

        @xarray_adapter(cf_metadata=None)
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        # input attrs should be preserved
        assert result.attrs["units"] == sample_monthly_precip_da.attrs["units"]
        assert result.attrs["long_name"] == sample_monthly_precip_da.attrs["long_name"]
        # version is always added
        assert "climate_indices_version" in result.attrs

    def test_registry_metadata_applied_to_output(self, sample_monthly_precip_da):
        """CF_METADATA registry values are correctly applied to output DataArray."""

        @xarray_adapter(cf_metadata=CF_METADATA["spi"])
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)
        assert result.attrs["long_name"] == "Standardized Precipitation Index"
        assert result.attrs["units"] == "dimensionless"
        assert "McKee" in result.attrs["references"]
        assert "standard_name" not in result.attrs


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
        # wrap the real SPI function with registry metadata
        wrapped_spi = xarray_adapter(cf_metadata=CF_METADATA["spi"])(indices.spi)

        # call with xarray DataArray - params will be inferred
        result = wrapped_spi(
            sample_monthly_precip_da,
            scale=3,
            distribution=indices.Distribution.gamma,
        )

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape
        assert result.attrs["long_name"] == "Standardized Precipitation Index"
        assert result.attrs["units"] == "dimensionless"
        assert "McKee" in result.attrs["references"]
        assert "standard_name" not in result.attrs

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


class TestBuildOutputDataarray:
    """Unit tests for _build_output_dataarray() coordinate preservation function."""

    def test_dimension_coords_preserved(self, coord_rich_1d_da):
        """All dimension coordinates present with identical values."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        assert "time" in output.coords
        np.testing.assert_array_equal(output.coords["time"].values, coord_rich_1d_da.coords["time"].values)

    def test_non_dimension_coords_preserved(self, coord_rich_1d_da):
        """Auxiliary coordinates (e.g., month) survive rewrap."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        assert "month" in output.coords
        np.testing.assert_array_equal(output.coords["month"].values, coord_rich_1d_da.coords["month"].values)

    def test_scalar_coords_preserved(self, coord_rich_1d_da):
        """Scalar coordinates (e.g., station_id) survive rewrap."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        assert "station_id" in output.coords
        assert output.coords["station_id"].values == coord_rich_1d_da.coords["station_id"].values

    def test_coord_attrs_preserved(self, coord_rich_1d_da):
        """Each coordinate's .attrs dict matches input."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        # check time coord attrs
        assert output.coords["time"].attrs["axis"] == "T"
        assert output.coords["time"].attrs["calendar"] == "standard"
        assert output.coords["time"].attrs["standard_name"] == "time"

        # check month coord attrs
        assert output.coords["month"].attrs["long_name"] == "month of year"

        # check station_id coord attrs
        assert output.coords["station_id"].attrs["long_name"] == "station identifier"
        assert output.coords["station_id"].attrs["cf_role"] == "timeseries_id"

    def test_da_attrs_preserved_without_cf(self, coord_rich_1d_da):
        """DA-level attrs match input when cf_metadata=None."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values, cf_metadata=None)

        assert output.attrs["units"] == "mm"
        assert output.attrs["long_name"] == "Monthly Precipitation"

    def test_cf_metadata_overrides_da_attrs(self, coord_rich_1d_da):
        """CF metadata updates DA attrs, not coord attrs."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        cf_metadata = {
            "long_name": "Standardized Precipitation Index",
            "units": "dimensionless",
        }
        output = _build_output_dataarray(coord_rich_1d_da, result_values, cf_metadata)

        # DA attrs should be overridden
        assert output.attrs["long_name"] == "Standardized Precipitation Index"
        assert output.attrs["units"] == "dimensionless"

    def test_cf_metadata_does_not_affect_coord_attrs(self, coord_rich_1d_da):
        """Coord-level attrs unchanged by CF overlay."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        cf_metadata = {
            "long_name": "Standardized Precipitation Index",
            "standard_name": "spi",
        }
        output = _build_output_dataarray(coord_rich_1d_da, result_values, cf_metadata)

        # coord attrs should remain unchanged
        assert output.coords["time"].attrs["standard_name"] == "time"
        assert output.coords["time"].attrs["long_name"] == "time"

    def test_coord_order_preserved(self, coord_rich_1d_da):
        """list(output.coords) == list(input.coords)."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        assert list(output.coords.keys()) == list(coord_rich_1d_da.coords.keys())

    def test_dim_order_preserved(self, coord_rich_1d_da):
        """output.dims == input.dims."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        assert output.dims == coord_rich_1d_da.dims

    def test_result_values_correct(self, coord_rich_1d_da):
        """output.values matches the raw result array."""
        result_values = np.arange(len(coord_rich_1d_da.values))
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        np.testing.assert_array_equal(output.values, result_values)

    def test_name_preserved(self, coord_rich_1d_da):
        """output.name == input.name."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        assert output.name == coord_rich_1d_da.name

    def test_coord_attrs_deep_copied(self, coord_rich_1d_da):
        """Mutating input coord attrs after call does not affect output."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        # mutate input coord attrs after building output
        coord_rich_1d_da.coords["time"].attrs["axis"] = "X"  # wrong!
        coord_rich_1d_da.coords["month"].attrs["mutated"] = "yes"

        # output should be unaffected
        assert output.coords["time"].attrs["axis"] == "T"
        assert "mutated" not in output.coords["month"].attrs

    def test_multi_dim_coords_preserved(self, multi_coord_da):
        """Test with 2D array - both time and lat coords preserved."""
        # only test first time slice to match 1D result shape
        input_1d = multi_coord_da.isel(lat=0)
        result_values = np.ones_like(input_1d.values)
        output = _build_output_dataarray(input_1d, result_values)

        assert "time" in output.coords
        assert output.coords["time"].attrs["axis"] == "T"

    def test_history_added_when_index_name_provided(self, coord_rich_1d_da):
        """History attribute should be added when index_name is provided."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(
            coord_rich_1d_da,
            result_values,
            calculation_metadata={"scale": 3, "distribution": indices.Distribution.gamma},
            index_name="SPI",
        )

        assert "history" in output.attrs
        assert "SPI-3 calculated using gamma distribution" in output.attrs["history"]
        assert "climate_indices v" in output.attrs["history"]

    def test_no_history_when_index_name_none(self, coord_rich_1d_da):
        """History attribute should not be added when index_name is None."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(
            coord_rich_1d_da,
            result_values,
            calculation_metadata={"scale": 3},
            index_name=None,
        )

        # history should not be present (backward compat for direct callers)
        assert "history" not in output.attrs


class TestCoordinatePreservationRoundTrip:
    """Integration tests for coordinate preservation through decorator pipeline."""

    def test_1d_rich_coord_attrs_round_trip(self, coord_rich_1d_da):
        """Time coord attrs survive full decorator pipeline."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(coord_rich_1d_da)

        # check time coord attrs survived
        assert result.coords["time"].attrs["axis"] == "T"
        assert result.coords["time"].attrs["calendar"] == "standard"
        assert result.coords["time"].attrs["standard_name"] == "time"
        assert result.coords["time"].attrs["bounds"] == "time_bounds"

    def test_non_dimension_coords_round_trip(self, coord_rich_1d_da):
        """Auxiliary coords survive decorator pipeline."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(coord_rich_1d_da)

        assert "month" in result.coords
        assert result.coords["month"].attrs["long_name"] == "month of year"
        np.testing.assert_array_equal(result.coords["month"].values, coord_rich_1d_da.coords["month"].values)

    def test_scalar_coords_round_trip(self, coord_rich_1d_da):
        """Scalar coords survive decorator pipeline."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(coord_rich_1d_da)

        assert "station_id" in result.coords
        assert result.coords["station_id"].values == "GHCN-12345"
        assert result.coords["station_id"].attrs["cf_role"] == "timeseries_id"

    def test_no_extra_coords_added(self, coord_rich_1d_da):
        """Output has exactly the same coord keys as input."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(coord_rich_1d_da)

        assert set(result.coords.keys()) == set(coord_rich_1d_da.coords.keys())

    def test_coord_values_unchanged(self, coord_rich_1d_da):
        """Coord values (not just keys) identical after round-trip."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(coord_rich_1d_da)

        for coord_name in coord_rich_1d_da.coords:
            np.testing.assert_array_equal(
                result.coords[coord_name].values,
                coord_rich_1d_da.coords[coord_name].values,
            )

    def test_coord_dtype_preserved(self, multi_coord_da):
        """Coord dtypes preserved (float64 lat stays float64)."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        # only test first lat slice for 1D
        input_1d = multi_coord_da.isel(lat=0)
        result = identity(input_1d)

        assert result.coords["time"].dtype == input_1d.coords["time"].dtype

    def test_empty_coord_attrs_preserved(self, sample_monthly_precip_da):
        """Coords with empty attrs {} don't crash."""
        # add a coord with no attrs
        sample_monthly_precip_da.coords["year"] = xr.DataArray(
            [pd.Timestamp(t).year for t in sample_monthly_precip_da.coords["time"].values],
            dims=["time"],
        )
        # explicitly set empty attrs
        sample_monthly_precip_da.coords["year"].attrs = {}

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        assert "year" in result.coords
        assert result.coords["year"].attrs == {}


class TestSerializeAttrValue:
    """Test _serialize_attr_value() helper function."""

    def test_enum_to_name_string(self):
        """Enum instances serialize to their .name string."""
        result = _serialize_attr_value(indices.Distribution.gamma)
        assert result == "gamma"
        assert isinstance(result, str)

    def test_enum_periodicity(self):
        """Periodicity enum serializes correctly."""
        result = _serialize_attr_value(compute.Periodicity.monthly)
        assert result == "monthly"
        assert isinstance(result, str)

    def test_string_passthrough(self):
        """String values pass through unchanged."""
        result = _serialize_attr_value("test_string")
        assert result == "test_string"
        assert isinstance(result, str)

    def test_int_passthrough(self):
        """Integer values pass through unchanged."""
        result = _serialize_attr_value(42)
        assert result == 42
        assert isinstance(result, int)

    def test_float_passthrough(self):
        """Float values pass through unchanged."""
        result = _serialize_attr_value(3.14)
        assert result == 3.14
        assert isinstance(result, float)

    def test_bool_passthrough(self):
        """Boolean values pass through unchanged."""
        result = _serialize_attr_value(True)
        assert result is True
        assert isinstance(result, bool)

    def test_numpy_int_to_python_int(self):
        """NumPy integer scalars convert to Python int."""
        result = _serialize_attr_value(np.int64(42))
        assert result == 42
        assert isinstance(result, int)
        assert not isinstance(result, np.integer)

    def test_numpy_float_to_python_float(self):
        """NumPy float scalars convert to Python float."""
        result = _serialize_attr_value(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)
        assert not isinstance(result, np.floating)

    def test_dict_raises_typeerror(self):
        """Dict values raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            _serialize_attr_value({"key": "value"})
        assert "Cannot serialize" in str(exc_info.value)

    def test_list_raises_typeerror(self):
        """List values raise TypeError."""
        with pytest.raises(TypeError) as exc_info:
            _serialize_attr_value([1, 2, 3])
        assert "Cannot serialize" in str(exc_info.value)


class TestCalculationMetadata:
    """Test calculation metadata capture and serialization."""

    def test_scale_captured(self, sample_monthly_precip_da):
        """scale parameter captured in output attrs."""

        @xarray_adapter(calculation_metadata_keys=["scale"])
        def needs_scale(values: np.ndarray, scale: int) -> np.ndarray:
            return values

        result = needs_scale(sample_monthly_precip_da, scale=3)

        assert result.attrs["scale"] == 3

    def test_enum_serialized(self, sample_monthly_precip_da):
        """Enum parameter serialized to .name string."""

        @xarray_adapter(calculation_metadata_keys=["distribution"])
        def needs_distribution(values: np.ndarray, distribution: indices.Distribution) -> np.ndarray:
            return values

        result = needs_distribution(sample_monthly_precip_da, distribution=indices.Distribution.gamma)

        assert result.attrs["distribution"] == "gamma"
        assert isinstance(result.attrs["distribution"], str)

    def test_multiple_keys(self, sample_monthly_precip_da):
        """Multiple calculation metadata keys captured."""

        @xarray_adapter(calculation_metadata_keys=["scale", "distribution"])
        def needs_both(values: np.ndarray, scale: int, distribution: indices.Distribution) -> np.ndarray:
            return values

        result = needs_both(sample_monthly_precip_da, scale=6, distribution=indices.Distribution.gamma)

        assert result.attrs["scale"] == 6
        assert result.attrs["distribution"] == "gamma"

    def test_missing_key_skipped(self, sample_monthly_precip_da):
        """Missing keys not added to attrs."""

        @xarray_adapter(calculation_metadata_keys=["scale", "nonexistent"])
        def optional_params(values: np.ndarray, scale: int) -> np.ndarray:
            return values

        result = optional_params(sample_monthly_precip_da, scale=3)

        assert result.attrs["scale"] == 3
        assert "nonexistent" not in result.attrs

    def test_none_keys_no_calc_attrs(self, sample_monthly_precip_da):
        """calculation_metadata_keys=None adds no calculation attrs."""

        @xarray_adapter(calculation_metadata_keys=None)
        def simple_func(values: np.ndarray, scale: int) -> np.ndarray:
            return values

        result = simple_func(sample_monthly_precip_da, scale=3)

        # scale not in attrs (no capture requested)
        assert "scale" not in result.attrs
        # but version is always present
        assert "climate_indices_version" in result.attrs

    def test_numpy_passthrough_unaffected(self):
        """NumPy inputs bypass calculation metadata capture."""

        @xarray_adapter(calculation_metadata_keys=["scale"])
        def needs_scale(values: np.ndarray, scale: int) -> np.ndarray:
            return values * scale

        numpy_input = np.array([1.0, 2.0, 3.0])
        result = needs_scale(numpy_input, scale=2)

        # numpy result has no attrs
        assert isinstance(result, np.ndarray)
        assert not hasattr(result, "attrs")


class TestLibraryVersionAttribute:
    """Test climate_indices_version attribute."""

    def test_version_always_present(self, sample_monthly_precip_da):
        """climate_indices_version always added to output."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        assert "climate_indices_version" in result.attrs

    def test_version_matches_package_version(self, sample_monthly_precip_da):
        """Version matches climate_indices.__version__."""
        from climate_indices import __version__

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        assert result.attrs["climate_indices_version"] == __version__

    def test_version_coexists_with_cf_metadata(self, sample_monthly_precip_da):
        """Version added alongside CF metadata."""

        @xarray_adapter(cf_metadata={"long_name": "Test Index", "units": "dimensionless"})
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        result = identity(sample_monthly_precip_da)

        assert result.attrs["long_name"] == "Test Index"
        assert result.attrs["units"] == "dimensionless"
        assert "climate_indices_version" in result.attrs

    def test_version_in_build_output_directly(self, coord_rich_1d_da):
        """_build_output_dataarray adds version even without decorator."""
        result_values = np.ones_like(coord_rich_1d_da.values)
        output = _build_output_dataarray(coord_rich_1d_da, result_values)

        assert "climate_indices_version" in output.attrs


class TestAttributeLayering:
    """Test attribute layering: input → CF → calc → version."""

    def test_full_layering(self, sample_monthly_precip_da):
        """All attribute sources layer correctly: input → CF → calc → version → history."""
        cf_meta = {"long_name": "Standardized Precipitation Index", "units": "dimensionless"}

        @xarray_adapter(cf_metadata=cf_meta, calculation_metadata_keys=["scale", "distribution"])
        def full_function(values: np.ndarray, scale: int, distribution: indices.Distribution) -> np.ndarray:
            return values

        result = full_function(sample_monthly_precip_da, scale=3, distribution=indices.Distribution.gamma)

        # input attrs preserved (but long_name/units overridden)
        # CF metadata present
        assert result.attrs["long_name"] == "Standardized Precipitation Index"
        assert result.attrs["units"] == "dimensionless"
        # calculation metadata present
        assert result.attrs["scale"] == 3
        assert result.attrs["distribution"] == "gamma"
        # version always present
        assert "climate_indices_version" in result.attrs
        # history present with expected content
        assert "history" in result.attrs
        assert "FULL_FUNCTION" in result.attrs["history"]
        assert "gamma distribution" in result.attrs["history"]

    def test_cf_and_calc_keys_dont_collide(self, sample_monthly_precip_da):
        """CF and calculation metadata use different keys."""
        cf_meta = {"long_name": "Test Index", "units": "dimensionless"}

        @xarray_adapter(cf_metadata=cf_meta, calculation_metadata_keys=["scale"])
        def no_collision(values: np.ndarray, scale: int) -> np.ndarray:
            return values

        result = no_collision(sample_monthly_precip_da, scale=6)

        # both sets of metadata present without collision
        assert result.attrs["long_name"] == "Test Index"
        assert result.attrs["units"] == "dimensionless"
        assert result.attrs["scale"] == 6


class TestEndToEndIntegration:
    """End-to-end integration test with real SPI and full metadata capture."""

    def test_spi_with_calculation_metadata_keys(self, sample_monthly_precip_da):
        """SPI with calculation_metadata_keys captures scale, distribution, and history."""
        # wrap SPI with calculation metadata capture
        wrapped_spi = xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale", "distribution"],
        )(indices.spi)

        result = wrapped_spi(
            sample_monthly_precip_da,
            scale=3,
            distribution=indices.Distribution.gamma,
        )

        # CF metadata present
        assert result.attrs["long_name"] == "Standardized Precipitation Index"
        assert result.attrs["units"] == "dimensionless"
        assert "McKee" in result.attrs["references"]

        # calculation metadata captured
        assert result.attrs["scale"] == 3
        assert result.attrs["distribution"] == "gamma"

        # version present
        assert "climate_indices_version" in result.attrs

        # history present with SPI and gamma distribution
        assert "history" in result.attrs
        assert "SPI" in result.attrs["history"]
        assert "gamma" in result.attrs["history"]

        # original input attrs preserved (not overridden by CF)
        # (input had "units" and "long_name" but they're overridden by CF)

        # result values are correct (not all NaN)
        assert not np.all(np.isnan(result.values))
        assert result.shape == sample_monthly_precip_da.shape


class TestBuildHistoryEntry:
    """Test history entry generation for provenance tracking."""

    def test_entry_with_scale_and_distribution(self) -> None:
        """History entry should include scale and distribution when both are provided."""
        entry = _build_history_entry(
            index_name="SPI",
            version="2.0.0",
            calculation_metadata={"scale": 3, "distribution": indices.Distribution.gamma},
        )

        assert "SPI-3 calculated using gamma distribution" in entry
        assert "(climate_indices v2.0.0)" in entry

    def test_entry_with_scale_only(self) -> None:
        """History entry should include scale but not distribution when only scale provided."""
        entry = _build_history_entry(
            index_name="SPI",
            version="2.0.0",
            calculation_metadata={"scale": 6},
        )

        assert "SPI-6 calculated" in entry
        assert "distribution" not in entry
        assert "(climate_indices v2.0.0)" in entry

    def test_entry_no_params(self) -> None:
        """History entry should have basic format when no calculation metadata provided."""
        entry = _build_history_entry(
            index_name="SPI",
            version="2.0.0",
            calculation_metadata=None,
        )

        assert "SPI calculated" in entry
        assert "(climate_indices v2.0.0)" in entry

    def test_entry_empty_metadata(self) -> None:
        """History entry should handle empty metadata dict same as None."""
        entry = _build_history_entry(
            index_name="SPI",
            version="2.0.0",
            calculation_metadata={},
        )

        assert "SPI calculated" in entry
        assert "(climate_indices v2.0.0)" in entry

    def test_timestamp_is_iso8601_utc(self) -> None:
        """History entry timestamp should match ISO 8601 UTC format."""
        entry = _build_history_entry(
            index_name="SPI",
            version="2.0.0",
        )

        # check for ISO 8601 timestamp pattern: YYYY-MM-DDTHH:MM:SSZ
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
        assert re.search(pattern, entry) is not None

    def test_entry_contains_version(self) -> None:
        """History entry should contain the library version string."""
        test_version = "1.2.3"
        entry = _build_history_entry(
            index_name="SPI",
            version=test_version,
        )

        assert f"climate_indices v{test_version}" in entry

    def test_enum_distribution_serialized(self) -> None:
        """Enum distribution values should be serialized to their .name attribute."""
        entry = _build_history_entry(
            index_name="SPI",
            version="2.0.0",
            calculation_metadata={"scale": 3, "distribution": indices.Distribution.pearson},
        )

        # should use enum .name, not the enum object
        assert "pearson distribution" in entry
        assert "Distribution.pearson" not in entry


class TestAppendHistory:
    """Test history attribute appending logic."""

    def test_no_existing_history(self) -> None:
        """When no existing history, should return just the new entry."""
        result = _append_history(
            existing_attrs={},
            new_entry="2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)",
        )

        assert result == "2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)"

    def test_append_to_existing(self) -> None:
        """Should append new entry to existing history with newline separator."""
        result = _append_history(
            existing_attrs={"history": "2026-02-06T09:00:00Z: Data prepared"},
            new_entry="2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)",
        )

        expected = "2026-02-06T09:00:00Z: Data prepared\n2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)"
        assert result == expected

    def test_append_to_whitespace_or_empty(self) -> None:
        """Empty or whitespace-only existing history should be treated as no history."""
        # empty string
        result = _append_history(
            existing_attrs={"history": ""},
            new_entry="2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)",
        )
        assert result == "2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)"

        # whitespace only
        result = _append_history(
            existing_attrs={"history": "   "},
            new_entry="2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)",
        )
        assert result == "2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)"

    def test_preserves_multi_line_existing(self) -> None:
        """Should preserve all existing history entries when appending."""
        existing_history = "2026-02-05T08:00:00Z: Data ingested\n2026-02-06T09:00:00Z: Quality control applied"

        result = _append_history(
            existing_attrs={"history": existing_history},
            new_entry="2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)",
        )

        expected = (
            "2026-02-05T08:00:00Z: Data ingested\n"
            "2026-02-06T09:00:00Z: Quality control applied\n"
            "2026-02-07T10:00:00Z: SPI calculated (climate_indices v2.0.0)"
        )
        assert result == expected


class TestHistoryProvenance:
    """Test provenance tracking in history attribute through the decorator."""

    def test_history_present(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """History attribute should be present in xarray output."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale"],
        )
        def mock_index(values: np.ndarray, scale: int, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        result = mock_index(sample_monthly_precip_da, scale=3)

        assert "history" in result.attrs

    def test_history_contains_timestamp(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """History should contain ISO 8601 UTC timestamp."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
        )
        def mock_index(values: np.ndarray, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        result = mock_index(sample_monthly_precip_da)

        # check for ISO 8601 pattern
        pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"
        assert re.search(pattern, result.attrs["history"]) is not None

    def test_history_contains_index_and_scale(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """History should include index name and scale when provided."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale"],
            index_display_name="SPI",
        )
        def mock_index(values: np.ndarray, scale: int, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        result = mock_index(sample_monthly_precip_da, scale=3)

        assert "SPI-3" in result.attrs["history"]

    def test_history_contains_distribution(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """History should include distribution when provided."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale", "distribution"],
            index_display_name="SPI",
        )
        def mock_index(values: np.ndarray, scale: int, distribution: indices.Distribution, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        result = mock_index(sample_monthly_precip_da, scale=3, distribution=indices.Distribution.gamma)

        assert "gamma distribution" in result.attrs["history"]

    def test_history_contains_library_version(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """History should include climate_indices version string."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
        )
        def mock_index(values: np.ndarray, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        result = mock_index(sample_monthly_precip_da)

        assert "climate_indices v" in result.attrs["history"]

    def test_default_index_name_from_function(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """When index_display_name not provided, should use uppercase function name."""

        @xarray_adapter(cf_metadata=CF_METADATA["spi"])
        def my_index(values: np.ndarray, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        result = my_index(sample_monthly_precip_da)

        assert "MY_INDEX calculated" in result.attrs["history"]

    def test_preserves_existing_history(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """Should preserve and append to existing history attribute."""
        # add existing history to input
        sample_monthly_precip_da.attrs["history"] = "2026-02-06T09:00:00Z: Data prepared"

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
            calculation_metadata_keys=["scale"],
        )
        def mock_index(values: np.ndarray, scale: int, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        result = mock_index(sample_monthly_precip_da, scale=3)

        # both old and new entries should be present
        assert "2026-02-06T09:00:00Z: Data prepared" in result.attrs["history"]
        assert "SPI-3 calculated" in result.attrs["history"]
        # should be newline-separated
        assert "\n" in result.attrs["history"]

    def test_not_present_on_numpy_passthrough(self) -> None:
        """History should not be present when input is NumPy array (passthrough path)."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
        )
        def mock_index(values: np.ndarray, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        # use numpy array input (passthrough path)
        result = mock_index(np.array([1.0, 2.0, 3.0]))

        # result is ndarray, no attrs
        assert isinstance(result, np.ndarray)
        assert not hasattr(result, "attrs")

    def test_history_with_no_calculation_metadata(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """History should be present even when no calculation metadata is captured."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
            # no calculation_metadata_keys
        )
        def mock_index(values: np.ndarray, **kwargs: int) -> np.ndarray:
            return np.random.randn(len(values))

        result = mock_index(sample_monthly_precip_da)

        assert "history" in result.attrs
        assert "SPI calculated" in result.attrs["history"]
        # should not have scale or distribution in history
        assert "SPI-" not in result.attrs["history"]
        assert "distribution" not in result.attrs["history"]
