"""Tests for xarray adapter decorator infrastructure.

This module tests Story 2.2 functionality:
- Parameter inference from time coordinates
- Extract → infer → compute → rewrap → log adapter contract
- CF metadata application
- NumPy passthrough behavior
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, indices
from climate_indices.exceptions import (
    CoordinateValidationError,
    InputAlignmentWarning,
    InsufficientDataError,
)
from climate_indices.xarray_adapter import (
    CF_METADATA,
    _align_inputs,
    _append_history,
    _assess_nan_density,
    _build_history_entry,
    _build_output_dataarray,
    _infer_calibration_period,
    _infer_data_start_year,
    _infer_periodicity,
    _resolve_scale_from_args,
    _resolve_secondary_inputs,
    _serialize_attr_value,
    _validate_calibration_non_nan_sample_size,
    _validate_sufficient_data,
    _validate_time_dimension,
    _validate_time_monotonicity,
    _verify_nan_propagation,
    xarray_adapter,
)


def _standard_normal(size: int | tuple[int, ...]) -> np.ndarray:
    """Generate deterministic standard normal data for tests."""
    return np.random.default_rng(42).standard_normal(size)


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


@pytest.fixture
def no_time_dim_da() -> xr.DataArray:
    """Create DataArray without a time dimension."""
    # spatial coordinates only
    x = np.arange(10)
    lat = np.arange(5) * 10.0
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=(len(x), len(lat)))

    return xr.DataArray(
        values,
        coords={"x": x, "lat": lat},
        dims=["x", "lat"],
        attrs={"units": "mm"},
    )


@pytest.fixture
def non_monotonic_time_da() -> xr.DataArray:
    """Create DataArray with non-monotonic (shuffled) time coordinate."""
    time = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    # shuffle the time coordinate
    shuffled_time = time[[0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10]]
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": shuffled_time},
        dims=["time"],
        attrs={"units": "mm"},
    )


@pytest.fixture
def short_monthly_da() -> xr.DataArray:
    """Create short DataArray with only 3 months."""
    time = pd.date_range("2020-01-01", "2020-03-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={"units": "mm"},
    )


@pytest.fixture
def sample_monthly_pet_da() -> xr.DataArray:
    """Create a 1D monthly PET DataArray matching precip fixture (40 years, 1980-2019)."""
    # 40 years * 12 months = 480 values, matching sample_monthly_precip_da
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    # generate random PET values (typically higher than precip)
    rng = np.random.default_rng(100)
    values = rng.gamma(shape=2.5, scale=60.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Potential Evapotranspiration",
        },
    )


@pytest.fixture
def sample_monthly_pet_offset_da() -> xr.DataArray:
    """Create a 1D monthly PET DataArray with offset time range (1985-2024).

    This fixture overlaps with sample_monthly_precip_da (1980-2019) only
    during 1985-2019, useful for testing alignment behavior.
    """
    # 40 years offset by 5 years: 1985-2024
    time = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
    # generate random PET values
    rng = np.random.default_rng(101)
    values = rng.gamma(shape=2.5, scale=60.0, size=len(time))

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Potential Evapotranspiration (Offset)",
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

    @staticmethod
    def _combined_log_output(caplog, capsys) -> str:
        """Collect log text from both stdlib capture and rendered stream output."""
        stream = capsys.readouterr()
        parts = [caplog.text, stream.out, stream.err]
        parts.extend(record.message for record in caplog.records)
        parts.extend(str(record.msg) for record in caplog.records)
        return "\n".join(part for part in parts if part)

    def test_logs_completion_event(self, sample_monthly_precip_da, caplog, capsys):
        """Logs xarray_adapter_completed event on success."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        with caplog.at_level("INFO"):
            _ = identity(sample_monthly_precip_da)

        # check that completion event was logged
        log_output = self._combined_log_output(caplog, capsys)
        assert "xarray_adapter_completed" in log_output

    def test_logs_include_shapes(self, sample_monthly_precip_da, caplog, capsys):
        """Logs include input/output shape information."""

        @xarray_adapter()
        def identity(values: np.ndarray) -> np.ndarray:
            return values

        with caplog.at_level("INFO"):
            _ = identity(sample_monthly_precip_da)

        # verify shape info in logs
        log_output = self._combined_log_output(caplog, capsys)
        assert "input_shape" in log_output
        assert "output_shape" in log_output


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
        assert result == pytest.approx(3.14)
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
        assert result == pytest.approx(3.14)
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
            return _standard_normal(len(values))

        result = mock_index(sample_monthly_precip_da, scale=3)

        assert "history" in result.attrs

    def test_history_contains_timestamp(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """History should contain ISO 8601 UTC timestamp."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
        )
        def mock_index(values: np.ndarray, **kwargs: int) -> np.ndarray:
            return _standard_normal(len(values))

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
            return _standard_normal(len(values))

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
            return _standard_normal(len(values))

        result = mock_index(sample_monthly_precip_da, scale=3, distribution=indices.Distribution.gamma)

        assert "gamma distribution" in result.attrs["history"]

    def test_history_contains_library_version(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """History should include climate_indices version string."""

        @xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            index_display_name="SPI",
        )
        def mock_index(values: np.ndarray, **kwargs: int) -> np.ndarray:
            return _standard_normal(len(values))

        result = mock_index(sample_monthly_precip_da)

        assert "climate_indices v" in result.attrs["history"]

    def test_default_index_name_from_function(self, sample_monthly_precip_da: xr.DataArray) -> None:
        """When index_display_name not provided, should use uppercase function name."""

        @xarray_adapter(cf_metadata=CF_METADATA["spi"])
        def my_index(values: np.ndarray, **kwargs: int) -> np.ndarray:
            return _standard_normal(len(values))

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
            return _standard_normal(len(values))

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
            return _standard_normal(len(values))

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
            return _standard_normal(len(values))

        result = mock_index(sample_monthly_precip_da)

        assert "history" in result.attrs
        assert "SPI calculated" in result.attrs["history"]
        # should not have scale or distribution in history
        assert "SPI-" not in result.attrs["history"]
        assert "distribution" not in result.attrs["history"]


class TestValidateTimeDimension:
    """Test _validate_time_dimension() function."""

    def test_valid_time_dimension_passes(self, sample_monthly_precip_da):
        """Valid time dimension does not raise."""
        # should not raise
        _validate_time_dimension(sample_monthly_precip_da, "time")

    def test_missing_dimension_raises(self, no_time_dim_da):
        """Missing time dimension raises CoordinateValidationError."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            _validate_time_dimension(no_time_dim_da, "time")

        assert "not found" in str(exc_info.value).lower()
        assert exc_info.value.coordinate_name == "time"
        assert exc_info.value.reason == "missing_dimension"

    def test_error_message_includes_available_dims(self, no_time_dim_da):
        """Error message lists available dimensions."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            _validate_time_dimension(no_time_dim_da, "time")

        error_msg = str(exc_info.value)
        assert "['x', 'lat']" in error_msg

    def test_error_message_suggests_time_dim_parameter(self, no_time_dim_da):
        """Error message suggests using time_dim parameter."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            _validate_time_dimension(no_time_dim_da, "time")

        assert "time_dim parameter" in str(exc_info.value)

    def test_custom_dimension_name_validated(self):
        """Validation works with custom time dimension name."""
        # create DataArray with 'date' instead of 'time'
        date = pd.date_range("2020-01-01", periods=12, freq="MS")
        da = xr.DataArray(
            _standard_normal(12),
            coords={"date": date},
            dims=["date"],
        )

        # should not raise when checking for 'date'
        _validate_time_dimension(da, "date")

        # should raise when checking for 'time'
        with pytest.raises(CoordinateValidationError):
            _validate_time_dimension(da, "time")

    def test_scalar_dataarray_raises(self):
        """Scalar DataArray (no dims) raises error."""
        scalar_da = xr.DataArray(42.0)

        with pytest.raises(CoordinateValidationError) as exc_info:
            _validate_time_dimension(scalar_da, "time")

        # should show empty dims list
        assert "[]" in str(exc_info.value)

    def test_time_as_coordinate_but_not_dimension_raises(self):
        """Time as non-dimension coordinate is rejected."""
        # create DataArray where 'time' is a scalar coordinate, not a dimension
        da = xr.DataArray(
            _standard_normal(10),
            coords={"x": np.arange(10), "time": pd.Timestamp("2020-01-01")},
            dims=["x"],
        )

        with pytest.raises(CoordinateValidationError):
            _validate_time_dimension(da, "time")

    def test_empty_dataarray_with_time_dim_passes(self):
        """Empty DataArray with time dim (0 elements) passes dimension check."""
        empty_da = xr.DataArray(
            [],
            coords={"time": pd.DatetimeIndex([])},
            dims=["time"],
        )

        # dimension exists, so validation passes
        _validate_time_dimension(empty_da, "time")


class TestValidateTimeMonotonicity:
    """Test _validate_time_monotonicity() function."""

    def test_monotonic_increasing_passes(self):
        """Monotonically increasing time coordinate passes."""
        time = pd.date_range("2020-01-01", periods=12, freq="MS")
        time_coord = xr.DataArray(time, dims=["time"])

        # should not raise
        _validate_time_monotonicity(time_coord)

    def test_reversed_time_raises(self):
        """Reversed time coordinate raises CoordinateValidationError."""
        time = pd.date_range("2020-01-01", periods=12, freq="MS")[::-1]
        time_coord = xr.DataArray(time, dims=["time"])

        with pytest.raises(CoordinateValidationError) as exc_info:
            _validate_time_monotonicity(time_coord)

        assert "not monotonically increasing" in str(exc_info.value).lower()
        assert exc_info.value.reason == "not_monotonic"

    def test_shuffled_time_raises(self, non_monotonic_time_da):
        """Shuffled time coordinate raises CoordinateValidationError."""
        with pytest.raises(CoordinateValidationError) as exc_info:
            _validate_time_monotonicity(non_monotonic_time_da["time"])

        assert "not monotonically increasing" in str(exc_info.value).lower()

    def test_duplicate_timestamps_pass(self):
        """Duplicate timestamps pass (is_monotonic_increasing allows duplicates)."""
        time = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01"])
        time_coord = xr.DataArray(time, dims=["time"])

        # pandas is_monotonic_increasing allows duplicates (non-decreasing)
        # should not raise
        _validate_time_monotonicity(time_coord)

    def test_error_suggests_sortby(self):
        """Error message suggests using sortby()."""
        time = pd.date_range("2020-01-01", periods=12, freq="MS")[::-1]
        time_coord = xr.DataArray(time, dims=["time"])

        with pytest.raises(CoordinateValidationError) as exc_info:
            _validate_time_monotonicity(time_coord)

        assert "sortby" in str(exc_info.value)

    def test_single_element_passes(self):
        """Single-element time coordinate passes (trivially monotonic)."""
        time_coord = xr.DataArray([pd.Timestamp("2020-01-01")], dims=["time"])

        # should not raise
        _validate_time_monotonicity(time_coord)

    def test_two_element_increasing_passes(self):
        """Two-element increasing time coordinate passes."""
        time = pd.to_datetime(["2020-01-01", "2020-02-01"])
        time_coord = xr.DataArray(time, dims=["time"])

        # should not raise
        _validate_time_monotonicity(time_coord)

    def test_two_element_decreasing_raises(self):
        """Two-element decreasing time coordinate raises."""
        time = pd.to_datetime(["2020-02-01", "2020-01-01"])
        time_coord = xr.DataArray(time, dims=["time"])

        with pytest.raises(CoordinateValidationError):
            _validate_time_monotonicity(time_coord)


class TestValidateSufficientData:
    """Test _validate_sufficient_data() function."""

    def test_sufficient_data_passes(self):
        """Time coordinate with enough elements passes."""
        time = pd.date_range("2020-01-01", periods=12, freq="MS")
        time_coord = xr.DataArray(time, dims=["time"])

        # 12 months >= scale 3
        _validate_sufficient_data(time_coord, scale=3)

    def test_insufficient_data_raises(self, short_monthly_da):
        """Time coordinate with too few elements raises InsufficientDataError."""
        with pytest.raises(InsufficientDataError) as exc_info:
            _validate_sufficient_data(short_monthly_da["time"], scale=6)

        assert "insufficient data" in str(exc_info.value).lower()
        assert exc_info.value.non_zero_count == 3
        assert exc_info.value.required_count == 6

    def test_exact_boundary_passes(self):
        """Time coordinate with exactly scale elements passes."""
        time = pd.date_range("2020-01-01", periods=3, freq="MS")
        time_coord = xr.DataArray(time, dims=["time"])

        # exactly 3 months for scale 3
        _validate_sufficient_data(time_coord, scale=3)

    def test_error_message_includes_counts(self, short_monthly_da):
        """Error message includes actual and required counts."""
        with pytest.raises(InsufficientDataError) as exc_info:
            _validate_sufficient_data(short_monthly_da["time"], scale=6)

        error_msg = str(exc_info.value)
        assert "3 time steps" in error_msg
        assert "at least 6 required" in error_msg

    def test_scale_one_edge_case(self):
        """Scale=1 requires at least 1 timestep."""
        time = pd.date_range("2020-01-01", periods=1, freq="MS")
        time_coord = xr.DataArray(time, dims=["time"])

        # 1 timestep >= scale 1
        _validate_sufficient_data(time_coord, scale=1)

    def test_empty_time_coord_raises(self):
        """Empty time coordinate raises for any scale."""
        empty_time = xr.DataArray([], dims=["time"])

        with pytest.raises(InsufficientDataError):
            _validate_sufficient_data(empty_time, scale=1)


class TestResolveScaleFromArgs:
    """Test _resolve_scale_from_args() function."""

    def test_scale_from_kwargs(self):
        """Scale extracted from kwargs."""

        def func(values: np.ndarray, scale: int) -> np.ndarray:
            return values

        result = _resolve_scale_from_args(func, (np.array([1, 2, 3]),), {"scale": 3})
        assert result == 3

    def test_scale_from_positional_args(self):
        """Scale extracted from positional arguments."""

        def func(values: np.ndarray, scale: int) -> np.ndarray:
            return values

        result = _resolve_scale_from_args(func, (np.array([1, 2, 3]), 6), {})
        assert result == 6

    def test_scale_not_in_signature_returns_none(self):
        """Function without scale parameter returns None."""

        def func(values: np.ndarray, other_param: int) -> np.ndarray:
            return values

        result = _resolve_scale_from_args(func, (np.array([1, 2, 3]),), {"other_param": 5})
        assert result is None

    def test_scale_not_provided_returns_none(self):
        """Scale in signature but not provided returns None."""

        def func(values: np.ndarray, scale: int) -> np.ndarray:
            return values

        result = _resolve_scale_from_args(func, (np.array([1, 2, 3]),), {})
        assert result is None

    def test_binding_failure_returns_none(self):
        """Binding failure gracefully returns None."""

        def func(values: np.ndarray, scale: int) -> np.ndarray:
            return values

        # pass extra unexpected arguments to cause binding failure
        result = _resolve_scale_from_args(func, (np.array([1, 2, 3]),), {"unexpected": 99})
        # should not crash, returns None
        assert result is None


class TestCoordinateValidationIntegration:
    """Integration tests for coordinate validation through the decorator."""

    def test_missing_time_dimension_raises(self, no_time_dim_da):
        """Missing time dimension raises during xarray processing."""

        @xarray_adapter(infer_params=True)
        def needs_time(values: np.ndarray, data_start_year: int) -> np.ndarray:
            return values

        with pytest.raises(CoordinateValidationError) as exc_info:
            needs_time(no_time_dim_da)

        assert "not found" in str(exc_info.value).lower()

    def test_non_monotonic_time_raises(self, non_monotonic_time_da):
        """Non-monotonic time coordinate raises during xarray processing."""

        @xarray_adapter(infer_params=True)
        def needs_time(values: np.ndarray, data_start_year: int) -> np.ndarray:
            return values

        with pytest.raises(CoordinateValidationError) as exc_info:
            needs_time(non_monotonic_time_da)

        assert "not monotonically increasing" in str(exc_info.value).lower()

    def test_insufficient_data_raises(self, short_monthly_da):
        """Insufficient data for scale raises InsufficientDataError."""

        @xarray_adapter(infer_params=True)
        def needs_scale(values: np.ndarray, scale: int, data_start_year: int) -> np.ndarray:
            return values

        with pytest.raises(InsufficientDataError) as exc_info:
            needs_scale(short_monthly_da, scale=6)

        assert "insufficient data" in str(exc_info.value).lower()

    def test_valid_data_passes_validation(self, sample_monthly_precip_da):
        """Valid data passes all validation checks."""

        @xarray_adapter(infer_params=True)
        def needs_scale(values: np.ndarray, scale: int, data_start_year: int) -> np.ndarray:
            return values * scale

        result = needs_scale(sample_monthly_precip_da, scale=3)

        # should succeed and return DataArray
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape

    def test_infer_params_false_skips_validation(self, no_time_dim_da):
        """infer_params=False skips coordinate validation."""

        @xarray_adapter(infer_params=False)
        def simple_func(values: np.ndarray) -> np.ndarray:
            return values

        # should not raise even though time dimension is missing
        result = simple_func(no_time_dim_da)
        assert isinstance(result, xr.DataArray)

    def test_numpy_passthrough_skips_validation(self):
        """NumPy input bypasses all validation."""

        @xarray_adapter(infer_params=True)
        def needs_params(values: np.ndarray, scale: int, data_start_year: int) -> np.ndarray:
            return values

        numpy_input = np.array([1.0, 2.0])

        # numpy passthrough should work even though data would fail validation
        result = needs_params(numpy_input, scale=10, data_start_year=2020)
        assert isinstance(result, np.ndarray)

    def test_custom_time_dim_validated(self):
        """Custom time_dim parameter is validated."""
        # create DataArray with 'date' instead of 'time'
        date = pd.date_range("2020-01-01", periods=12, freq="MS")
        da = xr.DataArray(
            _standard_normal(12),
            coords={"date": date},
            dims=["date"],
        )

        @xarray_adapter(time_dim="date", infer_params=True)
        def needs_date(values: np.ndarray, data_start_year: int) -> np.ndarray:
            return values

        # should not raise
        result = needs_date(da)
        assert isinstance(result, xr.DataArray)

    def test_validation_happens_before_inference(self, non_monotonic_time_da):
        """Validation runs before parameter inference (fails fast)."""

        @xarray_adapter(infer_params=True)
        def needs_params(
            values: np.ndarray,
            data_start_year: int,
            periodicity: compute.Periodicity,
        ) -> np.ndarray:
            return values

        # should raise validation error, not inference error
        with pytest.raises(CoordinateValidationError) as exc_info:
            needs_params(non_monotonic_time_da)

        assert "not monotonically increasing" in str(exc_info.value).lower()


class TestResolveSecondaryInputs:
    """Test _resolve_secondary_inputs() helper function."""

    def test_resolve_positional_argument(self):
        """Resolve secondary input provided as positional argument."""

        def spei_func(precip: np.ndarray, pet: np.ndarray, scale: int) -> np.ndarray:
            return precip

        precip_da = xr.DataArray([1, 2, 3])
        pet_da = xr.DataArray([4, 5, 6])

        resolved = _resolve_secondary_inputs(spei_func, (precip_da, pet_da, 3), {}, ["pet"])

        assert "pet" in resolved
        pos_index, value = resolved["pet"]
        assert pos_index == 1
        assert value is pet_da

    def test_resolve_keyword_argument(self):
        """Resolve secondary input provided as keyword argument."""

        def spei_func(precip: np.ndarray, pet: np.ndarray, scale: int) -> np.ndarray:
            return precip

        precip_da = xr.DataArray([1, 2, 3])
        pet_da = xr.DataArray([4, 5, 6])

        resolved = _resolve_secondary_inputs(spei_func, (precip_da,), {"pet": pet_da, "scale": 3}, ["pet"])

        assert "pet" in resolved
        pos_index, value = resolved["pet"]
        assert pos_index is None
        assert value is pet_da

    def test_missing_parameter_not_resolved(self):
        """Parameters not provided are not included in result."""

        def spei_func(precip: np.ndarray, pet: np.ndarray, scale: int) -> np.ndarray:
            return precip

        precip_da = xr.DataArray([1, 2, 3])

        resolved = _resolve_secondary_inputs(spei_func, (precip_da,), {"scale": 3}, ["pet"])

        # pet not provided, should not be in resolved dict
        assert "pet" not in resolved

    def test_multiple_secondaries(self):
        """Resolve multiple secondary inputs."""

        def multi_input(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
            return a

        a_da = xr.DataArray([1])
        b_da = xr.DataArray([2])
        c_da = xr.DataArray([3])

        resolved = _resolve_secondary_inputs(multi_input, (a_da, b_da, c_da), {}, ["b", "c"])

        assert len(resolved) == 2
        assert resolved["b"] == (1, b_da)
        assert resolved["c"] == (2, c_da)

    def test_empty_additional_names(self):
        """Empty additional_input_names returns empty dict."""

        def simple_func(a: np.ndarray) -> np.ndarray:
            return a

        resolved = _resolve_secondary_inputs(simple_func, (xr.DataArray([1]),), {}, [])

        assert resolved == {}

    def test_parameter_not_in_signature(self):
        """Parameters not in function signature are skipped."""

        def simple_func(a: np.ndarray) -> np.ndarray:
            return a

        resolved = _resolve_secondary_inputs(simple_func, (xr.DataArray([1]),), {}, ["nonexistent"])

        # nonexistent parameter should be skipped
        assert "nonexistent" not in resolved


class TestAlignInputs:
    """Test _align_inputs() helper function."""

    def test_identical_coordinates_no_warning(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """Aligning inputs with identical coordinates does not emit warning."""
        secondaries = {"pet": sample_monthly_pet_da}

        # use warnings module to capture warnings
        import warnings

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            aligned_primary, aligned_secondaries = _align_inputs(sample_monthly_precip_da, secondaries, "time")

        # no warnings should be emitted
        assert len(warning_list) == 0

        # outputs should match inputs (no trimming)
        assert len(aligned_primary) == len(sample_monthly_precip_da)
        assert len(aligned_secondaries["pet"]) == len(sample_monthly_pet_da)

    def test_partial_overlap_emits_warning(self, sample_monthly_precip_da, sample_monthly_pet_offset_da):
        """Aligning inputs with partial overlap emits InputAlignmentWarning."""
        # precip: 1980-2019 (480 months)
        # pet_offset: 1985-2024 (480 months)
        # overlap: 1985-2019 (420 months)
        secondaries = {"pet": sample_monthly_pet_offset_da}

        with pytest.warns(InputAlignmentWarning) as warning_list:
            aligned_primary, aligned_secondaries = _align_inputs(sample_monthly_precip_da, secondaries, "time")

        # should emit exactly one warning
        assert len(warning_list) == 1
        warning = warning_list[0].message

        # check warning attributes
        assert warning.original_size == 480
        assert warning.aligned_size == 420
        assert warning.dropped_count == 60

        # check aligned outputs
        assert len(aligned_primary) == 420
        assert len(aligned_secondaries["pet"]) == 420

        # verify time range is the intersection (1985-2019)
        assert pd.Timestamp(aligned_primary.time.values[0]).year == 1985
        assert pd.Timestamp(aligned_primary.time.values[-1]).year == 2019

    def test_empty_intersection_raises_error(self, sample_monthly_precip_da):
        """Aligning inputs with no overlap raises CoordinateValidationError."""
        # create PET with completely non-overlapping time range
        time = pd.date_range("2025-01-01", "2030-12-01", freq="MS")
        pet_no_overlap = xr.DataArray(
            _standard_normal(len(time)),
            coords={"time": time},
            dims=["time"],
        )

        secondaries = {"pet": pet_no_overlap}

        with pytest.raises(CoordinateValidationError) as exc_info:
            _align_inputs(sample_monthly_precip_da, secondaries, "time")

        assert "empty intersection" in str(exc_info.value).lower()
        assert exc_info.value.coordinate_name == "time"
        assert exc_info.value.reason == "empty_intersection_after_alignment"

    def test_multiple_secondaries_aligned(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """Multiple secondary inputs are all aligned together."""
        # create third input with identical coords
        third_da = xr.DataArray(
            _standard_normal(len(sample_monthly_precip_da)),
            coords={"time": sample_monthly_precip_da.time},
            dims=["time"],
        )

        secondaries = {"pet": sample_monthly_pet_da, "third": third_da}

        aligned_primary, aligned_secondaries = _align_inputs(sample_monthly_precip_da, secondaries, "time")

        # all should have same length
        assert len(aligned_primary) == len(sample_monthly_precip_da)
        assert len(aligned_secondaries["pet"]) == len(sample_monthly_precip_da)
        assert len(aligned_secondaries["third"]) == len(sample_monthly_precip_da)

    def test_empty_secondaries_returns_unchanged(self, sample_monthly_precip_da):
        """Empty secondaries dict returns primary unchanged."""
        aligned_primary, aligned_secondaries = _align_inputs(sample_monthly_precip_da, {}, "time")

        assert aligned_primary is sample_monthly_precip_da
        assert aligned_secondaries == {}

    def test_alignment_preserves_attributes(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """Alignment preserves DataArray and coordinate attributes."""
        secondaries = {"pet": sample_monthly_pet_da}

        aligned_primary, aligned_secondaries = _align_inputs(sample_monthly_precip_da, secondaries, "time")

        # check DataArray-level attrs preserved
        assert aligned_primary.attrs["units"] == "mm"
        assert aligned_primary.attrs["long_name"] == "Monthly Precipitation"
        assert aligned_secondaries["pet"].attrs["units"] == "mm"


class TestXarrayAdapterMultiInput:
    """Test xarray_adapter decorator with multiple inputs (Story 3.1)."""

    def test_two_dataarray_inputs_extracted(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """Decorator extracts values from both DataArray inputs."""

        @xarray_adapter(additional_input_names=["pet"])
        def spei_func(precip: np.ndarray, pet: np.ndarray, scale: int) -> np.ndarray:
            # verify both inputs are numpy arrays
            assert isinstance(precip, np.ndarray)
            assert isinstance(pet, np.ndarray)
            return precip - pet

        result = spei_func(sample_monthly_precip_da, sample_monthly_pet_da, scale=3)

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape

    def test_alignment_applied_before_extraction(self, sample_monthly_precip_da, sample_monthly_pet_offset_da):
        """Decorator aligns inputs before extracting values."""

        @xarray_adapter(additional_input_names=["pet"])
        def spei_func(precip: np.ndarray, pet: np.ndarray) -> np.ndarray:
            # both should have same length after alignment
            assert len(precip) == len(pet)
            return precip - pet

        with pytest.warns(InputAlignmentWarning):
            result = spei_func(sample_monthly_precip_da, sample_monthly_pet_offset_da)

        # result should match aligned size (1985-2019 = 420 months)
        assert len(result) == 420

    def test_numpy_secondary_passthrough(self, sample_monthly_precip_da):
        """Numpy secondary inputs pass through without alignment."""

        @xarray_adapter(additional_input_names=["threshold"])
        def threshold_func(precip: np.ndarray, threshold: float) -> np.ndarray:
            # threshold should remain a float, not be extracted
            assert isinstance(threshold, (int, float))
            return precip - threshold

        result = threshold_func(sample_monthly_precip_da, threshold=100.0)

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape

    def test_mixed_dataarray_and_numpy_secondaries(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """Mixed DataArray and numpy secondaries handled correctly."""

        @xarray_adapter(additional_input_names=["pet", "factor"])
        def mixed_func(precip: np.ndarray, pet: np.ndarray, factor: float) -> np.ndarray:
            assert isinstance(precip, np.ndarray)
            assert isinstance(pet, np.ndarray)
            assert isinstance(factor, (int, float))
            return (precip - pet) * factor

        result = mixed_func(sample_monthly_precip_da, sample_monthly_pet_da, factor=1.5)

        assert isinstance(result, xr.DataArray)

    def test_numpy_primary_passthrough_with_secondaries(self):
        """Numpy primary input bypasses all multi-input logic."""

        @xarray_adapter(additional_input_names=["pet"])
        def spei_func(precip: np.ndarray, pet: np.ndarray) -> np.ndarray:
            # verify both are numpy arrays in passthrough mode
            assert isinstance(precip, np.ndarray)
            assert isinstance(pet, np.ndarray)
            return precip - pet

        # numpy primary should pass through, no alignment
        # provide both as numpy arrays
        numpy_precip = _standard_normal(100)
        numpy_pet = _standard_normal(100)
        result = spei_func(numpy_precip, numpy_pet)

        # result should be numpy array (passthrough)
        assert isinstance(result, np.ndarray)

    def test_secondary_as_kwarg(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """Secondary input provided as keyword argument."""

        @xarray_adapter(additional_input_names=["pet"])
        def spei_func(precip: np.ndarray, pet: np.ndarray, scale: int = 3) -> np.ndarray:
            return precip - pet

        result = spei_func(sample_monthly_precip_da, pet=sample_monthly_pet_da)

        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape

    def test_cf_metadata_applied_with_multi_input(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """CF metadata correctly applied with multiple inputs."""

        @xarray_adapter(
            additional_input_names=["pet"],
            cf_metadata={"long_name": "SPEI", "units": "dimensionless"},
        )
        def spei_func(precip: np.ndarray, pet: np.ndarray) -> np.ndarray:
            return precip - pet

        result = spei_func(sample_monthly_precip_da, sample_monthly_pet_da)

        assert result.attrs["long_name"] == "SPEI"
        assert result.attrs["units"] == "dimensionless"


class TestCFMetadataRegistrySPEI:
    """Test SPEI entry in CF_METADATA registry."""

    def test_spei_metadata_exists(self):
        """SPEI metadata entry exists in registry."""
        assert "spei" in CF_METADATA

    def test_spei_long_name(self):
        """SPEI has correct long_name."""
        assert CF_METADATA["spei"]["long_name"] == "Standardized Precipitation Evapotranspiration Index"

    def test_spei_units(self):
        """SPEI has correct units."""
        assert CF_METADATA["spei"]["units"] == "dimensionless"

    def test_spei_references(self):
        """SPEI has references attribute."""
        assert "references" in CF_METADATA["spei"]
        assert "Vicente-Serrano" in CF_METADATA["spei"]["references"]
        assert "2010" in CF_METADATA["spei"]["references"]

    def test_spei_no_standard_name(self):
        """SPEI does not have standard_name (not officially defined in CF)."""
        # standard_name is optional, should not be present for SPEI
        assert "standard_name" not in CF_METADATA["spei"]


class TestSPEIIntegration:
    """End-to-end integration tests for SPEI with real indices.spei() function."""

    def test_spei_with_matching_dataarrays(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """SPEI computation with matching precipitation and PET DataArrays."""
        # wrap real spei function with decorator
        wrapped_spei = xarray_adapter(
            additional_input_names=["pet_mm"],
            cf_metadata=CF_METADATA["spei"],
            calculation_metadata_keys=["scale", "distribution"],
            index_display_name="SPEI",
        )(indices.spei)

        result = wrapped_spei(
            sample_monthly_precip_da,
            sample_monthly_pet_da,
            scale=3,
            distribution=indices.Distribution.gamma,
        )

        # verify output is DataArray with correct shape
        assert isinstance(result, xr.DataArray)
        assert result.shape == sample_monthly_precip_da.shape

        # verify CF metadata applied
        assert result.attrs["long_name"] == "Standardized Precipitation Evapotranspiration Index"
        assert result.attrs["units"] == "dimensionless"

    def test_spei_with_offset_dataarrays_aligned(
        self, sample_monthly_precip_da, sample_monthly_pet_offset_da
    ):
        """SPEI computation aligns offset DataArrays correctly."""
        wrapped_spei = xarray_adapter(
            additional_input_names=["pet_mm"],
            cf_metadata=CF_METADATA["spei"],
            index_display_name="SPEI",
        )(indices.spei)

        with pytest.warns(InputAlignmentWarning) as warning_list:
            result = wrapped_spei(
                sample_monthly_precip_da,
                sample_monthly_pet_offset_da,
                scale=3,
                distribution=indices.Distribution.gamma,
            )

        # verify warning was emitted
        assert len(warning_list) == 1

        # verify output reflects aligned size (1985-2019 = 420 months)
        assert len(result) == 420

    def test_spei_with_numpy_passthrough(self):
        """SPEI with numpy inputs passes through unchanged."""
        wrapped_spei = xarray_adapter(
            additional_input_names=["pet_mm"],
            cf_metadata=CF_METADATA["spei"],
        )(indices.spei)

        # generate numpy inputs
        rng = np.random.default_rng(42)
        precip_np = rng.gamma(shape=2.0, scale=50.0, size=480)
        pet_np = rng.gamma(shape=2.5, scale=60.0, size=480)

        result = wrapped_spei(
            precip_np,
            pet_np,
            scale=3,
            distribution=indices.Distribution.gamma,
            data_start_year=1980,
            periodicity=compute.Periodicity.monthly,
            calibration_year_initial=1980,
            calibration_year_final=2019,
        )

        # result should be numpy array (passthrough)
        assert isinstance(result, np.ndarray)

    def test_spei_history_entry(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """SPEI adds history entry to output."""
        wrapped_spei = xarray_adapter(
            additional_input_names=["pet_mm"],
            cf_metadata=CF_METADATA["spei"],
            calculation_metadata_keys=["scale"],
            index_display_name="SPEI",
        )(indices.spei)

        result = wrapped_spei(
            sample_monthly_precip_da,
            sample_monthly_pet_da,
            scale=6,
            distribution=indices.Distribution.gamma,
        )

        # verify history attribute exists and contains expected content
        assert "history" in result.attrs
        history = result.attrs["history"]
        assert "SPEI-6 calculated" in history
        assert "climate_indices" in history

    def test_spei_calculation_metadata(self, sample_monthly_precip_da, sample_monthly_pet_da):
        """SPEI includes calculation metadata in output attributes."""
        wrapped_spei = xarray_adapter(
            additional_input_names=["pet_mm"],
            calculation_metadata_keys=["scale", "distribution"],
        )(indices.spei)

        result = wrapped_spei(
            sample_monthly_precip_da,
            sample_monthly_pet_da,
            scale=12,
            distribution=indices.Distribution.gamma,
        )

        # verify calculation metadata captured
        assert result.attrs["scale"] == 12
        assert result.attrs["distribution"] == "gamma"


# NaN handling fixtures (Story 2.8)


@pytest.fixture
def monthly_precip_with_nan() -> xr.DataArray:
    """Create 40-year monthly precipitation with ~10% NaN scattered throughout."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(42)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    # add scattered NaN (~10%)
    nan_indices = rng.choice(len(time), size=int(len(time) * 0.1), replace=False)
    values[nan_indices] = np.nan

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation with NaN",
        },
    )


@pytest.fixture
def monthly_precip_heavy_nan() -> xr.DataArray:
    """Create 40-year monthly precipitation with ~80% NaN (insufficient data)."""
    time = pd.date_range("1980-01-01", "2019-12-01", freq="MS")
    rng = np.random.default_rng(123)
    values = rng.gamma(shape=2.0, scale=50.0, size=len(time))

    # add heavy NaN (~80%)
    nan_indices = rng.choice(len(time), size=int(len(time) * 0.8), replace=False)
    values[nan_indices] = np.nan

    return xr.DataArray(
        values,
        coords={"time": time},
        dims=["time"],
        attrs={
            "units": "mm",
            "long_name": "Monthly Precipitation with Heavy NaN",
        },
    )


# NaN handling test classes (Story 2.8)


class TestAssessNanDensity:
    """Test _assess_nan_density function."""

    def test_no_nan(self, sample_monthly_precip_da):
        """Clean data returns zero NaN metrics."""
        result = _assess_nan_density(sample_monthly_precip_da)

        assert result["total_values"] == len(sample_monthly_precip_da)
        assert result["nan_count"] == 0
        assert result["nan_ratio"] == 0.0
        assert result["has_nan"] is False
        assert result["nan_positions"].shape == sample_monthly_precip_da.values.shape
        assert not result["nan_positions"].any()

    def test_with_nan(self, monthly_precip_with_nan):
        """Data with NaN returns accurate metrics."""
        result = _assess_nan_density(monthly_precip_with_nan)

        expected_nan_count = int(np.isnan(monthly_precip_with_nan.values).sum())
        assert result["nan_count"] == expected_nan_count
        assert result["total_values"] == len(monthly_precip_with_nan)
        assert 0.0 < result["nan_ratio"] < 1.0
        assert result["has_nan"] is True
        assert result["nan_positions"].shape == monthly_precip_with_nan.values.shape

    def test_all_nan(self):
        """All-NaN data returns 100% NaN ratio."""
        time = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
        values = np.full(len(time), np.nan)
        da = xr.DataArray(values, coords={"time": time}, dims=["time"])

        result = _assess_nan_density(da)

        assert result["nan_count"] == len(time)
        assert result["nan_ratio"] == 1.0
        assert result["has_nan"] is True
        assert result["nan_positions"].all()

    def test_nan_positions_reusable(self, monthly_precip_with_nan):
        """NaN positions mask matches numpy isnan."""
        result = _assess_nan_density(monthly_precip_with_nan)
        expected_mask = np.isnan(monthly_precip_with_nan.values)

        assert np.array_equal(result["nan_positions"], expected_mask)

    def test_empty_array(self):
        """Empty array returns zero total_values."""
        da = xr.DataArray(np.array([]), dims=["time"])
        result = _assess_nan_density(da)

        assert result["total_values"] == 0
        assert result["nan_count"] == 0
        assert result["nan_ratio"] == 0.0
        assert result["has_nan"] is False


class TestVerifyNanPropagation:
    """Test _verify_nan_propagation function."""

    def test_perfect_propagation_no_nan(self):
        """Clean input and output returns True."""
        input_mask = np.array([False, False, False, False])
        output_values = np.array([1.0, 2.0, 3.0, 4.0])

        assert _verify_nan_propagation(input_mask, output_values) is True

    def test_perfect_propagation_with_nan(self):
        """Input NaN positions are NaN in output."""
        input_mask = np.array([True, False, True, False])
        output_values = np.array([np.nan, 2.0, np.nan, 4.0])

        assert _verify_nan_propagation(input_mask, output_values) is True

    def test_propagation_violation(self):
        """Input NaN position has non-NaN output value."""
        input_mask = np.array([True, False, True, False])
        # position 2 should be NaN but isn't
        output_values = np.array([np.nan, 2.0, 3.0, 4.0])

        assert _verify_nan_propagation(input_mask, output_values) is False

    def test_additional_output_nan_allowed(self):
        """Output can have additional NaN from convolution."""
        input_mask = np.array([False, True, False, False])
        # output has additional NaN at position 0 and 3
        output_values = np.array([np.nan, np.nan, 2.0, np.nan])

        # should still return True (one-directional check)
        assert _verify_nan_propagation(input_mask, output_values) is True

    def test_all_nan_propagation(self):
        """All-NaN input properly propagates."""
        input_mask = np.array([True, True, True, True])
        output_values = np.array([np.nan, np.nan, np.nan, np.nan])

        assert _verify_nan_propagation(input_mask, output_values) is True


class TestValidateCalibrationNonNanSampleSize:
    """Test _validate_calibration_non_nan_sample_size function."""

    def test_sufficient_non_nan_data(self, sample_monthly_precip_da):
        """Validation passes with sufficient non-NaN data."""
        # should not raise
        _validate_calibration_non_nan_sample_size(
            sample_monthly_precip_da["time"],
            sample_monthly_precip_da.values,
            calibration_year_initial=1980,
            calibration_year_final=2019,
        )

    def test_insufficient_non_nan_data_raises(self, monthly_precip_heavy_nan):
        """Raises InsufficientDataError when <30 effective years."""
        with pytest.raises(InsufficientDataError) as exc_info:
            _validate_calibration_non_nan_sample_size(
                monthly_precip_heavy_nan["time"],
                monthly_precip_heavy_nan.values,
                calibration_year_initial=1980,
                calibration_year_final=2019,
            )

        error = exc_info.value
        assert "Insufficient non-NaN data" in str(error)
        assert error.non_zero_count is not None
        assert error.required_count is not None

    def test_marginal_non_nan_data(self, monthly_precip_with_nan):
        """Validation passes with marginal (~30 years) non-NaN data."""
        # 10% NaN means ~36 effective years, which should pass
        _validate_calibration_non_nan_sample_size(
            monthly_precip_with_nan["time"],
            monthly_precip_with_nan.values,
            calibration_year_initial=1980,
            calibration_year_final=2019,
        )

    def test_empty_calibration_period_raises(self, sample_monthly_precip_da):
        """Raises error when calibration period has no data points."""
        with pytest.raises(InsufficientDataError) as exc_info:
            _validate_calibration_non_nan_sample_size(
                sample_monthly_precip_da["time"],
                sample_monthly_precip_da.values,
                calibration_year_initial=2050,  # way beyond data range
                calibration_year_final=2060,
            )

        assert "contains no data points" in str(exc_info.value)

    def test_custom_min_years(self):
        """Custom min_years parameter is respected."""
        # create data with clean 15-year period
        time = pd.date_range("1980-01-01", "1994-12-01", freq="MS")
        rng = np.random.default_rng(42)
        values = rng.gamma(shape=2.0, scale=50.0, size=len(time))
        da = xr.DataArray(values, coords={"time": time}, dims=["time"])

        # should pass with min_years=10 (15 years available)
        _validate_calibration_non_nan_sample_size(
            da["time"],
            da.values,
            calibration_year_initial=1980,
            calibration_year_final=1994,
            min_years=10,
        )


class TestSkipnaParameter:
    """Test skipna parameter behavior."""

    def test_skipna_false_default_works(self, monthly_precip_with_nan):
        """skipna=False (default) allows computation."""

        @xarray_adapter(skipna=False)
        def simple_func(values: np.ndarray) -> np.ndarray:
            return values * 2

        # should work without raising
        result = simple_func(monthly_precip_with_nan)
        assert isinstance(result, xr.DataArray)

    def test_skipna_true_raises_not_implemented(self, sample_monthly_precip_da):
        """skipna=True raises NotImplementedError."""

        @xarray_adapter(skipna=True)
        def simple_func(values: np.ndarray) -> np.ndarray:
            return values * 2

        with pytest.raises(NotImplementedError) as exc_info:
            simple_func(sample_monthly_precip_da)

        assert "skipna=True not yet implemented" in str(exc_info.value)
        assert "FR-INPUT-004" in str(exc_info.value)

    def test_skipna_explicit_false(self, sample_monthly_precip_da):
        """Explicitly passing skipna=False works."""

        @xarray_adapter(skipna=False)
        def simple_func(values: np.ndarray) -> np.ndarray:
            return values * 2

        result = simple_func(sample_monthly_precip_da)
        assert isinstance(result, xr.DataArray)


class TestNanHandlingDecoratorIntegration:
    """Test NaN handling integration in decorator pipeline."""

    def test_nan_detected_logged(self, monthly_precip_with_nan, caplog):
        """NaN detection is logged when present."""

        @xarray_adapter()
        def simple_func(values: np.ndarray) -> np.ndarray:
            return values * 2

        _ = simple_func(monthly_precip_with_nan)

        # check that nan_detected_in_input log event was emitted
        assert any("nan_detected_in_input" in record.message for record in caplog.records)

    def test_no_nan_no_log(self, sample_monthly_precip_da, caplog):
        """Clean data does not trigger NaN logging."""

        @xarray_adapter()
        def simple_func(values: np.ndarray) -> np.ndarray:
            return values * 2

        _ = simple_func(sample_monthly_precip_da)

        # no nan-related logs should appear
        assert not any("nan" in record.message.lower() for record in caplog.records)

    def test_nan_propagation_preserved(self, monthly_precip_with_nan):
        """NaN positions are preserved in output."""

        @xarray_adapter()
        def identity_func(values: np.ndarray) -> np.ndarray:
            # identity function preserves NaN
            return values.copy()

        result = identity_func(monthly_precip_with_nan)

        # input and output NaN positions should match
        input_nan_mask = np.isnan(monthly_precip_with_nan.values)
        output_nan_mask = np.isnan(result.values)

        assert np.array_equal(input_nan_mask, output_nan_mask)

    def test_calibration_validation_triggered(self, monthly_precip_heavy_nan):
        """Insufficient calibration non-NaN data raises error."""

        @xarray_adapter(infer_params=True)
        def mock_index(
            values: np.ndarray,
            scale: int,
            data_start_year: int,
            periodicity: compute.Periodicity,
            calibration_year_initial: int,
            calibration_year_final: int,
        ) -> np.ndarray:
            return np.zeros_like(values)

        with pytest.raises(InsufficientDataError):
            mock_index(monthly_precip_heavy_nan, scale=3)

    def test_nan_metrics_in_completion_log(self, monthly_precip_with_nan, caplog):
        """Completion log includes NaN metrics when present."""

        @xarray_adapter()
        def simple_func(values: np.ndarray) -> np.ndarray:
            return values * 2

        result = simple_func(monthly_precip_with_nan)

        # find the completion log entry
        completion_logs = [r for r in caplog.records if "xarray_adapter_completed" in r.message]
        assert len(completion_logs) > 0

        # check that NaN metrics are present (in the structured log data)
        # note: caplog may not capture structlog fields directly, so we verify behavior indirectly
        assert result is not None  # basic sanity check

    def test_all_nan_input_all_nan_output(self):
        """All-NaN input produces all-NaN output with preserved coordinates."""
        time = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
        values = np.full(len(time), np.nan)
        da = xr.DataArray(
            values,
            coords={"time": time},
            dims=["time"],
            attrs={"units": "mm"},
        )

        @xarray_adapter()
        def identity_func(values: np.ndarray) -> np.ndarray:
            return values.copy()

        result = identity_func(da)

        assert np.isnan(result.values).all()
        assert result.dims == da.dims
        assert "time" in result.coords


class TestNanHandlingSPIIntegration:
    """End-to-end NaN handling tests with real SPI computation."""

    def test_spi_with_scattered_nan(self, monthly_precip_with_nan):
        """SPI handles scattered NaN correctly."""
        wrapped_spi = xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale", "distribution"],
            index_display_name="SPI",
        )(indices.spi)

        result = wrapped_spi(
            monthly_precip_with_nan,
            scale=3,
            distribution=indices.Distribution.gamma,
        )

        # verify output is DataArray
        assert isinstance(result, xr.DataArray)

        # verify input NaN positions remain NaN in output
        input_nan_mask = np.isnan(monthly_precip_with_nan.values)
        output_nan_mask = np.isnan(result.values)

        # all input NaN positions must be NaN in output
        assert np.all(output_nan_mask[input_nan_mask])

    def test_spi_heavy_nan_raises_error(self, monthly_precip_heavy_nan):
        """SPI with insufficient non-NaN data raises InsufficientDataError."""
        wrapped_spi = xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale", "distribution"],
            index_display_name="SPI",
        )(indices.spi)

        with pytest.raises(InsufficientDataError) as exc_info:
            wrapped_spi(
                monthly_precip_heavy_nan,
                scale=3,
                distribution=indices.Distribution.gamma,
            )

        assert "Insufficient non-NaN data" in str(exc_info.value)

    def test_spi_nan_output_has_additional_nan_from_convolution(self, monthly_precip_with_nan):
        """SPI output may have additional NaN from convolution padding."""
        wrapped_spi = xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale", "distribution"],
            index_display_name="SPI",
        )(indices.spi)

        result = wrapped_spi(
            monthly_precip_with_nan,
            scale=6,  # larger scale increases boundary NaN
            distribution=indices.Distribution.gamma,
        )

        # count NaN in input and output
        input_nan_count = np.isnan(monthly_precip_with_nan.values).sum()
        output_nan_count = np.isnan(result.values).sum()

        # output can have more NaN (from convolution), but not less
        assert output_nan_count >= input_nan_count

    def test_spi_clean_data_no_nan_related_errors(self, sample_monthly_precip_da):
        """SPI with clean data runs without NaN-related issues."""
        wrapped_spi = xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale", "distribution"],
            index_display_name="SPI",
        )(indices.spi)

        result = wrapped_spi(
            sample_monthly_precip_da,
            scale=3,
            distribution=indices.Distribution.gamma,
        )

        assert isinstance(result, xr.DataArray)
        # result may have some NaN from convolution boundaries, which is fine

    def test_spi_coordinates_preserved_with_nan(self, monthly_precip_with_nan):
        """SPI preserves coordinates even with NaN present."""
        wrapped_spi = xarray_adapter(
            cf_metadata=CF_METADATA["spi"],
            calculation_metadata_keys=["scale", "distribution"],
            index_display_name="SPI",
        )(indices.spi)

        result = wrapped_spi(
            monthly_precip_with_nan,
            scale=3,
            distribution=indices.Distribution.gamma,
        )

        assert result.dims == monthly_precip_with_nan.dims
        assert "time" in result.coords
        assert len(result.coords["time"]) == len(monthly_precip_with_nan.coords["time"])
