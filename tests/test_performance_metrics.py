"""Tests for performance metrics logging (Story 1.8)."""

from __future__ import annotations

import json
import logging
from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest

from climate_indices import compute, indices
from climate_indices.eto import eto_hargreaves
from climate_indices.logging_config import _reset_logging_for_testing, configure_logging
from climate_indices.performance import (
    LARGE_ARRAY_THRESHOLD_BYTES,
    _reset_psutil_cache,
    check_large_array_memory,
    get_process_memory_mb,
)


@pytest.fixture(autouse=False)
def log_capture():
    """Set up JSON logging with output capture.

    This fixture must be explicitly requested by tests that need it.
    It configures structlog for JSON output and captures log events.
    """
    # ensure clean state
    _reset_logging_for_testing()

    # configure logging for JSON output
    configure_logging(log_format="json", log_level="INFO")

    # set up capture stream
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()

    # copy formatter from the configured handler
    if root.handlers:
        original_handler = root.handlers[0]
        handler.setFormatter(original_handler.formatter)
        root.handlers = [handler]

    yield stream

    # cleanup
    _reset_logging_for_testing()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


def parse_log_events(stream: StringIO) -> list[dict]:
    """Parse all JSON log lines from the captured stream."""
    stream.seek(0)
    events = []
    for line in stream.readlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def filter_events(events: list[dict], event_name: str) -> list[dict]:
    """Filter events by event field."""
    return [e for e in events if e.get("event") == event_name]


class TestGetProcessMemoryMb:
    """Test get_process_memory_mb function."""

    def test_returns_float_when_psutil_available(self):
        """get_process_memory_mb returns float when psutil is installed."""
        # reset cache to ensure fresh import attempt
        _reset_psutil_cache()

        # attempt to import psutil
        try:
            import psutil  # noqa: F401

            psutil_available = True
        except ImportError:
            psutil_available = False

        if psutil_available:
            result = get_process_memory_mb()
            assert isinstance(result, float)
            assert result > 0
        else:
            pytest.skip("psutil not installed")

    def test_returns_none_when_psutil_unavailable(self):
        """get_process_memory_mb returns None when psutil is not installed."""
        # reset cache
        _reset_psutil_cache()

        # mock psutil as unavailable
        with patch.dict("sys.modules", {"psutil": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                result = get_process_memory_mb()
                assert result is None

    def test_caches_psutil_availability(self):
        """get_process_memory_mb caches psutil availability check."""
        # reset cache
        _reset_psutil_cache()

        # first call
        result1 = get_process_memory_mb()

        # second call should use cached result (no new import attempt)
        result2 = get_process_memory_mb()

        # both calls should return same type
        assert type(result1) is type(result2)


class TestCheckLargeArrayMemory:
    """Test check_large_array_memory function."""

    def test_small_arrays_return_none(self):
        """check_large_array_memory returns None for small arrays."""
        # create small array (1000 elements * 8 bytes = 8KB)
        small_array = np.zeros(1000, dtype=np.float64)

        result = check_large_array_memory(small_array)

        assert result is None

    def test_large_single_array_returns_metrics(self):
        """check_large_array_memory returns metrics for large single array."""
        # create array just over 1GB threshold
        # 1GB = 1,073,741,824 bytes / 8 bytes per float64 = 134,217,728 elements
        elements = (LARGE_ARRAY_THRESHOLD_BYTES // 8) + 1000
        large_array = np.zeros(elements, dtype=np.float64)

        result = check_large_array_memory(large_array)

        assert result is not None
        assert "array_memory_mb" in result
        assert result["array_memory_mb"] > 1024.0

    def test_multiple_arrays_summed(self):
        """check_large_array_memory sums memory across multiple arrays."""
        # create two arrays that together exceed 1GB
        # each array is ~600MB
        elements_per_array = (LARGE_ARRAY_THRESHOLD_BYTES // 8) // 2 + 10_000_000
        array1 = np.zeros(elements_per_array, dtype=np.float64)
        array2 = np.zeros(elements_per_array, dtype=np.float64)

        result = check_large_array_memory(array1, array2)

        assert result is not None
        assert "array_memory_mb" in result
        # should be over 1GB
        assert result["array_memory_mb"] > 1024.0

    def test_boundary_exactly_threshold_returns_none(self):
        """check_large_array_memory returns None when exactly at threshold."""
        # create array exactly at 1GB (boundary condition)
        elements = LARGE_ARRAY_THRESHOLD_BYTES // 8
        boundary_array = np.zeros(elements, dtype=np.float64)

        result = check_large_array_memory(boundary_array)

        # at threshold should return None (only > threshold triggers)
        assert result is None

    def test_includes_process_memory_when_psutil_available(self):
        """check_large_array_memory includes process_memory_mb when psutil available."""
        # reset cache
        _reset_psutil_cache()

        # check if psutil is available
        try:
            import psutil  # noqa: F401

            psutil_available = True
        except ImportError:
            psutil_available = False

        if not psutil_available:
            pytest.skip("psutil not installed")

        # create large array
        elements = (LARGE_ARRAY_THRESHOLD_BYTES // 8) + 1000
        large_array = np.zeros(elements, dtype=np.float64)

        result = check_large_array_memory(large_array)

        assert result is not None
        assert "array_memory_mb" in result
        assert "process_memory_mb" in result
        assert isinstance(result["process_memory_mb"], float)
        assert result["process_memory_mb"] > 0

    def test_excludes_process_memory_when_psutil_unavailable(self):
        """check_large_array_memory excludes process_memory_mb when psutil unavailable."""
        # reset cache
        _reset_psutil_cache()

        # mock psutil as unavailable
        with patch.dict("sys.modules", {"psutil": None}):
            # trigger cache update
            get_process_memory_mb()

            class MockArray:
                nbytes = LARGE_ARRAY_THRESHOLD_BYTES + 1000

            result = check_large_array_memory(MockArray())

            assert result is not None
            assert "array_memory_mb" in result
            assert "process_memory_mb" not in result


class TestInputElementsInLogs:
    """Test that input_elements appears in log events."""

    def test_spi_logs_input_elements(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """SPI logs input_elements in calculation_started event."""
        indices.spi(
            values=precips_mm_monthly,
            scale=6,
            distribution=indices.Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")

        assert len(started_events) == 1
        started = started_events[0]
        assert "input_elements" in started
        assert started["input_elements"] == precips_mm_monthly.size

    def test_spei_logs_input_elements(
        self,
        log_capture,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """SPEI logs input_elements in calculation_started event."""
        indices.spei(
            precips_mm=precips_mm_monthly,
            pet_mm=pet_thornthwaite_mm,
            scale=6,
            distribution=indices.Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")

        assert len(started_events) == 1
        started = started_events[0]
        assert "input_elements" in started
        assert started["input_elements"] == precips_mm_monthly.size

    def test_percentage_of_normal_logs_input_elements(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """percentage_of_normal logs input_elements in calculation_started event."""
        flat_precips = precips_mm_monthly.flatten()
        indices.percentage_of_normal(
            values=flat_precips,
            scale=6,
            data_start_year=data_year_start_monthly,
            calibration_start_year=calibration_year_start_monthly,
            calibration_end_year=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")

        assert len(started_events) == 1
        started = started_events[0]
        assert "input_elements" in started
        assert started["input_elements"] == flat_precips.size

    def test_pet_logs_input_elements(
        self,
        log_capture,
        temps_celsius,
        latitude_degrees,
        data_year_start_monthly,
    ):
        """PET logs input_elements in calculation_started event."""
        indices.pet(
            temperature_celsius=temps_celsius,
            latitude_degrees=latitude_degrees,
            data_start_year=data_year_start_monthly,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")

        assert len(started_events) == 1
        started = started_events[0]
        assert "input_elements" in started
        assert started["input_elements"] == temps_celsius.size

    def test_pci_logs_input_elements(
        self,
        log_capture,
        rain_mm_366,
    ):
        """PCI logs input_elements in calculation_started event."""
        # PCI expects 1D array of 366 or 365 daily values
        flat_rain = rain_mm_366.flatten()
        indices.pci(flat_rain)

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")

        assert len(started_events) == 1
        started = started_events[0]
        assert "input_elements" in started
        assert started["input_elements"] == flat_rain.size

    def test_eto_hargreaves_logs_input_elements(
        self,
        log_capture,
        hargreaves_daily_tmin_celsius,
        hargreaves_daily_tmax_celsius,
        hargreaves_daily_tmean_celsius,
        hargreaves_latitude_degrees,
    ):
        """eto_hargreaves logs input_elements in calculation_started event."""
        eto_hargreaves(
            daily_tmin_celsius=hargreaves_daily_tmin_celsius,
            daily_tmax_celsius=hargreaves_daily_tmax_celsius,
            daily_tmean_celsius=hargreaves_daily_tmean_celsius,
            latitude_degrees=hargreaves_latitude_degrees,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")

        assert len(started_events) == 1
        started = started_events[0]
        assert "input_elements" in started
        assert started["input_elements"] == hargreaves_daily_tmean_celsius.size

    def test_input_elements_in_failed_events(
        self,
        log_capture,
        precips_mm_monthly,
    ):
        """input_elements appears in calculation_failed events (context-bound)."""
        # trigger failure with unsupported array shape (3D array)
        # this error occurs after logger is bound, inside the try block
        invalid_3d_array = np.zeros((10, 10, 10))
        with pytest.raises(ValueError, match="Invalid shape"):
            indices.spi(
                values=invalid_3d_array,
                scale=6,
                distribution=indices.Distribution.gamma,
                data_start_year=1895,
                calibration_year_initial=1895,
                calibration_year_final=1900,
                periodicity=compute.Periodicity.monthly,
            )

        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")

        assert len(failed_events) >= 1
        failed = failed_events[0]
        assert "input_elements" in failed
        assert failed["input_elements"] == invalid_3d_array.size


class TestMemoryMetricsInLogs:
    """Test that memory metrics appear in log events for large arrays."""

    def test_small_arrays_no_memory_metrics(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Small arrays do not log array_memory_mb in calculation_completed."""
        indices.spi(
            values=precips_mm_monthly,
            scale=6,
            distribution=indices.Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        events = parse_log_events(log_capture)
        completed_events = filter_events(events, "calculation_completed")

        assert len(completed_events) == 1
        completed = completed_events[0]
        # small arrays should not include memory metrics
        assert "array_memory_mb" not in completed
        assert "process_memory_mb" not in completed

    def test_large_arrays_include_memory_metrics(
        self,
        log_capture,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Large arrays include array_memory_mb in calculation_completed."""
        # mock check_large_array_memory to return metrics without allocating memory
        mock_metrics = {"array_memory_mb": 1500.5, "process_memory_mb": 2048.75}

        with patch("climate_indices.indices.check_large_array_memory", return_value=mock_metrics):
            # use small array but mocked check will return large metrics
            small_array = np.zeros(1000, dtype=np.float64)
            indices.spi(
                values=small_array,
                scale=6,
                distribution=indices.Distribution.gamma,
                data_start_year=data_year_start_monthly,
                calibration_year_initial=calibration_year_start_monthly,
                calibration_year_final=calibration_year_end_monthly,
                periodicity=compute.Periodicity.monthly,
            )

        events = parse_log_events(log_capture)
        completed_events = filter_events(events, "calculation_completed")

        assert len(completed_events) == 1
        completed = completed_events[0]
        # mocked large arrays should include memory metrics
        assert "array_memory_mb" in completed
        assert completed["array_memory_mb"] == 1500.5
        assert "process_memory_mb" in completed
        assert completed["process_memory_mb"] == 2048.75


class TestCustomMetricsViaContextBinding:
    """Test that custom metrics can be added via context binding API."""

    def test_custom_fields_propagate_to_events(self, log_capture):
        """User-bound fields propagate to all log events from that logger."""
        from climate_indices.logging_config import get_logger

        log = get_logger("test_custom_metrics")

        # bind custom metrics
        log = log.bind(
            custom_metric_1="test_value",
            custom_metric_2=42,
            custom_metric_3=3.14159,
        )

        # emit events
        log.info("test_event_1")
        log.info("test_event_2", additional_field="extra")

        events = parse_log_events(log_capture)

        # verify custom metrics appear in all events
        assert len(events) == 2
        for event in events:
            assert event["custom_metric_1"] == "test_value"
            assert event["custom_metric_2"] == 42
            assert event["custom_metric_3"] == 3.14159

        # verify additional field only in second event
        assert "additional_field" not in events[0]
        assert events[1]["additional_field"] == "extra"
