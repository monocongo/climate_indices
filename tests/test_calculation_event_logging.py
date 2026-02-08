"""Tests for calculation event logging (Story 1.6)."""

from __future__ import annotations

import json
import logging
from io import StringIO

import numpy as np
import pytest

from climate_indices import compute, indices
from climate_indices.eto import eto_hargreaves
from climate_indices.logging_config import _reset_logging_for_testing, configure_logging


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


class TestSPIEventLogging:
    """Test calculation event logging for SPI."""

    def test_spi_emits_started_and_completed(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """SPI emits both calculation_started and calculation_completed events."""
        # call spi
        result = indices.spi(
            values=precips_mm_monthly,
            scale=6,
            distribution=indices.Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        # parse log events
        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")
        completed_events = filter_events(events, "calculation_completed")

        # verify one started and one completed event
        assert len(started_events) == 1
        assert len(completed_events) == 1

        # verify started event fields
        started = started_events[0]
        assert started["index_type"] == "spi"
        assert started["scale"] == 6
        assert started["distribution"] == "gamma"
        assert "input_shape" in started

        # verify completed event fields
        completed = completed_events[0]
        assert "duration_ms" in completed
        assert isinstance(completed["duration_ms"], (int, float))
        assert completed["duration_ms"] > 0
        assert "output_shape" in completed
        # json serializes tuples as lists
        assert tuple(completed["output_shape"]) == result.shape

    def test_spi_all_nan_still_emits_completed(self, log_capture):
        """SPI with all-NaN input still emits calculation_completed."""
        nan_values = np.full(240, np.nan)

        result = indices.spi(
            values=nan_values,
            scale=3,
            distribution=indices.Distribution.gamma,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")
        completed_events = filter_events(events, "calculation_completed")

        assert len(started_events) == 1
        assert len(completed_events) == 1
        assert np.all(np.isnan(result))


class TestSPEIEventLogging:
    """Test calculation event logging for SPEI."""

    def test_spei_emits_started_and_completed(
        self,
        log_capture,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """SPEI emits both calculation_started and calculation_completed events."""
        result = indices.spei(
            precips_mm=precips_mm_monthly,
            pet_mm=pet_thornthwaite_mm,
            scale=6,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")
        completed_events = filter_events(events, "calculation_completed")

        assert len(started_events) == 1
        assert len(completed_events) == 1

        # verify fields
        started = started_events[0]
        assert started["index_type"] == "spei"
        assert started["scale"] == 6
        assert started["distribution"] == "gamma"
        assert "input_shape" in started

        completed = completed_events[0]
        assert completed["duration_ms"] > 0
        assert tuple(completed["output_shape"]) == result.shape

    def test_spei_all_nan_still_emits_completed(self, log_capture):
        """SPEI with all-NaN input still emits calculation_completed."""
        nan_precips = np.full(240, np.nan)
        nan_pet = np.full(240, np.nan)

        result = indices.spei(
            precips_mm=nan_precips,
            pet_mm=nan_pet,
            scale=3,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2019,
        )

        events = parse_log_events(log_capture)
        completed_events = filter_events(events, "calculation_completed")
        assert len(completed_events) == 1
        assert np.all(np.isnan(result))


class TestPercentageOfNormalEventLogging:
    """Test calculation event logging for percentage_of_normal."""

    def test_percentage_of_normal_emits_events(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Percentage of normal emits calculation events."""
        result = indices.percentage_of_normal(
            values=precips_mm_monthly.flatten(),
            scale=6,
            data_start_year=data_year_start_monthly,
            calibration_start_year=calibration_year_start_monthly,
            calibration_end_year=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")
        completed_events = filter_events(events, "calculation_completed")

        assert len(started_events) == 1
        assert len(completed_events) == 1

        # verify no distribution field for percentage_of_normal
        started = started_events[0]
        assert started["index_type"] == "percentage_of_normal"
        assert started["scale"] == 6
        assert "distribution" not in started
        assert "input_shape" in started

        completed = completed_events[0]
        assert completed["duration_ms"] > 0
        assert tuple(completed["output_shape"]) == result.shape


class TestPETEventLogging:
    """Test calculation event logging for PET (Thornthwaite)."""

    def test_pet_emits_events(
        self,
        log_capture,
        temps_celsius,
        latitude_degrees,
        data_year_start_monthly,
    ):
        """PET emits calculation events."""
        result = indices.pet(
            temperature_celsius=temps_celsius,
            latitude_degrees=latitude_degrees,
            data_start_year=data_year_start_monthly,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")
        completed_events = filter_events(events, "calculation_completed")

        assert len(started_events) == 1
        assert len(completed_events) == 1

        # verify no scale or distribution for pet
        started = started_events[0]
        assert started["index_type"] == "pet_thornthwaite"
        assert "scale" not in started
        assert "distribution" not in started
        assert "input_shape" in started

        completed = completed_events[0]
        assert completed["duration_ms"] > 0
        assert tuple(completed["output_shape"]) == result.shape

    def test_pet_all_nan_still_emits_completed(self, log_capture, latitude_degrees):
        """PET with all-NaN input still emits calculation_completed."""
        nan_temps = np.full(240, np.nan)

        result = indices.pet(
            temperature_celsius=nan_temps,
            latitude_degrees=latitude_degrees,
            data_start_year=2000,
        )

        events = parse_log_events(log_capture)
        completed_events = filter_events(events, "calculation_completed")
        assert len(completed_events) == 1
        assert np.all(np.isnan(result))


class TestPCIEventLogging:
    """Test calculation event logging for PCI."""

    def test_pci_emits_events(self, log_capture, rain_mm_366):
        """PCI emits calculation events."""
        indices.pci(rainfall_mm=rain_mm_366.flatten())

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")
        completed_events = filter_events(events, "calculation_completed")

        assert len(started_events) == 1
        assert len(completed_events) == 1

        # verify no scale or distribution for pci
        started = started_events[0]
        assert started["index_type"] == "pci"
        assert "scale" not in started
        assert "distribution" not in started
        assert "input_shape" in started

        completed = completed_events[0]
        assert completed["duration_ms"] > 0
        assert "output_shape" in completed

    def test_pci_all_nan_still_emits_completed(self, log_capture):
        """PCI with all-NaN input still emits calculation_completed."""
        nan_rainfall = np.full(366, np.nan)

        result = indices.pci(rainfall_mm=nan_rainfall)

        events = parse_log_events(log_capture)
        completed_events = filter_events(events, "calculation_completed")
        assert len(completed_events) == 1
        assert np.all(np.isnan(result))


class TestEtoHargreavesEventLogging:
    """Test calculation event logging for eto_hargreaves."""

    def test_eto_hargreaves_emits_events(
        self,
        log_capture,
        hargreaves_daily_tmin_celsius,
        hargreaves_daily_tmax_celsius,
        hargreaves_daily_tmean_celsius,
        hargreaves_latitude_degrees,
    ):
        """ETo Hargreaves emits calculation events."""
        result = eto_hargreaves(
            daily_tmin_celsius=hargreaves_daily_tmin_celsius,
            daily_tmax_celsius=hargreaves_daily_tmax_celsius,
            daily_tmean_celsius=hargreaves_daily_tmean_celsius,
            latitude_degrees=hargreaves_latitude_degrees,
        )

        events = parse_log_events(log_capture)
        started_events = filter_events(events, "calculation_started")
        completed_events = filter_events(events, "calculation_completed")

        assert len(started_events) == 1
        assert len(completed_events) == 1

        started = started_events[0]
        assert started["index_type"] == "pet_hargreaves"
        assert "input_shape" in started

        completed = completed_events[0]
        assert completed["duration_ms"] > 0
        assert tuple(completed["output_shape"]) == result.shape


class TestDurationAndShapeFields:
    """Test duration_ms and shape fields across all functions."""

    def test_duration_ms_is_positive_number(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Duration_ms is a positive number across all functions."""
        # call spi
        indices.spi(
            values=precips_mm_monthly,
            scale=1,
            distribution=indices.Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        events = parse_log_events(log_capture)
        completed_events = filter_events(events, "calculation_completed")
        assert len(completed_events) == 1

        duration = completed_events[0]["duration_ms"]
        assert isinstance(duration, (int, float))
        assert duration > 0
        # verify it's rounded to 2 decimal places or is an integer
        assert duration == round(duration, 2)

    def test_no_data_values_in_logs(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Log events do not contain actual data values (privacy check)."""
        indices.spi(
            values=precips_mm_monthly,
            scale=6,
            distribution=indices.Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        # get the raw log output
        log_capture.seek(0)
        raw_logs = log_capture.read()

        # verify no precipitation values appear in logs
        # check a few specific values from the precips array
        # flatten to iterate over individual values
        for val in precips_mm_monthly.flatten()[:10]:
            if not np.isnan(val):
                # convert to string representation
                val_str = str(float(val))
                assert val_str not in raw_logs, f"Found data value {val_str} in logs"

        # verify "values" field is not in any event (we only log shape)
        events = parse_log_events(StringIO(raw_logs))
        for event in events:
            assert "values" not in event
            assert "precips_mm" not in event
            assert "temperature_celsius" not in event
