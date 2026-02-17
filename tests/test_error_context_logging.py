"""Tests for error context logging (Story 1.7)."""

from __future__ import annotations

import json
import logging
from io import StringIO
from unittest import mock

import numpy as np
import pytest

from climate_indices import compute, indices
from climate_indices.eto import eto_hargreaves
from climate_indices.exceptions import InvalidArgumentError
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


class TestSPIErrorContextLogging:
    """Test calculation_failed event logging for SPI."""

    def test_gamma_cdf_failure_emits_calculation_failed(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Gamma CDF failure emits calculation_failed with full context."""
        # mock gamma.cdf to raise an error
        with mock.patch("scipy.stats.gamma.cdf", side_effect=ValueError("CDF computation failed")):
            with pytest.raises(compute.DistributionFittingError):
                indices.spi(
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
        failed_events = filter_events(events, "calculation_failed")

        # verify one calculation_failed event
        assert len(failed_events) == 1

        # verify all required fields
        failed = failed_events[0]
        assert failed["level"] == "error"
        assert failed["index_type"] == "spi"
        assert failed["scale"] == 6
        assert failed["distribution"] == "gamma"
        assert "input_shape" in failed
        assert failed["error_type"] == "DistributionFittingError"
        assert "CDF computation failed" in failed["error_message"]
        assert failed["calibration_period"] == f"{calibration_year_start_monthly}-{calibration_year_end_monthly}"
        assert "exception" in failed
        # verify traceback contains relevant information
        assert "CDF computation failed" in failed["exception"]

    def test_pearson_fallback_succeeds_no_calculation_failed(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Pearson fallback to gamma succeeds without calculation_failed."""

        def mock_pearson_fail(*args, **kwargs):
            raise compute.DistributionFittingError("Pearson failed")

        # mock pearson to fail but allow gamma to succeed (don't mock gamma)
        with mock.patch("climate_indices.compute.transform_fitted_pearson", side_effect=mock_pearson_fail):
            result = indices.spi(
                values=precips_mm_monthly,
                scale=3,
                distribution=indices.Distribution.pearson,
                data_start_year=data_year_start_monthly,
                calibration_year_initial=calibration_year_start_monthly,
                calibration_year_final=calibration_year_end_monthly,
                periodicity=compute.Periodicity.monthly,
            )

        # verify successful result
        assert result is not None
        # spi returns a 1D array
        assert len(result.shape) == 1
        assert result.size == precips_mm_monthly.size

        # verify no calculation_failed event (fallback succeeded)
        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 0

        # verify calculation_completed was emitted
        completed_events = filter_events(events, "calculation_completed")
        assert len(completed_events) == 1

    def test_pearson_and_gamma_both_fail_emits_calculation_failed(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Pearson fallback where gamma also fails emits calculation_failed."""
        # mock both to fail
        with mock.patch(
            "climate_indices.compute.transform_fitted_pearson",
            side_effect=ValueError("Pearson failed"),
        ):
            with mock.patch(
                "climate_indices.compute.transform_fitted_gamma",
                side_effect=RuntimeError("Gamma fallback failed"),
            ):
                with pytest.raises(RuntimeError, match="Gamma fallback failed"):
                    indices.spi(
                        values=precips_mm_monthly,
                        scale=6,
                        distribution=indices.Distribution.pearson,
                        data_start_year=data_year_start_monthly,
                        calibration_year_initial=calibration_year_start_monthly,
                        calibration_year_final=calibration_year_end_monthly,
                        periodicity=compute.Periodicity.monthly,
                    )

        # verify calculation_failed event
        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]
        assert failed["error_type"] == "RuntimeError"
        assert "Gamma fallback failed" in failed["error_message"]

    def test_invalid_shape_emits_calculation_failed(self, log_capture):
        """Invalid array shape emits calculation_failed."""
        # create 3-D array (invalid)
        invalid_array = np.random.rand(10, 12, 5)

        with pytest.raises(ValueError, match="Invalid shape"):
            indices.spi(
                values=invalid_array,
                scale=3,
                distribution=indices.Distribution.gamma,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2019,
                periodicity=compute.Periodicity.monthly,
            )

        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]
        assert failed["error_type"] == "ValueError"
        assert "Invalid shape" in failed["error_message"]


class TestSPEIErrorContextLogging:
    """Test calculation_failed event logging for SPEI."""

    def test_gamma_failure_emits_calculation_failed(
        self,
        log_capture,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """SPEI gamma failure emits calculation_failed."""
        with mock.patch("scipy.stats.gamma.cdf", side_effect=ValueError("Gamma CDF failed")):
            with pytest.raises(compute.DistributionFittingError):
                indices.spei(
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
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]
        assert failed["index_type"] == "spei"
        assert failed["scale"] == 6
        assert failed["distribution"] == "gamma"
        assert failed["error_type"] == "DistributionFittingError"
        assert "Gamma CDF failed" in failed["error_message"]
        assert failed["calibration_period"] == f"{calibration_year_start_monthly}-{calibration_year_end_monthly}"

    def test_incompatible_arrays_emits_calculation_failed(
        self,
        log_capture,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """SPEI with mismatched array sizes emits calculation_failed."""
        precips = np.random.rand(240)
        pet = np.random.rand(120)

        with pytest.raises(ValueError, match="Incompatible"):
            indices.spei(
                precips_mm=precips,
                pet_mm=pet,
                scale=3,
                distribution=indices.Distribution.gamma,
                periodicity=compute.Periodicity.monthly,
                data_start_year=data_year_start_monthly,
                calibration_year_initial=calibration_year_start_monthly,
                calibration_year_final=calibration_year_end_monthly,
            )

        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1


class TestPercentageOfNormalErrorContextLogging:
    """Test calculation_failed event logging for percentage_of_normal."""

    def test_invalid_calibration_emits_calculation_failed(self, log_capture):
        """Invalid calibration period emits calculation_failed."""
        values = np.random.rand(240)

        with pytest.raises(InvalidArgumentError, match="calibration start year"):
            indices.percentage_of_normal(
                values=values,
                scale=6,
                data_start_year=2010,
                calibration_start_year=2000,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]
        assert failed["index_type"] == "percentage_of_normal"
        assert failed["scale"] == 6
        assert "distribution" not in failed
        assert failed["error_type"] == "InvalidArgumentError"
        assert failed["calibration_period"] == "2000-2019"


class TestPETErrorContextLogging:
    """Test calculation_failed event logging for PET."""

    def test_invalid_latitude_emits_calculation_failed(self, log_capture):
        """PET with invalid latitude emits calculation_failed."""
        temps = np.random.rand(240)

        with pytest.raises(ValueError, match="Invalid latitude"):
            indices.pet(
                temperature_celsius=temps,
                latitude_degrees=100.0,
                data_start_year=2000,
            )

        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]
        assert failed["index_type"] == "pet_thornthwaite"
        assert "scale" not in failed
        assert "distribution" not in failed
        assert "calibration_period" not in failed
        assert failed["error_type"] == "ValueError"
        assert "Invalid latitude" in failed["error_message"]


class TestPCIErrorContextLogging:
    """Test calculation_failed event logging for PCI."""

    def test_invalid_length_emits_calculation_failed(self, log_capture):
        """PCI with invalid array length emits calculation_failed."""
        # array with 100 days (invalid, must be 365 or 366)
        rainfall = np.random.rand(100)

        with pytest.raises(InvalidArgumentError, match="365 or 366"):
            indices.pci(rainfall_mm=rainfall)

        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]
        assert failed["index_type"] == "pci"
        assert "scale" not in failed
        assert "distribution" not in failed
        assert "calibration_period" not in failed
        assert failed["error_type"] == "InvalidArgumentError"


class TestEtoHargreavesErrorContextLogging:
    """Test calculation_failed event logging for eto_hargreaves."""

    def test_computation_error_emits_calculation_failed(self, log_capture):
        """ETo Hargreaves computation error emits calculation_failed."""
        tmin = np.random.rand(366)
        tmax = np.random.rand(366)
        tmean = np.random.rand(366)

        # mock reshape_to_2d to raise an error
        with mock.patch("climate_indices.utils.reshape_to_2d", side_effect=RuntimeError("Reshape failed")):
            with pytest.raises(RuntimeError, match="Reshape failed"):
                eto_hargreaves(
                    daily_tmin_celsius=tmin,
                    daily_tmax_celsius=tmax,
                    daily_tmean_celsius=tmean,
                    latitude_degrees=40.0,
                )

        events = parse_log_events(log_capture)
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]
        assert failed["index_type"] == "pet_hargreaves"
        assert failed["error_type"] == "RuntimeError"
        assert "Reshape failed" in failed["error_message"]


class TestErrorLogPrivacy:
    """Test that error logs do not expose raw data."""

    def test_no_data_values_in_error_logs(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Error logs do not contain actual data values."""
        # mock to trigger error
        with mock.patch("scipy.stats.gamma.cdf", side_effect=ValueError("Test error")):
            with pytest.raises(compute.DistributionFittingError):
                indices.spi(
                    values=precips_mm_monthly,
                    scale=6,
                    distribution=indices.Distribution.gamma,
                    data_start_year=data_year_start_monthly,
                    calibration_year_initial=calibration_year_start_monthly,
                    calibration_year_final=calibration_year_end_monthly,
                    periodicity=compute.Periodicity.monthly,
                )

        # get raw log output
        log_capture.seek(0)
        raw_logs = log_capture.read()

        # verify no precipitation values appear in logs
        for val in precips_mm_monthly.flatten()[:10]:
            if not np.isnan(val):
                val_str = str(float(val))
                assert val_str not in raw_logs, f"Found data value {val_str} in error logs"

        # verify "values" field is not in any event
        events = parse_log_events(StringIO(raw_logs))
        for event in events:
            assert "values" not in event
            assert "precips_mm" not in event

    def test_no_calculation_completed_on_failure(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Calculation failure does not emit calculation_completed."""
        # mock to trigger error
        with mock.patch("scipy.stats.gamma.cdf", side_effect=ValueError("Test error")):
            with pytest.raises(compute.DistributionFittingError):
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
        completed_events = filter_events(events, "calculation_completed")
        failed_events = filter_events(events, "calculation_failed")

        # verify calculation_started present
        assert len(started_events) == 1

        # verify calculation_completed absent
        assert len(completed_events) == 0

        # verify calculation_failed present
        assert len(failed_events) == 1


class TestCalculationFailedFieldCompleteness:
    """Test that calculation_failed events contain all required fields."""

    def test_all_required_fields_present(
        self,
        log_capture,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ):
        """Calculation_failed event contains all required fields."""
        # mock to trigger error
        with mock.patch("scipy.stats.gamma.cdf", side_effect=ValueError("Test error")):
            with pytest.raises(compute.DistributionFittingError):
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
        failed_events = filter_events(events, "calculation_failed")
        assert len(failed_events) == 1

        failed = failed_events[0]

        # verify all required fields present
        required_fields = [
            "event",
            "level",
            "index_type",
            "scale",
            "distribution",
            "input_shape",
            "error_type",
            "error_message",
            "calibration_period",
            "exception",
        ]

        for field in required_fields:
            assert field in failed, f"Missing required field: {field}"

        # verify field types
        assert isinstance(failed["event"], str)
        assert failed["event"] == "calculation_failed"
        assert failed["level"] == "error"
        assert isinstance(failed["error_type"], str)
        assert isinstance(failed["error_message"], str)
        assert isinstance(failed["exception"], str)
        assert len(failed["exception"]) > 0
