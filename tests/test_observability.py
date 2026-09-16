"""Observability tests: logging configuration and calculation lifecycle events.

Consolidated from ``test_logging.py``, ``test_logging_config.py``,
``test_calculation_event_logging.py``, and ``test_error_context_logging.py``
(issue #915).  The calculation lifecycle contract is exercised once per public
index through the public API; JSON/console rendering, level resolution, and
failure context are covered by the shared cases in this module.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from io import StringIO
from unittest import mock

import numpy as np
import pytest

from climate_indices import compute, indices, palmer
from climate_indices.eto import eto_hargreaves
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.logging_config import (
    ENV_LOG_LEVEL,
    _reset_logging_for_testing,
    configure_logging,
    get_logger,
)

# fields every JSON log event must carry as top-level keys
REQUIRED_JSON_FIELDS = frozenset({"timestamp", "level", "event", "logger"})

# one NOAA climate division's monthly precipitation/PET, for the Palmer lifecycle
_PALMER_DIVISION_DIR = os.path.join(os.path.dirname(__file__), "fixture", "palmer", "0101")


@pytest.fixture(autouse=True)
def _clean_logging_state() -> None:
    """Reset structlog and root-logger state around every test in this module."""
    _reset_logging_for_testing()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    os.environ.pop(ENV_LOG_LEVEL, None)
    yield
    _reset_logging_for_testing()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    os.environ.pop(ENV_LOG_LEVEL, None)


@pytest.fixture(scope="module")
def palmer_division_precip_pet() -> tuple[np.ndarray, np.ndarray]:
    """Monthly precipitation and PET for one Palmer reference division."""
    return (
        np.load(os.path.join(_PALMER_DIVISION_DIR, "precips.npy")),
        np.load(os.path.join(_PALMER_DIVISION_DIR, "pet.npy")),
    )


def _capture_stream(log_format: str = "json", log_level: str = "DEBUG") -> StringIO:
    """Configure logging and route its formatted output into a fresh stream."""
    configure_logging(log_format=log_format, log_level=log_level)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    if root.handlers:
        handler.setFormatter(root.handlers[0].formatter)
    root.handlers = [handler]
    return stream


def _parse_events(stream: StringIO) -> list[dict]:
    """Parse each non-empty JSON line in the captured stream."""
    stream.seek(0)
    return [json.loads(line) for line in stream.read().splitlines() if line.strip()]


def _events_named(stream: StringIO, name: str) -> list[dict]:
    """Return the captured events whose ``event`` field equals ``name``."""
    return [event for event in _parse_events(stream) if event.get("event") == name]


class TestLogLevelResolution:
    """configure_logging resolves the level from argument, environment, or default."""

    @pytest.mark.parametrize(
        ("explicit", "env_value", "expected"),
        [
            (None, None, logging.INFO),
            ("debug", None, logging.DEBUG),
            ("INVALID", None, logging.INFO),
            (None, "warning", logging.WARNING),
            (None, "", logging.INFO),
            ("INFO", "ERROR", logging.INFO),
        ],
    )
    def test_level_resolution(self, explicit: str | None, env_value: str | None, expected: int) -> None:
        """Explicit level wins over the environment, invalid values fall back to INFO."""
        if env_value is not None:
            os.environ[ENV_LOG_LEVEL] = env_value
        if explicit is None:
            configure_logging()
        else:
            configure_logging(log_level=explicit)
        assert logging.getLogger().level == expected

    def test_level_filter_suppresses_below_threshold(self) -> None:
        """Events below the configured level are filtered out."""
        stream = _capture_stream(log_level="INFO")
        logger = get_logger("test.filter")
        logger.debug("debug event")
        logger.info("info event")

        assert "debug event" not in stream.getvalue()
        assert "info event" in stream.getvalue()

    def test_second_configure_call_is_noop(self) -> None:
        """The first configuration wins; later calls do not change the level."""
        configure_logging(log_level="INFO")
        configure_logging(log_level="DEBUG")
        assert logging.getLogger().level == logging.INFO

    def test_default_configuration_uses_a_single_stream_handler(self) -> None:
        """Default setup installs one StreamHandler and no FileHandler."""
        configure_logging()
        handlers = logging.getLogger().handlers
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)
        assert not isinstance(handlers[0], logging.FileHandler)


class TestJSONOutput:
    """JSON rendering meets the machine-parseable contract log aggregators need."""

    def test_event_carries_required_top_level_fields(self) -> None:
        """Required fields and bound context appear at the top level."""
        stream = _capture_stream()
        get_logger("test.module").bind(count=42).info("test event")

        event = _parse_events(stream)[0]
        assert REQUIRED_JSON_FIELDS.issubset(event)
        assert event["event"] == "test event"
        assert event["level"] == "info"
        assert event["logger"] == "test.module"
        assert event["count"] == 42

    def test_timestamp_is_utc_iso8601(self) -> None:
        """Timestamps parse as ISO 8601 with a UTC offset."""
        stream = _capture_stream()
        get_logger("test").info("test event")

        timestamp = _parse_events(stream)[0]["timestamp"]
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed.utcoffset() == timedelta(0)

    def test_output_is_one_json_object_per_line(self) -> None:
        """Each log line is an independently parseable JSON object (NDJSON)."""
        stream = _capture_stream()
        logger = get_logger("test")
        for message in ("first", "second", "third"):
            logger.info(message)

        assert [event["event"] for event in _parse_events(stream)] == ["first", "second", "third"]

    def test_special_characters_round_trip(self) -> None:
        """Quotes, backslashes, newlines, tabs, and unicode survive serialization."""
        stream = _capture_stream()
        message = 'quotes "x", path\\to, newline\nand tab\t + température 25°C'
        get_logger("test").info(message)

        assert _parse_events(stream)[0]["event"] == message

    def test_exception_traceback_is_included(self) -> None:
        """exc_info=True renders the exception type and message into the event."""
        stream = _capture_stream()
        logger = get_logger("test")
        try:
            raise ValueError("test exception")
        except ValueError:
            logger.error("caught exception", exc_info=True)

        exception = _parse_events(stream)[0]["exception"]
        assert "ValueError" in exception
        assert "test exception" in exception


class TestConsoleOutput:
    """Console rendering stays human-readable."""

    def test_event_text_and_level_appear(self) -> None:
        """The event text and level appear in console output."""
        stream = _capture_stream(log_format="console", log_level="INFO")
        get_logger("test").warning("console warning text")

        output = stream.getvalue()
        assert "console warning text" in output
        assert "warning" in output.lower()


class TestCalculationLifecycle:
    """Every public index emits exactly one started and one completed event."""

    def test_spi(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
        result = indices.spi(
            values=precips_mm_monthly,
            scale=6,
            distribution=indices.Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        started = _events_named(stream, "calculation_started")
        completed = _events_named(stream, "calculation_completed")
        assert len(started) == 1
        assert len(completed) == 1
        assert started[0]["index_type"] == "spi"
        assert started[0]["scale"] == 6
        assert started[0]["distribution"] == "gamma"
        assert "input_shape" in started[0]
        assert completed[0]["duration_ms"] > 0
        assert tuple(completed[0]["output_shape"]) == result.shape

    def test_spei(
        self,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
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

        started = _events_named(stream, "calculation_started")
        completed = _events_named(stream, "calculation_completed")
        assert len(started) == 1
        assert len(completed) == 1
        assert started[0]["index_type"] == "spei"
        assert started[0]["scale"] == 6
        assert started[0]["distribution"] == "gamma"
        assert "input_shape" in started[0]
        assert completed[0]["duration_ms"] > 0
        assert tuple(completed[0]["output_shape"]) == result.shape

    def test_percentage_of_normal(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
        result = indices.percentage_of_normal(
            values=precips_mm_monthly.flatten(),
            scale=6,
            data_start_year=data_year_start_monthly,
            calibration_start_year=calibration_year_start_monthly,
            calibration_end_year=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )

        started = _events_named(stream, "calculation_started")
        completed = _events_named(stream, "calculation_completed")
        assert len(started) == 1
        assert len(completed) == 1
        assert started[0]["index_type"] == "percentage_of_normal"
        assert started[0]["scale"] == 6
        assert "distribution" not in started[0]
        assert "input_shape" in started[0]
        assert completed[0]["duration_ms"] > 0
        assert tuple(completed[0]["output_shape"]) == result.shape

    def test_pet_thornthwaite(
        self,
        temps_celsius,
        latitude_degrees,
        data_year_start_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
        result = indices.pet(
            temperature_celsius=temps_celsius,
            latitude_degrees=latitude_degrees,
            data_start_year=data_year_start_monthly,
        )

        started = _events_named(stream, "calculation_started")
        completed = _events_named(stream, "calculation_completed")
        assert len(started) == 1
        assert len(completed) == 1
        assert started[0]["index_type"] == "pet_thornthwaite"
        assert "scale" not in started[0]
        assert "distribution" not in started[0]
        assert "input_shape" in started[0]
        assert completed[0]["duration_ms"] > 0
        assert tuple(completed[0]["output_shape"]) == result.shape

    def test_pci(self, rain_mm_366) -> None:
        stream = _capture_stream(log_level="INFO")
        indices.pci(rainfall_mm=rain_mm_366.flatten())

        started = _events_named(stream, "calculation_started")
        completed = _events_named(stream, "calculation_completed")
        assert len(started) == 1
        assert len(completed) == 1
        assert started[0]["index_type"] == "pci"
        assert "scale" not in started[0]
        assert "distribution" not in started[0]
        assert "input_shape" in started[0]
        assert completed[0]["duration_ms"] > 0
        assert "output_shape" in completed[0]

    def test_eto_hargreaves(
        self,
        hargreaves_daily_tmin_celsius,
        hargreaves_daily_tmax_celsius,
        hargreaves_daily_tmean_celsius,
        hargreaves_latitude_degrees,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
        result = eto_hargreaves(
            daily_tmin_celsius=hargreaves_daily_tmin_celsius,
            daily_tmax_celsius=hargreaves_daily_tmax_celsius,
            daily_tmean_celsius=hargreaves_daily_tmean_celsius,
            latitude_degrees=hargreaves_latitude_degrees,
        )

        started = _events_named(stream, "calculation_started")
        completed = _events_named(stream, "calculation_completed")
        assert len(started) == 1
        assert len(completed) == 1
        assert started[0]["index_type"] == "pet_hargreaves"
        assert "input_shape" in started[0]
        assert completed[0]["duration_ms"] > 0
        assert tuple(completed[0]["output_shape"]) == result.shape

    def test_pdsi(
        self,
        palmer_division_precip_pet,
        data_year_start_monthly,
        calibration_year_start_palmer,
        calibration_year_end_palmer,
    ) -> None:
        precips, pet = palmer_division_precip_pet
        stream = _capture_stream(log_level="INFO")
        result = palmer.pdsi(
            precips,
            pet,
            4.5,
            data_year_start_monthly,
            calibration_year_start_palmer,
            calibration_year_end_palmer,
        )

        started = _events_named(stream, "calculation_started")
        completed = _events_named(stream, "calculation_completed")
        assert len(started) == 1
        assert len(completed) == 1
        assert started[0]["index_type"] == "pdsi"
        assert tuple(started[0]["input_shape"]) == precips.shape
        assert completed[0]["duration_ms"] > 0
        assert completed[0]["output_elements"] == result[0].size


class TestCalculationFailureContext:
    """Every public index emits one calculation_failed event carrying context."""

    def test_spi_gamma_failure(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
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

        failed = _events_named(stream, "calculation_failed")
        assert len(failed) == 1
        event = failed[0]
        assert event["level"] == "error"
        assert event["index_type"] == "spi"
        assert event["scale"] == 6
        assert event["distribution"] == "gamma"
        assert "input_shape" in event
        assert event["error_type"] == "DistributionFittingError"
        assert "CDF computation failed" in event["error_message"]
        assert event["calibration_period"] == f"{calibration_year_start_monthly}-{calibration_year_end_monthly}"
        assert "CDF computation failed" in event["exception"]

    def test_spei_gamma_failure(
        self,
        precips_mm_monthly,
        pet_thornthwaite_mm,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
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

        failed = _events_named(stream, "calculation_failed")
        assert len(failed) == 1
        event = failed[0]
        assert event["index_type"] == "spei"
        assert event["scale"] == 6
        assert event["distribution"] == "gamma"
        assert event["error_type"] == "DistributionFittingError"
        assert "Gamma CDF failed" in event["error_message"]
        assert event["calibration_period"] == f"{calibration_year_start_monthly}-{calibration_year_end_monthly}"

    def test_spei_incompatible_arrays(
        self,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
        with pytest.raises(ValueError, match="Incompatible"):
            indices.spei(
                precips_mm=np.random.rand(240),
                pet_mm=np.random.rand(120),
                scale=3,
                distribution=indices.Distribution.gamma,
                periodicity=compute.Periodicity.monthly,
                data_start_year=data_year_start_monthly,
                calibration_year_initial=calibration_year_start_monthly,
                calibration_year_final=calibration_year_end_monthly,
            )

        failed = _events_named(stream, "calculation_failed")
        assert len(failed) == 1
        assert failed[0]["index_type"] == "spei"
        assert failed[0]["error_type"] == "ValueError"

    def test_percentage_of_normal_invalid_calibration(self) -> None:
        stream = _capture_stream(log_level="INFO")
        with pytest.raises(InvalidArgumentError, match="calibration start year"):
            indices.percentage_of_normal(
                values=np.random.rand(240),
                scale=6,
                data_start_year=2010,
                calibration_start_year=2000,
                calibration_end_year=2019,
                periodicity=compute.Periodicity.monthly,
            )

        failed = _events_named(stream, "calculation_failed")
        assert len(failed) == 1
        event = failed[0]
        assert event["index_type"] == "percentage_of_normal"
        assert event["scale"] == 6
        assert "distribution" not in event
        assert event["error_type"] == "InvalidArgumentError"
        assert event["calibration_period"] == "2000-2019"

    def test_pet_invalid_latitude(self) -> None:
        stream = _capture_stream(log_level="INFO")
        with pytest.raises(ValueError, match="Invalid latitude"):
            indices.pet(
                temperature_celsius=np.random.rand(240),
                latitude_degrees=100.0,
                data_start_year=2000,
            )

        failed = _events_named(stream, "calculation_failed")
        assert len(failed) == 1
        event = failed[0]
        assert event["index_type"] == "pet_thornthwaite"
        assert "scale" not in event
        assert "distribution" not in event
        assert "calibration_period" not in event
        assert event["error_type"] == "ValueError"
        assert "Invalid latitude" in event["error_message"]

    def test_pci_invalid_length(self) -> None:
        stream = _capture_stream(log_level="INFO")
        with pytest.raises(InvalidArgumentError, match="365 or 366"):
            indices.pci(rainfall_mm=np.random.rand(100))

        failed = _events_named(stream, "calculation_failed")
        assert len(failed) == 1
        event = failed[0]
        assert event["index_type"] == "pci"
        assert "scale" not in event
        assert "distribution" not in event
        assert "calibration_period" not in event
        assert event["error_type"] == "InvalidArgumentError"

    def test_eto_hargreaves_computation_error(self) -> None:
        stream = _capture_stream(log_level="INFO")
        with mock.patch("climate_indices.utils.reshape_to_2d", side_effect=RuntimeError("Reshape failed")):
            with pytest.raises(RuntimeError, match="Reshape failed"):
                eto_hargreaves(
                    daily_tmin_celsius=np.random.rand(366),
                    daily_tmax_celsius=np.random.rand(366),
                    daily_tmean_celsius=np.random.rand(366),
                    latitude_degrees=40.0,
                )

        failed = _events_named(stream, "calculation_failed")
        assert len(failed) == 1
        event = failed[0]
        assert event["index_type"] == "pet_hargreaves"
        assert event["error_type"] == "RuntimeError"
        assert "Reshape failed" in event["error_message"]


class TestFailureLifecycle:
    """Failure paths still honor the lifecycle contract and never leak data."""

    def test_failure_does_not_emit_completed(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
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

        assert len(_events_named(stream, "calculation_started")) == 1
        assert len(_events_named(stream, "calculation_failed")) == 1
        assert len(_events_named(stream, "calculation_completed")) == 0

    def test_pearson_fallback_success_emits_no_failure(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
        with mock.patch(
            "climate_indices.compute.transform_fitted_pearson",
            side_effect=compute.DistributionFittingError("Pearson failed"),
        ):
            result = indices.spi(
                values=precips_mm_monthly,
                scale=3,
                distribution=indices.Distribution.pearson,
                data_start_year=data_year_start_monthly,
                calibration_year_initial=calibration_year_start_monthly,
                calibration_year_final=calibration_year_end_monthly,
                periodicity=compute.Periodicity.monthly,
            )

        assert result.size == precips_mm_monthly.size
        assert len(_events_named(stream, "calculation_failed")) == 0
        assert len(_events_named(stream, "calculation_completed")) == 1

    def test_logs_never_contain_input_values(
        self,
        precips_mm_monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    ) -> None:
        stream = _capture_stream(log_level="INFO")
        indices.spi(
            values=precips_mm_monthly,
            scale=6,
            distribution=indices.Distribution.gamma,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )
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

        raw_logs = stream.getvalue()
        for value in precips_mm_monthly.flatten()[:10]:
            if not np.isnan(value):
                assert str(float(value)) not in raw_logs, f"Found data value {float(value)} in logs"
        for event in _parse_events(stream):
            assert "values" not in event
            assert "precips_mm" not in event
            assert "temperature_celsius" not in event
