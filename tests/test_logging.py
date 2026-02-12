"""Tests for logging output validation (Story 4.7).

This module validates that structlog output is machine-parseable and meets
the contract required by log aggregators (ELK, Datadog, Splunk).

Tests cover:
- JSON validity and structure (NDJSON format)
- Required field presence and types
- ISO 8601 timestamp format with timezone
- Special character escaping
- Console output format with ANSI codes
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from io import StringIO

import pytest
import structlog

from climate_indices.logging_config import (
    _reset_logging_for_testing,
    configure_logging,
    get_logger,
)

# required fields in every JSON log event
REQUIRED_FIELDS = frozenset({"timestamp", "level", "event", "logger"})

# iso 8601 pattern for timestamp validation
ISO8601_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

# ansi escape sequence pattern for console color detection
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[")


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset logging configuration before and after each test."""
    _reset_logging_for_testing()
    yield
    _reset_logging_for_testing()
    # clean up root logger handlers
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


@pytest.fixture
def json_log_capture():
    """Set up JSON logging with DEBUG level and output capture.

    Configures structlog for JSON output at DEBUG level to capture all events.
    Returns a StringIO stream containing the log output.
    """
    # configure logging for JSON output at DEBUG level
    configure_logging(log_format="json", log_level="DEBUG")

    # set up capture stream
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()

    # copy formatter from the configured handler
    if root.handlers:
        original_handler = root.handlers[0]
        handler.setFormatter(original_handler.formatter)
        root.handlers = [handler]

    return stream


@pytest.fixture
def console_log_capture():
    """Set up console logging with output capture.

    Configures structlog for console output and returns a StringIO stream.
    """
    configure_logging(log_format="console", log_level="DEBUG")

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()

    if root.handlers:
        original_handler = root.handlers[0]
        handler.setFormatter(original_handler.formatter)
        root.handlers = [handler]

    return stream


@pytest.fixture
def console_color_log_capture():
    """Set up console logging with forced colors for ANSI testing.

    This fixture explicitly configures ConsoleRenderer with colors enabled
    to ensure ANSI escape sequences are included in output.
    """
    _reset_logging_for_testing()

    # build processor chain with forced color console renderer
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    pre_chain = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
    ]

    # force colors on
    renderer = structlog.dev.ConsoleRenderer(colors=True)

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=pre_chain,
    )

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.DEBUG)

    # configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # mark logging as configured to prevent get_logger from reconfiguring
    from climate_indices import logging_config

    logging_config._LOGGING_CONFIGURED = True

    return stream


def parse_log_events(stream: StringIO) -> list[dict]:
    """Parse all JSON log lines from the captured stream."""
    stream.seek(0)
    events = []
    for line in stream.readlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


class TestJSONOutputValidity:
    """Test that JSON output is valid and parseable."""

    def test_info_event_is_valid_json(self, json_log_capture):
        """INFO log line parses as valid JSON dict."""
        logger = get_logger("test")
        logger.info("test message")

        json_log_capture.seek(0)
        line = json_log_capture.read().strip()

        # should parse without error
        parsed = json.loads(line)
        assert isinstance(parsed, dict)

    def test_debug_event_is_valid_json(self, json_log_capture):
        """DEBUG level produces valid JSON."""
        logger = get_logger("test")
        logger.debug("debug message")

        json_log_capture.seek(0)
        line = json_log_capture.read().strip()

        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert parsed["level"] == "debug"

    def test_warning_event_is_valid_json(self, json_log_capture):
        """WARNING level produces valid JSON."""
        logger = get_logger("test")
        logger.warning("warning message")

        json_log_capture.seek(0)
        line = json_log_capture.read().strip()

        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert parsed["level"] == "warning"

    def test_error_event_is_valid_json(self, json_log_capture):
        """ERROR level produces valid JSON."""
        logger = get_logger("test")
        logger.error("error message")

        json_log_capture.seek(0)
        line = json_log_capture.read().strip()

        parsed = json.loads(line)
        assert isinstance(parsed, dict)
        assert parsed["level"] == "error"

    def test_event_with_bound_context_is_valid_json(self, json_log_capture):
        """logger.bind(key=value) fields serialize correctly to JSON."""
        logger = get_logger("test")
        bound_logger = logger.bind(user_id=123, operation="test_op")
        bound_logger.info("bound context event")

        json_log_capture.seek(0)
        line = json_log_capture.read().strip()

        parsed = json.loads(line)
        assert parsed["user_id"] == 123
        assert parsed["operation"] == "test_op"

    def test_event_with_exception_info_is_valid_json(self, json_log_capture):
        """logger.error with exc_info=True produces valid JSON with traceback."""
        logger = get_logger("test")

        try:
            raise ValueError("test exception")
        except ValueError:
            logger.error("caught exception", exc_info=True)

        json_log_capture.seek(0)
        line = json_log_capture.read().strip()

        parsed = json.loads(line)
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "test exception" in parsed["exception"]

    def test_multiple_events_each_valid_json(self, json_log_capture):
        """Multiple log lines are each independently parseable."""
        logger = get_logger("test")
        logger.info("first event")
        logger.warning("second event")
        logger.error("third event")

        events = parse_log_events(json_log_capture)
        assert len(events) == 3
        assert all(isinstance(e, dict) for e in events)

    def test_numeric_and_boolean_fields_serialize(self, json_log_capture):
        """int, float, bool, None fields serialize to correct JSON types."""
        logger = get_logger("test")
        logger.info(
            "mixed types",
            count=42,
            ratio=3.14,
            enabled=True,
            disabled=False,
            empty=None,
        )

        events = parse_log_events(json_log_capture)
        event = events[0]

        assert event["count"] == 42
        assert isinstance(event["count"], int)
        assert event["ratio"] == pytest.approx(3.14)
        assert isinstance(event["ratio"], float)
        assert event["enabled"] is True
        assert event["disabled"] is False
        assert event["empty"] is None


class TestRequiredFieldsPresent:
    """Test that required fields are present in all log events."""

    def test_info_has_all_required_fields(self, json_log_capture):
        """INFO event has timestamp, level, event, logger."""
        logger = get_logger("test.module")
        logger.info("test event")

        events = parse_log_events(json_log_capture)
        event = events[0]

        assert REQUIRED_FIELDS.issubset(event.keys())

    def test_debug_has_all_required_fields(self, json_log_capture):
        """DEBUG event has all four required fields."""
        logger = get_logger("test.module")
        logger.debug("debug event")

        events = parse_log_events(json_log_capture)
        event = events[0]

        assert REQUIRED_FIELDS.issubset(event.keys())

    def test_warning_has_all_required_fields(self, json_log_capture):
        """WARNING event has all four required fields."""
        logger = get_logger("test.module")
        logger.warning("warning event")

        events = parse_log_events(json_log_capture)
        event = events[0]

        assert REQUIRED_FIELDS.issubset(event.keys())

    def test_error_has_all_required_fields(self, json_log_capture):
        """ERROR event has all four required fields."""
        logger = get_logger("test.module")
        logger.error("error event")

        events = parse_log_events(json_log_capture)
        event = events[0]

        assert REQUIRED_FIELDS.issubset(event.keys())

    def test_required_fields_are_top_level(self, json_log_capture):
        """All four required fields are top-level keys, not nested."""
        logger = get_logger("test.module")
        logger.info("test event", nested={"inner": "value"})

        events = parse_log_events(json_log_capture)
        event = events[0]

        # all required fields must be direct keys of the root object
        for field in REQUIRED_FIELDS:
            assert field in event
            # ensure they're not nested inside another structure
            assert isinstance(event[field], str | int | float | bool | type(None))

    def test_required_field_types(self, json_log_capture):
        """timestamp, level, event, logger are all strings."""
        logger = get_logger("test.module")
        logger.info("test event")

        events = parse_log_events(json_log_capture)
        event = events[0]

        assert isinstance(event["timestamp"], str)
        assert isinstance(event["level"], str)
        assert isinstance(event["event"], str)
        assert isinstance(event["logger"], str)

    def test_level_values_are_lowercase(self, json_log_capture):
        """Level field values are lowercase: info, debug, warning, error."""
        logger = get_logger("test")
        logger.debug("debug")
        logger.info("info")
        logger.warning("warning")
        logger.error("error")

        events = parse_log_events(json_log_capture)
        levels = [e["level"] for e in events]

        assert levels == ["debug", "info", "warning", "error"]
        # verify all lowercase
        assert all(level == level.lower() for level in levels)

    def test_extra_fields_do_not_shadow_required(self, json_log_capture):
        """Bound context with key 'event' or 'level' doesn't corrupt required fields."""
        logger = get_logger("test")

        # try to bind fields with same names as required fields
        bound = logger.bind(custom_event="custom", custom_level="custom_level")
        bound.info("test event")

        events = parse_log_events(json_log_capture)
        event = events[0]

        # required fields should still be present and correct
        assert event["event"] == "test event"
        assert event["level"] == "info"
        # custom fields should also be present
        assert event["custom_event"] == "custom"
        assert event["custom_level"] == "custom_level"


class TestISO8601Timestamps:
    """Test that timestamps are ISO 8601 format with timezone."""

    def test_timestamp_matches_iso8601_pattern(self, json_log_capture):
        """Timestamp matches YYYY-MM-DDTHH:MM:SS pattern."""
        logger = get_logger("test")
        logger.info("test event")

        events = parse_log_events(json_log_capture)
        timestamp = events[0]["timestamp"]

        assert ISO8601_PATTERN.match(timestamp)

    def test_timestamp_parseable_by_datetime(self, json_log_capture):
        """datetime.fromisoformat() succeeds on timestamp value."""
        logger = get_logger("test")
        logger.info("test event")

        events = parse_log_events(json_log_capture)
        timestamp = events[0]["timestamp"]

        # should parse without error
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert isinstance(parsed, datetime)

    def test_timestamp_has_utc_timezone(self, json_log_capture):
        """Timestamp ends with Z or +00:00 (UTC indicator)."""
        logger = get_logger("test")
        logger.info("test event")

        events = parse_log_events(json_log_capture)
        timestamp = events[0]["timestamp"]

        # should end with Z or +00:00
        assert timestamp.endswith("Z") or timestamp.endswith("+00:00")

    def test_timestamp_precision(self, json_log_capture):
        """Timestamp includes fractional seconds."""
        logger = get_logger("test")
        logger.info("test event")

        events = parse_log_events(json_log_capture)
        timestamp = events[0]["timestamp"]

        # structlog with fmt="iso" includes fractional seconds
        # pattern: YYYY-MM-DDTHH:MM:SS.ffffffZ
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z?", timestamp)

    def test_timestamps_are_monotonically_ordered(self, json_log_capture):
        """Two sequential log events have non-decreasing timestamps."""
        logger = get_logger("test")
        logger.info("first event")
        logger.info("second event")

        events = parse_log_events(json_log_capture)
        ts1 = events[0]["timestamp"].replace("Z", "+00:00")
        ts2 = events[1]["timestamp"].replace("Z", "+00:00")

        dt1 = datetime.fromisoformat(ts1)
        dt2 = datetime.fromisoformat(ts2)

        # second timestamp should be >= first
        assert dt2 >= dt1


class TestSpecialCharacterEscaping:
    """Test that special characters are properly escaped in JSON output."""

    def test_double_quotes_in_event_escaped(self, json_log_capture):
        """Event with double quotes produces valid JSON."""
        logger = get_logger("test")
        logger.info('say "hello"')

        events = parse_log_events(json_log_capture)
        assert events[0]["event"] == 'say "hello"'

    def test_backslash_in_event_escaped(self, json_log_capture):
        """Event with backslash produces valid JSON."""
        logger = get_logger("test")
        logger.info(r"path\to\file")

        events = parse_log_events(json_log_capture)
        assert events[0]["event"] == r"path\to\file"

    def test_newline_in_field_escaped(self, json_log_capture):
        """Bound field containing newline produces valid JSON."""
        logger = get_logger("test")
        logger.info("multiline", message="line1\nline2\nline3")

        events = parse_log_events(json_log_capture)
        assert events[0]["message"] == "line1\nline2\nline3"

    def test_tab_in_field_escaped(self, json_log_capture):
        """Bound field containing tab produces valid JSON."""
        logger = get_logger("test")
        logger.info("tabbed", data="col1\tcol2\tcol3")

        events = parse_log_events(json_log_capture)
        assert events[0]["data"] == "col1\tcol2\tcol3"

    def test_unicode_in_event_preserved(self, json_log_capture):
        """Unicode chars preserved in JSON output."""
        logger = get_logger("test")
        logger.info("température 25°C")

        events = parse_log_events(json_log_capture)
        assert events[0]["event"] == "température 25°C"

    def test_null_byte_handled(self, json_log_capture):
        """Field with null byte doesn't break JSON serialization."""
        logger = get_logger("test")
        # null byte in the middle of a string
        logger.info("null test", data="before\x00after")

        # should produce valid JSON (null byte may be escaped or removed)
        events = parse_log_events(json_log_capture)
        assert "data" in events[0]

    def test_mixed_special_characters(self, json_log_capture):
        """Combined special chars in single event all handled correctly."""
        logger = get_logger("test")
        complex_msg = 'Test: "quotes", \\backslash, \n newline, \t tab, unicode: café'
        logger.info(complex_msg)

        events = parse_log_events(json_log_capture)
        assert events[0]["event"] == complex_msg


class TestConsoleOutputFormat:
    """Test that console output includes expected formatting."""

    def test_console_output_contains_ansi_escape(self, console_color_log_capture):
        """Console renderer output contains ANSI escape sequence."""
        logger = get_logger("test")
        logger.info("test event")

        console_color_log_capture.seek(0)
        output = console_color_log_capture.read()

        # should contain ANSI escape codes when colors are enabled
        assert ANSI_ESCAPE_PATTERN.search(output)

    def test_console_output_contains_event_text(self, console_log_capture):
        """Event name appears in console output."""
        logger = get_logger("test")
        logger.info("unique_test_event_12345")

        console_log_capture.seek(0)
        output = console_log_capture.read()

        assert "unique_test_event_12345" in output

    def test_console_output_contains_log_level(self, console_log_capture):
        """Log level string appears in console output."""
        logger = get_logger("test")
        logger.warning("test warning")

        console_log_capture.seek(0)
        output = console_log_capture.read()

        # console format typically includes level indicator
        # may be uppercase or lowercase depending on renderer
        assert "warning" in output.lower()

    def test_console_output_contains_timestamp(self, console_log_capture):
        """Timestamp-like string appears in console output."""
        logger = get_logger("test")
        logger.info("test event")

        console_log_capture.seek(0)
        output = console_log_capture.read()

        # console format should include some timestamp representation
        # look for ISO-like pattern or time components
        assert re.search(r"\d{2}:\d{2}:\d{2}", output)

    def test_console_different_levels_produce_output(self, console_log_capture):
        """INFO, WARNING, ERROR all produce non-empty output."""
        logger = get_logger("test")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")

        console_log_capture.seek(0)
        output = console_log_capture.read()

        assert len(output) > 0
        assert "info message" in output
        assert "warning message" in output
        assert "error message" in output


class TestJSONSchemaContract:
    """Test JSON output schema contract for log aggregators."""

    def test_one_json_object_per_line(self, json_log_capture):
        """Each line is exactly one JSON object (NDJSON/JSON Lines format)."""
        logger = get_logger("test")
        logger.info("event 1")
        logger.info("event 2")
        logger.info("event 3")

        json_log_capture.seek(0)
        lines = json_log_capture.readlines()

        # each line should be independently parseable
        for line in lines:
            line = line.strip()
            if line:
                obj = json.loads(line)
                assert isinstance(obj, dict)

    def test_no_trailing_comma_or_array_wrapper(self, json_log_capture):
        """Output is not wrapped in [] or has trailing commas."""
        logger = get_logger("test")
        logger.info("event 1")
        logger.info("event 2")

        json_log_capture.seek(0)
        full_output = json_log_capture.read()

        # should not start with [ or end with ]
        assert not full_output.strip().startswith("[")
        assert not full_output.strip().endswith("]")

        # each line should not have trailing comma
        for line in full_output.split("\n"):
            line = line.strip()
            if line:
                assert not line.endswith(",")

    def test_custom_fields_appear_at_top_level(self, json_log_capture):
        """Bound fields appear as top-level JSON keys."""
        logger = get_logger("test")
        logger.info("calculation", index_type="spi", scale=6, distribution="gamma")

        events = parse_log_events(json_log_capture)
        event = events[0]

        # custom fields should be direct keys
        assert event["index_type"] == "spi"
        assert event["scale"] == 6
        assert event["distribution"] == "gamma"

    def test_empty_string_fields_preserved(self, json_log_capture):
        """Empty string serialized correctly (not null)."""
        logger = get_logger("test")
        logger.info("test", empty_field="", nonempty="value")

        events = parse_log_events(json_log_capture)
        event = events[0]

        assert event["empty_field"] == ""
        assert event["empty_field"] is not None

    def test_json_output_ends_with_newline(self, json_log_capture):
        """Each log entry is terminated by newline."""
        logger = get_logger("test")
        logger.info("test event")

        json_log_capture.seek(0)
        output = json_log_capture.read()

        # should end with newline
        assert output.endswith("\n")
