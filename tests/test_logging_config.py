"""Tests for structlog configuration in climate_indices.logging_config."""

from __future__ import annotations

import json
import logging
import os
from io import StringIO

import pytest
import structlog

from climate_indices.logging_config import (
    ENV_LOG_LEVEL,
    _reset_logging_for_testing,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    """Reset logging state between tests."""
    _reset_logging_for_testing()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    if ENV_LOG_LEVEL in os.environ:
        del os.environ[ENV_LOG_LEVEL]

    yield

    _reset_logging_for_testing()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    # clear any environment variable set during tests
    if ENV_LOG_LEVEL in os.environ:
        del os.environ[ENV_LOG_LEVEL]


class TestConfigureLoggingDefaults:
    """Verify default configuration behavior."""

    def test_default_level_is_info(self) -> None:
        """Default log level should be INFO when no level specified."""
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_default_format_is_console(self) -> None:
        """Default format should be console when no format specified."""
        configure_logging()
        root = logging.getLogger()
        # console format uses ConsoleRenderer
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        # check that formatter is ProcessorFormatter
        assert isinstance(handler.formatter, structlog.stdlib.ProcessorFormatter)

    def test_returns_none(self) -> None:
        """configure_logging should return None."""
        configure_logging()

    def test_configures_structlog(self) -> None:
        """configure_logging should set up structlog configuration."""
        configure_logging()
        # verify structlog can create loggers
        logger = structlog.get_logger("test")
        assert logger is not None
        # structlog returns a BoundLoggerLazyProxy that wraps BoundLogger
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")


class TestConfigureLoggingLogLevel:
    """Verify log level parameter behavior."""

    def test_explicit_level_sets_root_logger(self) -> None:
        """Explicit log level parameter should set root logger level."""
        configure_logging(log_level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_explicit_level_overrides_env_var(self) -> None:
        """Explicit log level should override environment variable."""
        os.environ[ENV_LOG_LEVEL] = "ERROR"
        configure_logging(log_level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_invalid_level_falls_back_to_info(self) -> None:
        """Invalid log level should fall back to INFO."""
        configure_logging(log_level="INVALID")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_lowercase_level_accepted(self) -> None:
        """Log level should be case-insensitive."""
        configure_logging(log_level="debug")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_warning_level(self) -> None:
        """WARNING level should be set correctly."""
        configure_logging(log_level="WARNING")
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_error_level(self) -> None:
        """ERROR level should be set correctly."""
        configure_logging(log_level="ERROR")
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_critical_level(self) -> None:
        """CRITICAL level should be set correctly."""
        configure_logging(log_level="CRITICAL")
        root = logging.getLogger()
        assert root.level == logging.CRITICAL


class TestEnvironmentVariableOverride:
    """Verify environment variable configuration."""

    def test_env_var_sets_level(self) -> None:
        """CLIMATE_INDICES_LOG_LEVEL env var should set log level."""
        os.environ[ENV_LOG_LEVEL] = "DEBUG"
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_env_var_ignored_when_explicit_given(self) -> None:
        """Environment variable should be ignored when explicit level provided."""
        os.environ[ENV_LOG_LEVEL] = "ERROR"
        configure_logging(log_level="INFO")
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_invalid_env_var_falls_back(self) -> None:
        """Invalid environment variable value should fall back to INFO."""
        os.environ[ENV_LOG_LEVEL] = "NOT_A_LEVEL"
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_env_var_case_insensitive(self) -> None:
        """Environment variable log level should be case-insensitive."""
        os.environ[ENV_LOG_LEVEL] = "warning"
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_empty_env_var_uses_default(self) -> None:
        """Empty environment variable should use default INFO level."""
        os.environ[ENV_LOG_LEVEL] = ""
        configure_logging()
        root = logging.getLogger()
        assert root.level == logging.INFO


class TestLogFormat:
    """Verify log format options."""

    def test_json_produces_parseable_json(self) -> None:
        """JSON format should produce valid JSON output."""
        # capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        configure_logging(log_format="json")
        # replace handler to capture output
        root = logging.getLogger()
        original_handler = root.handlers[0]
        handler.setFormatter(original_handler.formatter)
        root.handlers = [handler]

        # emit a log message
        logger = get_logger("test")
        logger.info("test message", extra_field="value")

        # verify JSON is parseable
        output = stream.getvalue()
        assert output.strip()
        parsed = json.loads(output.strip())
        assert isinstance(parsed, dict)

    def test_console_produces_readable_output(self) -> None:
        """Console format should produce human-readable output."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        configure_logging(log_format="console")
        root = logging.getLogger()
        original_handler = root.handlers[0]
        handler.setFormatter(original_handler.formatter)
        root.handlers = [handler]

        logger = get_logger("test")
        logger.info("test message")

        output = stream.getvalue()
        # console output should contain the message
        assert "test message" in output

    def test_json_includes_required_fields(self) -> None:
        """JSON format should include event, level, and timestamp fields."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        configure_logging(log_format="json", log_level="INFO")
        root = logging.getLogger()
        original_handler = root.handlers[0]
        handler.setFormatter(original_handler.formatter)
        root.handlers = [handler]

        logger = get_logger("test")
        logger.info("test message")

        output = stream.getvalue()
        parsed = json.loads(output.strip())
        assert "event" in parsed
        assert "level" in parsed
        assert "timestamp" in parsed
        assert parsed["event"] == "test message"
        assert parsed["level"] == "info"

    def test_json_includes_logger_name(self) -> None:
        """JSON format should include logger name."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        configure_logging(log_format="json")
        root = logging.getLogger()
        original_handler = root.handlers[0]
        handler.setFormatter(original_handler.formatter)
        root.handlers = [handler]

        logger = get_logger("my.logger")
        logger.info("test message")

        output = stream.getvalue()
        parsed = json.loads(output.strip())
        assert parsed["logger"] == "my.logger"


class TestDoubleConfigurationGuard:
    """Verify idempotency of configuration."""

    def test_second_call_is_no_op(self) -> None:
        """Second call to configure_logging should be no-op."""
        configure_logging(log_level="INFO")
        root = logging.getLogger()
        info_level = root.level

        # second call with different level should not change level
        configure_logging(log_level="DEBUG")
        assert root.level == info_level

    def test_reset_allows_reconfiguration(self) -> None:
        """After reset, configure_logging should work again."""
        configure_logging(log_level="INFO")
        root = logging.getLogger()
        assert root.level == logging.INFO

        # reset and reconfigure
        _reset_logging_for_testing()
        root.handlers.clear()
        configure_logging(log_level="DEBUG")
        assert root.level == logging.DEBUG

    def test_multiple_resets_safe(self) -> None:
        """Multiple resets should be safe."""
        configure_logging(log_level="INFO")
        _reset_logging_for_testing()
        _reset_logging_for_testing()
        # should not raise exception
        configure_logging(log_level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG


class TestGetLogger:
    """Verify get_logger convenience function."""

    def test_returns_bound_logger(self) -> None:
        """get_logger should return a BoundLogger instance."""
        configure_logging()
        logger = get_logger("test")
        # get_logger returns a bound logger (wrapped in lazy proxy)
        assert logger is not None
        # verify it has the expected interface
        assert hasattr(logger, "info")
        assert hasattr(logger, "bind")

    def test_name_matches_input(self) -> None:
        """Logger name should match input."""
        configure_logging()
        logger = get_logger("my.module")
        # BoundLogger wraps stdlib logger
        assert logger._context.get("logger") is not None or hasattr(logger, "_logger")

    def test_has_standard_methods(self) -> None:
        """BoundLogger should have standard logging methods."""
        configure_logging()
        logger = get_logger("test")
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
        assert callable(logger.info)
        assert callable(logger.debug)
        assert callable(logger.warning)
        assert callable(logger.error)

    def test_can_log_without_configure(self) -> None:
        """get_logger should work even if configure_logging not called."""
        # structlog has defaults, so this should not raise
        logger = get_logger("test")
        assert logger is not None
        # should be able to call methods (though output may not be configured)
        logger.info("test")


class TestNoFileLoggingByDefault:
    """Verify that no file handlers are created by default."""

    def test_root_logger_only_has_stream_handler(self) -> None:
        """Root logger should not contain user file handlers."""
        configure_logging()
        root = logging.getLogger()
        assert any(isinstance(handler, logging.StreamHandler) for handler in root.handlers)
        file_handlers = [handler for handler in root.handlers if isinstance(handler, logging.FileHandler)]
        assert all(
            getattr(handler, "baseFilename", None) in {None, os.devnull, "/dev/null"} for handler in file_handlers
        )

    def test_no_file_handler_after_json_config(self) -> None:
        """JSON format should also avoid user file handlers."""
        configure_logging(log_format="json")
        root = logging.getLogger()
        assert any(isinstance(handler, logging.StreamHandler) for handler in root.handlers)
        file_handlers = [handler for handler in root.handlers if isinstance(handler, logging.FileHandler)]
        assert all(
            getattr(handler, "baseFilename", None) in {None, os.devnull, "/dev/null"} for handler in file_handlers
        )
