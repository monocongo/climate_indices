"""Structlog configuration for climate_indices library."""

from __future__ import annotations

import logging
import os
from typing import Literal, cast

import structlog
from structlog.typing import Processor

# environment variable for log level configuration
ENV_LOG_LEVEL = "CLIMATE_INDICES_LOG_LEVEL"

# valid log level names
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# default log level when not specified
DEFAULT_LOG_LEVEL = "INFO"

# log format options
LogFormat = Literal["console", "json"]

# sentinel for distinguishing "no value provided" from "value provided"
_UNSET = object()

# guard against double-configuration
_LOGGING_CONFIGURED = False


def configure_logging(log_format: LogFormat = "console", log_level: object = _UNSET) -> None:
    """
    Configure structlog to emit either console-friendly or JSON logs.

    This function sets up structlog with a processor chain that bridges to stdlib logging,
    allowing both structlog and stdlib loggers to work together seamlessly. The log level
    can be set via parameter, environment variable, or defaults to INFO.

    Args:
        log_format: "console" for human-readable logs or "json" for structured output.
        log_level: Log level as string (e.g., "INFO", "DEBUG"). If not provided, uses
            CLIMATE_INDICES_LOG_LEVEL environment variable, falling back to INFO.
            Invalid values fall back to INFO.

    Returns:
        None
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    # resolve log level from parameter, env var, or default
    if log_level is _UNSET:
        level_str = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
    else:
        level_str = str(log_level)

    # normalize and validate
    level_str_upper = level_str.upper()
    if level_str_upper not in VALID_LOG_LEVELS:
        level_str_upper = DEFAULT_LOG_LEVEL

    resolved_level = logging.getLevelName(level_str_upper)

    # build processor chain for structlog
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    pre_chain: list[Processor] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
    ]

    # select renderer based on format
    if log_format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # configure stdlib logging with structlog formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=pre_chain,
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(resolved_level)

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

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog logger bound to the provided name.

    This is a convenience wrapper for internal library use. Library consumers should
    use configure_logging() to set up logging, then use structlog.get_logger() directly.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A structlog BoundLogger instance.
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def _reset_logging_for_testing() -> None:
    """
    Reset logging configuration state for testing purposes.

    This function is intended for use in test fixtures only. It resets the
    configuration guard flag and structlog's internal state, allowing tests
    to verify configuration behavior in isolation.

    Returns:
        None
    """
    global _LOGGING_CONFIGURED
    _LOGGING_CONFIGURED = False
    structlog.reset_defaults()
