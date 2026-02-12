"""Shared logging suppression fixture for tests.

This module provides a centralized logging suppression fixture to reduce
noise during test execution, particularly for property-based tests that
generate many test cases.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(scope="module")
def suppress_logging():
    """Suppress logging output during tests to reduce noise.

    This fixture temporarily disables logging at the CRITICAL level and above,
    preventing log messages from cluttering test output.

    Use this fixture explicitly in test modules that generate excessive logging:
        pytest_plugins = ["tests.helpers.logging"]

        @pytest.mark.usefixtures("suppress_logging")
        class TestMyModule:
            ...

    DO NOT use this fixture in test modules that verify logging behavior
    (e.g., test_logging.py, test_error_context_logging.py).
    """
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)
