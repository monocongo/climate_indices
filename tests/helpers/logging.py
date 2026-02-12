"""Shared logging suppression fixture for tests.

This module provides a centralized logging suppression fixture to reduce
noise during test execution, particularly for property-based tests that
generate many test cases.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(scope="module", autouse=True)
def suppress_logging():
    """Suppress logging output during tests to reduce noise.

    This fixture runs automatically for all test modules and temporarily
    disables logging at the CRITICAL level and above, preventing log
    messages from cluttering test output.
    """
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)
