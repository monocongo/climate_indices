"""Performance metrics utilities for computation tracking.

This module provides utilities for tracking performance metrics during climate index
calculations, including input size tracking and memory usage monitoring for large arrays.

Custom Metrics via Context Binding
-----------------------------------

The structlog library's `bind()` method allows you to attach custom metrics to a logger
context. All subsequent log events from that logger will include the bound fields:

Example:
    ::

        import structlog
        from climate_indices.performance import check_large_array_memory

        log = structlog.get_logger()
        log = log.bind(
            computation_type="spi",
            input_elements=values.size,
            scale_months=3,
        )

        # all subsequent log events will include the bound fields
        log.info("calculation_started")  # includes computation_type, input_elements, scale_months
        log.info("calculation_completed", duration_ms=123.45)  # same fields present

This pattern is used throughout the climate_indices package to ensure consistent
context across calculation lifecycle events (started, completed, failed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# 1 GB threshold for triggering memory metrics logging
LARGE_ARRAY_THRESHOLD_BYTES = 1_073_741_824

# cached psutil availability flag (set on first call to get_process_memory_mb)
_PSUTIL_AVAILABLE: bool | None = None


def get_process_memory_mb() -> float | None:
    """Get current process memory usage in megabytes.

    Uses psutil if available. On first call, attempts to import psutil and caches
    the result for subsequent calls.

    Returns:
        Current RSS (Resident Set Size) memory in MB, or None if psutil is not installed.
    """
    global _PSUTIL_AVAILABLE

    if _PSUTIL_AVAILABLE is None:
        try:
            import psutil  # noqa: F401

            _PSUTIL_AVAILABLE = True
        except ImportError:
            _PSUTIL_AVAILABLE = False

    if not _PSUTIL_AVAILABLE:
        return None

    import psutil

    process = psutil.Process()
    memory_bytes = process.memory_info().rss
    return memory_bytes / (1024 * 1024)


def check_large_array_memory(*arrays: np.ndarray) -> dict[str, float] | None:
    """Check if arrays exceed memory threshold and return metrics if so.

    Sums the memory usage of all input arrays. If the total exceeds 1 GB,
    returns a dictionary with memory metrics suitable for spreading into log events.

    Args:
        *arrays: One or more numpy arrays to check.

    Returns:
        Dictionary with 'array_memory_mb' (always) and 'process_memory_mb' (if psutil
        available) when total array memory exceeds threshold. Returns None if under
        threshold.

    Example:
        ::

            memory_metrics = check_large_array_memory(precips_mm, pet_mm)
            if memory_metrics:
                log.info("large_arrays_detected", **memory_metrics)
    """
    total_bytes = sum(arr.nbytes for arr in arrays)

    if total_bytes <= LARGE_ARRAY_THRESHOLD_BYTES:
        return None

    metrics = {"array_memory_mb": round(total_bytes / (1024 * 1024), 2)}

    process_memory = get_process_memory_mb()
    if process_memory is not None:
        metrics["process_memory_mb"] = round(process_memory, 2)

    return metrics


def _reset_psutil_cache() -> None:
    """Reset the cached psutil availability flag for test isolation.

    This function is intended for testing purposes only, allowing tests to reset
    the module-level cache and re-evaluate psutil availability.
    """
    global _PSUTIL_AVAILABLE
    _PSUTIL_AVAILABLE = None
