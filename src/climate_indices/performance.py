"""Performance metrics utilities for computation tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# 1 GB threshold for triggering memory metrics logging
LARGE_ARRAY_THRESHOLD_BYTES = 1_073_741_824


def get_process_memory_mb() -> float | None:
    """Get current process memory usage in megabytes.

    Uses psutil if available.

    Returns:
        Current RSS (Resident Set Size) memory in MB, or None if psutil is not installed.
    """
    try:
        import psutil
    except ImportError:
        return None

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
    """
    total_bytes = sum(arr.nbytes for arr in arrays)

    if total_bytes <= LARGE_ARRAY_THRESHOLD_BYTES:
        return None

    metrics = {"array_memory_mb": round(total_bytes / (1024 * 1024), 2)}

    process_memory = get_process_memory_mb()
    if process_memory is not None:
        metrics["process_memory_mb"] = round(process_memory, 2)

    return metrics
