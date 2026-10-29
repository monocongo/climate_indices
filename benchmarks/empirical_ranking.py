"""Compare four NumPy rank-count implementations in separate process runs.

Run ``uv run benchmarks/empirical_ranking.py --help`` for the workload controls.
Each invocation checks ties, NaNs, and infinities before timing one implementation.
The timer includes sorting, chunk copies, and the output allocation. It excludes
input generation and the plotting-position and inverse-normal transforms.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time

import numpy as np


def rank_counts(values: np.ndarray, calibration: np.ndarray, implementation: str, chunk_columns: int) -> np.ndarray:
    """Count calibration values at or below each value, independently per column."""
    counts = np.zeros(values.shape)
    if implementation == "loop":
        for year in calibration:
            counts += year <= values
        return counts

    years = calibration.shape[0]
    for start in range(0, values.shape[1], chunk_columns):
        stop = start + chunk_columns
        queries = values[:, start:stop]
        sample = calibration[:, start:stop]
        if implementation == "cube":
            counts[:, start:stop] = (sample[:, None, :] <= queries[None, :, :]).sum(axis=0)
        elif implementation == "merge":
            combined = np.concatenate((sample, queries), axis=0)
            # Stable sorting puts calibration values before equal queries for maximum-tie ranks.
            order = np.argsort(combined, axis=0, kind="stable")
            sorted_values = np.take_along_axis(combined, order, axis=0)
            cumulative = np.cumsum((order < years) & ~np.isnan(sorted_values), axis=0)
            ranks = np.empty_like(cumulative)
            np.put_along_axis(ranks, order, cumulative, axis=0)
            counts[:, start:stop] = np.where(np.isnan(queries), 0, ranks[years:])
        elif implementation == "binary":
            sorted_sample = np.sort(sample, axis=0)
            lower = np.zeros(queries.shape, dtype=np.intp)
            upper = np.full(queries.shape, years, dtype=np.intp)
            for _ in range(years.bit_length()):
                active = lower < upper
                middle = (lower + upper) // 2
                candidate = np.take_along_axis(sorted_sample, np.minimum(middle, years - 1), axis=0)
                right = active & (candidate <= queries) & ~np.isnan(candidate)
                lower = np.where(right, middle + 1, lower)
                upper = np.where(active & ~right, middle, upper)
            counts[:, start:stop] = lower
        else:
            raise ValueError(f"unknown implementation: {implementation}")
    return counts


def check_counts() -> None:
    """Check all implementations against direct comparisons, including chunk tails."""
    rng = np.random.default_rng(1435)
    values = rng.integers(-2, 3, size=(17, 7)).astype(float)
    values[0, :3] = [np.nan, -np.inf, np.inf]
    for years in (1, 2, 10, 17):
        calibration = values[:years]
        expected = (calibration[:, None, :] <= values[None, :, :]).sum(axis=0)
        for implementation in ("loop", "cube", "merge", "binary"):
            for chunk_columns in (1, 3, 7):
                np.testing.assert_array_equal(rank_counts(values, calibration, implementation, chunk_columns), expected)


def main() -> None:
    """Print one rank-only measurement as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("loop", "cube", "merge", "binary"), default="loop")
    parser.add_argument("--calibration-years", type=int, choices=(20, 60), default=20)
    parser.add_argument("--cells", type=int, default=65536)
    parser.add_argument("--chunk-columns", type=int, default=8192)
    args = parser.parse_args()
    if args.cells < 1 or args.chunk_columns < 1:
        parser.error("cells and chunk-columns must be positive")
    check_counts()
    values = np.random.default_rng(1435).gamma(2.0, size=(60, 12 * args.cells))
    calibration = values[: args.calibration_years]
    rss_unit = 1 if platform.system() == "Darwin" else 1024
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
    start = time.perf_counter()
    counts = rank_counts(values, calibration, args.implementation, args.chunk_columns)
    seconds = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
    np.testing.assert_array_equal(counts[:, :1], (calibration[:, None, :1] <= values[None, :, :1]).sum(axis=0))
    print(
        json.dumps(
            {
                "implementation": args.implementation,
                "calibration_years": args.calibration_years,
                "chunk_columns": args.chunk_columns,
                "shape": list(values.shape),
                "seconds": seconds,
                "input_bytes": values.nbytes,
                "peak_rss_bytes": after,
                "peak_increase_bytes": after - before,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            }
        )
    )


if __name__ == "__main__":
    main()
