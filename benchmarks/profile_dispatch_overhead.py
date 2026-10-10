"""Median/IQR dispatch accounting; prepared extension replay excludes Python prep.

Run from the repository root after ``uv run maturin develop --release``::

    uv run python benchmarks/profile_dispatch_overhead.py --label before

Raw times sum all extension calls captured during an untimed public call. They
include binding copies and allocation, not just pure Rust arithmetic. Public
minus raw is an estimate, not an independently isolated preparation timer.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib
import json
import platform
import pstats
import resource
import subprocess  # nosec B404 # fixed metadata commands and interpreter argv; no shell
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.rust_vs_python import native_policy, python_backend  # noqa: E402
from climate_indices import eto, indices, logging_config, pm_eto  # noqa: E402
from tests import conftest, parity_registry  # noqa: E402

SIZES = (365, 1826, 18262, 146000, 1000000)
SUPPLEMENTAL = parity_registry.Entry(
    "spei_pearson",
    parity_registry.MONTHLY,
    conftest.compute,
    lambda values: parity_registry._spei(values, indices.Distribution.pearson),
    frozenset({"pearson_parameters", "pearson_ks_statistics", "pearson_cdf"}),
)
DISPATCHES = (
    conftest.compute,
    conftest.eto,
    conftest.pm_eto,
    conftest.fire_native,
    conftest.flood_native,
    conftest.palmer,
)


class PreparedCalls:
    """Capture actual extension arguments once, without instrumenting timed calls."""

    def __init__(self, extension: Any) -> None:
        self.extension = extension
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        function = getattr(self.extension, name)
        if isinstance(function, type) or not callable(function):
            return function

        def record(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return function(*args, **kwargs)

        return record

    def replay(self) -> None:
        for name, args, kwargs in self.calls:
            getattr(self.extension, name)(*args, **kwargs)


def summary(samples: list[float]) -> dict[str, float]:
    """Seconds, including quartile endpoints rather than just IQR width."""
    q1, median, q3 = np.quantile(samples, [0.25, 0.5, 0.75])
    return {"median": float(median), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1)}


def measure(name: str, run: Callable[[], Any], repeats: int, elements: int, required: frozenset[str]) -> dict[str, Any]:
    """Alternate public backend order; all patches and policy setup are untimed."""
    extension = importlib.import_module("climate_indices._native")
    captured = PreparedCalls(extension)
    samples: dict[str, list[float]] = {key: [] for key in ("native", "python", "raw", "prep")}
    with native_policy():
        with ExitStack() as stack:
            for module in DISPATCHES:
                stack.enter_context(patch.object(module, "_native", captured))
            run()
        reached = {name for name, _, _ in captured.calls}
        if not required <= reached:
            raise RuntimeError(f"{name}: native kernels not reached: {required - reached}")
        with python_backend():
            run()
        captured.replay()
        for repetition in range(repeats):
            order = ("native", "python", "raw") if repetition % 2 else ("raw", "python", "native")
            for backend in order:
                with ExitStack() as stack:
                    if backend == "python":
                        stack.enter_context(python_backend())
                    start = time.perf_counter_ns()
                    (captured.replay if backend == "raw" else run)()
                    samples[backend].append((time.perf_counter_ns() - start) * 1e-9)
            samples["prep"].append(samples["native"][-1] - samples["raw"][-1])
    stats = {key: summary(values) for key, values in samples.items()}
    return {
        "name": name,
        "elements": elements,
        "calls": [name for name, _, _ in captured.calls],
        "samples_seconds": samples,
        "statistics_seconds": stats,
        "python_over_native": stats["python"]["median"] / stats["native"]["median"],
        "ns_per_element": {key: values["median"] * 1e9 / elements for key, values in stats.items()},
    }


def pet_cases(size: int) -> dict[str, tuple[Callable[[], Any], str]]:
    """Fixed seeded weather; realistic astronomy is separate from registry inputs."""
    rng = np.random.default_rng(1281)
    low = rng.uniform(0.0, 20.0, size)
    high = low + rng.uniform(1.0, 15.0, size)
    mean = (low + high) / 2.0
    monotonic_day = np.arange(1, size + 1, dtype=np.float64)
    day = (monotonic_day - 1.0) % 365.0 + 1.0
    return {
        "pm_eto": (lambda: pm_eto.pm_eto(high, 0.0, mean, 2.0, 2.0, 1.0, 0.2, 0.066), "pm_eto"),
        "fao56_eto_cycling": (lambda: pm_eto.penman_monteith_eto(low, high, 45.0, 100.0, 2.0, day), "fao56_eto"),
        "fao56_eto_monotonic": (
            lambda: pm_eto.penman_monteith_eto(low, high, 45.0, 100.0, 2.0, monotonic_day),
            "fao56_eto",
        ),
        "thornthwaite": (lambda: eto.eto_thornthwaite(mean, 45.0, 1981), "thornthwaite"),
        "hargreaves": (lambda: eto.eto_hargreaves(low, high, mean, 45.0), "hargreaves"),
    }


def rss_probe(size: int) -> dict[str, Any]:
    """Fresh-process RSS: three real arrays; every other PM operand is scalar."""
    low = np.full(size, 12.0)
    high = np.full(size, 24.0)
    day = np.arange(size, dtype=np.float64)
    np.remainder(day, 365.0, out=day)
    day += 1.0
    unit = 1 if sys.platform == "darwin" else 1024

    def peak() -> int:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit

    before = peak()
    with native_policy():
        prepared = pm_eto._native_arrays(low, high, 45.0, 100.0, 2.0, 2.0, day, 0.0, 0.23)
        if prepared is None:
            raise RuntimeError("RSS probe could not prepare native operands")
        after_prep = peak()
        result = pm_eto.penman_monteith_eto(low, high, 45.0, 100.0, 2.0, day)
        if not isinstance(result, np.ndarray) or result.shape != (size,):
            raise RuntimeError("RSS probe returned an unexpected result type or shape")
    return {
        "elements": size,
        "rss_before_bytes": before,
        "rss_after_prep_bytes": after_prep,
        "rss_after_call_bytes": peak(),
        "prep_peak_increase_bytes": after_prep - before,
        "prepared_array_bytes": sum(value.nbytes for value in prepared[1] if isinstance(value, np.ndarray)),
        "scalar_count": sum(isinstance(value, float) for value in prepared[1]),
        "note": "RSS high-water deltas are not exact allocation counts; real-array copies and output remain O(n).",
    }


def profile_pipeline(entry: parity_registry.Entry, repeats: int) -> dict[str, Any]:
    """cProfile with production logging; record the existing native-seam ceiling."""
    run = entry.run(parity_registry.SAMPLES[entry.family])
    with native_policy():
        run()
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(repeats):
            run()
        profiler.disable()
    stats: Any = pstats.Stats(profiler)  # pstats exposes its precise counters without typed attributes
    rows = [
        {
            "file": filename,
            "line": line,
            "function": function,
            "calls": values[1],
            "self_seconds": values[2],
            "cumulative_seconds": values[3],
        }
        for (filename, line, function), values in stats.stats.items()
    ]
    extension_seconds = sum(row["self_seconds"] for row in rows if "climate_indices._native" in row["function"])
    ks_seconds = sum(row["cumulative_seconds"] for row in rows if row["function"] == "_ks_d_statistic")
    gof_seconds = sum(
        row["cumulative_seconds"]
        for row in rows
        if row["function"] in ("_check_goodness_of_fit_gamma", "_check_goodness_of_fit_pearson")
    )
    nested_extension = sum(
        row["self_seconds"] for row in rows if "climate_indices._native.pearson_ks_statistics" in row["function"]
    )
    total = stats.total_tt
    return {
        "name": entry.name,
        "calls": repeats,
        "total_seconds": total,
        "extension_seconds": extension_seconds,
        "ks_seconds": ks_seconds,
        "gof_seconds": gof_seconds,
        "entire_gof_zero_time_upper_bound": total / (total - extension_seconds - gof_seconds + nested_extension),
        "existing_seams_zero_time_ceiling": total / (total - extension_seconds),
        "ks_plus_existing_seams_zero_time_ceiling": total / (total - extension_seconds - ks_seconds),
        "functions": sorted(rows, key=lambda row: row["cumulative_seconds"], reverse=True),
    }


def command_output(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()  # nosec B603


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--label", default="dispatch")
    parser.add_argument("--entries", nargs="*", help="Registry names; default: every entry")
    parser.add_argument("--sizes", nargs="*", type=int, default=SIZES, help="Empty list skips size sweep")
    parser.add_argument("--rss-worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--rss-size", type=int, default=10000000, help="Fresh-process scalar RSS probe; 0 skips")
    parser.add_argument("--profile", action="store_true", help="Profile SPI/SPEI/PNP after timing")
    args = parser.parse_args()
    if args.rss_worker is not None:
        print(json.dumps(rss_probe(args.rss_worker)))
        return
    if args.repeat < 30 or any(size <= 0 for size in args.sizes):
        parser.error("require >=30 repetitions and positive sizes")
    if Path(args.label).parts != (args.label,) or "\\" in args.label:
        parser.error("--label must be a plain name without path separators")
    available = {**parity_registry.ENTRIES_BY_NAME, SUPPLEMENTAL.name: SUPPLEMENTAL}
    entries = list(available.values()) if args.entries is None else [available[name] for name in args.entries]
    logging_config.configure_logging(log_format="console", log_level="INFO")
    extension_file = importlib.import_module("climate_indices._native").__file__
    if extension_file is None:
        raise RuntimeError("Native extension has no file path")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "revision": command_output("git", "rev-parse", "HEAD"),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory, suffix in (("src", ".py"), ("crates", ".rs"), ("benchmarks", ".py"))
            for path in sorted((ROOT / directory).rglob(f"*{suffix}"))
        },
        "extension_sha256": hashlib.sha256(Path(extension_file).read_bytes()).hexdigest(),
        "working_tree": command_output("git", "status", "--short"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu": command_output("sysctl", "-n", "machdep.cpu.brand_string")
        if sys.platform == "darwin"
        else platform.processor(),
        "python": sys.version,
        "rust": command_output("rustc", "--version"),
        "numpy": np.__version__,
        "scipy": importlib.import_module("scipy").__version__,
        "extension": extension_file,
        "argv": sys.argv,
        "repetitions": args.repeat,
        "warmups_per_backend": 1,
        "error_policy": "all=ignore",
        "logging": "library default console INFO",
    }
    output = ROOT / "benchmarks/results" / f"{args.label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    report: dict[str, Any] = {"metadata": metadata, "entries": [], "sweep": [], "completed": False}
    output.parent.mkdir(parents=True, exist_ok=True)

    def record(section: str, row: dict[str, Any]) -> None:
        report[section].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        stats = row["statistics_seconds"]
        print(
            f"{row['name']:30} n={row['elements']:7} native={stats['native']['median'] * 1e6:10.2f} us "
            f"IQR={stats['native']['iqr'] * 1e6:8.2f} python/native={row['python_over_native']:.3f}",
            flush=True,
        )

    for entry in entries:
        values = parity_registry.SAMPLES[entry.family]
        record("entries", measure(entry.name, entry.run(values), args.repeat, values.size, entry.kernels))
    for size in args.sizes:
        for name, (run, kernel) in pet_cases(size).items():
            record("sweep", measure(name, run, args.repeat, size, frozenset({kernel})))
    if args.profile:
        names = {"spi_gamma", "spi_pearson", "spei_gamma", "spei_pearson", "spei_loglogistic", "percentage_of_normal"}
        report["profiles"] = [profile_pipeline(entry, args.repeat) for entry in entries if entry.name in names]
    if args.rss_size:
        worker = subprocess.check_output(  # nosec B603
            [sys.executable, str(Path(__file__).resolve()), "--rss-worker", str(args.rss_size)], text=True
        )
        try:
            report["rss"] = json.loads(worker.splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as error:
            raise RuntimeError("RSS worker returned no valid JSON result") from error
    report["completed"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
