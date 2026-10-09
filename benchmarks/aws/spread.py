"""Per-sample spread for every parity-registry entry, through the harness's own sampler.

The committed benchmarks report each entry's best sample, which cannot show how
large a run's spread is and therefore which ratios are measurable at all. This
records every repetition instead, from the same interleaved sampler the harness
uses, and reports the median and interquartile range per backend.

This is the shape a `--spread` flag on `benchmarks/rust_vs_python.py` should
take once it has proven useful across a few runs.

Usage:
    taskset -c 0-7 python benchmarks/aws/spread.py [repeats]
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]


def load_harness() -> ModuleType:
    """Load ``benchmarks/rust_vs_python.py`` by path.

    Imported this way rather than with ``import rust_vs_python`` after a
    ``sys.path`` insert, because an import sorter will happily move a module-level
    import above the path setup and break it at run time. The harness resolves
    its own sibling and test imports from the repository root, so both entries go
    on the path first.
    """
    for entry in (str(ROOT), str(ROOT / "benchmarks")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    spec = importlib.util.spec_from_file_location("rust_vs_python", ROOT / "benchmarks" / "rust_vs_python.py")
    assert spec is not None
    assert spec.loader is not None
    if not (ROOT / "benchmarks" / "rust_vs_python.py").exists():
        raise FileNotFoundError(
            f"no harness at {ROOT / 'benchmarks' / 'rust_vs_python.py'}: run this with GIT_REF set to a "
            "ref that contains the RUST-011 harness (perf/1281-rust-benchmarks until #1322 merges)"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules["rust_vs_python"] = module
    spec.loader.exec_module(module)
    return module


def stats(samples: list[float]) -> dict[str, float]:
    """Minimum, quartiles, median, and the interquartile range as a percentage of the median."""
    ordered = sorted(samples)
    q1, _, q3 = statistics.quantiles(ordered, n=4)
    median = statistics.median(ordered)
    return {
        "min": ordered[0],
        "p25": q1,
        "median": median,
        "p75": q3,
        "max": ordered[-1],
        "iqr_pct": (q3 - q1) / median * 100.0,
    }


def separable(rust: dict[str, float], python: dict[str, float]) -> bool:
    """Whether the two backends' interquartile ranges fail to overlap, in either direction."""
    return rust["p75"] < python["p25"] or python["p75"] < rust["p25"]


def main(repeats: int) -> None:
    """Time every registry entry ``repeats`` times per backend and report the spread."""
    harness = load_harness()
    print(f"repeats={repeats}; backend_events interleaves which backend runs first", flush=True)

    rows = []
    for entry in harness.ENTRIES:
        drawn = harness.SAMPLES[entry.family]
        rust: list[float] = []
        python: list[float] = []
        for backend, seconds in harness.backend_events(entry.run(drawn), entry.run(drawn), repeats):
            (rust if backend == "rust" else python).append(seconds)
        rust_stats, python_stats = stats(rust), stats(python)
        rows.append({"name": entry.name, "family": entry.family, "rust": rust_stats, "python": python_stats})
        print(
            "{name:<30} rust med={rmed:9.3f}ms iqr={riqr:5.1f}%"
            " | py med={pmed:9.3f}ms iqr={piqr:5.1f}%"
            " | ratio med={ratio:7.3f} min={rmin:7.3f} separable={sep}".format(
                name=entry.name,
                rmed=rust_stats["median"] * 1e3,
                riqr=rust_stats["iqr_pct"],
                pmed=python_stats["median"] * 1e3,
                piqr=python_stats["iqr_pct"],
                ratio=python_stats["median"] / rust_stats["median"],
                rmin=python_stats["min"] / rust_stats["min"],
                sep=separable(rust_stats, python_stats),
            ),
            flush=True,
        )

    (ROOT / "spread.json").write_text(json.dumps(rows, indent=2))
    print("SPREAD_DONE")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
