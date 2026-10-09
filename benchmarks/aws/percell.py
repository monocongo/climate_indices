"""Per-cell Rust-versus-Python cost for a per-location-only index, on sampled real cells.

``palmer.scpdsi()`` cannot take a spatial block. ADR-0011 keeps it per-location because
its duration-factor fit and Wells recursion run per location, and the function rejects a
3+-D input with a ``ValueError`` naming that ADR. So scPDSI can never appear in the
harness's ``GRID_ENTRIES`` block tables, and a full-CONUS comparison is out of reach on
the Python side: at roughly 60 ms per cell it needs about 7.8 hours, against the Rust
path's ~11 minutes.

What is measurable is the per-cell cost over a cell sample large enough to pin the
ratio. The spread on this entry is tiny (interquartile range around 0.25%), so a few
thousand cells settle it in minutes, and the cells come from the real prepared grid via
the harness's own loader, so the series, the land mask, the spatial alignment and the
PET derivation are the ones a grid run uses.

scPDSI is the target because it is the family member with no grid measurement at all;
``pdsi`` is measured beside it for reference.

Usage:
    python benchmarks/aws/percell.py [prepared_prcp.nc] [raw_tavg.nc] [cells] [repeats]
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = (1991, 2020)
CONUS_LAND_CELLS = 469_758
GRID_AWC_INCHES = 6.0
MM_PER_INCH = 25.4


def load_harness() -> ModuleType:
    """Load ``benchmarks/rust_vs_python.py`` by path, as the spread driver does."""
    for entry in (str(ROOT), str(ROOT / "benchmarks")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    path = ROOT / "benchmarks" / "rust_vs_python.py"
    if not path.exists():
        raise FileNotFoundError(
            f"no harness at {path}: run this with GIT_REF set to a ref that contains the "
            "RUST-011 harness (perf/1281-rust-benchmarks until #1322 merges)"
        )
    spec = importlib.util.spec_from_file_location("rust_vs_python", path)
    assert spec is not None
    assert spec.loader is not None
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


def extrapolate(per_cell_seconds: float, cells: int) -> float:
    """Seconds the same per-cell cost would take over ``cells`` cells."""
    return per_cell_seconds * cells


def cell_calls(inputs: Any, cells: int, seed: int = 20261009) -> list[tuple[str, Callable[[], Any]]]:
    """One callable per (entry, sampled cell), from the harness loader's own blocks.

    ``load_grid`` already applies the land mask, the spatial roll that aligns temperature
    with precipitation, and the zero-to-0.01 mm treatment, so every series here is the one
    a full-grid run would compute.
    """
    from climate_indices import eto, palmer

    total = inputs.precipitation.shape[1]
    take = min(cells, total)
    picked = np.sort(np.random.default_rng(seed).choice(total, size=take, replace=False))
    print(f"sampled {take} of {total:,} land cells; {inputs.precipitation.shape[0]} months each", flush=True)

    calls: list[tuple[str, Callable[[], Any]]] = []
    for cell in picked:
        # load_grid stores inches-per-month-ready mm series; PDSI reads inches.
        precipitation = np.ascontiguousarray(inputs.precipitation[:, int(cell), 0]) / MM_PER_INCH
        temperature = np.ascontiguousarray(inputs.temperature[:, int(cell), 0])
        latitude = float(inputs.latitude[int(cell), 0])
        pet = eto.eto_thornthwaite(temperature, latitude, inputs.data_start_year) / MM_PER_INCH
        for name, run in (("palmer_pdsi", palmer.pdsi), ("palmer_scpdsi", palmer.scpdsi)):
            calls.append(
                (
                    name,
                    # the loop variables are bound as defaults: the callable outlives this iteration
                    lambda precipitation=precipitation, pet=pet, run=run: run(
                        precipitation,
                        pet,
                        GRID_AWC_INCHES,
                        inputs.data_start_year,
                        *CALIBRATION,
                    ),
                )
            )
    return calls


def main(prcp_path: Path, tavg_path: Path, cells: int, repeats: int) -> None:
    """Measure per-cell PDSI and scPDSI cost for both backends over a cell sample."""
    harness = load_harness()
    inputs = harness.load_grid(prcp_path, "prcp", tavg_path, "tavg")
    print(inputs.description, flush=True)

    samples: dict[str, dict[str, list[float]]] = {}
    for name, call in cell_calls(inputs, cells):
        bucket = samples.setdefault(name, {"rust": [], "python": []})
        for backend, seconds in harness.backend_events(call, call, repeats, warm_up=False):
            bucket[backend].append(seconds)

    results: dict[str, Any] = {}
    print("\nper-cell cost, sampled cells, both backends interleaved:", flush=True)
    for name, bucket in samples.items():
        rust_stats, python_stats = stats(bucket["rust"]), stats(bucket["python"])
        rust_full = extrapolate(rust_stats["median"], CONUS_LAND_CELLS) / 60.0
        python_full = extrapolate(python_stats["median"], CONUS_LAND_CELLS) / 60.0
        results[name] = {
            "cells": len(bucket["rust"]) // repeats,
            "repeats": repeats,
            "rust": rust_stats,
            "python": python_stats,
            "conus_minutes": {"rust": rust_full, "python": python_full},
        }
        print(
            "{name:<15} rust med={rmed:8.3f}ms iqr={riqr:5.1f}% | py med={pmed:9.3f}ms iqr={piqr:5.1f}%"
            " | ratio med={ratio:6.3f} | full CONUS: rust {rf:6.1f} min, py {pf:7.1f} min".format(
                name=name,
                rmed=rust_stats["median"] * 1e3,
                riqr=rust_stats["iqr_pct"],
                pmed=python_stats["median"] * 1e3,
                piqr=python_stats["iqr_pct"],
                ratio=python_stats["median"] / rust_stats["median"],
                rf=rust_full,
                pf=python_full,
            ),
            flush=True,
        )

    (ROOT / "percell.json").write_text(json.dumps(results, indent=2))
    print("\nPERCELL_DONE", flush=True)


if __name__ == "__main__":
    main(
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/nclimgrid_prcp_1981_2024.nc"),
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/nclimgrid_tavg.nc"),
        int(sys.argv[3]) if len(sys.argv) > 3 else 1000,
        int(sys.argv[4]) if len(sys.argv) > 4 else 2,
    )
