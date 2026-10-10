#!/usr/bin/env python3
"""Compare candidate AWC/AWS sources for gridded scPDSI: cost and output sensitivity.

Three soil-water sources are ingested and harmonized onto a climate grid
(``gridmet``, ``usgs``, ``polaris`` -- see :mod:`climate_indices.aws_ingest` for
what each one is and which of its facts are verified), the self-calibrating
Palmer index is computed from each, and the runs are compared.

Default test region
    ``--region -100,-99,38,39`` -- a 1 degree by 1 degree box in the Colorado /
    Kansas border country (west, east, south, north in degrees), snapped to the
    climate grid's cells. It is small enough to run on a laptop and spans a
    precipitation gradient with real soil variability. ``--region`` and
    ``--max-cells`` override it.

What is measured
    Ingest+harmonization and the scPDSI run are timed and their peak resident
    memory measured **separately**, each in its own subprocess (``ru_maxrss`` of
    that child), because once both sources are on the same grid the PDSI cost is
    nearly identical and the real difference is in ingest. Every source is run
    with the same period, calibration window, grid, and depth.

What is compared
    Per source: AWS field mean, range, and percent of cells filled. Per pair of
    sources, over the common finite mask and period: mean absolute difference of
    scPDSI, Pearson correlation, and the percentage of months agreeing on a
    drought category. Categories are the CPC Palmer severity ladder carrying the
    USDM D0-D4 labels (``D0``: -0.5 >= scPDSI > -1.0, ``D1``: -1.0 .. -2.0,
    ``D2``: -2.0 .. -3.0, ``D3``: -3.0 .. -4.0, ``D4``: <= -4.0, ``normal`` above
    -0.5). That is a **Palmer-derived proxy**, not the US Drought Monitor's
    expert classification, which is not a function of any single index.

Outputs
    ``<out>/aws_sources.csv`` (one tidy row per source and depth: cost, AWS field
    statistics, cache provenance) and ``<out>/aws_pairwise_scpdsi.csv`` (one row
    per source pair and depth: the scPDSI comparisons above). A short summary is
    printed. A source whose data cannot be read is never silently substituted:
    its row records the failure and the run continues with the others.

Usage::

    uv run benchmarks/benchmark_pdsi_soils.py \\
        --precip /path/nclimgrid_lowres_prcp.nc --pet /path/nclimgrid_lowres_pet.nc \\
        --region -100,-99,38,39 --depths native,1000,1500 \\
        --out /tmp/aws-benchmark

Reading remote POLARIS tiles needs the optional geospatial dependencies
(``pip install 'climate-indices[aws]'``); a harmonized GeoTIFF cache in
``--cache-dir`` removes that requirement on later runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import aws_ingest
from climate_indices.aws_ingest import AwsIngestError, HarmonizedAws
from climate_indices.exceptions import ClimateIndicesError
from climate_indices.logging_config import get_logger

_logger = get_logger(__name__)

#: Default comparison region as (west, east, south, north) degrees.
DEFAULT_REGION = (-100.0, -99.0, 38.0, 39.0)

#: Default soil depths: each source's own column, then a shared shallower column.
DEFAULT_DEPTHS = ("native", "1000")

#: Drought categories as (label, lower bound inclusive, upper bound exclusive),
#: most severe first, over the CPC Palmer severity ladder carrying USDM labels.
_PALMER_CATEGORIES: tuple[tuple[str, float, float], ...] = (
    ("D4", -np.inf, -4.0),
    ("D3", -4.0, -3.0),
    ("D2", -3.0, -2.0),
    ("D1", -2.0, -1.0),
    ("D0", -1.0, -0.5),
    ("normal", -0.5, np.inf),
)


# ---------------------------------------------------------------------------
# benchmark: phase runners (each executes in its own subprocess)
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Cost and output of one measured phase."""

    source: str
    depth_mm: float | str
    phase: str
    elapsed_s: float = float("nan")
    peak_rss_mb: float = float("nan")
    aws_mean_mm: float = float("nan")
    aws_min_mm: float = float("nan")
    aws_max_mm: float = float("nan")
    filled_percent: float = float("nan")
    cells: int = 0
    months: int = 0
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _peak_rss_mb() -> float:
    """Peak resident set size of this process in MiB (macOS reports bytes)."""
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return float(usage) / divisor


def _parse_region(text: str) -> tuple[float, float, float, float]:
    """Parse ``west,east,south,north`` and reject a degenerate box."""
    parts = [float(part) for part in text.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be four comma-separated degrees: west,east,south,north")
    west, east, south, north = parts
    if west >= east or south >= north:
        raise argparse.ArgumentTypeError(f"region must have west < east and south < north, got {parts!r}")
    return west, east, south, north


def _subset_region(
    field: xr.DataArray,
    region: tuple[float, float, float, float],
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray:
    """Clip a field to a region, taking the climate cells whose centres fall inside."""
    west, east, south, north = region
    latitudes = field[lat_dim].values
    ascending_lat = latitudes[0] <= latitudes[-1]
    lat_slice = slice(south, north) if ascending_lat else slice(north, south)
    return field.sel({lat_dim: lat_slice, lon_dim: slice(west, east)})


def _open_climate(path: str, variable: str | None, *, period: tuple[str, str] | None = None) -> xr.DataArray:
    """Open one climate NetCDF variable as a (time, lat, lon) DataArray in mm."""
    dataset = xr.open_dataset(path)
    if variable is None:
        candidates = [name for name in dataset.data_vars if name not in {"spatial_ref"}]
        if len(candidates) != 1:
            raise SystemExit(f"{path}: cannot pick a variable; --precip-var/--pet-var needed (found {candidates})")
        variable = candidates[0]
    field = dataset[variable]
    if period is not None:
        field = field.sel(time=slice(period[0], period[1]))
    units = str(field.attrs.get("units", "")).strip().lower()
    if units in {"inches", "inch"}:
        field = field * 25.4
        field.attrs["units"] = "mm"
    elif units not in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        raise SystemExit(f"{path}[{variable}]: unsupported units {units!r}; expected mm or inches")
    # the Palmer recursion indexes calendar months from January and needs a
    # continuous monthly record
    return field.transpose("time", "lat", "lon")


def _scpdsi_field(
    precip: xr.DataArray,
    pet: xr.DataArray,
    aws: xr.DataArray,
    *,
    data_start_year: int,
    calibration_period: tuple[int, int],
    max_cells: int,
) -> xr.DataArray:
    """Compute scPDSI for every cell of an AWS field.

    ``palmer.scpdsi()`` stays on the per-location path by design (ADR-0011), so
    the grid is looped here exactly as the CLI loops it, and a location whose
    calibration cannot fit usable factors is left missing instead of aborting
    the run.
    """
    from climate_indices import palmer
    from climate_indices.exceptions import ConvergenceError, InsufficientDataError

    latitudes = aws["lat"].values
    longitudes = aws["lon"].values
    if latitudes.size * longitudes.size > max_cells:
        raise SystemExit(
            f"region covers {latitudes.size * longitudes.size} cells, above --max-cells {max_cells}; "
            "narrow --region or raise --max-cells"
        )
    precip_values = _to_inches(precip).values.reshape(precip.sizes["time"], -1)
    pet_values = _to_inches(pet).values.reshape(pet.sizes["time"], -1)
    awc_values = aws_ingest.aws_mm_to_inches(aws).values.reshape(-1)
    if precip_values.shape != pet_values.shape:
        raise SystemExit("precipitation and PET must share the same time and cell dimensions")
    if precip_values.shape[1] != awc_values.size:
        raise SystemExit(
            f"climate grid has {precip_values.shape[1]} cells but the AWS field has {awc_values.size}; "
            "the soil field must be harmonized onto the climate grid first"
        )

    output = np.full(precip_values.shape, np.nan)
    failed = 0
    for cell in range(precip_values.shape[1]):
        if not np.isfinite(awc_values[cell]):
            continue
        try:
            output[:, cell] = palmer.scpdsi(
                precip_values[:, cell],
                pet_values[:, cell],
                float(awc_values[cell]),
                data_start_year,
                calibration_period[0],
                calibration_period[1],
            )[0]
        except (ConvergenceError, InsufficientDataError):
            failed += 1
            continue
    if failed:
        _logger.warning("scpdsi_locations_skipped", cells=failed, reason="unfittable duration factors")
    return xr.DataArray(
        output.reshape(precip.sizes["time"], latitudes.size, longitudes.size),
        coords={"time": precip["time"].values, "lat": latitudes, "lon": longitudes},
        dims=["time", "lat", "lon"],
        name="scpdsi",
    )


def _to_inches(field: xr.DataArray) -> xr.DataArray:
    """Convert a millimetre field to inches for the Palmer entry points."""
    return field / 25.4


def _category(values: np.ndarray) -> np.ndarray:
    """Map scPDSI values to drought-category labels."""
    labels = np.full(values.shape, "missing", dtype=object)
    finite = np.isfinite(values)
    for label, lower, upper in _PALMER_CATEGORIES:
        selected = finite & (values >= lower) & (values < upper)
        labels[selected] = label
    return labels


def _pairwise_metrics(scpdsi_a: np.ndarray, scpdsi_b: np.ndarray) -> dict[str, float]:
    """Mean absolute difference, correlation, and category agreement over a common mask."""
    common = np.isfinite(scpdsi_a) & np.isfinite(scpdsi_b)
    if not common.any():
        return {
            "common_values": 0.0,
            "mean_absolute_difference": float("nan"),
            "correlation": float("nan"),
            "category_agreement_percent": float("nan"),
        }
    left = scpdsi_a[common]
    right = scpdsi_b[common]
    if left.size > 1 and left.std() > 0.0 and right.std() > 0.0:
        correlation = float(np.corrcoef(left, right)[0, 1])
    else:
        correlation = float("nan")
    return {
        "common_values": float(common.sum()),
        "mean_absolute_difference": float(np.abs(left - right).mean()),
        "correlation": correlation,
        "category_agreement_percent": float((_category(left) == _category(right)).mean() * 100.0),
    }


# ---------------------------------------------------------------------------
# benchmark: the phases as standalone subprocess entry points
# ---------------------------------------------------------------------------


def _run_ingest_phase(args: argparse.Namespace, source: str, depth: str) -> PhaseResult:
    """Ingest and harmonize one source, cache it, and report cost and field statistics."""
    result = PhaseResult(source=source, depth_mm=depth, phase="ingest")
    started = time.perf_counter()
    try:
        harmonized = _ingest(args, source, depth)
    except (AwsIngestError, ClimateIndicesError, OSError) as error:
        result.elapsed_s = time.perf_counter() - started
        result.note = f"unavailable: {type(error).__name__}: {error}"
        _logger.warning("aws_source_unavailable", source=source, depth_mm=depth, error=str(error))
        return result

    result.elapsed_s = time.perf_counter() - started
    result.peak_rss_mb = _peak_rss_mb()
    values = harmonized.aws.values
    finite = np.isfinite(values)
    result.cells = int(finite.sum())
    result.aws_mean_mm = float(values[finite].mean()) if finite.any() else float("nan")
    result.aws_min_mm = float(values[finite].min()) if finite.any() else float("nan")
    result.aws_max_mm = float(values[finite].max()) if finite.any() else float("nan")
    filled_land = int(np.asarray(harmonized.filled.values, dtype=bool).sum())
    result.filled_percent = 100.0 * filled_land / result.cells if result.cells else float("nan")
    result.extra = {
        key: harmonized.aws.attrs.get(key)
        for key in ("aws_depth_mm", "aws_native_depth_mm", "aws_area_weighted", "aws_cells_at_surface_capacity")
    }
    result.note = "; ".join(json.loads(harmonized.aws.attrs.get("aws_unverified", "[]")))
    return result


def _ingest(args: argparse.Namespace, source: str, depth: str) -> HarmonizedAws:
    """Harmonize one source onto the climate region grid (region cells only)."""
    precip = _subset_region(_open_climate(args.precip, args.precip_var, period=args.period), args.region)
    return aws_ingest.load_aws(
        source,
        precip,
        depth_mm=aws_ingest.NATIVE_DEPTH if depth == "native" else float(depth),
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
    )


def _run_pdsi_phase(args: argparse.Namespace, source: str, depth: str) -> PhaseResult:
    """Compute scPDSI from one harmonized AWS field and report cost."""
    result = PhaseResult(source=source, depth_mm=depth, phase="pdsi")
    started = time.perf_counter()
    try:
        harmonized = _ingest(args, source, depth)
        precip = _subset_region(_open_climate(args.precip, args.precip_var, period=args.period), args.region)
        pet = _subset_region(_open_climate(args.pet, args.pet_var, period=args.period), args.region)
        if precip.sizes["time"] != pet.sizes["time"]:
            raise SystemExit("precipitation and PET must cover the same period")
        scpdsi = _scpdsi_field(
            precip,
            pet,
            harmonized.aws,
            data_start_year=int(pd.Timestamp(precip["time"].values[0]).year),
            calibration_period=args.calibration,
            max_cells=args.max_cells,
        )
        values = scpdsi.values
        result.cells = int(np.isfinite(values).any(axis=0).sum())
        result.months = int(precip.sizes["time"])
        _save_scpdsi(args, source, depth, scpdsi)
    except (AwsIngestError, ClimateIndicesError, OSError) as error:
        result.elapsed_s = time.perf_counter() - started
        result.note = f"unavailable: {type(error).__name__}: {error}"
        _logger.warning("scpdsi_phase_unavailable", source=source, depth_mm=depth, error=str(error))
        return result

    result.elapsed_s = time.perf_counter() - started
    result.peak_rss_mb = _peak_rss_mb()
    return result


def _scpdsi_cache_path(args: argparse.Namespace, source: str, depth: str) -> Path:
    """Location of one run's scPDSI field, used by the comparison phase."""
    return Path(args.out) / "scpdsi" / f"scpdsi_{source}_{depth}.nc"


def _save_scpdsi(args: argparse.Namespace, source: str, depth: str, values: xr.DataArray) -> None:
    """Persist one scPDSI field so the parent process can compare runs."""
    path = _scpdsi_cache_path(args, source, depth)
    path.parent.mkdir(parents=True, exist_ok=True)
    values.to_netcdf(path, engine="h5netcdf")


def _child_command(args: argparse.Namespace, phase: str, source: str, depth: str) -> list[str]:
    """Re-invoke this script for one measured phase."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--phase",
        phase,
        "--source",
        source,
        "--depth",
        depth,
        "--precip",
        str(args.precip),
        "--pet",
        str(args.pet),
        "--region",
        ",".join(str(value) for value in args.region),
        "--calibration",
        ",".join(str(year) for year in args.calibration),
        "--period",
        f"{args.calibration[0]}-01-01,{args.calibration[1]}-12-01",
        "--max-cells",
        str(args.max_cells),
        "--out",
        str(args.out),
        "--json",
    ]
    for name, value in (
        ("--precip-var", args.precip_var),
        ("--pet-var", args.pet_var),
        ("--raw-dir", args.raw_dir),
        ("--cache-dir", args.cache_dir),
    ):
        if value:
            command += [name, str(value)]
    return command


def _run_child(args: argparse.Namespace, phase: str, source: str, depth: str) -> PhaseResult:
    """Run one phase in a subprocess so its peak memory is that phase's alone."""
    completed = subprocess.run(  # noqa: S603 - argv is built here, not from user text
        _child_command(args, phase, source, depth),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return PhaseResult(
            source=source,
            depth_mm=depth,
            phase=phase,
            note=f"phase failed (exit {completed.returncode}): {completed.stderr.strip().splitlines()[-1:]}",
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    payload.setdefault("extra", {})
    return PhaseResult(**payload)


# ---------------------------------------------------------------------------
# benchmark: comparison and reporting
# ---------------------------------------------------------------------------


def _load_scpdsi(args: argparse.Namespace, source: str, depth: str) -> xr.DataArray | None:
    path = _scpdsi_cache_path(args, source, depth)
    if not path.exists():
        return None
    return xr.open_dataarray(path)


def _pairwise_rows(args: argparse.Namespace, available: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Compare every pair of successfully ingested sources for each depth."""
    rows: list[dict[str, Any]] = []
    depths = sorted({depth for _, depth in available}, key=lambda depth: (depth != "native", depth))
    for depth in depths:
        sources = sorted(source for source, source_depth in available if source_depth == depth)
        for index, source_a in enumerate(sources):
            for source_b in sources[index + 1 :]:
                field_a = _load_scpdsi(args, source_a, depth)
                field_b = _load_scpdsi(args, source_b, depth)
                if field_a is None or field_b is None:
                    continue
                aligned_a, aligned_b = xr.align(field_a, field_b, join="inner")
                metrics = _pairwise_metrics(aligned_a.values, aligned_b.values)
                rows.append({"depth": depth, "source_a": source_a, "source_b": source_b, **metrics})
    return rows


def _summary(ingest_rows: list[PhaseResult], pdsi_rows: list[PhaseResult], pairwise: list[dict[str, Any]]) -> str:
    """A short human-readable summary of the run."""
    lines = ["AWC/AWS source comparison (scPDSI)", ""]
    for row in ingest_rows:
        if row.note.startswith("unavailable") or row.note.startswith("phase failed"):
            lines.append(f"{row.source} ({row.depth_mm}): {row.note}")
            continue
        lines.append(
            f"{row.source} ({row.depth_mm}): ingest {row.elapsed_s:.1f}s / {row.peak_rss_mb:.0f} MiB, "
            f"AWS mean {row.aws_mean_mm:.1f} mm, range {row.aws_min_mm:.1f}-{row.aws_max_mm:.1f} mm, "
            f"filled {row.filled_percent:.2f}%"
        )
    for row in pdsi_rows:
        if row.note.startswith("unavailable") or row.note.startswith("phase failed"):
            lines.append(f"scPDSI {row.source} ({row.depth_mm}): {row.note}")
            continue
        lines.append(
            f"scPDSI {row.source} ({row.depth_mm}): {row.elapsed_s:.1f}s / {row.peak_rss_mb:.0f} MiB "
            f"over {row.cells} cells x {row.months} months"
        )
    if pairwise:
        lines.append("")
        for row in pairwise:
            lines.append(
                f"{row['source_a']} vs {row['source_b']} ({row['depth']}): "
                f"mean |diff| {row['mean_absolute_difference']:.3f}, r {row['correlation']:.4f}, "
                f"category agreement {row['category_agreement_percent']:.1f}%"
            )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precip", required=True, help="monthly precipitation NetCDF (mm or inches)")
    parser.add_argument("--pet", required=True, help="monthly PET NetCDF (mm or inches)")
    parser.add_argument("--precip-var", default=None, help="precipitation variable name in --precip")
    parser.add_argument("--pet-var", default=None, help="PET variable name in --pet")
    parser.add_argument(
        "--region",
        type=_parse_region,
        default=DEFAULT_REGION,
        help="west,east,south,north in degrees; defaults to the Colorado/Kansas border box -100,-99,38,39",
    )
    parser.add_argument(
        "--sources", default=",".join(sorted(aws_ingest.AWS_SOURCES)), help="comma-separated aws_source names"
    )
    parser.add_argument("--depths", default=",".join(DEFAULT_DEPTHS), help="comma-separated depths: native or mm")
    parser.add_argument(
        "--calibration",
        type=lambda text: tuple(int(year) for year in text.split(",")),
        default=(1981, 2010),
    )
    parser.add_argument(
        "--period",
        type=lambda text: tuple(text.split(",")),
        default=None,
        help="start,end dates for the run; defaults to the calibration years",
    )
    parser.add_argument("--max-cells", type=int, default=4096, help="refuse a region larger than this many cells")
    parser.add_argument("--raw-dir", default=None, help="directory holding raw soil rasters")
    parser.add_argument("--cache-dir", default=None, help="directory for the harmonized GeoTIFF cache")
    parser.add_argument("--out", default="aws-benchmark", help="output directory for CSVs and scPDSI fields")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=["ingest", "pdsi"], help=argparse.SUPPRESS)
    parser.add_argument("--source", help=argparse.SUPPRESS)
    parser.add_argument("--depth", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.period is None:
        args.period = (f"{args.calibration[0]}-01-01", f"{args.calibration[1]}-12-01")

    if args.phase is not None:
        if args.source is None or args.depth is None:
            raise SystemExit("--phase needs --source and --depth")
        runner = _run_ingest_phase if args.phase == "ingest" else _run_pdsi_phase
        result = runner(args, args.source, args.depth)
        print(
            json.dumps(
                {
                    "source": result.source,
                    "depth_mm": result.depth_mm,
                    "phase": result.phase,
                    "elapsed_s": result.elapsed_s,
                    "peak_rss_mb": result.peak_rss_mb,
                    "aws_mean_mm": result.aws_mean_mm,
                    "aws_min_mm": result.aws_min_mm,
                    "aws_max_mm": result.aws_max_mm,
                    "filled_percent": result.filled_percent,
                    "cells": result.cells,
                    "months": result.months,
                    "note": result.note,
                    "extra": result.extra,
                }
            )
        )
        return 0

    sources = [name.strip() for name in args.sources.split(",") if name.strip()]
    depths = [depth.strip() for depth in args.depths.split(",") if depth.strip()]
    _logger.info(
        "benchmark_started",
        region=args.region,
        sources=sources,
        depths=depths,
        calibration=args.calibration,
        period=args.period,
    )

    ingest_rows: list[PhaseResult] = []
    pdsi_rows: list[PhaseResult] = []
    available: list[tuple[str, str]] = []
    for source in sources:
        for depth in depths:
            ingest = _run_child(args, "ingest", source, depth)
            ingest_rows.append(ingest)
            if not ingest.note.startswith(("unavailable", "phase failed")):
                available.append((source, depth))
            pdsi_rows.append(_run_child(args, "pdsi", source, depth))

    ingest_table = pd.DataFrame([row.__dict__ for row in ingest_rows])
    ingest_table.insert(0, "kind", "ingest+harmonization")
    pdsi_table = pd.DataFrame([row.__dict__ for row in pdsi_rows])
    pdsi_table.insert(0, "kind", "scPDSI run")
    combined = pd.concat([ingest_table, pdsi_table], ignore_index=True)
    combined_path = Path(args.out) / "aws_sources.csv"
    combined.drop(columns=["extra"]).to_csv(combined_path, index=False)

    pairwise = _pairwise_rows(args, available)
    pairwise_path = Path(args.out) / "aws_pairwise_scpdsi.csv"
    pd.DataFrame(pairwise).to_csv(pairwise_path, index=False)

    summary = _summary(ingest_rows, pdsi_rows, pairwise)
    print(summary)
    print(f"\nwrote {combined_path}\nwrote {pairwise_path}")
    if not available:
        print(
            "\nNo source could be read in this environment. Check the 'note' column of "
            f"{combined_path} for the exact reason (missing rasters, missing rioxarray/rasterio, "
            "or unreachable tile server)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
