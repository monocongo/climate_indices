"""Time the legacy CLI's multiprocessing.Pool path against the xarray/Dask API.

`process_climate_indices` (`__main__.py`) parallelizes gridded NetCDF processing
with Python's `multiprocessing.Pool` over shared-memory arrays, split by
latitude, at a fixed worker count (ADR-0002). #1097
(`benchmarks/parallel_scaling.py --netcdf`) measured eager-vs-Dask on the
xarray/Dask API alone; neither side of that comparison goes through the CLI.
This script is the other half: run the same SPI-6 gamma workload through both
paths on one full-CONUS nClimGrid-Monthly precipitation grid, on the same
checkout, and assert the outputs match before any speedup is quoted.

Two subcommands:

``prepare`` trims a source nClimGrid NetCDF (the mutable, full period-of-record
object NCEI publishes) to one time span and applies the same land-mask-from-
first-step and zero-to-0.01mm treatment `parallel_scaling.load_netcdf_grid`
applies, so both paths compute from byte-identical input. The result is the
`--netcdf` argument to both `parallel_scaling.py` (the xarray/Dask side) and
this script's own ``time`` subcommand (the CLI side).

``time`` calls `climate_indices.__main__.main()` — the real CLI entry point,
default `--multiprocessing all_but_one` — the number of times the harness
convention calls for (one warm-up, then `--repeat` timed runs), and reports
Pool-map-only ("compute") and open+copy+compute+write ("total") seconds per
SPI distribution the CLI always computes (gamma and pearson; the CLI has no
flag to pick one). Before any timing is reported, the CLI's gamma output is
checked cell-for-cell against a `parallel_scaling.py --netcdf` xarray/Dask run
over the same prepared file.

Run from the repository root::

    uv run benchmarks/cli_multiprocessing.py prepare \\
        nclimgrid_prcp.nc nclimgrid_prcp_1981_2024.nc --start 1981 --end 2024
    uv run benchmarks/parallel_scaling.py --netcdf nclimgrid_prcp_1981_2024.nc \\
        --var-name prcp --scale 6 --cores 1,2,4,8 --repeat 3 \\
        --write-output /tmp/xarray_spi6_gamma.nc
    uv run benchmarks/cli_multiprocessing.py time nclimgrid_prcp_1981_2024.nc \\
        --var-name prcp --scale 6 --repeat 3 \\
        --output-dir /tmp/cli --xarray-output /tmp/xarray_spi6_gamma.nc
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from parallel_scaling import _environment, _format_samples, _hash_file, _revision

import climate_indices.__main__ as cli
from climate_indices.exceptions import GoodnessOfFitWarning
from climate_indices.logging_config import ENV_LOG_LEVEL

_MM_UNITS = {"mm", "millimeters", "millimeter", "mm/dy"}
_INCH_UNITS = {"inches", "inch"}
_DISTRIBUTIONS = ("gamma", "pearson")


def _safe_write_path(value: str, flag: str) -> Path:
    """Resolve ``value``, refusing a relative path that escapes the working directory.

    These arguments are agent-supplied in benchmark runs, so a relative value
    like ``../../.ssh`` must not create or overwrite files outside the checkout
    (pythonsecurity:S8707). Absolute paths are documented usage
    (``--output-dir /tmp/cli``) and pass through after normalisation.
    """
    resolved = os.path.realpath(value)
    if not os.path.isabs(value):
        base = os.path.realpath(os.getcwd())
        if resolved != base and not resolved.startswith(base + os.sep):
            raise SystemExit(f"{flag} {value!r} is outside the working directory")
    return Path(resolved)


def _prepare(args: argparse.Namespace) -> None:
    """Trim a source nClimGrid NetCDF to one time span and prepare it like the xarray harness.

    Land mask comes from the finite cells of the first prepared time step
    (matching ``parallel_scaling.load_netcdf_grid``); every other step is
    forced to that same mask, and zeros become 0.01 mm for the gamma fit. The
    result is written as a fresh dataset with no inherited packing encoding,
    so 0.01 cannot be re-rounded on write.
    """
    with xr.open_dataset(args.source) as source:
        if args.var_name not in source.variables:
            raise SystemExit(f"{args.source}: no variable named {args.var_name!r}")
        da = source[args.var_name]
        missing_dims = {"time", "lat", "lon"} - set(da.dims)
        if missing_dims:
            raise SystemExit(f"{args.source}: {args.var_name} is missing dims {sorted(missing_dims)}")
        da = da.transpose("time", "lat", "lon").sel(time=slice(f"{args.start}-01-01", f"{args.end}-12-31")).load()

    if da.sizes["time"] == 0:
        raise SystemExit(f"{args.source}: no time steps between {args.start} and {args.end}")

    units = str(da.attrs.get("units", "")).strip().lower()
    if units in _INCH_UNITS:
        da = da * 25.4
    elif units not in _MM_UNITS:
        raise SystemExit(f"{args.source}: unsupported units {units!r} for {args.var_name}")

    values = da.astype(np.float32).values
    valid_cells = np.isfinite(values[0])
    if not valid_cells.any():
        raise SystemExit(f"{args.source}: no finite cell in the first prepared time step")
    values[:, ~valid_cells] = np.nan
    zero_count = int((values == 0).sum())
    values[values == 0] = np.float32(0.01)

    # the CLI's shared-array transport accepts (lat, lon, time) only, time last
    # (climate_indices.__main__._TRANSPORT_DIMENSIONS); the xarray harness
    # transposes its own input on read, so it accepts either order
    prepared = xr.Dataset(
        {"prcp": (("lat", "lon", "time"), np.moveaxis(values, 0, -1), {"units": "mm"})},
        coords={"time": da["time"].values, "lat": da["lat"].values, "lon": da["lon"].values},
    )
    target = _safe_write_path(args.target, "target")
    prepared.to_netcdf(target, engine="h5netcdf")

    land_cells = int(valid_cells.sum())
    print(f"source: {os.path.abspath(args.source)}")
    print(f"target: {target} sha256={_hash_file(str(target))}")
    print(
        f"grid: time={values.shape[0]} lat={values.shape[1]} lon={values.shape[2]}; "
        f"land={land_cells} of {valid_cells.size} cells ({land_cells / valid_cells.size:.1%}); "
        f"zeros replaced with 0.01 mm: {zero_count}"
    )


def _cli_argv(
    prepared: str,
    var_name: str,
    scale: int,
    calibration_start: int,
    calibration_end: int,
    output_base: str,
    multiprocessing: str | None = None,
) -> list[str]:
    """Build the argv the real CLI entry point (`climate_indices.__main__.main`) would receive.

    ``multiprocessing`` is left unset in every real benchmark run, so the CLI's
    own default (``all_but_one``) applies -- ADR-0002's worker count is not
    configurable in this benchmark. The override exists only so a test can pin
    ``single`` for a fast, deterministic run.
    """
    argv = [
        "--index",
        "spi",
        "--periodicity",
        "monthly",
        "--scales",
        str(scale),
        "--calibration_start_year",
        str(calibration_start),
        "--calibration_end_year",
        str(calibration_end),
        "--netcdf_precip",
        prepared,
        "--var_name_precip",
        var_name,
        "--output_file_base",
        output_base,
    ]
    if multiprocessing is not None:
        argv += ["--multiprocessing", multiprocessing]
    return argv


def _assert_equivalence(cli_path: str, cli_var: str, xarray_path: str) -> None:
    """Fail loudly unless the CLI's gamma output matches the xarray/Dask output, cell for cell.

    `parallel_scaling.load_netcdf_grid` rolls its spatial axes so its
    calibration preflight samples a land cell; the CLI never rolls. Sorting
    both sides by lat/lon compares by coordinate label, not storage position,
    so the roll cannot hide a mismatch -- this assumes unique lat/lon values,
    true for nClimGrid's regular grid; a grid with duplicate coordinates could
    make the roll-then-sort round trip pair the wrong cells, though the fit
    values at swapped cells are never bit-identical in practice, so it would
    fail loudly here rather than pass silently.
    """
    with xr.open_dataset(cli_path) as cli_ds, xr.open_dataset(xarray_path) as xarray_ds:
        cli_da = cli_ds[cli_var].transpose("time", "lat", "lon").sortby(["lat", "lon"]).astype(np.float32).load()
        xarray_var = next(iter(xarray_ds.data_vars))
        xarray_da = (
            xarray_ds[xarray_var].transpose("time", "lat", "lon").sortby(["lat", "lon"]).astype(np.float32).load()
        )

    if cli_da.shape != xarray_da.shape:
        raise SystemExit(f"equivalence: shape mismatch {cli_da.shape} (CLI) vs {xarray_da.shape} (xarray)")
    np.testing.assert_array_equal(cli_da["lat"].values, xarray_da["lat"].values)
    np.testing.assert_array_equal(cli_da["lon"].values, xarray_da["lon"].values)
    np.testing.assert_array_equal(cli_da.values, xarray_da.values)
    finite = int(np.isfinite(cli_da.values).sum())
    print(f"equivalence: CLI {cli_var} == xarray {xarray_var}, {finite} finite cells, bit for bit")


def _time_cli(args: argparse.Namespace) -> None:
    """Time `climate_indices.__main__.main`'s default multiprocessing.Pool path.

    Wraps two module-level functions the CLI looks up by name at call time, so
    the wrap is transparent to everything the Pool spawns: `_compute_write_index`
    (open + shared-memory copy + compute + write, called once per distribution)
    gives "total"; `_parallel_process` (Pool spawn + map, called once inside
    each `_compute_write_index` call) gives "compute". Spawned workers reimport
    `climate_indices.__main__` fresh in their own process, so they run the real,
    unwrapped functions -- only the parent-side timing is patched.
    """
    os.environ[ENV_LOG_LEVEL] = "WARNING"
    warnings.filterwarnings("ignore", category=GoodnessOfFitWarning)

    output_dir = _safe_write_path(args.output_dir, "--output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = str(output_dir / "nclimgrid")
    argv = _cli_argv(
        args.prepared,
        args.var_name,
        args.scale,
        args.calibration_start,
        args.calibration_end,
        output_base,
        multiprocessing=args.multiprocessing,
    )

    compute_samples: dict[str, list[float]] = {name: [] for name in _DISTRIBUTIONS}
    total_samples: dict[str, list[float]] = {name: [] for name in _DISTRIBUTIONS}

    original_write = cli._compute_write_index
    original_parallel = cli._parallel_process

    def _timed_write(request: cli._IndexRequest) -> tuple[str, str] | None:
        start = time.perf_counter()
        result = original_write(request)
        assert request.distribution is not None
        total_samples[request.distribution.value].append(time.perf_counter() - start)
        return result

    def _timed_parallel(request: cli._IndexRequest, arguments: dict[str, Any]) -> None:
        start = time.perf_counter()
        original_parallel(request, arguments)
        assert request.distribution is not None
        compute_samples[request.distribution.value].append(time.perf_counter() - start)

    with xr.open_dataset(args.prepared) as prepared_ds:
        grid_line = (
            f"grid: time={prepared_ds.sizes['time']} lat={prepared_ds.sizes['lat']} lon={prepared_ds.sizes['lon']}"
        )
    mode = args.multiprocessing or "all_but_one"
    workers = {"single": 1, "all": os.cpu_count() or 1}.get(mode, (os.cpu_count() or 1) - 1)

    print(f"checkout revision: {_revision()}")
    print(f"fixture: {os.path.abspath(args.prepared)} sha256={_hash_file(args.prepared)}")
    print(grid_line)
    print(_environment())
    print(f"process: workers={workers} (--multiprocessing {mode}); {args.repeat} timed runs after a warm-up")

    cli._compute_write_index = _timed_write
    cli._parallel_process = _timed_parallel
    try:
        cli.main(argv)  # warm-up

        gamma_var = f"spi_gamma_{args.scale:02d}"
        _assert_equivalence(f"{output_base}_{gamma_var}.nc", gamma_var, args.xarray_output)

        for name in _DISTRIBUTIONS:
            compute_samples[name].clear()
            total_samples[name].clear()
        for _ in range(args.repeat):
            cli.main(argv)
    finally:
        cli._compute_write_index = original_write
        cli._parallel_process = original_parallel

    for name in _DISTRIBUTIONS:
        compute = tuple(compute_samples[name])
        total = tuple(total_samples[name])
        print(
            f"\ncli {name} (multiprocessing.Pool, {workers} workers): "
            f"compute samples=[{_format_samples(compute)}] min={min(compute):.3f} s; "
            f"total samples=[{_format_samples(total)}] min={min(total):.3f} s"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="trim and mask a source nClimGrid NetCDF")
    prepare.add_argument("source", help="source NetCDF, e.g. the NODD nclimgrid_prcp.nc period-of-record object")
    prepare.add_argument("target", help="prepared NetCDF to write")
    prepare.add_argument("--var-name", default="prcp", help="precipitation variable in --source (default: prcp)")
    prepare.add_argument("--start", type=int, required=True, help="first calendar year to keep")
    prepare.add_argument("--end", type=int, required=True, help="last calendar year to keep")

    time_cmd = subparsers.add_parser("time", help="time the CLI's multiprocessing.Pool path")
    time_cmd.add_argument("prepared", help="output of the prepare subcommand")
    time_cmd.add_argument("--var-name", default="prcp", help="precipitation variable in --prepared (default: prcp)")
    time_cmd.add_argument("--scale", type=int, default=6, help="SPI timescale (default: 6)")
    time_cmd.add_argument("--calibration-start", type=int, default=1991, help="first calibration year (default: 1991)")
    time_cmd.add_argument("--calibration-end", type=int, default=2020, help="last calibration year (default: 2020)")
    time_cmd.add_argument("--repeat", type=int, default=3, help="timed runs after a warm-up (default: 3)")
    time_cmd.add_argument("--output-dir", required=True, help="directory for the CLI's output NetCDF files")
    time_cmd.add_argument(
        "--xarray-output", required=True, help="parallel_scaling.py --write-output result to check the CLI against"
    )
    time_cmd.add_argument(
        "--multiprocessing",
        choices=["single", "all_but_one", "all"],
        default=None,
        help="override the CLI's own --multiprocessing default (all_but_one); "
        "leave unset for the benchmark, pin 'single' for a fast test run",
    )

    args = parser.parse_args(argv)
    if args.command == "time" and args.repeat < 1:
        parser.error("--repeat must be at least 1")
    return args


def main() -> None:
    """Run the requested subcommand."""
    args = _parse_args()
    if args.command == "prepare":
        _prepare(args)
    else:
        _time_cli(args)


if __name__ == "__main__":
    # spawn (the macOS/Windows default) reimports this module in every worker
    # process, so anything outside this guard would run again in each of them
    main()
