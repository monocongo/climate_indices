#!/usr/bin/env python3
"""Save a quick map of one gridded NetCDF variable/time slice.

Example (Cartopy and matplotlib are in the dev dependency group)::

    uv run --group dev scripts/plot_netcdf_map.py \
        --input /path/to/nclimgrid_spi_spei_gamma_03.nc --var spi_03 \
        --time 2020-01 --output spi_2020-01.png

    uv run --group dev scripts/plot_netcdf_map.py \
        --input /path/to/nclimgrid_spi_spei_gamma_03.nc --var spi_03 \
        --time 2020-01 --scale 3 --compare ncei --output comparison.png

Add --compare ncei to plot the matching NCEI/NIDIS nClimGrid monthly SPI/SPEI
slice beside the local grid with identical color limits. --scale confirms the
timescale inferred from --var (e.g. spi_03 or spi_gamma_03) or supplies it for
--var spi/spei. Unsuffixed variable names use Gamma unless their distribution
metadata specifies Pearson. NCEI uses the 1895–2014 Calibration Period; different
inputs or settings can produce different values. This is a visual reproduction
check, not independent scientific validation. An unavailable month fails without
writing an output.

Omit --time to plot the last time step. Supports 1-D lat/lon coordinates;
Cartopy may download Natural Earth boundaries on first use (60 s network timeout).
"""

import argparse
import re
import socket
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import fsspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

NCEI_TIMESCALES = {1, 2, 3, 6, 9, 12, 24, 36, 48, 60, 72}
NCEI_BASE = "https://www.ncei.noaa.gov/pub/data/nidis/indices/nclimgrid-monthly"


def _ncei_grid(index: str, distribution: str, scale: int, month: str) -> tuple[str, xr.DataArray]:
    """Read the exact monthly grid via HTTP ranges, without downloading the full archive."""
    if scale not in NCEI_TIMESCALES:
        raise ValueError(f"NCEI does not offer a {scale}-month timescale")
    if month < f"{1895 + (scale - 1) // 12:04d}-{(scale - 1) % 12 + 1:02d}":
        raise ValueError(f"NCEI has no complete {scale}-month accumulation for {month}")
    name = f"{index}_{scale:02d}"
    url = f"{NCEI_BASE}/{index}-{distribution}/nclimgrid-{index}-{distribution}-{scale:02d}.nc"
    try:
        with fsspec.open(url, block_size=2**20, timeout=15, allow_redirects=False) as file:
            if file.size is None or file.size > 4_000_000_000:
                raise ValueError("NCEI file has no known size or exceeds 4 GB")
            with xr.open_dataset(file, engine="h5netcdf") as dataset:
                if name not in dataset or month not in set(dataset.time.dt.strftime("%Y-%m").values):
                    raise ValueError(f"NCEI has no {index.upper()}-{scale} data for {month}")
                grid = dataset[name].sel(time=f"{month}-01").load()
                if set(grid.dims) != {"lat", "lon"}:
                    raise ValueError(f"NCEI returned unexpected grid dimensions: {grid.dims}")
                if not np.isfinite(grid.values).any():
                    raise ValueError(f"NCEI has no valid {index.upper()}-{scale} data for {month}")
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"NCEI retrieval failed for {url}: {exc}") from exc
    return url, grid


def select_time(data: xr.DataArray, time: str | None, parser: argparse.ArgumentParser) -> xr.DataArray:
    """Reduce ``data`` to the single time step ``time`` names, or the last one."""
    if "time" not in data.dims:
        if time:
            parser.error(f"{data.name!r} has no time dimension")
        return data
    try:
        data = data.sel(time=time) if time else data.isel(time=-1)
    except (KeyError, ValueError, IndexError) as exc:
        parser.error(f"cannot select time {time!r}: {exc}")
    if "time" in data.dims:  # A partial date such as 2020-01 can match several steps.
        if data.sizes["time"] != 1:
            parser.error(f"--time {time!r} matches {data.sizes['time']} time steps; give a full date")
        data = data.isel(time=0)
    return data


def _select_grid(dataset: xr.Dataset, variable: str, time: str | None, parser: argparse.ArgumentParser) -> xr.DataArray:
    """Return the single (lat, lon) slice of ``variable`` at ``time``."""
    if variable not in dataset.data_vars:
        parser.error(f"unknown variable {variable!r}; available: {', '.join(dataset.data_vars)}")
    data = select_time(dataset[variable], time, parser)
    if set(data.dims) != {"lat", "lon"} or not {"lat", "lon"} <= set(data.indexes):
        parser.error(f"expected a single (lat, lon) grid with lat/lon coordinates; got {data.dims}")
    lon = data.indexes["lon"]
    if not (lon.is_monotonic_increasing or lon.is_monotonic_decreasing):
        parser.error("longitudes must be monotonic; dateline-crossing grids are not supported")
    return data


def _comparison_scale(
    match: re.Match[str], data: xr.DataArray, scale: int | None, parser: argparse.ArgumentParser
) -> int:
    """Check the requested timescale against the variable name and metadata."""
    inferred_scale = int(match[3]) if match[3] else None
    if scale is None:
        scale = inferred_scale
    if scale is None or (inferred_scale is not None and scale != inferred_scale):
        parser.error("--scale must match the timescale in --var (or be supplied for spi/spei)")
    if "scale" in data.attrs:
        try:
            consistent = float(data.attrs["scale"]) == scale
        except (TypeError, ValueError):
            consistent = False
        if not consistent:
            parser.error(f"--scale/--var disagrees with variable scale metadata {data.attrs['scale']!r}")
    return scale


def _ncei_comparison(
    dataset: xr.Dataset, data: xr.DataArray, variable: str, scale: int | None, parser: argparse.ArgumentParser
) -> tuple[str, str, str, xr.DataArray]:
    """Check the local grid and return its label, month, NCEI URL, and matching grid."""
    match = re.fullmatch(r"(spi|spei)(?:_(gamma|pearson|loglogistic))?(?:_(\d+))?", variable, flags=re.IGNORECASE)
    if not match or "time" not in dataset[variable].dims:
        parser.error("comparison requires a time-dependent SPI/SPEI variable (e.g. spi_03)")
    scale = _comparison_scale(match, data, scale, parser)
    try:
        months = dataset[variable].time.dt.strftime("%Y-%m").values
        if len(set(months)) != months.size:
            raise ValueError("comparison requires monthly time steps; NCEI timescales are months")
        month = data.time.dt.strftime("%Y-%m").item()
        distribution = str(match[2] or data.attrs.get("distribution", "gamma")).lower()
        if match[2] and "distribution" in data.attrs and distribution != str(data.attrs["distribution"]).lower():
            raise ValueError("--var disagrees with variable distribution metadata")
        if distribution not in ("gamma", "pearson"):
            raise ValueError(f"NCEI does not offer the {distribution} distribution")
        url, grid = _ncei_grid(match[1].lower(), distribution, scale, month)
    except (AttributeError, ValueError) as exc:
        parser.error(str(exc))
    return f"{match[1].upper()}-{scale} ({distribution.title()})", month, url, grid


def main() -> None:
    """Plot one (lat, lon) slice of a NetCDF variable and print the PNG path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="NetCDF file")
    parser.add_argument("--var", dest="variable", required=True, help="data variable to plot")
    parser.add_argument("--time", help="date to select (e.g. 2020-01); defaults to last time step")
    parser.add_argument("--output", required=True, type=Path, help="PNG to write")
    parser.add_argument("--compare", choices=("ncei",), help="NCEI/NIDIS nClimGrid map beside local SPI/SPEI")
    parser.add_argument("--scale", type=int, help="monthly accumulation timescale (checked against --var)")
    args = parser.parse_args()

    if args.scale is not None and not args.compare:
        parser.error("--scale requires --compare")
    if args.output.exists() and args.output.samefile(args.input):
        parser.error("output must not overwrite input")
    socket.setdefaulttimeout(60)  # Cartopy's Natural Earth download has no timeout of its own.

    with xr.open_dataset(args.input) as dataset:
        data = _select_grid(dataset, args.variable, args.time, parser)

        title = data.attrs.get("long_name", args.variable)
        if "time" in data.coords:
            title += f" — {str(data.time.values)[:10]}"

        projection = ccrs.PlateCarree()
        if args.compare:
            label, month, url, reference = _ncei_comparison(dataset, data, args.variable, args.scale, parser)
            fig = plt.figure(figsize=(18, 7))
            ax = fig.add_subplot(121, projection=projection)
            reference_ax = fig.add_subplot(122, projection=projection)
            title = f"Local {title} ({label})"
            fig.text(0.5, 0.01, url, ha="center", fontsize=8)
        else:
            fig, ax = plt.subplots(figsize=(11, 6), subplot_kw={"projection": projection})
        plot_options = {"ax": ax, "transform": projection, "cbar_kwargs": {"label": args.variable, "shrink": 0.7}}
        if args.variable.lower().startswith(("spi", "spei")):
            plot_options.update(cmap="RdBu", vmin=-3, vmax=3)
        for map_ax, grid, heading in (
            [(ax, data, title), (reference_ax, reference, f"NCEI/NIDIS nClimGrid — {label} — {month}")]
            if args.compare
            else [(ax, data, title)]
        ):
            grid.plot.pcolormesh(x="lon", y="lat", **{**plot_options, "ax": map_ax})
            map_ax.set_extent(
                [float(data.lon.min()), float(data.lon.max()), float(data.lat.min()), float(data.lat.max())],
                crs=projection,
            )
            map_ax.coastlines(resolution="50m", linewidth=0.6)
            map_ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
            map_ax.set_title(heading)
        try:  # Boundaries are fetched while drawing, so download failures surface here.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.output, format="png", dpi=150, bbox_inches="tight")
        except OSError as exc:
            parser.error(f"cannot save map: {exc}")
        finally:
            plt.close(fig)

    print(args.output)


if __name__ == "__main__":
    main()
