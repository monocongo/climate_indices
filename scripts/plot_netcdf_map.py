#!/usr/bin/env python3
"""Save a quick map of one gridded NetCDF variable/time slice.

Example (Cartopy and matplotlib are in the dev dependency group)::

    uv run --group dev scripts/plot_netcdf_map.py \
        --input /path/to/nclimgrid_spi_spei_gamma_03.nc --var spi_03 \
        --time 2020-01 --output spi_2020-01.png

    uv run --group dev scripts/plot_netcdf_map.py \
        --input /path/to/nclimgrid_spi_spei_gamma_03.nc --var spi_03 \
        --time 2020-01 --scale 3 --compare wwdt --output comparison.png

Add --compare wwdt to place the matching WestWide Drought Tracker CONUS archive
image beside a local SPI/SPEI map. --scale confirms the timescale inferred from
--var (e.g. spi_03) or supplies it for --var spi/spei. An unavailable
image fails without writing an output. NOAA/NCEI is not supported until an
SPI/SPEI image endpoint is verified. WWDT uses PRISM data and may use a
different Calibration Period and methodology; visual comparison is not validation.

Omit --time to plot the last time step. Supports 1-D lat/lon coordinates;
Cartopy may download Natural Earth boundaries on first use (60 s network timeout).
"""

import argparse
import re
import socket
from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

WWDT_TIMESCALES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 24, 30, 36, 48, 60, 72}


def _wwdt_image(variable: str, scale: int, month: str) -> tuple[str, np.ndarray]:
    """Fetch the WWDT CONUS archive PNG for an exact index/timescale/month."""
    if scale not in WWDT_TIMESCALES:
        raise ValueError(f"WWDT does not offer a {scale}-month timescale")
    if month < f"{1895 + (scale - 1) // 12:04d}-{(scale - 1) % 12 + 1:02d}":
        raise ValueError(f"WWDT has no complete {scale}-month accumulation for {month}")
    url = f"https://wrcc-archive.dri.edu/wwdt/images/ARCHIVE/{variable}{scale}/{month.replace('-', '')}_us_cl.png"
    try:
        with urlopen(url, timeout=15) as response:
            if response.headers.get_content_type() != "image/png":
                raise ValueError(f"WWDT returned non-PNG content for {url}")
            content = response.read(5_000_001)
            if len(content) > 5_000_000:
                raise ValueError(f"WWDT image exceeds 5 MB: {url}")
    except HTTPError as exc:
        if exc.code in (404, 410):
            raise ValueError(f"WWDT image unavailable for {variable.upper()}-{scale}, {month}: {url}") from exc
        raise ValueError(f"WWDT request failed (HTTP {exc.code}): {url}") from exc
    except (OSError, IncompleteRead) as exc:
        raise ValueError(f"WWDT request failed: {url}: {exc}") from exc
    try:
        return url, plt.imread(BytesIO(content), format="png")
    except (OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"WWDT image could not be decoded: {url}") from exc


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


def main() -> None:
    """Plot one (lat, lon) slice of a NetCDF variable and print the PNG path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="NetCDF file")
    parser.add_argument("--var", dest="variable", required=True, help="data variable to plot")
    parser.add_argument("--time", help="date to select (e.g. 2020-01); defaults to last time step")
    parser.add_argument("--output", required=True, type=Path, help="PNG to write")
    parser.add_argument("--compare", choices=("wwdt", "noaa"), help="external map to display beside local SPI/SPEI")
    parser.add_argument("--scale", type=int, help="monthly accumulation timescale (checked against --var)")
    args = parser.parse_args()

    if args.compare == "noaa":
        parser.error("NOAA/NCEI comparison unavailable: no verified SPI/SPEI image endpoint")
    if args.scale is not None and not args.compare:
        parser.error("--scale requires --compare")
    if args.output.exists() and args.output.samefile(args.input):
        parser.error("output must not overwrite input")
    socket.setdefaulttimeout(60)  # Cartopy's Natural Earth download has no timeout of its own.

    with xr.open_dataset(args.input) as dataset:
        if args.variable not in dataset.data_vars:
            parser.error(f"unknown variable {args.variable!r}; available: {', '.join(dataset.data_vars)}")
        data = select_time(dataset[args.variable], args.time, parser)

        if set(data.dims) != {"lat", "lon"} or not {"lat", "lon"} <= set(data.indexes):
            parser.error(f"expected a single (lat, lon) grid with lat/lon coordinates; got {data.dims}")
        lon = data.indexes["lon"]
        if not (lon.is_monotonic_increasing or lon.is_monotonic_decreasing):
            parser.error("longitudes must be monotonic; dateline-crossing grids are not supported")

        title = data.attrs.get("long_name", args.variable)
        if "time" in data.coords:
            title += f" — {str(data.time.values)[:10]}"

        comparison = None
        if args.compare:
            match = re.fullmatch(r"(spi|spei)(?:_(\d+))?", args.variable, flags=re.IGNORECASE)
            if not match or "time" not in data.coords:
                parser.error("comparison requires a time-dependent SPI/SPEI variable (e.g. spi_03)")
            inferred_scale = int(match[2]) if match[2] else None
            scale = args.scale if args.scale is not None else inferred_scale
            if scale is None or (inferred_scale is not None and scale != inferred_scale):
                parser.error("--scale must match the timescale in --var (or be supplied for spi/spei)")
            if "scale" in data.attrs and str(data.attrs["scale"]) != str(scale):
                parser.error(f"--scale/--var disagrees with variable scale metadata {data.attrs['scale']!r}")
            try:
                month = data.time.dt.strftime("%Y-%m").item()
                comparison = _wwdt_image(match[1].lower(), scale, month)
            except (AttributeError, ValueError) as exc:
                parser.error(str(exc))

        projection = ccrs.PlateCarree()
        if comparison:
            fig = plt.figure(figsize=(18, 7))
            ax = fig.add_subplot(121, projection=projection)
            reference_ax = fig.add_subplot(122)
            reference_ax.imshow(comparison[1])
            reference_ax.axis("off")
            reference_ax.set_title(f"WWDT CONUS (PRISM) — {match[1].upper()}-{scale} — {month}")
            fig.text(0.5, 0.01, comparison[0], ha="center", fontsize=8)
            title = f"Local {title} ({match[1].upper()}-{scale})"
        else:
            fig, ax = plt.subplots(figsize=(11, 6), subplot_kw={"projection": projection})
        plot_options = {"ax": ax, "transform": projection, "cbar_kwargs": {"label": args.variable, "shrink": 0.7}}
        if args.variable.lower().startswith(("spi", "spei")):
            plot_options.update(cmap="RdBu", vmin=-3, vmax=3)
        data.plot.pcolormesh(x="lon", y="lat", **plot_options)
        ax.set_extent(
            [float(data.lon.min()), float(data.lon.max()), float(data.lat.min()), float(data.lat.max())],
            crs=projection,
        )
        ax.coastlines(resolution="50m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
        ax.set_title(title)
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
