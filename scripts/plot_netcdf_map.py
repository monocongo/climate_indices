#!/usr/bin/env python3
"""Save a quick map of one gridded NetCDF variable/time slice.

Example (Cartopy and matplotlib are in the dev dependency group)::

    uv run --group dev scripts/plot_netcdf_map.py \
        /path/to/nclimgrid_spi_spei_gamma_03.nc spi_03 \
        --time 2020-01 --output spi_2020-01.png

Omit --time to plot the last time step. Supports 1-D lat/lon coordinates;
Cartopy may download Natural Earth boundaries on first use (60 s network timeout).
"""

import argparse
import socket
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr


def main() -> None:
    """Plot one (lat, lon) slice of a NetCDF variable and print the PNG path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="NetCDF file")
    parser.add_argument("variable", help="data variable to plot")
    parser.add_argument("--time", help="date to select (e.g. 2020-01); defaults to last time step")
    parser.add_argument("--output", required=True, type=Path, help="PNG to write")
    args = parser.parse_args()

    if args.output.exists() and args.output.samefile(args.input):
        parser.error("output must not overwrite input")
    socket.setdefaulttimeout(60)  # Cartopy's Natural Earth download has no timeout of its own.

    with xr.open_dataset(args.input) as dataset:
        if args.variable not in dataset.data_vars:
            parser.error(f"unknown variable {args.variable!r}; available: {', '.join(dataset.data_vars)}")
        data = dataset[args.variable]
        if "time" in data.dims:
            try:
                data = data.sel(time=args.time) if args.time else data.isel(time=-1)
            except (KeyError, ValueError, IndexError) as exc:
                parser.error(f"cannot select time {args.time!r}: {exc}")
            if "time" in data.dims:  # A partial date such as 2020-01 can match several steps.
                if data.sizes["time"] != 1:
                    parser.error(f"--time {args.time!r} matches {data.sizes['time']} time steps; give a full date")
                data = data.isel(time=0)
        elif args.time:
            parser.error(f"{args.variable!r} has no time dimension")

        if set(data.dims) != {"lat", "lon"} or not {"lat", "lon"} <= set(data.indexes):
            parser.error(f"expected a single (lat, lon) grid with lat/lon coordinates; got {data.dims}")
        lon = data.indexes["lon"]
        if not (lon.is_monotonic_increasing or lon.is_monotonic_decreasing):
            parser.error("longitudes must be monotonic; dateline-crossing grids are not supported")

        title = data.attrs.get("long_name", args.variable)
        if "time" in data.coords:
            title += f" — {str(data.time.values)[:10]}"

        projection = ccrs.PlateCarree()
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
