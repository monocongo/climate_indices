#!/usr/bin/env python3
"""Prepare the ERA5 fire-weather demonstration inputs from ARCO-ERA5.

Run from the repository root:
    uv run --group dev scripts/prepare_fire_demo_inputs.py

Source and provenance
    Reads the publicly readable ARCO-ERA5 Zarr store on Google Cloud Storage
    (``gcp-public-data-arco-era5``, an Amazon Web Services / Google Cloud
    redistribution of the Copernicus ERA5 reanalysis). No credentials are
    required: the store is read anonymously over HTTPS. The coarse
    1959-2022 six-hourly store (1.5 degrees, 13 pressure levels) is used
    because its chunks carry several time steps per request; the 0.25 degree
    hourly stores chunk one time step at a time and would make a CONUS subset
    economically pointless to fetch.

Pipeline
    1. Subset every variable to the CONUS box (25-50 N, 235-295 E) on the
       store's 1.5 degree equiangular grid and convert the requested
       235-295 E window to the -125..-65 E convention (the realized grid
       coverage is 235.5-294.0 E, -124.5..-66.0 E).
    2. Aggregate the six-hourly surface variables to daily values over
       ``SURFACE_YEARS``: ``tmean_c`` (mean), ``tmax_c`` (maximum), ``tmin_c``
       (minimum), ``precip_mm`` (sum of the 6-hour accumulations) and
       ``wind_speed_ms`` (mean of the 10 m wind speed).
    3. Aggregate the six-hourly level variables to daily means over the
       ``SEASON_START``-``SEASON_END`` fire season, then derive
       ``relative_humidity_percent`` from specific humidity, temperature and
       the pressure-level value itself, and ``height_agl_m`` from geopotential
       minus the surface geopotential.
    4. Derive a single surface-level relative humidity per cell and day from
       the lowest pressure level above the surface.

Output contract (``data/fire-demo/manifest.json`` is the machine-readable
record)
    ``surface_daily_2018_2020.nc`` holds ``tmean_c``, ``tmax_c``, ``tmin_c``,
    ``precip_mm`` and ``wind_speed_ms`` with dims (time, latitude, longitude)
    at daily cadence timestamped 12:00.
    ``levels_daily_2020_season.nc`` holds ``temperature_c``,
    ``relative_humidity_percent``, ``wind_speed_ms`` and ``height_agl_m`` with
    dims (time, level, latitude, longitude), plus the derived surface
    ``surface_relative_humidity_percent`` with dims (time, latitude, longitude).

Approximations this module makes, all recorded in the manifest
    - Relative humidity is derived from ERA5 specific humidity against the
      saturation vapor pressure (FAO-56 Equation 11) at the same level. The
      lowest level above the surface stands in for the 2 m relative humidity
      that ERA5 does not publish in this store.
    - The pressure-level data below the surface is ERA5's own extrapolation,
      so the derived surface humidity is approximate over high terrain.
    - Daily means stand in for the noon-local-standard-time observations the
      FWI System defines its weather inputs on. Timestamps are set to 12:00
      and this remains a documented approximation.

Each run caches its per-variable downloads under ``data/fire-demo/cache/``, so
regeneration offline is possible once the cache exists; delete the cache to
force a fresh download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices.pm_eto import saturation_vapor_pressure

SOURCE_URL = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
)
SOURCE_DESCRIPTION = (
    "ARCO-ERA5 1959-2022 six-hourly reanalysis, 1.5 degree equiangular grid, "
    "13 pressure levels, anonymously readable on Google Cloud Storage"
)
DOMAIN_LATITUDE = (25.0, 50.0)
DOMAIN_LONGITUDE = (235.0, 295.0)
SURFACE_YEARS = (2018, 2019, 2020)
SEASON_START = "2020-03-01"
SEASON_END = "2020-10-31"
SURFACE_VARIABLES = ("2m_temperature", "total_precipitation_6hr", "10m_wind_speed")
LEVEL_VARIABLES = ("temperature", "specific_humidity", "wind_speed", "geopotential")
STATIC_VARIABLE = "geopotential_at_surface"
GRAVITY = 9.80665
WATER_VAPOR_RATIO = 0.622
OUTPUT_SURFACE = "surface_daily_2018_2020.nc"
OUTPUT_LEVELS = "levels_daily_2020_season.nc"


def _open_source() -> xr.Dataset:
    """Open the ARCO-ERA5 store lazily over anonymous HTTPS."""
    return xr.open_zarr(SOURCE_URL, chunks={}, consolidated=True, decode_times=True)


def _select(dataset: xr.Dataset, name: str, start: str, end: str) -> xr.DataArray:
    """Subset one variable to the CONUS box and inclusive time window."""
    variable = dataset[name]
    selection: dict[str, slice] = {
        "latitude": slice(*DOMAIN_LATITUDE),
        "longitude": slice(*DOMAIN_LONGITUDE),
    }
    if "time" in variable.dims:
        selection["time"] = slice(start, end)
    selected = variable.sel(**selection)
    # the store indexes longitude 0-360; report it on the -180-180 convention
    longitude = ((selected["longitude"] + 180.0) % 360.0) - 180.0
    return selected.assign_coords(longitude=longitude)


def _to_daily_surface(dataset: xr.Dataset, year: int) -> xr.Dataset:
    """Aggregate the six-hourly surface variables of one year to a daily dataset.

    ``dataset`` carries the whole first day of the next year, so the 31
    December precipitation bin is complete; the daily rows are trimmed back to
    the requested year here.
    """
    daily = xr.Dataset()
    temperature = dataset["2m_temperature"] - 273.15
    daily["tmean_c"] = temperature.resample(time="1D").mean()
    daily["tmax_c"] = temperature.resample(time="1D").max()
    daily["tmin_c"] = temperature.resample(time="1D").min()
    # ERA5 accumulates precipitation over the preceding six hours, so the
    # accumulation stamped 06:00 covers 00:00-06:00. Binning with closed="right"
    # groups the four accumulations that end 06, 12, 18, and 24 UTC: one
    # calendar day. Tiny negative values are numerical noise and would fail the
    # fire recurrences' non-negativity check.
    daily["precip_mm"] = (
        (dataset["total_precipitation_6hr"] * 1000.0)
        .clip(min=0.0)
        .resample(time="1D", closed="right", label="left")
        .sum()
    )
    daily["wind_speed_ms"] = dataset["10m_wind_speed"].resample(time="1D").mean()
    daily = daily.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    # midday timestamps: the CFFWIS adapter warns when a daily coordinate
    # clearly does not sample noon, and these daily summaries stand in for the
    # noon observations the system is defined on.
    return daily.assign_coords(time=daily.time + pd.Timedelta(hours=12))


def _relative_humidity(
    temperature_celsius: xr.DataArray,
    specific_humidity: xr.DataArray,
    pressure_hpa: xr.DataArray,
) -> xr.DataArray:
    """Derive relative humidity from specific humidity at a pressure level.

    ``saturation_vapor_pressure`` returns kPa (FAO-56 Equation 11); the mixing
    ratio conversion below works in hPa, and the saturation specific humidity
    uses the standard 0.378 (1 - 0.622) virtual-temperature coefficient.
    """
    # the shared helper is NumPy-only, so apply_ufunc reattaches the coordinate labels
    saturation_hpa = (
        xr.apply_ufunc(
            saturation_vapor_pressure,
            temperature_celsius,
            dask="parallelized",
            output_dtypes=[np.float64],
        )
        * 10.0
    )
    saturation_specific_humidity = (
        WATER_VAPOR_RATIO * saturation_hpa / (pressure_hpa - (1.0 - WATER_VAPOR_RATIO) * saturation_hpa)
    )
    return (100.0 * specific_humidity / saturation_specific_humidity).clip(0.0, 100.0)


def _heights_agl(geopotential: xr.DataArray, surface_geopotential: xr.DataArray) -> xr.DataArray:
    """Convert geopotential at levels to height above the surface, in meters."""
    return (geopotential - surface_geopotential) / GRAVITY


def _lowest_above_ground(values: xr.DataArray, height_agl: xr.DataArray) -> xr.DataArray:
    """Select each column's value at its lowest level above the surface.

    The AGL heights are used only to choose the level; ``argmin`` skips the
    levels masked as below ground, and every atmospheric column has at least
    one level above its surface.
    """
    selected = height_agl.where(height_agl >= 0.0).argmin(dim="level", skipna=True)
    return values.isel(level=selected)


def _to_daily_levels(dataset: xr.Dataset) -> xr.Dataset:
    """Aggregate and derive the season's daily level fields."""
    levels = xr.Dataset()
    surface_geopotential = dataset[STATIC_VARIABLE]
    if "time" in surface_geopotential.dims:
        surface_geopotential = surface_geopotential.isel(time=0, drop=True)

    def daily_mean(values: xr.DataArray) -> xr.DataArray:
        averaged = values.resample(time="1D").mean()
        return averaged.assign_coords(time=averaged.time + pd.Timedelta(hours=12))

    temperature = daily_mean(dataset["temperature"] - 273.15)
    humidity = daily_mean(dataset["specific_humidity"])
    wind = daily_mean(dataset["wind_speed"])
    geopotential = daily_mean(dataset["geopotential"])

    height = _heights_agl(geopotential, surface_geopotential)
    relative_humidity = _relative_humidity(temperature, humidity, temperature["level"])
    levels["temperature_c"] = temperature
    levels["relative_humidity_percent"] = relative_humidity
    levels["wind_speed_ms"] = wind
    levels["height_agl_m"] = height
    levels["surface_relative_humidity_percent"] = _lowest_above_ground(relative_humidity, height)
    return levels


def _cache_fingerprint() -> str:
    """Short digest of the source and subset a cached download belongs to.

    Folding this into the cache file name means a changed source URL, subset
    window, or domain can never reuse a download made for the old settings.
    """
    settings = {
        "source_url": SOURCE_URL,
        "domain_latitude": DOMAIN_LATITUDE,
        "domain_longitude": DOMAIN_LONGITUDE,
        "surface_years": SURFACE_YEARS,
        "season": [SEASON_START, SEASON_END],
    }
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]


def _cache(cache_dir: Path, key: str, variable: str, builder: Callable[[], xr.DataArray]) -> xr.DataArray:
    """Return a cached variable selection, building it from the source on a miss."""
    path = cache_dir / f"{_cache_fingerprint()}-{key}.nc"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache_dir) as temporary:
            staged = Path(temporary) / path.name
            builder().to_netcdf(staged)
            os.replace(staged, path)
    return xr.load_dataset(path)[variable]


def _surface_inputs(dataset: xr.Dataset, cache_dir: Path) -> xr.Dataset:
    """Build the long daily surface record, one cached year at a time.

    Each year's slice ends one six-hourly stamp into the next year so the 31
    December precipitation bin is complete; the cache key spells the slice out
    so a changed window can never reuse the old download.
    """
    years = []
    for year in SURFACE_YEARS:
        start, end = f"{year}-01-01", f"{year + 1}-01-01"
        parts = {}
        for name in SURFACE_VARIABLES:
            parts[name] = _cache(
                cache_dir,
                f"{name}_{start}_{end}",
                name,
                lambda name=name, start=start, end=end: _select(dataset, name, start, end),
            )
        years.append(_to_daily_surface(xr.Dataset(parts), year))
    return xr.concat(years, dim="time")


def _level_inputs(dataset: xr.Dataset, cache_dir: Path) -> xr.Dataset:
    """Build the season's daily level record, cached one variable at a time."""
    parts = {}
    for name in LEVEL_VARIABLES:
        parts[name] = _cache(
            cache_dir,
            f"{name}_{SEASON_START}_{SEASON_END}",
            name,
            lambda name=name: _select(dataset, name, SEASON_START, SEASON_END),
        )
    parts[STATIC_VARIABLE] = _cache(
        cache_dir,
        STATIC_VARIABLE,
        STATIC_VARIABLE,
        lambda: _select(dataset, STATIC_VARIABLE, SEASON_START, SEASON_END),
    )
    return _to_daily_levels(xr.Dataset(parts))


def _add_units(dataset: xr.Dataset) -> xr.Dataset:
    """Attach CF units and plot-friendly long names to the prepared variables."""
    units = {
        "tmean_c": ("degC", "Daily mean 2 m air temperature"),
        "tmax_c": ("degC", "Daily maximum 2 m air temperature"),
        "tmin_c": ("degC", "Daily minimum 2 m air temperature"),
        "precip_mm": ("mm", "Daily precipitation"),
        "wind_speed_ms": ("m s-1", "10 m wind speed"),
        "temperature_c": ("degC", "Air temperature"),
        "relative_humidity_percent": ("percent", "Relative humidity"),
        "surface_relative_humidity_percent": ("percent", "Relative humidity at the lowest level above the surface"),
        "height_agl_m": ("m", "Geopotential height above ground"),
    }
    for name, (unit, long_name) in units.items():
        if name not in dataset:
            continue
        if name == "wind_speed_ms" and "level" in dataset[name].dims:
            long_name = "Wind speed at the pressure level"
        dataset[name].attrs["units"] = unit
        dataset[name].attrs["long_name"] = long_name
    dataset["latitude"].attrs.update(units="degrees_north", long_name="latitude")
    dataset["longitude"].attrs.update(units="degrees_east", long_name="longitude")
    if "level" in dataset:
        dataset["level"].attrs.update(units="hPa", long_name="pressure level")
    return dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(dataset: xr.Dataset, path: Path) -> None:
    """Write a prepared dataset through a temporary file, then atomically publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    try:
        _add_units(dataset).to_netcdf(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_inputs(output_dir: Path) -> dict[str, Any]:
    """Prepare and publish the demonstration inputs, returning the manifest."""
    cache_dir = output_dir / "cache"
    source = _open_source()
    try:
        surface = _surface_inputs(source, cache_dir)
        levels = _level_inputs(source, cache_dir)
    finally:
        source.close()

    surface_path = output_dir / OUTPUT_SURFACE
    levels_path = output_dir / OUTPUT_LEVELS
    _publish(surface, surface_path)
    _publish(levels, levels_path)

    return {
        "source_url": SOURCE_URL,
        "source_description": SOURCE_DESCRIPTION,
        "source_access": "anonymous HTTPS, no credentials",
        "domain": {
            "requested_latitude": list(DOMAIN_LATITUDE),
            "requested_longitude": [DOMAIN_LONGITUDE[0] - 360.0, DOMAIN_LONGITUDE[1] - 360.0],
            "realized_latitude": [float(surface.latitude.min()), float(surface.latitude.max())],
            "realized_longitude": [float(surface.longitude.min()), float(surface.longitude.max())],
        },
        "surface_years": list(SURFACE_YEARS),
        "season": [SEASON_START, SEASON_END],
        "aggregation": {
            "tmean_c": "daily mean of 6-hourly 2 m temperature",
            "tmax_c": "daily maximum of 6-hourly 2 m temperature",
            "tmin_c": "daily minimum of 6-hourly 2 m temperature",
            "precip_mm": "sum of the four 6-hour accumulations ending 06, 12, 18, and 24 UTC",
            "wind_speed_ms": "daily mean of 6-hourly 10 m wind speed",
            "level_fields": "daily mean of the 6-hourly pressure-level fields",
            "level_wind_speed_ms": "daily mean of 6-hourly wind speed on the 13 pressure levels",
        },
        "approximations": [
            "relative humidity is derived from specific humidity and temperature at the same level",
            "the lowest level above the surface stands in for the 2 m relative humidity ERA5 does not publish here",
            "daily summaries timestamped 12:00 stand in for noon local-standard-time observations",
            "pressure-level values below the surface are ERA5's own extrapolation",
        ],
        "artifacts": {name: {"sha256": _sha256(output_dir / name)} for name in (OUTPUT_SURFACE, OUTPUT_LEVELS)},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    """Prepare the demonstration inputs under ``data/fire-demo``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fire-demo"),
        help="directory that receives the prepared NetCDF files and manifest (default: data/fire-demo)",
    )
    arguments = parser.parse_args()
    output_dir: Path = arguments.output_dir
    manifest = prepare_inputs(output_dir)
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(f".json.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"prepared {manifest_path}")


if __name__ == "__main__":
    main()
