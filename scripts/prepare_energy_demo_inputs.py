#!/usr/bin/env python3
"""Cache a small ERA5 subset for the offline energy demonstration.

Run ``uv run --group dev scripts/prepare_energy_demo_inputs.py`` from the root.
The anonymous ARCO-ERA5 store contains six-hourly samples on a conservatively
regridded 1.5 degree grid. Keep every second grid point in this European box
(12 cells); this is spatial subsampling, not a new spatial average. Preserve
both meteorological fields at six-hourly cadence so the notebook can apply
the nonlinear wind curve BEFORE daily averaging. This is not hourly ERA5.

Regeneration reads global time chunks (about 2 GB transferred, despite the
small output), caches each selected year, and requires network access. CI
uses only the committed NetCDF file, not this script or the remote store.
Data attribution and limitations are in notebooks/fixtures/energy_era5_README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SOURCE_URL = (
    "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/"
    "1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"
)
VARIABLES = {"2m_temperature": "temperature_k", "10m_wind_speed": "wind_speed_10m"}
LATITUDES = [48.0, 51.0, 54.0]
LONGITUDES = [3.0, 6.0, 9.0, 12.0]
FIRST_YEAR, LAST_YEAR = 1991, 2020
OUTPUT = Path("notebooks/fixtures/energy_era5_1991_2020.nc")


def validate(dataset: xr.Dataset, year: int) -> None:
    """Reject incomplete, misaligned, non-finite, or physically invalid inputs."""
    expected = pd.date_range(f"{year}-01-01", f"{year}-12-31T18:00", freq="6h")
    if not np.array_equal(dataset.time.values, expected.values):
        raise ValueError(f"{year}: expected a complete six-hourly calendar year")
    for coord, values in (("latitude", LATITUDES), ("longitude", LONGITUDES)):
        if not np.array_equal(dataset[coord].values, values):
            raise ValueError(f"{year}: unexpected {coord} coordinates")
    for name in VARIABLES.values():
        field = dataset[name]
        if field.dims != ("time", "latitude", "longitude") or not np.isfinite(field.values).all():
            raise ValueError(f"{year}: invalid {name} dimensions or missing values")
    if (dataset.temperature_k <= 0).any() or (dataset.wind_speed_10m < 0).any():
        raise ValueError(f"{year}: nonphysical temperature or wind speed")


def read_year(source: xr.Dataset, year: int, cache_dir: Path) -> xr.Dataset:
    """Read one selected year, reusing a validated local subset when present."""
    path = cache_dir / f"arco_era5_{year}.nc"
    if path.exists():
        with xr.open_dataset(path) as cached:
            subset: xr.Dataset = cached.load()
    else:
        # The store's coordinate floats differ slightly from exact degree values.
        subset = source[list(VARIABLES)].sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        subset = subset.sel(latitude=LATITUDES, longitude=LONGITUDES, method="nearest", tolerance=0.01)
        subset = subset.assign_coords(latitude=LATITUDES, longitude=LONGITUDES)
        subset = subset.rename(VARIABLES).transpose("time", "latitude", "longitude").load(num_workers=8)
        subset.temperature_k.attrs = {"units": "K", "long_name": "Six-hourly 2 m temperature"}
        subset.wind_speed_10m.attrs = {"units": "m s-1", "long_name": "Six-hourly 10 m wind speed"}
        validate(subset, year)
        subset.to_netcdf(path, engine="h5netcdf")
    validate(subset, year)
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/energy-demo/arco-cache"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    years = []
    with xr.open_zarr(SOURCE_URL, consolidated=True) as source:
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            print(f"reading {year}", flush=True)
            years.append(read_year(source, year, args.cache_dir))
    dataset = xr.concat(years, dim="time")
    dataset.attrs = {
        "title": "ERA5 energy demonstration: twelve European grid cells, six-hourly samples",
        "source": SOURCE_URL,
        "references": "https://doi.org/10.24381/cds.adbb2d47; https://github.com/google-research/arco-era5",
        "license": "CC-BY-4.0; see energy_era5_README.md for attribution and redistribution terms",
        "history": "Spatially subsampled ARCO-ERA5; coordinates rounded to exact degrees; no temporal averaging",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(
        args.output,
        engine="h5netcdf",
        encoding={name: {"dtype": "float32", "zlib": True, "complevel": 4} for name in dataset.data_vars},
    )
    provenance = {
        "source_url": SOURCE_URL,
        "source_variables": VARIABLES,
        "retrieved": datetime.now(timezone.utc).date().isoformat(),
        "period": [f"{FIRST_YEAR}-01-01T00:00", f"{LAST_YEAR}-12-31T18:00"],
        "latitude": LATITUDES,
        "longitude": LONGITUDES,
        "sampling": "six-hourly; every second 1.5-degree grid point; no new spatial or temporal averaging",
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "size_bytes": args.output.stat().st_size,
        "generator": "scripts/prepare_energy_demo_inputs.py",
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
