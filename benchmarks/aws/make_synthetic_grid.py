"""Write a large synthetic prepared grid for a grid-scale benchmark run.

The grid stages in `rust_vs_python.py` need prepared NetCDF inputs whose real
fixtures are deliberately not in the repository, which makes grid-scale runs
impossible to reproduce from a clone. This writes an equivalent-shape grid
instead: the same dimensions and the same preparation contract the loader
expects, so the library's gridded path is exercised at CONUS scale with no
external download. Values are synthetic, so results describe that code path
rather than any particular real dataset.

Writes ``prcp.nc`` and ``tavg.nc`` beside each other and prints their SHA-256,
matching the provenance the real full-grid artifact records.

Usage:
    python benchmarks/aws/make_synthetic_grid.py [rows] [cols] [outdir]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

START_YEAR = 1981
END_YEAR = 2024
CALIBRATION_START = 1991
CALIBRATION_END = 2020
SEED = 20261009


def sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks so a multi-gigabyte grid does not have to be held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_grid(
    path: Path, name: str, values: np.ndarray, time: pd.DatetimeIndex, lat: np.ndarray, lon: np.ndarray
) -> None:
    """Write one variable as a time/lat/lon NetCDF, float32 to halve the I/O."""
    dataset = xr.Dataset(
        {name: (("time", "lat", "lon"), values.astype("float32"))},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"units": "mm" if name == "prcp" else "degreesC"},
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def main(rows: int, cols: int, outdir: Path) -> None:
    """Generate a prepared precipitation and temperature grid of ``rows`` x ``cols`` cells."""
    outdir.mkdir(parents=True, exist_ok=True)
    time = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-01", freq="MS")
    months = time.size
    lat = np.linspace(25.0, 49.0, rows)
    lon = np.linspace(-125.0, -101.0, cols)
    cells = rows * cols
    print(
        f"grid {rows} x {cols} = {cells:,} cells, {months} months "
        f"({START_YEAR}-{END_YEAR}); calibration {CALIBRATION_START}-{CALIBRATION_END}",
        flush=True,
    )

    rng = np.random.default_rng(SEED)

    # Precipitation: a gamma draw per cell, as mm. Never zero, so the loader's
    # zero->0.01 mm substitution has nothing to do here.
    precipitation = rng.gamma(shape=2.0, scale=15.0, size=(months, rows, cols))

    # Temperature: a seasonal cycle plus per-cell offset and noise, in degrees C.
    seasonal = 10.0 * np.sin((np.arange(months) % 12) * np.pi / 6.0)[:, None, None]
    latitude_effect = ((49.0 - lat) / 24.0 * 10.0)[None, :, None]
    temperature = 15.0 + seasonal - latitude_effect + rng.normal(0.0, 2.0, size=(months, rows, cols))

    for name, values in (("prcp", precipitation), ("tavg", temperature)):
        path = outdir / f"{name}.nc"
        write_grid(path, name, values, time, lat, lon)
        print(f"{path}: {path.stat().st_size / 1e9:.2f} GB, sha256 {sha256(path)}", flush=True)

    print("SYNTHETIC_GRID_DONE", flush=True)


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 596,
        int(sys.argv[2]) if len(sys.argv) > 2 else 1385,
        Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/synth"),
    )
