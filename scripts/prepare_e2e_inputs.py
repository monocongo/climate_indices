#!/usr/bin/env python3
"""Generate the notebook's NetCDF and Zarr inputs from pinned nClimGrid examples.

Run from the repository root:
    uv run --with h5py --with zarr scripts/prepare_e2e_inputs.py

Source and provenance
    Downloads paired monthly precipitation/PET NetCDF files from SOURCE_COMMIT,
    a pinned commit of monocongo/example_climate_indices, itself a reduced
    sample of NOAA/NCEI nClimGrid data. See
    docs/research/nclimgrid-acquisition-and-redistribution.md for the upstream
    provenance and attribution constraints this sample inherits. Each download
    is cached under data/e2e/source/ and its SHA-256 is checked against
    SOURCES before every use, so a stale or tampered cache fails loudly
    instead of silently feeding bad data downstream.

Pipeline
    1. Subset both variables to the complete 1980-01 through 2016-12 monthly
       record and verify the resulting time coordinate has no gaps or extra
       months.
    2. Verify declared units are millimeters, transpose to (time, lat, lon),
       and write raw_precipitation.nc / raw_pet.nc.
    3. Call clean_and_prepare_inputs() (defined below) to align
       precipitation and PET on identical coordinates, derive an explanatory
       wb = precip - pet water-balance variable, and write a consolidated
       Zarr v2 store (cache_prepared_input.zarr) with the full time series in
       one chunk (required by xarray_adapter's Dask validation; see
       docs/adr/0003-dask-time-dimension-single-chunk.md) and 10x10 spatial
       blocks, giving multiple independent blocks on the 38x87 example grid.
    4. Reopen the prepared store and re-verify values, dims, and chunking
       against the freshly written raw NetCDF before trusting it.

Output contract (data/e2e/current/manifest.json is the machine-readable record)
    precip, pet, and wb share dims (time, lat, lon) and units "mm"; time runs
    1980-01-01 through 2016-12-01 monthly. The store opens with
    xr.open_zarr(path, consolidated=True) and has one time chunk plus 10x10
    spatial chunks. manifest.json additionally records the source URLs and
    checksums, the calibration period, and the realized dimensions/chunks.

Each run writes a private generation under data/e2e/generations/ and only
atomically switches data/e2e/current after validation. Consumers must resolve
current once before opening any artifacts. Rerunning needs no network access
once data/e2e/source/ holds checksum-valid downloads.
"""

import hashlib
import json
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError

SOURCE_COMMIT = "ae57c488af832c1ebfdf864c8ed7d16636e2e36f"
SOURCE_URL = f"https://raw.githubusercontent.com/monocongo/example_climate_indices/{SOURCE_COMMIT}/example/input"
SOURCES = {
    "prcp": "31689a564aa56993f5280a84050e32ad310ebc3b3b44c062ae846aba24ca7276",
    "pet": "caf705806052724f6db5f588b2c9347a8f632496896d2380cfa120fd7a53885d",
}
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "e2e"
CANONICAL_GRID = {"lat": 38, "lon": 87}
SPATIAL_CHUNK = 10


def _validate_monthly_time(time_values: np.ndarray) -> pd.DatetimeIndex:
    """Return complete month-start or month-end timestamps."""
    try:
        time = pd.DatetimeIndex(time_values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CoordinateValidationError(
            "Time coordinate must contain supported datetime values.",
            coordinate_name="time",
            reason="not datetime-like",
        ) from exc
    if time.empty:
        raise CoordinateValidationError(
            "Time coordinate must not be empty.", coordinate_name="time", reason="empty coordinate"
        )
    if time.is_month_start.all():
        expected_time = pd.date_range(time[0], periods=time.size, freq="MS")
    elif time.is_month_end.all():
        expected_time = pd.date_range(time[0], periods=time.size, freq=pd.offsets.MonthEnd())
    else:
        expected_time = pd.DatetimeIndex([])
    if not time.equals(expected_time):
        raise CoordinateValidationError(
            "Time coordinate must be a complete, chronological sequence of monthly "
            "month-start or month-end timestamps.",
            coordinate_name="time",
            reason="non-monotonic, irregular, or gapped monthly timestamps",
        )
    return time


def clean_and_prepare_inputs(precip_path: Path, pet_path: Path, zarr_prepared_path: Path) -> None:
    """Prepare aligned monthly totals in mm for blockwise SPI/SPEI computation.

    Args:
        precip_path: NetCDF file containing ``pr(time, lat, lon)`` in mm.
        pet_path: NetCDF file containing ``pet(time, lat, lon)`` in mm.
        zarr_prepared_path: Prepared Zarr store to create or replace.
    """
    with (
        xr.open_dataset(precip_path, chunks={"time": 1}) as ds_precip,
        xr.open_dataset(pet_path, chunks={"time": 1}) as ds_pet,
    ):
        pr, pet = xr.align(ds_precip["pr"], ds_pet["pet"], join="exact")
        if pr.attrs.get("units") != "mm" or pet.attrs.get("units") != "mm":
            raise InvalidArgumentError(
                "Inputs must contain monthly totals with units='mm'.",
                argument_name="units",
                argument_value=f"pr={pr.attrs.get('units')!r}, pet={pet.attrs.get('units')!r}",
                valid_values="mm",
            )
        _validate_monthly_time(pr["time"].values)
        ds_clean = xr.Dataset({"precip": pr, "pet": pet, "wb": pr - pet}).transpose("time", "lat", "lon")
        ds_clean["wb"].attrs = {"long_name": "Precipitation minus PET, monthly total", "units": "mm"}
        # Full time series per block; 10x10 gives multiple tasks on the small example grid.
        ds_clean = ds_clean.drop_encoding().chunk({"time": -1, "lat": 10, "lon": 10})
        print(f"Saving prepared inputs: {zarr_prepared_path}")
        ds_clean.to_zarr(zarr_prepared_path, mode="w", zarr_format=2, consolidated=True)


def _boundary_chunks(size: int, chunk_size: int) -> tuple[int, ...]:
    """Return dask's chunk-length sequence for a dimension, including the boundary remainder."""
    full, remainder = divmod(size, chunk_size)
    return (chunk_size,) * full + ((remainder,) if remainder else ())


def _cache_source(source_dir: Path, name: str, checksum: str) -> Path:
    """Return a checksum-valid source download."""
    path = source_dir / f"nclimgrid_lowres_{name}.nc"
    url = f"{SOURCE_URL}/{path.name}"
    if not path.exists():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=source_dir, prefix=f".{path.name}.", suffix=".download", delete=False
            ) as target:
                temporary = Path(target.name)
                with urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, target)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"SHA-256 mismatch: {url}")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"SHA-256 mismatch: {path}; remove it and rerun to download again.")
    return path


def prepare_inputs(output_dir: Path = DATA_DIR, expected_grid: dict[str, int] | None = None) -> None:
    """Download, normalize, validate, and atomically publish monthly inputs.

    Args:
        output_dir: Directory for shared source downloads and input generations.
        expected_grid: Canonical {"lat", "lon"} sizes the prepared store must match.
            Defaults to CANONICAL_GRID; tests substitute a smaller grid.
    """
    expected_grid = expected_grid or CANONICAL_GRID
    source_dir = output_dir / "source"
    generations_dir = output_dir / "generations"
    source_dir.mkdir(parents=True, exist_ok=True)
    generations_dir.mkdir(exist_ok=True)
    arrays = {}
    generation_id = uuid.uuid4().hex
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "period": ["1980-01-01", "2016-12-01"],
        "calibration_period": [1981, 2010],
        "sources": {},
    }
    for name, checksum in SOURCES.items():
        path = _cache_source(source_dir, name, checksum)
        with xr.open_dataset(path, engine="h5netcdf") as source:
            values = source[name].sel(time=slice("1980-01-01", "2016-12-31"))
            if values.attrs.get("units") not in {"millimeter", "millimeters", "mm"}:
                raise ValueError(f"Unexpected units for {name}: {values.attrs.get('units')}")
            # Both sources already contain monthly totals in mm: no numerical conversion.
            values = values.transpose("time", "lat", "lon").load().drop_encoding()
        np.testing.assert_array_equal(values.time.values, pd.date_range("1980-01-01", "2016-12-01", freq="MS").values)
        if np.isinf(values).any() or (values < 0).any() or not np.isfinite(values).any():
            raise ValueError(f"Invalid values in {name}; expected nonnegative totals and a NaN mask.")
        values.attrs["units"] = "mm"
        arrays[name] = values
        manifest["sources"][name] = {"url": f"{SOURCE_URL}/{path.name}", "sha256": checksum}

    pr, pet = xr.align(arrays["prcp"], arrays["pet"], join="exact")
    ds = xr.Dataset({"pr": pr, "pet": pet})
    ds.attrs = {
        "title": "nClimGrid monthly example inputs, 1980-2016",
        "source": SOURCE_URL,
        "history": "Selected complete years 1980-2016; renamed prcp to pr; transposed to time, lat, lon; units normalized to mm.",
    }
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=f".{generation_id}.") as temporary_dir:
        staging_dir = Path(temporary_dir)
        for name, filename in (("pr", "raw_precipitation.nc"), ("pet", "raw_pet.nc")):
            path = staging_dir / filename
            ds[[name]].to_netcdf(
                path,
                engine="h5netcdf",
                encoding={name: {"zlib": True, "complevel": 4, "chunksizes": (1, ds.sizes["lat"], ds.sizes["lon"])}},
            )
            with xr.open_dataset(path) as saved:
                xr.testing.assert_equal(saved[name], ds[name])

        prepared_path = staging_dir / "cache_prepared_input.zarr"
        clean_and_prepare_inputs(staging_dir / "raw_precipitation.nc", staging_dir / "raw_pet.nc", prepared_path)
        with xr.open_zarr(prepared_path, consolidated=True) as prepared:
            np.testing.assert_allclose(prepared.precip, pr)
            np.testing.assert_allclose(prepared.pet, pet)
            np.testing.assert_allclose(prepared.wb, pr - pet)
            if prepared.precip.dims != ("time", "lat", "lon"):
                raise ValueError(f"Unexpected dims: {prepared.precip.dims}")
            if prepared.sizes["lat"] != expected_grid["lat"] or prepared.sizes["lon"] != expected_grid["lon"]:
                raise ValueError(
                    f"Unexpected grid {prepared.sizes['lat']}x{prepared.sizes['lon']}; "
                    f"expected {expected_grid['lat']}x{expected_grid['lon']}"
                )
            expected_chunks = (
                (ds.sizes["time"],),
                _boundary_chunks(expected_grid["lat"], SPATIAL_CHUNK),
                _boundary_chunks(expected_grid["lon"], SPATIAL_CHUNK),
            )
            if prepared.precip.chunks != expected_chunks:
                raise ValueError(f"Unexpected chunk layout {prepared.precip.chunks}; expected {expected_chunks}")
            manifest["dimensions"] = dict(prepared.sizes)
            manifest["chunks"] = list(prepared.precip.encoding["chunks"])
        (staging_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        generation_path = generations_dir / generation_id
        staging_dir.rename(generation_path)
        current = output_dir / "current"
        pending_current = output_dir / f".current-{generation_id}"
        try:
            pending_current.symlink_to(generation_path.relative_to(output_dir), target_is_directory=True)
            pending_current.replace(current)
        finally:
            pending_current.unlink(missing_ok=True)
    print(f"Validated NetCDF + Zarr inputs in {output_dir / 'current'} ({dict(ds.sizes)})")


if __name__ == "__main__":
    prepare_inputs()
