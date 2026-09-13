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
    3. Call clean_and_prepare_inputs() (end_to_end_example.py) to align
       precipitation and PET on identical coordinates, derive an explanatory
       wb = precip - pet water-balance variable, and write a consolidated
       Zarr v2 store (cache_prepared_input.zarr) with the full time series in
       one chunk (required by xarray_adapter's Dask validation; see
       docs/adr/0003-dask-time-dimension-single-chunk.md) and 10x10 spatial
       blocks, giving multiple independent blocks on the 38x87 example grid.
    4. Reopen the prepared store and re-verify values, dims, and chunking
       against the freshly written raw NetCDF before trusting it.

Output contract (data/e2e/manifest.json is the machine-readable record)
    precip, pet, and wb share dims (time, lat, lon) and units "mm"; time runs
    1980-01-01 through 2016-12-01 monthly. The store opens with
    xr.open_zarr(path, consolidated=True) and has one time chunk plus 10x10
    spatial chunks. manifest.json additionally records the source URLs and
    checksums, the calibration period, and the realized dimensions/chunks.

Generated files live in data/e2e/ (ignored by Git) and are replaced on
reruns. Rerunning this script is the one documented way to regenerate or
re-verify the sample; it needs no network access once data/e2e/source/ holds
checksum-valid downloads.
"""

import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr
from end_to_end_example import clean_and_prepare_inputs

SOURCE_COMMIT = "ae57c488af832c1ebfdf864c8ed7d16636e2e36f"
SOURCE_URL = f"https://raw.githubusercontent.com/monocongo/example_climate_indices/{SOURCE_COMMIT}/example/input"
SOURCES = {
    "prcp": "31689a564aa56993f5280a84050e32ad310ebc3b3b44c062ae846aba24ca7276",
    "pet": "caf705806052724f6db5f588b2c9347a8f632496896d2380cfa120fd7a53885d",
}
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "e2e"


def prepare_inputs(output_dir: Path = DATA_DIR) -> None:
    """Download, normalize, and verify monthly precipitation/PET inputs.

    Args:
        output_dir: Directory for source downloads, derived files, and manifest.
    """
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "period": ["1980-01-01", "2016-12-01"],
        "calibration_period": [1981, 2010],
        "sources": {},
    }
    for name, checksum in SOURCES.items():
        path = source_dir / f"nclimgrid_lowres_{name}.nc"
        url = f"{SOURCE_URL}/{path.name}"
        if not path.exists():
            temporary = path.with_suffix(".download")
            try:
                with urlopen(url, timeout=120) as response, temporary.open("wb") as target:
                    shutil.copyfileobj(response, target)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                    raise ValueError(f"SHA-256 mismatch: {url}")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"SHA-256 mismatch: {path}; remove it and rerun to download again.")
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
        manifest["sources"][name] = {"url": url, "sha256": checksum}

    pr, pet = xr.align(arrays["prcp"], arrays["pet"], join="exact")
    ds = xr.Dataset({"pr": pr, "pet": pet})
    ds.attrs = {
        "title": "nClimGrid monthly example inputs, 1980-2016",
        "source": SOURCE_URL,
        "history": "Selected complete years 1980-2016; renamed prcp to pr; transposed to time, lat, lon; units normalized to mm.",
    }
    for name, filename in (("pr", "raw_precipitation.nc"), ("pet", "raw_pet.nc")):
        ds[[name]].to_netcdf(
            output_dir / filename,
            engine="h5netcdf",
            encoding={name: {"zlib": True, "complevel": 4, "chunksizes": (1, ds.sizes["lat"], ds.sizes["lon"])}},
        )
        with xr.open_dataset(output_dir / filename) as saved:
            xr.testing.assert_equal(saved[name], ds[name])

    prepared_path = output_dir / "cache_prepared_input.zarr"
    clean_and_prepare_inputs(output_dir / "raw_precipitation.nc", output_dir / "raw_pet.nc", prepared_path)
    with xr.open_zarr(prepared_path, consolidated=True) as prepared:
        np.testing.assert_allclose(prepared.precip, pr)
        np.testing.assert_allclose(prepared.pet, pet)
        np.testing.assert_allclose(prepared.wb, pr - pet)
        assert prepared.precip.dims == ("time", "lat", "lon")
        assert prepared.precip.chunks[0] == (ds.sizes["time"],)
        manifest["dimensions"] = dict(prepared.sizes)
        manifest["chunks"] = list(prepared.precip.encoding["chunks"])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated NetCDF + Zarr inputs in {output_dir} ({dict(ds.sizes)})")


if __name__ == "__main__":
    prepare_inputs()
