#!/usr/bin/env python3
"""Prepare the pinned Vieira Passos et al. (2024) subset without importing climate_indices.

``daily_spei`` adapts notebook cell 63ae9174 (monthly-to-daily preparation) and
``author_counts`` adapts the deterministic arithmetic of cell 93ef7024 (LMF) from
https://github.com/m-passos/hydroclimatic-hazards-hotspots at commit
6c88f255ee3034279dcebe2a2fc40c954dbe9575. No downloaded code or significance loop runs.
Existing output directories are refused; regenerate into a new directory for comparison.

The adapted code is MIT-licensed; the notice travels with this file because the
sdist excludes tests/fixture/compound/NOTICE:

    Copyright (c) 2023 Marlon Vieira Passos

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    THE SOFTWARE.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import xarray as xr

AUTHOR_COMMIT = "6c88f255ee3034279dcebe2a2fc40c954dbe9575"
DATASET_DOI = "10.6084/m9.figshare.25189118.v2"
SOURCES: dict[str, dict[str, int | str]] = {
    "SPEI12.nc": {
        "file_id": 44481290,
        "sha256": "bb2b32bc5c6d76ce3257a0b211696128ad6da13ae737650a1376524788e88dce",
    },
    "heatwave_SMHI.nc": {
        "file_id": 44481308,
        "sha256": "c0428087c22cd6586f77af83592185ca513d2f52a809b7173a85960a3497701b",
    },
    "Compound_patterns.ipynb": {
        "sha256": "a173677c9884bbadc8fad7c28e382f83adfd5378587042411179da98a6f8436a",
    },
}
# Nearest grid points to three fixed locations spanning Sweden; ties take the first index.
TARGETS = (("southern", 56.0, 14.0), ("central", 62.0, 15.0), ("northern", 67.0, 20.0))


def checksum(path: Path) -> str:
    """Hash a file incrementally so the full heatwave grid never occupies memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def daily_spei(monthly_time: np.ndarray, monthly_spei: np.ndarray) -> xr.DataArray:
    """Apply notebook cell 63ae9174: pad to days, then interpolate remaining NaNs."""
    data = xr.DataArray(monthly_spei, dims=("time", "cell"), coords={"time": monthly_time})
    return cast(xr.DataArray, data.resample(time="D").ffill().interpolate_na(dim="time", method="linear"))


def author_counts(hwi: np.ndarray, spei: np.ndarray) -> dict[str, int | float | None]:
    """Adapt the deterministic part of notebook cell 93ef7024; preserve its zero-joint NaN."""
    if hwi.ndim != 1 or hwi.shape != spei.shape or hwi.size == 0:
        raise ValueError("Reference inputs must be equal nonempty one-dimensional series.")
    if not np.isfinite(hwi).all() or not np.isfinite(spei).all():
        raise ValueError("The reference subset requires fully observed prepared series.")
    heat = hwi.copy()
    drought = spei.copy()
    heat[heat >= 0] = 1
    heat[heat < 0] = 0
    drought[drought > -1] = 0
    drought[drought <= -1] = 1
    joint = np.array(heat, dtype=int) & np.array(drought, dtype=int)
    n = len(joint)
    n_heat, n_drought, n_joint = int(np.sum(heat)), int(np.sum(drought)), int(np.sum(joint))
    p_joint = n_joint / n
    p_independent = (n_heat / n) * (n_drought / n)
    return {
        "n_days": n,
        "n_heat": n_heat,
        "n_drought": n_drought,
        "n_joint": n_joint,
        "joint_probability": p_joint,
        "independent_probability": p_independent,
        "lmf": p_joint / p_independent if n_joint > 0 else None,
    }


def prepare(source_dir: Path, output_dir: Path, download_date: str) -> None:
    """Verify the pinned files, extract three cells, and publish an offline fixture."""
    dt.date.fromisoformat(download_date)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use a new output directory.")
    for name, source in SOURCES.items():
        if checksum(source_dir / name) != source["sha256"]:
            raise ValueError(f"Unexpected source checksum: {name}")

    with (
        xr.open_dataset(source_dir / "SPEI12.nc", engine="h5netcdf") as spei,
        xr.open_dataset(source_dir / "heatwave_SMHI.nc", engine="h5netcdf") as hwi,
    ):
        np.testing.assert_array_equal(spei.lat, hwi.lat)
        np.testing.assert_array_equal(spei.lon, hwi.lon)
        cells = []
        monthly_columns, daily_columns = [], []
        for label, target_lat, target_lon in TARGETS:
            row = int(np.abs(spei.lat.values - target_lat).argmin())
            column = int(np.abs(spei.lon.values - target_lon).argmin())
            monthly = spei["SPEI"].isel(lat=row, lon=column).values
            cells.append(
                {
                    "name": label,
                    "target_lat": target_lat,
                    "target_lon": target_lon,
                    "lat_index": row,
                    "lon_index": column,
                    "latitude": float(spei.lat[row]),
                    "longitude": float(spei.lon[column]),
                    "monthly_missing_dates": spei.time.values[np.isnan(monthly)]
                    .astype("datetime64[D]")
                    .astype(str)
                    .tolist(),
                }
            )
            monthly_columns.append(monthly)
            daily_columns.append(hwi["heatwave index"].isel(lat=row, lon=column).values)
        monthly_time = spei.time.values
        daily_time = hwi.time.values
        monthly_values = np.stack(monthly_columns, axis=1)
        daily_values = np.stack(daily_columns, axis=1)

    np.testing.assert_array_equal(
        monthly_time[[0, -1]].astype("datetime64[D]"), [np.datetime64("1922-01-31"), np.datetime64("2021-12-31")]
    )
    np.testing.assert_array_equal(daily_time, np.arange("1922-01-01", "2022-01-01", dtype="datetime64[D]"))
    np.testing.assert_equal(monthly_values.shape, (1200, 3))
    np.testing.assert_equal(daily_values.shape, (36525, 3))
    prepared = daily_spei(monthly_time, monthly_values)
    positions = np.searchsorted(daily_time, prepared.time.values)
    np.testing.assert_array_equal(daily_time[positions], prepared.time.values)
    heat = daily_values[positions]
    cases = []
    for cell in range(3):
        for month in (6, 7, 8):
            selected = prepared.time.dt.month.values == month
            cases.append(
                {"cell": cell, "month": month, **author_counts(heat[selected, cell], prepared.values[selected, cell])}
            )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the target so a failed write leaves no partial directory to block the next run.
    with TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-") as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        np.savez_compressed(
            staging / "inputs.npz",
            monthly_time=monthly_time,
            daily_time=daily_time,
            monthly_spei=monthly_values,
            daily_hwi=daily_values,
        )
        reference: dict[str, Any] = {
            "dataset_doi": DATASET_DOI,
            "paper_doi": "10.1007/s00477-024-02783-3",
            "author_commit": AUTHOR_COMMIT,
            "sources": SOURCES,
            "environment": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "xarray", "pandas")},
            },
            "inputs_sha256": checksum(staging / "inputs.npz"),
            "cells": cells,
            "preparation": {
                "variables": {"SPEI12.nc": "SPEI", "heatwave_SMHI.nc": "heatwave index"},
                "monthly_to_daily": "resample(time='D').ffill().interpolate_na(dim='time', method='linear')",
                "start": "1922-01-31",
                "end": "2021-12-31",
                "heat_comparator": "HWI >= 0",
                "drought_comparator": "SPEI <= -1",
                "source_calibration_period": "not recorded in the supplied NetCDF attributes",
                "significance": "not reproduced; the author's IID resampling and random tie noise are excluded",
            },
            "cases": cases,
        }
        payload = (json.dumps(reference, indent=2, allow_nan=False) + "\n").encode()
        (staging / "reference.json").write_bytes(payload)
        provenance = {
            "source": "Marlon Vieira Passos; public study inputs and same-author deterministic LMF calculation",
            "url": "https://doi.org/" + DATASET_DOI,
            "download_date": download_date,
            "subset_description": "Three Swedish grid cells; monthly SPEI12 and daily HWI, 1922-2021; nine June-August LMF cases.",
            "checksum_sha256": hashlib.sha256(payload).hexdigest(),
            "fixture_version": "1.0.0",
            "validation_tolerance": {"count_atol": 0, "lmf_atol": 1e-12, "rtol": 0},
            "citation": "Vieira Passos, M., Kan, J.-C., Destouni, G., Barquet, K., and Kalantari, Z. (2024). Identifying regional hotspots of heatwaves, droughts, floods, and their co-occurrences. Stochastic Environmental Research and Risk Assessment 38, 3875-3893. https://doi.org/10.1007/s00477-024-02783-3",
            "doi": DATASET_DOI,
            "license": "Input data: CC BY 4.0. Adapted deterministic reference code: MIT; see NOTICE.",
            "notes": "reference.json is canonical and pins inputs.npz. This is an author-method fixture, not validation of a compound API or the complete Figure 8 map. Public source hashes, file IDs, cell-selection targets, source NaNs, original variable names, processing choices, and generation versions are in reference.json. The author returns NaN for zero joint counts; JSON uses null. No scientific bootstrap result is supplied. Normal tests are offline and import no compound module.",
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, output_dir)
        print(f"Wrote {len(cases)} author-method cases to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing ones are refused.")
    parser.add_argument("--download-date", required=True, help="Actual source retrieval date, YYYY-MM-DD.")
    args = parser.parse_args()
    prepare(args.source_dir, args.output_dir, args.download_date)


if __name__ == "__main__":
    main()
