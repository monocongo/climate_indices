#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "xarray",
#   "h5netcdf",
#   "netCDF4",
#   "pyshp",
# ]
# ///
"""Prepare CRU TS 4.09 input fixtures for the like-for-like SPEIbase comparison.

Downloads the CRU TS 4.09 monthly precipitation and FAO-56 Penman-Monteith PET
grids, selects the 0.5-degree cells whose centers fall inside the same three
CONUS climate divisions used by tests/fixture/speibase/, and stores each
division's per-cell precipitation (mm/month) and PET (mm/day) as fixtures for an
input-matched SPEI comparison against SPEIbase v2.11.

This is the input half of the like-for-like comparison owned by issue #1196.
tests/fixture/speibase/ holds SPEIbase's own per-cell-mean SPEI; this directory
holds the CRU TS inputs SPEIbase was computed from, so
tests/test_speibase_like_for_like.py can feed the same precipitation and PET
that SPEIbase standardized.

Usage:
    uv run scripts/prepare_speibase_cru_ts_inputs.py

The script will:
    1. Download CRU TS 4.09 pre and pet grids (1901-2024) and the NCEI CONUS
       climate division shapefile
    2. Select the grid cells inside the three divisions and extract each cell's
       monthly precipitation (mm/month) and PET (mm/day), 1901-2024
    3. Save one (n_cells, 1488) float32 array per division per variable
    4. Measure the agreement between per-cell climate_indices SPEI (log-logistic
       distribution, CRU TS P - PET) averaged over the division and the
       SPEIbase v2.11 reference, per scale
    5. Publish the fixtures with provenance.json (checksum, measurements, and
       floors derived from them)

Set CRU_TS_DOWNLOAD_DIR to reuse an existing download directory instead of
re-fetching the ~765 MB of grids.

Source:
    https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/ (Open Government
    Licence v3; Harris et al. 2020, https://doi.org/10.1038/s41597-020-0453-3)
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture"
OUTPUT_DIR = FIXTURE_DIR / "speibase_cru_ts"
_SPEIBASE_DIR = FIXTURE_DIR / "speibase"

# Reuse the shapefile download, polygon selection, and division parsing from the
# SPEIbase fixture script so both fixtures select identical grid cells.
sys.path.insert(0, str(Path(__file__).parent))
import prepare_speibase_fixtures as _speibase  # noqa: E402

_CRU_BASE_URL = "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/cruts.2503051245.v4.09"
_CRU_APPROVED_ORIGIN = "https://crudata.uea.ac.uk/"
# The uncompressed precipitation grid is ~2 GB; the compressed download is
# ~700 MB, so cap generously above it and below anything malformed.
_MAX_DOWNLOAD_BYTES = 1_200_000_000
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

_DIVISIONS = ("0101", "3405", "0205")
_DATA_START_YEAR = 1901
_DATA_END_YEAR = 2024  # SPEIbase v2.11's CRU TS 4.09 period
_N_MONTHS = (_DATA_END_YEAR - _DATA_START_YEAR + 1) * 12
_SCALES = (1, 3, 6, 12)
_LATITUDE_BAND = (24.0, 50.0)
_LONGITUDE_BAND = (-126.0, -66.0)

_CATEGORY_BOUNDARIES = (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0)
_FLOOR_METRICS = ("correlation", "sign_agreement", "category_agreement")
_RECORDED_METRICS = (*_FLOOR_METRICS, "mean_abs_difference")
_FLOOR_MARGINS = {"correlation": 0.02, "sign_agreement": 0.02, "category_agreement": 0.03}

# Frozen regression expectations, filled from the first generation. Every
# refresh re-measures the agreement and refuses to publish a drift beyond
# _EXPECTATION_TOLERANCE, so a silent upstream or library regression cannot
# quietly overwrite the recorded agreement with a worse one.
_EXPECTED_STATS: dict[str, dict[int, dict[str, float]]] = {
    "0101": {
        1: {
            "correlation": 0.999981,
            "sign_agreement": 0.999317,
            "category_agreement": 0.996585,
            "mean_abs_difference": 0.002692,
        },
        3: {
            "correlation": 0.999986,
            "sign_agreement": 0.998632,
            "category_agreement": 0.997948,
            "mean_abs_difference": 0.00321,
        },
        6: {
            "correlation": 0.999986,
            "sign_agreement": 0.997944,
            "category_agreement": 0.996573,
            "mean_abs_difference": 0.003625,
        },
        12: {
            "correlation": 0.999988,
            "sign_agreement": 0.999312,
            "category_agreement": 0.994494,
            "mean_abs_difference": 0.003961,
        },
    },
    "3405": {
        1: {
            "correlation": 0.999926,
            "sign_agreement": 0.997951,
            "category_agreement": 0.997268,
            "mean_abs_difference": 0.005229,
        },
        3: {
            "correlation": 0.999938,
            "sign_agreement": 0.997264,
            "category_agreement": 0.994528,
            "mean_abs_difference": 0.006224,
        },
        6: {
            "correlation": 0.999963,
            "sign_agreement": 0.996573,
            "category_agreement": 0.993831,
            "mean_abs_difference": 0.005808,
        },
        12: {
            "correlation": 0.999972,
            "sign_agreement": 0.995871,
            "category_agreement": 0.995182,
            "mean_abs_difference": 0.005748,
        },
    },
    "0205": {
        1: {
            "correlation": 0.999642,
            "sign_agreement": 0.994536,
            "category_agreement": 0.995219,
            "mean_abs_difference": 0.008025,
        },
        3: {
            "correlation": 0.999747,
            "sign_agreement": 0.997264,
            "category_agreement": 0.992476,
            "mean_abs_difference": 0.012789,
        },
        6: {
            "correlation": 0.999778,
            "sign_agreement": 0.995202,
            "category_agreement": 0.987663,
            "mean_abs_difference": 0.01396,
        },
        12: {
            "correlation": 0.999789,
            "sign_agreement": 0.993118,
            "category_agreement": 0.977977,
            "mean_abs_difference": 0.015958,
        },
    },
}
_EXPECTATION_TOLERANCE = 0.001


def _download(url: str, destination: Path) -> Path:
    """Fetch a URL to a local file, refusing off-origin redirects and oversized payloads."""
    if not url.startswith(_CRU_APPROVED_ORIGIN):
        raise ValueError(f"URL must use {_CRU_APPROVED_ORIGIN}, got: {url}")
    with urllib.request.urlopen(url, timeout=300) as response:  # noqa: S310 -- host validated above
        if not response.url.startswith(_CRU_APPROVED_ORIGIN):
            raise ValueError(f"download redirected off the approved origin: {response.url}")
        written = 0
        try:
            with destination.open("wb") as handle:
                while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > _MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"download exceeds the {_MAX_DOWNLOAD_BYTES} byte cap: {url}")
                    handle.write(chunk)
        except BaseException:
            destination.unlink(missing_ok=True)  # never leave a partial download behind
            raise
    if written == 0:
        destination.unlink(missing_ok=True)
        raise ValueError(f"download has implausible size ({written} bytes): {url}")
    return destination


def _fetch_grid(variable: str, downloads: Path) -> Path:
    """Download and decompress one CRU TS 4.09 variable grid."""
    compressed = downloads / f"cru_ts4.09.{_DATA_START_YEAR}.{_DATA_END_YEAR}.{variable}.dat.nc.gz"
    if not compressed.exists():
        url = f"{_CRU_BASE_URL}/{variable}/" + compressed.name
        print(f"Downloading {url} ...", file=sys.stderr)
        _download(url, compressed)
    netcdf = downloads / compressed.name.removesuffix(".gz")
    if not netcdf.exists():
        with gzip.open(compressed, "rb") as source, netcdf.open("wb") as target:
            shutil.copyfileobj(source, target)
    return netcdf


def _fetch_shapefile(downloads: Path) -> Path:
    """Return the CONUS climate division shapefile, reusing a cached extraction if present."""
    extracted = downloads / "divisions"
    if extracted.exists():
        shapes = sorted(extracted.glob("*.shp"))
        if len(shapes) == 1:
            return shapes[0]
    shutil.rmtree(extracted, ignore_errors=True)
    return _speibase._download_shapefile(downloads)


def _monthly_days(start_year: int, n_months: int) -> np.ndarray:
    """Days in each month, leap-aware, for a monthly series starting in January."""
    base = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    days = np.tile(base, n_months // 12 + 1)[:n_months]
    for index in range(n_months):
        year = start_year + index // 12
        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            if index % 12 == 1:
                days[index] = 29.0
    return days


def _cell_series(dataset, variable: str, mask: np.ndarray, latitudes: np.ndarray) -> np.ndarray:
    """Extract (n_cells, n_months) for the selected cells, in row-major grid order."""
    rows, columns = np.where(mask)
    values = dataset[variable].sel(lat=slice(*_LATITUDE_BAND), lon=slice(*_LONGITUDE_BAND)).values[:_N_MONTHS]
    if values.shape[0] < _N_MONTHS:
        raise RuntimeError(f"{variable} has only {values.shape[0]} months, need {_N_MONTHS}")
    return values[:, rows, columns].T  # (n_cells, n_months)


def _categories(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, _CATEGORY_BOUNDARIES)


def _agreement(computed: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Correlation, sign agreement, category agreement, and mean |difference|."""
    from scipy.stats import pearsonr

    both_present = ~np.isnan(computed) & ~np.isnan(reference)
    computed_values = computed[both_present].astype(np.float64)
    reference_values = reference[both_present].astype(np.float64)
    return {
        "correlation": float(pearsonr(computed_values, reference_values).statistic),
        "sign_agreement": float(np.mean(np.sign(computed_values) == np.sign(reference_values))),
        "category_agreement": float(np.mean(_categories(computed_values) == _categories(reference_values))),
        "mean_abs_difference": float(np.mean(np.abs(computed_values - reference_values))),
    }


def _computed_series(precip: np.ndarray, pet: np.ndarray, scale: int) -> np.ndarray:
    """Division-mean SPEI: standardize each cell with log-logistic, then average.

    Mirrors SPEIbase, which standardizes per cell before averaging. CRU TS PET
    is mm/day, converted to mm/month with the leap-aware month lengths.
    """
    from climate_indices import compute, indices

    pet_mm = pet * _monthly_days(_DATA_START_YEAR, _N_MONTHS)
    cells = [
        indices.spei(
            precip[row],
            pet_mm[row],
            scale,
            indices.Distribution.loglogistic,
            compute.Periodicity.monthly,
            _DATA_START_YEAR,
            _DATA_START_YEAR,
            _DATA_END_YEAR,
        )
        for row in range(precip.shape[0])
    ]
    with np.errstate(invalid="ignore"):
        with warnings.catch_warnings():
            # leading scale-1 months are all-NaN by construction (rolling-sum warmup)
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.vstack(cells), axis=0)


def _measure_agreement(inputs: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[int, dict[str, float]]]:
    """Measure the like-for-like agreement per division and timescale."""
    reference = {scale: np.load(_SPEIBASE_DIR / f"spei{scale:02d}.npy") for scale in _SCALES}
    reference_months = reference[_SCALES[0]].shape[1]
    measured = {}
    for row_index, division in enumerate(_DIVISIONS):
        measured[division] = {
            scale: _agreement(
                _computed_series(inputs[division]["pre"], inputs[division]["pet"], scale)[:reference_months],
                reference[scale][row_index],
            )
            for scale in _SCALES
        }
    return measured


def _check_expectations(measured: dict[str, dict[int, dict[str, float]]]) -> None:
    """Refuse a refresh whose agreement drifted away from the recorded expectations."""
    if not _EXPECTED_STATS:
        return
    for division, per_scale in measured.items():
        for scale, stats in per_scale.items():
            for metric, value in stats.items():
                expected = _EXPECTED_STATS[division][scale][metric]
                if abs(value - expected) > _EXPECTATION_TOLERANCE:
                    raise RuntimeError(
                        f"{division}_spei{scale:02d} {metric} measured {value:.4f}, expected {expected:.4f} "
                        f"(+-{_EXPECTATION_TOLERANCE}); investigate the drift, then re-record "
                        f"_EXPECTED_STATS deliberately"
                    )


def _floors(measured: dict[str, dict[int, dict[str, float]]]) -> dict[str, float]:
    return {
        f"{division}_spei{scale:02d}_{metric}": math.floor((stats[metric] - _FLOOR_MARGINS[metric]) * 100.0) / 100.0
        for division, per_scale in measured.items()
        for scale, stats in per_scale.items()
        for metric in _FLOOR_METRICS
    }


def _compute_checksum(directory: Path) -> str:
    hasher = hashlib.sha256()
    for npy_file in sorted(directory.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    return hasher.hexdigest()


def _write_provenance(directory: Path, checksum: str, measured: dict[str, dict[int, dict[str, float]]]) -> None:
    provenance = {
        "source": "Climatic Research Unit (CRU), University of East Anglia",
        "url": f"{_CRU_BASE_URL}/",
        "download_date": dt.date.today().isoformat(),
        "subset_description": (
            "CRU TS 4.09 monthly precipitation (mm/month) and FAO-56 Penman-Monteith potential "
            "evapotranspiration (mm/day), extracted at the 0.5-degree grid cells whose centers fall "
            "inside three CONUS climate divisions spanning an aridity gradient: 0101 (Northern Valley, "
            "Alabama; humid), 3405 (Central, Oklahoma; subhumid), and 0205 (Southwest, Arizona; arid). "
            "January 1901 through December 2024 (1488 months), the period SPEIbase v2.11 was computed "
            "from. Each pre_<division>.npy and pet_<division>.npy is a (n_cells, 1488) float32 array "
            "whose cell order matches divisions.json. Cell membership comes from the NCEI "
            "CONUS_CLIMATE_DIVISIONS.shp.zip polygons and matches tests/fixture/speibase/."
        ),
        "checksum_sha256": checksum,
        "fixture_version": "1.0.0",
        "validation_tolerance": _floors(measured),
        "measured_stats": {
            f"{division}_spei{scale:02d}": stats
            for division, per_scale in measured.items()
            for scale, stats in per_scale.items()
        },
        "citation": (
            "Harris, I., Osborn, T. J., Jones, P., and Lister, D. (2020). Version 4 of the CRU TS "
            "monthly high-resolution gridded multivariate climate dataset. Scientific Data 7, 109. "
            "https://doi.org/10.1038/s41597-020-0453-3. CRU TS 4.09 data developed by the Climatic "
            "Research Unit (CRU) at the University of East Anglia."
        ),
        "doi": "10.1038/s41597-020-0453-3",
        "license": (
            "Open Government Licence v3.0 (UK). Attribution: CRU TS 4.09 data developed by the "
            "Climatic Research Unit (CRU) at the University of East Anglia."
        ),
        "notes": (
            "INPUT FIXTURES FOR THE LIKE-FOR-LIKE SPEIBASE COMPARISON, NOT A STANDALONE REFERENCE "
            "SERIES. These are the exact inputs SPEIbase v2.11 was computed from (CRU TS 4.09 "
            "precipitation and FAO-56 Penman-Monteith PET), so tests/test_speibase_like_for_like.py "
            "can reproduce SPEIbase's pipeline: standardize each 0.5-degree cell with the "
            "log-logistic (generalized logistic) distribution using unbiased-PWM L-moments over "
            "1901-2024, then average the per-cell SPEI inside each climate division, and compare that "
            "mean against tests/fixture/speibase/. The remaining differences are the fitting "
            "implementation (climate_indices' L-moment code vs. R SPEI's ub-pwm), parameter-rounding, "
            "and any residual preprocessing, not the PET method, distribution family, precipitation "
            "input, or spatial-support confounds that separated the earlier gamma/Thornthwaite "
            "plausibility check. CRU TS PET is stored in mm/day and must be multiplied by the "
            "leap-aware days in each month before differencing with precipitation, matching "
            "SPEIbase's R/computeSPEI.R (`etp * ndays`)."
        ),
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def _write_divisions(directory: Path, rows: list[dict]) -> None:
    (directory / "divisions.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Download CRU TS 4.09, build the input fixtures, and write them to tests/fixture/speibase_cru_ts/."""
    import xarray as xr

    download_root = os.environ.get("CRU_TS_DOWNLOAD_DIR")
    downloads = Path(download_root) if download_root else Path(tempfile.mkdtemp(prefix="cru-ts-downloads-"))
    downloads.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".speibase-cru-ts-staging-", dir=FIXTURE_DIR))
    try:
        print("Downloading NCEI climate division shapefile ...", file=sys.stderr)
        shapefile_path = _fetch_shapefile(downloads)
        division_shapes = _speibase._load_division_shapes(shapefile_path)
        missing = [division for division in _DIVISIONS if division not in division_shapes]
        if missing:
            raise RuntimeError(f"divisions absent from the shapefile: {missing}")

        pre_path = _fetch_grid("pre", downloads)
        pet_path = _fetch_grid("pet", downloads)
        pre_dataset = xr.open_dataset(pre_path, engine="netcdf4")
        pet_dataset = xr.open_dataset(pet_path, engine="netcdf4")
        try:
            latitudes = pre_dataset.lat.sel(lat=slice(*_LATITUDE_BAND)).values
            longitudes = pre_dataset.lon.sel(lon=slice(*_LONGITUDE_BAND)).values
            grid_longitudes, grid_latitudes = np.meshgrid(longitudes, latitudes)

            speibase_divisions = {row["id"]: row for row in json.loads((_SPEIBASE_DIR / "divisions.json").read_text())}
            rows = []
            inputs: dict[str, dict[str, np.ndarray]] = {}
            for division in _DIVISIONS:
                mask = _speibase._polygon_mask(division_shapes[division]["shape"], grid_longitudes, grid_latitudes)
                expected_cells = speibase_divisions[division]["speibase_cells"]
                if int(mask.sum()) != expected_cells:
                    raise RuntimeError(
                        f"division {division}: selected {mask.sum()} CRU TS cells, but the SPEIbase "
                        f"fixture used {expected_cells}; the grids or polygon selection diverged"
                    )
                precip = _cell_series(pre_dataset, "pre", mask, latitudes)
                pet = _cell_series(pet_dataset, "pet", mask, latitudes)
                inputs[division] = {"pre": precip, "pet": pet}
                longitude, latitude = _speibase._polygon_centroid(division_shapes[division]["shape"])
                rows.append(
                    {
                        "id": division,
                        "name": division_shapes[division]["name"],
                        "state": division_shapes[division]["state"],
                        "latitude": round(latitude, 4),
                        "longitude": round(longitude, 4),
                        "cells": [
                            [round(float(la), 4), round(float(lo), 4)]
                            for la, lo in zip(latitudes[np.where(mask)[0]], longitudes[np.where(mask)[1]], strict=True)
                        ],
                    }
                )
                print(f"  division {division}: {mask.sum()} cells", file=sys.stderr)
        finally:
            pre_dataset.close()
            pet_dataset.close()

        print("Measuring the like-for-like agreement with SPEIbase ...", file=sys.stderr)
        measured = _measure_agreement(inputs)
        _check_expectations(measured)

        for division, arrays in inputs.items():
            np.save(staging / f"pre_{division}.npy", arrays["pre"].astype(np.float32))
            np.save(staging / f"pet_{division}.npy", arrays["pet"].astype(np.float32))
        _write_divisions(staging, rows)
        checksum = _compute_checksum(staging)
        _write_provenance(staging, checksum, measured)
        _publish(staging, OUTPUT_DIR)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if not download_root:
            shutil.rmtree(downloads, ignore_errors=True)

    print(f"Done. checksum_sha256={checksum}", file=sys.stderr)


def _publish(staging: Path, output_dir: Path) -> None:
    """Replace the published fixture directory with the staged one, as one transaction."""
    backup = output_dir.with_name(f".{output_dir.name}-backup")
    shutil.rmtree(backup, ignore_errors=True)
    if output_dir.exists():
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        if backup.exists():
            os.replace(backup, output_dir)
        raise
    shutil.rmtree(backup, ignore_errors=True)


if __name__ == "__main__":
    main()
