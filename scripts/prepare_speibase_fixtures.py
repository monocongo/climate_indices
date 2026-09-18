#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
#   "xarray",
#   "h5netcdf",
#   "pyshp",
# ]
# ///
"""Prepare SPEIbase v2.11 reference fixtures for SPEI plausibility tests.

Downloads the CSIC SPEIbase v2.11 global 0.5-degree monthly SPEI grids, selects
the 0.5-degree cell centers falling inside three CONUS climate divisions that
span an aridity gradient (humid Alabama, subhumid Oklahoma, arid southwest
Arizona), and prepares the per-cell SPEI mean as pytest fixtures for a
plausibility/agreement comparison against climate_indices' SPEI (gamma
distribution, Thornthwaite PET). This script must be run manually when
refreshing the reference data.

Usage:
    uv run scripts/prepare_speibase_fixtures.py

The script will:
    1. Download the NCEI CONUS climate division shapefile and the four
       SPEIbase v2.11 NetCDF files (SPEI-1/3/6/12, ~380 MB each)
    2. Select the grid cells whose centers fall inside each target division
       polygon, and average the per-cell SPEI values per month (a mean of
       standardized indices, not an index computed from averaged inputs --
       nClimDiv averages its inputs before computing its index)
    3. Truncate to January 1901 - December 2022, the period shared with the
       committed tests/fixture/palmer/<division>/{precips,temps}.npy inputs
    4. Save one (3, 1464) float32 array per timescale under
       tests/fixture/speibase/
    5. Measure the agreement between each freshly averaged series and this
       library's SPEI, per division and timescale
    6. Write divisions.json (row order, names, polygon centroids, cell counts)
       and provenance.json with a SHA-256 checksum, the step-5 measurements, and
       the floors derived from them; a refresh whose measurements drift beyond
       the recorded expectations is refused instead of published
    7. Publish the complete fixture directory as one transaction, so an
       interrupted refresh cannot leave arrays and metadata from different
       generations in place

Source:
    https://spei.csic.es/spei_database_2_11/ (CC-BY 4.0, Beguería et al. 2024)

This fixture is deliberately NOT an external numerical-validation oracle.
SPEIbase v2.11 uses FAO-56 Penman-Monteith PET (CRU TS 4.09) while
climate_indices' SPEI paths use Thornthwaite/Hargreaves PET, and SPEIbase
standardizes with the log-logistic distribution while these tests use gamma;
both differences are known, climate-dependent confounds (see
docs/research/spei-dataset-survey.md on branch research/spei-dataset-survey).
The test therefore asserts loose per-division agreement floors (correlation,
sign agreement, drought-category agreement) rather than an atol/rtol gate.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture"
OUTPUT_DIR = FIXTURE_DIR / "speibase"

_SPEIBASE_BASE_URL = "https://spei.csic.es/spei_database_2_11"
_SPEIBASE_APPROVED_ORIGIN = "https://spei.csic.es/"
_SHAPEFILE_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/CONUS_CLIMATE_DIVISIONS.shp.zip"
_SHAPEFILE_APPROVED_ORIGIN = "https://www.ncei.noaa.gov/"

_FILE_VERSION = "2.11.0"
_SCALES = (1, 3, 6, 12)

# NCEI climate division codes (old alphabetical state codes) and the CONUS
# aridity gradient they sample.
_DIVISIONS = ("0101", "3405", "0205")

_DATA_START_YEAR = 1901  # SPEIbase v2.11 starts January 1901
_DATA_END_YEAR = 2022  # matches tests/fixture/palmer/<division>/precips.npy length
_N_MONTHS = (_DATA_END_YEAR - _DATA_START_YEAR + 1) * 12

_LATITUDE_BAND = (24.0, 50.0)
_LONGITUDE_BAND = (-126.0, -66.0)

# SPEIbase files are ~380 MB each; anything larger is malformed.
_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024

# The compared series is computed from the committed nClimDiv inputs, and the
# agreement statistics bin against the standard SPEI drought-category
# boundaries (extreme <= -2, severe -2..-1.5, moderate -1.5..-1, etc.).
_PALMER_ROOT = FIXTURE_DIR / "palmer"
_PALMER_INPUT_START_YEAR = 1895  # tests/fixture/palmer/<division>/ inputs start here
_INPUT_OFFSET = (_DATA_START_YEAR - _PALMER_INPUT_START_YEAR) * 12
_CATEGORY_BOUNDARIES = (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0)

# Frozen regression expectations from the original fixture generation (GitHub
# issue #779): agreement between climate_indices.indices.spei() (gamma
# distribution, Thornthwaite PET, full-period-of-record calibration 1901-2022)
# and the grid-cell-mean SPEIbase v2.11 series, per division and timescale.
#
# These are expectations, not the provenance measurements: every refresh
# re-measures the agreement from the arrays it just built (_measure_agreement)
# and writes those measurements to provenance.json. A measurement outside
# _EXPECTATION_TOLERANCE of the expectation fails the refresh, so a silent
# upstream or library regression cannot quietly overwrite the recorded
# agreement with a worse one.
_EXPECTED_STATS = {
    "0101": {
        1: {
            "correlation": 0.9248,
            "sign_agreement": 0.8818,
            "category_agreement": 0.7671,
            "mean_abs_difference": 0.2904,
        },
        3: {
            "correlation": 0.9341,
            "sign_agreement": 0.8906,
            "category_agreement": 0.7722,
            "mean_abs_difference": 0.2772,
        },
        6: {
            "correlation": 0.9372,
            "sign_agreement": 0.8869,
            "category_agreement": 0.7724,
            "mean_abs_difference": 0.2781,
        },
        12: {
            "correlation": 0.9410,
            "sign_agreement": 0.9023,
            "category_agreement": 0.7632,
            "mean_abs_difference": 0.2766,
        },
    },
    "3405": {
        1: {
            "correlation": 0.9508,
            "sign_agreement": 0.9112,
            "category_agreement": 0.8128,
            "mean_abs_difference": 0.2301,
        },
        3: {
            "correlation": 0.9621,
            "sign_agreement": 0.9268,
            "category_agreement": 0.8269,
            "mean_abs_difference": 0.2040,
        },
        6: {
            "correlation": 0.9688,
            "sign_agreement": 0.9431,
            "category_agreement": 0.8554,
            "mean_abs_difference": 0.1890,
        },
        12: {
            "correlation": 0.9709,
            "sign_agreement": 0.9291,
            "category_agreement": 0.8362,
            "mean_abs_difference": 0.1841,
        },
    },
    "0205": {
        1: {
            "correlation": 0.8602,
            "sign_agreement": 0.8511,
            "category_agreement": 0.7063,
            "mean_abs_difference": 0.3984,
        },
        3: {
            "correlation": 0.8486,
            "sign_agreement": 0.8393,
            "category_agreement": 0.6772,
            "mean_abs_difference": 0.4220,
        },
        6: {
            "correlation": 0.8409,
            "sign_agreement": 0.8417,
            "category_agreement": 0.6429,
            "mean_abs_difference": 0.4391,
        },
        12: {
            "correlation": 0.8235,
            "sign_agreement": 0.8445,
            "category_agreement": 0.6001,
            "mean_abs_difference": 0.4809,
        },
    },
}
# Absolute band a re-measured statistic may sit outside before the refresh is
# refused. The recorded expectations are the same computation rounded to four
# decimals, so this only absorbs cross-platform numerical wobble.
_EXPECTATION_TOLERANCE = 0.001

# The floors keep this much slack below each measurement; the test's
# test_floors_keep_documented_slack pins the drift to a narrow, documented band
# so a floor cannot be widened to hide a regression.
_FLOOR_MARGINS = {"correlation": 0.05, "sign_agreement": 0.05, "category_agreement": 0.08}
_FLOOR_METRICS = ("correlation", "sign_agreement", "category_agreement")


def _download(url: str, destination: Path, approved_origin: str) -> Path:
    """Fetch a URL to a local file, refusing off-origin redirects and oversized payloads."""
    if not url.startswith(approved_origin):
        raise ValueError(f"URL must use {approved_origin}, got: {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 -- host validated above
        if not response.url.startswith(approved_origin):
            raise ValueError(f"download redirected off the approved origin: {response.url}")
        declared = response.headers.get("Content-Length")
        if declared is not None and declared.isdigit() and int(declared) > _MAX_DOWNLOAD_BYTES:
            raise ValueError(f"download declares {declared} bytes, over the {_MAX_DOWNLOAD_BYTES} cap: {url}")
        # Stream in bounded chunks and count as we go: Content-Length is optional
        # and untrusted, so the cap has to hold without it.
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


def _download_speibase_netcdf(scale: int, directory: Path) -> Path:
    """Fetch one SPEIbase v2.11 timescale NetCDF file."""
    url = f"{_SPEIBASE_BASE_URL}/nc/spei{scale:02d}.nc"
    return _download(url, directory / f"spei{scale:02d}.nc", _SPEIBASE_APPROVED_ORIGIN)


def _download_shapefile(directory: Path) -> Path:
    """Fetch and extract the NCEI CONUS climate division shapefile."""
    archive = _download(_SHAPEFILE_URL, directory / "CONUS_CLIMATE_DIVISIONS.shp.zip", _SHAPEFILE_APPROVED_ORIGIN)
    extract_dir = directory / "divisions"
    extract_dir.mkdir()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            path = PurePosixPath(member.filename)
            is_symlink = (member.external_attr >> 16) & 0o170000 == 0o120000
            if path.is_absolute() or ".." in path.parts or is_symlink:
                raise RuntimeError(f"refusing unsafe zip member: {member.filename}")
        zipped.extractall(extract_dir)
    shapes = sorted(extract_dir.glob("*.shp"))
    if len(shapes) != 1:
        raise RuntimeError(f"expected exactly one shapefile in the archive, found {shapes}")
    return shapes[0]


def _intersects_grid(points_x: np.ndarray, points_y: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd (ray casting) point-in-polygon for one closed ring.

    Loop over ring edges but vectorize across grid points. CONUS climate
    division rings are small and do not cross the antimeridian, so planar
    longitude/latitude intersection is sufficient.
    """
    inside = np.zeros(points_x.shape, dtype=bool)
    for index in range(len(ring) - 1):
        x1, y1 = ring[index]
        x2, y2 = ring[index + 1]
        if y1 == y2:
            continue
        crosses = (y1 > points_y) != (y2 > points_y)
        x_intersection = (x2 - x1) * (points_y - y1) / (y2 - y1) + x1
        inside ^= crosses & (points_x < x_intersection)
    return inside


def _shape_rings(shape) -> list[np.ndarray]:
    """Split a pyshp shape into closed exterior/interior rings."""
    parts = list(shape.parts) + [len(shape.points)]
    rings = []
    for start, stop in zip(parts[:-1], parts[1:], strict=True):
        ring = np.asarray(shape.points[start:stop], dtype=float)
        if not np.allclose(ring[0], ring[-1]):
            ring = np.vstack([ring, ring[0]])
        rings.append(ring)
    return rings


def _polygon_mask(shape, grid_longitudes: np.ndarray, grid_latitudes: np.ndarray) -> np.ndarray:
    """Boolean mask of grid cell centers inside the polygon (holes excluded)."""
    mask = np.zeros(grid_longitudes.shape, dtype=bool)
    for ring in _shape_rings(shape):
        mask ^= _intersects_grid(grid_longitudes, grid_latitudes, ring)
    return mask


def _polygon_centroid(shape) -> tuple[float, float]:
    """Area-weighted centroid of a (possibly multi-part) polygon."""
    total_area = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for ring in _shape_rings(shape):
        x, y = ring[:-1, 0], ring[:-1, 1]
        x_next, y_next = np.roll(x, -1), np.roll(y, -1)
        cross = x * y_next - x_next * y
        area = cross.sum() / 2.0
        # a degenerate ring contributes no area and no centroid; abs_tol is machine precision
        if math.isclose(area, 0.0, abs_tol=1e-12):
            continue
        total_area += area
        centroid_x += ((x + x_next) * cross).sum() / 6.0
        centroid_y += ((y + y_next) * cross).sum() / 6.0
    if math.isclose(total_area, 0.0, abs_tol=1e-12):
        raise ValueError("polygon has zero area, cannot compute a centroid")
    return centroid_x / total_area, centroid_y / total_area


def _load_division_shapes(shapefile_path: Path) -> dict[str, dict]:
    """Map NCEI climate division code to shape, name, and state."""
    import shapefile  # imported lazily so the module loads without the optional dependency

    reader = shapefile.Reader(str(shapefile_path))
    field_names = [field[0] for field in reader.fields[1:]]
    index_climdiv = field_names.index("CLIMDIV")
    index_name = field_names.index("NAME")
    index_state = field_names.index("ST_ABBRV")

    divisions = {}
    for record_index, record in enumerate(reader.records()):
        code = f"{record[index_climdiv]:04d}"
        divisions[code] = {
            "shape": reader.shape(record_index),
            "name": record[index_name],
            "state": record[index_state],
        }
    return divisions


def _write_divisions(directory: Path, divisions: list[dict]) -> None:
    (directory / "divisions.json").write_text(json.dumps(divisions, indent=2) + "\n", encoding="utf-8")


def _compute_checksum(directory: Path) -> str:
    hasher = hashlib.sha256()
    for npy_file in sorted(directory.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    return hasher.hexdigest()


def _load_temps_fahrenheit(division: str) -> np.ndarray:
    """Load a division's monthly temperatures as float, parsing legacy strings.

    The committed ``temps.npy`` arrays hold a legacy object dtype mixing floats
    and numeric strings; the nClimDiv source values are degrees Fahrenheit.
    """
    values = np.load(_PALMER_ROOT / division / "temps.npy", allow_pickle=True)
    return np.array([float(str(value).split()[0]) for value in values], dtype=float)


def _categories(values: np.ndarray) -> np.ndarray:
    """SPEI drought-category bins for the standard category boundaries."""
    return np.digitize(values, _CATEGORY_BOUNDARIES)


def _agreement(computed: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Correlation, sign agreement, category agreement, and mean |difference|.

    Statistics and their definitions match the assertions in
    tests/test_speibase_reference.py, measured over the months both series hold.
    """
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


def _computed_spei_series(division: str, latitude: float, scale: int) -> np.ndarray:
    """climate_indices SPEI for one division and timescale over 1901-2022.

    The same compared series tests/test_speibase_reference.py builds: the
    committed nClimDiv inputs (inches, Fahrenheit), Thornthwaite PET, gamma
    distribution, and full-period-of-record calibration.
    """
    from climate_indices import compute, eto, indices

    precip_mm = (
        np.load(_PALMER_ROOT / division / "precips.npy").astype(np.float64)[_INPUT_OFFSET : _INPUT_OFFSET + _N_MONTHS]
        * 25.4
    )  # committed nClimDiv precipitation is inches
    temps_c = (_load_temps_fahrenheit(division)[_INPUT_OFFSET : _INPUT_OFFSET + _N_MONTHS] - 32.0) * (5.0 / 9.0)
    pet_mm = eto.eto_thornthwaite(temps_c, latitude, _DATA_START_YEAR)
    return indices.spei(
        precip_mm,
        pet_mm,
        scale,
        indices.Distribution.gamma,
        compute.Periodicity.monthly,
        _DATA_START_YEAR,
        _DATA_START_YEAR,
        _DATA_END_YEAR,
    )


def _measure_agreement(arrays: dict[int, np.ndarray], rows: list[dict]) -> dict[str, dict[int, dict[str, float]]]:
    """Measure the agreement of the freshly built arrays, per division and timescale."""
    measured = {}
    for row_index, row in enumerate(rows):
        measured[row["id"]] = {
            scale: _agreement(_computed_spei_series(row["id"], row["latitude"], scale), arrays[scale][row_index])
            for scale in _SCALES
        }
    return measured


def _check_expectations(measured: dict[str, dict[int, dict[str, float]]]) -> None:
    """Refuse a refresh whose agreement drifted away from the recorded expectations."""
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


def _floors(measured: dict[str, dict[int, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Per-division, per-timescale floors, rounded down, from the measurements."""
    floors = {}
    for division, per_scale in measured.items():
        for scale, stats in per_scale.items():
            floors[f"{division}_spei{scale:02d}"] = {
                metric: math.floor((stats[metric] - _FLOOR_MARGINS[metric]) * 100.0) / 100.0
                for metric in _FLOOR_METRICS
            }
    return floors


def _write_provenance(directory: Path, checksum: str, measured: dict[str, dict[int, dict[str, float]]]) -> None:
    provenance = {
        "source": "Consejo Superior de Investigaciones Científicas (CSIC)",
        "url": f"{_SPEIBASE_BASE_URL}/",
        "download_date": dt.date.today().isoformat(),
        "subset_description": (
            "SPEIbase v2.11 global 0.5-degree monthly SPEI, averaged per month across the "
            "0.5-degree grid cells whose centers fall inside three CONUS climate divisions "
            "spanning an aridity gradient: 0101 (Northern Valley, Alabama; humid), 3405 "
            "(Central, Oklahoma; subhumid), and 0205 (Southwest, Arizona; arid). The stored "
            "series is the mean of the per-cell SPEI values, not an index computed from "
            "averaged inputs. Timescales 1, 3, 6 and 12 months, "
            "January 1901 through December 2022 (1464 months), truncated to the period shared "
            "with the committed tests/fixture/palmer/<division>/{precips,temps}.npy inputs. "
            "Each <scale>.npy is a (3, 1464) float32 array whose row order matches "
            "divisions.json. Cell membership comes from the NCEI CONUS_CLIMATE_DIVISIONS.shp.zip "
            "polygons."
        ),
        "checksum_sha256": checksum,
        "fixture_version": "1.0.0",
        "validation_tolerance": {
            f"{key}_{metric}": value for key, metrics in _floors(measured).items() for metric, value in metrics.items()
        },
        "measured_stats": {
            f"{division}_spei{scale:02d}": stats
            for division, per_scale in measured.items()
            for scale, stats in per_scale.items()
        },
        "citation": (
            "Beguería, S., Vicente-Serrano, S. M., Reig-Gracia, F., and Latorre Garcés, B. "
            "(2024). SPEIbase v.2.11 [Dataset]. DIGITAL.CSIC. "
            "https://doi.org/10.20350/digitalCSIC/16497. Methodology: Vicente-Serrano, S. M., "
            "Beguería, S., and López-Moreno, J. I. (2010), Journal of Climate 23(7), 1696-1718; "
            "Beguería, S. et al. (2014), International Journal of Climatology 34(10), 3001-3023."
        ),
        "doi": "10.20350/digitalCSIC/16497",
        "license": (
            "CC-BY 4.0 per the Google Earth Engine catalog entry; the spei.csic.es download "
            "site states the Open Database License (ODbL 1.0), which adds a share-alike "
            "condition. Attribution required either way; confirm the governing terms when "
            "refreshing."
        ),
        "notes": (
            "PLAUSIBILITY CHECK ONLY, NOT EXTERNAL NUMERICAL VALIDATION. Three known confounds "
            "separate these series: (1) SPEIbase v2.11 uses FAO-56 Penman-Monteith PET from CRU "
            "TS 4.09, while climate_indices' SPEI here uses Thornthwaite PET computed from the "
            "committed nClimDiv temperatures; the two PET families diverge with climate aridity "
            "(van der Schrier et al. 2011; see docs/research/spei-dataset-survey.md on branch "
            "research/spei-dataset-survey); (2) SPEIbase standardizes with the log-logistic "
            "distribution while the compared series uses gamma (climate_indices has no "
            "log-logistic implementation, see issue #106); (3) the reference is a 0.5-degree "
            "grid average inside a climate division polygon while the compared series is the "
            "division's station-derived areal average, and their precipitation inputs (CRU TS, "
            "nClimDiv) differ. Cell-center selection makes the reference a sparse sample (3 to 8 "
            "cells per division), not a full areal mean. The measured agreement is lowest in the "
            "arid Arizona division -- the direction the PET-family mismatch predicts -- and "
            "highest in subhumid Oklahoma; with only three divisions the ordering cannot isolate "
            "the PET-family confound from the distribution and precipitation-support confounds, "
            "which is why no tight numerical gate is defensible. Tests assert per-division, "
            "per-timescale correlation, sign-agreement, and drought-category-agreement floors "
            "derived from measured_stats with the slack recorded in validation_tolerance; they "
            "must not be tightened into an atol/rtol gate or described as independent validation. "
            "SPEIbase v2.11 spans 1901-2024; the fixtures retain 1901-2022 to share the "
            "committed input period, and the compared climate_indices series is calibrated "
            "against the full 1901-2022 period of record."
        ),
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def _grid_cell_mean(
    scale: int,
    netcdf_path: Path,
    masks: dict[str, np.ndarray],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    """Average one SPEIbase timescale over each division's selected grid cells.

    Unweighted: the selected cell centers are a sparse sample (3 to 8 cells per
    division), not a full areal mean.
    """
    import xarray as xr

    dataset = xr.open_dataset(netcdf_path, engine="h5netcdf")
    try:
        if tuple(dataset.spei.dims) != ("time", "lat", "lon"):
            raise RuntimeError(f"SPEI-{scale} has unexpected dims {tuple(dataset.spei.dims)}")
        if dataset.sizes["time"] < _N_MONTHS:
            raise RuntimeError(f"SPEI-{scale} has only {dataset.sizes['time']} months, need {_N_MONTHS}")
        time_first = str(dataset.time.values[0])[:7]
        if time_first != f"{_DATA_START_YEAR}-01":
            raise RuntimeError(f"SPEI-{scale} starts at {time_first}, expected {_DATA_START_YEAR}-01")
        file_latitudes = dataset.lat.sel(lat=slice(*_LATITUDE_BAND)).values
        file_longitudes = dataset.lon.sel(lon=slice(*_LONGITUDE_BAND)).values
        if not np.array_equal(file_latitudes, latitudes) or not np.array_equal(file_longitudes, longitudes):
            raise RuntimeError(f"SPEI-{scale} grid does not match the reference grid")
        values = dataset.spei.sel(lat=slice(*_LATITUDE_BAND), lon=slice(*_LONGITUDE_BAND)).values[:_N_MONTHS]
    finally:
        dataset.close()

    array = np.full((len(_DIVISIONS), _N_MONTHS), np.nan, dtype=np.float32)
    for row_index, division in enumerate(_DIVISIONS):
        selected = values[:, masks[division]]
        # leading scale-1 rows are all-NaN by construction (rolling-sum warmup),
        # so only the non-warmup rows are required to report values
        if np.isnan(selected).all() or np.isnan(selected[scale - 1 :]).all(axis=1).any():
            raise RuntimeError(f"SPEI-{scale} division {division}: selected cells are all-NaN")
        with warnings.catch_warnings():
            # leading scale-1 months are all-NaN by construction (rolling-sum warmup)
            warnings.simplefilter("ignore", RuntimeWarning)
            array[row_index] = np.nanmean(selected, axis=1)
    return array


def _publish(staging: Path) -> None:
    """Replace the published fixture directory with the staged one, as one transaction.

    Arrays and metadata must always come from a single generation: replacing
    the six files individually leaves a mixed set behind when a later
    replacement fails. The previous directory is kept as a backup until the
    swap succeeds, and restored if it does not.
    """
    backup = OUTPUT_DIR.with_name(f".{OUTPUT_DIR.name}-backup")
    shutil.rmtree(backup, ignore_errors=True)
    if OUTPUT_DIR.exists():
        os.replace(OUTPUT_DIR, backup)
    try:
        os.replace(staging, OUTPUT_DIR)
    except BaseException:
        if backup.exists():
            os.replace(backup, OUTPUT_DIR)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def main() -> None:
    """Download the dataset, build the fixtures, and write them to tests/fixture/speibase/."""
    import xarray as xr

    staging = Path(tempfile.mkdtemp(prefix=".speibase-staging-", dir=FIXTURE_DIR))
    downloads = Path(tempfile.mkdtemp(prefix="speibase-downloads-"))
    try:
        print("Downloading NCEI climate division shapefile ...", file=sys.stderr)
        shapefile_path = _download_shapefile(downloads)
        division_shapes = _load_division_shapes(shapefile_path)
        missing = [division for division in _DIVISIONS if division not in division_shapes]
        if missing:
            raise RuntimeError(f"divisions absent from the shapefile: {missing}")

        print(f"Downloading SPEIbase v2.11 SPEI-{', '.join(str(s) for s in _SCALES)} ...", file=sys.stderr)
        netcdf_paths = {scale: _download_speibase_netcdf(scale, downloads) for scale in _SCALES}

        reference = xr.open_dataset(netcdf_paths[_SCALES[0]], engine="h5netcdf")
        try:
            latitudes = reference.lat.sel(lat=slice(*_LATITUDE_BAND)).values
            longitudes = reference.lon.sel(lon=slice(*_LONGITUDE_BAND)).values
        finally:
            reference.close()
        grid_longitudes, grid_latitudes = np.meshgrid(longitudes, latitudes)
        if grid_longitudes.size == 0:
            raise RuntimeError("CONUS grid selection is empty, refusing to write fixtures")

        rows = []
        masks = {}
        for division in _DIVISIONS:
            shape = division_shapes[division]["shape"]
            mask = _polygon_mask(shape, grid_longitudes, grid_latitudes)
            if not mask.any():
                raise RuntimeError(f"no SPEIbase cells inside climate division {division}")
            longitude, latitude = _polygon_centroid(shape)
            masks[division] = mask
            rows.append(
                {
                    "id": division,
                    "name": division_shapes[division]["name"],
                    "state": division_shapes[division]["state"],
                    "latitude": round(latitude, 4),
                    "longitude": round(longitude, 4),
                    "speibase_cells": int(mask.sum()),
                }
            )
            print(
                f"  division {division} ({division_shapes[division]['state']}): "
                f"{mask.sum()} cells, centroid {latitude:.3f}, {longitude:.3f}",
                file=sys.stderr,
            )

        arrays = {}
        for scale in _SCALES:
            arrays[scale] = _grid_cell_mean(scale, netcdf_paths[scale], masks, latitudes, longitudes)
            print(f"  SPEI-{scale}: prepared {arrays[scale].shape}", file=sys.stderr)

        print("Measuring the agreement with this library's SPEI ...", file=sys.stderr)
        measured = _measure_agreement(arrays, rows)
        _check_expectations(measured)

        for scale, array in arrays.items():
            np.save(staging / f"spei{scale:02d}.npy", array)
        _write_divisions(staging, rows)
        checksum = _compute_checksum(staging)
        _write_provenance(staging, checksum, measured)
        _publish(staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(downloads, ignore_errors=True)

    print(f"Done. checksum_sha256={checksum}", file=sys.stderr)


if __name__ == "__main__":
    main()
