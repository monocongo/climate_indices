"""Cross-validation pilot: climate_indices against Earth Engine and Climate Engine.

Compares SPI, SPEI and EDDI computed by this library with two references, kept
separate in the report:

* A. the precomputed ``GRIDMET/DROUGHT`` pentad collection in the Earth Engine
  Data Catalog (non-parametric standardization, documented common baseline
  1981-2016);
* B. the Climate Engine API ``/timeseries/standard_index/coordinates`` endpoint
  (``distribution=nonparametric``, ``start_year=1981``, ``end_year=2016``).

Both references standardize by rank with a plotting position, so this script
uses ``Distribution.empirical`` (SEI plotting position ``(k + 1) / (n + 2)``)
for SPI and SPEI and :func:`climate_indices.eddi` (Tukey plotting position) rather than
gamma or log-logistic. The method alignment table and every known mismatch are
recorded in ``docs/research/earthengine-climate-engine-pilot.md``; nothing here
tunes a setting to a reference output.

Region: the Apalachicola-Chattahoochee-Flint basin bounding box, read on the
native 4-km gridMET grid. Monthly accumulation is performed server-side in
Earth Engine (one image per calendar month) because the daily collection holds
about 17,000 images; see the Xee guidance on consolidating work server-side.

Credentials (never logged, never committed):

* ``EE_PROJECT`` -- a Google Cloud project registered for Earth Engine.
  Authentication itself is OAuth2: run ``earthengine authenticate`` once, or set
  ``EE_SERVICE_ACCOUNT`` and ``EE_PRIVATE_KEY_FILE``. A Google Cloud *API key*
  cannot authenticate the Earth Engine API.
* ``CE_API_KEY`` -- a Climate Engine API key. When it is absent, reference B is
  reported as pending and only reference A runs.

Run: ``uv run --extra earthengine scripts/earthengine_climate_engine_pilot.py``
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from climate_indices import eddi, standardized_index
from climate_indices.drought.indices import Distribution
from climate_indices.provenance import software_identity

# Region and period. The bounding box is a rectangle around HUC 0313, not the
# basin polygon: the comparison is per grid cell, so a polygon clip would only
# change which cells are counted, and the rectangle keeps both references on one
# grid.
PILOT_BBOX = (-85.7, 29.5, -83.3, 35.0)
DAILY_COLLECTION = "IDAHO_EPSCOR/GRIDMET"
DROUGHT_COLLECTION = "GRIDMET/DROUGHT"
HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"
CE_ENDPOINT = "https://api.climateengine.org/timeseries/standard_index/coordinates"
CE_NATIVE_ENDPOINT = "https://api.climateengine.org/timeseries/native/coordinates"
CE_POINTS = ((-84.9, 30.7), (-84.4, 33.0), (-84.0, 34.5))

# The catalog's documented common baseline for the precomputed drought indices.
CALIBRATION_YEARS = (1981, 2016)
# Input window: 1981 minus the longest scale, so a 12-month value exists at 1981-01.
INPUT_YEARS = (1980, 2024)
# Comparison window: the whole record the calibration and the catalog cover.
COMPARISON_YEARS = (1981, 2024)

# climate_indices calendar-month scales mapped to the catalog's nearest fixed-day
# scale. The window definitions differ (calendar month versus trailing days); the
# report quantifies and states that difference rather than hiding it.
SCALES = {1: "30d", 3: "90d", 6: "180d", 12: "1y"}
SCALE_DAYS = {1: 30, 3: 90, 6: 180, 12: 365}

# The catalog's documented SPI/SPEI/EDDI interpretation bins, ordered from
# exceptional drought to exceptional wet, as inclusive lower edges.
USDM_EDGES = np.array([-np.inf, -2.0, -1.6, -1.3, -0.8, -0.5, 0.5, 0.8, 1.3, 1.6, 2.0, np.inf])

# Plausible domain-mean monthly totals for the pilot region, in millimetres. The
# bounds are wide and region-specific on purpose: they exist to catch a unit
# mistake, not to judge the climate.
MONTHLY_TOTAL_BANDS = {"precipitation": (5.0, 500.0), "reference_et": (1.0, 500.0), "alfalfa_et": (1.0, 600.0)}

# North-up degree grids only; the bounding box is EPSG:4326 and a metric grid would
# need ``fit_geometry`` instead of a crop.
DEGREE_GRID_LIMIT = 1.0


@dataclass(frozen=True)
class MonthEnd:
    """One calendar month and the last day that identifies it."""

    year: int
    month: int
    last_day: np.datetime64

    @property
    def label(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"


def month_ends(start_year: int, end_year: int) -> list[MonthEnd]:
    """Return every calendar month end from January ``start_year`` to December ``end_year``."""
    return [
        MonthEnd(
            year,
            month,
            np.datetime64(f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"),
        )
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


def last_pentad_index_per_month(stamps_ms: Any, months: list[MonthEnd]) -> list[int]:
    """Index the pentad image that ends each calendar month.

    Each month takes the latest pentad whose timestamp is on or before that
    month's last day, so a month never borrows a pentad from the next one.
    Returns ``-1`` for a month with no pentad at all, which the caller reports
    rather than interpolating.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(stamps_ms, dtype="int64"), unit="ms")).normalize()
    chosen: list[int] = []
    for month in months:
        candidates = np.nonzero(stamps <= month.last_day)[0]
        chosen.append(int(candidates[-1]) if candidates.size else -1)
    return chosen


def usdm_bin(values: np.ndarray) -> np.ndarray:
    """Bin a standardized index into the catalog's documented category integers."""
    binned = np.digitize(np.asarray(values, dtype="float64"), USDM_EDGES[1:-1], right=False)
    return np.asarray(binned, dtype=np.int8)


def comparison_metrics(ours: Any, theirs: Any) -> dict[str, float]:
    """Summarize agreement over the pairs where both values exist.

    ``bias`` is ``mean(ours - theirs)``; ``category_agreement`` is the fraction of
    pairs in the same documented category, and ``category_within_one`` allows one
    class of disagreement. Missing pairs are dropped before binning, so a NaN
    never counts as agreement.
    """
    a = np.asarray(ours, dtype="float64").ravel()
    b = np.asarray(theirs, dtype="float64").ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} versus {b.shape}")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size == 0:
        return {"n": 0.0}
    difference = a - b
    spread = float(np.std(b, ddof=1)) if b.size > 1 else float("nan")
    return {
        "n": float(a.size),
        # Reported separately so the report can show that part of the bias is a
        # property of the plotting-position convention rather than of the data.
        "mean_ours": float(np.mean(a)),
        "mean_theirs": float(np.mean(b)),
        "bias": float(np.mean(difference)),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "correlation": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan"),
        "max_abs_difference": float(np.max(np.abs(difference))),
        # scale-free error, comparable between indices and scales
        "normalized_rmse": float(np.sqrt(np.mean(difference**2)) / spread) if spread else float("nan"),
        "category_agreement": float(np.mean(usdm_bin(a) == usdm_bin(b))),
        "category_within_one": float(np.mean(np.abs(usdm_bin(a).astype(int) - usdm_bin(b).astype(int)) <= 1)),
    }


def require_ee_project() -> str:
    """Return ``EE_PROJECT`` or exit with the setup the user has to do."""
    project = os.environ.get("EE_PROJECT", "").strip()
    if not project:
        raise SystemExit(
            "EE_PROJECT is not set. Set it to a Google Cloud project registered for Earth Engine, e.g.\n"
            "  export EE_PROJECT=your-project-id\n"
            "and authenticate once with `uv run --extra earthengine earthengine authenticate`."
        )
    return project


def initialize_ee(project: str) -> Any:
    """Initialize Earth Engine without ever printing or logging a credential."""
    import ee

    service_account = os.environ.get("EE_SERVICE_ACCOUNT", "").strip()
    key_file = os.environ.get("EE_PRIVATE_KEY_FILE", "").strip()
    if service_account and key_file:
        credentials = ee.ServiceAccountCredentials(service_account, key_file)
        ee.Initialize(credentials=credentials, project=project, url=HIGH_VOLUME_URL)
    else:
        ee.Initialize(project=project, url=HIGH_VOLUME_URL)
    return ee


def monthly_server_side(ee: Any, collection: Any, months: list[MonthEnd]) -> Any:
    """Sum a daily collection over each calendar month, server-side.

    One image per month is what makes the daily collection affordable to read:
    about 550 images instead of about 17,000. Each image's timestamp is the
    month's last day, so the time coordinate matches the comparison window.
    """
    images = []
    for month in months:
        start = ee.Date.fromYMD(month.year, month.month, 1)
        last_day = start.advance(1, "month").advance(-1, "day")
        images.append(
            collection.filterDate(start, last_day.advance(1, "day")).sum().set("system:time_start", last_day.millis())
        )
    return ee.ImageCollection(images)


def native_grid(collection: Any) -> dict[str, Any]:
    """Native grid of a collection, before any cropping."""
    from xee import helpers

    return dict(helpers.extract_grid_params(collection))


def crop_grid_to_bbox(native: dict[str, Any], bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """Crop a north-up degree grid to the smallest pixel-aligned window covering ``bbox``.

    The crop keeps the source pixel edges, so Earth Engine returns stored pixel
    values instead of resampling them onto a different origin. Snapping to the
    native lattice is what makes the two references comparable cell by cell.
    """
    scale_x, shear_x, origin_x, shear_y, scale_y, origin_y = native["crs_transform"]
    if shear_x or shear_y:
        raise ValueError("only north-up grids without shear are supported")
    if not 0 < abs(scale_x) < 1 or not 0 < abs(scale_y) < 1:
        raise ValueError("grid pixel size is not in degrees, but the bounding box is in EPSG:4326")
    lon_min, lat_min, lon_max, lat_max = bbox
    column_first = math.floor((lon_min - origin_x) / scale_x)
    column_last = math.ceil((lon_max - origin_x) / scale_x)
    row_first = math.floor((origin_y - lat_max) / abs(scale_y))
    row_last = math.ceil((origin_y - lat_min) / abs(scale_y))
    cropped = dict(native)
    cropped["crs_transform"] = (
        scale_x,
        0.0,
        origin_x + column_first * scale_x,
        0.0,
        scale_y,
        origin_y - row_first * abs(scale_y),
    )
    cropped["shape_2d"] = (column_last - column_first, row_last - row_first)
    return cropped


def grids_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether two grid parameter sets describe the same pixels in the same projection."""
    return (
        left.get("crs") == right.get("crs")
        and left["shape_2d"] == right["shape_2d"]
        and bool(
            np.allclose(
                np.asarray(left["crs_transform"], dtype="float64"),
                np.asarray(right["crs_transform"], dtype="float64"),
                rtol=0.0,
                atol=1e-9,
            )
        )
    )


def check_monthly_totals(values: Any, name: str) -> None:
    """Reject negative or implausibly small or large monthly totals.

    Earth Engine does not carry band units into the xarray attributes, so the
    source unit is taken from the catalog (millimetres per day) instead of being
    read back. These wide bands catch the two silent failures that matter: a
    metres-versus-millimetres scaling, which is three orders of magnitude low, and
    a per-day value that was never summed.
    """
    low, high = MONTHLY_TOTAL_BANDS[name]
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError(f"{name} holds no finite values")
    if finite.min() < 0.0:
        raise RuntimeError(f"{name} holds negative values, which no monthly total of this band can be")
    domain_mean = float(np.mean(finite))
    if not low <= domain_mean <= high:
        raise RuntimeError(
            f"{name} has a domain mean of {domain_mean:.3g} mm per month, outside the plausible "
            f"{low:g}-{high:g} mm band for this region; check the source unit before trusting the result"
        )


def open_monthly_gridmet(ee: Any, bbox: tuple[float, float, float, float], grid: dict[str, Any]) -> xr.Dataset:
    """Open daily gridMET ``pr`` and ``eto`` as calendar-month sums on the cropped native grid.

    Units: gridMET publishes ``pr`` and ``eto`` as millimetres per day, so a
    monthly sum is millimetres per month, which is what the index functions take.
    Temperature is not read, so the kelvin bands never enter the comparison.
    """
    months = month_ends(*INPUT_YEARS)
    collection = (
        ee.ImageCollection(DAILY_COLLECTION)
        .filterDate(f"{INPUT_YEARS[0]}-01-01", f"{INPUT_YEARS[1] + 1}-01-01")
        .filterBounds(ee.Geometry.Rectangle(bbox))
        .select(["pr", "eto", "etr"])
    )
    dataset = xr.open_dataset(monthly_server_side(ee, collection, months), engine="ee", **grid)
    if dataset.sizes.get("time") != len(months):
        raise RuntimeError(f"expected {len(months)} monthly steps, got {dataset.sizes.get('time')}")
    dataset = dataset.assign_coords(time=("time", [month.last_day for month in months]))
    dataset = dataset.rename({"pr": "precipitation", "eto": "reference_et", "etr": "alfalfa_et"})
    for name in ("precipitation", "reference_et", "alfalfa_et"):
        check_monthly_totals(dataset[name].values, name)
    dataset["precipitation"].attrs.update(
        units="mm",
        long_name="precipitation total",
        standard_name="precipitation_amount",
        cell_methods="time: sum",
        comment="calendar-month sum of daily gridMET pr",
    )
    dataset["reference_et"].attrs.update(
        units="mm",
        long_name="ASCE grass reference evapotranspiration total",
        standard_name="water_potential_evapotranspiration_amount",
        cell_methods="time: sum",
        comment="calendar-month sum of daily gridMET eto",
    )
    dataset["alfalfa_et"].attrs.update(
        units="mm",
        long_name="ASCE alfalfa reference evapotranspiration total",
        standard_name="water_potential_evapotranspiration_amount",
        cell_methods="time: sum",
        comment="calendar-month sum of daily gridMET etr; the catalog does not say which reference crop its EDDI uses",
    )
    return dataset


def open_drought_reference(
    ee: Any, comparison_months: list[MonthEnd], grid: dict[str, Any]
) -> tuple[xr.Dataset, dict[str, float]]:
    """Open one pentad per month from ``GRIDMET/DROUGHT`` on the shared grid.

    Only one pentad per month enters the comparison, so the chosen images are
    listed client-side from a single timestamp query instead of reading every
    pentad. ``comparison_months`` covers the comparison window, which starts after
    the input window's first year. Returns the dataset and a summary of how far
    each chosen pentad sits from its month end, which the report reports as the
    sampling offset rather than hiding it.
    """
    collection = (
        ee.ImageCollection(DROUGHT_COLLECTION)
        .filterDate(f"{COMPARISON_YEARS[0]}-01-01", f"{COMPARISON_YEARS[1] + 1}-01-01")
        .sort("system:time_start")
    )
    stamps = collection.aggregate_array("system:time_start").getInfo()
    chosen = last_pentad_index_per_month(stamps, comparison_months)
    missing = [comparison_months[position].label for position, index in enumerate(chosen) if index < 0]
    if missing:
        raise RuntimeError(f"no pentad found for {len(missing)} months, first {missing[0]}")
    if len(set(chosen)) != len(chosen):
        raise RuntimeError("two months selected the same pentad; the sampling rule is ambiguous")
    images = collection.toList(collection.size())
    dataset = xr.open_dataset(
        ee.ImageCollection(ee.List(chosen).map(lambda index: images.get(index))), engine="ee", **grid
    )
    if dataset.sizes.get("time") != len(comparison_months):
        raise RuntimeError(f"expected {len(comparison_months)} pentads, got {dataset.sizes.get('time')}")

    # Read only the twelve bands the comparison uses; the collection carries 26.
    wanted_bands = sorted({f"{index}{suffix}" for suffix in SCALES.values() for index in ("spi", "spei", "eddi")})
    absent = [band for band in wanted_bands if band not in dataset]
    if absent:
        raise RuntimeError(f"{DROUGHT_COLLECTION} does not carry {absent}; check the catalog band names")
    dataset = dataset[wanted_bands]

    # Whole-day offsets from integer milliseconds, so a pentad stamped at 06:00 UTC
    # still counts as the calendar day it lands on.
    milliseconds_per_day = 86_400_000
    anchor_days: list[int] = []
    for position, index in enumerate(chosen):
        month_end_day = int(
            np.datetime64(comparison_months[position].last_day).astype("datetime64[ms]").astype("int64")
        )
        anchor_days.append(int(month_end_day // milliseconds_per_day - int(stamps[int(index)]) // milliseconds_per_day))
    anchor = {
        "pentads_read": float(len(chosen)),
        "anchor_days_before_month_end_min": float(min(anchor_days)),
        "anchor_days_before_month_end_mean": float(float(np.mean(anchor_days))),
        "anchor_days_before_month_end_max": float(max(anchor_days)),
    }
    return dataset.assign_coords(time=("time", [month.last_day for month in comparison_months])), anchor


def compute_indices(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Compute SPI, SPEI and EDDI at every scale with the reference-matching distribution.

    SPEI's input is a signed water balance, so ``clip_negatives`` is off: clipping
    would standardize a different series than either reference does.

    EDDI runs twice. The catalog does not say whether its EDDI uses grass reference
    ET (``eto``) or alfalfa reference ET (``etr``), and Climate Engine documents only
    ASCE grass reference ET. The grass variant is therefore the primary comparison,
    and the alfalfa variant is a sensitivity that tests the ambiguity instead of
    guessing at it.
    """
    precipitation = dataset["precipitation"]
    reference_et = dataset["reference_et"]
    initial, final = CALIBRATION_YEARS
    results: dict[str, xr.DataArray] = {}
    for scale in SCALES:
        results[f"spi_{scale}"] = standardized_index(
            precipitation,
            scale,
            Distribution.empirical,
            calibration_year_initial=initial,
            calibration_year_final=final,
        )
        results[f"spei_{scale}"] = standardized_index(
            precipitation - reference_et,
            scale,
            Distribution.empirical,
            calibration_year_initial=initial,
            calibration_year_final=final,
            clip_negatives=False,
        )
        for name, evaporative_demand in (("eddi", reference_et), ("eddi_alfalfa", dataset["alfalfa_et"])):
            results[f"{name}_{scale}"] = eddi(
                evaporative_demand,
                scale,
                calibration_year_initial=initial,
                calibration_year_final=final,
            )
    return results


def band_for(key: str) -> str:
    """Map a result key such as ``eddi_alfalfa_3`` to its catalog band ``eddi90d``."""
    index, _, scale = key.rpartition("_")
    index = index.split("_")[0]
    return f"{index}{SCALES[int(scale)]}"


def compare(
    ours: dict[str, xr.DataArray], reference: xr.Dataset
) -> tuple[dict[str, dict[str, float]], dict[str, xr.DataArray]]:
    """Compare every index and scale pair over the comparison window."""
    window = slice(f"{COMPARISON_YEARS[0]}-01-01", f"{COMPARISON_YEARS[1]}-12-31")
    metrics: dict[str, dict[str, float]] = {}
    fields: dict[str, xr.DataArray] = {}
    for key, ours_da in ours.items():
        band = band_for(key)
        if band not in reference:
            continue
        ours_window, reference_window = xr.align(
            ours_da.sel(time=window), reference[band].sel(time=window), join="inner"
        )
        metrics[key] = comparison_metrics(ours_window.values, reference_window.values)
        fields[key] = (ours_window - reference_window).assign_attrs(
            long_name=f"{key} difference (climate_indices minus {band})",
            units="standard deviations",
        )
    return metrics, fields


def climate_engine_request(endpoint: str, body: dict[str, Any], key: str) -> dict[str, Any]:
    """POST one request to the Climate Engine API and return the decoded JSON.

    The key travels in the ``Authorization`` header and is never written to any
    output, logged, or embedded in an error message.
    """
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:  # nosec B310 # fixed https endpoint
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # A remote error body can echo request headers, including the key.
        raise RuntimeError(f"Climate Engine API returned HTTP {error.code}") from None
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Climate Engine API returned a {type(decoded).__name__}, not a JSON object")
    return decoded


def climate_engine_series(key: str, scale: int, variable: str) -> dict[str, Any]:
    """Request one standard-index series from the Climate Engine API.

    The request body mirrors the published parameter set: ``accumulation`` is in
    the dataset's own unit (days for the daily GRIDMET dataset) and the
    calibration window is the catalog baseline.
    """
    body = {
        "coordinates": json.dumps([[lon, lat] for lon, lat in CE_POINTS]),
        "area_reducer": "mean",
        "dataset": "GRIDMET",
        "variable": variable,
        "distribution": "nonparametric",
        "accumulation": str(SCALE_DAYS[scale]),
        "start_year": CALIBRATION_YEARS[0],
        "end_year": CALIBRATION_YEARS[1],
        "start_date": f"{COMPARISON_YEARS[0]}-01-01",
        "end_date": f"{COMPARISON_YEARS[1]}-12-31",
    }
    return {"request": body, "response": climate_engine_request(CE_ENDPOINT, body, key)}


def climate_engine_native(key: str, dataset: str, variable: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Request one raw variable series from the Climate Engine API.

    The native endpoint returns the dataset's own values, which is what an index
    computed here needs as input rather than a finished standardized index.
    """
    body = {
        "coordinates": json.dumps([[lon, lat] for lon, lat in CE_POINTS]),
        "area_reducer": "mean",
        "dataset": dataset,
        "variable": variable,
        "start_date": start_date,
        "end_date": end_date,
    }
    return {"request": body, "response": climate_engine_request(CE_NATIVE_ENDPOINT, body, key)}


def parse_climate_engine_points(payload: dict[str, Any], variable: str) -> dict[tuple[float, float], pd.Series]:
    """Turn a Climate Engine time-series response into one dated Series per point.

    The response holds a list of blocks under ``Data``, one per requested
    coordinate and labelled by ``Metadata.DRI_OBJECTID``. Each block's ``Data`` is
    a list of ``{"Date": ..., "<variable>": value}`` records at the dataset's own
    cadence, so a daily dataset returns a daily index series. The API marks a
    missing observation with ``-9999``, which becomes NaN here so it cannot enter
    a comparison as a real value.
    """
    points: dict[tuple[float, float], pd.Series] = {}
    for block in payload["Data"]:
        label = block["Metadata"]["DRI_OBJECTID"]
        try:
            coordinates = json.loads(label)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"could not read the point label {label!r} from the response") from error
        if len(coordinates) != 2:  # noqa: PLR2004 - a point is a longitude and a latitude
            raise RuntimeError(f"expected a point coordinate, got {label!r}")
        records = block["Data"]
        if not records:
            raise RuntimeError(f"no records for point {label!r}")
        # The standard-index endpoint names the column after the variable; the
        # native endpoint appends the catalog unit, as in "pr (mm)".
        columns = records[0]
        value_column = variable if variable in columns else ""
        if not value_column:
            value_column = next((name for name in columns if name.split(" ")[0] == variable and name != "Date"), "")
        if not value_column:
            raise RuntimeError(f"records carry {sorted(columns)}, none of them {variable!r}")
        index = pd.to_datetime([record["Date"] for record in records])
        values = np.asarray([record[value_column] for record in records], dtype="float64")
        series = pd.Series(values, index=index, name=variable)
        points[(float(coordinates[0]), float(coordinates[1]))] = series.where(series > -9999.0)
    return points


def sample_at_month_ends(series: pd.Series, months: list[MonthEnd]) -> np.ndarray:
    """Take each month end's value from a possibly finer series.

    Uses the last observation at most four days before the month end. Longer
    gaps stay missing instead of copying a stale observation across months.
    """
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    values = series.reindex(
        pd.DatetimeIndex([month.last_day for month in months]), method="ffill", tolerance=pd.Timedelta(days=4)
    )
    return np.asarray(values.to_numpy(dtype="float64"))


def compare_climate_engine(
    ours: dict[str, xr.DataArray], key: str
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Compare every index and scale at the Climate Engine points.

    Returns pooled metrics per index and scale, plus the per-point metrics, the
    nearest grid cell used for each point, and the request that produced them.
    Only summaries are kept: the raw response is a daily series of about 16,000
    records per point.
    """
    window = pd.DatetimeIndex([month.last_day for month in month_ends(*COMPARISON_YEARS)])
    metrics: dict[str, dict[str, float]] = {}
    detail: dict[str, Any] = {}
    for scale in SCALES:
        for index in ("spi", "spei", "eddi"):
            key_name = f"{index}_{scale}"
            response = climate_engine_series(key, scale, index)
            points = parse_climate_engine_points(response["response"], index)
            if set(points) != set(CE_POINTS):
                raise RuntimeError(f"{key_name}: the API did not return exactly the requested points")
            pooled_ours: list[np.ndarray] = []
            pooled_theirs: list[np.ndarray] = []
            per_point: dict[str, dict[str, float]] = {}
            cells: list[list[float]] = []
            for (longitude, latitude), theirs in points.items():
                cell = ours[key_name].sel(x=longitude, y=latitude, method="nearest")
                # Select by the same month ends used to sample the API series, so
                # the two sides line up by date rather than by position.
                grid_series = np.asarray(cell.sel(time=window).values, dtype="float64")
                their_samples = sample_at_month_ends(theirs, month_ends(*COMPARISON_YEARS))
                if grid_series.shape != their_samples.shape:
                    raise RuntimeError(
                        f"{key_name}: {grid_series.shape[0]} grid steps against "
                        f"{their_samples.shape[0]} sampled API steps"
                    )
                per_point[f"{longitude},{latitude}"] = comparison_metrics(grid_series, their_samples)
                cells.append([float(cell["x"]), float(cell["y"])])
                pooled_ours.append(grid_series)
                pooled_theirs.append(their_samples)
            metrics[key_name] = comparison_metrics(np.concatenate(pooled_ours), np.concatenate(pooled_theirs))
            detail[key_name] = {
                "points": per_point,
                "grid_cells": cells,
                "request": response["request"],
            }
    return metrics, detail


def plot_differences(fields: dict[str, xr.DataArray], directory: Path) -> list[str]:
    """Write one map per index and scale showing where the largest differences sit."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    written = []
    for name, field in fields.items():
        mean_field = field.mean("time")
        magnitudes = np.abs(mean_field.values)
        finite = magnitudes[np.isfinite(magnitudes)]
        limit = (float(np.percentile(finite, 99)) if finite.size else 0.0) or 1.0
        figure, axis = plt.subplots(figsize=(6, 4.5))
        mesh = axis.pcolormesh(
            field["x"].values,
            field["y"].values,
            mean_field.values,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="auto",
        )
        figure.colorbar(mesh, ax=axis, label="standard deviations")
        axis.set_title(f"{name}: mean difference (climate_indices minus reference)")
        axis.set_xlabel("longitude")
        axis.set_ylabel("latitude")
        figure.tight_layout()
        figure.savefig(directory / f"difference-{name}.png", dpi=120)
        plt.close(figure)
        written.append(f"difference-{name}.png")
    return written


def collection_revision(ee: Any, asset_id: str) -> str:
    """Revision of a catalog asset, so the run can be tied to the inputs it read.

    The legacy Earth Engine catalog exposes no ``version`` for these
    collections, so their ``updateTime`` is the strongest revision the provider
    publishes; the caller pairs it with :func:`input_checksum`.
    """
    asset = ee.data.getAsset(asset_id)
    return str(asset.get("version") or asset.get("updateTime") or "unknown")


def input_checksum(dataset: xr.Dataset) -> str:
    """SHA-256 over one input's band names, time steps and read values.

    The catalog assets identify live products, not immutable snapshots, so this
    digest is what lets a reader detect a later provider update that would
    change the recorded metrics.
    """
    digest = hashlib.sha256()
    for band in sorted(str(band) for band in dataset.data_vars):
        values = np.ascontiguousarray(dataset[band].values, dtype="<f8")
        steps = np.ascontiguousarray(dataset[band]["time"].values, dtype="datetime64[s]").astype("int64")
        digest.update(f"{band}:{values.shape}:".encode())
        digest.update(steps.tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Run the pilot and write every number the report cites."""
    parser = argparse.ArgumentParser(description="Climate Engine and Earth Engine cross-validation pilot")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/research/earthengine-pilot-results"),
        help="directory for the metrics JSON and difference maps",
    )
    parser.add_argument("--skip-climate-engine", action="store_true", help="run reference A only")
    arguments = parser.parse_args(argv)

    project = require_ee_project()
    ee = initialize_ee(project)
    if ee.Number(1).getInfo() != 1:  # fail fast, before any collection is read
        raise SystemExit("Earth Engine did not answer a trivial computation")
    revisions = {
        "daily_revision": collection_revision(ee, DAILY_COLLECTION),
        "drought_revision": collection_revision(ee, DROUGHT_COLLECTION),
    }

    daily_native = native_grid(ee.ImageCollection(DAILY_COLLECTION).limit(1))
    drought_native = native_grid(ee.ImageCollection(DROUGHT_COLLECTION).limit(1))
    if not grids_match(daily_native, drought_native):
        raise SystemExit(
            "the daily and precomputed collections do not share a grid, so comparing them would resample; "
            "fix the grid handling before trusting any difference"
        )
    grid = crop_grid_to_bbox(daily_native, PILOT_BBOX)
    dataset = open_monthly_gridmet(ee, PILOT_BBOX, grid)
    reference, anchor = open_drought_reference(ee, month_ends(*COMPARISON_YEARS), grid)
    ours = compute_indices(dataset)
    metrics, fields = compare(ours, reference)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    # Stage every artifact, then move the set into place, so a failed run cannot
    # leave new maps beside the metrics JSON from an earlier run.
    staging = Path(tempfile.mkdtemp(prefix=".pilot-", dir=arguments.output_dir))
    payload: dict[str, Any] = {
        "software": software_identity(),
        "region": {"bbox": PILOT_BBOX, "crs": grid.get("crs"), "grid": "native gridMET 4 km"},
        "period": {
            "input_years": INPUT_YEARS,
            "calibration_years": CALIBRATION_YEARS,
            "comparison_years": COMPARISON_YEARS,
        },
        "method": {
            "ours": "SPI/SPEI: Distribution.empirical (SEI); EDDI: Tukey; per calendar month",
            "reference_a": "GRIDMET/DROUGHT, non-parametric, baseline 1981-2016, 5-day pentad",
            "scale_map": SCALES,
        },
        "collections": {
            "daily": DAILY_COLLECTION,
            "drought": DROUGHT_COLLECTION,
            "daily_input_sha256": input_checksum(dataset),
            "drought_input_sha256": input_checksum(reference),
            **revisions,
            **anchor,
        },
        "reference_a": metrics,
    }

    key = os.environ.get("CE_API_KEY", "").strip()
    if key and not arguments.skip_climate_engine:
        ce_metrics, ce_detail = compare_climate_engine(ours, key)
        payload["reference_b"] = {
            "endpoint": CE_ENDPOINT,
            "points": [list(point) for point in CE_POINTS],
            "sampling": "last daily record on or before each month end, at most four days earlier",
            "metrics": ce_metrics,
            "detail": ce_detail,
        }
    else:
        reason = "--skip-climate-engine was specified" if arguments.skip_climate_engine else "CE_API_KEY is not set"
        payload["reference_b"] = {"status": "pending", "reason": reason}

    try:
        payload["difference_maps"] = plot_differences(fields, staging)
        output = staging / "pilot-metrics.json"
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        for name in payload["difference_maps"]:
            (staging / name).replace(arguments.output_dir / name)
        output.replace(arguments.output_dir / output.name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"wrote {arguments.output_dir / output.name}", file=sys.stderr)
    frame = pd.DataFrame(metrics).T
    print(frame[["n", "bias", "mae", "rmse", "correlation", "max_abs_difference", "category_agreement"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
