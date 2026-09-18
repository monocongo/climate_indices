#!/usr/bin/env python3
# /// script
# dependencies = [
#   "cfgrib",
#   "numpy",
#   "xarray",
# ]
# ///
"""Prepare the Srock et al. (2018) Cedar Fire HDW validation fixture (issue #985).

Usage:
    uv run scripts/prepare_srock_hdw_fixtures.py

Writes ``tests/fixture/hdw_srock_cedar/``: the CFSR vertical profiles at the
grid point used for Figure 4a of Srock et al. (2018), plus the HDW series
digitized from that figure.

CFSR (NCEP Climate Forecast System Reanalysis) is free to use and needs no
registration; the NCEI object store serves 6-hourly pressure-level analyses as
global GRIB2 files with a companion ``.inv`` byte-offset inventory, so this
script fetches only the messages it needs with HTTP range requests. Downloads
are cached under the system temp directory, so re-runs after a partial failure
do not refetch completed messages.

The published series is transcribed from Figure 4a rather than downloaded:
MDPI publishes only the figure, no data table. Digitization calibrated the
y axis on the figure's own 0-500 gridline ticks (50-unit spacing) and the x
axis on the 29 daily tick marks (Oct 12 through Nov 09); each value is the
vertical center of the black marker at its tick, mapped through that linear
calibration, and the pixel pairs are committed alongside the values so the
mapping stays auditable. One pixel is about 0.94 HDW units, so the digitized
values carry a conservative +/-5 hPa m s-1 tolerance. Issue #985 records the
decision to assert event timing and series-shape agreement instead of a
numeric match: only the figure is published, and the paper's adiabatically
adjusted, independently maximized formulation is not the library's per-level
product.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cfgrib
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture" / "hdw_srock_cedar"
CACHE_DIR = Path(tempfile.gettempdir()) / "srock_hdw_cache"

_BASE_URL = "https://www.ncei.noaa.gov/oa/prod-cfs-reanalysis"
_DATASET_URL = "https://www.ncei.noaa.gov/oa/prod-cfs-reanalysis/"
_PDF_URL = "https://research.fs.usda.gov/download/treesearch/56562.pdf"
_PDF_SHA256 = "500c2961f37c3cf71ee5714273b8a5fd429754625aed5cbdde65a3dc5b9fc0e3"

# Figure 4a's grid point and window (the paper's Cedar Fire case study).
_GRID_LATITUDE = 33.0
_GRID_LONGITUDE = 243.5  # 116.5 W expressed in [0, 360)
_START_DATE = dt.date(2003, 10, 12)
_END_DATE = dt.date(2003, 11, 9)
_CYCLES_UTC = (0, 12, 18)  # the paper's 1200/1800/0000 UTC analyses
_PRESSURE_LEVELS_HPA = (925, 900, 875, 850, 825, 800)

# The library's layer is meters AGL, so profiles need a surface reference. The
# CFSR pressure-level files carry geopotential height, and the surface height
# is recovered from the lowest level above the surface with the hypsometric
# equation (dry-air constant and standard gravity, mean layer temperature).
_SURFACE_LEVEL_HEIGHT_METERS = 2.0
_R_DRY_AIR = 287.05
_GRAVITY = 9.80665

# Figure 4a of Srock et al. (2018), digitized 2026-09-18. Each entry is the
# pixel (column, row) of one day's black marker center, Oct 12 through Nov 9,
# in the figure's top-left-origin coordinates. The y axis was calibrated on
# the left-axis ticks for 500..0; every published value is the marker row
# mapped through that linear fit, and both the pixels and the fit inputs are
# committed so the transcription can be re-derived.
_Y_TICK_PIXELS = (118, 168, 220, 274, 328, 380, 434, 488, 542, 594, 648)
_Y_TICK_VALUES = (500.0, 450.0, 400.0, 350.0, 300.0, 250.0, 200.0, 150.0, 100.0, 50.0, 0.0)
_FIGURE4A_MARKER_PIXELS = (
    (164, 476),
    (217, 575),
    (270, 482),
    (322, 510),
    (375, 584),
    (427, 394),
    (480, 357),
    (532, 430),
    (585, 453),
    (638, 448),
    (690, 413),
    (743, 369),
    (795, 496),
    (848, 405),
    (900, 220),
    (953, 347),
    (1006, 504),
    (1058, 465),
    (1111, 584),
    (1163, 584),
    (1216, 600),
    (1268, 582),
    (1321, 590),
    (1374, 620),
    (1426, 620),
    (1479, 582),
    (1531, 587),
    (1584, 578),
    (1636, 575),
)

_MAX_DOWNLOAD_ATTEMPTS = 4
_REQUEST_TIMEOUT_SECONDS = 180
_DOWNLOAD_WORKERS = 8

# A per-run decode directory keeps simultaneous sessions (this repository's
# worktree workflow) from sharing a GRIB file or its index cache.
_DECODE_PATH = Path(tempfile.mkdtemp(prefix="srock_hdw_decode_")) / "decode.grb2"


def _timestamp_label(moment: dt.datetime) -> str:
    return moment.strftime("%Y%m%d%H")


def _timestamps() -> list[dt.datetime]:
    """Every analysis time in the Figure 4a window, in order."""
    moments: list[dt.datetime] = []
    day = _START_DATE
    while day <= _END_DATE:
        moments.extend(dt.datetime(day.year, day.month, day.day, hour, tzinfo=dt.timezone.utc) for hour in _CYCLES_UTC)
        day += dt.timedelta(days=1)
    return moments


def _fetch(url: str, offset: int | None = None, length: int | None = None) -> bytes:
    """Download a URL, optionally one byte range, with retries and a disk cache.

    Range requests are cached per URL and offset because a refresh run is a
    few thousand small transfers; a single transient failure should not cost
    the whole download. An open-ended range is allowed (``length`` is ``None``)
    for the last inventory record in a file.
    """
    key = hashlib.sha256(f"{url}:{offset}:{length}".encode()).hexdigest()
    cached = CACHE_DIR / key
    if cached.exists():
        payload = cached.read_bytes()
        if length is None or len(payload) == length:
            return payload
        cached.unlink(missing_ok=True)

    headers = {}
    if offset is not None:
        end = f"{offset + length - 1}" if length is not None else ""
        headers["Range"] = f"bytes={offset}-{end}"
    last_error: Exception | None = None
    for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
                payload = response.read()
            if offset is not None and length is not None and len(payload) != length:
                raise RuntimeError(f"short range response for {url}[{offset}:{length}]: {len(payload)} bytes")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            temporary = cached.with_name(cached.name + ".part")
            temporary.write_bytes(payload)
            temporary.replace(cached)
            return payload
        except urllib.error.HTTPError as error:
            if error.code in (400, 401, 403, 404):
                raise
            last_error = error
            time.sleep(2.0 * (attempt + 1))
        except Exception as error:  # retried below, surfaced after the final attempt
            last_error = error
            time.sleep(2.0 * (attempt + 1))
    assert last_error is not None
    raise RuntimeError(f"failed to download {url}[{offset}:{length}]: {last_error}") from last_error


def _parse_inventory(inventory: str) -> list[tuple[int, str]]:
    """``(byte offset, full descriptor)`` pairs from a wgrib2 inventory."""
    entries: list[tuple[int, str]] = []
    for line in inventory.splitlines():
        if not line.strip():
            continue
        parts = line.split(":", 3)
        entries.append((int(parts[1]), f"{parts[2]}:{parts[3]}"))
    return entries


def _message_range(entries: list[tuple[int, str]], match: str) -> tuple[int, int | None]:
    """The range of the first record whose descriptor contains ``match``.

    Multi-field messages (CFSR packs U and V into one GRIB2 message) share an
    offset across inventory rows, so the range runs to the next distinct
    offset and therefore carries every field of the message. The final record
    has no next offset and returns ``None`` for an open-ended read.
    """
    offsets = sorted({offset for offset, _ in entries})
    for offset, descriptor in entries:
        if match in descriptor:
            next_offset = next((candidate for candidate in offsets if candidate > offset), None)
            return offset, None if next_offset is None else next_offset - offset
    raise KeyError(match)


def _pressure_message(
    chunks: list[bytes], grib_url: str, entries: list[tuple[int, str]], level: int, variable: str
) -> None:
    """Append one pressure-level field's raw GRIB2 message from a parsed inventory."""
    offset, length = _message_range(entries, f"{variable}:{level} mb:")
    chunks.append(_fetch(grib_url, offset, length))


def _surface_message(chunks: list[bytes], moment: dt.datetime, variable: str, match: str) -> None:
    """Append one time-series surface record for ``moment``."""
    base = f"{_BASE_URL}/time-series/{moment.year}{moment.month:02d}/{variable}.gdas.{moment.year}{moment.month:02d}"
    entries = _parse_inventory(_fetch(f"{base}.grb2.inv").decode())
    label = _timestamp_label(moment)
    for index, (offset, descriptor) in enumerate(entries):
        if f"d={label}" in descriptor and match in descriptor:
            next_offset = entries[index + 1][0] if index + 1 < len(entries) else None
            chunks.append(_fetch(f"{base}.grb2", offset, None if next_offset is None else next_offset - offset))
            return
    raise KeyError(f"{variable}:{match} at {label}")


def _download_profile_messages(moment: dt.datetime) -> bytes | None:
    """Fetch every GRIB2 message needed for one timestamp, in parallel.

    Returns ``None`` only when the NCEI store answers HTTP 404 for the
    timestamp's pressure-level object -- the real 2003-10-29 12Z gap. Any
    other transport failure propagates so a transient outage cannot be
    mistaken for an archive gap.
    """
    label = _timestamp_label(moment)
    stem = f"{moment.year}{moment.month:02d}/{label[:8]}/pgbh00.gdas.{label}"
    grib_url = f"{_BASE_URL}/6-hourly-by-pressure-level/{moment.year}/{stem}.grb2"
    try:
        _fetch(grib_url, 0, 1)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise

    entries = _parse_inventory(
        _fetch(f"{_BASE_URL}/6-hourly-by-pressure-level/{moment.year}/{stem}.inv").decode()
    )

    chunks: list[bytes] = []
    surface_fields = (
        ("tmp2m", "TMP:2 m above ground:anl:"),
        ("q2m", "SPFH:2 m above ground:anl:"),
        ("pressfc", "PRES:surface:anl:"),
        ("wnd10m", "UGRD:10 m above ground:anl:"),
        ("wnd10m", "VGRD:10 m above ground:anl:"),
    )
    jobs = []
    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        for level in _PRESSURE_LEVELS_HPA:
            for variable in ("HGT", "TMP", "RH", "UGRD"):
                jobs.append(pool.submit(_pressure_message, chunks, grib_url, entries, level, variable))
        for variable, match in surface_fields:
            jobs.append(pool.submit(_surface_message, chunks, moment, variable, match))
        for job in jobs:
            job.result()
    return b"".join(chunks)


def _pick(datasets: list, name: str, level: float | None = None):
    """Select one variable at the fixture's grid point from decoded datasets."""
    for dataset in datasets:
        if name not in dataset.data_vars:
            continue
        selected = dataset[name].sel(
            latitude=_GRID_LATITUDE,
            longitude=_GRID_LONGITUDE,
            method="nearest",
        )
        selected_latitude = float(selected.latitude)
        selected_longitude = float(selected.longitude)
        longitude_error = abs(selected_longitude - _GRID_LONGITUDE)
        longitude_error = min(longitude_error, 360.0 - longitude_error)
        if abs(selected_latitude - _GRID_LATITUDE) > 0.25 or longitude_error > 0.25:
            raise RuntimeError(
                f"nearest grid point for {name} is ({selected_latitude:.2f}, {selected_longitude:.2f}), "
                f"not the fixture's ({_GRID_LATITUDE:.1f}, {_GRID_LONGITUDE - 360.0:.1f})"
            )
        if level is not None:
            selected = selected.sel(isobaricInhPa=level)
        return selected
    raise KeyError(name)


def _profile_from_messages(messages: bytes) -> tuple[list[float], list[float], list[float], list[float]]:
    """Extract the fixture profile arrays from one timestamp's GRIB2 messages."""
    _DECODE_PATH.write_bytes(messages)
    datasets = cfgrib.open_datasets(str(_DECODE_PATH), indexpath="")

    surface_hpa = float(_pick(datasets, "sp")) / 100.0
    surface_temperature_c = float(_pick(datasets, "t2m")) - 273.15
    surface_specific_humidity = float(_pick(datasets, "sh2"))
    surface_wind = float(np.hypot(_pick(datasets, "u10"), _pick(datasets, "v10")))

    # Specific humidity -> relative humidity at 2 m (FAO-56 saturation vapor
    # pressure; the same formula the library's HDW path uses via pm_eto).
    vapor_pressure_hpa = (
        surface_specific_humidity
        * surface_hpa
        / (0.622 + 0.378 * surface_specific_humidity)
    )
    saturation_hpa = 0.6108 * math.exp(17.27 * surface_temperature_c / (surface_temperature_c + 237.3)) * 10.0
    surface_humidity = min(100.0, max(0.0, 100.0 * vapor_pressure_hpa / saturation_hpa))

    geopotential = {
        level: float(_pick(datasets, "gh", float(level))) for level in _PRESSURE_LEVELS_HPA
    }
    levels_above_surface = [level for level in _PRESSURE_LEVELS_HPA if level < surface_hpa]
    if not levels_above_surface:
        raise RuntimeError(f"no fetched pressure level is above a surface pressure of {surface_hpa:.1f} hPa")
    lowest_level = max(levels_above_surface)
    mean_temperature_k = (
        surface_temperature_c + 273.15 + float(_pick(datasets, "t", float(lowest_level)))
    ) / 2.0
    surface_height = geopotential[lowest_level] - (
        _R_DRY_AIR * mean_temperature_k / _GRAVITY * math.log(surface_hpa / lowest_level)
    )

    temperatures = [surface_temperature_c]
    humidities = [surface_humidity]
    winds = [surface_wind]
    heights = [_SURFACE_LEVEL_HEIGHT_METERS]
    for level in _PRESSURE_LEVELS_HPA:
        temperatures.append(float(_pick(datasets, "t", float(level))) - 273.15)
        humidities.append(float(_pick(datasets, "r", float(level))))
        winds.append(float(np.hypot(_pick(datasets, "u", float(level)), _pick(datasets, "v", float(level)))))
        heights.append(geopotential[level] - surface_height)
    return temperatures, humidities, winds, heights


def _checksum_of_fixture(directory: Path) -> str:
    """SHA-256 over every committed ``.npy`` and ``.csv`` fixture artifact."""
    hasher = hashlib.sha256()
    for pattern in ("*.npy", "*.csv"):
        for path in sorted(directory.glob(pattern)):
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _save_array(name: str, array: np.ndarray) -> None:
    """Write one fixture array as a staged replace, so a partial run leaves no mixed fixture."""
    temporary = FIXTURE_DIR / (name + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(FIXTURE_DIR / name)


def _write_text(name: str, text: str) -> None:
    """Write one fixture text artifact as a staged replace."""
    temporary = FIXTURE_DIR / (name + ".part")
    temporary.write_text(text)
    temporary.replace(FIXTURE_DIR / name)


def main() -> None:
    moments = _timestamps()
    print(f"preparing {len(moments)} timestamps at ({_GRID_LATITUDE}, {_GRID_LONGITUDE - 360.0:.1f})")

    temperatures: list[list[float]] = []
    humidities: list[list[float]] = []
    winds: list[list[float]] = []
    heights: list[list[float]] = []
    valid_times: list[dt.datetime] = []
    missing: list[dt.datetime] = []
    for index, moment in enumerate(moments, start=1):
        try:
            messages = _download_profile_messages(moment)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            messages = None
        if messages is None:
            missing.append(moment)
            continue
        profile = _profile_from_messages(messages)
        for target, values in zip((temperatures, humidities, winds, heights), profile):
            target.append(values)
        valid_times.append(moment)
        if index % 10 == 0 or index == len(moments):
            print(f"  {index}/{len(moments)} {moment:%Y-%m-%dT%H:%M}Z")

    if not valid_times:
        raise RuntimeError("no CFSR analyses were retrieved; refusing to write an empty fixture")

    y_fit = np.polyfit(_Y_TICK_PIXELS, _Y_TICK_VALUES, 1)
    digitized = [
        (pixel_x, pixel_y, float(np.polyval(y_fit, pixel_y)))
        for pixel_x, pixel_y in _FIGURE4A_MARKER_PIXELS
    ]
    published_values = [value for _, _, value in digitized]

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _save_array("temperature_celsius.npy", np.asarray(temperatures, dtype=np.float64))
    _save_array("relative_humidity_percent.npy", np.asarray(humidities, dtype=np.float64))
    _save_array("wind_speed_meters_per_second.npy", np.asarray(winds, dtype=np.float64))
    _save_array("height_agl_meters.npy", np.asarray(heights, dtype=np.float64))
    _save_array(
        "valid_time.npy",
        np.asarray([moment.replace(tzinfo=None) for moment in valid_times], dtype="datetime64[ns]"),
    )

    day = _START_DATE
    rows = ["date,published_hdw_hpa_m_s,figure_pixel_x,figure_pixel_y"]
    for pixel_x, pixel_y, published in digitized:
        rows.append(f"{day.isoformat()},{published:.1f},{pixel_x},{pixel_y}")
        day += dt.timedelta(days=1)
    _write_text("reference_daily.csv", "\n".join(rows) + "\n")

    missing_labels = [moment.strftime("%Y-%m-%d %H:%M UTC") for moment in missing]
    event_day = _START_DATE + dt.timedelta(days=int(np.argmax(published_values)))
    provenance = {
        "source": "NCEP Climate Forecast System Reanalysis (CFSR), 6-hourly pressure-level and "
        "time-series analyses via the NOAA NCEI object store; published HDW series digitized from "
        "Srock et al. (2018) Figure 4a",
        "url": _DATASET_URL,
        "download_date": dt.date.today().isoformat(),
        "subset_description": (
            f"CFSR vertical profiles at the Figure 4a grid point ({_GRID_LATITUDE:.1f} N, "
            f"{_GRID_LONGITUDE - 360.0:.1f} W) for {_START_DATE.isoformat()} through {_END_DATE.isoformat()}, "
            f"at the paper's 0000/1200/1800 UTC analyses: 2 m temperature and specific humidity, "
            f"10 m wind, surface pressure, and temperature, relative humidity, wind speed, and "
            f"geopotential height at {', '.join(str(level) for level in _PRESSURE_LEVELS_HPA)} hPa. "
            "Heights are meters above the grid point's surface, recovered with the hypsometric equation "
            "from the lowest fetched level above the surface. The digitized Figure 4a series has one row "
            "per UTC date."
            + (
                f" The NCEI store has no complete pressure-level analysis for {', '.join(missing_labels)} "
                "(HTTP 404 on the absent object), so those daily maxima use the remaining analyses."
                if missing
                else ""
            )
        ),
        "checksum_sha256": _checksum_of_fixture(FIXTURE_DIR),
        "fixture_version": "1.0.0",
        "validation_tolerance": {
            "digitized_series_uncertainty_hpa_m_s": 5.0,
            "published_series_correlation": 0.80,
        },
        "measured_stats": {
            "digitized_series": {
                "event_day_hdw": float(max(published_values)),
            }
        },
        "citation": "Srock, A.F., Charney, J.J., Potter, B.E. and Goodrick, S.L. (2018) The Hot-Dry-Windy "
        "Index: A New Fire Weather Index. Atmosphere 9(7):279. https://doi.org/10.3390/atmos9070279",
        "doi": "10.3390/atmos9070279",
        "license": "CFSR is a U.S. Government product (public domain). The Srock et al. (2018) article is "
        "CC-BY 4.0 (MDPI); only a digitized derivative of Figure 4a is committed here, with attribution.",
        "notes": (
            f"Figure 4a's published series is figure-only; values were digitized from the CC-BY PDF "
            f"({_PDF_URL}, SHA-256 {_PDF_SHA256}) on 2026-09-18 using the marker pixels and left-axis "
            f"tick calibration recorded in _FIGURE4A_MARKER_PIXELS and _Y_TICK_PIXELS (the committed "
            f"figure_pixel_x/figure_pixel_y columns let the mapping be re-checked). One pixel is about "
            f"0.94 hPa m s-1; the +/-5 hPa m s-1 tolerance is a conservative bound on marker-center and "
            f"axis-calibration error. Each profile's surface row pairs the 2 m temperature and humidity "
            f"with the 10 m wind at a 2 m height, the library's single-height-per-level contract. The "
            f"library implements the per-level VPD x wind product, while "
            f"Srock et al. adiabatically adjust VPD to the surface and take the VPD and wind maxima "
            f"independently over the surface plus every pressure level up to and including the first "
            f"above surface + 50 hPa (Srock et al., 2018, Section 3). The 0600 UTC analyses are not "
            f"fetched, matching the paper's stated 1200/1800/0000 UTC daily maximum. The digitized "
            f"magnitudes cannot be reproduced from public metadata, so the reference test asserts event "
            f"timing and series-shape agreement, not a numerical match (issue #985); the measured "
            f"comparison is recorded in VALIDATION.md."
        ),
    }
    _write_text("provenance.json", json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {FIXTURE_DIR.relative_to(PROJECT_ROOT)} ({FIXTURE_DIR / 'provenance.json'})")
    print(f"digitized event day: {event_day.isoformat()} ({max(published_values):.1f} hPa m s-1)")


if __name__ == "__main__":
    main()
