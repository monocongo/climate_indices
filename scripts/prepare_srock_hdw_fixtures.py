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
calibration. One pixel is about 0.94 HDW units, so the digitized values carry
roughly +/-5 hPa m s-1 of uncertainty (issue #985 records the decision to use
event discrimination and a one-sided bound instead of a numeric match, because
the paper's adiabatically adjusted, independently maximized formulation
upper-bounds the library's per-level product and only the figure is published).
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
_PAPER_URL = "https://doi.org/10.3390/atmos9070279"
_PDF_URL = "https://research.fs.usda.gov/download/treesearch/56562.pdf"

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

# Figure 4a of Srock et al. (2018), digitized 2026-09-18. Index 0 is Oct 12
# and index 28 is Nov 9; see the module docstring for the method and its
# +/-5 hPa m s-1 uncertainty.
_FIGURE4A_HDW_HPA_M_S = (
    160.5,
    68.0,
    155.4,
    129.1,
    59.5,
    238.0,
    272.8,
    204.2,
    182.6,
    187.3,
    220.2,
    261.5,
    142.2,
    227.7,
    401.5,
    282.2,
    134.7,
    171.3,
    59.5,
    59.5,
    44.0,
    61.4,
    53.9,
    25.7,
    25.7,
    61.0,
    56.7,
    65.2,
    68.0,
)

_MAX_DOWNLOAD_ATTEMPTS = 4
_REQUEST_TIMEOUT_SECONDS = 180
_DOWNLOAD_WORKERS = 8

# cfgrib's hypercube loader expects its own index cache; running each
# timestamp through one temp file keeps the index files out of the tree.
_DECODE_PATH = Path(tempfile.gettempdir()) / "srock_hdw_decode.grb2"


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
    the whole download.
    """
    key = hashlib.sha256(f"{url}:{offset}:{length}".encode()).hexdigest()
    cached = CACHE_DIR / key
    if cached.exists():
        return cached.read_bytes()

    headers = {}
    if offset is not None:
        if length is None:
            raise ValueError("length is required with offset")
        headers["Range"] = f"bytes={offset}-{offset + length - 1}"
    last_error: Exception | None = None
    for attempt in range(_MAX_DOWNLOAD_ATTEMPTS):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
                payload = response.read()
            if offset is not None and len(payload) != length:
                raise RuntimeError(f"short range response for {url}[{offset}:{length}]: {len(payload)} bytes")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=CACHE_DIR, delete=False) as stream:
                stream.write(payload)
                temporary_cache = Path(stream.name)
            try:
                temporary_cache.replace(cached)
            finally:
                temporary_cache.unlink(missing_ok=True)
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


def _message_range(entries: list[tuple[int, str]], match: str) -> tuple[int, int]:
    """The range of the first record whose descriptor contains ``match``.

    Multi-field messages (CFSR packs U and V into one GRIB2 message) share an
    offset across inventory rows, so the range runs to the next distinct
    offset and therefore carries every field of the message.
    """
    offsets = sorted({offset for offset, _ in entries})
    for offset, descriptor in entries:
        if match in descriptor:
            next_offset = next((candidate for candidate in offsets if candidate > offset), offset + 5_000_000)
            return offset, next_offset - offset
    raise KeyError(match)


def _pressure_message(chunks: list[bytes], moment: dt.datetime, level: int, variable: str) -> None:
    """Append one pressure-level field's raw GRIB2 message for ``moment``."""
    label = _timestamp_label(moment)
    stem = f"{moment.year}{moment.month:02d}/{label[:8]}/pgbh00.gdas.{label}"
    inventory_url = f"{_BASE_URL}/6-hourly-by-pressure-level/{moment.year}/{stem}.inv"
    grib_url = f"{_BASE_URL}/6-hourly-by-pressure-level/{moment.year}/{stem}.grb2"
    entries = _parse_inventory(_fetch(inventory_url).decode())
    offset, length = _message_range(entries, f"{variable}:{level} mb:")
    chunks.append(_fetch(grib_url, offset, length))


def _surface_message(chunks: list[bytes], moment: dt.datetime, variable: str, match: str) -> None:
    """Append one time-series surface record for ``moment``."""
    base = f"{_BASE_URL}/time-series/{moment.year}{moment.month:02d}/{variable}.gdas.{moment.year}{moment.month:02d}"
    entries = _parse_inventory(_fetch(f"{base}.grb2.inv").decode())
    label = _timestamp_label(moment)
    for index, (offset, descriptor) in enumerate(entries):
        if f"d={label}" in descriptor and match in descriptor:
            next_offset = entries[index + 1][0] if index + 1 < len(entries) else offset + 700_000
            chunks.append(_fetch(f"{base}.grb2", offset, next_offset - offset))
            return
    raise KeyError(f"{variable}:{match} at {label}")


def _download_profile_messages(moment: dt.datetime) -> bytes | None:
    """Fetch every GRIB2 message needed for one timestamp, in parallel.

    Returns ``None`` when the NCEI store has no pressure-level analysis for the
    timestamp (one gap exists on 2003-10-29 12Z, and the surface product alone
    cannot supply the profile).
    """
    label = _timestamp_label(moment)
    stem = f"{moment.year}{moment.month:02d}/{label[:8]}/pgbh00.gdas.{label}"
    grib_url = f"{_BASE_URL}/6-hourly-by-pressure-level/{moment.year}/{stem}.grb2"
    try:
        _fetch(grib_url, 0, 1)
    except urllib.error.HTTPError:
        return None

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
                jobs.append(pool.submit(_pressure_message, chunks, moment, level, variable))
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
        if level is not None:
            selected = selected.sel(isobaricInhPa=level)
        return selected
    raise KeyError(name)


def _profile_from_messages(messages: bytes) -> tuple[list[float], list[float], list[float], list[float]]:
    """Extract the fixture profile arrays from one timestamp's GRIB2 messages."""
    _DECODE_PATH.write_bytes(messages)
    index_path = _DECODE_PATH.with_name(_DECODE_PATH.name + ".idx")
    index_path.unlink(missing_ok=True)
    try:
        datasets = cfgrib.open_datasets(str(_DECODE_PATH))
    finally:
        index_path.unlink(missing_ok=True)

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


def _checksum_of_arrays(directory: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(directory.glob("*.npy")):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


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
        messages = _download_profile_messages(moment)
        if messages is None:
            missing.append(moment)
            continue
        profile = _profile_from_messages(messages)
        for target, values in zip((temperatures, humidities, winds, heights), profile):
            target.append(values)
        valid_times.append(moment)
        if index % 10 == 0 or index == len(moments):
            print(f"  {index}/{len(moments)} {moment:%Y-%m-%dT%H:%M}Z")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURE_DIR / "temperature_celsius.npy", np.asarray(temperatures, dtype=np.float64))
    np.save(FIXTURE_DIR / "relative_humidity_percent.npy", np.asarray(humidities, dtype=np.float64))
    np.save(FIXTURE_DIR / "wind_speed_meters_per_second.npy", np.asarray(winds, dtype=np.float64))
    np.save(FIXTURE_DIR / "height_agl_meters.npy", np.asarray(heights, dtype=np.float64))
    np.save(
        FIXTURE_DIR / "valid_time.npy",
        np.asarray([moment.replace(tzinfo=None) for moment in valid_times], dtype="datetime64[ns]"),
    )

    day = _START_DATE
    rows = ["date,published_hdw_hpa_m_s"]
    for published in _FIGURE4A_HDW_HPA_M_S:
        rows.append(f"{day.isoformat()},{published:.1f}")
        day += dt.timedelta(days=1)
    (FIXTURE_DIR / "reference_daily.csv").write_text("\n".join(rows) + "\n")

    missing_labels = [moment.strftime("%Y-%m-%d %H:%M UTC") for moment in missing]
    event_day = _START_DATE + dt.timedelta(days=int(np.argmax(_FIGURE4A_HDW_HPA_M_S)))
    non_event = [value for index, value in enumerate(_FIGURE4A_HDW_HPA_M_S) if index != int(np.argmax(_FIGURE4A_HDW_HPA_M_S))]
    provenance = {
        "source": "NCEP Climate Forecast System Reanalysis (CFSR), 6-hourly pressure-level and "
        "time-series analyses via the NOAA NCEI object store; published HDW series digitized from "
        "Srock et al. (2018) Figure 4a",
        "url": _PDF_URL,
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
                f" The NCEI store has no pressure-level analysis for {', '.join(missing_labels)}, so "
                "those daily maxima use the remaining analyses."
                if missing
                else ""
            )
        ),
        "checksum_sha256": _checksum_of_arrays(FIXTURE_DIR),
        "fixture_version": "1.0.0",
        "validation_tolerance": {
            "digitized_series_uncertainty_hpa_m_s": 5.0,
        },
        "measured_stats": {
            "digitized_series": {
                "event_day_hdw": float(max(_FIGURE4A_HDW_HPA_M_S)),
                "max_non_event_day_hdw": float(max(non_event)),
            }
        },
        "citation": "Srock, A.F., Charney, J.J., Potter, B.E. and Goodrick, S.L. (2018) The Hot-Dry-Windy "
        "Index: A New Fire Weather Index. Atmosphere 9(7):279. https://doi.org/10.3390/atmos9070279",
        "doi": "10.3390/atmos9070279",
        "license": "CFSR is a U.S. Government product (public domain). The Srock et al. (2018) article is "
        "CC-BY 4.0 (MDPI); only a digitized derivative of Figure 4a is committed here, with attribution.",
        "notes": (
            f"Figure 4a's published series is figure-only; values were digitized from the CC-BY PDF "
            f"({_PDF_URL}) on 2026-09-18 by calibrating the y axis on the figure's 0-500 gridline ticks "
            f"and the x axis on the 29 daily tick marks, then reading the marker centers. One pixel is "
            f"about 0.94 HDW units, so the committed values carry roughly +/-5 hPa m s-1 of uncertainty. "
            f"The library implements the per-level VPD x wind product, while Srock et al. adiabatically "
            f"adjust VPD to the surface and take the VPD and wind maxima independently over a layer "
            f"ending at the first level above surface + 50 hPa. That formulation upper-bounds the "
            f"library's in theory, but the digitized series still cannot be reproduced "
            f"magnitude-for-magnitude from public metadata: the library's daily maximum exceeds the "
            f"digitized value on five of 29 days, always driven by the 0000 UTC analyses, and the "
            f"library's 1800 UTC values alone correlate with the published series at 0.945 with no "
            f"exceedances. The reference test therefore asserts event timing and series shape, not a "
            f"one-sided bound or a numerical match (issue #985)."
        ),
    }
    (FIXTURE_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {FIXTURE_DIR.relative_to(PROJECT_ROOT)} ({FIXTURE_DIR / 'provenance.json'})")
    print(f"digitized event day: {event_day.isoformat()} ({max(_FIGURE4A_HDW_HPA_M_S):.1f} hPa m s-1)")


if __name__ == "__main__":
    main()
