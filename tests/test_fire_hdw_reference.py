"""HDW case-study reference test against Srock et al. (2018) (#985).

This is event-discrimination evidence, not a numerical match. The committed
fixture under ``tests/fixture/hdw_srock_cedar/`` holds CFSR reanalysis
profiles at the grid point of Figure 4a in the paper's Cedar Fire case study
(33.0 N, 116.5 W; 12 October through 9 November 2003), and the test asserts
that ``fire.hot_dry_windy()`` produces its largest daily value on
26 October 2003 -- the significant fire-behavior day the paper reports.

The published magnitudes stay out of the assertion for two reasons:

- The paper publishes HDW only as figures (no table or supplement), so the
  reference series is digitized from Figure 4a and carries a documented
  +/-5 hPa m s-1 tolerance.
- The paper's formulation is not the library's: it adiabatically adjusts each
  level's VPD to the surface and takes the maximum VPD and maximum wind
  independently over the surface plus every level up to and including the
  first above surface + 50 hPa (Srock et al., 2018, Section 3), which is not
  the library's per-level VPD x wind product. The digitized series cannot be
  reproduced magnitude-for-magnitude from public metadata, so the test checks
  timing and shape and records the measured divergence in VALIDATION.md
  instead of asserting a numerical bound.

The daily maximum is over the three analyses the paper names (0000, 1200,
and 1800 UTC of each UTC date); the 0600 UTC analyses are not fetched. The
NCEI store has no pressure-level analysis for 2003-10-29 12Z, so that day's
maximum uses the remaining two analyses; the provenance records the gap.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from climate_indices import fire

pytestmark = pytest.mark.validation

FIXTURE_DIR = Path(__file__).parent / "fixture" / "hdw_srock_cedar"

_EVENT_DATE = "2003-10-26"

# Measured on the committed fixture (library output vs. the digitized Figure 4a
# series), recorded here for the failure messages and in VALIDATION.md: the
# event day reaches 262.1 against 223.8 the next-highest day, and the daily
# series correlates with the published series at r=0.8679. These are
# observations, not assertions; only the correlation floor from provenance is
# enforced, with headroom for deterministic refactors rather than stochastic
# noise.
_MEASURED_EVENT_DAY_HDW = 262.1
_MEASURED_NEXT_DAY_HDW = 223.8
_MEASURED_PUBLISHED_CORRELATION = 0.8679


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _load_provenance() -> dict[str, Any]:
    with (FIXTURE_DIR / "provenance.json").open() as handle:
        return json.load(handle)


def _load_published_daily() -> dict[str, float]:
    published: dict[str, float] = {}
    with (FIXTURE_DIR / "reference_daily.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            published[row["date"]] = float(row["published_hdw_hpa_m_s"])
    return published


def _checksum_of_fixture() -> str:
    hasher = hashlib.sha256()
    for pattern in ("*.npy", "*.csv"):
        for path in sorted(FIXTURE_DIR.glob(pattern)):
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _hdw_profiles() -> np.ndarray:
    temperature = np.load(FIXTURE_DIR / "temperature_celsius.npy")
    humidity = np.load(FIXTURE_DIR / "relative_humidity_percent.npy")
    wind = np.load(FIXTURE_DIR / "wind_speed_meters_per_second.npy")
    height = np.load(FIXTURE_DIR / "height_agl_meters.npy")
    return np.asarray(fire.hot_dry_windy(temperature, humidity, wind, height, level_axis=-1))


def _hdw_daily_maxima() -> dict[str, float]:
    times = np.load(FIXTURE_DIR / "valid_time.npy")
    daily: dict[str, float] = {}
    for date, value in zip(times.astype("datetime64[D]"), _hdw_profiles(), strict=True):
        key = str(date)
        daily[key] = max(daily.get(key, -np.inf), float(value))
    return daily


def test_fixture_checksum_matches_provenance() -> None:
    """The committed arrays and digitized series must be the artifact the provenance records."""
    assert _checksum_of_fixture() == _load_provenance()["checksum_sha256"]


def test_hdw_daily_maximum_falls_on_the_documented_fire_day() -> None:
    """The library's largest daily HDW is the paper's significant fire-behavior day.

    Srock et al. (2018) report that their CFSR-driven HDW time series peaks on
    26 October 2003 for the Cedar Fire, the day of most rapid spread. The
    fixture reproduces that claim for the library's formulation: with the
    committed CFSR profiles at the paper's grid point, the maximum daily HDW
    lands on the same date.
    """
    daily = _hdw_daily_maxima()
    ranked = sorted(daily.items(), key=lambda item: item[1], reverse=True)
    assert ranked[0][0] == _EVENT_DATE, (
        f"daily maximum fell on {ranked[0][0]} ({ranked[0][1]:.1f}), not {_EVENT_DATE} "
        f"(measured event day {_MEASURED_EVENT_DAY_HDW}, next-highest {_MEASURED_NEXT_DAY_HDW})"
    )


def test_hdw_daily_series_tracks_the_published_series() -> None:
    """The library's daily series has the published series' shape.

    Timing alone can pass while the series becomes unrelated to the paper's.
    The digitized Figure 4a series is the only published reference available,
    and its magnitudes are not comparable (see the module docstring), but the
    correlation still guards against a series that no longer tracks the
    published variation at all.
    """
    daily = _hdw_daily_maxima()
    published = _load_published_daily()
    shared_dates = [date for date in published if date in daily]
    assert len(shared_dates) == len(published)
    ours = np.array([daily[date] for date in shared_dates])
    reference = np.array([published[date] for date in shared_dates])
    correlation = float(np.corrcoef(ours, reference)[0, 1])
    floor = float(_load_provenance()["validation_tolerance"]["published_series_correlation"])
    assert correlation >= floor, (
        f"daily-series correlation {correlation:.4f} fell below the {floor} floor "
        f"(measured {_MEASURED_PUBLISHED_CORRELATION})"
    )


def test_hdw_profiles_produce_finite_values() -> None:
    """Every committed profile yields a finite HDW value.

    The fixture always carries a 2 m surface level inside the layer, so a NaN
    here means the extraction or the layer filter broke, not that the data is
    legitimately missing. This asserts on every profile, not the daily maxima:
    a NaN analysis would otherwise be hidden by a finite one on the same date.
    """
    assert np.isfinite(_hdw_profiles()).all()
