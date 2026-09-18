"""HDW case-study reference test against Srock et al. (2018) (#985).

This is event-discrimination evidence, not a numerical match. The committed
fixture under ``tests/fixture/hdw_srock_cedar/`` holds CFSR reanalysis
profiles at the grid point of Figure 4a in the paper's Cedar Fire case study
(33.0 N, 116.5 W; 12 October through 9 November 2003), and the test asserts
that ``fire.hot_dry_windy()`` produces its largest daily value on
26 October 2003 -- the significant fire-behavior day the paper reports.

Two limits keep the published magnitudes out of the assertion:

- The paper publishes HDW only as figures (no table or supplement), so the
  reference series is digitized from Figure 4a and carries roughly
  +/-5 hPa m s-1 of uncertainty.
- The paper's formulation is not the library's: it adiabatically adjusts each
  level's VPD to the surface and takes the maximum VPD and maximum wind
  independently over the surface + 50 hPa layer (including the first level
  above it), which upper-bounds the library's per-level VPD x wind product in
  theory. Empirically the digitized series still cannot be reproduced
  magnitude-for-magnitude: the days where the library's daily maximum exceeds
  the digitized value are all driven by the paper's 0000 UTC analyses, which
  the figure's published series does not appear to include (the library's
  1800 UTC values alone correlate at 0.945 with no exceedances). The test
  therefore checks timing and shape -- the paper's scientific claim -- and
  records the magnitude gap rather than asserting it.

The daily maximum is over the three analyses the paper names (0000, 1200,
and 1800 UTC of each UTC date). The NCEI store has no pressure-level analysis
for 2003-10-29 12Z, so that day's maximum uses the remaining two analyses;
the provenance records the gap.
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
# series): the event day sits 1.1709x above the next-highest day, and the daily
# series correlates with the published series at r=0.8681. The floors leave the
# documented headroom for deterministic refactors rather than stochastic noise.
_MEASURED_EVENT_SPIKE_RATIO = 1.1709
_MINIMUM_EVENT_SPIKE_RATIO = 1.10
_MEASURED_PUBLISHED_CORRELATION = 0.8681
_MINIMUM_PUBLISHED_CORRELATION = 0.80


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
    for path in sorted(FIXTURE_DIR.glob("*.npy")) + sorted(FIXTURE_DIR.glob("*.csv")):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _hdw_profiles() -> np.ndarray:
    temperature = np.load(FIXTURE_DIR / "temperature_celsius.npy")
    humidity = np.load(FIXTURE_DIR / "relative_humidity_percent.npy")
    wind = np.load(FIXTURE_DIR / "wind_speed_meters_per_second.npy")
    height = np.load(FIXTURE_DIR / "height_agl_meters.npy")
    return np.asarray(fire.hot_dry_windy(temperature, humidity, wind, height, level_axis=-1))


def _hdw_daily_maxima() -> dict[str, float]:
    hdw = _hdw_profiles()
    times = np.load(FIXTURE_DIR / "valid_time.npy")

    daily: dict[str, float] = {}
    for date, value in zip(times.astype("datetime64[D]"), np.asarray(hdw), strict=True):
        key = str(date)
        daily[key] = max(daily.get(key, -np.inf), float(value))
    return daily


def test_fixture_checksum_matches_provenance() -> None:
    """The committed arrays must be the exact artifact the provenance records."""
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
    assert max(daily, key=lambda date: daily[date]) == _EVENT_DATE


def test_hdw_event_day_stands_clear_of_the_rest_of_the_series() -> None:
    """The event-day maximum is a spike, not a tie with neighboring days.

    A regression that flattened the event response (for example, a level or
    wind-handling change) could still leave 26 October nominally on top.
    Require the documented margin over the next-highest day.
    """
    daily = _hdw_daily_maxima()
    ranked = sorted(daily.values(), reverse=True)
    ratio = ranked[0] / ranked[1]
    assert ratio >= _MINIMUM_EVENT_SPIKE_RATIO, (
        f"event-day spike ratio {ratio:.4f} fell below the {_MINIMUM_EVENT_SPIKE_RATIO} floor "
        f"(measured {_MEASURED_EVENT_SPIKE_RATIO})"
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
    assert correlation >= _MINIMUM_PUBLISHED_CORRELATION, (
        f"daily-series correlation {correlation:.4f} fell below the "
        f"{_MINIMUM_PUBLISHED_CORRELATION} floor (measured {_MEASURED_PUBLISHED_CORRELATION})"
    )


def test_hdw_profiles_produce_finite_values() -> None:
    """Every committed profile yields a finite HDW value.

    The fixture always carries a 2 m surface level inside the layer, so a NaN
    here means the extraction or the layer filter broke, not that the data is
    legitimately missing.
    """
    assert np.all(np.isfinite(_hdw_profiles()))
