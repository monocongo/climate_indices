"""PE/EDI characterization against Byun and Wilhite (1999) Table 5 (#1135).

This is **characterization, not exact-reference validation**. Table 5 reports
one-decimal summary rows for a variable-duration EDI over a 113-station record,
while the library implements the fixed 365-day Eq. (2) window (ADR-0014) and
this fixture is the single GHCN-Daily station ``USC00253825``. The two criteria
below were declared before the library was run on the record; the measured
minimum and drought bounds live in ``provenance.json`` and are not asserted.
See ``tests/fixture/flood_hickman/provenance.json`` for every deviation.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from climate_indices import flood
from climate_indices.utils import transform_to_366day

pytestmark = pytest.mark.validation

_DIR = Path(__file__).parent / "fixture" / "flood_hickman"
_FIRST_YEAR, _LAST_YEAR = 1960, 1996
_PAPER_DAY_ONE = dt.date(1995, 1, 1)


def _table5(equation: str, quantity: str) -> tuple[int, int, float | None]:
    with (_DIR / "table5.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["equation"] == equation and row["quantity"] == quantity:
                return int(row["day_start"]), int(row["day_end"]), float(row["value"]) if row["value"] else None
    raise KeyError((equation, quantity))


def _all_leap_position(date: dt.date) -> int:
    """Index of a Gregorian date in the 366-positions-per-year layout."""
    leap = date.year % 4 == 0 and (date.year % 100 != 0 or date.year % 400 == 0)
    day_of_year = date.timetuple().tm_yday + (1 if date.month >= 3 and not leap else 0)
    return (date.year - _FIRST_YEAR) * 366 + day_of_year - 1


@pytest.fixture(scope="module")
def hickman_edi() -> np.ndarray:
    """Library EDI for 1 Jan 1995 (paper day 1) through 31 Dec 1996 (day 731)."""
    with (_DIR / "hickman_prcp_1960_1996.csv").open() as handle:
        precipitation = np.array([float(row["prcp_mm"]) for row in csv.DictReader(handle)])
    pe = flood.effective_precipitation(transform_to_366day(precipitation, _FIRST_YEAR, _LAST_YEAR - _FIRST_YEAR + 1))
    edi = flood.edi(pe, _FIRST_YEAR, 1961, 1990)
    positions = [_all_leap_position(_PAPER_DAY_ONE + dt.timedelta(days=n)) for n in range(731)]
    return edi[positions]


def test_fixture_matches_provenance_and_paper_missing_data_limit() -> None:
    provenance = json.loads((_DIR / "provenance.json").read_text())
    data = (_DIR / "hickman_prcp_1960_1996.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == provenance["checksum_sha256"]
    rows = list(csv.DictReader(data.decode().splitlines()))
    assert len(rows) == 13515
    # The paper discarded stations missing more than 1% of the record.
    assert sum(int(row["filled_with_calendar_day_mean"]) for row in rows) / len(rows) < 0.01


def test_edi_is_in_drought_on_table5_minimum_day(hickman_edi: np.ndarray) -> None:
    """Criterion (a): EDI <= -1 on Table 5's Eq. (2) minimum-EDI day (469)."""
    day, _, _ = _table5("2", "min_edi")
    assert hickman_edi[day - 1] <= -1.0


def test_edi_minimum_falls_inside_table5_drought_duration(hickman_edi: np.ndarray) -> None:
    """Criterion (b): the 1995-1996 EDI minimum lies in Table 5's Eq. (2) drought run (259-493)."""
    first, last, _ = _table5("2", "drought_duration")
    assert first <= int(np.nanargmin(hickman_edi)) + 1 <= last
