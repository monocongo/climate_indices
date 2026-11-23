#!/usr/bin/env python3
"""Prepare the Hickman, Nebraska flood-index characterization fixture (issue #1135).

Usage:
    uv run scripts/prepare_flood_hickman_fixture.py

Downloads the GHCN-Daily file for ``USC00253825`` (HICKMAN) from NCEI, keeps
daily PRCP for 1960-1996 (the Byun & Wilhite 1999 record), fills each missing
or quality-flagged day with that station's mean for the same calendar day, and
writes ``tests/fixture/flood_hickman/hickman_prcp_1960_1996.csv``.

The paper first substitutes the nearest station within 100 km and only then the
calendar-day average; this script does the second step only, because the
113-station High Plains network is not reproducible here. The script does not
import ``climate_indices``.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import urllib.request
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "tests" / "fixture" / "flood_hickman"
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00253825.dly"
APPROVED_ORIGIN = "https://www.ncei.noaa.gov/"
FIRST_YEAR, LAST_YEAR = 1960, 1996
_MAX_BYTES = 8 * 1024 * 1024


def parse_dly(text: str) -> dict[dt.date, float]:
    """Return {date: mm} for valid PRCP days (fixed-width GHCN-Daily v2 layout)."""
    out: dict[dt.date, float] = {}
    for line in text.splitlines():
        if line[17:21] != "PRCP":
            continue
        year, month = int(line[11:15]), int(line[15:17])
        if not FIRST_YEAR <= year <= LAST_YEAR:
            continue
        for day in range(1, 32):
            cell = line[21 + 8 * (day - 1) : 29 + 8 * (day - 1)]
            value, qflag = int(cell[:5]), cell[6]
            try:
                date = dt.date(year, month, day)
            except ValueError:
                continue
            if value != -9999 and qflag == " ":
                out[date] = value / 10.0  # tenths of mm -> mm
    return out


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as response:  # noqa: S310  # nosec B310 # pinned https URL
        if not response.geturl().startswith(APPROVED_ORIGIN):
            raise SystemExit(f"unexpected redirect origin: {response.geturl()}")
        raw = response.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise SystemExit("download exceeds size cap")
    observed = parse_dly(raw.decode("ascii"))

    by_calendar_day: dict[tuple[int, int], list[float]] = defaultdict(list)
    for date, mm in observed.items():
        by_calendar_day[(date.month, date.day)].append(mm)

    rows, filled = [], 0
    date = dt.date(FIRST_YEAR, 1, 1)
    while date.year <= LAST_YEAR:
        if date in observed:
            rows.append((date.isoformat(), f"{observed[date]:.1f}", 0))
        else:
            values = by_calendar_day[(date.month, date.day)]
            rows.append((date.isoformat(), f"{sum(values) / len(values):.4f}", 1))
            filled += 1
        date += dt.timedelta(days=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("date", "prcp_mm", "filled_with_calendar_day_mean"))
    writer.writerows(rows)
    data = buffer.getvalue().encode()
    # Replace rather than truncate, so an interrupted write leaves the old fixture intact.
    target = OUT_DIR / "hickman_prcp_1960_1996.csv"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    # The provenance checksum covers both canonical CSVs, in sorted filename order
    # (tests/fixture/README.md). table5.csv is transcribed from the paper, not written here.
    hasher = hashlib.sha256(data)
    hasher.update((OUT_DIR / "table5.csv").read_bytes())
    print(f"days={len(rows)} filled={filled}")
    print(f"raw_dly_sha256={hashlib.sha256(raw).hexdigest()}")
    print(f"fixture_sha256={hasher.hexdigest()}")


if __name__ == "__main__":
    main()
