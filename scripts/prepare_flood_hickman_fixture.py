#!/usr/bin/env python3
"""Prepare the Hickman, Nebraska flood-index characterization fixture (issue #1135).

Usage:
    uv run scripts/prepare_flood_hickman_fixture.py

Downloads the GHCN-Daily file for ``USC00253825`` (HICKMAN) from NCEI, keeps
daily PRCP for 1960-1996 (the Byun & Wilhite 1999 record), fills each missing
or quality-flagged day with that station's mean for the same calendar day, and
writes ``tests/fixture/flood_hickman/hickman_prcp_1960_1996.csv`` only if
both resulting CSVs match the committed provenance checksum.

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
import json
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from http.client import HTTPMessage
from pathlib import Path
from typing import IO

OUT_DIR = Path(__file__).parent.parent / "tests" / "fixture" / "flood_hickman"
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/USC00253825.dly"
APPROVED_ORIGIN = "https://www.ncei.noaa.gov/"
FIRST_YEAR, LAST_YEAR = 1960, 1996
_MAX_BYTES = 8 * 1024 * 1024


class _ApprovedOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects before they contact any other origin."""

    def redirect_request(
        self, req: urllib.request.Request, fp: IO[bytes], code: int, msg: str, headers: HTTPMessage, newurl: str
    ) -> urllib.request.Request | None:
        if not newurl.startswith(APPROVED_ORIGIN):
            raise urllib.error.URLError(f"refusing redirect off {APPROVED_ORIGIN}: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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


def _publish_fixture(out_dir: Path, data: bytes) -> str:
    # Hash both CSVs in filename order before replacing the precipitation fixture.
    table = (out_dir / "table5.csv").read_bytes()
    checksum = hashlib.sha256(data + table).hexdigest()
    expected = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))["checksum_sha256"]
    if checksum != expected:
        raise SystemExit(f"fixture checksum mismatch: expected {expected}, got {checksum}")

    target = out_dir / "hickman_prcp_1960_1996.csv"
    with tempfile.NamedTemporaryFile(dir=out_dir, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.close()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return checksum


def main() -> None:
    opener = urllib.request.build_opener(_ApprovedOriginRedirectHandler)
    with opener.open(URL, timeout=60) as response:
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
            if not values:
                raise SystemExit(f"no valid precipitation observations for calendar day {date:%m-%d}")
            rows.append((date.isoformat(), f"{sum(values) / len(values):.4f}", 1))
            filled += 1
        date += dt.timedelta(days=1)

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("date", "prcp_mm", "filled_with_calendar_day_mean"))
    writer.writerows(rows)
    print(f"raw_dly_sha256={hashlib.sha256(raw).hexdigest()}")
    checksum = _publish_fixture(OUT_DIR, buffer.getvalue().encode())
    print(f"days={len(rows)} filled={filled}")
    print(f"fixture_sha256={checksum}")


if __name__ == "__main__":
    main()
