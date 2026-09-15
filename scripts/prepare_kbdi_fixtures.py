#!/usr/bin/env python3
"""Prepare the KBDI validation fixtures (issue #800).

Usage:
    uv run scripts/prepare_kbdi_fixtures.py

Writes two independently sourced fixture directories:

- ``tests/fixture/kbdi_se38_figure1/``: the 30-day worked example printed as
  Figure 1 of Keetch and Byram (1968), SE-38 pp. 11-13, with its published net
  rain and table drought factors. Figure 1 exercises the report's integer
  lookup tables, so it is a reference for the original table workflow rather
  than for the corrected continuous Equation 18 the library implements.
- ``tests/fixture/kbdi_ghcn/``: a complete 30-year GHCN-Daily PRCP/TMAX record
  from Fresno Yosemite International Airport (a fire-prone California
  station), with expected KBDI values generated here by a plain-Python
  transcription of the corrected Equation 18 contract. The expected column is
  regression coverage, not independent scientific validation: this script
  deliberately does not import ``climate_indices``, but it shares the
  project's platform decisions (metric evaluation, zero initialization,
  contiguous daily observations).

Run the script only when refreshing the fixtures; it downloads from NCEI and
rewrites the committed CSVs and provenance files.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixture"
FIGURE1_DIR = FIXTURE_DIR / "kbdi_se38_figure1"
GHCN_DIR = FIXTURE_DIR / "kbdi_ghcn"

# NOAA NCEI GHCN-Daily access service. Values arrive in tenths of a millimetre
# and tenths of a degree Celsius, as decimal integers.
_GHCN_STATION = "USW00093193"  # Fresno Yosemite International Airport, CA
_GHCN_STATION_NAME = "Fresno Yosemite International Airport, California"
_GHCN_START_DATE = "1991-01-01"
_GHCN_END_DATE = "2020-12-31"
_GHCN_ACCESS_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1"
    f"?dataset=daily-summaries&stations={_GHCN_STATION}"
    f"&startDate={_GHCN_START_DATE}&endDate={_GHCN_END_DATE}"
    "&dataTypes=PRCP,TMAX&format=csv&includeAttributes=false"
)
_APPROVED_ORIGIN = "https://www.ncei.noaa.gov/"
_APPROVED_ORIGIN_PARTS = urllib.parse.urlsplit(_APPROVED_ORIGIN)
_MAX_DOWNLOAD_BYTES = 32 * 1024 * 1024
_TENTHS_PER_UNIT = 10.0
_GHCN_START = dt.date.fromisoformat(_GHCN_START_DATE)
_GHCN_END = dt.date.fromisoformat(_GHCN_END_DATE)
_GHCN_EXPECTED_DATES = [
    (_GHCN_START + dt.timedelta(days=offset)).isoformat() for offset in range((_GHCN_END - _GHCN_START).days + 1)
]

# Figure 1 of SE-38, transcribed from the printed sample record. Column order
# follows the published form; blank rain cells are zero.
_FIGURE1_PRECIPITATION_IN = (
    0,
    0,
    0.66,
    0,
    0.23,
    0,
    0.16,
    0.09,
    0,
    0,
    0.08,
    0.03,
    0,
    0.22,
    0,
    0.21,
    0,
    0,
    0,
    0.01,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.25,
    0.16,
)
_FIGURE1_TEMPERATURE_F = (
    79,
    75,
    70,
    76,
    79,
    84,
    65,
    66,
    83,
    70,
    67,
    65,
    76,
    69,
    65,
    75,
    78,
    85,
    88,
    79,
    69,
    75,
    84,
    89,
    93,
    92,
    96,
    91,
    78,
    83,
)
_FIGURE1_NET_RAIN_IN = (
    0,
    0,
    0.46,
    0,
    0.03,
    0,
    0,
    0.05,
    0,
    0,
    0,
    0,
    0,
    0.02,
    0,
    0.01,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0.05,
    0.16,
)
_FIGURE1_TABLE_DROUGHT_FACTOR = (
    10,
    8,
    6,
    9,
    11,
    14,
    4,
    4,
    14,
    6,
    4,
    4,
    8,
    5,
    4,
    8,
    9,
    13,
    15,
    8,
    5,
    7,
    12,
    16,
    17,
    17,
    20,
    13,
    7,
    10,
)
_FIGURE1_PUBLISHED_KBDI = (
    174,
    182,
    142,
    151,
    159,
    173,
    177,
    176,
    190,
    196,
    200,
    204,
    212,
    215,
    219,
    226,
    235,
    248,
    263,
    271,
    276,
    283,
    295,
    311,
    328,
    345,
    365,
    378,
    380,
    374,
)
# The continuous Equation 18 cannot reproduce the table workflow exactly because
# the tables quantize temperature into 3 F bins, the deficit into 50-point
# columns, and the daily factor into integers. 4.0 hundredths of an inch bounds
# the accumulated deviation over the 30-day series (measured maximum with the
# library's corrected-equation implementation is 2.91; see
# tests/test_fire_kbdi_reference.py, which reads this tolerance from
# provenance.json).
_FIGURE1_ATOL = 4.0
_SE38_PDF_SHA256 = "4a000f5e4da1eb6b414724549b847b0556a1b6d5d56793c5459b18f01c45c03b"

# Corrected Equation 18 (Alexander 1990) and the SE-38 wet-spell rules, metric.
_KBDI_MAX_MM = 203.2
_KBDI_RAIN_THRESHOLD_MM = 5.08
_KBDI_DRYING_TEMPERATURE_CELSIUS = 10.0

# Tolerance for the GHCN fixture. Both this script and the library implement
# the same contract with the same constants; float summation order, numpy-vs-math
# exp, and CSV storage at ten decimal places are the only expected differences,
# so the bound is tight and is a regression guard, not an accuracy claim.
_GHCN_RTOL = 1e-12
_GHCN_ATOL_MM = 1e-9


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _csv_text(header: list[str], rows: list[list[str]]) -> tuple[str, str]:
    """Serialize a CSV with ``\\n`` line endings, returning its text and SHA-256."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    csv_text = buffer.getvalue()
    return csv_text, _sha256(csv_text.encode())


def _write_bundle(
    csv_path: Path,
    csv_text: str,
    provenance_path: Path,
    provenance: dict[str, object],
) -> None:
    """Replace a fixture CSV and its provenance as one staged unit.

    Both files are serialized and written beside their targets before either is
    swapped in, so a refresh that fails while downloading, serializing, or
    writing leaves the previous bundle intact instead of a truncated CSV or a
    CSV whose provenance describes different bytes.
    """
    staged = [
        (csv_path, csv_text),
        (provenance_path, json.dumps(provenance, indent=2) + "\n"),
    ]
    temps: list[tuple[Path, Path]] = []
    try:
        for target, contents in staged:
            temp = target.with_name(target.name + ".tmp")
            temp.write_text(contents)
            temps.append((temp, target))
        for temp, target in temps:
            temp.replace(target)
    finally:
        for temp, _ in temps:
            temp.unlink(missing_ok=True)


def _reference_kbdi_mm(
    precipitation_mm: list[float],
    temperature_celsius: list[float],
    mean_annual_precipitation_mm: float,
    initial_kbdi_mm: float = 0.0,
) -> list[float]:
    """Plain-Python transcription of corrected Equation 18 and the SE-38 rain rules.

    Deliberately independent of ``climate_indices``: one day at a time, plain
    arithmetic, ``math.exp``. It shares only the published contract with the
    production implementation.
    """
    kbdi_value = initial_kbdi_mm
    wet_spell = 0.0
    values: list[float] = []
    for precipitation, temperature in zip(precipitation_mm, temperature_celsius, strict=True):
        net_rain = 0.0
        if precipitation > 0.0:
            event_total = wet_spell + precipitation
            if wet_spell > _KBDI_RAIN_THRESHOLD_MM:
                net_rain = precipitation
            elif event_total > _KBDI_RAIN_THRESHOLD_MM:
                net_rain = event_total - _KBDI_RAIN_THRESHOLD_MM
            wet_spell = event_total
        else:
            wet_spell = 0.0
        kbdi_value = max(0.0, kbdi_value - net_rain)
        if temperature >= _KBDI_DRYING_TEMPERATURE_CELSIUS and kbdi_value < _KBDI_MAX_MM:
            drying = (
                (_KBDI_MAX_MM - kbdi_value)
                * (0.968 * math.exp(0.0875 * temperature + 1.5552) - 8.30)
                / (1.0 + 10.88 * math.exp(-0.001736 * mean_annual_precipitation_mm))
                * 1e-3
            )
            kbdi_value = min(_KBDI_MAX_MM, kbdi_value + max(drying, 0.0))
        values.append(kbdi_value)
    return values


def _is_approved_url(url: str) -> bool:
    """Whether ``url`` is on the approved HTTPS NCEI origin."""
    parts = urllib.parse.urlsplit(url)
    return (parts.scheme, parts.hostname) == (_APPROVED_ORIGIN_PARTS.scheme, _APPROVED_ORIGIN_PARTS.hostname)


class _ApprovedOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects that would leave the approved HTTPS origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _is_approved_url(newurl):
            raise urllib.error.URLError(f"refusing GHCN redirect off {_APPROVED_ORIGIN}: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download_ghcn_csv() -> str:
    """Download the fixed GHCN-Daily window, rejecting off-origin or oversized responses.

    Every redirect is checked before it is followed, and the accepted response
    URL is checked again, so a refresh can only ever read from the approved
    origin.
    """
    if not _is_approved_url(_GHCN_ACCESS_URL):
        raise ValueError(f"GHCN download must use {_APPROVED_ORIGIN}: {_GHCN_ACCESS_URL}")
    opener = urllib.request.build_opener(_ApprovedOriginRedirectHandler)
    with opener.open(_GHCN_ACCESS_URL, timeout=180) as response:
        if not _is_approved_url(response.geturl()):
            raise ValueError(f"GHCN download left {_APPROVED_ORIGIN}: {response.geturl()}")
        payload = response.read(_MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > _MAX_DOWNLOAD_BYTES:
        raise ValueError(f"GHCN response exceeded {_MAX_DOWNLOAD_BYTES} bytes")
    return payload.decode()


def _parse_ghcn_rows(text: str) -> tuple[list[str], list[float], list[float]]:
    """Parse the access-service CSV into dates and metric floats, rejecting missing cells."""
    dates: list[str] = []
    precipitation_mm: list[float] = []
    temperature_celsius: list[float] = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            precipitation = float(row["PRCP"]) / _TENTHS_PER_UNIT
            temperature = float(row["TMAX"]) / _TENTHS_PER_UNIT
        except (TypeError, ValueError) as exc:
            raise ValueError(f"missing or non-numeric PRCP/TMAX on {row['DATE']}") from exc
        if not (math.isfinite(precipitation) and math.isfinite(temperature)):
            raise ValueError(f"non-finite PRCP/TMAX on {row['DATE']}")
        dates.append(row["DATE"])
        precipitation_mm.append(precipitation)
        temperature_celsius.append(temperature)
    if dates != _GHCN_EXPECTED_DATES:
        raise ValueError(
            f"expected the ordered, unique {len(_GHCN_EXPECTED_DATES)} dates "
            f"{_GHCN_START_DATE}..{_GHCN_END_DATE}, got {len(dates)} rows"
        )
    return dates, precipitation_mm, temperature_celsius


def _write_figure1_fixture(today: str) -> None:
    rows = [
        [
            str(day),
            f"{precipitation:.2f}",
            str(temperature),
            f"{net_rain:.2f}",
            str(factor),
            str(published),
        ]
        for day, (precipitation, temperature, net_rain, factor, published) in enumerate(
            zip(
                _FIGURE1_PRECIPITATION_IN,
                _FIGURE1_TEMPERATURE_F,
                _FIGURE1_NET_RAIN_IN,
                _FIGURE1_TABLE_DROUGHT_FACTOR,
                _FIGURE1_PUBLISHED_KBDI,
                strict=True,
            ),
            start=1,
        )
    ]
    csv_text, checksum = _csv_text(
        [
            "day",
            "precipitation_in",
            "maximum_temperature_f",
            "net_rain_in",
            "table_drought_factor",
            "published_kbdi_hundredths_in",
        ],
        rows,
    )
    _write_bundle(
        FIGURE1_DIR / "figure1.csv",
        csv_text,
        FIGURE1_DIR / "provenance.json",
        {
            "source": "Keetch and Byram (1968), A Drought Index for Forest Fire Control, Research Paper SE-38, Figure 1 sample record (pp. 11-13)",
            "url": "https://research.fs.usda.gov/treesearch/40",
            "download_date": today,
            "subset_description": (
                "Transcription of the published 30-day Figure 1 worked example: daily precipitation, rounded "
                "maximum temperature, published net rain, Table 4 drought factor, and published KBDI. The series "
                "starts from the published previous-day KBDI of 164 hundredths of an inch and exercises the "
                "report's integer lookup tables (Table 4, 50-inch mean annual rainfall)."
            ),
            "checksum_sha256": checksum,
            "fixture_version": "1.0.0",
            "validation_tolerance": {"figure1_continuous_equation_atol_hundredths_in": _FIGURE1_ATOL},
            "citation": (
                "Keetch, J.J. and Byram, G.M. (1968) A Drought Index for Forest Fire Control. USDA Forest Service "
                "Research Paper SE-38. https://research.fs.usda.gov/treesearch/40"
            ),
            "license": (
                "U.S. Government work (17 USC 105); the source PDF is not redistributed. The corrected standard "
                "constant used by the test is documented by Alexander, M.E. (1990) Computer Calculation of the "
                "Keetch-Byram Drought Index - Programmers Beware! Fire Management Notes 51(4):23-25."
            ),
            "notes": (
                "Figure 1 is an exact oracle for the 1968 integer table workflow, not for the corrected continuous "
                "Equation 18 that fire.kbdi() implements. The tolerance is a bound on the accumulated table "
                "discretization (3 F temperature bins, 50-point deficit columns, integer drought factors), not a "
                "rounding allowance chosen to make the test pass. Source PDF SHA-256 at transcription: "
                f"{_SE38_PDF_SHA256}. The net_rain_in and table_drought_factor columns make the source's "
                "reduce-then-increase ordering auditable and are checked by the test independently of production code."
            ),
        },
    )


def _write_ghcn_fixture(today: str, text: str, raw_sha256: str) -> None:
    dates, precipitation_mm, temperature_celsius = _parse_ghcn_rows(text)
    mean_annual_mm = sum(precipitation_mm) / len(precipitation_mm) * 365.25
    expected = _reference_kbdi_mm(precipitation_mm, temperature_celsius, mean_annual_mm)
    rows = [
        [date, f"{precipitation:.1f}", f"{temperature:.1f}", f"{kbdi:.10f}"]
        for date, precipitation, temperature, kbdi in zip(
            dates, precipitation_mm, temperature_celsius, expected, strict=True
        )
    ]
    csv_text, checksum = _csv_text(
        ["date", "precipitation_mm", "maximum_temperature_c", "kbdi_mm"],
        rows,
    )
    _write_bundle(
        GHCN_DIR / "fresno_1991_2020.csv",
        csv_text,
        GHCN_DIR / "provenance.json",
        {
            "source": "NOAA NCEI GHCN-Daily, station USW00093193 (Fresno Yosemite International Airport, California)",
            "url": _GHCN_ACCESS_URL,
            "download_date": today,
            "subset_description": (
                f"Complete daily PRCP and TMAX for {_GHCN_START_DATE} through {_GHCN_END_DATE} (10958 days, no "
                "missing values). The expected kbdi_mm column was generated by the independent plain-Python "
                "calculator in scripts/prepare_kbdi_fixtures.py, starting from zero and deriving mean annual "
                f"precipitation from the record ({mean_annual_mm:.6f} mm/year)."
            ),
            "checksum_sha256": checksum,
            "fixture_version": "1.0.0",
            "validation_tolerance": {
                "regression_rtol": _GHCN_RTOL,
                "regression_atol_mm": _GHCN_ATOL_MM,
            },
            "citation": (
                "Menne, M.J., Durre, I., Vose, R.S., Gleason, B.E. and Houston, T.G. (2012) An Overview of the "
                "Global Historical Climatology Network-Daily Database. Journal of Atmospheric and Oceanic "
                "Technology 29(7):897-910. https://doi.org/10.1175/JTECH-D-11-00103.1"
            ),
            "license": "Public domain (U.S. Government work, 17 USC 105).",
            "notes": (
                "Regression coverage, not independent scientific validation: the expected values share the project's "
                "platform decisions (metric evaluation, zero initialization, contiguous daily observations) and "
                "differ from a production run only in summation order. Raw access-service download SHA-256: "
                f"{raw_sha256}. The station window was selected as the longest complete run in the station's "
                "1980-2024 record; 1991-2020 aligns with the current U.S. Climate Normals period and meets the "
                "library's 30-year derivation minimum."
            ),
        },
    )


def main() -> None:
    today = dt.date.today().isoformat()
    # Download before touching either committed bundle: a failed request must
    # leave the working tree's fixtures exactly as they were.
    text = _download_ghcn_csv()
    FIGURE1_DIR.mkdir(parents=True, exist_ok=True)
    GHCN_DIR.mkdir(parents=True, exist_ok=True)
    _write_figure1_fixture(today)
    _write_ghcn_fixture(today, text, _sha256(text.encode()))
    print(f"wrote {FIGURE1_DIR} and {GHCN_DIR}")


if __name__ == "__main__":
    main()
