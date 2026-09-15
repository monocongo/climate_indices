"""KBDI reference fixtures and cross-implementation checks (#800).

Three evidence classes live here, and none of them is independent scientific
validation of the corrected continuous Equation 18 on its own:

- ``kbdi_se38_figure1`` reproduces the published Figure 1 worked example from
  Keetch and Byram (1968). Figure 1 exercises the report's integer lookup
  tables, so the continuous equation tracks it rather than reproducing it; the
  tolerance is read from the fixture provenance.
- ``kbdi_ghcn`` is a complete 30-year GHCN-Daily station record with frozen
  expected values from an independent plain-Python calculator. It is regression
  coverage over real weather, not an external oracle.
- The xclim and WFAS checks are cross-implementation and operational smoke
  tests respectively; xclim implements the later Finkele (2006) Australian
  variant, and WFAS does not publish a reproducible point-input/output archive,
  so neither is authoritative.
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

FIXTURE_DIR = Path(__file__).parent / "fixture"
_FIGURE1_DIR = FIXTURE_DIR / "kbdi_se38_figure1"
_GHCN_DIR = FIXTURE_DIR / "kbdi_ghcn"
_WFAS_DIR = FIXTURE_DIR / "wfas_kbdi"

_WFAS_REQUIRED_FILES = ("provenance.json", "metadata.json", "station_kbdi.csv")


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_provenance(directory: Path) -> dict[str, Any]:
    with (directory / "provenance.json").open() as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _se38_net_rain_in(precipitation_in: list[float]) -> list[float]:
    """Transcribe the SE-38 wet-spell rule in inches, independently of fire.kbdi()."""
    net_rain: list[float] = []
    wet_spell = 0.0
    for precipitation in precipitation_in:
        effective = 0.0
        if precipitation > 0.0:
            event_total = wet_spell + precipitation
            if wet_spell > 0.20:
                effective = precipitation
            elif event_total > 0.20:
                effective = event_total - 0.20
            wet_spell = event_total
        else:
            wet_spell = 0.0
        net_rain.append(effective)
    return net_rain


@pytest.mark.parametrize(
    ("directory", "csv_name"),
    [
        (_FIGURE1_DIR, "figure1.csv"),
        (_GHCN_DIR, "fresno_1991_2020.csv"),
    ],
    ids=["se38_figure1", "ghcn_fresno"],
)
def test_fixture_checksum_matches_provenance(directory: Path, csv_name: str) -> None:
    """The committed CSV must be the exact artifact the provenance records."""
    assert _sha256(directory / csv_name) == _load_provenance(directory)["checksum_sha256"]


def test_se38_figure1_net_rain_matches_the_published_wet_spell_rule() -> None:
    """Audit the published net-rain column against the SE-38 rule.

    This does not exercise production code; it protects the fixture itself, so
    a transcription error cannot quietly weaken the Figure 1 reference test.
    Net-rain values are committed to the published 0.01-inch precision, so the
    0.005-inch bound is half a unit in the published last place.
    """
    rows = _read_csv(_FIGURE1_DIR / "figure1.csv")
    precipitation = [float(row["precipitation_in"]) for row in rows]
    published_net_rain = [float(row["net_rain_in"]) for row in rows]

    np.testing.assert_allclose(_se38_net_rain_in(precipitation), published_net_rain, atol=0.005)


def test_se38_figure1_drought_factors_reproduce_the_published_series() -> None:
    """Audit the published Table 4 drought-factor column against the published KBDI series.

    The SE-38 table workflow reduces the index by the net rain and then adds
    that day's Table 4 factor, all in hundredths of an inch, starting from the
    published previous-day KBDI of 164. Reconstructing the series this way
    catches any transcription error in the factor column without touching
    production code, which is what the fixture provenance claims.
    """
    rows = _read_csv(_FIGURE1_DIR / "figure1.csv")
    kbdi = 164.0
    for row in rows:
        net_rain = round(float(row["net_rain_in"]) * 100)
        kbdi = max(0.0, kbdi - net_rain) + int(row["table_drought_factor"])
        assert kbdi == float(row["published_kbdi_hundredths_in"]), f"day {row['day']}"


def test_se38_figure1_matches_the_published_series() -> None:
    """Reproduce the published Figure 1 KBDI series with the continuous equation.

    Figure 1 quantizes temperature into 3 F bins, the deficit into 50-point
    columns, and the daily factor into integers, so the continuous Equation 18
    cannot match it exactly. The tolerance comes from the fixture provenance:
    it bounds that accumulated discretization over the 30-day series (measured
    maximum deviation at fixture preparation was 2.91) and is not an accuracy
    claim for either workflow.
    """
    rows = _read_csv(_FIGURE1_DIR / "figure1.csv")
    precipitation = np.array([float(row["precipitation_in"]) for row in rows])
    temperature = np.array([float(row["maximum_temperature_f"]) for row in rows])
    published = np.array([float(row["published_kbdi_hundredths_in"]) for row in rows])
    tolerance = _load_provenance(_FIGURE1_DIR)["validation_tolerance"]
    atol = tolerance["figure1_continuous_equation_atol_hundredths_in"]

    values = fire.kbdi(precipitation, temperature, 50.0, units="imperial", initial_kbdi=164.0)

    np.testing.assert_allclose(values, published, atol=atol)


def test_ghcn_station_fixture_matches_frozen_expected_values() -> None:
    """Run 30 years of real Fresno weather against the frozen expected values.

    The expected column was produced by the plain-Python calculator in
    ``scripts/prepare_kbdi_fixtures.py``, which shares the published contract
    and the project's platform decisions but no production code. The test
    derives mean annual precipitation inside ``fire.kbdi()`` from the record
    itself, so the tolerances bound only numpy-vs-math summation and the
    ten-decimal CSV storage. This is regression coverage, not independent
    scientific validation.
    """
    rows = _read_csv(_GHCN_DIR / "fresno_1991_2020.csv")
    precipitation = np.array([float(row["precipitation_mm"]) for row in rows])
    temperature = np.array([float(row["maximum_temperature_c"]) for row in rows])
    expected = np.array([float(row["kbdi_mm"]) for row in rows])
    tolerance = _load_provenance(_GHCN_DIR)["validation_tolerance"]

    values = fire.kbdi(precipitation, temperature)

    np.testing.assert_allclose(
        values,
        expected,
        rtol=tolerance["regression_rtol"],
        atol=tolerance["regression_atol_mm"],
    )


def test_wfas_operational_values_are_close() -> None:
    """Cross-check against USFS WFAS published KBDI values when a fixture exists.

    WFAS publishes operational maps rather than a point-input/output archive,
    and no reproducible point oracle has been published, so this test skips by
    default. If a fixture is supplied it must contain ``provenance.json``, a
    ``metadata.json`` with ``mean_annual_precipitation_in``,
    ``initial_kbdi_hundredths_in`` and ``tolerance_atol_hundredths_in``, and a
    ``station_kbdi.csv`` with columns ``date``, ``precipitation_in``,
    ``maximum_temperature_f`` and ``kbdi_hundredths_in``. The tolerance is
    declared by the provider because WFAS does not publish its equation
    variant, initialization, or rounding; a disagreement is a review trigger,
    never evidence that WFAS is the more correct formulation.
    """
    missing = [name for name in _WFAS_REQUIRED_FILES if not (_WFAS_DIR / name).exists()]
    if len(missing) == len(_WFAS_REQUIRED_FILES):
        pytest.skip(
            f"WFAS point fixtures not found at {_WFAS_DIR}; WFAS publishes operational maps, not a "
            "reproducible point-input/output archive. Provide provenance.json, metadata.json, and "
            "station_kbdi.csv to enable this cross-check."
        )
    if missing:
        pytest.fail(f"incomplete WFAS fixture at {_WFAS_DIR}; missing {', '.join(missing)}")

    with (_WFAS_DIR / "metadata.json").open() as handle:
        metadata = json.load(handle)
    rows = _read_csv(_WFAS_DIR / "station_kbdi.csv")
    precipitation = np.array([float(row["precipitation_in"]) for row in rows])
    temperature = np.array([float(row["maximum_temperature_f"]) for row in rows])
    reference = np.array([float(row["kbdi_hundredths_in"]) for row in rows])

    values = fire.kbdi(
        precipitation,
        temperature,
        metadata["mean_annual_precipitation_in"],
        units="imperial",
        initial_kbdi=metadata["initial_kbdi_hundredths_in"],
    )

    np.testing.assert_allclose(values, reference, atol=metadata["tolerance_atol_hundredths_in"])


def test_xclim_finkele_variant_is_close_on_a_rain_free_series() -> None:
    """Compare against xclim's Finkele/FFDI variant where xclim is installed.

    xclim v0.62.0 implements the later Finkele (2006) Australian variant with a
    5.0 mm rain allowance and the ``-0.00173`` mean-annual exponent, not the
    original SE-38 procedure or Alexander's corrected ``-0.001736`` metric
    constant. On a rain-free series the allowance is never consumed, so the only
    difference is the denominator constant: 0.35% at 300 mm/year. The 0.25 mm
    bound covers that difference accumulating while both series approach the
    203.2 mm cap (measured maximum deviation is 0.12 mm), and xclim remains a
    sanity check, not an oracle.
    """
    xclim = pytest.importorskip("xclim")
    xr = pytest.importorskip("xarray")
    pd = pytest.importorskip("pandas")

    days = 120
    precipitation = np.zeros(days)
    temperature = np.full(days, 32.0)
    mean_annual_mm = 300.0

    time = pd.date_range("2000-01-01", periods=days)
    pr = xr.DataArray(precipitation, dims="time", coords={"time": time}, attrs={"units": "mm/day"})
    tasmax = xr.DataArray(temperature, dims="time", coords={"time": time}, attrs={"units": "degC"})
    pr_annual = xr.DataArray(mean_annual_mm, attrs={"units": "mm/year"})

    reference = xclim.indices.fire.keetch_byram_drought_index(pr, tasmax, pr_annual).values
    values = fire.kbdi(precipitation, temperature, mean_annual_mm)

    np.testing.assert_allclose(values, reference, atol=0.25)
