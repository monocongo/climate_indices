"""Characterization tests comparing SPI against NOAA NCEI's climdiv SPI product.

These tests compare climate_indices' ``indices.spi()`` (Pearson Type III path)
against the operational NOAA NCEI climate-divisional SPI reference arrays
committed under ``tests/fixture/ncei_spi/`` (see that directory's
``provenance.json``). They are loose regression ceilings on aggregated
statistics, not tight oracle/equality checks, for the same reason documented
in ``tests/test_nclimdiv_reference.py``: NCEI publishes only 2 decimal places
and (per Baldwin & Chen 2020) does not actually use its documented fixed
1931-1990 calibration window.

Resolves GitHub issue #777 (part of #769): NCEI's drought-readme.txt states a
fixed 1931-1990 calibration window, but empirically that window disagrees
badly with these files. Calibrating against the full input period of record
(1895-2022, matching the committed precipitation fixtures) reproduces NCEI's
values far more closely -- confirming Baldwin & Chen's finding -- so these
tests use full-period-of-record calibration, not the README's stated window.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from climate_indices import compute, indices

_FIXTURE_ROOT = Path(__file__).parent / "fixture"
_PALMER_ROOT = _FIXTURE_ROOT / "palmer"
_NCEI_SPI_ROOT = _FIXTURE_ROOT / "ncei_spi"

_DATA_START_YEAR = 1895
_DATA_END_YEAR = 2022  # matches tests/fixture/palmer/<division>/precips.npy length

_SCALES = (1, 2, 3, 6, 9, 12, 24)

_N_MONTHS = (_DATA_END_YEAR - _DATA_START_YEAR + 1) * 12

# Slack below the maximum comparable months per division (longer scales lose
# the leading `scale - 1` months to the rolling-sum warmup), tolerating a few
# legitimately missing NCEI months. Every division must clear this floor, so a
# regression that turns a division into NaN -- or a fixture whose rows are
# permuted or absent -- fails loudly instead of washing out in the aggregate.
_PER_DIVISION_SLACK = 12

# Measured across all 344 divisions (GitHub issue #777), full-period-of-record
# calibration (1895-2022). See tests/fixture/ncei_spi/provenance.json.
_MEASURED = {
    1: {"median": 0.0136, "p90": 0.0342, "max": 1.2900},
    2: {"median": 0.0129, "p90": 0.0298, "max": 0.9576},
    3: {"median": 0.0126, "p90": 0.0287, "max": 0.8376},
    6: {"median": 0.0126, "p90": 0.0275, "max": 0.6563},
    9: {"median": 0.0123, "p90": 0.0260, "max": 0.4088},
    12: {"median": 0.0122, "p90": 0.0252, "max": 0.3644},
    24: {"median": 0.0304, "p90": 0.0827, "max": 1.1598},
}
_CEILINGS = {
    1: {"median": 0.03, "p90": 0.07, "max": 2.2},
    2: {"median": 0.03, "p90": 0.06, "max": 1.6},
    3: {"median": 0.03, "p90": 0.06, "max": 1.4},
    6: {"median": 0.03, "p90": 0.06, "max": 1.1},
    9: {"median": 0.03, "p90": 0.06, "max": 0.7},
    12: {"median": 0.03, "p90": 0.06, "max": 0.65},
    24: {"median": 0.06, "p90": 0.17, "max": 1.9},
}

# (minimum, maximum) permitted ceiling-to-measurement ratio.
_HEADROOM_BOUNDS = {"max": (1.5, 1.8), "other": (1.8, 2.5)}


@pytest.fixture(scope="module")
def divisions() -> list[str]:
    return json.loads((_FIXTURE_ROOT / "nclimdiv" / "divisions.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def precips_by_division(divisions) -> dict[str, np.ndarray]:
    return {division: np.load(_PALMER_ROOT / division / "precips.npy") for division in divisions}


def _abs_diffs(computed: np.ndarray, reference_row: np.ndarray) -> np.ndarray:
    """Absolute differences at months where both series report a value."""
    reference = reference_row.astype(np.float64)
    both_present = ~np.isnan(reference) & ~np.isnan(computed)
    return np.abs(computed[both_present] - reference[both_present])


def _summarize(diffs: list[np.ndarray]) -> dict[str, float]:
    stacked = np.concatenate(diffs)
    return {
        "median": float(np.median(stacked)),
        "p90": float(np.percentile(stacked, 90)),
        "max": float(np.max(stacked)),
    }


def test_ceilings_keep_documented_headroom():
    """Ceilings must stay within the headroom the module docstring advertises.

    Without this, a ceiling can be widened to accommodate a regression while
    the stated rationale silently stops describing the assertions.
    """
    for scale, stats in _CEILINGS.items():
        for stat, ceiling in stats.items():
            low, high = _HEADROOM_BOUNDS["max" if stat == "max" else "other"]
            ratio = ceiling / _MEASURED[scale][stat]
            assert low <= ratio <= high, f"scale {scale} {stat}: ceiling is {ratio:.2f}x measured, want {low}-{high}x"


@pytest.fixture
def pearson_fallbacks(monkeypatch) -> list[str]:
    """Record any Pearson->gamma fallback inside ``indices.spi()``.

    The characterization claims Pearson Type III coverage; ``spi()`` silently
    falls back to gamma on fitting failure or excessive NaNs, so the test must
    observe and reject that fallback rather than comparing gamma values.
    """
    fallbacks: list[str] = []
    original = indices._fallback_strategy.log_fallback_warning

    def spy(reason: str, context: str = "") -> None:
        fallbacks.append(reason)
        original(reason, context)

    monkeypatch.setattr(indices._fallback_strategy, "log_fallback_warning", spy)
    return fallbacks


@pytest.mark.validation
@pytest.mark.parametrize("scale", _SCALES)
def test_spi_vs_noaa_ncei_climdiv_characterization(scale, divisions, precips_by_division, pearson_fallbacks):
    """Aggregate agreement between ``indices.spi()`` and NOAA's climdiv SPI.

    Uses the Pearson Type III distribution (matching NCEI's documented
    distribution choice) and full-period-of-record calibration (1895-2022),
    per the empirical finding recorded in
    ``tests/fixture/ncei_spi/provenance.json``.
    """
    reference = np.load(_NCEI_SPI_ROOT / f"sp{scale:02d}.npy")
    diffs = []
    for row, division in enumerate(divisions):
        computed = indices.spi(
            precips_by_division[division],
            scale,
            indices.Distribution.pearson,
            _DATA_START_YEAR,
            _DATA_START_YEAR,
            _DATA_END_YEAR,
            compute.Periodicity.monthly,
        )
        row_diffs = _abs_diffs(computed, reference[row])
        min_compared = _N_MONTHS - (scale - 1) - _PER_DIVISION_SLACK
        assert row_diffs.size >= min_compared, (
            f"scale {scale} division {division}: only {row_diffs.size} months compared, "
            f"expected at least {min_compared} -- fixture row missing/permuted or computed series is NaN"
        )
        diffs.append(row_diffs)

    assert not pearson_fallbacks, f"scale {scale}: Pearson fit fell back to gamma: {pearson_fallbacks}"

    summary = _summarize(diffs)
    for stat, ceiling in _CEILINGS[scale].items():
        assert summary[stat] < ceiling, f"scale {scale} {stat}: {summary}"


def test_provenance_characterization_matches_test_constants():
    """provenance.json must describe the criteria this module actually asserts."""
    provenance = json.loads((_NCEI_SPI_ROOT / "provenance.json").read_text(encoding="utf-8"))
    expected_ceilings = {
        f"sp{scale:02d}_{stat}": value for scale, stats in _CEILINGS.items() for stat, value in stats.items()
    }
    assert provenance["validation_tolerance"] == expected_ceilings
    assert provenance["measured_stats"] == {f"sp{scale:02d}": stats for scale, stats in _MEASURED.items()}


def test_division_rows_match_nclimdiv_index(divisions):
    """Every fixture row must line up with the shared 344-division index."""
    for scale in _SCALES:
        reference = np.load(_NCEI_SPI_ROOT / f"sp{scale:02d}.npy")
        assert reference.shape == (len(divisions), (_DATA_END_YEAR - _DATA_START_YEAR + 1) * 12)
