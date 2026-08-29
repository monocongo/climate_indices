"""Characterization tests comparing Palmer-family outputs against NOAA nClimDiv.

These tests compare climate_indices' ``pdsi()`` and ``scpdsi()`` outputs
against the operational NOAA NCEI nClimDiv reference arrays committed under
``tests/fixture/nclimdiv/`` (see that directory's ``provenance.json``). They
are loose regression ceilings on aggregated statistics, not tight
oracle/equality checks -- see VALIDATION.md's "Palmer Authoritative-Reference
Decision" section for why nClimDiv is treated as an external comparison
rather than an independently authoritative scientific reference.

Unlike ``test_scpdsi.py::test_climate_division_matches_scpdsi_oracle``, these
tests deliberately do not compare per-division, per-month values: NOAA's
national fixed K-factors and climate_indices' per-division self-calibration
are expected to diverge in the details even when both are implemented
correctly, so only aggregate statistics across all 344 divisions are
asserted. If ``pdsi()``/``scpdsi()`` unexpectedly raises for any division,
the test fails loudly (no broad except/continue) -- the 344-division oracle
test in ``test_scpdsi.py`` already proves both functions succeed for every
real division, so a swallowed exception here would hide a real regression.

Ceilings are derived from the measurements in ``_PDSI_MEASURED`` and
``_SCPDSI_MEASURED`` with the fixed headroom documented in
``_HEADROOM_BOUNDS`` and enforced by
``test_ceilings_keep_documented_headroom``. The computation is fully
deterministic -- committed ``.npy`` inputs and no RNG anywhere in the Palmer
path -- so the headroom absorbs tolerated library/platform drift and
refactoring, not stochastic run-to-run noise.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from climate_indices import palmer, self_calibration

_FIXTURE_ROOT = Path(__file__).parent / "fixture"
_PALMER_ROOT = _FIXTURE_ROOT / "palmer"
_NCLIMDIV_ROOT = _FIXTURE_ROOT / "nclimdiv"

_DATA_START_YEAR = 1895
_CALIBRATION_YEAR_INITIAL = 1931
_CALIBRATION_YEAR_FINAL = 1990

_SERIES = ("pdsi", "phdi", "pmdi", "zindex")

# provenance.json documents 528,384 division-months of overlap. Assert a floor
# well below that so a regression that turns whole divisions into NaN shrinks
# the compared sample loudly instead of silently improving the statistics.
_MIN_COMPARED_MONTHS = 400_000

# Measured across all 344 divisions (calibration period 1931-1990).
_PDSI_MEASURED = {
    "pdsi": {"median": 0.0127, "mean": 0.0432, "p90": 0.0638, "max": 7.6336},
    "phdi": {"median": 0.0156, "max": 3.8524},
    "pmdi": {"median": 0.0191, "max": 5.5748},
    "zindex": {"median": 0.0129, "max": 1.0176},
}
_PDSI_CEILINGS = {
    "pdsi": {"median": 0.03, "mean": 0.1, "p90": 0.15, "max": 12.0},
    "phdi": {"median": 0.04, "max": 6.0},
    "pmdi": {"median": 0.05, "max": 9.0},
    "zindex": {"median": 0.03, "max": 1.6},
}

# Measured across all 344 divisions (GitHub issue #755).
_SCPDSI_MEASURED = {
    "pdsi": {"median": 0.4731, "mean": 0.7574, "p90": 1.7786, "max": 9.3753},
    "phdi": {"median": 0.6055, "max": 9.0613},
    "pmdi": {"median": 0.5714, "max": 8.9270},
    "zindex": {"median": 0.1525, "max": 6.9305},
}
_SCPDSI_CEILINGS = {
    "pdsi": {"median": 0.9, "mean": 1.5, "p90": 3.0, "max": 15.0},
    "phdi": {"median": 1.1, "max": 14.0},
    "pmdi": {"median": 1.0, "max": 14.0},
    "zindex": {"median": 0.3, "max": 11.0},
}

# (minimum, maximum) permitted ceiling-to-measurement ratio, by statistic.
_HEADROOM_BOUNDS = {"max": (1.5, 1.7), "other": (1.65, 2.7)}


@pytest.fixture(scope="module")
def division_dirs() -> tuple[Path, ...]:
    return tuple(sorted(path for path in _PALMER_ROOT.iterdir() if path.name.isdigit()))


@pytest.fixture(scope="module")
def awcs(palmer_awcs) -> dict:
    return palmer_awcs


@pytest.fixture(scope="module")
def row_by_division() -> dict[str, int]:
    divisions = json.loads((_NCLIMDIV_ROOT / "divisions.json").read_text(encoding="utf-8"))
    return {division: row for row, division in enumerate(divisions)}


@pytest.fixture(scope="module")
def nclimdiv() -> dict[str, np.ndarray]:
    return {name: np.load(_NCLIMDIV_ROOT / f"{name}.npy") for name in _SERIES}


def _abs_diffs(actual: np.ndarray, reference_row: np.ndarray) -> np.ndarray:
    """Absolute differences at months where both series report a value."""
    reference = reference_row.astype(np.float64)
    both_present = ~np.isnan(reference) & ~np.isnan(actual)
    return np.abs(actual[both_present].astype(np.float64) - reference[both_present])


def _summarize(all_diffs: dict[str, list[np.ndarray]]) -> dict[str, dict[str, float]]:
    """Aggregate per-division absolute differences into comparable statistics."""
    summary = {}
    for name, diffs in all_diffs.items():
        stacked = np.concatenate(diffs)
        assert stacked.size >= _MIN_COMPARED_MONTHS, (
            f"{name}: only {stacked.size} division-months compared, expected at least "
            f"{_MIN_COMPARED_MONTHS} -- the reference and computed series barely overlap"
        )
        summary[name] = {
            "count": int(stacked.size),
            "median": float(np.median(stacked)),
            "mean": float(np.mean(stacked)),
            "p90": float(np.percentile(stacked, 90)),
            "max": float(np.max(stacked)),
        }
    return summary


def _collect_diffs(function, division_dirs, awcs, row_by_division, nclimdiv) -> dict[str, dict[str, float]]:
    all_diffs: dict[str, list[np.ndarray]] = {name: [] for name in _SERIES}
    for division_dir in division_dirs:
        division = division_dir.name
        values = function(
            np.load(division_dir / "precips.npy"),
            np.load(division_dir / "pet.npy"),
            awcs[division],
            _DATA_START_YEAR,
            _CALIBRATION_YEAR_INITIAL,
            _CALIBRATION_YEAR_FINAL,
        )
        assert values[4] is not None, f"{division}: {function.__name__}() returned no fitted parameters"
        row = row_by_division[division]
        for offset, name in enumerate(_SERIES):
            all_diffs[name].append(_abs_diffs(values[offset], nclimdiv[name][row]))
    return _summarize(all_diffs)


def test_division_directories_match_nclimdiv_index(division_dirs, row_by_division):
    """Every fixture division must have a reference row, and vice versa."""
    assert {path.name for path in division_dirs} == set(row_by_division)


def test_ceilings_keep_documented_headroom():
    """Ceilings must stay within the headroom the module docstring advertises.

    Without this, a ceiling can be widened to accommodate a regression while the
    stated rationale silently stops describing the assertions.
    """
    for label, ceilings, measured in (
        ("pdsi", _PDSI_CEILINGS, _PDSI_MEASURED),
        ("scpdsi", _SCPDSI_CEILINGS, _SCPDSI_MEASURED),
    ):
        for series, stats in ceilings.items():
            for stat, ceiling in stats.items():
                low, high = _HEADROOM_BOUNDS["max" if stat == "max" else "other"]
                ratio = ceiling / measured[series][stat]
                assert low <= ratio <= high, (
                    f"{label} {series} {stat}: ceiling is {ratio:.2f}x measured, want {low}-{high}x"
                )


@pytest.mark.validation
@pytest.mark.parametrize(
    ("function", "ceilings"),
    [(palmer.pdsi, _PDSI_CEILINGS), (palmer.scpdsi, _SCPDSI_CEILINGS)],
    ids=["pdsi", "scpdsi"],
)
def test_palmer_vs_noaa_nclimdiv_characterization(function, ceilings, division_dirs, awcs, row_by_division, nclimdiv):
    """Aggregate agreement between a Palmer entry point and the NOAA nClimDiv arrays.

    ``pdsi()`` uses the same fixed, nationally uniform duration/K factors as
    NOAA's operational nClimDiv product, so agreement is close; its measured
    median |Delta| of 0.0127 is consistent with
    ``tests/fixture/nclimdiv/provenance.json``, which records that same median
    alongside 86.2% of months within 0.05 (a fraction this test does not
    recompute). ``scpdsi()`` self-calibrates duration and K-prime factors per
    division (Wells, Goddard, and Hayes 2004) while NOAA uses fixed national
    K-factors, so the two diverge substantially by design -- expected
    divergence, not a defect -- and its ceilings are correspondingly wider.
    """
    summary = _collect_diffs(function, division_dirs, awcs, row_by_division, nclimdiv)
    for series, stats in ceilings.items():
        for stat, ceiling in stats.items():
            assert summary[series][stat] < ceiling, f"{series} {stat}: {summary[series]}"


@pytest.mark.validation
def test_scpdsi_calibration_anchor_lands_on_target(division_dirs, awcs):
    """Calibration-period 2nd/98th percentiles of scPDSI should land on -/+4.

    This is the defining property of Wells self-calibration, and the source of
    the figures published in VALIDATION.md's "scPDSI Calibration-Anchor
    Measurement" section. ``scpdsi()`` applies a fixed three rescaling passes
    rather than iterating to a fixed point, so unsettled divisions retain a
    residual deviation; the ceilings bound that residual.
    """
    start = (_CALIBRATION_YEAR_INITIAL - _DATA_START_YEAR) * 12
    end = (_CALIBRATION_YEAR_FINAL - _DATA_START_YEAR + 1) * 12

    low_deviations = []
    high_deviations = []
    for division_dir in division_dirs:
        division = division_dir.name
        scpdsi_values = palmer.scpdsi(
            np.load(division_dir / "precips.npy"),
            np.load(division_dir / "pet.npy"),
            awcs[division],
            _DATA_START_YEAR,
            _CALIBRATION_YEAR_INITIAL,
            _CALIBRATION_YEAR_FINAL,
        )[0]
        window = scpdsi_values[start:end]
        low_deviations.append(abs(self_calibration.nan_safe_percentile(window, 0.02) - (-4.0)))
        high_deviations.append(abs(self_calibration.nan_safe_percentile(window, 0.98) - 4.0))

    low = np.asarray(low_deviations)
    high = np.asarray(high_deviations)
    assert low.size == high.size == len(division_dirs)

    # Measured: 2nd median deviation 0.0102, max 0.9730, 74% within 0.05, 93%
    # within 0.25; 98th median deviation 0.0130, max 1.5421, 70% and 91%.
    for label, deviations, median_ceiling, max_ceiling, within_005, within_025 in (
        ("2nd", low, 0.03, 2.0, 0.60, 0.85),
        ("98th", high, 0.03, 2.5, 0.55, 0.80),
    ):
        assert float(np.median(deviations)) < median_ceiling, f"{label}: median deviation {np.median(deviations)}"
        assert float(np.max(deviations)) < max_ceiling, f"{label}: max deviation {np.max(deviations)}"
        assert float(np.mean(deviations <= 0.05)) > within_005, f"{label}: within 0.05 fraction too low"
        assert float(np.mean(deviations <= 0.25)) > within_025, f"{label}: within 0.25 fraction too low"
