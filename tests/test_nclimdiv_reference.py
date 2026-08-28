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
"""

import json
from pathlib import Path

import numpy as np
import pytest

from climate_indices import palmer

_FIXTURE_ROOT = Path(__file__).parent / "fixture"
_PALMER_ROOT = _FIXTURE_ROOT / "palmer"
_NCLIMDIV_ROOT = _FIXTURE_ROOT / "nclimdiv"
_DIVISION_DIRS = tuple(sorted(path for path in _PALMER_ROOT.iterdir() if path.name.isdigit()))
_AWCS = json.loads((_FIXTURE_ROOT / "palmer_awc.json").read_text(encoding="utf-8"))
_DIVISIONS = json.loads((_NCLIMDIV_ROOT / "divisions.json").read_text(encoding="utf-8"))
_ROW_BY_DIVISION = {division: row for row, division in enumerate(_DIVISIONS)}

_DATA_START_YEAR = 1895
_CALIBRATION_YEAR_INITIAL = 1931
_CALIBRATION_YEAR_FINAL = 1990


def _division_inputs(division_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    division = division_dir.name
    return (
        np.load(division_dir / "precips.npy"),
        np.load(division_dir / "pet.npy"),
        _AWCS[division],
    )


def _load_nclimdiv_arrays() -> dict[str, np.ndarray]:
    return {name: np.load(_NCLIMDIV_ROOT / f"{name}.npy") for name in ("pdsi", "phdi", "pmdi", "zindex")}


def _abs_diffs(actual: np.ndarray, reference_row: np.ndarray) -> np.ndarray:
    """Absolute differences at months where both series report a value."""
    reference = reference_row.astype(np.float64)
    finite = ~np.isnan(reference) & ~np.isnan(actual)
    return np.abs(actual[finite].astype(np.float64) - reference[finite])


def _summarize(all_diffs: dict[str, list[np.ndarray]]) -> dict[str, dict[str, float]]:
    summary = {}
    for name, diffs in all_diffs.items():
        stacked = np.concatenate(diffs)
        summary[name] = {
            "median": float(np.median(stacked)),
            "mean": float(np.mean(stacked)),
            "p90": float(np.percentile(stacked, 90)),
            "max": float(np.max(stacked)),
        }
    return summary


@pytest.mark.validation
def test_pdsi_vs_noaa_nclimdiv_characterization():
    """Aggregate agreement between plain pdsi() and the NOAA nClimDiv arrays.

    ``pdsi()`` uses the same fixed, nationally uniform duration/K factors as
    NOAA's operational nClimDiv product, so agreement is expected to be
    close. Measured across all 344 divisions: pdsi median |Delta| 0.0127
    (matches tests/fixture/nclimdiv/provenance.json's documented 86.2%
    within 0.05), mean 0.0432, p90 0.0638, max 7.6336; phdi median 0.0156,
    max 3.8524; pmdi median 0.0191, max 5.5748; zindex median 0.0129, max
    1.0176. Ceilings below give ~2x headroom on medians and ~1.5x headroom
    on maxima.
    """
    nclimdiv = _load_nclimdiv_arrays()
    all_diffs: dict[str, list[np.ndarray]] = {"pdsi": [], "phdi": [], "pmdi": [], "zindex": []}

    for division_dir in _DIVISION_DIRS:
        division = division_dir.name
        precips, pet, awc = _division_inputs(division_dir)
        pdsi_values, phdi_values, pmdi_values, zindex_values, _params = palmer.pdsi(
            precips, pet, awc, _DATA_START_YEAR, _CALIBRATION_YEAR_INITIAL, _CALIBRATION_YEAR_FINAL
        )
        row = _ROW_BY_DIVISION[division]
        all_diffs["pdsi"].append(_abs_diffs(pdsi_values, nclimdiv["pdsi"][row]))
        all_diffs["phdi"].append(_abs_diffs(phdi_values, nclimdiv["phdi"][row]))
        all_diffs["pmdi"].append(_abs_diffs(pmdi_values, nclimdiv["pmdi"][row]))
        all_diffs["zindex"].append(_abs_diffs(zindex_values, nclimdiv["zindex"][row]))

    summary = _summarize(all_diffs)

    # Measured: pdsi median 0.0127, mean 0.0432, p90 0.0638, max 7.6336.
    assert summary["pdsi"]["median"] < 0.03, summary["pdsi"]
    assert summary["pdsi"]["mean"] < 0.1, summary["pdsi"]
    assert summary["pdsi"]["p90"] < 0.15, summary["pdsi"]
    assert summary["pdsi"]["max"] < 12.0, summary["pdsi"]

    # Measured: phdi median 0.0156 max 3.8524; pmdi median 0.0191 max 5.5748.
    assert summary["phdi"]["median"] < 0.04, summary["phdi"]
    assert summary["phdi"]["max"] < 6.0, summary["phdi"]
    assert summary["pmdi"]["median"] < 0.05, summary["pmdi"]
    assert summary["pmdi"]["max"] < 9.0, summary["pmdi"]

    # Measured: zindex median 0.0129, max 1.0176.
    assert summary["zindex"]["median"] < 0.03, summary["zindex"]
    assert summary["zindex"]["max"] < 2.0, summary["zindex"]


@pytest.mark.validation
def test_scpdsi_vs_noaa_nclimdiv_characterization():
    """Aggregate agreement between scpdsi() and the NOAA nClimDiv arrays.

    Unlike plain pdsi(), scpdsi() self-calibrates duration and K-prime
    factors per climate division (Wells, Goddard, and Hayes 2004), while
    NOAA's operational nClimDiv product uses fixed national K-factors. The
    two are expected to diverge substantially by design -- this is expected
    divergence, not a defect -- so these ceilings are wide, calibrated from
    aggregate statistics measured across all 344 divisions (GitHub issue
    #755), with headroom for run-to-run noise rather than a tight
    per-division tolerance.
    """
    nclimdiv = _load_nclimdiv_arrays()
    all_diffs: dict[str, list[np.ndarray]] = {"pdsi": [], "phdi": [], "pmdi": [], "zindex": []}

    for division_dir in _DIVISION_DIRS:
        division = division_dir.name
        precips, pet, awc = _division_inputs(division_dir)
        scpdsi_values, scphdi_values, scpmdi_values, sczindex_values, params = palmer.scpdsi(
            precips, pet, awc, _DATA_START_YEAR, _CALIBRATION_YEAR_INITIAL, _CALIBRATION_YEAR_FINAL
        )
        assert params is not None, f"{division}: scpdsi() unexpectedly returned no fitted parameters"
        row = _ROW_BY_DIVISION[division]
        all_diffs["pdsi"].append(_abs_diffs(scpdsi_values, nclimdiv["pdsi"][row]))
        all_diffs["phdi"].append(_abs_diffs(scphdi_values, nclimdiv["phdi"][row]))
        all_diffs["pmdi"].append(_abs_diffs(scpmdi_values, nclimdiv["pmdi"][row]))
        all_diffs["zindex"].append(_abs_diffs(sczindex_values, nclimdiv["zindex"][row]))

    summary = _summarize(all_diffs)

    # Measured (issue #755, 344 divisions): pdsi median 0.4731, mean 0.7574,
    # p90 1.7786, max 9.3753. Ceilings below are ~1.5-2x median/mean/p90 and
    # ~1.5x max.
    assert summary["pdsi"]["median"] < 0.9, summary["pdsi"]
    assert summary["pdsi"]["mean"] < 1.5, summary["pdsi"]
    assert summary["pdsi"]["p90"] < 3.0, summary["pdsi"]
    assert summary["pdsi"]["max"] < 15.0, summary["pdsi"]

    # Measured: phdi median 0.6055, max 9.0613.
    assert summary["phdi"]["median"] < 1.1, summary["phdi"]
    assert summary["phdi"]["max"] < 14.0, summary["phdi"]

    # Measured: pmdi median 0.5714, max 8.9270.
    assert summary["pmdi"]["median"] < 1.0, summary["pmdi"]
    assert summary["pmdi"]["max"] < 14.0, summary["pmdi"]

    # Measured: zindex median 0.1525, max 6.9305.
    assert summary["zindex"]["median"] < 0.3, summary["zindex"]
    assert summary["zindex"]["max"] < 11.0, summary["zindex"]
