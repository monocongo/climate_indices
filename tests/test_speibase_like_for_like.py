"""Like-for-like SPEI comparison against the CSIC SPEIbase v2.11 grids.

The earlier ``tests/test_speibase_reference.py`` check compares climate_indices'
gamma/Thornthwaite SPEI against SPEIbase and can only claim plausibility: PET
method, distribution family, and precipitation input all differ. This module
removes those three confounds by feeding climate_indices the *same* inputs
SPEIbase was computed from — CRU TS 4.09 precipitation and FAO-56 Penman-
Monteith PET (``tests/fixture/speibase_cru_ts/``) — and standardizing with the
log-logistic distribution, then averaging the per-cell SPEI inside each climate
division exactly as SPEIbase does.

What remains is a small residual dominated by a PET day-length convention
difference — this test multiplies by leap-aware month lengths, matching the
public SPEIbase ``R/functions.R``, while a near-uniform month matches the
committed v2.11 grids more closely — plus the fit implementation and parameter
rounding. ``climate_indices.lmoments.fit_glo`` is a port of the same ``lmom``
PELGLO routine R SPEI uses, so this validates the port and the
per-cell-to-division pipeline rather than an algorithmically independent
estimator. The agreement floors are tight (correlation >= 0.97) rather than the
loose plausibility floors in the sibling module.

The compared series uses the corrected CRU TS units: precipitation is mm/month,
PET is mm/day, and PET is multiplied by the leap-aware month length before the
P - PET difference (following ``sbegueria/SPEIbase``'s ``R/functions.R``,
``spei.nc``).
"""

import hashlib
import json
import warnings
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.stats import pearsonr

from climate_indices import compute, indices

_FIXTURE_ROOT = Path(__file__).parent / "fixture"
_INPUT_ROOT = _FIXTURE_ROOT / "speibase_cru_ts"
_SPEIBASE_ROOT = _FIXTURE_ROOT / "speibase"
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"

_DATA_START_YEAR = 1901  # SPEIbase v2.11 / CRU TS 4.09 start year
_DATA_END_YEAR = 2024  # CRU TS 4.09 end year; SPEIbase calibrates on the full span
_N_MONTHS = (_DATA_END_YEAR - _DATA_START_YEAR + 1) * 12

_SCALES = (1, 3, 6, 12)
_FLOOR_METRICS = ("correlation", "sign_agreement", "category_agreement")
_RECORDED_METRICS = (*_FLOOR_METRICS, "mean_abs_difference")
# Keep in sync with scripts/prepare_speibase_cru_ts_inputs.py:_FLOOR_MARGINS.
_FLOOR_MARGINS = {"correlation": 0.02, "sign_agreement": 0.02, "category_agreement": 0.03}

_CATEGORY_BOUNDARIES = (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0)

_PROVENANCE = json.loads((_INPUT_ROOT / "provenance.json").read_text(encoding="utf-8"))
_DIVISIONS = json.loads((_INPUT_ROOT / "divisions.json").read_text(encoding="utf-8"))
_SPEIBASE_DIVISIONS = json.loads((_SPEIBASE_ROOT / "divisions.json").read_text(encoding="utf-8"))
_MEASURED: dict[str, dict[str, float]] = _PROVENANCE["measured_stats"]
_FLOORS: dict[str, dict[str, float]] = {}
for _key, _value in _PROVENANCE["validation_tolerance"].items():
    _division, _scale, _metric = _key.split("_", 2)
    _FLOORS.setdefault(f"{_division}_{_scale}", {})[_metric] = _value


def _load_fixture_script() -> ModuleType:
    """Load scripts/prepare_speibase_cru_ts_inputs.py without running its main()."""
    spec = spec_from_file_location(
        "prepare_speibase_cru_ts_inputs_test", _SCRIPTS_DIR / "prepare_speibase_cru_ts_inputs.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", autouse=True)
def _verify_fixture_checksum():
    """Reject a mixed or edited fixture generation before any assertion reads it."""
    hasher = hashlib.sha256()
    for npy_file in sorted(_INPUT_ROOT.glob("*.npy")):
        hasher.update(npy_file.read_bytes())
    assert hasher.hexdigest() == _PROVENANCE["checksum_sha256"], (
        "tests/fixture/speibase_cru_ts holds a mixed or edited fixture generation "
        f"(checksum {hasher.hexdigest()} != provenance {_PROVENANCE['checksum_sha256']}); "
        "rerun scripts/prepare_speibase_cru_ts_inputs.py"
    )


def _monthly_days(start_year: int, n_months: int) -> np.ndarray:
    """Days in each month, leap-aware, for a monthly series starting in January."""
    base = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=float)
    days = np.tile(base, n_months // 12 + 1)[:n_months]
    for index in range(n_months):
        year = start_year + index // 12
        if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
            if index % 12 == 1:
                days[index] = 29.0
    return days


def _load_inputs(division: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell CRU TS precipitation (mm/month) and PET (mm/day) for one division."""
    precip = np.load(_INPUT_ROOT / f"pre_{division}.npy").astype(np.float64)
    pet = np.load(_INPUT_ROOT / f"pet_{division}.npy").astype(np.float64)
    assert precip.shape == (len(_cells(division)), _N_MONTHS), (
        f"{division}: unexpected precipitation shape {precip.shape}"
    )
    assert pet.shape == precip.shape, f"{division}: PET shape {pet.shape} != precipitation {precip.shape}"
    return precip, pet


def _cells(division: str) -> list:
    return next(row["cells"] for row in _DIVISIONS if row["id"] == division)


def _categories(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, _CATEGORY_BOUNDARIES)


def _agreement(computed: np.ndarray, reference: np.ndarray) -> tuple[dict[str, float], int]:
    """Correlation, sign agreement, and category agreement over shared months."""
    both_present = ~np.isnan(computed) & ~np.isnan(reference)
    computed_values = computed[both_present].astype(np.float64)
    reference_values = reference[both_present].astype(np.float64)
    stats = {
        "correlation": float(pearsonr(computed_values, reference_values).statistic),
        "sign_agreement": float(np.mean(np.sign(computed_values) == np.sign(reference_values))),
        "category_agreement": float(np.mean(_categories(computed_values) == _categories(reference_values))),
    }
    return stats, int(np.count_nonzero(both_present))


def _computed_series(division: str, scale: int) -> np.ndarray:
    """Division-mean SPEI from the CRU TS inputs, calibrated over 1901-2024.

    Standardizes each cell with the log-logistic distribution and averages the
    per-cell values, matching SPEIbase's per-cell-then-mean pipeline.
    """
    precip, pet = _load_inputs(division)
    pet_mm = pet * _monthly_days(_DATA_START_YEAR, _N_MONTHS)
    cells = [
        indices.spei(
            precip[row],
            pet_mm[row],
            scale,
            indices.Distribution.loglogistic,
            compute.Periodicity.monthly,
            _DATA_START_YEAR,
            _DATA_START_YEAR,
            _DATA_END_YEAR,
        )
        for row in range(precip.shape[0])
    ]
    stacked = np.vstack(cells)
    assert not np.isnan(stacked[:, scale - 1 :]).all(axis=1).any(), (
        f"{division} SPEI-{scale}: a cell produced an all-NaN series"
    )
    with warnings.catch_warnings():
        # leading scale-1 months are all-NaN by construction (rolling-sum warmup)
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stacked, axis=0)


def test_input_cells_match_speibase_selection():
    """The CRU TS inputs must use the same cells SPEIbase was averaged over."""
    assert [row["id"] for row in _DIVISIONS] == [row["id"] for row in _SPEIBASE_DIVISIONS], (
        "speibase_cru_ts/divisions.json row order must match tests/fixture/speibase/divisions.json, "
        "which fixes the reference array row order"
    )
    reference_cells = {row["id"]: row["speibase_cells"] for row in _SPEIBASE_DIVISIONS}
    for row in _DIVISIONS:
        division = row["id"]
        assert row["cells"], f"{division}: no selected cells recorded"
        assert len(row["cells"]) == reference_cells[division], (
            f"{division}: {len(row['cells'])} CRU TS cells != {reference_cells[division]} SPEIbase cells"
        )
        _load_inputs(division)  # asserts the array shapes match the recorded cell count


def test_monthly_days_leap_handling():
    """The PET mm/day -> mm/month conversion must follow the Gregorian leap rule."""
    assert _monthly_days(1903, 12)[1] == 28.0
    leap = _monthly_days(1904, 12)
    assert leap[1] == 29.0 and leap[0] == 31.0  # February adjusts, January does not
    assert _monthly_days(1900, 12)[1] == 28.0  # century not divisible by 400
    assert _monthly_days(2000, 12)[1] == 29.0  # 400-year rule
    assert _monthly_days(1901, _N_MONTHS).sum() == 45291.0  # 1901-2024 day count


def test_provenance_declares_all_series():
    """provenance.json covers every division and timescale this module tests."""
    expected = {f"{row['id']}_spei{scale:02d}" for row in _DIVISIONS for scale in _SCALES}
    assert set(_MEASURED) == expected
    assert set(_FLOORS) == expected
    for series, metrics in _FLOORS.items():
        assert set(metrics) == set(_FLOOR_METRICS), (
            f"{series}: floors missing metrics {set(_FLOOR_METRICS) - set(metrics)}"
        )


def test_floors_keep_documented_slack():
    """Every floor must stay pinned to its measurement within the documented margin.

    Without this, a floor can be edited down to accommodate a regression while
    the stated rationale stops describing the assertions. Floors are quantized
    down to two decimals, so the slack sits in [margin, margin + 0.01).
    """
    for series, metrics in _FLOORS.items():
        for metric, floor in metrics.items():
            slack = _MEASURED[series][metric] - floor
            margin = _FLOOR_MARGINS[metric]
            assert margin <= slack <= margin + 0.01, (
                f"{series} {metric}: floor has {slack:.4f} slack, want {margin}-{margin + 0.01}"
            )


@pytest.mark.validation
@pytest.mark.parametrize("scale", _SCALES)
def test_spei_like_for_like(scale: int):
    """Input-matched SPEI must reproduce SPEIbase to the recorded tight floors."""
    reference = np.load(_SPEIBASE_ROOT / f"spei{scale:02d}.npy")
    reference_months = reference.shape[1]
    for row, division_row in enumerate(_DIVISIONS):
        division = division_row["id"]
        computed = _computed_series(division, scale)[:reference_months]
        stats, compared_months = _agreement(computed, reference[row])
        series = f"{division}_spei{scale:02d}"

        min_compared = reference_months - (scale - 1)
        assert compared_months >= min_compared, (
            f"{series}: only {compared_months} months compared, expected at least {min_compared} -- "
            f"fixture row missing/permuted or computed series is NaN"
        )
        for metric in _FLOOR_METRICS:
            assert stats[metric] >= _FLOORS[series][metric], (
                f"{series}: {metric} = {stats[metric]:.6f}, floor = {_FLOORS[series][metric]:.4f} "
                f"(measured {_MEASURED[series][metric]:.6f} with documented slack)"
            )


@pytest.mark.validation
def test_refresh_script_measurement_reproduces_recorded_expectations():
    """The refresh script must re-measure the agreement it records in provenance."""
    script = _load_fixture_script()
    inputs = {row["id"]: {"pre": _load_inputs(row["id"])[0], "pet": _load_inputs(row["id"])[1]} for row in _DIVISIONS}
    measured = script._measure_agreement(inputs)

    deviations = {
        f"{division}_spei{scale:02d} {metric}": abs(stats[metric] - script._EXPECTED_STATS[division][scale][metric])
        for division, per_scale in measured.items()
        for scale, stats in per_scale.items()
        for metric in _RECORDED_METRICS
    }
    assert len(deviations) == len(_DIVISIONS) * len(_SCALES) * len(_RECORDED_METRICS)
    worst_series = max(deviations, key=deviations.get)
    assert deviations[worst_series] <= script._EXPECTATION_TOLERANCE, (
        f"{worst_series} deviates {deviations[worst_series]:.6f} from the recorded expectation, "
        f"over the {script._EXPECTATION_TOLERANCE} band"
    )
