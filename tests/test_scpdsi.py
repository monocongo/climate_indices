"""Public-contract and oracle validation tests for :func:`palmer.scpdsi`."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from climate_indices import palmer
from climate_indices.exceptions import ConvergenceError

ATOL = 5e-5
RTOL = 0
_FIXTURE_ROOT = Path(__file__).parent / "fixture"
_PALMER_ROOT = _FIXTURE_ROOT / "palmer"
_DIVISION_DIRS = tuple(sorted(path for path in _PALMER_ROOT.iterdir() if path.name.isdigit()))
_AWCS = json.loads((_FIXTURE_ROOT / "palmer_awc.json").read_text(encoding="utf-8"))


def _division_inputs(division: str = "0101") -> tuple[np.ndarray, np.ndarray, float]:
    division_dir = _PALMER_ROOT / division
    return (
        np.load(division_dir / "precips.npy"),
        np.load(division_dir / "pet.npy"),
        _AWCS[division],
    )


def _call(division: str = "0101", fitting_params=None):
    precips, pet, awc = _division_inputs(division)
    return palmer.scpdsi(precips, pet, awc, 1895, 1931, 1990, fitting_params)


def test_public_signature_matches_pdsi_and_is_exported():
    assert "scpdsi" in palmer.__all__
    assert inspect.signature(palmer.scpdsi) == inspect.signature(palmer.pdsi)


def test_all_missing_input_returns_four_same_length_missing_arrays_and_no_params():
    precips = np.full(13, np.nan)
    pet = np.full(13, np.nan)

    scpdsi, scphdi, scpmdi, sczindex, params = palmer.scpdsi(precips, pet, 5.0, 2000, 2000, 2000)

    for values in (scpdsi, scphdi, scpmdi, sczindex):
        assert values.shape == precips.shape
        assert np.isnan(values).all()
    assert params is None


def test_mismatched_inputs_raise_the_same_error_as_pdsi():
    with pytest.raises(ValueError, match="Incompatible precipitation and PET arrays"):
        palmer.scpdsi(np.ones(60), np.ones(59), 5.0, 2000, 2000, 2004)


@pytest.mark.parametrize("function", [palmer.pdsi, palmer.scpdsi])
def test_mismatched_inputs_raise_before_the_all_missing_fast_path(function):
    with pytest.raises(ValueError, match="Incompatible precipitation and PET arrays"):
        function(np.full(60, np.nan), np.ones(59), 5.0, 2000, 2000, 2004)


@pytest.mark.parametrize(
    ("calibration_year_initial", "calibration_year_final"),
    [(1894, 1990), (1931, 2023), (1990, 1931)],
)
def test_invalid_calibration_period_raises_value_error(calibration_year_initial, calibration_year_final):
    precips, pet, awc = _division_inputs()

    with pytest.raises(ValueError, match="calibration period"):
        palmer.scpdsi(
            precips,
            pet,
            awc,
            1895,
            calibration_year_initial,
            calibration_year_final,
        )


@pytest.mark.parametrize(
    ("calibration_year_initial", "calibration_year_final"),
    [(1999, 2000), (2000, 2001), (2000, 1999)],
)
def test_all_missing_input_still_validates_the_calibration_period(calibration_year_initial, calibration_year_final):
    precips = np.full(12, np.nan)
    pet = np.full(12, np.nan)

    with pytest.raises(ValueError, match="calibration period"):
        palmer.scpdsi(
            precips,
            pet,
            5.0,
            2000,
            calibration_year_initial,
            calibration_year_final,
        )


@pytest.mark.parametrize("function", [palmer.pdsi, palmer.scpdsi])
def test_negative_precipitation_is_clipped_even_when_other_values_are_missing(function):
    precips, pet, awc = _division_inputs()
    mixed = precips.copy()
    mixed[-2] = -10.0
    mixed[-1] = np.nan
    expected_input = mixed.copy()
    expected_input[-2] = 0.0

    actual = function(mixed, pet, awc, 1895, 1931, 1990)
    expected = function(expected_input, pet, awc, 1895, 1931, 1990)

    for actual_values, expected_values in zip(actual[:4], expected[:4], strict=True):
        np.testing.assert_allclose(actual_values, expected_values, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("function", [palmer.pdsi, palmer.scpdsi])
def test_infinite_inputs_are_rejected(function):
    precips, pet, awc = _division_inputs()
    precips[-1] = np.inf

    with pytest.raises(ValueError, match="infinite"):
        function(precips, pet, awc, 1895, 1931, 1990)


def test_supplied_cafec_coefficients_are_reused_but_duration_factors_are_recalibrated():
    _, _, _, _, standard_params = _call()
    assert standard_params is not None
    supplied = dict(standard_params)
    supplied.update(wetm=-99.0, wetb=-99.0, drym=-99.0, dryb=-99.0)

    *_, params = _call(fitting_params=supplied)

    assert params is not None
    for name in ("alpha", "beta", "gamma", "delta"):
        np.testing.assert_array_equal(params[name], standard_params[name])
    expected = np.load(_PALMER_ROOT / "0101" / "scdurfact.npy")
    np.testing.assert_allclose(
        [params["wetm"], params["wetb"], params["drym"], params["dryb"]],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_recursion_runs_exactly_four_times_with_three_cumulative_rescalings(monkeypatch):
    seen_z: list[np.ndarray] = []

    def fake_calculate(z, **_kwargs):
        seen_z.append(np.asarray(z).copy())
        values = np.where(np.isnan(z), np.nan, 1.0)
        return SimpleNamespace(pdsi=values, phdi=values, pmdi=values)

    monkeypatch.setattr(palmer.self_calibration, "duration_factors", lambda _z, _sign: (1.0, 1.0))
    monkeypatch.setattr(
        palmer.self_calibration,
        "nan_safe_percentile",
        lambda _values, fraction: -2.0 if fraction == 0.02 else 2.0,
    )
    monkeypatch.setattr(palmer._palmer_wells, "calculate", fake_calculate)

    _call()

    assert len(seen_z) == 4
    np.testing.assert_allclose(seen_z[1], seen_z[0] * 2.0, atol=0, rtol=0, equal_nan=True)
    np.testing.assert_allclose(seen_z[2], seen_z[1] * 2.0, atol=0, rtol=0, equal_nan=True)
    np.testing.assert_allclose(seen_z[3], seen_z[2] * 2.0, atol=0, rtol=0, equal_nan=True)


def test_invalid_fitted_duration_factors_raise_convergence_error(monkeypatch):
    monkeypatch.setattr(palmer.self_calibration, "duration_factors", lambda _z, _sign: (-1.0, 1.0))

    with pytest.raises(ConvergenceError, match="duration factors"):
        _call()


@pytest.mark.parametrize(("dry", "wet"), [(0.0, 2.0), (-2.0, 0.0), (1.0, 2.0), (-2.0, -1.0)])
def test_invalid_calibration_percentiles_raise_convergence_error(monkeypatch, dry, wet):
    def fake_calculate(z, **_kwargs):
        values = np.where(np.isnan(z), np.nan, 1.0)
        return SimpleNamespace(pdsi=values, phdi=values, pmdi=values)

    monkeypatch.setattr(palmer.self_calibration, "duration_factors", lambda _z, _sign: (1.0, 1.0))
    monkeypatch.setattr(
        palmer.self_calibration,
        "nan_safe_percentile",
        lambda _values, fraction: dry if fraction == 0.02 else wet,
    )
    monkeypatch.setattr(palmer._palmer_wells, "calculate", fake_calculate)

    with pytest.raises(ConvergenceError, match="percentile"):
        _call()


def test_scpdsi_oracle_contains_all_climate_divisions():
    assert len(_DIVISION_DIRS) == 344


@pytest.mark.validation
@pytest.mark.parametrize("division_dir", _DIVISION_DIRS, ids=lambda path: path.name)
def test_climate_division_matches_scpdsi_oracle(division_dir):
    division = division_dir.name
    scpdsi, scphdi, scpmdi, sczindex, params = palmer.scpdsi(
        np.load(division_dir / "precips.npy"),
        np.load(division_dir / "pet.npy"),
        _AWCS[division],
        1895,
        1931,
        1990,
    )
    assert params is not None

    for name, actual in (
        ("scpdsi", scpdsi),
        ("scphdi", scphdi),
        ("scpmdi", scpmdi),
        ("sczindex", sczindex),
    ):
        np.testing.assert_allclose(
            actual,
            np.load(division_dir / f"{name}.npy"),
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=f"{division}: {name} mismatch",
        )

    np.testing.assert_allclose(
        [params["wetm"], params["wetb"], params["drym"], params["dryb"]],
        np.load(division_dir / "scdurfact.npy"),
        atol=ATOL,
        rtol=RTOL,
        err_msg=f"{division}: duration factors mismatch",
    )
