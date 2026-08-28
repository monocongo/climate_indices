"""Unit tests for the Wells-lineage Palmer recursion used by scPDSI."""

import numpy as np
import pytest

from climate_indices import _palmer_wells
from climate_indices.exceptions import ConvergenceError


def _run(
    z: list[float],
    *,
    wetm: float = 1.0,
    wetb: float = 1.0,
    drym: float = 1.0,
    dryb: float = 1.0,
) -> _palmer_wells.WellsResult:
    return _palmer_wells.calculate(
        np.asarray(z, dtype=float),
        wetm=wetm,
        wetb=wetb,
        drym=drym,
        dryb=dryb,
    )


def test_wet_and_dry_spells_establish_at_half_unit_thresholds():
    wet = _run([1.0])
    dry = _run([-1.0])

    assert wet.pdsi[0] == pytest.approx(0.5)
    assert wet.x3[0] == pytest.approx(0.5)
    assert wet.pmdi[0] == pytest.approx(0.5)
    assert dry.pdsi[0] == pytest.approx(-0.5)
    assert dry.x3[0] == pytest.approx(-0.5)


def test_wet_spell_abatement_updates_probability_and_established_index():
    result = _run([4.0, -0.1])

    # The generalized effective-moisture threshold is wetm / 2 = 0.5.
    assert result.probability[1] == pytest.approx(60.0)
    assert result.x3[1] == pytest.approx(0.95)
    assert result.pdsi[1] == pytest.approx(0.95)


def test_probability_termination_uses_reference_tolerance():
    # The second period gives Prob=99.999995, which is below 100 but within
    # the reference tolerance of spell termination.
    result = _run([4.0, -0.49999995])

    assert result.probability[1] == pytest.approx(100.0)
    assert result.x3[1] == pytest.approx(0.0)
    assert result.pdsi[1] == pytest.approx(-0.249999975)


def test_exact_zero_candidates_select_zero_without_leaving_pending_state():
    result = _run([0.0, 0.0])

    np.testing.assert_array_equal(result.pdsi, np.zeros(2))
    np.testing.assert_array_equal(result.x1, np.zeros(2))
    np.testing.assert_array_equal(result.x2, np.zeros(2))
    np.testing.assert_array_equal(result.x3, np.zeros(2))


def test_dry_coefficient_uses_wet_intercept_but_dry_z_denominator():
    result = _run([-1.0, -1.0], wetb=1.0, drym=1.0, dryb=3.0)

    # dryc = 1 - 1/(1+wetb) = 0.5, while Z is divided by drym+dryb = 4.
    assert result.x2[0] == pytest.approx(-0.25)
    assert result.x2[1] == pytest.approx(-0.375)


def test_spell_establishment_backtracks_tentative_candidate_values():
    result = _run([0.6, -0.1, -1.0])

    # At period two both candidates are non-zero, so its final value cannot be
    # selected until the dry spell establishes in period three.
    np.testing.assert_allclose(result.pdsi, [0.3, -0.05, -0.525], rtol=0, atol=1e-12)


def test_missing_periods_preserve_state_and_remain_missing_in_every_output():
    with_missing = _run([0.6, np.nan, -0.1, -1.0])
    compressed = _run([0.6, -0.1, -1.0])

    assert np.isnan(with_missing.pdsi[1])
    assert np.isnan(with_missing.phdi[1])
    assert np.isnan(with_missing.pmdi[1])
    assert np.isnan(with_missing.probability[1])
    np.testing.assert_allclose(
        with_missing.pdsi[[0, 2, 3]],
        compressed.pdsi,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("z_value", [-np.inf, np.inf])
def test_infinite_z_values_raise_convergence_error(z_value):
    with pytest.raises(ConvergenceError, match="non-finite"):
        _run([z_value])


@pytest.mark.parametrize(
    ("wetm", "wetb", "drym", "dryb"),
    [
        (-1.0, 1.0, 1.0, 1.0),
        (np.inf, 1.0, 1.0, 1.0),
        (1.0, 1.0, -1.0, 1.0),
        (1.0, 1.0, np.nan, 1.0),
        (1.0, -1.0, 1.0, 1.0),
        (3.0, -1.0, 1.0, 1.0),
    ],
)
def test_invalid_duration_factor_denominators_raise_convergence_error(wetm, wetb, drym, dryb):
    with pytest.raises(ConvergenceError, match="duration factors"):
        _run([0.0], wetm=wetm, wetb=wetb, drym=drym, dryb=dryb)


def test_zero_abatement_denominator_raises_convergence_error():
    with pytest.raises(ConvergenceError, match="abatement probability"):
        _run([2.0, 0.0])


@pytest.mark.parametrize(
    ("wetm", "wetb", "drym", "dryb", "coefficient"),
    [
        (0.0, 1.0, 1.0, 1.0, "wetc"),
        (10.0, -2.0, 1.0, 1.0, "dryc"),
        (1000.0, -100.0, -1.0, 3.0, "dry_spell_c"),
    ],
)
def test_duration_factor_magnitude_at_or_above_one_raises_convergence_error(wetm, wetb, drym, dryb, coefficient):
    with pytest.raises(ConvergenceError, match=coefficient):
        _run([0.0], wetm=wetm, wetb=wetb, drym=drym, dryb=dryb)
