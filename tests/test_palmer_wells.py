"""Unit tests for the Wells-lineage Palmer recursion used by scPDSI."""

from dataclasses import replace

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
        # Each tuple must trip its named coefficient and no other, so the match
        # pattern below is anchored to the message's "<name> = " fragment rather
        # than searching for the bare name. Computed values, for maintenance:
        #   wetc case:        wetc=1.000000  dryc=0.500000  dry_spell_c=0.500000
        #   dryc case:        wetc=-0.250000 dryc=2.000000  dry_spell_c=0.500000
        #   dry_spell_c case: wetc=-0.111111 dryc=0.990099  dry_spell_c=1.500000
        # Note the dry_spell_c case leaves dryc only ~1% below the threshold --
        # perturb those factors and it can start tripping dryc instead.
        (0.0, 1.0, 1.0, 1.0, "wetc"),
        (10.0, -2.0, 1.0, 1.0, "dryc"),
        (1000.0, -100.0, -1.0, 3.0, "dry_spell_c"),
    ],
    ids=["wetc", "dryc", "dry_spell_c"],
)
def test_duration_factor_magnitude_at_or_above_one_raises_convergence_error(wetm, wetb, drym, dryb, coefficient):
    with pytest.raises(ConvergenceError, match=rf"recursion: {coefficient} = "):
        _run([0.0], wetm=wetm, wetb=wetb, drym=drym, dryb=dryb)


def test_dry_candidate_clamp_does_not_bound_magnitude():
    """The x2 clamp is one-sided, which is why ``dryc`` needs a magnitude check.

    ``_candidate_values`` applies ``min(0.0, ...)`` to x2. That caps it at zero
    but leaves negative values free to grow, so a non-contracting ``dryc`` makes
    x2 diverge. This pins the mechanism the ``_validated_factors`` docstring
    cites as the justification for rejecting ``|dryc| >= 1``.
    """
    factors = _palmer_wells._Factors(
        wetm=1.0,
        wetb=1.0,
        drym=1.0,
        dryb=1.0,
        wet_denominator=2.0,
        dry_denominator=2.0,
        wetc=0.5,
        # The value the (10.0, -2.0, 1.0, 1.0) factors above produce.
        dryc=2.0,
        dry_spell_c=0.5,
    )
    state = _palmer_wells._State(x1=0.0, x2=-1.0, x3=0.0, v=0.0, probability=0.0)
    magnitudes = []
    for _ in range(5):
        _, x2 = _palmer_wells._candidate_values(state, 0.0, factors)
        magnitudes.append(abs(x2))
        state = replace(state, x2=x2)

    assert magnitudes == sorted(magnitudes)
    assert magnitudes[-1] > magnitudes[0]
