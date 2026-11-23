import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from climate_indices.drought import _palmer_pdi, palmer
from climate_indices.drought._palmer_duration import DurationFactors
from climate_indices.exceptions import ConvergenceError


def _prepared() -> palmer._PalmerPrepared:
    """A minimal, structurally-valid prepared struct for exercising the CAFEC stage."""
    return palmer._initialize_prepared(
        precips=np.zeros(12),
        pet=np.zeros(12),
        awc=1.0,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2000,
    )


def _state() -> _palmer_pdi._State:
    """A recursion state for one year of a single location, every Z value missing."""
    return _palmer_pdi._initialize_state(np.full((1, 12, 1), np.nan))


def test_initialize_prepared_sets_default_duration_factors() -> None:
    prepared = _prepared()

    # standard Palmer PDSI has no wet/dry distinction in its duration factors
    assert prepared.wetm == pytest.approx(prepared.drym)
    assert prepared.wetb == pytest.approx(prepared.dryb)

    # the implied duration-factor weighting fraction c = b / (m + b) must reproduce
    # Palmer's published constant (0.897), regardless of how m/b are derived
    c = prepared.wetb / (prepared.wetm + prepared.wetb)
    assert c == pytest.approx(0.897)

    # m + b must reproduce Palmer's published 1/q = 3.0
    assert (prepared.wetm + prepared.wetb) == pytest.approx(3.0)


def test_duration_factor_c_rejects_zero_factor_sum() -> None:
    with pytest.raises(ValueError, match="must not sum to zero"):
        _palmer_pdi._weighting_fraction(1.0, -1.0)


def test_weighting_fraction_keeps_the_division_form() -> None:
    """The pdi.f lineage divides, so its coefficient is ``b / (m + b)`` exactly.

    These constants make the division form differ bitwise from the complement
    form, so the test fails if the implementation switches forms.
    """
    m, b = 0.1, 0.2

    assert _palmer_pdi._weighting_fraction(m, b) == b / (m + b)
    assert _palmer_pdi._weighting_fraction(m, b) != 1.0 - m / (m + b)


def test_pdi_validation_accepts_factors_the_wells_cross_coefficient_would_reject() -> None:
    """The pdi.f recurrence never forms the Wells cross coefficient.

    ``c = b / (m + b)`` is -0.667 (wet) and 0.769 (dry), both contractions, so the
    pdi.f recursion can use these factors.  The Wells lineage additionally derives
    ``dryc = 1 - drym / (drym + wetb) = 4.0`` and rejects the same set.
    """
    factors = _palmer_pdi.PdiDurationFactors.from_fitted(1.0, -0.4, 0.3, 1.0)

    assert factors.wetm == 1.0
    with pytest.raises(ConvergenceError):
        DurationFactors.from_fitted(1.0, -0.4, 0.3, 1.0)

    # the same factors the Wells lineage rejects must drive the public pdsi()
    # override end to end, not just pass validation
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    override = {"wetm": 1.0, "wetb": -0.4, "drym": 0.3, "dryb": 1.0}

    overridden_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=override)

    assert np.isfinite(overridden_pdsi).any()


def test_pdi_validation_rejects_a_non_contracting_weighting_fraction() -> None:
    with pytest.raises(ConvergenceError, match="wetc"):
        _palmer_pdi.PdiDurationFactors.from_fitted(1.0, -0.5, 1.0, 1.0)


def test_pdi_validation_rejects_a_zero_denominator() -> None:
    with pytest.raises(ConvergenceError, match="standard PDSI recursion"):
        _palmer_pdi.PdiDurationFactors.from_fitted(1.0, -1.0, 1.0, 1.0)


def test_select_duration_factors_follows_the_sign_of_x3() -> None:
    """No established spell (x3 == 0) takes the wet pair; the sign of x3 selects after that."""
    factors = _palmer_pdi.PdiDurationFactors.from_fitted(1.0, 2.0, 3.0, 4.0)
    state = _state()

    state.x3 = np.array([0.0])
    m, b = _palmer_pdi._select_duration_factors(factors, state)
    np.testing.assert_array_equal(m, [1.0])
    np.testing.assert_array_equal(b, [2.0])

    state.x3 = np.array([-1.0])
    m, b = _palmer_pdi._select_duration_factors(factors, state)
    np.testing.assert_array_equal(m, [3.0])
    np.testing.assert_array_equal(b, [4.0])

    state.x3 = np.array([1.0])
    m, b = _palmer_pdi._select_duration_factors(factors, state)
    np.testing.assert_array_equal(m, [1.0])
    np.testing.assert_array_equal(b, [2.0])


def test_pdi_factors_are_not_interchangeable_between_wet_and_dry_spells() -> None:
    """Swapping the wet and dry pairs changes the recursion, through its interface."""
    z = np.array([2.0, 3.0, -1.0, -4.0, 0.5, -0.75, -0.25, 1.5, -2.0, 0.0, 2.5, -3.0]).reshape(1, 12, 1)

    wet_heavy = _palmer_pdi.calculate(z, _palmer_pdi.PdiDurationFactors.from_fitted(3.0, 1.0, 1.0, 1.0))
    dry_heavy = _palmer_pdi.calculate(z, _palmer_pdi.PdiDurationFactors.from_fitted(1.0, 1.0, 3.0, 1.0))

    assert not np.allclose(wet_heavy.pdsi, dry_heavy.pdsi, equal_nan=True)


def test_pdi_recursion_pins_a_z_sequence_through_its_interface() -> None:
    """A Z sequence driven through the recursion interface locks its arithmetic.

    The values were captured from the lineage-preserving refactor, so this is a
    regression pin, not an independent correctness check. ``test_palmer``
    corroborates the same path against the committed fixtures at ``atol=5e-5``,
    and only runs under ``-m validation``.
    """
    z = np.array([2.0, 3.0, -1.0, -4.0, 0.5, -0.75, -0.25, 1.5, -2.0, 0.0, 2.5, -3.0]).reshape(1, 12, 1)

    result = _palmer_pdi.calculate(z, _palmer_pdi.PdiDurationFactors.from_fitted(1.0, 2.0, 3.0, 4.0))

    expected = [
        0.666666666667,
        1.444444444444,
        -0.142857142857,
        -0.65306122449,
        -0.301749271137,
        -0.279571012078,
        -0.195469149759,
        0.5,
        -0.285714285714,
        -0.163265306122,
        0.833333333333,
        -0.428571428571,
    ]
    phdi_expected = [
        0.666666666667,
        1.444444444444,
        0.62962962963,
        -0.65306122449,
        -0.301749271137,
        -0.279571012078,
        -0.195469149759,
        0.5,
        -0.285714285714,
        -0.163265306122,
        0.833333333333,
        -0.428571428571,
    ]
    pmdi_expected = [
        0.666666666667,
        1.444444444444,
        -0.009989417989,
        -0.65306122449,
        -0.301749271137,
        -0.279571012078,
        -0.195469149759,
        0.5,
        -0.285714285714,
        -0.163265306122,
        0.833333333333,
        -0.428571428571,
    ]

    np.testing.assert_allclose(result.pdsi.reshape(-1), expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.phdi.reshape(-1), phdi_expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.pmdi.reshape(-1), pmdi_expected, rtol=0, atol=1e-12)


def test_pdi_recursion_rejects_infinite_z_values() -> None:
    """The pdi.f entry point rejects infinite Z like its Wells sibling."""
    z = np.zeros((1, 12, 1))
    z[0, 0, 0] = np.inf
    factors = _palmer_pdi.PdiDurationFactors.from_fitted(1.0, 1.0, 1.0, 1.0)

    with pytest.raises(ConvergenceError, match="non-finite"):
        _palmer_pdi.calculate(z, factors)


def test_pdi_recursion_rejects_a_non_twelve_month_axis() -> None:
    """The documented (years, 12, n_cells) contract covers the month axis, not just ndim."""
    factors = _palmer_pdi.PdiDurationFactors.from_fitted(1.0, 1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="shape"):
        _palmer_pdi.calculate(np.zeros((1, 11, 1)), factors)
    with pytest.raises(ValueError, match="shape"):
        _palmer_pdi.calculate(np.zeros((1, 13, 1)), factors)


def test_calc_cafec_zindex_writes_the_zindex() -> None:
    """The shared CAFEC/Z-index step feeds both the PDSI and scPDSI recursions.

    The CAFEC value and the Z-index are recomputed with the pre-refactor
    named-intermediate grouping and compared exactly: the recursion branches on
    exact comparisons downstream, so a 1-ulp reassociation is a behavior change.
    """
    prepared = _prepared()
    prepared.alpha = np.full((12,), 3.24)
    prepared.beta = np.full((12,), 1.52)
    prepared.gamma = np.full((12,), 6.51)
    prepared.delta = np.full((12,), 0.73)
    prepared.ak = np.full((12,), 6.0)
    prepared.pet[0, 0] = 5.36
    prepared.prdat[0, 0] = 3.66
    prepared.spdat[0, 0] = 0.59
    prepared.pldat[0, 0] = 5.07
    prepared.precips[0, 0] = 0.3
    z = np.full((1, 12, 1), np.nan)

    cafec = palmer._calc_cafec_zindex(prepared, z, 0, 0)

    cet = prepared.alpha[0] * prepared.pet[0, 0]
    cr = prepared.beta[0] * prepared.prdat[0, 0]
    cro = prepared.gamma[0] * prepared.spdat[0, 0]
    cl = prepared.delta[0] * prepared.pldat[0, 0]
    expected_cafec = cet + cr + cro - cl
    assert cafec == expected_cafec
    assert z[0, 0] == prepared.ak[0] * (prepared.precips[0, 0] - expected_cafec)


def test_custom_duration_factors_change_pdsi_output() -> None:
    """pdsi() accepts duration-factor overrides through its fitting parameters."""
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    custom = {"wetm": 1.0, "wetb": 2.0, "drym": 3.0, "dryb": 4.0}

    default_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003)
    custom_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=custom)

    assert np.isfinite(custom_pdsi).any()
    assert np.array_equal(np.isnan(default_pdsi), np.isnan(custom_pdsi))
    assert not np.allclose(default_pdsi, custom_pdsi, equal_nan=True)


def test_duration_factor_override_distinguishes_wet_from_dry() -> None:
    """Swapping the wet and dry pairs changes the recursion; wet and dry are not interchangeable."""
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    wet_heavy = {"wetm": 3.0, "wetb": 1.0, "drym": 1.0, "dryb": 1.0}
    dry_heavy = {"wetm": 1.0, "wetb": 1.0, "drym": 3.0, "dryb": 1.0}

    wet_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=wet_heavy)
    dry_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=dry_heavy)

    assert not np.allclose(wet_pdsi, dry_pdsi, equal_nan=True)


def test_pdsi_returned_params_reproduce_a_duration_factor_override() -> None:
    """The returned parameters echo the effective duration factors, so reuse is lossless."""
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    custom = {"wetm": 1.0, "wetb": 2.0, "drym": 3.0, "dryb": 4.0}

    custom_pdsi, *_, params = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=custom)
    assert params is not None
    assert [params[name] for name in ("wetm", "wetb", "drym", "dryb")] == [1.0, 2.0, 3.0, 4.0]

    rerun_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=params)

    np.testing.assert_array_equal(custom_pdsi, rerun_pdsi)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        pytest.param({"wetm": 1.0}, r"missing: wetb, drym, dryb", id="partial"),
        pytest.param(
            {"wetm": 1.0, "wetb": 1.0, "drym": 1.0, "dryb": [1.0]}, "dryb must be a finite scalar", id="non-scalar"
        ),
        pytest.param(
            {"wetm": 1.0, "wetb": 1.0, "drym": 1.0, "dryb": np.inf}, "dryb must be a finite scalar", id="non-finite"
        ),
        pytest.param(
            {"wetm": 1.0, "wetb": 1.0, "drym": 1.0, "dryb": "x"}, "dryb must be a finite scalar", id="non-numeric"
        ),
        pytest.param(
            {"wetm": 1.0, "wetb": 1.0, "drym": 1.0, "dryb": 10**1000},
            "dryb must be a finite scalar",
            id="overflow",
        ),
        pytest.param(
            {"wetm": np.ma.masked, "wetb": 1.0, "drym": 1.0, "dryb": 1.0},
            "wetm must be a finite scalar",
            id="masked",
        ),
        pytest.param(
            {"wetm": np.ma.array(1.0, mask=True), "wetb": 1.0, "drym": 1.0, "dryb": 1.0},
            "wetm must be a finite scalar",
            id="masked-array",
        ),
        pytest.param(
            # np.ma.is_masked reads a bare ``_mask`` attribute and raises
            # AttributeError on a non-masked object; the gate checks the type first
            {"wetm": types.SimpleNamespace(_mask="x"), "wetb": 1.0, "drym": 1.0, "dryb": 1.0},
            "wetm must be a finite scalar",
            id="bare-mask-attribute",
        ),
    ],
)
def test_invalid_duration_factor_override_is_rejected(override: dict[str, Any], message: str) -> None:
    """A malformed override is a caller error, not a silent default.

    A masked factor is missing data, not a number: ``np.asarray`` would drop the
    mask and hand the backing value to the recursion.
    """
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)

    with pytest.raises(ValueError, match=message):
        palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=override)


def test_palmer_default_factors_as_override_reproduce_the_default_run() -> None:
    """Palmer's own factors, supplied as an override, must reproduce the default run.

    Pins that a run given Palmer's constants through the override is bit-identical
    to the run that never saw them. Slot routing is pinned with distinct values by
    ``test_pdsi_returned_params_reproduce_a_duration_factor_override``; the default
    pair values are equal across wet and dry, so they cannot pin routing here.
    """
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    defaults = DurationFactors.from_defaults()
    as_override = {
        "wetm": defaults.wetm,
        "wetb": defaults.wetb,
        "drym": defaults.drym,
        "dryb": defaults.dryb,
    }

    default_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003)
    overridden_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=as_override)

    np.testing.assert_array_equal(default_pdsi, overridden_pdsi)


def test_duration_factor_override_does_not_preempt_input_validation() -> None:
    """An invalid override must not mask a data error the caller can act on."""
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    override = {"wetm": 1.0}

    with pytest.raises(ValueError, match="Incompatible precipitation and PET arrays"):
        palmer.pdsi(precips, pet[:-1], 5.0, 2000, 2000, 2003, fitting_params=override)

    infinite = precips.copy()
    infinite[0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        palmer.pdsi(infinite, pet, 5.0, 2000, 2000, 2003, fitting_params=override)

    # the calibration period is validated by the preparation stage that runs before
    # the override is resolved
    with pytest.raises(ValueError, match="calibration period"):
        palmer.pdsi(precips, pet, 5.0, 2000, 1990, 2003, fitting_params=override)


def test_duration_factor_override_is_not_validated_for_all_missing_input() -> None:
    """All-missing input keeps its documented NaN arrays and ``None`` parameters.

    The override is resolved after the all-missing fast path, as the CAFEC fitting
    parameters already are, so a partial override cannot turn that return into a raise.
    """
    all_missing = np.full(12 * 4, np.nan)

    pdsi_values, _, _, _, params = palmer.pdsi(
        all_missing, all_missing, 5.0, 2000, 2000, 2003, fitting_params={"wetm": 1.0}
    )

    assert params is None
    assert np.all(np.isnan(pdsi_values))


def test_cafec_ratio_substitutes_exact_zero_and_leaves_a_zero_denominator() -> None:
    """A month with no accumulated water-balance term takes ``both_zero``.

    A zero denominator with a nonzero numerator is not undefined the same way:
    the reference leaves 0.0 there. The sub-epsilon elements make either exact
    test fail if it becomes a tolerance.
    """
    tiny = np.finfo(float).tiny
    numerator = np.array([0.0, 2.0, 0.0, 1.0, tiny])
    denominator = np.array([0.0, 0.0, 4.0, tiny, 0.0])

    np.testing.assert_array_equal(
        palmer._calc_cafec_ratio(numerator, denominator, both_zero=1.0),
        np.array([1.0, 0.0, 0.0, 1.0 / tiny, 0.0]),
    )
    np.testing.assert_array_equal(
        palmer._calc_cafec_ratio(numerator, denominator, both_zero=0.0),
        np.array([0.0, 0.0, 0.0, 1.0 / tiny, 0.0]),
    )


def test_case_selects_near_normal_when_no_spell_is_established() -> None:
    """x3 is exactly 0.0 when no spell is established, and that exact zero --
    not a tolerance -- picks the larger-magnitude incipient index."""
    prob = np.array([50.0])
    x1 = np.array([1.5])
    x2 = np.array([-1.0])

    assert _palmer_pdi._case(prob, x1, x2, np.array([0.0]))[0] == 1.5
    # an established spell (x3 != 0) reports the interpolated severity instead
    assert _palmer_pdi._case(prob, x1, x2, np.array([-2.0]))[0] == -0.25
    # a sub-epsilon x3 is still an established spell under the exact test; a
    # tolerance would classify it as zero and return the near-normal 1.5
    assert _palmer_pdi._case(prob, x1, x2, np.array([-np.finfo(float).tiny]))[0] == 0.75


def test_record_index_values_falls_back_to_pdsi_when_no_spell_is_established() -> None:
    """PHDI has no severity of its own without an established spell (px3
    exactly 0.0), so it records the PDSI value; with a spell it keeps px3."""
    state = _state()
    state.px3[0, 0, 0] = 0.0
    state.px3[0, 1, 0] = -2.5
    state.px3[0, 2, 0] = -np.finfo(float).tiny
    values = np.array([3.0, 4.0, 5.0])

    _palmer_pdi._record_index_values(state, np.zeros(3, dtype=int), np.arange(3), values, np.array([0]))

    assert state.pdsi[0, 0, 0] == 3.0
    assert state.phdi[0, 0, 0] == 3.0  # no spell: the recorded PDSI value
    assert state.phdi[0, 1, 0] == -2.5  # established spell: its own severity
    assert state.phdi[0, 2, 0] == -np.finfo(float).tiny  # sub-epsilon, still a spell


def test_finish_up_falls_back_to_pdsi_when_no_spell_is_established() -> None:
    """_finish_up repeats the no-established-spell fallback for the months left
    pending when the record ends, under the same exact-zero test."""
    state = _state()
    state.k8 = np.array([2])
    state.indexj[0, 0], state.indexm[0, 0] = 0, 0
    state.indexj[1, 0], state.indexm[1, 0] = 0, 1
    state.x[0, 0, 0] = 3.0
    state.px3[0, 0, 0] = 0.0
    state.x[0, 1, 0] = 4.0
    state.px3[0, 1, 0] = -np.finfo(float).tiny

    _palmer_pdi._finish_up(state)

    assert state.pdsi[0, 0, 0] == 3.0
    assert state.phdi[0, 0, 0] == 3.0  # no spell: the PDSI value
    assert state.pdsi[0, 1, 0] == 4.0
    # a sub-epsilon px3 is still an established spell, so PHDI keeps it rather
    # than falling back to the PDSI value
    assert state.phdi[0, 1, 0] == -np.finfo(float).tiny


def test_finish_up_gives_each_pending_month_its_own_pmdi() -> None:
    """Each month flushed at the end of the record takes the CASE of its own
    ppr/px1/px2/px3, not the final month's (#1666)."""
    state = _state()
    state.k8 = np.array([2])
    state.indexj[0, 0], state.indexm[0, 0] = 0, 3
    state.indexj[1, 0], state.indexm[1, 0] = 0, 4
    state.x[0, 3, 0], state.x[0, 4, 0] = 1.0, 2.0
    # month 3: an established spell, 40% ending probability
    state.ppr[0, 3, 0], state.px1[0, 3, 0], state.px2[0, 3, 0], state.px3[0, 3, 0] = 40.0, 0.5, -0.5, 1.0
    # month 4: a different state
    state.ppr[0, 4, 0], state.px1[0, 4, 0], state.px2[0, 4, 0], state.px3[0, 4, 0] = 80.0, 1.5, -2.5, 2.0
    # the last calendar month of the record holds a third state that must not leak
    state.ppr[0, 11, 0], state.px1[0, 11, 0], state.px2[0, 11, 0], state.px3[0, 11, 0] = 10.0, 9.0, -9.0, 5.0

    _palmer_pdi._finish_up(state)

    # CASE by hand: (1 - ppr/100) * x3 + ppr/100 * x2 for x3 > 0
    assert state.wplm[0, 3, 0] == pytest.approx(0.6 * 1.0 + 0.4 * -0.5, abs=1e-15)
    assert state.wplm[0, 4, 0] == pytest.approx(0.2 * 2.0 + 0.8 * -2.5, abs=1e-15)


def test_pending_months_at_the_end_of_the_record_get_their_own_pmdi() -> None:
    """Through the whole recursion: the last three months of this record are still
    pending when it ends, and each carries the PMDI of its own state (#1666)."""
    factors = _palmer_pdi.PdiDurationFactors.from_fitted(0.309, 2.691, 0.309, 2.691)
    z = np.array([-0.65, -1.75, 2.61, -0.74, 0.49, -0.39, 2.38, 1.98, 0.95, -3.31, 0.08, 1.03]).reshape(1, 12, 1)

    state = _palmer_pdi._initialize_state(z)
    for month in range(12):
        _palmer_pdi._advance_month(factors, state, 0, month)
    pending = range(12 - int(state.k8[0]), 12)
    assert list(pending) == [9, 10, 11]

    result = _palmer_pdi.calculate(z, factors)

    own = [_palmer_pdi._case(state.ppr[0, m], state.px1[0, m], state.px2[0, m], state.px3[0, m])[0] for m in pending]
    np.testing.assert_array_equal(result.pmdi[0, 9:, 0], own)
    # the old flush wrote December's CASE into all three
    assert len(set(own)) == 3
    assert not np.any(np.isnan(result.pmdi))

    # The next month resolves the spell and changes PDSI, not PMDI.
    extended_z = np.r_[z.ravel(), -3.0, np.zeros(11)].reshape(2, 12, 1)
    extended = _palmer_pdi.calculate(extended_z, factors)
    assert not np.array_equal(result.pdsi[0, 9:, 0], extended.pdsi[0, 9:, 0])
    np.testing.assert_array_equal(result.pmdi[0, 9:, 0], extended.pmdi[0, 9:, 0])


def test_finish_up_leaves_months_of_a_resolved_spell_alone() -> None:
    """_finish_up flushes only the months still pending (#1665, ADR-0030).

    Slots at or beyond k8 hold months a closed spell already resolved; a second
    flush would overwrite their recorded values with the provisional ones. This is
    the unit-level check of the new semantics; the end-to-end regressions for the
    stale high-water mark are the two ``calculate()`` tests that follow.
    """
    state = _state()
    state.k8 = np.array([1])  # only the first slot is still pending
    for slot, month in enumerate((0, 1)):
        state.indexj[slot, 0], state.indexm[slot, 0] = 0, month
        state.x[0, month, 0] = 9.0
    state.pdsi[0, 1, 0], state.phdi[0, 1, 0], state.wplm[0, 1, 0] = 1.5, 2.5, 3.5

    _palmer_pdi._finish_up(state)

    assert state.pdsi[0, 0, 0] == 9.0  # pending: flushed
    assert (state.pdsi[0, 1, 0], state.phdi[0, 1, 0], state.wplm[0, 1, 0]) == (1.5, 2.5, 3.5)


def test_finish_up_flushes_each_cell_up_to_its_own_pending_count() -> None:
    """_finish_up keeps the per-cell k8 mask when cells have different runs pending.

    A regression that flushed the longest cell's slot count for every cell would
    overwrite the shorter cell's resolved slot 1 with its provisional values.
    """
    state = _palmer_pdi._initialize_state(np.full((1, 12, 2), np.nan))
    state.k8 = np.array([1, 2])
    for cell in (0, 1):
        for slot, month in enumerate((0, 1)):
            state.indexj[slot, cell], state.indexm[slot, cell] = 0, month
            state.x[0, month, cell] = 9.0
    # cell 0's slot 1 is a resolved month; cell 1's slot 1 is still pending
    state.pdsi[0, 1, 0], state.phdi[0, 1, 0], state.wplm[0, 1, 0] = 1.5, 2.5, 3.5

    _palmer_pdi._finish_up(state)

    assert state.pdsi[0, 0, 0] == 9.0  # cell 0 slot 0 pending: flushed
    assert (state.pdsi[0, 1, 0], state.phdi[0, 1, 0], state.wplm[0, 1, 0]) == (1.5, 2.5, 3.5)
    assert state.pdsi[0, 0, 1] == 9.0  # cell 1 both slots pending: flushed
    assert state.pdsi[0, 1, 1] == 9.0


def test_a_resolved_month_keeps_its_value_when_nothing_is_pending_at_the_end() -> None:
    """#1665: the REFIMPL-29 "F" sequence, where the stale ``k8max`` rewrote a resolved month.

    The spell resolves at the fifth month and the sixth resolves directly, so nothing is
    pending when the record ends. Before the fix the fifth month's PDSI was re-flushed to
    its provisional -0.618839 and its PMDI to the final month's 1.897. Values are the
    library's own, pinned to the same contract as the Rust test
    ``finish_up_leaves_months_of_a_resolved_spell_alone`` in ``palmer_pdi.rs``, which
    checks month 4 of the six-month series that this test pads to a whole year.
    """
    wetm = drym = (1.0 - 0.897) / (1.0 / 3.0)
    wetb = dryb = 0.897 / (1.0 / 3.0)
    factors = _palmer_pdi.PdiDurationFactors(wetm, wetb, drym, dryb)
    z = np.concatenate([np.zeros(6), [-2.0, -1.0, -1.3, -2.0, 3.0, 3.0]])  # whole year, no-op padding

    result = _palmer_pdi.calculate(z.reshape(1, 12, 1), factors)

    # same operation order as the Rust kernel, so agreement is to rounding
    assert result.pdsi.reshape(-1)[10] == pytest.approx(1.0, abs=1e-12)
    assert result.phdi.reshape(-1)[10] == pytest.approx(-0.6188390862539994, abs=1e-12)
    assert result.pmdi.reshape(-1)[10] == pytest.approx(0.9003972913896878, abs=1e-12)


def test_resolved_months_survive_truncating_the_record_division_0101() -> None:
    """#1665: truncating division 0101 at December 2002 must not change resolved months.

    Before the fix the stale ``k8max`` re-flushed February-November 2002, replacing their
    PMDI with the final month's 1.783. Only the months a spell still holds open at the
    truncation point (here December 2002) may differ from the full-record run.
    """
    z = np.load(Path(__file__).parent / "fixture" / "palmer" / "0101" / "zindex.npy")
    wetm = drym = (1.0 - 0.897) / (1.0 / 3.0)
    wetb = dryb = 0.897 / (1.0 / 3.0)
    factors = _palmer_pdi.PdiDurationFactors(wetm, wetb, drym, dryb)
    end = (2002 - 1895 + 1) * 12
    full = _palmer_pdi.calculate(z.reshape(-1, 12, 1), factors)
    cut = _palmer_pdi.calculate(z[:end].reshape(-1, 12, 1), factors)

    resolved = slice(0, end - 1)  # every month before December 2002
    for name in ("pdsi", "phdi", "pmdi"):
        # same arithmetic sequence in both runs, so agreement is to rounding only
        np.testing.assert_allclose(
            getattr(cut, name).reshape(-1)[resolved],
            getattr(full, name).reshape(-1)[resolved],
            rtol=0,
            atol=1e-12,
            err_msg=name,
        )
    assert not np.allclose(cut.pmdi.reshape(-1)[end - 11 : end - 1], 1.783, atol=1e-3)


def test_non_contracting_duration_factor_override_is_attributed_to_pdsi() -> None:
    """The override's ConvergenceError names the pdsi path, not the scPDSI calibration."""
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    non_contracting = {"wetm": 1.0, "wetb": -0.5, "drym": 1.0, "dryb": 1.0}

    with pytest.raises(ConvergenceError, match="duration-factor override") as error:
        palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=non_contracting)

    assert error.value.algorithm == "PDSI duration-factor override"


@pytest.mark.parametrize(
    ("x1", "x2", "expected"),
    [
        (0.5, -0.5, 0.5),  # exact |X1| == |X2| tie goes to X1, as in pdi.f CASE (878-879)
        (0.3, -0.7, -0.7),  # |X2| larger -> X2
        (0.7, -0.3, 0.7),  # |X1| larger -> X1
        (0.0, 0.0, 0.0),
    ],
)
def test_case_near_normal_selects_x2_only_when_strictly_larger(x1: float, x2: float, expected: float) -> None:
    result = _palmer_pdi._case(np.array([0.0]), np.array([x1]), np.array([x2]), np.array([0.0]))

    assert result[0] == expected
    assert np.signbit(result[0]) == np.signbit(expected)


def test_case_zero_tie_keeps_x1_sign() -> None:
    # |0.0| == |-0.0| is a tie; X1 (+0.0) wins, as in pdi.f.
    result = _palmer_pdi._case(np.array([0.0]), np.array([0.0]), np.array([-0.0]), np.array([0.0]))

    assert not np.signbit(result[0])


# --- records that end before December (#1663) ---------------------------------------------------

_FIXTURE_PALMER = Path(__file__).parent / "fixture" / "palmer"
_PDI_FIXTURE_FACTORS = _palmer_pdi.PdiDurationFactors.from_fitted(0.3, 2.7, 0.3, 2.7)
# 1530 months, January 1895 through June 2022: the record shape REFIMPL-46 measured against pdi.f
_JUNE_ENDING_MONTHS = 1530
# 1521 months, January 1895 through September 2021: a second partial final year
_SEPTEMBER_ENDING_MONTHS = 1521


def _division_z(months: int) -> np.ndarray:
    """Load divisions 1102 and 1407, which reproduce the June and September errors."""
    return np.stack(
        [np.load(_FIXTURE_PALMER / division / "zindex.npy")[:months] for division in ("1102", "1407")], axis=1
    )


def _nan_padded(z: np.ndarray) -> np.ndarray:
    """Pad (months, n_cells) with NaN to whole years, as ``_reshape_palmer_input`` does."""
    months, n_cells = z.shape
    padded = np.full((-(-months // 12) * 12, n_cells), np.nan)
    padded[:months] = z
    return padded.reshape(-1, 12, n_cells)


def _zero_prefixed_reference(z: np.ndarray, factors: _palmer_pdi.PdiDurationFactors) -> _palmer_pdi.PdiResult:
    """The recursion over ``z`` with the record's start padded by Z = 0 until it ends in December.

    A Z = 0 month leaves x1, x2, x3, pro, and v at zero. Leading zeros therefore
    preserve the initial state but let the run end in December without NaN padding.
    The months after the prefix are an internal regression reference, not a new
    comparison with ``pdi.f``. See ADR-0030 for the deliberate differences.
    """
    months, n_cells = z.shape
    prefix = -months % 12
    prefixed = np.concatenate([np.zeros((prefix, n_cells)), z]).reshape(-1, 12, n_cells)
    result = _palmer_pdi.calculate(prefixed, factors)
    return _palmer_pdi.PdiResult(
        pdsi=result.pdsi.reshape(-1, n_cells)[prefix:],
        phdi=result.phdi.reshape(-1, n_cells)[prefix:],
        pmdi=result.pmdi.reshape(-1, n_cells)[prefix:],
    )


@pytest.mark.parametrize(
    "months", [_JUNE_ENDING_MONTHS, _SEPTEMBER_ENDING_MONTHS], ids=["june-ending", "september-ending"]
)
def test_pdi_recursion_ends_at_the_last_real_month(months: int) -> None:
    """A record ending before December equals the same record run through a December end.

    The padded months must not drive the recursion: on the 344 fixture divisions
    they changed PDSI and PHDI by up to 0.67 on 14 divisions, while PMDI was
    unaffected (#1663). The reference is the zero-prefixed run rather than the full
    record truncated at the end month, because truncating a longer run keeps the
    longer record's pending-spell state at the cut. Both sides run the same
    operation sequence on the same values, so the comparison is exact.
    """
    z = _division_z(months)
    padded = _nan_padded(z)

    result = _palmer_pdi.calculate(padded, _PDI_FIXTURE_FACTORS, n_months=months)
    reference = _zero_prefixed_reference(z, _PDI_FIXTURE_FACTORS)

    n_cells = z.shape[1]
    for name in ("pdsi", "phdi", "pmdi"):
        actual = getattr(result, name).reshape(-1, n_cells)[:months]
        assert np.isfinite(actual).all(), name
        np.testing.assert_array_equal(actual, getattr(reference, name), err_msg=name)
        # the padding is never filled in: the caller trims it
        assert np.isnan(getattr(result, name).reshape(-1, n_cells)[months:]).all(), name


def test_pdi_recursion_default_runs_every_month_like_the_explicit_full_length() -> None:
    """The default and the explicit full length agree, and are finite for a December-ending record.

    That December-ending output is unchanged by the fix is pinned against the committed
    fixtures by ``test_palmer.py`` (the ``validation`` marker), not here.
    """
    rng = np.random.default_rng(1663)
    z = rng.normal(0.0, 1.5, size=(2, 12, 3))

    default = _palmer_pdi.calculate(z, _PDI_FIXTURE_FACTORS)
    explicit = _palmer_pdi.calculate(z, _PDI_FIXTURE_FACTORS, n_months=z.shape[0] * 12)

    for name in ("pdsi", "phdi", "pmdi"):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name), err_msg=name)
        assert np.isfinite(getattr(default, name)).all(), name


@pytest.mark.parametrize("months", [1, 5, 11, 12, 13, 23])
def test_pdi_recursion_short_records_end_at_the_last_real_month(months: int) -> None:
    """Records of one to two years: the flush reads the last real month, never a padded one."""
    rng = np.random.default_rng(1663)
    z = rng.normal(0.0, 1.5, size=(months, 3))

    result = _palmer_pdi.calculate(_nan_padded(z), _PDI_FIXTURE_FACTORS, n_months=months)
    reference = _zero_prefixed_reference(z, _PDI_FIXTURE_FACTORS)

    for name in ("pdsi", "phdi", "pmdi"):
        actual = getattr(result, name).reshape(-1, 3)[:months]
        assert np.isfinite(actual).all(), name
        np.testing.assert_array_equal(actual, getattr(reference, name), err_msg=name)


def test_pdi_recursion_stops_at_n_months_even_when_the_tail_is_not_nan() -> None:
    """``n_months`` ends the record even when the months after it are finite numbers.

    The caller pads with NaN, but the bound must be the month count rather than a
    scan for a trailing NaN, so a finite tail is never run.
    """
    z = np.r_[np.full(18, -2.0), np.full(6, 1.5)].reshape(2, 12, 1)

    result = _palmer_pdi.calculate(z, _PDI_FIXTURE_FACTORS, n_months=18)
    reference = _zero_prefixed_reference(z.reshape(-1, 1)[:18], _PDI_FIXTURE_FACTORS)

    for name in ("pdsi", "phdi", "pmdi"):
        actual = getattr(result, name).reshape(-1)[:18]
        np.testing.assert_array_equal(actual, getattr(reference, name)[:, 0], err_msg=name)
        assert np.isnan(getattr(result, name).reshape(-1)[18:]).all(), name


def test_pdi_recursion_zero_months_runs_nothing() -> None:
    """No real month: every output stays NaN and nothing is flushed."""
    result = _palmer_pdi.calculate(np.zeros((2, 12, 2)), _PDI_FIXTURE_FACTORS, n_months=0)

    for name in ("pdsi", "phdi", "pmdi"):
        assert np.isnan(getattr(result, name)).all(), name


@pytest.mark.parametrize("n_months", [-1, 25])
def test_pdi_recursion_rejects_an_n_months_outside_the_record(n_months: int) -> None:
    with pytest.raises(ValueError, match="n_months"):
        _palmer_pdi.calculate(np.zeros((2, 12, 1)), _PDI_FIXTURE_FACTORS, n_months=n_months)


@pytest.mark.parametrize("n_months", [5.0, 36.0, "5", True, False, np.True_])
def test_pdi_recursion_rejects_a_non_integer_n_months(n_months: object) -> None:
    with pytest.raises(TypeError):
        _palmer_pdi.calculate(np.zeros((3, 12, 1)), _PDI_FIXTURE_FACTORS, n_months=n_months)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        palmer._pdi_recursion(np.zeros((3, 12, 1)), _PDI_FIXTURE_FACTORS, n_months=n_months)  # type: ignore[arg-type]


def test_pdsi_on_a_june_ending_record_matches_the_zero_prefixed_reference() -> None:
    """End to end: ``pdsi()`` on a record the NaN padding would change.

    Division 1102 is one of the 14 of the 344 fixture divisions where the padded run
    differed from the reference, by 0.6706 in PDSI and PHDI. Division 0101 is not
    affected after #1666, so it would not catch a regression.
    """
    division = "1102"
    folder = _FIXTURE_PALMER / division
    precips = np.load(folder / "precips.npy")[:_JUNE_ENDING_MONTHS]
    pet = np.load(folder / "pet.npy")[:_JUNE_ENDING_MONTHS]
    with (Path(__file__).parent / "fixture" / "palmer_awc.json").open() as handle:
        awc = json.load(handle)[division]

    pdsi, phdi, pmdi, z, params = palmer.pdsi(precips, pet, awc, 1895, 1931, 1990)
    assert params is not None

    factors = _palmer_pdi.PdiDurationFactors.from_fitted(params["wetm"], params["wetb"], params["drym"], params["dryb"])
    reference = _zero_prefixed_reference(z.reshape(-1, 1), factors)
    for name, actual in (("pdsi", pdsi), ("phdi", phdi), ("pmdi", pmdi)):
        assert actual.shape == (_JUNE_ENDING_MONTHS,), name
        assert np.isfinite(actual).all(), name
        # the run may take the native kernel; its parity contract is 1e-10, and the
        # reference is exact in pure Python, so this is not a loosened regression bound
        np.testing.assert_allclose(actual, getattr(reference, name)[:, 0], rtol=0, atol=1e-10, err_msg=name)
