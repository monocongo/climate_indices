import numpy as np
import pytest

from climate_indices import palmer
from climate_indices._palmer_duration import DurationFactors


def _blank_state() -> tuple[palmer._PalmerPrepared, palmer._PalmerRecursion]:
    """A minimal, structurally-valid prepared struct and recursion state for
    exercising the recursion functions directly, without needing realistic
    precip/PET content."""
    prepared = palmer._initialize_prepared(
        precips=np.zeros(12),
        pet=np.zeros(12),
        awc=1.0,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2000,
    )
    return prepared, palmer._initialize_recursion(prepared)


# every recursion state field carries an internal n_cells == 1 cell axis (see
# ADR-0011); this is the "every cell" mask for a single-cell _blank_state()
_ACTIVE = np.array([True])


def test_initialize_prepared_sets_default_duration_factors():
    prepared, _ = _blank_state()

    # standard Palmer PDSI has no wet/dry distinction in its duration factors
    assert prepared.wetm == pytest.approx(prepared.drym)
    assert prepared.wetb == pytest.approx(prepared.dryb)

    # the implied duration-factor weighting fraction c = b / (m + b) must reproduce
    # Palmer's published constant (0.897), regardless of how m/b are derived
    c = prepared.wetb / (prepared.wetm + prepared.wetb)
    assert c == pytest.approx(0.897)

    # m + b must reproduce Palmer's published 1/q = 3.0
    assert (prepared.wetm + prepared.wetb) == pytest.approx(3.0)


def test_duration_factor_c_rejects_zero_factor_sum():
    with pytest.raises(ValueError, match="must not sum to zero"):
        DurationFactors.weighting_fraction(1.0, -1.0)


def test_weighting_fraction_keeps_the_division_form():
    """The pdi.f lineage divides, so its coefficient is ``b / (m + b)`` exactly.

    These constants make the division form differ bitwise from the complement
    form, so the test fails if the implementation switches forms.
    """
    m, b = 0.1, 0.2

    assert DurationFactors.weighting_fraction(m, b) == b / (m + b)
    assert DurationFactors.weighting_fraction(m, b) != 1.0 - m / (m + b)


def test_select_duration_factors_uses_wet_factors_when_x3_is_zero():
    prepared, state = _blank_state()
    prepared.wetm, prepared.wetb = 1.0, 2.0
    prepared.drym, prepared.dryb = 3.0, 4.0
    state.x3 = 0.0

    assert palmer._select_duration_factors(prepared, state) == (1.0, 2.0)


def test_statement_180_ze_uses_custom_dry_duration_factors():
    prepared, state = _blank_state()
    prepared.drym, prepared.dryb = 1.0, 2.0  # non-default, to prove they're used

    state.year, state.month = 0, 0
    state.x3 = -2.0  # an established drought
    state.v = 0.0
    # z chosen so that pv = (z + 0.15) + max(v, 0) > 0, falling into the
    # branch that actually computes ze (rather than short-circuiting to
    # _statement_210 for a fizzled abatement)
    state.z[0, 0] = 0.5

    # Calculate expected value before calling _statement_180 (which may modify x3)
    m, b = prepared.drym, prepared.dryb
    x3_original = state.x3
    expected_ze = -b * x3_original - 0.5 * (m + b)

    palmer._statement_180(prepared, state, _ACTIVE)

    assert state.ze == pytest.approx(expected_ze)


def test_statement_170_ze_uses_custom_wet_duration_factors():
    prepared, state = _blank_state()
    prepared.wetm, prepared.wetb = 3.0, 5.0  # non-default, to prove they're used

    state.year, state.month = 0, 0
    state.x3 = 2.0  # an established wet spell
    state.v = 0.0
    # z chosen so that pv = (z - 0.15) + min(v, 0) < 0, falling into the
    # branch that actually computes ze
    state.z[0, 0] = -0.5

    # Calculate expected value before calling _statement_170 (which may modify x3)
    m, b = prepared.wetm, prepared.wetb
    x3_original = state.x3
    expected_ze = -b * x3_original + 0.5 * (m + b)

    palmer._statement_170(prepared, state, _ACTIVE)

    assert state.ze == pytest.approx(expected_ze)


def test_statement_210_px3_selects_dry_factors_when_x3_negative():
    prepared, state = _blank_state()
    prepared.drym, prepared.dryb = 1.0, 4.0
    prepared.wetm, prepared.wetb = 99.0, 99.0  # deliberately different, must NOT be used

    state.year, state.month = 0, 0
    state.x3 = -1.5  # established drought -> dry factors expected
    state.z[0, 0] = 2.0

    # Calculate expected value before calling _statement_210, which
    # unconditionally calls _statement_220 and overwrites state.x3
    # with the freshly computed px3.
    m, b = prepared.drym, prepared.dryb
    c = DurationFactors.weighting_fraction(m, b)
    expected_px3 = c * state.x3 + state.z[0, 0] / (m + b)

    palmer._statement_210(prepared, state, _ACTIVE)

    assert state.px3[0, 0] == pytest.approx(expected_px3)


def test_statement_190_px3_selects_wet_factors_when_x3_positive():
    prepared, state = _blank_state()
    prepared.wetm, prepared.wetb = 2.0, 6.0
    prepared.drym, prepared.dryb = 99.0, 99.0  # deliberately different, must NOT be used

    state.year, state.month = 0, 0
    state.x3 = 2.0  # established wet spell -> wet factors expected
    state.pro = 0.0  # not 100, so q = ze + v
    state.ze = 10.0
    state.v = 0.0
    state.pv = 1.0  # ppr = (pv / q) * 100 = 10 < 100, falls into the px3 branch
    state.z[0, 0] = 4.0

    # Calculate expected value before calling _statement_190, which
    # unconditionally calls _statement_200 -> (on this code path)
    # _statement_220, overwriting state.x3 with the freshly computed px3.
    m, b = prepared.wetm, prepared.wetb
    c = DurationFactors.weighting_fraction(m, b)
    expected_px3 = c * state.x3 + state.z[0, 0] / (m + b)

    palmer._statement_190(prepared, state, _ACTIVE)

    assert state.px3[0, 0] == pytest.approx(expected_px3)


def test_statement_200_px1_always_uses_wet_factors_px2_always_dry():
    prepared, state = _blank_state()
    prepared.wetm, prepared.wetb = 1.0, 3.0
    prepared.drym, prepared.dryb = 2.0, 2.0

    state.year, state.month = 0, 0
    state.x1, state.x2 = 1.0, -1.0
    state.z[0, 0] = 0.4
    # a nonzero px3 prevents the early-return "new spell begins" branches,
    # so both px1 and px2 get computed and asserted on
    state.px3[0, 0] = 5.0
    # this code path falls through to the final bookkeeping section, which
    # unconditionally calls _statement_220 (needs pv/ppr present)
    state.pv = 0.0
    state.ppr[0, 0] = 0.0

    # Calculate expected values before calling _statement_200, which (on
    # this code path) unconditionally calls _statement_220, overwriting
    # state.x1 and state.x2 with the freshly computed px1/px2.
    x1_original, x2_original = state.x1, state.x2
    wetm, wetb = prepared.wetm, prepared.wetb
    drym, dryb = prepared.drym, prepared.dryb
    z = state.z[0, 0]

    c_wet = DurationFactors.weighting_fraction(wetm, wetb)
    expected_px1 = max(0.0, c_wet * x1_original + z / (wetm + wetb))

    c_dry = DurationFactors.weighting_fraction(drym, dryb)
    expected_px2 = min(0.0, c_dry * x2_original + z / (drym + dryb))

    palmer._statement_200(prepared, state, _ACTIVE)

    assert state.px1[0, 0] == pytest.approx(expected_px1)
    assert state.px2[0, 0] == pytest.approx(expected_px2)


def test_calc_cafec_zindex_writes_the_zindex():
    """The shared CAFEC/Z-index step serves both the PDSI and scPDSI recursions.

    The CAFEC value and the Z-index are recomputed with the pre-refactor
    named-intermediate grouping and compared exactly: the recursion branches on
    exact comparisons downstream, so a 1-ulp reassociation is a behavior change.
    The constants are chosen so that common reassociations (swapping the CAFEC
    terms, distributing ``ak`` over the departure) change the last bit.
    """
    prepared, state = _blank_state()
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

    cafec = palmer._calc_cafec_zindex(prepared, state, 0, 0)

    cet = prepared.alpha[0] * prepared.pet[0, 0]
    cr = prepared.beta[0] * prepared.prdat[0, 0]
    cro = prepared.gamma[0] * prepared.spdat[0, 0]
    cl = prepared.delta[0] * prepared.pldat[0, 0]
    expected_cafec = cet + cr + cro - cl
    assert cafec == expected_cafec
    assert state.z[0, 0] == prepared.ak[0] * (prepared.precips[0, 0] - expected_cafec)


def _run_zindex_pipeline(
    precips: np.ndarray, pet: np.ndarray, awc: float
) -> tuple[palmer._PalmerPrepared, palmer._PalmerRecursion]:
    """Runs the same internal pipeline palmer.pdsi() runs, up through
    _calc_kfactors (but not yet _calc_zindex), and returns the prepared inputs
    and recursion state. Exists so this test can inject custom duration factors
    between initialization and the recursion, without touching palmer.pdsi()'s
    public signature."""
    prepared = palmer._initialize_prepared(
        precips=precips,
        pet=pet,
        awc=awc,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2003,
    )
    palmer._calc_water_balances(prepared)
    palmer._calc_cafec_coefficients(prepared)
    palmer._calc_zindex_factors(prepared)
    palmer._calc_kfactors(prepared)
    return prepared, palmer._initialize_recursion(prepared)


def test_custom_duration_factors_change_pdsi_output():
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)

    prepared_default, state_default = _run_zindex_pipeline(precips, pet, awc=5.0)
    palmer._calc_zindex(prepared_default, state_default)
    palmer._finish_up(state_default)

    prepared_custom, state_custom = _run_zindex_pipeline(precips, pet, awc=5.0)
    prepared_custom.wetm, prepared_custom.wetb = 1.0, 1.0
    prepared_custom.drym, prepared_custom.dryb = 1.0, 1.0
    palmer._calc_zindex(prepared_custom, state_custom)
    palmer._finish_up(state_custom)

    assert not np.allclose(state_default.pdsi, state_custom.pdsi, equal_nan=True)


def test_cafec_ratio_substitutes_exact_zero_and_leaves_a_zero_denominator():
    """A month with no accumulated water-balance term takes ``both_zero``.

    A zero denominator with a nonzero numerator is not undefined the same way:
    the reference leaves 0.0 there. Both cases must stay exact, since a
    tolerance would fold genuinely small sums into them.
    """
    numerator = np.array([0.0, 2.0, 0.0])
    denominator = np.array([0.0, 0.0, 4.0])

    np.testing.assert_array_equal(
        palmer._calc_cafec_ratio(numerator, denominator, both_zero=1.0),
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        palmer._calc_cafec_ratio(numerator, denominator, both_zero=0.0),
        np.array([0.0, 0.0, 0.0]),
    )


def test_case_selects_near_normal_when_no_spell_is_established():
    """x3 is exactly 0.0 when no spell is established, and that exact zero --
    not a tolerance -- picks the larger-magnitude incipient index."""
    prob = np.array([50.0])
    x1 = np.array([1.5])
    x2 = np.array([-1.0])

    assert palmer._case(prob, x1, x2, np.array([0.0]))[0] == 1.5
    # an established spell (x3 != 0) reports the interpolated severity instead
    assert palmer._case(prob, x1, x2, np.array([-2.0]))[0] == -0.25


def test_record_index_values_falls_back_to_pdsi_when_no_spell_is_established():
    """PHDI has no severity of its own without an established spell (px3
    exactly 0.0), so it records the PDSI value; with a spell it keeps px3."""
    _, state = _blank_state()
    state.px3[0, 0, 0] = 0.0
    state.px3[0, 1, 0] = -2.5
    values = np.array([3.0, 4.0])

    palmer._record_index_values(state, np.array([0, 0]), np.array([0, 1]), values, np.array([0]))

    assert state.pdsi[0, 0, 0] == 3.0
    assert state.phdi[0, 0, 0] == 3.0  # no spell: the recorded PDSI value
    assert state.phdi[0, 1, 0] == -2.5  # established spell: its own severity
