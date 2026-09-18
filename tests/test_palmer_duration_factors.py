import types

import numpy as np
import pytest

from climate_indices import palmer
from climate_indices._palmer_duration import DurationFactors
from climate_indices.exceptions import ConvergenceError


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


def test_custom_duration_factors_change_pdsi_output():
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


def test_duration_factor_override_distinguishes_wet_from_dry():
    """Swapping the wet and dry pairs changes the recursion; wet and dry are not interchangeable."""
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    wet_heavy = {"wetm": 3.0, "wetb": 1.0, "drym": 1.0, "dryb": 1.0}
    dry_heavy = {"wetm": 1.0, "wetb": 1.0, "drym": 3.0, "dryb": 1.0}

    wet_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=wet_heavy)
    dry_pdsi, *_ = palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=dry_heavy)

    assert not np.allclose(wet_pdsi, dry_pdsi, equal_nan=True)


def test_pdsi_returned_params_reproduce_a_duration_factor_override():
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
def test_invalid_duration_factor_override_is_rejected(override, message):
    """A malformed override is a caller error, not a silent default.

    A masked factor is missing data, not a number: ``np.asarray`` would drop the
    mask and hand the backing value to the recursion.
    """
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)

    with pytest.raises(ValueError, match=message):
        palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=override)


def test_palmer_default_factors_as_override_reproduce_the_default_run():
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


def test_duration_factor_override_does_not_preempt_input_validation():
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


def test_duration_factor_override_is_not_validated_for_all_missing_input():
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


def test_cafec_ratio_substitutes_exact_zero_and_leaves_a_zero_denominator():
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


def test_case_selects_near_normal_when_no_spell_is_established():
    """x3 is exactly 0.0 when no spell is established, and that exact zero --
    not a tolerance -- picks the larger-magnitude incipient index."""
    prob = np.array([50.0])
    x1 = np.array([1.5])
    x2 = np.array([-1.0])

    assert palmer._case(prob, x1, x2, np.array([0.0]))[0] == 1.5
    # an established spell (x3 != 0) reports the interpolated severity instead
    assert palmer._case(prob, x1, x2, np.array([-2.0]))[0] == -0.25
    # a sub-epsilon x3 is still an established spell under the exact test; a
    # tolerance would classify it as zero and return the near-normal 1.5
    assert palmer._case(prob, x1, x2, np.array([-np.finfo(float).tiny]))[0] == 0.75


def test_record_index_values_falls_back_to_pdsi_when_no_spell_is_established():
    """PHDI has no severity of its own without an established spell (px3
    exactly 0.0), so it records the PDSI value; with a spell it keeps px3."""
    _, state = _blank_state()
    state.px3[0, 0, 0] = 0.0
    state.px3[0, 1, 0] = -2.5
    state.px3[0, 2, 0] = -np.finfo(float).tiny
    values = np.array([3.0, 4.0, 5.0])

    palmer._record_index_values(state, np.zeros(3, dtype=int), np.arange(3), values, np.array([0]))

    assert state.pdsi[0, 0, 0] == 3.0
    assert state.phdi[0, 0, 0] == 3.0  # no spell: the recorded PDSI value
    assert state.phdi[0, 1, 0] == -2.5  # established spell: its own severity
    assert state.phdi[0, 2, 0] == -np.finfo(float).tiny  # sub-epsilon, still a spell


def test_finish_up_falls_back_to_pdsi_when_no_spell_is_established():
    """_finish_up repeats the no-established-spell fallback for the months left
    pending when the record ends, under the same exact-zero test."""
    _, state = _blank_state()
    state.k8max = np.array([2])
    state.indexj[0, 0], state.indexm[0, 0] = 0, 0
    state.indexj[1, 0], state.indexm[1, 0] = 0, 1
    state.x[0, 0, 0] = 3.0
    state.px3[0, 0, 0] = 0.0
    state.x[0, 1, 0] = 4.0
    state.px3[0, 1, 0] = -np.finfo(float).tiny

    palmer._finish_up(state)

    assert state.pdsi[0, 0, 0] == 3.0
    assert state.phdi[0, 0, 0] == 3.0  # no spell: the PDSI value
    assert state.pdsi[0, 1, 0] == 4.0
    # a sub-epsilon px3 is still an established spell, so PHDI keeps it rather
    # than falling back to the PDSI value
    assert state.phdi[0, 1, 0] == -np.finfo(float).tiny


def test_non_contracting_duration_factor_override_is_attributed_to_pdsi():
    """The override's ConvergenceError names the pdsi path, not the scPDSI calibration."""
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    non_contracting = {"wetm": 1.0, "wetb": -0.5, "drym": 1.0, "dryb": 1.0}

    with pytest.raises(ConvergenceError, match="duration-factor override") as error:
        palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=non_contracting)

    assert error.value.algorithm == "PDSI duration-factor override"
