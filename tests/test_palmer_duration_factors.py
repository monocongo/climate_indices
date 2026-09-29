import types

import numpy as np
import pytest

from climate_indices import _palmer_pdi, palmer
from climate_indices._palmer_duration import DurationFactors
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


def test_initialize_prepared_sets_default_duration_factors():
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


def test_duration_factor_c_rejects_zero_factor_sum():
    with pytest.raises(ValueError, match="must not sum to zero"):
        _palmer_pdi._weighting_fraction(1.0, -1.0)


def test_weighting_fraction_keeps_the_division_form():
    """The pdi.f lineage divides, so its coefficient is ``b / (m + b)`` exactly.

    These constants make the division form differ bitwise from the complement
    form, so the test fails if the implementation switches forms.
    """
    m, b = 0.1, 0.2

    assert _palmer_pdi._weighting_fraction(m, b) == b / (m + b)
    assert _palmer_pdi._weighting_fraction(m, b) != 1.0 - m / (m + b)


def test_pdi_validation_accepts_factors_the_wells_cross_coefficient_would_reject():
    """The pdi.f recurrence never forms the Wells cross coefficient.

    ``c = b / (m + b)`` is -0.667 (wet) and 0.769 (dry), both contractions, so the
    pdi.f recursion can use these factors.  The Wells lineage additionally derives
    ``dryc = 1 - drym / (drym + wetb) = 4.0`` and rejects the same set.
    """
    factors = _palmer_pdi.PdiDurationFactors.from_fitted(1.0, -0.4, 0.3, 1.0)

    assert factors.wetm == 1.0
    with pytest.raises(ConvergenceError):
        DurationFactors.from_fitted(1.0, -0.4, 0.3, 1.0)


def test_pdi_validation_rejects_a_non_contracting_weighting_fraction():
    with pytest.raises(ConvergenceError, match="wetc"):
        _palmer_pdi.PdiDurationFactors.from_fitted(1.0, -0.5, 1.0, 1.0)


def test_pdi_validation_rejects_a_zero_denominator():
    with pytest.raises(ConvergenceError, match="standard PDSI recursion"):
        _palmer_pdi.PdiDurationFactors.from_fitted(1.0, -1.0, 1.0, 1.0)


def test_select_duration_factors_prefers_wet_factors_when_x3_is_zero():
    factors = _palmer_pdi.PdiDurationFactors.from_fitted(1.0, 2.0, 3.0, 4.0)
    state = _state()
    state.x3 = np.array([0.0])

    m, b = _palmer_pdi._select_duration_factors(factors, state)

    np.testing.assert_array_equal(m, [1.0])
    np.testing.assert_array_equal(b, [2.0])


def test_pdi_factors_are_not_interchangeable_between_wet_and_dry_spells():
    """Swapping the wet and dry pairs changes the recursion, through its interface."""
    z = np.array([2.0, 3.0, -1.0, -4.0, 0.5, -0.75, -0.25, 1.5, -2.0, 0.0, 2.5, -3.0]).reshape(1, 12, 1)

    wet_heavy = _palmer_pdi.calculate(z, _palmer_pdi.PdiDurationFactors.from_fitted(3.0, 1.0, 1.0, 1.0))
    dry_heavy = _palmer_pdi.calculate(z, _palmer_pdi.PdiDurationFactors.from_fitted(1.0, 1.0, 3.0, 1.0))

    assert not np.allclose(wet_heavy.pdsi, dry_heavy.pdsi, equal_nan=True)


def test_pdi_recursion_pins_a_z_sequence_through_its_interface():
    """A Z sequence driven through the recursion interface locks its arithmetic.

    The values were captured from the lineage-preserving refactor and are
    re-checked bit-for-bit against the NOAA/nClimDiv fixtures by ``test_palmer``.
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

    np.testing.assert_allclose(result.pdsi.reshape(-1), expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.phdi.reshape(-1), phdi_expected, rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.pmdi.reshape(-1), pmdi_expected, rtol=0, atol=1e-11)


def test_calc_cafec_zindex_writes_the_zindex():
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

    assert _palmer_pdi._case(prob, x1, x2, np.array([0.0]))[0] == 1.5
    # an established spell (x3 != 0) reports the interpolated severity instead
    assert _palmer_pdi._case(prob, x1, x2, np.array([-2.0]))[0] == -0.25
    # a sub-epsilon x3 is still an established spell under the exact test; a
    # tolerance would classify it as zero and return the near-normal 1.5
    assert _palmer_pdi._case(prob, x1, x2, np.array([-np.finfo(float).tiny]))[0] == 0.75


def test_record_index_values_falls_back_to_pdsi_when_no_spell_is_established():
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


def test_finish_up_falls_back_to_pdsi_when_no_spell_is_established():
    """_finish_up repeats the no-established-spell fallback for the months left
    pending when the record ends, under the same exact-zero test."""
    state = _state()
    state.k8max = np.array([2])
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


def test_non_contracting_duration_factor_override_is_attributed_to_pdsi():
    """The override's ConvergenceError names the pdsi path, not the scPDSI calibration."""
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)
    non_contracting = {"wetm": 1.0, "wetb": -0.5, "drym": 1.0, "dryb": 1.0}

    with pytest.raises(ConvergenceError, match="duration-factor override") as error:
        palmer.pdsi(precips, pet, 5.0, 2000, 2000, 2003, fitting_params=non_contracting)

    assert error.value.algorithm == "PDSI duration-factor override"
