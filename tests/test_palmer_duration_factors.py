import numpy as np
import pytest

from climate_indices import palmer


def _blank_data() -> dict:
    """A minimal, structurally-valid `data` dict for exercising the recursion
    functions directly, without needing realistic precip/PET content."""
    return palmer._initialize_data(
        precips=np.zeros(12),
        pet=np.zeros(12),
        awc=1.0,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2000,
    )


def test_initialize_data_sets_default_duration_factors():
    data = _blank_data()

    # standard Palmer PDSI has no wet/dry distinction in its duration factors
    assert data["wetm"] == pytest.approx(data["drym"])
    assert data["wetb"] == pytest.approx(data["dryb"])

    # the implied CAFEC-weighting fraction c = b / (m + b) must reproduce
    # Palmer's published constant (0.897), regardless of how m/b are derived
    c = data["wetb"] / (data["wetm"] + data["wetb"])
    assert c == pytest.approx(0.897)

    # m + b must reproduce Palmer's published 1/q = 3.0
    assert (data["wetm"] + data["wetb"]) == pytest.approx(3.0)


def test_duration_factor_c_rejects_zero_factor_sum():
    with pytest.raises(ValueError, match="must not sum to zero"):
        palmer._duration_factor_c(1.0, -1.0)


def test_select_duration_factors_uses_wet_factors_when_x3_is_zero():
    data = _blank_data()
    data["wetm"], data["wetb"] = 1.0, 2.0
    data["drym"], data["dryb"] = 3.0, 4.0
    data["x3"] = 0.0

    assert palmer._select_duration_factors(data) == (1.0, 2.0)


def test_statement_180_ze_uses_custom_dry_duration_factors():
    data = _blank_data()
    data["drym"], data["dryb"] = 1.0, 2.0  # non-default, to prove they're used

    data["year"], data["month"] = 0, 0
    data["x3"] = -2.0  # an established drought
    data["v"] = 0.0
    # z chosen so that pv = (z + 0.15) + max(v, 0) > 0, falling into the
    # branch that actually computes ze (rather than short-circuiting to
    # _statement_210 for a fizzled abatement)
    data["z"][0, 0] = 0.5

    # Calculate expected value before calling _statement_180 (which may modify x3)
    m, b = data["drym"], data["dryb"]
    x3_original = data["x3"]
    expected_ze = -b * x3_original - 0.5 * (m + b)

    palmer._statement_180(data)

    assert data["ze"] == pytest.approx(expected_ze)


def test_statement_170_ze_uses_custom_wet_duration_factors():
    data = _blank_data()
    data["wetm"], data["wetb"] = 3.0, 5.0  # non-default, to prove they're used

    data["year"], data["month"] = 0, 0
    data["x3"] = 2.0  # an established wet spell
    data["v"] = 0.0
    # z chosen so that pv = (z - 0.15) + min(v, 0) < 0, falling into the
    # branch that actually computes ze
    data["z"][0, 0] = -0.5

    # Calculate expected value before calling _statement_170 (which may modify x3)
    m, b = data["wetm"], data["wetb"]
    x3_original = data["x3"]
    expected_ze = -b * x3_original + 0.5 * (m + b)

    palmer._statement_170(data)

    assert data["ze"] == pytest.approx(expected_ze)


def test_statement_210_px3_selects_dry_factors_when_x3_negative():
    data = _blank_data()
    data["drym"], data["dryb"] = 1.0, 4.0
    data["wetm"], data["wetb"] = 99.0, 99.0  # deliberately different, must NOT be used

    data["year"], data["month"] = 0, 0
    data["x3"] = -1.5  # established drought -> dry factors expected
    data["z"][0, 0] = 2.0

    # Calculate expected value before calling _statement_210, which
    # unconditionally calls _statement_220 and overwrites data["x3"]
    # with the freshly computed px3.
    m, b = data["drym"], data["dryb"]
    c = palmer._duration_factor_c(m, b)
    expected_px3 = c * data["x3"] + data["z"][0, 0] / (m + b)

    palmer._statement_210(data)

    assert data["px3"][0, 0] == pytest.approx(expected_px3)


def test_statement_190_px3_selects_wet_factors_when_x3_positive():
    data = _blank_data()
    data["wetm"], data["wetb"] = 2.0, 6.0
    data["drym"], data["dryb"] = 99.0, 99.0  # deliberately different, must NOT be used

    data["year"], data["month"] = 0, 0
    data["x3"] = 2.0  # established wet spell -> wet factors expected
    data["pro"] = 0.0  # not 100, so q = ze + v
    data["ze"] = 10.0
    data["v"] = 0.0
    data["pv"] = 1.0  # ppr = (pv / q) * 100 = 10 < 100, falls into the px3 branch
    data["z"][0, 0] = 4.0

    # Calculate expected value before calling _statement_190, which
    # unconditionally calls _statement_200 -> (on this code path)
    # _statement_220, overwriting data["x3"] with the freshly computed px3.
    m, b = data["wetm"], data["wetb"]
    c = palmer._duration_factor_c(m, b)
    expected_px3 = c * data["x3"] + data["z"][0, 0] / (m + b)

    palmer._statement_190(data)

    assert data["px3"][0, 0] == pytest.approx(expected_px3)


def test_statement_200_px1_always_uses_wet_factors_px2_always_dry():
    data = _blank_data()
    data["wetm"], data["wetb"] = 1.0, 3.0
    data["drym"], data["dryb"] = 2.0, 2.0

    data["year"], data["month"] = 0, 0
    data["x1"], data["x2"] = 1.0, -1.0
    data["z"][0, 0] = 0.4
    # a nonzero px3 prevents the early-return "new spell begins" branches,
    # so both px1 and px2 get computed and asserted on
    data["px3"][0, 0] = 5.0
    # this code path falls through to the final bookkeeping section, which
    # unconditionally calls _statement_220 (needs pv/ppr present)
    data["pv"] = 0.0
    data["ppr"][0, 0] = 0.0

    # Calculate expected values before calling _statement_200, which (on
    # this code path) unconditionally calls _statement_220, overwriting
    # data["x1"] and data["x2"] with the freshly computed px1/px2.
    x1_original, x2_original = data["x1"], data["x2"]
    wetm, wetb = data["wetm"], data["wetb"]
    drym, dryb = data["drym"], data["dryb"]
    z = data["z"][0, 0]

    c_wet = palmer._duration_factor_c(wetm, wetb)
    expected_px1 = max(0.0, c_wet * x1_original + z / (wetm + wetb))

    c_dry = palmer._duration_factor_c(drym, dryb)
    expected_px2 = min(0.0, c_dry * x2_original + z / (drym + dryb))

    palmer._statement_200(data)

    assert data["px1"][0, 0] == pytest.approx(expected_px1)
    assert data["px2"][0, 0] == pytest.approx(expected_px2)


def _run_zindex_pipeline(precips: np.ndarray, pet: np.ndarray, awc: float) -> dict:
    """Runs the same internal pipeline palmer.pdsi() runs, up through
    _calc_kfactors (but not yet _calc_zindex), and returns the data dict.
    Exists so this test can inject custom duration factors between
    initialization and the recursion, without touching palmer.pdsi()'s
    public signature."""
    data = palmer._initialize_data(
        precips=precips,
        pet=pet,
        awc=awc,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2003,
    )
    palmer._calc_water_balances(data)
    palmer._calc_cafec_coefficients(data)
    palmer._calc_zindex_factors(data)
    palmer._calc_kfactors(data)
    return data


def test_custom_duration_factors_change_pdsi_output():
    rng = np.random.default_rng(42)
    precips = rng.uniform(0.0, 6.0, size=12 * 4)
    pet = rng.uniform(0.0, 4.0, size=12 * 4)

    data_default = _run_zindex_pipeline(precips, pet, awc=5.0)
    palmer._calc_zindex(data_default)
    palmer._finish_up(data_default)

    data_custom = _run_zindex_pipeline(precips, pet, awc=5.0)
    data_custom["wetm"], data_custom["wetb"] = 1.0, 1.0
    data_custom["drym"], data_custom["dryb"] = 1.0, 1.0
    palmer._calc_zindex(data_custom)
    palmer._finish_up(data_custom)

    assert not np.allclose(data_default["pdsi"], data_custom["pdsi"], equal_nan=True)
