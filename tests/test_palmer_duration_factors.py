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

    palmer._statement_190(data)

    m, b = data["wetm"], data["wetb"]
    c = palmer._duration_factor_c(m, b)
    expected_px3 = c * data["x3"] + data["z"][0, 0] / (m + b)
    assert data["px3"][0, 0] == pytest.approx(expected_px3)
