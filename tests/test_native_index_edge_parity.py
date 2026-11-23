"""End-to-end SPI, SPEI and EDDI backend parity on synthetic monthly edge cases."""

import numpy as np
import pytest

from climate_indices import compute
from climate_indices.drought import indices
from tests import conftest


@pytest.mark.parametrize(
    ("index", "case"),
    [
        (index, case)
        for index in ("spi", "spei", "eddi")
        for case in ("near_constant", "sparse", "four_years", "one_year")
    ]
    + [("spi", "near_constant_pearson"), ("spei", "near_constant_pearson")],
)
def test_monthly_index_edge_parity(monkeypatch: pytest.MonkeyPatch, index: str, case: str) -> None:
    years = {"near_constant": 8, "near_constant_pearson": 8, "sparse": 12, "four_years": 4, "one_year": 1}[case]
    rain = np.random.default_rng(582).gamma(2, 40, years * 12)
    pet = 35 + 0.1 * rain
    if case.startswith("near_constant"):
        rain = 40 + 1e-5 * rain
        pet = 35 + 0.1 * rain
    elif case == "sparse":
        rain[4 * 12 :] = np.nan
        pet[4 * 12 :] = np.nan
    window = (1981, 1981, 1980 + years)
    distribution = indices.Distribution.pearson if case == "near_constant_pearson" else indices.Distribution.gamma
    monthly = compute.Periodicity.monthly

    def run() -> np.ndarray:
        if index == "spi":
            return indices.spi(rain, 1, distribution, *window, monthly)
        if index == "spei":
            return indices.spei(rain, pet, 1, distribution, monthly, *window)
        return indices.eddi(pet, 1, *window, monthly)

    rust, python, calls = conftest.rust_and_python(monkeypatch, compute, run)
    if index == "eddi":
        assert {"tukey_probabilities", "hastings_inverse_normal"} <= calls
    elif distribution is indices.Distribution.pearson:
        assert {"pearson_parameters", "pearson_cdf"} <= calls
        assert "gamma_parameters" not in calls
    else:
        assert {"gamma_parameters", "gamma_probabilities", "norm_ppf"} <= calls
    conftest.assert_native_parity(rust, python)
    if case == "one_year":
        assert np.isnan(rust).all()
    else:
        assert np.isfinite(rust[:12]).all()
        if case == "sparse":
            assert np.isnan(rust[4 * 12 :]).all()
