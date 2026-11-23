"""Property checks for I_F and API on synthetic precipitation.

These verify algebraic and behavioral properties of the implementation on
seeded synthetic data. They are **not** validation against observations or an
external oracle, so they are deliberately outside the ``validation`` marker.
"""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import flood

_YEARS, _START = 40, 1980
_CAL_INITIAL, _CAL_FINAL = 1981, 2010  # year 1980 is PE warm-up, so excluded


def _rain(seed: int = 0, years: int = _YEARS) -> np.ndarray:
    """Wet-day gamma rainfall in the library's 366-positions-per-year layout."""
    rng = np.random.default_rng(seed)
    wet = rng.random(years * 366) < 0.3
    return np.where(wet, rng.gamma(0.8, 8.0, years * 366), 0.0)


def _flood_index(precipitation: np.ndarray) -> np.ndarray:
    pe = flood.effective_precipitation(precipitation)
    return flood.flood_index(pe, _START, _CAL_INITIAL, _CAL_FINAL, year_start_month=1)


def test_flood_index_annual_maxima_are_standardized_over_calibration() -> None:
    """I_F is affine in PE, so calibration annual maxima have mean 0 and ddof=0 SD 1.

    The index standardizes *annual maxima*, not daily values; a daily mean of 0 and
    SD of 1 would be wrong by construction. The identity is exact up to rounding,
    hence ``atol=1e-12`` on values of order one.
    """
    index = _flood_index(_rain()).reshape(_YEARS, 366)
    maxima = np.nanmax(index[_CAL_INITIAL - _START : _CAL_FINAL - _START + 1], axis=1)
    assert maxima.mean() == pytest.approx(0.0, abs=1e-12)
    assert maxima.std() == pytest.approx(1.0, abs=1e-12)


def test_flood_index_rises_from_a_precipitation_step_and_not_before() -> None:
    """A step added after the calibration window raises I_F from that day on.

    PE weights are all positive and the calibration maxima are untouched, so earlier
    days are identical (lag 0) and later days are strictly higher.
    """
    base = _rain()
    step = 35 * 366 + 100
    stepped = base.copy()
    stepped[step:] += 5.0
    before, after = _flood_index(base), _flood_index(stepped)
    np.testing.assert_array_equal(after[:step], before[:step])
    assert np.all(after[step:] > before[step:])


def test_flood_index_is_finite_after_spin_up() -> None:
    """Only the 364-day PE warm-up is NaN; nothing else is NaN or infinite."""
    index = _flood_index(_rain())
    assert np.isnan(index[:364]).all()
    assert np.isfinite(index[364:]).all()


def test_flood_index_of_constant_precipitation_is_nan_never_inf() -> None:
    """Zero annual-maximum variance gives NaN by contract (``flood_index`` docstring)."""
    index = _flood_index(np.full(_YEARS * 366, 3.0))
    assert not np.isinf(index).any()
    assert np.isnan(index).all()


@pytest.mark.parametrize("k", [0.3, 0.85, 0.95])
def test_api_single_pulse_decays_geometrically_at_k(k: float) -> None:
    """API of one pulse is ``P * k**t`` (Kohler & Linsley recurrence); ``rtol=1e-14`` allows
    a few ulps of accumulated multiplication."""
    pulse = np.zeros(60)
    pulse[0] = 12.0
    result = flood.antecedent_precipitation_index(pulse, k)
    np.testing.assert_allclose(result, 12.0 * k ** np.arange(60), rtol=1e-14)
    assert np.all(np.diff(result) < 0)


def test_api_is_linear() -> None:
    """The recurrence is linear, so API(a + b) = API(a) + API(b); ``rtol=1e-12`` covers
    reordering of floating-point sums over 2000 steps."""
    rng = np.random.default_rng(1)
    a, b = rng.gamma(0.8, 8.0, 2000), rng.gamma(0.8, 8.0, 2000)
    api = flood.antecedent_precipitation_index
    np.testing.assert_allclose(api(a + b, 0.9), api(a, 0.9) + api(b, 0.9), rtol=1e-12)


def test_api_is_nonnegative_and_bounded_by_the_constant_input_limit() -> None:
    """With ``0 <= P <= max(P)``, API stays in ``[0, max(P) / (1 - k)]``."""
    precipitation = _rain()
    result = flood.antecedent_precipitation_index(precipitation, 0.95)
    assert result.min() >= 0
    assert result.max() <= precipitation.max() / (1 - 0.95)
