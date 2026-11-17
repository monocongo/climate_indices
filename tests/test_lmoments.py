"""Tests for the L-moments Pearson Type III parameter estimate."""

from __future__ import annotations

from collections.abc import Callable
from math import pi, sqrt

import numpy as np
import pytest

from climate_indices import lmoments


@pytest.mark.parametrize("fitter", [lmoments.fit, lmoments.fit_glo, lmoments.fit_gev])
def test_scalar_fit_excludes_masked_observations(fitter: Callable[[np.ndarray], dict[str, float]]) -> None:
    sample = np.ma.array(np.arange(60, dtype=float), mask=np.isin(np.arange(60), [0, 7, 59]))
    sample[11] = np.nan
    expected = fitter(sample.filled(np.nan))

    assert fitter(sample) == pytest.approx(expected)
    np.testing.assert_array_equal(sample.mask, np.isin(np.arange(60), [0, 7, 59]))


@pytest.mark.parametrize("fitter", [lmoments.fit, lmoments.fit_glo, lmoments.fit_gev])
@pytest.mark.parametrize("valid_count", [0, 3])
def test_scalar_fit_minimum_excludes_masked_observations(
    fitter: Callable[[np.ndarray], dict[str, float]], valid_count: int
) -> None:
    sample = np.ma.array(np.arange(60), mask=np.arange(60) >= valid_count)
    with pytest.raises(ValueError, match=f"{valid_count} non-NaN values found"):
        fitter(sample)


def _gamma_ratio(alpha: float) -> float:
    """gamma(alpha) / gamma(alpha + 0.5) from its asymptotic series (DLMF 5.11.13).

    Six terms leave a truncation error below 1e-16 for alpha >= 1e3, and the series
    never subtracts two large numbers, so it is an independent reference for the
    large-alpha (near-zero skew) fits.
    """
    coefficients = (1.0, 1 / 8, 1 / 128, -5 / 1024, -21 / 32768, 399 / 262144)
    return sum(c / alpha**k for k, c in enumerate(coefficients)) / sqrt(alpha)


@pytest.mark.parametrize("tau3", [1e-3, -1e-4, 1e-5, 2e-6])
def test_pearson3_scale_is_accurate_for_near_zero_skew(tau3: float) -> None:
    # a small L-skewness gives alpha between ~1e5 and ~3e10, where the former
    # exp(gammaln(alpha) - gammaln(alpha + 0.5)) lost up to 1e-4 relative accuracy
    second_lmoment = 2.5
    moments = np.array([10.0, second_lmoment, tau3])

    single = lmoments._estimate_pearson3_parameters(moments)
    locs, scales, skews, valid = lmoments._estimate_pearson3_parameters_spatial(moments.reshape(3, 1), np.array([True]))

    for scale, skew in ((single["scale"], single["skew"]), (scales[0], skews[0])):
        assert np.sign(skew) == np.sign(tau3)
        alpha = 4.0 / skew**2
        expected = sqrt(pi) * second_lmoment * sqrt(alpha) * _gamma_ratio(alpha)
        assert scale == pytest.approx(expected, rel=1e-13)
    assert valid[0]
    assert locs[0] == single["loc"]
    assert scales[0] == single["scale"]
    assert skews[0] == single["skew"]
