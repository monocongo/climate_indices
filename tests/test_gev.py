"""Contract tests for the generalized extreme value (GEV) distribution (#1583, ADR-0028).

The distribution is Hosking's GEV fitted by L-moments. Its shape is Hosking's ``k``,
which is SciPy's ``genextreme`` ``c``, so SciPy is the independent reference for the
CDF and for the synthetic samples the fit must recover. GEV is a SPEI-only surface.
Cross-implementation fixtures against R ``lmom`` are optional (OQ-8) and not part of
this ticket.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr
from scipy import optimize, special, stats

import climate_indices
from climate_indices import compute, lmoments, spei
from climate_indices.drought import indices
from climate_indices.exceptions import InvalidArgumentError

_DATA_START_YEAR = 1950
_CALIBRATION_START_YEAR = 1950
_CALIBRATION_END_YEAR = 2009
_SHAPES = [-0.4, -1e-3, -1e-10, 0.0, 1e-10, 1e-3, 0.4]


def _synthetic_series(seed: int = 42, years: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    precip = np.clip(rng.gamma(2.0, 20.0, years * 12), 0.0, None)
    pet = np.clip(rng.gamma(3.0, 15.0, years * 12), 0.0, None)
    return precip, pet


def _spei_kwargs(scale: int = 3) -> dict[str, object]:
    return {
        "scale": scale,
        "distribution": indices.Distribution.gev,
        "periodicity": compute.Periodicity.monthly,
        "data_start_year": _DATA_START_YEAR,
        "calibration_year_initial": _CALIBRATION_START_YEAR,
        "calibration_year_final": _CALIBRATION_END_YEAR,
    }


def _theoretical_lmoments(loc: float, scale: float, shape: float) -> tuple[float, float, float]:
    """(lambda-1, lambda-2, tau-3) of a GEV, from Hosking's closed forms."""
    if shape == 0.0:
        # the Gumbel limit
        return loc + np.euler_gamma * scale, scale * np.log(2.0), 2.0 * np.log(3.0) / np.log(2.0) - 3.0
    gamma = special.gamma(1.0 + shape)
    lambda1 = loc + scale * (1.0 - gamma) / shape
    lambda2 = scale * (1.0 - 2.0**-shape) * gamma / shape
    tau3 = 2.0 * (1.0 - 3.0**-shape) / (1.0 - 2.0**-shape) - 3.0
    return lambda1, lambda2, tau3


def test_gev_display_name_is_lower_case() -> None:
    assert indices.Distribution.gev.value == "gev"
    assert indices.Distribution.gev.display_name == "gev"


def test_gamma_taylor_coefficients_match_the_zeta_expansion() -> None:
    """Gamma(1 + k) = exp(-gamma_E k + sum zeta(n) (-k)^n / n), exponentiated term by term."""
    log_terms = {1: -np.euler_gamma, **{n: special.zeta(n) * (-1) ** n / n for n in range(2, 6)}}
    coefficients = [1.0]
    for n in range(1, 6):
        coefficients.append(sum(j * log_terms[j] * coefficients[n - j] for j in range(1, n + 1)) / n)
    np.testing.assert_allclose(lmoments._GAMMA_TAYLOR, coefficients[1:], rtol=1e-14)


@pytest.mark.parametrize("tau3_offset", [-1e-12, 0.0, 1e-12])
def test_parameters_approach_the_gumbel_limit(tau3_offset: float) -> None:
    """Near k = 0 the fit is the Gumbel one: scale = lambda2 / ln 2, loc = lambda1 - gamma_E scale.

    This is where (1 - Gamma(1 + k)) / k, evaluated directly, rounds 1 + k to 1 and
    returns 0 instead of the Euler constant.
    """
    lambda1, lambda2 = 10.0, 3.0
    gumbel_tau3 = 2.0 * np.log(3.0) / np.log(2.0) - 3.0
    loc, scale, shape, valid = lmoments._gev_parameters(lambda1, lambda2, gumbel_tau3 + tau3_offset)
    assert bool(valid)
    assert abs(float(shape)) < 1e-10
    assert float(scale) == pytest.approx(lambda2 / np.log(2.0), rel=1e-9)
    assert float(loc) == pytest.approx(lambda1 - np.euler_gamma * lambda2 / np.log(2.0), rel=1e-9)


def test_location_is_continuous_across_the_series_switch() -> None:
    """The Taylor series below |k| = 1e-3 and the direct form above it agree there."""
    limit = lmoments._GEV_GAMMA_SERIES_LIMIT
    below, above = (
        lmoments._gev_parameters(10.0, 3.0, lmoments._gev_tau3(np.array(limit + offset)))[0] for offset in (-1e-9, 1e-9)
    )
    # the true change over 2e-9 of shape is about 2e-9 * (d loc / d k) < 1e-8
    assert abs(float(above) - float(below)) < 1e-8


@pytest.mark.parametrize("shape", [-0.9, -0.3, -1e-8, 0.0, 1e-8, 0.3, 2.0, 5.0])
def test_shape_solver_inverts_the_l_skewness(shape: float) -> None:
    tau3 = lmoments._gev_tau3(np.array(shape))
    assert float(lmoments._gev_shape(tau3)) == pytest.approx(shape, abs=1e-10)


@pytest.mark.parametrize("shape", [-0.7, -0.2, 0.0, 0.2, 0.6])
def test_parameters_invert_the_theoretical_l_moments(shape: float) -> None:
    """Hosking's closed-form L-moments of (10, 3, k) convert back to (10, 3, k)."""
    loc, scale = 10.0, 3.0
    lambda1, lambda2, tau3 = _theoretical_lmoments(loc, scale, shape)
    fitted_loc, fitted_scale, fitted_shape, valid = lmoments._gev_parameters(lambda1, lambda2, tau3)
    assert bool(valid)
    assert float(fitted_loc) == pytest.approx(loc, abs=1e-9)
    assert float(fitted_scale) == pytest.approx(scale, abs=1e-9)
    assert float(fitted_shape) == pytest.approx(shape, abs=1e-9)


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.25])
def test_gev_lmoment_fit_recovers_known_parameters(shape: float) -> None:
    """A large sample drawn from a known GEV recovers its parameters."""
    loc, scale = 12.0, 3.0
    sample = stats.genextreme.rvs(shape, loc=loc, scale=scale, size=200_000, random_state=11)
    params = lmoments.fit_gev(sample)
    assert params["loc"] == pytest.approx(loc, abs=0.05)
    assert params["scale"] == pytest.approx(scale, abs=0.05)
    assert params["shape"] == pytest.approx(shape, abs=0.02)


@pytest.mark.parametrize("shape", _SHAPES)
def test_cdf_matches_scipy_genextreme(shape: float) -> None:
    """The transform's probabilities agree with SciPy at rtol=1e-12, ends of support included."""
    x = np.linspace(-15.0, 30.0, 401)
    ours = compute._gev_fit(x, 3.0, 2.0, shape, output_scale="probability")
    reference = stats.genextreme.cdf(x, shape, loc=3.0, scale=2.0)
    np.testing.assert_allclose(ours, reference, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("shape", [-0.4, -1e-10, 0.0, 1e-10, 0.4])
def test_transform_round_trips_the_gev_quantile(shape: float) -> None:
    """Standardizing a GEV quantile (SciPy's ppf) gives the matching normal quantile."""
    probabilities = np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98])
    quantiles = stats.genextreme.ppf(probabilities, shape, loc=3.0, scale=2.0)
    # the error measured is ~1e-15; 1e-12 leaves room for the platform's libm
    np.testing.assert_allclose(
        compute._gev_fit(quantiles, 3.0, 2.0, shape), stats.norm.ppf(probabilities), rtol=0.0, atol=1e-12
    )


def test_a_value_beyond_the_finite_end_is_infinite_on_the_normal_scale() -> None:
    """k > 0 is bounded above and k < 0 below; past that end F is 1 or 0 and z is infinite."""
    upper = compute._gev_fit(np.array([100.0]), 0.0, 1.0, 0.5)
    lower = compute._gev_fit(np.array([-100.0]), 0.0, 1.0, -0.5)
    assert upper[0] == np.inf
    assert lower[0] == -np.inf


@pytest.mark.parametrize(
    ("shape", "loc", "expected"),
    [
        # the water balance sits near 1000, far above the upper end loc + scale / k = 2
        (0.5, 0.0, indices._FITTED_INDEX_VALID_MAX),
        # and far below the lower end loc + scale / k = 1998
        (-0.5, 2000.0, indices._FITTED_INDEX_VALID_MIN),
    ],
)
def test_spei_clips_values_beyond_the_fitted_support(shape: float, loc: float, expected: float) -> None:
    """The index layer clips the infinite z of an out-of-support value, on either bounded branch."""
    precip, pet = _synthetic_series()
    params = {"loc": np.full(12, loc), "scale": np.ones(12), "shape": np.full(12, shape)}
    # scale 1 keeps the offset water balance near 1000, between the two ends chosen above
    result = indices.spei(precip, pet, fitting_params=params, **_spei_kwargs(scale=1))
    finite = result[~np.isnan(result)]
    assert finite.size
    assert np.all(finite == expected)


@pytest.mark.parametrize(("output_scale", "lower", "upper"), [("probability", 0.0, 1.0), ("bounded", -1.0, 1.0)])
def test_spei_gev_output_scales_are_bounded(output_scale: compute.OutputScale, lower: float, upper: float) -> None:
    precip, pet = _synthetic_series()
    kwargs: dict[str, object] = {**_spei_kwargs(scale=6), "output_scale": output_scale}
    result = indices.spei(precip, pet, **kwargs)
    finite = result[np.isfinite(result)]
    assert finite.size
    assert finite.min() >= lower
    assert finite.max() <= upper


def test_a_value_beyond_the_fitted_end_is_probability_one() -> None:
    """The probability scale keeps the fitted end value instead of the normal-scale clip."""
    precip = np.full(60 * 12, 1000.0)
    pet = np.zeros(60 * 12)
    params = {"loc": np.zeros(12), "scale": np.ones(12), "shape": np.full(12, 0.5)}
    # the water balance near 2000 is past the upper end loc + scale / k = 2
    kwargs: dict[str, object] = {
        **_spei_kwargs(scale=1),
        "fitting_params": params,
        "output_scale": "probability",
    }
    result = indices.spei(precip, pet, **kwargs)
    finite = result[np.isfinite(result)]
    assert finite.size
    assert np.all(finite == 1.0)


def test_a_position_without_a_valid_fit_is_nan() -> None:
    result = compute._gev_fit(np.array([1.0, 2.0, 3.0]), np.zeros(3), np.array([1.0, 0.0, np.nan]), 0.1)
    assert np.isfinite(result[0])
    assert np.isnan(result[1:]).all()


@pytest.mark.parametrize(
    ("sample", "message"),
    [
        (np.array([1.0, 2.0, 3.0]), "Insufficient number of values"),  # under the L-moment minimum
        (np.full(30, 5.0), "Unable to calculate GEV parameters"),  # constant: a zero L-scale
        (np.full(30, 990.1), "Unable to calculate GEV parameters"),  # constant: a rounded-zero L-scale
        (np.array([np.nan] * 30), "Insufficient number of values"),
    ],
)
def test_an_unfittable_sample_raises(sample: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        lmoments.fit_gev(sample)


def test_a_short_sample_error_offers_no_gamma_remedy() -> None:
    """A GEV fit has no gamma fallback, so its error must not suggest one."""
    with pytest.raises(ValueError) as short:
        lmoments.fit_gev(np.array([1.0, 2.0, 3.0]))
    assert "Gamma" not in str(short.value)
    assert "Pearson" not in str(short.value)


@pytest.mark.parametrize("tau3", [1.0, -1.0, 1.5])
def test_l_skewness_outside_the_valid_range_is_invalid(tau3: float) -> None:
    *_, valid = lmoments._gev_parameters(np.array(1.0), np.array(1.0), np.array(tau3))
    assert not bool(valid)


def test_spatial_fit_matches_the_scalar_fit_per_cell() -> None:
    rng = np.random.default_rng(5)
    block = stats.genextreme.rvs(-0.1, loc=10.0, scale=2.0, size=(60, 3, 2), random_state=rng)
    block[:, 0, 0] = np.nan  # an empty cell is invalid, not an error
    locs, scales, shapes, valid = lmoments.fit_gev_spatial(block)
    assert not valid[0, 0]
    assert locs[0, 0] == scales[0, 0] == shapes[0, 0] == 0.0
    for i in range(3):
        for j in range(2):
            if (i, j) == (0, 0):
                continue
            expected = lmoments.fit_gev(block[:, i, j])
            assert valid[i, j]
            assert locs[i, j] == pytest.approx(expected["loc"], rel=1e-12)
            assert scales[i, j] == pytest.approx(expected["scale"], rel=1e-12)
            assert shapes[i, j] == pytest.approx(expected["shape"], rel=1e-12, abs=1e-14)


def _reference_gev_fit(sample: np.ndarray) -> tuple[float, float, float]:
    """(shape, loc, scale) from SciPy's sample L-moments and a root solve on the closed-form L-skewness."""
    lambda1, lambda2, tau3 = stats.lmoment(sample, order=[1, 2, 3])  # standardized: the third is tau3

    def residual(shape: float) -> float:
        ratio = 2.0 * np.log(3.0) / np.log(2.0) if shape == 0.0 else 2.0 * (1 - 3.0**-shape) / (1 - 2.0**-shape)
        return float(ratio - 3.0 - tau3)

    shape = optimize.brentq(residual, -0.999, 60.0, xtol=1e-14, rtol=1e-14)
    gamma = special.gamma(1.0 + shape)
    scale = lambda2 * shape / ((1.0 - 2.0**-shape) * gamma)
    return shape, lambda1 - scale * (1.0 - gamma) / shape, scale


def test_spei_gev_matches_an_independent_scipy_pipeline() -> None:
    """spei(gev) equals SciPy's L-moments, a root solve for the shape, genextreme.cdf and norm.ppf."""
    years = 60
    precip, pet = _synthetic_series(years=years)
    result = indices.spei(precip, pet, **_spei_kwargs(scale=1))
    balance = ((precip - pet) + 1000.0).reshape(years, 12)
    expected = np.empty_like(balance)
    for month in range(12):
        shape, loc, scale = _reference_gev_fit(balance[:, month])
        expected[:, month] = stats.norm.ppf(stats.genextreme.cdf(balance[:, month], shape, loc=loc, scale=scale))
    # in-sample values past a fitted finite end are infinite here and clipped by spei();
    # the difference measured is ~1e-12, so 1e-9 absorbs only platform wobble
    expected = np.clip(expected, indices._FITTED_INDEX_VALID_MIN, indices._FITTED_INDEX_VALID_MAX).reshape(-1)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-9)


def test_gev_parameters_treat_masked_observations_as_missing() -> None:
    precip, pet = _synthetic_series()
    balance = ((precip - pet) + 1000.0).reshape(60, 12)
    balance[11, 1] = np.nan
    mask = np.zeros_like(balance, dtype=bool)
    mask[7, 0] = True
    masked = np.ma.array(balance, mask=mask)
    args = (_DATA_START_YEAR, _CALIBRATION_START_YEAR, _CALIBRATION_END_YEAR, compute.Periodicity.monthly)

    actual = compute.gev_parameters(masked, *args)
    expected = compute.gev_parameters(masked.filled(np.nan), *args)
    spatial = compute.gev_parameters(masked[..., np.newaxis], *args)
    assert np.all(actual[1] > 0.0)
    for parameter, nan_parameter, spatial_parameter in zip(actual, expected, spatial, strict=True):
        np.testing.assert_allclose(parameter, nan_parameter, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(parameter, spatial_parameter[:, 0], rtol=1e-12, atol=1e-12)


def test_gev_parameters_mark_degenerate_steps_invalid() -> None:
    """A constant calibration step cannot be fitted and is marked by a zero scale."""
    locs, scales, shapes = compute.gev_parameters(
        np.full((60, 12), 5.0),
        _DATA_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        compute.Periodicity.monthly,
    )
    assert np.all(scales == 0.0)
    assert np.all(locs == 0.0)
    assert np.all(shapes == 0.0)


def test_gev_parameters_daily_shapes() -> None:
    """The daily path fits one parameter per day of the 366-day calendar."""
    rng = np.random.default_rng(13)
    series = np.clip(rng.gamma(2.0, 20.0, 366 * 60), 0.0, None)
    locs, scales, shapes = compute.gev_parameters(
        series, _DATA_START_YEAR, _CALIBRATION_START_YEAR, _CALIBRATION_END_YEAR, compute.Periodicity.daily
    )
    assert locs.shape == scales.shape == shapes.shape == (366,)
    assert np.isfinite(locs).all()
    assert np.isfinite(shapes).all()
    assert (scales > 0.0).all()


def test_gev_high_failure_rate_warning_names_gev(caplog: pytest.LogCaptureFixture) -> None:
    """The warning must not borrow the Pearson Type III wording or suggest a gamma fallback."""
    with caplog.at_level(logging.WARNING):
        compute.gev_parameters(
            np.full((60, 12), 5.0),
            _DATA_START_YEAR,
            _CALIBRATION_START_YEAR,
            _CALIBRATION_END_YEAR,
            compute.Periodicity.monthly,
        )
    assert "high_fitting_failure_rate" in caplog.text
    assert "gev" in caplog.text
    assert "Pearson" not in caplog.text
    assert "Gamma" not in caplog.text


def test_a_failed_step_is_nan_for_that_step_only_with_no_gamma_fallback() -> None:
    """One constant calendar month is missing; the other eleven are fitted."""
    years = 60
    precip, pet = _synthetic_series(years=years)
    precip[0::12], pet[0::12] = 100.0, 90.0  # January balance is constant
    result = indices.spei(precip, pet, **_spei_kwargs(scale=1)).reshape(years, 12)
    assert np.isnan(result[:, 0]).all()
    assert np.isfinite(result[:, 1:]).all()


def test_spei_gev_degenerate_series_is_missing() -> None:
    result = indices.spei(np.full(60 * 12, 100.0), np.full(60 * 12, 90.0), **_spei_kwargs(scale=6))
    assert np.all(np.isnan(result))


def test_spei_gev_non_integer_constant_series_is_missing() -> None:
    """The float residue of a non-integer constant is not a fit."""
    result = indices.spei(np.full(60 * 12, 100.1), np.full(60 * 12, 90.0), **_spei_kwargs(scale=1))
    assert np.all(np.isnan(result))


def test_spei_gev_all_missing_returns_missing() -> None:
    missing = np.full(60 * 12, np.nan)
    assert np.all(np.isnan(indices.spei(missing, missing, **_spei_kwargs(scale=6))))


def test_spei_gev_with_a_calibration_window_below_the_l_moment_minimum_is_missing() -> None:
    """Three calibration years is under the four L-moments need, so no step has a fit."""
    precip, pet = _synthetic_series()
    kwargs = {**_spei_kwargs(), "calibration_year_final": _CALIBRATION_START_YEAR + 2}
    assert np.all(np.isnan(indices.spei(precip, pet, **kwargs)))


def test_spei_gev_differs_from_loglogistic() -> None:
    """The families fit differently, so gev is not silently routed to the log-logistic path."""
    precip, pet = _synthetic_series()
    result = indices.spei(precip, pet, **_spei_kwargs())
    other = indices.spei(precip, pet, **{**_spei_kwargs(), "distribution": indices.Distribution.loglogistic})
    assert not np.allclose(result, other, equal_nan=True)


def test_spei_gev_fitting_params_round_trip() -> None:
    """A parameter set fitted by the public API reproduces the transform exactly."""
    precip, pet = _synthetic_series()
    kwargs = _spei_kwargs(scale=6)
    baseline = indices.spei(precip, pet, **kwargs)
    scaled = compute.prepare_scaled((precip - pet) + 1000.0, 6, compute.Periodicity.monthly, clip_negatives=False)
    locs, scales, shapes = compute.gev_parameters(
        scaled, _DATA_START_YEAR, _CALIBRATION_START_YEAR, _CALIBRATION_END_YEAR, compute.Periodicity.monthly
    )
    assert locs.shape == scales.shape == shapes.shape == (12,)
    repeated = indices.spei(precip, pet, fitting_params={"loc": locs, "scale": scales, "shape": shapes}, **kwargs)
    np.testing.assert_allclose(baseline, repeated, equal_nan=True)


def test_a_partial_fitting_params_set_raises() -> None:
    precip, pet = _synthetic_series()
    kwargs = _spei_kwargs()
    partial = {"loc": np.zeros(12)}
    with pytest.raises(ValueError, match="gev fitting parameters"):
        indices.spei(precip, pet, fitting_params=partial, **kwargs)


def test_spei_gev_block_matches_per_cell() -> None:
    precip, pet = _synthetic_series()
    block = np.stack([np.stack([precip, precip * 1.1], axis=-1)] * 2, axis=1)
    pet_block = np.stack([np.stack([pet, pet], axis=-1)] * 2, axis=1)
    block_result = indices.spei(block, pet_block, spatial_time_major=True, **_spei_kwargs())
    for i in range(block.shape[1]):
        for j in range(block.shape[2]):
            per_cell = indices.spei(block[:, i, j], pet_block[:, i, j], **_spei_kwargs())
            np.testing.assert_allclose(block_result[:, i, j], per_cell, equal_nan=True)


@pytest.mark.parametrize("lazy", [False, True])
def test_xarray_spei_gev_records_the_distribution(
    sample_monthly_precip_da: xr.DataArray, sample_monthly_pet_da: xr.DataArray, lazy: bool
) -> None:
    """The xarray route accepts gev, with time in one chunk when lazy, and labels the result."""
    precip, pet = sample_monthly_precip_da, sample_monthly_pet_da
    if lazy:
        precip, pet = precip.chunk({"time": -1}), pet.chunk({"time": -1})
    result = spei(precip, pet, scale=3, distribution=indices.Distribution.gev)
    if lazy:
        # still a graph, with time in a single chunk, before anything is computed
        assert result.chunks is not None
        assert len(result.chunks[result.dims.index("time")]) == 1
    else:
        assert result.chunks is None
    computed = result.compute() if lazy else result
    assert computed.attrs["distribution"] == "gev"
    assert "gev distribution" in computed.attrs["history"]
    eager = spei(sample_monthly_precip_da, sample_monthly_pet_da, scale=3, distribution=indices.Distribution.gev)
    np.testing.assert_allclose(computed.values, eager.values, equal_nan=True)


def test_other_surfaces_reject_gev(sample_monthly_precip_da: xr.DataArray) -> None:
    """GEV stays a SPEI-only surface: spi, standardized_index and fit_diagnostics refuse it."""
    precip = sample_monthly_precip_da
    with pytest.raises(InvalidArgumentError, match="SPEI is the supported GEV surface"):
        climate_indices.spi(precip, 3, indices.Distribution.gev)
    with pytest.raises(InvalidArgumentError, match="Unsupported distribution"):
        climate_indices.standardized_index(precip, 3, indices.Distribution.gev)
    sample = np.random.default_rng(0).gamma(2.0, 10.0, (40, 12))
    with pytest.raises(ValueError, match="Unsupported distribution"):
        compute.fit_diagnostics(
            sample,
            indices.Distribution.gev,
            _DATA_START_YEAR,
            _CALIBRATION_START_YEAR,
            _CALIBRATION_START_YEAR + 39,
            compute.Periodicity.monthly,
        )
    with pytest.raises(InvalidArgumentError, match="SPEI is the supported GEV surface"):
        indices.fit_diagnostics(
            sample,
            3,
            indices.Distribution.gev,
            _DATA_START_YEAR,
            _CALIBRATION_START_YEAR,
            _CALIBRATION_START_YEAR + 39,
            compute.Periodicity.monthly,
        )
