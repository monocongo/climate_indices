"""Contract tests for the log-logistic (generalized logistic) distribution (#106).

The distribution is Hosking's GLO with unbiased-PWM L-moments — what R's SPEI
package and SPEIbase fit under the name "log-Logistic". These tests pin the
parameter conversion and the transform, and the SPEI-only coverage boundary.
Cross-implementation fixtures against R SPEI are tracked by #1195.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from climate_indices import compute, indices, lmoments
from climate_indices.exceptions import InvalidArgumentError

_DATA_START_YEAR = 1950
_CALIBRATION_START_YEAR = 1950
_CALIBRATION_END_YEAR = 2009


def _glo_quantile(probabilities: np.ndarray, loc: float, scale: float, shape: float) -> np.ndarray:
    """Hosking's GLO quantile function (lmom's quaglo)."""
    if abs(shape) <= 1e-6:
        return loc + scale * np.log(probabilities / (1.0 - probabilities))
    return loc + (scale / shape) * (1.0 - ((1.0 - probabilities) / probabilities) ** shape)


def _synthetic_series(seed: int = 42, years: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    precip = np.clip(rng.gamma(2.0, 20.0, years * 12), 0.0, None)
    pet = np.clip(rng.gamma(3.0, 15.0, years * 12), 0.0, None)
    return precip, pet


def test_glo_parameters_match_pelglo_closed_form() -> None:
    """The L-moment conversion reproduces Hosking's PELGLO equations.

    The expected values are independent of the implementation: they are the PELGLO
    output for (lambda-1, lambda-2, tau-3) = (10, 2, 0.4).
    """
    params = lmoments._estimate_glo_parameters(np.array([10.0, 2.0, 0.4]))
    assert params["loc"] == pytest.approx(8.7841336432)
    assert params["scale"] == pytest.approx(1.5136534573)
    assert params["shape"] == pytest.approx(-0.4)


def test_glo_parameters_zero_shape_is_the_logistic_limit() -> None:
    """A negligible L-skewness yields an ordinary logistic fit (shape 0)."""
    params = lmoments._estimate_glo_parameters(np.array([5.0, 1.5, 0.0]))
    assert params == {"loc": 5.0, "scale": 1.5, "shape": 0.0}


@pytest.mark.parametrize("shape", [-0.3, 0.0, 0.25])
def test_glo_lmoment_fit_recovers_known_parameters(shape: float) -> None:
    """A large sample drawn from a known GLO recovers its parameters."""
    loc, scale = 12.0, 3.0
    rng = np.random.default_rng(11)
    sample = _glo_quantile(rng.uniform(1e-4, 1 - 1e-4, 200_000), loc, scale, shape)
    params = lmoments.fit_glo(sample)
    assert params["loc"] == pytest.approx(loc, abs=0.05)
    assert params["scale"] == pytest.approx(scale, abs=0.05)
    assert params["shape"] == pytest.approx(shape, abs=0.02)


@pytest.mark.parametrize("shape", [-0.4, 0.0, 0.4])
def test_transform_round_trips_the_glo_quantile(shape: float) -> None:
    """Standardizing a GLO quantile gives the matching normal quantile."""
    loc, scale = 2.0, 1.5
    probabilities = np.array([0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 0.98])
    padded = np.full(12, np.nan)
    padded[: probabilities.size] = _glo_quantile(probabilities, loc, scale, shape)
    standardized = compute._loglogistic_fit(padded.reshape(1, 12), loc, scale, shape)
    np.testing.assert_allclose(standardized[0, : probabilities.size], stats.norm.ppf(probabilities))


def test_transform_maps_out_of_support_like_cdfglo() -> None:
    """A value beyond the fitted support follows cdfglo's 0/1 rather than NaN.

    ``lmom::cdfglo`` clamps ``1 - shape*z`` at zero, so a value beyond the support
    yields a probability of 0 or 1 and the normal-scale z is infinite; the index
    layer clips that to the supported range, as it does for the other transforms.
    """
    values = np.full((1, 12), np.nan)
    values[0, 0] = 10.0
    probabilities = compute._loglogistic_fit(values, 0.0, 1.0, 0.5, output_scale="probability")
    assert probabilities[0, 0] == 1.0
    z_scores = compute._loglogistic_fit(values, 0.0, 1.0, 0.5)
    assert np.isposinf(z_scores[0, 0])
    assert np.isfinite(np.clip(z_scores[0, 0], -3.09, 3.09))


def test_transform_reports_nan_where_the_fit_is_invalid() -> None:
    """A non-positive scale or an out-of-support shape marks its positions NaN."""
    values = np.arange(1.0, 13.0).reshape(1, 12)
    scales = np.full(12, 1.0)
    scales[3] = 0.0
    shapes = np.zeros(12)
    shapes[5] = 1.0
    standardized = compute._loglogistic_fit(values, np.zeros(12), scales, shapes)
    assert np.isnan(standardized[0, 3])
    assert np.isnan(standardized[0, 5])
    assert np.isfinite(standardized[0, 0])


def test_loglogistic_parameters_are_location_equivariant() -> None:
    """An offset shifts only loc, which keeps the P−PET offset out of the result."""
    values, _ = _synthetic_series()
    values = values.reshape(60, 12)
    locs, scales, shapes = compute.loglogistic_parameters(
        values, _DATA_START_YEAR, _CALIBRATION_START_YEAR, _CALIBRATION_END_YEAR, compute.Periodicity.monthly
    )
    shifted_locs, shifted_scales, shifted_shapes = compute.loglogistic_parameters(
        values + 1000.0, _DATA_START_YEAR, _CALIBRATION_START_YEAR, _CALIBRATION_END_YEAR, compute.Periodicity.monthly
    )
    np.testing.assert_allclose(shifted_locs, locs + 1000.0)
    np.testing.assert_allclose(shifted_scales, scales)
    np.testing.assert_allclose(shifted_shapes, shapes)


def test_spei_loglogistic_is_invariant_to_a_constant_offset() -> None:
    """The GLO is a location family, so the P−PET offset cancels in standardization."""
    precip, pet = _synthetic_series()
    kwargs = {
        "scale": 6,
        "distribution": indices.Distribution.loglogistic,
        "periodicity": compute.Periodicity.monthly,
        "data_start_year": _DATA_START_YEAR,
        "calibration_year_initial": _CALIBRATION_START_YEAR,
        "calibration_year_final": _CALIBRATION_END_YEAR,
    }
    baseline = indices.spei(precip, pet, **kwargs)
    shifted = indices.spei(precip + 250.0, pet, **kwargs)
    np.testing.assert_allclose(baseline, shifted, equal_nan=True, atol=1e-9)


def test_spei_loglogistic_is_finite_and_bounded() -> None:
    precip, pet = _synthetic_series()
    result = indices.spei(
        precip,
        pet,
        scale=6,
        distribution=indices.Distribution.loglogistic,
        periodicity=compute.Periodicity.monthly,
        data_start_year=_DATA_START_YEAR,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    finite = result[np.isfinite(result)]
    assert finite.size > 0
    assert finite.min() >= indices._FITTED_INDEX_VALID_MIN
    assert finite.max() <= indices._FITTED_INDEX_VALID_MAX


def test_spei_loglogistic_probability_scale_is_a_probability() -> None:
    precip, pet = _synthetic_series()
    result = indices.spei(
        precip,
        pet,
        scale=6,
        distribution=indices.Distribution.loglogistic,
        periodicity=compute.Periodicity.monthly,
        data_start_year=_DATA_START_YEAR,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
        output_scale="probability",
    )
    finite = result[np.isfinite(result)]
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0


def test_spei_loglogistic_spatial_matches_per_cell() -> None:
    """A folded time-major block fits each cell exactly as a single series does."""
    precip, pet = _synthetic_series()
    block = np.stack([np.stack([precip, precip * 1.1], axis=-1)] * 2, axis=1)
    pet_block = np.stack([np.stack([pet, pet], axis=-1)] * 2, axis=1)
    kwargs = {
        "scale": 6,
        "distribution": indices.Distribution.loglogistic,
        "periodicity": compute.Periodicity.monthly,
        "data_start_year": _DATA_START_YEAR,
        "calibration_year_initial": _CALIBRATION_START_YEAR,
        "calibration_year_final": _CALIBRATION_END_YEAR,
    }
    block_result = indices.spei(block, pet_block, spatial_time_major=True, **kwargs)
    for i in range(block.shape[1]):
        for j in range(block.shape[2]):
            per_cell = indices.spei(block[:, i, j], pet_block[:, i, j], **kwargs)
            np.testing.assert_allclose(block_result[:, i, j], per_cell, equal_nan=True)


def test_spei_loglogistic_fitting_params_reproduce_the_fit() -> None:
    """A parameter set fitted by the public API reproduces the transform exactly."""
    precip, pet = _synthetic_series()
    kwargs = {
        "scale": 6,
        "distribution": indices.Distribution.loglogistic,
        "periodicity": compute.Periodicity.monthly,
        "data_start_year": _DATA_START_YEAR,
        "calibration_year_initial": _CALIBRATION_START_YEAR,
        "calibration_year_final": _CALIBRATION_END_YEAR,
    }
    baseline = indices.spei(precip, pet, **kwargs)
    scaled = compute.prepare_scaled((precip - pet) + 1000.0, 6, compute.Periodicity.monthly, clip_negatives=False)
    locs, scales, shapes = compute.loglogistic_parameters(
        scaled, _DATA_START_YEAR, _CALIBRATION_START_YEAR, _CALIBRATION_END_YEAR, compute.Periodicity.monthly
    )
    repeated = indices.spei(precip, pet, fitting_params={"loc": locs, "scale": scales, "shape": shapes}, **kwargs)
    np.testing.assert_allclose(baseline, repeated, equal_nan=True)


def test_spei_loglogistic_spatial_fitting_params_broadcast() -> None:
    """Period-only parameters broadcast across a folded block's cells."""
    precip, pet = _synthetic_series()
    block = np.stack([np.stack([precip, precip * 1.1], axis=-1)] * 2, axis=1)
    pet_block = np.stack([np.stack([pet, pet], axis=-1)] * 2, axis=1)
    locs, scales, shapes = compute.loglogistic_parameters(
        precip.reshape(-1, 12),
        _DATA_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        compute.Periodicity.monthly,
    )
    result = indices.spei(
        block,
        pet_block,
        scale=6,
        distribution=indices.Distribution.loglogistic,
        periodicity=compute.Periodicity.monthly,
        data_start_year=_DATA_START_YEAR,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
        fitting_params={"loc": locs, "scale": scales, "shape": shapes},
        spatial_time_major=True,
    )
    assert result.shape == block.shape
    assert np.isfinite(result).any()


def test_loglogistic_parameters_mark_degenerate_steps_invalid() -> None:
    """A constant calibration step cannot be fitted and is marked by a zero scale."""
    locs, scales, shapes = compute.loglogistic_parameters(
        np.full((60, 12), 5.0),
        _DATA_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
        compute.Periodicity.monthly,
    )
    assert np.all(scales == 0.0)
    assert np.all(locs == 0.0)
    assert np.all(shapes == 0.0)


def test_spei_loglogistic_degenerate_series_is_missing() -> None:
    """A series with no variability cannot be fitted and standardizes to NaN."""
    result = indices.spei(
        np.full(60 * 12, 100.0),
        np.full(60 * 12, 90.0),
        scale=6,
        distribution=indices.Distribution.loglogistic,
        periodicity=compute.Periodicity.monthly,
        data_start_year=_DATA_START_YEAR,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    assert np.all(np.isnan(result))


def test_spei_loglogistic_all_missing_returns_missing() -> None:
    missing = np.full(60 * 12, np.nan)
    result = indices.spei(
        missing,
        missing,
        scale=6,
        distribution=indices.Distribution.loglogistic,
        periodicity=compute.Periodicity.monthly,
        data_start_year=_DATA_START_YEAR,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
    )
    assert np.all(np.isnan(result))


def test_loglogistic_parameters_daily_shapes() -> None:
    """The daily path fits one parameter per day of the 366-day calendar."""
    rng = np.random.default_rng(13)
    series = np.clip(rng.gamma(2.0, 20.0, 366 * 60), 0.0, None)
    locs, scales, shapes = compute.loglogistic_parameters(
        series, _DATA_START_YEAR, _CALIBRATION_START_YEAR, _CALIBRATION_END_YEAR, compute.Periodicity.daily
    )
    assert locs.shape == scales.shape == shapes.shape == (366,)
    assert np.isfinite(scales).all()


@pytest.mark.parametrize("surface", ["spi", "standardized_index", "fit_diagnostics"])
def test_spi_family_rejects_loglogistic(surface: str) -> None:
    """The GLO is SPEI-only: a non-negative series has a zero mass it cannot place."""
    precip, _ = _synthetic_series()
    with pytest.raises(InvalidArgumentError, match="zero mass"):
        getattr(indices, surface)(
            precip,
            scale=6,
            distribution=indices.Distribution.loglogistic,
            periodicity=compute.Periodicity.monthly,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CALIBRATION_START_YEAR,
            calibration_year_final=_CALIBRATION_END_YEAR,
        )


def test_loglogistic_display_name_is_hyphenated() -> None:
    """Prose and metadata spell the distribution "log-logistic", not "loglogistic"."""
    assert indices.Distribution.loglogistic.display_name == "log-logistic"
    assert indices.Distribution.gamma.display_name == "gamma"
    assert indices.Distribution.pearson.display_name == "pearson"


def test_cli_spi_distribution_loop_excludes_loglogistic() -> None:
    from climate_indices import __main__ as cli_main

    assert indices.Distribution.loglogistic not in cli_main._SPI_DISTRIBUTIONS
    assert set(cli_main._SPI_DISTRIBUTIONS) == {indices.Distribution.gamma, indices.Distribution.pearson}
