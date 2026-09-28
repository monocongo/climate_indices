"""Output-scale support (probability/PIT and bounded) across the SPI/SPEI surfaces.

The probability scale is the fitted cumulative probability before the
inverse-normal transform, so ``norm.ppf(probability output)`` must reproduce the
default z-score output wherever that output was not clipped to the [-3.09, 3.09]
range. This holds for the NumPy API, the xarray adapter, and the CLI.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm

from climate_indices import compute, indices, spei, spi, typed_public_api
from climate_indices.exceptions import InvalidArgumentError

_DISTRIBUTIONS = (indices.Distribution.gamma, indices.Distribution.pearson)


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
def test_spi_probability_inverts_to_the_normal_output(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    distribution,
) -> None:
    """``norm.ppf`` of the probability output reproduces the z-scores within the clip range."""
    common = (
        precips_mm_monthly,
        6,
        distribution,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    z_scores = indices.spi(*common)
    probability = indices.spi(*common, output_scale="probability")
    bounded = indices.spi(*common, output_scale="bounded")

    assert np.nanmin(probability) >= 0.0
    assert np.nanmax(probability) <= 1.0
    np.testing.assert_allclose(bounded, (2.0 * probability) - 1.0, equal_nan=True)

    reconstructed = norm.ppf(probability)
    within_clip = np.abs(z_scores) < 3.09
    np.testing.assert_allclose(reconstructed[within_clip], z_scores[within_clip], atol=1e-9)


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
def test_spei_probability_inverts_to_the_normal_output(
    precips_mm_monthly,
    pet_thornthwaite_mm,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    distribution,
) -> None:
    z_scores = indices.spei(
        precips_mm_monthly,
        pet_thornthwaite_mm,
        6,
        distribution,
        compute.Periodicity.monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    )
    probability = indices.spei(
        precips_mm_monthly,
        pet_thornthwaite_mm,
        6,
        distribution,
        compute.Periodicity.monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        output_scale="probability",
    )

    reconstructed = norm.ppf(probability)
    within_clip = np.abs(z_scores) < 3.09
    np.testing.assert_allclose(reconstructed[within_clip], z_scores[within_clip], atol=1e-9)


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
def test_standardized_index_probability_inverts_to_the_normal_output(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    distribution,
) -> None:
    common = (
        precips_mm_monthly,
        3,
        distribution,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    z_scores = indices.standardized_index(*common)
    probability = indices.standardized_index(*common, output_scale="probability")

    reconstructed = norm.ppf(probability)
    within_clip = np.abs(z_scores) < 3.09
    np.testing.assert_allclose(reconstructed[within_clip], z_scores[within_clip], atol=1e-9)


def test_invalid_output_scale_raises(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
) -> None:
    with pytest.raises(InvalidArgumentError):
        indices.spi(
            precips_mm_monthly,
            3,
            indices.Distribution.gamma,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            compute.Periodicity.monthly,
            output_scale="percentile",
        )


@pytest.mark.parametrize("output_scale", ["probability", "bounded"])
def test_compute_transform_output_scales(
    output_scale,
    precips_mm_monthly,
    data_year_start_monthly,
    data_year_end_monthly,
    transformed_gamma_monthly,
) -> None:
    """The compute transforms expose the probabilities on the non-normal scales."""
    common = (
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )
    probabilities = compute.transform_fitted_gamma(*common, output_scale="probability")
    transformed = compute.transform_fitted_gamma(*common, output_scale=output_scale)
    expected = probabilities if output_scale == "probability" else (2.0 * probabilities) - 1.0
    np.testing.assert_allclose(transformed, expected, equal_nan=True)
    assert not np.allclose(transformed, transformed_gamma_monthly, equal_nan=True)


def test_fit_and_standardize_forwards_output_scale(
    precips_mm_monthly,
    data_year_start_monthly,
    data_year_end_monthly,
) -> None:
    probability = compute.fit_and_standardize(
        precips_mm_monthly,
        indices.Distribution.gamma,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
        output_scale="probability",
    )
    assert np.nanmin(probability) >= 0.0
    assert np.nanmax(probability) <= 1.0


def test_xarray_probability_output_carries_distinct_metadata(sample_monthly_precip_da) -> None:
    """The xarray path returns probabilities with probability metadata, not z-score metadata."""
    probability = spi(
        sample_monthly_precip_da,
        scale=6,
        distribution=indices.Distribution.gamma,
        output_scale="probability",
    )
    assert isinstance(probability, xr.DataArray)
    assert probability.attrs["long_name"] == "Standardized Precipitation Index probability"
    assert probability.attrs["units"] == "1"

    z_score = spi(
        sample_monthly_precip_da,
        scale=6,
        distribution=indices.Distribution.gamma,
    )
    assert z_score.attrs["units"] == "dimensionless"

    reconstructed = norm.ppf(probability.values)
    within_clip = np.abs(z_score.values) < 3.09
    np.testing.assert_allclose(reconstructed[within_clip], z_score.values[within_clip], atol=1e-9)


def test_xarray_spei_bounded_output_carries_distinct_metadata(sample_monthly_precip_da, sample_monthly_pet_da) -> None:
    bounded = spei(
        sample_monthly_precip_da,
        sample_monthly_pet_da,
        scale=6,
        distribution=indices.Distribution.gamma,
        output_scale="bounded",
    )
    assert bounded.attrs["long_name"] == ("Standardized Precipitation Evapotranspiration Index bounded probability")
    assert bounded.attrs["units"] == "1"


def test_typed_public_api_defaults_to_normal(sample_monthly_precip_da) -> None:
    """Omitting output_scale leaves the z-score output unchanged."""
    default = typed_public_api.spi(
        sample_monthly_precip_da,
        scale=6,
        distribution=indices.Distribution.gamma,
    )
    explicit = typed_public_api.spi(
        sample_monthly_precip_da,
        scale=6,
        distribution=indices.Distribution.gamma,
        output_scale="normal",
    )
    np.testing.assert_array_equal(default.values, explicit.values)
    assert default.attrs["units"] == explicit.attrs["units"] == "dimensionless"
    assert default.attrs["long_name"] == explicit.attrs["long_name"]
