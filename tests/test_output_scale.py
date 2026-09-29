"""Output-scale support (probability/PIT and bounded) across the SPI/SPEI surfaces.

The probability scale is the fitted cumulative probability before the
inverse-normal transform, so ``norm.ppf(probability output)`` must reproduce the
default z-score output wherever that output was not clipped to the [-3.09, 3.09]
range. The scale is deliberately unclipped, so the returned probabilities must
reach beyond that range's [0.001, 0.999]. This holds for the NumPy API, the
xarray adapter, and the CLI.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.stats import norm

from climate_indices import compute, indices, spei, spi, typed_public_api
from climate_indices.exceptions import InvalidArgumentError

_DISTRIBUTIONS = (indices.Distribution.gamma, indices.Distribution.pearson)

_METADATA_CASES = [
    ("spi", "probability", "Standardized Precipitation Index probability"),
    ("spi", "bounded", "Standardized Precipitation Index bounded probability"),
    ("spei", "probability", "Standardized Precipitation Evapotranspiration Index probability"),
    ("spei", "bounded", "Standardized Precipitation Evapotranspiration Index bounded probability"),
]


def _assert_unclipped_tails(probability: np.ndarray) -> None:
    """The PIT reaches beyond the z-clip's probability range, i.e. it is unclipped."""
    assert np.nanmin(probability) < norm.cdf(-3.09)
    assert np.nanmax(probability) > norm.cdf(3.09)


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
    _assert_unclipped_tails(probability)
    np.testing.assert_allclose(bounded, (2.0 * probability) - 1.0, equal_nan=True)

    reconstructed = norm.ppf(probability)
    within_clip = np.abs(z_scores) < 3.09
    assert within_clip.any()
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

    # the P−PET series is offset and its fitted range need not reach the tails,
    # so only the inversion is asserted here; the unclipped property is pinned on
    # the (zero-inflated) precipitation SPI test above
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

    _assert_unclipped_tails(probability)
    reconstructed = norm.ppf(probability)
    within_clip = np.abs(z_scores) < 3.09
    np.testing.assert_allclose(reconstructed[within_clip], z_scores[within_clip], atol=1e-9)


@pytest.mark.parametrize("distribution", _DISTRIBUTIONS)
def test_pearson_probability_is_not_shifted_at_the_support_boundary(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    distribution,
) -> None:
    """Pearson values at or below the fitted support floor take a PIT of 0, not the 0.0005 sentinel."""
    probability = indices.spi(
        precips_mm_monthly,
        6,
        distribution,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
        output_scale="probability",
    )
    assert np.nanmin(probability) < 0.0005


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
    """The compute transforms expose the fitted CDF for the non-normal scales."""
    transformed = compute.transform_fitted_gamma(
        precips_mm_monthly,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
        output_scale=output_scale,
    )
    # the independent expectation: the z-score fixture's own cumulative probability,
    # mapped to the requested scale, wherever the fixture was not clipped
    within_clip = np.abs(transformed_gamma_monthly) < 3.09
    expected = norm.cdf(transformed_gamma_monthly)
    if output_scale == "bounded":
        expected = (2.0 * expected) - 1.0
    np.testing.assert_allclose(transformed[within_clip], expected[within_clip], atol=1e-9)


def test_fit_and_standardize_forwards_both_non_normal_scales(
    precips_mm_monthly,
    data_year_start_monthly,
    data_year_end_monthly,
) -> None:
    common = (
        precips_mm_monthly,
        indices.Distribution.gamma,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )
    probability = compute.fit_and_standardize(*common, output_scale="probability")
    bounded = compute.fit_and_standardize(*common, output_scale="bounded")

    assert np.nanmin(probability) >= 0.0
    assert np.nanmax(probability) <= 1.0
    np.testing.assert_allclose(bounded, (2.0 * probability) - 1.0, equal_nan=True)


@pytest.mark.parametrize(("index_name", "output_scale", "long_name"), _METADATA_CASES)
def test_xarray_non_normal_output_carries_distinct_metadata(
    index_name, output_scale, long_name, sample_monthly_precip_da, sample_monthly_pet_da
) -> None:
    """Each non-normal scale returns probability metadata, not the z-score metadata."""
    if index_name == "spi":
        result = spi(
            sample_monthly_precip_da,
            scale=6,
            distribution=indices.Distribution.gamma,
            output_scale=output_scale,
        )
    else:
        result = spei(
            sample_monthly_precip_da,
            sample_monthly_pet_da,
            scale=6,
            distribution=indices.Distribution.gamma,
            output_scale=output_scale,
        )
    assert isinstance(result, xr.DataArray)
    assert result.attrs["long_name"] == long_name
    assert result.attrs["units"] == "1"
    assert result.attrs["climate_indices_variant"] == output_scale


def test_xarray_probability_inverts_to_the_normal_output(sample_monthly_precip_da) -> None:
    probability = spi(
        sample_monthly_precip_da,
        scale=6,
        distribution=indices.Distribution.gamma,
        output_scale="probability",
    )
    z_score = spi(
        sample_monthly_precip_da,
        scale=6,
        distribution=indices.Distribution.gamma,
    )
    assert z_score.attrs["units"] == "dimensionless"

    reconstructed = norm.ppf(probability.values)
    within_clip = np.abs(z_score.values) < 3.09
    np.testing.assert_allclose(reconstructed[within_clip], z_score.values[within_clip], atol=1e-9)


def test_output_scale_maps_cover_every_declared_scale() -> None:
    """A scale added to OUTPUT_SCALES must be given a CF_METADATA entry and a CF variant."""
    from climate_indices.cf_metadata_registry import CF_METADATA, standardized_output_bounds
    from climate_indices.typed_public_api import _SPEI_CF_METADATA_VARIANTS, _SPI_CF_METADATA_VARIANTS

    for scale in compute.OUTPUT_SCALES:
        assert set(standardized_output_bounds(scale)) == {"valid_min", "valid_max"}
    for base in ("spi", "spei"):
        assert base in CF_METADATA
        for scale in set(compute.OUTPUT_SCALES) - {"normal"}:
            assert f"{base}_{scale}" in CF_METADATA
    for variants in (_SPI_CF_METADATA_VARIANTS, _SPEI_CF_METADATA_VARIANTS):
        assert {"normal"} | set(variants) == set(compute.OUTPUT_SCALES)


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
