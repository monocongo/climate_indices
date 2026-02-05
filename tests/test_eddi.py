import logging

import numpy as np
import pytest
from scipy import stats

from climate_indices import compute, indices

# disable logging messages
logging.disable(logging.CRITICAL)

UNEXPECTED_ALL_NANS_MESSAGE = "All-NaN input array does not result in the expected all-NaN result"


def test_hastings_inverse_normal_accuracy():
    """Test that Hastings approximation closely matches scipy.stats.norm.ppf()."""
    # import the private function for testing
    from climate_indices.indices import _hastings_inverse_normal

    # test across a range of probability values
    test_probs = np.array([0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999])

    # compute using Hastings approximation
    hastings_result = _hastings_inverse_normal(test_probs)

    # compute using scipy (reference implementation)
    scipy_result = stats.norm.ppf(test_probs)

    # verify close agreement (Hastings approximation is accurate to ~0.00045)
    # using a slightly more generous tolerance for edge cases
    np.testing.assert_allclose(
        hastings_result,
        scipy_result,
        rtol=1e-3,
        atol=1e-3,
        err_msg="Hastings approximation deviates significantly from scipy.stats.norm.ppf",
    )


@pytest.mark.usefixtures(
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_eddi_all_nans(
    pet_thornthwaite_mm,
    data_year_start_monthly,
    data_year_end_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """Test that all-NaN input returns all-NaN output."""
    # confirm that an input array of all NaNs for PET results in the same array returned
    all_nans = np.full(pet_thornthwaite_mm.shape, np.nan)
    computed_eddi = indices.eddi(
        pet_values=all_nans,
        scale=1,
        data_start_year=data_year_start_monthly,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
        periodicity=compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_eddi,
        all_nans.flatten(),
        equal_nan=True,
        err_msg=UNEXPECTED_ALL_NANS_MESSAGE,
    )


@pytest.mark.usefixtures(
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_eddi_negative_values(
    pet_thornthwaite_mm,
    data_year_start_monthly,
    data_year_end_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """Test that negative PET values are clipped to zero with warning."""
    # create array with some negative values
    pet_with_negatives = pet_thornthwaite_mm.copy()
    pet_with_negatives[10:20] = -5.0

    # compute EDDI - should not raise error, just log warning
    computed_eddi = indices.eddi(
        pet_values=pet_with_negatives,
        scale=1,
        data_start_year=data_year_start_monthly,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
        periodicity=compute.Periodicity.monthly,
    )

    # verify we got a result (not all NaN)
    assert not np.all(np.isnan(computed_eddi))


@pytest.mark.usefixtures(
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_eddi_invalid_array_shape(
    pet_thornthwaite_mm,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """Test that 3-D+ arrays are rejected with clear error."""
    # create a 3-D array (should raise ValueError)
    pet_3d = np.random.rand(10, 12, 5)

    with pytest.raises(ValueError, match="Invalid shape of input array"):
        indices.eddi(
            pet_values=pet_3d,
            scale=1,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity=compute.Periodicity.monthly,
        )


@pytest.mark.usefixtures(
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_eddi_basic_computation(
    pet_thornthwaite_mm,
    data_year_start_monthly,
    data_year_end_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """Test EDDI computation with non-parametric empirical ranking."""
    # compute EDDI at 1-month scale
    computed_eddi_1month = indices.eddi(
        pet_values=pet_thornthwaite_mm,
        scale=1,
        data_start_year=data_year_start_monthly,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
        periodicity=compute.Periodicity.monthly,
    )

    # verify output shape matches input
    assert computed_eddi_1month.shape == pet_thornthwaite_mm.flatten().shape

    # verify output is within valid range (±3.09)
    valid_values = computed_eddi_1month[~np.isnan(computed_eddi_1month)]
    assert np.all(valid_values >= -3.09)
    assert np.all(valid_values <= 3.09)

    # verify we have some non-NaN values
    assert np.sum(~np.isnan(computed_eddi_1month)) > 0

    # compute EDDI at 6-month scale
    computed_eddi_6month = indices.eddi(
        pet_values=pet_thornthwaite_mm.flatten(),
        scale=6,
        data_start_year=data_year_start_monthly,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
        periodicity=compute.Periodicity.monthly,
    )

    # verify output shape matches input
    assert computed_eddi_6month.shape == pet_thornthwaite_mm.flatten().shape

    # verify output is within valid range
    valid_values = computed_eddi_6month[~np.isnan(computed_eddi_6month)]
    assert np.all(valid_values >= -3.09)
    assert np.all(valid_values <= 3.09)


@pytest.mark.usefixtures(
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_eddi_hastings_approximation(
    pet_thornthwaite_mm,
    data_year_start_monthly,
    data_year_end_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """Test that Hastings approximation produces reasonable z-scores."""
    # compute EDDI
    computed_eddi = indices.eddi(
        pet_values=pet_thornthwaite_mm,
        scale=6,
        data_start_year=data_year_start_monthly,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
        periodicity=compute.Periodicity.monthly,
    )

    # verify output shape matches input
    assert computed_eddi.shape == pet_thornthwaite_mm.flatten().shape

    # verify output is within valid range (±3.09)
    valid_values = computed_eddi[~np.isnan(computed_eddi)]
    assert np.all(valid_values >= -3.09)
    assert np.all(valid_values <= 3.09)

    # verify we have some non-NaN values
    assert np.sum(~np.isnan(computed_eddi)) > 0

    # verify we get both positive and negative values (drought and wet conditions)
    assert np.any(valid_values > 0)
    assert np.any(valid_values < 0)


@pytest.mark.usefixtures(
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_eddi_invalid_periodicity(
    pet_thornthwaite_mm,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """Test that invalid periodicity raises ValueError."""
    with pytest.raises(ValueError, match="Invalid periodicity argument"):
        indices.eddi(
            pet_values=pet_thornthwaite_mm.flatten(),
            scale=6,
            data_start_year=data_year_start_monthly,
            calibration_year_initial=calibration_year_start_monthly,
            calibration_year_final=calibration_year_end_monthly,
            periodicity="unsupported_value",
        )


@pytest.mark.usefixtures(
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_eddi_empirical_ranking(
    pet_thornthwaite_mm,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """Test that EDDI uses empirical ranking (non-parametric approach)."""
    # create a simple test case with known values
    # 10 years of monthly data (120 values)
    num_years = 10
    num_months = 12
    pet_simple = np.ones(num_years * num_months) * 100.0

    # add variation to a specific month (e.g., January)
    for year in range(num_years):
        # vary January values: 80, 85, 90, 95, 100, 105, 110, 115, 120, 125
        pet_simple[year * num_months] = 80.0 + (year * 5.0)

    # compute EDDI using middle years for calibration
    computed_eddi = indices.eddi(
        pet_values=pet_simple,
        scale=1,
        data_start_year=1900,
        calibration_year_initial=1902,  # skip first 2 years
        calibration_year_final=1907,  # skip last 2 years
        periodicity=compute.Periodicity.monthly,
    )

    # verify we get results
    assert not np.all(np.isnan(computed_eddi))

    # verify the ranking logic: highest PET should give highest (most positive) EDDI
    # extract January values (indices 0, 12, 24, ...)
    january_eddi = computed_eddi[::num_months]

    # remove NaN values
    january_eddi_valid = january_eddi[~np.isnan(january_eddi)]

    # verify we have monotonic relationship (higher PET -> higher EDDI for January)
    # (allowing for some NaNs at the beginning due to calibration)
    if len(january_eddi_valid) > 2:
        # check that later years (higher PET) have higher EDDI than earlier years
        assert january_eddi_valid[-1] > january_eddi_valid[0]


@pytest.mark.usefixtures(
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_eddi_2d_array(
    pet_thornthwaite_mm,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """Test that 2-D arrays are properly flattened and processed."""
    # reshape to 2-D
    num_months = pet_thornthwaite_mm.size
    num_years = num_months // 12
    pet_2d = pet_thornthwaite_mm.reshape(num_years, 12)

    # compute EDDI with 2-D input
    computed_eddi = indices.eddi(
        pet_values=pet_2d,
        scale=1,
        data_start_year=data_year_start_monthly,
        calibration_year_initial=calibration_year_start_monthly,
        calibration_year_final=calibration_year_end_monthly,
        periodicity=compute.Periodicity.monthly,
    )

    # verify output is 1-D
    assert len(computed_eddi.shape) == 1

    # verify output has same total size as input
    assert computed_eddi.size == pet_2d.size
