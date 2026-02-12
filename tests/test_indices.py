import numpy as np
import pytest

from climate_indices import compute, indices
from climate_indices.exceptions import InvalidArgumentError

UNEXPECTED_ALL_NANS_MESSAGE = "All-NaN input array does not result in the expected all-NaN result"


@pytest.mark.usefixtures(
    "temps_celsius",
    "latitude_degrees",
    "data_year_start_monthly",
)
def test_pet(
    temps_celsius,
    latitude_degrees,
    data_year_start_monthly,
):
    # confirm that an input temperature array of only NaNs
    # results in the same all NaNs array being returned
    all_nan_temps = np.full(temps_celsius.shape, np.nan)
    computed_pet = indices.pet(all_nan_temps, latitude_degrees, data_year_start_monthly)
    np.testing.assert_equal(
        computed_pet,
        all_nan_temps,
        UNEXPECTED_ALL_NANS_MESSAGE,
    )

    # confirm that a masked input temperature array of
    # only NaNs results in the same masked array being returned
    masked_all_nan_temps = np.ma.array(all_nan_temps)
    computed_pet = indices.pet(masked_all_nan_temps, latitude_degrees, data_year_start_monthly)
    np.testing.assert_equal(
        computed_pet,
        masked_all_nan_temps,
        UNEXPECTED_ALL_NANS_MESSAGE,
    )

    # confirm that a missing/None latitude value raises an error
    np.testing.assert_raises(ValueError, indices.pet, temps_celsius, None, data_year_start_monthly)

    # confirm that a missing/None latitude value raises an error
    np.testing.assert_raises(ValueError, indices.pet, temps_celsius, np.nan, data_year_start_monthly)

    # confirm that an invalid latitude value raises an error
    pytest.raises(
        ValueError,
        indices.pet,
        temps_celsius,
        91.0,  # latitude > 90 is invalid
        data_year_start_monthly,
    )

    # confirm that an invalid latitude value raises an error
    np.testing.assert_raises(
        ValueError,
        indices.pet,
        temps_celsius,
        -91.0,  # latitude < -90 is invalid
        data_year_start_monthly,
    )

    # compute PET from the monthly temperatures, latitude, and initial years -- if this runs without
    # error then this test passes, as the underlying method(s) being used to compute PET will be tested
    # in the relevant test_compute.py or test_eto.py codes
    indices.pet(temps_celsius, latitude_degrees, data_year_start_monthly)

    # compute PET from the monthly temperatures, latitude (as an array), and initial years -- if this runs without
    # error then this test passes, as the underlying method(s) being used to compute PET will be tested
    # in the relevant test_compute.py or test_eto.py codes
    indices.pet(temps_celsius, np.array([latitude_degrees]), data_year_start_monthly)

    # verify that 1-element array latitude produces identical results to scalar latitude
    # this validates the removal of the size > 1 guard in indices.py:721
    pet_scalar = indices.pet(temps_celsius, latitude_degrees, data_year_start_monthly)
    pet_array_1elem = indices.pet(temps_celsius, np.array([latitude_degrees]), data_year_start_monthly)
    np.testing.assert_array_equal(
        pet_scalar,
        pet_array_1elem,
        err_msg="1-element array latitude should produce identical results to scalar latitude",
    )

    # confirm that an empty latitude array raises a controlled ValueError
    with pytest.raises(ValueError, match="empty latitude array"):
        indices.pet(temps_celsius, np.array([]), data_year_start_monthly)


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "precips_mm_daily",
    "data_year_start_monthly",
    "data_year_start_daily",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
    "calibration_year_start_daily",
    "calibration_year_end_daily",
    "pnp_6month",
)
def test_pnp(
    precips_mm_monthly,
    precips_mm_daily,
    data_year_start_monthly,
    data_year_start_daily,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    calibration_year_start_daily,
    calibration_year_end_daily,
    pnp_6month,
):
    # confirm that an input precipitation array containing
    # only NaNs results in the same array returned
    all_nan_precips = np.full(precips_mm_monthly.shape, np.nan)
    computed_pnp = indices.percentage_of_normal(
        all_nan_precips,
        1,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_pnp.flatten(),
        all_nan_precips.flatten(),
        equal_nan=True,
        err_msg=UNEXPECTED_ALL_NANS_MESSAGE,
    )

    # compute PNP from the daily precipitation array
    computed_pnp_6month = indices.percentage_of_normal(
        precips_mm_monthly.flatten(),
        6,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )

    # confirm PNP is being computed as expected
    np.testing.assert_allclose(
        pnp_6month.flatten(),
        computed_pnp_6month.flatten(),
        atol=0.01,
        equal_nan=True,
        err_msg="PNP values not computed as expected",
    )

    # confirm we can compute PNP from the daily values without raising an error
    indices.percentage_of_normal(
        precips_mm_daily.flatten(),
        30,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )

    # invalid periodicity argument should raise InvalidArgumentError
    with pytest.raises(InvalidArgumentError):
        indices.percentage_of_normal(
            precips_mm_daily.flatten(),
            30,
            data_year_start_daily,
            calibration_year_start_daily,
            calibration_year_end_daily,
            "unsupported_value",
        )

    # invalid scale argument should raise InvalidArgumentError
    with pytest.raises(InvalidArgumentError):
        indices.percentage_of_normal(
            precips_mm_daily.flatten(),
            -3,
            data_year_start_daily,
            calibration_year_start_daily,
            calibration_year_end_daily,
            compute.Periodicity.daily,
        )
    with pytest.raises(InvalidArgumentError):
        indices.percentage_of_normal(
            precips_mm_daily.flatten(),
            None,
            data_year_start_daily,
            calibration_year_start_daily,
            calibration_year_end_daily,
            compute.Periodicity.daily,
        )


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "precips_mm_daily",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "data_year_start_daily",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
    "calibration_year_start_daily",
    "calibration_year_end_daily",
    "spi_1_month_gamma",
    "spi_6_month_gamma",
    "spi_6_month_pearson3",
)
def test_spi(
    precips_mm_monthly,
    precips_mm_daily,
    data_year_start_monthly,
    data_year_end_monthly,
    data_year_start_daily,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    calibration_year_start_daily,
    calibration_year_end_daily,
    spi_1_month_gamma,
    spi_6_month_gamma,
    spi_6_month_pearson3,
) -> None:
    # confirm that an input array of all NaNs for
    # precipitation results in the same array returned
    all_nans = np.full(precips_mm_monthly.shape, np.nan)
    computed_spi = indices.spi(
        all_nans,
        1,
        indices.Distribution.gamma,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_spi,
        all_nans.flatten(),
        equal_nan=True,
        err_msg="SPI/Gamma not handling all-NaN arrays as expected",
    )

    # confirm SPI/gamma is being computed as expected
    computed_spi = indices.spi(
        precips_mm_monthly,
        1,
        indices.Distribution.gamma,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_spi,
        spi_1_month_gamma,
        atol=0.001,
        err_msg="SPI/Gamma values for 1-month scale not computed as expected",
    )

    # confirm SPI/gamma is being computed as expected
    computed_spi = indices.spi(
        precips_mm_monthly.flatten(),
        6,
        indices.Distribution.gamma,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.monthly,
    )

    # confirm SPI/gamma is being computed as expected
    np.testing.assert_allclose(
        computed_spi,
        spi_6_month_gamma,
        atol=0.001,
        err_msg="SPI/Gamma values for 6-month scale not computed as expected",
    )

    # confirm we can also call the function with daily data,
    # if this completes without error then test passes
    indices.spi(
        precips_mm_daily,
        30,
        indices.Distribution.gamma,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )

    # invalid periodicity argument should raise InvalidArgumentError
    with pytest.raises(InvalidArgumentError):
        indices.spi(
            precips_mm_monthly.flatten(),
            6,
            indices.Distribution.gamma,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
            "unsupported_value",
        )

    # invalid distribution argument should raise InvalidArgumentError
    with pytest.raises(InvalidArgumentError):
        indices.spi(
            precips_mm_monthly.flatten(),
            6,
            None,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
            compute.Periodicity.monthly,
        )

    # input array argument that's neither 1-D nor 2-D should raise a ValueError
    np.testing.assert_raises(
        ValueError,
        indices.spi,
        np.array(np.zeros((4, 4, 8))),
        6,
        indices.Distribution.gamma,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        compute.Periodicity.daily,
    )

    # compute SPI/Pearson at 60-day scale, just make sure it completes without error
    # TODO compare against expected results
    indices.spi(
        precips_mm_daily.flatten(),
        60,
        indices.Distribution.pearson,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )

    # confirm SPI/Pearson is being computed as expected
    computed_spi = indices.spi(
        precips_mm_monthly.flatten(),
        6,
        indices.Distribution.pearson,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    np.testing.assert_allclose(
        computed_spi,
        spi_6_month_pearson3,
        atol=0.01,
        err_msg="SPI/Pearson values for 6-month scale not computed as expected",
    )

    # confirm we can compute from daily values without raising an error
    indices.spi(
        precips_mm_daily.flatten(),
        60,
        indices.Distribution.pearson,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )

    # invalid periodicity argument should raise InvalidArgumentError
    with pytest.raises(InvalidArgumentError):
        indices.spi(
            precips_mm_monthly.flatten(),
            6,
            indices.Distribution.pearson,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            "unsupported_value",
        )


def test_masked_array_edge_cases(
    precips_mm_monthly,
    data_year_start_monthly,
    data_year_end_monthly,
) -> None:
    """
    Test MaskedArray edge cases to validate the isinstance(x, np.ma.MaskedArray)
    and x.mask.all() pattern used throughout the codebase.

    Tests three scenarios:
    (a) MaskedArray with mask=False (no values masked)
    (b) MaskedArray with partial mask (some values masked)
    (c) MaskedArray with full mask (all values masked)
    """
    # setup: create test data
    scale = 6
    distribution = indices.Distribution.gamma
    periodicity = compute.Periodicity.monthly

    # (a) MaskedArray with mask=False - should work normally
    masked_no_mask = np.ma.array(precips_mm_monthly, mask=False)
    result_no_mask = indices.spi(
        masked_no_mask,
        scale,
        distribution,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        periodicity,
    )
    # should produce valid results (not all NaN)
    assert not np.all(np.isnan(result_no_mask)), "MaskedArray with mask=False should produce valid results"

    # (b) MaskedArray with partial mask - should handle partially masked data
    partial_mask = np.zeros(precips_mm_monthly.shape, dtype=bool)
    # mask the first 10% of values
    mask_count = max(1, precips_mm_monthly.size // 10)
    partial_mask.flat[:mask_count] = True
    masked_partial = np.ma.array(precips_mm_monthly, mask=partial_mask)
    result_partial = indices.spi(
        masked_partial,
        scale,
        distribution,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        periodicity,
    )
    # should produce results with some valid values (not all NaN)
    assert not np.all(np.isnan(result_partial)), "MaskedArray with partial mask should produce some valid results"
    # some values may be NaN due to masking
    assert result_partial.size > 0, "Result should have non-zero size"

    # (c) MaskedArray with full mask - should return quickly without computation
    masked_full = np.ma.array(precips_mm_monthly, mask=True)
    result_full = indices.spi(
        masked_full,
        scale,
        distribution,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        periodicity,
    )
    # should return all NaN or all masked
    assert np.all(np.isnan(result_full)) or (isinstance(result_full, np.ma.MaskedArray) and result_full.mask.all()), (
        "MaskedArray with full mask should return all NaN or fully masked result"
    )

    # also test a compute function directly: transform_fitted_gamma
    # this validates the pattern at the compute module level
    # create monthly test data (12 months minimum required)
    test_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 15.0, 25.0, 35.0, 45.0, 55.0, 12.0, 22.0])

    # (a) no mask
    masked_values_no_mask = np.ma.array(test_values, mask=False)
    result_compute_no_mask = compute.transform_fitted_gamma(
        masked_values_no_mask,
        1900,  # data_start_year
        1900,  # calibration_start_year
        1901,  # calibration_end_year
        compute.Periodicity.monthly,
    )
    assert not np.all(np.isnan(result_compute_no_mask)), (
        "compute.transform_fitted_gamma with mask=False should produce valid results"
    )

    # (c) full mask - should trigger early return
    masked_values_full = np.ma.array(test_values, mask=True)
    result_compute_full = compute.transform_fitted_gamma(
        masked_values_full,
        1900,
        1900,
        1901,
        compute.Periodicity.monthly,
    )
    # should return the input masked array (early return)
    np.testing.assert_array_equal(
        result_compute_full,
        masked_values_full,
        err_msg="compute.transform_fitted_gamma should return input when fully masked",
    )


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "data_year_end_monthly",
    "spei_6_month_gamma",
    "spei_6_month_pearson3",
)
def test_spei(
    precips_mm_monthly,
    pet_thornthwaite_mm,
    data_year_start_monthly,
    data_year_end_monthly,
    spei_6_month_gamma,
    spei_6_month_pearson3,
) -> None:
    # confirm that an input precipitation array containing
    # only NaNs results in the same array being returned
    all_nans = np.full(precips_mm_monthly.shape, np.nan)
    computed_spei = indices.spei(
        all_nans,
        all_nans,
        1,
        indices.Distribution.gamma,
        compute.Periodicity.monthly,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        {"alpha": None, "beta": None},
    )
    np.testing.assert_allclose(
        computed_spei,
        all_nans,
        equal_nan=True,
        err_msg="SPEI/Gamma not handling all-NaN arrays as expected",
    )

    # compute SPEI/gamma at 6-month scale
    computed_spei = indices.spei(
        precips_mm_monthly,
        pet_thornthwaite_mm,
        6,
        indices.Distribution.gamma,
        compute.Periodicity.monthly,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        # Deprecated fitting keys
        {"alphas": None, "betas": None},
    )

    # confirm SPEI/gamma is being computed as expected
    np.testing.assert_allclose(
        computed_spei,
        spei_6_month_gamma,
        atol=0.01,
        err_msg="SPEI/Gamma values for 6-month scale not computed as expected",
    )

    # compute SPEI/Pearson at 6-month scale
    computed_spei = indices.spei(
        precips_mm_monthly,
        pet_thornthwaite_mm,
        6,
        indices.Distribution.pearson,
        compute.Periodicity.monthly,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        # Deprecated fitting keys
        {"probabilities_of_zero": None, "locs": None, "scales": None, "skews": None},
    )

    # confirm SPEI/Pearson is being computed as expected
    np.testing.assert_allclose(
        computed_spei,
        spei_6_month_pearson3,
        atol=0.01,
        err_msg="SPEI/Pearson values for 6-month scale not computed as expected",
    )

    # invalid periodicity argument should raise InvalidArgumentError
    with pytest.raises(InvalidArgumentError):
        indices.spei(
            precips_mm_monthly,
            pet_thornthwaite_mm,
            6,
            indices.Distribution.pearson,
            "unsupported_value",
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
        )

    # having both precipitation and PET input array arguments
    # with incongruent dimensions should raise a ValueError
    np.testing.assert_raises(
        ValueError,
        indices.spei,
        precips_mm_monthly,
        np.array((200, 200), dtype=float),
        6,
        indices.Distribution.pearson,
        compute.Periodicity.monthly,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
    )


@pytest.mark.usefixtures(
    "rain_mm",
    "rain_mm_365",
    "rain_mm_366",
)
def test_pci(
    rain_mm,
    rain_mm_365,
    rain_mm_366,
):
    # confirm that an input rainfall array of only NaNs
    # results in the same all NaNs array being returned
    all_nan_rainfall = np.full(rain_mm.shape, np.nan)
    computed_pci = indices.pci(all_nan_rainfall)
    np.testing.assert_equal(
        computed_pci,
        all_nan_rainfall,
        UNEXPECTED_ALL_NANS_MESSAGE,
    )

    # confirm that a masked input rainfall array of
    # only NaNs results in the same masked array being returned
    masked_all_nan_rainfall = np.ma.array(all_nan_rainfall)
    computed_pci = indices.pci(masked_all_nan_rainfall)
    np.testing.assert_equal(
        computed_pci,
        masked_all_nan_rainfall,
        "All-NaN masked input array does not result in the expected all-NaN masked result",
    )

    # Compute PCI for 366 days
    indices.pci(rain_mm_366[0])

    # Compute PCI for 365 days
    indices.pci(rain_mm_365[0])

    # confirm that an invalid number of days raises an error
    np.testing.assert_raises(ValueError, indices.pci, np.array(list(range(300))))
