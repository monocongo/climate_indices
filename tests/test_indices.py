import logging

import numpy as np
import pytest

from climate_indices import compute, indices

# disable logging messages
logging.disable(logging.CRITICAL)

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

    # invalid periodicity argument should raise an Error
    np.testing.assert_raises(
        ValueError,
        indices.percentage_of_normal,
        precips_mm_daily.flatten(),
        30,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        "unsupported_value",
    )

    # invalid scale argument should raise an Error
    np.testing.assert_raises(
        ValueError,
        indices.percentage_of_normal,
        precips_mm_daily.flatten(),
        -3,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )
    np.testing.assert_raises(
        ValueError,
        indices.percentage_of_normal,
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

    # invalid periodicity argument should raise a ValueError
    np.testing.assert_raises(
        ValueError,
        indices.spi,
        precips_mm_monthly.flatten(),
        6,
        indices.Distribution.gamma,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
        "unsupported_value",
    )

    # invalid distribution argument should raise a ValueError
    np.testing.assert_raises(
        ValueError,
        indices.spi,
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

    # invalid periodicity argument should raise a ValueError
    np.testing.assert_raises(
        ValueError,
        indices.spi,
        precips_mm_monthly.flatten(),
        6,
        indices.Distribution.pearson,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        "unsupported_value",
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

    # invalid periodicity argument should raise a ValueError
    np.testing.assert_raises(
        ValueError,
        indices.spei,
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
