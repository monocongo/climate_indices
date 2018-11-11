import logging
import numpy as np
import unittest

from tests import fixtures
from climate_indices import compute, indices

# ----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# ----------------------------------------------------------------------------------------------------------------------
class IndicesTestCase(fixtures.FixturesTestCase):
    """
    Tests for `indices.py`.
    """

    # ---------------------------------------------------------------------------------------
    def test_pdsi(self):

        # the indices.pdsi() function is a wrapper for palmer.pdsi(), so we'll
        # just confirm that this function can be called without raising an error and
        # the compute.pdsi() function itself being tested within test_palmer.py
        indices.pdsi(
            self.fixture_precips_mm_monthly,
            self.fixture_pet_mm,
            self.fixture_awc_inches,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
        )

    # ---------------------------------------------------------------------------------------
    def test_scpdsi(self):

        # the indices.scpdsi() function is a wrapper for palmer.pdsi(), so we'll
        # just confirm that this function can be called without raising an error and
        # the compute.pdsi() function itself being tested within test_palmer.py
        indices.scpdsi(
            self.fixture_precips_mm_monthly,
            self.fixture_pet_mm,
            self.fixture_awc_inches,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
        )

    # ----------------------------------------------------------------------------------------
    def test_pet(self):

        # confirm that an input array of all NaNs for temperature results in the same array returned
        all_nan_temps = np.full(self.fixture_temps_celsius.shape, np.NaN)
        computed_pet = indices.pet(
            all_nan_temps,
            self.fixture_latitude_degrees,
            self.fixture_data_year_start_monthly,
        )
        np.testing.assert_equal(
            computed_pet,
            all_nan_temps,
            "All-NaN input array does not result in the expected all-NaN result",
        )

        # confirm that a masked input array of all NaNs for temperature results in the same masked array returned
        masked_all_nan_temps = np.ma.array(all_nan_temps)
        computed_pet = indices.pet(
            masked_all_nan_temps,
            self.fixture_latitude_degrees,
            self.fixture_data_year_start_monthly,
        )
        np.testing.assert_equal(
            computed_pet,
            masked_all_nan_temps,
            "All-NaN masked input array does not result in the expected all-NaN masked result",
        )

        # confirm that a missing/None latitude value raises an error
        np.testing.assert_raises(
            ValueError,
            indices.pet,
            self.fixture_temps_celsius,
            None,
            self.fixture_data_year_start_monthly,
        )

        # confirm that a missing/None latitude value raises an error
        np.testing.assert_raises(
            ValueError,
            indices.pet,
            self.fixture_temps_celsius,
            np.NaN,
            self.fixture_data_year_start_monthly,
        )

        # confirm that an invalid latitude value raises an error
        self.assertRaises(
            ValueError,
            indices.pet,
            self.fixture_temps_celsius,
            91.0,  # latitude > 90 is invalid
            self.fixture_data_year_start_monthly,
        )

        # confirm that an invalid latitude value raises an error
        np.testing.assert_raises(
            ValueError,
            indices.pet,
            self.fixture_temps_celsius,
            -91.0,  # latitude < -90 is invalid
            self.fixture_data_year_start_monthly,
        )

        # compute PET from the monthly temperatures, latitude, and initial years -- if this runs without
        # error then this test passes, as the underlying method(s) being used to compute PET will be tested
        # in the relevant test_compute.py or test_eto.py codes
        indices.pet(
            self.fixture_temps_celsius,
            self.fixture_latitude_degrees,
            self.fixture_data_year_start_monthly,
        )

        # compute PET from the monthly temperatures, latitude (as an array), and initial years -- if this runs without
        # error then this test passes, as the underlying method(s) being used to compute PET will be tested
        # in the relevant test_compute.py or test_eto.py codes
        indices.pet(
            self.fixture_temps_celsius,
            np.array([self.fixture_latitude_degrees]),
            self.fixture_data_year_start_monthly,
        )

    # ----------------------------------------------------------------------------------------
    def test_pnp(self):

        # confirm that an input array of all NaNs for precipitation results in the same array returned
        all_nan_precips = np.full(self.fixture_precips_mm_monthly.shape, np.NaN)
        computed_pnp = indices.percentage_of_normal(
            all_nan_precips,
            1,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_pnp.flatten(),
            all_nan_precips.flatten(),
            equal_nan=True,
            err_msg="All-NaN input array does not result in the expected all-NaN result",
        )

        # compute PNP from the daily precipitation array
        computed_pnp_6month = indices.percentage_of_normal(
            self.fixture_precips_mm_monthly.flatten(),
            6,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )

        # confirm PNP is being computed as expected
        np.testing.assert_allclose(
            self.fixture_pnp_6month.flatten(),
            computed_pnp_6month.flatten(),
            atol=0.01,
            equal_nan=True,
            err_msg="PNP values not computed as expected",
        )

        # confirm we can compute PNP from the daily values without raising an error
        indices.percentage_of_normal(
            self.fixture_precips_mm_daily.flatten(),
            30,
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.daily,
        )

        # invalid periodicity argument should raise an Error
        np.testing.assert_raises(
            ValueError,
            indices.percentage_of_normal,
            self.fixture_precips_mm_daily.flatten(),
            30,
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            "unsupported_value",
        )

        # invalid scale argument should raise an Error
        np.testing.assert_raises(
            ValueError,
            indices.percentage_of_normal,
            self.fixture_precips_mm_daily.flatten(),
            -3,
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.daily,
        )
        np.testing.assert_raises(
            ValueError,
            indices.percentage_of_normal,
            self.fixture_precips_mm_daily.flatten(),
            None,
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.daily,
        )

    # ----------------------------------------------------------------------------------------
    def test_spi(self):

        # confirm that an input array of all NaNs for precipitation results in the same array returned
        all_nans = np.full(self.fixture_precips_mm_monthly.shape, np.NaN)
        computed_spi = indices.spi(
            all_nans,
            1,
            indices.Distribution.gamma,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
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
            self.fixture_precips_mm_monthly,
            1,
            indices.Distribution.gamma,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_spi,
            self.fixture_spi_1_month_gamma,
            atol=0.001,
            err_msg="SPI/Gamma values for 1-month scale not computed as expected",
        )

        # confirm SPI/gamma is being computed as expected
        computed_spi = indices.spi(
            self.fixture_precips_mm_monthly.flatten(),
            6,
            indices.Distribution.gamma,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
            compute.Periodicity.monthly,
        )

        # confirm SPI/gamma is being computed as expected
        np.testing.assert_allclose(
            computed_spi,
            self.fixture_spi_6_month_gamma,
            atol=0.001,
            err_msg="SPI/Gamma values for 6-month scale not computed as expected",
        )

        # confirm we can also call the function with daily data, if this completes without error then test passes
        indices.spi(
            self.fixture_precips_mm_daily,
            30,
            indices.Distribution.gamma,
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.daily,
        )

        # invalid periodicity argument should raise a ValueError
        np.testing.assert_raises(
            ValueError,
            indices.spi,
            self.fixture_precips_mm_monthly.flatten(),
            6,
            indices.Distribution.gamma,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
            "unsupported_value",
        )

        # invalid distribution argument should raise a ValueError
        np.testing.assert_raises(
            ValueError,
            indices.spi,
            self.fixture_precips_mm_monthly.flatten(),
            6,
            None,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
            compute.Periodicity.monthly,
        )

        # input array argument that's neither 1-D nor 2-D should raise a ValueError
        np.testing.assert_raises(
            ValueError,
            indices.spi,
            np.array(np.zeros((4, 4, 8))),
            6,
            indices.Distribution.gamma,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
            compute.Periodicity.daily,
        )

        # compute SPI/Pearson at 60-day scale, just make sure it completes without error
        # TODO compare against expected results
        indices.spi(
            self.fixture_precips_mm_daily.flatten(),
            60,
            indices.Distribution.pearson,
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.daily,
        )

        # confirm SPI/Pearson is being computed as expected
        computed_spi = indices.spi(
            self.fixture_precips_mm_monthly.flatten(),
            6,
            indices.Distribution.pearson,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )
        np.testing.assert_allclose(
            computed_spi,
            self.fixture_spi_6_month_pearson3,
            atol=0.01,
            err_msg="SPI/Pearson values for 6-month scale not computed as expected",
        )

        # confirm we can compute from daily values without raising an error
        indices.spi(
            self.fixture_precips_mm_daily.flatten(),
            60,
            indices.Distribution.pearson,
            self.fixture_data_year_start_daily,
            self.fixture_calibration_year_start_daily,
            self.fixture_calibration_year_end_daily,
            compute.Periodicity.daily,
        )

        # invalid periodicity argument should raise a ValueError
        np.testing.assert_raises(
            ValueError,
            indices.spi,
            self.fixture_precips_mm_monthly.flatten(),
            6,
            indices.Distribution.pearson,
            self.fixture_data_year_start_monthly,
            self.fixture_calibration_year_start_monthly,
            self.fixture_calibration_year_end_monthly,
            "unsupported_value",
        )

    # ----------------------------------------------------------------------------------------
    def test_spei(self):

        # confirm that an input array of all NaNs for precipitation results in the same array returned
        all_nans = np.full(self.fixture_precips_mm_monthly.shape, np.NaN)
        computed_spei = indices.spei(
            all_nans,
            all_nans,
            1,
            indices.Distribution.gamma,
            compute.Periodicity.monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
        )
        np.testing.assert_allclose(
            computed_spei,
            all_nans,
            equal_nan=True,
            err_msg="SPEI/Gamma not handling all-NaN arrays as expected",
        )

        # compute SPEI/gamma at 6-month scale
        computed_spei = indices.spei(
            self.fixture_precips_mm_monthly,
            self.fixture_pet_mm,
            6,
            indices.Distribution.gamma,
            compute.Periodicity.monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
        )

        # confirm SPEI/gamma is being computed as expected
        np.testing.assert_allclose(
            computed_spei,
            self.fixture_spei_6_month_gamma,
            atol=0.01,
            err_msg="SPEI/Gamma values for 6-month scale not computed as expected",
        )

        # compute SPEI/Pearson at 6-month scale
        computed_spei = indices.spei(
            self.fixture_precips_mm_monthly,
            self.fixture_pet_mm,
            6,
            indices.Distribution.pearson,
            compute.Periodicity.monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
        )

        # confirm SPEI/Pearson is being computed as expected
        np.testing.assert_allclose(
            computed_spei,
            self.fixture_spei_6_month_pearson3,
            atol=0.01,
            err_msg="SPEI/Pearson values for 6-month scale not computed as expected",
        )

        # invalid periodicity argument should raise a ValueError
        np.testing.assert_raises(
            ValueError,
            indices.spei,
            self.fixture_precips_mm_monthly,
            self.fixture_pet_mm,
            6,
            indices.Distribution.pearson,
            "unsupported_value",
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
        )

        # having both precipitation and PET input array arguments with incongruent dimensions should raise a ValueError
        np.testing.assert_raises(
            ValueError,
            indices.spei,
            self.fixture_precips_mm_monthly,
            np.array((200, 200), dtype=float),
            6,
            indices.Distribution.pearson,
            compute.Periodicity.monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_start_monthly,
            self.fixture_data_year_end_monthly,
        )


# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
