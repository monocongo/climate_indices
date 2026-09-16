import logging
from unittest import mock

import numpy as np
import pytest

from climate_indices import compute, indices
from climate_indices.exceptions import DataShapeError, InvalidArgumentError

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

    # The geographic poles are valid latitude coordinates.
    for polar_latitude in (-90.0, 90.0):
        result = indices.pet(temps_celsius, polar_latitude, data_year_start_monthly)
        assert result.size == temps_celsius.size

    # Invalid latitude values must not bypass validation for all-NaN temperatures.
    for invalid_latitude in (-91.0, 91.0):
        with pytest.raises(ValueError):
            indices.pet(all_nan_temps, invalid_latitude, data_year_start_monthly)

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


def test_pnp_calibration_period_extends_past_data():
    """Calibration windows with a trailing partial period still average per calendar time step."""
    # 481 monthly values starting 1900, i.e. 40 years plus one extra month
    values = np.arange(481, dtype=float)

    # the calibration period starts past the data start and ends past the data end,
    # so the calibration window is 121 values, i.e. 10 whole years plus one month
    computed_pnp = indices.percentage_of_normal(
        values,
        1,
        1900,
        1930,
        1969,
        compute.Periodicity.monthly,
    )

    # reference: per calendar time step average of the truncated calibration window
    calibration_period_sums = values[(1930 - 1900) * 12 :]
    averages = np.array([np.nanmean(calibration_period_sums[i::12]) for i in range(12)])
    expected = np.full(values.shape, np.nan)
    for i in range(values.size):
        divisor = averages[i % 12]
        if divisor > 0.0:
            expected[i] = values[i] / divisor

    np.testing.assert_allclose(computed_pnp, expected, equal_nan=True)


def test_pnp_calibration_period_beyond_data_returns_missing():
    """A calibration window past the end of the data yields all-NaN percentages, without raising."""
    values = np.arange(240, dtype=float)  # 20 years of monthly values starting 1900

    computed_pnp = indices.percentage_of_normal(
        values,
        1,
        1900,
        1921,
        1925,
        compute.Periodicity.monthly,
    )

    assert computed_pnp.shape == values.shape
    assert np.isnan(computed_pnp).all()


def test_pnp_2d_input_matches_flattened_1d():
    """A 2-D (years, periods) input is flattened, matching the equivalent 1-D series."""
    values = np.arange(240, dtype=float).reshape(20, 12)

    computed_2d = indices.percentage_of_normal(
        values,
        3,
        1900,
        1900,
        1919,
        compute.Periodicity.monthly,
    )
    computed_1d = indices.percentage_of_normal(
        values.flatten(),
        3,
        1900,
        1900,
        1919,
        compute.Periodicity.monthly,
    )

    assert computed_2d.shape == computed_1d.shape
    np.testing.assert_allclose(computed_2d, computed_1d, equal_nan=True)


def test_pnp_3d_input_raises():
    """An input array with more than two dimensions raises DataShapeError."""
    values = np.zeros((2, 3, 4))

    with pytest.raises(DataShapeError):
        indices.percentage_of_normal(
            values,
            1,
            1900,
            1900,
            1901,
            compute.Periodicity.monthly,
        )


def test_spi_ambiguous_3d_input_raises():
    """A gridded array whose first cell axis is the period length raises ValueError.

    That shape is equally readable as time-major (time, 12, *cells) and as the legacy
    (years, periods, *cells) layout, so it must be declared with spatial_time_major=True
    instead of being silently re-read along the wrong axis (#923).

    Unlike eddi()/percentage_of_normal(), spi()'s dimension errors are pinned to
    plain ValueError by tests/test_backward_compat.py::TestErrorHierarchyDocumented,
    so this stays on the shared preparation seam's ValueError rather than switching
    to DataShapeError.
    """
    values = np.zeros((2, 12, 4))

    with pytest.raises(ValueError, match="Invalid shape of input array"):
        indices.spi(
            values,
            1,
            indices.Distribution.gamma,
            1900,
            1900,
            1901,
            compute.Periodicity.monthly,
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

    # a gridded array whose first cell axis is the period length is ambiguous with a
    # (years, periods, *cells) array, so it has to be declared rather than read
    np.testing.assert_raises(
        ValueError,
        indices.spi,
        np.array(np.zeros((4, 366, 8))),
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
    np.testing.assert_raises(InvalidArgumentError, indices.pci, np.array(list(range(300))))


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "pet_thornthwaite_mm",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_fitting_indices_share_one_preparation_seam(
    precips_mm_monthly,
    pet_thornthwaite_mm,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """SPI, SPEI, EDDI, and PNP all prepare their scaled values through one seam."""
    precips = precips_mm_monthly.flatten()
    pet = pet_thornthwaite_mm.flatten()

    with mock.patch.object(compute, "prepare_scaled", wraps=compute.prepare_scaled) as prepare_scaled:
        indices.spi(
            precips,
            3,
            indices.Distribution.gamma,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )
        indices.spei(
            precips,
            pet,
            3,
            indices.Distribution.gamma,
            compute.Periodicity.monthly,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
        )
        indices.eddi(
            pet_thornthwaite_mm,
            3,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )
        indices.percentage_of_normal(
            precips,
            3,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )

    # SPI and EDDI prepare with the defaults; SPEI and PNP opt out of clipping and
    # reshaping. The spatial-block declaration is not part of that contract, so only
    # those two keywords are compared.
    assert prepare_scaled.call_count == 4
    prep_kwargs = [
        {key: value for key, value in call.kwargs.items() if key in {"clip_negatives", "reshape"}}
        for call in prepare_scaled.call_args_list
    ]
    assert prep_kwargs.count({}) == 2
    assert [call for call in prep_kwargs if call] == [{"clip_negatives": False, "reshape": False}] * 2


def test_fitting_indices_share_one_fit_seam(
    precips_mm_monthly,
    pet_thornthwaite_mm,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """SPI and SPEI both fit and standardize through one seam, stating the fall-back policy."""
    precips = precips_mm_monthly.flatten()
    pet = pet_thornthwaite_mm.flatten()

    with mock.patch.object(compute, "fit_and_standardize", wraps=compute.fit_and_standardize) as fit_and_standardize:
        indices.spi(
            precips,
            3,
            indices.Distribution.gamma,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )
        indices.spei(
            precips,
            pet,
            3,
            indices.Distribution.gamma,
            compute.Periodicity.monthly,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
        )

    assert fit_and_standardize.call_count == 2
    # SPI falls back from a failed Pearson Type III fit to gamma, SPEI does not
    assert [call.kwargs.get("fallback_to_gamma", False) for call in fit_and_standardize.call_args_list] == [
        True,
        False,
    ]


def test_spi_accepts_deprecated_fitting_parameter_keys(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """SPI normalizes its fitting parameters through the shared seam, so the deprecated
    aliases that SPEI accepts work here too, and the parameters given are the ones used."""
    precips = precips_mm_monthly.flatten()
    parameters = (
        (
            indices.Distribution.gamma,
            {"alpha": np.full(12, 4.0), "beta": np.full(12, 8.0)},
            {"alphas": np.full(12, 4.0), "betas": np.full(12, 8.0)},
        ),
        (
            indices.Distribution.pearson,
            {
                "prob_zero": np.full(12, 0.1),
                "loc": np.full(12, 1.0),
                "scale": np.full(12, 2.0),
                "skew": np.full(12, 0.5),
            },
            {
                "probabilities_of_zero": np.full(12, 0.1),
                "locs": np.full(12, 1.0),
                "scales": np.full(12, 2.0),
                "skews": np.full(12, 0.5),
            },
        ),
    )

    for distribution, canonical, deprecated in parameters:
        computed = [
            indices.spi(
                precips,
                6,
                distribution,
                data_year_start_monthly,
                calibration_year_start_monthly,
                calibration_year_end_monthly,
                compute.Periodicity.monthly,
                fitting_params,
            )
            for fitting_params in (canonical, deprecated)
        ]
        np.testing.assert_array_equal(computed[0], computed[1])

        # the supplied parameters are the ones used, rather than refitted from the data
        refitted = indices.spi(
            precips,
            6,
            distribution,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )
        assert not np.array_equal(computed[0], refitted)


def test_spei_accepts_explicit_none_fitting_parameters(
    precips_mm_monthly,
    pet_thornthwaite_mm,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """An explicit None for a canonical fitting-parameter key means "fit it from the
    data" rather than dropping the key, which used to raise KeyError."""
    precips = precips_mm_monthly.flatten()
    pet = pet_thornthwaite_mm.flatten()
    with_explicit_none = indices.spei(
        precips,
        pet,
        6,
        indices.Distribution.gamma,
        compute.Periodicity.monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        {"alpha": None, "beta": None},
    )
    without_parameters = indices.spei(
        precips,
        pet,
        6,
        indices.Distribution.gamma,
        compute.Periodicity.monthly,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    )
    np.testing.assert_array_equal(with_explicit_none, without_parameters)


def test_spatial_pearson_deprecated_fitting_keys_warn_once(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    """A deprecated fitting-parameter key warns once per top-level spatial operation,
    not once for every cell the Pearson Type III dispatch fits."""
    block = np.asarray(precips_mm_monthly).reshape(-1, 1, 1) * np.ones((1, 3, 2))
    deprecated = {"probabilities_of_zero": None, "locs": None, "scales": None, "skews": None}

    with mock.patch.object(compute, "_logger") as warning_logger:
        indices.spi(
            block,
            6,
            indices.Distribution.pearson,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            compute.Periodicity.monthly,
            deprecated,
            spatial_time_major=True,
        )
    assert warning_logger.warning.call_count == len(deprecated)

    with mock.patch.object(compute, "_logger") as warning_logger:
        indices.spei(
            block,
            np.full_like(block, 10.0),
            6,
            indices.Distribution.pearson,
            compute.Periodicity.monthly,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            deprecated,
            spatial_time_major=True,
        )
    assert warning_logger.warning.call_count == len(deprecated)
