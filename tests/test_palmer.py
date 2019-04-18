import logging

import numpy as np
import pytest

from climate_indices import palmer

# ----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


# Tests for `climate_indices.palmer.py`


# ---------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "pet_thornthwaite_mm",
    "awc_inches",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
    "palmer_pdsi_monthly",
    "palmer_phdi_monthly",
    "palmer_pmdi_monthly",
    "palmer_zindex_monthly",
)
def test_pdsi(
    precips_mm_monthly,
    pet_thornthwaite_mm,
    awc_inches,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    palmer_pdsi_monthly,
    palmer_phdi_monthly,
    palmer_pmdi_monthly,
    palmer_zindex_monthly,
):

    pdsi, phdi, pmdi, zindex = palmer.pdsi(
        precips_mm_monthly,
        pet_thornthwaite_mm,
        awc_inches,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    )

    np.testing.assert_allclose(
        pdsi,
        palmer_pdsi_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="PDSI not computed as expected from monthly inputs",
    )

    np.testing.assert_allclose(
        phdi,
        palmer_phdi_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="PHDI not computed as expected from monthly inputs",
    )

    np.testing.assert_allclose(
        pmdi,
        palmer_pmdi_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="PMDI not computed as expected from monthly inputs",
    )

    np.testing.assert_allclose(
        zindex,
        palmer_zindex_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="Z-Index not computed as expected from monthly inputs",
    )


# ---------------------------------------------------------------------------------------
@pytest.mark.usefixtures("palmer_pdsi_monthly",
                         "palmer_phdi_monthly",
                         "palmer_pmdi_monthly",
                         "palmer_zindex_monthly")
def test_pdsi_from_zindex(palmer_pdsi_monthly,
                          palmer_phdi_monthly,
                          palmer_pmdi_monthly,
                          palmer_zindex_monthly):

    pdsi, phdi, pmdi = palmer._pdsi_from_zindex(palmer_zindex_monthly)

    np.testing.assert_allclose(
        pdsi,
        palmer_pdsi_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="PDSI not computed as expected from monthly Z-Index fixture",
    )

    np.testing.assert_allclose(
        phdi,
        palmer_phdi_monthly,
        atol=0.01,
        equal_nan=True,
        err_msg="PHDI not computed as expected from monthly Z-Index fixture",
    )

    np.testing.assert_allclose(pmdi,
                               palmer_pmdi_monthly,
                               atol=0.01,
                               equal_nan=True,
                               err_msg="PMDI not computed as expected "
                                       "from monthly Z-Index fixture")


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "palmer_precip",
    "palmer_pet",
    "palmer_et",
    "palmer_pr",
    "palmer_r",
    "palmer_ro",
    "palmer_pro",
    "palmer_l",
    "palmer_pl",
    "palmer_zindex",
    "data_year_start_palmer",
    "calibration_year_start_palmer",
    "calibration_year_end_palmer",
)
def test_z_index(
    palmer_precip,
    palmer_pet,
    palmer_et,
    palmer_pr,
    palmer_r,
    palmer_ro,
    palmer_pro,
    palmer_l,
    palmer_pl,
    palmer_zindex,
    data_year_start_palmer,
    calibration_year_start_palmer,
    calibration_year_end_palmer,
):
    """
    Test for the palmer._z_index() function
    """

    # call the _z_index() function
    zindex = palmer._z_index(palmer_precip,
                             palmer_pet,
                             palmer_et,
                             palmer_pr,
                             palmer_r,
                             palmer_ro,
                             palmer_pro,
                             palmer_l,
                             palmer_pl,
                             data_year_start_palmer,
                             calibration_year_start_palmer,
                             calibration_year_end_palmer)

    # compare against expected results
    np.testing.assert_allclose(zindex,
                               palmer_zindex,
                               atol=0.01,
                               err_msg="Not computing the Z-Index as expected")


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures("palmer_alpha",
                         "palmer_beta",
                         "palmer_gamma",
                         "palmer_delta",
                         "palmer_precip",
                         "palmer_pet",
                         "palmer_r",
                         "palmer_pr",
                         "palmer_ro",
                         "palmer_pro",
                         "palmer_l",
                         "palmer_pl",
                         "palmer_K",
                         "data_year_start_palmer",
                         "calibration_year_start_palmer",
                         "calibration_year_end_palmer")
def test_climatic_characteristic(palmer_alpha,
                                 palmer_beta,
                                 palmer_gamma,
                                 palmer_delta,
                                 palmer_precip,
                                 palmer_pet,
                                 palmer_r,
                                 palmer_pr,
                                 palmer_ro,
                                 palmer_pro,
                                 palmer_l,
                                 palmer_pl,
                                 palmer_K,
                                 data_year_start_palmer,
                                 calibration_year_start_palmer,
                                 calibration_year_end_palmer):
    """
    Test for the palmer._climatic_characteristic() function
    """

    # call the _cafec_coefficients() function
    computed_palmer_K = \
        palmer._climatic_characteristic(palmer_alpha,
                                        palmer_beta,
                                        palmer_gamma,
                                        palmer_delta,
                                        palmer_precip,
                                        palmer_pet,
                                        palmer_r,
                                        palmer_pr,
                                        palmer_ro,
                                        palmer_pro,
                                        palmer_l,
                                        palmer_pl,
                                        data_year_start_palmer,
                                        calibration_year_start_palmer,
                                        calibration_year_end_palmer)

    # compare against expected results
    np.testing.assert_allclose(computed_palmer_K,
                               palmer_K,
                               atol=0.01,
                               err_msg="Not computing the K as expected"
    )


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures("palmer_zindex_monthly")
def test_cafec_compute_X(palmer_zindex_monthly):
    """
    Test for the palmer._compute_X() function
    """

    # simulate computation of X at an initial step (with all zeros for intermediate value arrays)
    k = 0
    PPe = np.zeros(palmer_zindex_monthly.shape)
    X1 = 0.0
    X2 = 0.0
    PX1 = np.zeros(palmer_zindex_monthly.shape)
    PX2 = np.zeros(palmer_zindex_monthly.shape)
    PX3 = np.zeros(palmer_zindex_monthly.shape)
    X = np.zeros(palmer_zindex_monthly.shape)
    BT = np.zeros(palmer_zindex_monthly.shape)
    PX1, PX2, PX3, X, BT = palmer._compute_X(
        palmer_zindex_monthly, k, PPe, X1, X2, PX1, PX2, PX3, X, BT
    )
    if PX1[0] != 0.0:
        raise AssertionError("PX1 value not computed as expected at initial step")
    if PX2[0] != -0.34:
        raise AssertionError("PX2 value not computed as expected at initial step")
    if PX3[0] != 0.0:
        raise AssertionError("PX3 value not computed as expected at initial step")
    if X[0] != -0.34:
        raise AssertionError("X value not computed as expected at initial step")
    if BT[0] != 2:
        raise AssertionError("Backtrack value not computed as expected at initial step")


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "pet_thornthwaite_mm",
    "awc_inches",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
    "palmer_pdsi_from_scpdsi_monthly",
    "palmer_scpdsi_monthly",
    "palmer_scphdi_monthly",
    "palmer_scpmdi_monthly",
    "palmer_sczindex_monthly",
)
def test_scpdsi(
    precips_mm_monthly,
    pet_thornthwaite_mm,
    awc_inches,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    palmer_pdsi_from_scpdsi_monthly,
    palmer_scpdsi_monthly,
    palmer_scphdi_monthly,
    palmer_scpmdi_monthly,
    palmer_sczindex_monthly,
):
    """
    Test for the palmer.scpdsi() function
    """

    scpdsi, pdsi, phdi, pmdi, zindex = palmer.scpdsi(
        precips_mm_monthly,
        pet_thornthwaite_mm,
        awc_inches,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
    )

    np.testing.assert_allclose(
        scpdsi,
        palmer_scpdsi_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="PDSI not computed as expected from monthly inputs",
    )

    np.testing.assert_allclose(
        pdsi,
        palmer_pdsi_from_scpdsi_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="PDSI not computed as expected from monthly inputs",
    )

    np.testing.assert_allclose(
        phdi,
        palmer_scphdi_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="PHDI not computed as expected from monthly inputs",
    )

    np.testing.assert_allclose(
        pmdi,
        palmer_scpmdi_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="PMDI not computed as expected from monthly inputs",
    )

    np.testing.assert_allclose(
        zindex,
        palmer_sczindex_monthly,
        atol=0.001,
        equal_nan=True,
        err_msg="Z-Index not computed as expected from monthly inputs",
    )


# ------------------------------------------------------------------------------------------------------------------
def test_cafec_coeff_ufunc():
    """
    Test for the palmer._cafec_coeff_ufunc() function
    """

    if palmer._cafec_coeff_ufunc(0, 0) != 1:
        raise AssertionError()
    if palmer._cafec_coeff_ufunc(5, 0) != 0:
        raise AssertionError()
    if palmer._cafec_coeff_ufunc(5, 10) != 0.5:
        raise AssertionError()


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "palmer_alpha",
    "palmer_beta",
    "palmer_gamma",
    "palmer_delta",
    "palmer_pet",
    "palmer_et",
    "palmer_pr",
    "palmer_r",
    "palmer_ro",
    "palmer_pro",
    "palmer_l",
    "palmer_pl",
    "data_year_start_palmer",
    "calibration_year_start_palmer",
    "calibration_year_end_palmer",
)
def test_cafec_coefficients(
    palmer_alpha,
    palmer_beta,
    palmer_gamma,
    palmer_delta,
    palmer_pet,
    palmer_et,
    palmer_pr,
    palmer_r,
    palmer_ro,
    palmer_pro,
    palmer_l,
    palmer_pl,
    data_year_start_palmer,
    calibration_year_start_palmer,
    calibration_year_end_palmer,
):
    """
    Test for the palmer._cafec_coefficients() function
    """

    # call the _cafec_coefficients() function
    alpha, beta, gamma, delta = palmer._cafec_coefficients(
        palmer_pet,
        palmer_et,
        palmer_pr,
        palmer_r,
        palmer_ro,
        palmer_pro,
        palmer_l,
        palmer_pl,
        data_year_start_palmer,
        calibration_year_start_palmer,
        calibration_year_end_palmer,
    )

    # verify that the function performed as expected
    arrays = [
        ["Alpha", alpha, palmer_alpha],
        ["Beta", beta, palmer_beta],
        ["Gamma", gamma, palmer_gamma],
        ["Delta", delta, palmer_delta],
    ]

    for lst in arrays:

        name = lst[0]
        actual = lst[1]
        expected = lst[2]

        # compare against expected results
        np.testing.assert_allclose(
            actual,
            expected,
            atol=0.01,
            err_msg="Not computing the {0} as expected".format(name),
        )


# ------------------------------------------------------------------------------------------------------------------
def test_phdi_select_ufunc():
    """
    Test for the palmer._phdi_select_ufunc() function
    """

    if palmer._phdi_select_ufunc(0, 5) != 5:
        raise AssertionError()
    if palmer._phdi_select_ufunc(8, 5) != 8:
        raise AssertionError()


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "palmer_awc",
    "palmer_pet",
    "palmer_precip",
    "palmer_et",
    "palmer_pr",
    "palmer_r",
    "palmer_ro",
    "palmer_pro",
    "palmer_l",
    "palmer_pl",
)
def test_water_balance(palmer_awc,
                       palmer_pet,
                       palmer_precip,
                       palmer_et,
                       palmer_pr,
                       palmer_r,
                       palmer_ro,
                       palmer_pro,
                       palmer_l,
                       palmer_pl):
    """
    Test for the palmer._water_balance() function
    """

    # call the water balance accounting function, providing AL-01 climate division input data
    palmer_ET, palmer_PR, palmer_R, palmer_RO, palmer_PRO, palmer_L, palmer_PL = palmer._water_balance(
        palmer_awc + 1.0, palmer_pet, palmer_precip
    )

    arrays = [
        ["ET", palmer_ET, palmer_et],
        ["PR", palmer_PR, palmer_pr],
        ["R", palmer_R, palmer_r],
        ["RO", palmer_RO, palmer_ro],
        ["PRO", palmer_PRO, palmer_pro],
        ["L", palmer_L, palmer_l],
        ["PL", palmer_PL, palmer_pl],
    ]

    # verify that the function performed as expected
    for lst in arrays:
        name = lst[0]
        actual = lst[1]
        expected = lst[2]

        np.testing.assert_allclose(actual,
                                   expected,
                                   atol=0.01,
                                   err_msg=f"Not computing the {name} as expected")

    # verify that the function can be called with an AWC value of zero (no error == passed test)
    palmer._water_balance(0.0, palmer_pet, palmer_precip)
