import logging

import numpy as np
import pytest

from climate_indices import palmer

# ----------------------------------------------------------------------------------------------------------------------
# disable logging messages
logging.disable(logging.CRITICAL)


"""
Tests for `palmer.py`.
"""


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
@pytest.mark.usefixtures(
    "palmer_pdsi_monthly",
    "palmer_phdi_monthly",
    "palmer_pmdi_monthly",
    "palmer_zindex_monthly",
)
def test_pdsi_from_zindex(
    palmer_pdsi_monthly, palmer_phdi_monthly, palmer_pmdi_monthly, palmer_zindex_monthly
):

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

    np.testing.assert_allclose(
        pmdi,
        palmer_pmdi_monthly,
        atol=0.01,
        equal_nan=True,
        err_msg="PMDI not computed as expected from monthly Z-Index fixture",
    )


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "palmer_precip_AL01",
    "palmer_pet_AL01",
    "palmer_et_AL01",
    "palmer_pr_AL01",
    "palmer_r_AL01",
    "palmer_ro_AL01",
    "palmer_pro_AL01",
    "palmer_l_AL01",
    "palmer_pl_AL01",
    "palmer_zindex_AL01",
    "data_year_start_palmer",
    "calibration_year_start_palmer",
    "calibration_year_end_palmer",
)
def test_z_index(
    palmer_precip_AL01,
    palmer_pet_AL01,
    palmer_et_AL01,
    palmer_pr_AL01,
    palmer_r_AL01,
    palmer_ro_AL01,
    palmer_pro_AL01,
    palmer_l_AL01,
    palmer_pl_AL01,
    palmer_zindex_AL01,
    data_year_start_palmer,
    calibration_year_start_palmer,
    calibration_year_end_palmer,
):
    """
    Test for the palmer._z_index() function
    """

    # call the _z_index() function
    zindex = palmer._z_index(
        palmer_precip_AL01,
        palmer_pet_AL01,
        palmer_et_AL01,
        palmer_pr_AL01,
        palmer_r_AL01,
        palmer_ro_AL01,
        palmer_pro_AL01,
        palmer_l_AL01,
        palmer_pl_AL01,
        data_year_start_palmer,
        calibration_year_start_palmer,
        calibration_year_end_palmer,
    )

    # compare against expected results
    np.testing.assert_allclose(
        zindex,
        palmer_zindex_AL01,
        atol=0.01,
        err_msg="Not computing the Z-Index as expected",
    )


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "palmer_alpha_AL01",
    "palmer_beta_AL01",
    "palmer_gamma_AL01",
    "palmer_delta_AL01",
    "palmer_precip_AL01",
    "palmer_pet_AL01",
    "palmer_r_AL01",
    "palmer_pr_AL01",
    "palmer_ro_AL01",
    "palmer_pro_AL01",
    "palmer_l_AL01",
    "palmer_pl_AL01",
    "palmer_K_AL01",
    "data_year_start_palmer",
    "calibration_year_start_palmer",
    "calibration_year_end_palmer",
)
def test_climatic_characteristic(
    palmer_alpha_AL01,
    palmer_beta_AL01,
    palmer_gamma_AL01,
    palmer_delta_AL01,
    palmer_precip_AL01,
    palmer_pet_AL01,
    palmer_r_AL01,
    palmer_pr_AL01,
    palmer_ro_AL01,
    palmer_pro_AL01,
    palmer_l_AL01,
    palmer_pl_AL01,
    palmer_K_AL01,
    data_year_start_palmer,
    calibration_year_start_palmer,
    calibration_year_end_palmer,
):
    """
    Test for the palmer._climatic_characteristic() function
    """

    # call the _cafec_coefficients() function
    palmer_K = palmer._climatic_characteristic(
        palmer_alpha_AL01,
        palmer_beta_AL01,
        palmer_gamma_AL01,
        palmer_delta_AL01,
        palmer_precip_AL01,
        palmer_pet_AL01,
        palmer_r_AL01,
        palmer_pr_AL01,
        palmer_ro_AL01,
        palmer_pro_AL01,
        palmer_l_AL01,
        palmer_pl_AL01,
        data_year_start_palmer,
        calibration_year_start_palmer,
        calibration_year_end_palmer,
    )

    # compare against expected results
    np.testing.assert_allclose(
        palmer_K, palmer_K_AL01, atol=0.01, err_msg="Not computing the K as expected"
    )


# # ------------------------------------------------------------------------------------------------------------------
# def test_cafec_compute_X(self):
#     """
#     Test for the palmer._compute_X() function
#     """
#
#     # simulate computation of X at an initial step (with all zeros for intermediate value arrays)
#     Z = palmer_zindex_monthly
#     k = 0
#     PPe = np.zeros(Z.shape)
#     X1 = 0.0
#     X2 = 0.0
#     PX1 = np.zeros(Z.shape)
#     PX2 = np.zeros(Z.shape)
#     PX3 = np.zeros(Z.shape)
#     X = np.zeros(Z.shape)
#     BT = np.zeros(Z.shape)
#     PX1, PX2, PX3, X, BT = palmer._compute_X(
#         Z, k, PPe, X1, X2, PX1, PX2, PX3, X, BT
#     )
#     .assertEqual(
#         PX1[0], 0.0, "PX1 value not computed as expected at initial step"
#     )
#     .assertEqual(
#         PX2[0], -0.34, "PX2 value not computed as expected at initial step"
#     )
#     .assertEqual(
#         PX3[0], 0.0, "PX3 value not computed as expected at initial step"
#     )
#     .assertEqual(
#         X[0], -0.34, "X value not computed as expected at initial step"
#     )
#     .assertEqual(
#         BT[0], 2, "Backtrack value not computed as expected at initial step"
#     )
#
# # ------------------------------------------------------------------------------------------------------------------
# def test_scpdsi():
#     """
#     Test for the palmer.scpdsi() function
#     """
#
#     scpdsi, pdsi, phdi, pmdi, zindex = palmer.scpdsi(
#         precips_mm_monthly,
#         pet_mm,
#         awc_inches,
#         data_year_start_monthly,
#         calibration_year_start_monthly,
#         calibration_year_end_monthly,
#     )
#
#     np.testing.assert_allclose(
#         scpdsi,
#         palmer_scpdsi_monthly,
#         atol=0.001,
#         equal_nan=True,
#         err_msg="PDSI not computed as expected from monthly inputs",
#     )
#
#     np.testing.assert_allclose(
#         phdi,
#         palmer_scphdi_monthly,
#         atol=0.001,
#         equal_nan=True,
#         err_msg="PHDI not computed as expected from monthly inputs",
#     )
#
#     np.testing.assert_allclose(
#         pmdi,
#         palmer_scpmdi_monthly,
#         atol=0.001,
#         equal_nan=True,
#         err_msg="PMDI not computed as expected from monthly inputs",
#     )
#
#     np.testing.assert_allclose(
#         zindex,
#         palmer_sczindex_monthly,
#         atol=0.001,
#         equal_nan=True,
#         err_msg="Z-Index not computed as expected from monthly inputs",
#     )
#
# ------------------------------------------------------------------------------------------------------------------
def test_cafec_coeff_ufunc():
    """
    Test for the palmer._cafec_coeff_ufunc() function
    """

    assert palmer._cafec_coeff_ufunc(0, 0) == 1
    assert palmer._cafec_coeff_ufunc(5, 0) == 0
    assert palmer._cafec_coeff_ufunc(5, 10) == 0.5


# # ------------------------------------------------------------------------------------------------------------------
# def test_cafec_coefficients():
#     """
#     Test for the palmer._cafec_coefficients() function
#     """
#
#     # call the _cafec_coefficients() function
#     alpha, beta, gamma, delta = palmer._cafec_coefficients(
#         palmer_pet_AL01,
#         palmer_et_AL01,
#         palmer_pr_AL01,
#         palmer_r_AL01,
#         palmer_ro_AL01,
#         palmer_pro_AL01,
#         palmer_l_AL01,
#         palmer_pl_AL01,
#         palmer_data_begin_year,
#         palmer_calibration_begin_year,
#         palmer_calibration_end_year,
#     )
#
#     # verify that the function performed as expected
#     arys = [
#         ["Alpha", alpha, palmer_alpha_AL01],
#         ["Beta", beta, palmer_beta_AL01],
#         ["Gamma", gamma, palmer_gamma_AL01],
#         ["Delta", delta, palmer_delta_AL01],
#     ]
#
#     for lst in arys:
#
#         name = lst[0]
#         actual = lst[1]
#         expected = lst[2]
#
#         # compare against expected results
#         np.testing.assert_allclose(
#             actual,
#             expected,
#             atol=0.01,
#             err_msg="Not computing the {0} as expected".format(name),
#         )


# ------------------------------------------------------------------------------------------------------------------
def test_phdi_select_ufunc():
    """
    Test for the palmer._phdi_select_ufunc() function
    """

    assert palmer._phdi_select_ufunc(0, 5) == 5
    assert palmer._phdi_select_ufunc(8, 5) == 8


# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "palmer_awc_AL01",
    "palmer_pet_AL01",
    "palmer_precip_AL01",
    "palmer_et_AL01",
    "palmer_pr_AL01",
    "palmer_r_AL01",
    "palmer_ro_AL01",
    "palmer_pro_AL01",
    "palmer_l_AL01",
    "palmer_pl_AL01",
)
def test_water_balance(
    palmer_awc_AL01,
    palmer_pet_AL01,
    palmer_precip_AL01,
    palmer_et_AL01,
    palmer_pr_AL01,
    palmer_r_AL01,
    palmer_ro_AL01,
    palmer_pro_AL01,
    palmer_l_AL01,
    palmer_pl_AL01,
):
    """
    Test for the palmer._water_balance() function
    """

    # call the water balance accounting function, providing AL-01 climate division input data
    palmer_ET, palmer_PR, palmer_R, palmer_RO, palmer_PRO, palmer_L, palmer_PL = palmer._water_balance(
        palmer_awc_AL01 + 1.0, palmer_pet_AL01, palmer_precip_AL01
    )

    arrays = [
        ["ET", palmer_ET, palmer_et_AL01],
        ["PR", palmer_PR, palmer_pr_AL01],
        ["R", palmer_R, palmer_r_AL01],
        ["RO", palmer_RO, palmer_ro_AL01],
        ["PRO", palmer_PRO, palmer_pro_AL01],
        ["L", palmer_L, palmer_l_AL01],
        ["PL", palmer_PL, palmer_pl_AL01],
    ]

    # verify that the function performed as expected
    for lst in arrays:
        name = lst[0]
        actual = lst[1]
        expected = lst[2]

        np.testing.assert_allclose(
            actual,
            expected,
            atol=0.01,
            err_msg="Not computing the {0} as expected".format(name),
        )

    # verify that the function can be called with an AWC value of zero (no error == passed test)
    palmer._water_balance(0.0, palmer_pet_AL01, palmer_precip_AL01)
