import logging
import os

import numpy as np
import pytest
from glob import glob

from climate_indices import palmer

# Tests for `climate_indices.palmer.py`
ATOL = 5E-5
RTOL = 0

# ---------------------------------------------------------------------------------------
@pytest.mark.usefixtures(
    "data_year_start_monthly",
    "calibration_year_start_palmer",
    "calibration_year_end_palmer",
    "palmer_awcs",
)
def test_pdsi(
    data_year_start_monthly,
    calibration_year_start_palmer,
    calibration_year_end_palmer,
    palmer_awcs,
):
    # Run test for each climate division
    for testpath in glob(os.path.join(os.path.split(__file__)[0], "fixture", "palmer", "*")):
        test_id = testpath[-4:]
        awc = palmer_awcs[test_id]
        precips = np.load(f"{testpath}/precips.npy")
        pet = np.load(f"{testpath}/pet.npy")
        alphas = np.load(f"{testpath}/alphas.npy")
        betas = np.load(f"{testpath}/betas.npy")
        gammas = np.load(f"{testpath}/gammas.npy")
        deltas = np.load(f"{testpath}/deltas.npy")
        noaa_pdsi = np.load(f"{testpath}/pdsi.npy")
        noaa_phdi = np.load(f"{testpath}/phdi.npy")
        noaa_pmdi = np.load(f"{testpath}/pmdi.npy")
        noaa_zindex = np.load(f"{testpath}/zindex.npy")

        pdsi, phdi, pmdi, zindex, params = palmer.pdsi(
            precips,
            pet,
            awc,
            data_year_start_monthly,
            calibration_year_start_palmer,
            calibration_year_end_palmer,
        )

        np.testing.assert_allclose(
            pdsi,
            noaa_pdsi,
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=(
                f"{test_id}: PDSI not computed as expected, awc: {awc} "
                f"start: {data_year_start_monthly} calyr: {calibration_year_start_palmer} "
                f"calyrend: {calibration_year_end_palmer}"
            ),
        )

        np.testing.assert_allclose(
            phdi,
            noaa_phdi,
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=f"{test_id}: PHDI not computed as expected",
        )

        np.testing.assert_allclose(
            pmdi,
            noaa_pmdi,
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=f"{test_id}: PMDI not computed as expected",
        )

        np.testing.assert_allclose(
            zindex,
            noaa_zindex,
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=f"{test_id}: Z-Index not computed as expected",
        )

        np.testing.assert_allclose(
            params["alpha"],
            alphas,
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=f"{test_id}: Alphas not computed as expected",
        )

        np.testing.assert_allclose(
            params["beta"],
            betas,
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=f"{test_id}: Betas not computed as expected",
        )

        np.testing.assert_allclose(
            params["gamma"],
            gammas,
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=f"{test_id}: Gammas not computed as expected",
        )

        np.testing.assert_allclose(
            params["delta"],
            deltas,
            atol=ATOL,
            rtol=RTOL,
            equal_nan=True,
            err_msg=f"{test_id}: Deltas not computed as expected",
        )
