import os

import numpy as np
import pytest

# Tests for `climate_indices.palmer.py`
ATOL = 5e-5
RTOL = 0

pytestmark = pytest.mark.validation

_FIXTURE_ROOT = os.path.join(os.path.split(__file__)[0], "fixture", "palmer")


# ---------------------------------------------------------------------------------------
def test_pdsi(
    palmer_pdsi_results,
    data_year_start_monthly,
    calibration_year_start_palmer,
    calibration_year_end_palmer,
    palmer_awcs,
):
    # Run test for each climate division, reusing the session-cached pdsi() sweep
    for test_id, (pdsi, phdi, pmdi, zindex, params) in palmer_pdsi_results.items():
        testpath = os.path.join(_FIXTURE_ROOT, test_id)
        awc = palmer_awcs[test_id]
        alphas = np.load(f"{testpath}/alphas.npy")
        betas = np.load(f"{testpath}/betas.npy")
        gammas = np.load(f"{testpath}/gammas.npy")
        deltas = np.load(f"{testpath}/deltas.npy")
        noaa_pdsi = np.load(f"{testpath}/pdsi.npy")
        noaa_phdi = np.load(f"{testpath}/phdi.npy")
        noaa_pmdi = np.load(f"{testpath}/pmdi.npy")
        noaa_zindex = np.load(f"{testpath}/zindex.npy")

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
