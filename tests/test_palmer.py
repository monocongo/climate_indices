import os
from glob import glob

import numpy as np
import pytest

from climate_indices import palmer

# Tests for `climate_indices.palmer.py`
ATOL = 5e-5
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


def test_monthly_climatology_prefers_calibration_window() -> None:
    values = np.concatenate([np.arange(12, dtype=float), np.arange(12, dtype=float) + 100.0])
    climatology = palmer._compute_monthly_climatology(
        values,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2000,
    )
    np.testing.assert_allclose(climatology, np.arange(12, dtype=float))


def test_monthly_climatology_fallbacks_to_full_period() -> None:
    values = np.concatenate([np.full(12, np.nan), np.arange(12, dtype=float) + 100.0])
    climatology = palmer._compute_monthly_climatology(
        values,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2000,
    )
    np.testing.assert_allclose(climatology, np.arange(12, dtype=float) + 100.0)


def test_missing_policy_climatology_fills_values() -> None:
    precips = np.full(24, 1.0)
    pet = np.full(24, 0.5)
    precips[0:12] = np.nan
    pet[0:12] = np.nan
    precip_climatology = palmer._compute_monthly_climatology(
        precips,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
    )
    pet_climatology = palmer._compute_monthly_climatology(
        pet,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
    )
    filled_precips = palmer._fill_missing_with_climatology(precips, precip_climatology, precips.size)
    filled_pet = palmer._fill_missing_with_climatology(pet, pet_climatology, pet.size)

    pdsi_filled, _, _, _, _ = palmer.pdsi(
        filled_precips,
        filled_pet,
        awc=1.0,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        missing_policy="strict",
    )
    pdsi_climo, _, _, _, _ = palmer.pdsi(
        precips,
        pet,
        awc=1.0,
        data_start_year=2000,
        calibration_year_initial=2000,
        calibration_year_final=2001,
        missing_policy="climatology",
    )

    np.testing.assert_allclose(pdsi_climo, pdsi_filled, equal_nan=True)


def test_leap_year_rule_affects_daily_aggregation() -> None:
    daily_noaa = np.ones(366)
    monthly_noaa = palmer._aggregate_daily_to_monthly(
        daily_noaa,
        data_start_year=1900,
        leap_year_rule="noaa",
    )
    assert monthly_noaa[1] == 29.0

    daily_gregorian = np.ones(365)
    monthly_gregorian = palmer._aggregate_daily_to_monthly(
        daily_gregorian,
        data_start_year=1900,
        leap_year_rule="gregorian",
    )
    assert monthly_gregorian[1] == 28.0


def test_pet_source_requires_inputs() -> None:
    precips = np.full(24, 1.0)
    temps = np.full(24, 10.0)
    with pytest.raises(ValueError, match="PET input"):
        palmer.pdsi(
            precips,
            pet=None,
            awc=1.0,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2001,
            pet_source="input",
        )

    with pytest.raises(ValueError, match="fortran_b"):
        palmer.pdsi(
            precips,
            pet=None,
            awc=1.0,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2001,
            pet_source="fortran",
            temperature_celsius=temps,
            latitude_degrees=45.0,
        )

    with pytest.raises(ValueError, match="Daily Tmin"):
        palmer.pdsi(
            precips,
            pet=None,
            awc=1.0,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2001,
            pet_source="hargreaves",
            latitude_degrees=45.0,
        )
