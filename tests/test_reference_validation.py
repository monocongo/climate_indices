"""Reference dataset validation tests.

This module validates scientific correctness by testing SPI and SPEI computations
against NOAA-validated reference datasets. Tests both the NumPy computation path
(indices.spi/spei) and the xarray computation path (typed_public_api.spi/spei)
against the same reference data.

Reference data provenance is documented in tests/data/PROVENANCE.md.

Satisfies FR-TEST-004: Reference dataset validation at 1e-5 tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import indices, spei, spi
from climate_indices.compute import Periodicity
from climate_indices.indices import Distribution

# reference tolerance per FR-TEST-004
REFERENCE_TOLERANCE = 1e-5

# data parameters documented in tests/data/PROVENANCE.md
DATA_YEAR_START = 1895
DATA_YEAR_END = 2017
CALIBRATION_YEAR_START_PEARSON = 1981  # SPI Pearson only
CALIBRATION_YEAR_END_PEARSON = 2010  # SPI Pearson only

_DATA_DIR = Path(__file__).parent / "data"
_SPI_DIR = _DATA_DIR / "spi"
_SPEI_DIR = _DATA_DIR / "spei"


# ==============================================================================
# Fixtures (session-scoped, self-contained in test file)
# ==============================================================================


@pytest.fixture(scope="session")
def ref_precips_mm() -> np.ndarray:
    """Load reference precipitation input (1476 months, 1895-2017)."""
    return np.load(_SPI_DIR / "precips_mm_monthly.npy")


@pytest.fixture(scope="session")
def ref_pet_thornthwaite() -> np.ndarray:
    """Load reference PET input (Thornthwaite method, 1476 months)."""
    return np.load(_SPEI_DIR / "pet_thornthwaite.npy")


@pytest.fixture(scope="session")
def ref_spi_01_gamma() -> np.ndarray:
    """Load SPI 1-month gamma reference output."""
    return np.load(_SPI_DIR / "spi_01_gamma.npy")


@pytest.fixture(scope="session")
def ref_spi_06_gamma() -> np.ndarray:
    """Load SPI 6-month gamma reference output."""
    return np.load(_SPI_DIR / "spi_06_gamma.npy")


@pytest.fixture(scope="session")
def ref_spi_06_pearson() -> np.ndarray:
    """Load SPI 6-month Pearson Type III reference output."""
    return np.load(_SPI_DIR / "spi_06_pearson3.npy")


@pytest.fixture(scope="session")
def ref_spei_06_gamma() -> np.ndarray:
    """Load SPEI 6-month gamma reference output."""
    return np.load(_SPEI_DIR / "spei_06_gamma.npy")


@pytest.fixture(scope="session")
def ref_spei_06_pearson() -> np.ndarray:
    """Load SPEI 6-month Pearson Type III reference output."""
    return np.load(_SPEI_DIR / "spei_06_pearson3.npy")


@pytest.fixture(scope="session")
def ref_precips_da(ref_precips_mm: np.ndarray) -> xr.DataArray:
    """Wrap precipitation as xarray DataArray with monthly time coordinate."""
    # flatten 2D array (123, 12) to 1D (1476,) for time series
    precips_flat = ref_precips_mm.flatten()
    time = pd.date_range("1895-01-01", periods=len(precips_flat), freq="MS")
    return xr.DataArray(
        precips_flat,
        coords={"time": time},
        dims=["time"],
        name="precipitation",
    )


@pytest.fixture(scope="session")
def ref_pet_da(ref_pet_thornthwaite: np.ndarray) -> xr.DataArray:
    """Wrap PET as xarray DataArray with monthly time coordinate."""
    # flatten 2D array (123, 12) to 1D (1476,) for time series
    pet_flat = ref_pet_thornthwaite.flatten()
    time = pd.date_range("1895-01-01", periods=len(pet_flat), freq="MS")
    return xr.DataArray(
        pet_flat,
        coords={"time": time},
        dims=["time"],
        name="pet",
    )


# ==============================================================================
# Test Class 1: Reference Data Integrity
# ==============================================================================


class TestReferenceDataIntegrity:
    """Validate reference data files exist and have expected structure."""

    def test_spi_reference_files_exist(self) -> None:
        """All SPI reference files exist in tests/data/spi/."""
        assert (_SPI_DIR / "precips_mm_monthly.npy").exists()
        assert (_SPI_DIR / "spi_01_gamma.npy").exists()
        assert (_SPI_DIR / "spi_06_gamma.npy").exists()
        assert (_SPI_DIR / "spi_06_pearson3.npy").exists()

    def test_spei_reference_files_exist(self) -> None:
        """All SPEI reference files exist in tests/data/spei/."""
        assert (_SPEI_DIR / "precips_mm_monthly.npy").exists()
        assert (_SPEI_DIR / "pet_thornthwaite.npy").exists()
        assert (_SPEI_DIR / "spei_06_gamma.npy").exists()
        assert (_SPEI_DIR / "spei_06_pearson3.npy").exists()

    def test_provenance_file_exists(self) -> None:
        """PROVENANCE.md documents reference data sources."""
        assert (_DATA_DIR / "PROVENANCE.md").exists()

    def test_input_array_sizes_match_expected(
        self, ref_precips_mm: np.ndarray, ref_pet_thornthwaite: np.ndarray
    ) -> None:
        """Input arrays have expected size (1476 months = 123 years).

        Note: Arrays are stored as 2D (123, 12) but total size is 1476 months.
        """
        expected_months = (DATA_YEAR_END - DATA_YEAR_START + 1) * 12
        assert ref_precips_mm.size == expected_months, f"Expected {expected_months} months"
        assert ref_pet_thornthwaite.size == expected_months, f"Expected {expected_months} months"

    def test_precip_and_pet_arrays_consistent(
        self, ref_precips_mm: np.ndarray, ref_pet_thornthwaite: np.ndarray
    ) -> None:
        """Precipitation and PET arrays have matching sizes."""
        assert ref_precips_mm.shape == ref_pet_thornthwaite.shape


# ==============================================================================
# Test Class 2: SPI Reference Validation
# ==============================================================================


class TestSPIReferenceValidation:
    """Validate SPI computations against NOAA reference data via both NumPy and xarray paths."""

    def test_spi_1_month_gamma_numpy(
        self,
        ref_precips_mm: np.ndarray,
        ref_spi_01_gamma: np.ndarray,
    ) -> None:
        """SPI 1-month gamma (NumPy path) matches reference at 1e-5 tolerance."""
        result = indices.spi(
            ref_precips_mm,
            1,
            Distribution.gamma,
            DATA_YEAR_START,
            DATA_YEAR_START,
            DATA_YEAR_END,
            Periodicity.monthly,
        )

        np.testing.assert_allclose(
            result,
            ref_spi_01_gamma,
            atol=REFERENCE_TOLERANCE,
            err_msg=f"SPI 1-month gamma (NumPy) does not match reference at {REFERENCE_TOLERANCE} tolerance",
        )

    def test_spi_6_month_gamma_numpy(
        self,
        ref_precips_mm: np.ndarray,
        ref_spi_06_gamma: np.ndarray,
    ) -> None:
        """SPI 6-month gamma (NumPy path) matches reference at 1e-5 tolerance."""
        result = indices.spi(
            ref_precips_mm.flatten(),
            6,
            Distribution.gamma,
            DATA_YEAR_START,
            DATA_YEAR_START,
            DATA_YEAR_END,
            Periodicity.monthly,
        )

        np.testing.assert_allclose(
            result,
            ref_spi_06_gamma,
            atol=REFERENCE_TOLERANCE,
            err_msg=f"SPI 6-month gamma (NumPy) does not match reference at {REFERENCE_TOLERANCE} tolerance",
        )

    def test_spi_6_month_pearson_numpy(
        self,
        ref_precips_mm: np.ndarray,
        ref_spi_06_pearson: np.ndarray,
    ) -> None:
        """SPI 6-month Pearson (NumPy path) matches reference at 1e-5 tolerance."""
        result = indices.spi(
            ref_precips_mm.flatten(),
            6,
            Distribution.pearson,
            DATA_YEAR_START,
            CALIBRATION_YEAR_START_PEARSON,
            CALIBRATION_YEAR_END_PEARSON,
            Periodicity.monthly,
        )

        np.testing.assert_allclose(
            result,
            ref_spi_06_pearson,
            atol=REFERENCE_TOLERANCE,
            err_msg=f"SPI 6-month Pearson (NumPy) does not match reference at {REFERENCE_TOLERANCE} tolerance",
        )

    def test_spi_1_month_gamma_xarray(
        self,
        ref_precips_da: xr.DataArray,
        ref_spi_01_gamma: np.ndarray,
    ) -> None:
        """SPI 1-month gamma (xarray path) matches reference at 1e-5 tolerance."""
        result = spi(
            values=ref_precips_da,
            scale=1,
            distribution=Distribution.gamma,
            calibration_year_initial=DATA_YEAR_START,
            calibration_year_final=DATA_YEAR_END,
        )

        np.testing.assert_allclose(
            result.values,
            ref_spi_01_gamma,
            atol=REFERENCE_TOLERANCE,
            err_msg=f"SPI 1-month gamma (xarray) does not match reference at {REFERENCE_TOLERANCE} tolerance",
        )

    def test_spi_6_month_gamma_xarray(
        self,
        ref_precips_da: xr.DataArray,
        ref_spi_06_gamma: np.ndarray,
    ) -> None:
        """SPI 6-month gamma (xarray path) matches reference at 1e-5 tolerance."""
        result = spi(
            values=ref_precips_da,
            scale=6,
            distribution=Distribution.gamma,
            calibration_year_initial=DATA_YEAR_START,
            calibration_year_final=DATA_YEAR_END,
        )

        np.testing.assert_allclose(
            result.values,
            ref_spi_06_gamma,
            atol=REFERENCE_TOLERANCE,
            err_msg=f"SPI 6-month gamma (xarray) does not match reference at {REFERENCE_TOLERANCE} tolerance",
        )

    def test_spi_6_month_pearson_xarray(
        self,
        ref_precips_da: xr.DataArray,
        ref_spi_06_pearson: np.ndarray,
    ) -> None:
        """SPI 6-month Pearson (xarray path) matches reference at 1e-5 tolerance."""
        result = spi(
            values=ref_precips_da,
            scale=6,
            distribution=Distribution.pearson,
            calibration_year_initial=CALIBRATION_YEAR_START_PEARSON,
            calibration_year_final=CALIBRATION_YEAR_END_PEARSON,
        )

        np.testing.assert_allclose(
            result.values,
            ref_spi_06_pearson,
            atol=REFERENCE_TOLERANCE,
            err_msg=f"SPI 6-month Pearson (xarray) does not match reference at {REFERENCE_TOLERANCE} tolerance",
        )


# ==============================================================================
# Test Class 3: SPEI Reference Validation
# ==============================================================================


class TestSPEIReferenceValidation:
    """Validate SPEI computations against NOAA reference data via both NumPy and xarray paths.

    Note: SPEI Pearson Type III may require relaxed tolerance (~1e-3) due to inherent
    numerical sensitivity in L-moments fitting, as documented in test_backward_compat.py.
    """

    def test_spei_6_month_gamma_numpy(
        self,
        ref_precips_mm: np.ndarray,
        ref_pet_thornthwaite: np.ndarray,
        ref_spei_06_gamma: np.ndarray,
    ) -> None:
        """SPEI 6-month gamma (NumPy path) matches reference at 1e-5 tolerance."""
        result = indices.spei(
            ref_precips_mm,
            ref_pet_thornthwaite,
            6,
            Distribution.gamma,
            Periodicity.monthly,
            DATA_YEAR_START,
            DATA_YEAR_START,
            DATA_YEAR_END,
        )

        np.testing.assert_allclose(
            result,
            ref_spei_06_gamma,
            atol=REFERENCE_TOLERANCE,
            err_msg=f"SPEI 6-month gamma (NumPy) does not match reference at {REFERENCE_TOLERANCE} tolerance",
        )

    def test_spei_6_month_pearson_numpy(
        self,
        ref_precips_mm: np.ndarray,
        ref_pet_thornthwaite: np.ndarray,
        ref_spei_06_pearson: np.ndarray,
    ) -> None:
        """SPEI 6-month Pearson (NumPy path) matches reference at relaxed 1e-3 tolerance.

        Pearson Type III L-moments fitting has inherently more numerical noise than gamma,
        requiring a relaxed tolerance as documented in test_backward_compat.py.
        """
        result = indices.spei(
            ref_precips_mm,
            ref_pet_thornthwaite,
            6,
            Distribution.pearson,
            Periodicity.monthly,
            DATA_YEAR_START,
            DATA_YEAR_START,
            DATA_YEAR_END,
        )

        # use relaxed tolerance for Pearson (1e-3 instead of 1e-5)
        np.testing.assert_allclose(
            result,
            ref_spei_06_pearson,
            atol=1e-3,
            err_msg="SPEI 6-month Pearson (NumPy) does not match reference at 1e-3 tolerance",
        )

    def test_spei_6_month_gamma_xarray(
        self,
        ref_precips_da: xr.DataArray,
        ref_pet_da: xr.DataArray,
        ref_spei_06_gamma: np.ndarray,
    ) -> None:
        """SPEI 6-month gamma (xarray path) matches reference at 1e-5 tolerance."""
        result = spei(
            precips_mm=ref_precips_da,
            pet_mm=ref_pet_da,
            scale=6,
            distribution=Distribution.gamma,
            calibration_year_initial=DATA_YEAR_START,
            calibration_year_final=DATA_YEAR_END,
        )

        np.testing.assert_allclose(
            result.values,
            ref_spei_06_gamma,
            atol=REFERENCE_TOLERANCE,
            err_msg=f"SPEI 6-month gamma (xarray) does not match reference at {REFERENCE_TOLERANCE} tolerance",
        )

    def test_spei_6_month_pearson_xarray(
        self,
        ref_precips_da: xr.DataArray,
        ref_pet_da: xr.DataArray,
        ref_spei_06_pearson: np.ndarray,
    ) -> None:
        """SPEI 6-month Pearson (xarray path) matches reference at relaxed 1e-3 tolerance.

        Pearson Type III L-moments fitting has inherently more numerical noise than gamma,
        requiring a relaxed tolerance as documented in test_backward_compat.py.
        """
        result = spei(
            precips_mm=ref_precips_da,
            pet_mm=ref_pet_da,
            scale=6,
            distribution=Distribution.pearson,
            calibration_year_initial=DATA_YEAR_START,
            calibration_year_final=DATA_YEAR_END,
        )

        # use relaxed tolerance for Pearson (1e-3 instead of 1e-5)
        np.testing.assert_allclose(
            result.values,
            ref_spei_06_pearson,
            atol=1e-3,
            err_msg="SPEI 6-month Pearson (xarray) does not match reference at 1e-3 tolerance",
        )
