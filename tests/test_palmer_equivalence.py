"""Tests for Palmer NumPy vs xarray equivalence (Story 4.8).

Validates that the xarray path (palmer_xarray -> xr.Dataset) produces
numerically identical results to the NumPy path (pdsi -> tuple) for
all 4 Palmer output variables.

Tolerance: 1e-8 per NFR-PATTERN-EQUIV.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices.palmer import palmer_xarray
from climate_indices.palmer import pdsi as numpy_pdsi

_RNG = np.random.default_rng(42)
_N_MONTHS = 240  # 20 years
_DATA_START_YEAR = 2000
_CAL_YEAR_INITIAL = 2000
_CAL_YEAR_FINAL = 2019
_AWC = 5.0
_EQUIV_TOL = 1e-8


def _make_test_data(
    n_months: int = _N_MONTHS,
    start_year: int = _DATA_START_YEAR,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Generate consistent test precipitation and PET data.

    Returns:
        Tuple of (precip_array, pet_array, time_index).
    """
    time_index = pd.date_range(f"{start_year}-01", periods=n_months, freq="MS")
    months = np.arange(n_months) % 12

    # seasonal precipitation pattern ~2-4 in/month
    precip = 2.5 + 1.5 * np.sin(2 * np.pi * months / 12)
    precip = precip + _RNG.normal(0, 0.5, n_months)
    precip = np.clip(precip, 0.0, None)

    # seasonal PET pattern ~1-5 in/month
    pet = 3.0 + 2.0 * np.sin(2 * np.pi * (months - 3) / 12)
    pet = pet + _RNG.normal(0, 0.3, n_months)
    pet = np.clip(pet, 0.1, None)

    return precip, pet, time_index


# ---------------------------------------------------------------------------
# NumPy vs xarray equivalence tests
# ---------------------------------------------------------------------------


class TestPalmerEquivalence:
    """Validate numerical equivalence between numpy and xarray paths."""

    def test_pdsi_equivalence(self) -> None:
        """PDSI values from numpy and xarray paths are identical."""
        precip, pet, time_index = _make_test_data()

        # numpy path
        pdsi_np, _, _, _, _ = numpy_pdsi(
            precips=precip,
            pet=pet,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        # xarray path
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(
            precip_da,
            pet_da,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        np.testing.assert_allclose(pdsi_np, ds["pdsi"].values, atol=_EQUIV_TOL)

    def test_all_four_variables_equivalent(self) -> None:
        """All 4 Palmer outputs (pdsi, phdi, pmdi, z_index) are equivalent."""
        precip, pet, time_index = _make_test_data()

        pdsi_np, phdi_np, pmdi_np, z_np, _ = numpy_pdsi(
            precips=precip,
            pet=pet,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(
            precip_da,
            pet_da,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        np.testing.assert_allclose(pdsi_np, ds["pdsi"].values, atol=_EQUIV_TOL)
        np.testing.assert_allclose(phdi_np, ds["phdi"].values, atol=_EQUIV_TOL)
        np.testing.assert_allclose(pmdi_np, ds["pmdi"].values, atol=_EQUIV_TOL)
        np.testing.assert_allclose(z_np, ds["z_index"].values, atol=_EQUIV_TOL)

    @pytest.mark.parametrize("var_name,idx", [("pdsi", 0), ("phdi", 1), ("pmdi", 2), ("z_index", 3)])
    def test_individual_variable_equivalence(self, var_name: str, idx: int) -> None:
        """Parameterized test for each variable independently."""
        precip, pet, time_index = _make_test_data()

        result_tuple = numpy_pdsi(
            precips=precip,
            pet=pet,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(
            precip_da,
            pet_da,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        np.testing.assert_allclose(
            result_tuple[idx],
            ds[var_name].values,
            atol=_EQUIV_TOL,
            err_msg=f"{var_name} numpy vs xarray mismatch",
        )

    def test_equivalence_with_inferred_params(self) -> None:
        """Equivalence holds when xarray infers temporal parameters."""
        precip, pet, time_index = _make_test_data()

        # numpy path with explicit params
        pdsi_np, phdi_np, pmdi_np, z_np, _ = numpy_pdsi(
            precips=precip,
            pet=pet,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        # xarray path with inferred params
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)

        np.testing.assert_allclose(pdsi_np, ds["pdsi"].values, atol=_EQUIV_TOL)
        np.testing.assert_allclose(phdi_np, ds["phdi"].values, atol=_EQUIV_TOL)
        np.testing.assert_allclose(pmdi_np, ds["pmdi"].values, atol=_EQUIV_TOL)
        np.testing.assert_allclose(z_np, ds["z_index"].values, atol=_EQUIV_TOL)


# ---------------------------------------------------------------------------
# params_dict equivalence tests
# ---------------------------------------------------------------------------


class TestPalmerParamsEquivalence:
    """Validate that calibration parameters match between paths."""

    def test_params_dict_equivalent(self) -> None:
        """Calibration params from numpy match xarray Dataset attrs."""
        precip, pet, time_index = _make_test_data()

        _, _, _, _, params_np = numpy_pdsi(
            precips=precip,
            pet=pet,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(
            precip_da,
            pet_da,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        assert params_np is not None
        for param_name in ("alpha", "beta", "gamma", "delta"):
            np_values = params_np[param_name].tolist()
            xr_values = ds.attrs[f"palmer_{param_name}"]
            assert np_values == xr_values, f"{param_name} params differ"

    def test_params_json_matches_numpy(self) -> None:
        """JSON-serialized params from xarray match numpy params."""
        precip, pet, time_index = _make_test_data()

        _, _, _, _, params_np = numpy_pdsi(
            precips=precip,
            pet=pet,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(
            precip_da,
            pet_da,
            awc=_AWC,
            data_start_year=_DATA_START_YEAR,
            calibration_year_initial=_CAL_YEAR_INITIAL,
            calibration_year_final=_CAL_YEAR_FINAL,
        )

        params_json = json.loads(ds.attrs["palmer_params"])
        assert params_np is not None
        for param_name in ("alpha", "beta", "gamma", "delta"):
            np_values = params_np[param_name].tolist()
            json_values = params_json[param_name]
            assert np_values == json_values, f"JSON {param_name} differs from numpy"


# ---------------------------------------------------------------------------
# Scalar vs DataArray AWC equivalence tests
# ---------------------------------------------------------------------------


class TestPalmerAWCEquivalence:
    """Validate that scalar and DataArray AWC produce equivalent results."""

    def test_scalar_vs_0d_dataarray_awc(self) -> None:
        """Scalar AWC and 0-d DataArray AWC produce identical results."""
        precip, pet, time_index = _make_test_data()
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])

        ds_scalar = palmer_xarray(precip_da, pet_da, awc=_AWC)
        ds_da = palmer_xarray(precip_da, pet_da, awc=xr.DataArray(_AWC))

        for var_name in ("pdsi", "phdi", "pmdi", "z_index"):
            np.testing.assert_allclose(
                ds_scalar[var_name].values,
                ds_da[var_name].values,
                atol=_EQUIV_TOL,
                err_msg=f"Scalar vs DataArray AWC differ for {var_name}",
            )


# ---------------------------------------------------------------------------
# Provenance metadata tests
# ---------------------------------------------------------------------------


class TestPalmerProvenance:
    """Validate provenance metadata on xarray output."""

    def test_provenance_includes_version(self) -> None:
        """Dataset includes climate_indices_version attr."""
        precip, pet, time_index = _make_test_data()
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        assert "climate_indices_version" in ds.attrs
        assert isinstance(ds.attrs["climate_indices_version"], str)

    def test_provenance_history_records_palmer(self) -> None:
        """History attr records Palmer computation."""
        precip, pet, time_index = _make_test_data()
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        assert "history" in ds.attrs
        assert "Palmer" in ds.attrs["history"]

    def test_provenance_history_includes_timestamp(self) -> None:
        """History attr starts with a timestamp."""
        precip, pet, time_index = _make_test_data()
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        history = ds.attrs["history"]
        # format: YYYY-MM-DDTHH:MM:SSZ ...
        assert len(history) > 20
        assert "T" in history[:20]

    def test_each_variable_has_references(self) -> None:
        """Each output variable includes references from CF metadata registry."""
        precip, pet, time_index = _make_test_data()
        precip_da = xr.DataArray(precip, coords={"time": time_index}, dims=["time"])
        pet_da = xr.DataArray(pet, coords={"time": time_index}, dims=["time"])
        ds = palmer_xarray(precip_da, pet_da, awc=_AWC)
        for var_name in ("pdsi", "phdi", "pmdi", "z_index"):
            assert "references" in ds[var_name].attrs
            assert len(ds[var_name].attrs["references"]) > 0
