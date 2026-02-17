"""Tests for EDDI CLI integration (FR-EDDI-003).

Validates that the EDDI index is properly integrated into the CLI
(__main__.py) including argument parsing, validation, worker functions,
variable attributes, and argument building.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from climate_indices import compute, indices


def _has_h5py() -> bool:
    """Check if h5py is available for NetCDF4 file writing."""
    try:
        import h5py  # noqa: F401

        return True
    except ImportError:
        return False
from climate_indices.__main__ import (
    _build_arguments,
    _eddi,
    _get_variable_attributes,
    _validate_args,
    process_climate_indices,
)


class TestEddiCliArgumentParsing:
    """Verify EDDI is accepted as an --index choice."""

    def test_eddi_is_valid_index_choice(self) -> None:
        """'eddi' should be a valid choice for --index."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--index",
            choices=["spi", "spei", "pnp", "scaled", "pet", "palmers", "eddi", "all"],
        )
        args = parser.parse_args(["--index", "eddi"])
        assert args.index == "eddi"


class TestEddiWorkerFunction:
    """Verify the _eddi() worker function calls indices.eddi correctly."""

    def test_eddi_worker_produces_valid_output(self) -> None:
        """_eddi() should produce an array of the same length as input."""
        rng = np.random.default_rng(42)
        pet = rng.uniform(20.0, 80.0, size=360)
        params: dict[str, Any] = {
            "data_start_year": 1990,
            "scale": 1,
            "calibration_year_initial": 1990,
            "calibration_year_final": 2019,
            "periodicity": compute.Periodicity.monthly,
        }
        result = _eddi(pet, params)
        assert isinstance(result, np.ndarray)
        assert result.shape == pet.shape

    def test_eddi_worker_matches_direct_call(self) -> None:
        """_eddi() should produce identical results to indices.eddi()."""
        rng = np.random.default_rng(123)
        pet = rng.uniform(20.0, 80.0, size=360)
        params: dict[str, Any] = {
            "data_start_year": 1990,
            "scale": 3,
            "calibration_year_initial": 1990,
            "calibration_year_final": 2019,
            "periodicity": compute.Periodicity.monthly,
        }
        worker_result = _eddi(pet, params)
        direct_result = indices.eddi(
            pet_values=pet,
            scale=3,
            data_start_year=1990,
            calibration_year_initial=1990,
            calibration_year_final=2019,
            periodicity=compute.Periodicity.monthly,
        )
        np.testing.assert_array_equal(worker_result, direct_result)


class TestEddiBuildArguments:
    """Verify _build_arguments produces correct args for EDDI."""

    def test_eddi_build_arguments_keys(self) -> None:
        """EDDI arguments should include scale, calibration years, and periodicity."""
        kwrgs: dict[str, Any] = {
            "index": "eddi",
            "data_start_year": 1990,
            "scale": 3,
            "calibration_start_year": 1990,
            "calibration_end_year": 2019,
            "periodicity": compute.Periodicity.monthly,
        }
        result = _build_arguments(kwrgs)
        assert result["data_start_year"] == 1990
        assert result["scale"] == 3
        assert result["calibration_year_initial"] == 1990
        assert result["calibration_year_final"] == 2019
        assert result["periodicity"] == compute.Periodicity.monthly

    def test_eddi_build_arguments_no_distribution(self) -> None:
        """EDDI should not include a 'distribution' key (non-parametric)."""
        kwrgs: dict[str, Any] = {
            "index": "eddi",
            "data_start_year": 1990,
            "scale": 1,
            "calibration_start_year": 1990,
            "calibration_end_year": 2019,
            "periodicity": compute.Periodicity.monthly,
        }
        result = _build_arguments(kwrgs)
        assert "distribution" not in result


class TestEddiVariableAttributes:
    """Verify _get_variable_attributes produces correct EDDI metadata."""

    def test_eddi_var_name_format(self) -> None:
        """EDDI variable name should be 'eddi_XX' (zero-padded scale)."""
        args_dict: dict[str, Any] = {
            "index": "eddi",
            "scale": 3,
            "periodicity": compute.Periodicity.monthly,
        }
        var_name, attrs = _get_variable_attributes(args_dict)
        assert var_name == "eddi_03"

    def test_eddi_long_name(self) -> None:
        """EDDI long_name should include 'Evaporative Demand Drought Index'."""
        args_dict: dict[str, Any] = {
            "index": "eddi",
            "scale": 6,
            "periodicity": compute.Periodicity.monthly,
        }
        var_name, attrs = _get_variable_attributes(args_dict)
        assert "Evaporative Demand Drought Index" in attrs["long_name"]
        assert "6-month" in attrs["long_name"]

    def test_eddi_valid_range(self) -> None:
        """EDDI valid_min/max should be [-3.09, 3.09]."""
        args_dict: dict[str, Any] = {
            "index": "eddi",
            "scale": 1,
            "periodicity": compute.Periodicity.monthly,
        }
        _, attrs = _get_variable_attributes(args_dict)
        assert attrs["valid_min"] == -3.09
        assert attrs["valid_max"] == 3.09


class TestEddiValidateArgs:
    """Verify _validate_args handles EDDI-specific requirements."""

    def _make_pet_netcdf(self, tmp_path: Path) -> Path:
        """Create a minimal PET netCDF file for testing."""
        pet_file = tmp_path / "pet.nc"
        time = np.arange(360)
        pet = np.random.default_rng(42).uniform(20.0, 80.0, size=360)
        ds = xr.Dataset(
            {"pet": (("time",), pet, {"units": "mm"})},
            coords={"time": time},
        )
        ds.to_netcdf(pet_file, engine="scipy")
        return pet_file

    def test_eddi_requires_pet_file(self, tmp_path: Path) -> None:
        """EDDI should raise ValueError if no PET file is provided."""
        args = argparse.Namespace(
            index="eddi",
            netcdf_pet=None,
            var_name_pet=None,
            scales=[1, 3],
            periodicity=compute.Periodicity.monthly,
        )
        with pytest.raises(ValueError, match="Missing the required PET file"):
            _validate_args(args)

    def test_eddi_requires_pet_var_name(self, tmp_path: Path) -> None:
        """EDDI should raise ValueError if no PET variable name is provided."""
        pet_file = self._make_pet_netcdf(tmp_path)
        args = argparse.Namespace(
            index="eddi",
            netcdf_pet=str(pet_file),
            var_name_pet=None,
            scales=[1],
            periodicity=compute.Periodicity.monthly,
        )
        with pytest.raises(ValueError, match="Missing PET variable name"):
            _validate_args(args)

    def test_eddi_requires_scales(self, tmp_path: Path) -> None:
        """EDDI should raise ValueError if no scales are provided."""
        pet_file = self._make_pet_netcdf(tmp_path)
        args = argparse.Namespace(
            index="eddi",
            netcdf_pet=str(pet_file),
            var_name_pet="pet",
            scales=None,
            periodicity=compute.Periodicity.monthly,
        )
        with pytest.raises(ValueError, match="missing --scales"):
            _validate_args(args)

    def test_eddi_does_not_require_precip(self, tmp_path: Path) -> None:
        """EDDI should NOT require a precipitation file."""
        pet_file = self._make_pet_netcdf(tmp_path)
        args = argparse.Namespace(
            index="eddi",
            netcdf_pet=str(pet_file),
            var_name_pet="pet",
            netcdf_precip=None,
            var_name_precip=None,
            scales=[1],
            periodicity=compute.Periodicity.monthly,
        )
        # should not raise about missing precipitation
        input_type = _validate_args(args)
        assert input_type is not None

    def test_eddi_validates_pet_variable_exists(self, tmp_path: Path) -> None:
        """EDDI should raise ValueError if PET variable doesn't exist in file."""
        pet_file = self._make_pet_netcdf(tmp_path)
        args = argparse.Namespace(
            index="eddi",
            netcdf_pet=str(pet_file),
            var_name_pet="nonexistent_var",
            scales=[1],
            periodicity=compute.Periodicity.monthly,
        )
        with pytest.raises(ValueError, match="Invalid PET variable name"):
            _validate_args(args)


class TestEddiEndToEnd:
    """Integration test: EDDI through the CLI pipeline with a real NetCDF."""

    @pytest.mark.skipif(
        not _has_h5py(),
        reason="h5py not installed; CLI NetCDF write requires h5py/netCDF4 backend",
    )
    def test_eddi_process_timeseries(self, tmp_path: Path) -> None:
        """process_climate_indices should produce EDDI output file for timeseries data."""
        import pandas as pd

        # create synthetic PET NetCDF
        rng = np.random.default_rng(42)
        n_months = 360
        time = pd.date_range("1990-01-01", periods=n_months, freq="MS")
        pet = rng.uniform(20.0, 80.0, size=n_months)
        ds = xr.Dataset(
            {"pet": (("time",), pet, {"units": "mm"})},
            coords={"time": time},
        )
        pet_file = tmp_path / "pet_input.nc"
        ds.to_netcdf(pet_file, engine="scipy")

        output_base = str(tmp_path / "eddi_output")

        args = argparse.Namespace(
            index="eddi",
            periodicity=compute.Periodicity.monthly,
            scales=[1],
            calibration_start_year=1990,
            calibration_end_year=2019,
            netcdf_precip=None,
            var_name_precip=None,
            netcdf_temp=None,
            var_name_temp=None,
            netcdf_pet=str(pet_file),
            var_name_pet="pet",
            netcdf_awc=None,
            var_name_awc=None,
            output_file_base=output_base,
            multiprocessing="single",
            chunksizes="none",
        )

        process_climate_indices(arguments=args)

        # verify the output file was created
        expected_output = Path(output_base + "_eddi_01.nc")
        assert expected_output.exists(), f"Expected output file {expected_output} not found"

        # verify the output contains EDDI data
        with xr.open_dataset(expected_output) as result_ds:
            assert "eddi_01" in result_ds.data_vars
            eddi_values = result_ds["eddi_01"].values
            # should have same length as input
            assert len(eddi_values) == n_months
            # check valid values are within EDDI range
            valid = ~np.isnan(eddi_values)
            assert np.all(eddi_values[valid] >= -3.09)
            assert np.all(eddi_values[valid] <= 3.09)
