"""CLI regression tests for the Palmers path that do not need a full CLI run.

The historical `--index palmers` dispatch bug (`_parallel_process()` had no
Palmers branch, so every CLI invocation of `--index palmers` raised
`ValueError` before computing anything) is guarded end-to-end by
``tests/test_cli_e2e.py::test_palmers_writes_all_four_outputs_matching_in_process_computation``.

Retained here: `_validate_args()` must reject a time-dependent AWC, and a direct
worker call must write all four Palmer outputs into the shared arrays without
a `scpdsi` array -- self-calibration isn't implemented (see CONTEXT.md / issue
#716). The e2e test covers the same contract through the CLI, including the
absence of a `scpdsi` output file.
"""

import argparse
import multiprocessing
import os

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import palmer
from climate_indices.__main__ import InputType

_DIVISION_ID = "0101"
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixture", "palmer", _DIVISION_ID)


@pytest.fixture
def division_precip_pet():
    precips = np.load(os.path.join(_FIXTURE_DIR, "precips.npy"))
    pet = np.load(os.path.join(_FIXTURE_DIR, "pet.npy"))
    return precips, pet


def _make_shared_array(values: np.ndarray, shape: tuple[int, ...]) -> dict:
    shared = multiprocessing.Array("d", int(np.prod(shape)))
    view = np.frombuffer(shared.get_obj()).reshape(shape)
    np.copyto(view, values.reshape(shape))
    return {cli_main._KEY_ARRAY: shared, cli_main._KEY_SHAPE: shape}


def _make_empty_shared_array(shape: tuple[int, ...]) -> dict:
    shared = multiprocessing.Array("d", int(np.prod(shape)))
    return {cli_main._KEY_ARRAY: shared, cli_main._KEY_SHAPE: shape}


class TestAWCDimensions:
    @pytest.mark.parametrize("index", ["palmers", "all"])
    def test_rejects_time_dependent_awc(self, monkeypatch, index):
        """AWC with a time dimension cannot be consumed by Palmer workers."""
        coords = {"division": [_DIVISION_ID], "time": np.arange(12)}
        datasets = {
            "precip.nc": xr.Dataset({"precip": (("division", "time"), np.ones((1, 12)))}, coords=coords),
            "pet.nc": xr.Dataset({"pet": (("division", "time"), np.ones((1, 12)))}, coords=coords),
            "awc.nc": xr.Dataset({"awc": (("time", "division"), np.ones((12, 1)))}, coords=coords),
        }
        monkeypatch.setattr(cli_main.xr, "open_dataset", datasets.__getitem__)
        arguments = argparse.Namespace(
            index=index,
            netcdf_precip="precip.nc",
            var_name_precip="precip",
            netcdf_temp=None,
            netcdf_pet="pet.nc",
            var_name_pet="pet",
            netcdf_awc="awc.nc",
            var_name_awc="awc",
        )

        with pytest.raises(ValueError) as error:
            cli_main._validate_args(arguments)

        assert str(error.value) == (
            "Invalid dimensions of the AWC variable: ('time', 'division') (expected names and order: [('division',)])"
        )


class TestPalmersWorker:
    def test_writes_all_four_palmer_outputs(
        self,
        monkeypatch,
        division_precip_pet,
        data_year_start_monthly,
        calibration_year_start_palmer,
        calibration_year_end_palmer,
        palmer_awcs,
    ):
        precips, pet = division_precip_pet
        awc = palmer_awcs[_DIVISION_ID]
        n_time = precips.shape[0]
        shape = (1, n_time)

        shared_arrays = {
            "precip": _make_shared_array(precips, shape),
            "pet": _make_shared_array(pet, shape),
            "awc": _make_shared_array(np.array([awc]), (1,)),
            cli_main._KEY_RESULT_PDSI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_PHDI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_PMDI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_ZINDEX: _make_empty_shared_array(shape),
        }
        monkeypatch.setattr(cli_main, "_global_shared_arrays", shared_arrays)

        params = {
            "func1d": cli_main._palmers,
            "sub_array_start": 0,
            "sub_array_end": None,
            "var_name_precip": "precip",
            "var_name_pet": "pet",
            "var_name_awc": "awc",
            "output_var_name": cli_main._KEY_RESULT_PDSI,
            "input_type": InputType.divisions,
            "args": {
                "data_start_year": data_year_start_monthly,
                "calibration_start_year": calibration_year_start_palmer,
                "calibration_end_year": calibration_year_end_palmer,
            },
        }

        # Should not raise (e.g. KeyError for a missing scpdsi shared array).
        cli_main._apply_along_axis_palmers(params)

        def _read(key):
            entry = shared_arrays[key]
            return np.frombuffer(entry[cli_main._KEY_ARRAY].get_obj()).reshape(entry[cli_main._KEY_SHAPE])[0]

        expected_pdsi, expected_phdi, expected_pmdi, expected_zindex, _ = palmer.pdsi(
            precips,
            pet,
            awc,
            data_year_start_monthly,
            calibration_year_start_palmer,
            calibration_year_end_palmer,
        )
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_PDSI), expected_pdsi, equal_nan=True)
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_PHDI), expected_phdi, equal_nan=True)
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_PMDI), expected_pmdi, equal_nan=True)
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_ZINDEX), expected_zindex, equal_nan=True)
