"""Tests for the CLI's Palmers dispatch path in climate_indices.__main__.

Covers a real bug: `_parallel_process()` had no branch for index="palmers",
so every CLI invocation of `--index palmers` raised `ValueError` before
computing anything. There was also no `_palmers()` wrapper function, and
`_apply_along_axis_palmers()` expected a nonexistent "scpdsi" value (five
outputs) even though `palmer.pdsi()` only produces four (pdsi, phdi, pmdi,
zindex) -- self-calibration isn't implemented (see CONTEXT.md / issue #716).
"""

import multiprocessing
import os

import numpy as np
import pytest

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


class TestPalmersWrapper:
    def test_palmers_wrapper_returns_four_arrays_matching_palmer_pdsi(
        self,
        division_precip_pet,
        data_year_start_monthly,
        calibration_year_start_palmer,
        calibration_year_end_palmer,
        palmer_awcs,
    ):
        precips, pet = division_precip_pet
        awc = palmer_awcs[_DIVISION_ID]
        parameters = {
            "data_start_year": data_year_start_monthly,
            "calibration_start_year": calibration_year_start_palmer,
            "calibration_end_year": calibration_year_end_palmer,
        }

        result = cli_main._palmers(precips, pet, awc, parameters=parameters)

        assert len(result) == 4

        expected_pdsi, expected_phdi, expected_pmdi, expected_zindex, _ = palmer.pdsi(
            precips,
            pet,
            awc,
            data_year_start_monthly,
            calibration_year_start_palmer,
            calibration_year_end_palmer,
        )
        np.testing.assert_allclose(result[0], expected_pdsi, equal_nan=True)
        np.testing.assert_allclose(result[1], expected_phdi, equal_nan=True)
        np.testing.assert_allclose(result[2], expected_pmdi, equal_nan=True)
        np.testing.assert_allclose(result[3], expected_zindex, equal_nan=True)


class TestApplyAlongAxisPalmers:
    def test_writes_four_arrays_with_no_scpdsi_key_required(
        self,
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
        cli_main._global_shared_arrays = shared_arrays

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

    def test_no_scpdsi_shared_array_key_exists_on_module(self):
        assert not hasattr(cli_main, "_KEY_RESULT_SCPDSI")


class TestParallelProcessPalmersDispatch:
    def test_does_not_raise_unsupported_index_for_palmers(
        self,
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

        args = {
            "data_start_year": data_year_start_monthly,
            "calibration_start_year": calibration_year_start_palmer,
            "calibration_end_year": calibration_year_end_palmer,
        }

        cli_main._parallel_process(
            "palmers",
            shared_arrays,
            {"var_name_precip": "precip", "var_name_pet": "pet", "var_name_awc": "awc"},
            cli_main._KEY_RESULT_PDSI,
            input_type=InputType.divisions,
            args=args,
        )

        expected_pdsi, _, _, _, _ = palmer.pdsi(
            precips,
            pet,
            awc,
            data_year_start_monthly,
            calibration_year_start_palmer,
            calibration_year_end_palmer,
        )
        entry = shared_arrays[cli_main._KEY_RESULT_PDSI]
        actual_pdsi = np.frombuffer(entry[cli_main._KEY_ARRAY].get_obj()).reshape(entry[cli_main._KEY_SHAPE])[0]
        np.testing.assert_allclose(actual_pdsi, expected_pdsi, equal_nan=True)
