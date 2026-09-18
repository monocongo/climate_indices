"""CLI regression tests for the Palmers path that do not need a full CLI run.

The historical `--index palmers` dispatch bug (`_parallel_process()` had no
Palmers branch, so every CLI invocation of `--index palmers` raised
`ValueError` before computing anything) is guarded end-to-end by
``tests/test_cli_e2e.py::test_palmers_writes_all_five_outputs_matching_in_process_computation``.

Retained here: `_validate_args()` must reject a time-dependent AWC, and a direct
worker call must write all five Palmer outputs, including `scpdsi`, into the
shared arrays. The e2e test covers the same contract through the CLI.
"""

import argparse
import multiprocessing
import os

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import palmer
from climate_indices.__main__ import DatasetLayout

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

    def test_rejects_awc_for_a_timeseries_input(self, monkeypatch):
        """A timeseries layout carries no per-location form for AWC."""
        coords = {"time": np.arange(12)}
        datasets = {
            "precip.nc": xr.Dataset({"precip": (("time",), np.ones(12))}, coords=coords),
            "pet.nc": xr.Dataset({"pet": (("time",), np.ones(12))}, coords=coords),
            "awc.nc": xr.Dataset({"awc": (("time",), np.ones(12))}, coords=coords),
        }
        monkeypatch.setattr(cli_main.xr, "open_dataset", datasets.__getitem__)
        arguments = argparse.Namespace(
            index="palmers",
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

        assert str(error.value) == "Available water capacity input requires gridded or US climate division data"


class TestScalesRequirement:
    def test_all_without_scales_raises_value_error(self, monkeypatch):
        """`--index all` computes SPI/PNP and must require --scales like they do.

        Regression test: `_validate_args` used to omit "all" from the scales
        requirement, so a missing `--scales` reached the SPI/PNP loop in
        `process_climate_indices` and raised a bare `TypeError` instead of a
        clear `ValueError` from validation.
        """
        coords = {"division": [_DIVISION_ID], "time": np.arange(12)}
        datasets = {
            "precip.nc": xr.Dataset({"precip": (("division", "time"), np.ones((1, 12)))}, coords=coords),
            "pet.nc": xr.Dataset({"pet": (("division", "time"), np.ones((1, 12)))}, coords=coords),
            "awc.nc": xr.Dataset({"awc": (("division",), np.ones(1))}, coords={"division": [_DIVISION_ID]}),
        }
        monkeypatch.setattr(cli_main.xr, "open_dataset", datasets.__getitem__)
        arguments = argparse.Namespace(
            index="all",
            scales=None,
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
            "Scaled indices (SPI, SPEI, and/or PNP) specified without "
            "including one or more time scales (missing --scales argument)"
        )

    def test_all_with_empty_scales_raises_value_error(self, monkeypatch):
        """An explicitly empty `--scales` list must be rejected like a missing one.

        Regression test: `--scales` uses `nargs="*"`, so `--scales` with no
        values parses to `[]` rather than `None`. The prior `is None` check
        let `[]` through, so SPI/SPEI/PNP scale loops ran zero iterations and
        `--index all` silently wrote only its unscaled outputs.
        """
        coords = {"division": [_DIVISION_ID], "time": np.arange(12)}
        datasets = {
            "precip.nc": xr.Dataset({"precip": (("division", "time"), np.ones((1, 12)))}, coords=coords),
            "pet.nc": xr.Dataset({"pet": (("division", "time"), np.ones((1, 12)))}, coords=coords),
            "awc.nc": xr.Dataset({"awc": (("division",), np.ones(1))}, coords={"division": [_DIVISION_ID]}),
        }
        monkeypatch.setattr(cli_main.xr, "open_dataset", datasets.__getitem__)
        arguments = argparse.Namespace(
            index="all",
            scales=[],
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
            "Scaled indices (SPI, SPEI, and/or PNP) specified without "
            "including one or more time scales (missing --scales argument)"
        )


class TestPalmersWorker:
    def test_writes_all_five_palmer_outputs(
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
            cli_main._KEY_RESULT_SCPDSI: _make_empty_shared_array(shape),
        }
        monkeypatch.setattr(cli_main, "_global_shared_arrays", shared_arrays)

        palmers = cli_main._registry_for("palmers")
        params = {
            "func1d": palmers.kernel,
            "sub_array_start": 0,
            "sub_array_end": None,
            "input_var_names": ["precip", "pet", "awc"],
            "output_var_names": list(palmers.output_keys),
            "coordinate_input": False,
            "input_type": DatasetLayout.DIVISIONS,
            "args": {
                "data_start_year": data_year_start_monthly,
                "calibration_start_year": calibration_year_start_palmer,
                "calibration_end_year": calibration_year_end_palmer,
            },
        }

        # the registration's worker is reachable directly, without monkeypatched dispatch.
        # Should not raise (e.g. KeyError for a missing scpdsi shared array).
        assert palmers.worker is not None
        palmers.worker(params)

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
        expected_scpdsi = palmer.scpdsi(
            precips,
            pet,
            awc,
            data_year_start_monthly,
            calibration_year_start_palmer,
            calibration_year_end_palmer,
        )[0]
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_PDSI), expected_pdsi, equal_nan=True)
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_PHDI), expected_phdi, equal_nan=True)
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_PMDI), expected_pmdi, equal_nan=True)
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_ZINDEX), expected_zindex, equal_nan=True)
        np.testing.assert_allclose(_read(cli_main._KEY_RESULT_SCPDSI), expected_scpdsi, equal_nan=True)

    def test_grid_worker_applies_the_supplied_callable(
        self,
        monkeypatch,
    ):
        """The grid branch must call ``func1d`` (as the divisions branch does)
        instead of hard-coding ``palmer.pdsi``, passing the block with a private
        ``spatial_time_major=True`` so the callable reads it as time-major."""
        lat, lon, n_time = 2, 2, 24
        shape = (lat, lon, n_time)
        shared_arrays = {
            "precip": _make_shared_array(np.ones(shape), shape),
            "pet": _make_shared_array(np.ones(shape), shape),
            "awc": _make_shared_array(np.full((lat, lon), 5.0), (lat, lon)),
            cli_main._KEY_RESULT_PDSI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_PHDI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_PMDI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_ZINDEX: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_SCPDSI: _make_empty_shared_array(shape),
        }
        monkeypatch.setattr(cli_main, "_global_shared_arrays", shared_arrays)

        calls: list[tuple[tuple[int, ...], bool]] = []

        def recording_palmers(precips, pet, awc, parameters):
            calls.append((precips.shape, parameters.get("spatial_time_major", False)))
            return tuple(np.full(precips.shape, position + 1.0) for position in range(5))

        params = {
            "func1d": recording_palmers,
            "sub_array_start": 0,
            "sub_array_end": None,
            "input_var_names": ("precip", "pet", "awc"),
            "output_var_names": cli_main._registry_for("palmers").output_keys,
            "input_type": DatasetLayout.GRID,
            "args": {"data_start_year": 1980, "calibration_start_year": 1980, "calibration_end_year": 1981},
        }

        cli_main._apply_along_axis_palmers(params)

        # the callable saw the time-major block (time, lat, lon) ...
        assert calls == [((n_time, lat, lon), True)]
        # ... and each of its five outputs landed in its own shared array, in
        # registration order, not just the first one
        for position, key in enumerate(cli_main._registry_for("palmers").output_keys):
            entry = shared_arrays[key]
            written = np.frombuffer(entry[cli_main._KEY_ARRAY].get_obj()).reshape(entry[cli_main._KEY_SHAPE])
            np.testing.assert_array_equal(written, np.full(shape, position + 1.0))

    def test_grid_worker_matches_per_division_computation(
        self,
        monkeypatch,
        division_precip_pet,
        data_year_start_monthly,
        calibration_year_start_palmer,
        calibration_year_end_palmer,
        palmer_awcs,
    ):
        """A 2x2 grid chunk is computed with palmer.pdsi()'s spatial block path
        (#937) rather than a Python loop over grid cells, and must still match
        calling palmer.pdsi() once per cell."""
        precips, pet = division_precip_pet
        n_time = precips.shape[0]
        lat, lon = 2, 2
        shape = (lat, lon, n_time)

        # four distinct AWCs so a broadcasting bug (one AWC applied to every
        # cell) would be caught
        awcs = np.array([[6.0, 7.0], [5.0, 8.0]])
        precip_grid = np.broadcast_to(precips, (lat, lon, n_time)).copy()
        pet_grid = np.broadcast_to(pet, (lat, lon, n_time)).copy()

        shared_arrays = {
            "precip": _make_shared_array(precip_grid, shape),
            "pet": _make_shared_array(pet_grid, shape),
            "awc": _make_shared_array(awcs, (lat, lon)),
            cli_main._KEY_RESULT_PDSI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_PHDI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_PMDI: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_ZINDEX: _make_empty_shared_array(shape),
            cli_main._KEY_RESULT_SCPDSI: _make_empty_shared_array(shape),
        }
        monkeypatch.setattr(cli_main, "_global_shared_arrays", shared_arrays)

        palmers = cli_main._registry_for("palmers")
        params = {
            "func1d": palmers.kernel,
            "sub_array_start": 0,
            "sub_array_end": None,
            "input_var_names": ("precip", "pet", "awc"),
            "output_var_names": palmers.output_keys,
            "input_type": DatasetLayout.GRID,
            "args": {
                "data_start_year": data_year_start_monthly,
                "calibration_start_year": calibration_year_start_palmer,
                "calibration_end_year": calibration_year_end_palmer,
            },
        }

        cli_main._apply_along_axis_palmers(params)

        def _read(key):
            entry = shared_arrays[key]
            return np.frombuffer(entry[cli_main._KEY_ARRAY].get_obj()).reshape(entry[cli_main._KEY_SHAPE])

        grid_pdsi = _read(cli_main._KEY_RESULT_PDSI)
        grid_phdi = _read(cli_main._KEY_RESULT_PHDI)
        grid_pmdi = _read(cli_main._KEY_RESULT_PMDI)
        grid_zindex = _read(cli_main._KEY_RESULT_ZINDEX)
        grid_scpdsi = _read(cli_main._KEY_RESULT_SCPDSI)

        for i in range(lat):
            for j in range(lon):
                expected_pdsi, expected_phdi, expected_pmdi, expected_zindex, _ = palmer.pdsi(
                    precips,
                    pet,
                    awcs[i, j],
                    data_year_start_monthly,
                    calibration_year_start_palmer,
                    calibration_year_end_palmer,
                )
                expected_scpdsi = palmer.scpdsi(
                    precips,
                    pet,
                    awcs[i, j],
                    data_year_start_monthly,
                    calibration_year_start_palmer,
                    calibration_year_end_palmer,
                )[0]
                np.testing.assert_array_equal(grid_pdsi[i, j], expected_pdsi)
                np.testing.assert_array_equal(grid_phdi[i, j], expected_phdi)
                np.testing.assert_array_equal(grid_pmdi[i, j], expected_pmdi)
                np.testing.assert_array_equal(grid_zindex[i, j], expected_zindex)
                np.testing.assert_array_equal(grid_scpdsi[i, j], expected_scpdsi)

    def test_uncalibratable_location_is_left_missing(
        self,
        division_precip_pet,
        data_year_start_monthly,
        calibration_year_start_palmer,
        calibration_year_end_palmer,
        palmer_awcs,
    ):
        """A calibration gap that defeats the duration-factor fit must not abort the run.

        `pdsi()` tolerates the gap and still computes, so scPDSI is left missing for
        that location (the all-missing contract) while the other outputs survive.
        """
        precips, pet = division_precip_pet
        awc = palmer_awcs[_DIVISION_ID]
        gappy_precips = precips.copy()
        gappy_precips[12 * (calibration_year_start_palmer - data_year_start_monthly)] = np.nan
        parameters = {
            "data_start_year": data_year_start_monthly,
            "calibration_start_year": calibration_year_start_palmer,
            "calibration_end_year": calibration_year_end_palmer,
        }

        # divisions path: the single location's scPDSI is missing, its PDSI is not
        pdsi, _phdi, _pmdi, _zindex, scpdsi = cli_main._palmers(gappy_precips, pet, awc, parameters)
        assert np.isfinite(pdsi).any()
        assert np.isnan(scpdsi).all()

        # grid path: only the gap-bearing cell is missing, the run still completes
        block = np.empty((precips.shape[0], 1, 2))
        block[:, 0, 0] = gappy_precips
        block[:, 0, 1] = precips
        pet_block = np.broadcast_to(pet[:, None, None], block.shape).copy()
        grid_pdsi, _grid_phdi, _grid_pmdi, _grid_zindex, grid_scpdsi = cli_main._palmers(
            block,
            pet_block,
            awc,
            {**parameters, "spatial_time_major": True},
        )
        assert np.isfinite(grid_pdsi).all()
        assert np.isnan(grid_scpdsi[:, 0, 0]).all()
        assert np.isfinite(grid_scpdsi[:, 0, 1]).all()
