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
import os

import numpy as np
import pytest
import xarray as xr

from climate_indices import __main__ as cli_main
from climate_indices import _cli_transport, compute, palmer
from climate_indices.__main__ import DatasetLayout

_DIVISION_ID = "0101"
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixture", "palmer", _DIVISION_ID)


@pytest.fixture
def division_precip_pet():
    precips = np.load(os.path.join(_FIXTURE_DIR, "precips.npy"))
    pet = np.load(os.path.join(_FIXTURE_DIR, "pet.npy"))
    return precips, pet


def _palmer_transport(precips, pet, awc, shape: tuple[int, ...], awc_shape: tuple[int, ...]):
    """A transport with the three Palmer inputs and five empty result buffers."""
    transport = _cli_transport.Transport(1)
    transport.store.write("precip", np.asarray(precips).reshape(shape))
    transport.store.write("pet", np.asarray(pet).reshape(shape))
    transport.store.write("awc", np.asarray(awc).reshape(awc_shape))
    for key in _cli_transport.PALMER_RESULT_KEYS:
        transport.allocate(key, shape)
    return transport


def _run_palmer_worker(transport, item: _cli_transport.WorkItem) -> None:
    """Run one Palmer work item through the in-process executor (the test seam)."""
    _cli_transport.InlineExecutor(transport.store).map(_cli_transport.run_palmers, [item])


class TestCompanionDimensions:
    def test_rejects_pet_dimensions_outside_the_layout(self, monkeypatch):
        """A PET variable must carry one of the precipitation layout's accepted orders."""
        context = cli_main._InputContext(
            input_type=DatasetLayout.GRID,
            dimensions=("lat", "lon", "time"),
            times=np.arange(12),
            latitudes=np.array([25.0, 26.0]),
            longitudes=np.array([-100.0, -99.0]),
        )
        dataset = xr.Dataset(
            {"pet": (("lat", "time"), np.ones((2, 12)))},
            coords={"lat": [25.0, 26.0], "time": np.arange(12)},
        )
        monkeypatch.setattr(cli_main.xr, "open_dataset", lambda *args, **kwargs: dataset)

        with pytest.raises(ValueError) as error:
            cli_main._validate_matching_input_file(context, "PET", "pet.nc", "pet")

        assert str(error.value) == (
            "Invalid dimensions of the PET variable: ('lat', 'time') "
            "(expected names and order: [('lat', 'lon', 'time'), ('time', 'lat', 'lon')])"
        )


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

        transport = _palmer_transport(precips, pet, np.array([awc]), shape, (1,))

        palmers = cli_main._registry_for("palmers")
        assert palmers.worker is not None
        item = _cli_transport.WorkItem(
            kernel=palmers.kernel,
            input_names=("precip", "pet", "awc"),
            output_names=palmers.output_keys,
            coordinate_input=False,
            layout=DatasetLayout.DIVISIONS,
            arguments={
                "data_start_year": data_year_start_monthly,
                "calibration_start_year": calibration_year_start_palmer,
                "calibration_end_year": calibration_year_end_palmer,
            },
            start=0,
            end=None,
        )

        # the registration's worker is reachable through the in-process executor.
        # Should not raise (e.g. KeyError for a missing scpdsi shared array).
        _run_palmer_worker(transport, item)

        def _read(key):
            return transport.read(key, shape)[0]

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
        np.testing.assert_allclose(_read(_cli_transport.PALMER_RESULT_KEYS[0]), expected_pdsi, equal_nan=True)
        np.testing.assert_allclose(_read(_cli_transport.PALMER_RESULT_KEYS[1]), expected_phdi, equal_nan=True)
        np.testing.assert_allclose(_read(_cli_transport.PALMER_RESULT_KEYS[2]), expected_pmdi, equal_nan=True)
        np.testing.assert_allclose(_read(_cli_transport.PALMER_RESULT_KEYS[3]), expected_zindex, equal_nan=True)
        np.testing.assert_allclose(_read(_cli_transport.PALMER_RESULT_KEYS[4]), expected_scpdsi, equal_nan=True)

    def test_writer_trims_copied_input_chunks_to_the_output_shape(self, tmp_path, caplog):
        """An oversized copied chunk must not reach any of the five output writers."""
        n_time = 24
        shape = (1, n_time)
        transport = _cli_transport.Transport(1)
        for key, _var_name, _cf_key, _index_name in cli_main._PALMER_OUTPUTS:
            transport.allocate(key, shape)
        request = cli_main._IndexRequest(
            index="palmers",
            output_file_base=str(tmp_path / "out"),
            input_type=DatasetLayout.DIVISIONS,
            periodicity=compute.Periodicity.monthly,
            chunksizes="input",
            var_name_precip="precip",
        )
        context = cli_main._ComputeContext(
            request=request,
            dataset=xr.Dataset({"precip": (("division",), np.zeros(1))}),
            output_dims=("division", "time"),
            output_shape=shape,
            output_encodings={"chunksizes": (1, 100)},
            output_engine="h5netcdf",
            arguments={},
            transport=transport,
        )

        cli_main._write_palmer_outputs(context)

        assert caplog.messages.count("Trimming copied input chunksizes (1, 100) to the output shape (1, 24)") == 1
        for _key, var_name, _cf_key, _index_name in cli_main._PALMER_OUTPUTS:
            with xr.open_dataset(tmp_path / f"out_{var_name}.nc", engine="h5netcdf") as written:
                assert written[var_name].encoding["chunksizes"] == shape

    def test_grid_worker_applies_the_supplied_callable(
        self,
    ):
        """The grid branch must call ``func1d`` (as the divisions branch does)
        instead of hard-coding ``palmer.pdsi``, passing the block with a private
        ``spatial_time_major=True`` so the callable reads it as time-major."""
        lat, lon, n_time = 2, 2, 24
        shape = (lat, lon, n_time)
        transport = _palmer_transport(np.ones(shape), np.ones(shape), np.full((lat, lon), 5.0), shape, (lat, lon))

        calls: list[tuple[tuple[int, ...], bool]] = []

        def recording_palmers(precips, pet, awc, parameters):
            calls.append((precips.shape, parameters.get("spatial_time_major", False)))
            return tuple(np.full(precips.shape, position + 1.0) for position in range(5))

        output_keys = cli_main._registry_for("palmers").output_keys
        item = _cli_transport.WorkItem(
            kernel=recording_palmers,
            input_names=("precip", "pet", "awc"),
            output_names=output_keys,
            coordinate_input=False,
            layout=DatasetLayout.GRID,
            arguments={"data_start_year": 1980, "calibration_start_year": 1980, "calibration_end_year": 1981},
            start=0,
            end=None,
        )

        _run_palmer_worker(transport, item)

        # the callable saw the time-major block (time, lat, lon) ...
        assert calls == [((n_time, lat, lon), True)]
        # ... and each of its five outputs landed in its own shared array, in
        # registration order, not just the first one
        for position, key in enumerate(output_keys):
            np.testing.assert_array_equal(transport.read(key, shape), np.full(shape, position + 1.0))

    def test_grid_worker_matches_per_division_computation(
        self,
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

        transport = _palmer_transport(precip_grid, pet_grid, awcs, shape, (lat, lon))

        palmers = cli_main._registry_for("palmers")
        item = _cli_transport.WorkItem(
            kernel=palmers.kernel,
            input_names=("precip", "pet", "awc"),
            output_names=palmers.output_keys,
            coordinate_input=False,
            layout=DatasetLayout.GRID,
            arguments={
                "data_start_year": data_year_start_monthly,
                "calibration_start_year": calibration_year_start_palmer,
                "calibration_end_year": calibration_year_end_palmer,
            },
            start=0,
            end=None,
        )

        _run_palmer_worker(transport, item)

        grid_pdsi = transport.read(palmers.output_keys[0], shape)
        grid_phdi = transport.read(palmers.output_keys[1], shape)
        grid_pmdi = transport.read(palmers.output_keys[2], shape)
        grid_zindex = transport.read(palmers.output_keys[3], shape)
        grid_scpdsi = transport.read(palmers.output_keys[4], shape)

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
