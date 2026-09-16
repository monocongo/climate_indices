"""End-to-end tests for the ``climate_indices`` command-line interface.

Each test drives ``__main__.main()`` with real netCDF files written under
``tmp_path`` -- no subprocess and no monkeypatched ``xarray.open_dataset`` --
so argument parsing, input-type detection, file I/O, and index dispatch run
the way a user invokes them. Expected values come from calling the library
functions directly on the same fixture arrays, so a dispatch or reshape
mistake cannot pass by comparing the CLI against itself.

The Palmers test is the end-to-end regression guard for the historical
``--index palmers`` dispatch bug documented in ``tests/test_main_palmers.py``:
before the fix, ``_parallel_process()`` raised ``ValueError`` for every CLI
Palmers invocation and wrote no output.
"""

import numpy as np
import pytest
import xarray as xr

from climate_indices import compute, indices, palmer
from climate_indices.__main__ import main

# the fixture data starts in January of this year (see tests/conftest.py)
_DATA_START_YEAR = 1895
_CALIBRATION_START_YEAR = 1981
_CALIBRATION_END_YEAR = 2010
_DIVISION = "0101"
_LATITUDES = [25.0, 35.0]
_LONGITUDES = [250.0, 260.0]


def _months(periods: int):
    return xr.date_range(f"{_DATA_START_YEAR}-01-01", periods=periods, freq="MS")


def _write_timeseries(path, values, var_name="precip", units="mm") -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    xr.Dataset(
        {var_name: ("time", values, {"units": units})},
        coords={"time": _months(values.size)},
    ).to_netcdf(path)


def _write_divisions(path, values, var_name="precip", units="mm") -> None:
    values = np.asarray(values, dtype=float).reshape(1, -1)
    xr.Dataset(
        {
            var_name: (("division", "time"), values, {"units": units}),
            # the CLI reads a per-division latitude variable from divisions input
            "lat": (("division",), [_LATITUDES[0]]),
        },
        coords={"division": [_DIVISION], "time": _months(values.shape[-1])},
    ).to_netcdf(path)


def _write_grid(path, values, var_name="precip", units="mm") -> None:
    values = np.asarray(values, dtype=float)
    xr.Dataset(
        {var_name: (("lat", "lon", "time"), values, {"units": units})},
        coords={"lat": _LATITUDES, "lon": _LONGITUDES, "time": _months(values.shape[-1])},
    ).to_netcdf(path)


def _common_arguments(index, precip_path, output_base) -> list[str]:
    return [
        "--index",
        index,
        "--periodicity",
        "monthly",
        "--calibration_start_year",
        str(_CALIBRATION_START_YEAR),
        "--calibration_end_year",
        str(_CALIBRATION_END_YEAR),
        "--netcdf_precip",
        str(precip_path),
        "--var_name_precip",
        "precip",
        "--output_file_base",
        str(output_base),
        "--multiprocessing",
        "single",
    ]


def _spi_arguments(precip_path, output_base, scales=("6",)) -> list[str]:
    return [*_common_arguments("spi", precip_path, output_base), "--scales", *scales]


def test_timeseries_spi_matches_in_process_computation(tmp_path, precips_mm_monthly):
    values = precips_mm_monthly.reshape(-1)
    precip_path = tmp_path / "precip.nc"
    _write_timeseries(precip_path, values)
    output_base = tmp_path / "spi_timeseries"

    main(_spi_arguments(precip_path, output_base))

    expected = indices.spi(
        values=values,
        scale=6,
        distribution=indices.Distribution.gamma,
        data_start_year=_DATA_START_YEAR,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
        periodicity=compute.Periodicity.monthly,
    )
    with xr.open_dataset(tmp_path / "spi_timeseries_spi_gamma_06.nc") as dataset:
        np.testing.assert_allclose(dataset["spi_gamma_06"].values, expected, equal_nan=True)


def test_gridded_spi_matches_in_process_computation(tmp_path, precips_mm_monthly):
    values = precips_mm_monthly.reshape(-1)
    # offset each grid cell's seasonal cycle so a mis-indexed cell cannot match
    grid = np.stack([np.roll(values, (i * 2 + j) * 3) for i in range(len(_LATITUDES)) for j in range(len(_LONGITUDES))])
    grid = grid.reshape(len(_LATITUDES), len(_LONGITUDES), values.size)
    precip_path = tmp_path / "precip_grid.nc"
    _write_grid(precip_path, grid)
    output_base = tmp_path / "spi_grid"

    main(_spi_arguments(precip_path, output_base))

    with xr.open_dataset(tmp_path / "spi_grid_spi_gamma_06.nc") as dataset:
        written = dataset["spi_gamma_06"].values
        assert written.shape == grid.shape
        np.testing.assert_array_equal(dataset["lat"].values, _LATITUDES)
        np.testing.assert_array_equal(dataset["lon"].values, _LONGITUDES)
        for i in range(len(_LATITUDES)):
            for j in range(len(_LONGITUDES)):
                expected = indices.spi(
                    values=grid[i, j],
                    scale=6,
                    distribution=indices.Distribution.gamma,
                    data_start_year=_DATA_START_YEAR,
                    calibration_year_initial=_CALIBRATION_START_YEAR,
                    calibration_year_final=_CALIBRATION_END_YEAR,
                    periodicity=compute.Periodicity.monthly,
                )
                np.testing.assert_allclose(written[i, j], expected, equal_nan=True, err_msg=f"cell ({i}, {j})")


def test_spei_uses_provided_pet_file_and_matches_in_process_computation(
    tmp_path, precips_mm_monthly, pet_thornthwaite_mm
):
    precips = precips_mm_monthly.reshape(-1)
    pet = pet_thornthwaite_mm.reshape(-1)
    precip_path = tmp_path / "precip.nc"
    pet_path = tmp_path / "pet.nc"
    output_base = tmp_path / "spei_divisions"

    # a divisions dataset is used because the grid/divisions dispatch covers
    # the two-array worker path the timeseries shape does not
    _write_divisions(precip_path, precips)
    _write_divisions(pet_path, pet, var_name="pet")
    main(
        [
            *_common_arguments("spei", precip_path, output_base),
            "--scales",
            "6",
            "--netcdf_pet",
            str(pet_path),
            "--var_name_pet",
            "pet",
        ]
    )

    expected = indices.spei(
        precips_mm=precips,
        pet_mm=pet,
        scale=6,
        distribution=indices.Distribution.gamma,
        data_start_year=_DATA_START_YEAR,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
        periodicity=compute.Periodicity.monthly,
    )
    with xr.open_dataset(tmp_path / "spei_divisions_spei_gamma_06.nc") as dataset:
        np.testing.assert_allclose(dataset["spei_gamma_06"].values[0], expected, equal_nan=True)


def test_palmers_writes_all_four_outputs_matching_in_process_computation(
    tmp_path, precips_mm_monthly, pet_thornthwaite_mm, palmer_awcs
):
    precips = precips_mm_monthly.reshape(-1)
    pet = pet_thornthwaite_mm.reshape(-1)
    awc = palmer_awcs[_DIVISION]
    precip_path = tmp_path / "precip.nc"
    pet_path = tmp_path / "pet.nc"
    awc_path = tmp_path / "awc.nc"
    _write_divisions(precip_path, precips)
    _write_divisions(pet_path, pet, var_name="pet")
    xr.Dataset({"awc": ("division", np.array([awc]))}, coords={"division": [_DIVISION]}).to_netcdf(awc_path)
    output_base = tmp_path / "palmers"

    main(
        [
            *_common_arguments("palmers", precip_path, output_base),
            "--netcdf_pet",
            str(pet_path),
            "--var_name_pet",
            "pet",
            "--netcdf_awc",
            str(awc_path),
            "--var_name_awc",
            "awc",
        ]
    )

    expected_pdsi, expected_phdi, expected_pmdi, expected_zindex, _ = palmer.pdsi(
        precips,
        pet,
        awc,
        _DATA_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
    )
    for variable_name, expected in [
        ("pdsi", expected_pdsi),
        ("phdi", expected_phdi),
        ("pmdi", expected_pmdi),
        ("zindex", expected_zindex),
    ]:
        with xr.open_dataset(tmp_path / f"palmers_{variable_name}.nc") as dataset:
            np.testing.assert_allclose(dataset[variable_name].values[0], expected, equal_nan=True)

    # self-calibration isn't implemented (CONTEXT.md / issue #716), so the CLI
    # must not write a fifth scpdsi output
    assert not (tmp_path / "palmers_scpdsi.nc").exists()


def test_invalid_scale_raises_and_writes_no_output(tmp_path, precips_mm_monthly):
    precip_path = tmp_path / "precip.nc"
    _write_timeseries(precip_path, precips_mm_monthly.reshape(-1))
    output_base = tmp_path / "no_output"

    arguments = _spi_arguments(precip_path, output_base, scales=("-6",))

    with pytest.raises(ValueError, match="negative scale"):
        main(arguments)

    assert not list(tmp_path.glob("no_output*"))


def test_output_carries_cf_metadata_and_coordinates(tmp_path, precips_mm_monthly):
    precip_path = tmp_path / "precip.nc"
    _write_divisions(precip_path, precips_mm_monthly)
    output_base = tmp_path / "spi_metadata"

    main(_spi_arguments(precip_path, output_base))

    # the CLI computes every distribution, so both output files must exist
    assert (tmp_path / "spi_metadata_spi_pearson_06.nc").is_file()

    with xr.open_dataset(tmp_path / "spi_metadata_spi_gamma_06.nc") as dataset:
        variable = dataset["spi_gamma_06"]
        assert list(dataset.data_vars) == ["spi_gamma_06"]
        assert variable.dims == ("division", "time")
        assert variable.attrs["long_name"] == "Standardized Precipitation Index (Gamma distribution), 6-month"
        assert variable.attrs["valid_min"] == -3.09
        assert variable.attrs["valid_max"] == 3.09
        assert dataset["division"].values.tolist() == [_DIVISION]
        np.testing.assert_array_equal(dataset["time"].values, _months(precips_mm_monthly.size).values)
