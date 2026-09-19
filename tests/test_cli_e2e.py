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

from climate_indices import compute, indices, palmer, utils
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


def _write_time_major_grid(path, values, var_name="precip", units="mm") -> None:
    values = np.asarray(values, dtype=float)
    xr.Dataset(
        {var_name: (("time", "lat", "lon"), values, {"units": units})},
        coords={"time": _months(values.shape[0]), "lat": _LATITUDES, "lon": _LONGITUDES},
    ).to_netcdf(path)


def _write_time_major_divisions(path, values, var_name="precip", units="mm") -> None:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    xr.Dataset(
        {
            var_name: (("time", "division"), values, {"units": units}),
            "lat": (("division",), [_LATITUDES[0]]),
        },
        coords={"time": _months(values.shape[0]), "division": [_DIVISION]},
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


def _run_daily_spi(precip_path, output_base) -> None:
    main(
        [
            "--index",
            "spi",
            "--periodicity",
            "daily",
            "--calibration_start_year",
            "1981",
            "--calibration_end_year",
            "2010",
            "--netcdf_precip",
            str(precip_path),
            "--var_name_precip",
            "precip",
            "--output_file_base",
            str(output_base),
            "--multiprocessing",
            "single",
            "--scales",
            "30",
        ]
    )


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


def test_time_major_gridded_input_is_rejected(tmp_path, precips_mm_monthly):
    """
    A grid stored time-first is rejected rather than standardized wrongly.

    The shared-array transport copies storage order and the kernels index the
    grid's time axis last, so accepting this order computes each cell's index
    from its longitude series instead of raising.
    """
    values = precips_mm_monthly.reshape(-1)
    cells = np.stack([values] * (len(_LATITUDES) * len(_LONGITUDES)), axis=-1)
    precip_path = tmp_path / "precip_time_major.nc"
    _write_time_major_grid(precip_path, cells.reshape(values.size, len(_LATITUDES), len(_LONGITUDES)))

    arguments = _spi_arguments(precip_path, tmp_path / "spi_time_major")

    with pytest.raises(ValueError) as error:
        main(arguments)

    assert str(error.value) == (
        "Invalid dimensions for variable 'precip': ('time', 'lat', 'lon') "
        "(expected one of [('lat', 'lon', 'time'), ('lat', 'lon')])"
    )


def test_time_major_divisions_input_is_rejected(tmp_path, precips_mm_monthly):
    """
    A divisions variable stored time-first is rejected rather than standardized wrongly.

    The shared-array transport copies storage order and the kernels index a
    division's time axis at position 1, so accepting this order computes each
    index from its division series instead of raising.
    """
    values = precips_mm_monthly.reshape(-1)
    precip_path = tmp_path / "precip_time_major.nc"
    _write_time_major_divisions(precip_path, values)

    arguments = _spi_arguments(precip_path, tmp_path / "spi_time_major_divisions")

    with pytest.raises(ValueError) as error:
        main(arguments)

    assert str(error.value) == (
        "Invalid dimensions for variable 'precip': ('time', 'division') "
        "(expected one of [('division', 'time'), ('division',)])"
    )


def test_mixed_order_divisions_companion_is_rejected(tmp_path, precips_mm_monthly, pet_thornthwaite_mm):
    """
    A time-major PET companion is rejected alongside a time-last precipitation variable.

    Both variables ride the same transport and are zipped positionally into the
    two-array kernels, so a companion stored time-first would pair each
    division's series with the wrong PET series rather than raising.
    """
    precips = precips_mm_monthly.reshape(-1)
    pet = pet_thornthwaite_mm.reshape(-1)
    precip_path = tmp_path / "precip.nc"
    pet_path = tmp_path / "pet_time_major.nc"
    _write_divisions(precip_path, precips)
    _write_time_major_divisions(pet_path, pet, var_name="pet")

    arguments = [
        *_common_arguments("spei", precip_path, tmp_path / "spei_mixed_order"),
        "--scales",
        "6",
        "--netcdf_pet",
        str(pet_path),
        "--var_name_pet",
        "pet",
    ]

    with pytest.raises(ValueError) as error:
        main(arguments)

    assert str(error.value) == (
        "Invalid dimensions of the PET variable: ('time', 'division') (expected names and order: [('division', 'time')])"
    )


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


def test_spei_with_temperature_input_computes_and_consumes_pet(tmp_path, precips_mm_monthly, temps_celsius):
    """A temperature-only SPEI run writes PET as a side effect and consumes it."""
    precips = precips_mm_monthly.reshape(-1)
    temps = temps_celsius.reshape(-1)
    precip_path = tmp_path / "precip.nc"
    temp_path = tmp_path / "temp.nc"
    output_base = tmp_path / "spei_temp_only"

    _write_divisions(precip_path, precips)
    _write_divisions(temp_path, temps, var_name="temp", units="degrees_celsius")
    main(
        [
            *_common_arguments("spei", precip_path, output_base),
            "--scales",
            "6",
            "--netcdf_temp",
            str(temp_path),
            "--var_name_temp",
            "temp",
        ]
    )

    pet_path = tmp_path / "spei_temp_only_pet_thornthwaite.nc"
    assert pet_path.exists()
    with xr.open_dataset(pet_path) as dataset:
        pet = dataset["pet_thornthwaite"].values[0]

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
    with xr.open_dataset(tmp_path / "spei_temp_only_spei_gamma_06.nc") as dataset:
        np.testing.assert_allclose(dataset["spei_gamma_06"].values[0], expected, equal_nan=True)


def _length_in(values_inches, units):
    """Express values known in inches under the given length unit label."""
    return values_inches if units in ("inches", None) else values_inches * 25.4


@pytest.mark.parametrize("precip_units", ["mm", "inches"])
@pytest.mark.parametrize("awc_units", ["mm", "millimeters", "inches", None])
def test_palmers_writes_all_five_outputs_matching_in_process_computation(
    tmp_path, precips_mm_monthly, pet_thornthwaite_mm, palmer_awcs, precip_units, awc_units
):
    precips = precips_mm_monthly.reshape(-1)
    pet = pet_thornthwaite_mm.reshape(-1)
    awc = palmer_awcs[_DIVISION]
    precip_path = tmp_path / "precip.nc"
    pet_path = tmp_path / "pet.nc"
    awc_path = tmp_path / "awc.nc"
    # both labelings describe the same physical inputs, so the computed indices
    # must match the in-process computation on the inches palmer.pdsi() takes
    _write_divisions(precip_path, _length_in(precips / 25.4, precip_units), units=precip_units)
    _write_divisions(pet_path, _length_in(pet / 25.4, precip_units), var_name="pet", units=precip_units)
    awc_attrs = {} if awc_units is None else {"units": awc_units}
    xr.Dataset(
        {"awc": ("division", np.array([_length_in(awc, awc_units)]), awc_attrs)},
        coords={"division": [_DIVISION]},
    ).to_netcdf(awc_path)
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
        precips / 25.4,
        pet / 25.4,
        awc,
        _DATA_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
    )
    expected_scpdsi = palmer.scpdsi(
        precips / 25.4,
        pet / 25.4,
        awc,
        _DATA_START_YEAR,
        _CALIBRATION_START_YEAR,
        _CALIBRATION_END_YEAR,
    )[0]
    for variable_name, expected in [
        ("pdsi", expected_pdsi),
        ("phdi", expected_phdi),
        ("pmdi", expected_pmdi),
        ("zindex", expected_zindex),
        ("scpdsi", expected_scpdsi),
    ]:
        with xr.open_dataset(tmp_path / f"palmers_{variable_name}.nc") as dataset:
            np.testing.assert_allclose(dataset[variable_name].values[0], expected, equal_nan=True)

    # scPDSI has no hard valid range, so it is written without valid_min/valid_max
    with xr.open_dataset(tmp_path / "palmers_scpdsi.nc") as dataset:
        attrs = dataset["scpdsi"].attrs
        assert attrs["long_name"] == "Self-calibrated Palmer Drought Severity Index"
        assert "valid_min" not in attrs
        assert "valid_max" not in attrs

    # the CLI exposes all five Palmer outputs, including the self-calibrating scPDSI
    assert {path.name for path in tmp_path.glob("palmers_*.nc")} == {
        "palmers_pdsi.nc",
        "palmers_phdi.nc",
        "palmers_pmdi.nc",
        "palmers_zindex.nc",
        "palmers_scpdsi.nc",
    }


def test_palmers_rejects_an_awc_variable_with_unsupported_units(
    tmp_path, precips_mm_monthly, pet_thornthwaite_mm, palmer_awcs
):
    precip_path = tmp_path / "precip.nc"
    pet_path = tmp_path / "pet.nc"
    awc_path = tmp_path / "awc.nc"
    _write_divisions(precip_path, precips_mm_monthly.reshape(-1))
    _write_divisions(pet_path, pet_thornthwaite_mm.reshape(-1), var_name="pet")
    xr.Dataset(
        {"awc": ("division", np.array([palmer_awcs[_DIVISION]]), {"units": "kg m-2"})},
        coords={"division": [_DIVISION]},
    ).to_netcdf(awc_path)

    arguments = [
        *_common_arguments("palmers", precip_path, tmp_path / "palmers"),
        "--netcdf_pet",
        str(pet_path),
        "--var_name_pet",
        "pet",
        "--netcdf_awc",
        str(awc_path),
        "--var_name_awc",
        "awc",
    ]

    with pytest.raises(ValueError, match="Unsupported available water capacity units"):
        main(arguments)

    assert not list(tmp_path.glob("palmers_*"))


def test_palmers_rejects_a_precipitation_rate_label(tmp_path, precips_mm_monthly, pet_thornthwaite_mm, palmer_awcs):
    precip_path = tmp_path / "precip.nc"
    pet_path = tmp_path / "pet.nc"
    awc_path = tmp_path / "awc.nc"
    _write_divisions(precip_path, precips_mm_monthly.reshape(-1), units="mm/dy")
    _write_divisions(pet_path, pet_thornthwaite_mm.reshape(-1), var_name="pet")
    xr.Dataset(
        {"awc": ("division", np.array([palmer_awcs[_DIVISION]]), {"units": "inches"})},
        coords={"division": [_DIVISION]},
    ).to_netcdf(awc_path)

    arguments = [
        *_common_arguments("palmers", precip_path, tmp_path / "palmers"),
        "--netcdf_pet",
        str(pet_path),
        "--var_name_pet",
        "pet",
        "--netcdf_awc",
        str(awc_path),
        "--var_name_awc",
        "awc",
    ]

    with pytest.raises(ValueError, match="mm/dy"):
        main(arguments)

    assert not list(tmp_path.glob("palmers_*"))


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


def test_pnp_matches_in_process_computation(tmp_path, precips_mm_monthly):
    values = precips_mm_monthly.reshape(-1)
    precip_path = tmp_path / "precip.nc"
    _write_timeseries(precip_path, values)
    output_base = tmp_path / "pnp_timeseries"

    main([*_common_arguments("pnp", precip_path, output_base), "--scales", "6"])

    expected = indices.percentage_of_normal(
        values,
        scale=6,
        data_start_year=_DATA_START_YEAR,
        calibration_start_year=_CALIBRATION_START_YEAR,
        calibration_end_year=_CALIBRATION_END_YEAR,
        periodicity=compute.Periodicity.monthly,
    )
    with xr.open_dataset(tmp_path / "pnp_timeseries_pnp_06.nc") as dataset:
        np.testing.assert_allclose(dataset["pnp_06"].values, expected, equal_nan=True)


def test_all_runs_each_index_into_its_own_output(tmp_path, precips_mm_monthly, pet_thornthwaite_mm, palmer_awcs):
    """`all` runs SPI, SPEI, PNP, and Palmers, skipping PET when one is provided."""
    precips = precips_mm_monthly.reshape(-1)
    pet = pet_thornthwaite_mm.reshape(-1)
    precip_path = tmp_path / "precip.nc"
    pet_path = tmp_path / "pet.nc"
    awc_path = tmp_path / "awc.nc"
    _write_divisions(precip_path, precips)
    _write_divisions(pet_path, pet, var_name="pet")
    xr.Dataset(
        {"awc": ("division", [palmer_awcs[_DIVISION]])},
        coords={"division": [_DIVISION]},
    ).to_netcdf(awc_path)
    output_base = tmp_path / "all"

    main(
        [
            *_common_arguments("all", precip_path, output_base),
            "--scales",
            "1",
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

    # SPI and SPEI run once per scale and distribution, PNP once per scale, and
    # Palmers once into its five outputs; no PET file is written because the
    # provided PET input is used instead
    assert {path.name for path in tmp_path.glob("all_*.nc")} == {
        "all_spi_gamma_01.nc",
        "all_spi_pearson_01.nc",
        "all_spei_gamma_01.nc",
        "all_spei_pearson_01.nc",
        "all_pnp_01.nc",
        "all_pdsi.nc",
        "all_phdi.nc",
        "all_pmdi.nc",
        "all_zindex.nc",
        "all_scpdsi.nc",
    }

    expected_pnp = indices.percentage_of_normal(
        precips,
        scale=1,
        data_start_year=_DATA_START_YEAR,
        calibration_start_year=_CALIBRATION_START_YEAR,
        calibration_end_year=_CALIBRATION_END_YEAR,
        periodicity=compute.Periodicity.monthly,
    )
    with xr.open_dataset(tmp_path / "all_pnp_01.nc") as dataset:
        np.testing.assert_allclose(dataset["pnp_01"].values[0], expected_pnp, equal_nan=True)

    expected_spi = indices.spi(
        values=precips,
        scale=1,
        distribution=indices.Distribution.gamma,
        data_start_year=_DATA_START_YEAR,
        calibration_year_initial=_CALIBRATION_START_YEAR,
        calibration_year_final=_CALIBRATION_END_YEAR,
        periodicity=compute.Periodicity.monthly,
    )
    with xr.open_dataset(tmp_path / "all_spi_gamma_01.nc") as dataset:
        np.testing.assert_allclose(dataset["spi_gamma_01"].values[0], expected_spi, equal_nan=True)


def test_daily_gridded_spi_accepts_a_partial_final_year(tmp_path):
    """
    A daily input ending mid-year computes instead of failing in the conversion.

    The Gregorian-to-366-day conversion assumed whole calendar years: a partial
    final year either raised in the conversion (short or leap-year tails) or
    restored to the full Gregorian span and then failed the output size check.
    The shared calendar plan pads the partial final year and restores the
    observed days on output.
    """
    start_year = 1981
    time = xr.date_range(f"{start_year}-01-01", "2010-06-15", freq="D")
    generator = np.random.default_rng(seed=8675309)
    values = generator.gamma(shape=2.0, scale=10.0, size=(len(_LATITUDES), len(_LONGITUDES), time.size))
    precip_path = tmp_path / "precip_daily_grid.nc"
    xr.Dataset(
        {"precip": (("lat", "lon", "time"), values, {"units": "mm"})},
        coords={"lat": _LATITUDES, "lon": _LONGITUDES, "time": time},
    ).to_netcdf(precip_path)

    _run_daily_spi(precip_path, tmp_path / "spi_daily")

    # 2010 is a partial final year: 166 observed days padded to 366 for the core
    plan = utils.DailyCalendarPlan.from_year_span(start_year, 2011 - start_year, time.size)
    assert plan.observed_days_by_year[-1] == 166

    with xr.open_dataset(tmp_path / "spi_daily_spi_gamma_30.nc") as dataset:
        written = dataset["spi_gamma_30"].values
        assert written.shape == values.shape
        np.testing.assert_array_equal(dataset["time"].values, time.values)
        # the observed tail is real output, not NaN left over from the padding
        assert np.isfinite(written[..., -1]).all()
        for i in range(len(_LATITUDES)):
            for j in range(len(_LONGITUDES)):
                expected = plan.to_gregorian(
                    indices.spi(
                        values=plan.to_all_leap(values[i, j]),
                        scale=30,
                        distribution=indices.Distribution.gamma,
                        data_start_year=start_year,
                        calibration_year_initial=start_year,
                        calibration_year_final=2010,
                        periodicity=compute.Periodicity.daily,
                    )
                )
                np.testing.assert_allclose(written[i, j], expected, equal_nan=True, err_msg=f"cell ({i}, {j})")


def test_daily_divisions_spi_converts_and_restores(tmp_path):
    """A daily divisions input uses the same calendar plan as the grid transport."""
    start_year = 1981
    time = xr.date_range(f"{start_year}-01-01", "2010-06-15", freq="D")
    generator = np.random.default_rng(seed=112358)
    values = generator.gamma(shape=2.0, scale=10.0, size=(1, time.size))
    precip_path = tmp_path / "precip_daily_divisions.nc"
    xr.Dataset(
        {
            "precip": (("division", "time"), values, {"units": "mm"}),
            "lat": (("division",), [_LATITUDES[0]]),
        },
        coords={"division": [_DIVISION], "time": time},
    ).to_netcdf(precip_path)

    _run_daily_spi(precip_path, tmp_path / "spi_daily_divisions")

    plan = utils.DailyCalendarPlan.from_year_span(start_year, 2011 - start_year, time.size)
    with xr.open_dataset(tmp_path / "spi_daily_divisions_spi_gamma_30.nc") as dataset:
        written = dataset["spi_gamma_30"].values
        assert written.shape == values.shape
        expected = plan.to_gregorian(
            indices.spi(
                values=plan.to_all_leap(values[0]),
                scale=30,
                distribution=indices.Distribution.gamma,
                data_start_year=start_year,
                calibration_year_initial=start_year,
                calibration_year_final=2010,
                periodicity=compute.Periodicity.daily,
            )
        )
        np.testing.assert_allclose(written[0], expected, equal_nan=True)


def test_daily_input_starting_mid_year_is_rejected(tmp_path):
    """A daily series that does not begin January 1 is rejected, not silently shifted."""
    time = xr.date_range("1981-06-01", "2010-12-31", freq="D")
    precip_path = tmp_path / "precip_mid_year.nc"
    xr.Dataset(
        {"precip": (("time",), np.ones(time.size), {"units": "mm"})},
        coords={"time": time},
    ).to_netcdf(precip_path)

    with pytest.raises(ValueError, match="begin on January 1"):
        _run_daily_spi(precip_path, tmp_path / "spi_mid_year")


def test_daily_input_with_a_gap_is_rejected(tmp_path):
    """A daily series with a missing day is rejected, not positionally compacted."""
    time = xr.date_range("1981-01-01", "2010-12-31", freq="D").delete(100)
    precip_path = tmp_path / "precip_gap.nc"
    xr.Dataset(
        {"precip": (("time",), np.ones(time.size), {"units": "mm"})},
        coords={"time": time},
    ).to_netcdf(precip_path)

    with pytest.raises(ValueError, match="contiguous daily steps"):
        _run_daily_spi(precip_path, tmp_path / "spi_gap")
