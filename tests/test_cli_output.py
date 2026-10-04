"""Unit tests for the CLI's single output module and shared unit normalization."""

from __future__ import annotations

from pathlib import Path

import dask
import numpy as np
import pytest
import xarray as xr

from climate_indices import _cli_output
from climate_indices._units import _convert_precipitation_units
from climate_indices.exceptions import InvalidArgumentError


def _precip(units: str) -> xr.DataArray:
    return xr.DataArray(np.array([30.0, 60.0]), dims=("time",), attrs={"units": units})


@pytest.mark.parametrize("units", ["mm", "millimeters", "mm/month", "mm month-1"])
def test_monthly_depth_units_are_accepted(units: str) -> None:
    result = _convert_precipitation_units(_precip(units), "mm", monthly=True)
    np.testing.assert_array_equal(result.values, [30.0, 60.0])


@pytest.mark.parametrize("units", ["mm/day", "mm day-1", "kg m-2 s-1"])
def test_a_rate_is_rejected_for_a_monthly_depth(units: str) -> None:
    data = _precip(units)
    with pytest.raises(InvalidArgumentError, match="per-day rate"):
        _convert_precipitation_units(data, "mm", monthly=True)


@pytest.mark.parametrize("units", ["mm/month", "mm month-1"])
def test_a_monthly_depth_is_rejected_for_a_daily_series(units: str) -> None:
    data = _precip(units)
    with pytest.raises(InvalidArgumentError, match="Unsupported precipitation units"):
        _convert_precipitation_units(data, "mm")


def test_atomic_write_cleans_up_the_temporary_file_on_failure(monkeypatch, tmp_path) -> None:
    target = tmp_path / "out.nc"
    target.write_text("previous")

    def fail(self, path, **kwargs):
        Path(path).write_text("partial")
        raise RuntimeError("write failed")

    monkeypatch.setattr(xr.DataArray, "to_netcdf", fail)
    data = _precip("mm")
    with pytest.raises(RuntimeError, match="write failed"):
        _cli_output.write_netcdf_atomic(data, str(target))

    assert target.read_text() == "previous"
    assert list(tmp_path.glob("out.nc.*.tmp")) == []


def test_atomic_write_uses_a_unique_temporary_file(monkeypatch, tmp_path) -> None:
    target = tmp_path / "out.nc"
    seen: list[str] = []

    def record(self, path, **kwargs):
        seen.append(str(path))
        Path(path).write_text("written")

    monkeypatch.setattr(xr.DataArray, "to_netcdf", record)
    data = _precip("mm")
    _cli_output.write_netcdf_atomic(data, str(target))
    _cli_output.write_netcdf_atomic(data, str(target))

    assert len(set(seen)) == 2
    assert all(path != str(target) for path in seen)


def test_build_index_attrs_drops_input_cell_methods() -> None:
    source = xr.DataArray(
        np.array([1.0]),
        dims=("time",),
        attrs={"units": "mm", "cell_methods": "time: sum"},
    )
    attrs = _cli_output.build_index_attrs(source, "spi", index_name="SPI")
    assert "cell_methods" not in attrs


def test_build_index_attrs_drops_the_inputs_own_valid_range() -> None:
    source = xr.DataArray(
        np.array([1.0]),
        dims=("time",),
        attrs={
            "units": "mm",
            "valid_min": 0.0,
            "valid_max": 500.0,
            "valid_range": [0.0, 500.0],
            "actual_range": [0.0, 10.0],
        },
    )
    attrs = _cli_output.build_index_attrs(
        source, "spi", index_name="SPI", extra={"valid_min": -3.09, "valid_max": 3.09}
    )
    assert attrs["valid_min"] == -3.09
    assert attrs["valid_max"] == 3.09
    assert "valid_range" not in attrs
    assert "actual_range" not in attrs


_SCALE = 1e-4


def _grid(values: np.ndarray, name: str = "z") -> xr.Dataset:
    return xr.Dataset({name: (("lat", "lon", "time"), values)})


def _write_packed(dataset: xr.Dataset, path: Path) -> xr.Dataset:
    """Write ``dataset`` with packing on, then reopen it fully loaded."""
    _cli_output.write_netcdf_atomic(dataset, str(path), engine="h5netcdf", pack=True)
    with xr.open_dataset(path, engine="h5netcdf") as written:
        return written.load()


def test_in_range_values_pack_to_int16_within_half_a_step(tmp_path) -> None:
    rng = np.random.default_rng(0)
    values = rng.uniform(-3.09, 3.09, (2, 3, 24))
    values[0, 0, :4] = np.nan
    dataset = _grid(values)
    dataset["empty"] = (("lat", "lon", "time"), np.full((2, 3, 24), np.nan))

    encoding = _cli_output.choose_netcdf_encoding(dataset)
    assert encoding["z"] == {
        "dtype": "int16",
        "scale_factor": _SCALE,
        "add_offset": 0.0,
        "_FillValue": -32768,
        "zlib": True,
        "complevel": 4,
    }
    assert encoding["empty"]["dtype"] == "int16"

    written = _write_packed(dataset, tmp_path / "packed.nc")
    assert written["z"].encoding["dtype"] == np.dtype("int16")
    np.testing.assert_array_equal(np.isnan(written["z"].values), np.isnan(values))
    assert np.nanmax(np.abs(written["z"].values - values)) <= _SCALE / 2 + 1e-12
    assert np.isnan(written["empty"].values).all()


def test_a_value_beyond_the_int16_range_falls_back_to_exact_float32(tmp_path) -> None:
    values = np.random.default_rng(1).uniform(-1.0, 1.0, (2, 3, 24))
    values[1, 2, 5] = 3.5
    dataset = _grid(values)

    assert _cli_output.choose_netcdf_encoding(dataset)["z"] == {
        "dtype": "float32",
        "_FillValue": np.nan,
        "zlib": True,
        "complevel": 4,
    }
    written = _write_packed(dataset, tmp_path / "wide.nc")
    assert written["z"].dtype == np.dtype("float32")
    np.testing.assert_array_equal(written["z"].values, values.astype("float32"))


def test_the_limit_is_32767_steps_inclusive() -> None:
    limit = 32767 * _SCALE
    assert _cli_output.choose_netcdf_encoding(_grid(np.full((1, 1, 2), limit)))["z"]["dtype"] == "int16"
    assert _cli_output.choose_netcdf_encoding(_grid(np.full((1, 1, 2), limit + 1e-3)))["z"]["dtype"] == "float32"
    assert _cli_output.choose_netcdf_encoding(_grid(np.full((1, 1, 2), -limit)))["z"]["dtype"] == "int16"


def test_an_infinite_value_forces_float32(tmp_path) -> None:
    values = np.random.default_rng(2).uniform(-1.0, 1.0, (2, 3, 24))
    values[0, 1, 3] = np.inf
    values[1, 0, 7] = -np.inf
    dataset = _grid(values)

    assert _cli_output.choose_netcdf_encoding(dataset)["z"]["dtype"] == "float32"
    written = _write_packed(dataset, tmp_path / "inf.nc")
    np.testing.assert_array_equal(written["z"].values, values.astype("float32"))


def test_the_scale_and_complevel_are_parameters() -> None:
    dataset = _grid(np.full((1, 1, 2), 2.0))
    assert _cli_output.choose_netcdf_encoding(dataset, scale=1e-3, complevel=1)["z"] == {
        "dtype": "int16",
        "scale_factor": 1e-3,
        "add_offset": 0.0,
        "_FillValue": -32768,
        "zlib": True,
        "complevel": 1,
    }
    # 2.0 no longer fits a 1e-5 step: 32767 * 1e-5 = 0.32767
    assert _cli_output.choose_netcdf_encoding(dataset, scale=1e-5)["z"]["dtype"] == "float32"


def test_a_stale_packed_encoding_never_leaks_into_the_output(tmp_path) -> None:
    source = tmp_path / "packed_input.nc"
    _write_packed(_grid(np.full((2, 3, 24), 1.0)), source)
    with xr.open_dataset(source, engine="h5netcdf") as opened:
        dataset = opened.load()
    assert dataset["z"].encoding["dtype"] == np.dtype("int16")
    assert dataset["z"].encoding["scale_factor"] == _SCALE

    # an int16 write at the inherited 1e-4 step would wrap 5.0 -> 50000
    dataset["z"].values[0, 0, 0] = 5.0
    written = _write_packed(dataset, tmp_path / "widened.nc")

    assert written["z"].encoding["dtype"] == np.dtype("float32")
    assert "scale_factor" not in written["z"].encoding
    np.testing.assert_array_equal(written["z"].values, dataset["z"].values.astype("float32"))


def test_packing_clears_conflicting_keys_from_the_variables_own_encoding() -> None:
    dataset = _grid(np.full((1, 1, 2), 1.0))
    dataset["z"].encoding.update(
        {"dtype": "int16", "scale_factor": 0.5, "add_offset": 3.0, "_FillValue": -1, "missing_value": -1, "units": "x"}
    )
    _cli_output.choose_netcdf_encoding(dataset)
    assert dataset["z"].encoding == {"units": "x"}


def test_dask_chunking_is_kept_in_the_written_file(tmp_path) -> None:
    values = np.random.default_rng(3).uniform(-1.0, 1.0, (2, 4, 30))
    dataset = _grid(values).chunk({"lat": 1, "lon": 2, "time": -1})

    assert _cli_output.choose_netcdf_encoding(dataset)["z"]["chunksizes"] == (1, 2, 30)
    written = _write_packed(dataset, tmp_path / "chunked.nc")
    assert written["z"].encoding["chunksizes"] == (1, 2, 30)


def test_a_copied_input_chunk_encoding_wins_and_is_clamped_to_the_shape() -> None:
    dataset = _grid(np.zeros((2, 4, 30))).chunk({"lat": 1, "lon": 2, "time": -1})
    dataset["z"].encoding["chunksizes"] = (2, 3, 1000)
    assert _cli_output.choose_netcdf_encoding(dataset)["z"]["chunksizes"] == (2, 3, 30)


def test_an_unchunked_variable_gets_no_chunksizes() -> None:
    assert "chunksizes" not in _cli_output.choose_netcdf_encoding(_grid(np.zeros((2, 4, 30))))["z"]


def test_every_variable_is_ranged_in_a_single_dask_pass() -> None:
    dataset = _grid(np.ones((2, 4, 30)))
    dataset["w"] = (("lat", "lon", "time"), np.full((2, 4, 30), 9.0))
    dataset = dataset.chunk({"lat": 1, "lon": 2, "time": -1})
    calls: list[int] = []

    def counting_get(graph, keys, **kwargs):
        calls.append(1)
        return dask.get(graph, keys, **kwargs)

    with dask.config.set(scheduler=counting_get):
        encoding = _cli_output.choose_netcdf_encoding(dataset)

    assert len(calls) == 1
    assert (encoding["z"]["dtype"], encoding["w"]["dtype"]) == ("int16", "float32")


def test_packing_is_logged(caplog) -> None:
    dataset = _grid(np.ones((1, 1, 2)))
    dataset["w"] = (("lat", "lon", "time"), np.full((1, 1, 2), 9.0))
    with caplog.at_level("INFO", logger=_cli_output.__name__):
        _cli_output.choose_netcdf_encoding(dataset)
    assert "int16: ['z']" in caplog.text
    assert "float32: ['w']" in caplog.text


def test_writing_without_packing_keeps_the_default_encoding(tmp_path) -> None:
    values = np.random.default_rng(4).uniform(-1.0, 1.0, (2, 3, 24))
    path = tmp_path / "plain.nc"
    _cli_output.write_netcdf_atomic(_grid(values), str(path), engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf", mask_and_scale=False) as written:
        assert written["z"].dtype == np.dtype("float64")
        np.testing.assert_array_equal(written["z"].values, values)
