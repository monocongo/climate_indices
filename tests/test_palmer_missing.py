"""pdsi() must never turn a missing or uncomputable month into a number (#1710)."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climate_indices
from climate_indices.drought import palmer
from climate_indices.exceptions import ConvergenceError, MissingDataWarning
from tests.conftest import NativeRecorder, import_native

_N_MONTHS = 480
_AWC = 6.0
_YEARS = (1981, 1981, 2000)  # data start, calibration initial, calibration final
_CALIBRATION_END = 20 * 12  # first month after the calibration period


@pytest.fixture(params=["native", "python"], autouse=True)
def backend(request, monkeypatch):
    """Run every test on the Rust kernels and on the pure-Python path."""
    if request.param == "python":
        monkeypatch.setattr(palmer, "_native", None)
        yield None
        return
    recorder = NativeRecorder(import_native())
    monkeypatch.setattr(palmer, "_native", recorder)
    # the kernels run only when no NumPy floating-point reporting is requested
    with np.errstate(all="ignore"):
        yield recorder


@pytest.fixture
def series() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    precips = rng.gamma(2.0, 1.4, _N_MONTHS)
    pet = 2.5 - 2.3 * np.cos(2 * np.pi * (np.arange(_N_MONTHS) % 12) / 12)
    return precips, pet


def _with_gap(values: np.ndarray, month: int) -> np.ndarray:
    gappy = values.copy()
    gappy[month] = np.nan
    return gappy


@pytest.mark.parametrize("month", [0, 200, _CALIBRATION_END - 1])
@pytest.mark.parametrize("target", ["precips", "pet"])
def test_gap_before_calibration_ends_raises_for_a_single_series(series, month, target):
    inputs = dict(zip(("precips", "pet"), series, strict=True))
    inputs[target] = _with_gap(inputs[target], month)

    with pytest.raises(ConvergenceError, match=f"month index {month} is missing"):
        palmer.pdsi(inputs["precips"], inputs["pet"], _AWC, *_YEARS)


def test_zero_departure_record_raises_for_a_single_series(series):
    """Identical years give D-bar = 0, so K is not finite with no data missing."""
    precips, pet = series

    with pytest.raises(ConvergenceError, match="K factors are not finite"):
        palmer.pdsi(np.tile(precips[:12], _N_MONTHS // 12), pet, _AWC, *_YEARS)


@pytest.mark.parametrize("month", range(_CALIBRATION_END, _N_MONTHS))
def test_gap_after_calibration_ends_the_record(backend, series, month):
    precips, pet = series

    gapped = palmer.pdsi(_with_gap(precips, month), pet, _AWC, *_YEARS)[:4]
    truncated = palmer.pdsi(precips[:month], pet[:month], _AWC, *_YEARS)[:4]

    if backend is not None:
        assert "palmer_pdi" in backend.calls
    for values, expected in zip(gapped, truncated, strict=True):
        np.testing.assert_array_equal(values[:month], expected)
        assert np.isfinite(values[:month]).all()
        assert np.isnan(values[month:]).all()


def test_block_isolates_uncomputable_cells(series):
    precips, pet = series
    cells = [
        precips,
        _with_gap(precips, 200),  # inside calibration: uncomputable
        _with_gap(precips, 300),  # after calibration: truncated
        np.full(_N_MONTHS, np.nan),  # no data at all
        np.tile(precips[:12], _N_MONTHS // 12),  # D-bar = 0
    ]
    block = np.stack(cells, axis=1)[:, :, None]
    pet_block = np.broadcast_to(pet[:, None, None], block.shape).copy()

    with pytest.warns(MissingDataWarning, match=r"2 cell.*1 cell") as caught:
        result = palmer.pdsi(block, pet_block, _AWC, *_YEARS, spatial_time_major=True)[:4]

    assert caught[0].filename == __file__  # points at the caller of pdsi()

    clean = palmer.pdsi(precips, pet, _AWC, *_YEARS)[:4]
    truncated = palmer.pdsi(precips[:300], pet[:300], _AWC, *_YEARS)[:4]
    for values, expected, head in zip(result, clean, truncated, strict=True):
        np.testing.assert_array_equal(values[:, 0, 0], expected)
        np.testing.assert_array_equal(values[:300, 2, 0], head)
        assert np.isnan(values[300:, 2, 0]).all()
        for uncomputable in (1, 3, 4):
            assert np.isnan(values[:, uncomputable, 0]).all()


def test_block_with_only_fully_missing_cells_does_not_warn(series):
    precips, pet = series
    block = np.stack([precips, np.full(_N_MONTHS, np.nan)], axis=1)[:, :, None]
    pet_block = np.broadcast_to(pet[:, None, None], block.shape).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingDataWarning)
        result = palmer.pdsi(block, pet_block, _AWC, *_YEARS, spatial_time_major=True)

    assert np.isfinite(result[0][:, 0, 0]).all()
    assert np.isnan(result[0][:, 1, 0]).all()


def test_xarray_station_dimension_isolates_uncomputable_stations(series):
    """A (time, station) input reaches pdsi() one station at a time."""
    precips, pet = series
    coords = {"time": pd.date_range("1981-01-01", periods=_N_MONTHS, freq="MS"), "station": [0, 1]}

    def stations(first: np.ndarray, second: np.ndarray) -> xr.DataArray:
        values = np.stack([first, second], axis=1)
        return xr.DataArray(values, coords=coords, dims=("time", "station"), attrs={"units": "inches"})

    with pytest.warns(MissingDataWarning):
        result = climate_indices.pdsi(
            stations(precips, _with_gap(precips, 200)),
            stations(pet, pet),
            _AWC,
            calibration_year_initial=1981,
            calibration_year_final=2000,
        )

    expected = palmer.pdsi(precips, pet, _AWC, *_YEARS)[0]
    np.testing.assert_array_equal(result["pdsi"].isel(station=0).values, expected)
    for name in result.data_vars:
        assert np.isnan(result[name].isel(station=1).values).all()


def test_partial_final_calibration_year_is_named_as_the_reason(series):
    """No month is missing from the data: the calibration period outruns the record."""
    precips, pet = series

    with pytest.raises(ConvergenceError, match="final calibration year is incomplete.*month index 234"):
        palmer.pdsi(precips[:235], pet[:235], _AWC, *_YEARS)


def _monthly(values: np.ndarray, *dims: str) -> xr.DataArray:
    coords = {"time": pd.date_range("1981-01-01", periods=values.shape[0], freq="MS")}
    return xr.DataArray(values, coords=coords, dims=("time", *dims), attrs={"units": "inches"})


def test_xarray_default_calibration_ends_at_the_last_complete_year(series):
    precips, pet = series

    result = climate_indices.pdsi(_monthly(precips[:475]), _monthly(pet[:475]), _AWC)

    expected = palmer.pdsi(precips[:475], pet[:475], _AWC, 1981, 1981, 2019)[0]
    np.testing.assert_array_equal(result["pdsi"].values, expected)
    assert np.isfinite(expected).all()


def test_xarray_cell_dimension_only_on_pet_isolates_uncomputable_cells(series):
    precips, pet = series
    pets = _monthly(np.stack([pet, _with_gap(pet, 200)], axis=1), "station")

    with pytest.warns(MissingDataWarning):
        result = climate_indices.pdsi(
            _monthly(precips), pets, _AWC, calibration_year_initial=1981, calibration_year_final=2000
        )

    values = result["pdsi"].transpose("time", "station").values
    np.testing.assert_array_equal(values[:, 0], palmer.pdsi(precips, pet, _AWC, *_YEARS)[0])
    assert np.isnan(values[:, 1]).all()


def test_xarray_record_without_a_complete_year_names_the_incomplete_year(series):
    precips, pet = series

    with pytest.raises(ConvergenceError, match="final calibration year is incomplete"):
        climate_indices.pdsi(_monthly(precips[:6]), _monthly(pet[:6]), _AWC)
