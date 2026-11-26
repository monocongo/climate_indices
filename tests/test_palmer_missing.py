"""pdsi() must never turn a missing or uncomputable month into a number (#1710)."""

import warnings
from collections.abc import Iterator
from typing import Literal

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

_Series = tuple[np.ndarray, np.ndarray]

pytestmark = pytest.mark.usefixtures("backend")


@pytest.fixture(params=["native", "python"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[NativeRecorder | None]:
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
def series() -> _Series:
    rng = np.random.default_rng(0)
    precips = rng.gamma(2.0, 1.4, _N_MONTHS)
    pet = 2.5 - 2.3 * np.cos(2 * np.pi * (np.arange(_N_MONTHS) % 12) / 12)
    return precips, pet


def _with_gap(values: np.ndarray, month: int) -> np.ndarray:
    gappy = values.copy()
    gappy[month] = np.nan
    return gappy


@pytest.mark.parametrize("month, calibration_start", [(0, 1981), (60, 1991), (200, 1981), (_CALIBRATION_END - 1, 1981)])
@pytest.mark.parametrize("target", ["precips", "pet"])
def test_gap_before_calibration_ends_raises_for_a_single_series(
    series: _Series, month: int, calibration_start: int, target: Literal["precips", "pet"]
) -> None:
    inputs = dict(zip(("precips", "pet"), series, strict=True))
    inputs[target] = _with_gap(inputs[target], month)

    with pytest.raises(ConvergenceError, match=f"month index {month} is missing"):
        palmer.pdsi(inputs["precips"], inputs["pet"], _AWC, 1981, calibration_start, 2000)


def test_zero_departure_record_raises_for_a_single_series(series: _Series) -> None:
    """Identical years give D-bar = 0, so K is not finite with no data missing."""
    precips, pet = series

    with pytest.raises(ConvergenceError, match="K factors are not finite"):
        palmer.pdsi(np.tile(precips[:12], _N_MONTHS // 12), pet, _AWC, *_YEARS)


@pytest.mark.filterwarnings("error::climate_indices.exceptions.MissingDataWarning")
@pytest.mark.parametrize(
    "month, target", [(month, "precips") for month in range(_CALIBRATION_END, _N_MONTHS)] + [(335, "pet")]
)
def test_gap_after_calibration_ends_the_record(
    backend: NativeRecorder | None, series: _Series, month: int, target: Literal["precips", "pet"]
) -> None:
    precips, pet = series
    inputs = dict(zip(("precips", "pet"), series, strict=True))
    inputs[target] = _with_gap(inputs[target], month)

    gapped = palmer.pdsi(inputs["precips"], inputs["pet"], _AWC, *_YEARS)[:4]
    truncated = palmer.pdsi(precips[:month], pet[:month], _AWC, *_YEARS)[:4]

    if backend is not None:
        assert "palmer_pdi" in backend.calls
    for values, expected in zip(gapped, truncated, strict=True):
        np.testing.assert_array_equal(values[:month], expected)
        assert np.isfinite(values[:month]).all()
        assert np.isnan(values[month:]).all()


def test_block_isolates_uncomputable_cells(series: _Series) -> None:
    precips, pet = series
    cells = [
        precips,
        _with_gap(precips, 200),  # inside calibration: uncomputable
        _with_gap(precips, 300),  # after calibration: truncated
        np.full(_N_MONTHS, np.nan),  # no data at all
        np.tile(precips[:12], _N_MONTHS // 12),  # D-bar = 0
        precips,  # later PET gap, with a different record length
    ]
    block = np.stack(cells, axis=1)[:, :, None]
    pet_block = np.broadcast_to(pet[:, None, None], block.shape).copy()
    pet_block[335, 5, 0] = np.nan

    with pytest.warns(MissingDataWarning, match=r"2 cell.*2 cell") as caught:
        result = palmer.pdsi(block, pet_block, _AWC, *_YEARS, spatial_time_major=True)[:4]

    assert len(caught) == 1
    assert caught[0].filename == __file__  # points at the caller of pdsi()

    clean = palmer.pdsi(precips, pet, _AWC, *_YEARS)[:4]
    for values, expected in zip(result, clean, strict=True):
        np.testing.assert_array_equal(values[:, 0, 0], expected)
        for uncomputable in (1, 3, 4):
            assert np.isnan(values[:, uncomputable, 0]).all()
    for cell, end in ((2, 300), (5, 335)):
        truncated = palmer.pdsi(precips[:end], pet[:end], _AWC, *_YEARS)[:4]
        for values, expected in zip(result, truncated, strict=True):
            np.testing.assert_array_equal(values[:end, cell, 0], expected)
            assert np.isnan(values[end:, cell, 0]).all()


def test_zero_water_supply_and_demand_do_not_abort_block(series: _Series) -> None:
    precips, pet = series
    block = np.stack([precips, np.zeros_like(precips)], axis=1)[:, :, None]
    pet_block = np.stack([pet, np.zeros_like(pet)], axis=1)[:, :, None]

    with np.errstate(divide="raise", invalid="raise"), pytest.warns(MissingDataWarning, match=r"1 cell"):
        result = palmer.pdsi(block, pet_block, _AWC, *_YEARS, spatial_time_major=True)

    for values in result[:4]:
        assert np.isfinite(values[:, 0, 0]).all()
        assert np.isnan(values[:, 1, 0]).all()


def test_block_with_only_fully_missing_cells_does_not_warn(series: _Series) -> None:
    precips, pet = series
    block = np.stack([precips, np.full(_N_MONTHS, np.nan)], axis=1)[:, :, None]
    pet_block = np.broadcast_to(pet[:, None, None], block.shape).copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingDataWarning)
        result = palmer.pdsi(block, pet_block, _AWC, *_YEARS, spatial_time_major=True)

    assert np.isfinite(result[0][:, 0, 0]).all()
    assert np.isnan(result[0][:, 1, 0]).all()


def test_xarray_station_dimension_isolates_uncomputable_stations(series: _Series) -> None:
    """The per-cell path passes each station to pdsi() as a singleton block."""
    precips, pet = series
    coords = {"time": pd.date_range("1981-01-01", periods=_N_MONTHS, freq="MS"), "station": [0, 1]}

    def stations(first: np.ndarray, second: np.ndarray) -> xr.DataArray:
        values = np.stack([first, second], axis=1)
        return xr.DataArray(values, coords=coords, dims=("time", "station"), attrs={"units": "inches"})

    station_precips = stations(precips, _with_gap(precips, 200))
    station_pet = stations(pet, pet)
    with pytest.warns(MissingDataWarning):
        result = climate_indices.pdsi(
            station_precips,
            station_pet,
            _AWC,
            calibration_year_initial=1981,
            calibration_year_final=2000,
        )

    expected = palmer.pdsi(precips, pet, _AWC, *_YEARS)[0]
    np.testing.assert_array_equal(result["pdsi"].isel(station=0).values, expected)
    for name in result.data_vars:
        assert np.isnan(result[name].isel(station=1).values).all()


def test_partial_final_calibration_year_is_named_as_the_reason(series: _Series) -> None:
    """No month is missing from the data: the calibration period outruns the record."""
    precips, pet = series

    with pytest.raises(ConvergenceError, match="final calibration year is incomplete.*month index 234"):
        palmer.pdsi(precips[:235], pet[:235], _AWC, *_YEARS)


def _monthly(values: np.ndarray, *dims: str) -> xr.DataArray:
    coords = {"time": pd.date_range("1981-01-01", periods=values.shape[0], freq="MS")}
    return xr.DataArray(values, coords=coords, dims=("time", *dims), attrs={"units": "inches"})


def test_xarray_default_calibration_ends_at_the_last_complete_year(series: _Series) -> None:
    precips, pet = series

    result = climate_indices.pdsi(_monthly(precips[:475]), _monthly(pet[:475]), _AWC)

    expected = palmer.pdsi(precips[:475], pet[:475], _AWC, 1981, 1981, 2019)[0]
    np.testing.assert_array_equal(result["pdsi"].values, expected)
    assert np.isfinite(expected).all()


def test_xarray_cell_dimension_only_on_pet_isolates_uncomputable_cells(series: _Series) -> None:
    precips, pet = series
    rain = _monthly(precips)
    pets = _monthly(np.stack([pet, _with_gap(pet, 200)], axis=1), "station")

    with pytest.warns(MissingDataWarning):
        result = climate_indices.pdsi(rain, pets, _AWC, calibration_year_initial=1981, calibration_year_final=2000)

    values = result["pdsi"].transpose("time", "station").values
    np.testing.assert_array_equal(values[:, 0], palmer.pdsi(precips, pet, _AWC, *_YEARS)[0])
    assert np.isnan(values[:, 1]).all()


def test_xarray_record_without_a_complete_year_names_the_incomplete_year(series: _Series) -> None:
    precips, pet = series

    rain = _monthly(precips[:6])
    evaporation = _monthly(pet[:6])
    with pytest.raises(ConvergenceError, match="final calibration year is incomplete"):
        climate_indices.pdsi(rain, evaporation, _AWC)
