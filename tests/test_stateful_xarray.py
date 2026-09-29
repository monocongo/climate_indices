"""Tests for the shared stateful-recurrence xarray adapter (#1222)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices._stateful_xarray import StatefulAlignment, stateful_recurrence_xarray
from climate_indices.exceptions import CoordinateValidationError, InputAlignmentWarning
from climate_indices.xarray_adapter import _wrap_spatial


def _kernel(values: np.ndarray, *state: np.ndarray, offset: float = 1.0, spin_up: int = 0) -> np.ndarray:
    """A fake recurrence: add ``offset`` to every day, plus any supplied state."""
    total = np.moveaxis(values, -1, 0) + offset
    if state:
        total = total + state[0]
    return np.moveaxis(total[spin_up:], 0, -1)


def _time_series(periods: int = 6, cells: int = 3) -> xr.DataArray:
    dates = pd.date_range("2001-01-01", periods=periods)
    return xr.DataArray(
        np.arange(float(periods * cells)).reshape(periods, cells),
        dims=["time", "cell"],
        coords={
            "time": ("time", dates, {"long_name": "observation time"}),
            "cell": np.arange(cells),
        },
    )


def test_run_is_eager_preserves_time_attributes_and_extra_coordinates() -> None:
    data = _time_series().assign_coords(doy=("time", np.arange(6), {"long_name": "day of year"}))
    result = stateful_recurrence_xarray(
        [("values", data)],
        _kernel,
        time_dim="time",
        spin_up=2,
        output_core_dims=["time"],
        output_dtypes=[float],
        index_display_name="Fake",
        kernel_kwargs={"offset": 1.0, "spin_up": 2},
    )
    (values,) = result.outputs
    assert values.dims == ("time", "cell")
    assert values.time.attrs == {"long_name": "observation time"}
    assert values.doy.attrs == {"long_name": "day of year"}
    np.testing.assert_array_equal(values.doy, data.doy.isel(time=slice(2, None)))
    assert values.sizes["time"] == 4
    np.testing.assert_array_equal(values.values, data.values[2:] + 1.0)


def test_spatial_chunks_stay_lazy_and_are_chunked_to_the_primary_input() -> None:
    data = _time_series().chunk({"time": -1, "cell": 1})
    operand = _wrap_spatial(np.zeros(3), (3,), ("cell",), chunks={"cell": (1, 1, 1)})
    assert operand.chunks is not None
    result = stateful_recurrence_xarray(
        [("values", data)],
        _kernel,
        time_dim="time",
        spin_up=0,
        output_core_dims=["time"],
        output_dtypes=[float],
        index_display_name="Fake",
        build_extra_inputs=lambda alignment: [(operand, None)],
    )
    (values,) = result.outputs
    assert values.chunks is not None
    np.testing.assert_array_equal(values.values, data.values + 1.0)


def test_static_operand_receives_the_weather_chunk_targets() -> None:
    data = _time_series().chunk({"time": -1, "cell": 1})
    captured: list[StatefulAlignment] = []

    def build(alignment: StatefulAlignment) -> list[tuple[xr.DataArray, None]]:
        captured.append(alignment)
        operand = _wrap_spatial(
            np.zeros(3), alignment.spatial_shape, alignment.spatial_dims, chunks=alignment.spatial_chunks
        )
        assert operand.chunks is not None
        return [(operand, None)]

    stateful_recurrence_xarray(
        [("values", data)],
        _kernel,
        time_dim="time",
        spin_up=0,
        output_core_dims=["time"],
        output_dtypes=[float],
        index_display_name="Fake",
        build_extra_inputs=build,
    )
    assert captured[0].spatial_chunks == {"cell": (1, 1, 1)}


def _pair_kernel(values: np.ndarray, other: np.ndarray, offset: float = 1.0) -> np.ndarray:
    """A two-input fake recurrence: it reads the second input only to keep apply_ufunc happy."""
    del other
    return np.moveaxis(np.moveaxis(values, -1, 0) + offset, 0, -1)


def test_alignment_drops_only_warns_for_time_and_errors_for_space() -> None:
    values = _time_series()
    shorter = values.isel(time=slice(1, None))
    with pytest.warns(InputAlignmentWarning):
        result = stateful_recurrence_xarray(
            [("values", values), ("shorter", shorter)],
            _pair_kernel,
            time_dim="time",
            spin_up=0,
            output_core_dims=["time"],
            output_dtypes=[float],
            index_display_name="Fake",
        )
    assert result.outputs[0].sizes["time"] == 5

    disjoint_cells = values.assign_coords(cell=[1, 2, 3])
    with pytest.raises(CoordinateValidationError) as lost:
        stateful_recurrence_xarray(
            [("values", values), ("disjoint", disjoint_cells)],
            _pair_kernel,
            time_dim="time",
            spin_up=0,
            output_core_dims=["time"],
            output_dtypes=[float],
            index_display_name="Fake",
        )
    assert lost.value.reason == "non_time_alignment_dropped_coordinates"


def test_empty_time_intersection_raises() -> None:
    values = _time_series()
    disjoint_time = values.assign_coords(time=pd.date_range("2002-01-01", periods=6))
    with pytest.raises(CoordinateValidationError) as empty:
        stateful_recurrence_xarray(
            [("values", values), ("other", disjoint_time)],
            _kernel,
            time_dim="time",
            spin_up=0,
            output_core_dims=["time"],
            output_dtypes=[float],
            index_display_name="Fake",
        )
    assert empty.value.reason == "empty_intersection_after_alignment"


def test_extra_time_series_must_match_the_aligned_time_length() -> None:
    values = _time_series()
    with pytest.raises(CoordinateValidationError) as mismatch:
        stateful_recurrence_xarray(
            [("values", values)],
            _kernel,
            time_dim="time",
            spin_up=0,
            output_core_dims=["time"],
            output_dtypes=[float],
            index_display_name="Fake",
            build_extra_inputs=lambda alignment: [(xr.DataArray(np.zeros(3, dtype=np.int64), dims=["time"]), "time")],
        )
    assert mismatch.value.reason == "extra_input_time_length_mismatch"


def test_state_outputs_are_returned_eagerly() -> None:
    data = _time_series()
    result = stateful_recurrence_xarray(
        [("values", data)],
        _state_kernel,
        time_dim="time",
        spin_up=1,
        output_core_dims=["time", None],
        output_dtypes=[float, float],
        index_display_name="Fake",
        kernel_kwargs={"spin_up": 1},
    )
    values, final = result.outputs
    np.testing.assert_array_equal(final.values, data.values[-1] + 5.0)
    assert values.dims == ("time", "cell")


def _state_kernel(values: np.ndarray, spin_up: int = 0) -> tuple[np.ndarray, np.ndarray]:
    time_first = np.moveaxis(values, -1, 0)
    trimmed = time_first[spin_up:]
    return np.moveaxis(trimmed + 5.0, 0, -1), time_first[-1] + 5.0
