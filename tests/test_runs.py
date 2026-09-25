"""Tests for ``climate_indices.runs`` (run theory event identification)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from climate_indices import runs
from climate_indices.exceptions import DataShapeError, DimensionMismatchError, InvalidArgumentError

# the study copy is an optional, local-only oracle (`precip-index`); the comparison
# tests skip when it is not checked out (e.g. CI)
STUDY_SRC = Path(os.environ.get("PRECIP_INDEX_STUDY_SRC", "~/git/precip-index-study/src")).expanduser()
STUDY_RUNTHEORY = STUDY_SRC / "runtheory.py"

# 1-D series with two runs below -1.0 (indices 1-2 and 5-6)
BELOW_SERIES = np.array([0.5, -1.2, -1.5, -0.8, 0.3, -1.1, -1.3])
# 1-D series with two runs above 1.0 (indices 1-2 and 4-5)
ABOVE_SERIES = np.array([-0.5, 1.2, 1.5, 0.8, 1.1, 1.3])


def test_below_threshold_reference_case() -> None:
    found = runs.identify_runs(BELOW_SERIES, threshold=-1.0, direction="below")

    assert len(found) == 2
    np.testing.assert_array_equal(found.start_index, [1, 5])
    np.testing.assert_array_equal(found.end_index, [2, 6])
    np.testing.assert_array_equal(found.duration, [2, 2])
    np.testing.assert_allclose(found.magnitude, [0.7, 0.4])
    np.testing.assert_allclose(found.intensity, [0.35, 0.2])
    np.testing.assert_allclose(found.peak_value, [-1.5, -1.3])
    np.testing.assert_array_equal(found.peak_index, [2, 6])
    np.testing.assert_allclose(found.interarrival, [4.0, np.nan], equal_nan=True)


def test_above_threshold_reference_case() -> None:
    found = runs.identify_runs(ABOVE_SERIES, threshold=1.0, direction="above")

    assert len(found) == 2
    np.testing.assert_array_equal(found.start_index, [1, 4])
    np.testing.assert_array_equal(found.end_index, [2, 5])
    np.testing.assert_array_equal(found.duration, [2, 2])
    np.testing.assert_allclose(found.magnitude, [0.7, 0.4])
    np.testing.assert_allclose(found.intensity, [0.35, 0.2])
    np.testing.assert_allclose(found.peak_value, [1.5, 1.3])
    np.testing.assert_array_equal(found.peak_index, [2, 5])
    np.testing.assert_allclose(found.interarrival, [3.0, np.nan], equal_nan=True)


def test_peak_is_minimum_below_and_maximum_above() -> None:
    series = np.array([-0.2, -1.4, -1.1, -1.9, -0.4, 0.2, 1.4, 1.1, 1.9, 0.4])

    below = runs.identify_runs(series, threshold=-1.0, direction="below")
    above = runs.identify_runs(series, threshold=1.0, direction="above")

    np.testing.assert_allclose(below.peak_value, [-1.9])
    np.testing.assert_array_equal(below.peak_index, [3])
    np.testing.assert_allclose(below.magnitude, [0.4 + 0.1 + 0.9])
    np.testing.assert_allclose(above.peak_value, [1.9])
    np.testing.assert_array_equal(above.peak_index, [8])
    np.testing.assert_allclose(above.magnitude, [0.4 + 0.1 + 0.9])


def test_direction_is_explicit_never_inferred_from_threshold_sign() -> None:
    # "above" with a negative threshold: runs are values above -0.5
    found_above = runs.identify_runs(np.array([-1.2, -0.3, -0.1, -0.9]), threshold=-0.5, direction="above")
    assert len(found_above) == 1
    np.testing.assert_array_equal(found_above.start_index, [1])
    np.testing.assert_array_equal(found_above.end_index, [2])
    np.testing.assert_allclose(found_above.magnitude, [0.6])
    np.testing.assert_allclose(found_above.peak_value, [-0.1])

    # "below" with a positive threshold: runs are values below 1.0
    found_below = runs.identify_runs(np.array([0.2, 1.5, 0.5]), threshold=1.0, direction="below")
    np.testing.assert_array_equal(found_below.start_index, [0, 2])
    np.testing.assert_array_equal(found_below.duration, [1, 1])


def test_runs_touching_array_boundaries_are_detected() -> None:
    found = runs.identify_runs(np.array([-1.5, -1.2, 0.5, -1.3]), threshold=-1.0)

    np.testing.assert_array_equal(found.start_index, [0, 3])
    np.testing.assert_array_equal(found.end_index, [1, 3])
    np.testing.assert_array_equal(found.duration, [2, 1])
    np.testing.assert_allclose(found.interarrival, [3.0, np.nan], equal_nan=True)


def test_nan_terminates_runs() -> None:
    found = runs.identify_runs(np.array([-1.2, np.nan, -1.3, -1.4]), threshold=-1.0)

    # the NaN breaks what would otherwise be one run into two
    assert len(found) == 2
    np.testing.assert_array_equal(found.start_index, [0, 2])
    np.testing.assert_array_equal(found.end_index, [0, 3])
    np.testing.assert_array_equal(found.duration, [1, 2])
    np.testing.assert_allclose(found.interarrival, [2.0, np.nan], equal_nan=True)


def test_value_equal_to_threshold_is_not_in_a_run() -> None:
    below = runs.identify_runs(np.array([-1.0, -1.01]), threshold=-1.0)
    above = runs.identify_runs(np.array([1.0, 1.01]), threshold=1.0, direction="above")

    np.testing.assert_array_equal(below.start_index, [1])
    np.testing.assert_array_equal(above.start_index, [1])


def test_min_duration_filters_runs_and_recomputes_interarrival() -> None:
    series = np.array([-1.2, 0.0, -1.1, -1.3, -1.5])

    unfiltered = runs.identify_runs(series, threshold=-1.0, min_duration=1)
    filtered = runs.identify_runs(series, threshold=-1.0, min_duration=2)
    long_only = runs.identify_runs(series, threshold=-1.0, min_duration=3)

    np.testing.assert_array_equal(unfiltered.duration, [1, 3])
    np.testing.assert_allclose(unfiltered.interarrival, [2.0, np.nan], equal_nan=True)

    assert len(filtered) == 1
    np.testing.assert_array_equal(filtered.start_index, [2])
    np.testing.assert_array_equal(filtered.end_index, [4])
    np.testing.assert_allclose(filtered.magnitude, [0.9])
    np.testing.assert_allclose(filtered.interarrival, [np.nan], equal_nan=True)

    # interarrival is computed between surviving runs only
    assert len(long_only) == 1


def test_empty_input_returns_empty_run_set() -> None:
    found = runs.identify_runs(np.array([]), threshold=-1.0)

    assert len(found) == 0
    assert found.duration.size == 0


def test_all_nan_input_returns_empty_run_set() -> None:
    found = runs.identify_runs(np.array([np.nan, np.nan, np.nan]), threshold=-1.0)

    assert len(found) == 0


@pytest.mark.parametrize("series", [np.array([]), np.array([np.nan, np.nan])])
def test_empty_run_set_has_stable_dtypes(series: np.ndarray) -> None:
    found = runs.identify_runs(series, threshold=-1.0)

    assert found.start_index.dtype == np.int64
    assert found.end_index.dtype == np.int64
    assert found.duration.dtype == np.int64
    assert found.peak_index.dtype == np.int64
    assert found.magnitude.dtype == np.float64
    assert found.intensity.dtype == np.float64
    assert found.peak_value.dtype == np.float64
    assert found.interarrival.dtype == np.float64


def test_list_input_is_accepted() -> None:
    found = runs.identify_runs([0.5, -1.2, -1.5], threshold=-1.0)

    np.testing.assert_array_equal(found.start_index, [1])
    np.testing.assert_array_equal(found.duration, [2])


def test_run_set_equality_compares_values_and_treats_nan_interarrival_as_equal() -> None:
    first = runs.identify_runs(BELOW_SERIES, threshold=-1.0)
    second = runs.identify_runs(BELOW_SERIES, threshold=-1.0)
    different = runs.identify_runs(BELOW_SERIES, threshold=-1.5)

    assert first == second
    assert first != different
    assert first != "not a run set"


def test_two_dimensional_numpy_input_is_rejected() -> None:
    with pytest.raises(DataShapeError, match="one-dimensional"):
        runs.identify_runs(np.zeros((12, 3)), threshold=-1.0)


def test_invalid_direction_raises() -> None:
    with pytest.raises(InvalidArgumentError, match="direction"):
        runs.identify_runs(BELOW_SERIES, threshold=-1.0, direction="sideways")  # type: ignore[arg-type]


@pytest.mark.parametrize("min_duration", [0, -1, 1.5])
def test_invalid_min_duration_raises(min_duration: float) -> None:
    with pytest.raises(InvalidArgumentError, match="min_duration"):
        runs.identify_runs(BELOW_SERIES, threshold=-1.0, min_duration=min_duration)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_non_finite_threshold_raises(threshold: float) -> None:
    with pytest.raises(InvalidArgumentError, match="threshold"):
        runs.identify_runs(BELOW_SERIES, threshold=threshold)


def test_xarray_1d_input_returns_run_set_directly() -> None:
    data = xr.DataArray(BELOW_SERIES, dims=["time"], coords={"time": np.arange(BELOW_SERIES.size)})

    found = runs.identify_runs_xarray(data, threshold=-1.0)

    assert isinstance(found, runs.RunSet)
    np.testing.assert_array_equal(found.start_index, [1, 5])


def test_xarray_2d_cells_matches_per_cell_core() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(36, 4))
    values[:, 2] = np.nan  # one all-NaN cell
    data = xr.DataArray(values, dims=["time", "cells"], coords={"cells": list("abcd")})

    found = runs.identify_runs_xarray(data, threshold=-1.0, min_duration=2)

    assert found.dims == ("cells",)
    assert found.dtype == object
    assert found.coords["cells"].values.tolist() == list("abcd")
    for cell in range(values.shape[1]):
        expected = runs.identify_runs(values[:, cell], threshold=-1.0, min_duration=2)
        assert found.values[cell] == expected
    assert len(found.values[2]) == 0


def test_xarray_3d_preserves_coords_and_matches_per_cell_core() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(24, 3, 2))
    data = xr.DataArray(
        values,
        dims=["time", "lat", "lon"],
        coords={"time": np.arange(24), "lat": [10.0, 20.0, 30.0], "lon": [-5.0, 5.0]},
    )

    found = runs.identify_runs_xarray(data, threshold=-0.5, direction="above")

    assert found.dims == ("lat", "lon")
    np.testing.assert_array_equal(found.coords["lat"], data.coords["lat"])
    np.testing.assert_array_equal(found.coords["lon"], data.coords["lon"])
    for i in range(3):
        for j in range(2):
            expected = runs.identify_runs(values[:, i, j], threshold=-0.5, direction="above")
            assert found.values[i, j] == expected


def test_xarray_dask_matches_eager() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(48, 3, 2))
    eager = xr.DataArray(values, dims=["time", "lat", "lon"])

    dask_data = eager.chunk({"time": 7, "lat": 2})
    found = runs.identify_runs_xarray(dask_data, threshold=-1.0)

    # the time axis is rechunked whole, so runs spanning chunk boundaries survive
    assert found.chunks is not None  # still lazy
    computed = found.compute()
    expected = runs.identify_runs_xarray(eager, threshold=-1.0)
    for i in range(3):
        for j in range(2):
            assert computed.values[i, j] == expected.values[i, j]


def test_xarray_custom_time_dim() -> None:
    data = xr.DataArray(BELOW_SERIES, dims=["month"])
    found = runs.identify_runs_xarray(data, threshold=-1.0, time_dim="month")

    assert isinstance(found, runs.RunSet)
    np.testing.assert_array_equal(found.start_index, [1, 5])


def test_xarray_missing_time_dim_raises() -> None:
    data = xr.DataArray(np.zeros((6, 2)), dims=["month", "cells"])

    with pytest.raises(DimensionMismatchError, match="time"):
        runs.identify_runs_xarray(data, threshold=-1.0, time_dim="time")


def test_xarray_attrs_record_the_options() -> None:
    data = xr.DataArray(np.zeros((6, 2)), dims=["time", "cells"])

    found = runs.identify_runs_xarray(data, threshold=-1.5, direction="above", min_duration=3)

    assert found.attrs["threshold"] == -1.5
    assert found.attrs["direction"] == "above"
    assert found.attrs["min_duration"] == 3


def _compare_with_study(series: np.ndarray, threshold: float, min_duration: int, study: object) -> None:
    """Compare our dry (below) runs against the study copy's event table."""
    ours = runs.identify_runs(series, threshold=threshold, direction="below", min_duration=min_duration)
    theirs = study.identify_events(series, threshold=threshold, min_duration=min_duration)  # type: ignore[attr-defined]

    if len(ours) == 0:
        assert len(theirs) == 0
        return

    np.testing.assert_array_equal(ours.start_index, theirs["start_idx"].to_numpy())
    np.testing.assert_array_equal(ours.end_index, theirs["end_idx"].to_numpy())
    np.testing.assert_array_equal(ours.duration, theirs["duration"].to_numpy())
    np.testing.assert_array_equal(ours.peak_index, theirs["peak_idx"].to_numpy())
    np.testing.assert_allclose(ours.magnitude, theirs["magnitude"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(ours.intensity, theirs["intensity"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(ours.peak_value, theirs["peak"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(ours.interarrival, theirs["interarrival"].to_numpy(), rtol=1e-12, equal_nan=True)


@pytest.mark.skipif(not STUDY_RUNTHEORY.is_file(), reason=f"study copy not available at {STUDY_SRC}")
def test_study_copy_dry_run_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(STUDY_SRC))
    study = pytest.importorskip("runtheory")

    for threshold in (-1.0, -1.5, -2.0):
        for min_duration in (1, 2, 3):
            _compare_with_study(BELOW_SERIES, threshold, min_duration, study)


@pytest.mark.skipif(not STUDY_RUNTHEORY.is_file(), reason=f"study copy not available at {STUDY_SRC}")
def test_study_copy_dry_run_comparison_on_random_series_with_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(STUDY_SRC))
    study = pytest.importorskip("runtheory")

    rng = np.random.default_rng(20250925)
    series = rng.normal(size=240) * 1.2
    series[[0, 17, 18, 19, 100, 239]] = np.nan
    series[40:44] = -2.5  # a long, unambiguous drought

    for threshold in (-0.5, -1.0, -1.5):
        for min_duration in (1, 3):
            _compare_with_study(series, threshold, min_duration, study)
