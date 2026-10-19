"""Energy event statistics use the generic run-theory contract."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, energy, indices
from climate_indices.exceptions import CoordinateValidationError, InsufficientDataError, InvalidArgumentError
from climate_indices.runs import Direction, RunStatistics


def test_energy_events_report_runs_and_period_statistics() -> None:
    result = energy.energy_drought_events(
        [2, 0, 3, 0, 0, 4],
        threshold=1.0,
        direction="above",
        min_duration=2,
        max_gap=1,
        periods=[2020, 2020, 2021, 2021, 2021, 2021],
    )

    assert isinstance(result, RunStatistics)
    np.testing.assert_array_equal(result.events.start_index, [0])
    np.testing.assert_array_equal(result.events.end_index, [2])
    np.testing.assert_array_equal(result.count, [1, 0])
    np.testing.assert_array_equal(result.total_duration, [3, 0])
    # Integer deviations and these binary fractions are exactly representable.
    np.testing.assert_allclose(result.max_magnitude, [3.0, 0.0], rtol=0, atol=0)
    np.testing.assert_allclose(result.frequency, [0.5, 0.0], rtol=0, atol=0)


@pytest.mark.parametrize(
    "source, direction",
    [("spi", "below"), ("capacity_mean_1_normal", "below"), ("residual_sum_1_normal", "above")],
)
def test_events_accept_spi_and_saved_srepi_srli_index_series(source: str, direction: Direction) -> None:
    if source == "spi":
        rain = np.random.default_rng(1173).gamma(2, 20, 40 * 12)
        index = indices.spi(rain, 1, indices.Distribution.gamma, 1980, 1980, 2019, compute.Periodicity.monthly)
    else:
        # Saved SEI outputs cover energy indices before #1171/#1172 expose their wrappers.
        with np.load(Path(__file__).parent / "fixture" / "energy_sei" / "reference.npz") as saved:
            index = saved[source]
    threshold = -1.0 if direction == "below" else 1.0
    result = energy.energy_drought_events(index, threshold=threshold, direction=direction)

    expected = index < threshold if direction == "below" else index > threshold
    observed = np.zeros(index.size, dtype=bool)
    for start, end in zip(result.events.start_index, result.events.end_index, strict=True):
        observed[start : end + 1] = True
    np.testing.assert_array_equal(observed, expected)
    # Count starts directly, independently of the vectorized run detector.
    count = sum(active and (i == 0 or not expected[i - 1]) for i, active in enumerate(expected))
    assert len(result.events) == count > 0
    np.testing.assert_array_equal(result.count, [count])
    np.testing.assert_array_equal(result.valid_count, [np.count_nonzero(~np.isnan(index))])
    np.testing.assert_allclose(result.frequency, [count / result.valid_count[0]], rtol=0, atol=0)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("labeled_periods", [False, True])
def test_dataarray_series_returns_the_same_statistics(lazy: bool, labeled_periods: bool) -> None:
    """A 1-D DataArray returns the NumPy RunStatistics unchanged."""
    values = np.array([2.0, 0.0, 3.0, 0.0, 0.0, 4.0])
    da = xr.DataArray(values, dims="time", coords={"time": pd.date_range("2020-01-01", periods=values.size)})
    periods = np.array([2020, 2020, 2021, 2021, 2021, 2021])
    labels = xr.DataArray(periods, dims="time", coords={"time": da.time}) if labeled_periods else periods
    if lazy:
        da = da.chunk({"time": 2})

    result = energy.energy_drought_events(
        da, threshold=1.0, direction="above", min_duration=2, max_gap=1, periods=labels
    )
    expected = energy.energy_drought_events(
        values, threshold=1.0, direction="above", min_duration=2, max_gap=1, periods=periods
    )

    assert isinstance(result, RunStatistics)
    np.testing.assert_array_equal(result.events.start_index, expected.events.start_index)
    np.testing.assert_array_equal(result.count, expected.count)
    np.testing.assert_array_equal(result.total_duration, expected.total_duration)


def test_gridded_events_report_summaries_and_a_per_cell_run_set() -> None:
    """A gridded DataArray yields gridded summaries plus each cell's tidy run structure."""
    values = np.array(
        [
            [-2.0, -2.0, 0.0, -2.0, -2.0, -2.0],
            [0.0, -2.0, -2.0, -2.0, 0.0, -2.0],
        ]
    ).T
    periods = pd.date_range("2020-01-01", periods=values.shape[0], freq="MS")
    da = xr.DataArray(values, dims=("time", "site"), coords={"time": periods, "site": ["a", "b"]})

    result = energy.energy_drought_events(da, threshold=-1.0, periods=da.time.dt.year)

    assert isinstance(result, xr.Dataset)
    assert set(result.data_vars) == {"count", "total_duration", "max_magnitude", "valid_count", "frequency", "events"}
    assert result["count"].dims == ("period", "site")
    np.testing.assert_array_equal(result["period"].values, [2020])
    for site in ("a", "b"):
        expected = energy.energy_drought_events(da.sel(site=site), threshold=-1.0, periods=da.time.dt.year.values)
        assert isinstance(expected, RunStatistics)
        np.testing.assert_array_equal(result["count"].sel(site=site).values, expected.count)
        np.testing.assert_array_equal(result["total_duration"].sel(site=site).values, expected.total_duration)
        cell_events = result["events"].sel(site=site).item()
        np.testing.assert_array_equal(cell_events.start_index, expected.events.start_index)
        np.testing.assert_array_equal(cell_events.duration, expected.events.duration)


def test_gridded_events_match_the_numpy_series_without_periods() -> None:
    """With no period labels every summary sits under the single whole-series period."""
    values = np.array([[-2.0, 0.0, -2.0, -2.0], [0.0, -2.0, -2.0, 0.0]]).T
    da = xr.DataArray(values, dims=("time", "site"), coords={"site": ["a", "b"]})

    result = energy.energy_drought_events(da, threshold=-1.0)

    assert isinstance(result, xr.Dataset)
    assert result["count"].dims == ("period", "site")
    np.testing.assert_array_equal(result["period"].values, [0])
    # site a: two below-threshold runs (lengths 1 and 2); site b: one run of length 2
    np.testing.assert_array_equal(result["count"].values.ravel(), [2, 1])
    np.testing.assert_array_equal(result["total_duration"].values.ravel(), [3, 2])


def test_gridded_events_keep_lazy_dask_summaries() -> None:
    """A spatial Dask chunk stays lazy; the time axis must be a single chunk for runs."""
    values = np.array([[-2.0, 0.0, -2.0, -2.0], [0.0, -2.0, -2.0, 0.0]]).T
    da = xr.DataArray(values, dims=("time", "site"), coords={"site": ["a", "b"]}).chunk({"time": -1, "site": 1})

    result = energy.energy_drought_events(da, threshold=-1.0)

    assert isinstance(result, xr.Dataset)
    assert result["count"].chunks is not None
    np.testing.assert_array_equal(result["count"].compute().values.ravel(), [2, 1])


def test_gridded_events_reject_a_split_time_chunk() -> None:
    """Runs span the whole series, so a chunked time axis is rejected instead of silently mis-counted."""
    da = xr.DataArray(
        np.zeros((4, 2)), dims=("time", "site"), coords={"time": pd.date_range("2020-01-01", periods=4)}
    ).chunk({"time": 2})

    with pytest.raises(CoordinateValidationError, match="chunk"):
        energy.energy_drought_events(da)


def test_period_label_count_must_match_the_time_axis() -> None:
    """A short period label sequence is rejected rather than broadcast."""
    da = xr.DataArray(np.zeros((4, 2)), dims=("time", "site"), coords={"time": pd.date_range("2020-01-01", periods=4)})

    with pytest.raises(ValueError, match="periods"):
        energy.energy_drought_events(da, periods=[2020, 2021])


@pytest.mark.parametrize("gridded", [False, True])
def test_events_require_the_declared_time_dimension(gridded: bool) -> None:
    da = xr.DataArray([-2.0, 0.0], dims="step")
    if gridded:
        da = da.expand_dims(site=[0])

    with pytest.raises(CoordinateValidationError, match="time"):
        energy.energy_drought_events(da)
    result = energy.energy_drought_events(da, time_dim="step")
    assert isinstance(result, xr.Dataset if gridded else RunStatistics)


@pytest.mark.parametrize("gridded", [False, True])
@pytest.mark.parametrize("label_contract", ["reordered", "missing", "wrong_dimension"])
def test_events_require_aligned_period_coordinates(gridded: bool, label_contract: str) -> None:
    da = xr.DataArray([-2.0, 0.0], dims="time", coords={"time": [1, 2]})
    if gridded:
        da = da.expand_dims(site=[0])
    labels = xr.DataArray([2021, 2020], dims="time", coords={"time": [2, 1]})
    if label_contract == "missing":
        labels = labels.drop_vars("time")
    elif label_contract == "wrong_dimension":
        labels = labels.rename(time="step")

    with pytest.raises(CoordinateValidationError, match="coordinate"):
        energy.energy_drought_events(da, periods=labels)


@pytest.mark.parametrize("gridded", [False, True])
@pytest.mark.parametrize(
    "options, argument",
    [
        ({"direction": "sideways"}, "direction"),
        ({"threshold": np.array([-1.0, -2.0])}, "threshold"),
        ({"threshold": "-1.0"}, "threshold"),
        ({"threshold": True}, "threshold"),
        ({"threshold": np.nan}, "threshold"),
        ({"threshold": np.ma.masked}, "threshold"),
        ({"min_duration": 0}, "min_duration"),
        ({"min_duration": True}, "min_duration"),
        ({"max_gap": -1}, "max_gap"),
        ({"max_gap": 0.5}, "max_gap"),
    ],
)
def test_event_options_use_the_run_contract_before_dispatch(
    gridded: bool, options: dict[str, Any], argument: str
) -> None:
    da = xr.DataArray([-2.0, 0.0], dims="time")
    if gridded:
        da = da.expand_dims(site=[0])
    da = da.chunk({"time": -1})

    with pytest.raises(InvalidArgumentError) as numpy_error:
        energy.energy_drought_events([-2.0, 0.0], **options)
    with pytest.raises(InvalidArgumentError) as xarray_error:
        energy.energy_drought_events(da, **options)
    assert xarray_error.value.argument_name == numpy_error.value.argument_name == argument
    assert str(xarray_error.value) == str(numpy_error.value)


@pytest.mark.parametrize("gridded", [False, True])
def test_event_tests_reject_a_non_monotonic_time_coordinate(gridded: bool) -> None:
    """Steps are run in array order, so an out-of-order time coordinate is rejected."""
    da = xr.DataArray(
        [[-2.0], [-2.0], [0.0]],
        dims=("time", "site"),
        coords={"time": np.array(["2020-01-01", "2020-01-03", "2020-01-02"], dtype="datetime64[D]")},
    )

    if not gridded:
        da = da.isel(site=0)
    with pytest.raises(CoordinateValidationError, match="monotonic"):
        energy.energy_drought_events(da)


def test_event_tests_reject_misaligned_period_labels() -> None:
    """Labels attached to different time steps would attribute events to the wrong period."""
    da = xr.DataArray([[-2.0], [0.0]], dims=("time", "site"), coords={"time": [1, 2]})
    labels = xr.DataArray([2021, 2020], dims="time", coords={"time": [2, 1]})

    with pytest.raises(CoordinateValidationError, match="coordinate"):
        energy.energy_drought_events(da, periods=labels)


def test_event_tests_reject_masked_period_labels() -> None:
    """A masked label is missing, which the NumPy contract already rejects."""
    da = xr.DataArray([[-2.0], [0.0]], dims=("time", "site"))

    with pytest.raises(InvalidArgumentError, match="missing labels"):
        energy.energy_drought_events(da, periods=np.ma.array([2020, 2021], mask=[False, True]))


def test_event_tests_reject_an_empty_time_axis() -> None:
    """A gridded empty series has no steps to run over; the series path reports empty statistics."""
    da = xr.DataArray(np.empty((0, 2)), dims=("time", "site"))

    with pytest.raises(InsufficientDataError):
        energy.energy_drought_events(da)
