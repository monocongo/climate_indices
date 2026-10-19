"""Standardized energy indices as xarray adapters (#1174).

SREPI, the standardized demand index, and SRLI wrap the generic seam, so these tests
pin what the adapter layer adds: beta dispatch, the wrapper defaults, registry metadata,
gridded and Dask parity against the NumPy functions, and the retained restrictions.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from climate_indices import compute, energy, indices
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.exceptions import InvalidArgumentError

MONTHLY = compute.Periodicity.monthly
DATA_START_YEAR = 1985
CALIBRATION_START = 1990
CALIBRATION_END = 2019
ARGS = (3, DATA_START_YEAR, CALIBRATION_START, CALIBRATION_END, MONTHLY)
_CAL_START = CALIBRATION_START
_CAL_END = DATA_START_YEAR + 34


def _values(seed: int = 0, years: int = 35) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.05, 0.95, years * 12)


def _monthly(seed: int = 0) -> xr.DataArray:
    values = _values(seed)
    time = xr.date_range(f"{DATA_START_YEAR}-01-01", periods=values.size, freq="MS")
    return xr.DataArray(values, dims="time", coords={"time": time}, attrs={"units": "1"})


def _gridded(seed: int = 0) -> xr.DataArray:
    values = np.random.default_rng(seed).uniform(0.05, 0.95, (35 * 12, 2, 3))
    time = xr.date_range(f"{DATA_START_YEAR}-01-01", periods=values.shape[0], freq="MS")
    return xr.DataArray(
        values, dims=("time", "lat", "lon"), coords={"time": time, "lat": [10.0, 20.0], "lon": [1.0, 2.0, 3.0]}
    )


@pytest.mark.parametrize(
    "function, aggregation",
    [(energy.srepi, "mean"), (energy.standardized_demand_index, "sum")],
)
def test_one_dimensional_series_matches_numpy(function: object, aggregation: str) -> None:
    """A 1-D DataArray standardizes exactly as the NumPy wrapper does."""
    result = function(  # type: ignore[operator]
        _monthly(), 3, calibration_year_initial=CALIBRATION_START, calibration_year_final=CALIBRATION_END
    )
    expected = function(_values(), *ARGS)  # type: ignore[operator]

    np.testing.assert_array_equal(result.values, expected)
    assert result.dims == ("time",)
    assert result.attrs["aggregation"] == aggregation


def test_srli_matches_numpy() -> None:
    """SRLI keeps its signed-series path through the adapter."""
    series = np.random.default_rng(3).normal(0.0, 1.0, 35 * 12)
    time = xr.date_range(f"{DATA_START_YEAR}-01-01", periods=series.size, freq="MS")
    da = xr.DataArray(series, dims="time", coords={"time": time})

    result = energy.srli(da, 3, calibration_year_initial=CALIBRATION_START, calibration_year_final=CALIBRATION_END)

    np.testing.assert_array_equal(result.values, energy.srli(series, *ARGS))
    assert result.attrs["aggregation"] == "sum"


def test_temporal_parameters_are_inferred_from_the_time_coordinate() -> None:
    """An omitted Calibration Period is inferred, as for the other adapters."""
    result = energy.srepi(_monthly(), 3)
    expected = energy.srepi(_values(), 3, DATA_START_YEAR, DATA_START_YEAR, DATA_START_YEAR + 34, MONTHLY)

    np.testing.assert_array_equal(result.values, expected)
    assert result.attrs["calibration_year_initial"] == DATA_START_YEAR
    assert result.attrs["calibration_year_final"] == DATA_START_YEAR + 34


@pytest.mark.parametrize("index_name", ["srepi", "standardized_demand_index", "srli"])
def test_metadata_comes_from_the_energy_registry_entry(index_name: str) -> None:
    """Each wrapper stamps its own registry entry, including the supply/demand caveat."""
    function = getattr(energy, index_name)
    result = function(_monthly(), 3, calibration_year_initial=_CAL_START, calibration_year_final=_CAL_END)

    assert result.attrs["long_name"] == CF_METADATA[index_name]["long_name"]
    assert result.attrs["units"] == "1"
    assert result.attrs["references"] == CF_METADATA[index_name]["references"]
    assert "climate_indices" in result.attrs["history"]
    assert result.attrs["scale"] == 3


@pytest.mark.parametrize("index_name", ["srepi", "standardized_demand_index", "srli"])
def test_gridded_input_matches_a_per_cell_numpy_run(index_name: str) -> None:
    """Each grid cell is standardized independently with the same arguments."""
    function = getattr(energy, index_name)
    cube = _gridded()
    result = function(cube, 3, calibration_year_initial=_CAL_START, calibration_year_final=_CAL_END)

    assert result.dims == cube.dims
    for lat_index in range(cube.sizes["lat"]):
        for lon_index in range(cube.sizes["lon"]):
            expected = function(cube.isel(lat=lat_index, lon=lon_index).values, *ARGS)
            np.testing.assert_array_equal(result.isel(lat=lat_index, lon=lon_index).values, expected)


def test_gridded_input_matches_numpy_with_tied_values() -> None:
    """Tied capacity factors keep the NumPy ranks, so a grid cell cannot drift."""
    tied = np.concatenate([np.random.default_rng(7).permutation(np.arange(1, 13) / 13) for _ in range(35)])
    time = xr.date_range(f"{DATA_START_YEAR}-01-01", periods=tied.size, freq="MS")
    cube = xr.DataArray(
        tied[:, None, None], dims=("time", "lat", "lon"), coords={"time": time, "lat": [1.0], "lon": [1.0]}
    )

    result = energy.srepi(cube, 12)

    np.testing.assert_array_equal(
        result.values[:, 0, 0], energy.srepi(tied, 12, DATA_START_YEAR, DATA_START_YEAR, DATA_START_YEAR + 34, MONTHLY)
    )


def test_source_nans_propagate_as_missing() -> None:
    """A missing source step stays missing in the output."""
    cube = _gridded()
    cube[0, 0, 0] = np.nan

    result = energy.srepi(cube, 3, calibration_year_initial=_CAL_START, calibration_year_final=_CAL_END)

    assert np.isnan(result.values[0, 0, 0])


@pytest.mark.parametrize("index_name", ["srepi", "srli"])
def test_dask_backed_input_matches_the_eager_result(index_name: str) -> None:
    """A time-chunked Dask cube stays lazy and produces the same values."""
    function = getattr(energy, index_name)
    cube = _gridded().chunk({"time": -1, "lat": 1})
    result = function(cube, 3, calibration_year_initial=_CAL_START, calibration_year_final=_CAL_END)

    assert result.chunks is not None
    np.testing.assert_array_equal(
        result.compute().values,
        function(_gridded(), 3, calibration_year_initial=_CAL_START, calibration_year_final=_CAL_END).values,
    )


def test_srepi_defaults_to_mean_aggregation() -> None:
    """A capacity factor is averaged, not accumulated, unless the caller asks for sum."""
    summed = energy.srepi(
        _monthly(), 3, calibration_year_initial=_CAL_START, calibration_year_final=_CAL_END, aggregation="sum"
    )
    averaged = energy.srepi(_monthly(), 3, calibration_year_initial=_CAL_START, calibration_year_final=_CAL_END)

    assert averaged.attrs["aggregation"] == "mean"
    assert summed.attrs["aggregation"] == "sum"
    assert not np.array_equal(summed.values, averaged.values)


def test_beta_distribution_standardizes_a_capacity_factor() -> None:
    """Beta stays available for a bounded capacity factor with mean aggregation."""
    result = energy.srepi(
        _monthly(),
        3,
        calibration_year_initial=_CAL_START,
        calibration_year_final=_CAL_END,
        distribution=indices.Distribution.beta,
    )

    assert result.attrs["distribution"] == "beta"
    assert np.isfinite(result.values[3:]).all()


def test_srepi_rejects_a_distribution_the_wrapper_does_not_offer() -> None:
    """Gamma is not an energy-wrapper option, and the seam reports it."""
    with pytest.raises(InvalidArgumentError, match="empirical and beta"):
        energy.srepi(
            _monthly(),
            3,
            calibration_year_initial=_CAL_START,
            calibration_year_final=_CAL_END,
            distribution=indices.Distribution.gamma,
        )


def test_srli_rejects_a_parametric_distribution() -> None:
    """A signed series keeps the empirical rank path on the xarray arm too."""
    with pytest.raises(InvalidArgumentError, match="empirical distribution"):
        energy.srli(
            _monthly(),
            3,
            calibration_year_initial=_CAL_START,
            calibration_year_final=_CAL_END,
            distribution=indices.Distribution.beta,
        )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize(
    "index_name, distribution, message",
    [
        ("srepi", indices.Distribution.gamma, "empirical and beta"),
        ("srepi", indices.Distribution.pearson, "empirical and beta"),
        ("standardized_demand_index", indices.Distribution.gamma, "empirical and beta"),
        ("standardized_demand_index", indices.Distribution.pearson, "empirical and beta"),
        ("srli", indices.Distribution.gamma, "empirical distribution"),
        ("srli", indices.Distribution.pearson, "empirical distribution"),
        ("srli", indices.Distribution.beta, "empirical distribution"),
    ],
)
def test_gridded_distribution_restrictions_apply_at_call_time(
    index_name: str, distribution: indices.Distribution, message: str, lazy: bool
) -> None:
    cube = _gridded()
    if lazy:
        cube = cube.chunk({"time": -1, "lat": 1})
    function = getattr(energy, index_name)

    with pytest.raises(InvalidArgumentError, match=message):
        function(cube, 3, distribution=distribution)


def test_probability_output_scale_is_unclipped() -> None:
    """The output scale reaches the seam through the adapter."""
    result = energy.srepi(
        _monthly(), 3, calibration_year_initial=_CAL_START, calibration_year_final=_CAL_END, output_scale="probability"
    )

    assert np.nanmin(result.values) >= 0.0
    assert np.nanmax(result.values) <= 1.0
