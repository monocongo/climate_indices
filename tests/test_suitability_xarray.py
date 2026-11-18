"""Tests for the xarray/Dask route of ``suitability.compare_distributions`` (#1585, ADR-0028)."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import compute, suitability
from climate_indices.exceptions import (
    BetaFeatureWarning,
    ClimateIndicesWarning,
    CoordinateValidationError,
    InsufficientDataError,
    InvalidArgumentError,
)
from climate_indices.vocabulary import Distribution
from climate_indices.xarray_adapter import _build_calendar_plan

MONTHLY = compute.Periodicity.monthly
START, YEARS = 1950, 60
END = START + YEARS - 1

DAILY = compute.Periodicity.daily

pytestmark = [
    pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning"),
    # the fits take the mean of a fully masked cell, which warns but is the documented all-missing outcome
    pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning"),
]

CANDIDATE_VARIABLES = ("log_likelihood", "aic", "aicc", "bic", "n_parameters", "fit_failed")
STEP_VARIABLES = ("n", "best", "delta_to_runner_up")
# every ADR-0028 section 6 attribute that applies to the xarray route
ADR_ATTRIBUTES = (
    "climate_indices_candidates",
    "climate_indices_fitting_methods",
    "climate_indices_criterion",
    "climate_indices_log_likelihood_at",
    "scale",
    "calibration_year_initial",
    "calibration_year_final",
    "climate_indices_version",
    "climate_indices_provenance",
)


def _times(years: int = YEARS) -> pd.DatetimeIndex:
    return pd.date_range(f"{START}-01-01", periods=years * 12, freq="MS")


def _precip(seed: int, *shape: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    season = 40.0 + 30.0 * np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    base = np.tile(season, YEARS).reshape((YEARS * 12,) + (1,) * len(shape))
    return rng.gamma(shape=2.0, scale=base / 2.0, size=(YEARS * 12, *shape))


def _grid(seed: int = 0, *, masked_cell: bool = True) -> xr.DataArray:
    data = _precip(seed, 3, 2)
    if masked_cell:
        data[:, 0, 0] = np.nan
    return xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": _times()}, attrs={"units": "mm"})


def _compare(values: xr.DataArray, scale: int, **kwargs: Any) -> xr.Dataset:
    """Call the public comparison on one DataArray and keep its Dataset result."""
    result = suitability.compare_distributions(values, scale, **kwargs)
    assert isinstance(result, xr.Dataset)
    return result


def _core(values: np.ndarray, scale: int, **kwargs: Any) -> suitability.DistributionComparison:
    """The NumPy core on the same block, the way the xarray route calls it."""
    kwargs.setdefault("index", "spi")
    result = suitability._compare_numpy(
        values,
        scale,
        data_start_year=START,
        calibration_year_initial=START,
        calibration_year_final=END,
        periodicity=MONTHLY,
        require_valid=False,
        **kwargs,
    )
    assert isinstance(result, suitability.DistributionComparison)
    return result


def _assert_matches_core(dataset: xr.Dataset, core: suitability.DistributionComparison) -> None:
    labels = [d.value for d in core.candidates]
    assert list(dataset["candidate"].values) == labels
    for name in ("log_likelihood", "aic", "aicc", "bic"):
        np.testing.assert_allclose(dataset[name].values, getattr(core, name), equal_nan=True, rtol=1e-12)
    np.testing.assert_array_equal(dataset["n_parameters"].values, core.n_parameters)
    np.testing.assert_array_equal(dataset["fit_failed"].values, core.fit_failed)
    np.testing.assert_array_equal(dataset["n"].values, core.n)
    np.testing.assert_allclose(dataset["delta_to_runner_up"].values, core.delta_to_runner_up, equal_nan=True)
    expected_best = np.where(core.best >= 0, np.array(labels + ["none"])[core.best], "none")
    np.testing.assert_array_equal(dataset["best"].values, expected_best)
    for position, distribution in enumerate(core.candidates):
        for name, array in core.parameters[distribution].items():
            np.testing.assert_allclose(dataset[name].isel(candidate=position).values, array, equal_nan=True, rtol=1e-12)


def test_three_dimensional_grid_matches_the_numpy_core() -> None:
    grid = _grid()
    dataset = _compare(grid, 3, index="spi")
    assert isinstance(dataset, xr.Dataset)
    assert dataset["log_likelihood"].dims == ("candidate", "month", "lat", "lon")
    assert dataset["best"].dims == ("month", "lat", "lon")
    assert list(dataset["month"].values) == list(range(1, 13))
    _assert_matches_core(dataset, _core(grid.values, 3, spatial_time_major=True))


def test_one_dimensional_series_matches_the_numpy_core() -> None:
    series = _grid(masked_cell=False).isel(lat=1, lon=1)
    dataset = _compare(series, 1, index="spi")
    assert dataset["log_likelihood"].dims == ("candidate", "month")
    _assert_matches_core(dataset, _core(series.values, 1))


def test_two_dimensional_input_is_read_cell_by_cell() -> None:
    grid = _grid(masked_cell=False).isel(lon=0)
    dataset = _compare(grid, 1, index="spi")
    assert dataset["log_likelihood"].dims == ("candidate", "month", "lat")
    for cell in range(grid.sizes["lat"]):
        _assert_matches_core(dataset.isel(lat=cell), _core(grid.isel(lat=cell).values, 1))


def test_spei_route_scores_the_water_balance_with_the_log_logistic_and_gev_candidates() -> None:
    precip = _grid(seed=1, masked_cell=False)
    pet = precip * 0.6
    dataset = _compare(precip, 3, index="spei", pet_mm=pet)
    assert list(dataset["candidate"].values) == ["pearson", "gamma", "loglogistic", "gev"]
    # the log-logistic and GEV families have no zero mass, and gamma carries the only unused `shape`
    assert np.isnan(dataset["shape"].sel(candidate="gamma")).all()
    assert np.isfinite(dataset["shape"].sel(candidate="loglogistic")).any()
    assert np.isfinite(dataset["shape"].sel(candidate="gev")).any()
    _assert_matches_core(dataset, _core(precip.values, 3, index="spei", pet_mm=pet.values, spatial_time_major=True))


def test_standardized_index_route_applies_the_aggregation() -> None:
    grid = _grid(seed=2, masked_cell=False)
    dataset = _compare(grid, 3, index="standardized_index", aggregation="mean")
    _assert_matches_core(
        dataset, _core(grid.values, 3, index="standardized_index", aggregation="mean", spatial_time_major=True)
    )
    assert dataset["best"].attrs["aggregation"] == "mean"


def _masked_cases() -> dict[str, tuple[xr.DataArray, dict[str, int]]]:
    """Inputs whose cell ``lat=0`` is all missing: the series, the cell and a whole Dask chunk."""
    grid = _grid()
    return {
        "3d_block": (grid, {"lat": 0, "lon": 0}),
        "1d_series": (grid.isel(lat=0, lon=0), {}),
        "2d_cell": (grid.isel(lon=0), {"lat": 0}),
        "3d_dask_chunk": (grid.chunk({"time": -1, "lat": 1, "lon": 1}), {"lat": 0, "lon": 0}),
        "2d_dask": (grid.isel(lon=0).chunk({"time": -1, "lat": 1}), {"lat": 0}),
    }


@pytest.mark.parametrize("case", list(_masked_cases()))
def test_all_missing_series_cell_or_chunk_has_no_best_candidate(case: str) -> None:
    values, masked = _masked_cases()[case]
    dataset = _compare(values, 3, index="spi").compute()
    cell = dataset.isel(masked)
    assert (cell["best"] == "none").all()
    assert (cell["n"] == 0).all()
    assert cell["delta_to_runner_up"].isnull().all()
    assert cell["fit_failed"].all()
    if masked:
        other = dataset.isel(dict.fromkeys(masked, 1))
        assert (other["best"] != "none").any()


def test_all_missing_spei_cell_has_no_best_candidate() -> None:
    precip = _grid()
    dataset = _compare(precip.isel(lon=0), 3, index="spei", pet_mm=precip.isel(lon=0) * 0.6)
    assert (dataset.isel(lat=0)["best"] == "none").all()
    assert (dataset.isel(lat=1)["best"] != "none").any()


def test_daily_input_matches_the_numpy_core_on_the_366_day_calendar() -> None:
    times = pd.date_range("1980-01-01", "2011-12-31", freq="D")
    values = np.random.default_rng(3).gamma(2.0, 3.0, size=(len(times), 2))
    daily = xr.DataArray(values, dims=("time", "cell"), coords={"time": times})
    dataset = _compare(daily, 7, index="spi")
    assert dataset["log_likelihood"].dims == ("candidate", "dayofyear", "cell")
    assert list(dataset["dayofyear"].values) == list(range(1, 367))
    assert dataset["best"].attrs["periodicity"] == "daily"
    assert (dataset["best"] != "none").any()
    plan = _build_calendar_plan(daily["time"], DAILY)
    assert plan is not None
    for cell in range(2):
        core = suitability._compare_numpy(
            plan.to_all_leap(values[:, cell]),
            7,
            index="spi",
            data_start_year=1980,
            calibration_year_initial=1980,
            calibration_year_final=2011,
            periodicity=DAILY,
            require_valid=False,
        )
        _assert_matches_core(dataset.isel(cell=cell), core)


@pytest.mark.parametrize("chunks", [{"lat": 1}, {"lat": 2, "lon": 1}])
def test_dask_spatial_chunks_match_the_in_memory_route_and_stay_lazy(chunks: dict[str, int]) -> None:
    grid = _grid(seed=4)
    eager = _compare(grid, 3, index="spi")
    lazy = _compare(grid.chunk({"time": -1, **chunks}), 3, index="spi")
    assert all(variable.chunks is not None for variable in lazy.data_vars.values())
    xr.testing.assert_equal(lazy.compute(), eager)


def test_dask_spei_matches_the_in_memory_route() -> None:
    precip = _grid(seed=5, masked_cell=False)
    pet = precip * 0.5
    eager = _compare(precip, 3, index="spei", pet_mm=pet)
    lazy = _compare(precip.chunk({"time": -1, "lat": 1}), 3, index="spei", pet_mm=pet.chunk({"time": -1, "lat": 1}))
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_dask_pet_chunked_differently_from_values_matches_the_in_memory_route() -> None:
    precip = _grid(seed=5, masked_cell=False)
    pet = precip * 0.5
    eager = _compare(precip, 3, index="spei", pet_mm=pet)
    lazy = _compare(precip.chunk({"time": -1, "lat": 1}), 3, index="spei", pet_mm=pet.chunk({"time": -1, "lat": 2}))
    xr.testing.assert_allclose(lazy.compute(), eager)


@pytest.mark.parametrize(
    ("index", "candidates", "criterion"),
    [
        ("spi", [Distribution.gamma], "aic"),
        ("spi", [Distribution.gamma, Distribution.pearson], "bic"),
        ("spei", [Distribution.loglogistic, Distribution.gamma], "aicc"),
    ],
)
def test_candidate_subsets_and_criteria_match_the_core(
    index: str, candidates: list[Distribution], criterion: str
) -> None:
    precip = _grid(seed=10, masked_cell=False)
    extra: dict[str, Any] = {"pet_mm": precip * 0.6} if index == "spei" else {}
    dataset = _compare(precip, 3, index=index, candidates=candidates, criterion=criterion, **extra)
    core = _core(
        precip.values,
        3,
        index=index,
        candidates=candidates,
        criterion=criterion,
        spatial_time_major=True,
        **({"pet_mm": extra["pet_mm"].values} if extra else {}),
    )
    _assert_matches_core(dataset, core)
    attributes = dataset["best"].attrs
    # the candidates are recorded in canonical enum order, whatever order the caller named them
    assert json.loads(attributes["climate_indices_candidates"]) == [d.value for d in core.candidates]
    assert attributes["climate_indices_criterion"] == criterion


def test_non_default_time_dimension_name_and_position() -> None:
    grid = _grid(seed=11, masked_cell=False)
    expected = _compare(grid, 3, index="spi")
    renamed = _compare(grid.rename(time="t"), 3, index="spi", time_dim="t")
    xr.testing.assert_equal(renamed, expected)
    last = _compare(grid.transpose("lat", "lon", "time"), 3, index="spi")
    xr.testing.assert_equal(last, expected)


def test_non_default_calibration_window_matches_the_core() -> None:
    grid = _grid(seed=12, masked_cell=False)
    dataset = _compare(grid, 3, index="spi", calibration_year_initial=1960, calibration_year_final=1990)
    core = suitability._compare_numpy(
        grid.values,
        3,
        index="spi",
        data_start_year=START,
        calibration_year_initial=1960,
        calibration_year_final=1990,
        periodicity=MONTHLY,
        spatial_time_major=True,
    )
    _assert_matches_core(dataset, core)


@pytest.mark.parametrize("name", ["n", "best", "scale", "candidate", "month"])
def test_a_dimension_named_like_an_output_is_rejected(name: str) -> None:
    grid = _grid(masked_cell=False).rename(lat=name)
    with pytest.raises(InvalidArgumentError, match="named like an output"):
        _compare(grid, 3, index="spi")


@pytest.mark.parametrize("name", ["n", "best", "scale"])
def test_a_scalar_coordinate_named_like_an_output_is_rejected(name: str) -> None:
    grid = _grid(masked_cell=False).assign_coords({name: 3})
    with pytest.raises(InvalidArgumentError, match="named like an output"):
        _compare(grid, 3, index="spi")


def test_a_time_indexed_coordinate_named_like_an_output_is_accepted() -> None:
    # apply_ufunc drops a coordinate over the core time dimension, so it cannot collide
    grid = _grid(seed=16, masked_cell=False)
    tagged = grid.assign_coords(month=("time", grid["time"].dt.month.values))
    xr.testing.assert_equal(
        _compare(tagged, 3, index="spi"),
        _compare(grid, 3, index="spi"),
    )


def test_a_time_dimension_named_like_an_output_is_rejected() -> None:
    grid = _grid(masked_cell=False).rename(time="month")
    with pytest.raises(InvalidArgumentError, match="named like an output"):
        _compare(grid, 3, index="spi", time_dim="month")


def test_time_split_across_dask_chunks_raises() -> None:
    grid = _grid().chunk({"time": 120})
    with pytest.raises(CoordinateValidationError, match="single chunk"):
        _compare(grid, 3, index="spi")


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_every_variable_carries_the_adr_attributes(lazy: bool) -> None:
    grid = _grid(seed=6)
    dataset = _compare(grid.chunk({"time": -1, "lat": 1}) if lazy else grid, 3, index="spi")
    assert set(dataset.data_vars) >= {*CANDIDATE_VARIABLES, *STEP_VARIABLES, "alpha", "beta", "loc", "scale", "skew"}
    assert dataset["fit_failed"].dtype == bool
    assert dataset["n"].dtype.kind == dataset["n_parameters"].dtype.kind == "i"
    assert dataset["best"].dtype.kind == "U"
    for name, variable in dataset.data_vars.items():
        attributes = variable.attrs
        for key in ADR_ATTRIBUTES:
            assert key in attributes, f"{name} lacks {key}"
        assert json.loads(attributes["climate_indices_candidates"]) == ["pearson", "gamma"]
        assert json.loads(attributes["climate_indices_fitting_methods"]) == {
            "pearson": "l_moments",
            "gamma": "thom_mle_approximation",
        }
        assert attributes["climate_indices_log_likelihood_at"] == "fitted_estimates"
        assert attributes["climate_indices_criterion"] == "aicc"
        assert attributes["scale"] == "3"
        assert (attributes["calibration_year_initial"], attributes["calibration_year_final"]) == (str(START), str(END))
        assert attributes.get("units", "1") != "mm", f"{name} inherited the precipitation units"
        assert all(not isinstance(value, bool | list | type(None)) for value in attributes.values())
    # a calibration window beyond the record is recorded as the years actually used
    clamped = _compare(grid, 3, index="spi", calibration_year_initial=START - 10, calibration_year_final=END + 10)
    assert clamped["best"].attrs["calibration_year_initial"] == str(START)
    assert clamped["best"].attrs["calibration_year_final"] == str(END)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_netcdf_round_trip_keeps_values_and_attributes(tmp_path: Path, lazy: bool) -> None:
    grid = _grid(seed=7)
    dataset = _compare(grid.chunk({"time": -1, "lat": 1}) if lazy else grid, 3, index="spi")
    path = tmp_path / "comparison.nc"
    dataset.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as reopened:
        loaded = reopened.load()
    assert list(loaded["candidate"].values) == ["pearson", "gamma"]
    assert loaded["fit_failed"].dtype == bool
    assert loaded["n"].dtype.kind == "i"
    np.testing.assert_array_equal(loaded["best"].values, dataset["best"].values)
    np.testing.assert_array_equal(loaded["fit_failed"].values, dataset["fit_failed"].values)
    np.testing.assert_allclose(loaded["aicc"].values, dataset["aicc"].values, equal_nan=True)
    for name in dataset.data_vars:
        for key in ADR_ATTRIBUTES:
            assert loaded[name].attrs[key] == dataset[name].attrs[key], f"{name}.{key}"


def test_beta_warning_is_emitted_once_per_call_not_once_per_block() -> None:
    grid = _grid(seed=8).chunk({"time": -1, "lat": 1})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _compare(grid, 3, index="spi").compute()
    assert sum(issubclass(item.category, BetaFeatureWarning) for item in caught) == 1


def test_numpy_route_warnings_point_at_the_caller() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        suitability.compare_distributions(
            _precip(0),
            13,
            index="spi",
            data_start_year=START,
            calibration_year_initial=START,
            calibration_year_final=END,
            periodicity=MONTHLY,
        )
    overlap = [
        item for item in caught if issubclass(item.category, ClimateIndicesWarning) and "exceeds" in str(item.message)
    ]
    assert overlap
    assert all(Path(item.filename).name == Path(__file__).name for item in overlap)


def test_inferred_and_explicit_temporal_arguments_agree() -> None:
    grid = _grid(seed=9, masked_cell=False)
    inferred = _compare(grid, 3, index="spi")
    explicit = _compare(
        grid,
        3,
        index="spi",
        data_start_year=START,
        calibration_year_initial=START,
        calibration_year_final=END,
        periodicity=MONTHLY,
    )
    xr.testing.assert_equal(inferred, explicit)


def test_numpy_input_still_requires_the_temporal_arguments() -> None:
    precip = _precip(0)
    with pytest.raises(InvalidArgumentError, match="required for NumPy input"):
        suitability.compare_distributions(precip, 1, index="spi")


def test_numpy_input_still_returns_the_dataclass() -> None:
    result = suitability.compare_distributions(
        _precip(0),
        1,
        index="spi",
        data_start_year=START,
        calibration_year_initial=START,
        calibration_year_final=END,
        periodicity=MONTHLY,
    )
    assert isinstance(result, suitability.DistributionComparison)
    assert result.candidates == (Distribution.pearson, Distribution.gamma)


def test_non_numeric_daily_input_is_rejected_before_calendar_adaptation() -> None:
    times = pd.date_range("1980-01-01", "2011-12-31", freq="D")
    numeric = xr.DataArray(np.random.default_rng(0).gamma(2.0, 3.0, times.size), dims="time", coords={"time": times})
    as_text = numeric.astype(str)
    with pytest.raises(InvalidArgumentError, match="values must hold numeric values"):
        suitability.compare_distributions(as_text, 7, index="spi")
    with pytest.raises(InvalidArgumentError, match="pet_mm must hold numeric values"):
        suitability.compare_distributions(numeric, 7, index="spei", pet_mm=as_text)


def test_pet_must_be_a_matching_dataarray() -> None:
    grid = _grid(masked_cell=False)
    as_array, one_column = grid.values, grid.isel(lon=0)
    with pytest.raises(InvalidArgumentError, match="DataArray"):
        suitability.compare_distributions(grid, 3, index="spei", pet_mm=as_array)
    with pytest.raises(InvalidArgumentError, match="same dimensions"):
        suitability.compare_distributions(grid, 3, index="spei", pet_mm=one_column)
    with pytest.raises(InvalidArgumentError, match="requires pet_mm"):
        suitability.compare_distributions(grid, 3, index="spei")


def test_pet_must_be_an_array_for_numpy_values() -> None:
    grid = _grid(masked_cell=False)
    # pairing a DataArray by position would ignore labels, so a transposed PET is refused
    with pytest.raises(InvalidArgumentError, match="NumPy array"):
        suitability.compare_distributions(
            grid.values,
            3,
            index="spei",
            data_start_year=START,
            calibration_year_initial=START,
            calibration_year_final=END,
            periodicity=MONTHLY,
            pet_mm=grid.transpose("time", "lon", "lat") * 0.5,
        )
    with pytest.raises(InvalidArgumentError, match="applies only to index='spei'"):
        suitability.compare_distributions(
            grid.values,
            3,
            index="spi",
            data_start_year=START,
            calibration_year_initial=START,
            calibration_year_final=END,
            periodicity=MONTHLY,
            pet_mm=grid,
        )


def test_non_january_start_is_rejected() -> None:
    grid = _grid(masked_cell=False).isel(time=slice(1, None))
    with pytest.raises(CoordinateValidationError):
        suitability.compare_distributions(grid, 3, index="spi")


def test_a_missing_time_dimension_is_rejected() -> None:
    grid = _grid(masked_cell=False).rename(time="t")
    with pytest.raises(CoordinateValidationError, match="Time dimension 'time' not found"):
        _compare(grid, 3, index="spi")


def test_a_time_dimension_without_a_coordinate_is_rejected() -> None:
    grid = _grid(masked_cell=False).drop_vars("time")
    with pytest.raises(CoordinateValidationError, match="Unsupported calendar"):
        _compare(grid, 3, index="spi")


def test_a_non_monotonic_time_coordinate_is_rejected() -> None:
    grid = _grid(masked_cell=False)
    times = grid["time"].values.copy()
    times[[0, 1]] = times[[1, 0]]
    with pytest.raises(CoordinateValidationError, match="not monotonically increasing"):
        _compare(grid.assign_coords(time=times), 3, index="spi")


def test_a_time_coord_with_a_timezone_is_rejected() -> None:
    times = pd.date_range(f"{START}-01-01", periods=YEARS * 12, freq="MS", tz="UTC")
    values = xr.DataArray(_precip(17), dims=("time",), coords={"time": times})
    with pytest.raises(CoordinateValidationError, match="Unsupported calendar"):
        _compare(values, 3, index="spi")


def test_a_cell_coordinate_named_like_an_output_is_rejected() -> None:
    grid = _grid(seed=18, masked_cell=False).assign_coords(best=(("lat", "lon"), np.zeros((3, 2))))
    with pytest.raises(InvalidArgumentError, match="named like an output"):
        _compare(grid, 3, index="spi")


def test_a_non_datetime_time_coordinate_is_rejected() -> None:
    grid = _grid(masked_cell=False).assign_coords(time=np.arange(YEARS * 12))
    with pytest.raises(CoordinateValidationError, match="Unsupported calendar"):
        _compare(grid, 3, index="spi")


def test_hourly_input_is_rejected() -> None:
    times = pd.date_range("2000-01-01", periods=48, freq="h")
    values = xr.DataArray(np.linspace(1.0, 2.0, len(times)), dims=("time",), coords={"time": times})
    with pytest.raises(InvalidArgumentError, match="Hourly periodicity"):
        _compare(values, 3, index="spi")


def test_a_periodicity_that_mismatches_the_coordinate_is_rejected() -> None:
    times = pd.date_range("1980-01-01", periods=24 * 366, freq="D")
    values = xr.DataArray(np.linspace(1.0, 2.0, len(times)), dims=("time",), coords={"time": times})
    with pytest.raises(CoordinateValidationError, match="periodicity"):
        _compare(values, 3, index="spi", periodicity=MONTHLY)


def test_a_record_shorter_than_the_scale_raises_insufficient_data() -> None:
    # three month-starts beginning in January: a valid calendar, too short for a 12-step scale
    times = pd.date_range("1950-01-01", periods=3, freq="MS")
    values = xr.DataArray([1.0, 2.0, 3.0], dims=("time",), coords={"time": times})
    with pytest.raises(InsufficientDataError, match="Insufficient data for scale=12"):
        _compare(values, 12, index="spi")


def test_an_explicit_data_start_year_must_match_the_time_coordinate() -> None:
    grid = _grid(seed=14, masked_cell=False)
    with pytest.raises(InvalidArgumentError, match="match the first year"):
        _compare(grid, 3, index="spi", data_start_year=START + 1)
    xr.testing.assert_equal(
        _compare(grid, 3, index="spi", data_start_year=START),
        _compare(grid, 3, index="spi"),
    )


def test_a_grid_whose_first_cell_axis_is_a_calendar_length_matches_the_core() -> None:
    # a (time, 12, 2) block reads as (years, periods, *cells) unless it is declared time-major
    data = _precip(15, 12, 2)
    grid = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": _times()})
    dataset = _compare(grid, 3, index="spi")
    assert isinstance(dataset, xr.Dataset)
    assert dataset["log_likelihood"].dims == ("candidate", "month", "lat", "lon")
    _assert_matches_core(dataset, _core(grid.values, 3, spatial_time_major=True))


def test_unknown_candidate_still_raises_before_any_work() -> None:
    grid = _grid()
    with pytest.raises(InvalidArgumentError):
        _compare(grid, 3, index="spi", candidates=[Distribution.beta])
