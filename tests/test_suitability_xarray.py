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
from climate_indices.exceptions import BetaFeatureWarning, CoordinateValidationError, InvalidArgumentError
from climate_indices.vocabulary import Distribution

MONTHLY = compute.Periodicity.monthly
START, YEARS = 1950, 60
END = START + YEARS - 1

pytestmark = pytest.mark.filterwarnings("ignore::climate_indices.exceptions.BetaFeatureWarning")

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
    dataset = suitability.compare_distributions(grid, 3, index="spi")
    assert isinstance(dataset, xr.Dataset)
    assert dataset["log_likelihood"].dims == ("candidate", "month", "lat", "lon")
    assert dataset["best"].dims == ("month", "lat", "lon")
    assert list(dataset["month"].values) == list(range(1, 13))
    _assert_matches_core(dataset, _core(grid.values, 3, spatial_time_major=True))


def test_one_dimensional_series_matches_the_numpy_core() -> None:
    series = _grid(masked_cell=False).isel(lat=1, lon=1)
    dataset = suitability.compare_distributions(series, 1, index="spi")
    assert dataset["log_likelihood"].dims == ("candidate", "month")
    _assert_matches_core(dataset, _core(series.values, 1))


def test_two_dimensional_input_is_read_cell_by_cell() -> None:
    grid = _grid(masked_cell=False).isel(lon=0)
    dataset = suitability.compare_distributions(grid, 1, index="spi")
    assert dataset["log_likelihood"].dims == ("candidate", "month", "lat")
    for cell in range(grid.sizes["lat"]):
        _assert_matches_core(dataset.isel(lat=cell), _core(grid.isel(lat=cell).values, 1))


def test_spei_route_scores_the_water_balance_with_the_log_logistic_candidate() -> None:
    precip = _grid(seed=1, masked_cell=False)
    pet = precip * 0.6
    dataset = suitability.compare_distributions(precip, 3, index="spei", pet_mm=pet)
    assert list(dataset["candidate"].values) == ["pearson", "gamma", "loglogistic"]
    # the log-logistic family has no zero mass, and its parameters are the only `shape` ones
    assert np.isnan(dataset["shape"].sel(candidate="gamma")).all()
    assert np.isfinite(dataset["shape"].sel(candidate="loglogistic")).any()
    _assert_matches_core(dataset, _core(precip.values, 3, index="spei", pet_mm=pet.values, spatial_time_major=True))


def test_standardized_index_route_applies_the_aggregation() -> None:
    grid = _grid(seed=2, masked_cell=False)
    dataset = suitability.compare_distributions(grid, 3, index="standardized_index", aggregation="mean")
    _assert_matches_core(
        dataset, _core(grid.values, 3, index="standardized_index", aggregation="mean", spatial_time_major=True)
    )
    assert dataset["best"].attrs["aggregation"] == "mean"


def test_all_missing_cell_has_no_best_candidate() -> None:
    dataset = suitability.compare_distributions(_grid(), 3, index="spi")
    cell = dataset.isel(lat=0, lon=0)
    assert (cell["best"] == "none").all()
    assert (cell["n"] == 0).all()
    assert cell["delta_to_runner_up"].isnull().all()
    assert cell["fit_failed"].all()
    assert (dataset["best"].isel(lat=1, lon=1) != "none").any()


def test_daily_input_uses_a_dayofyear_dimension() -> None:
    times = pd.date_range("1980-01-01", "2011-12-31", freq="D")
    rng = np.random.default_rng(3)
    daily = xr.DataArray(rng.gamma(2.0, 3.0, size=(len(times), 2)), dims=("time", "cell"), coords={"time": times})
    dataset = suitability.compare_distributions(daily, 7, index="spi")
    assert dataset["log_likelihood"].dims == ("candidate", "dayofyear", "cell")
    assert dataset.sizes["dayofyear"] == 366
    assert dataset["best"].attrs["periodicity"] == "daily"
    assert set(np.unique(dataset["best"].values)) <= {"pearson", "gamma", "none"}


@pytest.mark.parametrize("chunks", [{"lat": 1}, {"lat": 2, "lon": 1}])
def test_dask_spatial_chunks_match_the_in_memory_route_and_stay_lazy(chunks: dict[str, int]) -> None:
    grid = _grid(seed=4)
    eager = suitability.compare_distributions(grid, 3, index="spi")
    lazy = suitability.compare_distributions(grid.chunk({"time": -1, **chunks}), 3, index="spi")
    assert all(variable.chunks is not None for variable in lazy.data_vars.values())
    xr.testing.assert_equal(lazy.compute(), eager)


def test_dask_spei_matches_the_in_memory_route() -> None:
    precip = _grid(seed=5, masked_cell=False)
    pet = precip * 0.5
    eager = suitability.compare_distributions(precip, 3, index="spei", pet_mm=pet)
    lazy = suitability.compare_distributions(
        precip.chunk({"time": -1, "lat": 1}), 3, index="spei", pet_mm=pet.chunk({"time": -1, "lat": 1})
    )
    xr.testing.assert_allclose(lazy.compute(), eager)


def test_time_split_across_dask_chunks_raises() -> None:
    with pytest.raises(CoordinateValidationError, match="single chunk"):
        suitability.compare_distributions(_grid().chunk({"time": 120}), 3, index="spi")


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_every_variable_carries_the_adr_attributes(lazy: bool) -> None:
    grid = _grid(seed=6)
    dataset = suitability.compare_distributions(grid.chunk({"time": -1, "lat": 1}) if lazy else grid, 3, index="spi")
    assert set(dataset.data_vars) >= {*CANDIDATE_VARIABLES, *STEP_VARIABLES, "alpha", "beta", "loc", "scale", "skew"}
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
    clamped = suitability.compare_distributions(
        grid, 3, index="spi", calibration_year_initial=START - 10, calibration_year_final=END + 10
    )
    assert clamped["best"].attrs["calibration_year_initial"] == str(START)
    assert clamped["best"].attrs["calibration_year_final"] == str(END)


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_netcdf_round_trip_keeps_values_and_attributes(tmp_path: Path, lazy: bool) -> None:
    grid = _grid(seed=7)
    dataset = suitability.compare_distributions(grid.chunk({"time": -1, "lat": 1}) if lazy else grid, 3, index="spi")
    path = tmp_path / "comparison.nc"
    dataset.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as reopened:
        loaded = reopened.load()
    assert list(loaded["candidate"].values) == ["pearson", "gamma"]
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
        suitability.compare_distributions(grid, 3, index="spi")
    assert sum(issubclass(item.category, BetaFeatureWarning) for item in caught) == 1


def test_inferred_and_explicit_temporal_arguments_agree() -> None:
    grid = _grid(seed=9, masked_cell=False)
    inferred = suitability.compare_distributions(grid, 3, index="spi")
    explicit = suitability.compare_distributions(
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
    with pytest.raises(InvalidArgumentError, match="required for NumPy input"):
        suitability.compare_distributions(_precip(0), 1, index="spi")


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


def test_pet_must_be_a_matching_dataarray() -> None:
    grid = _grid(masked_cell=False)
    with pytest.raises(InvalidArgumentError, match="DataArray"):
        suitability.compare_distributions(grid, 3, index="spei", pet_mm=grid.values)
    with pytest.raises(InvalidArgumentError, match="same dimensions"):
        suitability.compare_distributions(grid, 3, index="spei", pet_mm=grid.isel(lon=0))
    with pytest.raises(InvalidArgumentError, match="requires pet_mm"):
        suitability.compare_distributions(grid, 3, index="spei")


def test_non_january_start_is_rejected() -> None:
    grid = _grid(masked_cell=False).isel(time=slice(1, None))
    with pytest.raises(CoordinateValidationError):
        suitability.compare_distributions(grid, 3, index="spi")


def test_unknown_candidate_still_raises_before_any_work() -> None:
    with pytest.raises(InvalidArgumentError):
        suitability.compare_distributions(_grid(), 3, index="spi", candidates=[Distribution.beta])
