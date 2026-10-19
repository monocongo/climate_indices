"""Tests for the beta xarray Dataset routes of the compound statistics (CMP-08, ADR-0022).

The routes add labels, metadata, and a Dask contract to the NumPy core, so the central claim is
equivalence: every result field equals the NumPy route's exactly (counts, flags, and NaN positions;
estimates come from the same arithmetic on the same counts). This is interface equivalence, not
external validation. The remaining tests pin what the adapter itself owns: nothing is evaluated,
rechunked, or broadcast when a graph is built; unsupported layouts fail before a graph exists; the
metadata is registry-derived and inherits nothing from a source; and the warning and logging policy
describes lazy scheduling honestly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import warnings
from collections.abc import Callable
from typing import Any
from unittest import mock

import dask
import numpy as np
import pytest
import xarray as xr

from climate_indices import compound
from climate_indices.cf_metadata_registry import CF_METADATA
from climate_indices.compound import _xarray as xarray_module
from climate_indices.exceptions import (
    BetaFeatureWarning,
    ClimateIndicesWarning,
    CoordinateValidationError,
    DimensionMismatchError,
    InputAlignmentWarning,
    InputTypeError,
    InvalidArgumentError,
)

YEARS, LAT, LON = 40, 3, 4
MONTHS = np.arange("1980-01", f"{1980 + YEARS}-01", dtype="datetime64[M]")
SPATIAL = ("lat", "lon")
DECLARATIONS: dict[str, Any] = {
    "cadence": {"drought": "monthly", "heat": "monthly"},
    "input_scale": {"drought": "probability", "heat": "indicator"},
    "thresholds": {"drought": 0.45, "heat": 0.5},
    "directions": {"drought": "below", "heat": "above"},
    "target_cadence": "monthly",
    "source_metadata": {"drought": {"calibration_year_initial": 1991, "calibration_year_final": 2020}},
}
# every result field other than these is an array the Dataset must reproduce
_NOT_ARRAYS = {"names", "groupby", "metadata", "statistic", "given", "target", "lag_window"}


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture(autouse=True)
def quiet_beta_and_support_warnings():
    """Let the routine route warnings pass; tests that assert warnings record them explicitly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ClimateIndicesWarning)
        yield


def _values(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return seeded ``(time, lat, lon)`` drought probabilities (with gaps) and a heat indicator."""
    rng = np.random.default_rng(seed)
    drought = rng.uniform(size=(len(MONTHS), LAT, LON))
    heat = rng.uniform(size=(len(MONTHS), LAT, LON)) > 0.4
    drought[5:9, 0, 0] = np.nan
    drought[100:103, 2, 3] = np.nan
    return drought, heat


def _coords(time: np.ndarray = MONTHS, time_dim: str = "time") -> dict[str, Any]:
    return {time_dim: time.astype("datetime64[ns]"), "lat": np.arange(LAT), "lon": np.arange(LON)}


def sources(*, lazy: bool = False, chunks: dict[str, int] | None = None, seed: int = 0) -> dict[str, xr.DataArray]:
    """Return the two source DataArrays, optionally Dask-backed with one whole time chunk."""
    drought, heat = _values(seed)
    arrays = {
        "drought": xr.DataArray(drought, dims=("time", *SPATIAL), coords=_coords()),
        "heat": xr.DataArray(heat, dims=("time", *SPATIAL), coords=_coords()),
    }
    if lazy:
        arrays = {name: array.chunk(chunks or {"time": -1, "lat": 2, "lon": 2}) for name, array in arrays.items()}
    return arrays


def numpy_aligned(seed: int = 0, **overrides: Any) -> compound.AlignedHazards:
    """Return the NumPy route's prepared hazards for the same data."""
    drought, heat = _values(seed)
    return compound.harmonize(
        {"drought": drought, "heat": heat},
        time={"drought": MONTHS, "heat": MONTHS},
        spatial_time_major=True,
        **{**DECLARATIONS, **overrides},
    )


def prepared(*, lazy: bool = False, chunks: dict[str, int] | None = None) -> xr.Dataset:
    """Return the xarray route's prepared Dataset."""
    return compound.harmonize(sources(lazy=lazy, chunks=chunks), **DECLARATIONS)


def assert_matches_numpy(dataset: xr.Dataset, result: Any) -> None:
    """Require every NumPy result field to equal the Dataset variable of the same name."""
    for field in dataclasses.fields(result):
        if field.name in _NOT_ARRAYS:
            continue
        variable = dataset[field.name]
        dims = [dim for dim in ("hazard", "month") if dim in variable.dims] + [d for d in SPATIAL if d in variable.dims]
        np.testing.assert_array_equal(variable.transpose(*dims).values, getattr(result, field.name), err_msg=field.name)


# --- harmonize -------------------------------------------------------------------------------------


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_harmonize_matches_numpy(lazy: bool) -> None:
    """Return the NumPy arrays exactly, with the documented dimension order and labels."""
    dataset = prepared(lazy=lazy)
    aligned = numpy_aligned()
    assert dataset["exceedance"].dims == ("hazard", "time", "lat", "lon")
    assert bool(dataset.chunks) is lazy
    assert dataset["hazard"].values.tolist() == ["drought", "heat"]
    np.testing.assert_array_equal(dataset["time"].values.astype("datetime64[D]"), aligned.time)
    np.testing.assert_array_equal(dataset["exceedance"].values, aligned.exceedance)
    np.testing.assert_array_equal(dataset["valid"].values, aligned.valid)
    assert dataset["exceedance"].dtype == np.bool_
    assert dataset["valid"].dtype == np.bool_
    assert json.loads(dataset.attrs["compound_manifest"]) == json.loads(json.dumps(aligned.metadata))


def test_harmonize_accepts_custom_and_transposed_dimensions() -> None:
    """Read dates from a caller-named time dimension in any dimension order."""
    renamed = {
        name: array.rename({"time": "t"}).transpose("lat", "t", "lon") for name, array in sources(lazy=True).items()
    }
    dataset = compound.harmonize(renamed, time_dim="t", **DECLARATIONS)
    assert dataset["exceedance"].dims == ("hazard", "time", "lat", "lon")
    np.testing.assert_array_equal(dataset["exceedance"].values, numpy_aligned().exceedance)


def test_nan_and_explicit_validity_masks_are_equivalent() -> None:
    """Treat a NaN and a validity-masked element identically, eager or lazy."""
    drought, _ = _values()
    base = sources()
    filled = base["drought"].copy(data=np.where(np.isnan(drought), 0.0, drought))
    mask = xr.DataArray(~np.isnan(drought), dims=base["drought"].dims, coords=base["drought"].coords)
    expected = prepared()[["exceedance", "valid"]]
    for lazy in (False, True):

        def place(array: xr.DataArray, lazy: bool = lazy) -> xr.DataArray:
            return array.chunk({"time": -1, "lat": 2, "lon": 2}) if lazy else array

        result = compound.harmonize(
            {"drought": place(filled), "heat": place(base["heat"])},
            validity={"drought": place(mask)},
            **DECLARATIONS,
        )
        xr.testing.assert_equal(result[["exceedance", "valid"]].compute(), expected)


def test_daily_sources_reduce_to_months_like_numpy() -> None:
    """Reduce daily indicators with any/all on a grid, matching the NumPy route."""
    days = np.arange("2000-01-01", "2002-01-01", dtype="datetime64[D]")
    rng = np.random.default_rng(3)
    drought = rng.uniform(size=(len(days), 2, 2))
    heat = rng.uniform(size=(len(days), 2, 2)) > 0.5
    drought[40, 0, 0] = np.nan
    declared: dict[str, Any] = {
        "cadence": {"drought": "daily", "heat": "daily"},
        "input_scale": {"drought": "probability", "heat": "indicator"},
        "thresholds": {"drought": 0.3, "heat": 0.5},
        "directions": {"drought": "below", "heat": "above"},
        "target_cadence": "monthly",
        "aggregation": {"drought": "any", "heat": "all"},
    }
    coords = {"time": days.astype("datetime64[ns]"), "lat": [0, 1], "lon": [0, 1]}
    arrays = {
        "drought": xr.DataArray(drought, dims=("time", "lat", "lon"), coords=coords),
        "heat": xr.DataArray(heat, dims=("time", "lat", "lon"), coords=coords),
    }
    expected = compound.harmonize(
        {"drought": drought, "heat": heat}, time={"drought": days, "heat": days}, spatial_time_major=True, **declared
    )
    for lazy in (False, True):
        inputs = {k: v.chunk({"time": -1, "lat": 1, "lon": 2}) if lazy else v for k, v in arrays.items()}
        dataset = compound.harmonize(inputs, **declared)
        np.testing.assert_array_equal(dataset["exceedance"].values, expected.exceedance)
        np.testing.assert_array_equal(dataset["valid"].values, expected.valid)


def test_inner_join_drops_boundaries_with_a_warning_before_any_computation() -> None:
    """Intersect different coverage like NumPy, warning at the call even when the input is lazy."""
    short = MONTHS[:36]
    shifted = MONTHS[2:38]
    rng = np.random.default_rng(5)
    left, right = rng.uniform(size=(36, 2)), rng.uniform(size=(36, 2)) > 0.5
    declared = {**DECLARATIONS, "join": "inner"}
    expected = compound.harmonize(
        {"drought": left, "heat": right},
        time={"drought": short, "heat": shifted},
        spatial_time_major=True,
        **declared,
    )
    for lazy in (False, True):
        arrays = {
            "drought": xr.DataArray(left, dims=("time", "site"), coords={"time": short, "site": [0, 1]}),
            "heat": xr.DataArray(right, dims=("time", "site"), coords={"time": shifted, "site": [0, 1]}),
        }
        if lazy:
            arrays = {k: v.chunk({"time": -1, "site": 1}) for k, v in arrays.items()}
        with pytest.warns(InputAlignmentWarning, match="dropped"):
            dataset = compound.harmonize(arrays, **declared)
        np.testing.assert_array_equal(dataset["exceedance"].values, expected.exceedance)
        assert dataset.sizes["time"] == expected.time.shape[0]


def test_point_series_returns_datasets() -> None:
    """Return a Dataset even for one location, with no spatial dimensions."""
    drought, heat = _values()
    series = {
        "drought": xr.DataArray(drought[:, 1, 1], dims="time", coords={"time": MONTHS}),
        "heat": xr.DataArray(heat[:, 1, 1], dims="time", coords={"time": MONTHS}),
    }
    dataset = compound.harmonize(series, **DECLARATIONS)
    assert dataset["exceedance"].dims == ("hazard", "time")
    joint = compound.joint_statistics(dataset, min_samples=1)
    expected = compound.joint_statistics(
        compound.harmonize(
            {"drought": drought[:, 1, 1], "heat": heat[:, 1, 1]},
            time={"drought": MONTHS, "heat": MONTHS},
            **DECLARATIONS,
        ),
        min_samples=1,
    )
    assert joint["n_valid"].dims == ("month",)
    assert joint["n_marginal"].dims == ("hazard", "month")
    for field in ("n_valid", "n_joint", "quality_flags"):
        np.testing.assert_array_equal(joint[field].values, getattr(expected, field))
    np.testing.assert_array_equal(joint["n_marginal"].values, expected.n_marginal)


# --- statistics ------------------------------------------------------------------------------------


def _routes(aligned: compound.AlignedHazards) -> list[tuple[str, Callable[[Any], Any], Any]]:
    """Return ``(id, xarray-or-numpy callable, NumPy result)`` for each statistics route and grouping."""
    cases: list[tuple[str, Callable[[Any], Any], Any]] = []
    for groupby in ("month", "pooled"):
        joint = {"groupby": groupby, "min_samples": 1}
        cond = {"given": "drought", "target": "heat", "lag_window": (0, 2), "groupby": groupby, "min_samples": 1}
        boot = {
            "statistic": "joint_probability",
            "groupby": groupby,
            "seed": 11,
            "n_resamples": 29,
            "block_length_years": 3,
        }
        cases += [
            (
                f"joint-{groupby}",
                lambda h, o=joint: compound.joint_statistics(h, **o),
                compound.joint_statistics(aligned, **joint),
            ),
            (
                f"conditional-{groupby}",
                lambda h, o=cond: compound.conditional_probability(h, **o),
                compound.conditional_probability(aligned, **cond),
            ),
            (
                f"bootstrap-{groupby}",
                lambda h, o=boot: compound.bootstrap_dependence(h, **o),
                compound.bootstrap_dependence(aligned, **boot),
            ),
        ]
    return cases


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_every_statistics_route_matches_numpy(lazy: bool) -> None:
    """Match each result field of the NumPy route, per month and pooled, eager and lazy."""
    aligned = numpy_aligned()
    dataset = prepared(lazy=lazy)
    for case, call, expected in _routes(aligned):
        result = call(dataset)
        assert isinstance(result, xr.Dataset), case
        assert bool(result.chunks) is lazy, case
        assert_matches_numpy(result.compute(scheduler="synchronous") if lazy else result, expected)
        assert json.loads(result.attrs["compound_manifest"]) == json.loads(json.dumps(expected.metadata)), case


def test_bootstrap_inference_is_actually_exercised() -> None:
    """Guard the equivalence test above: some cells reach finite intervals and p-values."""
    expected = compound.bootstrap_dependence(
        numpy_aligned(), statistic="joint_probability", seed=11, n_resamples=29, block_length_years=3
    )
    assert np.isfinite(expected.p_value).any()
    assert np.isfinite(expected.confidence_low).any()


def test_conditional_and_lmf_bootstrap_match_numpy_with_selected_hazards() -> None:
    """Label the bootstrap channel dimension with the selected hazards, not every source."""
    aligned = numpy_aligned()
    options = {
        "statistic": "conditional_probability",
        "given": "heat",
        "target": "drought",
        "lag_window": (0, 1),
        "seed": 4,
        "n_resamples": 19,
        "block_length_years": 2,
        "alternative": "less",
    }
    result = compound.bootstrap_dependence(prepared(lazy=True), **options)
    expected = compound.bootstrap_dependence(aligned, **options)
    assert result["hazard"].values.tolist() == ["heat", "drought"]
    assert_matches_numpy(result.compute(scheduler="synchronous"), expected)
    lmf = compound.bootstrap_dependence(
        prepared(), statistic="likelihood_multiplication_factor", seed=2, n_resamples=19, block_length_years=2
    )
    assert_matches_numpy(
        lmf,
        compound.bootstrap_dependence(
            aligned, statistic="likelihood_multiplication_factor", seed=2, n_resamples=19, block_length_years=2
        ),
    )


def test_bootstrap_is_invariant_to_spatial_tiling() -> None:
    """Give every cell the same result whatever the Dask tile boundaries."""
    options = {"statistic": "joint_probability", "seed": 7, "n_resamples": 29, "block_length_years": 3}
    results = [
        compound.bootstrap_dependence(prepared(lazy=True, chunks={"time": -1, **tile}), **options).compute(
            scheduler="synchronous"
        )
        for tile in ({"lat": 1, "lon": 1}, {"lat": 2, "lon": 3}, {"lat": -1, "lon": -1})
    ]
    expected = compound.bootstrap_dependence(numpy_aligned(), **options)
    for result in results:
        assert_matches_numpy(result, expected)


def test_prepared_dimension_order_does_not_matter() -> None:
    """Accept a prepared Dataset whose variables are transposed, as a reloaded file may be."""
    dataset = prepared()
    shuffled = dataset.transpose("lon", "lat", "time", "hazard")
    result = compound.joint_statistics(shuffled, min_samples=1)
    assert_matches_numpy(result, compound.joint_statistics(numpy_aligned(), min_samples=1))


# --- laziness and the Dask contract ------------------------------------------------------------------


def _forbid_compute(*_: Any, **__: Any) -> Any:
    raise AssertionError("a Dask graph was evaluated while it was being built")


def test_building_a_graph_never_evaluates_an_input() -> None:
    """Build every route lazily without a single task running, then compute on request."""
    inputs = sources(lazy=True)
    with dask.config.set(scheduler=_forbid_compute):
        dataset = compound.harmonize(inputs, **DECLARATIONS)
        joint = compound.joint_statistics(dataset, min_samples=1)
        cond = compound.conditional_probability(dataset, given="drought", target="heat", min_samples=1)
        boot = compound.bootstrap_dependence(dataset, statistic="joint_probability", seed=1, n_resamples=9)
    for result in (dataset, joint, cond, boot):
        assert result.chunks
    assert_matches_numpy(joint.compute(), compound.joint_statistics(numpy_aligned(), min_samples=1))


def test_value_errors_surface_when_a_lazy_graph_is_computed() -> None:
    """Validate values inside the task: eager input fails at the call, lazy input at compute."""
    bad = sources()
    bad["drought"] = bad["drought"].copy(data=np.full((len(MONTHS), LAT, LON), 1.5))
    with pytest.raises(InvalidArgumentError, match="not clipped"):
        compound.harmonize(bad, **DECLARATIONS)
    lazy = {name: array.chunk({"time": -1, "lat": 2, "lon": 2}) for name, array in bad.items()}
    dataset = compound.harmonize(lazy, **DECLARATIONS)
    with pytest.raises(InvalidArgumentError, match="not clipped"):
        dataset.compute(scheduler="synchronous")


@pytest.mark.parametrize(
    ("mutate", "error", "reason"),
    [
        (lambda s: {k: v.chunk({"time": 120}) for k, v in s.items()}, CoordinateValidationError, "multi_chunked"),
        (
            lambda s: {"drought": s["drought"].chunk({"lat": 2}), "heat": s["heat"].chunk({"lat": 1})},
            CoordinateValidationError,
            "incompatible_spatial_chunks",
        ),
        (lambda s: {"drought": s["drought"], "heat": s["heat"].compute()}, InvalidArgumentError, None),
        (lambda s: {k: v.drop_vars("lat") for k, v in s.items()}, CoordinateValidationError, "missing_coordinate"),
        (
            lambda s: {"drought": s["drought"], "heat": s["heat"].assign_coords(lat=[5, 6, 7])},
            CoordinateValidationError,
            "mismatched_coordinates",
        ),
        (
            lambda s: {"drought": s["drought"], "heat": s["heat"].isel(lon=slice(0, 3))},
            CoordinateValidationError,
            "mismatched_coordinates",
        ),
        (
            lambda s: {"drought": s["drought"], "heat": s["heat"].isel(lat=0, drop=True)},
            DimensionMismatchError,
            "mismatched_spatial_dimensions",
        ),
        (lambda s: {k: v.drop_vars("time") for k, v in s.items()}, CoordinateValidationError, "missing_coordinate"),
        (lambda s: {k: v.rename(lat="month") for k, v in s.items()}, DimensionMismatchError, "reserved"),
    ],
    ids=[
        "split-time-chunk",
        "mismatched-spatial-chunks",
        "mixed-backends",
        "missing-spatial-labels",
        "shifted-spatial-labels",
        "different-grid-size",
        "broadcast-a-point",
        "dimension-only-time",
        "reserved-spatial-name",
    ],
)
def test_harmonize_rejects_unsupported_layouts_before_building_a_graph(
    mutate: Callable[[dict[str, xr.DataArray]], dict[str, xr.DataArray]], error: type[Exception], reason: str | None
) -> None:
    """Report each incompatible grid, chunk, backend, or label problem with a structured error."""
    inputs = mutate(sources(lazy=True))
    with dask.config.set(scheduler=_forbid_compute), pytest.raises(error) as caught:
        compound.harmonize(inputs, **DECLARATIONS)
    if reason is not None:
        assert reason in str(getattr(caught.value, "reason", "")) or reason in str(caught.value)


def test_statistics_reject_split_hazard_and_time_chunks() -> None:
    """Require one chunk along every core dimension of a prepared Dataset."""
    dataset = prepared(lazy=True)
    for chunking in ({"hazard": 1}, {"time": 120}):
        split = dataset.chunk(chunking)
        with pytest.raises(CoordinateValidationError, match="split across"):
            compound.joint_statistics(split, min_samples=1)


def test_statistics_reject_variables_chunked_differently() -> None:
    """Never reconcile two variables of one Dataset whose spatial chunks disagree."""
    dataset = prepared(lazy=True)
    dataset["valid"] = dataset["valid"].chunk({"lat": 1})
    with pytest.raises(CoordinateValidationError, match="chunked differently"):
        compound.joint_statistics(dataset, min_samples=1)


def test_statistics_reject_unprepared_and_malformed_datasets() -> None:
    """Reject a Dataset that is not harmonize() output, with the NumPy route's own errors."""
    unprepared = xr.Dataset({"x": ("time", [1.0])})
    with pytest.raises(InvalidArgumentError, match="compound.harmonize"):
        compound.joint_statistics(unprepared)
    dataset = prepared()
    integer_valid = dataset.assign(valid=dataset["valid"].astype(int))
    with pytest.raises(InputTypeError, match="Boolean"):
        compound.joint_statistics(integer_valid)
    hourly = dataset.assign_attrs(cadence="hourly")
    with pytest.raises(InvalidArgumentError, match="cadence"):
        compound.joint_statistics(hourly)
    malformed = dataset.assign_attrs(compound_manifest="{not json")
    with pytest.raises(InvalidArgumentError, match="JSON"):
        compound.joint_statistics(malformed)
    one_hazard = dataset.isel(hazard=[0])
    with pytest.raises(InvalidArgumentError, match="two or three"):
        compound.joint_statistics(one_hazard)
    unlabeled = dataset.drop_vars("hazard")
    with pytest.raises(CoordinateValidationError, match="missing_coordinate_labels|labeled"):
        compound.joint_statistics(unlabeled)
    sub_daily = dataset.assign_coords(time=dataset["time"] + np.timedelta64(3, "h"))
    with pytest.raises(CoordinateValidationError, match="whole civil days"):
        compound.joint_statistics(sub_daily)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda d: compound.joint_statistics(d, groupby="seasonal"), "groupby"),
        (lambda d: compound.joint_statistics(d, min_samples=0), "positive integer"),
        (lambda d: compound.conditional_probability(d, given="drought", target="drought"), "different sources"),
        (lambda d: compound.conditional_probability(d, given="drought", target="heat", lag_window=(2, 1)), "ordered"),
        (lambda d: compound.conditional_probability(d, given="nope", target="heat"), "given must name"),
        (lambda d: compound.bootstrap_dependence(d, statistic="joint_probability"), "seed"),
        (lambda d: compound.bootstrap_dependence(d, statistic="joint_probability", seed=-1), "seed"),
        (
            lambda d: compound.bootstrap_dependence(d, statistic="joint_probability", seed=0, n_resamples=10**9),
            "n_resamples",
        ),
        (lambda d: compound.bootstrap_dependence(d, statistic="mean", seed=0), "statistic"),
        (
            lambda d: compound.bootstrap_dependence(d, statistic="joint_probability", seed=0, given="drought"),
            "apply only",
        ),
    ],
)
def test_statistics_validate_options_before_building_a_graph(call: Callable[[xr.Dataset], Any], match: str) -> None:
    """Raise the NumPy route's option errors at the call, even for a lazy Dataset."""
    dataset = prepared(lazy=True)
    with dask.config.set(scheduler=_forbid_compute), pytest.raises((InvalidArgumentError, TypeError), match=match):
        call(dataset)


def test_gapped_time_is_rejected_for_a_forward_window() -> None:
    """Reuse the NumPy complete-cadence rule for conditional windows on a Dataset with a missing step."""
    dataset = prepared()
    gapped = dataset.isel(time=[i for i in range(dataset.sizes["time"]) if i != 10])
    with pytest.raises(CoordinateValidationError, match="complete cadence"):
        compound.conditional_probability(gapped, given="drought", target="heat", min_samples=1)
    compound.joint_statistics(gapped, min_samples=1)


def test_daily_cadence_is_rejected_for_bootstrap() -> None:
    """Keep the monthly-only inference rule on the Dataset route."""
    days = np.arange("2000-01-01", "2000-03-01", dtype="datetime64[D]")
    rng = np.random.default_rng(1)
    arrays = {
        name: xr.DataArray(rng.uniform(size=len(days)) > 0.5, dims="time", coords={"time": days}) for name in ("a", "b")
    }
    declared: dict[str, Any] = {
        "cadence": dict.fromkeys(arrays, "daily"),
        "input_scale": dict.fromkeys(arrays, "indicator"),
        "thresholds": dict.fromkeys(arrays, 0.5),
        "directions": dict.fromkeys(arrays, "above"),
        "target_cadence": "daily",
    }
    dataset = compound.harmonize(arrays, **declared)
    with pytest.raises(InvalidArgumentError, match="monthly"):
        compound.bootstrap_dependence(dataset, statistic="joint_probability", seed=1)


# --- harmonize argument contract ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"time": {"drought": MONTHS, "heat": MONTHS}}, "NumPy inputs"),
        ({"time": None}, "NumPy inputs"),
        ({"spatial_time_major": True}, "NumPy inputs"),
        ({"time_dim": 3}, "time_dim"),
    ],
)
def test_harmonize_rejects_numpy_only_arguments_for_dataarrays(extra: dict[str, Any], match: str) -> None:
    """Take dates from the coordinate and the layout from named dimensions, never from NumPy options."""
    arrays = sources()
    with pytest.raises(InvalidArgumentError, match=match):
        compound.harmonize(arrays, **DECLARATIONS, **extra)


def test_harmonize_rejects_mixed_inputs_and_a_numpy_time_dim() -> None:
    """Never mix a DataArray with a NumPy array, and never accept time_dim for NumPy input."""
    drought, heat = _values()
    mixed = {"drought": drought, "heat": sources()["heat"]}
    with pytest.raises(InputTypeError, match="all DataArrays or all NumPy"):
        compound.harmonize(mixed, **DECLARATIONS)
    arrays = sources()
    with pytest.raises(InputTypeError, match="validity"):
        compound.harmonize(arrays, validity={"drought": np.ones(len(MONTHS), bool)}, **DECLARATIONS)
    with pytest.raises(InvalidArgumentError, match="time_dim"):
        compound.harmonize(
            {"drought": drought, "heat": heat},
            time={"drought": MONTHS, "heat": MONTHS},
            time_dim="t",
            spatial_time_major=True,
            **DECLARATIONS,
        )


def test_numpy_route_keeps_its_contract_for_a_missing_or_none_time() -> None:
    """Raise TypeError for an omitted time, as when it was a required keyword, and keep the None message."""
    drought, heat = _values()
    inputs = {"drought": drought, "heat": heat}
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'time'"):
        compound.harmonize(inputs, spatial_time_major=True, **DECLARATIONS)
    with pytest.raises(InvalidArgumentError, match="time must be a mapping keyed by source name"):
        compound.harmonize(inputs, time=None, spatial_time_major=True, **DECLARATIONS)


def test_harmonize_rejects_unsupported_calendars_and_conflicting_scales() -> None:
    """Refuse non-Gregorian calendars and a declared probability whose metadata says it is a z-score."""
    arrays = sources()
    arrays["drought"].coords["time"].attrs["calendar"] = "noleap"
    with pytest.raises(CoordinateValidationError, match="Unsupported calendar"):
        compound.harmonize(arrays, **DECLARATIONS)
    scored = sources()
    scored["drought"].attrs["output_scale"] = "normal"
    with pytest.raises(InvalidArgumentError, match="output_scale"):
        compound.harmonize(scored, **DECLARATIONS)
    scored["drought"].attrs["output_scale"] = "probability"
    compound.harmonize(scored, **DECLARATIONS)


def test_harmonize_rejects_cftime_calendars() -> None:
    """Reject cftime axes, which no datetime64 calendar conversion may paper over."""
    pytest.importorskip("cftime")
    arrays = sources()
    cftime_axis = xr.date_range("1980-01-01", periods=len(MONTHS), freq="MS", calendar="noleap", use_cftime=True)
    arrays = {name: array.assign_coords(time=cftime_axis) for name, array in arrays.items()}
    with pytest.raises(CoordinateValidationError, match="Unsupported calendar"):
        compound.harmonize(arrays, **DECLARATIONS)


def test_harmonize_reuses_the_numpy_declaration_errors() -> None:
    """Raise the NumPy route's structured errors at the call, from labels and dtypes alone."""
    arrays = sources()
    with pytest.raises(InvalidArgumentError, match="z-score"):
        compound.harmonize(arrays, **{**DECLARATIONS, "input_scale": {"drought": "normal", "heat": "indicator"}})
    wrong = sources()
    wrong["heat"] = wrong["heat"].astype(float)
    with pytest.raises(InputTypeError, match="Boolean"):
        compound.harmonize(wrong, **DECLARATIONS)
    short = sources()
    short["heat"] = short["heat"].isel(time=slice(0, 100))
    with pytest.raises(CoordinateValidationError, match="coverage"):
        compound.harmonize(short, **DECLARATIONS)


# --- metadata and serialization ----------------------------------------------------------------------


def test_results_carry_registry_metadata_and_inherit_nothing_from_a_source() -> None:
    """Apply the registry attributes, bounds, and flags; never a source's standard name or bounds."""
    arrays = sources()
    for array in arrays.values():
        array.attrs.update(standard_name="air_temperature", units="K", valid_min=-50.0, valid_max=60.0, long_name="T")
    dataset = compound.harmonize(arrays, **DECLARATIONS)
    joint = compound.joint_statistics(dataset, min_samples=1)
    cond = compound.conditional_probability(dataset, given="drought", target="heat", min_samples=1)
    boot = compound.bootstrap_dependence(dataset, statistic="likelihood_multiplication_factor", seed=1, n_resamples=9)
    for result in (dataset, joint, cond, boot):
        assert "standard_name" not in result.attrs
        assert result.attrs["climate_indices_version"]
        assert "history" in result.attrs
        for name, variable in result.variables.items():
            assert "standard_name" not in variable.attrs, name
            assert variable.attrs.get("units") != "K", name
    for name, key in (
        ("joint_probability", "compound_joint_probability"),
        ("marginal_frequency", "compound_marginal_probability"),
        ("independent_probability", "compound_independent_probability"),
    ):
        attrs = joint[name].attrs
        assert attrs["long_name"] == CF_METADATA[key]["long_name"]
        assert attrs["units"] == "1"
        assert (attrs["valid_min"], attrs["valid_max"]) == (0.0, 1.0)
        assert attrs["references"] == CF_METADATA[key]["references"]
    lmf = joint["likelihood_multiplication_factor"].attrs
    assert lmf["valid_min"] == 0.0
    assert "valid_max" not in lmf
    assert (
        cond["conditional_probability"].attrs["long_name"]
        == CF_METADATA["compound_conditional_probability"]["long_name"]
    )
    assert (
        cond["response_window_frequency"].attrs["long_name"]
        == (CF_METADATA["compound_response_window_probability"]["long_name"])
    )
    assert (
        boot["estimate"]
        .attrs["long_name"]
        .startswith(CF_METADATA["compound_likelihood_multiplication_factor"]["long_name"])
    )
    assert "valid_max" not in boot["estimate"].attrs
    assert "unadjusted" in boot["p_value"].attrs["description"]


def test_counts_and_flags_are_numeric_with_cf_flag_attributes() -> None:
    """Keep counts int64 and flags numeric bit fields described by flag_masks and flag_meanings."""
    joint = compound.joint_statistics(prepared(), min_samples=1)
    boot = compound.bootstrap_dependence(prepared(), statistic="joint_probability", seed=1, n_resamples=9)
    for result in (joint, boot):
        flags = result["quality_flags"]
        assert flags.dtype == np.int64
        assert list(flags.attrs["flag_masks"])
        assert flags.attrs["flag_meanings"].split()
    assert (
        joint["quality_flags"].attrs["flag_meanings"]
        == "insufficient_support undefined_denominator sparse_occurrences short_record"
    )
    boot_flags = boot["quality_flags"].attrs
    described = dict(zip(boot_flags["flag_meanings"].split(), boot_flags["flag_masks"].tolist(), strict=True))
    assert described == json.loads(boot.attrs["compound_manifest"])["quality_flags"]
    for name in ("n_valid", "n_joint", "n_marginal"):
        assert joint[name].dtype == np.int64
    assert boot["inference_start"].dtype.kind == "M"


def test_flag_attribute_arrays_are_not_shared_between_results() -> None:
    """Give each Dataset its own flag_masks, so mutating one result cannot alter another."""
    first = compound.joint_statistics(prepared(), min_samples=1)
    first["quality_flags"].attrs["flag_masks"][0] = 99
    second = compound.joint_statistics(prepared(), min_samples=1)
    assert second["quality_flags"].attrs["flag_masks"][0] == 1


def test_results_are_ordinary_writable_arrays() -> None:
    """Return writable arrays, unlike the NumPy dataclasses whose arrays are read-only."""
    dataset = prepared()
    dataset["exceedance"].values[0, 0, 0, 0] = True
    joint = compound.joint_statistics(dataset, min_samples=1)
    joint["n_valid"].values[0, 0, 0] = 0


@pytest.mark.parametrize("engine", ["h5netcdf", "zarr"])
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_numeric_datasets_round_trip_and_stay_usable(engine: str, lazy: bool, tmp_path: Any) -> None:
    """Preserve values, coordinates, flags, and JSON choices through a file, then reuse the Dataset."""
    if engine == "zarr":
        pytest.importorskip("zarr")
    dataset = prepared(lazy=lazy)
    stats = compound.bootstrap_dependence(
        dataset, statistic="joint_probability", seed=3, n_resamples=19, block_length_years=3
    )
    reloaded: dict[str, xr.Dataset] = {}
    for label, original in (("prepared", dataset), ("stats", stats)):
        path = tmp_path / f"{label}.{'zarr' if engine == 'zarr' else 'nc'}"
        if engine == "zarr":
            original.to_zarr(path, mode="w", consolidated=False)
            reloaded[label] = xr.open_zarr(path, consolidated=False).compute()
        else:
            original.to_netcdf(path, engine="h5netcdf")
            with xr.open_dataset(path, engine="h5netcdf") as opened:
                reloaded[label] = opened.load()
        expected = original.compute()
        assert reloaded[label].attrs["compound_manifest"] == expected.attrs["compound_manifest"]
        assert reloaded[label]["hazard"].values.tolist() == expected["hazard"].values.tolist()
        for name in expected.data_vars:
            np.testing.assert_array_equal(reloaded[label][name].values, expected[name].values, err_msg=name)
            # a file decodes dates at nanosecond resolution; the kind (and every value) must survive
            assert reloaded[label][name].dtype.kind == expected[name].dtype.kind, name
    flags = reloaded["stats"]["quality_flags"].attrs
    assert flags["flag_meanings"] == stats["quality_flags"].attrs["flag_meanings"]
    assert list(flags["flag_masks"]) == list(stats["quality_flags"].attrs["flag_masks"])
    again = compound.joint_statistics(reloaded["prepared"], min_samples=1)
    assert_matches_numpy(again, compound.joint_statistics(numpy_aligned(), min_samples=1))


# --- warnings and lifecycle logs ----------------------------------------------------------------------


def _record(call: Callable[[], Any]) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    return caught


def test_every_route_emits_one_beta_warning_at_the_caller() -> None:
    """Point the beta warning, and every replayed NumPy warning, at the code that called the public function."""
    dataset = prepared()
    calls: list[Callable[[], Any]] = [
        lambda: compound.harmonize(sources(), **DECLARATIONS),
        lambda: compound.joint_statistics(dataset, min_samples=1),
        lambda: compound.conditional_probability(dataset, given="drought", target="heat", min_samples=1),
        lambda: compound.bootstrap_dependence(dataset, statistic="joint_probability", seed=1, n_resamples=5),
    ]
    for call in calls:
        caught = _record(call)
        beta = [w for w in caught if issubclass(w.category, BetaFeatureWarning)]
        assert len(beta) == 1
        assert all(w.filename == __file__ for w in caught), [(w.category.__name__, w.filename) for w in caught]


def numpy_aligned_year() -> compound.AlignedHazards:
    """Return the NumPy prepared hazards for the first calendar year only."""
    drought, heat = _values()
    return compound.harmonize(
        {"drought": drought[:12], "heat": heat[:12]},
        time={"drought": MONTHS[:12], "heat": MONTHS[:12]},
        spatial_time_major=True,
        **DECLARATIONS,
    )


def test_eager_input_warns_like_numpy_and_dask_input_leaves_support_to_flags() -> None:
    """Replay the NumPy support warnings for in-memory input; under Dask the quality flags carry them."""
    one_year = slice(0, 12)
    eager = prepared().isel(time=one_year)
    lazy = prepared(lazy=True).isel(time=one_year).chunk({"time": -1})
    numpy_caught = _record(lambda: compound.joint_statistics(numpy_aligned_year(), min_samples=1))
    eager_caught = _record(lambda: compound.joint_statistics(eager, min_samples=1))
    numpy_messages = {str(w.message) for w in numpy_caught}
    assert numpy_messages
    assert {str(w.message) for w in eager_caught} - {xarray_module._BETA_MESSAGE} == numpy_messages

    def lazy_run() -> Any:
        return compound.joint_statistics(lazy, min_samples=1).compute(scheduler="synchronous")

    caught = _record(lazy_run)
    assert [w for w in caught if not issubclass(w.category, BetaFeatureWarning)] == []
    flags = lazy_run()["quality_flags"]
    assert (flags != 0).any(), "the condition the warning would have named must be visible in the flags"


@pytest.mark.parametrize("statistic", ["joint_statistics", "conditional_probability"])
@pytest.mark.parametrize("groupby", ["pooled", "month"])
@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_support_flags_report_sparse_occurrences_and_short_records(statistic: str, groupby: str, lazy: bool) -> None:
    """Keep both suppressed warnings visible, independently of tile boundaries and min_samples."""
    occurrence = np.ones((2, len(MONTHS), 3), dtype=bool)
    occurrence[:, :, 0] = False
    occurrence[:, : 5 if groupby == "pooled" else 60, 0] = True
    valid = np.ones_like(occurrence)
    valid[:, :-60, 1] = False
    dataset = xr.Dataset(
        {"exceedance": (("hazard", "time", "cell"), occurrence), "valid": (("hazard", "time", "cell"), valid)},
        coords={"hazard": ["drought", "heat"], "time": MONTHS.astype("datetime64[ns]"), "cell": [0, 1, 2]},
        attrs={"cadence": "monthly"},
    )
    if lazy:
        dataset = dataset.chunk({"hazard": -1, "time": -1, "cell": 1})
    options = {"given": "drought", "target": "heat"} if statistic == "conditional_probability" else {}
    result = getattr(compound, statistic)(dataset, groupby=groupby, min_samples=1, **options)
    manifest = json.loads(result.attrs["compound_manifest"])
    meanings = dict(
        zip(
            result["quality_flags"].attrs["flag_meanings"].split(),
            result["quality_flags"].attrs["flag_masks"],
            strict=True,
        )
    )
    assert meanings == manifest["quality_flags"]
    assert meanings["sparse_occurrences"] == 4
    assert meanings["short_record"] == 8
    flags = result["quality_flags"].compute(scheduler="synchronous")
    assert (flags.sel(cell=0) == 4).all()
    assert (flags.sel(cell=1) == (8 if groupby == "pooled" else 12)).all()
    assert (flags.sel(cell=2) == 0).all()


def test_missing_calibration_period_warns_at_the_call_for_lazy_input() -> None:
    """Emit the harmonization warnings from labels alone, so a lazy call reports them immediately."""
    declared = {**DECLARATIONS, "source_metadata": None}
    arrays = sources(lazy=True)
    with pytest.warns(ClimateIndicesWarning, match="Calibration Period metadata is missing"):
        compound.harmonize(arrays, **declared)


def test_lifecycle_events_describe_scheduling_honestly() -> None:
    """Say a lazy route completed graph construction, and an eager one completed its computation."""
    for lazy, expected in ((False, "eager"), (True, "graph_construction")):
        logger = mock.MagicMock()
        logger.bind.return_value = logger
        with mock.patch.object(xarray_module, "_logger", logger):
            dataset = compound.harmonize(sources(lazy=lazy), **DECLARATIONS)
            compound.joint_statistics(dataset, min_samples=1)
        bound = [call.kwargs for call in logger.bind.call_args_list]
        assert [b["index_type"] for b in bound] == ["compound_harmonize", "compound_joint_statistics"]
        assert all(b["route"] == "xarray" and "names" not in b for b in bound)
        events = [call.args[0] for call in logger.info.call_args_list]
        assert events == ["calculation_started", "calculation_completed"] * 2
        completed = [call.kwargs for call in logger.info.call_args_list if call.args[0] == "calculation_completed"]
        assert [c["evaluation"] for c in completed] == [expected, expected]
        assert completed[0]["output_shape"] == (2, len(MONTHS), LAT, LON)


def test_a_rejected_call_logs_failure_and_propagates() -> None:
    """Emit calculation_failed for a call rejected before any graph exists."""
    logger = mock.MagicMock()
    logger.bind.return_value = logger
    dataset = prepared()
    with (
        mock.patch.object(xarray_module, "_logger", logger),
        pytest.raises(InvalidArgumentError, match="groupby"),
    ):
        compound.joint_statistics(dataset, groupby="seasonal")
    assert [call.args[0] for call in logger.info.call_args_list] == ["calculation_started"]
    failed = [call for call in logger.error.call_args_list if call.args[0] == "calculation_failed"]
    assert len(failed) == 1
    assert failed[0].kwargs["error_type"] == "InvalidArgumentError"


def test_inputs_are_never_mutated() -> None:
    """Leave the caller's DataArrays, their attrs, and their coordinates untouched."""
    arrays = sources()
    before = {
        name: (array.values.copy(), dict(array.attrs), array["time"].values.copy()) for name, array in arrays.items()
    }
    compound.joint_statistics(compound.harmonize(arrays, **DECLARATIONS), min_samples=1)
    for name, array in arrays.items():
        np.testing.assert_array_equal(array.values, before[name][0])
        assert dict(array.attrs) == before[name][1]
        np.testing.assert_array_equal(array["time"].values, before[name][2])


# --- non-index coordinates ---------------------------------------------------------------------------


def _with_aux_coords(arrays: dict[str, xr.DataArray], *, lazy_aux: bool = False) -> dict[str, xr.DataArray]:
    """Give each source a 2-D auxiliary coordinate, a scalar coordinate, and CF attributes on lat and lon."""
    grid = np.arange(LAT * LON, dtype=float).reshape(LAT, LON)
    out = {}
    for name, array in arrays.items():
        aux = xr.DataArray(grid, dims=SPATIAL)
        if lazy_aux:
            aux = aux.chunk({"lat": 1, "lon": 1})  # a distinct Dask graph per source, as when each is opened separately
        array = array.assign_coords(lat2d=aux, level=500 if name == "drought" else 850)
        array["lat"].attrs.update(standard_name="latitude", units="degrees_north", axis="Y")
        array["lon"].attrs.update(standard_name="longitude", units="degrees_east", axis="X")
        out[name] = array
    return out


def test_a_lazy_auxiliary_coordinate_is_never_computed_while_building_a_graph() -> None:
    """Drop non-index coordinates before apply_ufunc merges, and so compares, them across inputs."""
    inputs = _with_aux_coords(sources(lazy=True), lazy_aux=True)
    with dask.config.set(scheduler=_forbid_compute):
        dataset = compound.harmonize(inputs, **DECLARATIONS)
        compound.joint_statistics(dataset, min_samples=1)
    assert "lat2d" not in dataset.coords
    assert "level" not in dataset.coords


@pytest.mark.parametrize("reserved", ["hazard", "source", "month", "time_of_day"])
def test_scalar_coordinates_never_break_the_harmonize_to_statistics_chain(reserved: str) -> None:
    """Accept a scalar coordinate named like a result dimension; none is carried onto any result."""
    inputs = {name: array.assign_coords({reserved: "x"}) for name, array in sources().items()}
    dataset = compound.harmonize(inputs, **DECLARATIONS)
    assert reserved not in dataset.coords or reserved == "hazard"
    for result in (
        compound.joint_statistics(dataset, min_samples=1),
        compound.conditional_probability(dataset, given="drought", target="heat", min_samples=1),
        compound.bootstrap_dependence(dataset, statistic="joint_probability", seed=1, n_resamples=5),
    ):
        assert reserved not in result.coords or reserved in ("hazard", "month")


def test_prepared_dataset_with_an_auxiliary_coordinate_chunked_differently_still_runs() -> None:
    """Ignore a prepared Dataset's auxiliary coordinates, whatever their chunking, instead of failing late."""
    dataset = prepared(lazy=True)
    dataset = dataset.assign_coords(
        lat2d=xr.DataArray(np.zeros((LAT, LON)), dims=SPATIAL).chunk({"lat": 1, "lon": 1}), level=500
    )
    result = compound.joint_statistics(dataset, min_samples=1)
    assert_matches_numpy(
        result.compute(scheduler="synchronous"), compound.joint_statistics(numpy_aligned(), min_samples=1)
    )
    assert "lat2d" not in result.coords


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "dask"])
def test_index_coordinate_attributes_survive_every_route(lazy: bool) -> None:
    """Keep standard_name, units, and axis on lat and lon so the result grid stays self-describing."""
    inputs = _with_aux_coords(sources(lazy=lazy))
    dataset = compound.harmonize(inputs, **DECLARATIONS)
    results = [
        dataset,
        compound.joint_statistics(dataset, min_samples=1),
        compound.conditional_probability(dataset, given="drought", target="heat", min_samples=1),
        compound.bootstrap_dependence(dataset, statistic="joint_probability", seed=1, n_resamples=5),
    ]
    for result in results:
        assert result["lat"].attrs == {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
        assert result["lon"].attrs == {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}
        for name, variable in result.data_vars.items():
            assert "axis" not in variable.attrs, name
            assert variable.attrs.get("standard_name") is None, name


# --- the warning sink -------------------------------------------------------------------------------


def _all_routes(dataset: xr.Dataset) -> list[Callable[[], Any]]:
    return [
        lambda: compound.harmonize(sources(), **DECLARATIONS),
        lambda: compound.harmonize(sources(lazy=True), **DECLARATIONS),
        lambda: compound.joint_statistics(dataset, min_samples=1),
        lambda: compound.conditional_probability(dataset, given="drought", target="heat", min_samples=1),
        lambda: compound.bootstrap_dependence(dataset, statistic="joint_probability", seed=1, n_resamples=5),
    ]


def test_no_route_captures_warnings_with_catch_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never enter the process-global, thread-unsafe catch_warnings from compound code."""

    class Guard(warnings.catch_warnings):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            caller = sys._getframe(1).f_globals.get("__name__", "")
            assert not caller.startswith("climate_indices.compound"), f"{caller} captured warnings"
            super().__init__(*args, **kwargs)

    dataset = prepared()
    monkeypatch.setattr(warnings, "catch_warnings", Guard)
    for call in _all_routes(dataset):
        call()


def test_default_filters_show_each_route_warning_once_per_location() -> None:
    """Keep the once-per-location rule, which a catch_warnings block would reset on every call."""
    # Isolate route handling: pandas 2.x index construction resets the warning registry.
    dataset = xr.Dataset({"n_valid": 1})
    support_message = "Insufficient compound support."

    def build(queue: list[Warning]) -> xr.Dataset:
        queue.append(ClimateIndicesWarning(support_message))
        return dataset

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(3):
            xarray_module._route("compound_joint_statistics", {}, build, "n_valid")
    messages = [str(w.message) for w in caught]
    assert messages.count(xarray_module._BETA_MESSAGE) == 1
    assert messages.count(support_message) == 1


def test_a_warning_raised_as_an_error_still_logs_a_terminal_event() -> None:
    """Log calculation_failed when a beta or support warning is turned into an exception."""
    dataset = prepared().isel(time=slice(0, 12))
    for ignore_beta, error_type in ((False, "BetaFeatureWarning"), (True, "ClimateIndicesWarning")):
        logger = mock.MagicMock()
        logger.bind.return_value = logger
        with warnings.catch_warnings(), mock.patch.object(xarray_module, "_logger", logger):
            warnings.simplefilter("error", ClimateIndicesWarning)
            if ignore_beta:
                warnings.simplefilter("ignore", BetaFeatureWarning)
            with pytest.raises(ClimateIndicesWarning):
                compound.joint_statistics(dataset, min_samples=1)
        assert [call.args[0] for call in logger.info.call_args_list] == ["calculation_started"]
        failed = [call for call in logger.error.call_args_list if call.args[0] == "calculation_failed"]
        assert len(failed) == 1
        assert failed[0].kwargs["error_type"] == error_type


def test_a_lazy_result_of_zero_dimensional_variables_is_logged_as_graph_construction() -> None:
    """Read laziness from the result's own variable, since Dataset.chunks ignores 0-d variables."""
    logger = mock.MagicMock()
    logger.bind.return_value = logger
    point = {name: array.isel(lat=1, lon=1, drop=True).chunk({"time": -1}) for name, array in sources().items()}
    with mock.patch.object(xarray_module, "_logger", logger):
        result = compound.conditional_probability(
            compound.harmonize(point, **DECLARATIONS), given="drought", target="heat", groupby="pooled", min_samples=1
        )
    assert result.chunks == {}
    assert result["n_anchors"].chunks is not None
    completed = [call.kwargs for call in logger.info.call_args_list if call.args[0] == "calculation_completed"]
    assert [c["evaluation"] for c in completed] == ["graph_construction", "graph_construction"]


def test_numpy_route_warnings_point_at_the_caller() -> None:
    """Keep the NumPy functions' warning attribution on the caller's line."""
    aligned = numpy_aligned_year()
    declared = {**DECLARATIONS, "source_metadata": None}
    drought, heat = _values()
    calls: list[Callable[[], Any]] = [
        lambda: compound.harmonize(
            {"drought": drought, "heat": heat},
            time={"drought": MONTHS, "heat": MONTHS},
            spatial_time_major=True,
            **declared,
        ),
        lambda: compound.joint_statistics(aligned, min_samples=1),
        lambda: compound.conditional_probability(aligned, given="drought", target="heat", min_samples=1),
        lambda: compound.bootstrap_dependence(
            numpy_aligned(), statistic="joint_probability", seed=1, n_resamples=5, block_length_years=40
        ),
    ]
    for call in calls:
        caught = _record(call)
        assert caught, "each call is expected to warn"
        assert all(w.filename == __file__ for w in caught), [(w.category.__name__, w.filename) for w in caught]


# --- truthful lazy dtypes, descriptions, and time errors -------------------------------------------------


def test_bootstrap_datetime_dtype_is_the_same_eager_lazy_and_computed() -> None:
    """Declare datetime64[s] up front, so a lazy variable's dtype matches what computing it returns."""
    options = {"statistic": "joint_probability", "seed": 1, "n_resamples": 9, "block_length_years": 3}
    eager = compound.bootstrap_dependence(prepared(), **options)
    lazy = compound.bootstrap_dependence(prepared(lazy=True), **options)
    for name in ("inference_start", "inference_end"):
        computed = lazy[name].compute()
        assert eager[name].dtype == lazy[name].dtype == computed.dtype == np.dtype("datetime64[s]"), name
        np.testing.assert_array_equal(computed.values, eager[name].values)


def test_confidence_bounds_and_estimates_are_described_by_their_role() -> None:
    """Describe a bound as a bound and each estimate by its support, not as the bare point statistic."""
    result = compound.bootstrap_dependence(
        prepared(), statistic="joint_probability", seed=1, n_resamples=9, block_length_years=3
    )
    registry = CF_METADATA["compound_joint_probability"]["description"]
    leads = {
        "estimate": "inference segment",
        "full_record_estimate": "every eligible",
        "confidence_low": "Lower bound",
        "confidence_high": "Upper bound",
    }
    descriptions = {name: result[name].attrs["description"] for name in leads}
    for name, phrase in leads.items():
        assert phrase in descriptions[name], name
        assert registry in descriptions[name], name
    assert len(set(descriptions.values())) == len(descriptions)


def test_an_undecoded_time_coordinate_says_it_needs_decoded_labels() -> None:
    """Name the real problem, undecoded numeric time, rather than an unsupported calendar."""
    dataset = prepared()
    undecoded = dataset.assign_coords(time=np.arange(dataset.sizes["time"]))
    with pytest.raises(CoordinateValidationError, match="decoded datetime64"):
        compound.joint_statistics(undecoded, min_samples=1)
    numeric = {name: array.assign_coords(time=np.arange(len(MONTHS))) for name, array in sources().items()}
    with pytest.raises(CoordinateValidationError, match="decoded datetime64"):
        compound.harmonize(numeric, **DECLARATIONS)


# --- documented guarantees that each route, not just one, must keep ---------------------------------------


@pytest.mark.parametrize(
    "tile", [{"lat": 1, "lon": 1}, {"lat": 2, "lon": 3}, {"lat": -1, "lon": -1}], ids=["1x1", "2x3", "whole-grid"]
)
def test_count_statistics_match_numpy_across_spatial_tilings(tile: dict[str, int]) -> None:
    """Give every cell the NumPy result whatever the Dask tile boundaries, for the joint and conditional routes."""
    aligned = numpy_aligned()
    dataset = prepared(lazy=True, chunks={"time": -1, **tile})
    joint = compound.joint_statistics(dataset, min_samples=1).compute(scheduler="synchronous")
    assert_matches_numpy(joint, compound.joint_statistics(aligned, min_samples=1))
    options = {"given": "drought", "target": "heat", "lag_window": (0, 2), "min_samples": 1}
    conditional = compound.conditional_probability(dataset, **options).compute(scheduler="synchronous")
    assert_matches_numpy(conditional, compound.conditional_probability(aligned, **options))


def _messages(caught: list[warnings.WarningMessage]) -> set[str]:
    return {str(w.message) for w in caught if not issubclass(w.category, BetaFeatureWarning)}


def test_eager_conditional_and_bootstrap_warn_exactly_like_numpy() -> None:
    """Replay the conditional and bootstrap support warnings, and the same ones, for in-memory input."""
    year = prepared().isel(time=slice(0, 12))
    conditional = {"given": "drought", "target": "heat", "min_samples": 1}
    expected = _messages(_record(lambda: compound.conditional_probability(numpy_aligned_year(), **conditional)))
    assert expected
    assert _messages(_record(lambda: compound.conditional_probability(year, **conditional))) == expected

    boot = {"statistic": "joint_probability", "seed": 1, "n_resamples": 5, "block_length_years": 40}
    expected = _messages(_record(lambda: compound.bootstrap_dependence(numpy_aligned(), **boot)))
    assert any("limited support" in message for message in expected)
    assert _messages(_record(lambda: compound.bootstrap_dependence(prepared(), **boot))) == expected
    lazy = prepared(lazy=True)
    assert (
        _messages(_record(lambda: compound.bootstrap_dependence(lazy, **boot).compute(scheduler="synchronous")))
        == set()
    )


def test_every_route_returns_a_dataset_for_a_single_location() -> None:
    """Return Datasets with no spatial dimension from all four routes, equal to the NumPy point results."""
    drought, heat = _values()
    series = {
        "drought": xr.DataArray(drought[:, 1, 1], dims="time", coords={"time": MONTHS}),
        "heat": xr.DataArray(heat[:, 1, 1], dims="time", coords={"time": MONTHS}),
    }
    point = compound.harmonize(series, **DECLARATIONS)
    aligned = compound.harmonize(
        {"drought": drought[:, 1, 1], "heat": heat[:, 1, 1]}, time={"drought": MONTHS, "heat": MONTHS}, **DECLARATIONS
    )
    conditional = {"given": "drought", "target": "heat", "lag_window": (0, 1), "min_samples": 1}
    boot = {"statistic": "joint_probability", "seed": 1, "n_resamples": 9, "block_length_years": 3}
    for result, expected in (
        (
            compound.conditional_probability(point, **conditional),
            compound.conditional_probability(aligned, **conditional),
        ),
        (compound.bootstrap_dependence(point, **boot), compound.bootstrap_dependence(aligned, **boot)),
    ):
        assert isinstance(result, xr.Dataset)
        assert not set(result.dims) - {"month", "hazard"}
        assert_matches_numpy(result, expected)
