"""Interface equivalence and numeric serialization for eager compound catalogs."""

from __future__ import annotations

import dataclasses
import json
import warnings
from typing import Any, Literal

import dask
import dask.config
import numpy as np
import pytest
import xarray as xr

from climate_indices import compound
from climate_indices.compound._catalog import _MAX_EVENTS
from climate_indices.exceptions import (
    BetaFeatureWarning,
    ClimateIndicesWarning,
    CoordinateValidationError,
    DimensionMismatchError,
    InputTypeError,
    InvalidArgumentError,
)
from climate_indices.runs import identify_runs

TIME = np.arange("2000-01", "2001-01", dtype="datetime64[M]")
EVENT_FIELDS = [f.name for f in dataclasses.fields(compound.CompoundEventSet) if f.name not in ("mode", "metadata")]


@pytest.fixture(autouse=True)
def quiet_warnings():
    """Suppress routine beta and source-metadata warnings, except when explicitly recorded."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ClimateIndicesWarning)
        yield


def prepared(*, cadence: Literal["daily", "monthly"] = "monthly", grid: bool = True) -> xr.Dataset:
    """Prepare three hazards; the third source never has valid data and must not affect a pair."""
    time = TIME if cadence == "monthly" else np.arange("2000-02-24", "2000-03-07", dtype="datetime64[D]")
    first = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1], dtype=bool)
    second = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], dtype=bool)
    dims: tuple[str, ...] = ("time",)
    coords: dict[str, Any] = {"time": time.astype("datetime64[s]")}
    if grid:
        first = np.broadcast_to(first[:, None, None], (12, 2, 2)).copy()
        second = np.broadcast_to(second[:, None, None], (12, 2, 2)).copy()
        first[:, 1, 1] = False
        dims = ("time", "lat", "station")
        coords.update(lat=[40.0, 50.0], station=["west", "east"])
    inputs = {name: xr.DataArray(value, dims=dims, coords=coords) for name, value in (("dry", first), ("hot", second))}
    inputs["unused"] = xr.zeros_like(inputs["dry"])
    masks = {"hot": xr.ones_like(inputs["hot"], dtype=bool), "unused": xr.zeros_like(inputs["dry"], dtype=bool)}
    masks["hot"][4] = False
    result = compound.harmonize(
        inputs,
        cadence=dict.fromkeys(inputs, cadence),
        input_scale=dict.fromkeys(inputs, "indicator"),
        thresholds=dict.fromkeys(inputs, 0.5),
        directions=dict.fromkeys(inputs, "above"),
        target_cadence=cadence,
        validity=masks,
        source_metadata={"hot": {"physical_threshold": 35.0, "comparator": ">="}},
    )
    if grid:
        result["lat"].attrs = {"units": "degrees_north", "long_name": "Latitude"}
    return result


def catalog(hazards: xr.Dataset, **options: Any) -> xr.Dataset:
    return compound.compound_events_xarray(hazards, first="dry", second="hot", **{"mode": "concurrent", **options})


@pytest.mark.parametrize("cadence", ["daily", "monthly"])
@pytest.mark.parametrize("mode,window", [("concurrent", None), ("sequential", None), ("sequential", (1, 3))])
@pytest.mark.parametrize("duration", [1, 2])
def test_grid_catalog_matches_each_series(
    cadence: Literal["daily", "monthly"], mode: Literal["concurrent", "sequential"], window: Any, duration: int
) -> None:
    """Every row, timestamp, count, and censor flag equals the existing NumPy route exactly."""
    hazards = prepared(cadence=cadence).transpose("station", "time", "lat", "hazard")
    before = hazards.copy(deep=True)
    result = catalog(hazards, mode=mode, lag_window=window, min_duration=duration)
    assert result.sizes["cell"] == 4
    np.testing.assert_array_equal(result["station"], ["west", "west", "east", "east"])
    np.testing.assert_array_equal(result["lat"], [40, 50, 40, 50])
    assert result["lat"].attrs == hazards["lat"].attrs
    for cell, (station, lat) in enumerate(np.ndindex(2, 2)):
        point = hazards.isel(station=station, lat=lat)
        shared = point["valid"].sel(hazard=["dry", "hot"]).all("hazard").values
        runs = [
            identify_runs(
                np.where(shared, point["exceedance"].sel(hazard=name).values, np.nan),
                threshold=0.5,
                direction="above",
                min_duration=duration,
            )
            for name in ("dry", "hot")
        ]
        expected = compound.compound_events(
            *runs, time=point.time.values, valid=shared, cadence=cadence, mode=mode, lag_window=window
        )
        rows = result.isel(event=np.flatnonzero(result["cell_index"].values == cell))
        for field in EVENT_FIELDS:
            np.testing.assert_array_equal(rows[field].values, getattr(expected, field), err_msg=field)
        for bound in ("start", "end"):
            np.testing.assert_array_equal(rows[f"{bound}_time"], point.time.values[getattr(expected, f"{bound}_index")])
        assert result["n_valid"][cell] == shared.sum()
        assert result["n_first_runs"][cell] == len(runs[0])
        assert result["n_second_runs"][cell] == len(runs[1])
        assert result["n_pairs"][cell] == len(expected)
        assert result["n_unknown_bridges"][cell] == expected.metadata["n_unknown_bridges"]
        single = catalog(point, mode=mode, lag_window=window, min_duration=duration)
        assert single.sizes["cell"] == 1
        for field in EVENT_FIELDS:
            np.testing.assert_array_equal(single[field], rows[field])
    xr.testing.assert_identical(hazards, before)


def test_shared_mask_splits_runs_and_censors_boundaries() -> None:
    """An unknown second-source step splits a first-source run; neither run can cross it."""
    hazards = prepared(grid=False)
    result = catalog(hazards)
    np.testing.assert_array_equal(result.start_index, [1, 7, 9, 11])
    np.testing.assert_array_equal(result.end_index, [1, 7, 9, 11])
    # min_duration applies before matching; only the first first-source run survives
    filtered = catalog(hazards, min_duration=2)
    np.testing.assert_array_equal(filtered.first_run_id, [0])
    np.testing.assert_array_equal(filtered.start_index, [1])
    manifest = json.loads(result.attrs["compound_manifest"])
    assert manifest["observation_support"].startswith("Both selected hazards")
    assert manifest["sources"]["first"] == {
        "name": "dry",
        "threshold": 0.5,
        "direction": "above",
        "min_duration": 1,
        "max_gap": 0,
    }
    assert manifest["prepared_hazards"] == json.loads(hazards.attrs["compound_manifest"])
    assert manifest["schema_version"] == 1
    assert "n_valid" not in manifest["grid"]
    assert manifest["censoring_support"].startswith("Censor flags use the shared support")
    assert str(_MAX_EVENTS) in manifest["row_limit"]


def test_censor_flags_follow_the_shared_support() -> None:
    """A boundary next to a step unknown in the other source is censored, and flagged per source."""
    hazards = prepared(grid=False)
    # index 4 is observed by dry and unknown by hot, so the shared mask invalidates it and the
    # dry run starting at index 5 must be flagged as left-censored; a per-source mask would not
    assert hazards["valid"].sel(hazard="dry").values[4]
    assert not hazards["valid"].sel(hazard="hot").values[4]
    result = catalog(hazards, mode="sequential")
    censored = np.flatnonzero(result["first_run_id"].values == 2)
    assert censored.size == 3
    np.testing.assert_array_equal(result["start_index"].values[censored], 5)
    np.testing.assert_array_equal(result["first_left_censored"].values[censored], 1)
    later = np.flatnonzero(result["first_run_id"].values == 3)
    assert later.size == 2
    np.testing.assert_array_equal(result["first_left_censored"].values[later], 0)


def test_selecting_a_non_default_pair_swaps_the_source_roles() -> None:
    """Reversing the pair swaps the run-ID columns and keeps every interval unchanged."""
    hazards = prepared(grid=False)
    forward = catalog(hazards)
    reverse = compound.compound_events_xarray(hazards, first="hot", second="dry", mode="concurrent")
    np.testing.assert_array_equal(reverse["first_run_id"], forward["second_run_id"])
    np.testing.assert_array_equal(reverse["second_run_id"], forward["first_run_id"])
    np.testing.assert_array_equal(reverse["start_index"], forward["start_index"])
    np.testing.assert_array_equal(reverse["end_index"], forward["end_index"])
    assert reverse["n_first_runs"].values.tolist() != forward["n_first_runs"].values.tolist()


def test_extra_sources_beyond_the_statistics_limit_are_allowed() -> None:
    """Select two of four prepared sources: the statistics' three-source cap does not apply."""
    base = prepared(grid=False)
    # the spare source exceeds at index 3 and reports index 10 unobserved, so including it in the
    # shared support would change both the mask and the run sequence
    spare = base[["exceedance", "valid"]].isel(hazard=[2]).copy(deep=True)
    spare["valid"].values[:] = True
    spare["valid"].values[0, 10] = False
    spare["exceedance"].values[:] = False
    spare["exceedance"].values[0, 3] = True
    spare = spare.assign_coords(hazard=["spare"])
    hazards = xr.Dataset(
        {name: xr.concat([base[name], spare[name]], dim="hazard") for name in ("exceedance", "valid")},
        attrs=base.attrs,
    )
    assert hazards.sizes["hazard"] == 4
    result = catalog(hazards)
    expected = catalog(base)
    assert expected.sizes["event"] > 0
    for field in EVENT_FIELDS:
        np.testing.assert_array_equal(result[field], expected[field], err_msg=field)
    # the unselected source neither adds runs nor changes the shared support
    np.testing.assert_array_equal(result["n_pairs"], expected["n_pairs"])
    np.testing.assert_array_equal(result["n_valid"], expected["n_valid"])
    np.testing.assert_array_equal(result["n_first_runs"], expected["n_first_runs"])
    np.testing.assert_array_equal(result["n_second_runs"], expected["n_second_runs"])


@pytest.mark.parametrize("empty", ["no_pairs", "all_invalid", "zero_cells", "long_duration"])
def test_empty_catalog_retains_dtypes_and_cells(empty: str) -> None:
    hazards = prepared()
    options: dict[str, Any] = {}
    if empty == "no_pairs":
        hazards["exceedance"][:] = False
    elif empty == "all_invalid":
        hazards["valid"][:] = False
    elif empty == "zero_cells":
        hazards = hazards.isel(lat=slice(0, 0))
    else:
        options["min_duration"] = 20
    result = catalog(hazards, **options)
    assert result.sizes == {"event": 0, "cell": 0 if empty == "zero_cells" else 4}
    for name in ("first_run_id", "second_run_id", "start_index", "end_index", "overlap_duration", "cell_index"):
        assert result[name].dtype == np.int64
    assert result.lag.dtype == np.float64
    assert result.start_time.dtype.kind == "M"
    for name in EVENT_FIELDS:
        if name.endswith("censored"):
            assert result[name].dtype == np.int8


def test_cell_coordinates_drop_dangling_boundary_attributes() -> None:
    """A bounds, axis, or cf_role attribute would reference a variable the catalog never carries."""
    hazards = prepared()
    hazards["lat"].attrs.update(bounds="lat_bnds", axis="Y", cf_role="timeseries_id", units="degrees_north")
    result = catalog(hazards)
    assert result["lat"].attrs == {"units": "degrees_north", "long_name": "Latitude"}


@pytest.mark.parametrize("engine", ["h5netcdf", "zarr"])
@pytest.mark.parametrize("empty", [False, True])
def test_numeric_catalog_round_trip(engine: str, empty: bool, tmp_path: Any) -> None:
    """Both installed storage backends preserve rows, cell coordinates, flags, and JSON choices."""
    original = catalog(prepared(), mode="sequential", lag_window=(1, 4), min_duration=20 if empty else 1)
    assert all(variable.dtype.kind != "O" for variable in original.variables.values())
    assert "featureType" not in original.attrs
    for name, variable in original.data_vars.items():
        assert "long_name" in variable.attrs
        assert "standard_name" not in variable.attrs
        if str(name).endswith("censored"):
            np.testing.assert_array_equal(variable.attrs["flag_values"], [0, 1])
    path = tmp_path / engine
    if engine == "zarr":
        original.to_zarr(path, mode="w", consolidated=False)
        with xr.open_zarr(path, consolidated=False) as opened:
            loaded = opened.compute()
    else:
        original.to_netcdf(path, engine="h5netcdf")
        with xr.open_dataset(path, engine="h5netcdf") as opened:
            loaded = opened.load()
    xr.testing.assert_equal(loaded, original)
    assert loaded.attrs["compound_manifest"] == original.attrs["compound_manifest"]
    flags = loaded["first_left_censored"].attrs
    assert flags["flag_meanings"] == "not_censored censored"
    np.testing.assert_array_equal(flags["flag_values"], [0, 1])
    assert flags["long_name"] == "First Source Left Censored"
    assert flags["units"] == "1"
    for name in loaded.data_vars:
        assert loaded[name].attrs["long_name"] == original[name].attrs["long_name"], name
    for dim in ("lat", "station"):
        assert loaded[dim].attrs == original[dim].attrs, dim
    for name in original.data_vars:
        assert loaded[name].dtype.kind == original[name].dtype.kind


@pytest.mark.parametrize("lazy", ["all", "mixed", "auxiliary"])
def test_lazy_input_is_rejected_without_evaluation(lazy: str) -> None:
    hazards = prepared()
    if lazy == "all":
        hazards = hazards.chunk({"time": 3})
    elif lazy == "mixed":
        hazards["valid"] = hazards["valid"].chunk({"time": 3})
    else:
        hazards = hazards.assign_coords(aux=xr.DataArray(np.arange(12), dims="time").chunk({"time": 3}))

    def fail_scheduler(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("The catalog evaluated lazy input.")

    with (
        dask.config.set(scheduler=fail_scheduler),
        pytest.raises(InvalidArgumentError, match="Select a bounded subset"),
    ):
        catalog(hazards)


@pytest.mark.parametrize(
    "options",
    [
        {"first": "absent"},
        {"second": "dry"},
        {"mode": "invalid"},
        {"lag_window": (1, 2)},
        {"mode": "sequential", "lag_window": (0, 2)},
        {"min_duration": 0},
        {"min_duration": True},
        {"min_duration": 1.5},
    ],
)
def test_invalid_options_are_rejected_even_without_cells(options: dict[str, Any]) -> None:
    hazards = prepared().isel(lat=slice(0, 0))
    with pytest.raises(InvalidArgumentError):
        compound.compound_events_xarray(hazards, **{"first": "dry", "second": "hot", "mode": "concurrent", **options})


@pytest.mark.parametrize(
    "problem,error",
    [
        ("type", InputTypeError),
        ("dtype", InputTypeError),
        ("missing", InvalidArgumentError),
        ("gap", CoordinateValidationError),
        ("duplicate", CoordinateValidationError),
        ("manifest", InvalidArgumentError),
        ("manifest_nan", InvalidArgumentError),
        ("reserved", DimensionMismatchError),
        ("object_labels", InputTypeError),
    ],
)
def test_malformed_prepared_input_is_rejected(problem: str, error: type[Exception]) -> None:
    hazards: Any = prepared()
    if problem == "type":
        hazards = []
    elif problem == "dtype":
        hazards["valid"] = hazards["valid"].astype(int)
    elif problem == "missing":
        hazards = hazards.drop_vars("valid")
    elif problem == "gap":
        hazards = hazards.isel(time=[0, 2, 3])
    elif problem == "duplicate":
        hazards = hazards.assign_coords(time=[hazards.time.values[0]] * 12)
    elif problem == "manifest":
        hazards.attrs["compound_manifest"] = "[]"
    elif problem == "manifest_nan":
        hazards.attrs["compound_manifest"] = '{"threshold": NaN}'
    elif problem == "reserved":
        hazards = hazards.rename(lat="event")
    else:
        hazards = hazards.assign_coords(station=np.array([object(), object()], dtype=object))
    with pytest.raises(error):
        catalog(hazards)


@pytest.mark.parametrize(
    "labels,repeated",
    [
        (["dry", "dry", "unused"], ["dry"]),
        (["z", "a", "z", "a", "dry", "hot"], ["a", "z"]),
        (["z", "a"] * 1000 + ["dry", "hot"], ["a", "z"]),
    ],
)
def test_duplicate_hazard_labels_are_rejected(labels: list[str], repeated: list[str]) -> None:
    """Report each duplicate once in sorted order, including inputs with many repeated labels."""
    hazards = prepared(grid=False).isel(hazard=[0] * len(labels)).assign_coords(hazard=labels)
    with pytest.raises(InvalidArgumentError, match="unique") as error:
        catalog(hazards)
    assert f"{repeated} appears more than once" in str(error.value)


def test_row_budget_is_compared_with_the_candidate_count() -> None:
    """The row builder refuses an over-budget candidate block and names the remaining rows and the limit."""
    from climate_indices.compound import _events as events_module

    second = events_module._SourceRuns(
        start=np.array([2, 4, 6]),
        end=np.array([2, 4, 6]),
        left_censored=np.zeros(3, dtype=bool),
        right_censored=np.zeros(3, dtype=bool),
    )
    valid = np.ones(9, dtype=bool)
    prefix = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    unknown = np.arange(prefix.size) - prefix
    rejected, rows = events_module._sequential_rows(second, 0, 0, 9, (1, 5), unknown, budget=(2, 10))
    assert rejected == 0
    assert rows is not None
    assert rows[0].size == 2
    with pytest.raises(InvalidArgumentError, match="10-row limit this result allows \\(1 row remaining\\)"):
        events_module._sequential_rows(second, 0, 0, 9, (1, 5), unknown, budget=(1, 10))


def test_concurrent_row_budget_matches_the_candidate_count() -> None:
    """The concurrent path applies the same budget before building a row block."""
    from climate_indices.compound import _events as events_module

    second = events_module._SourceRuns(
        start=np.array([1, 2, 3]),
        end=np.array([1, 2, 3]),
        left_censored=np.zeros(3, dtype=bool),
        right_censored=np.zeros(3, dtype=bool),
    )
    rows = events_module._concurrent_rows(second, 1, 3, budget=(3, 10))
    assert rows is not None
    assert rows[0].size == 3
    with pytest.raises(InvalidArgumentError, match="One source run matches 3 runs.*\\(2 rows remaining\\)"):
        events_module._concurrent_rows(second, 1, 3, budget=(2, 10))


@pytest.mark.parametrize("limit,grid", [(5, False), (4, False), (5, True), (10, True), (15, True), (9, True)])
def test_event_row_cap_bounds_the_catalog(monkeypatch: pytest.MonkeyPatch, limit: int, grid: bool) -> None:
    """The cap counts rows exactly, across cells, and fails before the whole catalog is allocated."""
    monkeypatch.setattr("climate_indices.compound._catalog._MAX_EVENTS", limit)
    hazards = prepared(grid=grid)
    total = 15 if grid else 5
    if limit >= total:
        assert catalog(hazards, mode="sequential").sizes["event"] == total
    else:
        with pytest.raises(InvalidArgumentError, match="row limit"):
            catalog(hazards, mode="sequential")


def test_beta_warning_and_lifecycle(caplog: Any) -> None:
    """Warn once at the caller and log eager completion or structured failure."""
    hazards = prepared(grid=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compound.compound_events_xarray(hazards, first="dry", second="hot", mode="concurrent")
    beta = [warning for warning in caught if issubclass(warning.category, BetaFeatureWarning)]
    assert len(beta) == 1
    assert beta[0].filename == __file__
    records = [record.msg for record in caplog.records if isinstance(record.msg, dict)]
    completed = [record for record in records if record.get("index_type") == "compound_events_xarray"]
    assert any(
        record.get("event") == "calculation_completed" and record.get("evaluation") == "eager" for record in completed
    )
    with pytest.raises(InvalidArgumentError):
        catalog(hazards, min_duration=0)
    failures = [
        record
        for record in (entry.msg for entry in caplog.records)
        if isinstance(record, dict) and record.get("event") == "calculation_failed"
    ]
    assert len(failures) == 1
    assert failures[0]["index_type"] == "compound_events_xarray"
    assert failures[0]["error_type"] == "InvalidArgumentError"
