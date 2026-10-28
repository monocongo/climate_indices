"""Run Theory parity, routing, and inherited event-summary behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
import xarray as xr

from climate_indices import energy, flood, runs
from tests import conftest

native = conftest.import_native()


def test_identify_runs_reaches_the_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(runs, "_native", recorder, raising=False)
    with np.errstate(all="ignore"):
        found = runs.identify_runs([0.5, -1.2, -1.5, -0.8, 0.3, -1.1, -1.3])
    assert recorder.calls == {"identify_runs"}
    np.testing.assert_array_equal(found.start_index, [1, 5])
    np.testing.assert_array_equal(found.duration, [2, 2])


@pytest.mark.parametrize("direction, sign", [("below", 1), ("above", -1)])
@pytest.mark.parametrize("max_gap", [0, 1, 2, 10**400])
@pytest.mark.parametrize("min_duration", [1, 3, 10**400])
@pytest.mark.parametrize(
    "values",
    [[], [np.nan], [0], [-1], [-2], [-2, -2, 0, -3, np.nan, -4, 0, 0, -5], [-np.inf, 0, -2, np.inf, -3]],
)
def test_run_metrics_match_python(
    monkeypatch: pytest.MonkeyPatch,
    direction: runs.Direction,
    sign: int,
    max_gap: int,
    min_duration: int,
    values: list[float],
) -> None:
    series = np.asarray(values, dtype=float) * sign
    rust, python, calls = conftest.rust_and_python(
        monkeypatch,
        runs,
        lambda: runs.identify_runs(
            series, threshold=-1.0 * sign, direction=direction, min_duration=min_duration, max_gap=max_gap
        ),
    )
    assert calls == {"identify_runs"}
    conftest.assert_native_parity(rust, python)
    for field in fields(rust):
        actual, expected = getattr(rust, field.name), getattr(python, field.name)
        assert actual.dtype == expected.dtype
        # The shared pairwise reduction preserves NumPy's grouping, not just its tolerance.
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("length", [7, 8, 127, 128, 129, 8191, 8192, 8193, 36500])
@pytest.mark.parametrize("max_gap", [0, 1])
def test_magnitude_preserves_pairwise_sum_of_only_active_values(
    monkeypatch: pytest.MonkeyPatch, length: int, max_gap: int
) -> None:
    deviations = np.random.default_rng(length).lognormal(0, 8, length)
    series = -deviations
    if max_gap:
        series = np.column_stack((series, np.zeros(length))).ravel()
    rust, python, calls = conftest.rust_and_python(
        monkeypatch, runs, lambda: runs.identify_runs(series, threshold=0, max_gap=max_gap)
    )
    assert calls == {"identify_runs"}
    # NumPy's reduction order above the block size is platform-dependent (x86 SIMD
    # differs from the scalar reference the Rust port follows), so compare a few ULP.
    np.testing.assert_allclose(rust.magnitude, python.magnitude, rtol=1e-12, atol=0)
    assert rust.magnitude[0] == pytest.approx(deviations.sum(), rel=1e-12, abs=0)


@pytest.mark.parametrize("layout", ["strided", "reversed", "masked", "float32", "unaligned"])
def test_prepared_input_layouts(monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    values = np.array([-2, -2, 0, -3, np.nan, -4, 0, -5], dtype=float)
    if layout == "strided":
        values = np.repeat(values, 2)[::2]
    elif layout == "reversed":
        values = values[::-1]
    elif layout == "masked":
        values = np.ma.masked_invalid(values)
    elif layout == "float32":
        values = values.astype(np.float32)
    else:
        values = np.ndarray(values.shape, dtype=float, buffer=bytearray(values.nbytes + 1), offset=1)
        values[:] = [-2, -2, 0, -3, np.nan, -4, 0, -5]
    before = values.copy()
    rust, python, calls = conftest.rust_and_python(monkeypatch, runs, lambda: runs.identify_runs(values, max_gap=1))
    assert calls == ({"identify_runs"} if layout != "unaligned" else set())
    conftest.assert_native_parity(rust, python)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("entry", [runs.summarize_runs, energy.energy_drought_events, flood.flood_event_summary])
def test_event_summaries_inherit_run_metrics(monkeypatch: pytest.MonkeyPatch, entry: Callable[..., Any]) -> None:
    values = np.array([-2, 0, -3, np.nan, 2, 0, 3, 3], dtype=float)
    options = {} if entry is flood.flood_event_summary else {"max_gap": 1}
    rust, python, calls = conftest.rust_and_python(
        monkeypatch, runs, lambda: entry(values, periods=np.array([2000] * 4 + [2001] * 4), **options)
    )
    assert calls == {"identify_runs"}
    conftest.assert_native_parity(rust, python)


@pytest.mark.parametrize("lazy", [False, True])
def test_xarray_per_cell_path_inherits_run_metrics(monkeypatch: pytest.MonkeyPatch, lazy: bool) -> None:
    data = xr.DataArray([[-2, 0], [0, -3], [-4, -3], [np.nan, 2]], dims=("time", "site"))
    if lazy:
        data = data.chunk({"time": -1, "site": 1})

    def run() -> xr.DataArray:
        # Dask thread workers have their own NumPy policy.
        with np.errstate(all="ignore"):
            found = runs.identify_runs_xarray(data, max_gap=1)
            assert isinstance(found, xr.DataArray)
            result = found.compute(scheduler="synchronous")
            assert isinstance(result, xr.DataArray)
            return result

    rust, python, calls = conftest.rust_and_python(monkeypatch, runs, run)
    assert calls == {"identify_runs"}
    np.testing.assert_array_equal(rust.values, python.values)


def test_stale_extension_keeps_python_and_runtime_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runs, "_native", SimpleNamespace())
    with np.errstate(all="ignore"):
        assert runs.identify_runs([-2]).duration.tolist() == [1]

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("kernel failed")

    monkeypatch.setattr(runs, "_native", SimpleNamespace(identify_runs=broken))
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="kernel failed"):
        runs.identify_runs([-2])


@pytest.mark.parametrize(
    "series, threshold",
    [
        ([-0.0, 0.0, -1.0], 0.0),
        ([-np.nextafter(0.0, 1.0), 0.0, -np.nextafter(0.0, 1.0)], 0.0),
        ([np.nextafter(1.0, 0.0), 2.0, np.nextafter(1.0, 0.0)], 1.0),
        ([-1e308, 0.0, -1e308], 1e308),
    ],
)
def test_small_deviations_and_overflow_match_python(
    monkeypatch: pytest.MonkeyPatch, series: list[float], threshold: float
) -> None:
    rust, python, calls = conftest.rust_and_python(
        monkeypatch, runs, lambda: runs.identify_runs(series, threshold=threshold, max_gap=1)
    )
    assert calls == {"identify_runs"}
    conftest.assert_native_parity(rust, python)
    np.testing.assert_array_equal(rust.magnitude, python.magnitude)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_warning_promotion_keeps_python(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(runs, "_native", recorder)
    with np.errstate(all="ignore"):
        runs.identify_runs([-2, 0, -3])
    assert not recorder.calls


@pytest.mark.parametrize("policy", ["warn", "raise", "call", "print", "log"])
def test_reporting_policies_keep_python(
    monkeypatch: pytest.MonkeyPatch, policy: Literal["warn", "raise", "call", "print", "log"]
) -> None:
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(runs, "_native", recorder)
    with np.errstate(all=policy):
        runs.identify_runs([-2, 0, -3])
    assert not recorder.calls


@pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
def test_direct_binding_empty_axes(shape: tuple[int, int]) -> None:
    result = native.identify_runs(np.empty(shape), -1.0, "below", 1, 0)
    assert len(result) == 9
    assert all(array.size == 0 for array in result)


def test_direct_binding_rejects_invalid_direction_and_unaligned_storage() -> None:
    with pytest.raises(ValueError, match="direction"):
        native.identify_runs(np.zeros((1, 1)), -1.0, "sideways", 1, 0)
    unaligned = np.ndarray((2, 1), dtype=float, buffer=bytearray(17), offset=1)
    with pytest.raises(ValueError, match="unaligned"):
        native.identify_runs(unaligned, -1.0, "below", 1, 0)


def test_block_kernel_orders_cells_and_matches_each_series(monkeypatch: pytest.MonkeyPatch) -> None:
    values = np.random.default_rng(1436).normal(size=(321, 4))
    values[30:32] = np.nan
    # A borrowed view with negative strides exercises both axes of the Rust input.
    values = values[::-1, ::-1]
    with np.errstate(all="ignore"):
        result = native.identify_runs(values, -0.5, "below", 3, 1)
    cell_indices, *metrics = result
    monkeypatch.setattr(runs, "_native", None)
    assert np.all(cell_indices[1:] >= cell_indices[:-1])
    for cell in range(values.shape[1]):
        expected = runs.identify_runs(values[:, cell], threshold=-0.5, min_duration=3, max_gap=1)
        for field, actual in zip(fields(expected), metrics, strict=True):
            np.testing.assert_array_equal(actual[cell_indices == cell], getattr(expected, field.name))
