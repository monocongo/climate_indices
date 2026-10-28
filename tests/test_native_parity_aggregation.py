"""Timescale Aggregation parity, including seams that must retain NumPy."""

import warnings
from typing import Any

import numpy as np
import pytest
import xarray as xr

from climate_indices import compute
from tests import conftest

native = conftest.import_native()


def _compare(monkeypatch: pytest.MonkeyPatch, values: np.ndarray, scale: int) -> set[str]:
    recorder = conftest.NativeRecorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(compute, "_native", recorder)
        actual = compute.sum_to_scale(values, scale)
        monkeypatch.setattr(compute, "_native", None)
        expected = compute.sum_to_scale(values, scale)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    # ADR-0017 requires both tolerances, including cancellation near zero.
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(np.signbit(actual[actual == 0]), np.signbit(expected[expected == 0]))
    return recorder.calls


@pytest.mark.parametrize("cells", [None, 1, 3])
@pytest.mark.parametrize("scale", [2, 3, 7, 8, 12, 128, 129, 720])
@pytest.mark.parametrize(
    "case",
    ["positive", "signed", "missing", "all_missing", "infinity", "cancellation", "overflow", "underflow", "zero"],
)
def test_window_parity(monkeypatch: pytest.MonkeyPatch, cells: int | None, scale: int, case: str) -> None:
    rng = np.random.default_rng(1485)
    values: np.ndarray = rng.uniform(0, 100, size=2 * scale + 13)
    if case == "signed":
        values -= 50
    elif case == "missing":
        values[scale : scale + 2] = np.nan
    elif case == "all_missing":
        values[:] = np.nan
    elif case == "infinity":
        values[scale] = np.inf
        values[scale + 1] = -np.inf
    elif case == "cancellation":
        values = np.resize([1e16, 1, 1, 1, 1, 1, 1, -1e16], values.size)
    elif case == "overflow":
        values = np.resize([1e308, 1e308, -1e308, -1e308, 1], values.size)
    elif case == "underflow":
        values *= np.nextafter(0.0, 1.0)
    elif case == "zero":
        values[:] = -0.0
    if cells is not None:
        values = np.broadcast_to(values[:, None, None], (values.size, cells, 1)).copy()
    calls = _compare(monkeypatch, values, scale)
    assert calls == ({"trailing_sum"} if cells is not None or scale < 8 else set())


def test_independent_reference_and_mean_is_divided_once(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    values = np.array([3.0, 4, 6, 2, 1, 3, 5, 8, 5])
    expected = np.array([np.nan, np.nan, 13, 12, 9, 6, 9, 16, 18])
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(compute.sum_to_scale(values, 3), expected)
        mean = compute.prepare_scaled(values, 3, compute.Periodicity.monthly, reshape=False, aggregation="mean")
    np.testing.assert_allclose(mean, expected / 3, rtol=1e-10, atol=1e-10)
    assert recorder.calls == {"trailing_sum"}


@pytest.mark.parametrize("shape", [(), (0,), (1,), (0, 2, 1), (1, 2, 1), (8, 2, 1), (8, 1, 1), (8, 0, 1)])
@pytest.mark.parametrize("scale", [0, -1, 1, 2, 8, 9, 1.5, True])
def test_direct_boundary_behavior_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, shape: tuple[int, ...], scale: Any
) -> None:
    values = np.zeros(shape)
    outcomes: list[Any] = []
    for backend in (native, None):
        monkeypatch.setattr(compute, "_native", backend)
        with np.errstate(all="ignore"):
            try:
                outcomes.append(compute.sum_to_scale(values, scale))
            except (ValueError, TypeError) as error:
                outcomes.append((type(error), str(error)))
    if isinstance(outcomes[0], tuple):
        assert outcomes[0] == outcomes[1]
    else:
        np.testing.assert_array_equal(outcomes[0], outcomes[1])
        if scale == 1:
            assert outcomes[0] is values


@pytest.mark.parametrize(
    "kind", ["float32", "integer", "subclass", "fortran", "strided", "unaligned", "data_array", "two_dimensional"]
)
def test_ineligible_layouts_retain_python(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    values: Any = np.arange(120.0).reshape(20, 3, 2)
    if kind == "float32":
        values = values.astype(np.float32)
    elif kind == "integer":
        values = values.astype(np.int64)
    elif kind == "subclass":

        class Subclass(np.ndarray):
            pass

        values = values.view(Subclass)
    elif kind == "fortran":
        values = np.asfortranarray(values)
    elif kind == "strided":
        values = values[::-2, ::-1]
    elif kind == "unaligned":
        values = np.ndarray(values.shape, dtype=np.float64, buffer=bytearray(values.nbytes + 1), offset=1)
        values.fill(1)
    elif kind == "data_array":
        # a duck array has no `flags`; dispatch must decline it before reading them
        values = xr.DataArray(values)
    elif kind == "two_dimensional":
        values = values.reshape(20, 6)
    if kind == "two_dimensional":
        # ignore reporting so the dispatch predicate itself must decline the 2-D input
        recorder = conftest.NativeRecorder(native)
        monkeypatch.setattr(compute, "_native", recorder)
        with np.errstate(all="ignore"), pytest.raises(ValueError):
            compute.sum_to_scale(values, 3)
        assert not recorder.calls
    else:
        assert _compare(monkeypatch, values, 3) == set()


def test_mask_normalization_and_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    values = np.ma.array(np.arange(24.0), mask=np.arange(24) == 8)
    assert _compare(monkeypatch, values, 3) == {"trailing_sum"}
    assert _compare(monkeypatch, np.ma.array(np.zeros(24), mask=True), 3) == {"trailing_sum"}
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    assert compute.sum_to_scale(values, 1) is values
    assert not recorder.calls
    assert _compare(monkeypatch, np.arange(24.0)[::-2], 3) == {"trailing_sum"}


@pytest.mark.parametrize("policy", ["warn", "raise", "call", "log", "print"])
def test_reporting_policies_retain_python(monkeypatch: pytest.MonkeyPatch, policy: Any) -> None:
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with np.errstate(all=policy):
        compute.sum_to_scale(np.ones((12, 2, 1)), 3)
    assert not recorder.calls


def test_runtime_warning_error_filter_retain_python(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("error", RuntimeWarning)
        compute.sum_to_scale(np.ones(12), 3)
    assert not recorder.calls


def test_stale_extension_and_runtime_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stale:
        pass

    monkeypatch.setattr(compute, "_native", Stale())
    with np.errstate(all="ignore"):
        assert compute.sum_to_scale(np.ones(12), 3)[-1] == 3

    class Broken:
        def trailing_sum(self, *args: Any) -> np.ndarray:
            raise RuntimeError("kernel failed")

    monkeypatch.setattr(compute, "_native", Broken())
    with np.errstate(all="ignore"), pytest.raises(RuntimeError, match="kernel failed"):
        compute.sum_to_scale(np.ones(12), 3)


def test_binding_checks_alignment_and_scale() -> None:
    values = np.ones((12, 2))
    with pytest.raises(ValueError):
        native.trailing_sum(values, 0)
    unaligned = np.ndarray((12, 2), dtype=np.float64, buffer=bytearray(values.nbytes + 1), offset=1)
    with pytest.raises(ValueError, match="unaligned"):
        native.trailing_sum(unaligned, 3)
    empty = np.ones((0, 2))[::-1]
    assert native.trailing_sum(empty, 3).shape == empty.shape
