"""Parity of the Rust flood kernels with the Python reference implementations.

Each test computes the same result twice through the public ``climate_indices.flood``
API: once with ``climate_indices.flood._native._native`` replaced by a recorder
around the Rust extension, and once with it set to None, which runs the
pure-Python implementations. The recorder proves the first run reached the Rust
kernels, so the comparison is never Python against Python. The contract is
``rtol = atol = 1e-10`` with matching NaN positions, and, for the API, an
identical returned state.

Native dispatch also requires NumPy floating-point errors to be ignored, as in
``compute._native_float64``, so every run here is wrapped in ``np.errstate``.
Skipped when the extension is not built (``uv run maturin develop --release``),
unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1`` is set, as in CI's native legs.

No external numeric oracle exists for the flood family (``tests/fixture/flood/README.md``);
the real-record cases run the Fresno GHCN daily rainfall the KBDI reference uses.
"""

from __future__ import annotations

import csv
import logging
import traceback
import types
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_indices import _recurrence as recurrence_runner
from climate_indices import flood
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.flood import _native as flood_native
from climate_indices.utils import transform_to_366day
from tests import conftest

native = conftest.import_native()

_FRESNO = Path(__file__).parent / "fixture" / "kbdi_ghcn" / "fresno_1991_2020.csv"
_FRESNO_START, _FRESNO_YEARS = 1991, 30


def _rust_and_python(monkeypatch: pytest.MonkeyPatch, run: Callable[[], Any]) -> tuple[Any, Any, set[str]]:
    return conftest.rust_and_python(monkeypatch, flood_native, run)


_assert_parity = conftest.assert_native_parity


def _fresno_rain() -> np.ndarray:
    """The Fresno GHCN daily precipitation record, 1991-2020, in its Gregorian layout."""
    with _FRESNO.open(newline="") as handle:
        return np.array([float(row["precipitation_mm"]) for row in csv.DictReader(handle)])


def _fresno_all_leap_rain() -> np.ndarray:
    """The Fresno record in the 366-day layout EDI and the Flood Index read."""
    return transform_to_366day(_fresno_rain(), _FRESNO_START, _FRESNO_YEARS)


def _fresno_pe() -> np.ndarray:
    """Python-path PE of the all-leap Fresno record, a fixed input for EDI and I_F."""
    with np.errstate(all="ignore"):
        return flood.effective_precipitation(_fresno_all_leap_rain())


def _synthetic_rain(shape: tuple[int, ...], seed: int, missing: float = 0.0) -> np.ndarray:
    """Showery daily rain: about half the days dry, the rest gamma-distributed, some NaN."""
    rng = np.random.default_rng(seed)
    rain = rng.gamma(0.5, 8.0, shape) * (rng.random(shape) < 0.5)
    rain[rng.random(shape) < missing] = np.nan
    return rain


# Effective precipitation.


@pytest.mark.parametrize("duration", [365, 30, 2, 1])
def test_effective_precipitation_on_the_fresno_record(monkeypatch, duration: int) -> None:
    run = partial(flood.effective_precipitation, _fresno_all_leap_rain(), duration=duration)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"effective_precipitation"}
    _assert_parity(rust, python)
    assert np.isnan(rust[: duration - 1]).all()
    assert np.isfinite(rust[duration - 1 :]).all()


def test_effective_precipitation_shorter_than_its_window_is_all_nan(monkeypatch) -> None:
    """A series shorter than the window has nothing to compute, so neither path runs a kernel."""
    rust, python, calls = _rust_and_python(
        monkeypatch, partial(flood.effective_precipitation, np.ones(20), duration=30)
    )
    assert calls == set()
    _assert_parity(rust, python)
    assert np.isnan(rust).all()


def test_effective_precipitation_with_nan_and_masked_days(monkeypatch) -> None:
    rain = _synthetic_rain((3 * 366,), seed=1, missing=0.01)
    masked = np.ma.masked_array(rain, mask=np.zeros(rain.shape, dtype=bool))
    masked[400:403] = np.ma.masked
    rust, python, calls = _rust_and_python(monkeypatch, partial(flood.effective_precipitation, masked, duration=60))
    assert calls == {"effective_precipitation"}
    _assert_parity(rust, python)
    # every 60-day window holding one of the masked days 400-402 is NaN
    assert np.isnan(rust[400:462]).all()


@pytest.mark.parametrize(
    ("shape", "spatial_time_major"),
    [((2 * 366, 2, 3), False), ((2 * 366, 12, 1), True), ((4, 366), False)],
    ids=["spatial-block", "declared-calendar-shaped-block", "years-by-days"],
)
def test_effective_precipitation_layouts(monkeypatch, shape: tuple[int, ...], spatial_time_major: bool) -> None:
    rain = _synthetic_rain(shape, seed=2, missing=0.005)
    run = partial(flood.effective_precipitation, rain, duration=45, spatial_time_major=spatial_time_major)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"effective_precipitation"}
    _assert_parity(rust, python)
    assert rust.shape == shape


# EDI.


def test_edi_on_the_fresno_record(monkeypatch) -> None:
    # 1991 is the PE warm-up year, so the Calibration Period starts in 1992
    run = partial(flood.edi, _fresno_pe(), _FRESNO_START, 1992, 2020)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"edi"}
    _assert_parity(rust, python)
    assert np.isfinite(rust[366:]).mean() > 0.99


def test_edi_with_a_partial_final_year(monkeypatch) -> None:
    pe = _fresno_pe()[:-100]
    rust, python, calls = _rust_and_python(monkeypatch, partial(flood.edi, pe, _FRESNO_START, 1992, 2019))
    assert calls == {"edi"}
    _assert_parity(rust, python)
    assert rust.shape == pe.shape


def _degenerate_pe_block(years: int) -> np.ndarray:
    """A three-cell PE block: a regular cell, a constant one, and one with a single calibration year."""
    # strictly positive values, so the regular cell never has a zero-variance calendar day
    pe = np.random.default_rng(3).gamma(2.0, 3.0, (years * 366, 3, 1)) + 0.1
    pe[:, 1] = 5.0
    pe[366:, 2] = np.nan
    return pe


def test_edi_zero_variance_and_one_sample_cells_are_nan(monkeypatch) -> None:
    pe = _degenerate_pe_block(6)
    rust, python, calls = _rust_and_python(monkeypatch, partial(flood.edi, pe, 2000, 2000, 2005))
    assert calls == {"edi"}
    _assert_parity(rust, python)
    assert np.isfinite(rust[:, 0]).all()
    assert np.isnan(rust[:, 1]).all()
    assert np.isnan(rust[:, 2]).all()


def test_edi_years_by_days_layout(monkeypatch) -> None:
    pe = _fresno_pe().reshape(_FRESNO_YEARS, 366)
    rust, python, calls = _rust_and_python(monkeypatch, partial(flood.edi, pe, _FRESNO_START, 1992, 2020))
    assert calls == {"edi"}
    _assert_parity(rust, python)
    assert rust.shape == pe.shape


# Flood Index.


@pytest.mark.parametrize("year_start_month", [1, 3, 7, 12])
def test_flood_index_on_the_fresno_record(monkeypatch, year_start_month: int) -> None:
    # a single series reduces its annual maxima pairwise, as NumPy does
    last = 2020 if year_start_month == 1 else 2019
    run = partial(flood.flood_index, _fresno_pe(), _FRESNO_START, 1992, last, year_start_month=year_start_month)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"flood_index"}
    _assert_parity(rust, python)


@pytest.mark.parametrize("year_start_month", [1, 10])
def test_flood_index_over_a_spatial_block(monkeypatch, year_start_month: int) -> None:
    # several cells reduce their annual maxima sequentially, as NumPy does
    pe = flood.effective_precipitation(_synthetic_rain((12 * 366, 2, 2), seed=4, missing=0.002), duration=30)
    run = partial(flood.flood_index, pe, 2000, 2001, 2010, year_start_month=year_start_month)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"flood_index"}
    _assert_parity(rust, python)


def test_flood_index_zero_variance_and_one_sample_cells_are_nan(monkeypatch) -> None:
    pe = _degenerate_pe_block(6)
    run = partial(flood.flood_index, pe, 2000, 2000, 2004, year_start_month=1)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"flood_index"}
    _assert_parity(rust, python)
    assert np.isfinite(rust[:, 0]).all()
    assert np.isnan(rust[:, 1]).all()
    assert np.isnan(rust[:, 2]).all()


# Summation order and the rounding guard.
#
# The 1e-10 parity tolerance cannot see which order a kernel sums a calibration
# sample in: reordering a few dozen rainfall values moves the mean by about 1e-16
# relative. These cases use a sample whose sum depends on the order, so the
# results compare exactly.

_ABSORBING_YEARS = 16


def _absorbing_annual_maxima() -> np.ndarray:
    """Sixteen annual maxima, one 1e16 among ones.

    ``1e16`` absorbs a ``1`` added to it (the spacing there is 2), so a left-to-right
    sum loses all fifteen ones while NumPy's pairwise sum keeps fourteen of them.
    """
    maxima = np.ones(_ABSORBING_YEARS)
    maxima[0] = 1e16
    return maxima


def _constant_years(maxima: np.ndarray, cells: tuple[int, ...] | None) -> np.ndarray:
    """A PE series whose every day of year ``y`` holds ``maxima[y]``, one series or a ``(time, *cells)`` block."""
    series = np.repeat(maxima, 366)
    return series if cells is None else series.reshape(-1, *([1] * len(cells))) * np.ones((1, *cells))


def test_the_absorbing_sample_separates_numpys_two_axis_zero_sums() -> None:
    """The premise of the sum-order tests below, so they cannot pass vacuously.

    NumPy sums a 1-D sample, and a one-column block, pairwise; it sums a block of
    several columns left to right over the rows.
    """
    maxima = _absorbing_annual_maxima()
    assert maxima.sum() - 1e16 == 14.0
    assert maxima.reshape(-1, 1).sum(axis=0)[0] - 1e16 == 14.0
    assert np.tile(maxima[:, None], (1, 2)).sum(axis=0)[0] - 1e16 == 0.0


@pytest.mark.parametrize("cells", [None, (1, 1), (2, 1)], ids=["one-series", "one-cell-block", "two-cell-block"])
def test_flood_index_sums_its_annual_maxima_in_numpys_order(monkeypatch, cells) -> None:
    """One column sums pairwise and several sum sequentially; either branch swapped changes the bits."""
    pe = _constant_years(_absorbing_annual_maxima(), cells)
    run = partial(flood.flood_index, pe, 2000, 2000, 2015, year_start_month=1)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"flood_index"}
    assert np.isfinite(rust).all()
    np.testing.assert_array_equal(rust, python)


@pytest.mark.parametrize("cells", [None, (2, 1)], ids=["years-by-days", "two-cell-block"])
def test_edi_sums_each_calendar_day_in_numpys_order(monkeypatch, cells) -> None:
    """EDI's samples always have 366 or more columns, so they sum left to right over the years."""
    pe = _constant_years(_absorbing_annual_maxima(), cells)
    rust, python, calls = _rust_and_python(monkeypatch, partial(flood.edi, pe, 2000, 2000, 2015))
    assert calls == {"edi"}
    assert np.isfinite(rust).all()
    np.testing.assert_array_equal(rust, python)


def _guard_block() -> np.ndarray:
    """A two-cell PE block of ten years around 1e6: cell 0 varies by one ulp steps, cell 1 by 1e-6 steps."""
    mean = 1e6
    steps = np.random.default_rng(13).integers(0, 4, 10)
    spacing = np.array([np.spacing(mean), 1e-6])
    annual = mean + steps[:, None] * spacing
    return np.repeat(annual, 366, axis=0).reshape(-1, 2, 1)


def test_the_rounding_guard_premise() -> None:
    """Cell 0 has variance, but its spread is under ``8 * eps * |mean|``; cell 1's is far over it."""
    annual = _guard_block()[::366, :, 0]
    guard = 8 * np.finfo(np.float64).eps * np.abs(annual.mean(axis=0))
    assert (annual.std(axis=0) > 0).all()
    assert annual.std(axis=0)[0] < guard[0]
    assert annual.std(axis=0)[1] > guard[1]


def test_edi_rounding_guard_leaves_a_sample_of_last_bit_noise_nan(monkeypatch) -> None:
    rust, python, calls = _rust_and_python(monkeypatch, partial(flood.edi, _guard_block(), 2000, 2000, 2009))
    assert calls == {"edi"}
    _assert_parity(rust, python)
    assert np.isnan(rust[:, 0]).all()
    assert np.isfinite(rust[:, 1]).all()


def test_flood_index_rounding_guard_leaves_a_sample_of_last_bit_noise_nan(monkeypatch) -> None:
    run = partial(flood.flood_index, _guard_block(), 2000, 2000, 2009, year_start_month=1)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"flood_index"}
    _assert_parity(rust, python)
    assert np.isnan(rust[:, 0]).all()
    assert np.isfinite(rust[:, 1]).all()


# Antecedent Precipitation Index.


@pytest.mark.parametrize("k", [0.85, 0.9, np.float32(0.95)])
def test_api_on_the_fresno_record(monkeypatch, k: float) -> None:
    run = partial(flood.antecedent_precipitation_index, _fresno_rain(), k, return_state=True)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"antecedent_precipitation_index"}
    _assert_parity(rust, python)


def test_api_spin_up(monkeypatch) -> None:
    run = partial(flood.antecedent_precipitation_index, _fresno_rain(), 0.9, spin_up=365, return_state=True)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"antecedent_precipitation_index"}
    _assert_parity(rust, python)
    assert rust.values.shape == (_fresno_rain().size - 365,)


@pytest.mark.parametrize(
    ("nan_policy", "max_gap_days"),
    [("propagate", 0), ("bridge", 1), ("bridge", 4)],
    ids=["propagate", "bridge-1", "bridge-4"],
)
def test_api_missing_day_policies(monkeypatch, nan_policy: str, max_gap_days: int) -> None:
    rain = _fresno_rain()
    rng = np.random.default_rng(5)
    rain[rng.random(rain.size) < 0.01] = np.nan
    rain[4000:4006] = np.nan
    run = partial(
        flood.antecedent_precipitation_index,
        rain,
        0.9,
        nan_policy=nan_policy,
        max_gap_days=max_gap_days,
        return_state=True,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"antecedent_precipitation_index"}
    _assert_parity(rust, python)


def test_api_over_a_spatial_block_with_masked_days(monkeypatch) -> None:
    rain = _synthetic_rain((900, 3, 2), seed=6, missing=0.01)
    masked = np.ma.masked_array(rain, mask=np.zeros(rain.shape, dtype=bool))
    masked[100:105, 0, 1] = np.ma.masked
    run = partial(
        flood.antecedent_precipitation_index, masked, 0.88, nan_policy="bridge", max_gap_days=3, return_state=True
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"antecedent_precipitation_index"}
    _assert_parity(rust, python)


@pytest.mark.parametrize(
    ("nan_policy", "max_gap_days", "split", "live_state"),
    [
        ("propagate", 0, 3000, True),
        ("propagate", 0, 5000, False),
        ("bridge", 3, 4003, True),
        ("bridge", 3, 4004, False),
    ],
    ids=["propagate-live-state", "propagate-poisoned-state", "bridge-inside-a-gap", "bridge-past-the-allowance"],
)
def test_api_resumed_from_its_state_is_bitwise_a_single_pass(
    monkeypatch, nan_policy: str, max_gap_days: int, split: int, live_state: bool
) -> None:
    """A run split at ``split`` and resumed from the returned APIState equals one pass, on both paths.

    The six-day gap starts on day 4000. The propagate split at 3000 resumes a live
    state, so the second run computes real values until it reaches the gap; the one
    at 5000 resumes a state the gap already poisoned. The bridge splits land
    inside the gap, so the resumed state carries a nonzero trailing gap count; the
    second one resumes once the allowance is already exceeded, so its state is
    poisoned.
    """
    rain = _fresno_rain()
    rain[4000:4006] = np.nan
    api = partial(flood.antecedent_precipitation_index, k=0.9, nan_policy=nan_policy, max_gap_days=max_gap_days)

    def single_and_split() -> tuple[Any, Any, Any]:
        single = api(rain, return_state=True)
        first = api(rain[:split], return_state=True)
        second = api(rain[split:], initial_state=first.state, return_state=True)
        return single, first, second

    (rust_single, rust_first, rust_second), python, calls = _rust_and_python(monkeypatch, single_and_split)
    assert calls == {"antecedent_precipitation_index"}
    _assert_parity((rust_single, rust_first, rust_second), python)
    assert bool(np.all(np.isfinite(rust_first.state.api))) == live_state
    if nan_policy == "propagate" and live_state:
        # the resumed run starts from real values and computes them until the gap
        assert np.isfinite(rust_second.values[: 4000 - split]).all()
        assert np.isnan(rust_second.values[4000 - split :]).all()
    if max_gap_days:
        assert rust_first.state.trailing_gap_days is not None
        assert int(rust_first.state.trailing_gap_days) > 0
    np.testing.assert_array_equal(np.concatenate([rust_first.values, rust_second.values]), rust_single.values)
    np.testing.assert_array_equal(rust_second.state.api, rust_single.state.api)
    np.testing.assert_array_equal(rust_second.state.trailing_gap_days, rust_single.state.trailing_gap_days)


def test_api_overflow_raises_the_python_error_on_both_paths(monkeypatch) -> None:
    rain = np.full(3, np.finfo(np.float64).max)
    message = "antecedent_precipitation_index produced a non-finite value from finite inputs"
    recorder = conftest.NativeRecorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(flood_native, "_native", recorder)
        with pytest.raises(InvalidArgumentError, match=message):
            flood.antecedent_precipitation_index(rain, 0.9)
        monkeypatch.setattr(flood_native, "_native", None)
        with pytest.raises(InvalidArgumentError, match=message):
            flood.antecedent_precipitation_index(rain, 0.9)
    # the dispatch also reads the extension's NonFiniteResultError to translate it
    assert "antecedent_precipitation_index" in recorder.calls


def test_the_api_kernel_returns_the_history_it_builds_in_the_requested_shape(monkeypatch) -> None:
    """The kernel builds the one history and returns it in the shape the runner asks for."""
    rain = _synthetic_rain((40, 3), seed=10)
    api = np.zeros(3)
    with np.errstate(all="ignore"):
        recurrence = flood_native.api_recurrence(
            rain, 0.9, api, np.isfinite(rain), np.ones(3, dtype=np.bool_), np.zeros(3, dtype=np.int64)
        )
        assert recurrence is not None
        result = recurrence((30, 3), 10, "propagate", 0)
    assert result is not None
    history, _ = result
    assert history is not None
    assert history.shape == (30, 3)
    # the values are the Python recurrence's, not merely the right shape: the
    # first ten days are spin-up, which the history leaves out; the public API reads
    # a 2-D array as one flattened series, so the three cells go in as (40, 3, 1)
    monkeypatch.setattr(flood_native, "_native", None)
    with np.errstate(all="ignore"):
        python = flood.antecedent_precipitation_index(rain.reshape(40, 3, 1), 0.9, spin_up=10)
    assert np.isfinite(history).all()
    _assert_parity(history, python.reshape(30, 3))


def test_the_runner_leaves_a_native_api_history_to_the_kernel(monkeypatch) -> None:
    """Through the shared runner, a native component allocates no Python history slot."""
    rain = _synthetic_rain((40, 3), seed=10)
    recorder = conftest.NativeRecorder(native)
    allocations: list[tuple[int, ...]] = []
    real_allocate = recurrence_runner._allocate_history

    def counting_allocate(component, keep, n_days, spin_up):
        history = real_allocate(component, keep, n_days, spin_up)
        if history is not None:
            allocations.append(history.shape)
        return history

    monkeypatch.setattr(recurrence_runner, "_allocate_history", counting_allocate)
    monkeypatch.setattr(flood_native, "_native", recorder)
    with np.errstate(all="ignore"):
        result = flood.antecedent_precipitation_index(rain, 0.9)
    assert "antecedent_precipitation_index" in recorder.calls
    assert allocations == []
    assert np.isfinite(result).any()


# Dispatch policy.


def _unaligned(values: np.ndarray) -> np.ndarray:
    """``values`` copied into a float64 array that starts one byte off an 8-byte boundary."""
    raw = np.zeros(values.size * 8 + 1, dtype=np.uint8)
    unaligned = raw[1:].view(np.float64)
    unaligned[:] = values
    assert not unaligned.flags.aligned
    return unaligned


def _api_recurrence(precipitation: np.ndarray, cells: tuple[int, ...] = ()) -> Any:
    """``flood_native.api_recurrence`` over ``precipitation`` with every cell valid and unstarted."""
    return flood_native.api_recurrence(
        precipitation,
        0.9,
        np.zeros(cells),
        np.isfinite(precipitation),
        np.ones(cells, dtype=np.bool_),
        np.zeros(cells, dtype=np.int64),
    )


def test_the_dispatch_declines_unaligned_and_empty_arrays() -> None:
    """The public entry points copy these into aligned arrays first, so the guard is checked on the dispatch itself."""
    series = _synthetic_rain((2 * 366,), seed=14)
    unaligned = _unaligned(series)
    no_days = np.empty((0,))
    no_cells = np.empty((2 * 366, 0))
    with np.errstate(all="ignore"):
        assert flood_native.effective_precipitation(unaligned, 30) is None
        assert flood_native.edi(unaligned, 0, 1) is None
        assert flood_native.flood_index(unaligned, 0, 2) is None
        assert _api_recurrence(unaligned) is None
        for empty in (no_days, no_cells):
            assert flood_native.effective_precipitation(empty, 30) is None
            assert flood_native.edi(empty, 0, 1) is None
            assert flood_native.flood_index(empty, 0, 2) is None
        assert _api_recurrence(no_days) is None
        assert _api_recurrence(no_cells, cells=(0,)) is None
        # control: the same series, aligned, reaches the kernels
        assert flood_native.effective_precipitation(np.array(unaligned), 30) is not None
        assert _api_recurrence(np.array(unaligned)) is not None


def _layouts() -> dict[str, np.ndarray]:
    """One ``(time, 3, 4)`` rain block stored four ways; every layout holds the same values."""
    rain = _synthetic_rain((2 * 366, 3, 4), seed=15, missing=0.005)
    time_last = np.ascontiguousarray(np.moveaxis(rain, 0, -1))
    wider_grid = np.zeros((2 * 366, 3, 9))
    wider_grid[:, :, :4] = rain
    return {
        "C-order": rain,
        "time-last": np.moveaxis(time_last, -1, 0),
        "regional-slice": wider_grid[:, :, :4],
        "Fortran-order": np.asfortranarray(rain),
    }


def test_a_time_last_block_is_viewed_and_a_layout_that_cannot_merge_is_copied_once() -> None:
    """The kernel's block adds no Python copy where NumPy can reshape to a view, and one where it cannot."""
    layouts = _layouts()
    time_last = layouts["time-last"]
    viewed = flood_native._block(time_last, 2 * 366, 12)
    assert np.shares_memory(viewed, time_last)
    for name in ("regional-slice", "Fortran-order"):
        copied = flood_native._block(layouts[name], 2 * 366, 12)
        assert not np.shares_memory(copied, layouts[name])
        assert copied.flags.c_contiguous
        np.testing.assert_array_equal(copied, layouts["C-order"].reshape(2 * 366, 12))


@pytest.mark.parametrize("layout", ["C-order", "time-last", "regional-slice", "Fortran-order"])
def test_effective_precipitation_reads_every_layout_the_same_way(monkeypatch, layout: str) -> None:
    layouts = _layouts()
    with np.errstate(all="ignore"):
        # the dispatch itself, which a direct caller of flood._native can hand any layout
        got = flood_native.effective_precipitation(layouts[layout], 30)
        expected = flood_native.effective_precipitation(layouts["C-order"], 30)
    assert got is not None
    np.testing.assert_array_equal(got, expected)
    run = partial(flood.effective_precipitation, layouts[layout], duration=30, spatial_time_major=True)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"effective_precipitation"}
    _assert_parity(rust, python)


@pytest.mark.parametrize("layout", ["time-last", "regional-slice", "Fortran-order"])
def test_the_api_kernel_takes_every_layout_inside_the_runner(layout: str) -> None:
    """The kernel's arrays are built when the runner calls it, from whatever layout the component holds."""
    layouts = _layouts()

    def run(block: np.ndarray) -> tuple[Any, Any]:
        recurrence = _api_recurrence(block, cells=(3, 4))
        assert recurrence is not None
        result = recurrence((2 * 366, 3, 4), 0, "bridge", 2)
        assert result is not None
        return result

    with np.errstate(all="ignore"):
        history, gaps = run(layouts[layout])
        expected_history, expected_gaps = run(layouts["C-order"])
    assert history is not None
    np.testing.assert_array_equal(history, expected_history)
    assert (gaps is None) == (expected_gaps is None)
    if gaps is not None:
        np.testing.assert_array_equal(gaps, expected_gaps)


def test_the_api_arrays_are_converted_inside_the_runners_guarded_region(monkeypatch, caplog) -> None:
    """A failed conversion (an allocation a layout needs) surfaces through the runner, which reports it."""

    def failing_block(*_: Any) -> np.ndarray:
        raise MemoryError("no room for the kernel's block")

    monkeypatch.setattr(flood_native, "_native", native)
    monkeypatch.setattr(flood_native, "_block", failing_block)
    caplog.set_level(logging.ERROR)
    with np.errstate(all="ignore"), pytest.raises(MemoryError) as exc_info:
        flood.antecedent_precipitation_index(_synthetic_rain((40, 3), seed=16), 0.9)
    frames = {frame.name for frame in traceback.extract_tb(exc_info.value.__traceback__)}
    assert "run_daily_recurrences" in frames
    failures = [
        record.msg
        for record in caplog.records
        if isinstance(record.msg, dict) and record.msg.get("event") == "calculation_failed"
    ]
    assert len(failures) == 1
    assert failures[0]["index_type"] == "antecedent_precipitation_index"
    assert failures[0]["error_type"] == "MemoryError"


def test_a_decay_constant_wider_than_float64_stays_in_python(monkeypatch) -> None:
    """An extended ``np.longdouble`` promotes the Python step beyond float64, which the kernel does not reproduce.

    Where ``long double`` is plain float64 (MSVC), NumPy treats the two as one
    type and the kernel takes it.
    """
    run = partial(flood.antecedent_precipitation_index, _synthetic_rain((50,), seed=7), np.longdouble(0.9))
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == (set() if np.dtype(np.longdouble) != np.float64 else {"antecedent_precipitation_index"})
    _assert_parity(rust, python)


@pytest.mark.parametrize(
    "options",
    [
        {"nan_policy": "bridge", "max_gap_days": 2**63},
        {"spin_up": 2**63},
    ],
)
def test_an_api_option_wider_than_the_binding_keeps_the_python_path(monkeypatch, options) -> None:
    """Options the Rust integer types cannot represent fall back instead of overflowing."""
    rain = _synthetic_rain((30,), seed=11, missing=0.1)
    recorder = conftest.NativeRecorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(flood_native, "_native", recorder)
        rust = flood.antecedent_precipitation_index(rain, 0.9, **options)
        monkeypatch.setattr(flood_native, "_native", None)
        python = flood.antecedent_precipitation_index(rain, 0.9, **options)
    assert recorder.calls == set()
    _assert_parity(rust, python)


def test_an_extension_that_predates_the_flood_kernels_keeps_the_python_path(monkeypatch) -> None:
    """A loaded extension without the flood kernels (a stale build) leaves every entry point in Python."""
    rain = _synthetic_rain((2 * 366,), seed=12)

    def run() -> tuple[Any, ...]:
        with np.errstate(all="ignore"):
            pe = flood.effective_precipitation(rain, duration=30)
            return (
                pe,
                flood.edi(pe, 2000, 2000, 2001),
                flood.flood_index(pe, 2000, 2000, 2001, year_start_month=1),
                flood.antecedent_precipitation_index(rain, 0.9),
            )

    monkeypatch.setattr(flood_native, "_native", None)
    python = run()
    monkeypatch.setattr(flood_native, "_native", types.ModuleType("climate_indices._native"))
    stale = run()
    for stale_item, python_item in zip(stale, python, strict=True):
        _assert_parity(stale_item, python_item)


def _time_last_moved_first(block: np.ndarray) -> np.ndarray:
    """A time-first view of a time-last array, as from ``np.moveaxis`` on an xarray block: not contiguous."""
    return np.moveaxis(np.ascontiguousarray(np.moveaxis(block, 0, -1)), -1, 0)


@pytest.mark.parametrize(
    "layout",
    [_time_last_moved_first, lambda block: np.concatenate([block, block], axis=2)[:, :, ::2], np.asfortranarray],
    ids=["time-last-moved-first", "every-other-cell", "fortran-order"],
)
def test_non_contiguous_layouts_reach_the_kernels(monkeypatch, layout: Callable[[np.ndarray], np.ndarray]) -> None:
    """Non-contiguous input reaches all four kernels and matches Python; validation makes it C-contiguous first."""
    rain = layout(_synthetic_rain((6 * 366, 3, 2), seed=13, missing=0.002))
    assert not rain.flags.c_contiguous

    def run() -> tuple[Any, ...]:
        pe = layout(flood.effective_precipitation(rain, duration=30))
        return (
            pe,
            flood.edi(pe, 2000, 2001, 2005),
            flood.flood_index(pe, 2000, 2001, 2005, year_start_month=1),
            flood.antecedent_precipitation_index(rain, 0.9),
        )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"effective_precipitation", "edi", "flood_index", "antecedent_precipitation_index"}
    for rust_item, python_item in zip(rust, python, strict=True):
        _assert_parity(rust_item, python_item)


def test_default_numpy_error_policies_keep_the_python_path(monkeypatch) -> None:
    recorder = conftest.NativeRecorder(native)
    monkeypatch.setattr(flood_native, "_native", recorder)
    rain = _synthetic_rain((2 * 366,), seed=8)
    with np.errstate(all="warn"):
        pe = flood.effective_precipitation(rain, duration=30)
        flood.edi(pe, 2000, 2000, 2001)
        flood.flood_index(pe, 2000, 2000, 2001, year_start_month=1)
        flood.antecedent_precipitation_index(rain, 0.9)
    assert recorder.calls == set()


def test_the_xarray_adapters_reach_the_kernels(monkeypatch) -> None:
    dates = pd.date_range("2000-01-01", "2004-12-31", freq="D")
    rain = xr.DataArray(
        _synthetic_rain((dates.size, 2, 3), seed=9),
        dims=("time", "lat", "lon"),
        coords={"time": dates, "lat": [10, 20], "lon": [0, 1, 2]},
        attrs={"units": "mm"},
    )

    def run() -> tuple[Any, ...]:
        pe = flood.effective_precipitation(rain, duration=30)
        return (
            pe,
            flood.edi(pe),
            flood.flood_index(pe, year_start_month=1),
            flood.antecedent_precipitation_index(rain, 0.9),
        )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"effective_precipitation", "edi", "flood_index", "antecedent_precipitation_index"}
    for rust_item, python_item in zip(rust, python, strict=True):
        _assert_parity(rust_item, python_item)
