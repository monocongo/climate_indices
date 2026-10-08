"""Parity of the Rust flood kernels with the Python reference implementations.

Each test computes the same result twice through the public ``climate_indices.flood``
API: once with ``climate_indices.flood._native._native`` replaced by a recorder
around the Rust extension, and once with it set to None, which runs the
pure-Python implementations. The recorder proves the first run reached the Rust
kernels, so the comparison is never Python against Python. The contract is
``rtol = atol = 1e-10`` with matching NaN positions, and — for the API — an
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
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from climate_indices import flood
from climate_indices.exceptions import InvalidArgumentError
from climate_indices.flood import _native as flood_native
from climate_indices.utils import transform_to_366day
from tests import conftest

native = conftest.import_native()

RTOL = 1e-10
ATOL = 1e-10

_FRESNO = Path(__file__).parent / "fixture" / "kbdi_ghcn" / "fresno_1991_2020.csv"
_FRESNO_START, _FRESNO_YEARS = 1991, 30


class _Recorder:
    """Stand-in for the extension module that records which kernels were called."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self.calls: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        self.calls.add(name)
        return getattr(self._module, name)


def _rust_and_python(monkeypatch: pytest.MonkeyPatch, run: Callable[[], Any]) -> tuple[Any, Any, set[str]]:
    recorder = _Recorder(native)
    # the Rust kernels do not implement NumPy's floating-point reporting policies
    with np.errstate(all="ignore"):
        monkeypatch.setattr(flood_native, "_native", recorder)
        rust = run()
        monkeypatch.setattr(flood_native, "_native", None)
        python = run()
    return rust, python, recorder.calls


def _assert_parity(rust: Any, python: Any) -> None:
    """Compare two runs' results at the kernel tolerance, field by field.

    A dataclass result (``APIResult``, ``APIState``) is compared field by field,
    so the returned state is covered as well as the values; arrays compare with
    ``allclose`` and matching NaN positions, while integer arrays (the gap
    counts) compare exactly.
    """
    fields = getattr(rust, "__dataclass_fields__", None)
    if fields is not None:
        for name in fields:
            _assert_parity(getattr(rust, name), getattr(python, name))
        return
    if isinstance(rust, tuple):
        assert len(rust) == len(python)
        for rust_item, python_item in zip(rust, python, strict=True):
            _assert_parity(rust_item, python_item)
        return
    if rust is None or python is None:
        assert rust is None and python is None
        return
    if isinstance(rust, xr.DataArray):
        assert isinstance(python, xr.DataArray)
        assert rust.dims == python.dims
        _assert_parity(rust.values, python.values)
        return
    rust_array = np.asarray(rust)
    python_array = np.asarray(python)
    assert rust_array.shape == python_array.shape
    assert rust_array.dtype == python_array.dtype
    if np.issubdtype(rust_array.dtype, np.integer):
        np.testing.assert_array_equal(rust_array, python_array)
    else:
        np.testing.assert_allclose(rust_array, python_array, rtol=RTOL, atol=ATOL, equal_nan=True)


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
    ("nan_policy", "max_gap_days", "split"),
    [("propagate", 0, 5000), ("bridge", 3, 4003), ("bridge", 3, 4004)],
    ids=["propagate", "bridge-inside-a-gap", "bridge-past-the-allowance"],
)
def test_api_resumed_from_its_state_is_bitwise_a_single_pass(
    monkeypatch, nan_policy: str, max_gap_days: int, split: int
) -> None:
    """A run split at ``split`` and resumed from the returned APIState equals one pass, on both paths.

    The bridge splits land inside a six-day gap, so the resumed state carries a
    nonzero trailing gap count; the second one resumes once the allowance is
    already exceeded, so the state it carries is poisoned.
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
    if max_gap_days:
        assert rust_first.state.trailing_gap_days is not None
        assert int(rust_first.state.trailing_gap_days) > 0
    np.testing.assert_array_equal(np.concatenate([rust_first.values, rust_second.values]), rust_single.values)
    np.testing.assert_array_equal(rust_second.state.api, rust_single.state.api)
    np.testing.assert_array_equal(rust_second.state.trailing_gap_days, rust_single.state.trailing_gap_days)


def test_api_overflow_raises_the_python_error_on_both_paths(monkeypatch) -> None:
    rain = np.full(3, np.finfo(np.float64).max)
    message = "antecedent_precipitation_index produced a non-finite value from finite inputs"
    recorder = _Recorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(flood_native, "_native", recorder)
        with pytest.raises(InvalidArgumentError, match=message):
            flood.antecedent_precipitation_index(rain, 0.9)
        monkeypatch.setattr(flood_native, "_native", None)
        with pytest.raises(InvalidArgumentError, match=message):
            flood.antecedent_precipitation_index(rain, 0.9)
    # the dispatch also reads the extension's NonFiniteResultError to translate it
    assert "antecedent_precipitation_index" in recorder.calls


# Dispatch policy.


def test_a_decay_constant_wider_than_float64_stays_in_python(monkeypatch) -> None:
    """An extended ``np.longdouble`` promotes the Python step beyond float64, which the kernel does not reproduce.

    Where ``long double`` is plain float64 (MSVC), NumPy treats the two as one
    type and the kernel takes it.
    """
    run = partial(flood.antecedent_precipitation_index, _synthetic_rain((50,), seed=7), np.longdouble(0.9))
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == (set() if np.dtype(np.longdouble) != np.float64 else {"antecedent_precipitation_index"})
    _assert_parity(rust, python)


def test_default_numpy_error_policies_keep_the_python_path(monkeypatch) -> None:
    recorder = _Recorder(native)
    monkeypatch.setattr(flood_native, "_native", recorder)
    rain = _synthetic_rain((2 * 366,), seed=8)
    with np.errstate(all="warn"):
        pe = flood.effective_precipitation(rain, duration=30)
        flood.edi(pe, 2000, 2000, 2001)
        flood.flood_index(pe, 2000, 2000, 2001, year_start_month=1)
        flood.antecedent_precipitation_index(rain, 0.9)
    assert recorder.calls == set()
