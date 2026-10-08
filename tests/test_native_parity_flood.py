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
