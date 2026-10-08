"""Parity of the Rust PET kernels with the Python reference implementations.

Each test computes the same PET twice through the public API: once with
``eto._native``/``pm_eto._native`` replaced by a recorder around the Rust
extension, and once with them set to None, which runs the pure-Python reference.
The recorder proves the first run reached the Rust kernels, so the comparison is
never Python against Python. The contract is ``rtol = atol = 1e-10`` with
matching NaN positions, for every method and pathway issue #1276 ports.

Skipped when the extension is not built (``uv run maturin develop --release``),
unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1`` is set, as in CI's native legs, where a
missing extension is a collection error.
"""

import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from climate_indices import eto, pm_eto
from climate_indices.exceptions import InvalidArgumentError
from tests import conftest

native = conftest.import_native()

RTOL = 1e-10
ATOL = 1e-10

# the FAO-56 Example 18 (page 74) intermediates, as tests/test_pm_eto.py states them
_FAO56_EXAMPLE_18 = {
    "daily_tmin_celsius": 12.3,
    "daily_tmax_celsius": 21.5,
    "latitude_degrees": 50.80,
    "elevation_m": 100.0,
    "wind_speed_m_s": 2.78,
    "day_of_year": 187,
    "wind_speed_height_m": 10.0,
    "humidity": pm_eto.HumidityInputs(rh_min=63.0, rh_max=84.0),
    "radiation": pm_eto.RadiationInputs(sunshine_hours=9.25),
}

_HUMIDITY_PATHWAYS = [
    pytest.param(pm_eto.HumidityInputs(tdew_celsius=12.05), id="dewpoint"),
    pytest.param(pm_eto.HumidityInputs(rh_min=63.0, rh_max=84.0), id="rhmin_rhmax"),
    pytest.param(pm_eto.HumidityInputs(rh_max=84.0), id="rhmax"),
    pytest.param(pm_eto.HumidityInputs(rh_mean=70.0), id="rhmean"),
    pytest.param(None, id="tmin_fallback"),
]

_RADIATION_PATHWAYS = [
    pytest.param(pm_eto.RadiationInputs(solar_radiation_mj_m2_day=22.0), id="supplied"),
    pytest.param(pm_eto.RadiationInputs(sunshine_hours=9.25), id="sunshine"),
    pytest.param(pm_eto.RadiationInputs(coastal=True), id="temperature_range_coastal"),
    pytest.param(pm_eto.RadiationInputs(), id="temperature_range_interior"),
]


class _Recorder:
    """Stand-in for the extension module that records which kernels were called."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self.calls: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        self.calls.add(name)
        return getattr(self._module, name)


def _rust_and_python(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    run: Callable[[], Any],
) -> tuple[Any, Any, set[str]]:
    """Run ``run`` against the Rust kernels, then against the Python reference."""
    recorder = _Recorder(native)
    # native kernels do not implement NumPy's floating-point reporting policies
    with np.errstate(all="ignore"):
        monkeypatch.setattr(module, "_native", recorder)
        rust = run()
        monkeypatch.setattr(module, "_native", None)
        python = run()
    return rust, python, recorder.calls


def _assert_parity(rust: Any, python: Any) -> None:
    rust = np.asarray(rust)
    python = np.asarray(python)
    assert rust.shape == python.shape
    assert rust.dtype == python.dtype
    assert np.array_equal(np.isnan(rust), np.isnan(python)), "NaN positions differ between the two paths"
    np.testing.assert_allclose(rust, python, rtol=RTOL, atol=ATOL)


def _temperatures(time_steps: int, cells: tuple[int, ...] = (), seed: int = 32) -> np.ndarray:
    """A seeded monthly temperature block: warm months, cold months, and some NaN."""
    rng = np.random.default_rng(seed)
    values = rng.normal(15.0, 8.0, size=(time_steps, *cells))
    values.reshape(-1)[::37] = np.nan
    return values


# ---------------------------------------------------------------------------
# Thornthwaite
# ---------------------------------------------------------------------------


def test_thornthwaite_matches_the_committed_fixture(
    monkeypatch, temps_celsius, latitude_degrees, data_year_start_monthly, pet_thornthwaite_mm
):
    rust, python, calls = _rust_and_python(
        monkeypatch, eto, lambda: eto.eto_thornthwaite(temps_celsius, latitude_degrees, data_year_start_monthly)
    )
    assert calls == {"thornthwaite"}
    _assert_parity(rust, python)
    # the kernel also has to reproduce the reference fixture, not just the Python path
    np.testing.assert_allclose(rust, pet_thornthwaite_mm.flatten(), atol=0.001, equal_nan=True)


def test_thornthwaite_literature_watson_parity(
    monkeypatch, thornthwaite_literature_monthly_temps_celsius, thornthwaite_literature_expected_pet_mm
):
    temps = thornthwaite_literature_monthly_temps_celsius.copy()
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_thornthwaite(temps, 43.0, 2001))
    assert calls == {"thornthwaite"}
    _assert_parity(rust, python)
    np.testing.assert_allclose(rust, thornthwaite_literature_expected_pet_mm, atol=4.0)


@pytest.mark.parametrize("start_year", [1999, 2000])
def test_thornthwaite_leap_year_start(monkeypatch, start_year):
    temps = _temperatures(24, seed=start_year)
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_thornthwaite(temps, 40.0, start_year))
    assert calls == {"thornthwaite"}
    _assert_parity(rust, python)


def test_thornthwaite_spatial_block_with_per_cell_latitude(monkeypatch):
    # a time-major block that does not end on a year boundary, with one latitude per cell
    temps = _temperatures(26, (2, 2))
    latitude = np.array([[40.0, 35.0], [30.0, 25.0]])
    rust, python, calls = _rust_and_python(
        monkeypatch, eto, lambda: eto.eto_thornthwaite(temps, latitude, 2001, spatial_time_major=True)
    )
    assert calls == {"thornthwaite"}
    assert rust.shape == (26, 2, 2)
    _assert_parity(rust, python)


def test_thornthwaite_passes_a_scalar_latitude_as_a_view(monkeypatch):
    """One latitude per cell, read from a view of the scalar rather than a block of it.

    The kernel copies what it reads, so materializing the scalar into a per-cell block
    here would allocate in proportion to the request rather than to the scalar.
    """
    temps = _temperatures(26, (2, 2))
    seen: list[tuple[np.ndarray, ...]] = []

    def capture(*arrays: np.ndarray) -> np.ndarray:
        seen.append(arrays)
        return native.thornthwaite(*arrays)

    with np.errstate(all="ignore"):
        monkeypatch.setattr(eto, "_native", SimpleNamespace(thornthwaite=capture))
        result = eto.eto_thornthwaite(temps, 35.0, 2001, spatial_time_major=True)
        monkeypatch.setattr(eto, "_native", None)
        reference = eto.eto_thornthwaite(temps, 35.0, 2001, spatial_time_major=True)

    assert len(seen) == 1
    latitude = seen[0][1]
    assert latitude.strides == (0,), "a scalar latitude was materialized into a per-cell block"
    assert latitude.base is not None and latitude.base.size == 1
    _assert_parity(result, reference)


@pytest.mark.parametrize("latitude", [90.0, -90.0])
def test_thornthwaite_polar_latitudes(monkeypatch, latitude):
    temps = _temperatures(24, seed=3)
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_thornthwaite(temps, latitude, 2001))
    assert calls == {"thornthwaite"}
    _assert_parity(rust, python)


def test_thornthwaite_all_nan_month_column_stays_on_python(monkeypatch):
    temps = np.full((2, 12), 15.0)
    temps[:, 3] = np.nan
    with warnings.catch_warnings():
        # the Python path reports np.nanmean's empty slice, which the kernel cannot
        warnings.simplefilter("ignore", RuntimeWarning)
        rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_thornthwaite(temps, 40.0, 2001))
    assert calls == set()
    _assert_parity(rust, python)


def test_thornthwaite_all_nan_month_column_warns_before_an_invalid_latitude(monkeypatch):
    # the Python path reports np.nanmean's empty slice before its daylight term
    # rejects the latitude, so the guard has to fall back before checking it
    temps = np.full((2, 12), 15.0)
    temps[:, 3] = np.nan
    with np.errstate(all="ignore"):
        monkeypatch.setattr(eto, "_native", _Recorder(native))
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            with pytest.raises(InvalidArgumentError) as native_error:
                eto.eto_thornthwaite(temps, 91.0, 2001)
        monkeypatch.setattr(eto, "_native", None)
        with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
            with pytest.raises(InvalidArgumentError) as python_error:
                eto.eto_thornthwaite(temps, 91.0, 2001)
    assert str(native_error.value) == str(python_error.value)


def test_thornthwaite_float32_and_masked_inputs_stay_on_python(monkeypatch):
    temps = _temperatures(24, seed=11)
    float32_temps = temps.astype(np.float32)
    result, reference, calls = _rust_and_python(
        monkeypatch, eto, lambda: eto.eto_thornthwaite(float32_temps, 40.0, 2001)
    )
    assert calls == set()
    _assert_parity(result, reference)
    # float32 loses precision in the Python reference as well, so the comparison
    # against the float64 reference carries the variant's own tolerance
    np.testing.assert_allclose(result, eto.eto_thornthwaite(temps, 40.0, 2001), rtol=1e-5, atol=1e-5, equal_nan=True)

    # a masked array is not a plain float64 ndarray, so it keeps the Python path,
    # and keeps whatever a mask means to the Python expressions
    masked = np.ma.masked_array(temps, mask=False)
    result, reference, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_thornthwaite(masked, 40.0, 2001))
    assert calls == set()
    _assert_parity(result, reference)


def test_thornthwaite_all_zero_temperatures(monkeypatch):
    # every temperature at zero divides by a zero heat index, as the Python path does
    temps = np.zeros(24)
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_thornthwaite(temps, 40.0, 2001))
    assert calls == {"thornthwaite"}
    _assert_parity(rust, python)


def test_thornthwaite_strided_input_matches_python(monkeypatch):
    temps = _temperatures(24, seed=5)[::-1]
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_thornthwaite(temps, 40.0, 2001))
    assert calls == {"thornthwaite"}
    _assert_parity(rust, python)


def test_thornthwaite_invalid_latitude_raises_on_both_paths(monkeypatch):
    temps = _temperatures(24, seed=13)
    for latitude in (91.0, -91.0, np.nan, None):
        with np.errstate(all="ignore"):
            monkeypatch.setattr(eto, "_native", _Recorder(native))
            with pytest.raises((InvalidArgumentError, TypeError)) as native_error:
                eto.eto_thornthwaite(temps, latitude, 2001)
            monkeypatch.setattr(eto, "_native", None)
            with pytest.raises(type(native_error.value)) as python_error:
                eto.eto_thornthwaite(temps, latitude, 2001)
        assert str(native_error.value) == str(python_error.value)


# ---------------------------------------------------------------------------
# Hargreaves
# ---------------------------------------------------------------------------


def test_hargreaves_reference_series(
    monkeypatch,
    hargreaves_daily_tmin_celsius,
    hargreaves_daily_tmax_celsius,
    hargreaves_daily_tmean_celsius,
    hargreaves_latitude_degrees,
):
    run = lambda: eto.eto_hargreaves(  # noqa: E731
        hargreaves_daily_tmin_celsius,
        hargreaves_daily_tmax_celsius,
        hargreaves_daily_tmean_celsius,
        hargreaves_latitude_degrees,
    )
    rust, python, calls = _rust_and_python(monkeypatch, eto, run)
    assert calls == {"hargreaves"}
    _assert_parity(rust, python)


def test_hargreaves_partial_year_block_with_per_cell_latitude(monkeypatch):
    # 900 days is two whole 366-day years plus a partial one; a block is three
    # dimensions or more, with one value and one latitude per cell
    days = 900
    rng = np.random.default_rng(17)
    tmin = rng.normal(10.0, 4.0, size=(days, 2, 2))
    tmax = tmin + 12.0
    tmean = (tmin + tmax) / 2.0
    latitude = np.array([[35.0, -35.0], [55.0, -55.0]])
    run = lambda: eto.eto_hargreaves(  # noqa: E731
        tmin, tmax, tmean, latitude, spatial_time_major=True
    )
    rust, python, calls = _rust_and_python(monkeypatch, eto, run)
    assert calls == {"hargreaves"}
    assert rust.shape == (days, 2, 2)
    _assert_parity(rust, python)


@pytest.mark.parametrize("latitude", [90.0, -90.0, 0.0])
def test_hargreaves_polar_and_equatorial_latitudes(monkeypatch, latitude):
    rng = np.random.default_rng(19)
    tmin = rng.normal(5.0, 6.0, size=732)
    tmax = tmin + 10.0
    tmean = (tmin + tmax) / 2.0
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_hargreaves(tmin, tmax, tmean, latitude))
    assert calls == {"hargreaves"}
    _assert_parity(rust, python)


def test_hargreaves_literature_mehta_parity(
    monkeypatch, hargreaves_literature_tmin_tmax_tmean_celsius, hargreaves_literature_expected_eto_mm_per_day
):
    tmin, tmax, tmean = hargreaves_literature_tmin_tmax_tmean_celsius
    daily_tmin = np.full(366, tmin)
    daily_tmax = np.full(366, tmax)
    daily_tmean = np.full(366, tmean)
    rust, python, calls = _rust_and_python(
        monkeypatch, eto, lambda: eto.eto_hargreaves(daily_tmin, daily_tmax, daily_tmean, -12.3)
    )
    assert calls == {"hargreaves"}
    _assert_parity(rust, python)
    np.testing.assert_allclose(rust[57], hargreaves_literature_expected_eto_mm_per_day[0], atol=0.05)


def test_hargreaves_nan_temperatures(monkeypatch):
    rng = np.random.default_rng(23)
    tmin = rng.normal(8.0, 5.0, size=732)
    tmax = tmin + 9.0
    tmean = (tmin + tmax) / 2.0
    tmin[::53] = np.nan
    tmean[1::97] = np.nan
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_hargreaves(tmin, tmax, tmean, 35.0))
    assert calls == {"hargreaves"}
    _assert_parity(rust, python)


def test_hargreaves_inverted_and_equal_temperatures(monkeypatch):
    # tmin > tmax makes the square root's argument negative, and an equal pair makes
    # it zero; both have to land on the same values as the Python path
    rng = np.random.default_rng(37)
    tmin = rng.normal(15.0, 4.0, size=732)
    tmax = tmin - 5.0
    tmean = (tmin + tmax) / 2.0
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_hargreaves(tmin, tmax, tmean, 35.0))
    assert calls == {"hargreaves"}
    _assert_parity(rust, python)

    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_hargreaves(tmin, tmin, tmin, 35.0))
    assert calls == {"hargreaves"}
    _assert_parity(rust, python)


def test_hargreaves_float32_inputs_stay_on_python(monkeypatch):
    rng = np.random.default_rng(29)
    tmin = rng.normal(8.0, 5.0, size=732)
    tmax = tmin + 9.0
    tmean = (tmin + tmax) / 2.0
    reference = eto.eto_hargreaves(tmin, tmax, tmean, 35.0)
    recorder = _Recorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(eto, "_native", recorder)
        result = eto.eto_hargreaves(tmin.astype(np.float32), tmax.astype(np.float32), tmean.astype(np.float32), 35.0)
    assert recorder.calls == set()
    np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_hargreaves_mismatched_block_shapes_stay_on_python(monkeypatch):
    # equal sizes in different layouts: the kernel would pair the flattened cells,
    # while the Python path fails to assign the broadcast result into its output
    tmean = np.full((5, 2, 2), 15.0)
    tmin = np.full((1, 5, 2, 2), 10.0)
    recorder = _Recorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(eto, "_native", recorder)
        with pytest.raises(ValueError) as native_error:
            eto.eto_hargreaves(tmin, tmean + 5.0, tmean, 35.0, spatial_time_major=True)
        monkeypatch.setattr(eto, "_native", None)
        with pytest.raises(ValueError) as python_error:
            eto.eto_hargreaves(tmin, tmean + 5.0, tmean, 35.0, spatial_time_major=True)
    assert recorder.calls == set()
    assert str(native_error.value) == str(python_error.value)


def test_masked_input_with_a_padded_length_reaches_the_kernel(monkeypatch):
    """A padded masked input is de-masked before either path computes.

    The 1-D/2-D path pads through ``utils.reshape_to_2d``, whose ``np.pad`` drops the
    MaskedArray subclass, so both paths operate on the same plain data: the kernel
    is dispatched, and the Python path ignores the mask in exactly the same way.
    Only a whole-year masked input, which padding never touches, keeps its mask and
    therefore stays on the Python path.
    """
    days = 900  # not a whole number of 366-day years, so the input is padded
    rng = np.random.default_rng(41)
    tmin = rng.normal(8.0, 5.0, size=days)
    tmax = tmin + 10.0
    tmean = (tmin + tmax) / 2.0
    masked = np.ma.masked_array(tmin, mask=np.zeros(days, dtype=bool))
    masked.mask[:60] = True
    rust, python, calls = _rust_and_python(monkeypatch, eto, lambda: eto.eto_hargreaves(masked, tmax, tmean, 35.0))
    assert calls == {"hargreaves"}
    _assert_parity(rust, python)
    np.testing.assert_array_equal(python, eto.eto_hargreaves(np.asarray(masked), tmax, tmean, 35.0))


def test_hargreaves_passes_a_scalar_latitude_as_a_view(monkeypatch):
    """One latitude per cell, read from a view of the scalar rather than a block of it.

    The kernel copies what it reads, so materializing the scalar into a per-cell block
    here would allocate in proportion to the request rather than to the scalar.
    """
    rng = np.random.default_rng(43)
    tmin = rng.normal(8.0, 5.0, size=(732, 2, 2))
    tmax = tmin + 9.0
    tmean = (tmin + tmax) / 2.0
    seen: list[tuple[np.ndarray, ...]] = []

    def capture(*arrays: np.ndarray) -> np.ndarray:
        seen.append(arrays)
        return native.hargreaves(*arrays)

    with np.errstate(all="ignore"):
        monkeypatch.setattr(eto, "_native", SimpleNamespace(hargreaves=capture))
        result = eto.eto_hargreaves(tmin, tmax, tmean, 35.0, spatial_time_major=True)
        monkeypatch.setattr(eto, "_native", None)
        reference = eto.eto_hargreaves(tmin, tmax, tmean, 35.0, spatial_time_major=True)

    assert len(seen) == 1
    latitude = seen[0][3]
    assert latitude.strides == (0,), "a scalar latitude was materialized into a per-cell block"
    assert latitude.base is not None and latitude.base.size == 1
    _assert_parity(result, reference)


@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided_blocks"])
def test_hargreaves_reports_the_bytes_the_native_route_holds(monkeypatch, strided):
    """The recorded footprint covers the copies the kernel route makes.

    The kernel copies each block and the latitude it reads before it releases the GIL,
    and a block that does not lie contiguously is flattened here as well, so the route's
    own bytes are reported beside the arrays the caller holds rather than left out.
    """
    days = 3
    rng = np.random.default_rng(47)
    # (days, cells, 3) so every block is a strided view of the array that holds them
    values = rng.normal(8.0, 5.0, size=(days, 2, 3, 3))
    values[..., 1] = values[..., 0] + 9.0
    values[..., 2] = (values[..., 0] + values[..., 1]) / 2.0
    if strided:
        tmin, tmax, tmean = (values[..., index] for index in range(3))
    else:
        tmin, tmax, tmean = (values[..., index].copy() for index in range(3))

    block_bytes = days * 2 * 3 * np.dtype(np.float64).itemsize
    copies = 3 * block_bytes + 2 * 3 * np.dtype(np.float64).itemsize
    expected = copies + (3 * block_bytes if strided else 0)
    recorded: list[int] = []

    def capture(*arrays: np.ndarray, extra_bytes: int = 0) -> None:
        recorded.append(extra_bytes)
        return None

    monkeypatch.setattr(eto, "check_large_array_memory", capture)
    with np.errstate(all="ignore"):
        result = eto.eto_hargreaves(tmin, tmax, tmean, 35.0, spatial_time_major=True)
    assert recorded == [expected]
    assert result.shape == (days, 2, 3)

    # the Python path holds no native copies to report
    monkeypatch.setattr(eto, "_native", None)
    with np.errstate(all="ignore"):
        fallback = eto.eto_hargreaves(tmin, tmax, tmean, 35.0, spatial_time_major=True)
    assert recorded == [expected, 0]
    _assert_parity(result, fallback)


def test_hargreaves_invalid_latitude_raises_on_both_paths(monkeypatch):
    tmin = np.full(366, 10.0)
    tmax = np.full(366, 25.0)
    for latitude in (91.0, -91.0, np.nan):
        with np.errstate(all="ignore"):
            monkeypatch.setattr(eto, "_native", _Recorder(native))
            with pytest.raises(InvalidArgumentError, match="Latitude outside valid range") as native_error:
                eto.eto_hargreaves(tmin, tmax, tmin, latitude)
            monkeypatch.setattr(eto, "_native", None)
            with pytest.raises(InvalidArgumentError, match="Latitude outside valid range") as python_error:
                eto.eto_hargreaves(tmin, tmax, tmin, latitude)
        assert str(native_error.value) == str(python_error.value)


# ---------------------------------------------------------------------------
# Penman-Monteith
# ---------------------------------------------------------------------------


def test_pm_eto_core_parity(monkeypatch):
    arguments = (
        np.array([13.28, 12.5, 14.0]),
        np.array([0.14, 0.0, 0.3]),
        np.array([16.9, 20.0, 5.0]),
        np.array([2.078, 1.5, 3.0]),
        np.array([1.997, 2.3, 0.9]),
        np.array([1.409, 1.1, 0.7]),
        np.array([0.122, 0.145, 0.06]),
        np.array([0.0666, 0.067, 0.05]),
    )
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.pm_eto(*arguments))
    assert calls == {"pm_eto"}
    _assert_parity(rust, python)


def test_all_scalar_pm_eto_stays_on_python(monkeypatch):
    scalars = (13.28, 0.14, 16.9, 2.078, 1.997, 1.409, 0.122, 0.0666)
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.pm_eto(*scalars))
    assert calls == set()
    _assert_parity(rust, python)
    assert isinstance(rust, np.float64)


def test_pm_eto_float32_inputs_stay_on_python(monkeypatch):
    arguments = tuple(
        np.array([value, value], dtype=np.float32) for value in (13.28, 0.14, 16.9, 2.078, 1.997, 1.409, 0.122, 0.0666)
    )
    reference = pm_eto.pm_eto(*(array.astype(np.float64) for array in arguments))
    recorder = _Recorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(pm_eto, "_native", recorder)
        result = pm_eto.pm_eto(*arguments)
    assert recorder.calls == set()
    np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-6)


def test_pm_eto_passes_a_scalar_operand_as_a_view(monkeypatch):
    """A scalar operand reaches the kernel as a zero-stride view, not a full array.

    The kernel reads one element per broadcast position and the binding copies what it
    reads, so materializing the scalar here would allocate a full-size array, and copy
    it, for a value that occupies one element.
    """
    net_radiation = np.array([13.28, 12.5, 14.0])
    scalars = (0.14, 16.9, 2.078, 1.997, 1.409, 0.122, 0.0666)
    seen: list[tuple[np.ndarray, ...]] = []

    def capture(*arrays: np.ndarray) -> np.ndarray:
        seen.append(arrays)
        return native.pm_eto(*arrays)

    with np.errstate(all="ignore"):
        monkeypatch.setattr(pm_eto, "_native", SimpleNamespace(pm_eto=capture))
        result = pm_eto.pm_eto(net_radiation, *scalars)
        monkeypatch.setattr(pm_eto, "_native", None)
        reference = pm_eto.pm_eto(net_radiation, *scalars)

    assert len(seen) == 1
    for operand in seen[0][1:]:
        assert operand.strides == (0,), "a scalar operand was materialized into a full-size array"
        assert operand.base is not None and operand.base.size == 1
    _assert_parity(result, reference)


def test_penman_monteith_eto_matches_fao56_example_18(monkeypatch):
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.penman_monteith_eto(**_FAO56_EXAMPLE_18))
    # scalars alone keep the Python path, so the example is checked against it directly
    assert calls == set()
    _assert_parity(rust, python)
    assert rust == pytest.approx(3.88, abs=0.05)


def _example_18_arrays(length: int = 3) -> dict[str, Any]:
    """The Example 18 inputs as arrays, which is what the native dispatch takes."""
    rng = np.random.default_rng(31)
    jitter = rng.normal(0.0, 0.5, size=length)
    arrays = {
        key: np.full(length, value) + jitter
        for key, value in _FAO56_EXAMPLE_18.items()
        if isinstance(value, (int, float))
    }
    arrays["humidity"] = _FAO56_EXAMPLE_18["humidity"]
    arrays["radiation"] = _FAO56_EXAMPLE_18["radiation"]
    return arrays


@pytest.mark.parametrize("humidity", _HUMIDITY_PATHWAYS)
def test_penman_monteith_eto_humidity_pathways(monkeypatch, humidity):
    arguments = _example_18_arrays()
    arguments["humidity"] = humidity
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.penman_monteith_eto(**arguments))
    assert calls == {"fao56_eto"}
    _assert_parity(rust, python)


@pytest.mark.parametrize("radiation", _RADIATION_PATHWAYS)
def test_penman_monteith_eto_radiation_pathways(monkeypatch, radiation):
    arguments = _example_18_arrays()
    arguments["radiation"] = radiation
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.penman_monteith_eto(**arguments))
    assert calls == {"fao56_eto"}
    _assert_parity(rust, python)


def test_penman_monteith_eto_broadcasts_scalars_against_arrays(monkeypatch):
    # a scalar pathway input against array meteorology still reaches the kernel
    arguments = _example_18_arrays()
    arguments["elevation_m"] = 100.0
    arguments["humidity"] = pm_eto.HumidityInputs(rh_max=84.0)
    arguments["radiation"] = None
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.penman_monteith_eto(**arguments))
    assert calls == {"fao56_eto"}
    _assert_parity(rust, python)


def test_penman_monteith_eto_nan_inputs(monkeypatch):
    arguments = _example_18_arrays()
    arguments["daily_tmin_celsius"] = np.array([12.3, np.nan, 12.3])
    arguments["wind_speed_m_s"] = np.array([2.78, 2.78, np.nan])
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.penman_monteith_eto(**arguments))
    assert calls == {"fao56_eto"}
    _assert_parity(rust, python)


def test_penman_monteith_eto_float32_inputs_stay_on_python(monkeypatch):
    arguments = _example_18_arrays()
    arguments = {
        key: (value.astype(np.float32) if isinstance(value, np.ndarray) else value) for key, value in arguments.items()
    }
    reference = pm_eto.penman_monteith_eto(**_example_18_arrays())
    recorder = _Recorder(native)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(pm_eto, "_native", recorder)
        result = pm_eto.penman_monteith_eto(**arguments)
    assert recorder.calls == set()
    np.testing.assert_allclose(result, reference, rtol=1e-5, atol=1e-5, equal_nan=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"humidity": pm_eto.HumidityInputs(rh_min=63.0)}, "rh_min was provided without rh_max"),
        ({"wind_speed_height_m": 0.0}, "Wind measurement height must be positive"),
    ],
)
def test_penman_monteith_eto_errors_match_on_both_paths(monkeypatch, overrides, message):
    arguments = _example_18_arrays()
    arguments.update(overrides)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(pm_eto, "_native", _Recorder(native))
        with pytest.raises(InvalidArgumentError, match=message) as native_error:
            pm_eto.penman_monteith_eto(**arguments)
        monkeypatch.setattr(pm_eto, "_native", None)
        with pytest.raises(InvalidArgumentError, match=message) as python_error:
            pm_eto.penman_monteith_eto(**arguments)
    assert str(native_error.value) == str(python_error.value)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"latitude_degrees": "north", "wind_speed_height_m": 0.0}, id="latitude_before_wind_height"),
        pytest.param(
            {"wind_speed_m_s": "calm", "humidity": pm_eto.HumidityInputs(rh_min=63.0)},
            id="wind_speed_before_humidity_pathway",
        ),
    ],
)
def test_penman_monteith_eto_conversion_errors_precede_native_checks(monkeypatch, overrides):
    # an operand the kernel cannot take keeps the Python path before the native
    # checks run, so both paths raise the Python path's first error
    arguments = _example_18_arrays()
    arguments.update(overrides)
    with np.errstate(all="ignore"):
        monkeypatch.setattr(pm_eto, "_native", _Recorder(native))
        with pytest.raises(TypeError) as native_error:
            pm_eto.penman_monteith_eto(**arguments)
        monkeypatch.setattr(pm_eto, "_native", None)
        with pytest.raises(TypeError) as python_error:
            pm_eto.penman_monteith_eto(**arguments)
    assert str(native_error.value) == str(python_error.value)


def test_penman_monteith_eto_polar_night_parity(monkeypatch):
    # at either pole a winter day has no daylight, so the sunshine pathway divides
    # zero by zero and the temperature-range estimate is capped by a zero clear-sky
    # value; the two paths have to agree on where that becomes NaN or infinite
    arguments = _example_18_arrays()
    arguments["latitude_degrees"] = np.array([90.0, -90.0, 0.0])
    for radiation in (
        pm_eto.RadiationInputs(sunshine_hours=np.zeros(3)),
        pm_eto.RadiationInputs(),
        pm_eto.RadiationInputs(coastal=True),
    ):
        arguments["radiation"] = radiation
        rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.penman_monteith_eto(**arguments))
        assert calls == {"fao56_eto"}
        _assert_parity(rust, python)


def test_penman_monteith_eto_broadcast_shapes_parity(monkeypatch):
    # (3, 1) temperatures against (1, 2) temperatures, with scalar pathway inputs
    arguments = _example_18_arrays()
    arguments["daily_tmin_celsius"] = np.full((3, 1), 12.3)
    arguments["daily_tmax_celsius"] = np.full((1, 2), 21.5)
    for key in ("latitude_degrees", "elevation_m", "wind_speed_m_s", "wind_speed_height_m", "day_of_year"):
        arguments[key] = np.asarray(arguments[key])[:1]
    arguments["humidity"] = pm_eto.HumidityInputs(rh_max=84.0)
    arguments["radiation"] = pm_eto.RadiationInputs()
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.penman_monteith_eto(**arguments))
    assert calls == {"fao56_eto"}
    assert rust.shape == (3, 2)
    _assert_parity(rust, python)


def test_penman_monteith_eto_read_only_inputs(monkeypatch):
    # a read-only block is still a plain aligned float64 array, so it dispatches and
    # must come back unchanged
    arguments = _example_18_arrays()
    arguments["daily_tmin_celsius"].flags.writeable = False
    before = arguments["daily_tmin_celsius"].copy()
    rust, python, calls = _rust_and_python(monkeypatch, pm_eto, lambda: pm_eto.penman_monteith_eto(**arguments))
    assert calls == {"fao56_eto"}
    _assert_parity(rust, python)
    np.testing.assert_array_equal(arguments["daily_tmin_celsius"], before)


def test_penman_monteith_eto_default_error_policy_stays_on_python(monkeypatch):
    # the dispatch requires NumPy's floating-point errors to be ignored
    arguments = _example_18_arrays()
    recorder = _Recorder(native)
    monkeypatch.setattr(pm_eto, "_native", recorder)
    rust = pm_eto.penman_monteith_eto(**arguments)
    assert recorder.calls == set()
    monkeypatch.setattr(pm_eto, "_native", None)
    python = pm_eto.penman_monteith_eto(**arguments)
    _assert_parity(rust, python)
