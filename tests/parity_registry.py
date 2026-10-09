"""One parity registry and harness shared by every ported Rust kernel.

Each entry names a Python entry point, the extension module its dispatch lives
behind, the ``climate-core`` kernels that entry must reach, and the input family
its values are drawn from. ``run_parity`` computes the same entry twice -- once
through the Rust extension behind a recorder, once with the dispatch module's
``_native`` set to None -- and compares the outcomes at the entry's tolerance,
so a test can never pass by comparing Python against Python.

The input families are Hypothesis strategies rather than fixed arrays: a draw
varies the series length, the NaN pattern, the zero runs, an injected extreme
magnitude, and the number of spatial cells. ``SAMPLES`` holds one fixed array per
family for the deterministic case and for ``scripts/native_parity_maxima.py``,
which reports each entry's largest deviation for the tolerance table in
``docs/architecture.md``.

The expected kernels are asserted as a subset, not an equality: an entry may
reach another ported kernel on its way (a flood entry that prepares its input
through ``flood.effective_precipitation``), and the coverage test in
``tests/test_native_parity_registry.py`` asserts the registry's union against the
extension's own function list, so a new kernel cannot be added without a entry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import xarray as xr
from hypothesis import strategies as st

from climate_indices import compute, eto, fire, flood, indices, palmer, pm_eto
from climate_indices.fire import _native as fire_native
from climate_indices.flood import _native as flood_native
from tests import conftest

RTOL = 1e-10
ATOL = 1e-10

# The record every fitting entry is calibrated from: a series that starts in this
# year, a window two years in so the fit has a warm-up, and one that ends a year
# before the record does.
DATA_START_YEAR = 1981

# The extension's non-kernel attributes; every other callable it exposes is a kernel.
NON_KERNELS = frozenset({"__version__", "NonFiniteResultError", "NoConvergenceError"})

MONTHLY = "monthly"
DAILY = "daily"
BLOCK = "block"


@dataclass(frozen=True)
class _Error:
    """A raised exception, captured so both paths can be compared as outcomes."""

    error: type[BaseException]
    message: str


@dataclass(frozen=True)
class Entry:
    """One ported kernel surface: its dispatch module, its run, and its contract."""

    name: str
    family: str
    dispatch: ModuleType
    run: Callable[[np.ndarray], Callable[[], Any]]
    kernels: frozenset[str]
    rtol: float = RTOL
    atol: float = ATOL


@dataclass(frozen=True)
class ParityRun:
    """The two outcomes of one entry, the kernels the Rust run reached, and its error."""

    entry: Entry
    rust: Any
    python: Any
    calls: frozenset[str]
    max_absolute: float
    max_relative: float


# ---------------------------------------------------------------------------
# Input families
# ---------------------------------------------------------------------------


def _monthly_years(years: int) -> int:
    return years * 12


def _daily_years(years: int) -> int:
    return years * 365


def _precipitation(rng: np.random.Generator, size: int, zero_fraction: float) -> np.ndarray:
    values = rng.gamma(2.0, 40.0, size=size)
    values[rng.random(size) < zero_fraction] = 0.0
    return values


def _monthly_values(years: int, seed: int, zero_fraction: float, missing_fraction: float, extreme: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = _precipitation(rng, _monthly_years(years), zero_fraction)
    values[rng.random(values.size) < missing_fraction] = np.nan
    if not np.isnan(extreme):
        values[rng.integers(0, values.size)] = extreme
    return values


def _daily_values(years: int, seed: int, zero_fraction: float, missing_fraction: float, extreme: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.gamma(1.5, 8.0, size=_daily_years(years))
    values[rng.random(values.size) < zero_fraction] = 0.0
    values[rng.random(values.size) < missing_fraction] = np.nan
    if not np.isnan(extreme):
        values[rng.integers(0, values.size)] = extreme
    return values


def _block_values(years: int, cells: int, seed: int, zero_fraction: float, missing_fraction: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = _precipitation(rng, _monthly_years(years) * cells, zero_fraction).reshape(_monthly_years(years), cells)
    values[rng.random(values.shape) < missing_fraction] = np.nan
    return values


@st.composite
def monthly(draw: st.DrawFn) -> np.ndarray:
    """A monthly series long enough for a 30-year calibration window."""
    return _monthly_values(
        years=draw(st.integers(min_value=34, max_value=38)),
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        zero_fraction=draw(st.floats(min_value=0.0, max_value=0.4)),
        missing_fraction=draw(st.floats(min_value=0.0, max_value=0.1)),
        extreme=draw(st.sampled_from([1.0, 1e3, 1e6, np.nan])),
    )


@st.composite
def daily(draw: st.DrawFn) -> np.ndarray:
    """A whole number of 365-day years of daily values, long enough for a two-year flood calibration."""
    return _daily_values(
        years=draw(st.integers(min_value=5, max_value=7)),
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        zero_fraction=draw(st.floats(min_value=0.0, max_value=0.5)),
        missing_fraction=draw(st.floats(min_value=0.0, max_value=0.1)),
        extreme=draw(st.sampled_from([1.0, 1e3, 1e6, np.nan])),
    )


@st.composite
def block(draw: st.DrawFn) -> np.ndarray:
    """A time-major Spatial Block: monthly time on the first axis, three cells on the second."""
    return _block_values(
        years=draw(st.integers(min_value=34, max_value=38)),
        cells=draw(st.integers(min_value=2, max_value=4)),
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        zero_fraction=draw(st.floats(min_value=0.0, max_value=0.3)),
        missing_fraction=draw(st.floats(min_value=0.0, max_value=0.1)),
    )


STRATEGIES: dict[str, st.SearchStrategy[np.ndarray]] = {MONTHLY: monthly(), DAILY: daily(), BLOCK: block()}

SAMPLES: dict[str, np.ndarray] = {
    MONTHLY: _monthly_values(35, 20261009, 0.2, 0.02, 1e3),
    DAILY: _daily_values(5, 20261010, 0.3, 0.02, 1e3),
    BLOCK: _block_values(35, 3, 20261011, 0.15, 0.02),
}


# ---------------------------------------------------------------------------
# Derivations: every entry reads one drawn array per family
# ---------------------------------------------------------------------------


def _window(length: int, period_length: int = 12) -> tuple[int, int, int]:
    """The ``(data_start_year, calibration_start, calibration_end)`` window of a drawn series."""
    years = length // period_length
    return DATA_START_YEAR, DATA_START_YEAR + 2, DATA_START_YEAR + years - 1


def _monthly_window(values: np.ndarray) -> tuple[int, int, int]:
    return _window(values.size, 12)


def _daily_window(values: np.ndarray) -> tuple[int, int, int]:
    return _window(values.size, 365)


def _pet(values: np.ndarray) -> np.ndarray:
    """A PET series in the same shape as the drawn precipitation it accompanies."""
    return 0.4 * values + 25.0


def _temperatures(values: np.ndarray) -> np.ndarray:
    """Monthly mean temperatures from a drawn precipitation series."""
    return values / 12.0 - 10.0


def _daily_temperature(values: np.ndarray) -> np.ndarray:
    """Daily mean temperatures from a drawn series."""
    return values / 2.0 - 5.0


def _relative_humidity(values: np.ndarray) -> np.ndarray:
    return 20.0 + np.mod(np.nan_to_num(values), 60.0)


def _wind_speed(values: np.ndarray) -> np.ndarray:
    return np.mod(np.nan_to_num(values), 8.0) / 2.0


def _months(values: np.ndarray) -> np.ndarray:
    return np.tile(np.arange(1, 13), values.size // 12 + 1)[: values.size]


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

_GAMMA_KERNELS = frozenset({"gamma_parameters", "gamma_probabilities", "norm_ppf"})
# The L-moment fits feed the kernel's own CDF; the normal transform of SPI's Pearson
# and GLO surfaces is the Python inverse normal, so `norm_ppf` is not among them.
_PEARSON_KERNELS = frozenset({"pearson_parameters", "pearson_cdf"})
_GLO_KERNELS = frozenset({"loglogistic_parameters", "loglogistic_cdf"})
_PALMER_WATER_BALANCE = frozenset({"palmer_water_balance", "palmer_k_prime", "palmer_raw_zindex"})


def _spi(values: np.ndarray, distribution: indices.Distribution, **kwargs: Any) -> Callable[[], Any]:
    start, calibration_start, calibration_end = _monthly_window(values)
    spatial = values.ndim > 1
    return lambda: indices.spi(
        values,
        6,
        distribution,
        start,
        calibration_start,
        calibration_end,
        compute.Periodicity.monthly,
        spatial_time_major=spatial,
        **kwargs,
    )


def _spei(values: np.ndarray, distribution: indices.Distribution = indices.Distribution.gamma) -> Callable[[], Any]:
    start, calibration_start, calibration_end = _monthly_window(values)
    return lambda: indices.spei(
        values,
        _pet(values),
        6,
        distribution,
        compute.Periodicity.monthly,
        start,
        calibration_start,
        calibration_end,
    )


def _pnp(values: np.ndarray) -> Callable[[], Any]:
    start, calibration_start, calibration_end = _monthly_window(values)
    return lambda: indices.percentage_of_normal(
        values, 6, start, calibration_start, calibration_end, compute.Periodicity.monthly
    )


def _eddi(values: np.ndarray) -> Callable[[], Any]:
    start, calibration_start, calibration_end = _monthly_window(values)
    return lambda: indices.eddi(
        _pet(values),
        6,
        start,
        calibration_start,
        calibration_end,
        compute.Periodicity.monthly,
        spatial_time_major=values.ndim > 1,
    )


def _pci(values: np.ndarray) -> Callable[[], Any]:
    # a calendar year of rain: the only lengths the entry point accepts, and no NaN
    return lambda: indices.pci(np.nan_to_num(values[:365]))


def _fit_diagnostics(values: np.ndarray) -> Callable[[], Any]:
    start, calibration_start, calibration_end = _monthly_window(values)
    return lambda: compute.fit_diagnostics(
        values, indices.Distribution.gamma, start, calibration_start, calibration_end, compute.Periodicity.monthly
    )


def _thornthwaite(values: np.ndarray) -> Callable[[], Any]:
    start, _, _ = _monthly_window(values)
    return lambda: eto.eto_thornthwaite(_temperatures(values), 40.0, start)


def _hargreaves(values: np.ndarray) -> Callable[[], Any]:
    temperature = _daily_temperature(values)
    return lambda: eto.eto_hargreaves(temperature - 6.0, temperature + 6.0, temperature, 40.0)


def _penman_monteith(values: np.ndarray) -> Callable[[], Any]:
    temperature = _daily_temperature(values)
    return lambda: pm_eto.penman_monteith_eto(
        temperature - 6.0,
        temperature + 6.0,
        40.0,
        100.0,
        2.5,
        # a float64 day of year is what the extension takes; an integer one keeps the Python path
        np.arange(values.size, dtype=float) + 1.0,
        10.0,
        pm_eto.HumidityInputs(rh_min=40.0, rh_max=80.0),
        pm_eto.RadiationInputs(sunshine_hours=9.0),
    )


def _pm_eto_intermediates(values: np.ndarray) -> Callable[[], Any]:
    """The FAO-56 equation itself, over the intermediate chain a caller supplies."""
    net_radiation = 12.0 + np.mod(values, 8.0)
    temperature = _daily_temperature(values)
    wind_2m = 1.5 + np.mod(values, 4.0) / 2.0
    saturation_vp = 1.0 + np.mod(values, 3.0)
    delta = 0.05 + np.mod(values, 2.0) / 10.0
    gamma = 0.06 + np.mod(values, 1.0) / 20.0
    return lambda: pm_eto.pm_eto(
        net_radiation,
        np.zeros(values.size),
        temperature,
        wind_2m,
        saturation_vp,
        saturation_vp - 0.5,
        delta,
        gamma,
    )


def _fire_runs(values: np.ndarray) -> dict[str, Callable[[], Any]]:
    temperature = _daily_temperature(values)
    humidity = _relative_humidity(values)
    wind = _wind_speed(values)
    precipitation = np.mod(np.nan_to_num(values), 20.0)
    months = _months(values)
    return {
        "ffmc": lambda: fire.ffmc(temperature, humidity, wind, precipitation),
        "duff_moisture_code": lambda: fire.duff_moisture_code(temperature, humidity, precipitation, 40.0, months),
        "drought_code": lambda: fire.drought_code(temperature, precipitation, 40.0, months),
        "kbdi": lambda: fire.kbdi(precipitation, temperature, float(np.nansum(precipitation))),
    }


def _flood_precipitation(values: np.ndarray) -> Callable[[], Any]:
    return lambda: flood.effective_precipitation(values, duration=60)


def _flood_runs(values: np.ndarray) -> dict[str, Callable[[], Any]]:
    years = values.size // 365
    # the calibration window has to sit inside the record, span at least two complete
    # annual periods, and leave the first year to the PE warm-up
    start, calibration_start, calibration_end = DATA_START_YEAR, DATA_START_YEAR + 2, DATA_START_YEAR + years - 2
    return {
        "pe": _flood_precipitation(values),
        "edi": lambda: flood.edi(
            flood.effective_precipitation(values, duration=60), start, calibration_start, calibration_end, duration=60
        ),
        "flood_index": lambda: flood.flood_index(
            flood.effective_precipitation(values, duration=60),
            start,
            calibration_start,
            calibration_end,
            year_start_month=1,
        ),
        "api": lambda: flood.antecedent_precipitation_index(values, 0.9, spin_up=365),
    }


def _palmer_precipitation(values: np.ndarray) -> np.ndarray:
    """Monthly precipitation with a positive floor: the scPDSI calibration needs neither a zero nor a gap."""
    return np.clip(np.nan_to_num(values, nan=0.0), 5.0, 300.0)


def _palmer_pet(values: np.ndarray) -> np.ndarray:
    return 100.0 + 0.1 * _palmer_precipitation(values)


def _pdsi(values: np.ndarray) -> Callable[[], Any]:
    start, calibration_start, calibration_end = _monthly_window(values)
    precipitation = _palmer_precipitation(values)
    return lambda: palmer.pdsi(precipitation, _palmer_pet(values), 4.0, start, calibration_start, calibration_end)


def _scpdsi(values: np.ndarray) -> Callable[[], Any]:
    start, calibration_start, calibration_end = _monthly_window(values)
    precipitation = _palmer_precipitation(values)
    return lambda: palmer.scpdsi(precipitation, _palmer_pet(values), 4.0, start, calibration_start, calibration_end)


def _fire_entries() -> Iterator[Entry]:
    sample = SAMPLES[DAILY]
    for name in _fire_runs(sample):
        yield Entry(
            name=f"fire_{name}",
            family=DAILY,
            dispatch=fire_native,
            run=lambda values, key=name: _fire_runs(values)[key],
            kernels=frozenset({name}),
        )


def _flood_entries() -> Iterator[Entry]:
    sample = SAMPLES[DAILY]
    kernels = {
        "pe": frozenset({"effective_precipitation"}),
        "edi": frozenset({"effective_precipitation", "edi"}),
        "flood_index": frozenset({"effective_precipitation", "flood_index"}),
        "api": frozenset({"antecedent_precipitation_index"}),
    }
    for name in _flood_runs(sample):
        yield Entry(
            name=f"flood_{name}",
            family=DAILY,
            dispatch=flood_native,
            run=lambda values, key=name: _flood_runs(values)[key],
            kernels=kernels[name],
        )


ENTRIES: tuple[Entry, ...] = (
    Entry("spi_gamma", MONTHLY, compute, lambda values: _spi(values, indices.Distribution.gamma), _GAMMA_KERNELS),
    Entry(
        "spi_gamma_mean_zero",
        MONTHLY,
        compute,
        lambda values: _spi(values, indices.Distribution.gamma, zero_handling="mean_zero"),
        _GAMMA_KERNELS,
    ),
    Entry("spi_pearson", MONTHLY, compute, lambda values: _spi(values, indices.Distribution.pearson), _PEARSON_KERNELS),
    Entry(
        "spei_loglogistic",
        MONTHLY,
        compute,
        lambda values: _spei(values, indices.Distribution.loglogistic),
        _GLO_KERNELS,
    ),
    Entry(
        "spi_gamma_spatial_block",
        BLOCK,
        compute,
        lambda values: _spi(values, indices.Distribution.gamma),
        _GAMMA_KERNELS,
    ),
    Entry("spei_gamma", MONTHLY, compute, _spei, _GAMMA_KERNELS),
    Entry("percentage_of_normal", MONTHLY, compute, _pnp, frozenset({"pnp_normals", "pnp_percentages"})),
    Entry("eddi", MONTHLY, compute, _eddi, frozenset({"tukey_probabilities", "hastings_inverse_normal"})),
    Entry("eddi_spatial_block", BLOCK, compute, _eddi, frozenset({"tukey_probabilities", "hastings_inverse_normal"})),
    Entry("pci", DAILY, compute, _pci, frozenset({"pci"})),
    Entry("fit_diagnostics", MONTHLY, compute, _fit_diagnostics, frozenset({"gamma_parameters"})),
    Entry("thornthwaite", MONTHLY, eto, _thornthwaite, frozenset({"thornthwaite"})),
    Entry("hargreaves", DAILY, eto, _hargreaves, frozenset({"hargreaves"})),
    Entry("penman_monteith", DAILY, pm_eto, _penman_monteith, frozenset({"fao56_eto"})),
    Entry("pm_eto_intermediates", DAILY, pm_eto, _pm_eto_intermediates, frozenset({"pm_eto"})),
    *_fire_entries(),
    *_flood_entries(),
    Entry("palmer_pdsi", MONTHLY, palmer, _pdsi, _PALMER_WATER_BALANCE | {"palmer_pdi"}),
    Entry(
        "palmer_scpdsi",
        MONTHLY,
        palmer,
        _scpdsi,
        _PALMER_WATER_BALANCE | {"scpdsi_duration_factors", "palmer_wells"},
    ),
)

ENTRIES_BY_NAME: dict[str, Entry] = {entry.name: entry for entry in ENTRIES}
REGISTERED_KERNELS: frozenset[str] = frozenset().union(*(entry.kernels for entry in ENTRIES))


# ---------------------------------------------------------------------------
# Routing: which inputs the documented dispatch sends to Rust, and which stay Python
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingCase:
    """One documented dispatch decision: the input, the kernels it must reach, and the ones it must not."""

    name: str
    documented: str
    dispatch: ModuleType
    run: Callable[[], Any]
    expected: frozenset[str] = frozenset()
    forbidden: frozenset[str] = frozenset()


def _routing_cases() -> tuple[RoutingCase, ...]:
    values = SAMPLES[MONTHLY].copy()
    start, calibration_start, calibration_end = _monthly_window(values)
    daily_values = SAMPLES[DAILY].copy()
    fitted = np.nan_to_num(values, nan=0.0)
    stride = np.full(values.size * 2, np.nan)
    stride[::2] = fitted
    overflowing = fitted.reshape(-1, 12).copy()
    # the guard reads the calibration window, so the infinity has to land inside it
    overflowing[2, 3] = np.inf
    temperatures = _temperatures(values)
    monthly = values.reshape(-1, 12)
    return (
        RoutingCase(
            "float32",
            "a float32 series is fitted in float32 by NumPy, so it keeps the Python fit",
            compute,
            lambda: compute.gamma_parameters(
                values.astype(np.float32), start, calibration_start, calibration_end, compute.Periodicity.monthly
            ),
            forbidden=frozenset({"gamma_parameters"}),
        ),
        RoutingCase(
            "masked",
            "a masked array is not a plain float64 ndarray, so it keeps the Python path",
            eto,
            lambda: eto.eto_thornthwaite(np.ma.masked_invalid(temperatures), 40.0, start),
            forbidden=frozenset({"thornthwaite"}),
        ),
        RoutingCase(
            "strided",
            "a non-contiguous float64 series reaches Rust",
            eto,
            lambda: eto.eto_thornthwaite(temperatures[::-1], 40.0, start),
            expected=frozenset({"thornthwaite"}),
        ),
        RoutingCase(
            "year_varying_parameters",
            "year-varying fitting parameters keep the Python CDF",
            compute,
            lambda: compute.transform_fitted_gamma(
                monthly,
                start,
                calibration_start,
                calibration_end,
                compute.Periodicity.monthly,
                np.full(monthly.shape, 2.0),
                np.full(monthly.shape, 30.0),
            ),
            forbidden=frozenset({"gamma_probabilities"}),
        ),
        RoutingCase(
            "overflowing_lmoment_block",
            "an infinite value in an L-moment block keeps the Python fit",
            compute,
            lambda: compute.pearson_parameters(
                overflowing, start, calibration_start, calibration_end, compute.Periodicity.monthly
            ),
            forbidden=frozenset({"pearson_parameters"}),
        ),
        RoutingCase(
            "plain_daily_series",
            "a plain float64 daily series reaches Rust",
            eto,
            lambda: eto.eto_hargreaves(
                _daily_temperature(daily_values) - 6.0,
                _daily_temperature(daily_values) + 6.0,
                _daily_temperature(daily_values),
                40.0,
            ),
            expected=frozenset({"hargreaves"}),
        ),
    )


ROUTING: tuple[RoutingCase, ...] = _routing_cases()


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _outcome(run: Callable[[], Any]) -> Any:
    try:
        return run()
    except Exception as error:  # noqa: BLE001 - the raised outcome is what is compared
        return _Error(type(error), str(error))


def _leaves(value: Any) -> Iterator[Any]:
    """Every numeric leaf of a result: a dataclass, tuple, dict, DataArray, or array."""
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        for name in fields:
            yield from _leaves(getattr(value, name))
        return
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _leaves(value[key])
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _leaves(item)
        return
    if isinstance(value, xr.DataArray):
        yield from _leaves(value.values)
        return
    if value is None:
        return
    if not np.issubdtype(np.asarray(value).dtype, np.number):
        # a distribution enum or a unit name is not a deviation to measure
        return
    yield value


def _numeric_pairs(rust: Any, python: Any) -> Iterator[tuple[float, float]]:
    for rust_leaf, python_leaf in zip(_leaves(rust), _leaves(python), strict=True):
        rust_array = np.asarray(rust_leaf, dtype=float).reshape(-1)
        python_array = np.asarray(python_leaf, dtype=float).reshape(-1)
        for left, right in zip(rust_array, python_array, strict=True):
            yield float(left), float(right)


def parity_errors(rust: Any, python: Any) -> tuple[float, float]:
    """The largest absolute and relative deviation over a Rust/Python outcome pair."""
    max_absolute = 0.0
    max_relative = 0.0
    for left, right in _numeric_pairs(rust, python):
        difference = abs(left - right)
        if np.isnan(difference):
            continue
        max_absolute = max(max_absolute, difference)
        if right != 0.0:
            max_relative = max(max_relative, difference / abs(right))
    return max_absolute, max_relative


def assert_outcomes_parity(entry: Entry, rust: Any, python: Any) -> None:
    """Assert two outcomes of one entry agree: values at its tolerance, errors exactly."""
    if isinstance(python, _Error) or isinstance(rust, _Error):
        assert isinstance(rust, _Error) and isinstance(python, _Error), f"{entry.name}: {rust!r} vs {python!r}"
        assert (rust.error, rust.message) == (python.error, python.message)
        return
    assert entry.rtol == RTOL and entry.atol == ATOL, (
        f"{entry.name} carries a loosened tolerance; compare it explicitly with its recorded evidence"
    )
    conftest.assert_native_parity(rust, python)


def run_parity(monkeypatch: pytest.MonkeyPatch, entry: Entry, values: np.ndarray | None = None) -> ParityRun:
    """Run one entry through the Rust kernels and the Python reference, and compare them."""
    drawn = SAMPLES[entry.family] if values is None else values
    recorder = conftest.NativeRecorder(conftest.import_native())
    # the Rust kernels do not implement NumPy's floating-point reporting policies;
    # the context also restores every dispatch module, so one entry never leaves
    # another entry's module patched to None
    with np.errstate(all="ignore"), monkeypatch.context() as patch:
        patch.setattr(entry.dispatch, "_native", recorder)
        rust = _outcome(entry.run(drawn))
        patch.setattr(entry.dispatch, "_native", None)
        python = _outcome(entry.run(drawn))
    calls = frozenset(recorder.calls)
    assert entry.kernels <= calls, f"{entry.name} did not reach {sorted(entry.kernels - calls)}"
    assert calls <= REGISTERED_KERNELS | entry.kernels, f"{entry.name} reached an unregistered kernel: {calls}"
    assert_outcomes_parity(entry, rust, python)
    max_absolute, max_relative = parity_errors(rust, python)
    return ParityRun(entry, rust, python, calls, max_absolute, max_relative)


def routing_calls(monkeypatch: pytest.MonkeyPatch, case: RoutingCase) -> frozenset[str]:
    """The kernels a routing case reached on its documented path."""
    recorder = conftest.NativeRecorder(conftest.import_native())
    with np.errstate(all="ignore"), monkeypatch.context() as patch:
        patch.setattr(case.dispatch, "_native", recorder)
        case.run()
    return frozenset(recorder.calls)


def measure(monkeypatch: pytest.MonkeyPatch, entries: tuple[Entry, ...] = ENTRIES) -> list[ParityRun]:
    """Run every entry once and return its measured parity."""
    return [run_parity(monkeypatch, entry) for entry in entries]
