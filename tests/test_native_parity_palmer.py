"""Parity of the Rust Palmer-family kernels with the Python reference implementation.

Each test computes the same result twice through ``palmer.pdsi``/``palmer.scpdsi``
(or a ``palmer`` dispatch helper): once with ``palmer._native`` replaced by a
recorder around the Rust extension, and once with it set to None, which runs the
pure-Python reference. The recorder proves the first run reached the Rust kernels,
so the comparison is never Python against Python. The contract is
``rtol = atol = 1e-10`` with matching NaN positions; an error must be the same
type with the same message.

Skipped when the extension is not built (``uv run maturin develop --release``),
unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1`` is set, as in CI's native legs.
"""

import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from climate_indices import palmer, self_calibration
from climate_indices._palmer_duration import DurationFactors
from climate_indices._palmer_pdi import PdiDurationFactors
from climate_indices.exceptions import ConvergenceError, InsufficientDataError, InvalidArgumentError
from tests import conftest

native = conftest.import_native()

RTOL = 1e-10
ATOL = 1e-10

_START = 1895
_CALIBRATION = (1931, 1990)
_SHARED = {"palmer_water_balance", "palmer_k_prime", "palmer_raw_zindex"}
_PDSI_KERNELS = _SHARED | {"palmer_pdi"}
_SCPDSI_KERNELS = _SHARED | {"scpdsi_duration_factors", "palmer_wells"}


class _Recorder:
    """Stand-in for the extension module that records which kernels were called.

    Only a call is recorded: the dispatch probes kernels with ``hasattr`` before it
    decides to stay in Python, and reads the exception types it translates.
    """

    def __init__(self, module: Any) -> None:
        self._module = module
        self.calls: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._module, name)
        if isinstance(attribute, type):
            return attribute

        def record(*args: Any) -> Any:
            self.calls.add(name)
            return attribute(*args)

        return record


def _outcome(run: Callable[[], Any]) -> Any:
    """The run's result, or the type and message of the error it raised."""
    try:
        return run()
    except Exception as error:  # noqa: BLE001 - the comparison is the point
        return type(error), str(error)


def _rust_and_python(monkeypatch: pytest.MonkeyPatch, run: Callable[[], Any]) -> tuple[Any, Any, set[str]]:
    recorder = _Recorder(native)
    # Native kernels do not implement NumPy's floating-point reporting policies.
    with np.errstate(all="ignore"):
        monkeypatch.setattr(palmer, "_native", recorder)
        rust = _outcome(run)
        monkeypatch.setattr(palmer, "_native", None)
        python = _outcome(run)
    return rust, python, recorder.calls


def _assert_parity(rust: Any, python: Any) -> None:
    if isinstance(python, tuple) and python and isinstance(python[0], type):
        assert rust == python
        return
    if isinstance(python, dict):
        assert rust.keys() == python.keys()
        for key in python:
            _assert_parity(rust[key], python[key])
        return
    if isinstance(python, tuple):
        assert len(rust) == len(python)
        for rust_item, python_item in zip(rust, python, strict=True):
            _assert_parity(rust_item, python_item)
        return
    if python is None:
        assert rust is None
        return
    rust, python = np.asarray(rust), np.asarray(python)
    assert rust.shape == python.shape
    assert rust.dtype == python.dtype
    np.testing.assert_allclose(rust, python, rtol=RTOL, atol=ATOL, equal_nan=True)
    # the backtracking selects a candidate (x1 >= 0, x2 <= 0, or none at exactly 0.0),
    # and that categorical choice must match exactly, not within tolerance
    np.testing.assert_array_equal(np.sign(rust), np.sign(python))


def _pdsi(precips: np.ndarray, pet: np.ndarray, awc: Any, **kwargs: Any) -> Callable[[], Any]:
    return lambda: palmer.pdsi(precips, pet, awc, _START, *_CALIBRATION, **kwargs)


def _scpdsi(precips: np.ndarray, pet: np.ndarray, awc: Any, **kwargs: Any) -> Callable[[], Any]:
    return lambda: palmer.scpdsi(precips, pet, awc, _START, *_CALIBRATION, **kwargs)


@pytest.mark.parametrize("months", [0, 1, 11, 13])
def test_water_balance_rejects_non_monthly_shapes(months: int) -> None:
    values = np.zeros((1, months, 1))
    with pytest.raises(ValueError, match=f"months has length {months}, expected 12"):
        native.palmer_water_balance(values, values, np.ones(1), 0, 0)


def test_native_water_balance_releases_unused_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = palmer._initialize_prepared(np.zeros(24), np.zeros(24), 4.0, 2000, 2000, 2001)
    placeholders = [weakref.ref(getattr(prepared, name)) for name in palmer._WATER_BALANCE_MONTHLY]
    kernel = native.palmer_water_balance

    def water_balance(*args: Any) -> Any:
        assert all(reference() is None for reference in placeholders)
        return kernel(*args)

    monkeypatch.setattr(native, "palmer_water_balance", water_balance)
    monkeypatch.setattr(palmer, "_native", native)
    with np.errstate(all="ignore"):
        assert palmer._native_water_balances(prepared)
    for name in palmer._WATER_BALANCE_MONTHLY:
        assert getattr(prepared, name).shape == (2, 12, 1)


@pytest.mark.parametrize("cells", [0, 3])
def test_cafec_kernels_accept_borrowed_strided_and_broadcast_arrays(cells: int) -> None:
    monthly = np.random.default_rng(1316).uniform(1.0, 2.0, (2, 12, cells))[::-1, ::-1, ::-1]
    coefficients = tuple(np.broadcast_to(value, (12, cells)) for value in (0.5, 0.3, 0.2, 0.1))
    factors = np.broadcast_to(2.0, (12, cells))
    operands = (monthly,) * 5 + coefficients
    alpha, beta, gamma, delta = coefficients
    cafec = alpha * monthly + beta * monthly + gamma * monthly - delta * monthly
    expected_dbar = np.abs(monthly - cafec).sum(axis=0) / 2
    with np.errstate(all="ignore"):
        expected_k = 1.5 * np.log10((factors + 2.8) / expected_dbar) + 0.5
        _assert_parity(native.palmer_k_prime(*operands, factors, 0, 1), (expected_dbar, expected_k))
    _assert_parity(native.palmer_raw_zindex(*operands, factors), factors * (monthly - cafec))


@pytest.mark.parametrize("kernel", ["palmer_k_prime", "palmer_raw_zindex"])
@pytest.mark.parametrize("argument", range(10))
def test_cafec_kernels_reject_unaligned_operands(kernel: str, argument: int) -> None:
    operands: list[Any] = [np.ones((2, 12, 1)) for _ in range(5)] + [np.ones((12, 1)) for _ in range(5)]
    original = operands[argument]
    operands[argument] = np.ndarray(original.shape, dtype=np.float64, buffer=bytearray(original.nbytes + 1), offset=1)
    kernel_callable = getattr(native, kernel)
    if kernel == "palmer_k_prime":
        operands.extend((0, 1))
    with pytest.raises(ValueError, match="unaligned float64 array"):
        kernel_callable(*operands)


def test_every_division_matches_for_pdsi(monkeypatch, palmer_division_dir: Path, palmer_division_inputs) -> None:
    precips, pet, awc = palmer_division_inputs[palmer_division_dir.name]
    rust, python, calls = _rust_and_python(monkeypatch, _pdsi(precips, pet, awc))
    assert calls == _PDSI_KERNELS
    _assert_parity(rust, python)


def test_every_division_matches_for_scpdsi(monkeypatch, palmer_division_dir: Path, palmer_division_inputs) -> None:
    precips, pet, awc = palmer_division_inputs[palmer_division_dir.name]
    rust, python, calls = _rust_and_python(monkeypatch, _scpdsi(precips, pet, awc))
    assert calls == _SCPDSI_KERNELS
    # the fitted duration factors are in the parameter dictionary
    assert {"wetm", "wetb", "drym", "dryb"} <= python[4].keys()
    _assert_parity(rust, python)


@pytest.mark.parametrize("per_cell_awc", [True, False], ids=["per-cell-awc", "scalar-awc"])
def test_spatial_block_with_fully_missing_cells(monkeypatch, palmer_division_inputs, per_cell_awc: bool) -> None:
    names = ("0101", "0405", "0909", "1209", "1606")
    columns = [palmer_division_inputs[name] for name in names]
    precips = np.stack([p for p, _, _ in columns] + [np.full_like(columns[0][0], np.nan)], axis=1).reshape(-1, 2, 3)
    pet = np.stack([e for _, e, _ in columns] + [columns[0][1]], axis=1).reshape(-1, 2, 3)
    awc = np.array([a for _, _, a in columns] + [5.0]).reshape(2, 3) if per_cell_awc else 6.0

    rust, python, calls = _rust_and_python(monkeypatch, _pdsi(precips, pet, awc, spatial_time_major=True))
    assert calls == _PDSI_KERNELS
    assert np.isnan(python[0][:, 1, 2]).all()
    assert np.isfinite(python[0][:, 0, 0]).all()
    _assert_parity(rust, python)


def test_supplied_fitting_params_match(monkeypatch, palmer_division_inputs) -> None:
    precips, pet, awc = palmer_division_inputs["2101"]
    with np.errstate(all="ignore"):
        fitted = palmer.pdsi(precips, pet, awc, _START, *_CALIBRATION)[4]
    for run in (_pdsi(precips, pet, awc, fitting_params=fitted), _scpdsi(precips, pet, awc, fitting_params=fitted)):
        rust, python, calls = _rust_and_python(monkeypatch, run)
        assert _SHARED <= calls
        _assert_parity(rust, python)


def test_duration_factor_override_matches(monkeypatch, palmer_division_inputs) -> None:
    precips, pet, awc = palmer_division_inputs["2404"]
    override = {"wetm": 0.3, "wetb": 2.6, "drym": 0.4, "dryb": 2.5}
    rust, python, calls = _rust_and_python(monkeypatch, _pdsi(precips, pet, awc, fitting_params=override))
    assert calls == _PDSI_KERNELS
    assert python[4]["wetm"] == 0.3
    _assert_parity(rust, python)


@pytest.mark.parametrize("awc", [0.0, 0.5, 1.0, 5, np.float64(9.5), np.array(3.0), np.array([4.0])])
@pytest.mark.parametrize("index", [_pdsi, _scpdsi], ids=["pdsi", "scpdsi"])
def test_awc_edge_cases_match(monkeypatch, palmer_division_inputs, awc: Any, index) -> None:
    precips, pet, _ = palmer_division_inputs["3301"]
    rust, python, calls = _rust_and_python(monkeypatch, index(precips, pet, awc))
    assert "palmer_water_balance" in calls
    _assert_parity(rust, python)


@pytest.mark.parametrize("awc", [np.float32(5.0), True, np.array([4.0, 5.0])])
def test_awc_without_float64_semantics_keeps_the_python_water_balance(monkeypatch, palmer_division_inputs, awc) -> None:
    precips, pet, _ = palmer_division_inputs["3601"]
    rust, python, calls = _rust_and_python(monkeypatch, _pdsi(precips, pet, awc))
    assert "palmer_water_balance" not in calls
    _assert_parity(rust, python)


@pytest.mark.parametrize("index", [_pdsi, _scpdsi], ids=["pdsi", "scpdsi"])
def test_missing_months_match(monkeypatch, palmer_division_inputs, index) -> None:
    precips, pet, awc = palmer_division_inputs["4002"]
    precips = precips.copy()
    rng = np.random.default_rng(1278)
    precips[rng.random(precips.shape) < 0.03] = np.nan
    precips[600:640] = np.nan
    rust, python, calls = _rust_and_python(monkeypatch, index(precips, pet, awc))
    assert "palmer_water_balance" in calls
    _assert_parity(rust, python)


def test_short_calibration_raises_like_python(monkeypatch, palmer_division_inputs) -> None:
    precips, pet, awc = palmer_division_inputs["4406"]
    run = lambda: palmer.scpdsi(precips, pet, awc, _START, 1931, 1933)  # noqa: E731
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert python[0] is InsufficientDataError
    assert "scpdsi_duration_factors" not in calls
    _assert_parity(rust, python)


@pytest.mark.parametrize("sign", [self_calibration.WET_SIGN, self_calibration.DRY_SIGN])
def test_degenerate_duration_factor_fit_raises_like_python(monkeypatch, sign: int) -> None:
    # every wet window sums to zero, so the ten regression ordinates are identical
    calibration_z = np.zeros(60)
    rust, python, calls = _rust_and_python(monkeypatch, lambda: palmer._scpdsi_duration_factors(calibration_z, sign))
    assert python[0] is ConvergenceError
    assert calls == {"scpdsi_duration_factors"}
    _assert_parity(rust, python)


def test_wells_abatement_failure_raises_like_python(monkeypatch) -> None:
    # an established wet spell at x3 = 1 makes the abatement denominator exactly zero
    factors = DurationFactors.from_fitted(1.0, 1.0, 1.0, 1.0)
    z = np.array([2.0, 0.0])
    rust, python, calls = _rust_and_python(monkeypatch, lambda: palmer._wells_recursion(z, factors))
    assert python[0] is ConvergenceError
    assert calls == {"palmer_wells"}
    _assert_parity(rust, python)


_INFINITE_Z = np.where(np.arange(12) == 5, np.inf, 0.0)
_PDI_FACTORS = PdiDurationFactors.from_fitted(0.3, 2.7, 0.3, 2.7)


@pytest.mark.parametrize(
    ("run", "error"),
    [
        (lambda: palmer._pdi_recursion(_INFINITE_Z.reshape(1, 12, 1), _PDI_FACTORS), ConvergenceError),
        (lambda: palmer._pdi_recursion(np.zeros((2, 3)), _PDI_FACTORS), ValueError),
        (lambda: palmer._wells_recursion(_INFINITE_Z, DurationFactors.from_defaults()), ConvergenceError),
        (lambda: palmer._scpdsi_duration_factors(np.ones(60), 5), InvalidArgumentError),
    ],
    ids=["pdi-infinite-z", "pdi-shape", "wells-infinite-z", "duration-factor-sign"],
)
def test_inputs_python_rejects_keep_the_python_error(monkeypatch, run, error) -> None:
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert python[0] is error
    assert not calls
    _assert_parity(rust, python)
