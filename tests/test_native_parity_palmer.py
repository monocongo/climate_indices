"""Parity of the Rust Palmer-family kernels with the Python reference implementation.

Each test computes the same result twice through ``palmer.pdsi``/``palmer.scpdsi``
(or a ``palmer`` dispatch helper): once with ``palmer._native`` replaced by a
recorder around the Rust extension, and once with it set to None, which runs the
pure-Python reference. The recorder proves the first run reached the Rust kernels,
so the comparison is never Python against Python. The contract is
``rtol = atol = 1e-10`` with matching NaN positions; only explicitly expected
errors may pass, with the same type and message.

Skipped when the extension is not built (``uv run maturin develop --release``),
unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1`` is set, as in CI's native legs.
"""

import json
import platform
import weakref
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from climate_indices.drought import palmer, self_calibration
from climate_indices.drought._palmer_duration import DurationFactors
from climate_indices.drought._palmer_pdi import PdiDurationFactors
from climate_indices.exceptions import ConvergenceError, InsufficientDataError, InvalidArgumentError
from tests import conftest

native = conftest.import_native()

RTOL = 1e-10
ATOL = 1e-10
# REFIMPL-34 (#1636) measured every output of pdsi() and scpdsi() on the 344 divisions
# as bit-identical between the two paths on macOS arm64 (docs/research/reference-papers/
# palmer-rust-parity.md), so that platform pins a maximum of exactly 0.0. Other
# platforms keep the registry contract until the reproducibility workflow reports them.
_PINNED = (platform.system(), platform.machine()) == ("Darwin", "arm64")
PINNED_RTOL, PINNED_ATOL = (0.0, 0.0) if _PINNED else (RTOL, ATOL)

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


def _assert_parity(
    rust: Any, python: Any, *, error: type[Exception] | None = None, rtol: float = RTOL, atol: float = ATOL
) -> None:
    if isinstance(python, tuple) and python and isinstance(python[0], type):
        assert python[0] is error, f"unexpected error: {python}"
        assert rust == python
        return
    assert error is None, f"expected error: {error}"
    if isinstance(python, dict):
        assert rust.keys() == python.keys()
        for key in python:
            _assert_parity(rust[key], python[key], rtol=rtol, atol=atol)
        return
    if isinstance(python, tuple):
        assert len(rust) == len(python)
        for rust_item, python_item in zip(rust, python, strict=True):
            _assert_parity(rust_item, python_item, rtol=rtol, atol=atol)
        return
    if python is None:
        assert rust is None
        return
    rust, python = np.asarray(rust), np.asarray(python)
    assert rust.shape == python.shape
    assert rust.dtype == python.dtype
    np.testing.assert_allclose(rust, python, rtol=rtol, atol=atol, equal_nan=True)
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


def test_nominal_parity_rejects_matching_unexpected_errors() -> None:
    outcome = (RuntimeError, "valid fixture unexpectedly failed")
    with pytest.raises(AssertionError, match="unexpected error"):
        _assert_parity(outcome, outcome)


def test_every_division_matches_for_pdsi(monkeypatch, palmer_division_dir: Path, palmer_division_inputs) -> None:
    precips, pet, awc = palmer_division_inputs[palmer_division_dir.name]
    rust, python, calls = _rust_and_python(monkeypatch, _pdsi(precips, pet, awc))
    assert calls == _PDSI_KERNELS
    _assert_parity(rust, python, rtol=PINNED_RTOL, atol=PINNED_ATOL)


def test_every_division_matches_for_scpdsi(monkeypatch, palmer_division_dir: Path, palmer_division_inputs) -> None:
    precips, pet, awc = palmer_division_inputs[palmer_division_dir.name]
    rust, python, calls = _rust_and_python(monkeypatch, _scpdsi(precips, pet, awc))
    assert calls == _SCPDSI_KERNELS
    # the fitted duration factors are in the parameter dictionary
    assert {"wetm", "wetb", "drym", "dryb"} <= python[4].keys()
    _assert_parity(rust, python, rtol=PINNED_RTOL, atol=PINNED_ATOL)


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
    _assert_parity(rust, python, error=ValueError if isinstance(awc, np.ndarray) else None)


@pytest.mark.parametrize("index", [_pdsi, _scpdsi], ids=["pdsi", "scpdsi"])
@pytest.mark.parametrize("missing_in_calibration", [True, False], ids=["calibration-gaps", "outside-calibration"])
def test_missing_months_match(monkeypatch, palmer_division_inputs, index, missing_in_calibration: bool) -> None:
    original, pet, awc = palmer_division_inputs["4002"]
    precips = original.copy()
    rng = np.random.default_rng(1278)
    precips[rng.random(precips.shape) < 0.03] = np.nan
    precips[600:640] = np.nan
    if not missing_in_calibration:
        calibration = slice((_CALIBRATION[0] - _START) * 12, (_CALIBRATION[1] - _START + 1) * 12)
        precips[calibration] = original[calibration]
    rust, python, calls = _rust_and_python(monkeypatch, index(precips, pet, awc))
    assert "palmer_water_balance" in calls
    # Calibration gaps invalidate scPDSI K-prime; gaps outside it still produce indices.
    error = ConvergenceError if index is _scpdsi and missing_in_calibration else None
    _assert_parity(rust, python, error=error)


def test_short_calibration_raises_like_python(monkeypatch, palmer_division_inputs) -> None:
    precips, pet, awc = palmer_division_inputs["4406"]
    run = lambda: palmer.scpdsi(precips, pet, awc, _START, 1931, 1933)  # noqa: E731
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert python[0] is InsufficientDataError
    assert "scpdsi_duration_factors" not in calls
    _assert_parity(rust, python, error=InsufficientDataError)


@pytest.mark.parametrize("sign", [self_calibration.WET_SIGN, self_calibration.DRY_SIGN])
def test_degenerate_duration_factor_fit_raises_like_python(monkeypatch, sign: int) -> None:
    # every wet window sums to zero, so the ten regression ordinates are identical
    calibration_z = np.zeros(60)
    rust, python, calls = _rust_and_python(monkeypatch, lambda: palmer._scpdsi_duration_factors(calibration_z, sign))
    assert python[0] is ConvergenceError
    assert calls == {"scpdsi_duration_factors"}
    _assert_parity(rust, python, error=ConvergenceError)


def test_wells_abatement_failure_raises_like_python(monkeypatch) -> None:
    # an established wet spell at x3 = 1 makes the abatement denominator exactly zero
    factors = DurationFactors.from_fitted(1.0, 1.0, 1.0, 1.0)
    z = np.array([2.0, 0.0])
    rust, python, calls = _rust_and_python(monkeypatch, lambda: palmer._wells_recursion(z, factors))
    assert python[0] is ConvergenceError
    assert calls == {"palmer_wells"}
    _assert_parity(rust, python, error=ConvergenceError)


_INFINITE_Z = np.where(np.arange(12) == 5, np.inf, 0.0)
_PDI_FACTORS = PdiDurationFactors.from_fitted(0.3, 2.7, 0.3, 2.7)


@pytest.mark.parametrize("shape", [(0, 12, 0), (0, 12, 3), (2, 12, 0)])
def test_pdi_recursion_preserves_empty_shapes(monkeypatch, shape: tuple[int, int, int]) -> None:
    z = np.empty(shape)

    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = palmer._pdi_recursion(z, _PDI_FACTORS)
        return result.pdsi, result.phdi, result.pmdi

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"palmer_pdi"}
    _assert_parity(rust, python)
    _assert_parity(rust, (z, z, z))


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
    _assert_parity(rust, python, error=error)


@pytest.mark.parametrize(
    ("native", "shape"),
    [(None, (1, 12, 1)), (SimpleNamespace(palmer_pdi=None), (2, 3))],
    ids=["extension-absent", "non-monthly-shape"],
)
def test_pdi_dispatch_scans_z_for_infinities_only_when_the_kernel_can_take_it(monkeypatch, native, shape) -> None:
    """The dispatch guard does not pay for a scan the Python path repeats for its own error."""
    monkeypatch.setattr(palmer, "_palmer_native", lambda *args: native)
    monkeypatch.setattr(palmer._palmer_pdi, "calculate", lambda *args: "python path")
    monkeypatch.setattr(np, "isinf", lambda values: pytest.fail("the dispatch guard scanned z"))

    assert palmer._pdi_recursion(np.zeros(shape), _PDI_FACTORS) == "python path"


# --- branch stability (REFIMPL-34, #1636) ---------------------------------------------
# The recursions branch on exact zeros and fixed thresholds (docs/algorithm_refs/palmer.md,
# "Exact-zero comparisons"), so a one-ulp difference upstream could flip a branch. The
# corpus puts boundary values (-0.0, which equals 0, and +-0.15, +-0.5, +-1, each with one ulp either side) into
# random Z series, then repeats each series with one boundary entry moved one ulp down
# and up. Both paths must agree on every series; the liveness floors prove that moving
# an entry by one ulp does change the answer for some series, so agreement is not vacuous.

_BOUNDARIES = (0.0, 0.15, 0.5, 1.0)
_CORPUS_SEED = 1636
_MONTHS = 60
_LIVE_DIFFERENCE = 1e-6  # a smooth response to one ulp is ~1e-16; a flipped branch moves the index


def _boundary_values() -> np.ndarray:
    edges = {-0.0}
    for boundary in _BOUNDARIES:
        for signed in (boundary, -boundary):
            edges.update((signed, np.nextafter(signed, np.inf), np.nextafter(signed, -np.inf)))
    return np.array(sorted(edges, key=lambda value: (value, np.signbit(value))))


def _straddling_corpus(series: int) -> np.ndarray:
    """``(_MONTHS, 3 * series)`` Z series; columns ``3k..3k+2`` are one series with one
    boundary entry one ulp below, on, and one ulp above the boundary."""
    rng = np.random.default_rng(_CORPUS_SEED)
    edges = _boundary_values()
    base = np.where(
        rng.random((_MONTHS, series)) < 0.6,
        rng.choice(edges, (_MONTHS, series)),
        rng.normal(0.0, 1.5, (_MONTHS, series)),
    )
    corpus = np.repeat(base, 3, axis=1)
    for column in range(series):
        # the entry moved is a boundary value already, so it is the one at stake
        positions = np.flatnonzero(np.isin(base[:, column], edges))
        if positions.size:
            month = rng.choice(positions)
            value = base[month, column]
            corpus[month, 3 * column] = np.nextafter(value, -np.inf)
            corpus[month, 3 * column + 2] = np.nextafter(value, np.inf)
    return corpus


def _live_triples(result: np.ndarray, completed: np.ndarray | None = None) -> int:
    """Count complete (below, on, above) triples with a change above the liveness floor."""
    triples = result.reshape(result.shape[0], -1, 3)
    if completed is not None:
        triples = triples[:, completed.reshape(-1, 3).all(axis=1), :]
    spread = np.abs(triples[:, :, [0, 2]] - triples[:, :, [1]]).max(axis=(0, 2))
    return int(np.count_nonzero(spread > _LIVE_DIFFERENCE))


def test_live_triples_excludes_failed_runs() -> None:
    # two months and three triples, so a transposed grouping of `completed` gives a different count
    results = np.array([[0.0] * 9, [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]])
    completed = np.array([True, False, False, True, True, True, True, True, True])
    assert _live_triples(results, completed) == 1


def _same_outcome(rust: Any, python: Any) -> None:
    """Both paths return the same values, or raise the same error with the same message."""
    if isinstance(python, tuple) and python and isinstance(python[0], type):
        assert rust == python
        return
    _assert_parity(rust, python)
    # a branch flip moves a value to the other side of a threshold, far above 1e-10
    for rust_item, python_item in zip(rust, python, strict=True):
        np.testing.assert_array_equal(rust_item == 0.0, python_item == 0.0)


def _pdi_run(z: np.ndarray, factors: PdiDurationFactors = _PDI_FACTORS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = palmer._pdi_recursion(z, factors)
    return result.pdsi, result.phdi, result.pmdi


def test_pdi_recursion_takes_the_same_branches_on_boundary_series(monkeypatch) -> None:
    corpus = _straddling_corpus(series=1500)
    z = corpus.reshape(_MONTHS // 12, 12, corpus.shape[1])
    rust, python, calls = _rust_and_python(monkeypatch, lambda: _pdi_run(z))
    assert calls == {"palmer_pdi"}
    _same_outcome(rust, python)
    # the record's H3 table measured 28, 14 and 6 live triples for these three outputs. With
    # both flush fixes (#1665, #1666; ADR-0030) this corpus measures 27, 13 and 0: #1665
    # stopped the resolved months being rewritten, which moved PMDI to 3, and the 3 that
    # remained all came from the flush copying the final month's CASE into every pending
    # month. Since #1666 each month keeps its own, the CASE is continuous away from its
    # branch points. PMDI's liveness is asserted on the spell-threshold corpus below; here
    # its parity is 1e-10 with an identical zero and sign pattern.
    for output, floor in zip(python[:2], (27, 13), strict=True):
        assert _live_triples(output.reshape(_MONTHS, -1)) >= floor


def _wells_corpus_outcomes(monkeypatch, corpus: np.ndarray, factors: DurationFactors) -> tuple[np.ndarray, np.ndarray]:
    """Run each corpus column through both paths; return the Python PDSI and the completed columns."""
    results = np.zeros(corpus.shape)
    completed = np.zeros(corpus.shape[1], dtype=bool)
    for column in range(corpus.shape[1]):
        z = np.ascontiguousarray(corpus[:, column])
        rust, python, calls = _rust_and_python(monkeypatch, lambda z=z: palmer._wells_recursion(z, factors))
        assert calls == {"palmer_wells"} or isinstance(python[0], type)
        _same_outcome(rust, python)
        if not isinstance(python[0], type):
            results[:, column] = python[0]
            completed[column] = True
    errors = completed.size - np.count_nonzero(completed)
    assert errors < corpus.shape[1] // 10, "the corpus must mostly complete, or it tests only the error path"
    return results, completed


def test_wells_recursion_takes_the_same_branches_on_boundary_series(monkeypatch) -> None:
    results, completed = _wells_corpus_outcomes(
        monkeypatch, _straddling_corpus(series=600), DurationFactors.from_defaults()
    )
    assert _live_triples(results, completed) >= 4  # measured 4, record's H3 table


def _spell_threshold_corpus(factors: DurationFactors) -> np.ndarray:
    """``(24, 3 * n)`` series: an established spell, then a threshold value and one ulp either side.

    The Wells continuation test ``direction * z >= 0.5 * slope`` and the ``pv`` sign tests
    of the ``pdi.f`` abatement statements only run once a spell is established, which random
    draws rarely reach with the entry exactly on the boundary.
    """
    wet, dry = 0.5 * (factors.wetm + factors.wetb), 0.5 * (factors.drym + factors.dryb)
    # abatement_z = slope * 0.5 in _continue_spell; the establishment test is x1 = z / (m + b) >= 0.5
    thresholds = {0.15, 0.5, 1.0, wet, dry, 0.5 * factors.wetm, 0.5 * factors.drym}
    thresholds |= {2 * wet, 2 * dry, factors.wetm + factors.wetb, factors.drym + factors.dryb}
    thresholds |= {-value for value in thresholds}
    columns = []
    for prefix_sign in (1.0, -1.0):
        for magnitude in (1.0, 3.0):
            for length in (1, 2, 4, 6):
                for threshold in sorted(thresholds):
                    for tail in (0.0, -prefix_sign * 2.0):
                        for value in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)):
                            z = np.zeros(24)
                            z[10 : 10 + length] = prefix_sign * magnitude
                            z[10 + length] = value
                            z[11 + length :] = tail
                            columns.append(z)
    return np.stack(columns, axis=1)


def test_pdi_recursion_takes_the_same_branches_at_spell_thresholds(monkeypatch) -> None:
    corpus = _spell_threshold_corpus(DurationFactors.from_defaults())
    z = corpus.reshape(2, 12, corpus.shape[1])
    rust, python, calls = _rust_and_python(monkeypatch, lambda: _pdi_run(z))
    assert calls == {"palmer_pdi"}
    _same_outcome(rust, python)
    assert _live_triples(python[0].reshape(24, -1)) >= 26  # measured 26, record's H3 table
    # measured 14; since #1666 this corpus, not the straddling one, keeps PMDI's CASE branches live
    assert _live_triples(python[2].reshape(24, -1)) >= 14


def test_wells_recursion_takes_the_same_branches_at_spell_thresholds(monkeypatch) -> None:
    factors = DurationFactors.from_defaults()
    results, completed = _wells_corpus_outcomes(monkeypatch, _spell_threshold_corpus(factors), factors)
    assert _live_triples(results, completed) >= 30  # measured 30, record's H3 table


@pytest.mark.parametrize("direction", [1.0, -1.0], ids=["wet", "dry"])
def test_wells_continuation_threshold_right_after_a_spell_ends(monkeypatch, direction: float) -> None:
    """``direction * z >= 0.5 * slope`` is not redundant while a terminated spell's carry is open.

    After a spell terminates and a new one is established in the same month, the state has
    ``probability == 100`` and a nonzero ``v``. A ``z`` exactly on the threshold then restarts
    the spell (``v = 0``), while one ulp below it continues the abatement (REFIMPL-34 review).
    """
    factors = DurationFactors.from_fitted(0.897, 0.333, 0.897, 0.333)
    # the two directions are exact negatives of each other while wetm == drym
    threshold = 0.5 * (factors.wetm if direction > 0 else factors.drym)
    outputs = []
    for value in (np.nextafter(threshold, 0.0), threshold, np.nextafter(threshold, 1.0)):
        z = direction * np.array([-1.0, 3.0, 0.0, value, value, value])
        rust, python, calls = _rust_and_python(monkeypatch, lambda z=z: palmer._wells_recursion(z, factors))
        assert calls == {"palmer_wells"}
        _same_outcome(rust, python)
        outputs.append(python[0])
    below, on, above = outputs
    assert abs(below[4] - on[4]) > 0.1  # one ulp changes the answer, so agreement is not vacuous
    np.testing.assert_allclose(on, above, atol=1e-12)  # same branch, a smooth ulp-level response


_SEQUENCES = Path(__file__).parents[1] / "docs/research/reference-papers/palmer-recursion-divergence-sequences.json"


@pytest.mark.skipif(not _SEQUENCES.exists(), reason="the research record is not part of an installed copy")
@pytest.mark.parametrize("label", sorted(json.loads(_SEQUENCES.read_text())) if _SEQUENCES.exists() else [])
def test_divergence_sequences_match_between_paths(monkeypatch, label: str) -> None:
    """The minimal sequences on which pdi.f and Wells disagree, run on Rust and Python alike."""
    record = json.loads(_SEQUENCES.read_text())[label]
    z = np.asarray(record["z"], dtype=float)
    z = np.concatenate([np.zeros(-len(z) % 12), z])
    wells = DurationFactors.from_defaults()
    pdi = PdiDurationFactors.from_fitted(wells.wetm, wells.wetb, wells.drym, wells.dryb)

    rust, python, calls = _rust_and_python(monkeypatch, lambda: palmer._wells_recursion(z, wells))
    assert calls == {"palmer_wells"}
    _same_outcome(rust, python)

    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = palmer._pdi_recursion(z.reshape(-1, 12, 1), pdi)
        return result.pdsi, result.phdi, result.pmdi

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"palmer_pdi"}
    _same_outcome(rust, python)


@pytest.mark.parametrize("index", [_pdsi, _scpdsi], ids=["pdsi", "scpdsi"])
def test_zero_precipitation_months_match(monkeypatch, palmer_division_inputs, index) -> None:
    """Dry months and a whole dry year: the registry's input builders never produce zero precipitation."""
    precips, pet, awc = palmer_division_inputs["4002"]
    precips = precips.copy()
    precips[np.random.default_rng(1636).random(precips.shape) < 0.08] = 0.0
    precips[600:612] = 0.0
    rust, python, calls = _rust_and_python(monkeypatch, index(precips, pet, awc))
    assert {"palmer_water_balance", "palmer_pdi" if index is _pdsi else "palmer_wells"} <= calls
    _assert_parity(rust, python)


def test_pmdi_exact_tie_goes_to_x1_on_both_backends(monkeypatch) -> None:
    """#1662: with Z = (0.5, -0.125) and unit factors, every month after the first has |X1| == |X2|.

    pdi.f CASE and the Wells getWPLM give the tie to X1, so the PMDI stays positive. The dyadic
    values keep each tie exact in floating point, so a one-sided revert of the comparison flips
    the PMDI sign here. (The PDI lineage's tied values are the end-of-record flush of #1665.)
    """
    z = np.zeros(24)
    z[:2] = (0.5, -0.125)

    pdi_factors = PdiDurationFactors.from_fitted(1.0, 1.0, 1.0, 1.0)
    rust, python, calls = _rust_and_python(monkeypatch, lambda: _pdi_run(z.reshape(2, 12, 1), pdi_factors))
    assert calls == {"palmer_pdi"}
    _same_outcome(rust, python)
    assert np.all(python[2].reshape(24)[1:] > 0.0)

    wells_factors = DurationFactors.from_fitted(1.0, 1.0, 1.0, 1.0)
    rust, python, calls = _rust_and_python(monkeypatch, lambda: palmer._wells_recursion(z, wells_factors))
    assert calls == {"palmer_wells"}
    _same_outcome(rust, python)
    pmdi = np.asarray(python[2])
    assert pmdi[1] == 0.0625
    assert np.all(pmdi[1:] > 0.0)
