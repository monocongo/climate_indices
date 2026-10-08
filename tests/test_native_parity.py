"""Parity of the Rust kernels with the Python reference implementation.

Each test computes the same result twice through the public or compute-level API:
once with ``compute._native`` replaced by a recorder around the Rust extension, and
once with it set to None, which runs the pure-Python reference. The recorder proves
the first run reached the Rust kernels, so the comparison is never Python against
Python. The contract is ``rtol = atol = 1e-10`` with matching NaN positions.

Skipped when the extension is not built (``uv run maturin develop --release``),
unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1`` is set, as in CI's native legs, where a
missing extension is a collection error. This module deliberately has no other skip
(the Python 3.14-only context-aware-warnings check lives in test_native_backend.py),
so a native leg that skips anything here is a bug.
"""

import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.stats

from climate_indices import compute, exceptions, indices
from tests import conftest

native = conftest.import_native()

RTOL = 1e-10
ATOL = 1e-10

_DATA_START = 1895
_KERNELS = {"gamma_parameters", "gamma_probabilities", "norm_ppf"}
_EDDI_KERNELS = {"tukey_probabilities", "hastings_inverse_normal"}


class _Recorder:
    """Stand-in for the extension module that records which kernels were called."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self.calls: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        self.calls.add(name)
        return getattr(self._module, name)


def _rust_and_python(monkeypatch: pytest.MonkeyPatch, run: Callable[[], np.ndarray]) -> tuple[Any, Any, set[str]]:
    recorder = _Recorder(native)
    # Native kernels do not implement NumPy's floating-point reporting policies.
    with np.errstate(all="ignore"):
        monkeypatch.setattr(compute, "_native", recorder)
        rust = run()
        monkeypatch.setattr(compute, "_native", None)
        python = run()
    return rust, python, recorder.calls


def _assert_parity(rust: np.ndarray, python: np.ndarray) -> None:
    assert rust.shape == python.shape
    assert rust.dtype == python.dtype
    np.testing.assert_allclose(rust, python, rtol=RTOL, atol=ATOL, equal_nan=True)


def _spi(values: np.ndarray, scale: int, start: int, end: int, **kwargs: Any) -> Callable[[], np.ndarray]:
    periodicity = kwargs.pop("periodicity", compute.Periodicity.monthly)
    data_start = kwargs.pop("data_start", _DATA_START)
    return lambda: indices.spi(values, scale, indices.Distribution.gamma, data_start, start, end, periodicity, **kwargs)


def _with_zeros(values: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dry = values.copy()
    dry[rng.random(dry.shape) < fraction] = 0.0
    return dry


@pytest.mark.parametrize("scale", [1, 3, 6, 12, 24])
def test_spi_monthly_reference_series(monkeypatch, precips_mm_monthly, scale):
    rust, python, calls = _rust_and_python(monkeypatch, _spi(precips_mm_monthly, scale, 1981, 2010))
    assert calls == _KERNELS
    _assert_parity(rust, python)
    # the leading scale - 1 months have no complete sum on either path
    assert np.isnan(rust[: scale - 1]).all()


@pytest.mark.parametrize("scale", [1, 30, 90])
def test_spi_daily_series_with_zeros_and_gaps(
    monkeypatch,
    precips_mm_daily,
    data_year_start_daily,
    calibration_year_start_daily,
    calibration_year_end_daily,
    scale,
):
    run = _spi(
        precips_mm_daily,
        scale,
        calibration_year_start_daily,
        calibration_year_end_daily,
        periodicity=compute.Periodicity.daily,
        data_start=data_year_start_daily,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    # Scale 1 has calendar steps with no non-zero calibration value.
    assert calls == ({"gamma_probabilities", "norm_ppf"} if scale == 1 else _KERNELS)
    _assert_parity(rust, python)


def test_spi_matches_the_committed_gamma_fixtures(
    monkeypatch,
    precips_mm_monthly,
    data_year_start_monthly,
    data_year_end_monthly,
    spi_1_month_gamma,
    spi_6_month_gamma,
):
    for scale, fixture in ((1, spi_1_month_gamma), (6, spi_6_month_gamma)):
        run = _spi(precips_mm_monthly, scale, data_year_start_monthly, data_year_end_monthly)
        rust, python, calls = _rust_and_python(monkeypatch, run)
        assert calls == _KERNELS
        _assert_parity(rust, python)
        # the same tolerance tests/test_indices.py holds the Python path to
        np.testing.assert_allclose(rust, fixture, atol=0.001, equal_nan=True)


@pytest.mark.parametrize("fraction", [0.2, 0.6])
@pytest.mark.parametrize("zero_handling", ["classic", "center_of_mass", "mean_zero"])
def test_spi_with_zero_precipitation(monkeypatch, precips_mm_monthly, fraction, zero_handling):
    dry = _with_zeros(precips_mm_monthly, fraction, seed=1)
    run = _spi(dry, 3, 1981, 2010, zero_handling=zero_handling)
    rust, python, _ = _rust_and_python(monkeypatch, run)
    _assert_parity(rust, python)


def test_spi_with_an_all_zero_calendar_month(monkeypatch, precips_mm_monthly):
    """A month that never rains has p0 = 1, which the transform resets to no zero mass."""
    dry = precips_mm_monthly.copy()
    dry[:, 6] = 0.0
    rust, python, calls = _rust_and_python(monkeypatch, _spi(dry, 1, 1981, 2010))
    assert calls == {"gamma_probabilities", "norm_ppf"}
    _assert_parity(rust, python)


def test_spi_with_missing_values(monkeypatch, precips_mm_monthly):
    gappy = precips_mm_monthly.copy().flatten()
    rng = np.random.default_rng(2)
    gappy[rng.random(gappy.size) < 0.15] = np.nan
    gappy[:30] = np.nan
    rust, python, _ = _rust_and_python(monkeypatch, _spi(gappy, 6, 1981, 2010))
    _assert_parity(rust, python)


@pytest.mark.parametrize(
    "series",
    [
        pytest.param(np.zeros(240), id="all-zero"),
        pytest.param(np.full(240, 25.0), id="constant"),
        pytest.param(25.0 + 1e-9 * np.arange(240), id="near-constant"),
        pytest.param(np.r_[np.full(239, np.nan), 5.0], id="one-value"),
    ],
)
def test_spi_on_degenerate_series(monkeypatch, series):
    rust, python, _ = _rust_and_python(monkeypatch, _spi(series, 1, 1895, 1914))
    _assert_parity(rust, python)


def test_spi_all_missing_series_never_reaches_a_kernel(monkeypatch):
    rust, python, calls = _rust_and_python(monkeypatch, _spi(np.full(120, np.nan), 3, 1895, 1904))
    assert calls == set()
    _assert_parity(rust, python)


def test_spi_series_shorter_than_scale_raises_on_both_paths(monkeypatch):
    run = _spi(np.ones(5), 6, 1895, 1895)
    for backend in (_Recorder(native), None):
        monkeypatch.setattr(compute, "_native", backend)
        with pytest.raises(exceptions.InsufficientDataError):
            run()


@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(1895, 2017, id="full-record"),
        pytest.param(2010, 2010, id="single-year"),
        pytest.param(2008, 2017, id="end-of-record"),
        pytest.param(1850, 1920, id="starts-before-record"),
        pytest.param(2000, 2050, id="ends-after-record"),
    ],
)
def test_spi_calibration_periods(monkeypatch, precips_mm_monthly, start, end):
    rust, python, _ = _rust_and_python(monkeypatch, _spi(precips_mm_monthly, 3, start, end))
    _assert_parity(rust, python)


@pytest.mark.parametrize("output_scale", ["probability", "bounded"])
def test_spi_output_scales(monkeypatch, precips_mm_monthly, output_scale):
    dry = _with_zeros(precips_mm_monthly, 0.2, seed=3)
    rust, python, calls = _rust_and_python(monkeypatch, _spi(dry, 3, 1981, 2010, output_scale=output_scale))
    # the probability scales skip the inverse-normal transform
    assert calls == {"gamma_parameters", "gamma_probabilities"}
    _assert_parity(rust, python)


def test_spi_spatial_block_with_masked_ocean(monkeypatch, precips_mm_monthly):
    rng = np.random.default_rng(4)
    block = precips_mm_monthly.flatten()[:, None, None] * rng.uniform(0.5, 1.5, (1, 3, 4))
    block = _with_zeros(block, 0.1, seed=5)
    block[:, 0, 0] = np.nan
    rust, python, calls = _rust_and_python(monkeypatch, _spi(block, 6, 1981, 2010, spatial_time_major=True))
    assert calls == {"gamma_probabilities", "norm_ppf"}
    _assert_parity(rust, python)


def test_masked_transform_is_normalized_to_nan_before_dispatch(monkeypatch, precips_mm_monthly):
    """A mask is a missing marker: the transform reads it as NaN, then may use Rust."""
    masked = np.ma.masked_less(precips_mm_monthly, 1.0)
    run = lambda: compute.transform_fitted_gamma(  # noqa: E731
        masked, _DATA_START, 1981, 2010, compute.Periodicity.monthly
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _KERNELS
    _assert_parity(rust, python)

    # the documented guarantee: a partial mask has the result of the explicitly NaN-filled input
    filled = np.ma.filled(masked.astype(float), np.nan)
    _assert_parity(
        python,
        compute.transform_fitted_gamma(filled, _DATA_START, 1981, 2010, compute.Periodicity.monthly),
    )


def test_supplied_fitting_params_are_transformed_by_the_kernel(monkeypatch, precips_mm_monthly):
    alphas, betas = compute.gamma_parameters(precips_mm_monthly, _DATA_START, 1981, 2010, compute.Periodicity.monthly)
    run = _spi(precips_mm_monthly, 1, 1981, 2010, fitting_params={"alpha": alphas, "beta": betas})
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"gamma_probabilities", "norm_ppf"}
    _assert_parity(rust, python)


def test_year_varying_parameters_keep_the_python_cdf(monkeypatch, precips_mm_monthly):
    """Only a caller can pass parameters that vary by year; the kernel takes one per step."""
    alphas = np.full(precips_mm_monthly.shape, 2.0)
    betas = np.full(precips_mm_monthly.shape, 30.0)
    run = lambda: compute.transform_fitted_gamma(  # noqa: E731
        precips_mm_monthly, _DATA_START, 1981, 2010, compute.Periodicity.monthly, alphas, betas
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert "gamma_probabilities" not in calls
    _assert_parity(rust, python)


def test_float32_input_keeps_the_python_fit(monkeypatch, precips_mm_monthly):
    """A scale-1 float32 series is fitted in float32 by NumPy, so it stays on the Python path."""
    rust, python, calls = _rust_and_python(monkeypatch, _spi(precips_mm_monthly.astype(np.float32), 1, 1981, 2010))
    assert "gamma_parameters" not in calls
    _assert_parity(rust, python)


def test_spei_gamma_uses_the_same_kernels(
    monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, data_year_start_monthly, data_year_end_monthly
):
    def run() -> np.ndarray:
        return indices.spei(
            precips_mm_monthly,
            pet_thornthwaite_mm,
            6,
            indices.Distribution.gamma,
            compute.Periodicity.monthly,
            data_year_start_monthly,
            data_year_start_monthly,
            data_year_end_monthly,
        )

    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _KERNELS
    _assert_parity(rust, python)


@pytest.mark.parametrize("periodicity", [compute.Periodicity.monthly, compute.Periodicity.daily])
def test_unclipped_transform_tails(monkeypatch, precips_mm_monthly, precips_mm_daily, periodicity):
    """``transform_fitted_gamma`` is not clipped to +/-3.09, so its tails are compared too."""
    values = precips_mm_monthly if periodicity is compute.Periodicity.monthly else precips_mm_daily
    start = _DATA_START if periodicity is compute.Periodicity.monthly else 1998
    run = lambda: compute.transform_fitted_gamma(values, start, start, start + 9, periodicity)  # noqa: E731
    rust, python, calls = _rust_and_python(monkeypatch, run)
    expected = {"gamma_probabilities", "norm_ppf"} if periodicity is compute.Periodicity.daily else _KERNELS
    assert calls == expected
    assert np.nanmax(np.abs(python)) > 3.09
    _assert_parity(rust, python)


def test_climate_indices_warnings_are_unchanged(monkeypatch, precips_mm_monthly):
    """Warnings come from the Python orchestration around the kernels, on both paths."""
    gappy = _with_zeros(precips_mm_monthly, 0.3, seed=6).flatten()
    gappy[: 12 * 100] = np.nan

    def warning_types(backend) -> tuple[list[type[Warning]], set[str]]:
        recorder = _Recorder(backend) if backend is not None else None
        monkeypatch.setattr(compute, "_native", recorder)
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
            warnings.simplefilter("always")
            _spi(gappy, 3, 1990, 2017)()
        return sorted(
            {w.category for w in caught if issubclass(w.category, exceptions.ClimateIndicesWarning)},
            key=lambda category: category.__name__,
        ), (recorder.calls if recorder is not None else set())

    rust, rust_calls = warning_types(native)
    python, _ = warning_types(None)
    assert rust == python
    assert rust, "the case is meant to trigger at least one data-quality warning"
    assert rust_calls, "the native arm must actually reach the Rust kernels"


@pytest.mark.parametrize("parameter_name", ["alpha", "beta"])
def test_unaligned_supplied_parameters_keep_the_python_cdf(monkeypatch, parameter_name):
    params = {"alpha": np.full(12, 2.0), "beta": np.ones(12), "prob_zero": np.zeros(12)}
    packed = np.zeros(12, dtype=[("value", "f8"), ("flag", "u1")])["value"]
    packed[:] = params[parameter_name]
    params[parameter_name] = packed
    assert not packed.flags.aligned
    run = _spi(np.arange(1.0, 25.0), 1, 1895, 1896, fitting_params=params, output_scale="probability")
    rust, python, calls = _rust_and_python(monkeypatch, run)
    _assert_parity(rust, python)
    assert "gamma_probabilities" not in calls


@pytest.mark.parametrize("layout", ["packed", "offset", "empty-offset"])
@pytest.mark.parametrize(
    ("kernel", "argument_index"),
    [
        ("gamma_parameters", 0),
        *[("gamma_probabilities", i) for i in range(4)],
        ("norm_ppf", 0),
        ("pnp_normals", 0),
        ("pnp_percentages", 0),
        ("pnp_percentages", 1),
        ("pci", 0),
    ],
)
def test_native_boundary_rejects_unaligned_arrays(layout, kernel, argument_index):
    arguments = {
        "gamma_parameters": [np.arange(1.0, 25.0).reshape(2, 12)],
        "gamma_probabilities": [np.ones((2, 12)), np.full(12, 2.0), np.ones(12), np.zeros(12)],
        "norm_ppf": [np.full((2, 12), 0.5)],
        "pnp_normals": [np.ones((2, 12))],
        "pnp_percentages": [np.ones((2, 12)), np.ones((12, 12))],
        "pci": [np.ones(366)],
    }[kernel]
    original = arguments[argument_index]
    if layout == "empty-offset":
        unaligned = np.ndarray((0,) * original.ndim, dtype=np.float64, buffer=bytearray(1), offset=1)
        assert unaligned.flags.aligned  # NumPy ignores pointer alignment for empty arrays.
        assert unaligned.ctypes.data % 8 != 0
    else:
        if layout == "packed":
            unaligned = np.zeros(original.shape, dtype=[("value", "f8"), ("flag", "u1")])["value"]
        else:
            unaligned = np.ndarray(original.shape, dtype=np.float64, buffer=bytearray(original.nbytes + 1), offset=1)
        unaligned[:] = original
        assert not unaligned.flags.aligned
    arguments[argument_index] = unaligned
    kernel_function = getattr(native, kernel)
    with pytest.raises(ValueError, match="unaligned float64 array"):
        kernel_function(*arguments)


@pytest.mark.parametrize("stride", [-1, -16])
@pytest.mark.parametrize("shape", [(0, 12), (2, 0)])
def test_native_empty_negative_stride_blocks(stride, shape):
    empty = np.ndarray(shape, dtype=np.float64, buffer=np.empty(1), strides=(stride, stride))
    assert empty.flags.aligned
    assert empty.ctypes.data % 8 == 0
    alphas, betas = native.gamma_parameters(empty)
    assert alphas.shape == (shape[1],)
    assert betas.shape == (shape[1],)
    assert np.isnan(alphas).all()
    assert np.isnan(betas).all()
    parameters = np.ones(shape[1])
    result = native.gamma_probabilities(empty, parameters, parameters, parameters)
    assert result.shape == shape
    assert result.dtype == np.float64
    quantiles = native.norm_ppf(empty)
    assert quantiles.shape == shape
    assert quantiles.dtype == np.float64
    normals = native.pnp_normals(empty)
    assert normals.shape == (shape[1],)
    percentages = native.pnp_percentages(empty, np.ones((1, shape[1])))
    assert percentages.shape == shape
    assert percentages.dtype == np.float64


@pytest.mark.parametrize("stride", [-1, -16])
@pytest.mark.parametrize("argument_index", [1, 2, 3])
def test_native_empty_negative_stride_parameters(stride, argument_index):
    empty = np.ndarray((0,), dtype=np.float64, buffer=np.empty(1), strides=(stride,))
    arguments = [np.empty((2, 0)), np.empty(0), np.empty(0), np.empty(0)]
    arguments[argument_index] = empty
    result = native.gamma_probabilities(*arguments)
    assert result.shape == (2, 0)
    assert result.dtype == np.float64
    arguments = [np.ones((2, 12)), np.ones(12), np.ones(12), np.zeros(12)]
    arguments[argument_index] = empty
    with pytest.raises(ValueError, match="12"):
        native.gamma_probabilities(*arguments)


def test_native_norm_ppf_preserves_fortran_layout_values():
    probabilities = np.asfortranarray(np.array([[0.1, 0.2], [0.7, 0.9]]))
    _assert_parity(native.norm_ppf(probabilities), scipy.stats.norm.ppf(probabilities))


def test_native_norm_ppf_accepts_numpy_maximum_dimensions():
    ndim = 64 if np.lib.NumpyVersion(np.__version__) >= "2.0.0" else 32
    probabilities = np.array([0.25, 0.75]).reshape((1,) * (ndim - 1) + (2,))
    result = native.norm_ppf(probabilities)
    assert result.shape == probabilities.shape
    # SciPy/NumPy broadcasting itself is limited to 32 dimensions; compare flattened values.
    _assert_parity(result.reshape(-1), scipy.stats.norm.ppf(probabilities.reshape(-1)))


def test_aligned_strided_parameters_still_use_native(monkeypatch):
    params = {"alpha": np.full(24, 2.0)[::-2], "beta": np.ones(24)[::2], "prob_zero": np.zeros(12)}
    run = _spi(np.arange(1.0, 25.0), 1, 1895, 1896, fitting_params=params, output_scale="probability")
    rust, python, calls = _rust_and_python(monkeypatch, run)
    _assert_parity(rust, python)
    assert calls == {"gamma_probabilities"}


def test_numpy_error_policy_preserves_fit_exceptions(monkeypatch):
    run = _spi(np.ones(360), 1, 1895, 1924)
    for backend in (_Recorder(native), None):
        monkeypatch.setattr(compute, "_native", backend)
        with np.errstate(divide="raise"), pytest.raises(FloatingPointError, match="divide by zero"):
            run()
        if backend is not None:
            assert "gamma_parameters" not in backend.calls


def test_runtime_warnings_promoted_to_exceptions_stay_on_python(monkeypatch):
    run = _spi(np.ones(360), 1, 1895, 1924)
    for backend in (_Recorder(native), None):
        monkeypatch.setattr(compute, "_native", backend)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning, match="divide by zero"):
                run()
        if backend is not None:
            assert "gamma_parameters" not in backend.calls


def test_numpy_warning_policy_preserves_fit_warnings(monkeypatch):
    recorded = []
    for backend in (_Recorder(native), None):
        monkeypatch.setattr(compute, "_native", backend)
        with warnings.catch_warnings(record=True) as caught, np.errstate(divide="warn"):
            warnings.simplefilter("always", RuntimeWarning)
            _spi(np.ones(360), 1, 1895, 1924)()
        recorded.append([(warning.category, str(warning.message)) for warning in caught])
        if backend is not None:
            assert not backend.calls
    assert recorded[0] == recorded[1]
    assert any("divide by zero" in message for _, message in recorded[0])


def test_empty_slice_warnings_survive_ignored_floating_point_errors(monkeypatch):
    values = np.arange(1.0, 361.0).reshape(30, 12)
    values[:, 0] = 0.0
    recorder = _Recorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always", RuntimeWarning)
        compute.gamma_parameters(values, 1895, 1895, 1924, compute.Periodicity.monthly)
    assert any("Mean of empty slice" in str(warning.message) for warning in caught)
    assert "gamma_parameters" not in recorder.calls


def test_no_positive_calibration_column_stays_on_python(monkeypatch):
    """A column of only negative values logs to all-NaN, warning independently of errstate."""
    values = np.arange(1.0, 361.0).reshape(30, 12)
    values[:, 0] = -1.0
    recorder = _Recorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always", RuntimeWarning)
        alphas, betas = compute.gamma_parameters(values, 1895, 1895, 1924, compute.Periodicity.monthly)
    assert any("Mean of empty slice" in str(warning.message) for warning in caught)
    assert "gamma_parameters" not in recorder.calls
    assert np.isnan(alphas[0])
    assert np.isnan(betas[0])


def test_runtime_warning_error_filter_disables_native(monkeypatch):
    """An error filter for RuntimeWarning keeps the Python path even under all="ignore"."""
    monkeypatch.setattr(compute, "_native", native)
    calibration = np.ones((2, 12))
    with np.errstate(all="ignore"):
        assert compute._native_float64(calibration)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert not compute._native_float64(calibration)


def test_numpy_error_callback_stays_on_python(monkeypatch):
    def run() -> list[tuple[str, int]]:
        errors = []
        old_callback = np.geterrcall()
        np.seterrcall(lambda error, flag: errors.append((error, flag)))
        try:
            with np.errstate(divide="call"):
                _spi(np.ones(360), 1, 1895, 1924)()
        finally:
            np.seterrcall(old_callback)
        return errors

    monkeypatch.setattr(compute, "_native", native)
    rust = run()
    monkeypatch.setattr(compute, "_native", None)
    assert rust == run()
    assert rust


def test_numpy_error_policy_preserves_cdf_exceptions(monkeypatch):
    params = {"alpha": np.ones(12), "beta": np.full(12, 1e-300), "prob_zero": np.zeros(12)}
    run = _spi(np.full(24, 1e300), 1, 1895, 1896, fitting_params=params, output_scale="probability")
    for backend in (_Recorder(native), None):
        monkeypatch.setattr(compute, "_native", backend)
        with np.errstate(over="raise"), pytest.raises(exceptions.DistributionFittingError) as caught:
            run()
        assert isinstance(caught.value.underlying_error, FloatingPointError)
        if backend is not None:
            assert "gamma_probabilities" not in backend.calls


def test_kernels_match_scipy_primitives():
    """The kernels against the SciPy calls the Python path makes, across every igam branch."""
    rng = np.random.default_rng(7)
    alphas = np.exp(rng.uniform(np.log(1e-3), np.log(2000.0), 20_000))
    betas = np.exp(rng.uniform(-3.0, 3.0, alphas.size))
    values = alphas * np.exp(rng.normal(0.0, 1.5, alphas.size)) * betas
    probabilities = native.gamma_probabilities(values[None, :], alphas, betas, np.zeros(alphas.size))[0]
    np.testing.assert_allclose(
        probabilities, scipy.stats.gamma.cdf(values, a=alphas, scale=betas), rtol=RTOL, atol=ATOL
    )

    quantiles = np.concatenate([rng.uniform(0.0, 1.0, 5000), 1.0 - 10.0 ** rng.uniform(-16, -1, 2000), [0.0, 1.0]])
    np.testing.assert_allclose(native.norm_ppf(quantiles), scipy.stats.norm.ppf(quantiles), rtol=RTOL, atol=ATOL)


# --- PNP and PCI kernels (RUST-007) ---------------------------------------------------

_PNP_KERNELS = {"pnp_normals", "pnp_percentages"}


def _pnp(values: np.ndarray, scale: int, start: int, end: int, **kwargs: Any) -> Callable[[], np.ndarray]:
    periodicity = kwargs.pop("periodicity", compute.Periodicity.monthly)
    data_start = kwargs.pop("data_start", _DATA_START)
    return lambda: indices.percentage_of_normal(values, scale, data_start, start, end, periodicity, **kwargs)


@pytest.mark.parametrize("scale", [1, 3, 6])
def test_pnp_monthly_reference_series(monkeypatch, precips_mm_monthly, scale):
    rust, python, calls = _rust_and_python(monkeypatch, _pnp(precips_mm_monthly.flatten(), scale, 1981, 2010))
    assert calls == _PNP_KERNELS
    _assert_parity(rust, python)
    # the leading scale - 1 months have no complete sum on either path
    assert np.isnan(rust[: scale - 1]).all()


def test_pnp_matches_the_committed_fixture(
    monkeypatch,
    precips_mm_monthly,
    pnp_6month,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
):
    run = _pnp(
        precips_mm_monthly.flatten(),
        6,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        data_start=data_year_start_monthly,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _PNP_KERNELS
    _assert_parity(rust, python)
    # the same tolerance tests/test_indices.py holds the Python path to
    np.testing.assert_allclose(rust, pnp_6month, atol=0.01, equal_nan=True)


def test_pnp_spatial_block(
    monkeypatch, precips_mm_monthly, calibration_year_start_monthly, calibration_year_end_monthly
):
    rng = np.random.default_rng(9)
    block = precips_mm_monthly.flatten()[:, None, None] * rng.uniform(0.5, 1.5, (1, 3, 4))
    block[::37, 0, 0] = np.nan
    block[:, 1, 2] = 0.0
    run = _pnp(
        block,
        3,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        spatial_time_major=True,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _PNP_KERNELS
    _assert_parity(rust, python)
    # a cell whose calibration period has no positive value has no normal at all
    assert np.isnan(rust[:, 1, 2]).all()


def test_pnp_with_zero_and_missing_calendar_steps(monkeypatch):
    """A step with no positive calibration value has no normal, so it carries no percentage."""
    rng = np.random.default_rng(8)
    values = rng.gamma(2.0, 20.0, 12 * 40)
    values[6::12] = 0.0  # July never rains
    values[11::12][:5] = np.nan  # the first five Decembers are missing
    rust, python, calls = _rust_and_python(monkeypatch, _pnp(values, 1, 1990, 2029, data_start=1990))
    assert calls == _PNP_KERNELS
    _assert_parity(rust, python)
    assert np.isnan(rust[6::12]).all()
    assert np.isnan(rust[11::12][:5]).all()
    assert np.isfinite(rust[11::12][5:]).all()


def test_pnp_calibration_window_ending_in_a_partial_year(monkeypatch):
    """A trailing partial calibration period is padded with NaN before the kernel averages."""
    values = np.arange(481, dtype=float)
    rust, python, calls = _rust_and_python(monkeypatch, _pnp(values, 1, 1930, 1940, data_start=1900))
    assert calls == _PNP_KERNELS
    _assert_parity(rust, python)


def test_pnp_all_missing_series(monkeypatch):
    rust, python, calls = _rust_and_python(monkeypatch, _pnp(np.full(120, np.nan), 3, 1895, 1904))
    assert calls == _PNP_KERNELS
    _assert_parity(rust, python)
    assert np.isnan(rust).all()


def test_pnp_masked_input_stays_on_the_python_path(monkeypatch):
    """A fully masked input short-circuits before the kernels, as the Python path does."""
    rust, python, calls = _rust_and_python(monkeypatch, _pnp(np.ma.masked_all(120), 3, 1895, 1904))
    assert calls == set()
    assert np.ma.allequal(rust, python)


@pytest.mark.parametrize("days", [365, 366])
def test_pci_daily_year(monkeypatch, rain_mm_365, rain_mm_366, days):
    rainfall = (rain_mm_365 if days == 365 else rain_mm_366)[0]
    rust, python, calls = _rust_and_python(monkeypatch, lambda: indices.pci(rainfall))
    assert calls == {"pci"}
    _assert_parity(rust, python)
    assert np.isfinite(rust).all()


def test_pci_all_zero_rainfall(monkeypatch):
    rust, python, calls = _rust_and_python(monkeypatch, lambda: indices.pci(np.zeros(365)))
    assert calls == {"pci"}
    _assert_parity(rust, python)
    assert np.isnan(rust).all()


def test_pci_mask_or_length_stays_on_the_python_path(monkeypatch):
    all_masked = _rust_and_python(monkeypatch, lambda: indices.pci(np.ma.masked_all(366)))
    assert all_masked[2] == set()
    # a masked value is missing, so the year's totals are NaN on either path
    partially_masked = _rust_and_python(
        monkeypatch,
        lambda: indices.pci(np.ma.array(np.ones(366), mask=np.arange(366) < 31)),
    )
    assert partially_masked[2] == set()
    assert np.isnan(partially_masked[0]).all()
    assert np.isnan(partially_masked[1]).all()

    def invalid_length() -> np.ndarray:
        with pytest.raises(exceptions.InvalidArgumentError):
            indices.pci(np.ones(300))
        return np.empty(0)

    rust, python, calls = _rust_and_python(monkeypatch, invalid_length)
    assert calls == set()
    assert rust.size == python.size == 0


def test_pnp_with_infinite_values(monkeypatch):
    """inf inside the calibration period makes its normal inf; inf outside it survives the ratio."""
    rng = np.random.default_rng(11)
    values = rng.gamma(2.0, 20.0, 12 * 40)
    values[3] = np.inf  # an April the calibration period also averages
    values[12 * 20 + 4] = np.inf  # a May after the calibration period
    rust, python, calls = _rust_and_python(monkeypatch, _pnp(values, 1, 1990, 1999, data_start=1990))
    assert calls == _PNP_KERNELS
    _assert_parity(rust, python)
    # inside the period the normal is inf: inf/inf is missing and every other April is zero
    assert np.isnan(rust[3::12]).sum() == 1
    assert (rust[3::12] == 0.0).sum() == 39
    # outside it the normal is finite, so the ratio keeps the infinity
    assert np.isposinf(rust[12 * 20 + 4])


@pytest.mark.parametrize("days", [365, 366])
@pytest.mark.parametrize("case", ["monthly", "monthly-cancel", "monthly-inf", "annual"])
def test_pci_cancellation_matches_numpy_reduction_order(monkeypatch, days, case):
    rainfall = np.zeros(days)
    if case == "monthly":
        rainfall[:31] = np.r_[1e16, np.ones(28), -1e16, 0.0]
        rainfall[31] = 1.0
    elif case in {"monthly-cancel", "monthly-inf"}:
        rainfall[:3] = [1e16, -1e16, 1.0]
        rainfall[31] = 1.0 if case == "monthly-cancel" else -1.0
    else:
        rainfall[indices._PCI_MONTH_STARTS[days]] = np.r_[1e16, np.ones(10), -1e16]
    rust, python, calls = _rust_and_python(monkeypatch, lambda: indices.pci(rainfall))
    assert calls == {"pci"}
    _assert_parity(rust, python)


@pytest.mark.parametrize("layout", ["C", "F", "strided-C", "strided-F"])
@pytest.mark.parametrize("years", [7, 8, 40, 128, 129, 256])
@pytest.mark.parametrize("columns", [1, 12])
def test_pnp_normals_match_numpy_layout_and_cancellation(layout, years, columns):
    order = layout[-1]
    calibration = np.ones((years, columns), order=order)
    calibration[0] = 1e16
    calibration[-1] = -1e16
    # NaNs must occupy their zero-valued slots in NumPy's pairwise grouping,
    # rather than being dropped from the sequence before summation.
    calibration[2, 0] = np.nan
    if layout.startswith("strided"):
        storage = np.empty((years * 2, columns * 2), order=order)
        storage[::2, ::2] = calibration
        calibration = storage[::2, ::2]
    counts = np.sum(~np.isnan(calibration), axis=0)
    expected = np.nansum(calibration, axis=0) / np.maximum(counts, 1)
    expected = np.where((counts > 0) & (expected > 0.0), expected, np.nan)
    _assert_parity(native.pnp_normals(calibration), expected)


@pytest.mark.parametrize("columns", [0, 2])
@pytest.mark.parametrize("rows", [0, 1])
def test_pnp_percentages_reject_empty_normal_period(rows, columns):
    with pytest.raises(ValueError, match="normals.*at least one"):
        native.pnp_percentages(np.ones((rows, columns)), np.empty((0, columns)))


@pytest.mark.parametrize("days", [365, 366])
def test_pci_explicit_nan_days_preserve_python_validation(monkeypatch, days):
    rainfall = np.ones(days)
    rainfall[31] = np.nan
    for backend in (_Recorder(native), None):
        monkeypatch.setattr(compute, "_native", backend)
        with np.errstate(all="ignore"), pytest.raises(exceptions.InvalidArgumentError):
            indices.pci(rainfall)
        if backend is not None:
            assert backend.calls == set()
    missing = np.full(days, np.nan)
    rust, python, calls = _rust_and_python(monkeypatch, lambda: indices.pci(missing))
    assert calls == set()
    assert rust is missing
    assert python is missing


def test_pnp_partial_mask_is_prepared_before_native_dispatch(monkeypatch):
    values = np.ma.array(np.arange(1.0, 241.0), mask=np.arange(240) % 17 == 0)
    rust, python, calls = _rust_and_python(monkeypatch, _pnp(values, 1, 1895, 1914))
    assert calls == _PNP_KERNELS
    _assert_parity(rust, python)


# EDDI: the empirical rank count, Tukey plotting position, and Hastings inverse normal


def _eddi(values: np.ndarray, scale: int, start: int, end: int, **kwargs: Any) -> Callable[[], np.ndarray]:
    periodicity = kwargs.pop("periodicity", compute.Periodicity.monthly)
    data_start = kwargs.pop("data_start", _DATA_START)
    return lambda: indices.eddi(values, scale, data_start, start, end, periodicity, **kwargs)


def _daily_pet(years: int) -> np.ndarray:
    """A seasonal daily PET series, as ``tests/test_eddi.py`` builds for daily EDDI."""
    rng = np.random.default_rng(seed=42)
    day_of_year = np.tile(np.arange(366), years)
    seasonal_pattern = 100.0 + 50.0 * np.sin(2 * np.pi * day_of_year / 366)
    return seasonal_pattern + rng.uniform(-10.0, 10.0, years * 366)


def _spatial_pet_block(pet_thornthwaite_mm: np.ndarray) -> np.ndarray:
    """A (time, 2, 2) PET block whose cells rank differently from one another.

    Rescalings of one series are not enough: EDDI ranks within each cell, so a
    positive rescaling leaves every cell's probabilities identical.
    """
    series = np.asarray(pet_thornthwaite_mm).reshape(-1)
    rng = np.random.default_rng(seed=7)
    cells = series[:, None] * rng.uniform(0.4, 2.0, size=(1, 4))
    cells = cells + rng.normal(0.0, 25.0, size=cells.shape)
    cells[rng.random(cells.shape) < 0.02] = np.nan
    return cells.reshape(series.size, 2, 2)


@pytest.mark.parametrize("scale", [1, 3, 6])
def test_eddi_monthly_reference_series(monkeypatch, pet_thornthwaite_mm, scale):
    rust, python, calls = _rust_and_python(monkeypatch, _eddi(pet_thornthwaite_mm, scale, 1981, 2010))
    _assert_parity(rust, python)
    assert calls == _EDDI_KERNELS


def test_eddi_daily_series(monkeypatch):
    values = _daily_pet(19)
    rust, python, calls = _rust_and_python(
        monkeypatch,
        _eddi(values, 1, 1998, 2016, data_start=1998, periodicity=compute.Periodicity.daily),
    )
    _assert_parity(rust, python)
    assert calls == _EDDI_KERNELS


def test_eddi_missing_values_and_ties(monkeypatch, pet_thornthwaite_mm):
    values = pet_thornthwaite_mm.copy()
    values[::37] = np.nan
    values[100:140] = 42.0
    rust, python, calls = _rust_and_python(monkeypatch, _eddi(values, 3, 1981, 2010))
    _assert_parity(rust, python)
    assert calls == _EDDI_KERNELS


def test_eddi_short_calibration_and_missing_climatology(monkeypatch):
    values = np.random.default_rng(42).uniform(50.0, 150.0, 5 * 12)
    values[1::12] = np.nan  # every February missing: that calendar period has no ranking
    for start, end in ((2000, 2004), (2002, 2003)):
        rust, python, calls = _rust_and_python(monkeypatch, _eddi(values, 1, start, end, data_start=2000))
        _assert_parity(rust, python)
        assert calls == _EDDI_KERNELS


def test_eddi_spatial_time_major_block(monkeypatch, pet_thornthwaite_mm):
    block = _spatial_pet_block(pet_thornthwaite_mm)
    rust, python, calls = _rust_and_python(monkeypatch, _eddi(block, 3, 1981, 2010, spatial_time_major=True))
    _assert_parity(rust, python)
    assert calls == _EDDI_KERNELS


def test_eddi_ranking_matches_the_python_path_chunked_by_cells(monkeypatch, pet_thornthwaite_mm):
    """Rust walks every column at once; the chunked NumPy rank must agree cell for cell."""
    block = _spatial_pet_block(pet_thornthwaite_mm)
    monkeypatch.setattr(indices, "_EDDI_RANK_COMPARISON_ELEMENT_BUDGET", 4)
    rust, python, calls = _rust_and_python(monkeypatch, _eddi(block, 3, 1981, 2010, spatial_time_major=True))
    _assert_parity(rust, python)
    assert calls == _EDDI_KERNELS


def test_eddi_matches_the_committed_noaa_fixtures(monkeypatch):
    """The Rust path holds the NOAA PSL agreement, with the leading-scale pads in play.

    The fixtures calibrate from the data's first year, so in scales 1 and 3 the
    pads fall inside the calibration rows and the Rust kernel receives non-zero
    pad counts. The tolerance is the one ``tests/test_noaa_eddi_reference.py``
    holds the Python path to.
    """
    fixture_root = Path(__file__).parent / "fixture"
    fixtures = [(scale, fixture_root / f"noaa-eddi-{scale}month") for scale in (1, 3, 6)]
    missing = [directory.name for _, directory in fixtures if not (directory / "pet_input.npy").is_file()]
    assert not missing, f"Missing NOAA EDDI fixtures: {', '.join(missing)}"
    for scale, directory in fixtures:
        metadata = json.loads((directory / "metadata.json").read_text())
        run = _eddi(
            np.load(directory / "pet_input.npy"),
            scale,
            metadata["calibration_year_initial"],
            metadata["calibration_year_final"],
            data_start=metadata["data_start_year"],
        )
        rust, python, calls = _rust_and_python(monkeypatch, run)
        assert calls == _EDDI_KERNELS
        _assert_parity(rust, python)
        reference = np.load(directory / "eddi_reference.npy")
        valid = ~np.isnan(reference)
        np.testing.assert_allclose(rust[valid], reference[valid], rtol=1e-5, atol=1e-5)


def test_eddi_all_missing_never_reaches_a_kernel(monkeypatch, pet_thornthwaite_mm):
    series = np.full(pet_thornthwaite_mm.size, np.nan)
    block = np.full((pet_thornthwaite_mm.size, 2, 2), np.nan)
    for values, spatial in ((series, False), (block, True)):
        rust, python, calls = _rust_and_python(monkeypatch, _eddi(values, 3, 1981, 2010, spatial_time_major=spatial))
        _assert_parity(rust, python)
        assert calls == set()
        assert np.isnan(rust).all()


def test_native_eddi_kernels_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="climatology"):
        native.tukey_probabilities(np.ones((2, 3)), np.ones((2, 2)), np.ones(2))
    with pytest.raises(ValueError, match="pads"):
        native.tukey_probabilities(np.ones((2, 2)), np.ones((2, 2)), np.ones(3))


def test_native_eddi_kernels_handle_empty_negative_stride_blocks():
    empty = np.ndarray((0,), dtype=np.float64, buffer=np.empty(1), strides=(-1,))
    result = native.hastings_inverse_normal(empty)
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_eddi_hastings_covers_both_tails(monkeypatch):
    probabilities = np.concatenate(
        [
            np.linspace(0.0, 1.0, 257),
            1.0 - 10.0 ** np.arange(-16.0, -1.0),
            np.array([np.nan]),
        ]
    )
    rust, python, calls = _rust_and_python(monkeypatch, lambda: indices._hastings_inverse_normal(probabilities))
    _assert_parity(rust, python)
    assert calls == {"hastings_inverse_normal"}
