"""Parity of the Rust gamma kernels with the Python reference implementation.

Each test computes the same result twice through the public or compute-level API:
once with ``compute._native`` replaced by a recorder around the Rust extension, and
once with it set to None, which runs the pure-Python reference. The recorder proves
the first run reached the Rust kernels, so the comparison is never Python against
Python. The contract is ``rtol = atol = 1e-10`` with matching NaN positions.

Skipped when the extension is not built (``uv run maturin develop --release``),
unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1`` is set, as in CI's native legs, where a
missing extension is a collection error.
"""

import subprocess
import sys
import warnings
from collections.abc import Callable
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
    [("gamma_parameters", 0), *[("gamma_probabilities", i) for i in range(4)], ("norm_ppf", 0)],
)
def test_native_boundary_rejects_unaligned_arrays(layout, kernel, argument_index):
    arguments = {
        "gamma_parameters": [np.arange(1.0, 25.0).reshape(2, 12)],
        "gamma_probabilities": [np.ones((2, 12)), np.full(12, 2.0), np.ones(12), np.zeros(12)],
        "norm_ppf": [np.full((2, 12), 0.5)],
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


@pytest.mark.skipif(sys.version_info < (3, 14), reason="context-aware warnings require Python 3.14")
def test_context_aware_warning_filters_stay_on_python():
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "context_aware_warnings=1",
            "-c",
            """
import sys
import warnings
import numpy as np
from climate_indices import compute, _native
assert sys.flags.context_aware_warnings
class NoNativeFit:
    def gamma_parameters(self, *args):
        raise AssertionError("native fit called")
compute._native = NoNativeFit()
with np.errstate(all="ignore"):
    assert not compute._native_float64(np.ones((30, 12)))
with warnings.catch_warnings(), np.errstate(divide="warn"):
    warnings.simplefilter("error", RuntimeWarning)
    try:
        compute.gamma_parameters(np.ones((30, 12)), 1895, 1895, 1924, compute.Periodicity.monthly)
    except RuntimeWarning as error:
        assert "divide by zero" in str(error)
    else:
        raise AssertionError("RuntimeWarning suppressed")
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
