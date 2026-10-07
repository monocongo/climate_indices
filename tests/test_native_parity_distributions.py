"""Parity of the Rust Pearson Type III and generalized logistic kernels with Python.

The companion of ``test_native_parity.py`` for the L-moment distributions: SPI with
Pearson Type III, and SPEI with Pearson Type III or the log-logistic (GLO). Each case
runs once through the Rust kernels, proved by the recorder, and once through the
pure-Python reference, at ``rtol = atol = 1e-10`` with matching NaN positions.

Skipped when the extension is not built, unless ``CLIMATE_INDICES_REQUIRE_NATIVE=1``
is set, as in CI's native legs (see ``test_native_parity.py``).
"""

import warnings

import numpy as np
import pytest
import scipy.special
import scipy.stats

from climate_indices import compute, indices, lmoments
from tests import conftest
from tests.test_native_parity import _DATA_START, _assert_parity, _Recorder, _rust_and_python, _with_zeros

native = conftest.import_native()

_PEARSON = {"pearson_parameters", "pearson_cdf"}
_GLO = {"loglogistic_parameters", "loglogistic_cdf"}
_MONTHLY = compute.Periodicity.monthly


def _spi_pearson(values: np.ndarray, scale: int, start: int, end: int, **kwargs):
    periodicity = kwargs.pop("periodicity", _MONTHLY)
    data_start = kwargs.pop("data_start", _DATA_START)
    return lambda: indices.spi(
        values, scale, indices.Distribution.pearson, data_start, start, end, periodicity, **kwargs
    )


def _spei(precips, pet, scale, distribution, **kwargs):
    return lambda: indices.spei(precips, pet, scale, distribution, _MONTHLY, _DATA_START, 1981, 2010, **kwargs)


# --- SPI with Pearson Type III -------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 3, 6, 12])
def test_spi_pearson_monthly_reference_series(monkeypatch, precips_mm_monthly, scale):
    rust, python, calls = _rust_and_python(monkeypatch, _spi_pearson(precips_mm_monthly, scale, 1981, 2010))
    assert calls == _PEARSON
    _assert_parity(rust, python)
    assert np.isnan(rust[: scale - 1]).all()


def test_spi_pearson_matches_the_committed_fixture(
    monkeypatch,
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    spi_6_month_pearson3,
):
    run = _spi_pearson(
        precips_mm_monthly.flatten(),
        6,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        data_start=data_year_start_monthly,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _PEARSON
    _assert_parity(rust, python)
    # the same tolerance tests/test_indices.py holds the Python path to
    np.testing.assert_allclose(rust, spi_6_month_pearson3, atol=0.01, equal_nan=True)


@pytest.mark.parametrize("zero_handling", ["classic", "center_of_mass", "mean_zero"])
@pytest.mark.parametrize("output_scale", ["normal", "probability", "bounded"])
def test_spi_pearson_zeros_zero_handling_and_output_scales(
    monkeypatch, precips_mm_monthly, zero_handling, output_scale
):
    dry = _with_zeros(precips_mm_monthly, 0.25, seed=11)
    run = _spi_pearson(dry, 3, 1981, 2010, zero_handling=zero_handling, output_scale=output_scale)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _PEARSON
    _assert_parity(rust, python)


def test_spi_pearson_trace_values(monkeypatch, precips_mm_monthly):
    """A value below 0.0005 is a trace value when the step has no zero mass."""
    traced = precips_mm_monthly.copy()
    rng = np.random.default_rng(12)
    traced[rng.random(traced.shape) < 0.08] = 0.0003
    rust, python, calls = _rust_and_python(monkeypatch, _spi_pearson(traced, 1, 1981, 2010))
    assert calls == _PEARSON
    _assert_parity(rust, python)


def test_spi_pearson_with_missing_values(monkeypatch, precips_mm_monthly):
    gappy = precips_mm_monthly.copy().flatten()
    gappy[np.random.default_rng(13).random(gappy.size) < 0.15] = np.nan
    gappy[:30] = np.nan
    rust, python, _ = _rust_and_python(monkeypatch, _spi_pearson(gappy, 6, 1981, 2010))
    _assert_parity(rust, python)


def test_spi_pearson_fewer_than_four_non_zero_values_in_one_month(monkeypatch, precips_mm_monthly):
    """One calendar month with three non-zero calibration values fails its fit on both paths."""
    sparse = precips_mm_monthly.copy()
    sparse[:, 6] = 0.0
    sparse[[90, 95, 100], 6] = [12.0, 30.0, 8.0]
    rust, python, calls = _rust_and_python(monkeypatch, _spi_pearson(sparse, 1, 1981, 2010))
    assert calls == _PEARSON
    _assert_parity(rust, python)


@pytest.mark.parametrize("fraction", [0.2, 0.6])
def test_pearson_parameters_match_the_python_fit(monkeypatch, precips_mm_monthly, fraction):
    dry = _with_zeros(precips_mm_monthly, fraction, seed=14)
    run = lambda: np.stack(  # noqa: E731
        compute.pearson_parameters(dry, _DATA_START, 1981, 2010, _MONTHLY)
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"pearson_parameters"}
    _assert_parity(rust, python)


def test_pearson_parameters_daily(monkeypatch, precips_mm_daily, data_year_start_daily):
    run = lambda: np.stack(  # noqa: E731
        compute.pearson_parameters(
            precips_mm_daily,
            data_year_start_daily,
            data_year_start_daily,
            data_year_start_daily + 14,
            compute.Periodicity.daily,
        )
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"pearson_parameters"}
    _assert_parity(rust, python)


def test_the_gamma_fallback_fires_on_the_same_inputs(monkeypatch, precips_mm_monthly):
    """A block whose months mostly fail the Pearson fit is refitted with gamma on both paths."""
    # nearly dry through the calibration years 1981-2010 (rows 86-115) and wet elsewhere
    sparse = precips_mm_monthly.copy()
    sparse[86:116] = _with_zeros(sparse[86:116], 0.97, seed=15)
    healthy = precips_mm_monthly

    def run(values):
        def fallback() -> tuple[bool, np.ndarray]:
            standardized, used, _ = compute._fit_pearson_with_fallback(
                values.copy(), indices.Distribution.pearson, _DATA_START, 1981, 2010, _MONTHLY, None, "test"
            )
            return used, standardized

        return fallback

    for values, expected in ((sparse, True), (healthy, False)):
        recorder = _Recorder(native)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            monkeypatch.setattr(compute, "_native", recorder)
            rust_used, rust = run(values)()
            monkeypatch.setattr(compute, "_native", None)
            python_used, python = run(values)()
        assert rust_used is python_used is expected
        assert {"pearson_parameters", "pearson_cdf"} <= recorder.calls
        _assert_parity(rust, python)


# --- SPEI ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("distribution", "kernels"),
    [(indices.Distribution.pearson, _PEARSON), (indices.Distribution.loglogistic, _GLO)],
    ids=["pearson", "loglogistic"],
)
@pytest.mark.parametrize("scale", [1, 3, 6, 12])
def test_spei_at_several_scales(monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, distribution, kernels, scale):
    run = _spei(precips_mm_monthly, pet_thornthwaite_mm, scale, distribution)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == kernels
    _assert_parity(rust, python)


@pytest.mark.parametrize("distribution", [indices.Distribution.pearson, indices.Distribution.loglogistic])
@pytest.mark.parametrize("output_scale", ["probability", "bounded"])
def test_spei_output_scales(monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, distribution, output_scale):
    run = _spei(precips_mm_monthly, pet_thornthwaite_mm, 3, distribution, output_scale=output_scale)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls
    _assert_parity(rust, python)


def test_spei_loglogistic_with_zero_precipitation(
    monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, data_year_start_monthly, data_year_end_monthly
):
    """SPEI keeps zeros as ordinary values in its GLO fit, so a dry series is parity-tested too."""
    dry = _with_zeros(precips_mm_monthly, 0.3, seed=16)
    run = lambda: indices.spei(  # noqa: E731
        dry,
        pet_thornthwaite_mm,
        3,
        indices.Distribution.loglogistic,
        _MONTHLY,
        data_year_start_monthly,
        data_year_start_monthly,
        data_year_end_monthly,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == _GLO
    _assert_parity(rust, python)


def test_spei_pearson_with_negative_skew(monkeypatch, precips_mm_monthly):
    """PET above precipitation by a gamma amount leaves a negatively skewed water balance."""
    gamma_excess = np.random.default_rng(17).gamma(2.0, 20.0, precips_mm_monthly.shape)
    pet = precips_mm_monthly + gamma_excess
    skews = compute.pearson_parameters(precips_mm_monthly - pet + 1000.0, _DATA_START, 1981, 2010, _MONTHLY)[3]
    assert (skews < 0).all()
    rust, python, calls = _rust_and_python(monkeypatch, _spei(precips_mm_monthly, pet, 1, indices.Distribution.pearson))
    assert calls == _PEARSON
    _assert_parity(rust, python)


# --- the transforms with caller-supplied parameters ----------------------------------


def test_pearson_support_limits_with_negative_and_positive_skew(monkeypatch):
    """Values beyond the lower (positive skew) or upper (negative skew) limit take the sentinels."""
    values = np.tile(np.linspace(-20.0, 150.0, 12), (30, 1))
    skews = np.array([-1.0, -0.4, 1.0, 0.4, 0.0, 2e-5, -2e-5, 1e-5, 1.5, -1.5, 0.05, -0.05])
    run = lambda: compute.transform_fitted_pearson(  # noqa: E731
        values,
        _DATA_START,
        _DATA_START,
        _DATA_START + 29,
        _MONTHLY,
        np.zeros(12),
        np.full(12, 40.0),
        np.full(12, 25.0),
        skews,
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"pearson_cdf"}
    _assert_parity(rust, python)
    # both limits are exercised: the sentinels 0.0005 and 0.9995 map to the same z-scores
    assert np.isclose(python, scipy.stats.norm.ppf(0.0005)).any()
    assert np.isclose(python, scipy.stats.norm.ppf(0.9995)).any()


def test_invalid_pearson_parameters_give_the_same_missing_values(monkeypatch):
    values = np.tile(np.linspace(1.0, 100.0, 12), (30, 1))
    run = lambda: compute.transform_fitted_pearson(  # noqa: E731
        values,
        _DATA_START,
        _DATA_START,
        _DATA_START + 29,
        _MONTHLY,
        np.zeros(12),
        np.zeros(12),
        np.zeros(12),
        np.zeros(12),
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"pearson_cdf"}
    _assert_parity(rust, python)


@pytest.mark.parametrize("shape", [-0.6, -0.2, 0.0, 5e-7, 0.3, 0.9, float("nan")])
def test_loglogistic_cdf_beyond_its_support(monkeypatch, shape):
    """A finite value past the support maps to probability 0 or 1; invalid parameters to NaN."""
    values = np.tile(np.linspace(-200.0, 400.0, 12), (30, 1))
    run = lambda: compute.transform_fitted_loglogistic(  # noqa: E731
        values,
        _DATA_START,
        _DATA_START,
        _DATA_START + 29,
        _MONTHLY,
        np.full(12, 50.0),
        np.full(12, 20.0),
        np.full(12, shape),
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"loglogistic_cdf"}
    _assert_parity(rust, python)


# --- spatial blocks ------------------------------------------------------------------


def _spatial_block(precips_mm_monthly: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = precips_mm_monthly.flatten()[:, None, None] * rng.uniform(0.5, 1.5, (1, 3, 4))
    block = _with_zeros(block, 0.1, seed=seed + 1)
    block[:, 0, 0] = np.nan
    block[:, 1, 1] = 0.0
    return block


def test_spi_pearson_spatial_block(monkeypatch, precips_mm_monthly):
    block = _spatial_block(precips_mm_monthly, 20)
    run = _spi_pearson(block, 6, 1981, 2010, spatial_time_major=True)
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls >= _PEARSON
    _assert_parity(rust, python)


@pytest.mark.parametrize("distribution", [indices.Distribution.pearson, indices.Distribution.loglogistic])
def test_spei_spatial_block(monkeypatch, precips_mm_monthly, pet_thornthwaite_mm, distribution):
    precips = _spatial_block(precips_mm_monthly, 22)
    pet = pet_thornthwaite_mm.flatten()[:, None, None] * np.ones((1, 3, 4))
    run = lambda: indices.spei(  # noqa: E731
        precips, pet, 3, distribution, _MONTHLY, _DATA_START, 1981, 2010, spatial_time_major=True
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls
    _assert_parity(rust, python)


def test_loglogistic_parameters_spatial_failure_count_matches(monkeypatch, precips_mm_monthly):
    """The failed-fit count that drives the high-failure-rate warning is the same on both paths."""
    block = precips_mm_monthly[:, :, None] * np.ones((1, 1, 3))
    block[:, :, 0] = np.nan
    block[:, 4, 1] = np.nan
    run = lambda: np.stack(  # noqa: E731
        compute.loglogistic_parameters(block, _DATA_START, 1981, 2010, _MONTHLY)
    )
    rust, python, calls = _rust_and_python(monkeypatch, run)
    assert calls == {"loglogistic_parameters"}
    _assert_parity(rust, python)


# --- routing -------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf, 1e101])
def test_unbounded_calibration_values_keep_the_python_fit(monkeypatch, precips_mm_monthly, bad_value):
    """An infinity or an overflowing weight makes the L-moments NaN; the Python fits differ on that."""
    values = precips_mm_monthly.copy()
    values[90, 3] = bad_value
    for fit in (compute.pearson_parameters, compute.loglogistic_parameters):
        recorder = _Recorder(native)
        monkeypatch.setattr(compute, "_native", recorder)
        with np.errstate(all="ignore"):
            fit(values, _DATA_START, 1981, 2010, _MONTHLY)
        assert not recorder.calls, fit.__name__


def test_float32_and_masked_blocks_keep_the_python_fit(monkeypatch, precips_mm_monthly):
    recorder = _Recorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with np.errstate(all="ignore"):
        compute.pearson_parameters(precips_mm_monthly.astype(np.float32), _DATA_START, 1981, 2010, _MONTHLY)
        compute.loglogistic_parameters(np.ma.masked_less(precips_mm_monthly, 1.0), _DATA_START, 1981, 2010, _MONTHLY)
    assert not recorder.calls


def test_numpy_error_policy_keeps_the_python_fit(monkeypatch, precips_mm_monthly):
    recorder = _Recorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute.pearson_parameters(precips_mm_monthly, _DATA_START, 1981, 2010, _MONTHLY)
        compute.loglogistic_parameters(precips_mm_monthly, _DATA_START, 1981, 2010, _MONTHLY)
    assert not recorder.calls, "the default NumPy error policy is not all-ignore"


def test_climate_indices_warnings_are_unchanged(monkeypatch, precips_mm_monthly):
    """The failure-rate and data-quality warnings come from Python on both paths."""
    gappy = _with_zeros(precips_mm_monthly, 0.9, seed=30).flatten()
    gappy[: 12 * 100] = np.nan

    def warning_types(backend):
        recorder = _Recorder(backend) if backend is not None else None
        monkeypatch.setattr(compute, "_native", recorder)
        with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
            warnings.simplefilter("always")
            _spi_pearson(gappy, 3, 1990, 2017)()
        categories = sorted(
            {
                w.category
                for w in caught
                if issubclass(w.category, Warning) and "climate_indices" in w.category.__module__
            },
            key=lambda category: category.__name__,
        )
        return categories, (recorder.calls if recorder is not None else set())

    rust, rust_calls = warning_types(native)
    python, _ = warning_types(None)
    assert rust == python
    assert rust, "the case is meant to trigger at least one data-quality warning"
    assert rust_calls, "the native arm must actually reach the Rust kernels"


# --- the kernels against their SciPy and Python primitives ---------------------------


def test_pearson_cdf_matches_scipy_across_parameters():
    rng = np.random.default_rng(40)
    count = 30_000
    skews = np.concatenate([rng.uniform(-3.0, 3.0, count), rng.uniform(-3e-5, 3e-5, 500), [0.0, 1.6e-5, -1.6e-5]])
    locs = rng.uniform(-50.0, 50.0, skews.size)
    scales = np.exp(rng.uniform(-2.0, 4.0, skews.size))
    values = locs + scales * rng.normal(0.0, 2.5, skews.size)
    ours = native.pearson_cdf(values[None, :], skews, locs, scales)[0]
    np.testing.assert_allclose(
        ours, scipy.stats.pearson3.cdf(values, skews, loc=locs, scale=scales), rtol=1e-10, atol=1e-10
    )


def test_pearson_cdf_edge_arguments_match_scipy():
    values = np.array([0.0, 1.0, np.inf, -np.inf, np.nan, 5.0, 5.0, 5.0, 5.0, 5.0])
    skews = np.array([1.0, 1.0, 1.0, -1.0, 1.0, np.nan, np.inf, 1.0, 1e300, 1e-300])
    locs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0])
    scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    scales[2] = 0.0
    expected = scipy.stats.pearson3.cdf(values, skews, loc=locs, scale=scales)
    ours = native.pearson_cdf(values[None, :], skews, locs, scales)[0]
    np.testing.assert_allclose(ours, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.parametrize("sample_size", [4, 5, 12, 30, 61])
def test_fits_match_the_python_lmoments_modules(sample_size):
    """The Rust fits against ``lmoments.fit`` and ``fit_glo`` on skewed, symmetric, and tied samples."""
    rng = np.random.default_rng(50 + sample_size)
    columns = [
        rng.gamma(0.8, 30.0, sample_size),
        rng.gamma(30.0, 3.0, sample_size),
        200.0 - rng.gamma(2.0, 20.0, sample_size),
        rng.normal(100.0, 15.0, sample_size),
        np.round(rng.gamma(2.0, 10.0, sample_size)),
    ]
    block = np.stack(columns, axis=1)
    p0, locs, scales, skews, pearson_valid = native.pearson_parameters(block)
    glo_locs, glo_scales, glo_shapes, glo_valid = native.loglogistic_parameters(block)
    for index, column in enumerate(columns):
        try:
            expected = lmoments.fit(column)
            assert pearson_valid[index]
            np.testing.assert_allclose(
                [locs[index], scales[index], skews[index]],
                [expected["loc"], expected["scale"], expected["skew"]],
                rtol=1e-10,
                atol=1e-10,
            )
        except ValueError:
            assert not pearson_valid[index]
        try:
            expected = lmoments.fit_glo(column)
            assert glo_valid[index]
            np.testing.assert_allclose(
                [glo_locs[index], glo_scales[index], glo_shapes[index]],
                [expected["loc"], expected["scale"], expected["shape"]],
                rtol=1e-10,
                atol=1e-10,
            )
        except ValueError:
            assert not glo_valid[index]
    assert (p0 == 0.0).all()


def test_small_skew_cdf_is_the_normal_cdf_in_both_tails():
    """A zero skew is the normal CDF, i.e. Cephes ``ndtr``, including its far tails."""
    z = np.array([-40.0, -8.0, -1.0, -0.1, 0.0, 0.1, 1.0, 8.0, 40.0])
    ours = native.pearson_cdf(z[None, :], np.zeros(z.size), np.zeros(z.size), np.ones(z.size))[0]
    np.testing.assert_allclose(ours, scipy.special.ndtr(z), rtol=1e-12, atol=0.0)


def test_loglogistic_cdf_matches_the_numpy_formula():
    rng = np.random.default_rng(60)
    count = 20_000
    shapes = np.concatenate([rng.uniform(-0.95, 0.95, count), [0.0, 1e-6, -1e-6, 2e-6, np.nan, 1.5]])
    locs = rng.uniform(-50.0, 50.0, shapes.size)
    scales = np.exp(rng.uniform(-2.0, 4.0, shapes.size))
    values = locs + scales * rng.normal(0.0, 3.0, shapes.size)
    ours = native.loglogistic_cdf(values[None, :], locs, scales, shapes)[0]
    with np.errstate(all="ignore"):
        z = (values - locs) / scales
        y = np.where(np.abs(shapes) <= 1e-6, z, -np.log(np.maximum(0.0, 1.0 - shapes * z)) / shapes)
        expected = np.clip(1.0 / (1.0 + np.exp(-y)), 0.0, 1.0)
    np.testing.assert_allclose(ours, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


# --- the native boundary -------------------------------------------------------------


@pytest.mark.parametrize(
    ("kernel", "arguments"),
    [
        ("pearson_cdf", [np.ones((2, 12)), np.full(12, 0.5), np.zeros(12), np.ones(12)]),
        ("loglogistic_cdf", [np.ones((2, 12)), np.zeros(12), np.ones(12), np.full(12, 0.1)]),
    ],
)
def test_cdf_kernels_reject_a_parameter_of_the_wrong_length(kernel, arguments):
    arguments[2] = np.ones(11)
    with pytest.raises(ValueError, match="11"):
        getattr(native, kernel)(*arguments)


@pytest.mark.parametrize("shape", [(0, 12), (2, 0), (0, 0)])
def test_kernels_accept_empty_blocks(shape):
    empty = np.empty(shape)
    parameters = np.ones(shape[1])
    *pearson, valid = native.pearson_parameters(empty)
    assert all(array.shape == (shape[1],) for array in pearson) and valid.shape == (shape[1],)
    *glo, valid = native.loglogistic_parameters(empty)
    assert all(array.shape == (shape[1],) for array in glo) and valid.shape == (shape[1],)
    assert not valid.any()
    assert native.pearson_cdf(empty, parameters, parameters, parameters).shape == shape
    assert native.loglogistic_cdf(empty, parameters, parameters, parameters).shape == shape


@pytest.mark.parametrize("layout", ["packed", "offset"])
def test_kernels_reject_unaligned_arrays(layout):
    def unaligned_like(original: np.ndarray) -> np.ndarray:
        if layout == "packed":
            array = np.zeros(original.shape, dtype=[("value", "f8"), ("flag", "u1")])["value"]
        else:
            array = np.ndarray(original.shape, dtype=np.float64, buffer=bytearray(original.nbytes + 1), offset=1)
        array[:] = original
        assert not array.flags.aligned
        return array

    block = np.arange(1.0, 25.0).reshape(2, 12)
    parameters = np.full(12, 0.5)
    for kernel in (native.pearson_parameters, native.loglogistic_parameters):
        with pytest.raises(ValueError, match="unaligned float64 array"):
            kernel(unaligned_like(block))
    for kernel in (native.pearson_cdf, native.loglogistic_cdf):
        for position in range(4):
            arguments = [block, parameters, parameters, parameters]
            arguments[position] = unaligned_like(arguments[position])
            with pytest.raises(ValueError, match="unaligned float64 array"):
                kernel(*arguments)
