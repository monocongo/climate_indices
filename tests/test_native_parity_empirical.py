"""Exact native parity for the empirical distribution and its dispatch guards."""

from types import SimpleNamespace

import numpy as np
import pytest

from climate_indices import compute, indices
from tests.test_native_parity import _Recorder, _rust_and_python, native


@pytest.mark.parametrize("periodicity", list(compute.Periodicity))
@pytest.mark.parametrize("cells", [(), (2, 3)])
@pytest.mark.parametrize("output_scale", compute.OUTPUT_SCALES)
def test_empirical_transform_is_bit_identical(monkeypatch, periodicity, cells, output_scale):
    values = np.random.default_rng(1435).integers(-5, 6, size=(14, periodicity.value, *cells)).astype(float)
    values[0, 0] = np.nan  # Nine valid calibration values fail; ten at the next step succeed.
    values[-1, 1] = np.nan
    values[-2, 2] = -np.inf
    values[-1, 2] = np.inf
    values[0, 3] = -0.0
    values[1, 3] = 0.0
    values[2, 3] = np.nextafter(0.0, 1.0)
    original = values.copy()
    rust, python, calls = _rust_and_python(
        monkeypatch,
        lambda: compute.transform_fitted_empirical(values, 2000, 2000, 2009, periodicity, output_scale),
    )
    assert calls == {"empirical_transform"}
    # Integer counts and the Cephes ndtri use the reference's exact arithmetic.
    np.testing.assert_array_equal(rust, python)
    np.testing.assert_array_equal(values, original)
    assert np.isnan(rust[:, 0]).all()
    assert np.isfinite(rust[:-1, 1]).all()


@pytest.mark.parametrize("output_scale", compute.OUTPUT_SCALES)
def test_empirical_extrema_ties_and_infinities_have_exact_positions(monkeypatch, output_scale):
    sample = [-np.inf, -2.0, -2.0, -0.0, 0.0, 1.0, 2.0, 3.0, np.inf, np.inf, np.nan]
    values = np.tile(sample, (12, 1)).T.copy()
    rust, python, calls = _rust_and_python(
        monkeypatch,
        lambda: compute.transform_fitted_empirical(values, 2000, 2000, 2010, compute.Periodicity.monthly, output_scale),
    )
    assert calls == {"empirical_transform"}
    np.testing.assert_array_equal(rust, python)
    if output_scale == "probability":
        np.testing.assert_array_equal(rust[:, 0], np.array([2, 4, 4, 6, 6, 7, 8, 9, 11, 11, np.nan]) / 12)


@pytest.mark.parametrize("layout", ["fortran", "reversed", "sliced"])
def test_empirical_strided_blocks(monkeypatch, layout):
    values = np.random.default_rng(1).normal(size=(15, 12, 4))
    if layout == "fortran":
        values = np.asfortranarray(values)
    elif layout == "reversed":
        values = values[::-1, :, ::-1]
    else:
        values = values[:, :, ::2]
    rust, python, calls = _rust_and_python(monkeypatch, lambda: compute._empirical_fit(values, values[:10], "normal"))
    assert calls == {"empirical_transform"}
    np.testing.assert_array_equal(rust, python)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_empirical_non_float64_keeps_python(monkeypatch, dtype):
    values = np.arange(120).reshape(10, 12).astype(dtype)
    rust, python, calls = _rust_and_python(monkeypatch, lambda: compute._empirical_fit(values, values, "probability"))
    assert not calls
    np.testing.assert_array_equal(rust, python)


def test_empirical_broadcast_sample_keeps_python(monkeypatch):
    values = np.zeros((10, 12, 3))
    sample = np.zeros((10, 12, 1))
    rust, python, calls = _rust_and_python(monkeypatch, lambda: compute._empirical_fit(values, sample, "probability"))
    assert not calls
    np.testing.assert_array_equal(rust, python)


@pytest.mark.parametrize("policy", ["warn", "raise"])
def test_empirical_reporting_policy_keeps_python(monkeypatch, policy):
    values = np.ones((10, 12))
    recorder = _Recorder(native)
    monkeypatch.setattr(compute, "_native", recorder)
    with np.errstate(all=policy):
        result = compute._empirical_fit(values, values, "probability")
    assert not recorder.calls
    np.testing.assert_array_equal(result, np.full(values.shape, 11 / 12))


def test_empirical_stale_extension_keeps_python(monkeypatch):
    values = np.ones((10, 12))
    monkeypatch.setattr(compute, "_native", SimpleNamespace())
    with np.errstate(all="ignore"):
        result = compute._empirical_fit(values, values, "probability")
    np.testing.assert_array_equal(result, np.full(values.shape, 11 / 12))


def test_empirical_native_error_is_not_retried(monkeypatch):
    def fail(*args):
        raise ValueError("native failure")

    monkeypatch.setattr(compute, "_native", SimpleNamespace(empirical_transform=fail))
    values = np.ones((10, 12))
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="native failure"):
        compute._empirical_fit(values, values, "probability")


def test_empirical_binding_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        native.empirical_transform(np.ones((10, 2)), np.ones((1, 3)), False)


@pytest.mark.parametrize("shape", [(0, 12), (10, 0)])
def test_empirical_binding_accepts_empty_blocks(shape):
    values = np.empty(shape)
    result = native.empirical_transform(values, values, True)
    assert result.shape == shape
    assert result.flags["C_CONTIGUOUS"]
    assert result.flags["WRITEABLE"]


def _bits(values: np.ndarray) -> np.ndarray:
    """Exact float64 bit patterns, so two NaN payloads still compare equal."""
    return values.view(np.uint64)


def test_empirical_binding_result_is_independent_c_order_float64():
    """The in-place rank returns its own buffer, not a view of either input."""
    calibration = np.random.default_rng(7).normal(size=(20, 6))
    values = np.random.default_rng(8).normal(size=(24, 6))
    values[0, 0] = np.nan
    calibration_before, values_before = calibration.copy(), values.copy()
    result = native.empirical_transform(calibration, values, True)
    assert result.dtype == np.float64
    assert result.shape == values.shape
    assert result.flags["C_CONTIGUOUS"]
    assert result.flags["WRITEABLE"]
    assert not np.shares_memory(result, calibration)
    assert not np.shares_memory(result, values)
    np.testing.assert_array_equal(_bits(calibration), _bits(calibration_before))
    np.testing.assert_array_equal(_bits(values), _bits(values_before))
    assert np.isnan(result[0, 0])


def test_empirical_binding_aliased_inputs_stay_unchanged():
    values = np.random.default_rng(9).normal(size=(15, 4))
    values[0, 0] = np.nan
    before = values.copy()
    result = native.empirical_transform(values, values, False)
    np.testing.assert_array_equal(_bits(values), _bits(before))
    assert not np.shares_memory(result, values)
    expected = native.empirical_transform(before, before, False)
    np.testing.assert_array_equal(_bits(result), _bits(expected))


def test_empirical_binding_normalizes_a_fortran_input():
    """A non-standard layout keeps the C-ordered result the allocating kernel produced."""
    calibration = np.asfortranarray(np.random.default_rng(10).normal(size=(20, 5)))
    values = np.asfortranarray(np.random.default_rng(11).normal(size=(24, 5)))
    result = native.empirical_transform(calibration, values, True)
    assert result.flags["C_CONTIGUOUS"]
    expected = native.empirical_transform(np.ascontiguousarray(calibration), np.ascontiguousarray(values), True)
    np.testing.assert_array_equal(_bits(result), _bits(expected))


def test_empirical_binding_normalizes_a_reversed_input():
    """A negative-stride input is read in logical order and copied out C-ordered."""
    calibration = np.random.default_rng(12).normal(size=(20, 8))
    values = np.random.default_rng(13).normal(size=(24, 8))
    expected = native.empirical_transform(calibration, values, True)
    result = native.empirical_transform(calibration[::-1], values[::-1], True)
    assert result.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(_bits(result), _bits(expected[::-1]))


@pytest.mark.parametrize("argument", [0, 1])
def test_empirical_binding_rejects_unaligned_inputs(argument):
    values = np.ones((10, 12))
    unaligned = np.ndarray(values.shape, dtype=np.float64, buffer=bytearray(values.nbytes + 1), offset=1)
    arguments = [values, values]
    arguments[argument] = unaligned
    with pytest.raises(ValueError, match="unaligned float64 array"):
        native.empirical_transform(*arguments, False)
    with np.errstate(all="ignore"):
        assert compute._native_empirical_fit(arguments[0], arguments[1], "normal") is None


def test_empirical_masked_array_keeps_python():
    values = np.ma.array(np.ones((10, 12)), mask=False)
    with np.errstate(all="ignore"):
        assert compute._native_empirical_fit(values, values, "normal") is None


def test_signed_standardized_index_reaches_empirical_kernel(monkeypatch):
    values = np.random.default_rng(1435).normal(size=40 * 12)
    rust, python, calls = _rust_and_python(
        monkeypatch,
        lambda: indices.standardized_index(
            values,
            3,
            indices.Distribution.empirical,
            1981,
            1983,
            2019,
            compute.Periodicity.monthly,
            clip_negatives=False,
        ),
    )
    assert "empirical_transform" in calls
    np.testing.assert_array_equal(rust, python)
