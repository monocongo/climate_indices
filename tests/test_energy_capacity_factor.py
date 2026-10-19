"""Wind capacity-factor specification checks, not observed-generation validation."""

from fractions import Fraction
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr

from climate_indices import energy, performance
from climate_indices.energy import _wind
from climate_indices.exceptions import InputTypeError, InvalidArgumentError


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_generic_curve_regions_and_exact_boundaries(dtype: Any) -> None:
    wind = np.array([0, 3, 3.5, 5, 8, 12, 13, 20, 25, 30], dtype=dtype)
    expected = (
        [0, 0, 0]
        + [float((Fraction(v) ** 3 - Fraction("3.5") ** 3) / (13**3 - Fraction("3.5") ** 3)) for v in [5, 8, 12]]
        + [1, 1, 0, 0]
    )
    result = energy.wind_capacity_factor(wind, "generic")
    # Exact rational arithmetic is independent of the float64 curve evaluation.
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result[[0, 1, 2, 6, 7, 8, 9]], [0, 0, 0, 1, 1, 0, 0])


def test_curve_boundary_neighbors() -> None:
    wind = np.array([np.nextafter(3.5, 0), np.nextafter(3.5, 13), np.nextafter(13, 0), np.nextafter(25, 0), 25])
    result = energy.wind_capacity_factor(wind, "generic")
    assert result[0] == 0
    assert 0 < result[1] < result[2] < 1
    np.testing.assert_array_equal(result[3:], [1, 0])


@pytest.mark.parametrize("cut_in,rated,cut_out", [(0, 2, 4), (1e300, 2e300, 3e300), (1, np.nextafter(1.0, 2), 2)])
def test_custom_thresholds_avoid_overflow_and_cancellation(cut_in: float, rated: float, cut_out: float) -> None:
    wind = np.array([cut_in, cut_in + (rated - cut_in) / 2, rated, cut_out])
    ci, r = Fraction(cut_in), Fraction(rated)
    expected = [float((Fraction(float(v)) ** 3 - ci**3) / (r**3 - ci**3)) for v in wind[:2]] + [1, 0]
    with np.errstate(all="raise"):
        result = energy.wind_capacity_factor(wind, "generic", cut_in=cut_in, rated=rated, cut_out=cut_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)


def test_tabulated_knots_linear_interpolation_and_zero_extrapolation() -> None:
    curve = ([3, 5, 10, 20, 25], [0, 0.25, 1, 1, 0])
    wind = np.array([[0, 3, 4, 5, 7.5], [10, 20, 22.5, 25, 30]])
    result = energy.wind_capacity_factor(wind, curve)
    np.testing.assert_array_equal(result, [[0, 0, 0.125, 0.25, 0.625], [1, 1, 0.5, 0, 0]])
    np.testing.assert_array_equal(energy.wind_capacity_factor([0, 1, 2, 3], ([1, 2], [0.2, 0.8])), [0, 0.2, 0.8, 0])


@pytest.mark.parametrize("curve", ["generic", ([0, 10, 25], [0, 1, 0])])
def test_density_scaling_broadcasts_before_curve_evaluation(curve: Any) -> None:
    wind = np.array([[5.0], [15.0]])
    density = np.array([1.225, 1.225 * 8, 1.225 / 8])
    result = energy.wind_capacity_factor(wind, curve, air_density=density)
    expected = energy.wind_capacity_factor([[5, 10, 2.5], [15, 30, 7.5]], curve)
    # Cube-root rounding permits a few float64 operations of absolute error.
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("curve", ["generic", ([0, 10, 25], [0, 1, 0])])
def test_missing_invalid_strided_masked_inputs_and_immutability(curve: Any) -> None:
    wind = np.ma.array([0, 2, -1, 4, np.nan, 6, np.inf, 8, -np.inf, 10, 5, 12], mask=[False] * 10 + [True, False])[::2]
    original = wind.copy()
    with np.errstate(all="raise"):
        result = energy.wind_capacity_factor(wind, curve)
    np.testing.assert_array_equal(result, [0, np.nan, np.nan, np.nan, np.nan, np.nan])
    np.testing.assert_array_equal(wind.data, original.data)
    np.testing.assert_array_equal(wind.mask, original.mask)
    density = np.ma.array([1.225, 0, -1, np.nan, np.inf, 1.225], mask=[False] * 5 + [True])
    with np.errstate(all="raise"):
        result = energy.wind_capacity_factor(0, curve, air_density=density)
    np.testing.assert_array_equal(result, [0, np.nan, np.nan, np.nan, np.nan, np.nan])


def test_tabulated_subnormal_interval_does_not_overflow() -> None:
    with np.errstate(all="raise"):
        result = energy.wind_capacity_factor([0, 5e-324, 1e-323], ([0, 1e-323], [0, 1]))
    np.testing.assert_array_equal(result, [0, 0.5, 1])


def test_scalar_empty_and_extreme_winds() -> None:
    result = energy.wind_capacity_factor(13, "generic")
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result == 1
    assert energy.wind_capacity_factor([], "generic").shape == (0,)
    wind = [0, np.nextafter(0.0, 1.0), 1e308]
    with np.errstate(all="raise"):
        result = energy.wind_capacity_factor(wind, "generic")
    np.testing.assert_array_equal(result, [0, 0, 0])


@pytest.mark.parametrize(
    "curve",
    [
        None,
        "unknown",
        [],
        ([1], [0]),
        ([1, 2], [0]),
        ([[1, 2]], [[0, 1]]),
        ([2, 1], [0, 1]),
        ([1, 1], [0, 1]),
        ([-1, 2], [0, 1]),
        ([1, np.nan], [0, 1]),
        ([1, np.inf], [0, 1]),
        ([1, 2], [-0.1, 1]),
        ([1, 2], [0, 1.1]),
        ([1, 2], [0, np.nan]),
        ([1, 2], np.ma.array([0, 1], mask=[True, False])),
    ],
)
def test_invalid_curves_raise(curve: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="power_curve"):
        energy.wind_capacity_factor([5], curve)


@pytest.mark.parametrize(
    "thresholds",
    [
        (3.5, 3.5, 25),
        (13, 3.5, 25),
        (3.5, 13, 13),
        (-1, 13, 25),
        (np.nan, 13, 25),
        (3.5, np.inf, 25),
        (True, 13, 25),
        ([3.5], 13, 25),
        (np.ma.masked, 13, 25),
    ],
)
def test_invalid_generic_thresholds_raise(thresholds: tuple[Any, Any, Any]) -> None:
    with pytest.raises(InvalidArgumentError, match="cut_in|rated|cut_out"):
        energy.wind_capacity_factor([5], "generic", cut_in=thresholds[0], rated=thresholds[1], cut_out=thresholds[2])


def test_density_shape_must_broadcast() -> None:
    with pytest.raises(InvalidArgumentError, match="broadcast"):
        energy.wind_capacity_factor([1, 2], "generic", air_density=[1, 2, 3])


@pytest.mark.parametrize("invalid", [["5"], [1j], [object()], np.array(["2000-01-01"], dtype="datetime64[D]")])
def test_nonnumeric_inputs_raise(invalid: npt.ArrayLike) -> None:
    with pytest.raises(InputTypeError):
        energy.wind_capacity_factor(invalid, "generic")
    with pytest.raises(InputTypeError):
        energy.wind_capacity_factor([5], "generic", air_density=invalid)
    with pytest.raises(InputTypeError):
        energy.wind_capacity_factor([5], (invalid, [0]))


def test_numpy_only_and_explicit_curve_contract() -> None:
    wind = xr.DataArray([5])
    density = xr.DataArray([1.225])
    with pytest.raises(InputTypeError, match="NumPy"):
        energy.wind_capacity_factor(wind, "generic")
    with pytest.raises(InputTypeError, match="NumPy"):
        energy.wind_capacity_factor([5], "generic", air_density=density)


def test_calculation_logs_memory_and_failure() -> None:
    logger = mock.MagicMock()
    logger.bind.return_value = logger
    with (
        mock.patch.object(_wind, "_logger", logger),
        mock.patch.object(_wind, "check_large_array_memory", return_value={}) as memory,
    ):
        energy.wind_capacity_factor([5], "generic")
        memory.assert_called_once()
        assert logger.bind.call_args_list[0].kwargs["index_type"] == "wind_capacity_factor"
        assert logger.bind.call_args.kwargs == {"input_shape": (1,), "input_elements": 1}
        assert [call.args[0] for call in logger.info.call_args_list] == ["calculation_started", "calculation_completed"]
        with (
            mock.patch.object(np, "searchsorted", side_effect=MemoryError("synthetic failure")),
            pytest.raises(MemoryError),
        ):
            energy.wind_capacity_factor([5], ([0, 10], [0, 1]))
        assert logger.error.call_args.args == ("calculation_failed",)


@pytest.mark.parametrize("curve", ["generic", ([0, 10], [0, 1])])
@pytest.mark.parametrize(
    "target,prior_calls", [("_as_float_array", 0), ("_as_float_array", 1), ("np.broadcast_arrays", 0)]
)
def test_input_preparation_failure_is_logged(curve: Any, target: str, prior_calls: int) -> None:
    logger = mock.MagicMock()
    logger.bind.return_value = logger
    failure = MemoryError("synthetic allocation failure")
    with (
        mock.patch.object(_wind, "_logger", logger),
        mock.patch(f"{_wind.__name__}.{target}", side_effect=[np.array([5.0])] * prior_calls + [failure]),
        pytest.raises(MemoryError) as caught,
    ):
        energy.wind_capacity_factor(
            np.array([5], dtype=np.float32), curve, air_density=np.array([1.225], dtype=np.float32)
        )
    assert caught.value is failure
    logger.error.assert_called_once()
    assert logger.error.call_args.args == ("calculation_failed",)


def test_optional_memory_inspection_failure_does_not_abort_calculation() -> None:
    wind = [5, 13, 25]
    expected = energy.wind_capacity_factor(wind, "generic")
    psutil = mock.Mock()
    psutil.Process.return_value.memory_info.side_effect = OSError("RSS unavailable")
    with (
        mock.patch.object(performance, "LARGE_ARRAY_THRESHOLD_BYTES", 0),
        mock.patch.dict("sys.modules", {"psutil": psutil}),
    ):
        result = energy.wind_capacity_factor(wind, "generic")
    np.testing.assert_array_equal(result, expected)
    psutil.Process.return_value.memory_info.assert_called_once()
