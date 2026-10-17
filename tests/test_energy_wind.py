"""Wind-profile contracts; external numerical evidence lives in test_energy_reference."""

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest

import climate_indices
from climate_indices import energy
from climate_indices.exceptions import InputTypeError, InvalidArgumentError


def test_energy_namespace_is_exported_without_root_functions() -> None:
    assert "energy" in climate_indices.__all__
    assert set(energy.__all__) == {"hub_height_wind_speed", "wind_shear_exponent"}
    for name in energy.__all__:
        assert callable(getattr(energy, name))
        assert not hasattr(climate_indices, name)


@pytest.mark.parametrize("method,kwargs", [("power", {}), ("log", {"roughness_length": 0.03})])
def test_wind_profile_shapes_masks_and_input_immutability(
    method: Literal["power", "log"], kwargs: dict[str, Any]
) -> None:
    wind = np.ma.array([[0, 2, 4], [6, 8, 10]], mask=[[False, True, False], [False, False, False]])
    original = wind.copy()
    result = energy.hub_height_wind_speed(wind, 10, 10, method=method, **kwargs)
    assert result.shape == wind.shape
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, [[0, np.nan, 4], [6, 8, 10]], rtol=0, atol=0)
    np.testing.assert_array_equal(wind.data, original.data)
    np.testing.assert_array_equal(wind.mask, original.mask)


@pytest.mark.parametrize("method,kwargs", [("power", {}), ("log", {"roughness_length": 0.1})])
def test_invalid_winds_are_missing_without_runtime_warnings(
    method: Literal["power", "log"], kwargs: dict[str, Any]
) -> None:
    with np.errstate(all="raise"):
        result = energy.hub_height_wind_speed([0, -1, np.inf, -np.inf, np.nan, 2], 10, 100, method=method, **kwargs)
    assert result[0] == 0
    assert np.isnan(result[1:5]).all()
    assert result[5] > 2


def test_power_exponents_broadcast_and_preserve_inversions() -> None:
    wind = np.array([[0.0], [4.0]])
    alpha = np.ma.array([-1, 0, 1, np.nan, np.inf, 2], mask=[False, False, False, False, False, True])
    result = energy.hub_height_wind_speed(wind, 10, 100, alpha=alpha)
    np.testing.assert_allclose(result, [[0, 0, 0, np.nan, np.nan, np.nan], [0.4, 4, 40, np.nan, np.nan, np.nan]])


def test_overflow_is_missing_but_calm_stays_zero() -> None:
    with np.errstate(all="raise"):
        result = energy.hub_height_wind_speed([0, 1], 10, 100, alpha=1000)
    np.testing.assert_allclose(result, [0, np.nan])


@pytest.mark.parametrize("wind,reference,hub,alpha", [(1.0, 100.0, 10.0, 1000), (1e-300, 1e100, 1.0, 1)])
def test_underflow_rounds_to_zero_without_runtime_warnings(
    wind: float, reference: float, hub: float, alpha: float
) -> None:
    with np.errstate(all="raise"):
        result = energy.hub_height_wind_speed([wind], reference, hub, alpha=alpha)
    np.testing.assert_array_equal(result, [0.0])


@pytest.mark.parametrize("invalid_method", ["unknown", None, 1, np.array(["power", "log"])])
def test_invalid_method_raises_argument_error(invalid_method: Any) -> None:
    with pytest.raises(InvalidArgumentError, match="method"):
        energy.hub_height_wind_speed([1], 10, 100, method=invalid_method)


@pytest.mark.parametrize("name", ["reference_height", "hub_height"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, True, np.bool_(True), "10", [10], 10**400])
def test_profile_rejects_invalid_heights(name: str, value: Any) -> None:
    kwargs: dict[str, Any] = {"reference_height": 10, "hub_height": 100, name: value}
    with pytest.raises(InvalidArgumentError, match=name):
        energy.hub_height_wind_speed([1], **kwargs)


@pytest.mark.parametrize("roughness", [None, 0, -1, np.nan, np.inf, 10, 100, True])
def test_log_profile_requires_positive_roughness_below_both_heights(roughness: float | None) -> None:
    with pytest.raises(InvalidArgumentError, match="roughness_length"):
        energy.hub_height_wind_speed([1], 10, 100, method="log", roughness_length=roughness)
    with pytest.raises(InvalidArgumentError, match="roughness_length"):
        energy.hub_height_wind_speed([1], 100, 10, method="log", roughness_length=10)


def test_invalid_shapes_raise() -> None:
    with pytest.raises(InvalidArgumentError, match="broadcast"):
        energy.hub_height_wind_speed([1, 2], 10, 100, alpha=[0, 0, 0])
    with pytest.raises(InvalidArgumentError, match="broadcast"):
        energy.wind_shear_exponent([1, 2], [1, 2, 3])


@pytest.mark.parametrize("invalid", [["5"], [1j], [object()], np.array(["2000-01-01"], dtype="datetime64[D]")])
def test_nonnumeric_inputs_are_rejected(invalid: npt.ArrayLike) -> None:
    with pytest.raises(InputTypeError):
        energy.hub_height_wind_speed(invalid, 10, 100)
    with pytest.raises(InputTypeError):
        energy.hub_height_wind_speed([1], 10, 100, alpha=invalid)
    with pytest.raises(InputTypeError):
        energy.wind_shear_exponent([1], invalid)


def test_shear_recovers_positive_zero_and_negative_exponents() -> None:
    lower = np.array([[2.0], [5.0]])
    exponent = np.array([-0.3, 0.0, 0.2])
    upper = energy.hub_height_wind_speed(lower, 10, 100, alpha=exponent)
    recovered = energy.wind_shear_exponent(lower, upper)
    np.testing.assert_allclose(recovered, np.broadcast_to(exponent, (2, 3)), rtol=0, atol=2e-16)
    assert (recovered[:, 0] < 0).all()
    np.testing.assert_array_equal(recovered[:, 1], 0)


def test_shear_invalid_pairs_and_masks_are_missing() -> None:
    lower = np.ma.array([0, 1, -1, 1, np.inf, 1, np.nan, 2], mask=[False] * 7 + [True])
    upper = [1, 0, 1, -1, 1, np.inf, 1, 4]
    with np.errstate(all="raise"):
        result = energy.wind_shear_exponent(lower, upper)
    assert np.isnan(result).all()


def test_shear_avoids_speed_ratio_overflow() -> None:
    with np.errstate(all="raise"):
        result = energy.wind_shear_exponent([1e-300], [1e300])
    np.testing.assert_allclose(result, [600], rtol=0, atol=1e-12)


@pytest.mark.parametrize("lower,upper", [(0, 100), (10, 10), (100, 10), (10, np.inf), (np.nan, 100)])
def test_shear_requires_ordered_positive_heights(lower: float, upper: float) -> None:
    with pytest.raises(InvalidArgumentError):
        energy.wind_shear_exponent([1], [2], lower_height=lower, upper_height=upper)


def test_scalar_empty_and_custom_height_inputs() -> None:
    result = energy.hub_height_wind_speed(5, np.float64(20), np.int64(80), alpha=0.5)
    assert isinstance(result, np.ndarray) and result.shape == () and result.dtype == np.float64
    assert result == 10
    np.testing.assert_array_equal(energy.hub_height_wind_speed([2], 2**70, 2**70), [2])
    shear = energy.wind_shear_exponent(5, 10, lower_height=20, upper_height=80)
    assert isinstance(shear, np.ndarray) and shear.shape == () and shear.dtype == np.float64
    assert shear == pytest.approx(0.5)
    assert energy.hub_height_wind_speed([], 10, 100).shape == (0,)
    assert energy.wind_shear_exponent([], []).shape == (0,)
