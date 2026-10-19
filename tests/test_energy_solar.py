"""Jerez PV specification checks and a matched pvlib power-term comparison."""

import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from climate_indices import energy
from climate_indices.exceptions import InputTypeError, InvalidArgumentError


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_jerez_model_terms(dtype: npt.DTypeLike) -> None:
    # Hand-calculated equations 1–3 isolate irradiance, air temperature, and wind terms.
    result = energy.pv_capacity_factor(
        np.array([500, 500, 500, 1000], dtype=dtype),
        np.array([0, 20, 0, 0], dtype=dtype),
        np.array([0, 0, 2, 0], dtype=dtype),
    )
    np.testing.assert_allclose(result, [0.51675, 0.4696, 0.52439, 0.9635], rtol=0, atol=1e-12)
    assert result.dtype == np.float64


def test_cell_reference_temperature_is_not_air_reference_temperature() -> None:
    # With G=1000 and v=2, this air temperature gives Tcell=25 exactly in real arithmetic.
    air_temperature = (25 - 4.3 - 28 + 1.528 * 2) / 0.943
    np.testing.assert_allclose(energy.pv_capacity_factor(1000, air_temperature, 2), 1, rtol=0, atol=1e-12)
    assert energy.pv_capacity_factor(1000, 25, 2) < 1


def test_night_and_project_clipping_policy() -> None:
    result = energy.pv_capacity_factor([0, -0.0, 1000, 500], [20, 20, -100, 400], 0)
    np.testing.assert_array_equal(result, [0, 0, 1, 0])
    assert not np.signbit(result).any()


@pytest.mark.parametrize("position", range(3))
def test_missing_and_nonfinite_inputs_stay_missing_at_night(position: int) -> None:
    inputs = [np.zeros(4), np.full(4, 20.0), np.ones(4)]
    inputs[position] = np.array([np.nan, np.inf, -np.inf, 1.0])
    with np.errstate(all="raise"):
        result = energy.pv_capacity_factor(*inputs)
    assert np.isnan(result[:3]).all()
    assert np.isfinite(result[3])


@pytest.mark.parametrize("position", [0, 2])
def test_negative_radiation_and_wind_are_missing(position: int) -> None:
    inputs = [np.full(2, 500.0), np.full(2, 20.0), np.ones(2)]
    inputs[position][0] = -1
    result = energy.pv_capacity_factor(*inputs)
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


def test_broadcasting_masks_strides_and_input_immutability() -> None:
    radiation = (np.arange(12.0) * 100).reshape(3, 4)[:, ::2]
    temperature = np.ma.array([[0], [20], [40]], mask=[[False], [True], [False]])
    wind = np.array([0.0, 2.0])
    copies = [value.copy() for value in (radiation, temperature, wind)]
    result = energy.pv_capacity_factor(radiation, temperature, wind)
    assert result.shape == (3, 2)
    assert np.isnan(result[1]).all()
    np.testing.assert_allclose(result[0], [0, 0.218156], rtol=0, atol=1e-12)
    for value, original in zip((radiation, temperature, wind), copies, strict=True):
        np.testing.assert_array_equal(value, original)
    np.testing.assert_array_equal(temperature.mask, np.ma.getmaskarray(copies[1]))


@pytest.mark.parametrize("position", range(3))
def test_each_input_mask_is_preserved(position: int) -> None:
    inputs = [np.ma.array([500.0, 500]), np.ma.array([20.0, 20]), np.ma.array([1.0, 1])]
    inputs[position][0] = np.ma.masked
    assert np.isnan(energy.pv_capacity_factor(*inputs)[0])


def test_scalar_and_empty_inputs() -> None:
    assert energy.pv_capacity_factor(500, 20, 2).shape == ()
    assert energy.pv_capacity_factor([], 20, 2).shape == (0,)


def test_unbroadcastable_inputs_raise() -> None:
    with pytest.raises(InvalidArgumentError, match="broadcast"):
        energy.pv_capacity_factor([0, 500], [10, 20, 30], 2)


@pytest.mark.parametrize("position", range(3))
@pytest.mark.parametrize("invalid", [["5"], [1j], [object()], np.array(["2000-01-01"], dtype="datetime64[D]")])
def test_nonnumeric_inputs_raise(position: int, invalid: npt.ArrayLike) -> None:
    inputs: list[npt.ArrayLike] = [500, 20, 2]
    inputs[position] = invalid
    with pytest.raises(InputTypeError):
        energy.pv_capacity_factor(*inputs)


def test_extremes_do_not_warn_or_turn_overflow_into_capacity() -> None:
    with np.errstate(all="raise"):
        result = energy.pv_capacity_factor([1e308, 1e-300, 500], [20, 20, -1e308], [0, 0, 1e308])
    assert np.isnan(result[[0, 2]]).all()
    assert 0 <= result[1] < 1e-300


def test_accumulated_era5_radiation_uses_actual_interval_duration() -> None:
    # Exact unit conversion: J m-2 / seconds = W m-2, including non-hourly intervals.
    seconds = np.array([1800, 3600, 10800])
    flux = np.array([500, 500, 500])
    np.testing.assert_array_equal(
        energy.pv_capacity_factor(flux * seconds / seconds, 20, 2),
        energy.pv_capacity_factor(flux, 20, 2),
    )


def test_matched_pvlib_power_temperature_term() -> None:
    directory = Path(__file__).parent / "fixture" / "energy_solar"
    payload = (directory / "reference.json").read_bytes()
    provenance = json.loads((directory / "provenance.json").read_text())
    reference = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == provenance["checksum_sha256"]
    assert reference["pvlib_version"] == "0.15.0"
    radiation = np.array(reference["irradiance_w_m2"])
    cell = np.array(reference["cell_temperature_c"])
    wind = 2.0
    air = (cell - 4.3 - 0.028 * radiation + 1.528 * wind) / 0.943
    expected = np.clip(reference["pvwatts_dc"], 0, 1)
    # Matched float64 power terms allow only cross-platform roundoff, not model differences.
    np.testing.assert_allclose(
        energy.pv_capacity_factor(radiation, air, wind), expected, **provenance["validation_tolerance"]
    )
