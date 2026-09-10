"""Tests for the Fosberg Fire Weather Index (#808)."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from climate_indices import fire
from climate_indices.exceptions import InvalidArgumentError


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _emc(temperature_fahrenheit: float, relative_humidity_percent: float) -> float:
    return float(
        fire._equilibrium_moisture_content(
            np.asarray(temperature_fahrenheit, dtype=np.float64),
            np.asarray(relative_humidity_percent, dtype=np.float64),
        )
    )


# ------------------------------------------------------------------------------
# equilibrium moisture content


@pytest.mark.parametrize(
    ("relative_humidity", "expected"),
    [
        pytest.param(5.0, 1.235355, id="low_range"),
        pytest.param(30.0, 5.99582, id="mid_range"),
        pytest.param(80.0, 16.06068, id="high_range"),
    ],
)
def test_emc_matches_simard_on_each_range(relative_humidity: float, expected: float) -> None:
    """Hand-evaluated Simard (1968) regressions at 70 F."""
    assert _emc(70.0, relative_humidity) == pytest.approx(expected, rel=1e-12)


def test_emc_breakpoints_belong_to_the_lower_range() -> None:
    """10% and 50% use the lower regression, as in NCEP's GEMPAK code.

    The published equations are written "h < 10" and "10 < h <= 50", which
    leaves exactly 10% in neither range; GEMPAK closes both ranges from above.
    """
    just_above_10 = np.nextafter(10.0, np.inf)
    just_above_50 = np.nextafter(50.0, np.inf)

    assert _emc(70.0, 10.0) == pytest.approx(2.43842, rel=1e-12)
    assert _emc(70.0, just_above_10) == pytest.approx(2.79368, rel=1e-9)
    assert _emc(70.0, 50.0) == pytest.approx(9.19796, rel=1e-12)
    assert _emc(70.0, just_above_50) == pytest.approx(9.58815, rel=1e-9)


def test_emc_is_discontinuous_at_the_breakpoints() -> None:
    """The three regressions do not meet, so continuity cannot be asserted.

    Pinned so that nobody "fixes" the published equations into a smooth curve
    by accident. At 10% the jump even changes sign with temperature.
    """
    below_10, above_10 = _emc(32.0, 10.0), _emc(32.0, np.nextafter(10.0, np.inf))
    below_50, above_50 = _emc(32.0, 50.0), _emc(32.0, np.nextafter(50.0, np.inf))
    assert above_10 - below_10 == pytest.approx(0.697412, abs=1e-6)
    assert above_50 - below_50 == pytest.approx(0.493398, abs=1e-6)

    hot_jump = _emc(110.0, np.nextafter(10.0, np.inf)) - _emc(110.0, 10.0)
    assert hot_jump == pytest.approx(-0.0049, abs=1e-6)


# ------------------------------------------------------------------------------
# damping coefficient and index


def test_calibration_point() -> None:
    """Zero moisture and a 30 mph wind give 100, to the precision of 0.3002."""
    value = float(fire._ffwi(np.asarray(0.0), np.asarray(30.0)))
    assert value == pytest.approx(99.988881, abs=1e-6)
    assert abs(value - 100.0) < 0.02


def test_damping_changes_sign_exactly_at_30() -> None:
    """eta = (1 - x)(0.5x^2 - x + 1) with the second factor always positive."""
    x = np.linspace(0.0, 1.5, 301)
    polynomial = 1.0 - 2.0 * x + 1.5 * x**2 - 0.5 * x**3
    np.testing.assert_allclose(polynomial, (1.0 - x) * (0.5 * x**2 - x + 1.0), atol=1e-12)
    assert np.all(0.5 * x**2 - x + 1.0 > 0.0)

    damping = fire._moisture_damping(np.array([0.0, 15.0, 30.0, 40.0]))
    assert damping[0] == pytest.approx(1.0)
    assert 0.0 < damping[1] < 1.0
    assert damping[2] == pytest.approx(0.0, abs=1e-12)
    assert damping[3] == pytest.approx(0.0, abs=1e-12)


def test_saturated_cold_air_gives_zero_not_a_negative_index() -> None:
    """Below about -43 C at 100% humidity the moisture content exceeds 30."""
    assert _emc(-58.0, 100.0) > 30.0
    for cap in (True, False):
        assert float(fire.fosberg_ffwi(-50.0, 100.0, 5.0, cap_at_100=cap)) == 0.0


@pytest.mark.parametrize(
    ("temperature", "humidity", "wind", "expected"),
    [
        pytest.param(30.0, 15.0, 10.0, 59.242107, id="hot_dry_breezy"),
        pytest.param(25.0, 20.0, 10.0, 55.430582, id="warm_dry_breezy"),
    ],
)
def test_reference_values(temperature: float, humidity: float, wind: float, expected: float) -> None:
    assert float(fire.fosberg_ffwi(temperature, humidity, wind)) == pytest.approx(expected, rel=1e-7)


def test_cap_is_applied_by_default_and_can_be_turned_off() -> None:
    assert float(fire.fosberg_ffwi(40.0, 5.0, 25.0)) == 100.0
    assert float(fire.fosberg_ffwi(40.0, 5.0, 25.0, cap_at_100=False)) == pytest.approx(172.5894, abs=1e-4)


def _gempak_pd_fosb(tmpc: np.ndarray, relh: np.ndarray, sped: np.ndarray) -> np.ndarray:
    """Transcription of NCEP GEMPAK's pd_fosb, in float32 with its own constants."""
    f32 = np.float32
    tmpc, relh, sped = (np.asarray(a, dtype=f32) for a in (tmpc, relh, sped))
    tf = tmpc * f32(9.0 / 5.0) + f32(32.0)
    smph = sped * f32(1.9425) / f32(0.868976)
    fw = np.where(
        relh <= f32(10.0),
        f32(0.03229) + f32(0.281073) * relh - f32(0.000578) * relh * tf,
        np.where(
            relh <= f32(50.0),
            f32(2.22749) + f32(0.160107) * relh - f32(0.014784) * tf,
            f32(21.0606) + f32(0.005565) * relh * relh - f32(0.00035) * relh * tf - f32(0.483199) * relh,
        ),
    )
    fwd = fw / f32(30.0)
    damping = f32(1.0) - f32(2.0) * fwd + f32(1.5) * fwd**2 - f32(0.5) * fwd**3
    return damping * np.sqrt(f32(1.0) + smph * smph) / f32(0.3002)


def test_matches_ncep_gempak() -> None:
    """Agrees with NCEP's operational implementation behind FOSINDX.

    The tolerance covers GEMPAK's m/s to mph factor, 1.9425 / 0.868976, which is
    0.069% below the exact 1 / 0.44704. Temperatures stay above -10 C, where
    GEMPAK's unclamped moisture content cannot exceed 30.
    """
    temperature, humidity, wind = np.meshgrid(
        np.arange(-10.0, 46.0, 5.0),
        np.array([0.0, 3.0, 10.0, 10.5, 25.0, 50.0, 50.5, 70.0, 100.0]),
        np.array([0.0, 1.0, 3.0, 7.0, 12.0, 20.0, 30.0]),
        indexing="ij",
    )
    ours = fire.fosberg_ffwi(temperature, humidity, wind, cap_at_100=False)
    np.testing.assert_allclose(ours, _gempak_pd_fosb(temperature, humidity, wind), rtol=1e-3)


# ------------------------------------------------------------------------------
# inputs


def test_out_of_range_inputs_are_nan() -> None:
    result = fire.fosberg_ffwi(
        np.array([20.0, 20.0, 20.0, 20.0, np.nan, 20.0]),
        np.array([-1.0, 100.5, 40.0, 40.0, 40.0, np.nan]),
        np.array([5.0, 5.0, -0.1, 5.0, 5.0, 5.0]),
    )
    assert np.isnan(result[[0, 1, 2, 4, 5]]).all()
    assert np.isfinite(result[3])


def test_edges_of_the_valid_range_are_valid() -> None:
    result = fire.fosberg_ffwi(20.0, np.array([0.0, 100.0]), 0.0)
    assert np.isfinite(result).all()


def test_inputs_broadcast() -> None:
    assert fire.fosberg_ffwi(20.0, 30.0, 5.0).shape == ()
    grid = fire.fosberg_ffwi(np.zeros((3, 1)), np.full((1, 4), 30.0), 5.0)
    assert grid.shape == (3, 4)


def test_incompatible_shapes_raise() -> None:
    with pytest.raises(InvalidArgumentError, match="must broadcast together"):
        fire.fosberg_ffwi(np.zeros(3), np.zeros(4), 5.0)


def test_chunked_time_axis_matches_eager() -> None:
    """Elementwise, so chunking along time changes nothing."""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask")

    rng = np.random.default_rng(808)
    shape = (48, 3)
    dims = ("time", "station")
    temperature = xr.DataArray(rng.uniform(-5.0, 40.0, shape), dims=dims)
    humidity = xr.DataArray(rng.uniform(0.0, 100.0, shape), dims=dims)
    wind = xr.DataArray(rng.uniform(0.0, 20.0, shape), dims=dims)

    eager = fire.fosberg_ffwi(temperature.values, humidity.values, wind.values)
    chunked = xr.apply_ufunc(
        fire.fosberg_ffwi,
        temperature.chunk({"time": 12}),
        humidity.chunk({"time": 12}),
        wind.chunk({"time": 12}),
        dask="parallelized",
        output_dtypes=[np.float64],
    )
    np.testing.assert_array_equal(chunked.compute().values, eager)


# ------------------------------------------------------------------------------
# properties

_temperature = st.floats(min_value=-60.0, max_value=55.0, allow_nan=False)
_humidity = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
_wind = st.floats(min_value=0.0, max_value=60.0, allow_nan=False)


@given(temperature=_temperature, humidity=_humidity, wind=_wind)
@settings(max_examples=200, deadline=None)
def test_capped_index_stays_within_0_and_100(temperature: float, humidity: float, wind: float) -> None:
    value = float(fire.fosberg_ffwi(temperature, humidity, wind))
    assert 0.0 <= value <= 100.0


@given(temperature=_temperature, humidity=_humidity, wind=_wind)
@settings(max_examples=200, deadline=None)
def test_uncapped_index_is_never_negative(temperature: float, humidity: float, wind: float) -> None:
    assert float(fire.fosberg_ffwi(temperature, humidity, wind, cap_at_100=False)) >= 0.0


@given(temperature=_temperature, humidity=_humidity, wind=_wind, extra=_wind)
@settings(max_examples=200, deadline=None)
def test_index_does_not_decrease_with_wind(temperature: float, humidity: float, wind: float, extra: float) -> None:
    slower = float(fire.fosberg_ffwi(temperature, humidity, wind, cap_at_100=False))
    faster = float(fire.fosberg_ffwi(temperature, humidity, wind + extra, cap_at_100=False))
    assert faster >= slower - 1e-12
