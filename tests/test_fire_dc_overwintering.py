"""Tests for Drought Code overwintering and the seasonal carry contract (#806).

The frozen reference values in this module were produced with the NRCan
reference implementation (``cffdrs_py`` commit 0f57fcca2a6a84b69fe8d50f29d947f0be34d5f6,
``cffdrs/overwinter_drought_code.py``), the same lineage the CFFWIS
moisture-code fixtures use. They pin the overwintering equations and their
default parameters; they are regression fixtures, not an independent
scientific validation.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from climate_indices import fire
from climate_indices.exceptions import InvalidArgumentError

# (final_fall_dc, overwinter_precipitation_mm, carry_over_fraction, wetting_efficiency, expected)
_OVERWINTER_CASES = [
    (300.0, 110.0, 0.75, 0.75, 109.46569214268736),
    (300.0, 110.0, 1.0, 0.9, 16.353151574046638),
    (500.0, 60.0, 0.75, 0.75, 331.583464683451),
    (250.0, 0.0, 0.75, 0.75, 365.0728289807123),
    (400.0, 99.0, 0.75, 0.75, 177.52180542264173),
    # full recharge: the start-up code is the published seed
    (100.0, 200.0, 0.75, 0.75, 15.0),
]

_OVERWINTER_TOLERANCE = 1e-9

# two five-day fire seasons separated by a four-day off-season, on a two-cell
# grid. The off-season weather is deliberately extreme (40 C, 50 mm/day), so a
# leak into the recurrence would show up as a mismatch.
_SEASON_MASK = np.array([True] * 5 + [False] * 4 + [True] * 5)
_SEASON_TEMPERATURE = np.tile(np.array([25.0] * 5 + [40.0] * 4 + [25.0] * 5)[:, np.newaxis], (1, 2))
_SEASON_PRECIPITATION = np.tile(np.array([0.0] * 5 + [50.0] * 4 + [0.0] * 5)[:, np.newaxis], (1, 2))
_SEASON_MONTH = np.full(14, 7, dtype=np.int64)
_SEASON_LATITUDE = 46.0

# cell 0: a dry winter, so the autumn DC largely carries over and the spring
# start-up stays well above the seed. Cell 1: a fully recharging winter.
_WINTER_PRECIPITATION = np.array([20.0, 250.0])


@pytest.fixture(scope="module", autouse=True)
def disable_logging():
    """Silence the calculation lifecycle events during these tests."""
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


def _season_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return the multi-season temperature and precipitation inputs."""
    return _SEASON_TEMPERATURE.copy(), _SEASON_PRECIPITATION.copy()


def _run_season(
    start: int,
    stop: int,
    *,
    initial_dc: np.ndarray | float | None = None,
) -> np.ndarray:
    """Run one season's days on the multi-season grid."""
    temperature, precipitation = _season_inputs()
    return np.asarray(
        fire.drought_code(
            temperature[start:stop],
            precipitation[start:stop],
            _SEASON_LATITUDE,
            _SEASON_MONTH[start:stop],
            initial_dc=initial_dc,
        )
    )


# ------------------------------------------------------------------------------
# reference values


def test_public_api_exports_overwintering() -> None:
    """The overwintering function is exported from fire and nowhere unqualified."""
    assert "overwinter_drought_code" in fire.__all__


@pytest.mark.parametrize(
    ("final_fall_dc", "overwinter_precipitation", "carry_over_fraction", "wetting_efficiency", "expected"),
    _OVERWINTER_CASES,
)
def test_overwinter_drought_code_matches_the_reference_values(
    final_fall_dc: float,
    overwinter_precipitation: float,
    carry_over_fraction: float,
    wetting_efficiency: float,
    expected: float,
) -> None:
    """The overwintering equations and defaults reproduce the NRCan reference."""
    result = fire.overwinter_drought_code(
        final_fall_dc,
        overwinter_precipitation,
        carry_over_fraction=carry_over_fraction,
        wetting_efficiency=wetting_efficiency,
    )
    assert np.isclose(result, expected, rtol=0.0, atol=_OVERWINTER_TOLERANCE)


def test_overwinter_drought_code_broadcasts_elementwise() -> None:
    """Array inputs broadcast against each other and stay independent per cell."""
    final_fall_dc = np.array([400.0, 300.0, 250.0])
    overwinter_precipitation = np.array([99.0, 110.0, 200.0])
    result = fire.overwinter_drought_code(final_fall_dc, overwinter_precipitation)
    np.testing.assert_allclose(
        result,
        [177.52180542264173, 109.46569214268736, 15.0],
        rtol=0.0,
        atol=_OVERWINTER_TOLERANCE,
    )


def test_overwinter_drought_code_masks_invalid_inputs() -> None:
    """NaN and negative inputs are missing or invalid, not values to compute on."""
    result = fire.overwinter_drought_code(
        np.array([np.nan, 300.0, -1.0, 300.0, 300.0]),
        np.array([110.0, np.nan, 110.0, -5.0, 110.0]),
    )
    assert np.isnan(result[:4]).all()
    assert np.isclose(result[4], 109.46569214268736, rtol=0.0, atol=_OVERWINTER_TOLERANCE)


def test_overwinter_drought_code_leaves_a_zero_moisture_equivalent_undefined() -> None:
    """No carry-over and no winter wetting has no defined start-up code."""
    assert np.isnan(fire.overwinter_drought_code(300.0, 0.0, carry_over_fraction=0.0, wetting_efficiency=0.0))


@pytest.mark.parametrize("fraction", [-0.01, 1.01, np.nan, np.inf, True, "0.75"])
@pytest.mark.parametrize("name", ["carry_over_fraction", "wetting_efficiency"])
def test_overwinter_drought_code_rejects_invalid_fractions(name: str, fraction: object) -> None:
    """Both tunable fractions are validated rather than silently assumed."""
    with pytest.raises(InvalidArgumentError):
        fire.overwinter_drought_code(300.0, 110.0, **{name: fraction})


def test_overwinter_drought_code_rejects_incompatible_shapes() -> None:
    """Shapes that cannot broadcast raise rather than aligning by position."""
    with pytest.raises(InvalidArgumentError):
        fire.overwinter_drought_code(np.zeros((3, 2)), np.zeros((4,)))


# ------------------------------------------------------------------------------
# seasonal carry


def test_an_all_in_season_mask_matches_no_mask() -> None:
    """``in_season=None`` is exactly an all-true mask, bitwise."""
    temperature, precipitation = _season_inputs()
    unmasked = fire.drought_code(temperature, precipitation, _SEASON_LATITUDE, _SEASON_MONTH)
    masked = fire.drought_code(
        temperature,
        precipitation,
        _SEASON_LATITUDE,
        _SEASON_MONTH,
        in_season=np.ones(len(_SEASON_MONTH), dtype=bool),
    )
    np.testing.assert_array_equal(unmasked, masked)


def test_off_season_days_ignore_the_weather_and_carry_the_code() -> None:
    """Off-season days freeze the recurrence and emit the carried DC."""
    temperature, precipitation = _season_inputs()
    values = np.asarray(
        fire.drought_code(
            temperature,
            precipitation,
            _SEASON_LATITUDE,
            _SEASON_MONTH,
            in_season=_SEASON_MASK,
        )
    )
    # the four off-season days emit the last in-season value verbatim
    np.testing.assert_array_equal(values[5:9], np.repeat(values[4][np.newaxis], 4, axis=0))
    # the next season resumes from that frozen state rather than resetting
    assert (values[9] > values[4]).all()

    # replacing the off-season weather changes nothing at all
    temperature[5:9] = -30.0
    precipitation[5:9] = 0.0
    unchanged = fire.drought_code(
        temperature,
        precipitation,
        _SEASON_LATITUDE,
        _SEASON_MONTH,
        in_season=_SEASON_MASK,
    )
    np.testing.assert_array_equal(values, unchanged)


def test_off_season_days_are_not_missing_days() -> None:
    """Off-season NaNs neither poison nor bridge the recurrence."""
    temperature, precipitation = _season_inputs()
    expected = fire.drought_code(
        temperature,
        precipitation,
        _SEASON_LATITUDE,
        _SEASON_MONTH,
        in_season=_SEASON_MASK,
    )

    temperature[5:9] = np.nan
    precipitation[5:9] = np.nan
    observed = fire.drought_code(
        temperature,
        precipitation,
        _SEASON_LATITUDE,
        _SEASON_MONTH,
        in_season=_SEASON_MASK,
    )
    np.testing.assert_array_equal(expected, observed)

    # an in-season missing day still poisons under the default policy
    temperature[1] = np.nan
    poisoned = fire.drought_code(
        temperature,
        precipitation,
        _SEASON_LATITUDE,
        _SEASON_MONTH,
        in_season=_SEASON_MASK,
    )
    assert np.isnan(poisoned[1:]).all()


def test_a_run_can_be_resumed_across_an_off_season() -> None:
    """A state returned at the end of an off-season resumes bitwise."""
    temperature, precipitation = _season_inputs()
    whole = np.asarray(
        fire.drought_code(
            temperature,
            precipitation,
            _SEASON_LATITUDE,
            _SEASON_MONTH,
            in_season=_SEASON_MASK,
        )
    )
    state = fire.drought_code(
        temperature[:9],
        precipitation[:9],
        _SEASON_LATITUDE,
        _SEASON_MONTH[:9],
        in_season=_SEASON_MASK[:9],
        return_state=True,
    ).state
    resumed = fire.drought_code(
        temperature[9:],
        precipitation[9:],
        _SEASON_LATITUDE,
        _SEASON_MONTH[9:],
        initial_state=state,
    )
    np.testing.assert_array_equal(whole[9:], resumed)


def test_season_mask_broadcasts_and_validates() -> None:
    """A one-dimensional season mask is shared across cells; bad masks raise."""
    temperature, precipitation = _season_inputs()
    shared = fire.drought_code(
        temperature,
        precipitation,
        _SEASON_LATITUDE,
        _SEASON_MONTH,
        in_season=_SEASON_MASK,
    )
    per_cell = fire.drought_code(
        temperature,
        precipitation,
        _SEASON_LATITUDE,
        _SEASON_MONTH,
        in_season=np.stack([_SEASON_MASK, _SEASON_MASK], axis=1),
    )
    np.testing.assert_array_equal(shared, per_cell)

    for invalid in (np.ones(14, dtype=np.int64), np.ones((14, 3), dtype=bool), np.array([True, False])):
        with pytest.raises(InvalidArgumentError):
            fire.drought_code(
                temperature,
                precipitation,
                _SEASON_LATITUDE,
                _SEASON_MONTH,
                in_season=invalid,
            )


def test_multi_year_overwintering_diverges_in_the_expected_direction() -> None:
    """A dry winter carries the DC over; a wet one recharges it to the seed."""
    fall_dc = _run_season(0, 5)[-1]
    spring_dc = fire.overwinter_drought_code(fall_dc, _WINTER_PRECIPITATION)

    # (1) dry winter: the overwintered start-up is drier than the seed, and the
    # difference is exactly the start-up difference on the first day
    assert spring_dc[0] > 15.0
    assert spring_dc[1] == 15.0
    overwintered = _run_season(9, 14, initial_dc=spring_dc)
    non_overwintered = _run_season(9, 14)
    assert (overwintered[:, 0] > non_overwintered[:, 0]).all()
    assert np.isclose(overwintered[0, 0] - non_overwintered[0, 0], spring_dc[0] - 15.0, rtol=0.0, atol=1e-9)

    # (2) wet winter: full recharge, so the overwintered run is the seed run
    np.testing.assert_array_equal(overwintered[:, 1], non_overwintered[:, 1])

    # (3) carrying the autumn DC through without the overwinter equation keeps
    # the wet cell drier than a recharged start, which is the bias the
    # overwintering step exists to remove
    temperature, precipitation = _season_inputs()
    carried_through = np.asarray(
        fire.drought_code(
            temperature,
            precipitation,
            _SEASON_LATITUDE,
            _SEASON_MONTH,
            in_season=_SEASON_MASK,
        )
    )[9:]
    assert (carried_through[:, 1] > overwintered[:, 1]).all()
