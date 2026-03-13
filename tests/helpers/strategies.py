"""Reusable Hypothesis strategies for property-based testing.

This module provides composite strategies for generating valid climate data
for property-based tests. These strategies ensure generated data satisfies
domain constraints (e.g., non-negative precipitation, valid latitudes).

Usage:
    from tests.helpers.strategies import monthly_precipitation_array, valid_scale

    @given(precip=monthly_precipitation_array(), scale=valid_scale())
    def test_spi_properties(precip, scale):
        result = indices.spi(precip, scale=scale, ...)
        assert result.shape == precip.shape
"""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst


@st.composite
def monthly_precipitation_array(draw: st.DrawFn, num_years: int | None = None) -> np.ndarray:
    """Generate valid monthly precipitation array.

    Args:
        draw: Hypothesis draw function
        num_years: Number of years (if None, randomly chosen between 30-50)

    Returns:
        Array of monthly precipitation values >= 0
    """
    if num_years is None:
        num_years = draw(st.integers(min_value=30, max_value=50))

    length = num_years * 12
    # use gamma distribution for realistic precipitation (skewed, non-negative)
    return draw(
        npst.arrays(
            dtype=np.float64,
            shape=length,
            elements=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        )
    )


@st.composite
def monthly_temperature_array(draw: st.DrawFn, num_years: int | None = None) -> np.ndarray:
    """Generate valid monthly temperature array with seasonal variation.

    Args:
        draw: Hypothesis draw function
        num_years: Number of years (if None, randomly chosen between 30-50)

    Returns:
        Array of monthly temperatures with realistic seasonal sinusoidal pattern
    """
    if num_years is None:
        num_years = draw(st.integers(min_value=30, max_value=50))

    length = num_years * 12
    # base mean temperature
    base_temp = draw(st.floats(min_value=-10.0, max_value=30.0))
    # seasonal amplitude
    amplitude = draw(st.floats(min_value=5.0, max_value=25.0))

    # create seasonal sinusoid
    months = np.arange(length)
    seasonal_pattern = base_temp + amplitude * np.sin(2 * np.pi * months / 12)

    # add random noise
    noise = draw(
        npst.arrays(
            dtype=np.float64,
            shape=length,
            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        )
    )

    return seasonal_pattern + noise


@st.composite
def daily_temperature_triplet(
    draw: st.DrawFn, num_years: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate valid daily temperature triplet (tmin, tmax, tmean).

    Args:
        draw: Hypothesis draw function
        num_years: Number of years (if None, randomly chosen between 5-15 for speed)

    Returns:
        Tuple of (tmin, tmax, tmean) where tmin <= tmean <= tmax
    """
    if num_years is None:
        # use fewer years for daily data to keep tests fast
        num_years = draw(st.integers(min_value=5, max_value=15))

    length = num_years * 366

    # generate base mean temperature
    base_mean = draw(st.floats(min_value=-10.0, max_value=30.0))

    # generate daily range (tmax - tmin)
    daily_range = draw(
        npst.arrays(
            dtype=np.float64,
            shape=length,
            elements=st.floats(min_value=3.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        )
    )

    # generate tmean with seasonal variation
    days = np.arange(length)
    seasonal_amplitude = draw(st.floats(min_value=10.0, max_value=25.0))
    tmean = base_mean + seasonal_amplitude * np.sin(2 * np.pi * days / 366)

    # derive tmin and tmax from tmean and range
    half_range = daily_range / 2.0
    tmin = tmean - half_range
    tmax = tmean + half_range

    return tmin, tmax, tmean


@st.composite
def valid_latitude(draw: st.DrawFn) -> float:
    """Generate valid latitude avoiding pole singularities.

    Returns:
        Latitude in degrees, range (-89.0, 89.0)
    """
    return draw(st.floats(min_value=-89.0, max_value=89.0, allow_nan=False, allow_infinity=False))


@st.composite
def valid_scale(draw: st.DrawFn) -> int:
    """Generate valid scale parameter for SPI/SPEI.

    Returns:
        Scale value in [1, 24] (capped for reasonable test execution time)
    """
    return draw(st.integers(min_value=1, max_value=24))


@st.composite
def precip_with_uniform_offset(draw: st.DrawFn, num_years: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate paired precipitation arrays where higher[i] > lower[i] everywhere.

    Used for monotonicity tests.

    Args:
        draw: Hypothesis draw function
        num_years: Number of years (if None, randomly chosen between 30-50)

    Returns:
        Tuple of (lower, higher) precipitation arrays
    """
    if num_years is None:
        num_years = draw(st.integers(min_value=30, max_value=50))

    length = num_years * 12

    # generate base array
    lower = draw(
        npst.arrays(
            dtype=np.float64,
            shape=length,
            elements=st.floats(min_value=0.0, max_value=400.0, allow_nan=False, allow_infinity=False),
        )
    )

    # generate uniform offset
    offset = draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False))

    higher = lower + offset

    return lower, higher
