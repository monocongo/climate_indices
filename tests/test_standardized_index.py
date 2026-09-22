"""Parity tests for the generic public standardized_index() API (issue #1113)."""

from __future__ import annotations

import numpy as np
import pytest

from climate_indices import compute, indices


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
@pytest.mark.parametrize("scale", [1, 6])
def test_standardized_index_matches_spi_for_gamma(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
    scale: int,
) -> None:
    """The generic API must reproduce SPI exactly for the gamma distribution."""
    values = np.asarray(precips_mm_monthly).flatten()

    standardized = indices.standardized_index(
        values,
        scale,
        indices.Distribution.gamma,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    spi = indices.spi(
        values,
        scale,
        indices.Distribution.gamma,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )

    np.testing.assert_array_equal(standardized, spi)


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_standardized_index_matches_spi_for_pearson(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
) -> None:
    """The generic API must reproduce SPI exactly for the Pearson Type III distribution."""
    values = np.asarray(precips_mm_monthly).flatten()

    standardized = indices.standardized_index(
        values,
        6,
        indices.Distribution.pearson,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )
    spi = indices.spi(
        values,
        6,
        indices.Distribution.pearson,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )

    np.testing.assert_array_equal(standardized, spi)


@pytest.mark.usefixtures(
    "precips_mm_daily",
    "data_year_start_daily",
    "calibration_year_start_daily",
    "calibration_year_end_daily",
)
def test_standardized_index_matches_spi_for_daily_periodicity(
    precips_mm_daily,
    data_year_start_daily,
    calibration_year_start_daily,
    calibration_year_end_daily,
) -> None:
    """Daily series use the same pipeline, including the 366-day calendar handling."""
    values = np.asarray(precips_mm_daily).flatten()

    standardized = indices.standardized_index(
        values,
        60,
        indices.Distribution.gamma,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )
    spi = indices.spi(
        values,
        60,
        indices.Distribution.gamma,
        data_year_start_daily,
        calibration_year_start_daily,
        calibration_year_end_daily,
        compute.Periodicity.daily,
    )

    np.testing.assert_array_equal(standardized, spi)


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_standardized_index_uses_supplied_fitting_params(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
) -> None:
    """Pre-computed fitting parameters reach the shared fit/standardize seam unchanged."""
    values = np.asarray(precips_mm_monthly).flatten()
    fitting_params = {"alpha": np.full(12, 4.0), "beta": np.full(12, 8.0)}

    standardized = indices.standardized_index(
        values,
        6,
        indices.Distribution.gamma,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
        fitting_params,
    )
    spi = indices.spi(
        values,
        6,
        indices.Distribution.gamma,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
        fitting_params,
    )

    np.testing.assert_array_equal(standardized, spi)


def test_standardized_index_is_exported_from_package_root() -> None:
    """The generic NumPy API is public from both the package root and its module."""
    import climate_indices

    assert climate_indices.standardized_index is indices.standardized_index
    assert "standardized_index" in climate_indices.__all__
    assert "standardized_index" in indices.__all__
