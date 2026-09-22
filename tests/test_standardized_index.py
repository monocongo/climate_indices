"""Parity and contract tests for the generic public standardized_index() API (issue #1113)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from climate_indices import compute, indices

# scale, distribution, calibration-start fixture, calibration-end fixture, golden SPI fixture
_PARITY_CASES = (
    (1, indices.Distribution.gamma, "data_year_start_monthly", "data_year_end_monthly", "spi_1_month_gamma"),
    (6, indices.Distribution.gamma, "data_year_start_monthly", "data_year_end_monthly", "spi_6_month_gamma"),
    (
        6,
        indices.Distribution.pearson,
        "calibration_year_start_monthly",
        "calibration_year_end_monthly",
        "spi_6_month_pearson3",
    ),
)


@pytest.mark.parametrize(
    ("scale", "distribution", "calibration_start", "calibration_end", "expected_fixture"), _PARITY_CASES
)
def test_standardized_index_matches_pinned_spi_result(
    request,
    precips_mm_monthly,
    data_year_start_monthly,
    scale: int,
    distribution: indices.Distribution,
    calibration_start: str,
    calibration_end: str,
    expected_fixture: str,
) -> None:
    """Golden SPI fixtures pin standardized_index independently of spi() itself."""
    expected = request.getfixturevalue(expected_fixture)
    calibration_initial = request.getfixturevalue(calibration_start)
    calibration_final = request.getfixturevalue(calibration_end)
    values = np.asarray(precips_mm_monthly).flatten()

    computed = indices.standardized_index(
        values=values,
        scale=scale,
        distribution=distribution,
        data_start_year=data_year_start_monthly,
        calibration_year_initial=calibration_initial,
        calibration_year_final=calibration_final,
        periodicity=compute.Periodicity.monthly,
    )
    spi = indices.spi(
        values=values,
        scale=scale,
        distribution=distribution,
        data_start_year=data_year_start_monthly,
        calibration_year_initial=calibration_initial,
        calibration_year_final=calibration_final,
        periodicity=compute.Periodicity.monthly,
    )

    np.testing.assert_array_equal(computed, spi)
    np.testing.assert_allclose(computed, expected, atol=1e-8)


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


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_standardized_index_accepts_declared_spatial_blocks(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
) -> None:
    """The ambiguous (time, 12, *cells) layout needs the same declaration spi() requires."""
    block = np.asarray(precips_mm_monthly).reshape(123, 12, 1)
    arguments = (
        block,
        6,
        indices.Distribution.gamma,
        data_year_start_monthly,
        calibration_year_start_monthly,
        calibration_year_end_monthly,
        compute.Periodicity.monthly,
    )

    with pytest.raises(ValueError, match="spatial_time_major=True"):
        indices.standardized_index(*arguments)

    standardized = indices.standardized_index(*arguments, spatial_time_major=True)
    spi = indices.spi(*arguments, spatial_time_major=True)

    assert standardized.shape == block.shape
    np.testing.assert_array_equal(standardized, spi)


@pytest.mark.usefixtures(
    "precips_mm_monthly",
    "data_year_start_monthly",
    "calibration_year_start_monthly",
    "calibration_year_end_monthly",
)
def test_standardized_index_reports_its_own_fallback_context(
    precips_mm_monthly,
    data_year_start_monthly,
    calibration_year_start_monthly,
    calibration_year_end_monthly,
) -> None:
    """A Pearson-to-gamma fallback names the generic index, not SPI, in its warning."""
    values = np.full(np.asarray(precips_mm_monthly).size, 5.0)

    with mock.patch.object(compute._default_fallback_strategy, "log_fallback_warning") as log_fallback_warning:
        indices.standardized_index(
            values,
            6,
            indices.Distribution.pearson,
            data_year_start_monthly,
            calibration_year_start_monthly,
            calibration_year_end_monthly,
            compute.Periodicity.monthly,
        )

    assert log_fallback_warning.call_count == 1
    assert log_fallback_warning.call_args.kwargs["context"] == "standardized index computation"


def test_standardized_index_is_exported_from_the_indices_module() -> None:
    """The generic NumPy API is public through the module that carries the stable NumPy API."""
    assert "standardized_index" in indices.__all__
    assert callable(indices.standardized_index)
