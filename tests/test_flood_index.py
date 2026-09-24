"""Annual-maximum calibration and daily layout contracts for the Flood Index."""

import numpy as np
import pytest

from climate_indices import flood
from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError


def test_flood_index_uses_complete_annual_maxima_and_preserves_daily_pe() -> None:
    values = np.full((4, 366), 2.0)
    values[0, 10], values[1, 10], values[2, 10], values[3, 10] = 4, 6, 8, 100
    values[1, 11] = np.nan
    result = flood.flood_index(values, 2000, 2000, 2002, year_start_month=1)
    np.testing.assert_allclose(result[:, 10], (np.array([4, 6, 8, 100]) - 6) / np.sqrt(8 / 3))
    assert result[0, 0] == pytest.approx((2 - 6) / np.sqrt(8 / 3))
    assert np.isnan(result[1, 11])
    np.testing.assert_allclose(
        flood.flood_index(values.ravel(), 2000, 2000, 2002, year_start_month=1), result.ravel(), equal_nan=True
    )


def test_flood_index_groups_cross_year_maxima_and_excludes_partial_periods() -> None:
    # October starts at positional day 274 in the all-leap calendar.
    values = np.ones(4 * 366)
    values[0] = 1000  # incomplete leading calendar portion, not an annual maximum
    for year, peak in enumerate((4, 6, 8)):
        values[year * 366 + 274] = peak
    values[3 * 366 + 300] = 2000  # incomplete trailing annual period
    result = flood.flood_index(values, 2000, 2001, 2002, year_start_month=10)
    # Starts in 2000–2002 are complete; 2003 is partial.
    assert result[0] == pytest.approx((1000 - 7) / 1)
    assert result[3 * 366 + 300] == pytest.approx((2000 - 7) / 1)
    with pytest.raises(InvalidArgumentError):
        flood.flood_index(values, 2000, 2000, 2003, year_start_month=10)


def test_flood_index_spatial_and_missing_calibration_maxima() -> None:
    years = np.stack([np.full(366, level, dtype=float) for level in (2, 4, 6)])
    block = np.stack([years.ravel(), years.ravel() * 2], axis=1).reshape(1098, 1, 2)
    result = flood.flood_index(block, 2000, 2000, 2002, year_start_month=1)
    np.testing.assert_allclose(result[:, 0, 0], result[:, 0, 1])
    ambiguous = np.ones((1098, 366, 1))
    with pytest.raises(ValueError, match="spatial_time_major=True"):
        flood.flood_index(ambiguous, 2000, 2000, 2002, year_start_month=1)
    assert (
        flood.flood_index(ambiguous, 2000, 2000, 2002, year_start_month=1, spatial_time_major=True).shape
        == ambiguous.shape
    )
    block[:732, 0, 0] = np.nan  # one finite maximum cannot define a SD
    assert np.isnan(flood.flood_index(block, 2000, 2000, 2002, year_start_month=1)[:, 0, 0]).all()
    masked = np.ma.array(years, mask=False)
    masked.mask[0, 0] = True
    assert np.isnan(flood.flood_index(masked, 2000, 2000, 2002, year_start_month=1)[0, 0])


@pytest.mark.parametrize("month", [0, 13, 1.5, True])
def test_flood_index_rejects_invalid_month(month: object) -> None:
    with pytest.raises(InvalidArgumentError):
        flood.flood_index(np.ones(732), 2000, 2000, 2001, year_start_month=month)  # type: ignore[arg-type]


def test_flood_index_rejects_invalid_inputs() -> None:
    for years in ((1999, 2001), (2000, 2000), (2000, 2002), (2000.5, 2001)):
        with pytest.raises(InvalidArgumentError):
            flood.flood_index(np.ones(732), 2000, *years, year_start_month=1)
    with pytest.raises(DataShapeError):
        flood.flood_index(np.ones((2, 365)), 2000, 2000, 2001, year_start_month=1)
    with pytest.raises(InputTypeError):
        flood.flood_index(["rain"], 2000, 2000, 2001, year_start_month=1)
    for invalid in (-1.0, np.inf):
        values = np.full(732, 1.0)
        values[-1] = invalid
        with pytest.raises(InvalidArgumentError):
            flood.flood_index(values, 2000, 2000, 2001, year_start_month=1)
