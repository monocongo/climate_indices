"""Fixed-window EDI identities and daily Calibration Period contracts."""

import numpy as np
import pytest

from climate_indices import flood
from climate_indices.exceptions import DataShapeError, InputTypeError, InvalidArgumentError


def _pe_years(*levels: float) -> np.ndarray:
    return np.stack([np.full(366, level, dtype=float) for level in levels])


def test_edi_standardizes_each_calendar_day_on_calibration_years_only() -> None:
    values = _pe_years(1, 3, 5, 9)
    result = flood.edi(values, 2000, 2000, 2002)
    np.testing.assert_allclose(result[:, 0], (np.array([1, 3, 5, 9]) - 3) / np.sqrt(8 / 3))
    assert result.shape == values.shape
    # Scaling PE (and matching duration) cancels in DEP / SD(PRN).
    np.testing.assert_allclose(flood.edi(values * 7, 2000, 2000, 2002, duration=2), result)
    np.testing.assert_allclose(flood.edi(values.ravel(), 2000, 2000, 2002), result.ravel())


def test_edi_accepts_pe_kernel_output_and_keeps_window_nans() -> None:
    rain = _pe_years(1, 3, 5)
    pe = flood.effective_precipitation(rain, duration=3)
    result = flood.edi(pe, 2000, 2000, 2002, duration=3)
    assert np.isnan(result[0, :2]).all()
    assert np.isfinite(result[0, 2:]).all()


def test_edi_uses_daily_baselines_and_preserves_missing_observations() -> None:
    values = _pe_years(1, 3, 5)
    values[:, 1] = [2, 4, 6]
    values[0, 2] = np.nan  # only two calibration values: still defined
    values[1, 3] = np.nan  # two values: still defined
    values[:2, 4] = np.nan  # insufficient calibration data
    values[:, 5] = 7  # zero variance
    result = flood.edi(values, 2000, 2000, 2002)
    assert result[2, 0] == pytest.approx((5 - 3) / np.sqrt(8 / 3))
    assert result[2, 1] == pytest.approx((6 - 4) / np.sqrt(8 / 3))
    assert result[2, 2] == pytest.approx(1)
    assert np.isnan(result[1, 3])
    assert np.isnan(result[:, 4:6]).all()
    masked = np.ma.array(values, mask=False)
    masked.mask[0, 0] = True
    assert np.isnan(flood.edi(masked, 2000, 2000, 2002)[0, 0])


def test_edi_partial_year_and_independent_spatial_cells() -> None:
    values = _pe_years(1, 3, 5)
    block = np.stack((values.ravel(), values.ravel() * 2), axis=1).reshape(1098, 1, 2)
    result = flood.edi(block, 2000, 2000, 2001)
    np.testing.assert_allclose(result[:, 0, 0], result[:, 0, 1])
    np.testing.assert_allclose(result[:, 0, 0], flood.edi(values.ravel(), 2000, 2000, 2001))
    partial = np.concatenate((values[:2].ravel(), [7.0]))
    assert flood.edi(partial, 2000, 2000, 2001)[-1] == pytest.approx(5)
    with pytest.raises(InvalidArgumentError):
        flood.edi(partial, 2000, 2000, 2002)
    ambiguous = np.ones((732, 366, 1))
    with pytest.raises(ValueError, match="spatial_time_major=True"):
        flood.edi(ambiguous, 2000, 2000, 2001)
    assert flood.edi(ambiguous, 2000, 2000, 2001, spatial_time_major=True).shape == ambiguous.shape
    empty = np.empty((732, 0, 2))
    assert flood.edi(empty, 2000, 2000, 2001).shape == empty.shape


@pytest.mark.parametrize("years", [(2001, 2001), (1999, 2001), (2000, 2002), (2000, 1999), (2000.5, 2001)])
def test_edi_rejects_invalid_calibration_period(years: tuple[int, int]) -> None:
    values = _pe_years(1, 3)
    with pytest.raises(InvalidArgumentError):
        flood.edi(values, 2000, *years)


@pytest.mark.parametrize("duration", [0, -1, 1.5, True])
def test_edi_rejects_invalid_duration(duration: object) -> None:
    values = _pe_years(1, 3)
    with pytest.raises(InvalidArgumentError):
        flood.edi(values, 2000, 2000, 2001, duration=duration)  # type: ignore[arg-type]


def test_edi_rejects_invalid_input() -> None:
    with pytest.raises(DataShapeError):
        flood.edi(np.ones((2, 365)), 2000, 2000, 2001)
    with pytest.raises(InputTypeError):
        flood.edi(["wet"], 2000, 2000, 2001)
    for invalid in (-1.0, np.inf):
        values = _pe_years(1, invalid)
        with pytest.raises(InvalidArgumentError):
            flood.edi(values, 2000, 2000, 2001)
