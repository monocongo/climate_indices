"""Tests for Calibration Period resolution (#1214)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climate_indices as ci
from climate_indices import compute, indices
from climate_indices._calibration_period import CalibrationPeriodError, resolve_calibration_period
from climate_indices.exceptions import CalibrationPeriodClampedWarning, InvalidArgumentError, ShortCalibrationWarning

MONTHLY = compute.Periodicity.monthly
DAILY = compute.Periodicity.daily

# reversed windows against a 1981-2009 record: inside it, straddling its start, straddling
# its end, before it and after it
REVERSED_WINDOWS = [(2005, 2002), (1990, 1970), (2020, 2000), (1975, 1970), (2020, 2010)]


def _record(n_years: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).gamma(2.0, 30.0, size=n_years * 12)


def _short_calibration_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, ShortCalibrationWarning)]


class TestClamp:
    def resolve(self, start: int, end: int):
        # a 29-year record, 1981-2009
        return resolve_calibration_period(1981, 29, start, end, policy="clamp")

    def test_exact_fit_is_kept(self):
        period = self.resolve(1981, 2009)
        assert (period.start_year, period.end_year, period.n_years) == (1981, 2009, 29)
        assert (period.start_index, period.end_index) == (0, 28)

    def test_window_inside_record_is_kept(self):
        period = self.resolve(1985, 1995)
        assert (period.start_year, period.end_year, period.n_years) == (1985, 1995, 11)
        assert period.rows == slice(4, 15)

    @pytest.mark.parametrize("window", [(1981, 2010), (1981, 2030), (1970, 2000), (1970, 2030), (2020, 2030)])
    def test_window_the_record_does_not_cover_is_clamped_to_the_true_record(self, window):
        # the last data year is 2009, so the record holds 29 years, not 30
        period = self.resolve(*window)
        assert (period.start_year, period.end_year, period.n_years) == (1981, 2009, 29)
        assert period.rows == slice(0, 29)

    def test_window_one_year_past_the_record_keeps_its_start(self):
        # long-standing behaviour: only the phantom year is dropped, not the start
        period = self.resolve(1985, 2010)
        assert (period.start_year, period.end_year, period.n_years) == (1985, 2009, 25)
        assert period.rows == slice(4, 29)

    def test_window_two_years_past_the_record_falls_back_to_the_whole_record(self):
        period = self.resolve(1985, 2011)
        assert (period.start_year, period.end_year) == (1981, 2009)

    @pytest.mark.parametrize("window", REVERSED_WINDOWS)
    def test_reversed_window_raises_wherever_it_lies(self, window):
        # it used to select no rows inside the record and, ending before the record,
        # wrap to some other rows (#1231)
        with pytest.raises(CalibrationPeriodError, match="initial year"):
            self.resolve(*window)

    def test_single_year_window_is_kept(self):
        period = self.resolve(1985, 1985)
        assert (period.n_years, period.rows) == (1, slice(4, 5))


class TestReject:
    def resolve(self, start: int, end: int):
        return resolve_calibration_period(2000, 10, start, end, policy="reject")

    def test_window_the_record_covers_resolves(self):
        period = self.resolve(2000, 2009)
        assert (period.start_year, period.end_year, period.start_index, period.end_index) == (2000, 2009, 0, 9)

    @pytest.mark.parametrize(
        ("window", "argument_name", "message"),
        [
            ((2005, 2002), "calibration_year_initial", "initial year"),
            ((1999, 2005), "calibration_year_initial", "calibration start year"),
            ((2002, 2010), "calibration_year_final", "calibration end year"),
        ],
    )
    def test_window_the_record_does_not_cover_raises(self, window, argument_name, message):
        with pytest.raises(CalibrationPeriodError, match=message) as excinfo:
            self.resolve(*window)
        assert excinfo.value.argument_name == argument_name

    def test_error_keeps_both_contracts(self):
        # EDDI has always raised InvalidArgumentError and Palmer ValueError
        with pytest.raises(CalibrationPeriodError) as excinfo:
            self.resolve(1999, 2005)
        assert isinstance(excinfo.value, InvalidArgumentError)
        assert isinstance(excinfo.value, ValueError)


class TestThroughIndices:
    """The same edge cases through the public entry points."""

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    @pytest.mark.parametrize("window", [(1981, 2009), (1981, 2010), (1981, 2030), (1970, 2009)])
    def test_spi_short_calibration_warning_counts_the_true_record(self, distribution, window):
        # 29 years of data: a window running past the record used to be widened one
        # year beyond it, and that phantom year hid the short-calibration warning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            indices.spi(_record(29), 1, distribution, 1981, *window, MONTHLY)
        assert len(_short_calibration_warnings(caught)) == 1

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    def test_spi_window_one_year_past_the_record_keeps_its_start(self, distribution):
        # the WMO 1981-2010 window on a record ending in 2009 fits 1981-2009, not the
        # whole 1976-2009 record; the phantom 2010 must not change that fit
        values = _record(34)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            past = indices.spi(values, 3, distribution, 1976, 1981, 2010, MONTHLY)
        cut = indices.spi(values, 3, distribution, 1976, 1981, 2009, MONTHLY)
        whole = indices.spi(values, 3, distribution, 1976, 1976, 2009, MONTHLY)
        np.testing.assert_array_equal(past, cut)
        assert not np.array_equal(past, whole, equal_nan=True)
        # 1981-2009 is 29 years
        assert len(_short_calibration_warnings(caught)) == 1

    def test_spi_covering_thirty_year_record_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            indices.spi(_record(30), 1, indices.Distribution.gamma, 1981, 1981, 2040, MONTHLY)
        assert _short_calibration_warnings(caught) == []

    # a window starting before the record catches a site that slices by raw year
    # offsets, since a negative start index wraps; numpy clips one past the end, so
    # a window that only runs past the record cannot tell
    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    @pytest.mark.parametrize("window", [(1970, 2015), (1981, 2050)])
    def test_spi_window_outside_the_record_equals_the_whole_record(self, distribution, window):
        values = _record(35)
        outside = indices.spi(values, 3, distribution, 1981, *window, MONTHLY)
        whole = indices.spi(values, 3, distribution, 1981, 1981, 2015, MONTHLY)
        np.testing.assert_array_equal(outside, whole)

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    @pytest.mark.parametrize("window", [(1970, 2015), (1981, 2050)])
    def test_fit_diagnostics_window_outside_the_record_equals_the_whole_record(self, distribution, window):
        values = _record(35)
        outside = indices.fit_diagnostics(values, 3, distribution, 1981, *window, MONTHLY)
        whole = indices.fit_diagnostics(values, 3, distribution, 1981, 1981, 2015, MONTHLY)
        assert outside.n_valid.tolist() == whole.n_valid.tolist()
        np.testing.assert_array_equal(outside.ks_statistic, whole.ks_statistic)

    def test_transform_fitted_gamma_zero_mass_ignores_a_window_before_the_record(self):
        values = _record(35, seed=3)
        values[::7] = 0.0
        alphas, betas = np.full(12, 2.0), np.full(12, 30.0)
        whole = compute.transform_fitted_gamma(values, 1981, 1981, 2015, MONTHLY, alphas=alphas, betas=betas)
        before = compute.transform_fitted_gamma(values, 1981, 1970, 2015, MONTHLY, alphas=alphas, betas=betas)
        np.testing.assert_array_equal(before, whole)

    def test_eddi_rejects_a_window_one_year_past_the_record(self):
        pet = np.full(10 * 12, 100.0)
        with pytest.raises(InvalidArgumentError, match="calibration end year"):
            indices.eddi(pet, 1, 2000, 2000, 2010, MONTHLY)

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    @pytest.mark.parametrize("window", REVERSED_WINDOWS)
    def test_spi_rejects_a_reversed_window(self, distribution, window):
        # the Pearson to gamma fallback must not swallow it
        values = _record(29)
        with pytest.raises(CalibrationPeriodError, match="initial year"):
            indices.spi(values, 1, distribution, 1981, *window, MONTHLY)

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    @pytest.mark.parametrize("window", REVERSED_WINDOWS)
    def test_fit_diagnostics_rejects_a_reversed_window(self, distribution, window):
        values = _record(29)
        with pytest.raises(CalibrationPeriodError, match="initial year"):
            indices.fit_diagnostics(values, 1, distribution, 1981, *window, MONTHLY)

    @pytest.mark.parametrize("window", REVERSED_WINDOWS)
    def test_transform_fitted_gamma_rejects_a_reversed_window(self, window):
        values = _record(29, seed=3)
        values[::7] = 0.0
        with pytest.raises(CalibrationPeriodError, match="initial year"):
            compute.transform_fitted_gamma(
                values, 1981, *window, MONTHLY, alphas=np.full(12, 2.0), betas=np.full(12, 30.0)
            )

    @pytest.mark.parametrize("window", REVERSED_WINDOWS)
    def test_spi_rejects_a_reversed_window_with_saved_fitting_parameters(self, window):
        # a complete supplied parameter set skips the transforms' own resolver calls
        values = _record(29, seed=3)
        values[::7] = 0.0
        fitting_params = {
            "alpha": np.full(12, 2.0),
            "beta": np.full(12, 30.0),
            "prob_zero": np.zeros(12),
        }
        with pytest.raises(CalibrationPeriodError, match="initial year"):
            indices.spi(values, 1, indices.Distribution.gamma, 1981, *window, MONTHLY, fitting_params=fitting_params)

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    @pytest.mark.parametrize("window", REVERSED_WINDOWS)
    def test_spei_rejects_a_reversed_window(self, distribution, window):
        values = _record(29)
        pet = _record(29, seed=1) / 2
        with pytest.raises(CalibrationPeriodError, match="initial year"):
            indices.spei(values, pet, 1, distribution, MONTHLY, 1981, *window)

    @pytest.mark.parametrize("window", REVERSED_WINDOWS)
    def test_eddi_rejects_a_reversed_window(self, window):
        with pytest.raises(CalibrationPeriodError, match="initial year"):
            indices.eddi(np.full(29 * 12, 100.0), 1, 1981, *window, MONTHLY)


class TestPercentageOfNormalWindow:
    """Percentage of normal rejects a window its record does not cover (#1230)."""

    @pytest.mark.parametrize("window", [(2000, 2009), (2003, 2007), (2009, 2009)])
    def test_monthly_window_the_record_covers_resolves(self, window):
        result = indices.percentage_of_normal(_record(10), 1, 2000, *window, MONTHLY)
        assert np.isfinite(result).all()

    @pytest.mark.parametrize(
        "window",
        [
            (1999, 2005),  # starts before the record
            (2005, 2012),  # partly past it: it used to average the years that exist
            (2010, 2012),  # wholly past it: it used to be all-NaN
            (2000, 2012),  # longer than the record
            (2005, 2002),  # reversed: it used to be all-NaN
        ],
    )
    def test_monthly_window_the_record_does_not_cover_raises(self, window):
        values = _record(10)
        with pytest.raises(CalibrationPeriodError):
            indices.percentage_of_normal(values, 1, 2000, *window, MONTHLY)

    @pytest.mark.parametrize("window", [(2005, 2002), (2010, 2012)])
    def test_all_missing_input_still_rejects_an_invalid_window(self, window):
        values = np.ma.masked_all((10, 12))
        with pytest.raises(CalibrationPeriodError):
            indices.percentage_of_normal(values, 1, 2000, *window, MONTHLY)

    def test_a_trailing_partial_year_counts_as_a_record_year(self):
        # 121 months from 2000 reach into 2010
        values = np.random.default_rng(0).gamma(2.0, 30.0, size=121)
        assert np.isfinite(indices.percentage_of_normal(values, 1, 2000, 2000, 2010, MONTHLY)).all()
        with pytest.raises(CalibrationPeriodError):
            indices.percentage_of_normal(values, 1, 2000, 2000, 2011, MONTHLY)

    @pytest.mark.parametrize("window", [(2000, 2002), (2001, 2002)])
    def test_daily_window_the_record_covers_resolves(self, window):
        # 1096 Gregorian days: 2000 (leap), 2001 and 2002
        values = np.random.default_rng(1).gamma(2.0, 3.0, size=1096)
        assert np.isfinite(indices.percentage_of_normal(values, 1, 2000, *window, DAILY)).any()

    @pytest.mark.parametrize("window", [(2000, 2003), (1999, 2001), (2002, 2000), (2003, 2004)])
    def test_daily_window_the_record_does_not_cover_raises(self, window):
        # (2000, 2003) used to pass a check that counted 12 steps per year
        values = np.random.default_rng(1).gamma(2.0, 3.0, size=1096)
        with pytest.raises(CalibrationPeriodError):
            indices.percentage_of_normal(values, 1, 2000, *window, DAILY)


def _clamped_warnings(caught: list[warnings.WarningMessage]) -> list[CalibrationPeriodClampedWarning]:
    return [w.message for w in caught if issubclass(w.category, CalibrationPeriodClampedWarning)]


def _monthly_precip(first_year: int, n_years: int) -> xr.DataArray:
    time = pd.date_range(f"{first_year}-01-01", periods=n_years * 12, freq="MS")
    return xr.DataArray(
        _record(n_years),
        coords={"time": time},
        dims=("time",),
        name="precipitation",
        attrs={"units": "mm/month"},
    )


class TestReportsTheWindowUsed:
    """A clamped window is announced, and the metadata names the years the fit used (#1050)."""

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    @pytest.mark.parametrize(
        ("data_start", "n_years", "window", "used"),
        [
            (1990, 31, (1981, 2010), (1990, 2020)),  # starts before the record
            (1976, 34, (1981, 2010), (1981, 2009)),  # ends one year past it, keeps its start
            (1990, 31, (1981, 2040), (1990, 2020)),  # neither end covered
        ],
    )
    def test_spi_warns_once_when_the_window_is_clamped(self, distribution, data_start, n_years, window, used):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            indices.spi(_record(n_years), 3, distribution, data_start, *window, MONTHLY)
        (warning,) = _clamped_warnings(caught)
        assert (warning.requested_years, warning.effective_years) == (window, used)
        assert f"{window[0]}-{window[1]}" in str(warning)
        assert f"{used[0]}-{used[1]}" in str(warning)

    def test_spei_warns_when_the_window_is_clamped(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            indices.spei(_record(31), _record(31, 1) * 0.5, 3, indices.Distribution.gamma, MONTHLY, 1990, 1981, 2010)
        assert len(_clamped_warnings(caught)) == 1

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    @pytest.mark.parametrize("window", [(1990, 2020), (1995, 2015)])
    def test_spi_is_silent_when_the_record_covers_the_window(self, distribution, window):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            indices.spi(_record(31), 3, distribution, 1990, *window, MONTHLY)
        assert _clamped_warnings(caught) == []

    def test_xarray_spi_attrs_report_the_window_used(self):
        # the #1050 reproduction
        precip = _monthly_precip(1990, 31)
        requested = ci.spi(
            precip,
            scale=6,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=1981,
            calibration_year_final=2010,
        )
        explicit = ci.spi(
            precip,
            scale=6,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=1990,
            calibration_year_final=2020,
        )
        assert (requested.attrs["calibration_year_initial"], requested.attrs["calibration_year_final"]) == (1990, 2020)
        np.testing.assert_array_equal(requested.values, explicit.values)

    def test_xarray_spi_attrs_keep_a_window_the_record_covers(self):
        result = ci.spi(
            _monthly_precip(1990, 31),
            scale=6,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=1995,
            calibration_year_final=2015,
        )
        assert (result.attrs["calibration_year_initial"], result.attrs["calibration_year_final"]) == (1995, 2015)

    def test_xarray_sample_size_check_reads_the_window_the_fit_uses(self):
        # 1981-2010 selects only 21 of this record's years, but the fit clamps to all 31 and
        # a single missing month leaves them well above the 30-year minimum
        precip = _monthly_precip(1990, 31)
        precip[100] = np.nan
        result = ci.spi(
            precip,
            scale=6,
            distribution=indices.Distribution.gamma,
            calibration_year_initial=1981,
            calibration_year_final=2010,
        )
        assert result.attrs["calibration_year_initial"] == 1990
