"""Hourly periodicity (#1177): the 8784-slot calendar and hour x day-of-year strata."""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climate_indices
from climate_indices import __main__ as cli_main
from climate_indices import _cli, compute, energy, indices, utils
from climate_indices.exceptions import CoordinateValidationError, InvalidArgumentError

HOURS_PER_YEAR = 8784
HOURS_PER_DAY = 24
CALENDAR_START = 2001
CALENDAR_YEARS = 12


def _hourly_values(years: range, start_year: int) -> np.ndarray:
    """Return a Gregorian hourly series starting on January 1 of ``start_year``."""
    return np.concatenate(
        [np.arange((366 if calendar.isleap(year) else 365) * HOURS_PER_DAY, dtype=float) for year in years]
    )


def _all_leap(values: np.ndarray, start_year: int, total_years: int) -> np.ndarray:
    """Fold a Gregorian hourly series into the NumPy core's 8784-slot year layout."""
    plan = utils.CalendarPlan.from_year_span(start_year, total_years, values.size, steps_per_day=HOURS_PER_DAY)
    return plan.to_all_leap(values)


class TestHourlyPeriodicityMember:
    """The enum member carries the 8784-step year and the hour unit."""

    def test_period_length_and_unit(self) -> None:
        assert compute.Periodicity.hourly.period_length == HOURS_PER_YEAR
        assert compute.Periodicity.hourly.unit() == "hour"
        assert compute.Periodicity.from_string("hourly") is compute.Periodicity.hourly


class TestHourlyCalendarPlan:
    """The calendar plan inserts 24 synthetic February 29 hours in a non-leap year."""

    def test_synthetic_hours_ramp_between_the_gap_endpoints(self) -> None:
        values = _hourly_values(range(2001, 2003), 2001)
        folded = _all_leap(values, 2001, 2)

        assert folded.size == 2 * HOURS_PER_YEAR
        first_year = folded[:HOURS_PER_YEAR]
        synthetic_start = 59 * HOURS_PER_DAY
        synthetic = first_year[synthetic_start : synthetic_start + HOURS_PER_DAY]

        # the gap runs from February 28 23:00 (hour 1415) to March 1 00:00 (hour 1416)
        assert first_year[synthetic_start - 1] == 1415.0
        assert first_year[synthetic_start + HOURS_PER_DAY] == 1416.0
        assert np.all(synthetic > 1415.0)
        assert np.all(synthetic < 1416.0)
        assert np.all(np.diff(synthetic) > 0)

    def test_leap_year_is_copied_without_a_synthetic_block(self) -> None:
        values = _hourly_values(range(2000, 2001), 2000)
        folded = _all_leap(values, 2000, 1)

        assert folded.size == HOURS_PER_YEAR
        assert np.array_equal(folded, values)

    def test_round_trip_restores_the_gregorian_series(self) -> None:
        values = _hourly_values(range(2001, 2004), 2001)
        plan = utils.CalendarPlan.from_year_span(2001, 3, values.size, steps_per_day=HOURS_PER_DAY)

        restored = plan.to_gregorian(plan.to_all_leap(values))

        assert np.array_equal(restored, values)

    def test_partial_final_year_is_padded(self) -> None:
        values = _hourly_values(range(2001, 2003), 2001)[: 365 * HOURS_PER_DAY + 100]
        plan = utils.CalendarPlan.from_year_span(2001, 2, values.size, steps_per_day=HOURS_PER_DAY)
        folded = plan.to_all_leap(values)

        assert folded.size == 2 * HOURS_PER_YEAR
        assert np.all(np.isnan(folded[-100:]))
        assert plan.populated_length == HOURS_PER_YEAR + 100

    @pytest.mark.parametrize("steps_per_day", [0, -1, 1.0, 1.5, np.nan, np.inf])
    def test_steps_per_day_must_be_a_positive_integer(self, steps_per_day) -> None:
        with pytest.raises(ValueError, match="steps per day"):
            utils.CalendarPlan(2001, (0,), steps_per_day=steps_per_day)
        with pytest.raises(ValueError):
            utils.CalendarPlan.from_year_span(2001, 1, 0, steps_per_day=steps_per_day)

    @pytest.mark.parametrize("steps_per_day", [1, 24, np.int64(24)])
    def test_integral_steps_per_day_are_accepted(self, steps_per_day) -> None:
        plan = utils.CalendarPlan.from_year_span(2001, 1, 1, steps_per_day=steps_per_day)
        values = np.ones(1)

        np.testing.assert_array_equal(plan.to_gregorian(plan.to_all_leap(values)), values)


class TestHourlyEmpiricalTransform:
    """Standardization ranks each hour x day-of-year stratum against itself."""

    # twelve years, with the first ten as the calibration period, so every stratum has
    # at least the ten valid calibration values the empirical transform requires
    ROWS = 12
    CALIBRATION_FIRST = 2000
    CALIBRATION_LAST = 2009

    @classmethod
    def _transform(cls, values: np.ndarray, output_scale: compute.OutputScale) -> np.ndarray:
        result: np.ndarray = indices.standardized_index(
            values.flatten(),
            1,
            indices.Distribution.empirical,
            data_start_year=cls.CALIBRATION_FIRST,
            calibration_year_initial=cls.CALIBRATION_FIRST,
            calibration_year_final=cls.CALIBRATION_LAST,
            periodicity=compute.Periodicity.hourly,
            output_scale=output_scale,
        ).reshape(cls.ROWS, HOURS_PER_YEAR)
        return result

    def test_probability_matches_the_plotting_position(self) -> None:
        # every column carries the same values, so each stratum's ranks are known
        values = np.repeat(np.arange(self.ROWS, dtype=float)[:, np.newaxis], HOURS_PER_YEAR, axis=1)

        probabilities = self._transform(values, "probability")

        # ten calibration values (0..9): p = (1 + count(calibration <= value)) / (10 + 2)
        assert np.allclose(probabilities[0], 2.0 / 12.0)
        assert np.allclose(probabilities[5], 7.0 / 12.0)
        assert np.allclose(probabilities[10], 11.0 / 12.0)

    def test_strata_are_ranked_independently(self) -> None:
        values = np.repeat(np.arange(self.ROWS, dtype=float)[:, np.newaxis], HOURS_PER_YEAR, axis=1)
        # keep the evaluation years inside the calibration range, then swap them in the
        # right half, so both halves carry the same value set with ranks on other rows
        values[10] = 4.0
        values[11] = 6.0
        values[10, HOURS_PER_YEAR // 2 :] = 6.0
        values[11, HOURS_PER_YEAR // 2 :] = 4.0

        probabilities = self._transform(values, "probability")

        assert np.allclose(np.sort(probabilities[:, 0]), np.sort(probabilities[:, -1]))
        assert probabilities[10, 0] == probabilities[11, -1]
        assert probabilities[10, 0] != probabilities[10, -1]

    @pytest.mark.parametrize("output_scale", compute.OUTPUT_SCALES)
    def test_output_reuses_the_rank_buffer(self, monkeypatch, output_scale) -> None:
        values = np.repeat(np.arange(self.ROWS, dtype=float)[:, np.newaxis], HOURS_PER_YEAR, axis=1)
        original = values.copy()
        buffers = []
        zeros = np.zeros

        def record_zeros(*args, **kwargs):
            buffer = zeros(*args, **kwargs)
            buffers.append(buffer)
            return buffer

        monkeypatch.setattr(compute.np, "zeros", record_zeros)

        result = compute.transform_fitted_empirical(values, 2000, 2000, 2009, compute.Periodicity.hourly, output_scale)

        assert result is buffers[0]
        np.testing.assert_array_equal(values, original)

    def test_normal_scale_is_finite_and_ordered(self) -> None:
        values = np.repeat(np.arange(self.ROWS, dtype=float)[:, np.newaxis], HOURS_PER_YEAR, axis=1)

        result = self._transform(values, "normal")

        assert np.all(np.isfinite(result))
        # the calibration years are strictly ordered; both later years exceed the
        # calibration maximum, so they share its top plotting position
        assert np.all(np.diff(result[:10], axis=0) > 0.0)
        assert np.allclose(result[10:], result[9])


class TestHourlyRejections:
    """Only the empirical transform accepts hourly data, and scale is bounded in hours."""

    @pytest.mark.parametrize("distribution", [indices.Distribution.gamma, indices.Distribution.pearson])
    def test_parametric_distributions_are_rejected(self, distribution: indices.Distribution) -> None:
        values = np.zeros(2 * HOURS_PER_YEAR)

        with pytest.raises(InvalidArgumentError, match="Hourly periodicity supports only the empirical distribution"):
            indices.standardized_index(
                values,
                1,
                distribution,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2001,
                periodicity=compute.Periodicity.hourly,
            )

    def test_beta_is_rejected_through_an_energy_index(self) -> None:
        values = np.repeat(np.linspace(0.0, 1.0, HOURS_PER_YEAR)[np.newaxis, :], 2, axis=0).flatten()

        with pytest.raises(InvalidArgumentError, match="Hourly periodicity supports only the empirical distribution"):
            energy.srepi(
                values,
                24,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2001,
                periodicity=compute.Periodicity.hourly,
                distribution=indices.Distribution.beta,
            )

    def test_scale_is_bounded_in_hours(self) -> None:
        values = np.zeros(2 * HOURS_PER_YEAR)

        with pytest.raises(InvalidArgumentError, match="Scale must be an integer"):
            indices.standardized_index(
                values,
                6 * HOURS_PER_YEAR + 1,
                indices.Distribution.empirical,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2001,
                periodicity=compute.Periodicity.hourly,
            )

    def test_non_standardized_surfaces_reject_hourly(self) -> None:
        values = np.repeat(np.arange(2, dtype=float)[:, np.newaxis], HOURS_PER_YEAR, axis=1).flatten()

        with pytest.raises(InvalidArgumentError, match="supported only by the standardized indices"):
            indices.percentage_of_normal(
                values,
                1,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2001,
                periodicity=compute.Periodicity.hourly,
            )

    def test_annual_scale_is_accepted(self) -> None:
        values = np.repeat(np.arange(2, dtype=float)[:, np.newaxis], HOURS_PER_YEAR, axis=1).flatten()

        result = indices.standardized_index(
            values,
            HOURS_PER_YEAR,
            indices.Distribution.empirical,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2001,
            periodicity=compute.Periodicity.hourly,
            aggregation="mean",
        )

        assert result.size == values.size


class TestHourlyXarray:
    """The xarray adapter infers hourly cadence and restores the Gregorian axis."""

    @staticmethod
    def _hourly_data() -> xr.DataArray:
        time = pd.date_range("2001-01-01", "2012-12-31 23:00", freq="h")
        values = np.sin(np.arange(time.size, dtype=float) * 2.0 * np.pi / HOURS_PER_DAY)
        return xr.DataArray(values, coords={"time": time}, dims=("time",), name="supply")

    def test_srepi_restores_the_gregorian_axis(self) -> None:
        data = self._hourly_data()

        result = energy.srepi(data, scale=24)

        assert result.dims == data.dims
        assert np.array_equal(result["time"].values, data["time"].values)
        # only the leading scale - 1 hours of the rolling window are missing
        assert np.isfinite(result.values[23:]).all()
        assert result.attrs["long_name"] == "Standardized Renewable Energy Production Index"

    def test_periodicity_mismatch_is_rejected(self) -> None:
        data = self._hourly_data().isel(time=slice(0, 2 * HOURS_PER_YEAR))

        with pytest.raises(CoordinateValidationError, match="does not match the requested daily"):
            energy.srepi(data, scale=1, periodicity=compute.Periodicity.daily)

    def test_sub_hourly_cadence_is_not_read_as_hourly(self) -> None:
        time = pd.date_range("2001-01-01", periods=48, freq="30min")
        data = xr.DataArray(np.arange(48, dtype=float), coords={"time": time}, dims=("time",))

        with pytest.raises(CoordinateValidationError):
            energy.srepi(data, scale=1)

    @pytest.mark.parametrize("start", ["2001-01-01 06:00", "2001-01-01 00:30"])
    def test_hourly_input_must_start_at_the_hour_zero_of_january_first(self, start: str) -> None:
        time = pd.date_range(start, periods=48, freq="h")
        data = xr.DataArray(np.arange(48, dtype=float), coords={"time": time}, dims=("time",))

        with pytest.raises(CoordinateValidationError, match="calendar-hour"):
            energy.srepi(data, scale=1)


class TestHourlyEnergyAdapters:
    """Every standardized energy adapter runs hourly and rejects the beta family eagerly."""

    @staticmethod
    def _data(sign: float = 1.0) -> xr.DataArray:
        time = pd.date_range("2001-01-01", "2012-12-31 23:00", freq="h")
        values = sign * (np.sin(np.arange(time.size, dtype=float) * 2.0 * np.pi / HOURS_PER_DAY) + 2.0)
        return xr.DataArray(values, coords={"time": time}, dims=("time",), name="series")

    @staticmethod
    def _numpy_reference(adapter, data: xr.DataArray) -> np.ndarray:
        """Standardize through the NumPy path and the calendar plan the adapter must reproduce."""
        values = data.values
        plan = utils.CalendarPlan.from_year_span(2001, 12, values.size, steps_per_day=HOURS_PER_DAY)
        standardized = adapter(
            plan.to_all_leap(values),
            scale=24,
            data_start_year=2001,
            calibration_year_initial=2001,
            calibration_year_final=2012,
            periodicity=compute.Periodicity.hourly,
        )
        reference: np.ndarray = plan.to_gregorian(standardized)
        return reference

    @pytest.mark.parametrize(
        ("adapter", "sign"),
        [
            (energy.srepi, 1.0),
            (energy.srli, 1.0),
            (energy.srli, -1.0),
            (energy.standardized_demand_index, 1.0),
        ],
        ids=["srepi", "srli-positive", "srli-signed", "demand-index"],
    )
    def test_hourly_result_matches_the_numpy_path(self, adapter, sign) -> None:
        data = self._data(sign)

        result = adapter(data, scale=24)
        reference = self._numpy_reference(adapter, data)

        assert np.array_equal(result["time"].values, data["time"].values)
        # the first 23 windows are incomplete, and nothing after them is missing
        assert np.isnan(reference[:23]).all()
        assert np.isfinite(reference[23:]).all()
        assert np.abs(reference[23:]).max() > 1.0
        np.testing.assert_allclose(result.values, reference, equal_nan=True)

    def test_hourly_beta_is_rejected_before_the_computation_runs(self) -> None:
        data = self._data().chunk({"time": -1})

        with pytest.raises(InvalidArgumentError, match="Hourly periodicity supports only the empirical distribution"):
            energy.srepi(data, scale=24, distribution=indices.Distribution.beta)

    def test_hourly_dask_input_standardizes(self) -> None:
        data = self._data().chunk({"time": -1})

        result = energy.srepi(data, scale=24)

        assert result.chunks is not None
        np.testing.assert_allclose(result.compute().values, energy.srepi(self._data(), scale=24).values, equal_nan=True)


class TestHourlyScaleBounds:
    """The hourly scale bound is six years expressed in hours."""

    def test_the_bound_is_accepted_and_one_more_is_rejected(self) -> None:
        indices._validate_scale(indices.SCALE_MAX_HOURLY, compute.Periodicity.hourly)

        with pytest.raises(InvalidArgumentError, match="Scale must be an integer"):
            indices._validate_scale(indices.SCALE_MAX_HOURLY + 1, compute.Periodicity.hourly)

    def test_the_bound_spans_six_hourly_years(self) -> None:
        assert indices.SCALE_MAX_HOURLY == 6 * HOURS_PER_YEAR


class TestHourlyTimeMajorBlock:
    """A time-major hourly block is standardized cell by cell, partial final year included."""

    YEARS = 12
    TAIL_HOURS = 100
    CELLS = 3

    @classmethod
    def _series(cls, cell: int) -> np.ndarray:
        """Return one cell's series, `TAIL_HOURS` short of whole years."""
        rng = np.random.default_rng(cell)
        return rng.normal(size=cls.YEARS * HOURS_PER_YEAR - cls.TAIL_HOURS)

    @staticmethod
    def _standardize(values: np.ndarray, **kwargs) -> np.ndarray:
        result: np.ndarray = indices.standardized_index(
            values,
            1,
            indices.Distribution.empirical,
            data_start_year=2000,
            calibration_year_initial=2000,
            calibration_year_final=2009,
            periodicity=compute.Periodicity.hourly,
            **kwargs,
        )
        return result

    def test_block_matches_the_per_cell_series(self) -> None:
        series = [self._series(cell) for cell in range(self.CELLS)]
        block = np.stack(series, axis=1)[..., np.newaxis]

        result = self._standardize(block, spatial_time_major=True)

        assert result.shape == block.shape
        for cell, cell_series in enumerate(series):
            np.testing.assert_allclose(result[:, cell, 0], self._standardize(cell_series), equal_nan=True)
        # cells carry different data, so a transform that mixed or ignored them would show here
        assert not np.allclose(result[:, 0, 0], result[:, 1, 0])
        assert not np.allclose(result[:, 0, 0], block[:, 0, 0])

    def test_unambiguous_shape_needs_no_declaration(self) -> None:
        block = np.stack([self._series(cell) for cell in range(self.CELLS)], axis=1)[..., np.newaxis]

        np.testing.assert_array_equal(self._standardize(block), self._standardize(block, spatial_time_major=True))

    def test_a_first_cell_axis_of_the_period_length_must_be_declared(self) -> None:
        ambiguous = np.zeros((self.YEARS, HOURS_PER_YEAR, 1))

        with pytest.raises(ValueError, match="spatial_time_major"):
            self._standardize(ambiguous)


class TestHourlySurfaceGates:
    """Surfaces that cannot standardize hourly data reject it eagerly."""

    def test_spi_and_spei_reject_hourly_with_the_surface_message(self) -> None:
        values = np.ones(2 * HOURS_PER_YEAR)
        gamma = indices.Distribution.gamma
        hourly = compute.Periodicity.hourly

        with pytest.raises(InvalidArgumentError, match="supported only by the standardized indices"):
            indices.spi(values, 1, gamma, 2000, 2000, 2001, hourly)
        with pytest.raises(InvalidArgumentError, match="supported only by the standardized indices"):
            indices.spei(values, values, 1, gamma, hourly, 2000, 2000, 2001)

    def test_fit_diagnostics_rejects_hourly(self) -> None:
        values = np.zeros(2 * HOURS_PER_YEAR)

        with pytest.raises(InvalidArgumentError, match="supported only by the standardized indices"):
            indices.fit_diagnostics(
                values,
                1,
                indices.Distribution.gamma,
                data_start_year=2000,
                calibration_year_initial=2000,
                calibration_year_final=2001,
                periodicity=compute.Periodicity.hourly,
            )

    def test_fit_diagnostics_adapter_rejects_hourly_before_building_a_graph(self) -> None:
        time = pd.date_range("2001-01-01", "2012-12-31 23:00", freq="h")
        data = xr.DataArray(np.ones(time.size), coords={"time": time}, dims=("time",)).chunk({"time": -1})

        with pytest.raises(InvalidArgumentError, match="supported only by the standardized indices"):
            climate_indices.fit_diagnostics(data, 24, indices.Distribution.gamma)

    def test_compute_seam_rejects_a_parametric_hourly_fit(self) -> None:
        values = np.tile(np.arange(2, dtype=float)[:, np.newaxis], (1, HOURS_PER_YEAR)).ravel()

        with pytest.raises(InvalidArgumentError, match="Hourly periodicity supports only the empirical distribution"):
            compute.fit_and_standardize(
                values,
                indices.Distribution.gamma,
                data_start_year=2000,
                calibration_start_year=2000,
                calibration_end_year=2001,
                periodicity=compute.Periodicity.hourly,
            )

    @pytest.mark.parametrize(
        ("transform", "parameter_names"),
        [
            (compute.transform_fitted_gamma, ("alphas", "betas", "probabilities_of_zero")),
            (compute.transform_fitted_beta, ("alphas", "betas", "probabilities_of_zero", "probabilities_of_one")),
            (compute.transform_fitted_pearson, ("probabilities_of_zero", "locs", "scales", "skews")),
            (compute.transform_fitted_loglogistic, ("locs", "scales", "shapes")),
        ],
    )
    @pytest.mark.parametrize("value", [0.5, np.nan])
    @pytest.mark.parametrize("supplied", [False, True])
    def test_public_parametric_transforms_reject_hourly(self, transform, parameter_names, value, supplied) -> None:
        values = np.full((10, HOURS_PER_YEAR), value)
        parameters = {name: np.full(HOURS_PER_YEAR, 0.1) for name in parameter_names} if supplied else {}

        with pytest.raises(InvalidArgumentError, match="Hourly periodicity supports only the empirical distribution"):
            transform(values, 2000, 2000, 2009, compute.Periodicity.hourly, **parameters)

    def test_the_cli_does_not_offer_hourly(self) -> None:
        assert compute.Periodicity.hourly not in _cli.CLI_PERIODICITIES

        parser = cli_main._build_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["--periodicity", "hourly", "--index", "srepi"])
