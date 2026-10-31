"""Smoothed seasonal-diurnal climatology for hourly empirical standardization (#1407).

``pool_window_days`` widens each hour x day-of-year stratum to the same hour on the
neighbouring days, so a thin calibration step pools more values. The default stays the
shipped hour x day-of-year strata, so every existing hourly result is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import climate_indices
from climate_indices import compute, energy, indices, utils
from climate_indices.exceptions import InvalidArgumentError

HOURS_PER_YEAR = compute.Periodicity.hourly.period_length
HOURS_PER_DAY = 24
FEBRUARY_29_START = 59 * HOURS_PER_DAY
CALIBRATION_FIRST = 2000
CALIBRATION_LAST = 2011  # twelve years, so every calendar step clears the ten-value minimum
POOL_WINDOW_DAYS = 7


def _ramp_rows(years: int = 12) -> np.ndarray:
    """Return one value per year, constant across the year, so every step has known ranks."""
    return np.repeat(np.arange(years, dtype=float)[:, np.newaxis], HOURS_PER_YEAR, axis=1)


def _probabilities(
    values: np.ndarray,
    pool_window_days: int = 0,
    calibration_last: int = CALIBRATION_LAST,
    scale: int = 1,
    calibration_first: int = CALIBRATION_FIRST,
) -> np.ndarray:
    """Standardize a (years, 8784) block on the probability scale in the hourly layout."""
    result = indices.standardized_index(
        values.ravel(),
        scale,
        indices.Distribution.empirical,
        data_start_year=CALIBRATION_FIRST,
        calibration_year_initial=calibration_first,
        calibration_year_final=calibration_last,
        periodicity=compute.Periodicity.hourly,
        output_scale="probability",
        pool_window_days=pool_window_days,
        clip_negatives=False,
    )
    return np.asarray(result).reshape(values.shape)


def _pooled_reference(sample: np.ndarray, row: int, column: int, pool_window_days: int) -> float:
    """Rebuild one step's pooled plotting position with explicit wrap and day indexing."""
    window = [
        (column + HOURS_PER_DAY * offset) % HOURS_PER_YEAR for offset in range(-pool_window_days, pool_window_days + 1)
    ]
    window = [index for index in window if not FEBRUARY_29_START <= index < FEBRUARY_29_START + HOURS_PER_DAY]
    pooled = sample[:, window].ravel()
    pooled = pooled[~np.isnan(pooled)]
    value = sample[row, column]
    if np.isnan(value) or pooled.size < 10:
        return float("nan")
    return float((1.0 + np.count_nonzero(pooled <= value)) / (pooled.size + 2.0))


@pytest.fixture
def observed_hourly_temperatures() -> np.ndarray:
    """Load observed Fresno January temperatures into the all-leap hourly layout."""
    directory = Path(__file__).parent / "fixture" / "hourly_observed"
    path = directory / "fresno_january_2000_2011.csv.gz"
    provenance = json.loads((directory / "provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["checksum_sha256"]
    observations = pd.read_csv(path, parse_dates=["valid"])
    assert len(observations) == 8893
    assert np.all(observations["station"].to_numpy() == "FAT")
    assert np.all(observations["valid"].dt.month.to_numpy() == 1)
    # Keep the last routine report in each UTC hour; never fill missing observations.
    observations = observations.loc[~observations["valid"].dt.floor("h").duplicated(keep="last")]
    time = observations["valid"].dt
    values = np.full((12, HOURS_PER_YEAR), np.nan)
    values[time.year - CALIBRATION_FIRST, (time.day - 1) * HOURS_PER_DAY + time.hour] = observations["tmpf"]
    return values


class TestDefaultWindow:
    """A zero window keeps the shipped hour x day-of-year strata."""

    def test_default_matches_the_plotting_position(self) -> None:
        probabilities = _probabilities(_ramp_rows())

        # twelve calibration values: p = (1 + count(calibration <= value)) / (12 + 2)
        np.testing.assert_allclose(probabilities[0], 2.0 / 14.0)
        np.testing.assert_allclose(probabilities[6], 8.0 / 14.0)
        np.testing.assert_allclose(probabilities[11], 13.0 / 14.0)

    def test_default_window_is_the_omitted_argument(self) -> None:
        values = _ramp_rows()

        np.testing.assert_array_equal(_probabilities(values), _probabilities(values, pool_window_days=0))


class TestPooledProbability:
    """The pooled sample is the same hour on the neighbouring days, year boundary included."""

    def test_matches_an_explicit_window_reference(self) -> None:
        rng = np.random.default_rng(1407)
        sample = rng.normal(size=(12, HOURS_PER_YEAR))
        probabilities = _probabilities(sample, POOL_WINDOW_DAYS)

        columns = (
            0,
            1,
            HOURS_PER_DAY,
            FEBRUARY_29_START - 1,
            FEBRUARY_29_START,
            FEBRUARY_29_START + 1,
            365 * HOURS_PER_DAY,
            HOURS_PER_YEAR - 1,
        )
        for column in columns:
            for row in (0, 5, 11):
                expected = _pooled_reference(sample, row, column, POOL_WINDOW_DAYS)
                assert probabilities[row, column] == pytest.approx(expected, abs=1e-15), (column, row)

    def test_pooling_ranks_against_more_calibration_values(self) -> None:
        probabilities = _probabilities(_ramp_rows(), POOL_WINDOW_DAYS)

        # fifteen pooled days of twelve years, none on 29 February: p = (1 + 15) / (180 + 2)
        assert probabilities[0, 100 * HOURS_PER_DAY] == pytest.approx(16.0 / 182.0)
        # the same value scores lower than under its own stratum, which has twelve values
        assert probabilities[0, 100 * HOURS_PER_DAY] < (2.0 / 14.0)

    def test_integer_observations_match_floating_point(self) -> None:
        values = _ramp_rows().astype(int)
        original = values.copy()

        np.testing.assert_array_equal(
            _probabilities(values, POOL_WINDOW_DAYS), _probabilities(values.astype(float), POOL_WINDOW_DAYS)
        )
        np.testing.assert_array_equal(values, original)

    def test_largest_window_counts_each_calendar_day_once(self) -> None:
        values = _ramp_rows()
        probabilities = _probabilities(values, 182)

        for column in (0, FEBRUARY_29_START, 200 * HOURS_PER_DAY):
            expected = _pooled_reference(values, 0, column, 182)
            assert probabilities[0, column] == pytest.approx(expected, abs=1e-15)

    def test_probability_is_monotone_in_the_value(self) -> None:
        probabilities = _probabilities(_ramp_rows(), POOL_WINDOW_DAYS)

        assert np.all(np.diff(probabilities[:, 0]) > 0.0)

    def test_scale_one_output_keeps_the_input_shape(self) -> None:
        values = _ramp_rows()

        assert _probabilities(values, POOL_WINDOW_DAYS).shape == values.shape


class TestFebruary29AndYearBoundary:
    """29 February is held out of every pooled sample, and the window wraps the year."""

    def test_february_29_is_held_out_of_every_pooled_sample(self) -> None:
        # every value is a tie, so each step's probability depends only on its sample size
        probabilities = _probabilities(np.zeros((12, HOURS_PER_YEAR)), POOL_WINDOW_DAYS)

        # a step far from 29 February pools fifteen days of twelve years
        assert probabilities[0, 100 * HOURS_PER_DAY] == pytest.approx(181.0 / 182.0)
        # 28 February and 29 February each pool fourteen days, because the held-out day
        # reaches both windows and is never counted
        assert probabilities[0, 58 * HOURS_PER_DAY] == pytest.approx(169.0 / 170.0)
        assert probabilities[0, FEBRUARY_29_START] == pytest.approx(169.0 / 170.0)

    def test_synthesized_february_29_values_change_no_other_step(self) -> None:
        sample = _ramp_rows()
        baseline = _probabilities(sample, POOL_WINDOW_DAYS)
        sample[:, FEBRUARY_29_START : FEBRUARY_29_START + HOURS_PER_DAY] = 1.0e9

        perturbed = _probabilities(sample, POOL_WINDOW_DAYS)

        # the held-out day never enters another step's sample, so only its own target moves
        mask = np.ones(HOURS_PER_YEAR, dtype=bool)
        mask[FEBRUARY_29_START : FEBRUARY_29_START + HOURS_PER_DAY] = False
        np.testing.assert_array_equal(perturbed[:, mask], baseline[:, mask])

    def test_holdout_does_not_depend_on_the_calibration_memory_layout(self) -> None:
        calibration = np.random.default_rng(1407).normal(size=(12, HOURS_PER_YEAR))

        c_order = compute._pooled_empirical_counts(calibration, calibration, POOL_WINDOW_DAYS)
        fortran_order = compute._pooled_empirical_counts(calibration, np.asfortranarray(calibration), POOL_WINDOW_DAYS)

        for expected, actual in zip(c_order, fortran_order, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_scaled_windows_spanning_february_29_are_held_out(self) -> None:
        scale = 2
        sample = np.random.default_rng(1407).normal(size=(12, HOURS_PER_YEAR))
        baseline = _probabilities(sample, POOL_WINDOW_DAYS, scale=scale)
        sample[:, FEBRUARY_29_START : FEBRUARY_29_START + HOURS_PER_DAY] = 1.0e9

        perturbed = _probabilities(sample, POOL_WINDOW_DAYS, scale=scale)

        # every step whose own scaled window spans 29 February is the only step that can
        # move; every other step held out the contaminated scaled values from its sample
        untouched = np.ones(HOURS_PER_YEAR, dtype=bool)
        untouched[FEBRUARY_29_START : FEBRUARY_29_START + HOURS_PER_DAY + scale - 1] = False
        np.testing.assert_array_equal(perturbed[:, untouched], baseline[:, untouched])

    def test_scaled_windows_before_the_calibration_block_are_held_out(self) -> None:
        # a scale that reaches from the previous year's synthesized 29 February into the
        # calibration block; the block's flat positions start at the first calibration
        # year, so the spill-in has to be marked there rather than from the record start
        scale = 8000
        pool_window_days = 168  # wide enough that the holdout cannot empty a pooled sample
        values = np.random.default_rng(1407).normal(size=(13, HOURS_PER_YEAR))
        baseline = _probabilities(values, pool_window_days, scale=scale, calibration_first=CALIBRATION_FIRST + 1)
        values[0, FEBRUARY_29_START : FEBRUARY_29_START + HOURS_PER_DAY] = 1.0e9

        perturbed = _probabilities(values, pool_window_days, scale=scale, calibration_first=CALIBRATION_FIRST + 1)

        # only a target whose own scaled window spans the perturbed day may move; no
        # other step may count the contaminated scaled values from the year before
        contaminated = np.zeros(values.size, dtype=bool)
        contaminated[FEBRUARY_29_START : FEBRUARY_29_START + HOURS_PER_DAY + scale - 1] = True
        np.testing.assert_array_equal(perturbed.ravel()[~contaminated], baseline.ravel()[~contaminated])

    def test_the_pooled_window_wraps_the_year_boundary(self) -> None:
        values = np.zeros((12, HOURS_PER_YEAR))
        baseline = _probabilities(values, POOL_WINDOW_DAYS)
        values[:, 365 * HOURS_PER_DAY : 366 * HOURS_PER_DAY] = 1.0e6

        perturbed = _probabilities(values, POOL_WINDOW_DAYS)

        # 31 December reaches 1 January's window, and a step far from the boundary does not move
        assert perturbed[0, 0] != baseline[0, 0]
        assert perturbed[0, 100 * HOURS_PER_DAY] == baseline[0, 100 * HOURS_PER_DAY]


class TestTailResolution:
    """Pooling refines the tail resolution of a thin hourly calibration."""

    def test_pooling_adds_distinct_tail_probabilities(self, observed_hourly_temperatures: np.ndarray) -> None:
        values = _ramp_rows()
        # eleven calibration years: above the ten-value minimum, below the 100-value
        # recommendation recorded in tests/fixture/energy_sei/warnings.txt
        default = _probabilities(values, 0, calibration_last=2010)
        pooled = _probabilities(values, POOL_WINDOW_DAYS, calibration_last=2010)

        assert len(np.unique(default)) == 11
        assert len(np.unique(pooled)) > len(np.unique(default))
        # the rarest pooled value scores lower, so a rarer event is resolvable
        assert pooled.min() < default.min()

        observed = observed_hourly_temperatures
        calibration, targets = observed[:-1], observed[-1]
        # Score 2011 independently; January 8-24 keeps every seven-day window in January.
        columns = np.arange(7 * HOURS_PER_DAY, 24 * HOURS_PER_DAY)
        columns = columns[np.isfinite(observed[:, columns]).all(axis=0)]
        strata = (1 + np.count_nonzero(calibration[:, columns] <= targets[columns], axis=0)) / 13.0
        columns = columns[strata <= 0.2]
        assert columns.size == 49

        rms_changes = []
        full_tails = []
        for window in (0, POOL_WINDOW_DAYS):
            full = _probabilities(observed, window, calibration_last=2010)[-1, columns]
            # Recount observed same-hour samples independently of either implementation.
            neighbours = columns[:, np.newaxis] + HOURS_PER_DAY * np.arange(-window, window + 1)
            sample = calibration[:, neighbours]
            expected = (1 + np.count_nonzero(sample <= targets[columns, np.newaxis], axis=(0, 2))) / (
                2 + np.count_nonzero(np.isfinite(sample), axis=(0, 2))
            )
            np.testing.assert_allclose(full, expected, rtol=0, atol=1e-15)
            full_tails.append(full)

            changes = []
            for year in range(calibration.shape[0]):
                reduced = observed.copy()
                reduced[year] = np.nan
                tail = _probabilities(reduced, window, calibration_last=2010)[-1, columns]
                assert np.isfinite(tail).all()
                changes.append(tail - full)
            rms_changes.append(np.sqrt(np.mean(np.square(changes))))

        assert np.unique(full_tails[1]).size > np.unique(full_tails[0]).size
        assert full_tails[1].min() < full_tails[0].min()
        # This fixture measures year-removal sensitivity, not independent-sample uncertainty.
        assert 0 < rms_changes[1] < rms_changes[0]


class TestPoolingRejections:
    """The window is a non-negative integer on the hourly empirical path only."""

    @pytest.mark.parametrize("pool_window_days", [-1, 183, 366, 10**9, 1.0, 1.5, np.nan, True, "3", None])
    def test_invalid_window_is_rejected(self, pool_window_days: object) -> None:
        with pytest.raises(InvalidArgumentError, match="pool_window_days"):
            indices.standardized_index(
                np.zeros(2 * HOURS_PER_YEAR),
                1,
                indices.Distribution.empirical,
                data_start_year=CALIBRATION_FIRST,
                calibration_year_initial=CALIBRATION_FIRST,
                calibration_year_final=CALIBRATION_FIRST + 1,
                periodicity=compute.Periodicity.hourly,
                pool_window_days=pool_window_days,
            )

    @pytest.mark.parametrize("pool_window_days", [-1, 183, "3", 1])
    def test_all_missing_numpy_input_still_validates_the_window(self, pool_window_days: object) -> None:
        with pytest.raises(InvalidArgumentError, match="pool_window_days"):
            indices.standardized_index(
                np.full(36, np.nan),
                1,
                indices.Distribution.gamma,
                CALIBRATION_FIRST,
                CALIBRATION_FIRST,
                CALIBRATION_FIRST + 2,
                compute.Periodicity.monthly,
                pool_window_days=pool_window_days,
            )

    @pytest.mark.parametrize("pool_window_days", [-1, 183, "3", True, 1])
    @pytest.mark.parametrize("distribution", [indices.Distribution.empirical, indices.Distribution.gamma])
    def test_reusable_fit_validates_the_window(
        self, pool_window_days: object, distribution: indices.Distribution
    ) -> None:
        with pytest.raises(InvalidArgumentError, match="pool_window_days"):
            compute.FittedDistribution.resolve(
                np.ones((12, 12)),
                distribution,
                CALIBRATION_FIRST,
                CALIBRATION_FIRST,
                CALIBRATION_LAST,
                compute.Periodicity.monthly,
                pool_window_days=pool_window_days,
            )

    @pytest.mark.parametrize("distribution", [indices.Distribution.empirical, indices.Distribution.gamma])
    def test_zero_window_remains_valid_for_monthly_fits(self, distribution: indices.Distribution) -> None:
        fitted = compute.FittedDistribution.resolve(
            np.random.default_rng(1407).uniform(1.0, 10.0, size=(12, 12)),
            distribution,
            CALIBRATION_FIRST,
            CALIBRATION_FIRST,
            CALIBRATION_LAST,
            compute.Periodicity.monthly,
            pool_window_days=0,
        )
        assert fitted.pool_window_days == 0

    def test_a_non_hourly_calendar_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError, match="hourly empirical"):
            indices.standardized_index(
                np.zeros(3 * 12),
                1,
                indices.Distribution.empirical,
                data_start_year=CALIBRATION_FIRST,
                calibration_year_initial=CALIBRATION_FIRST,
                calibration_year_final=CALIBRATION_FIRST + 1,
                periodicity=compute.Periodicity.monthly,
                pool_window_days=1,
            )

    def test_a_non_empirical_distribution_is_rejected(self) -> None:
        with pytest.raises(InvalidArgumentError, match="hourly empirical"):
            compute._validate_pool_window_days(1, compute.Periodicity.hourly, indices.Distribution.gamma)

    def test_a_full_cycle_scale_is_rejected_with_pooling(self) -> None:
        rows = _ramp_rows()
        with pytest.raises(InvalidArgumentError, match="full hourly cycle"):
            _probabilities(rows, POOL_WINDOW_DAYS, scale=8761)

    def test_a_scaled_february_29_holdout_cannot_cover_the_pooled_window(self) -> None:
        rows = _ramp_rows()
        with pytest.raises(InvalidArgumentError, match="would empty some pooled samples") as exc_info:
            _probabilities(rows, pool_window_days=1, scale=26)

        assert exc_info.value.argument_name == "pool_window_days"
        assert exc_info.value.argument_value == "1"
        assert exc_info.value.valid_values == "at least 2 days for scale 26"

    @pytest.mark.parametrize("scale", [0, -1, 1.5, True, "24", None])
    def test_a_pooled_window_requires_a_positive_integer_scale(self, scale: object) -> None:
        with pytest.raises(InvalidArgumentError, match="scale") as exc_info:
            compute._validate_pool_window_days(
                POOL_WINDOW_DAYS,
                compute.Periodicity.hourly,
                indices.Distribution.empirical,
                scale=scale,  # type: ignore[arg-type]
            )

        assert exc_info.value.argument_name == "scale"

    def test_the_narrow_window_error_names_no_window_when_pooling_is_unavailable(self) -> None:
        validate = compute._validate_pool_window_days
        hourly, empirical = compute.Periodicity.hourly, indices.Distribution.empirical
        # the largest scale whose 29 February holdout a 182-day window still outlasts
        validate(182, hourly, empirical, scale=8713)
        with pytest.raises(InvalidArgumentError, match="would empty some pooled samples") as exc_info:
            validate(182, hourly, empirical, scale=8714)

        assert exc_info.value.valid_values == "none: pooling is unavailable for scale 8714"

    def test_the_public_transform_rejects_a_full_cycle_scale(self) -> None:
        with pytest.raises(InvalidArgumentError, match="full hourly cycle"):
            compute.transform_fitted_empirical(
                np.zeros((2, HOURS_PER_YEAR)),
                CALIBRATION_FIRST,
                CALIBRATION_FIRST,
                CALIBRATION_FIRST + 1,
                compute.Periodicity.hourly,
                pool_window_days=POOL_WINDOW_DAYS,
                scale=8761,
            )

    def test_the_public_transform_and_seam_reject_the_same(self) -> None:
        monthly = np.zeros((3, 12))
        with pytest.raises(InvalidArgumentError, match="hourly empirical"):
            compute.transform_fitted_empirical(
                monthly,
                CALIBRATION_FIRST,
                CALIBRATION_FIRST,
                CALIBRATION_FIRST + 1,
                compute.Periodicity.monthly,
                pool_window_days=1,
            )
        with pytest.raises(InvalidArgumentError, match="hourly empirical"):
            compute.fit_and_standardize(
                monthly,
                indices.Distribution.empirical,
                data_start_year=CALIBRATION_FIRST,
                calibration_start_year=CALIBRATION_FIRST,
                calibration_end_year=CALIBRATION_FIRST + 1,
                periodicity=compute.Periodicity.monthly,
                pool_window_days=1,
            )


class TestEnergyWrappers:
    """Every standardized energy wrapper forwards the window on both tiers."""

    NUMPY_KWARGS = {
        "data_start_year": CALIBRATION_FIRST,
        "calibration_year_initial": CALIBRATION_FIRST,
        "calibration_year_final": CALIBRATION_LAST,
        "periodicity": compute.Periodicity.hourly,
    }

    @pytest.mark.parametrize("adapter", [energy.srepi, energy.standardized_demand_index, energy.srli])
    def test_numpy_forwarding_changes_the_result(self, adapter: Callable[..., Any]) -> None:
        values = _ramp_rows()

        default = adapter(values, 1, **self.NUMPY_KWARGS)
        pooled = adapter(values, 1, pool_window_days=POOL_WINDOW_DAYS, **self.NUMPY_KWARGS)

        assert not np.allclose(default, pooled, equal_nan=True)

    @staticmethod
    def _gregorian_hourly() -> tuple[pd.DatetimeIndex, np.ndarray]:
        time = pd.date_range(f"{CALIBRATION_FIRST}-01-01", f"{CALIBRATION_LAST}-12-31 23:00", freq="h")
        values = np.sin(np.arange(time.size, dtype=float) * 2.0 * np.pi / HOURS_PER_DAY)
        return time, values

    def test_xarray_result_matches_the_numpy_path(self) -> None:
        time, values = self._gregorian_hourly()
        data = xr.DataArray(values, coords={"time": time}, dims=("time",), name="supply")

        result = energy.srepi(
            data,
            scale=1,
            aggregation="mean",
            pool_window_days=POOL_WINDOW_DAYS,
            data_start_year=CALIBRATION_FIRST,
            calibration_year_initial=CALIBRATION_FIRST,
            calibration_year_final=CALIBRATION_LAST,
        )
        plan = utils.CalendarPlan.from_year_span(
            CALIBRATION_FIRST, CALIBRATION_LAST - CALIBRATION_FIRST + 1, values.size, steps_per_day=HOURS_PER_DAY
        )
        reference = plan.to_gregorian(
            energy.srepi(
                plan.to_all_leap(values),
                1,
                aggregation="mean",
                pool_window_days=POOL_WINDOW_DAYS,
                **self.NUMPY_KWARGS,
            )
        )

        assert np.array_equal(result["time"].values, time.values)
        np.testing.assert_allclose(result.values, reference, equal_nan=True)

    def test_xarray_rejects_an_invalid_window_before_building_a_graph(self) -> None:
        time, values = self._gregorian_hourly()
        data = xr.DataArray(values, coords={"time": time}, dims=("time",), name="supply").chunk({"time": -1})

        with pytest.raises(InvalidArgumentError, match="pool_window_days"):
            energy.srepi(data, scale=1, pool_window_days=-1)

    @pytest.mark.parametrize(
        "adapter", [energy.srepi, energy.standardized_demand_index, energy.srli, climate_indices.standardized_index]
    )
    @pytest.mark.parametrize("pool_window_days", [0, POOL_WINDOW_DAYS])
    def test_xarray_records_the_calibration_window(self, adapter: Callable[..., Any], pool_window_days: int) -> None:
        time, values = self._gregorian_hourly()
        data = xr.DataArray(values, coords={"time": time}, dims=("time",)).chunk({"time": -1})
        result = adapter(
            data,
            scale=1,
            distribution=indices.Distribution.empirical,
            pool_window_days=pool_window_days,
        )

        assert result.attrs["pool_window_days"] == pool_window_days
        provenance = json.loads(result.attrs["climate_indices_provenance"])
        assert provenance["parameters"]["pool_window_days"] == pool_window_days

    @pytest.mark.parametrize("adapter", [energy.srepi, energy.standardized_demand_index, energy.srli])
    @pytest.mark.parametrize("hourly", [False, True])
    def test_xarray_rejects_a_malformed_distribution(self, adapter: Callable[..., Any], hourly: bool) -> None:
        time = pd.date_range("2000-01-01", periods=36, freq="h" if hourly else "MS")
        data = xr.DataArray(np.zeros(time.size), coords={"time": time}, dims=("time",)).chunk({"time": -1})

        with pytest.raises(InvalidArgumentError, match="distribution"):
            adapter(data, scale=1, distribution="empirical", pool_window_days=1)

    def test_generic_standardized_index_forwards_the_window(self) -> None:
        values = _ramp_rows()

        default = climate_indices.standardized_index(
            values.ravel(),
            1,
            indices.Distribution.empirical,
            **self.NUMPY_KWARGS,
        )
        pooled = climate_indices.standardized_index(
            values.ravel(),
            1,
            indices.Distribution.empirical,
            pool_window_days=POOL_WINDOW_DAYS,
            **self.NUMPY_KWARGS,
        )

        assert not np.allclose(default, pooled, equal_nan=True)
