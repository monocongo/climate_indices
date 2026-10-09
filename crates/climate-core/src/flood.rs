//! Flood-family kernels: effective precipitation, EDI, the Flood Index, and the
//! Antecedent Precipitation Index recurrence.
//!
//! Ports of `climate_indices.flood._pe`, `_edi`, `_if`, and `_antecedent`, which
//! stay the parity oracles. Python keeps the validation, the all-leap calendar
//! layout, and the Calibration Period resolution; these kernels take the
//! prepared time-first float64 blocks and the calibration rows Python resolved.
//!
//! The operation order is the Python one, so the recorded paths in
//! `tests/test_native_parity_flood.py` agree at `rtol = atol = 1e-10` with
//! identical NaN positions:
//!
//! * effective precipitation follows `scipy.ndimage.correlate1d`'s general
//!   (non-symmetric) loop, which starts from the newest day's term and then adds
//!   the window from its oldest day;
//! * the calibration sums follow NumPy's axis-0 reduction, which accumulates the
//!   years sequentially when the block has more than one column and pairwise
//!   when it has exactly one ([`crate::reduction::pairwise_sum`]).

use std::ops::Range;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::ClimateError;
use crate::recurrence::{RecurrenceInputs, RecurrenceRun, run};
use crate::reduction::pairwise_sum;

/// Positional days per year in the all-leap layout EDI and the Flood Index read
/// (`climate_indices.flood._common._DAYS_PER_YEAR`).
pub const DAYS_PER_YEAR: usize = 366;

/// The correlation filter of Byun and Wilhite (1999), Eq. 2, oldest day first.
///
/// `w_m = sum(n=m..D, 1/n)` weights the day `m - 1` days before the window's
/// last one, so the oldest day carries `1 / D` and the newest `H_D`. Built
/// exactly as `np.cumsum((1.0 / np.arange(1, D + 1))[::-1])` (the Python weights
/// reversed for `correlate1d`): a running sum from `1 / D` up to `1`.
fn harmonic_filter(duration: usize) -> Vec<f64> {
    let mut running = 0.0;
    (1..=duration)
        .rev()
        .map(|n| {
            running += 1.0 / n as f64;
            running
        })
        .collect()
}

/// Daily effective precipitation over a fixed window (Byun and Wilhite, 1999, Eq. 2).
///
/// `precipitation` is `(days, cells)`. The first `duration - 1` days, and any
/// day whose window holds a NaN, are NaN; a series shorter than the window is
/// all NaN.
pub fn effective_precipitation(
    precipitation: ArrayView2<'_, f64>,
    duration: usize,
) -> Result<Array2<f64>, ClimateError> {
    if duration == 0 {
        return Err(ClimateError::EmptyPeriod {
            argument: "duration",
        });
    }
    let (days, cells) = precipitation.dim();
    let mut result = Array2::from_elem((days, cells), f64::NAN);
    if days < duration {
        return Ok(result);
    }
    let filter = harmonic_filter(duration);
    let (older, newest) = filter.split_at(duration - 1);
    let newest = newest[0];
    let windows = days - duration + 1;
    let mut line = Vec::with_capacity(days);
    let mut sums = vec![0.0; windows];
    for cell in 0..cells {
        line.clear();
        line.extend(precipitation.column(cell).iter().copied());
        // every window adds its terms in correlate1d's order (the newest day,
        // then the oldest onward); taking one weight across all windows at a
        // time keeps that order per window while vectorizing over the days
        for (sum, &rain) in sums.iter_mut().zip(&line[duration - 1..]) {
            *sum = rain * newest;
        }
        for (offset, &weight) in older.iter().enumerate() {
            for (sum, &rain) in sums.iter_mut().zip(&line[offset..offset + windows]) {
                *sum += rain * weight;
            }
        }
        for (value, &sum) in result
            .column_mut(cell)
            .iter_mut()
            .skip(duration - 1)
            .zip(&sums)
        {
            *value = sum;
        }
    }
    Ok(result)
}

/// The per-column mean and population variance of a calibration sample.
struct Climatology {
    mean: Array1<f64>,
    variance: Array1<f64>,
}

/// NumPy's `values.sum(axis=0)` of one column, in the order NumPy reduces it.
fn column_sum(column: ArrayView1<'_, f64>, columns: usize) -> f64 {
    if columns == 1 {
        pairwise_sum(0..column.len(), |row| column[row])
    } else {
        column
            .iter()
            .copied()
            .reduce(|sum, value| sum + value)
            .unwrap_or(0.0)
    }
}

/// The mean and population (`ddof=0`) variance of each column's finite values.
///
/// A column without finite values has a NaN mean, and one with fewer than two a
/// NaN variance. Non-finite entries contribute zero, as in the Python
/// `np.where(valid, ..., 0)` sums.
fn climatology(sample: ArrayView2<'_, f64>) -> Climatology {
    let columns = sample.ncols();
    let mut mean = Array1::from_elem(columns, f64::NAN);
    let mut variance = Array1::from_elem(columns, f64::NAN);
    let mut terms = Array1::<f64>::zeros(sample.nrows());
    for (column, values) in sample.columns().into_iter().enumerate() {
        let count = values.iter().filter(|value| value.is_finite()).count();
        terms.zip_mut_with(&values, |term, &value| {
            *term = if value.is_finite() { value } else { 0.0 };
        });
        let column_mean = if count > 0 {
            column_sum(terms.view(), columns) / count as f64
        } else {
            f64::NAN
        };
        mean[column] = column_mean;
        terms.zip_mut_with(&values, |term, &value| {
            let deviation = if value.is_finite() {
                value - column_mean
            } else {
                0.0
            };
            *term = deviation * deviation;
        });
        if count > 1 {
            variance[column] = column_sum(terms.view(), columns) / count as f64;
        }
    }
    Climatology { mean, variance }
}

/// Standardize every row of `values` against its column's climatology.
///
/// A column whose variance does not exceed the squared rounding guard
/// `8 * eps * |mean|` (including a NaN mean or variance) is NaN throughout.
fn standardize(values: ArrayView2<'_, f64>, climatology: &Climatology) -> Array2<f64> {
    // a rejected column divides by NaN, which keeps it NaN without a branch
    let deviation = Array1::from_iter(climatology.mean.iter().zip(&climatology.variance).map(
        |(&mean, &variance)| {
            let rounding = 8.0 * f64::EPSILON * mean.abs();
            if variance > rounding * rounding {
                variance.sqrt()
            } else {
                f64::NAN
            }
        },
    ));
    let mut result = Array2::from_elem(values.raw_dim(), f64::NAN);
    // walk the time-first block a row at a time, in memory order
    for (mut output, input) in result.rows_mut().into_iter().zip(values.rows()) {
        ndarray::Zip::from(&mut output)
            .and(&input)
            .and(&climatology.mean)
            .and(&deviation)
            .for_each(|standardized, &value, &mean, &deviation| {
                *standardized = (value - mean) / deviation;
            });
    }
    result
}

/// Check that a calibration window lies inside an axis of `length` rows.
fn check_rows(
    argument: &'static str,
    rows: &Range<usize>,
    length: usize,
) -> Result<(), ClimateError> {
    if rows.start <= rows.end && rows.end <= length {
        Ok(())
    } else {
        Err(ClimateError::RowsOutOfRange {
            argument,
            start: rows.start,
            end: rows.end,
            length,
        })
    }
}

/// The fixed-window Effective Drought Index of an all-leap `(years, columns)` block.
///
/// Each column is one calendar day of one cell; `calibration` selects the years
/// whose finite values fit that column's climatology. The result is
/// `(PE - mean) / SD` with the population SD, in the input's layout.
pub fn edi(
    years: ArrayView2<'_, f64>,
    calibration: Range<usize>,
) -> Result<Array2<f64>, ClimateError> {
    check_rows("calibration", &calibration, years.nrows())?;
    let climatology = climatology(years.slice(ndarray::s![calibration, ..]));
    Ok(standardize(years, &climatology))
}

/// The Flood Index of a `(days, cells)` effective-precipitation block.
///
/// `first_start` is the first day of the first calibration year's annual period
/// and `calibration_years` the number of consecutive [`DAYS_PER_YEAR`]-day
/// periods from there. Each period's maximum over its finite days (minus
/// infinity when it has none, which the climatology then excludes) forms the
/// calibration sample, and every day is standardized against it.
pub fn flood_index(
    pe: ArrayView2<'_, f64>,
    first_start: usize,
    calibration_years: usize,
) -> Result<Array2<f64>, ClimateError> {
    let (days, cells) = pe.dim();
    let end = calibration_years
        .checked_mul(DAYS_PER_YEAR)
        .and_then(|span| span.checked_add(first_start))
        .unwrap_or(usize::MAX);
    check_rows("calibration", &(first_start..end), days)?;
    let mut maxima = Array2::from_elem((calibration_years, cells), f64::NEG_INFINITY);
    for (year, mut row) in maxima.rows_mut().into_iter().enumerate() {
        let start = first_start + year * DAYS_PER_YEAR;
        // a running maximum per cell, day by day in memory order; the maximum
        // of the finite days does not depend on the order they are visited
        for day in pe
            .slice(ndarray::s![start..start + DAYS_PER_YEAR, ..])
            .rows()
        {
            row.zip_mut_with(&day, |largest, &value| {
                if value.is_finite() && value > *largest {
                    *largest = value;
                }
            });
        }
    }
    let climatology = climatology(maxima.view());
    Ok(standardize(pe, &climatology))
}

/// Run the Antecedent Precipitation Index over the whole time axis.
///
/// `API_t = k * API_(t-1) + P_t` (Kohler and Linsley, 1951), advanced one cell
/// at a time by [`crate::recurrence::run`], so the ADR-0007 missing-day policy,
/// the spin-up, and the recorded-history NaNs are the Python driver's.
pub fn antecedent_precipitation_index(
    precipitation_mm: ArrayView2<'_, f64>,
    k: f64,
    initial_api: ArrayView1<'_, f64>,
    inputs: &RecurrenceInputs<'_>,
    record: bool,
) -> Result<RecurrenceRun<f64>, ClimateError> {
    if precipitation_mm.dim() != inputs.weather_valid.dim() {
        return Err(ClimateError::ShapeMismatch {
            argument: "precipitation_mm",
            expected: inputs.weather_valid.len(),
            actual: precipitation_mm.len(),
        });
    }
    let initial = initial_api.to_vec();
    run(
        "antecedent_precipitation_index",
        &initial,
        inputs,
        record,
        |api| *api,
        |api| *api = f64::NAN,
        |previous, day, cell| k * previous + precipitation_mm[[day, cell]],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::MissingDayPolicy;
    use ndarray::{Array, array};

    /// Relative closeness for closed-form references, scaled by `tol`.
    fn close(actual: f64, expected: f64, tol: f64) -> bool {
        (actual - expected).abs() <= tol * expected.abs().max(1.0)
    }

    /// Eq. 2 as written: `PE_t = sum(n=1..D, sum(m=1..n, P[t-m+1]) / n)`.
    fn double_sum(series: &[f64], day: usize, duration: usize) -> f64 {
        (1..=duration)
            .map(|n| (1..=n).map(|m| series[day + 1 - m]).sum::<f64>() / n as f64)
            .sum()
    }

    fn rainfall(days: usize) -> Vec<f64> {
        // a deterministic mix of dry days and showers of different sizes
        (0..days)
            .map(|day| match day % 7 {
                0 | 3 => 0.0,
                1 => 12.5,
                2 => 0.25,
                4 => 3.0,
                5 => 40.0,
                _ => 1.0,
            })
            .collect()
    }

    #[test]
    fn the_filter_has_the_harmonic_endpoint_weights() {
        let filter = harmonic_filter(4);
        // oldest day 1/D, newest day H_D
        assert_eq!(filter[0], 0.25);
        assert!(close(filter[3], 1.0 + 0.5 + 1.0 / 3.0 + 0.25, 1e-15));
        assert!(filter.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn effective_precipitation_matches_the_equation_2_double_sum() {
        let series = rainfall(60);
        let column = Array::from_shape_vec((60, 1), series.clone()).unwrap();
        for duration in [1, 2, 3, 15] {
            let result = effective_precipitation(column.view(), duration).unwrap();
            for day in 0..duration - 1 {
                assert!(result[[day, 0]].is_nan());
            }
            for day in duration - 1..60 {
                // the double sum reassociates the same terms: a few ulps of a
                // value built from at most D * D additions
                let expected = double_sum(&series, day, duration);
                assert!(
                    close(result[[day, 0]], expected, 1e-13),
                    "D={duration} day={day}"
                );
            }
        }
    }

    #[test]
    fn the_two_day_window_is_the_selected_identity() {
        // EP_2 = P_1 + (P_1 + P_2) / 2 (ADR-0014)
        let result = effective_precipitation(array![[4.0], [10.0]].view(), 2).unwrap();
        assert!(result[[0, 0]].is_nan());
        assert_eq!(result[[1, 0]], 10.0 + (10.0 + 4.0) / 2.0);
    }

    #[test]
    fn a_missing_day_blanks_only_the_windows_that_hold_it() {
        let mut series = rainfall(30);
        series[10] = f64::NAN;
        let column = Array::from_shape_vec((30, 1), series).unwrap();
        let result = effective_precipitation(column.view(), 5).unwrap();
        assert!(result[[9, 0]].is_finite());
        assert!((10..15).all(|day| result[[day, 0]].is_nan()));
        assert!(result[[15, 0]].is_finite());
    }

    #[test]
    fn a_series_shorter_than_the_window_is_all_nan_and_cells_are_independent() {
        let short = effective_precipitation(Array2::zeros((3, 2)).view(), 4).unwrap();
        assert!(short.iter().all(|value| value.is_nan()));
        let block = array![[1.0, 0.0], [2.0, f64::NAN], [3.0, 5.0]];
        let result = effective_precipitation(block.view(), 1).unwrap();
        // a one-day window is the day's own rain
        assert_eq!(result.column(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert!(result[[1, 1]].is_nan());
        assert_eq!(result[[2, 1]], 5.0);
        assert_eq!(
            effective_precipitation(block.view(), 0).unwrap_err(),
            ClimateError::EmptyPeriod {
                argument: "duration"
            }
        );
    }

    #[test]
    fn edi_is_the_population_z_score_of_each_column() {
        // two columns, three calibration years; the fourth year is standardized only
        let years = array![[1.0, 4.0], [2.0, 4.0], [3.0, 4.0], [5.0, 8.0]];
        let result = edi(years.view(), 0..3).unwrap();
        // column 0: mean 2, population SD sqrt(2/3)
        let sd = (2.0_f64 / 3.0).sqrt();
        assert!(close(result[[0, 0]], -1.0 / sd, 1e-15));
        assert_eq!(result[[1, 0]], 0.0);
        assert!(close(result[[3, 0]], 3.0 / sd, 1e-15));
        // column 1 has zero calibration variance, so it is NaN in every year
        assert!(result.column(1).iter().all(|value| value.is_nan()));
    }

    #[test]
    fn edi_needs_two_finite_calibration_values_per_column() {
        let years = array![
            [1.0, f64::NAN, f64::NAN],
            [f64::NAN, 2.0, f64::NAN],
            [4.0, 6.0, f64::NAN]
        ];
        let result = edi(years.view(), 0..2).unwrap();
        // one finite value per column inside the calibration rows is not enough
        assert!(result.iter().all(|value| value.is_nan()));
        let result = edi(years.view(), 0..3).unwrap();
        assert!(close(result[[2, 0]], 1.0, 1e-15));
        assert!(result[[1, 0]].is_nan());
        assert!(close(result[[1, 1]], -1.0, 1e-15));
        assert!(result.column(2).iter().all(|value| value.is_nan()));
    }

    #[test]
    fn the_rounding_guard_rejects_a_variance_of_rounding_error_alone() {
        // values that differ only in their last bit around a large mean
        let base = 1.0e6;
        let years = array![[base], [base + base * f64::EPSILON], [base]];
        let result = edi(years.view(), 0..3).unwrap();
        assert!(result.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn a_calibration_window_outside_the_rows_is_rejected() {
        let years = Array2::<f64>::zeros((3, 2));
        assert_eq!(
            edi(years.view(), 1..4).unwrap_err(),
            ClimateError::RowsOutOfRange {
                argument: "calibration",
                start: 1,
                end: 4,
                length: 3
            }
        );
        assert!(flood_index(years.view(), 0, 1).is_err());
    }

    #[test]
    fn a_single_column_sums_pairwise_and_several_sequentially() {
        // 1e16 swamps every one a sequential sum adds to it, but NumPy's
        // eight-lane pairwise grouping of a single column adds the ones together
        let mut values = vec![1.0e16];
        values.extend([1.0; 8]);
        values.push(-1.0e16);
        let column = Array::from(values.clone());
        let pairwise = column_sum(column.view(), 1);
        assert_eq!(pairwise, pairwise_sum(0..values.len(), |row| values[row]));
        assert!(pairwise > 0.0);
        assert_eq!(column_sum(column.view(), 2), 0.0);
    }

    fn annual_periods(maxima: &[f64], offset: usize) -> Array2<f64> {
        // a flat 1 mm/day record with one peak per period and a leading offset
        let mut pe = Array2::from_elem((offset + maxima.len() * DAYS_PER_YEAR + 10, 1), 1.0);
        for (year, &maximum) in maxima.iter().enumerate() {
            pe[[offset + year * DAYS_PER_YEAR + 100, 0]] = maximum;
        }
        pe
    }

    #[test]
    fn flood_index_standardizes_against_the_annual_maxima() {
        let pe = annual_periods(&[10.0, 20.0, 30.0], 31);
        let result = flood_index(pe.view(), 31, 3).unwrap();
        // maxima 10, 20, 30: mean 20, population SD sqrt(200 / 3)
        let sd = (200.0_f64 / 3.0).sqrt();
        assert!(close(result[[0, 0]], (1.0 - 20.0) / sd, 1e-15));
        assert!(close(result[[31 + 100, 0]], -10.0 / sd, 1e-15));
        assert!(close(
            result[[31 + 2 * DAYS_PER_YEAR + 100, 0]],
            10.0 / sd,
            1e-15
        ));
    }

    #[test]
    fn a_period_without_finite_days_drops_out_of_the_flood_index_sample() {
        let mut pe = annual_periods(&[10.0, 20.0, 30.0], 0);
        pe.slice_mut(ndarray::s![0..DAYS_PER_YEAR, ..])
            .fill(f64::NAN);
        let result = flood_index(pe.view(), 0, 3).unwrap();
        // the sample is {20, 30}: mean 25, SD 5
        assert!(close(result[[DAYS_PER_YEAR + 100, 0]], -1.0, 1e-15));
        assert!(result[[0, 0]].is_nan());
        // one remaining finite maximum is not enough
        pe.slice_mut(ndarray::s![DAYS_PER_YEAR..2 * DAYS_PER_YEAR, ..])
            .fill(f64::NAN);
        let result = flood_index(pe.view(), 0, 3).unwrap();
        assert!(result.iter().all(|value| value.is_nan()));
    }

    fn api_inputs<'a>(
        weather: &'a Array2<bool>,
        static_valid: &'a Array1<bool>,
        gaps: &'a Array1<i64>,
        spin_up: usize,
        policy: MissingDayPolicy,
    ) -> RecurrenceInputs<'a> {
        RecurrenceInputs {
            weather_valid: weather.view(),
            static_valid: static_valid.view(),
            in_season: None,
            trailing_gap_days: gaps.view(),
            spin_up,
            policy,
        }
    }

    #[test]
    fn constant_rain_follows_the_geometric_closed_form() {
        let (days, k, rain) = (200, 0.9, 5.0);
        let precipitation = Array2::from_elem((days, 1), rain);
        let weather = Array2::from_elem((days, 1), true);
        let (static_valid, gaps) = (array![true], array![-1]);
        let run = antecedent_precipitation_index(
            precipitation.view(),
            k,
            array![0.0].view(),
            &api_inputs(
                &weather,
                &static_valid,
                &gaps,
                0,
                MissingDayPolicy::Propagate,
            ),
            true,
        )
        .unwrap();
        let values = run.values.unwrap();
        for day in 0..days {
            // API_t = P (1 - k^(t+1)) / (1 - k); about one ulp per day of recurrence
            let expected = rain * (1.0 - k.powi(day as i32 + 1)) / (1.0 - k);
            assert!(close(values[[day, 0]], expected, 1e-13), "day {day}");
        }
        // the state is the last recorded day, within k^days of the P / (1 - k) limit
        assert_eq!(run.state[0], values[[days - 1, 0]]);
        let limit = rain / (1.0 - k);
        assert!((limit - run.state[0]) / limit <= 2.0 * k.powi(days as i32));
        assert_eq!(run.trailing_gap_days.unwrap(), array![0]);
    }

    #[test]
    fn a_bridged_gap_decays_nothing_and_a_long_one_poisons_the_cell() {
        let precipitation = array![[10.0, 10.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 2.0]];
        let mut weather = Array2::from_elem((5, 2), true);
        weather[[1, 0]] = false;
        for day in 1..4 {
            weather[[day, 1]] = false;
        }
        let (static_valid, gaps) = (array![true, true], array![-1, -1]);
        let run = antecedent_precipitation_index(
            precipitation.view(),
            0.5,
            array![0.0, 0.0].view(),
            &api_inputs(
                &weather,
                &static_valid,
                &gaps,
                1,
                MissingDayPolicy::Bridge { max_gap_days: 2 },
            ),
            true,
        )
        .unwrap();
        let values = run.values.unwrap();
        // spin-up omits day 0; cell 0's missing day holds the API (NaN on the
        // day), then it decays from 10 on the next valid day
        assert!(values[[0, 0]].is_nan());
        assert_eq!(values[[1, 0]], 5.0);
        assert_eq!(values[[3, 0]], 0.5 * 2.5 + 2.0);
        // cell 1's three missing days exceed the allowance
        assert!(values.column(1).iter().all(|value| value.is_nan()));
        assert!(run.state[1].is_nan());
    }

    #[test]
    fn the_precipitation_block_must_match_the_validity_mask() {
        let weather = Array2::from_elem((3, 1), true);
        let (static_valid, gaps) = (array![true], array![-1]);
        let error = antecedent_precipitation_index(
            Array2::zeros((2, 1)).view(),
            0.9,
            array![0.0].view(),
            &api_inputs(
                &weather,
                &static_valid,
                &gaps,
                0,
                MissingDayPolicy::Propagate,
            ),
            true,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "precipitation_mm",
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn an_overflowing_step_is_a_non_finite_error() {
        let precipitation = Array2::from_elem((2, 1), f64::MAX);
        let weather = Array2::from_elem((2, 1), true);
        let (static_valid, gaps) = (array![true], array![-1]);
        let error = antecedent_precipitation_index(
            precipitation.view(),
            0.9,
            array![0.0].view(),
            &api_inputs(
                &weather,
                &static_valid,
                &gaps,
                0,
                MissingDayPolicy::Propagate,
            ),
            true,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::NonFinite {
                index_type: "antecedent_precipitation_index"
            }
        );
    }
}
