//! Flood-family kernels: effective precipitation and EDI.
//!
//! Ports of `climate_indices.flood._pe` and `_edi`, which stay the parity
//! oracles. Python keeps the validation, the all-leap calendar layout, and the
//! Calibration Period resolution; these kernels take the prepared time-first
//! float64 blocks and the calibration rows Python resolved.
//!
//! The operation order is the Python one:
//!
//! * effective precipitation follows `scipy.ndimage.correlate1d`'s general
//!   (non-symmetric) loop, which starts from the newest day's term and then adds
//!   the window from its oldest day;
//! * the calibration sums follow NumPy's axis-0 reduction, which accumulates the
//!   years sequentially when the block has more than one column and pairwise
//!   when it has exactly one ([`crate::reduction::pairwise_sum`]).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::ClimateError;
use crate::reduction::pairwise_sum;

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
    let mut line = Vec::with_capacity(days);
    for cell in 0..cells {
        line.clear();
        line.extend(precipitation.column(cell).iter().copied());
        let mut output = result.column_mut(cell);
        for (window, value) in line
            .windows(duration)
            .zip(output.iter_mut().skip(duration - 1))
        {
            let (window_older, window_newest) = window.split_at(duration - 1);
            *value = window_older
                .iter()
                .zip(older)
                .fold(window_newest[0] * newest, |sum, (&rain, &weight)| {
                    sum + rain * weight
                });
        }
    }
    Ok(result)
}

/// The per-column mean and population variance of a calibration sample.
#[allow(dead_code)]
struct Climatology {
    mean: Array1<f64>,
    variance: Array1<f64>,
}

/// NumPy's `values.sum(axis=0)` of one column, in the order NumPy reduces it.
#[allow(dead_code)]
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
#[allow(dead_code)]
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
#[allow(dead_code)]
fn standardize(values: ArrayView2<'_, f64>, climatology: &Climatology) -> Array2<f64> {
    let mut result = Array2::from_elem(values.raw_dim(), f64::NAN);
    for (column, (mut output, input)) in result
        .columns_mut()
        .into_iter()
        .zip(values.columns())
        .enumerate()
    {
        let mean = climatology.mean[column];
        let variance = climatology.variance[column];
        let rounding = 8.0 * f64::EPSILON * mean.abs();
        if variance > rounding * rounding {
            let deviation = variance.sqrt();
            output.zip_mut_with(&input, |standardized, &value| {
                *standardized = (value - mean) / deviation;
            });
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
