//! Percentage-of-normal precipitation (PNP): the calibration normal of each
//! calendar step, and the ratio of every scaled total to its step's normal.
//!
//! Reference: `docs/algorithms.md` § Percentage of Normal Precipitation. The
//! Python source is the `valid_counts`/`averages` block and the `np.divide`
//! calls of `indices.percentage_of_normal`, which stays the reference oracle.
//! Resolving and rejecting the calibration period, padding a trailing partial
//! period with NaN, the deprecated calibration keywords, logging, and warnings
//! all stay in Python; these kernels take the prepared scale sums and return
//! the percentages.

use ndarray::{Array1, Array2, ArrayView2};

use crate::ClimateError;
use crate::reduction::pairwise_sum;

/// The calibration normal of each column of a (calibration years, columns) block.
///
/// - Python source: the `valid_counts`, `averages`, and `averages > 0.0` lines
///   of `indices.percentage_of_normal`.
/// - Inputs: `calibration`, shaped (calibration years, columns), each column one
///   calendar step of one cell, already sliced to the calibration period and
///   padded with NaN to whole periods -- the caller reshapes the block the way
///   Python's `reshape(-1, period_length, *cells)` does before reducing axis 0.
/// - Outputs: one normal per column, in the same order.
/// - NaN semantics: NaN is missing and excluded from both the count and the sum.
///   A column with no value has the NaN normal that Python's
///   `np.maximum(count, 1)` and `valid_counts > 0` produce.
/// - Zero and negative semantics: a normal that is not a positive value is NaN,
///   so its calendar step carries no percentage; the Python `averages > 0.0`
///   filter does the same.
/// - Numerics: NumPy reduces unit-stride year columns with pairwise summation;
///   other columns are accumulated sequentially. Missing values keep their
///   zero-valued positions in the pairwise grouping, as `np.nansum` does.
pub fn pnp_normals(calibration: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut normals = Array1::<f64>::zeros(calibration.ncols());
    for (normal, column) in normals.iter_mut().zip(calibration.columns()) {
        let pairwise = column.strides()[0].unsigned_abs() == 1;
        let (mut sum, mut count) = (0.0, 0.0_f64);
        for &value in column {
            if value.is_nan() {
                continue;
            }
            if !pairwise {
                sum += value;
            }
            count += 1.0;
        }
        if pairwise {
            sum = pairwise_sum(0..column.len(), |index| {
                let value = column[index];
                if value.is_nan() { 0.0 } else { value }
            });
        }
        let mean = sum / count.max(1.0);
        *normal = if count > 0.0 && mean > 0.0 {
            mean
        } else {
            f64::NAN
        };
    }
    normals
}

/// Each value's percentage of its calendar step's normal.
///
/// - Python source: the `whole_periods` and remainder `np.divide` calls of
///   `indices.percentage_of_normal`, which leave a value NaN wherever its
///   normal is NaN.
/// - Inputs: `scale_sums`, shaped (time, columns), the sliding scale sums in
///   time order; `normals`, shaped (period length, columns), the calendar-step
///   normals of `pnp_normals`, with at least one row.
/// - Outputs: percentages shaped like `scale_sums`. A row past a whole number of
///   periods uses the normals of the calendar steps it covers, as Python's
///   remainder branch does.
/// - NaN semantics: a NaN value or a NaN normal gives a NaN percentage, and a
///   NaN normal is never zero, so no division by zero is reachable here.
///
/// Returns [`ClimateError::ShapeMismatch`] when `normals` does not have one
/// value per column of `scale_sums`, or [`ClimateError::EmptyPeriod`] when it
/// contains no calendar steps.
pub fn pnp_percentages(
    scale_sums: ArrayView2<'_, f64>,
    normals: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, ClimateError> {
    if normals.ncols() != scale_sums.ncols() {
        return Err(ClimateError::ShapeMismatch {
            argument: "normals",
            expected: scale_sums.ncols(),
            actual: normals.ncols(),
        });
    }
    let period_length = normals.nrows();
    if period_length == 0 {
        return Err(ClimateError::EmptyPeriod {
            argument: "normals",
        });
    }

    let mut percentages = Array2::<f64>::zeros(scale_sums.raw_dim());
    for (step, (mut out, values)) in percentages
        .rows_mut()
        .into_iter()
        .zip(scale_sums.rows())
        .enumerate()
    {
        let calendar_step = normals.row(step % period_length);
        for ((percentage, &value), &normal) in out.iter_mut().zip(&values).zip(&calendar_step) {
            *percentage = value / normal;
        }
    }
    Ok(percentages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn normals_average_each_column_and_exclude_missing_values() {
        let block = array![[2.0, 1.0], [4.0, f64::NAN], [f64::NAN, 3.0]];
        let normals = pnp_normals(block.view());
        assert_eq!(normals[0], 3.0);
        assert_eq!(normals[1], 2.0);
    }

    #[test]
    fn a_normal_that_is_not_positive_leaves_its_step_without_a_percentage() {
        let block = array![[0.0, -1.0, f64::NAN], [0.0, -3.0, f64::NAN]];
        let normals = pnp_normals(block.view());
        assert!(normals.iter().all(|normal| normal.is_nan()));
    }

    #[test]
    fn percentages_use_the_calendar_step_of_each_row() {
        let scale_sums = array![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]];
        let normals = array![[10.0, 20.0], [30.0, 40.0]];
        let percentages = pnp_percentages(scale_sums.view(), normals.view()).unwrap();
        assert_eq!(percentages, array![[1.0, 1.0], [1.0, 1.0], [5.0, 3.0]]);
    }

    #[test]
    fn percentages_are_missing_where_the_value_or_its_normal_is() {
        let scale_sums = array![[f64::NAN, 1.0], [1.0, 1.0]];
        let normals = array![[f64::NAN, f64::NAN], [2.0, f64::NAN]];
        let percentages = pnp_percentages(scale_sums.view(), normals.view()).unwrap();
        assert!(percentages[[0, 0]].is_nan() && percentages[[0, 1]].is_nan());
        assert_eq!(percentages[[1, 0]], 0.5);
        assert!(percentages[[1, 1]].is_nan());
    }

    #[test]
    fn percentages_reject_a_period_with_no_calendar_steps() {
        let error = pnp_percentages(
            Array2::<f64>::ones((1, 2)).view(),
            Array2::<f64>::zeros((0, 2)).view(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::EmptyPeriod {
                argument: "normals"
            }
        );
    }

    #[test]
    fn percentages_reject_a_normal_of_the_wrong_width() {
        let error = pnp_percentages(array![[1.0, 2.0]].view(), array![[1.0]].view()).unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "normals",
                expected: 2,
                actual: 1,
            }
        );
    }
}
