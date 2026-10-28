//! Empirical distribution probabilities for signed climate and energy series.
//!
//! This ports the rank count and SEI plotting position in Python's
//! `compute._empirical_fit`. Calendar preparation, Calibration Period selection,
//! output scaling, validation, and logging remain the caller's responsibility.

use ndarray::{Array2, ArrayView2, Zip};

use crate::ClimateError;

/// Rank each value against its column's Calibration Period sample.
///
/// # Numerical contract
///
/// `calibration` has shape `(calibration years, columns)` and `values` has shape
/// `(years, columns)`, with `column = step * cells + cell`. Both use the same
/// input units. Monthly, daily, and hourly calendars are prepared by the caller.
/// Strided and empty views are accepted. Outputs are dimensionless probabilities
/// with the shape of `values`.
///
/// The SEI plotting position is `(1 + k) / (n + 2)`, where `k` counts calibration
/// values at or below the value and `n` counts non-NaN calibration values.
/// Ties take the maximum rank. Signed zeros compare equal; infinities rank as
/// ordinary values. NaN calibration values never count. A NaN value, or a column
/// with fewer than ten valid calibration values, produces NaN. Constant columns
/// receive `(n + 1) / (n + 2)` for their constant value. Probabilities remain
/// strictly between zero and one; no extrapolation or normal transform occurs.
/// Integer counts convert to f64 before the same arithmetic as Python.
///
/// Each column sorts once, then uses binary searches. Scratch space holds at most
/// one calibration column in addition to the output array.
/// Returns [`ClimateError::ShapeMismatch`] if column counts differ.
/// See the measured workflow rows `empirical` and `empirical_spatial_block` in
/// the crate's `Measured accuracy` section.
pub fn plotting_positions(
    calibration: ArrayView2<'_, f64>,
    values: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, ClimateError> {
    if calibration.ncols() != values.ncols() {
        return Err(ClimateError::ShapeMismatch {
            argument: "calibration",
            expected: values.ncols(),
            actual: calibration.ncols(),
        });
    }
    let mut probabilities = Array2::from_elem(values.raw_dim(), f64::NAN);
    let mut sorted = Vec::with_capacity(calibration.nrows());
    Zip::from(probabilities.columns_mut())
        .and(calibration.columns())
        .and(values.columns())
        .for_each(|mut out, calibration, values| {
            sorted.clear();
            sorted.extend(calibration.iter().copied().filter(|value| !value.is_nan()));
            if sorted.len() < 10 {
                return;
            }
            sorted.sort_unstable_by(f64::total_cmp);
            let denominator = sorted.len() as f64 + 2.0;
            for (probability, &value) in out.iter_mut().zip(values.iter()) {
                if !value.is_nan() {
                    let count = sorted.partition_point(|&sample| sample <= value);
                    *probability = (count as f64 + 1.0) / denominator;
                }
            }
        });
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Axis, Slice, array};

    #[test]
    fn ranks_ties_signed_zeros_and_infinities_exactly() {
        let calibration = array![
            [f64::NEG_INFINITY],
            [-2.0],
            [-2.0],
            [-0.0],
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [f64::INFINITY],
            [f64::INFINITY],
            [f64::NAN]
        ];
        let values = array![
            [f64::NEG_INFINITY],
            [-2.0],
            [-0.0],
            [0.0],
            [f64::INFINITY],
            [f64::NAN]
        ];
        let result = plotting_positions(calibration.view(), values.view()).unwrap();
        for (row, count) in [1.0, 3.0, 5.0, 5.0, 10.0].iter().enumerate() {
            assert_eq!(result[[row, 0]], (count + 1.0) / 12.0);
        }
        assert!(result[[5, 0]].is_nan());
    }

    #[test]
    fn minimum_sample_and_outside_range() {
        let mut calibration = Array2::from_shape_fn((10, 2), |(row, _)| row as f64);
        calibration[[0, 0]] = f64::NAN;
        let result = plotting_positions(
            calibration.view(),
            array![[-1.0, -1.0], [10.0, 10.0]].view(),
        )
        .unwrap();
        assert!(result.column(0).iter().all(|value| value.is_nan()));
        assert_eq!(result[[0, 1]], 1.0 / 12.0);
        assert_eq!(result[[1, 1]], 11.0 / 12.0);
    }

    #[test]
    fn strided_and_degenerate_views() {
        let calibration = Array2::from_elem((10, 4), 0.0);
        let values = Array2::from_elem((2, 4), -0.0);
        let result = plotting_positions(
            calibration
                .slice_axis(Axis(0), Slice::new(0, None, -1))
                .slice_axis_move(Axis(1), Slice::new(0, None, 2)),
            values.slice_axis(Axis(1), Slice::new(0, None, 2)),
        )
        .unwrap();
        assert_eq!(result, Array2::from_elem((2, 2), 11.0 / 12.0));
        let empty = Array2::zeros((0, 2));
        assert!(
            plotting_positions(
                empty.view(),
                values.slice_axis(Axis(1), Slice::new(0, None, 2))
            )
            .unwrap()
            .iter()
            .all(|value| value.is_nan())
        );
        assert_eq!(
            plotting_positions(calibration.view(), Array2::zeros((0, 4)).view())
                .unwrap()
                .dim(),
            (0, 4)
        );
        let missing = Array2::from_elem((10, 4), f64::NAN);
        assert!(
            plotting_positions(missing.view(), values.view())
                .unwrap()
                .iter()
                .all(|value| value.is_nan())
        );
    }

    #[test]
    fn mismatched_columns_are_an_error() {
        assert_eq!(
            plotting_positions(Array2::zeros((10, 2)).view(), Array2::zeros((1, 3)).view())
                .unwrap_err(),
            ClimateError::ShapeMismatch {
                argument: "calibration",
                expected: 3,
                actual: 2
            }
        );
    }
}
