//! Run Theory: threshold runs and their duration, magnitude, intensity, and peak.
//!
//! Python retains validation, masked-array conversion, and period summaries.
//! This kernel also serves Rust callers with borrowed `(steps, cells)` views.

use ndarray::ArrayView2;

use crate::reduction::pairwise_sum;

/// The strict side of the threshold that defines an active step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Direction {
    /// Values strictly below the threshold.
    Below,
    /// Values strictly above the threshold.
    Above,
}

impl Direction {
    fn active(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::Below => value < threshold,
            Self::Above => value > threshold,
        }
    }

    fn deviation(self, value: f64, threshold: f64) -> f64 {
        match self {
            Self::Below => threshold - value,
            Self::Above => value - threshold,
        }
    }
}

/// Flat run metrics, ordered by cell and then by start, with equally sized fields.
#[derive(Debug, Default)]
pub struct Runs {
    /// Zero-based input column of each run.
    pub cell: Vec<usize>,
    /// Zero-based first step of each run.
    pub start: Vec<usize>,
    /// Zero-based last step of each run, inclusive.
    pub end: Vec<usize>,
    /// Elapsed steps, including pooled gaps.
    pub duration: Vec<usize>,
    /// Sum of threshold deviations over active steps only, in input units.
    pub magnitude: Vec<f64>,
    /// Magnitude divided by elapsed duration, in input units per step.
    pub intensity: Vec<f64>,
    /// Most extreme value; minimum for below, maximum for above.
    pub peak_value: Vec<f64>,
    /// Zero-based step of the first occurrence of the peak.
    pub peak_index: Vec<usize>,
    /// Steps between retained starts in the same cell; NaN for its last run.
    pub interarrival: Vec<f64>,
}

/// Identify threshold runs in a `(steps, cells)` block (Run Theory; Yevjevich, 1967).
///
/// # Numerical contract
///
/// Values and `threshold` share units. Comparisons are strict and NaN always
/// separates runs. Pooling joins runs across at most `max_gap` non-NaN steps;
/// `min_duration` filters the elapsed duration after pooling. Magnitude excludes
/// gaps and uses NumPy's float64 pairwise reduction order. Intensity divides
/// magnitude by elapsed duration. A tied peak keeps its first position.
/// Interarrival uses retained starts within each cell, never across cells.
///
/// # Preconditions and edge cases
///
/// The caller supplies a finite threshold and `min_duration >= 1`. This kernel
/// performs no option validation, calendar conversion, or period grouping.
/// Infinite input values retain their comparisons and may give infinite metrics;
/// finite deviations or sums can also overflow. Empty axes and inputs without
/// retained runs return empty fields. Strided views are supported.
/// Work is linear in the input size; a reusable buffer holds one run's active
/// deviations so its reduction excludes pooled gaps without changing grouping.
/// Registry entry `runs_identify` in `tests/parity_registry.py` compares both
/// directions, with and without pooling, against the Python reference at
/// `rtol = atol = 1e-10`, with matching NaN positions and exact integer fields.
/// See the Measured accuracy section of [`crate`] for the fixed-corpus errors.
///
/// ```
/// use climate_indices::runs::{Direction, identify_runs};
/// use ndarray::{Axis, array};
///
/// let spi = array![0.5, -1.2, -1.5, -0.8, 0.3, -1.1, -1.3];
/// let found = identify_runs(spi.view().insert_axis(Axis(1)), -1.0, Direction::Below, 1, 0);
/// assert_eq!(found.start, [1, 5]);
/// assert_eq!(found.duration, [2, 2]);
/// ```
pub fn identify_runs(
    values: ArrayView2<'_, f64>,
    threshold: f64,
    direction: Direction,
    min_duration: usize,
    max_gap: usize,
) -> Runs {
    let mut result = Runs::default();
    let mut deviations = Vec::new();
    for (cell, series) in values.columns().into_iter().enumerate() {
        let mut step = 0;
        let first_run = result.start.len();
        while step < series.len() {
            if !direction.active(series[step], threshold) {
                step += 1;
                continue;
            }
            let start = step;
            let mut end = step;
            let mut peak = step;
            deviations.clear();
            while step < series.len() {
                let value = series[step];
                if direction.active(value, threshold) {
                    end = step;
                    deviations.push(direction.deviation(value, threshold));
                    if direction.active(value, series[peak]) {
                        peak = step;
                    }
                } else if value.is_nan() || step - end > max_gap {
                    break;
                }
                step += 1;
            }
            let duration = end - start + 1;
            if duration < min_duration {
                continue;
            }
            if result.start.len() > first_run {
                let previous = result.start.len() - 1;
                result.interarrival[previous] = (start - result.start[previous]) as f64;
            }
            let magnitude = pairwise_sum(0..deviations.len(), |index| deviations[index]);
            result.cell.push(cell);
            result.start.push(start);
            result.end.push(end);
            result.duration.push(duration);
            result.magnitude.push(magnitude);
            result.intensity.push(magnitude / duration as f64);
            result.peak_value.push(series[peak]);
            result.peak_index.push(peak);
            result.interarrival.push(f64::NAN);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Axis, Slice, array};

    #[test]
    fn pooled_duration_filters_after_pooling_and_magnitude_excludes_gaps() {
        let values = array![[-2.0], [0.0], [-3.0], [0.0], [0.0], [-4.0]];
        let result = identify_runs(values.view(), -1.0, Direction::Below, 2, 1);
        assert_eq!(result.cell, [0]);
        assert_eq!(result.start, [0]);
        assert_eq!(result.end, [2]);
        assert_eq!(result.duration, [3]);
        assert_eq!(result.magnitude, [3.0]);
        assert_eq!(result.intensity, [1.0]);
        assert_eq!(result.peak_index, [2]);
        assert_eq!(result.peak_value, [-3.0]);
        assert!(result.interarrival[0].is_nan());
    }

    #[test]
    fn nan_separates_runs_and_peak_ties_keep_the_first_step() {
        for (direction, sign) in [(Direction::Below, 1.0), (Direction::Above, -1.0)] {
            let values = array![[-2.0], [-2.0], [0.0], [-3.0], [f64::NAN], [-4.0], [-4.0]] * sign;
            let result = identify_runs(values.view(), -sign, direction, 1, usize::MAX);
            assert_eq!(result.start, [0, 5]);
            assert_eq!(result.end, [3, 6]);
            assert_eq!(result.duration, [4, 2]);
            assert_eq!(result.magnitude, [4.0, 6.0]);
            assert_eq!(result.peak_index, [3, 5]);
            assert_eq!(result.interarrival[0], 5.0);
            assert!(result.interarrival[1].is_nan());
        }
    }

    #[test]
    fn retained_runs_define_interarrival_within_each_cell() {
        let values = array![
            [-2.0, -3.0],
            [-2.0, -3.0],
            [0.0, 0.0],
            [-3.0, 0.0],
            [0.0, -2.0],
            [-4.0, -2.0],
            [-4.0, -2.0]
        ];
        let result = identify_runs(values.view(), -1.0, Direction::Below, 2, 0);
        assert_eq!(result.cell, [0, 0, 1, 1]);
        assert_eq!(result.start, [0, 5, 0, 4]);
        assert_eq!(result.interarrival[0], 5.0);
        assert!(result.interarrival[1].is_nan());
        assert_eq!(result.interarrival[2], 4.0);
        assert!(result.interarrival[3].is_nan());
    }

    #[test]
    fn infinities_and_signed_zero_keep_strict_comparisons() {
        let values = array![[f64::NEG_INFINITY], [0.0], [-2.0], [f64::INFINITY], [-3.0]];
        let result = identify_runs(values.view(), -1.0, Direction::Below, 1, 1);
        assert_eq!(result.duration, [5]);
        assert_eq!(result.magnitude, [f64::INFINITY]);
        assert_eq!(result.peak_index, [0]);
        let zero = array![[-0.0], [0.0]];
        for direction in [Direction::Below, Direction::Above] {
            assert!(
                identify_runs(zero.view(), 0.0, direction, 1, 0)
                    .start
                    .is_empty()
            );
        }
    }

    #[test]
    fn empty_axes_no_exceedance_and_impossible_duration_return_empty_fields() {
        for shape in [(0, 3), (3, 0), (1, 1)] {
            let values = Array2::zeros(shape);
            assert!(
                identify_runs(values.view(), -1.0, Direction::Below, 1, 0)
                    .start
                    .is_empty()
            );
        }
        let values = array![[-2.0]];
        assert!(
            identify_runs(values.view(), -1.0, Direction::Below, 2, 0)
                .start
                .is_empty()
        );
    }

    #[test]
    fn borrowed_negative_stride_views_keep_step_and_cell_order() {
        let values = array![[-2.0, 0.0], [0.0, -3.0], [-4.0, -3.0]];
        let reversed = values.slice_axis(Axis(0), Slice::new(0, None, -1));
        let result = identify_runs(reversed, -1.0, Direction::Below, 1, 0);
        assert_eq!(result.cell, [0, 0, 1]);
        assert_eq!(result.start, [0, 2, 0]);
        assert_eq!(result.end, [0, 2, 1]);
        assert_eq!(result.magnitude, [3.0, 1.0, 4.0]);
    }
}
