//! Self-calibrating PDSI duration-factor fit.
//!
//! Reference: Wells, Goddard, and Hayes (2004), "A Self-Calibrating Palmer
//! Drought Severity Index", J. Climate 17(12), 2335-2351, and the reference
//! implementation's `get_Z_sum()`, `LeastSquares()`, `CalcDurFact()`, and
//! `safe_percentile()`.
//!
//! The Python implementation in `climate_indices.self_calibration` stays the
//! reference oracle; `tests/test_native_parity_palmer.py` checks this kernel
//! against it at `rtol = atol = 1e-10`. Sign and window validation, the
//! calibration-period slice, the short-record `InsufficientDataError`, and the
//! duration-factor validation stay in Python.

use std::collections::VecDeque;

use crate::ClimateError;

/// The spell side a duration factor is fitted for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Spell {
    Wet,
    Dry,
}

impl Spell {
    /// Python's `WET_SIGN`/`DRY_SIGN`, as the float it multiplies into.
    fn sign(self) -> f64 {
        match self {
            Self::Wet => 1.0,
            Self::Dry => -1.0,
        }
    }
}

/// Spell durations, in months, the duration-factor regression samples.
pub const DURATION_FACTOR_WINDOW_LENGTHS: [usize; 10] = [3, 6, 9, 12, 18, 24, 30, 36, 42, 48];

const EXTREME_PERCENTILE: f64 = 0.98;
const REASONABLE_TOLERANCE: f64 = 1.25;
const CORRELATION_TOLERANCE: f64 = 0.85;
const MIN_REGRESSION_POINTS: usize = 4;
const PDSI_ANCHOR: f64 = 4.0;

/// The 1-indexed `k`th smallest value, or NaN when `k` is out of range.
///
/// Python: `self_calibration.kth_smallest`. The values carry no NaN; ties
/// between `-0.0` and `0.0` may resolve to either zero, which no caller can
/// tell apart.
fn kth_smallest(values: &mut [f64], k: usize) -> f64 {
    if k < 1 || k > values.len() {
        return f64::NAN;
    }
    *values.select_nth_unstable_by(k - 1, f64::total_cmp).1
}

/// The truncated-rank order statistic of the non-NaN values.
///
/// Python: `self_calibration.nan_safe_percentile`, for a `fraction` Python has
/// already checked lies in `[0, 1]`.
pub fn nan_safe_percentile(values: &[f64], fraction: f64) -> f64 {
    let mut present: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    let k = (fraction * present.len() as f64) as usize;
    kth_smallest(&mut present, k)
}

/// The most extreme rolling sum that is not a freak anomaly.
///
/// Python: `self_calibration._highest_reasonable`, only reached for wet spells.
fn highest_reasonable(sums: &[f64], spell: Spell) -> f64 {
    let sign = spell.sign();
    let threshold = nan_safe_percentile(sums, EXTREME_PERCENTILE);
    let mut highest = 0.0;
    for &value in sums {
        if sign * value <= 0.0 {
            continue;
        }
        let is_reasonable = if threshold.is_nan() {
            // too few sums for a percentile: the reference lets every candidate through
            true
        } else if threshold == 0.0 {
            false
        } else {
            (value / threshold) < REASONABLE_TOLERANCE
        };
        if is_reasonable && sign * value > sign * highest {
            highest = value;
        }
    }
    highest
}

/// The representative extreme rolling Z sum for one window length, or `None`
/// when the series never fills a window.
///
/// Python: `self_calibration.extreme_z_sum`, including its subtract-then-add
/// slide and the wet-only anomaly filter.
fn extreme_z_sum(series: &[f64], window_length: usize, spell: Spell) -> Option<f64> {
    let sign = spell.sign();
    let mut window = VecDeque::with_capacity(window_length);
    let mut running = 0.0;
    let mut values = series.iter().copied();
    while window.len() < window_length {
        let value = values.next()?;
        if !value.is_nan() {
            running += value;
            window.push_back(value);
        }
    }

    let mut extreme = running;
    let mut sums = vec![running];
    for value in values {
        if !value.is_nan() {
            running -= window.pop_front().unwrap_or(0.0);
            running += value;
            window.push_back(value);
            sums.push(running);
        }
        if sign * running > sign * extreme {
            extreme = running;
        }
    }

    Some(match spell {
        Spell::Dry => extreme,
        Spell::Wet => highest_reasonable(&sums, spell),
    })
}

/// The degenerate-variance or non-finite failure of the adaptive least squares.
fn degenerate(ss_x: f64, ss_y: f64) -> Result<(), ClimateError> {
    if !ss_x.is_finite() || !ss_y.is_finite() || ss_x <= 0.0 || ss_y <= 0.0 {
        return Err(ClimateError::NoConvergence {
            message: "least-squares fit has degenerate variance",
        });
    }
    Ok(())
}

/// The reference's adaptive least-squares `(slope, intercept)`.
///
/// Python: `self_calibration.least_squares_fit`, for its ten-point call.
fn least_squares_fit(x: &[f64], y: &[f64], spell: Spell) -> Result<(f64, f64), ClimateError> {
    let sign = spell.sign();
    if !x.iter().chain(y).all(|v| v.is_finite()) {
        return Err(ClimateError::NoConvergence {
            message: "least-squares fit received non-finite values",
        });
    }

    let count = x.len();
    let (mut sum_x, mut sum_y, mut sum_x2, mut sum_y2, mut sum_xy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (&this_x, &this_y) in x.iter().zip(y) {
        sum_x += this_x;
        sum_y += this_y;
        sum_x2 += this_x * this_x;
        sum_y2 += this_y * this_y;
        sum_xy += this_x * this_y;
    }

    let n = count as f64;
    let mut ss_x = sum_x2 - (sum_x * sum_x) / n;
    let mut ss_y = sum_y2 - (sum_y * sum_y) / n;
    let mut ss_xy = sum_xy - (sum_x * sum_y) / n;
    degenerate(ss_x, ss_y)?;
    let mut correlation = ss_xy / (ss_x.sqrt() * ss_y.sqrt());

    // drop trailing points until the fit correlates well enough
    let mut last = count - 1;
    while sign * correlation < CORRELATION_TOLERANCE && last > MIN_REGRESSION_POINTS - 1 {
        let (this_x, this_y) = (x[last], y[last]);
        sum_x -= this_x;
        sum_y -= this_y;
        sum_x2 -= this_x * this_x;
        sum_y2 -= this_y * this_y;
        sum_xy -= this_x * this_y;

        let n = last as f64;
        ss_x = sum_x2 - (sum_x * sum_x) / n;
        ss_y = sum_y2 - (sum_y * sum_y) / n;
        ss_xy = sum_xy - (sum_x * sum_y) / n;
        degenerate(ss_x, ss_y)?;
        correlation = ss_xy / (ss_x.sqrt() * ss_y.sqrt());
        last -= 1;
    }

    let slope = ss_xy / ss_x;

    // anchor the line on the retained point with the most extreme residual
    let mut max_residual = 0.0;
    let (mut anchor_x, mut anchor_y) = (x[0], 0.0);
    for (&this_x, &this_y) in x.iter().zip(y).take(last + 1) {
        let residual = this_y - slope * this_x;
        if sign * residual > sign * max_residual {
            max_residual = residual;
            anchor_x = this_x;
            anchor_y = this_y;
        }
    }
    Ok((slope, anchor_y - slope * anchor_x))
}

/// The wet or dry duration factors `(m, b)` for one location.
///
/// - Python source: `self_calibration.duration_factors`, with the
///   `extreme_z_sum`, `_highest_reasonable`, `nan_safe_percentile`,
///   `kth_smallest`, and `least_squares_fit` it calls.
/// - Inputs: `z`, the calibration-period Z-index series; NaN is missing.
/// - Outputs: `(m, b)`, the slope and intercept normalized against PDSI +/-4.
/// - Zero semantics: a wet-side window with no surviving sum contributes the
///   reference's `0.0`, and a zero percentile threshold rejects every sum.
/// - NaN semantics: missing periods never enter a rolling window.
/// - Short series: fewer non-missing values than the longest window is
///   [`ClimateError::InsufficientData`]; Python keeps that case and raises.
/// - Degenerate fits: non-finite sums or degenerate variance are
///   [`ClimateError::NoConvergence`] with Python's message.
/// - Numerics: rolling sums subtract the leaving value before adding the
///   entering one, and every regression sum runs sequentially, as in Python.
pub fn duration_factors(z: &[f64], spell: Spell) -> Result<(f64, f64), ClimateError> {
    let mut sums = [0.0; DURATION_FACTOR_WINDOW_LENGTHS.len()];
    for (sum, &length) in sums.iter_mut().zip(&DURATION_FACTOR_WINDOW_LENGTHS) {
        *sum = extreme_z_sum(z, length, spell).ok_or_else(|| ClimateError::InsufficientData {
            required: length,
            available: z.iter().filter(|v| !v.is_nan()).count(),
        })?;
    }
    let lengths = DURATION_FACTOR_WINDOW_LENGTHS.map(|length| length as f64);
    let (slope, intercept) = least_squares_fit(&lengths, &sums, spell)?;
    let anchor = spell.sign() * PDSI_ANCHOR;
    Ok((slope / anchor, intercept / anchor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_truncates_the_rank_and_skips_nan() {
        let values = [5.0, f64::NAN, 1.0, 3.0, 2.0, 4.0];
        // int(0.5 * 5) = 2 -> the second smallest
        assert_eq!(nan_safe_percentile(&values, 0.5), 2.0);
        // int(0.1 * 5) = 0 -> no rank
        assert!(nan_safe_percentile(&values, 0.1).is_nan());
    }

    #[test]
    fn dry_extreme_sum_is_the_most_negative_window() {
        let z = [1.0, -2.0, -3.0, f64::NAN, 4.0, -1.0];
        // windows of two non-missing values: -1, -5, 1, 3
        assert_eq!(extreme_z_sum(&z, 2, Spell::Dry), Some(-5.0));
        assert_eq!(extreme_z_sum(&z, 6, Spell::Dry), None);
    }

    #[test]
    fn a_short_record_is_insufficient_data() {
        let z = [1.0; 47];
        assert_eq!(
            duration_factors(&z, Spell::Wet),
            Err(ClimateError::InsufficientData {
                required: 48,
                available: 47
            })
        );
    }

    #[test]
    fn identical_sums_have_degenerate_variance() {
        let x = [3.0, 6.0, 9.0, 12.0];
        let y = [1.0; 4];
        assert!(matches!(
            least_squares_fit(&x, &y, Spell::Wet),
            Err(ClimateError::NoConvergence { .. })
        ));
    }

    #[test]
    fn collinear_points_fit_their_line() {
        let x = [3.0, 6.0, 9.0, 12.0];
        let y = [7.0, 13.0, 19.0, 25.0];
        let (slope, intercept) = least_squares_fit(&x, &y, Spell::Wet).unwrap();
        assert!((slope - 2.0).abs() < 1e-12);
        assert!((intercept - 1.0).abs() < 1e-12);
    }
}
