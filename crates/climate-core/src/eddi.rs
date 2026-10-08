//! EDDI kernels: the empirical rank count and Tukey plotting position, and the
//! Hastings inverse-normal approximation.
//!
//! References: Hobbins et al. (2016), "The Evaporative Demand Drought Index",
//! J. Hydrometeorology, doi:10.1175/JHM-D-15-0121.1, and Abramowitz & Stegun
//! (1965), eq. 26.2.23, the approximation the NOAA PSL Fortran reference uses.
//!
//! The Python implementation in `climate_indices.indices` stays the reference
//! oracle; `tests/test_native_parity.py` checks these kernels against it at
//! `rtol = atol = 1e-10` with identical NaN positions. Validation,
//! calibration-period resolution, the leading-scale-pad mask, the unfolding to
//! the caller's layout, logging, and warnings all stay in Python. SciPy is not
//! involved: EDDI is non-parametric, so there is no special function to port.

use ndarray::{Array2, ArrayView1, ArrayView2, Zip};

use crate::ClimateError;

// Abramowitz & Stegun (1965) 26.2.23, digit for digit from `indices._HASTINGS_*`.
const HASTINGS_C0: f64 = 2.515517;
const HASTINGS_C1: f64 = 0.802853;
const HASTINGS_C2: f64 = 0.010328;
const HASTINGS_D1: f64 = 1.432788;
const HASTINGS_D2: f64 = 0.189269;
const HASTINGS_D3: f64 = 0.001308;

/// Probability of each value in one calendar period, from its rank against the
/// period's calibration climatology (the Tukey plotting position).
///
/// - Python source: the rank count and `probabilities = ...` lines of the
///   per-period loop in `indices.eddi`. Rust has no equivalent of Python's
///   `_EDDI_RANK_COMPARISON_ELEMENT_BUDGET` chunking: the result is the same
///   whichever cells a chunk holds, so this walks every column in one pass.
/// - Inputs: `climatology`, shaped (climatology years, columns), and `values`,
///   shaped (years, columns), both already restricted to the calibration rows
///   and one calendar period; `pads`, one value per column, the count of
///   leading scale pads among that period's climatology rows.
/// - Outputs: probabilities shaped like `values`.
/// - Ranking: `below` counts the climatology values strictly less than a value,
///   the operation order NumPy's `count_nonzero(... < ...)` uses.
/// - NaN semantics: a missing climatology value never compares below anything;
///   a missing value's probability is NaN. A column whose climatology holds
///   fewer than two valid values has no ranking at all and is all NaN.
/// - Zero semantics: zero is a value like any other and ranks by the same
///   strict comparison.
/// - Ties: the strict `<` comparison is part of the contract, so an equal
///   climatology value never counts as below; equal values rank exactly as the
///   Python path ranks them.
///
/// Returns [`ClimateError::ShapeMismatch`] when a block has a different number
/// of columns, or when `pads` is not one value per column.
pub fn tukey_probabilities(
    climatology: ArrayView2<'_, f64>,
    values: ArrayView2<'_, f64>,
    pads: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, ClimateError> {
    let columns = values.ncols();
    if climatology.ncols() != columns {
        return Err(ClimateError::ShapeMismatch {
            argument: "climatology",
            expected: columns,
            actual: climatology.ncols(),
        });
    }
    if pads.len() != columns {
        return Err(ClimateError::ShapeMismatch {
            argument: "pads",
            expected: columns,
            actual: pads.len(),
        });
    }

    let mut probabilities = Array2::<f64>::zeros(values.raw_dim());
    Zip::from(probabilities.columns_mut())
        .and(climatology.columns())
        .and(values.columns())
        .and(pads)
        .for_each(|mut out, climatology, values, &pad| {
            let valid = climatology.iter().filter(|value| !value.is_nan()).count() as f64;
            for (probability, &value) in out.iter_mut().zip(values.iter()) {
                *probability = if value.is_nan() || valid < 2.0 {
                    f64::NAN
                } else {
                    let below = climatology
                        .iter()
                        .filter(|climatology| **climatology < value)
                        .count() as f64;
                    (pad + below + 0.66) / (valid + pad + 0.33)
                };
            }
        });
    Ok(probabilities)
}

/// Abramowitz & Stegun (1965) 26.2.23, the inverse normal approximation EDDI
/// uses, matching `indices._hastings_inverse_normal` and the NOAA PSL Fortran.
///
/// - Inputs: a cumulative probability. It is clipped to `[1e-10, 1 - 1e-10]`
///   first, so a value at or outside that range saturates rather than raising.
/// - Numerics: the lower tail is used and the sign flipped above 0.5, keeping
///   the Python expression's operation order and every constant exact.
/// - NaN semantics: NaN propagates to NaN.
pub fn hastings_inverse_normal(probability: f64) -> f64 {
    let probability = probability.clamp(1e-10, 1.0 - 1e-10);

    // work in the lower tail; flip if p > 0.5
    let sign = if probability <= 0.5 { -1.0 } else { 1.0 };
    let lower_tail = if probability <= 0.5 {
        probability
    } else {
        1.0 - probability
    };

    let t = (-2.0 * lower_tail.ln()).sqrt();
    let numerator = HASTINGS_C0 + t * (HASTINGS_C1 + t * HASTINGS_C2);
    let denominator = 1.0 + t * (HASTINGS_D1 + t * (HASTINGS_D2 + t * HASTINGS_D3));
    sign * (t - numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn probabilities_rank_strictly_and_place_the_pads() {
        // two climatology years, one column: values 1 and 2, no pads
        let probabilities = tukey_probabilities(
            array![[1.0], [2.0]].view(),
            array![[1.0], [2.0], [3.0]].view(),
            array![0.0].view(),
        )
        .unwrap();
        assert_eq!(probabilities[[0, 0]], 0.283_261_802_575_107_3);
        assert_eq!(probabilities[[1, 0]], 0.712_446_351_931_330_5);
        assert_eq!(probabilities[[2, 0]], 1.141_630_901_287_553_6);

        // one leading pad ranks below both observations, as NOAA counts it
        let padded = tukey_probabilities(
            array![[1.0], [2.0]].view(),
            array![[1.0], [3.0]].view(),
            array![1.0].view(),
        )
        .unwrap();
        assert_eq!(padded[[0, 0]], 0.498_498_498_498_498_53);
        assert_eq!(padded[[1, 0]], 1.099_099_099_099_099_2);
    }

    #[test]
    fn ties_rank_by_the_strict_comparison() {
        let probabilities = tukey_probabilities(
            array![[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]].view(),
            array![[2.0, 2.0]].view(),
            array![0.0, 0.0].view(),
        )
        .unwrap();
        // 2.0 is below only 1.0 in the first column and below nothing in the second
        assert_eq!(probabilities[[0, 0]], 0.498_498_498_498_498_53);
        assert_eq!(probabilities[[0, 1]], 0.198_198_198_198_198_2);
    }

    #[test]
    fn missing_values_and_short_climatologies_are_nan() {
        let probabilities = tukey_probabilities(
            array![[1.0, f64::NAN], [2.0, 3.0], [3.0, f64::NAN]].view(),
            array![[2.0, 3.0], [f64::NAN, 3.0]].view(),
            array![0.0, 0.0].view(),
        )
        .unwrap();
        assert_eq!(probabilities[[0, 0]], 0.498_498_498_498_498_53);
        assert!(probabilities[[1, 0]].is_nan());
        // the second column has one valid climatology value, so it has no ranking
        assert!(probabilities[[0, 1]].is_nan());
        assert!(probabilities[[1, 1]].is_nan());
    }

    #[test]
    fn probabilities_reject_mismatched_shapes() {
        let error = tukey_probabilities(
            array![[1.0, 2.0]].view(),
            array![[1.0]].view(),
            array![0.0].view(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "climatology",
                expected: 1,
                actual: 2
            }
        );
        let error = tukey_probabilities(
            array![[1.0]].view(),
            array![[1.0]].view(),
            array![0.0, 0.0].view(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "pads",
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn inverse_normal_matches_the_python_approximation() {
        // values from the Python `_hastings_inverse_normal`
        let cases = [
            (0.5, 1.010_066_754_680_849_5e-7),
            (0.66 / 3.33, -0.847_915_598_085_17),
            (1.66 / 3.33, -0.003_752_519_260_830_755_6),
            (1e-10, -6.360_938_869_405_1),
            (1.0 - 1e-10, 6.360_938_856_699_430_5),
            (0.025, -1.960_394_916_925_34),
            (0.975, 1.960_394_916_925_339_6),
        ];
        for (probability, expected) in cases {
            let actual = hastings_inverse_normal(probability);
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 1e-10,
                "hastings_inverse_normal({probability}): actual={actual}, expected={expected}"
            );
        }
        // clipping saturates the tails and NaN propagates
        assert_eq!(hastings_inverse_normal(0.0), hastings_inverse_normal(1e-10));
        assert_eq!(
            hastings_inverse_normal(1.0),
            hastings_inverse_normal(1.0 - 1e-10)
        );
        assert!(hastings_inverse_normal(f64::NAN).is_nan());
    }
}
