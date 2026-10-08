//! Pearson Type III kernels behind SPI and SPEI when fitted to Pearson Type III.
//!
//! Reference: Hosking (1990) for the L-moment fit, and the `pearson3` subroutine
//! of Hosking's IBM Research Report RC20525 (1996) for its parameter estimate;
//! the CDF follows `scipy.stats.pearson3` (Vogel and McMartin, 1991).
//!
//! The Python implementations in `climate_indices.compute` and
//! `climate_indices.lmoments` stay the reference oracle;
//! `tests/test_native_parity_distributions.py` checks these kernels against them at
//! `rtol = atol = 1e-10`. Calibration-period selection, the fallback to gamma,
//! goodness-of-fit checks, support-limit masks, zero placement, output scales,
//! warnings, and logging all stay in Python.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

use crate::ClimateError;
use crate::gamma::{gamma_cdf, gamma_sf};
use crate::lmoments::sample_lmoments;
use crate::special::{lgam, ndtr};

/// Hosking `pearson3` coefficients (c1, c2, c3, d1, d2, d3, d4, d5, d6).
const C1: f64 = 0.2906;
const C2: f64 = 0.1882;
const C3: f64 = 0.0442;
const D1: f64 = 0.36067;
const D2: f64 = -0.59567;
const D3: f64 = 0.25361;
const D4: f64 = -2.78861;
const D5: f64 = 2.56096;
const D6: f64 = -0.77045;

/// Fewest non-zero values a column needs for a Pearson Type III fit.
const MIN_NON_ZERO_VALUES: usize = 4;

/// SciPy's brute-force divide between the normal and Pearson Type III CDF.
const NORMAL_TRANSITION_SKEW: f64 = 0.000_016;

/// Pearson Type III fit of every column of a calibration block.
///
/// A column that cannot be fitted has `valid == false` and zero parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PearsonFit {
    pub probabilities_of_zero: Array1<f64>,
    pub locs: Array1<f64>,
    pub scales: Array1<f64>,
    pub skews: Array1<f64>,
    pub valid: Array1<bool>,
}

/// Probability of zero and L-moment Pearson Type III `(loc, scale, skew)` of one column.
fn fit_column(column: ArrayView1<'_, f64>) -> Option<[f64; 4]> {
    let mut zeros = 0_usize;
    let mut non_missing = 0_usize;
    for &x in column {
        if !x.is_nan() {
            non_missing += 1;
            if x == 0.0 {
                zeros += 1;
            }
        }
    }
    if non_missing - zeros < MIN_NON_ZERO_VALUES {
        return None;
    }
    let probability_of_zero = if zeros > 0 {
        zeros as f64 / non_missing as f64
    } else {
        0.0
    };

    let [loc, second, skewness] = sample_lmoments(column)?;
    let t3 = skewness.abs();
    // negated so that a NaN L-moment is invalid
    if !(second > 0.0 && t3 < 1.0) {
        return None;
    }

    let (scale, skew) = if t3 <= 1e-6 {
        // skewness is effectively zero
        (second * PI.sqrt(), 0.0)
    } else {
        let alpha = if t3 < 0.333_333_333 {
            let t = PI * 3.0 * t3 * t3;
            (1.0 + (C1 * t)) / (t * (1.0 + (t * (C2 + (t * C3)))))
        } else {
            let t = 1.0 - t3;
            t * (D1 + (t * (D2 + (t * D3)))) / (1.0 + (t * (D4 + (t * (D5 + (t * D6))))))
        };
        let alpha_root = alpha.sqrt();
        let beta = PI.sqrt() * second * (lgam(alpha) - lgam(alpha + 0.5)).exp();
        let skew = if skewness < 0.0 {
            -2.0 / alpha_root
        } else {
            2.0 / alpha_root
        };
        (beta * alpha_root, skew)
    };
    Some([probability_of_zero, loc, scale, skew])
}

/// L-moment Pearson Type III parameters and probability of zero for each column.
///
/// - Python source: `compute._pearson_parameters_spatial` with
///   `lmoments.fit_spatial`, which `compute.calculate_time_step_params` mirrors
///   one time step at a time.
/// - Inputs: `calibration`, shaped (years, columns), one column per calendar step
///   (and grid cell), already restricted to the calibration years.
/// - Outputs: a [`PearsonFit`] with one value per column.
/// - Zero semantics: zeros count toward the probability of zero, `zeros /
///   non-missing`, and stay in the L-moment sample, as in Python.
/// - NaN semantics: NaN is missing. A column with fewer than four non-zero
///   values, fewer than four non-missing values, a non-positive second
///   L-moment, or `|tau_3| >= 1` (or a NaN there) is invalid.
/// - Numerics: `scale` and `skew` follow Hosking's `pearson3`, with
///   `gammaln` as Cephes `lgam`. A near-zero `tau_3` (at most 1e-6) is a zero
///   skew and `scale = lambda_2 * sqrt(pi)`.
pub fn pearson_parameters(calibration: ArrayView2<'_, f64>) -> PearsonFit {
    let columns = calibration.ncols();
    let mut fit = PearsonFit {
        probabilities_of_zero: Array1::zeros(columns),
        locs: Array1::zeros(columns),
        scales: Array1::zeros(columns),
        skews: Array1::zeros(columns),
        valid: Array1::from_elem(columns, false),
    };
    Zip::from(&mut fit.probabilities_of_zero)
        .and(&mut fit.locs)
        .and(&mut fit.scales)
        .and(&mut fit.skews)
        .and(&mut fit.valid)
        .and(calibration.columns())
        .for_each(|p0, loc, scale, skew, valid, column| {
            if let Some(parameters) = fit_column(column) {
                [*p0, *loc, *scale, *skew] = parameters;
                *valid = true;
            }
        });
    fit
}

/// `scipy.stats.pearson3.cdf(x, skew, loc=loc, scale=scale)`, including its argument handling.
///
/// NaN when the skew is not finite, the scale is not positive, or the
/// standardized value is NaN; 1 at +inf and 0 at -inf; the normal CDF when
/// `|skew| < 1.6e-5`; otherwise the gamma CDF (positive skew) or survival
/// function (negative skew) of `beta * (z - zeta)` with shape `alpha = beta^2`.
pub fn pearson_cdf(x: f64, skew: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    if !(skew.is_finite() && scale > 0.0) || z.is_nan() {
        return f64::NAN;
    }
    if z == f64::INFINITY {
        return 1.0;
    }
    if z == f64::NEG_INFINITY {
        return 0.0;
    }
    if skew.abs() < NORMAL_TRANSITION_SKEW {
        return ndtr(z);
    }
    let beta = 2.0 / skew;
    let alpha = beta * beta;
    let zeta = 0.0 - alpha / beta;
    let transformed = beta * (z - zeta);
    if skew > 0.0 {
        gamma_cdf(transformed, alpha, 1.0)
    } else {
        gamma_sf(transformed, alpha)
    }
}

/// Pearson Type III CDF of each value, one `(skew, loc, scale)` per column.
///
/// - Python source: the `scipy.stats.pearson3.cdf` call in `compute._pearson_fit`.
/// - Inputs: `values` shaped (years, columns); `skews`, `locs`, and `scales`
///   with one value per column.
/// - Outputs: the CDF shaped like `values`, before Python applies the zero,
///   trace, and support-limit masks and the probability of zero.
/// - Zero semantics: a zero is an ordinary value here; its zero mass is applied
///   by Python afterwards.
/// - NaN semantics: a NaN value, and every value under a parameter set whose
///   skew is not finite or whose scale is not positive (or NaN), is NaN.
/// - Invalid input: that is the only handling; nothing is rejected, as in SciPy.
///
/// Returns [`ClimateError::ShapeMismatch`] when a parameter's length is not the
/// number of columns.
pub fn pearson_cdf_block(
    values: ArrayView2<'_, f64>,
    skews: ArrayView1<'_, f64>,
    locs: ArrayView1<'_, f64>,
    scales: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, ClimateError> {
    let columns = values.ncols();
    for (argument, parameter) in [("skews", &skews), ("locs", &locs), ("scales", &scales)] {
        if parameter.len() != columns {
            return Err(ClimateError::ShapeMismatch {
                argument,
                expected: columns,
                actual: parameter.len(),
            });
        }
    }

    let mut cdf = Array2::<f64>::zeros(values.raw_dim());
    Zip::from(cdf.columns_mut())
        .and(values.columns())
        .and(&skews)
        .and(&locs)
        .and(&scales)
        .for_each(|mut out, column, &skew, &loc, &scale| {
            Zip::from(&mut out)
                .and(&column)
                .for_each(|p, &x| *p = pearson_cdf(x, skew, loc, scale));
        });
    Ok(cdf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fit_of_a_symmetric_sample_has_zero_skew() {
        // 1..=7 has tau_3 = 0, so scale = lambda_2 * sqrt(pi)
        let sample = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
        let fit = pearson_parameters(sample.view());
        assert!(fit.valid[0]);
        assert_eq!(fit.skews[0], 0.0);
        assert_eq!(fit.locs[0], 4.0);
        assert!((fit.scales[0] - 1.333_333_333_333_333_3 * PI.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn fit_reports_zero_mass_and_rejects_too_few_non_zero_values() {
        let sample = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [3.0, 3.0],
            [5.0, 4.0],
            [4.0, 9.0]
        ];
        let fit = pearson_parameters(sample.view());
        assert_eq!(fit.probabilities_of_zero[0], 0.0);
        assert!(
            !fit.valid[0],
            "three non-zero values is below the minimum of four"
        );
        assert!(fit.valid[1]);
        assert_eq!(fit.probabilities_of_zero[1], 2.0 / 6.0);
        // invalid columns keep zero parameters
        assert_eq!((fit.locs[0], fit.scales[0], fit.skews[0]), (0.0, 0.0, 0.0));
    }

    #[test]
    fn fit_rejects_a_column_with_too_few_non_missing_values() {
        let sample = array![[1.0], [f64::NAN], [2.0], [3.0], [f64::NAN], [f64::NAN]];
        assert!(!pearson_parameters(sample.view()).valid[0]);
    }

    // Reference values from climate_indices.lmoments.fit (the Python oracle) and
    // scipy.stats.pearson3.cdf 1.17.0, on samples chosen to take each branch of the
    // estimate: tau_3 < 1/3, tau_3 >= 1/3, and a negative skew.
    #[test]
    fn fit_matches_the_python_oracle_on_each_branch() {
        let sample = array![
            [12.0, 1.0, 99.0],
            [15.5, 1.0, 99.0],
            [9.1, 1.0, 99.0],
            [30.2, 2.0, 98.0],
            [22.4, 2.0, 98.0],
            [18.8, 3.0, 97.0],
            [11.3, 4.0, 96.0],
            [40.7, 8.0, 92.0],
            [25.0, 20.0, 80.0],
            [14.2, 60.0, 40.0],
        ];
        let expected = [
            // (loc, scale, skew)
            (19.92, 10.997_734_974_299_528, 1.728_920_122_795_316),
            (10.2, 26.672_309_243_115_933, 5.910_576_578_609_104),
            (89.8, 26.672_309_243_116_08, -5.910_576_578_609_143_5),
        ];
        let fit = pearson_parameters(sample.view());
        for (column, (loc, scale, skew)) in expected.into_iter().enumerate() {
            assert!(fit.valid[column]);
            assert_eq!(fit.probabilities_of_zero[column], 0.0);
            for (name, actual, wanted) in [
                ("loc", fit.locs[column], loc),
                ("scale", fit.scales[column], scale),
                ("skew", fit.skews[column], skew),
            ] {
                assert!(
                    (actual - wanted).abs() <= 1e-13 * wanted.abs(),
                    "column {column} {name} = {actual:e}, expected {wanted:e}"
                );
            }
        }
    }

    #[test]
    fn fit_keeps_a_tiny_but_non_zero_skew() {
        // tau_3 is about 1.6e-4: above the 1e-6 zero-skew cut-off, where alpha is
        // large and `scale` is only reproducible to the conditioning of gammaln, so
        // only loosely compared; tests/test_native_parity_distributions.py checks it exactly
        let fit = pearson_parameters(array![[1.0], [2.0], [3.0], [4.001]].view());
        assert!(fit.valid[0]);
        assert!(fit.skews[0] > 0.0, "skew = {}", fit.skews[0]);
        assert!((fit.scales[0] - 1.477_488_148).abs() < 1e-6);
    }

    #[test]
    fn cdf_matches_scipy_for_both_skew_signs() {
        // (x, skew, loc, scale, scipy.stats.pearson3.cdf)
        let cases = [
            (1.0, 0.8, 3.0, 2.0, 0.149_591_916_675_002_43),
            (4.0, 0.8, 3.0, 2.0, 0.726_357_546_160_562_9),
            (9.0, 0.8, 3.0, 2.0, 0.991_621_686_655_470_5),
            (-3.0, -0.8, 3.0, 2.0, 0.008_378_313_344_529_589),
            (4.0, -0.8, 3.0, 2.0, 0.656_058_361_824_302_2),
            (1.0, 1.5, 3.0, 2.0, 0.108_815_120_615_328_28),
            (9.0, 1.5, 3.0, 2.0, 0.985_210_904_298_800_1),
            (-3.0, -1.5, 3.0, 2.0, 0.014_789_095_701_199_942),
            (4.0, -1.5, 3.0, 2.0, 0.625_760_220_566_874_7),
            // at and beyond the support limit
            (-3.0, 0.8, 3.0, 2.0, 0.0),
            (9.0, -0.8, 3.0, 2.0, 1.0),
        ];
        for (x, skew, loc, scale, expected) in cases {
            let actual = pearson_cdf(x, skew, loc, scale);
            assert!(
                (actual - expected).abs() <= 1e-14 * expected,
                "pearson_cdf({x}, {skew}, {loc}, {scale}) = {actual:e}, expected {expected:e}"
            );
        }
    }

    #[test]
    fn cdf_mirrors_scipy_argument_handling() {
        assert!(pearson_cdf(1.0, f64::NAN, 0.0, 1.0).is_nan());
        assert!(pearson_cdf(1.0, f64::INFINITY, 0.0, 1.0).is_nan());
        assert!(pearson_cdf(1.0, 1.0, 0.0, 0.0).is_nan());
        assert!(pearson_cdf(1.0, 1.0, 0.0, -1.0).is_nan());
        assert!(pearson_cdf(f64::NAN, 1.0, 0.0, 1.0).is_nan());
        assert_eq!(pearson_cdf(f64::INFINITY, 1.0, 0.0, 1.0), 1.0);
        assert_eq!(pearson_cdf(f64::NEG_INFINITY, -1.0, 0.0, 1.0), 0.0);
        // a near-zero skew is the normal CDF
        assert_eq!(pearson_cdf(0.0, 1e-6, 0.0, 1.0), 0.5);
    }

    #[test]
    fn cdf_of_opposite_skews_reflects_about_the_location() {
        let positive = pearson_cdf(0.7, 0.8, 0.0, 1.0);
        let negative = pearson_cdf(-0.7, -0.8, 0.0, 1.0);
        assert!((positive + negative - 1.0).abs() < 1e-15);
    }

    #[test]
    fn cdf_block_rejects_a_parameter_of_the_wrong_length() {
        let error = pearson_cdf_block(
            array![[1.0, 2.0]].view(),
            array![1.0, 1.0].view(),
            array![0.0].view(),
            array![1.0, 1.0].view(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "locs",
                expected: 2,
                actual: 1
            }
        );
    }
}
