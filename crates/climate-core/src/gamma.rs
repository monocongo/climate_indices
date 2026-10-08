//! Gamma-distribution kernels behind SPI (and SPEI or the standardized index
//! when fitted to gamma).
//!
//! Reference: McKee, Doesken, and Kleist (1993), "The relationship of drought
//! frequency and duration to time scales"; the shape estimate is Thom's (1958)
//! approximation to the maximum-likelihood estimate, "A note on the gamma
//! distribution", Monthly Weather Review 86(4), 117-122.
//!
//! The Python implementations in `climate_indices.compute` stay the reference
//! oracle; `tests/test_native_parity.py` checks these kernels against them at
//! `rtol = atol = 1e-10`. Calibration-period selection, the probability of zero,
//! its resets, zero placement, output scales, warnings, and logging all stay in
//! Python; these kernels take numbers that Python already prepared.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

use crate::ClimateError;
use crate::special::{igam, igamc};

/// Method-of-moments gamma shape and scale for each column of a calibration block.
///
/// - Python source: the `means` ... `betas` block of `compute.gamma_parameters`.
/// - Inputs: `calibration`, shaped (years, columns), one column per calendar
///   step (and grid cell), already restricted to the calibration years.
/// - Outputs: `(alphas, betas)`, one value per column.
/// - Zero semantics: zeros (including -0.0) are the separate zero mass and are
///   excluded from both means, as `gamma_parameters` replaces them with NaN.
/// - NaN semantics: NaN is missing and excluded. A column with no positive
///   value has a NaN mean, so its alpha and beta are NaN.
/// - Invalid input: a negative value enters the arithmetic mean but not the
///   log mean, whose `np.log` is NaN there, matching `np.nanmean`.
/// - Short series / degenerate columns: no minimum length is enforced; a
///   column of identical values gives `A = 0`, an infinite alpha, and a zero
///   beta, which the CDF treats as invalid, exactly as NumPy does.
/// - Numerics: `A = ln(mean) - mean(ln x)`,
///   `alpha = (1 + sqrt(1 + 4A/3)) / (4A)`, `beta = mean / alpha`. Sums run
///   sequentially over years, the order NumPy reduces axis 0 in, and every
///   expression keeps NumPy's operation order.
pub fn gamma_parameters(calibration: ArrayView2<'_, f64>) -> (Array1<f64>, Array1<f64>) {
    let columns = calibration.ncols();
    let mut alphas = Array1::<f64>::zeros(columns);
    let mut betas = Array1::<f64>::zeros(columns);
    Zip::from(&mut alphas)
        .and(&mut betas)
        .and(calibration.columns())
        .for_each(|alpha, beta, column| {
            let (mut sum, mut count) = (0.0, 0.0);
            let (mut log_sum, mut log_count) = (0.0, 0.0);
            for &x in column {
                if x.is_nan() || x == 0.0 {
                    continue;
                }
                sum += x;
                count += 1.0;
                let log = x.ln();
                if !log.is_nan() {
                    log_sum += log;
                    log_count += 1.0;
                }
            }
            let mean = sum / count;
            let a = mean.ln() - log_sum / log_count;
            *alpha = (1.0 + (1.0 + 4.0 * a / 3.0).sqrt()) / (4.0 * a);
            *beta = mean / *alpha;
        });
    (alphas, betas)
}

/// `scipy.stats.gamma.cdf(x, a=alpha, scale=beta)`, including its argument handling.
///
/// NaN when alpha or beta is not positive (or NaN) or `x / beta` is NaN; 1 at
/// `x / beta = +inf`; 0 at or below zero; otherwise Cephes `igam(alpha, x / beta)`.
pub fn gamma_cdf(x: f64, alpha: f64, beta: f64) -> f64 {
    let scaled = x / beta;
    if !(alpha > 0.0 && beta > 0.0) || scaled.is_nan() {
        f64::NAN
    } else if scaled == f64::INFINITY {
        1.0
    } else if scaled > 0.0 {
        igam(alpha, scaled)
    } else {
        0.0
    }
}

/// `scipy.stats.gamma.sf(x, a=alpha)` at unit scale, including its argument handling.
///
/// NaN when alpha is not positive (or NaN) or `x` is NaN; 1 at or below zero; 0 at
/// `x = +inf`; otherwise Cephes `igamc(alpha, x)`. The negative-skew Pearson Type
/// III CDF is this survival function.
pub fn gamma_sf(x: f64, alpha: f64) -> f64 {
    if alpha.is_nan() || alpha <= 0.0 || x.is_nan() {
        f64::NAN
    } else if x <= 0.0 {
        1.0
    } else if x == f64::INFINITY {
        0.0
    } else {
        igamc(alpha, x)
    }
}

/// Cumulative probability of each value under a gamma fit with a zero mass.
///
/// - Python source: the gamma CDF and mixing step of
///   `compute.transform_fitted_gamma`: `p0 + (1 - p0) * gamma.cdf(x)`, where a
///   zero value's gamma probability is 0 (it lies at the top of the zero mass).
/// - Inputs: `values` shaped (years, columns); `alphas`, `betas`, and the
///   effective `probabilities_of_zero` with one value per column. "Effective"
///   means after Python's resets: a step with every calibration value zero, or
///   with no defined zero mass, arrives here with `p0 = 0`.
/// - Outputs: probabilities shaped like `values`.
/// - Zero semantics: a zero (or -0.0) value has probability `p0`, whatever the fit.
/// - NaN semantics: a NaN value, or a non-zero value under an invalid fit
///   (non-positive or NaN alpha or beta), has NaN probability.
/// - Negative values: probability `p0`, since the gamma CDF is 0 below its
///   support; the index functions clip negatives before reaching here.
///
/// Returns [`ClimateError::ShapeMismatch`] when a parameter's length is not the
/// number of columns.
pub fn gamma_probabilities(
    values: ArrayView2<'_, f64>,
    alphas: ArrayView1<'_, f64>,
    betas: ArrayView1<'_, f64>,
    probabilities_of_zero: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, ClimateError> {
    let columns = values.ncols();
    for (argument, parameter) in [
        ("alphas", &alphas),
        ("betas", &betas),
        ("probabilities_of_zero", &probabilities_of_zero),
    ] {
        if parameter.len() != columns {
            return Err(ClimateError::ShapeMismatch {
                argument,
                expected: columns,
                actual: parameter.len(),
            });
        }
    }

    let mut probabilities = Array2::<f64>::zeros(values.raw_dim());
    Zip::from(probabilities.columns_mut())
        .and(values.columns())
        .and(&alphas)
        .and(&betas)
        .and(&probabilities_of_zero)
        .for_each(|mut out, column, &alpha, &beta, &p0| {
            Zip::from(&mut out).and(&column).for_each(|p, &x| {
                let gamma_probability = if x == 0.0 {
                    0.0
                } else {
                    gamma_cdf(x, alpha, beta)
                };
                *p = p0 + (1.0 - p0) * gamma_probability;
            });
        });
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fit_excludes_zeros_and_missing_values() {
        let with_gaps = array![[1.0, 0.0], [f64::NAN, 2.0], [3.0, 0.0], [2.0, 4.0]];
        let (alphas, betas) = gamma_parameters(with_gaps.view());
        let (expected_alphas, expected_betas) =
            gamma_parameters(array![[1.0, 2.0], [3.0, 4.0], [2.0, 2.0]].view());
        // column 0 has values 1, 3, 2 and column 1 has values 2, 4 in both blocks
        assert_eq!(alphas[0], expected_alphas[0]);
        assert_eq!(betas[0], expected_betas[0]);
        let mean: f64 = 3.0;
        let a = mean.ln() - (2.0_f64.ln() + 4.0_f64.ln()) / 2.0;
        let alpha = (1.0 + (1.0 + 4.0 * a / 3.0).sqrt()) / (4.0 * a);
        assert_eq!(alphas[1], alpha);
        assert_eq!(betas[1], mean / alpha);
    }

    #[test]
    fn fit_of_an_empty_column_is_nan() {
        let (alphas, betas) = gamma_parameters(array![[0.0], [f64::NAN]].view());
        assert!(alphas[0].is_nan() && betas[0].is_nan());
    }

    #[test]
    fn probabilities_place_zeros_at_the_zero_mass() {
        let values = array![[0.0, 1.0], [2.0, f64::NAN]];
        let p = gamma_probabilities(
            values.view(),
            array![2.0, f64::NAN].view(),
            array![1.0, 1.0].view(),
            array![0.25, 0.5].view(),
        )
        .unwrap();
        assert_eq!(p[[0, 0]], 0.25);
        assert_eq!(p[[1, 0]], 0.25 + 0.75 * igam(2.0, 2.0));
        assert!(
            p[[0, 1]].is_nan(),
            "a non-zero value under an invalid fit is NaN"
        );
        assert!(p[[1, 1]].is_nan());
    }

    #[test]
    fn probabilities_reject_a_parameter_of_the_wrong_length() {
        let error = gamma_probabilities(
            array![[1.0, 2.0]].view(),
            array![1.0].view(),
            array![1.0, 1.0].view(),
            array![0.0, 0.0].view(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "alphas",
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn survival_function_matches_scipy_including_a_deep_tail() {
        // reference values from scipy.stats.gamma.sf(x, a) 1.17.0
        let cases = [
            (0.5, 2.0, 0.909_795_989_568_950_1),
            (3.0, 0.5, 0.014_305_878_435_429_645),
            (10.0, 4.0, 0.010_336_050_675_925_726),
            (30.0, 5.0, 3.624_300_952_061_492_4e-9),
            (1e-3, 0.5, 0.964_329_408_270_320_1),
        ];
        for (x, alpha, expected) in cases {
            let actual = gamma_sf(x, alpha);
            // 1 - igam would lose the deep tail's digits, so the check is relative
            assert!(
                (actual - expected).abs() <= 1e-14 * expected,
                "gamma_sf({x}, {alpha}) = {actual:e}, expected {expected:e}"
            );
        }
    }

    #[test]
    fn survival_function_mirrors_scipy_argument_handling() {
        assert!(gamma_sf(1.0, 0.0).is_nan());
        assert!(gamma_sf(1.0, f64::NAN).is_nan());
        assert!(gamma_sf(f64::NAN, 1.0).is_nan());
        assert_eq!(gamma_sf(0.0, 1.0), 1.0);
        assert_eq!(gamma_sf(-3.0, 1.0), 1.0);
        assert_eq!(gamma_sf(f64::INFINITY, 1.0), 0.0);
    }

    #[test]
    fn cdf_mirrors_scipy_argument_handling() {
        assert!(gamma_cdf(1.0, 0.0, 1.0).is_nan());
        assert!(gamma_cdf(1.0, 1.0, -1.0).is_nan());
        assert!(gamma_cdf(f64::INFINITY, 1.0, f64::INFINITY).is_nan());
        assert_eq!(gamma_cdf(f64::INFINITY, 1.0, 1.0), 1.0);
        assert_eq!(gamma_cdf(-1.0, 1.0, 1.0), 0.0);
        assert_eq!(gamma_cdf(1.0, 1.0, f64::INFINITY), 0.0);
        assert_eq!(gamma_cdf(1.0, f64::INFINITY, 1.0), 0.0);
    }
}
