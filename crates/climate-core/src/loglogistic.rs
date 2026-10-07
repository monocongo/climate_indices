//! Generalized logistic (log-logistic, GLO) kernels behind SPEI's reference distribution.
//!
//! Reference: Hosking (1990); the parameter estimate is `PELGLO` and the CDF is
//! `CDFGLO` from Hosking's IBM Research Report RC20525 (1996) as distributed in
//! the R package `lmom`. Vicente-Serrano, Beguería, and López-Moreno (2010)
//! define SPEI on it; ADR-0016 records the choice.
//!
//! The Python implementations in `climate_indices.compute` and
//! `climate_indices.lmoments` stay the reference oracle;
//! `tests/test_native_parity.py` checks these kernels against them at
//! `rtol = atol = 1e-10`. Calibration-period selection, validity masking of the
//! fitted parameters, output scales, warnings, and logging stay in Python.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};

use crate::ClimateError;
use crate::lmoments::sample_lmoments;

/// A shape of at most this magnitude is the ordinary logistic, `SMALL` in `PELGLO`.
const SHAPE_TOLERANCE: f64 = 1e-6;

/// GLO fit of every column of a calibration block.
///
/// A column that cannot be fitted has `valid == false` and zero parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct LogLogisticFit {
    pub locs: Array1<f64>,
    pub scales: Array1<f64>,
    pub shapes: Array1<f64>,
    pub valid: Array1<bool>,
}

/// `(loc, scale, shape)` of one column from its L-moments.
fn fit_column(column: ArrayView1<'_, f64>) -> Option<[f64; 3]> {
    let [first, second, skewness] = sample_lmoments(column)?;
    let shape = -skewness;
    // negated so that a NaN L-moment is invalid
    if !(second > 0.0 && shape.abs() < 1.0) {
        return None;
    }
    if shape.abs() <= SHAPE_TOLERANCE {
        return Some([first, second, 0.0]);
    }
    let gg = shape * PI / (shape * PI).sin();
    let scale = second / gg;
    Some([first - scale * (1.0 - gg) / shape, scale, shape])
}

/// L-moment GLO location, scale, and shape for each column.
///
/// - Python source: `lmoments.fit_glo_spatial`, which `lmoments.fit_glo` mirrors
///   one time step at a time, via `compute._loglogistic_parameters_spatial`.
/// - Inputs: `calibration`, shaped (years, columns), one column per calendar step
///   (and grid cell), already restricted to the calibration years.
/// - Outputs: a [`LogLogisticFit`] with one value per column.
/// - Zero semantics: every value takes part in the fit, zeros included; unlike
///   gamma and Pearson Type III there is no separate zero mass.
/// - NaN semantics: NaN is missing. A column with fewer than four non-missing
///   values, a non-positive second L-moment, or `|shape| >= 1` (or a NaN
///   there) is invalid.
/// - Numerics: `shape = -tau_3`; a shape of at most 1e-6 in magnitude is
///   exactly zero (the ordinary logistic), otherwise `PELGLO`'s `gg`.
pub fn loglogistic_parameters(calibration: ArrayView2<'_, f64>) -> LogLogisticFit {
    let columns = calibration.ncols();
    let mut fit = LogLogisticFit {
        locs: Array1::zeros(columns),
        scales: Array1::zeros(columns),
        shapes: Array1::zeros(columns),
        valid: Array1::from_elem(columns, false),
    };
    Zip::from(&mut fit.locs)
        .and(&mut fit.scales)
        .and(&mut fit.shapes)
        .and(&mut fit.valid)
        .and(calibration.columns())
        .for_each(|loc, scale, shape, valid, column| {
            if let Some(parameters) = fit_column(column) {
                [*loc, *scale, *shape] = parameters;
                *valid = true;
            }
        });
    fit
}

/// Generalized logistic CDF, `cdfglo`, clipped to [0, 1].
///
/// With `z = (x - loc) / scale`: `F = 1 / (1 + exp(-y))`, where `y = z` for a
/// shape of at most 1e-6 in magnitude, else `y = -ln(max(0, 1 - shape * z)) / shape`.
/// A NaN input gives NaN. This is the bare formula: the caller decides which
/// parameter sets are valid, as `compute._loglogistic_fit` does.
pub fn loglogistic_cdf(x: f64, loc: f64, scale: f64, shape: f64) -> f64 {
    let z = (x - loc) / scale;
    let y = if shape.abs() <= SHAPE_TOLERANCE {
        z
    } else {
        // f64::max would drop a NaN, which np.maximum propagates
        let argument = 1.0 - shape * z;
        let argument = if argument.is_nan() {
            argument
        } else {
            argument.max(0.0)
        };
        -argument.ln() / shape
    };
    (1.0 / (1.0 + (-y).exp())).clamp(0.0, 1.0)
}

/// GLO CDF of each value, one `(loc, scale, shape)` per column.
///
/// - Python source: the probability step of `compute._loglogistic_fit`.
/// - Inputs: `values` shaped (years, columns); `locs`, `scales`, and `shapes`
///   with one value per column.
/// - Outputs: probabilities shaped like `values`, before Python masks the
///   positions whose parameters are invalid and maps to the output scale.
///
/// Returns [`ClimateError::ShapeMismatch`] when a parameter's length is not the
/// number of columns.
pub fn loglogistic_cdf_block(
    values: ArrayView2<'_, f64>,
    locs: ArrayView1<'_, f64>,
    scales: ArrayView1<'_, f64>,
    shapes: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, ClimateError> {
    let columns = values.ncols();
    for (argument, parameter) in [("locs", &locs), ("scales", &scales), ("shapes", &shapes)] {
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
        .and(&locs)
        .and(&scales)
        .and(&shapes)
        .for_each(|mut out, column, &loc, &scale, &shape| {
            Zip::from(&mut out)
                .and(&column)
                .for_each(|p, &x| *p = loglogistic_cdf(x, loc, scale, shape));
        });
    Ok(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fit_of_a_symmetric_sample_is_the_ordinary_logistic() {
        // tau_3 = 0, so shape = 0 and (loc, scale) = (lambda_1, lambda_2)
        let sample = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]];
        let fit = loglogistic_parameters(sample.view());
        assert!(fit.valid[0]);
        assert_eq!(fit.shapes[0], 0.0);
        assert_eq!(fit.locs[0], 4.0);
        assert!((fit.scales[0] - 4.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn fit_keeps_zeros_and_rejects_a_short_column() {
        let sample = array![
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, f64::NAN],
            [4.0, 3.0],
            [9.0, f64::NAN]
        ];
        let fit = loglogistic_parameters(sample.view());
        assert!(fit.valid[0], "zeros are ordinary values in a GLO fit");
        assert!(
            !fit.valid[1],
            "three non-missing values is below the minimum"
        );
        assert_eq!((fit.locs[1], fit.scales[1], fit.shapes[1]), (0.0, 0.0, 0.0));
    }

    #[test]
    fn cdf_is_the_logistic_at_zero_shape_and_clips_beyond_the_support() {
        assert_eq!(loglogistic_cdf(2.0, 2.0, 3.0, 0.0), 0.5);
        // shape 0.5, scale 1: the support ends at z = 2
        assert_eq!(loglogistic_cdf(3.0, 0.0, 1.0, 0.5), 1.0);
        assert_eq!(loglogistic_cdf(-3.0, 0.0, 1.0, -0.5), 0.0);
        assert!(loglogistic_cdf(f64::NAN, 0.0, 1.0, 0.5).is_nan());
    }

    #[test]
    fn cdf_block_rejects_a_parameter_of_the_wrong_length() {
        let error = loglogistic_cdf_block(
            array![[1.0, 2.0]].view(),
            array![0.0, 0.0].view(),
            array![1.0, 1.0].view(),
            array![0.1].view(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "shapes",
                expected: 2,
                actual: 1
            }
        );
    }
}
