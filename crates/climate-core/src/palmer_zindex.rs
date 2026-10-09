//! Palmer CAFEC precipitation and Z-index kernels.
//!
//! Reference: Palmer (1965), "Meteorological Drought", U.S. Weather Bureau
//! Research Paper 45, equations for the climatically appropriate for existing
//! conditions (CAFEC) precipitation, the mean absolute moisture departure, and
//! the K-weighted Z-index.
//!
//! The Python implementation in `climate_indices.palmer` stays the reference
//! oracle; `tests/test_native_parity_palmer.py` checks these kernels against it
//! at `rtol = atol = 1e-10`. The CAFEC ratios, the T ratio, the K-factor
//! normalization, the scPDSI finiteness check, validation, and logging stay in
//! Python; these kernels take the water balance and coefficients Python already
//! prepared.

use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

use crate::ClimateError;

/// The water balance and CAFEC coefficients a month's CAFEC precipitation reads.
///
/// The monthly arrays are shaped `(n_years, 12, n_cells)`; the coefficients
/// `(12, n_cells)`, one row per calendar month.
pub struct CafecInputs<'a> {
    pub precips: ArrayView3<'a, f64>,
    pub pet: ArrayView3<'a, f64>,
    pub prdat: ArrayView3<'a, f64>,
    pub spdat: ArrayView3<'a, f64>,
    pub pldat: ArrayView3<'a, f64>,
    pub alpha: ArrayView2<'a, f64>,
    pub beta: ArrayView2<'a, f64>,
    pub gamma: ArrayView2<'a, f64>,
    pub delta: ArrayView2<'a, f64>,
}

impl CafecInputs<'_> {
    fn validate(&self) -> Result<(), ClimateError> {
        let monthly = self.precips.dim();
        for (argument, values) in [
            ("pet", &self.pet),
            ("prdat", &self.prdat),
            ("spdat", &self.spdat),
            ("pldat", &self.pldat),
        ] {
            if values.dim() != monthly {
                return Err(ClimateError::ShapeMismatch {
                    argument,
                    expected: self.precips.len(),
                    actual: values.len(),
                });
            }
        }
        for (argument, values) in [
            ("alpha", &self.alpha),
            ("beta", &self.beta),
            ("gamma", &self.gamma),
            ("delta", &self.delta),
        ] {
            check_coefficients(argument, values, monthly)?;
        }
        Ok(())
    }

    /// CAFEC precipitation for one month, one cell.
    ///
    /// Python: the `phat`/`cafec` expression shared by
    /// `palmer._calc_k_prime_and_dbar` and `palmer._calc_cafec_zindex`, summed
    /// left to right as NumPy evaluates it.
    fn cafec(&self, year: usize, month: usize, cell: usize) -> f64 {
        self.alpha[[month, cell]] * self.pet[[year, month, cell]]
            + self.beta[[month, cell]] * self.prdat[[year, month, cell]]
            + self.gamma[[month, cell]] * self.spdat[[year, month, cell]]
            - self.delta[[month, cell]] * self.pldat[[year, month, cell]]
    }
}

/// Reject a `(12, n_cells)` coefficient array that does not match a monthly block.
fn check_coefficients(
    argument: &'static str,
    values: &ArrayView2<'_, f64>,
    (_, months, n_cells): (usize, usize, usize),
) -> Result<(), ClimateError> {
    if values.dim() != (months, n_cells) {
        return Err(ClimateError::ShapeMismatch {
            argument,
            expected: months * n_cells,
            actual: values.len(),
        });
    }
    Ok(())
}

/// Monthly mean absolute moisture departure (`dbar`) and the raw K-prime factors.
///
/// - Python source: `palmer._calc_k_prime_and_dbar`.
/// - Inputs: the water balance and CAFEC coefficients in `inputs`; `trat`, the
///   `(12, n_cells)` moisture-demand ratio; the inclusive calibration year rows.
/// - Outputs: `(dbar, k_prime)`, each `(12, n_cells)`.
/// - Zero semantics: a zero `dbar` divides as NumPy does, giving an infinite
///   or NaN K-prime that Python's callers reject or carry, as they always have.
/// - NaN semantics: a NaN departure propagates into its month's sum.
/// - Invalid input: a calibration row outside the record is an error, which the
///   Python period resolution makes unreachable.
/// - Numerics: departures accumulate sequentially over the calibration years;
///   `dbar = sum / n_calibration_years` and
///   `k_prime = 1.5 * log10((trat + 2.8) / dbar) + 0.5`.
pub fn k_prime_and_dbar(
    inputs: &CafecInputs<'_>,
    trat: ArrayView2<'_, f64>,
    calibration_year_initial_idx: usize,
    calibration_year_final_idx: usize,
) -> Result<(Array2<f64>, Array2<f64>), ClimateError> {
    inputs.validate()?;
    let (n_years, months, n_cells) = inputs.precips.dim();
    check_coefficients("trat", &trat, (n_years, months, n_cells))?;
    if calibration_year_initial_idx > calibration_year_final_idx
        || calibration_year_final_idx >= n_years
    {
        return Err(ClimateError::IndexOutOfRange {
            argument: "calibration_year_final_idx",
            value: calibration_year_final_idx as i64,
            minimum: calibration_year_initial_idx as i64,
            maximum: n_years as i64 - 1,
        });
    }

    let mut sabsd = Array2::<f64>::zeros((months, n_cells));
    for year in calibration_year_initial_idx..=calibration_year_final_idx {
        for month in 0..months {
            for cell in 0..n_cells {
                let phat = inputs.cafec(year, month, cell);
                sabsd[[month, cell]] += (inputs.precips[[year, month, cell]] - phat).abs();
            }
        }
    }

    let n_calb_years = (calibration_year_final_idx - calibration_year_initial_idx + 1) as f64;
    let dbar = sabsd.mapv(|sum| sum / n_calb_years);
    let mut k_prime = Array2::<f64>::zeros((months, n_cells));
    ndarray::Zip::from(&mut k_prime)
        .and(&dbar)
        .and(&trat)
        .for_each(|k, &d, &t| *k = 1.5 * ((t + 2.8) / d).log10() + 0.5);
    Ok((dbar, k_prime))
}

/// The K-weighted Z-index for every month of the record.
///
/// - Python source: `palmer._calc_raw_zindex` and the `_calc_cafec_zindex` it
///   calls once per month.
/// - Inputs: the water balance and CAFEC coefficients in `inputs`; `ak`, the
///   `(12, n_cells)` K factors (normalized for PDSI, raw K-prime for scPDSI).
/// - Outputs: `z`, shaped `(n_years, 12, n_cells)`.
/// - Zero and NaN semantics: none of its own; NaN inputs propagate.
/// - Numerics: `z = ak * (precips - cafec)`, with the CAFEC sum evaluated
///   left to right.
pub fn raw_zindex(
    inputs: &CafecInputs<'_>,
    ak: ArrayView2<'_, f64>,
) -> Result<Array3<f64>, ClimateError> {
    inputs.validate()?;
    let dim = inputs.precips.dim();
    check_coefficients("ak", &ak, dim)?;
    Ok(Array3::from_shape_fn(dim, |(year, month, cell)| {
        let cafec = inputs.cafec(year, month, cell);
        ak[[month, cell]] * (inputs.precips[[year, month, cell]] - cafec)
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn inputs<'a>(
        monthly: &'a Array3<f64>,
        coefficients: &'a Array2<f64>,
        delta: &'a Array2<f64>,
    ) -> CafecInputs<'a> {
        CafecInputs {
            precips: monthly.view(),
            pet: monthly.view(),
            prdat: monthly.view(),
            spdat: monthly.view(),
            pldat: monthly.view(),
            alpha: coefficients.view(),
            beta: coefficients.view(),
            gamma: coefficients.view(),
            delta: delta.view(),
        }
    }

    #[test]
    fn zindex_weights_the_cafec_departure() {
        // every term 2.0, alpha=beta=gamma=0.5, delta=0.25: cafec = 1+1+1-0.5 = 2.5
        let monthly = Array3::<f64>::from_elem((1, 1, 1), 2.0);
        let coefficients = array![[0.5]];
        let delta = array![[0.25]];
        let ak = array![[2.0]];
        let z = raw_zindex(&inputs(&monthly, &coefficients, &delta), ak.view()).unwrap();
        assert_eq!(z[[0, 0, 0]], 2.0 * (2.0 - 2.5));
    }

    #[test]
    fn k_prime_averages_only_the_calibration_years() {
        // year 0 departs by 0.5, year 1 by 1.5 (precips 4.0 vs the 2.5 cafec)
        let monthly = Array3::<f64>::from_elem((2, 1, 1), 2.0);
        let coefficients = array![[0.5]];
        let delta = array![[0.25]];
        let trat = array![[1.2]];
        let mut precips = monthly.clone();
        precips[[1, 0, 0]] = 4.0;
        let mut cafec = inputs(&monthly, &coefficients, &delta);
        cafec.precips = precips.view();
        let (dbar, k_prime) = k_prime_and_dbar(&cafec, trat.view(), 1, 1).unwrap();
        assert_eq!(dbar[[0, 0]], 1.5);
        assert_eq!(k_prime[[0, 0]], 1.5 * (4.0_f64 / 1.5).log10() + 0.5);
    }

    #[test]
    fn rejects_a_calibration_row_past_the_record() {
        let monthly = Array3::<f64>::from_elem((1, 1, 1), 2.0);
        let coefficients = array![[0.5]];
        let trat = array![[1.0]];
        let error = k_prime_and_dbar(
            &inputs(&monthly, &coefficients, &coefficients),
            trat.view(),
            0,
            1,
        )
        .unwrap_err();
        assert!(matches!(error, ClimateError::IndexOutOfRange { .. }));
    }
}
