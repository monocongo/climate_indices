//! Palmer-family water-balance kernel.
//!
//! Reference: Palmer (1965), "Meteorological Drought", U.S. Weather Bureau
//! Research Paper 45, and the algorithmic lineage the Python package carries for
//! PDSI/scPDSI. The Python implementation in `climate_indices.palmer` stays the
//! reference oracle; `tests/test_native_parity_palmer.py` checks this kernel
//! against it at `rtol = atol = 1e-10`. The spell recursions, CAFEC/Z-index
//! stages, validation, calibration-period resolution, warnings, logging, and
//! fitting-parameter handling stay in Python; this kernel takes numbers Python
//! already prepared and returns numbers.

use ndarray::{Array2, Array3, ArrayView1, ArrayView3};

use crate::ClimateError;

/// Surface-layer available water capacity, in inches (`palmer.AWCTOP`).
pub const AWCTOP: f64 = 1.0;

/// Python's builtin `max(a, b)`: `b` only if `b > a`, so NaN order matters.
pub(crate) fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's builtin `min(a, b)`: `b` only if `b < a`.
pub(crate) fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// Available water capacity of the under layer, in inches.
///
/// Python: `palmer._get_awc_bot`, `max(awc - AWCTOP, 0.0)` through the
/// builtin-order `_py_max`.
fn awc_bottom(awc: f64) -> f64 {
    py_max(awc - AWCTOP, 0.0)
}

/// Potential loss for one month, one cell.
///
/// Python: `palmer._calc_potential_loss`.
fn potential_loss(pet: f64, ss: f64, su: f64, awc: f64) -> f64 {
    let awc_bot = awc_bottom(awc);
    let candidate = py_min(ss + su, ((pet - ss) * su) / (awc_bot + AWCTOP) + ss);
    if ss >= pet { pet } else { candidate }
}

/// The recharge/runoff/loss outputs `_calc_recharge` returns for one month, one cell.
struct Recharge {
    et: f64,
    tl: f64,
    r: f64,
    ro: f64,
    sss: f64,
    ssu: f64,
}

/// Recharge, runoff, residual moisture, and loss to both layers for one month, one cell.
///
/// Python: `palmer._calc_recharge`, reproducing its elementwise branch order
/// exactly. Both arms are evaluated and selected with the same comparisons, so a
/// single location and a spatial cell take the same path.
fn recharge(p: f64, pet: f64, ss: f64, su: f64, awc: f64) -> Recharge {
    let awc_bot = awc_bottom(awc);

    // precipitation exceeds potential evaporation
    let excess = p - pet;
    let rs = AWCTOP - ss;
    let both_layers_take_it_all = (excess - rs) < (awc_bot - su);
    let ru = if both_layers_take_it_all {
        excess - rs
    } else {
        awc_bot - su
    };
    let mut ro_recharge_case = if both_layers_take_it_all {
        0.0
    } else {
        excess - rs - ru
    };
    let under_recharged = excess > (AWCTOP - ss);
    let r_recharge_case = if under_recharged { rs + ru } else { excess };
    let sss_recharge_case = if under_recharged { AWCTOP } else { ss + excess };
    let ssu_recharge_case = if under_recharged { su + ru } else { su };
    if !under_recharged {
        ro_recharge_case = 0.0;
    }
    let et_recharge_case = pet;
    let tl_recharge_case = 0.0;

    // evaporation exceeds precipitation
    let deficit = pet - p;
    let ul_both = py_min(su, (deficit - ss) * su / awc);
    let surface_only = ss >= deficit;
    let sl = if surface_only { deficit } else { ss };
    let ul = if surface_only { 0.0 } else { ul_both };
    let sss_evap_case = if surface_only { ss - sl } else { 0.0 };
    let ssu_evap_case = if surface_only { su } else { su - ul };
    let tl_evap_case = sl + ul;
    let et_evap_case = p + sl + ul;
    let r_evap_case = 0.0;
    let ro_evap_case = 0.0;

    let precip_exceeds = p >= pet;
    Recharge {
        et: if precip_exceeds {
            et_recharge_case
        } else {
            et_evap_case
        },
        tl: if precip_exceeds {
            tl_recharge_case
        } else {
            tl_evap_case
        },
        r: if precip_exceeds {
            r_recharge_case
        } else {
            r_evap_case
        },
        ro: if precip_exceeds {
            ro_recharge_case
        } else {
            ro_evap_case
        },
        sss: if precip_exceeds {
            sss_recharge_case
        } else {
            sss_evap_case
        },
        ssu: if precip_exceeds {
            ssu_recharge_case
        } else {
            ssu_evap_case
        },
    }
}

/// Every water-balance array and calibration-period monthly sum.
///
/// The monthly arrays are shaped `(n_years, 12, n_cells)`; the sums are
/// `(12, n_cells)`, one row per calendar month.
#[derive(Debug)]
pub struct WaterBalance {
    pub spdat: Array3<f64>,
    pub pldat: Array3<f64>,
    pub prdat: Array3<f64>,
    pub rdat: Array3<f64>,
    pub tldat: Array3<f64>,
    pub etdat: Array3<f64>,
    pub rodat: Array3<f64>,
    pub sssdat: Array3<f64>,
    pub ssudat: Array3<f64>,
    pub psum: Array2<f64>,
    pub spsum: Array2<f64>,
    pub petsum: Array2<f64>,
    pub plsum: Array2<f64>,
    pub prsum: Array2<f64>,
    pub rsum: Array2<f64>,
    pub tlsum: Array2<f64>,
    pub etsum: Array2<f64>,
    pub rosum: Array2<f64>,
}

/// The Palmer water balance over a `(n_years, 12, n_cells)` record.
///
/// - Python source: `palmer._calc_water_balances`, including its per-month
///   [`potential_loss`] and [`recharge`] calls and its calibration-period sum
///   accumulation.
/// - Inputs: `precips` and `pet` shaped `(n_years, 12, n_cells)`, already folded
///   to a calendar-year block with a trailing cell axis; `awc` the total
///   available water capacity per cell, in inches.
/// - Outputs: the nine monthly arrays and their calibration-period monthly sums.
/// - Zero and NaN semantics: every branch keeps NumPy's comparison order, so a
///   NaN precipitation or PET propagates through the same expressions the Python
///   path evaluates, and an exact-zero denominator (`awc == 0`) reaches the same
///   `inf`/`NaN` arithmetic.
pub fn water_balance(
    precips: ArrayView3<'_, f64>,
    pet: ArrayView3<'_, f64>,
    awc: ArrayView1<'_, f64>,
    calibration_year_initial_idx: usize,
    calibration_year_final_idx: usize,
) -> Result<WaterBalance, ClimateError> {
    let (n_years, months, n_cells) = precips.dim();
    if months != 12 {
        return Err(ClimateError::ShapeMismatch {
            argument: "months",
            expected: 12,
            actual: months,
        });
    }
    if pet.dim() != precips.dim() {
        return Err(ClimateError::ShapeMismatch {
            argument: "pet",
            expected: precips.len(),
            actual: pet.len(),
        });
    }
    if awc.len() != n_cells {
        return Err(ClimateError::ShapeMismatch {
            argument: "awc",
            expected: n_cells,
            actual: awc.len(),
        });
    }

    let mut out = WaterBalance {
        spdat: Array3::<f64>::zeros((n_years, months, n_cells)),
        pldat: Array3::<f64>::zeros((n_years, months, n_cells)),
        prdat: Array3::<f64>::zeros((n_years, months, n_cells)),
        rdat: Array3::<f64>::zeros((n_years, months, n_cells)),
        tldat: Array3::<f64>::zeros((n_years, months, n_cells)),
        etdat: Array3::<f64>::zeros((n_years, months, n_cells)),
        rodat: Array3::<f64>::zeros((n_years, months, n_cells)),
        sssdat: Array3::<f64>::zeros((n_years, months, n_cells)),
        ssudat: Array3::<f64>::zeros((n_years, months, n_cells)),
        psum: Array2::<f64>::zeros((12, n_cells)),
        spsum: Array2::<f64>::zeros((12, n_cells)),
        petsum: Array2::<f64>::zeros((12, n_cells)),
        plsum: Array2::<f64>::zeros((12, n_cells)),
        prsum: Array2::<f64>::zeros((12, n_cells)),
        rsum: Array2::<f64>::zeros((12, n_cells)),
        tlsum: Array2::<f64>::zeros((12, n_cells)),
        etsum: Array2::<f64>::zeros((12, n_cells)),
        rosum: Array2::<f64>::zeros((12, n_cells)),
    };

    for cell in 0..n_cells {
        let awc_cell = awc[cell];
        let awc_bot = awc_bottom(awc_cell);
        let mut ss = AWCTOP;
        let mut su = awc_bot;
        for year in 0..n_years {
            for month in 0..months {
                let p = precips[[year, month, cell]];
                let pet_value = pet[[year, month, cell]];
                let sp = ss + su;
                let pr = awc_bot + AWCTOP - sp;

                let pl = potential_loss(pet_value, ss, su, awc_cell);
                let outcomes = recharge(p, pet_value, ss, su, awc_cell);

                let in_calibration =
                    year >= calibration_year_initial_idx && year <= calibration_year_final_idx;
                if in_calibration {
                    out.psum[[month, cell]] += p;
                    out.spsum[[month, cell]] += sp;
                    out.petsum[[month, cell]] += pet_value;
                    out.plsum[[month, cell]] += pl;
                    out.prsum[[month, cell]] += pr;
                    out.rsum[[month, cell]] += outcomes.r;
                    out.tlsum[[month, cell]] += outcomes.tl;
                    out.etsum[[month, cell]] += outcomes.et;
                    out.rosum[[month, cell]] += outcomes.ro;
                }

                out.spdat[[year, month, cell]] = sp;
                out.pldat[[year, month, cell]] = pl;
                out.prdat[[year, month, cell]] = pr;
                out.rdat[[year, month, cell]] = outcomes.r;
                out.tldat[[year, month, cell]] = outcomes.tl;
                out.etdat[[year, month, cell]] = outcomes.et;
                out.rodat[[year, month, cell]] = outcomes.ro;
                out.sssdat[[year, month, cell]] = outcomes.sss;
                out.ssudat[[year, month, cell]] = outcomes.ssu;

                ss = outcomes.sss;
                su = outcomes.ssu;
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    /// A wet month (precipitation at or above PET) with both layers already full.
    #[test]
    fn recharge_case_matches_the_python_branches() {
        // awc 4.0 -> awc_bot 3.0; ss 1.0, su 3.0; p 2.0, pet 0.5
        let precips = Array3::from_elem((1, 12, 1), 2.0);
        let pet = Array3::from_elem((1, 12, 1), 0.5);
        let awc = array![4.0];
        let out = water_balance(precips.view(), pet.view(), awc.view(), 0, 0).unwrap();

        assert_eq!(out.spdat[[0, 0, 0]], 4.0);
        assert_eq!(out.pldat[[0, 0, 0]], 0.5);
        assert_eq!(out.prdat[[0, 0, 0]], 0.0);
        assert_eq!(out.rdat[[0, 0, 0]], 0.0);
        assert_eq!(out.tldat[[0, 0, 0]], 0.0);
        assert_eq!(out.etdat[[0, 0, 0]], 0.5);
        assert_eq!(out.rodat[[0, 0, 0]], 1.5);
        assert_eq!(out.sssdat[[0, 0, 0]], 1.0);
        assert_eq!(out.ssudat[[0, 0, 0]], 3.0);
        assert_eq!(out.psum[[0, 0]], 2.0);
        assert_eq!(out.etsum[[0, 0]], 0.5);
        assert_eq!(out.rosum[[0, 0]], 1.5);
    }

    /// A dry month (PET above precipitation): surface layer empties first.
    #[test]
    fn evaporation_case_matches_the_python_branches() {
        // awc 4.0 -> awc_bot 3.0; ss 1.0, su 3.0; p 0.5, pet 2.0
        let precips = Array3::from_elem((1, 12, 1), 0.5);
        let pet = Array3::from_elem((1, 12, 1), 2.0);
        let awc = array![4.0];
        let out = water_balance(precips.view(), pet.view(), awc.view(), 0, 0).unwrap();

        assert_eq!(out.spdat[[0, 0, 0]], 4.0);
        assert_eq!(out.pldat[[0, 0, 0]], 1.75);
        assert_eq!(out.prdat[[0, 0, 0]], 0.0);
        assert_eq!(out.rdat[[0, 0, 0]], 0.0);
        assert_eq!(out.tldat[[0, 0, 0]], 1.375);
        assert_eq!(out.etdat[[0, 0, 0]], 1.875);
        assert_eq!(out.rodat[[0, 0, 0]], 0.0);
        assert_eq!(out.sssdat[[0, 0, 0]], 0.0);
        assert_eq!(out.ssudat[[0, 0, 0]], 2.625);
    }

    /// The sums only accrue over the inclusive calibration years, while the
    /// monthly arrays cover every year.
    #[test]
    fn sums_are_limited_to_the_calibration_years() {
        let mut precips = Array3::<f64>::zeros((3, 12, 1));
        let mut pet = Array3::<f64>::zeros((3, 12, 1));
        for month in 0..12 {
            precips[[0, month, 0]] = 1.0;
            precips[[1, month, 0]] = 2.0;
            precips[[2, month, 0]] = 3.0;
            pet[[0, month, 0]] = 0.5;
            pet[[1, month, 0]] = 0.5;
            pet[[2, month, 0]] = 0.5;
        }
        let awc = array![4.0];
        let out = water_balance(precips.view(), pet.view(), awc.view(), 1, 2).unwrap();

        // years 1 and 2 only: 2.0 + 3.0
        assert_eq!(out.psum[[0, 0]], 5.0);
        // the monthly arrays keep all three years
        assert_eq!(out.spdat[[0, 0, 0]], 4.0);
        assert_eq!(out.spdat[[2, 0, 0]], 4.0);
    }

    #[test]
    fn awc_bottom_never_goes_negative() {
        assert_eq!(awc_bottom(0.5), 0.0);
        assert_eq!(awc_bottom(4.0), 3.0);
    }

    #[test]
    fn rejects_non_monthly_shapes() {
        for months in [0, 1, 11, 13] {
            let values = Array3::zeros((1, months, 1));
            let awc = array![4.0];
            let error = water_balance(values.view(), values.view(), awc.view(), 0, 0).unwrap_err();
            assert_eq!(
                error,
                ClimateError::ShapeMismatch {
                    argument: "months",
                    expected: 12,
                    actual: months,
                }
            );
        }
    }

    #[test]
    fn rejects_a_mismatched_awc_length() {
        let precips = Array3::from_elem((1, 12, 2), 1.0);
        let pet = Array3::from_elem((1, 12, 2), 0.5);
        let awc = Array1::from(vec![4.0]);
        let error = water_balance(precips.view(), pet.view(), awc.view(), 0, 0).unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "awc",
                expected: 2,
                actual: 1
            }
        );
    }
}
