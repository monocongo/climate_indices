//! NCEI `pdi.f`-lineage Palmer spell recursion behind standard PDSI, PHDI, and PMDI.
//!
//! Reference: Palmer (1965), "Meteorological Drought", U.S. Weather Bureau
//! Research Paper 45, as implemented by NCEI's `pdi.f` (statements 170-220 and
//! the backtracking that resolves a deferred spell).
//!
//! The Python implementation in `climate_indices._palmer_pdi` stays the
//! reference oracle; `tests/test_native_parity_palmer.py` checks this kernel
//! against it at `rtol = atol = 1e-10`. Python vectorizes the state machine
//! across cells with masks that partition every cell into exactly one branch,
//! so each cell's recursion is independent; this kernel runs it one cell at a
//! time with the same branch order and exact comparisons. Duration-factor
//! validation and the infinite-Z check stay in Python.

use ndarray::{Array2, ArrayView2};

use crate::palmer::{py_max, py_min};

/// Duration factors the `pdi.f` recurrence reads, already validated by
/// `PdiDurationFactors.from_fitted`.
#[derive(Debug, Clone, Copy)]
pub struct PdiDurationFactors {
    pub wetm: f64,
    pub wetb: f64,
    pub drym: f64,
    pub dryb: f64,
}

impl PdiDurationFactors {
    /// The wet or dry `(m, b)` for an `x3` sign; zero keeps the wet factors.
    fn select(&self, x3: f64) -> (f64, f64) {
        if x3 >= 0.0 {
            (self.wetm, self.wetb)
        } else {
            (self.drym, self.dryb)
        }
    }
}

/// `pdi.f`'s weighted carry `b / (m + b) * previous + z / (m + b)`.
///
/// Python: `_weighting_fraction(m, b) * previous + z / (m + b)`; the division
/// form is load-bearing, since the recursion branches on exact comparisons.
fn carry(m: f64, b: f64, previous: f64, z: f64) -> f64 {
    b / (m + b) * previous + z / (m + b)
}

/// The final PDSI, PHDI, and PMDI series.
#[derive(Debug)]
pub struct PdiResult {
    pub pdsi: Array2<f64>,
    pub phdi: Array2<f64>,
    pub pmdi: Array2<f64>,
}

/// The preliminary (near-real-time) PDSI, PMDI. Python: `_palmer_pdi._case`.
fn case(prob: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    let near_normal = if x1.abs() > x2.abs() { x1 } else { x2 };
    let pro = prob / 100.0;
    let interpolated = if x3 <= 0.0 {
        (1.0 - pro) * x3 + pro * x1
    } else {
        (1.0 - pro) * x3 + pro * x2
    };
    let established = if prob <= 0.0 || prob >= 100.0 {
        x3
    } else {
        interpolated
    };
    if x3 == 0.0 { near_normal } else { established }
}

/// One cell's recursion state; Python: `_palmer_pdi._State` restricted to a cell.
///
/// The per-month arrays are indexed by the flat month `t = year * 12 + month`,
/// and the K8 window's `index` holds the flat month Python keeps as the
/// `indexj`/`indexm` pair. Python's `ud`/`uw` are written but never read, so
/// they are not carried.
struct Cell<'a> {
    factors: PdiDurationFactors,
    z: &'a [f64],
    t: usize,
    ppr: Vec<f64>,
    px1: Vec<f64>,
    px2: Vec<f64>,
    px3: Vec<f64>,
    x: Vec<f64>,
    pdsi: Vec<f64>,
    phdi: Vec<f64>,
    wplm: Vec<f64>,
    index: Vec<usize>,
    sx: Vec<f64>,
    sx1: Vec<f64>,
    sx2: Vec<f64>,
    sx3: Vec<f64>,
    k8: usize,
    k8max: usize,
    iass: u8,
    v: f64,
    pro: f64,
    x1: f64,
    x2: f64,
    x3: f64,
    ze: f64,
    pv: f64,
}

impl<'a> Cell<'a> {
    fn new(z: &'a [f64], factors: PdiDurationFactors) -> Self {
        let n = z.len();
        Self {
            factors,
            z,
            t: 0,
            ppr: vec![0.0; n],
            px1: vec![0.0; n],
            px2: vec![0.0; n],
            px3: vec![0.0; n],
            x: vec![0.0; n],
            pdsi: vec![f64::NAN; n],
            phdi: vec![f64::NAN; n],
            wplm: vec![f64::NAN; n],
            index: vec![0; n],
            sx: vec![0.0; n],
            sx1: vec![0.0; n],
            sx2: vec![0.0; n],
            sx3: vec![0.0; n],
            k8: 0,
            k8max: 0,
            iass: 0,
            v: 0.0,
            pro: 0.0,
            x1: 0.0,
            x2: 0.0,
            x3: 0.0,
            ze: 0.0,
            pv: 0.0,
        }
    }

    /// Python: `_record_index_values` for one entry.
    fn record(&mut self, t: usize, value: f64) {
        let px3 = self.px3[t];
        self.pdsi[t] = value;
        // no established spell (px3 exactly 0.0): PHDI falls back to the PDSI value
        self.phdi[t] = if px3 == 0.0 { value } else { px3 };
        self.wplm[t] = case(self.ppr[t], self.px1[t], self.px2[t], px3);
    }

    /// Python: `_backtrack_assigned_values`.
    fn backtrack(&mut self) {
        let mut isave = self.iass;
        for i in (0..self.k8).rev() {
            if isave == 2 {
                if self.sx2[i] == 0.0 {
                    isave = 1;
                    self.sx[i] = self.sx1[i];
                } else {
                    isave = 2;
                    self.sx[i] = self.sx2[i];
                }
            } else if self.sx1[i] == 0.0 {
                isave = 2;
                self.sx[i] = self.sx2[i];
            } else {
                isave = 1;
                self.sx[i] = self.sx1[i];
            }
        }
    }

    /// Python: `_flush_spells`.
    fn flush(&mut self) {
        if self.iass == 3 {
            for i in 0..self.k8 {
                self.sx[i] = self.sx3[i];
            }
        } else {
            self.backtrack();
        }
        for i in 0..=self.k8 {
            self.record(self.index[i], self.sx[i]);
        }
    }

    /// Python: `_assign`.
    fn assign(&mut self) {
        let t = self.t;
        self.sx[self.k8] = self.x[t];
        if self.k8 == 0 {
            self.record(t, self.x[t]);
        } else {
            self.flush();
        }
        self.k8 = 0;
    }

    /// Save this month's variables for next month. Python: `_statement_220`.
    fn statement_220(&mut self) {
        let t = self.t;
        self.v = self.pv;
        self.pro = self.ppr[t];
        self.x1 = self.px1[t];
        self.x2 = self.px2[t];
        self.x3 = self.px3[t];
    }

    /// Prob(end) returns to 0; accept the x3 value. Python: `_statement_210`.
    fn statement_210(&mut self) {
        let t = self.t;
        self.pv = 0.0;
        self.px1[t] = 0.0;
        self.px2[t] = 0.0;
        self.ppr[t] = 0.0;
        let (m, b) = self.factors.select(self.x3);
        self.px3[t] = carry(m, b, self.x3, self.z[t]);
        self.x[t] = self.px3[t];
        self.iass = 3;
        self.assign();
        self.statement_220();
    }

    /// Continue x1/x2 and promote a new x3. Python: `_statement_200`.
    fn statement_200(&mut self) {
        let t = self.t;
        let PdiDurationFactors {
            wetm,
            wetb,
            drym,
            dryb,
        } = self.factors;
        let px1 = carry(wetm, wetb, self.x1, self.z[t]);
        self.px1[t] = if px1 > 0.0 { px1 } else { 0.0 };

        // px3/px1/px2 exactly 0.0 mean no established spell / no incipient index
        let branch1 = self.px1[t] >= 1.0 && self.px3[t] == 0.0;
        if branch1 {
            self.px3[t] = self.px1[t];
            self.x[t] = self.px1[t];
            self.px1[t] = 0.0;
            self.iass = 1;
        } else {
            let px2 = carry(drym, dryb, self.x2, self.z[t]);
            self.px2[t] = if px2 < 0.0 { px2 } else { 0.0 };
        }

        let branch2 = !branch1 && self.px2[t] <= -1.0 && self.px3[t] == 0.0;
        if branch2 {
            self.px3[t] = self.px2[t];
            self.x[t] = self.px2[t];
            self.px2[t] = 0.0;
            self.iass = 2;
        }

        let px3_still_zero = !(branch1 || branch2) && self.px3[t] == 0.0;
        let branch3 = px3_still_zero && self.px1[t] == 0.0;
        if branch3 {
            self.x[t] = self.px2[t];
            self.iass = 2;
        }
        let branch4 = px3_still_zero && !branch3 && self.px2[t] == 0.0;
        if branch4 {
            self.x[t] = self.px1[t];
            self.iass = 1;
        }

        if branch1 || branch2 || branch3 || branch4 {
            self.assign();
        } else {
            // no determined value yet: save x1, x2, and x3 until x3 resolves the spell
            let k8 = self.k8;
            self.sx1[k8] = self.px1[t];
            self.sx2[k8] = self.px2[t];
            self.sx3[k8] = self.px3[t];
            self.x[t] = self.px3[t];
            self.k8 += 1;
            self.k8max = self.k8;
        }
        self.statement_220();
    }

    /// A spell continues; calculate prob(end). Python: `_statement_190`.
    fn statement_190(&mut self) {
        let t = self.t;
        // pro is 100.0 exactly where ppr was clamped to that endpoint
        let q = if self.pro == 100.0 {
            self.ze
        } else {
            self.ze + self.v
        };
        let ppr = (self.pv / q) * 100.0;
        let (m, b) = self.factors.select(self.x3);
        let px3 = carry(m, b, self.x3, self.z[t]);
        if ppr >= 100.0 {
            self.ppr[t] = 100.0;
            self.px3[t] = 0.0;
        } else {
            self.ppr[t] = ppr;
            self.px3[t] = px3;
        }
        self.statement_200();
    }

    /// Drought abatement is possible. Python: `_statement_180`.
    fn statement_180(&mut self) {
        let z = self.z[self.t];
        self.pv = (z + 0.15) + py_max(self.v, 0.0);
        // during a drought, PV <= 0 implies prob(end) has returned to 0
        if self.pv <= 0.0 {
            self.statement_210();
        } else {
            let PdiDurationFactors { drym, dryb, .. } = self.factors;
            self.ze = -dryb * self.x3 - 0.5 * (drym + dryb);
            self.statement_190();
        }
    }

    /// Wet-spell abatement is possible. Python: `_statement_170`.
    fn statement_170(&mut self) {
        let z = self.z[self.t];
        self.pv = (z - 0.15) + py_min(self.v, 0.0);
        // during a wet spell, PV >= 0 implies prob(end) has returned to 0
        if self.pv >= 0.0 {
            self.statement_210();
        } else {
            let PdiDurationFactors { wetm, wetb, .. } = self.factors;
            self.ze = -wetb * self.x3 + 0.5 * (wetm + wetb);
            self.statement_190();
        }
    }

    /// Python: `_advance_month`, including its NaN-`x3` fallthrough to statement 170.
    fn advance_month(&mut self, t: usize) {
        self.t = t;
        self.index[self.k8] = t;
        self.ze = 0.0;
        let z = self.z[t];
        let x3 = self.x3;
        // pro takes its endpoints exactly; either endpoint means an established spell
        if self.pro == 100.0 || self.pro == 0.0 {
            if (-0.5..=0.5).contains(&x3) {
                // end of drought or wet: check for a new spell
                self.pv = 0.0;
                self.ppr[t] = 0.0;
                self.px3[t] = 0.0;
                self.statement_200();
            } else if x3 > 0.5 {
                if z >= 0.15 {
                    self.statement_210();
                } else {
                    self.statement_170();
                }
            } else if x3 < -0.5 {
                if z <= -0.15 {
                    self.statement_210();
                } else {
                    self.statement_180();
                }
            } else {
                // a NaN x3 satisfies no comparison
                self.statement_170();
            }
        } else if x3 > 0.0 || x3.is_nan() {
            self.statement_170();
        } else {
            self.statement_180();
        }
    }

    /// Flush a spell still open when the record ends. Python: `_finish_up`.
    ///
    /// Reads `k8max`, not `k8`, exactly as Python does.
    fn finish_up(&mut self) {
        if self.k8max == 0 {
            return;
        }
        let last = self.z.len() - 1;
        let final_wplm = case(
            self.ppr[last],
            self.px1[last],
            self.px2[last],
            self.px3[last],
        );
        for k in 0..self.k8max {
            let t = self.index[k];
            let x = self.x[t];
            let px3 = self.px3[t];
            self.pdsi[t] = x;
            self.phdi[t] = if px3 == 0.0 { x } else { px3 };
            self.wplm[t] = final_wplm;
        }
    }
}

/// The `pdi.f` Palmer recursion over a precomputed Z-index series.
///
/// - Python source: `_palmer_pdi.calculate` (statements 170-220,
///   `_advance_month`, `_assign`, `_flush_spells`, `_backtrack_assigned_values`,
///   `_finish_up`).
/// - Inputs: `z`, shaped `(n_months, n_cells)` (Python's `(years, 12, n_cells)`
///   with the calendar axes flattened); `factors`, already validated.
/// - Outputs: PDSI, PHDI, and PMDI shaped like `z`.
/// - Zero semantics: `px1`/`px2`/`px3`/`sx1`/`sx2` exactly `0.0` mark "no
///   index", and `pro` exactly `0.0`/`100.0` marks an established spell, as in
///   Python; no tolerance is applied.
/// - NaN semantics: a NaN Z propagates through the same arithmetic, and a NaN
///   `x3` takes Python's fallthrough to statement 170.
/// - Invalid input: infinite Z values are rejected by Python before dispatch.
/// - Numerics: every expression keeps the Python operation order; the
///   weighting fraction is the division `b / (m + b)`.
pub fn calculate(z: ArrayView2<'_, f64>, factors: PdiDurationFactors) -> PdiResult {
    let (n_months, n_cells) = z.dim();
    let mut result = PdiResult {
        pdsi: Array2::<f64>::zeros((n_months, n_cells)),
        phdi: Array2::<f64>::zeros((n_months, n_cells)),
        pmdi: Array2::<f64>::zeros((n_months, n_cells)),
    };
    for cell in 0..n_cells {
        let series = z.column(cell).to_vec();
        let mut state = Cell::new(&series, factors);
        for t in 0..n_months {
            state.advance_month(t);
        }
        state.finish_up();
        result
            .pdsi
            .column_mut(cell)
            .assign(&ndarray::ArrayView1::from(&state.pdsi));
        result
            .phdi
            .column_mut(cell)
            .assign(&ndarray::ArrayView1::from(&state.phdi));
        result
            .pmdi
            .column_mut(cell)
            .assign(&ndarray::ArrayView1::from(&state.wplm));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn palmer_defaults() -> PdiDurationFactors {
        // Palmer (1965) p = 0.897, q = 1/3
        let m = (1.0 - 0.897) / (1.0 / 3.0);
        let b = 0.897 / (1.0 / 3.0);
        PdiDurationFactors {
            wetm: m,
            wetb: b,
            drym: m,
            dryb: b,
        }
    }

    #[test]
    fn a_wet_month_starts_a_spell_at_x1() {
        let factors = palmer_defaults();
        let z = Array2::from_elem((1, 1), 4.0);
        let result = calculate(z.view(), factors);
        // px1 = z / (m + b) = 4 / 3 >= 1 starts the wet spell at once
        let expected = 4.0 / (factors.wetm + factors.wetb);
        assert_eq!(result.pdsi[[0, 0]], expected);
        assert_eq!(result.phdi[[0, 0]], expected);
    }

    #[test]
    fn an_abating_spell_left_open_is_resolved_by_finish_up() {
        let factors = palmer_defaults();
        // month 0 starts a wet spell; month 1's dry Z defers it with x3 still set
        let z = ndarray::array![[4.0], [-1.0]];
        let result = calculate(z.view(), factors);
        let x3 = carry(factors.wetm, factors.wetb, result.pdsi[[0, 0]], -1.0);
        assert_eq!(result.pdsi[[1, 0]], x3);
        assert_eq!(result.phdi[[1, 0]], x3);
        // the final month's PMDI interpolates toward the incipient dry index
        assert!(result.pmdi[[1, 0]] < x3);
    }

    #[test]
    fn cells_recurse_independently() {
        let factors = palmer_defaults();
        let z = ndarray::array![[4.0, -4.0], [0.5, -0.5], [f64::NAN, 2.0]];
        let block = calculate(z.view(), factors);
        for cell in 0..2 {
            let single = calculate(z.slice(ndarray::s![.., cell..cell + 1]), factors);
            for t in 0..3 {
                for (a, b) in [
                    (block.pdsi[[t, cell]], single.pdsi[[t, 0]]),
                    (block.phdi[[t, cell]], single.phdi[[t, 0]]),
                    (block.pmdi[[t, cell]], single.pmdi[[t, 0]]),
                ] {
                    assert!(a == b || (a.is_nan() && b.is_nan()));
                }
            }
        }
    }
}
