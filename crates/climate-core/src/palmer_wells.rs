//! Wells-lineage Palmer recursion behind self-calibrating PDSI.
//!
//! Reference: Wells, Goddard, and Hayes (2004), "A Self-Calibrating Palmer
//! Drought Severity Index", J. Climate 17(12), 2335-2351.
//!
//! The Python implementation in `climate_indices._palmer_wells` stays the
//! reference oracle; `tests/test_native_parity_palmer.py` checks this kernel
//! against it at `rtol = atol = 1e-10`. Duration-factor validation and the
//! derivation of the recurrence coefficients (`_palmer_duration.DurationFactors`)
//! and the infinite-Z check stay in Python; this kernel takes the coefficients
//! Python derived.

use ndarray::{Array1, ArrayView1};

use crate::ClimateError;
use crate::palmer::{py_max, py_min};

const TOLERANCE: f64 = 1e-5;
const SPELL_THRESHOLD: f64 = 0.5;

/// Validated duration factors and the Wells recurrence coefficients Python derived.
///
/// Python: `_palmer_duration.DurationFactors`.
#[derive(Debug, Clone, Copy)]
pub struct WellsFactors {
    pub wetm: f64,
    pub wetb: f64,
    pub drym: f64,
    pub dryb: f64,
    pub wet_denominator: f64,
    pub dry_denominator: f64,
    pub wetc: f64,
    pub dryc: f64,
    pub dry_spell_c: f64,
}

/// The final scPDSI, scPHDI, and scPMDI series.
#[derive(Debug)]
pub struct WellsResult {
    pub pdsi: Array1<f64>,
    pub phdi: Array1<f64>,
    pub pmdi: Array1<f64>,
}

/// State carried between non-missing periods. Python: `_palmer_wells._State`.
#[derive(Debug, Clone, Copy, Default)]
struct State {
    x1: f64,
    x2: f64,
    x3: f64,
    v: f64,
    probability: f64,
}

/// One period's candidate state. Python: `_palmer_wells._Transition`.
#[derive(Debug, Clone, Copy, Default)]
struct Transition {
    x1: f64,
    x2: f64,
    x3: f64,
    v: f64,
    probability: f64,
    selection: u8,
    spell_terminated: bool,
}

/// A deferred period. Python: `_palmer_wells._Tentative`.
#[derive(Debug, Clone, Copy)]
struct Tentative {
    index: usize,
    x1: f64,
    x2: f64,
    x3: f64,
}

/// Python: `_palmer_wells._candidate_values`, with its builtin `max`/`min`.
fn candidate_values(state: &State, z: f64, factors: &WellsFactors) -> (f64, f64) {
    let x1 = py_max(0.0, factors.wetc * state.x1 + z / factors.wet_denominator);
    let x2 = py_min(0.0, factors.dryc * state.x2 + z / factors.dry_denominator);
    (x1, x2)
}

/// Resolve tentative periods. Python: `_palmer_wells._backtrack`.
fn backtrack(pdsi: &mut Array1<f64>, tentative: &mut Vec<Tentative>, mut selection: u8) {
    if selection == 3 {
        for values in tentative.iter() {
            pdsi[values.index] = values.x3;
        }
        tentative.clear();
        return;
    }
    for values in tentative.iter().rev() {
        if selection == 2 {
            if values.x2 == 0.0 {
                selection = 1;
                pdsi[values.index] = values.x1;
            } else {
                pdsi[values.index] = values.x2;
            }
        } else if values.x1 == 0.0 {
            selection = 2;
            pdsi[values.index] = values.x2;
        } else {
            pdsi[values.index] = values.x1;
        }
    }
    tentative.clear();
}

/// Python: `_palmer_wells._pmdi`.
fn pmdi(probability: f64, x1: f64, x2: f64, x3: f64) -> f64 {
    if x3 == 0.0 {
        return if x1.abs() > x2.abs() { x1 } else { x2 };
    }
    if probability <= 0.0 || probability >= 100.0 {
        return x3;
    }
    let fraction = probability / 100.0;
    if x3 <= 0.0 {
        (1.0 - fraction) * x3 + fraction * x1
    } else {
        (1.0 - fraction) * x3 + fraction * x2
    }
}

/// Advance an established spell while abatement is possible.
///
/// Python: `_palmer_wells._abatement_transition`.
fn abatement_transition(
    state: &State,
    z: f64,
    factors: &WellsFactors,
    transition: Transition,
    wet_spell: bool,
) -> Result<Transition, ClimateError> {
    let (coefficient, denominator, slope, own_intercept, direction) = if wet_spell {
        (
            factors.wetc,
            factors.wet_denominator,
            factors.wetm,
            factors.wetb,
            1.0,
        )
    } else {
        (
            factors.dry_spell_c,
            factors.dry_denominator,
            factors.drym,
            factors.dryb,
            -1.0,
        )
    };
    let abatement_z = slope * SPELL_THRESHOLD;
    let carry = if wet_spell {
        py_min(state.v + TOLERANCE, 0.0)
    } else {
        py_max(state.v - TOLERANCE, 0.0)
    };
    let new_v = z - direction * abatement_z + carry;

    if direction * new_v >= 0.0 {
        return Ok(Transition {
            x3: coefficient * state.x3 + z / denominator,
            selection: 3,
            ..Transition::default()
        });
    }

    let ze = direction * SPELL_THRESHOLD * (slope + own_intercept) - own_intercept * state.x3;
    let q = if state.probability >= 100.0 - TOLERANCE {
        ze
    } else {
        ze + state.v
    };
    if q == 0.0 || !q.is_finite() {
        return Err(ClimateError::NoConvergence {
            message: "Wells recursion could not calculate an abatement probability",
        });
    }

    let new_probability = (new_v / q) * 100.0;
    if new_probability >= 100.0 - TOLERANCE {
        return Ok(Transition {
            x1: transition.x1,
            x2: transition.x2,
            v: new_v,
            probability: 100.0,
            spell_terminated: true,
            ..Transition::default()
        });
    }
    Ok(Transition {
        x1: transition.x1,
        x2: transition.x2,
        x3: coefficient * state.x3 + z / denominator,
        v: new_v,
        probability: new_probability,
        ..Transition::default()
    })
}

/// Continue or abate an established spell. Python: `_palmer_wells._continue_spell`.
fn continue_spell(
    state: &State,
    z: f64,
    factors: &WellsFactors,
    transition: Transition,
) -> Result<Transition, ClimateError> {
    if state.x3 == 0.0 {
        return Ok(transition);
    }
    let wet_spell = state.x3 >= 0.0;
    let (coefficient, denominator, slope, direction) = if wet_spell {
        (factors.wetc, factors.wet_denominator, factors.wetm, 1.0)
    } else {
        (
            factors.dry_spell_c,
            factors.dry_denominator,
            factors.drym,
            -1.0,
        )
    };
    // half the fitted slope replaces Palmer's fixed 0.15 effective-moisture threshold
    let abatement_z = slope * SPELL_THRESHOLD;
    let abatement_underway = !(state.probability == 0.0 || state.probability == 100.0);
    if !abatement_underway && direction * z >= abatement_z {
        return Ok(Transition {
            x3: coefficient * state.x3 + z / denominator,
            selection: 3,
            ..Transition::default()
        });
    }
    abatement_transition(state, z, factors, transition, wet_spell)
}

/// Candidate selection and spell-establishment tie-breaks.
///
/// Python: `_palmer_wells._establish_spell`. Each new transition is built field
/// by field, as Python builds it, so an `x3` of `-0.0` becomes `0.0` where
/// Python's default does.
fn establish_spell(transition: Transition) -> Transition {
    if transition.x3 != 0.0 {
        return transition;
    }
    let terminated = transition.spell_terminated;
    let (v, probability) = if terminated {
        (transition.v, transition.probability)
    } else {
        (0.0, 0.0)
    };
    if transition.x1 >= SPELL_THRESHOLD {
        return Transition {
            x1: 0.0,
            x2: transition.x2,
            x3: transition.x1,
            v,
            probability,
            selection: 1,
            spell_terminated: terminated,
        };
    }
    if transition.x2 <= -SPELL_THRESHOLD {
        return Transition {
            x1: transition.x1,
            x2: 0.0,
            x3: transition.x2,
            v,
            probability,
            selection: 2,
            spell_terminated: terminated,
        };
    }
    let selection = if transition.x1 == 0.0 {
        2
    } else if transition.x2 == 0.0 {
        1
    } else {
        return transition;
    };
    Transition {
        x1: transition.x1,
        x2: transition.x2,
        x3: 0.0,
        v: transition.v,
        probability: transition.probability,
        selection,
        spell_terminated: terminated,
    }
}

/// Assign or defer one period's PDSI. Python: `_palmer_wells._assign_period`.
fn assign_period(
    pdsi: &mut Array1<f64>,
    tentative: &mut Vec<Tentative>,
    index: usize,
    transition: &Transition,
) {
    if transition.selection == 0 {
        pdsi[index] = transition.x3;
        tentative.push(Tentative {
            index,
            x1: transition.x1,
            x2: transition.x2,
            x3: transition.x3,
        });
        return;
    }
    backtrack(pdsi, tentative, transition.selection);
    pdsi[index] = match transition.selection {
        1 if transition.x3 == 0.0 => transition.x1,
        2 if transition.x3 == 0.0 => transition.x2,
        _ => transition.x3,
    };
}

/// The Wells Palmer recursion over a Z-index series.
///
/// - Python source: `_palmer_wells.calculate`.
/// - Inputs: `z`, the chronological Z-index series (NaN missing); `factors`,
///   validated with coefficients derived by Python.
/// - Outputs: scPDSI, scPHDI, and scPMDI, each shaped like `z`.
/// - Zero semantics: exact `0.0` (Python's `_is_exact_zero`) selects and
///   backtracks candidates; no tolerance is applied.
/// - NaN semantics: a missing period is missing in every output and does not
///   advance the state.
/// - Invalid input: infinite Z values are rejected by Python before dispatch.
///   A zero or non-finite abatement denominator is
///   [`ClimateError::NoConvergence`] with Python's message.
/// - Numerics: every expression keeps the Python operation order, including
///   the `1e-5` tolerance adjustments.
pub fn calculate(
    z: ArrayView1<'_, f64>,
    factors: &WellsFactors,
) -> Result<WellsResult, ClimateError> {
    let size = z.len();
    let mut pdsi = Array1::<f64>::from_elem(size, f64::NAN);
    let mut x1_values = Array1::<f64>::from_elem(size, f64::NAN);
    let mut x2_values = Array1::<f64>::from_elem(size, f64::NAN);
    let mut x3_values = Array1::<f64>::from_elem(size, f64::NAN);
    let mut probability_values = Array1::<f64>::from_elem(size, f64::NAN);

    let mut state = State::default();
    let mut tentative = Vec::new();
    for (index, &z_value) in z.iter().enumerate() {
        if z_value.is_nan() {
            continue;
        }
        let (x1, x2) = candidate_values(&state, z_value, factors);
        let transition = Transition {
            x1,
            x2,
            ..Transition::default()
        };
        let transition = continue_spell(&state, z_value, factors, transition)?;
        let transition = establish_spell(transition);
        assign_period(&mut pdsi, &mut tentative, index, &transition);

        state = State {
            x1: transition.x1,
            x2: transition.x2,
            x3: transition.x3,
            v: transition.v,
            probability: transition.probability,
        };
        x1_values[index] = transition.x1;
        x2_values[index] = transition.x2;
        x3_values[index] = transition.x3;
        probability_values[index] = transition.probability;
    }

    let phdi = Array1::from_shape_fn(size, |i| {
        if pdsi[i].is_nan() {
            f64::NAN
        } else if x3_values[i] == 0.0 {
            pdsi[i]
        } else {
            x3_values[i]
        }
    });
    let pmdi = Array1::from_shape_fn(size, |i| {
        if z[i].is_nan() {
            f64::NAN
        } else {
            pmdi(
                probability_values[i],
                x1_values[i],
                x2_values[i],
                x3_values[i],
            )
        }
    });
    Ok(WellsResult { pdsi, phdi, pmdi })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn factors(m: f64, b: f64) -> WellsFactors {
        WellsFactors {
            wetm: m,
            wetb: b,
            drym: m,
            dryb: b,
            wet_denominator: m + b,
            dry_denominator: m + b,
            wetc: 1.0 - m / (m + b),
            dryc: 1.0 - m / (m + b),
            dry_spell_c: 1.0 - m / (m + b),
        }
    }

    #[test]
    fn missing_periods_stay_missing_and_do_not_advance_the_state() {
        let f = factors(0.3, 2.7);
        let with_gap = calculate(array![3.0, f64::NAN, 3.0].view(), &f).unwrap();
        let without = calculate(array![3.0, 3.0].view(), &f).unwrap();
        assert!(with_gap.pdsi[1].is_nan());
        assert!(with_gap.phdi[1].is_nan());
        assert!(with_gap.pmdi[1].is_nan());
        assert_eq!(with_gap.pdsi[2], without.pdsi[1]);
    }

    #[test]
    fn a_wet_candidate_above_the_threshold_establishes_the_spell() {
        let f = factors(0.3, 2.7);
        let result = calculate(array![3.0].view(), &f).unwrap();
        // x1 = z / (m + b) = 1.0 >= 0.5 becomes x3
        assert_eq!(result.pdsi[0], 1.0);
        assert_eq!(result.phdi[0], 1.0);
    }

    #[test]
    fn a_zero_abatement_denominator_does_not_converge() {
        // the established wet x3 = 1.0 makes ze = 0.5 * (m + b) - b * x3 = 0
        let f = factors(1.0, 1.0);
        let error = calculate(array![2.0, 0.0].view(), &f).unwrap_err();
        assert!(matches!(error, ClimateError::NoConvergence { .. }));
    }
}
