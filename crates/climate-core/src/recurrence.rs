//! The daily-recurrence engine behind the fire kernels.
//!
//! A port of the shared day loop in `climate_indices._recurrence`
//! (`run_daily_recurrences` and `_apply_gap_policy`), which is the reference for
//! every semantic here: the ADR-0007 missing-day policy, the ADR-0010 seasonal
//! carry mask, the spin-up offset, and the recorded-history NaNs. State comes in
//! and goes out per cell, so a kernel owns the whole time axis and Python keeps
//! the orchestration around it (validation, calendars, warnings, logging).
//!
//! The Python implementation stays the parity oracle; see `docs/architecture.md`.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::ClimateError;

/// How a started recurrence treats a missing observation (ADR-0007).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingDayPolicy {
    /// One missing observation poisons the cell for the rest of the run.
    Propagate,
    /// Up to `max_gap_days` consecutive missing days are bridged.
    Bridge { max_gap_days: i64 },
}

/// The prepared, time-first inputs one recurrence reads.
#[derive(Debug)]
pub struct RecurrenceInputs<'a> {
    /// `(days, cells)`: whether each day holds an observation for this code.
    pub weather_valid: ArrayView2<'a, bool>,
    /// Per cell: whether the static input (climatology, latitude) is usable.
    /// An unusable cell never starts and is not an elapsed missing day.
    pub static_valid: ArrayView1<'a, bool>,
    /// `(days, cells)`: the days inside the fire season, or `None` when the
    /// recurrence has no seasonal shutdown (ADR-0010).
    pub in_season: Option<ArrayView2<'a, bool>>,
    /// Per cell: the gap count the run starts from, `-1` when never started.
    pub trailing_gap_days: ArrayView1<'a, i64>,
    /// Leading days to compute but omit from the recorded history.
    pub spin_up: usize,
    pub policy: MissingDayPolicy,
}

/// One recurrence's recorded history and final per-cell bookkeeping.
#[derive(Debug)]
pub struct RecurrenceRun<S> {
    /// `(days - spin_up, cells)`, or `None` when the caller records no history.
    pub values: Option<Array2<f64>>,
    /// Final per-cell state.
    pub state: Vec<S>,
    /// Final per-cell gap counts, or `None` when no cell ever started.
    pub trailing_gap_days: Option<Array1<i64>>,
}

/// Run one daily recurrence over the whole time axis.
///
/// * `initial_state` - one state per cell; a state whose value is NaN is a
///   poisoned cell, exactly like `_initialize_component`.
/// * `record` - whether to allocate and return the daily history.
/// * `value_of` - the code value of a state, the number the history records and
///   the finiteness check reads.
/// * `poison` - turn a state into the poisoned one (value NaN); the Python
///   driver assigns `np.nan` to the code value and leaves the rest of the state.
/// * `step` - advance one active cell by one day.
///
/// A step that returns a non-finite value from finite inputs is an error, as in
/// `_advance_component`; the driver returns before assigning any of that day's
/// results, so the caller's state is whatever the previous days left.
pub fn run<S: Copy>(
    index_type: &'static str,
    initial_state: &[S],
    inputs: &RecurrenceInputs<'_>,
    record: bool,
    value_of: impl Fn(&S) -> f64,
    poison: impl Fn(&mut S),
    mut step: impl FnMut(S, usize, usize) -> S,
) -> Result<RecurrenceRun<S>, ClimateError> {
    let (days, cells) = inputs.weather_valid.dim();
    let expect = |argument: &'static str, length: usize| {
        if length == cells {
            Ok(())
        } else {
            Err(ClimateError::ShapeMismatch {
                argument,
                expected: cells,
                actual: length,
            })
        }
    };
    expect("initial_state", initial_state.len())?;
    expect("static_valid", inputs.static_valid.len())?;
    expect("trailing_gap_days", inputs.trailing_gap_days.len())?;
    if let Some(in_season) = inputs.in_season.as_ref()
        && in_season.dim() != (days, cells)
    {
        return Err(ClimateError::ShapeMismatch {
            argument: "in_season",
            expected: days * cells,
            actual: in_season.len(),
        });
    }

    let mut state = initial_state.to_vec();
    let mut gaps: Vec<i64> = inputs.trailing_gap_days.iter().copied().collect();
    let mut started: Vec<bool> = gaps.iter().map(|&gap| gap >= 0).collect();
    let mut poisoned: Vec<bool> = state.iter().map(|cell| value_of(cell).is_nan()).collect();
    let mut values = if record {
        Some(Array2::<f64>::from_elem(
            (days.saturating_sub(inputs.spin_up), cells),
            f64::NAN,
        ))
    } else {
        None
    };

    let mut active = vec![false; cells];
    let mut updates: Vec<(usize, S)> = Vec::with_capacity(cells);
    for day in 0..days {
        let season_day = inputs.in_season.as_ref().map(|mask| mask.row(day));
        for cell in 0..cells {
            // an off-season day is neither an observation nor a missing day: it
            // never advances the recurrence and never counts against the allowance
            let in_season = season_day.is_none_or(|season| season[cell]);
            let observed =
                inputs.static_valid[cell] && in_season && inputs.weather_valid[[day, cell]];
            if observed {
                // a valid day is the return point's last day, so any run of gaps
                // closes here, whether or not a missing day poisoned the cell
                gaps[cell] = 0;
                active[cell] = !poisoned[cell];
                if active[cell] {
                    started[cell] = true;
                }
                continue;
            }
            active[cell] = false;
            // a cell whose static input is unusable has no recurrence to
            // gap-manage: it never starts, so it is not an elapsed missing day
            if !inputs.static_valid[cell] || !in_season || !(started[cell] || poisoned[cell]) {
                continue;
            }
            match inputs.policy {
                MissingDayPolicy::Propagate => {
                    poison(&mut state[cell]);
                    poisoned[cell] = true;
                    gaps[cell] = gaps[cell].max(0) + 1;
                }
                MissingDayPolicy::Bridge { max_gap_days } => {
                    let next_gap_days = gaps[cell].max(0) + 1;
                    gaps[cell] = next_gap_days;
                    if next_gap_days > max_gap_days {
                        poison(&mut state[cell]);
                        poisoned[cell] = true;
                    }
                }
            }
        }

        updates.clear();
        for cell in 0..cells {
            if !active[cell] {
                continue;
            }
            let updated = step(state[cell], day, cell);
            if !value_of(&updated).is_finite() {
                return Err(ClimateError::NonFinite { index_type });
            }
            updates.push((cell, updated));
        }
        for &(cell, updated) in &updates {
            state[cell] = updated;
        }

        if let Some(values) = values.as_mut()
            && day >= inputs.spin_up
        {
            let row = day - inputs.spin_up;
            for cell in 0..cells {
                // an off-season cell emits the carried state instead of a gap NaN
                let carried = season_day.is_some_and(|season| !season[cell])
                    && inputs.static_valid[cell]
                    && started[cell];
                values[[row, cell]] = if active[cell] || carried {
                    value_of(&state[cell])
                } else {
                    f64::NAN
                };
            }
        }
    }

    let trailing_gap_days = started
        .iter()
        .zip(&poisoned)
        .any(|(&started, &poisoned)| started || poisoned)
        .then(|| Array1::from(gaps));
    Ok(RecurrenceRun {
        values,
        state,
        trailing_gap_days,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    /// The three cells' static validity, which outlives every test input.
    const STATIC_VALID: [bool; 3] = [true, true, true];

    /// A three-cell recurrence whose middle cell starts poisoned (a NaN state).
    fn inputs<'a>(weather: &'a Array2<bool>, gaps: &'a Array1<i64>) -> RecurrenceInputs<'a> {
        RecurrenceInputs {
            weather_valid: weather.view(),
            static_valid: ArrayView1::from(&STATIC_VALID),
            in_season: None,
            trailing_gap_days: gaps.view(),
            spin_up: 0,
            policy: MissingDayPolicy::Propagate,
        }
    }

    fn increment(state: f64, _day: usize, _cell: usize) -> f64 {
        state + 1.0
    }

    fn run_increment(
        weather: &Array2<bool>,
        gaps: &Array1<i64>,
        record: bool,
    ) -> Result<RecurrenceRun<f64>, ClimateError> {
        run(
            "test",
            &[0.0, f64::NAN, 0.0],
            &inputs(weather, gaps),
            record,
            |state| *state,
            |state| *state = f64::NAN,
            increment,
        )
    }

    /// One column's entries, with a NaN recorded as None so it can be compared.
    fn column(values: &Array2<f64>, column: usize) -> Vec<Option<f64>> {
        values
            .column(column)
            .iter()
            .map(|value| (!value.is_nan()).then_some(*value))
            .collect()
    }

    /// The values a cell reaches on the days it is active, in order.
    fn advancing(values: &Array2<f64>, cell: usize) -> Vec<f64> {
        column(values, cell).into_iter().flatten().collect()
    }

    #[test]
    fn a_valid_day_advances_starts_and_records_every_cell() {
        let weather = Array2::from_elem((5, 3), true);
        let run = run_increment(&weather, &array![-1, -1, -1], true).unwrap();
        let values = run.values.unwrap();
        assert_eq!(advancing(&values, 0), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(advancing(&values, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        // the middle cell starts from the NaN state it was handed, so it is
        // poisoned and never advances
        assert!(values.column(1).iter().all(|value| value.is_nan()));
        assert_eq!(run.state[0], 5.0);
        assert!(run.state[1].is_nan());
        assert_eq!(run.state[2], 5.0);
        assert_eq!(run.trailing_gap_days.unwrap(), array![0, 0, 0]);
    }

    #[test]
    fn a_missing_day_poisons_a_started_cell_and_records_nan() {
        let mut weather = Array2::from_elem((4, 3), true);
        weather[[2, 0]] = false;
        let run = run_increment(&weather, &array![-1, -1, -1], true).unwrap();
        // cell 0 advances on days 0 and 1, its missing day 2 poisons it, and the
        // last valid day records the carried NaN rather than resuming
        let values = run.values.unwrap();
        assert_eq!(column(&values, 0), vec![Some(1.0), Some(2.0), None, None]);
        assert!(run.state[0].is_nan());
        // a valid day still closes the gap count, as in the Python driver
        assert_eq!(run.trailing_gap_days.unwrap(), array![0, 0, 0]);
    }

    #[test]
    fn a_missing_day_before_any_observation_is_not_an_elapsed_gap() {
        let mut weather = Array2::from_elem((3, 3), true);
        weather[[0, 0]] = false;
        let run = run_increment(&weather, &array![-1, -1, -1], true).unwrap();
        // day 0 never started cell 0, so it stays unstarted and records nothing,
        // and the two valid days then run it from zero
        let values = run.values.unwrap();
        assert_eq!(column(&values, 0), vec![None, Some(1.0), Some(2.0)]);
        assert_eq!(run.trailing_gap_days.unwrap()[0], 0);
    }

    #[test]
    fn bridging_tolerates_the_allowance_and_poisons_past_it() {
        let mut weather = Array2::from_elem((5, 3), true);
        weather[[1, 0]] = false;
        weather[[1, 1]] = false;
        weather[[2, 1]] = false;
        weather[[3, 1]] = false;
        let bridged = run(
            "test",
            &[0.0, 0.0, 0.0],
            &RecurrenceInputs {
                policy: MissingDayPolicy::Bridge { max_gap_days: 2 },
                ..inputs(&weather, &array![-1, -1, -1])
            },
            true,
            |state| *state,
            |state| *state = f64::NAN,
            increment,
        )
        .unwrap();
        // cell 0's single missing day is inside the allowance: it records a NaN
        // day and resumes; cell 1's three exceed it and poison the cell
        let values = bridged.values.unwrap();
        assert_eq!(
            column(&values, 0),
            vec![Some(1.0), None, Some(2.0), Some(3.0), Some(4.0)]
        );
        assert!(values.column(1).iter().skip(1).all(|value| value.is_nan()));
        assert!(bridged.state[1].is_nan());
        // the last valid day closes cell 1's gap count, as in the Python driver
        assert_eq!(bridged.trailing_gap_days.unwrap(), array![0, 0, 0]);
    }

    #[test]
    fn an_unusable_static_input_never_starts_the_cell() {
        let weather = Array2::from_elem((3, 3), true);
        let gaps = array![-1, -1, -1];
        let static_valid = array![true, false, true];
        let run = run(
            "test",
            &[0.0, 0.0, 0.0],
            &RecurrenceInputs {
                static_valid: static_valid.view(),
                ..inputs(&weather, &gaps)
            },
            true,
            |state| *state,
            |state| *state = f64::NAN,
            increment,
        )
        .unwrap();
        let values = run.values.unwrap();
        assert!(values.column(1).iter().all(|value| value.is_nan()));
        assert_eq!(run.state[1], 0.0);
    }

    #[test]
    fn an_off_season_day_carries_the_state_without_gap_managing_it() {
        let weather = Array2::from_elem((4, 3), true);
        // the even days are in season, the odd ones are not
        let in_season = Array2::from_shape_vec(
            (4, 3),
            [
                true, true, true, false, false, false, true, true, true, false, false, false,
            ]
            .to_vec(),
        )
        .unwrap();
        let run = run(
            "test",
            &[0.0, 0.0, 0.0],
            &RecurrenceInputs {
                in_season: Some(in_season.view()),
                ..inputs(&weather, &array![-1, -1, -1])
            },
            true,
            |state| *state,
            |state| *state = f64::NAN,
            increment,
        )
        .unwrap();
        // an off-season day emits the carried value instead of a gap NaN, and
        // never counts as a missing day
        let values = run.values.unwrap();
        assert_eq!(
            column(&values, 0),
            vec![Some(1.0), Some(1.0), Some(2.0), Some(2.0)]
        );
        assert_eq!(run.trailing_gap_days.unwrap(), array![0, 0, 0]);
    }

    #[test]
    fn spin_up_days_are_computed_but_not_recorded() {
        let weather = Array2::from_elem((4, 3), true);
        let run = run(
            "test",
            &[0.0, 0.0, 0.0],
            &RecurrenceInputs {
                spin_up: 2,
                ..inputs(&weather, &array![-1, -1, -1])
            },
            true,
            |state| *state,
            |state| *state = f64::NAN,
            increment,
        )
        .unwrap();
        // the two spin-up days advanced the state without being recorded
        let values = run.values.unwrap();
        assert_eq!(column(&values, 0), vec![Some(3.0), Some(4.0)]);
        assert_eq!(run.state[0], 4.0);
    }

    #[test]
    fn recording_can_be_skipped_while_the_state_still_advances() {
        let weather = Array2::from_elem((3, 3), true);
        let run = run_increment(&weather, &array![-1, -1, -1], false).unwrap();
        assert!(run.values.is_none());
        assert_eq!(run.state[0], 3.0);
        assert!(run.state[1].is_nan());
        assert_eq!(run.state[2], 3.0);
    }

    #[test]
    fn a_nan_state_the_caller_resumed_stays_poisoned() {
        // a NaN value with a started gap count is the state of a cell whose gap
        // the caller did not close, and it never advances
        let weather = Array2::from_elem((2, 3), true);
        let run = run_increment(&weather, &array![-1, 5, -1], true).unwrap();
        let values = run.values.unwrap();
        assert!(values.column(1).iter().all(|value| value.is_nan()));
        assert!(run.state[1].is_nan());
    }

    #[test]
    fn a_non_finite_step_result_is_an_error() {
        let weather = Array2::from_elem((2, 3), true);
        let error = run(
            "kbdi",
            &[0.0, 0.0, 0.0],
            &inputs(&weather, &array![-1, -1, -1]),
            true,
            |state| *state,
            |state| *state = f64::NAN,
            |_state, _day, _cell| f64::INFINITY,
        )
        .unwrap_err();
        assert_eq!(error, ClimateError::NonFinite { index_type: "kbdi" });
    }

    #[test]
    fn a_parameter_of_the_wrong_length_is_rejected() {
        let weather = Array2::from_elem((2, 3), true);
        let error = run(
            "test",
            &[0.0, 0.0],
            &inputs(&weather, &array![-1, -1, -1]),
            true,
            |state| *state,
            |state| *state = f64::NAN,
            increment,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "initial_state",
                expected: 3,
                actual: 2
            }
        );
    }

    #[test]
    fn a_seasonal_mask_of_the_wrong_shape_is_rejected() {
        let weather = Array2::from_elem((2, 3), true);
        let in_season = Array2::from_elem((3, 3), true);
        let error = run(
            "test",
            &[0.0, 0.0, 0.0],
            &RecurrenceInputs {
                in_season: Some(in_season.view()),
                ..inputs(&weather, &array![-1, -1, -1])
            },
            true,
            |state| *state,
            |state| *state = f64::NAN,
            increment,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "in_season",
                expected: 6,
                actual: 9
            }
        );
    }
}
