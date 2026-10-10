//! Fire-weather kernels: the KBDI and CFFWIS moisture-code daily recursions.
//!
//! Ports of `climate_indices.fire._kbdi` and
//! `climate_indices.fire._cffwis_codes`, which stay the parity oracles. Each
//! kernel runs one recurrence over the whole time axis, one cell at a time, and
//! returns the recorded history plus the final state and gap counts; the
//! missing-day policy, seasonal carry, and spin-up come from [`crate::recurrence`].
//!
//! The equations follow Van Wagner and Pickett (1985) as implemented by the
//! NRCan reference code (`cffdrs_r` and its Python port `cffdrs_py`) for the
//! moisture codes, and the corrected continuous Eq. 18 of Keetch and Byram
//! (1968) with Alexander's (1990) corrected 8.30 constant for KBDI. Everything
//! is evaluated in the source's operational units (km/h wind, mm rain, degrees
//! Celsius), and the operation order is the Python one: the recording paths in
//! `tests/test_native_parity_fire.py` compare the two at
//! `rtol = atol = 1e-10` with identical NaN positions.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::ClimateError;
use crate::recurrence::{RecurrenceInputs, run};

// The CFFWIS constants of `fire/_cffwis_codes.py`. The published FFMC equations
// print 147.2 for the moisture-content conversion; the reference code uses the
// exact 250 * 59.5 / 101 in both directions, and this kernel matches it.
const FFMC_COEFFICIENT: f64 = 250.0 * 59.5 / 101.0;
const FFMC_MAXIMUM: f64 = 101.0;
const FFMC_MOISTURE_CAP: f64 = 250.0;
const FFMC_PRECIPITATION_THRESHOLD_MM: f64 = 0.5;
const FFMC_MOISTURE_FOR_RAIN_CORRECTION: f64 = 150.0;

const DMC_PRECIPITATION_THRESHOLD_MM: f64 = 1.5;
const DMC_TEMPERATURE_FLOOR_CELSIUS: f64 = -1.1;

const DC_PRECIPITATION_THRESHOLD_MM: f64 = 2.8;
const DC_TEMPERATURE_FLOOR_CELSIUS: f64 = -2.8;

// The KBDI constants of `fire/_kbdi.py`.
const KBDI_MAX_MM: f64 = 203.2;
const KBDI_RAIN_THRESHOLD_MM: f64 = 5.08;
const KBDI_DRYING_TEMPERATURE_CELSIUS: f64 = 10.0;

/// `np.maximum`, which propagates NaN instead of dropping it.
fn maximum(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left >= right {
        left
    } else {
        right
    }
}

/// `np.minimum`, which propagates NaN instead of dropping it.
fn minimum(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left <= right {
        left
    } else {
        right
    }
}

/// Advance the FFMC one day (Van Wagner and Pickett, 1985, Eq. 1-10).
fn ffmc_next(
    previous: f64,
    temperature_celsius: f64,
    relative_humidity_percent: f64,
    wind_speed_kilometers_per_hour: f64,
    precipitation_mm: f64,
) -> f64 {
    // Eq. 1: previous FFMC to fine fuel moisture content, percent
    let mut moisture = FFMC_COEFFICIENT * (101.0 - previous) / (59.5 + previous);
    let rained = precipitation_mm > FFMC_PRECIPITATION_THRESHOLD_MM;
    let effective_rain = if rained {
        precipitation_mm - FFMC_PRECIPITATION_THRESHOLD_MM
    } else {
        precipitation_mm
    };
    // Eqs. 3a and 3b: rain adds moisture, with an amendment above 150 percent
    let mut rain_moisture = 42.5
        * effective_rain
        * (-100.0 / (251.0 - moisture)).exp()
        * (1.0 - (-6.93 / effective_rain).exp());
    rain_moisture += if moisture > FFMC_MOISTURE_FOR_RAIN_CORRECTION {
        0.0015 * (moisture - FFMC_MOISTURE_FOR_RAIN_CORRECTION).powi(2) * effective_rain.sqrt()
    } else {
        0.0
    };
    moisture = if rained {
        minimum(moisture + rain_moisture, FFMC_MOISTURE_CAP)
    } else {
        moisture
    };

    // Eqs. 4 and 5: equilibrium moisture content for drying and wetting
    let temperature_term =
        0.18 * (21.1 - temperature_celsius) * (1.0 - (-0.115 * relative_humidity_percent).exp());
    let drying_equilibrium = 0.942 * relative_humidity_percent.powf(0.679)
        + 11.0 * ((relative_humidity_percent - 100.0) / 10.0).exp()
        + temperature_term;
    let wetting_equilibrium = 0.618 * relative_humidity_percent.powf(0.753)
        + 10.0 * ((relative_humidity_percent - 100.0) / 10.0).exp()
        + temperature_term;

    // Eqs. 6-9: dry toward the drying equilibrium or wet toward the wetting
    // equilibrium, whichever side of it the fuel is on
    let humidity_fraction = relative_humidity_percent / 100.0;
    let wind_root = wind_speed_kilometers_per_hour.sqrt();
    let temperature_scale = 0.581 * (0.0365 * temperature_celsius).exp();
    let drying_rate = (0.424 * (1.0 - humidity_fraction.powf(1.7))
        + 0.0694 * wind_root * (1.0 - humidity_fraction.powi(8)))
        * temperature_scale;
    let wetting_rate = (0.424 * (1.0 - (1.0 - humidity_fraction).powf(1.7))
        + 0.0694 * wind_root * (1.0 - (1.0 - humidity_fraction).powi(8)))
        * temperature_scale;
    let dried = drying_equilibrium + (moisture - drying_equilibrium) * 10.0_f64.powf(-drying_rate);
    let wetted =
        wetting_equilibrium - (wetting_equilibrium - moisture) * 10.0_f64.powf(-wetting_rate);
    moisture = if moisture > drying_equilibrium {
        dried
    } else if moisture < wetting_equilibrium {
        wetted
    } else {
        moisture
    };

    // Eq. 10: final FFMC conversion, clamped to the published range
    let result = 59.5 * (250.0 - moisture) / (FFMC_COEFFICIENT + moisture);
    minimum(maximum(result, 0.0), FFMC_MAXIMUM)
}

/// Advance the DMC one day (Van Wagner and Pickett, 1985, Eq. 11-16).
fn dmc_next(
    previous: f64,
    temperature_celsius: f64,
    relative_humidity_percent: f64,
    precipitation_mm: f64,
    effective_day_length_hours: f64,
) -> f64 {
    // Eq. 16: the log drying rate, with its temperature floor
    let temperature = maximum(temperature_celsius, DMC_TEMPERATURE_FLOOR_CELSIUS);
    let drying_rate = 1.894
        * (temperature + 1.1)
        * (100.0 - relative_humidity_percent)
        * effective_day_length_hours
        * 1e-4;

    // Eqs. 11-15: rain above 1.5 mm rewets the duff layer
    let rained = precipitation_mm > DMC_PRECIPITATION_THRESHOLD_MM;
    let effective_rain = 0.92 * precipitation_mm - 1.27;
    let moisture_before = 20.0 + 280.0 / (0.023 * previous).exp();
    // Eq. 13's piecewise slope of the moisture-content relation
    let slope = if previous <= 33.0 {
        100.0 / (0.5 + 0.3 * previous)
    } else if previous <= 65.0 {
        14.0 - 1.3 * previous.ln()
    } else {
        6.2 * previous.ln() - 17.2
    };
    let moisture_after =
        moisture_before + 1000.0 * effective_rain / (48.77 + slope * effective_rain);
    // Eq. 15 in the reference code's more accurate form
    let after_rain = maximum(43.43 * (5.6348 - (moisture_after - 20.0).ln()), 0.0);

    let previous = if rained { after_rain } else { previous };
    maximum(previous + drying_rate, 0.0)
}

/// DC to its layer moisture equivalent (Van Wagner and Pickett, 1985, Eq. 20).
fn dc_moisture_equivalent(dc: f64) -> f64 {
    800.0 * (-dc / 400.0).exp()
}

/// Advance the DC one day (Van Wagner and Pickett, 1985, Eq. 18-23).
fn dc_next(
    previous: f64,
    temperature_celsius: f64,
    precipitation_mm: f64,
    day_length_adjustment: f64,
) -> f64 {
    // Eq. 22: potential evapotranspiration, floored at zero for winter
    let temperature = maximum(temperature_celsius, DC_TEMPERATURE_FLOOR_CELSIUS);
    let potential_evapotranspiration = maximum(
        (0.36 * (temperature + 2.8) + day_length_adjustment) / 2.0,
        0.0,
    );

    // Eqs. 18-21: rain above 2.8 mm reduces the drought code
    let rained = precipitation_mm > DC_PRECIPITATION_THRESHOLD_MM;
    let effective_rain = 0.83 * precipitation_mm - 1.27;
    let moisture_before = dc_moisture_equivalent(previous);
    let after_rain = maximum(
        previous - 400.0 * (1.0 + 3.937 * effective_rain / moisture_before).ln(),
        0.0,
    );

    let previous = if rained { after_rain } else { previous };
    maximum(previous + potential_evapotranspiration, 0.0)
}

/// The KBDI state of one cell: the index and the wet spell it carries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KbdiCell {
    pub kbdi: f64,
    pub wet_spell_precipitation: f64,
}

/// Advance KBDI one day (corrected continuous Eq. 18 of Keetch and Byram, 1968).
///
/// Consecutive positive-rain days form one wet spell: only rain above its first
/// 5.08 mm reduces KBDI. Days below 10 C add no drought factor.
fn kbdi_next(
    previous: KbdiCell,
    precipitation_mm: f64,
    maximum_temperature_celsius: f64,
    mean_annual_precipitation_mm: f64,
) -> KbdiCell {
    let rained = precipitation_mm > 0.0;
    let event_total = previous.wet_spell_precipitation + precipitation_mm;
    let crossing_threshold = rained
        && previous.wet_spell_precipitation <= KBDI_RAIN_THRESHOLD_MM
        && event_total > KBDI_RAIN_THRESHOLD_MM;
    let continuing_wet_spell = rained && previous.wet_spell_precipitation > KBDI_RAIN_THRESHOLD_MM;
    let net_rain = if crossing_threshold {
        event_total - KBDI_RAIN_THRESHOLD_MM
    } else if continuing_wet_spell {
        precipitation_mm
    } else {
        0.0
    };
    let wet_spell_precipitation = if rained { event_total } else { 0.0 };

    let after_rain = maximum(0.0, previous.kbdi - net_rain);
    let drought_day =
        maximum_temperature_celsius >= KBDI_DRYING_TEMPERATURE_CELSIUS && after_rain < KBDI_MAX_MM;
    let drying = if drought_day {
        (KBDI_MAX_MM - after_rain)
            * (0.968 * (0.0875 * maximum_temperature_celsius + 1.5552).exp() - 8.30)
            / (1.0 + 10.88 * (-0.001736 * mean_annual_precipitation_mm).exp())
            * 1e-3
    } else {
        0.0
    };
    KbdiCell {
        kbdi: minimum(KBDI_MAX_MM, after_rain + maximum(drying, 0.0)),
        wet_spell_precipitation,
    }
}

/// The recorded history and final state of one moisture code.
#[derive(Debug)]
pub struct CodeRun {
    pub values: Option<Array2<f64>>,
    pub code: Array1<f64>,
    pub trailing_gap_days: Option<Array1<i64>>,
}

/// The recorded history and final state of a KBDI run.
#[derive(Debug)]
pub struct KbdiRun {
    pub values: Option<Array2<f64>>,
    pub state: Vec<KbdiCell>,
    pub trailing_gap_days: Option<Array1<i64>>,
}

/// Convert a whole-axis run of single-value cells into the code's result.
fn code_run(
    index_type: &'static str,
    initial: ArrayView1<'_, f64>,
    inputs: &RecurrenceInputs<'_>,
    record: bool,
    step: impl FnMut(f64, usize, usize) -> f64,
) -> Result<CodeRun, ClimateError> {
    let initial = initial.to_vec();
    let run = run(
        index_type,
        &initial,
        inputs,
        record,
        |value| *value,
        |value| *value = f64::NAN,
        step,
    )?;
    Ok(CodeRun {
        values: run.values,
        code: Array1::from(run.state),
        trailing_gap_days: run.trailing_gap_days,
    })
}

/// Run the Fine Fuel Moisture Code over the whole time axis.
pub fn ffmc(
    temperature_celsius: ArrayView2<'_, f64>,
    relative_humidity_percent: ArrayView2<'_, f64>,
    wind_speed_kilometers_per_hour: ArrayView2<'_, f64>,
    precipitation_mm: ArrayView2<'_, f64>,
    initial_ffmc: ArrayView1<'_, f64>,
    inputs: &RecurrenceInputs<'_>,
    record: bool,
) -> Result<CodeRun, ClimateError> {
    for (argument, array) in [
        ("temperature_celsius", &temperature_celsius),
        ("relative_humidity_percent", &relative_humidity_percent),
        (
            "wind_speed_kilometers_per_hour",
            &wind_speed_kilometers_per_hour,
        ),
        ("precipitation_mm", &precipitation_mm),
    ] {
        if array.dim() != inputs.weather_valid.dim() {
            return Err(ClimateError::ShapeMismatch {
                argument,
                expected: inputs.weather_valid.len(),
                actual: array.len(),
            });
        }
    }
    code_run(
        "ffmc",
        initial_ffmc,
        inputs,
        record,
        |previous, day, cell| {
            ffmc_next(
                previous,
                temperature_celsius[[day, cell]],
                relative_humidity_percent[[day, cell]],
                wind_speed_kilometers_per_hour[[day, cell]],
                precipitation_mm[[day, cell]],
            )
        },
    )
}

/// The day-length inputs a moisture code reads.
///
/// A port of the Python lookup `TABLE[band, month - 1]`: the latitude/month
/// table, the band each cell selects, and the calendar month of every day.
pub struct DayLength<'a> {
    pub table: ArrayView2<'a, f64>,
    pub band: ArrayView1<'a, i64>,
    pub months: ArrayView2<'a, i64>,
}

/// Run the Duff Moisture Code over the whole time axis.
pub fn dmc(
    temperature_celsius: ArrayView2<'_, f64>,
    relative_humidity_percent: ArrayView2<'_, f64>,
    precipitation_mm: ArrayView2<'_, f64>,
    day_length: &DayLength<'_>,
    initial_dmc: ArrayView1<'_, f64>,
    inputs: &RecurrenceInputs<'_>,
    record: bool,
) -> Result<CodeRun, ClimateError> {
    let (days, cells) = inputs.weather_valid.dim();
    for (argument, array) in [
        ("temperature_celsius", &temperature_celsius),
        ("relative_humidity_percent", &relative_humidity_percent),
        ("precipitation_mm", &precipitation_mm),
    ] {
        if array.dim() != (days, cells) {
            return Err(ClimateError::ShapeMismatch {
                argument,
                expected: days * cells,
                actual: array.len(),
            });
        }
    }
    check_day_length(day_length, days, cells)?;
    code_run(
        "duff_moisture_code",
        initial_dmc,
        inputs,
        record,
        |previous, day, cell| {
            dmc_next(
                previous,
                temperature_celsius[[day, cell]],
                relative_humidity_percent[[day, cell]],
                precipitation_mm[[day, cell]],
                day_length_lookup(day_length, day, cell),
            )
        },
    )
}

/// Run the Drought Code over the whole time axis.
pub fn dc(
    temperature_celsius: ArrayView2<'_, f64>,
    precipitation_mm: ArrayView2<'_, f64>,
    day_length: &DayLength<'_>,
    initial_dc: ArrayView1<'_, f64>,
    inputs: &RecurrenceInputs<'_>,
    record: bool,
) -> Result<CodeRun, ClimateError> {
    let (days, cells) = inputs.weather_valid.dim();
    for (argument, array) in [
        ("temperature_celsius", &temperature_celsius),
        ("precipitation_mm", &precipitation_mm),
    ] {
        if array.dim() != (days, cells) {
            return Err(ClimateError::ShapeMismatch {
                argument,
                expected: days * cells,
                actual: array.len(),
            });
        }
    }
    check_day_length(day_length, days, cells)?;
    code_run(
        "drought_code",
        initial_dc,
        inputs,
        record,
        |previous, day, cell| {
            dc_next(
                previous,
                temperature_celsius[[day, cell]],
                precipitation_mm[[day, cell]],
                day_length_lookup(day_length, day, cell),
            )
        },
    )
}

/// The day's day-length value for one cell: `TABLE[band, month - 1]`.
fn day_length_lookup(day_length: &DayLength<'_>, day: usize, cell: usize) -> f64 {
    day_length.table[[
        day_length.band[cell] as usize,
        day_length.months[[day, cell]] as usize - 1,
    ]]
}

/// Validate a day-length table, the band each cell selects, and the calendar
/// months of every day, so the lookup above needs no bounds check.
fn check_day_length(
    day_length: &DayLength<'_>,
    days: usize,
    cells: usize,
) -> Result<(), ClimateError> {
    if day_length.months.dim() != (days, cells) {
        return Err(ClimateError::ShapeMismatch {
            argument: "months",
            expected: days * cells,
            actual: day_length.months.len(),
        });
    }
    if day_length.band.len() != cells {
        return Err(ClimateError::ShapeMismatch {
            argument: "day_length_band",
            expected: cells,
            actual: day_length.band.len(),
        });
    }
    let bands = day_length.table.dim().0;
    if day_length.table.dim().1 != 12 {
        return Err(ClimateError::ShapeMismatch {
            argument: "day_length_table",
            expected: 12,
            actual: day_length.table.dim().1,
        });
    }
    for &row in day_length.band {
        if row < 0 || row as usize >= bands {
            return Err(ClimateError::IndexOutOfRange {
                argument: "day_length_band",
                value: row,
                minimum: 0,
                maximum: bands as i64 - 1,
            });
        }
    }
    for &month in day_length.months {
        if !(1..=12).contains(&month) {
            return Err(ClimateError::IndexOutOfRange {
                argument: "months",
                value: month,
                minimum: 1,
                maximum: 12,
            });
        }
    }
    Ok(())
}

/// Run KBDI over the whole time axis.
pub fn kbdi(
    precipitation_mm: ArrayView2<'_, f64>,
    maximum_temperature_celsius: ArrayView2<'_, f64>,
    mean_annual_precipitation_mm: ArrayView1<'_, f64>,
    initial_state: &[KbdiCell],
    inputs: &RecurrenceInputs<'_>,
    record: bool,
) -> Result<KbdiRun, ClimateError> {
    let (days, cells) = inputs.weather_valid.dim();
    for (argument, array) in [
        ("precipitation_mm", &precipitation_mm),
        ("maximum_temperature_celsius", &maximum_temperature_celsius),
    ] {
        if array.dim() != (days, cells) {
            return Err(ClimateError::ShapeMismatch {
                argument,
                expected: days * cells,
                actual: array.len(),
            });
        }
    }
    // the step indexes one climatology value per cell, so the caller's vector
    // must be exactly that long
    if mean_annual_precipitation_mm.len() != cells {
        return Err(ClimateError::ShapeMismatch {
            argument: "mean_annual_precipitation_mm",
            expected: cells,
            actual: mean_annual_precipitation_mm.len(),
        });
    }
    let run = run(
        "kbdi",
        initial_state,
        inputs,
        record,
        |cell| cell.kbdi,
        |cell| cell.kbdi = f64::NAN,
        |previous, day, cell| {
            kbdi_next(
                previous,
                precipitation_mm[[day, cell]],
                maximum_temperature_celsius[[day, cell]],
                mean_annual_precipitation_mm[cell],
            )
        },
    )?;
    Ok(KbdiRun {
        values: run.values,
        state: run.state,
        trailing_gap_days: run.trailing_gap_days,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::recurrence_inputs;
    use ndarray::{Array2, array};

    #[test]
    fn ffmc_rain_below_the_threshold_moistens_by_the_wetting_side() {
        // 0.5 mm exactly is not rain for FFMC: it stays on the wetting branch,
        // which dries a dry code toward the wetting equilibrium
        let dry = ffmc_next(85.0, 20.0, 40.0, 10.0, 0.0);
        let below_threshold = ffmc_next(85.0, 20.0, 40.0, 10.0, 0.5);
        let above_threshold = ffmc_next(85.0, 20.0, 40.0, 10.0, 5.0);
        assert_eq!(below_threshold, dry);
        // rain adds moisture to the fuel, which lowers the code
        assert!(above_threshold < dry);
        assert!(above_threshold >= 0.0 && dry <= FFMC_MAXIMUM);
    }

    #[test]
    fn dmc_dries_without_rain_and_rewets_over_its_threshold() {
        let dried = dmc_next(50.0, 25.0, 30.0, 0.0, 12.8);
        assert!(dried > 50.0);
        // a heavy rain day pulls the code back toward the wet end
        assert!(dmc_next(50.0, 25.0, 30.0, 200.0, 12.8) < 50.0);
    }

    #[test]
    fn dc_pe_floor_keeps_a_freezing_day_from_drying() {
        // -2.8 C and a negative day-length adjustment floor the evaporation at zero
        assert_eq!(dc_next(100.0, -10.0, 0.0, -1.6), 100.0);
        assert!(dc_next(100.0, 25.0, 0.0, 6.4) > 100.0);
        assert!(dc_next(100.0, 25.0, 30.0, 6.4) < 100.0);
    }

    #[test]
    fn kbdi_ignores_rain_inside_the_first_wet_spell() {
        let dry = KbdiCell {
            kbdi: 100.0,
            wet_spell_precipitation: 0.0,
        };
        let no_rain = kbdi_next(dry, 0.0, 25.0, 800.0);
        // 3 mm is inside the first 5.08 mm of the wet spell, so the index holds
        // and only the spell total moves
        let light_rain = kbdi_next(dry, 3.0, 25.0, 800.0);
        assert_eq!(light_rain.kbdi, no_rain.kbdi);
        assert_eq!(light_rain.wet_spell_precipitation, 3.0);
        // rain above that opening 5.08 mm reduces the index
        let more_rain = kbdi_next(light_rain, 10.0, 25.0, 800.0);
        assert!(more_rain.kbdi < light_rain.kbdi);
        // a freezing day adds no drought factor
        let freezing = kbdi_next(dry, 0.0, 5.0, 800.0);
        assert_eq!(freezing.kbdi, 100.0);
        assert_eq!(freezing.wet_spell_precipitation, 0.0);
    }

    #[test]
    fn kbdi_uses_the_first_day_rain_above_the_threshold_of_a_crossing_spell() {
        let dry = KbdiCell {
            kbdi: 150.0,
            wet_spell_precipitation: 0.0,
        };
        // a freezing day isolates the rain term: the first 5.08 mm only opens
        // the spell, so the 2.92 mm above it reduces the index
        let crossing = kbdi_next(dry, 8.0, 5.0, 800.0);
        assert_eq!(crossing.kbdi, 150.0 - (8.0 - KBDI_RAIN_THRESHOLD_MM));
        assert_eq!(crossing.wet_spell_precipitation, 8.0);
    }

    #[test]
    fn a_run_over_the_time_axis_matches_a_day_by_day_loop() {
        let precipitation = array![[0.0, 30.0], [5.0, 0.0], [12.0, 0.0], [0.0, 10.0]];
        let temperature = array![[25.0, 25.0], [22.0, 30.0], [18.0, 28.0], [26.0, 24.0]];
        let weather_valid: Array2<bool> = Array2::from_elem((4, 2), true);
        let gaps = array![-1, -1];
        let state = [KbdiCell {
            kbdi: 0.0,
            wet_spell_precipitation: 0.0,
        }; 2];
        let run = kbdi(
            precipitation.view(),
            temperature.view(),
            array![800.0, 900.0].view(),
            &state,
            &recurrence_inputs(
                weather_valid.view(),
                array![true, true].view(),
                None,
                gaps.view(),
                0,
                "propagate",
                0,
            )
            .unwrap(),
            true,
        )
        .unwrap();
        let values = run.values.unwrap();
        // replay the same days one at a time, the way the Python day loop does
        let mut replayed = vec![
            KbdiCell {
                kbdi: 0.0,
                wet_spell_precipitation: 0.0,
            };
            2
        ];
        for day in 0..4 {
            for cell in 0..2 {
                replayed[cell] = kbdi_next(
                    replayed[cell],
                    precipitation[[day, cell]],
                    temperature[[day, cell]],
                    if cell == 0 { 800.0 } else { 900.0 },
                );
                assert_eq!(values[[day, cell]], replayed[cell].kbdi);
            }
        }
        assert_eq!(run.state, replayed);
        assert_eq!(run.trailing_gap_days.unwrap(), array![0, 0]);
    }

    #[test]
    fn a_seasonal_mask_carries_the_code_instead_of_recording_a_gap() {
        let temperature = Array2::from_elem((3, 1), 20.0);
        let precipitation = Array2::from_elem((3, 1), 0.0);
        let in_season = array![[true], [false], [true]];
        let gaps = array![-1];
        let run = kbdi(
            precipitation.view(),
            temperature.view(),
            array![800.0].view(),
            &[KbdiCell {
                kbdi: 0.0,
                wet_spell_precipitation: 0.0,
            }],
            &recurrence_inputs(
                Array2::from_elem((3, 1), true).view(),
                array![true].view(),
                Some(in_season.view()),
                gaps.view(),
                0,
                "propagate",
                0,
            )
            .unwrap(),
            true,
        )
        .unwrap();
        let values = run.values.unwrap();
        // the off-season day emits the carried value, not a gap NaN
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values[[1, 0]] <= values[[2, 0]]);
    }

    #[test]
    fn a_step_out_of_range_latitude_band_is_rejected() {
        let band = array![0, 5];
        let months = array![[6, 6]];
        let table = Array2::from_elem((5, 12), 12.8);
        let error = check_day_length(
            &DayLength {
                table: table.view(),
                band: band.view(),
                months: months.view(),
            },
            1,
            2,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::IndexOutOfRange {
                argument: "day_length_band",
                value: 5,
                minimum: 0,
                maximum: 4
            }
        );
    }

    #[test]
    fn a_kbdi_climatology_of_the_wrong_length_is_rejected() {
        // the step indexes one climatology value per cell, so a shorter vector
        // must be an error rather than an out-of-bounds read
        let precipitation = Array2::from_elem((2, 3), 1.0);
        let temperature = Array2::from_elem((2, 3), 20.0);
        let weather_valid: Array2<bool> = Array2::from_elem((2, 3), true);
        let state = [KbdiCell {
            kbdi: 0.0,
            wet_spell_precipitation: 0.0,
        }; 3];
        let error = kbdi(
            precipitation.view(),
            temperature.view(),
            array![800.0].view(),
            &state,
            &recurrence_inputs(
                weather_valid.view(),
                array![true, true, true].view(),
                None,
                array![-1, -1, -1].view(),
                0,
                "propagate",
                0,
            )
            .unwrap(),
            true,
        )
        .unwrap_err();
        assert_eq!(
            error,
            ClimateError::ShapeMismatch {
                argument: "mean_annual_precipitation_mm",
                expected: 3,
                actual: 1
            }
        );
    }
}
