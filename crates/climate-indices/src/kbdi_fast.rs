//! KBDI over complete input: every day valid, every cell usable, no season.
//!
//! Under that contract the general recurrence driver's
//! driver's missing-day bookkeeping never fires, so a run is the step applied
//! to every cell on every day, recorded after the spin-up. This module computes
//! that faster, within the `rtol = atol = 1e-10` parity tolerance rather than
//! bit for bit:
//!
//! - the drying fraction `k = max(heat · 1e-3 / den, 0)` (0 below 10 °C) does
//!   not depend on the index, so it is computed off the dependency chain, and
//!   the index step is `Q = min(203.2, Q1 + (203.2 − Q1) k)`;
//! - `heat = 0.968 exp(0.0875 T + 1.5552) − 8.30` comes from a 256-entry
//!   `2^(j/256)` table and a short polynomial, with no library call, fused
//!   multiply-add or float-to-integer conversion, so it vectorizes on every
//!   target and gives the same bits on every platform;
//! - eight adjacent cells advance together in lanes, over tiles of
//!   [`TILE_CELLS`] cells and [`TILE_DAYS`] days.

use std::ops::Range;

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::fire::{KBDI_DRYING_TEMPERATURE_CELSIUS, KBDI_MAX_MM, KBDI_RAIN_THRESHOLD_MM, KbdiCell};
use crate::kbdi_exp_table::{EXP2_TABLE_BITS, HEAT_OFFSET};

/// Cells advanced together, each in its own lane.
const LANES: usize = 8;
/// Cells per tile: a tile's lane groups share the rows they read and write.
pub(crate) const TILE_CELLS: usize = 256;
/// Days a tile advances before the next tile.
pub(crate) const TILE_DAYS: usize = 32;

// `f64::max` is one instruction on AArch64 (`fmaxnm`) but a NaN fix-up
// sequence on x86, where the compare-select is one `maxpd`. They agree on the
// finite values this module sees.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fmax(a: f64, b: f64) -> f64 {
    a.max(b)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fmin(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn fmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn fmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// `0.968 exp(0.0875 T + 1.5552) − 8.30`, the drying term's temperature
/// factor.
///
/// `0.968` is folded into the argument as `+ ln 0.968` ([`HEAT_OFFSET`]). The
/// argument is reduced as `x = (m + j/256) ln 2 + r`, `|r| <= ln 2 / 512`;
/// `2^(j/256)` comes from the table and `exp(r) − 1` from a degree-4
/// polynomial (truncation below 4e-17 relative). The argument is clamped at
/// 709 so no temperature overflows; there is no lower clamp, because below
/// 10 °C [`drying_fraction`] discards the value, and a very negative argument
/// only yields a finite or NaN result there (the table index is masked).
#[inline(always)]
pub(crate) fn heat(temperature_celsius: f64) -> f64 {
    const SHIFT: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52: rounds to an integer
    const N_OVER_LN2: f64 = 256.0 / std::f64::consts::LN_2;
    // fdlibm's split of ln 2, over 256: n * LN2_HI is exact for |n| < 2^21
    const LN2_HI: f64 = f64::from_bits(0x3fe6_2e42_fee0_0000) / 256.0;
    const LN2_LO: f64 = f64::from_bits(0x3dea_39ef_3579_3c76) / 256.0;
    let x = fmin(0.0875 * temperature_celsius + HEAT_OFFSET, 709.0);
    let shifted = x * N_OVER_LN2 + SHIFT;
    let n = shifted - SHIFT;
    let r = (x - n * LN2_HI) - n * LN2_LO;
    // the shifted value's low mantissa bits hold n
    let bits = shifted.to_bits().wrapping_sub(SHIFT.to_bits());
    let fraction = EXP2_TABLE_BITS[(bits & 255) as usize];
    let exponent = (((bits as i64) >> 8) << 52) as u64;
    let scale = f64::from_bits(fraction.wrapping_add(exponent));
    let r2 = r * r;
    let poly = r + r2 * ((0.5 + r * (1.0 / 6.0)) + r2 * (1.0 / 24.0));
    (scale + scale * poly) - 8.30
}

/// `1e-3 / (1 + 10.88 exp(−0.001736 R))`, once per cell.
pub(crate) fn climatology_scale(mean_annual_precipitation_mm: f64) -> f64 {
    1e-3 / (1.0 + 10.88 * (-0.001736 * mean_annual_precipitation_mm).exp())
}

/// The share of the remaining deficit `203.2 − Q1` one day dries, 0 below
/// 10 °C. Computed for every day and selected, so it stays branch-free.
#[inline(always)]
pub(crate) fn drying_fraction(temperature_celsius: f64, scale: f64) -> f64 {
    let fraction = fmax(heat(temperature_celsius) * scale, 0.0);
    if temperature_celsius >= KBDI_DRYING_TEMPERATURE_CELSIUS {
        fraction
    } else {
        0.0
    }
}

/// One day: the wet-spell bookkeeping of `kbdi_next` (the same values,
/// selected instead of branched), then the index update with the fraction.
#[inline(always)]
fn step(kbdi: f64, wet_spell: f64, precipitation_mm: f64, fraction: f64) -> (f64, f64) {
    let rained = precipitation_mm > 0.0;
    let event_total = wet_spell + precipitation_mm;
    let crossing = fmax(event_total - KBDI_RAIN_THRESHOLD_MM, 0.0);
    let net = if wet_spell > KBDI_RAIN_THRESHOLD_MM {
        precipitation_mm
    } else {
        crossing
    };
    let (net, wet_spell) = if rained {
        (net, event_total)
    } else {
        (0.0, 0.0)
    };
    let after_rain = fmax(kbdi - net, 0.0);
    let kbdi = fmin(
        after_rain + (KBDI_MAX_MM - after_rain) * fraction,
        KBDI_MAX_MM,
    );
    (kbdi, wet_spell)
}

/// The time-first rows `days` of `cells` columns, and where they go.
struct Rows<'a> {
    precipitation: &'a [f64],
    temperature: &'a [f64],
    cells: usize,
    days: Range<usize>,
    /// The history's first row is day `days.start`; `None` records nothing.
    history: Option<&'a mut [f64]>,
}

/// Advance the `N` adjacent cells from `first` over `days`, from and back into
/// their state, which stays in registers in between; returns the sum of every
/// precipitation and temperature value read (see [`advance`]).
#[inline(always)]
fn lanes<const N: usize>(
    rows: &mut Rows<'_>,
    first: usize,
    days: Range<usize>,
    scales: &[f64],
    kbdi: &mut [f64],
    wet_spell: &mut [f64],
) -> f64 {
    let cells = rows.cells;
    let cell_range = first..first + N;
    let scales: &[f64; N] = scales[cell_range.clone()].try_into().expect("N lanes");
    let mut q: [f64; N] = kbdi[cell_range.clone()].try_into().expect("N lanes");
    let mut w: [f64; N] = wet_spell[cell_range.clone()].try_into().expect("N lanes");
    let mut total = [0.0; N];
    for day in days {
        let at = day * cells + first;
        let p: &[f64; N] = rows.precipitation[at..at + N].try_into().expect("N lanes");
        let t: &[f64; N] = rows.temperature[at..at + N].try_into().expect("N lanes");
        let k: [f64; N] = std::array::from_fn(|lane| drying_fraction(t[lane], scales[lane]));
        for lane in 0..N {
            total[lane] += p[lane] + t[lane];
            (q[lane], w[lane]) = step(q[lane], w[lane], p[lane], k[lane]);
        }
        if let Some(history) = rows.history.as_deref_mut() {
            let at = (day - rows.days.start) * cells + first;
            history[at..at + N].copy_from_slice(&q);
        }
    }
    kbdi[cell_range.clone()].copy_from_slice(&q);
    wet_spell[cell_range].copy_from_slice(&w);
    total.iter().sum()
}

/// Advance every cell over `rows.days`; returns the sum of every
/// precipitation and temperature value read.
///
/// The sum is finite only if every value is: a NaN or infinite input makes it
/// NaN or infinite. Finite inputs large enough to overflow it (near 1e308)
/// make it infinite too, which only sends a valid run to the general driver.
/// One add per value, cheaper than testing each value.
fn advance(rows: &mut Rows<'_>, scales: &[f64], kbdi: &mut [f64], wet_spell: &mut [f64]) -> f64 {
    let cells = rows.cells;
    let grouped = cells / LANES * LANES;
    let mut total = 0.0;
    for tile in (0..grouped).step_by(TILE_CELLS) {
        let tile_end = (tile + TILE_CELLS).min(grouped);
        for block in rows.days.clone().step_by(TILE_DAYS) {
            let block = block..(block + TILE_DAYS).min(rows.days.end);
            for first in (tile..tile_end).step_by(LANES) {
                total += lanes::<LANES>(rows, first, block.clone(), scales, kbdi, wet_spell);
            }
        }
    }
    // the fewer than eight cells left over (every cell of a small run) advance
    // together over the whole span, in as many lanes as there are cells; one
    // cell is the single-series loop, latency-bound on its index chain
    let days = rows.days.clone();
    total += match cells - grouped {
        0 => 0.0,
        1 => lanes::<1>(rows, grouped, days, scales, kbdi, wet_spell),
        2 => lanes::<2>(rows, grouped, days, scales, kbdi, wet_spell),
        3 => lanes::<3>(rows, grouped, days, scales, kbdi, wet_spell),
        4 => lanes::<4>(rows, grouped, days, scales, kbdi, wet_spell),
        5 => lanes::<5>(rows, grouped, days, scales, kbdi, wet_spell),
        6 => lanes::<6>(rows, grouped, days, scales, kbdi, wet_spell),
        _ => lanes::<7>(rows, grouped, days, scales, kbdi, wet_spell),
    };
    total
}

/// A complete-input run's history (when recorded) and final state.
pub(crate) type CompleteRun = (Option<Array2<f64>>, Vec<KbdiCell>);

/// Run KBDI over complete, time-first input.
///
/// The caller guarantees the contract in the module docs and that every
/// `mean_annual_precipitation_mm`, every initial state and every initial
/// index at most 203.2 mm are finite. Returns `None` when a precipitation or
/// temperature value is not finite, so the caller can rerun the general
/// driver, which reports it.
pub(crate) fn run(
    precipitation_mm: ArrayView2<'_, f64>,
    maximum_temperature_celsius: ArrayView2<'_, f64>,
    mean_annual_precipitation_mm: ArrayView1<'_, f64>,
    initial_state: &[KbdiCell],
    spin_up: usize,
    record: bool,
) -> Option<CompleteRun> {
    let (days, cells) = precipitation_mm.dim();
    let precipitation = precipitation_mm.as_standard_layout();
    let temperature = maximum_temperature_celsius.as_standard_layout();
    let scales: Vec<f64> = mean_annual_precipitation_mm
        .iter()
        .map(|&r| climatology_scale(r))
        .collect();
    let mut kbdi: Vec<f64> = initial_state.iter().map(|cell| cell.kbdi).collect();
    let mut wet_spell: Vec<f64> = initial_state
        .iter()
        .map(|cell| cell.wet_spell_precipitation)
        .collect();
    let spin_up = spin_up.min(days);
    // every recorded value is written, so the zeroed (lazily committed) array
    // is never read before it is overwritten
    let mut values = record.then(|| Array2::<f64>::zeros((days - spin_up, cells)));
    let (precipitation, temperature) = (
        precipitation.as_slice().expect("standard layout"),
        temperature.as_slice().expect("standard layout"),
    );
    let mut total = 0.0;
    for (days, history) in [
        (0..spin_up, None),
        (
            spin_up..days,
            values
                .as_mut()
                .map(|values| values.as_slice_mut().expect("standard layout")),
        ),
    ] {
        let mut rows = Rows {
            precipitation,
            temperature,
            cells,
            days,
            history,
        };
        total += advance(&mut rows, &scales, &mut kbdi, &mut wet_spell);
    }
    if !total.is_finite() {
        return None;
    }
    let state = kbdi
        .into_iter()
        .zip(wet_spell)
        .map(|(kbdi, wet_spell_precipitation)| KbdiCell {
            kbdi,
            wet_spell_precipitation,
        })
        .collect();
    Some((values, state))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The heat term as the general step computes it, with the platform `exp`.
    fn reference_heat(temperature_celsius: f64) -> f64 {
        0.968 * (0.0875 * temperature_celsius + 1.5552).exp() - 8.30
    }

    #[test]
    fn the_table_heat_term_matches_the_platform_exp_where_it_dries() {
        // every hundredth of a degree from the drying threshold to 80 C: the
        // table form differs from the platform `exp` by a few ulps of the term
        let mut worst: f64 = 0.0;
        for hundredths in 1_000..=8_000 {
            let temperature = f64::from(hundredths) / 100.0;
            let want = reference_heat(temperature);
            worst = worst.max(((heat(temperature) - want) / want).abs());
        }
        assert!(worst < 1e-14, "worst relative error {worst:e}");
    }

    #[test]
    fn the_drying_fraction_is_zero_below_ten_degrees_and_finite_at_any_temperature() {
        let scale = climatology_scale(800.0);
        for temperature in [-1e300, -1e4, -40.0, 0.0, 9.999_999] {
            assert_eq!(drying_fraction(temperature, scale), 0.0);
        }
        for temperature in [10.0, 45.0, 1e4, 1e300] {
            let fraction = drying_fraction(temperature, scale);
            assert!(
                fraction.is_finite() && fraction > 0.0,
                "{temperature}: {fraction}"
            );
        }
    }

    #[test]
    fn a_non_finite_weather_value_declines_the_run() {
        let precipitation = ndarray::array![[0.0, 1.0], [f64::INFINITY, 0.0]];
        let temperature = ndarray::array![[20.0, 20.0], [20.0, 20.0]];
        let state = [KbdiCell {
            kbdi: 0.0,
            wet_spell_precipitation: 0.0,
        }; 2];
        let mean = ndarray::array![800.0, 800.0];
        let run = |p: ArrayView2<'_, f64>, t: ArrayView2<'_, f64>| {
            run(p, t, mean.view(), &state, 0, true)
        };
        assert!(run(precipitation.view(), temperature.view()).is_none());
        let mut cold = temperature.clone();
        cold[[0, 1]] = f64::NAN;
        assert!(run(temperature.view(), cold.view()).is_none());
        assert!(run(temperature.view(), temperature.view()).is_some());
    }
}
