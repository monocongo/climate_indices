//! KBDI's drying factor, `0.968 exp(0.0875 T + 1.5552) − 8.30`, without a
//! library `exp`.
//!
//! It comes from a 256-entry `2^(j/256)` table and a short polynomial, with no
//! library call, fused multiply-add or float-to-integer conversion, so it
//! vectorizes on every target and gives the same bits on every platform.

use crate::fire::KBDI_DRYING_TEMPERATURE_CELSIUS;
use crate::kbdi_exp_table::{EXP2_TABLE_BITS, HEAT_OFFSET};

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
}
