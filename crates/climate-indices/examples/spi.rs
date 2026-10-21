//! The Standardized Precipitation Index (SPI) from a plain monthly array.
//!
//! The crate publishes kernels, not a workflow. This example performs the steps a
//! caller owns: it accumulates a trailing Timescale window, estimates the
//! discrete zero mass, fits a gamma distribution per calendar month over the
//! Calibration Period, and transforms the probabilities to standard-normal
//! values. A negative result is dry, a positive result is wet.
//!
//! Run with `cargo run --example spi`.

use std::f64::consts::PI;

use climate_indices::ClimateError;
use climate_indices::gamma::{gamma_parameters, gamma_probabilities};
use climate_indices::special::norm_ppf;
use ndarray::{Array1, Array2, ArrayView2};

const YEARS: usize = 40;
const MONTHS: usize = 12;
/// The Timescale: a three-month accumulation gives SPI-3.
const SCALE: usize = 3;

fn main() -> Result<(), ClimateError> {
    let precipitation_mm = monthly_precipitation();
    let accumulated_mm = trailing_sum(precipitation_mm.view(), SCALE);

    // The Calibration Period is the caller's choice. This example calibrates on
    // every year; the fit excludes the NaN warm-up windows by itself.
    let (alphas, betas) = gamma_parameters(accumulated_mm.view());
    let probability_of_zero = zero_mass(accumulated_mm.view());
    let probabilities = gamma_probabilities(
        accumulated_mm.view(),
        alphas.view(),
        betas.view(),
        probability_of_zero.view(),
    )?;
    let spi = probabilities.mapv(norm_ppf);

    let finite = spi.iter().filter(|value| value.is_finite()).count();
    println!("SPI-{SCALE}: {finite} finite values of {}", spi.len());
    println!("year  Jan     Feb     Mar     Apr");
    for year in 1..4 {
        print!("{year:>4}");
        for month in 0..4 {
            print!("  {:>6.2}", spi[(year, month)]);
        }
        println!();
    }

    assert_eq!(spi.dim(), (YEARS, MONTHS));
    assert!(finite > (YEARS * MONTHS) / 2);
    // The synthetic record holds one dry season, whose zero accumulation gives
    // at least one calendar month a positive zero mass.
    assert!(probability_of_zero.iter().any(|mass| *mass > 0.0));
    for month in 0..SCALE - 1 {
        assert!(
            spi[(0, month)].is_nan(),
            "a month without a full window is missing"
        );
    }
    // A gamma CDF and the standard-normal quantile both increase, so the wettest
    // accumulated value of a calendar month must also carry the largest SPI.
    for month in 0..MONTHS {
        let (mut wettest_mm, mut wettest_index, mut largest_index) =
            (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for year in 0..YEARS {
            let (accumulated, index) = (accumulated_mm[(year, month)], spi[(year, month)]);
            if !index.is_finite() {
                continue;
            }
            if accumulated > wettest_mm {
                (wettest_mm, wettest_index) = (accumulated, index);
            }
            largest_index = largest_index.max(index);
        }
        assert!(
            (wettest_index - largest_index).abs() < 1e-9,
            "the largest SPI of month {month} is not its wettest accumulation"
        );
    }

    Ok(())
}

/// Sum the `SCALE` months ending at each position, across year boundaries.
///
/// The crate leaves accumulation to the caller. A window that holds a missing
/// value is itself missing, so the leading `SCALE - 1` positions stay NaN.
fn trailing_sum(values: ArrayView2<'_, f64>, scale: usize) -> Array2<f64> {
    let (years, months) = values.dim();
    let mut accumulated = Array2::<f64>::from_elem((years, months), f64::NAN);
    for index in (scale - 1)..(years * months) {
        let window: Vec<f64> = (index + 1 - scale..=index)
            .map(|position| values[[position / months, position % months]])
            .collect();
        if window.iter().all(|value| !value.is_nan()) {
            accumulated[[index / months, index % months]] = window.iter().sum();
        }
    }
    accumulated
}

/// The empirical fraction of exact zeros in each calendar month.
///
/// This is the discrete mass that [`gamma_probabilities`] expects; it does not
/// estimate the mass itself. A month whose calibration values are all zero
/// carries no separate mass, and a month with no value at all has no defined
/// one: both arrive at the kernel as an effective mass of zero.
fn zero_mass(calibration: ArrayView2<'_, f64>) -> Array1<f64> {
    let mut masses = Array1::<f64>::zeros(calibration.ncols());
    for (month, column) in calibration.columns().into_iter().enumerate() {
        let valid = column.iter().filter(|value| !value.is_nan()).count();
        let zeros = column.iter().filter(|value| **value == 0.0).count();
        masses[month] = if zeros == valid {
            0.0
        } else {
            zeros as f64 / valid as f64
        };
    }
    masses
}

/// A deterministic synthetic record: a wet summer season with one dry season,
/// so a zero accumulation window and a positive zero mass both occur.
fn monthly_precipitation() -> Array2<f64> {
    Array2::from_shape_fn((YEARS, MONTHS), |(year, month)| {
        if year == 7 && (4..7).contains(&month) {
            return 0.0;
        }
        let season_mm = 70.0 + 40.0 * ((month as f64 - 5.0) * PI / 6.0).cos();
        let noise = scramble(year * MONTHS + month);
        if noise < 0.12 {
            0.0
        } else {
            season_mm * (0.3 + 1.4 * noise)
        }
    })
}

/// A deterministic value in [0, 1) from a position index (splitmix64).
fn scramble(index: usize) -> f64 {
    let mut state = (index as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 27;
    (state >> 11) as f64 / (1_u64 << 53) as f64
}
