//! The Standardized Precipitation Evapotranspiration Index (SPEI) from plain arrays.
//!
//! SPEI standardizes accumulated (precipitation - PET) with the same
//! methodology as SPI. The crate publishes kernels, not a workflow: this example
//! subtracts the two supplied records, accumulates a trailing Timescale window,
//! fits a generalized logistic distribution per calendar month over the
//! Calibration Period, and transforms the probabilities to standard-normal
//! values.
//!
//! The generalized logistic fit used here has no discrete zero mass, so unlike
//! the gamma path there is no probability-of-zero step. Negative offsets are
//! ordinary values. A nonzero shape gives the fitted distribution finite support
//! ends, and a value beyond one of those limits saturates to a probability of 0
//! or 1.
//!
//! Run with `cargo run --example spei`.

use std::f64::consts::PI;

use climate_indices::ClimateError;
use climate_indices::aggregation::trailing_sum;
use climate_indices::loglogistic::{loglogistic_cdf_block, loglogistic_parameters};
use climate_indices::special::norm_ppf;
use ndarray::Array2;

const YEARS: usize = 40;
const MONTHS: usize = 12;
/// The Timescale: a six-month accumulation gives SPEI-6.
const SCALE: usize = 6;

fn main() -> Result<(), ClimateError> {
    // SPEI takes the difference of two records the caller supplies: an observed
    // precipitation record and a PET record. See `examples/pet.rs` for PET.
    let water_mm = &monthly_precipitation() - &monthly_pet();
    // Flatten the calendar before Aggregation, then restore the fitting columns.
    let steps = water_mm
        .view()
        .into_shape_with_order((YEARS * MONTHS, 1))
        .unwrap();
    let accumulated_mm = trailing_sum(steps, SCALE)?
        .into_shape_with_order((YEARS, MONTHS))
        .unwrap();

    // The Calibration Period is the caller's choice. This example calibrates on
    // every year; the fit excludes the NaN warm-up windows by itself.
    let fit = loglogistic_parameters(accumulated_mm.view());
    let mut probabilities = loglogistic_cdf_block(
        accumulated_mm.view(),
        fit.locs.view(),
        fit.scales.view(),
        fit.shapes.view(),
    )?;
    // The block CDF applies no fit-validity mask, so the caller decides what an
    // invalid column means. Here it is missing rather than a meaningless value.
    for (month, mut column) in probabilities.columns_mut().into_iter().enumerate() {
        if !fit.valid[month] {
            column.fill(f64::NAN);
        }
    }
    let spei = probabilities.mapv(norm_ppf);

    let valid_columns = fit.valid.iter().filter(|valid| **valid).count();
    let finite = spei.iter().filter(|value| value.is_finite()).count();
    println!("SPEI-{SCALE}: {finite} finite values, {valid_columns} of {MONTHS} fits valid");
    println!("year  Jan     Feb     Mar     Apr");
    for year in 1..4 {
        print!("{year:>4}");
        for month in 0..4 {
            print!("  {:>6.2}", spei[(year, month)]);
        }
        println!();
    }

    assert_eq!(spei.dim(), (YEARS, MONTHS));
    assert_eq!(
        valid_columns, MONTHS,
        "every synthetic month has a usable fit"
    );
    assert!(finite > (YEARS * MONTHS) / 2);
    // A generalized logistic CDF and the standard-normal quantile both increase,
    // so the wettest accumulated offset of a month must carry the largest SPEI.
    for month in 0..MONTHS {
        let (mut wettest_mm, mut wettest_index, mut largest_index) =
            (f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for year in 0..YEARS {
            let (accumulated, index) = (accumulated_mm[(year, month)], spei[(year, month)]);
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
            "the largest SPEI of month {month} is not its wettest accumulation"
        );
    }

    Ok(())
}

/// A deterministic synthetic precipitation record, in mm.
fn monthly_precipitation() -> Array2<f64> {
    Array2::from_shape_fn((YEARS, MONTHS), |(year, month)| {
        let season_mm = 70.0 + 40.0 * ((month as f64 - 5.0) * PI / 6.0).cos();
        season_mm * (0.3 + 1.4 * scramble(year * MONTHS + month))
    })
}

/// A deterministic synthetic PET record, in mm: a summer demand peak.
fn monthly_pet() -> Array2<f64> {
    Array2::from_shape_fn((YEARS, MONTHS), |(_, month)| {
        80.0 + 60.0 * ((month as f64 - 6.0) * PI / 6.0).cos()
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
