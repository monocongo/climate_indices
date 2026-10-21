//! Monthly potential evapotranspiration (PET) with the Thornthwaite method.
//!
//! The kernel takes monthly temperatures shaped `(years, 12, cells)` with
//! negative temperatures already clamped to zero, one latitude per cell in
//! radians, and one leap-year flag per year. It computes each cell's heat index
//! over every input year, so there is no separate Calibration Period and no
//! aggregation step to perform here. The result has the input shape, in mm per
//! month.
//!
//! For daily PET, `eto::hargreaves` takes a
//! `(days, cells)` all-leap-year layout, and `pm_eto::pm_eto` implements the
//! FAO-56 reference equation from full meteorological inputs.
//!
//! Run with `cargo run --example pet`.

use std::f64::consts::PI;

use climate_indices::ClimateError;
use climate_indices::eto::thornthwaite;
use ndarray::{Array1, Array3};

const YEARS: usize = 10;
const MONTHS: usize = 12;
const CELLS: usize = 2;

fn main() -> Result<(), ClimateError> {
    let temperatures_celsius = monthly_temperatures_celsius();
    let latitudes_radians = Array1::from_vec(vec![33.4_f64.to_radians(), 47.6_f64.to_radians()]);
    // A real calendar rule belongs to the caller; this example marks every
    // fourth year, which is the leap cycle the kernel only reads flags for.
    let leap_years = Array1::from_shape_fn(YEARS, |year| year % 4 == 3);

    let pet_mm = thornthwaite(
        temperatures_celsius.view(),
        latitudes_radians.view(),
        leap_years.view(),
    )?;

    println!("PET in mm, year 0 (cold cell, warm cell)");
    println!("month  cold    warm");
    for (month, name) in [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    .iter()
    .enumerate()
    {
        println!(
            "{name:>5}  {:>6.1}  {:>6.1}",
            pet_mm[(0, month, 0)],
            pet_mm[(0, month, 1)]
        );
    }
    for cell in 0..CELLS {
        let annual_mm: f64 = (0..MONTHS).map(|month| pet_mm[(0, month, cell)]).sum();
        println!("cell {cell}: {annual_mm:.1} mm in year 0");
    }

    assert_eq!(pet_mm.dim(), (YEARS, MONTHS, CELLS));
    assert!(
        pet_mm
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
    // The temperature cycle peaks in July, so northern-hemisphere PET must too.
    for cell in 0..CELLS {
        let peak = (0..MONTHS)
            .max_by(|left, right| {
                pet_mm[(0, *left, cell)]
                    .partial_cmp(&pet_mm[(0, *right, cell)])
                    .expect("PET is finite")
            })
            .expect("twelve months");
        assert_eq!(peak, 6, "cell {cell} peaks in July");
    }

    Ok(())
}

/// A deterministic synthetic temperature record in degrees C: a July peak, with
/// negative values clamped to zero as the kernel documents.
fn monthly_temperatures_celsius() -> Array3<f64> {
    Array3::from_shape_fn((YEARS, MONTHS, CELLS), |(year, month, cell)| {
        let season = 12.0 + 12.0 * ((month as f64 - 6.0) * PI / 6.0).cos();
        let warming = 2.0 * cell as f64 + 0.2 * year as f64;
        (season + warming - 0.5).max(0.0)
    })
}
