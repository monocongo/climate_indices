//! Thornthwaite and Hargreaves PET kernels, mirroring `climate_indices.eto`.
//!
//! Both kernels are ports of the Python reference implementation, which stays
//! the parity oracle: the same operation order, the same libm call sites, and
//! the same NaN propagation as NumPy. Validation, warnings, reshaping, and
//! truncation stay in the Python package.

use std::f64::consts::PI;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};

use crate::ClimateError;

/// Days of each calendar month, for non-leap and leap years, as `eto.py` has them.
const MONTH_DAYS_NONLEAP: [f64; 12] = [
    31.0, 28.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0,
];
const MONTH_DAYS_LEAP: [f64; 12] = [
    31.0, 29.0, 31.0, 30.0, 31.0, 30.0, 31.0, 31.0, 30.0, 31.0, 30.0, 31.0,
];

/// Solar constant [MJ m-2 min-1] (FAO-56 Eq 21).
const SOLAR_CONSTANT: f64 = 0.0820;

/// `np.clip` for one value: NaN in, NaN out (Rust's `f64::clamp` returns a bound).
fn numpy_clip(value: f64, minimum: f64, maximum: f64) -> f64 {
    if value.is_nan() {
        f64::NAN
    } else if value < minimum {
        minimum
    } else if value > maximum {
        maximum
    } else {
        value
    }
}

/// Solar declination, the port of `eto._solar_declination` (FAO-56 Eq 24).
fn solar_declination(day_of_year: f64) -> f64 {
    0.409 * (((2.0 * PI / 365.0) * day_of_year) - 1.39).sin()
}

/// Sunset hour angle, the port of `eto._sunset_hour_angle` (FAO-56 Eq 25).
fn sunset_hour_angle(latitude_radians: f64, solar_declination_radians: f64) -> f64 {
    let cosine = -latitude_radians.tan() * solar_declination_radians.tan();
    numpy_clip(cosine, -1.0, 1.0).acos()
}

/// Mean daylight hours of each calendar month, the port of
/// `eto._monthly_mean_daylight_hours`: one row per month, one column per cell.
fn monthly_mean_daylight_hours(latitude_radians: ArrayView1<'_, f64>, leap: bool) -> Array2<f64> {
    let month_days = if leap {
        MONTH_DAYS_LEAP
    } else {
        MONTH_DAYS_NONLEAP
    };
    let cells = latitude_radians.len();
    let mut means = Array2::<f64>::zeros((12, cells));
    let mut day_of_year = 1.0;
    for (month, days_in_month) in month_days.iter().enumerate() {
        for _ in 0..(*days_in_month as usize) {
            let declination = solar_declination(day_of_year);
            for cell in 0..cells {
                means[(month, cell)] +=
                    (24.0 / PI) * sunset_hour_angle(latitude_radians[cell], declination);
            }
            day_of_year += 1.0;
        }
        for cell in 0..cells {
            means[(month, cell)] /= *days_in_month;
        }
    }
    means
}

/// Thornthwaite (1948) monthly PET, the port of `eto.eto_thornthwaite`.
///
/// `monthly_temps_celsius` is (years, 12, cells) with negative temperatures
/// already clamped to zero, `latitude_radians` carries one latitude per cell,
/// and `leap_years` one flag per year. The result has the input shape.
pub fn thornthwaite(
    monthly_temps_celsius: ArrayView3<'_, f64>,
    latitude_radians: ArrayView1<'_, f64>,
    leap_years: ArrayView1<'_, bool>,
) -> Result<Array3<f64>, ClimateError> {
    let (years, months, cells) = monthly_temps_celsius.dim();
    for (argument, actual, expected) in [
        ("months", months, 12usize),
        ("latitude_radians", latitude_radians.len(), cells),
        ("leap_years", leap_years.len(), years),
    ] {
        if actual != expected {
            return Err(ClimateError::ShapeMismatch {
                argument,
                expected,
                actual,
            });
        }
    }

    // monthly means over the year axis, NaN-ignoring as np.nanmean is, and the heat
    // index they accumulate into
    let mut heat_index = Array1::<f64>::zeros(cells);
    for month in 0..months {
        for cell in 0..cells {
            let (mut sum, mut count) = (0.0_f64, 0_usize);
            for year in 0..years {
                let value = monthly_temps_celsius[(year, month, cell)];
                if !value.is_nan() {
                    sum += value;
                    count += 1;
                }
            }
            let mean = if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            };
            heat_index[cell] += (mean / 5.0).powf(1.514);
        }
    }

    // Thornthwaite's exponent, per cell
    let mut exponent = Array1::<f64>::zeros(cells);
    for cell in 0..cells {
        let index = heat_index[cell];
        exponent[cell] =
            6.75e-07 * index.powf(3.0) - 7.71e-05 * index.powf(2.0) + 1.792e-02 * index + 0.49239;
    }

    // mean daylight hours of each calendar month, for non-leap and leap years
    let daylight_nonleap = monthly_mean_daylight_hours(latitude_radians, false);
    let daylight_leap = monthly_mean_daylight_hours(latitude_radians, true);

    let mut pet = Array3::<f64>::from_elem((years, months, cells), f64::NAN);
    for year in 0..years {
        let (month_days, daylight) = if leap_years[year] {
            (&MONTH_DAYS_LEAP, &daylight_leap)
        } else {
            (&MONTH_DAYS_NONLEAP, &daylight_nonleap)
        };
        for month in 0..months {
            let month_length = month_days[month] / 30.0;
            for cell in 0..cells {
                let mean_daylight_hours = daylight[(month, cell)] / 12.0;
                let ratio = 10.0 * monthly_temps_celsius[(year, month, cell)] / heat_index[cell];
                pet[(year, month, cell)] =
                    16.0 * mean_daylight_hours * month_length * ratio.powf(exponent[cell]);
            }
        }
    }
    Ok(pet)
}

/// Hargreaves (1985) daily PET, the port of `eto.eto_hargreaves`.
///
/// The three daily blocks are (time, cells) in year-major 366-day order and
/// `latitude_radians` carries one latitude per cell. Rows that no day of the
/// year reaches stay NaN, the way the Python result's padded rows do.
pub fn hargreaves(
    daily_tmin_celsius: ArrayView2<'_, f64>,
    daily_tmax_celsius: ArrayView2<'_, f64>,
    daily_tmean_celsius: ArrayView2<'_, f64>,
    latitude_radians: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, ClimateError> {
    let (time, cells) = daily_tmean_celsius.dim();
    if daily_tmin_celsius.dim() != (time, cells) {
        return Err(ClimateError::ShapeMismatch {
            argument: "daily_tmin_celsius",
            expected: time * cells,
            actual: daily_tmin_celsius.len(),
        });
    }
    if daily_tmax_celsius.dim() != (time, cells) {
        return Err(ClimateError::ShapeMismatch {
            argument: "daily_tmax_celsius",
            expected: time * cells,
            actual: daily_tmax_celsius.len(),
        });
    }
    if latitude_radians.len() != cells {
        return Err(ClimateError::ShapeMismatch {
            argument: "latitude_radians",
            expected: cells,
            actual: latitude_radians.len(),
        });
    }

    let mut pet = Array2::<f64>::from_elem((time, cells), f64::NAN);
    for day_of_year in 1..=366 {
        let declination = solar_declination(day_of_year as f64);

        // FAO-56 Eq 23: inverse relative distance between earth and sun
        let inverse_relative_distance =
            1.0 + 0.033 * ((2.0 * PI / 365.0) * day_of_year as f64).cos();
        let scale = ((24.0 * 60.0) / PI) * SOLAR_CONSTANT * inverse_relative_distance;

        for cell in 0..cells {
            let latitude = latitude_radians[cell];
            let sunset = sunset_hour_angle(latitude, declination);

            // FAO-56 Eq 21: extraterrestrial radiation
            let extraterrestrial = scale
                * (sunset * latitude.sin() * declination.sin()
                    + latitude.cos() * declination.cos() * sunset.sin());

            // the rows of this day of the year: one per whole year, plus the row of
            // a trailing partial year once the day is reached
            let mut row = day_of_year - 1;
            while row < time {
                pet[(row, cell)] = 0.0023
                    * (daily_tmean_celsius[(row, cell)] + 17.8)
                    * (daily_tmax_celsius[(row, cell)] - daily_tmin_celsius[(row, cell)]).powf(0.5)
                    * 0.408
                    * extraterrestrial;
                row += 366;
            }
        }
    }
    Ok(pet)
}
