//! Penman-Monteith (FAO-56) PET kernels, mirroring `climate_indices.pm_eto`.
//!
//! The kernels are ports of the Python reference implementation, which stays the
//! parity oracle: the same operation order, the same libm call sites, and the
//! same NaN propagation as NumPy. Validation, warnings, logging, and the
//! humidity/radiation pathway precedence stay in the Python package; the caller
//! resolves the pathways and hands each kernel one array per broadcast element,
//! so every input of a kernel has the same length.

use std::f64::consts::PI;

use ndarray::{Array1, ArrayView1};

use crate::ClimateError;

// physical constants, as `pm_eto.py` declares them
const ATMOSPHERIC_PRESSURE_SEA_LEVEL: f64 = 101.3;
const TEMPERATURE_LAPSE_RATE: f64 = 0.0065;
const BASE_TEMPERATURE_K: f64 = 293.0;
const PRESSURE_EXPONENT: f64 = 5.26;
const SOLAR_CONSTANT: f64 = 0.0820;
const STEFAN_BOLTZMANN: f64 = 4.903e-9;
const KELVIN_OFFSET: f64 = 273.16;

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

/// `np.minimum` for one pair: NaN in, NaN out (Rust's `f64::min` ignores NaN).
fn numpy_minimum(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left < right {
        left
    } else {
        right
    }
}

/// `np.maximum` for one pair: NaN in, NaN out (Rust's `f64::max` ignores NaN).
fn numpy_maximum(left: f64, right: f64) -> f64 {
    if left.is_nan() || right.is_nan() {
        f64::NAN
    } else if left > right {
        left
    } else {
        right
    }
}

/// FAO-56 Eq 11: saturation vapour pressure at a temperature.
fn saturation_vapor_pressure(temperature_celsius: f64) -> f64 {
    0.6108 * ((17.27 * temperature_celsius) / (temperature_celsius + 237.3)).exp()
}

/// FAO-56 Eq 13: slope of the saturation vapour pressure curve.
fn vapor_pressure_slope(temperature_celsius: f64) -> f64 {
    4098.0 * saturation_vapor_pressure(temperature_celsius)
        / (temperature_celsius + 237.3).powf(2.0)
}

/// The FAO-56 actual-vapour-pressure pathway the caller selected (Eq 14-19).
pub enum HumidityPathway<'a> {
    /// Eq 14: from dewpoint temperature.
    Dewpoint { tdew_celsius: ArrayView1<'a, f64> },
    /// Eq 17: from minimum and maximum relative humidity.
    MinimumAndMaximumRelativeHumidity {
        rh_min: ArrayView1<'a, f64>,
        rh_max: ArrayView1<'a, f64>,
    },
    /// Eq 18: from maximum relative humidity alone.
    MaximumRelativeHumidity { rh_max: ArrayView1<'a, f64> },
    /// Eq 19: from mean relative humidity.
    MeanRelativeHumidity { rh_mean: ArrayView1<'a, f64> },
    /// The arid-region estimate, `e0(Tmin - 2)`.
    MinimumTemperature,
}

/// The FAO-56 solar-radiation pathway the caller selected (Eq 35, 50).
pub enum RadiationPathway<'a> {
    /// Measured solar radiation.
    Supplied {
        solar_radiation_mj_m2_day: ArrayView1<'a, f64>,
    },
    /// Eq 35: from sunshine duration.
    SunshineHours { sunshine_hours: ArrayView1<'a, f64> },
    /// Eq 50: from the daily temperature range, with `kRs = 0.16` (interior) or
    /// `0.19` (coastal). The estimate is limited to the clear-sky radiation.
    TemperatureRange { coastal: bool },
}

/// FAO-56 Eq 6: the eight already-broadcast inputs of the Penman-Monteith equation.
pub struct PmEtoInputs<'a> {
    pub net_radiation: ArrayView1<'a, f64>,
    pub soil_heat_flux: ArrayView1<'a, f64>,
    pub temperature_celsius: ArrayView1<'a, f64>,
    pub wind_speed_2m: ArrayView1<'a, f64>,
    pub saturation_vp: ArrayView1<'a, f64>,
    pub actual_vp: ArrayView1<'a, f64>,
    pub delta: ArrayView1<'a, f64>,
    pub gamma: ArrayView1<'a, f64>,
}

/// The meteorological inputs of `pm_eto.penman_monteith_eto`, already broadcast to
/// one element per position, with the humidity and radiation pathways resolved.
pub struct MetInputs<'a> {
    pub daily_tmin_celsius: ArrayView1<'a, f64>,
    pub daily_tmax_celsius: ArrayView1<'a, f64>,
    pub latitude_degrees: ArrayView1<'a, f64>,
    pub elevation_m: ArrayView1<'a, f64>,
    pub wind_speed_m_s: ArrayView1<'a, f64>,
    pub wind_speed_height_m: ArrayView1<'a, f64>,
    pub day_of_year: ArrayView1<'a, f64>,
    pub soil_heat_flux_mj_m2_day: ArrayView1<'a, f64>,
    pub albedo: ArrayView1<'a, f64>,
    pub humidity: HumidityPathway<'a>,
    pub radiation: RadiationPathway<'a>,
}

/// A named argument whose length is not the length of the other arguments.
fn length_mismatch(argument: &'static str, actual: usize, expected: usize) -> Option<ClimateError> {
    (actual != expected).then_some(ClimateError::ShapeMismatch {
        argument,
        expected,
        actual,
    })
}

/// FAO-56 Eq 6: Penman-Monteith reference evapotranspiration.
pub fn pm_eto(inputs: &PmEtoInputs<'_>) -> Result<Array1<f64>, ClimateError> {
    let length = inputs.temperature_celsius.len();
    let mismatch = [
        ("net_radiation", inputs.net_radiation.len()),
        ("soil_heat_flux", inputs.soil_heat_flux.len()),
        ("wind_speed_2m", inputs.wind_speed_2m.len()),
        ("saturation_vp", inputs.saturation_vp.len()),
        ("actual_vp", inputs.actual_vp.len()),
        ("delta", inputs.delta.len()),
        ("gamma", inputs.gamma.len()),
    ]
    .into_iter()
    .find_map(|(argument, actual)| length_mismatch(argument, actual, length));
    if let Some(error) = mismatch {
        return Err(error);
    }

    let mut result = Array1::<f64>::zeros(length);
    for index in 0..length {
        // the radiation term and the aerodynamic term of the numerator
        let numerator = 0.408
            * inputs.delta[index]
            * (inputs.net_radiation[index] - inputs.soil_heat_flux[index])
            + inputs.gamma[index]
                * (900.0 / (inputs.temperature_celsius[index] + 273.0))
                * inputs.wind_speed_2m[index]
                * (inputs.saturation_vp[index] - inputs.actual_vp[index]);
        let denominator =
            inputs.delta[index] + inputs.gamma[index] * (1.0 + 0.34 * inputs.wind_speed_2m[index]);
        result[index] = numerator / denominator;
    }
    Ok(result)
}

/// FAO-56 Eq 14-19: actual vapour pressure for the selected pathway.
fn actual_vapor_pressure(
    pathway: &HumidityPathway<'_>,
    index: usize,
    tmin_celsius: f64,
    e_tmin: f64,
    e_tmax: f64,
    e_s: f64,
) -> f64 {
    match pathway {
        HumidityPathway::Dewpoint { tdew_celsius } => {
            saturation_vapor_pressure(tdew_celsius[index])
        }
        HumidityPathway::MinimumAndMaximumRelativeHumidity { rh_min, rh_max } => {
            (e_tmin * rh_max[index] / 100.0 + e_tmax * rh_min[index] / 100.0) / 2.0
        }
        HumidityPathway::MaximumRelativeHumidity { rh_max } => e_tmin * rh_max[index] / 100.0,
        HumidityPathway::MeanRelativeHumidity { rh_mean } => e_s * rh_mean[index] / 100.0,
        HumidityPathway::MinimumTemperature => saturation_vapor_pressure(tmin_celsius - 2.0),
    }
}

/// Derive the FAO-56 intermediates and evaluate Eq 6, the port of
/// `pm_eto.penman_monteith_eto`.
pub fn penman_monteith_eto(inputs: &MetInputs<'_>) -> Result<Array1<f64>, ClimateError> {
    let length = inputs.daily_tmin_celsius.len();
    let mismatch = [
        ("daily_tmax_celsius", inputs.daily_tmax_celsius.len()),
        ("latitude_degrees", inputs.latitude_degrees.len()),
        ("elevation_m", inputs.elevation_m.len()),
        ("wind_speed_m_s", inputs.wind_speed_m_s.len()),
        ("wind_speed_height_m", inputs.wind_speed_height_m.len()),
        ("day_of_year", inputs.day_of_year.len()),
        (
            "soil_heat_flux_mj_m2_day",
            inputs.soil_heat_flux_mj_m2_day.len(),
        ),
        ("albedo", inputs.albedo.len()),
    ]
    .into_iter()
    .find_map(|(argument, actual)| length_mismatch(argument, actual, length))
    .or_else(|| match &inputs.humidity {
        HumidityPathway::Dewpoint { tdew_celsius } => {
            length_mismatch("tdew_celsius", tdew_celsius.len(), length)
        }
        HumidityPathway::MinimumAndMaximumRelativeHumidity { rh_min, rh_max } => {
            length_mismatch("rh_min", rh_min.len(), length)
                .or_else(|| length_mismatch("rh_max", rh_max.len(), length))
        }
        HumidityPathway::MaximumRelativeHumidity { rh_max } => {
            length_mismatch("rh_max", rh_max.len(), length)
        }
        HumidityPathway::MeanRelativeHumidity { rh_mean } => {
            length_mismatch("rh_mean", rh_mean.len(), length)
        }
        HumidityPathway::MinimumTemperature => None,
    })
    .or_else(|| match &inputs.radiation {
        RadiationPathway::Supplied {
            solar_radiation_mj_m2_day,
        } => length_mismatch(
            "solar_radiation_mj_m2_day",
            solar_radiation_mj_m2_day.len(),
            length,
        ),
        RadiationPathway::SunshineHours { sunshine_hours } => {
            length_mismatch("sunshine_hours", sunshine_hours.len(), length)
        }
        RadiationPathway::TemperatureRange { .. } => None,
    });
    if let Some(error) = mismatch {
        return Err(error);
    }

    let mut temperature_celsius = Array1::<f64>::zeros(length);
    let mut wind_speed_2m = Array1::<f64>::zeros(length);
    let mut gamma = Array1::<f64>::zeros(length);
    let mut saturation_vp = Array1::<f64>::zeros(length);
    let mut delta = Array1::<f64>::zeros(length);
    let mut actual_vp = Array1::<f64>::zeros(length);
    let mut net_radiation = Array1::<f64>::zeros(length);

    for index in 0..length {
        let tmin_celsius = inputs.daily_tmin_celsius[index];
        let tmax_celsius = inputs.daily_tmax_celsius[index];
        temperature_celsius[index] = (tmin_celsius + tmax_celsius) / 2.0;
        let elevation_m = inputs.elevation_m[index];
        let latitude_radians = inputs.latitude_degrees[index].to_radians();
        let day_of_year = inputs.day_of_year[index];

        // FAO-56 Eq 47: wind speed at the 2 m standard height, which leaves the
        // input unchanged at the standard height
        let measurement_height_m = inputs.wind_speed_height_m[index];
        wind_speed_2m[index] = if measurement_height_m == 2.0 {
            inputs.wind_speed_m_s[index]
        } else {
            inputs.wind_speed_m_s[index] * 4.87 / (67.8 * measurement_height_m - 5.42).ln()
        };

        // FAO-56 Eq 7-8: atmospheric pressure and the psychrometric constant
        let atmospheric_pressure = ATMOSPHERIC_PRESSURE_SEA_LEVEL
            * ((BASE_TEMPERATURE_K - TEMPERATURE_LAPSE_RATE * elevation_m) / BASE_TEMPERATURE_K)
                .powf(PRESSURE_EXPONENT);
        gamma[index] = 0.665e-3 * atmospheric_pressure;

        // FAO-56 Eq 11-13: saturation vapour pressure and the slope of its curve
        let e_tmin = saturation_vapor_pressure(tmin_celsius);
        let e_tmax = saturation_vapor_pressure(tmax_celsius);
        saturation_vp[index] = (e_tmin + e_tmax) / 2.0;
        delta[index] = vapor_pressure_slope(temperature_celsius[index]);

        // FAO-56 Eq 14-19: the actual vapour pressure of the selected pathway
        actual_vp[index] = actual_vapor_pressure(
            &inputs.humidity,
            index,
            tmin_celsius,
            e_tmin,
            e_tmax,
            saturation_vp[index],
        );

        // FAO-56 Eq 21-34: extraterrestrial radiation and daylight hours
        let declination = 0.409 * (((2.0 * PI / 365.0) * day_of_year) - 1.39).sin();
        let sunset_hour_angle =
            numpy_clip(-latitude_radians.tan() * declination.tan(), -1.0, 1.0).acos();
        let inverse_relative_distance = 1.0 + 0.033 * ((2.0 * PI / 365.0) * day_of_year).cos();
        let extraterrestrial_radiation = ((24.0 * 60.0) / PI)
            * SOLAR_CONSTANT
            * inverse_relative_distance
            * (sunset_hour_angle * latitude_radians.sin() * declination.sin()
                + latitude_radians.cos() * declination.cos() * sunset_hour_angle.sin());
        let daylight_hours = (24.0 / PI) * sunset_hour_angle;

        // FAO-56 Eq 37: clear-sky solar radiation
        let clear_sky_solar_radiation = (0.75 + 2.0e-5 * elevation_m) * extraterrestrial_radiation;

        // FAO-56 Eq 35, 50: solar radiation. A temperature-range estimate is
        // limited to the clear-sky radiation, as FAO-56 requires.
        let solar_radiation = match &inputs.radiation {
            RadiationPathway::Supplied {
                solar_radiation_mj_m2_day,
            } => solar_radiation_mj_m2_day[index],
            RadiationPathway::SunshineHours { sunshine_hours } => {
                (0.25 + 0.50 * sunshine_hours[index] / daylight_hours) * extraterrestrial_radiation
            }
            RadiationPathway::TemperatureRange { coastal } => {
                let angstrom_coefficient = if *coastal { 0.19 } else { 0.16 };
                let temperature_range = numpy_maximum(tmax_celsius - tmin_celsius, 0.0);
                numpy_minimum(
                    angstrom_coefficient * temperature_range.sqrt() * extraterrestrial_radiation,
                    clear_sky_solar_radiation,
                )
            }
        };

        // FAO-56 Eq 38-40: net shortwave and net longwave radiation
        let relative_solar_radiation =
            numpy_minimum(solar_radiation / clear_sky_solar_radiation, 1.0);
        net_radiation[index] = (1.0 - inputs.albedo[index]) * solar_radiation
            - STEFAN_BOLTZMANN
                * (((tmax_celsius + KELVIN_OFFSET).powf(4.0)
                    + (tmin_celsius + KELVIN_OFFSET).powf(4.0))
                    / 2.0)
                * (0.34 - 0.14 * actual_vp[index].sqrt())
                * (1.35 * relative_solar_radiation - 0.35);
    }

    pm_eto(&PmEtoInputs {
        net_radiation: net_radiation.view(),
        soil_heat_flux: inputs.soil_heat_flux_mj_m2_day,
        temperature_celsius: temperature_celsius.view(),
        wind_speed_2m: wind_speed_2m.view(),
        saturation_vp: saturation_vp.view(),
        actual_vp: actual_vp.view(),
        delta: delta.view(),
        gamma: gamma.view(),
    })
}
