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
    // NumPy's power(..., 2) fast path is np.square, not libm pow.
    let shifted = temperature_celsius + 237.3;
    4098.0 * saturation_vapor_pressure(temperature_celsius) / (shifted * shifted)
}

/// A borrowed array or an unexpanded constant operand.
#[derive(Clone, Copy)]
pub enum Operand<'a> {
    Scalar(f64),
    Array(ArrayView1<'a, f64>),
}

impl Operand<'_> {
    fn len(&self) -> Option<usize> {
        match self {
            Self::Scalar(_) => None,
            Self::Array(array) => Some(array.len()),
        }
    }

    #[inline]
    pub fn at(&self, index: usize) -> f64 {
        self[index]
    }
}

impl std::ops::Index<usize> for Operand<'_> {
    type Output = f64;

    #[inline]
    fn index(&self, index: usize) -> &f64 {
        match self {
            Self::Scalar(value) => value,
            Self::Array(array) => &array[index],
        }
    }
}

/// The FAO-56 actual-vapour-pressure pathway the caller selected (Eq 14-19).
pub enum HumidityPathway<'a> {
    /// Eq 14: from dewpoint temperature.
    Dewpoint { tdew_celsius: Operand<'a> },
    /// Eq 17: from minimum and maximum relative humidity.
    MinimumAndMaximumRelativeHumidity {
        rh_min: Operand<'a>,
        rh_max: Operand<'a>,
    },
    /// Eq 18: from maximum relative humidity alone.
    MaximumRelativeHumidity { rh_max: Operand<'a> },
    /// Eq 19: from mean relative humidity.
    MeanRelativeHumidity { rh_mean: Operand<'a> },
    /// The arid-region estimate, `e0(Tmin - 2)`.
    MinimumTemperature,
}

/// The FAO-56 solar-radiation pathway the caller selected (Eq 35, 50).
pub enum RadiationPathway<'a> {
    /// Measured solar radiation.
    Supplied {
        solar_radiation_mj_m2_day: Operand<'a>,
    },
    /// Eq 35: from sunshine duration.
    SunshineHours { sunshine_hours: Operand<'a> },
    /// Eq 50: from the daily temperature range, with `kRs = 0.16` (interior) or
    /// `0.19` (coastal). The estimate is limited to the clear-sky radiation.
    TemperatureRange { coastal: bool },
}

/// FAO-56 Eq 6: the eight already-broadcast inputs of the Penman-Monteith equation.
pub struct PmEtoInputs<'a> {
    pub net_radiation: Operand<'a>,
    pub soil_heat_flux: Operand<'a>,
    pub temperature_celsius: Operand<'a>,
    pub wind_speed_2m: Operand<'a>,
    pub saturation_vp: Operand<'a>,
    pub actual_vp: Operand<'a>,
    pub delta: Operand<'a>,
    pub gamma: Operand<'a>,
}

/// The meteorological inputs of `pm_eto.penman_monteith_eto`, already broadcast to
/// one element per position, with the humidity and radiation pathways resolved.
pub struct MetInputs<'a> {
    pub daily_tmin_celsius: Operand<'a>,
    pub daily_tmax_celsius: Operand<'a>,
    pub latitude_degrees: Operand<'a>,
    pub elevation_m: Operand<'a>,
    pub wind_speed_m_s: Operand<'a>,
    pub wind_speed_height_m: Operand<'a>,
    pub day_of_year: Operand<'a>,
    pub soil_heat_flux_mj_m2_day: Operand<'a>,
    pub albedo: Operand<'a>,
    pub humidity: HumidityPathway<'a>,
    pub radiation: RadiationPathway<'a>,
}

/// A named argument whose length is not the length of the other arguments.
fn length_mismatch(
    argument: &'static str,
    actual: Option<usize>,
    expected: usize,
) -> Option<ClimateError> {
    actual.and_then(|actual| {
        (actual != expected).then_some(ClimateError::ShapeMismatch {
            argument,
            expected,
            actual,
        })
    })
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn fao56_equation_6(
    rn: f64,
    g: f64,
    t: f64,
    u2: f64,
    es: f64,
    ea: f64,
    delta: f64,
    gamma: f64,
) -> f64 {
    let numerator = 0.408 * delta * (rn - g) + gamma * (900.0 / (t + 273.0)) * u2 * (es - ea);
    let denominator = delta + gamma * (1.0 + 0.34 * u2);
    numerator / denominator
}

/// Cache one station term on exact input bits, including signed zero and NaNs.
fn station_term<T: Copy>(
    cache: &mut Option<(u64, T)>,
    value: f64,
    evaluate: impl FnOnce(f64) -> T,
) -> T {
    let bits = value.to_bits();
    if let Some((key, result)) = *cache
        && key == bits
    {
        return result;
    }
    let result = evaluate(value);
    *cache = Some((bits, result));
    result
}

fn latitude_terms(latitude: f64) -> [f64; 3] {
    let radians = latitude.to_radians();
    [radians.sin(), radians.cos(), radians.tan()]
}

fn astronomy(day: f64, latitude: [f64; 3]) -> [f64; 2] {
    let declination = 0.409 * (((2.0 * PI / 365.0) * day) - 1.39).sin();
    let sunset = numpy_clip(-latitude[2] * declination.tan(), -1.0, 1.0).acos();
    let distance = 1.0 + 0.033 * ((2.0 * PI / 365.0) * day).cos();
    let ra = ((24.0 * 60.0) / PI)
        * SOLAR_CONSTANT
        * distance
        * (sunset * latitude[0] * declination.sin()
            + latitude[1] * declination.cos() * sunset.sin());
    [ra, (24.0 / PI) * sunset]
}

fn astronomy_slot(day_bits: u64, latitude_bits: u64) -> usize {
    let key = day_bits ^ latitude_bits.rotate_left(23);
    (((key ^ (key >> 32)).wrapping_mul(0x9e3779b97f4a7c15)) >> 54) as usize
}

/// FAO-56 Eq 6: Penman-Monteith reference evapotranspiration.
pub fn pm_eto(inputs: &PmEtoInputs<'_>) -> Result<Array1<f64>, ClimateError> {
    let length = [
        inputs.temperature_celsius,
        inputs.net_radiation,
        inputs.soil_heat_flux,
        inputs.wind_speed_2m,
        inputs.saturation_vp,
        inputs.actual_vp,
        inputs.delta,
        inputs.gamma,
    ]
    .iter()
    .find_map(Operand::len)
    .unwrap_or(1);
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
    // Common all-array calls pay for variant selection once, not eight times per element.
    if let (
        Operand::Array(rn),
        Operand::Array(g),
        Operand::Array(t),
        Operand::Array(u2),
        Operand::Array(es),
        Operand::Array(ea),
        Operand::Array(delta),
        Operand::Array(gamma),
    ) = (
        inputs.net_radiation,
        inputs.soil_heat_flux,
        inputs.temperature_celsius,
        inputs.wind_speed_2m,
        inputs.saturation_vp,
        inputs.actual_vp,
        inputs.delta,
        inputs.gamma,
    ) {
        for index in 0..length {
            result[index] = fao56_equation_6(
                rn[index],
                g[index],
                t[index],
                u2[index],
                es[index],
                ea[index],
                delta[index],
                gamma[index],
            );
        }
        return Ok(result);
    }
    for index in 0..length {
        result[index] = fao56_equation_6(
            inputs.net_radiation[index],
            inputs.soil_heat_flux[index],
            inputs.temperature_celsius[index],
            inputs.wind_speed_2m[index],
            inputs.saturation_vp[index],
            inputs.actual_vp[index],
            inputs.delta[index],
            inputs.gamma[index],
        );
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
    let humidity_length = match &inputs.humidity {
        HumidityPathway::Dewpoint { tdew_celsius } => tdew_celsius.len(),
        HumidityPathway::MinimumAndMaximumRelativeHumidity { rh_min, rh_max } => {
            rh_min.len().or(rh_max.len())
        }
        HumidityPathway::MaximumRelativeHumidity { rh_max } => rh_max.len(),
        HumidityPathway::MeanRelativeHumidity { rh_mean } => rh_mean.len(),
        HumidityPathway::MinimumTemperature => None,
    };
    let radiation_length = match &inputs.radiation {
        RadiationPathway::Supplied {
            solar_radiation_mj_m2_day,
        } => solar_radiation_mj_m2_day.len(),
        RadiationPathway::SunshineHours { sunshine_hours } => sunshine_hours.len(),
        RadiationPathway::TemperatureRange { .. } => None,
    };
    let length = [
        inputs.daily_tmin_celsius,
        inputs.daily_tmax_celsius,
        inputs.latitude_degrees,
        inputs.elevation_m,
        inputs.wind_speed_m_s,
        inputs.wind_speed_height_m,
        inputs.day_of_year,
        inputs.soil_heat_flux_mj_m2_day,
        inputs.albedo,
    ]
    .iter()
    .find_map(Operand::len)
    .or(humidity_length)
    .or(radiation_length)
    .unwrap_or(1);
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

    let mut result = Array1::<f64>::zeros(length);
    let mut latitude_cache = None;
    let mut elevation_cache = None;
    let mut height_cache = None;
    let mut astronomy_cache = [None; 1024];

    for index in 0..length {
        let tmin_celsius = inputs.daily_tmin_celsius[index];
        let tmax_celsius = inputs.daily_tmax_celsius[index];
        let temperature_celsius = (tmin_celsius + tmax_celsius) / 2.0;
        let elevation_m = inputs.elevation_m[index];
        let latitude_degrees = inputs.latitude_degrees[index];
        let latitude = station_term(&mut latitude_cache, latitude_degrees, latitude_terms);
        let day_of_year = inputs.day_of_year[index];

        // FAO-56 Eq 47: wind speed at the 2 m standard height, which leaves the
        // input unchanged at the standard height
        let measurement_height_m = inputs.wind_speed_height_m[index];
        let height_log = station_term(&mut height_cache, measurement_height_m, |height| {
            (67.8 * height - 5.42).ln()
        });
        let wind_speed_2m = if measurement_height_m == 2.0 {
            inputs.wind_speed_m_s[index]
        } else {
            inputs.wind_speed_m_s[index] * 4.87 / height_log
        };

        // FAO-56 Eq 7-8: atmospheric pressure and the psychrometric constant
        let [gamma, clear_sky_factor] =
            station_term(&mut elevation_cache, elevation_m, |elevation| {
                let pressure = ATMOSPHERIC_PRESSURE_SEA_LEVEL
                    * ((BASE_TEMPERATURE_K - TEMPERATURE_LAPSE_RATE * elevation)
                        / BASE_TEMPERATURE_K)
                        .powf(PRESSURE_EXPONENT);
                [0.665e-3 * pressure, 0.75 + 2.0e-5 * elevation]
            });

        // FAO-56 Eq 11-13: saturation vapour pressure and the slope of its curve
        let e_tmin = saturation_vapor_pressure(tmin_celsius);
        let e_tmax = saturation_vapor_pressure(tmax_celsius);
        let saturation_vp = (e_tmin + e_tmax) / 2.0;
        let delta = vapor_pressure_slope(temperature_celsius);

        // FAO-56 Eq 14-19: the actual vapour pressure of the selected pathway
        let actual_vp = actual_vapor_pressure(
            &inputs.humidity,
            index,
            tmin_celsius,
            e_tmin,
            e_tmax,
            saturation_vp,
        );

        // FAO-56 Eq 21-34: extraterrestrial radiation and daylight hours
        let day_bits = day_of_year.to_bits();
        let latitude_bits = latitude_degrees.to_bits();
        let slot = &mut astronomy_cache[astronomy_slot(day_bits, latitude_bits)];
        let terms = match *slot {
            Some((day, lat, terms)) if day == day_bits && lat == latitude_bits => terms,
            _ => {
                let terms = astronomy(day_of_year, latitude);
                *slot = Some((day_bits, latitude_bits, terms));
                terms
            }
        };
        let [extraterrestrial_radiation, daylight_hours] = terms;

        // FAO-56 Eq 37: clear-sky solar radiation
        let clear_sky_solar_radiation = clear_sky_factor * extraterrestrial_radiation;

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
        let net_radiation = (1.0 - inputs.albedo[index]) * solar_radiation
            - STEFAN_BOLTZMANN
                * (((tmax_celsius + KELVIN_OFFSET).powf(4.0)
                    + (tmin_celsius + KELVIN_OFFSET).powf(4.0))
                    / 2.0)
                * (0.34 - 0.14 * actual_vp.sqrt())
                * (1.35 * relative_solar_radiation - 0.35);
        result[index] = fao56_equation_6(
            net_radiation,
            inputs.soil_heat_flux_mj_m2_day[index],
            temperature_celsius,
            wind_speed_2m,
            saturation_vp,
            actual_vp,
            delta,
            gamma,
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent, uncached evaluation of the original scalar expressions.
    fn reference(inputs: &MetInputs<'_>, i: usize) -> f64 {
        let tmin = inputs.daily_tmin_celsius[i];
        let tmax = inputs.daily_tmax_celsius[i];
        let t = (tmin + tmax) / 2.0;
        let lat = inputs.latitude_degrees[i].to_radians();
        let day = inputs.day_of_year[i];
        let height = inputs.wind_speed_height_m[i];
        let u2 = if height == 2.0 {
            inputs.wind_speed_m_s[i]
        } else {
            inputs.wind_speed_m_s[i] * 4.87 / (67.8 * height - 5.42).ln()
        };
        let pressure = ATMOSPHERIC_PRESSURE_SEA_LEVEL
            * ((BASE_TEMPERATURE_K - TEMPERATURE_LAPSE_RATE * inputs.elevation_m[i])
                / BASE_TEMPERATURE_K)
                .powf(PRESSURE_EXPONENT);
        let gamma = 0.665e-3 * pressure;
        let es = (saturation_vapor_pressure(tmin) + saturation_vapor_pressure(tmax)) / 2.0;
        let shifted = t + 237.3;
        let delta = 4098.0 * saturation_vapor_pressure(t) / (shifted * shifted);
        let ea = saturation_vapor_pressure(tmin - 2.0);
        let declination = 0.409 * (((2.0 * PI / 365.0) * day) - 1.39).sin();
        let sunset = numpy_clip(-lat.tan() * declination.tan(), -1.0, 1.0).acos();
        let distance = 1.0 + 0.033 * ((2.0 * PI / 365.0) * day).cos();
        let ra = ((24.0 * 60.0) / PI)
            * SOLAR_CONSTANT
            * distance
            * (sunset * lat.sin() * declination.sin()
                + lat.cos() * declination.cos() * sunset.sin());
        let daylight = (24.0 / PI) * sunset;
        let rso = (0.75 + 2.0e-5 * inputs.elevation_m[i]) * ra;
        let sunshine = match &inputs.radiation {
            RadiationPathway::SunshineHours { sunshine_hours } => sunshine_hours[i],
            _ => panic!("test reference uses sunshine hours"),
        };
        let rs = (0.25 + 0.50 * sunshine / daylight) * ra;
        let rn = (1.0 - inputs.albedo[i]) * rs
            - STEFAN_BOLTZMANN
                * (((tmax + KELVIN_OFFSET).powf(4.0) + (tmin + KELVIN_OFFSET).powf(4.0)) / 2.0)
                * (0.34 - 0.14 * ea.sqrt())
                * (1.35 * numpy_minimum(rs / rso, 1.0) - 0.35);
        let numerator = 0.408 * delta * (rn - inputs.soil_heat_flux_mj_m2_day[i])
            + gamma * (900.0 / (t + 273.0)) * u2 * (es - ea);
        numerator / (delta + gamma * (1.0 + 0.34 * u2))
    }

    #[test]
    fn caches_and_fusion_are_bit_identical_to_uncached_reference() {
        let n = 4096;
        for cycle in [0, 365, 366] {
            let mut low = Array1::from_shape_fn(n, |i| 5.0 + (i % 37) as f64 / 3.0);
            let high = &low + 9.0;
            let mut latitude = Array1::from_shape_fn(n, |i| if i < n / 2 { 40.0 } else { -35.0 });
            let mut elevation = Array1::from_shape_fn(n, |i| (i % 53) as f64 * 50.0);
            let height = Array1::from_shape_fn(n, |i| if i % 3 == 0 { 2.0 } else { 10.0 });
            let day =
                Array1::from_shape_fn(n, |i| (if cycle == 0 { i } else { i % cycle }) as f64 + 1.0);
            low[57] = f64::NAN;
            latitude[101] = f64::NAN;
            elevation[31] = f64::NAN;
            let wind = Array1::from_elem(n, 2.5);
            let soil = Array1::zeros(n);
            let albedo = Array1::from_elem(n, 0.23);
            let sunshine = Array1::from_elem(n, 9.0);
            let inputs = MetInputs {
                daily_tmin_celsius: Operand::Array(low.view()),
                daily_tmax_celsius: Operand::Array(high.view()),
                latitude_degrees: Operand::Array(latitude.view()),
                elevation_m: Operand::Array(elevation.view()),
                wind_speed_m_s: Operand::Array(wind.view()),
                wind_speed_height_m: Operand::Array(height.view()),
                day_of_year: Operand::Array(day.view()),
                soil_heat_flux_mj_m2_day: Operand::Array(soil.view()),
                albedo: Operand::Array(albedo.view()),
                humidity: HumidityPathway::MinimumTemperature,
                radiation: RadiationPathway::SunshineHours {
                    sunshine_hours: Operand::Array(sunshine.view()),
                },
            };
            if cycle == 0 {
                let slots: std::collections::HashSet<_> = day
                    .iter()
                    .map(|d| astronomy_slot(d.to_bits(), 40.0_f64.to_bits()))
                    .collect();
                assert!(slots.len() < day.len()); // >1024 distinct keys necessarily collide
            }
            let cached = penman_monteith_eto(&inputs).unwrap();
            for i in 0..n {
                assert_eq!(
                    cached[i].to_bits(),
                    reference(&inputs, i).to_bits(),
                    "cycle={cycle}, i={i}"
                );
            }
        }
    }
}
