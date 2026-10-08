//! Array conversion for the PET kernels; the equations live in `climate-core`.
//!
//! Every array argument must be a float64 ndarray in the layout the matching
//! kernel documents. The kernels receive one element per broadcast position, so
//! the caller flattens the broadcast inputs and reshapes the result.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use climate_core::pm_eto::{HumidityPathway, MetInputs, PmEtoInputs, RadiationPathway};

// selectors of the FAO-56 actual-vapour-pressure pathway (Eq 14-19); the Python
// dispatch passes these codes and must match `pm_eto.py`'s constants
const HUMIDITY_DEWPOINT: u8 = 0;
const HUMIDITY_RH_MIN_MAX: u8 = 1;
const HUMIDITY_RH_MAX: u8 = 2;
const HUMIDITY_RH_MEAN: u8 = 3;
const HUMIDITY_TMIN: u8 = 4;

// selectors of the FAO-56 solar-radiation pathway (Eq 35, 50)
const RADIATION_SUPPLIED: u8 = 0;
const RADIATION_SUNSHINE: u8 = 1;
const RADIATION_TEMPERATURE_RANGE: u8 = 2;

/// Copy an optional float64 array, rejecting an unaligned one as `checked_copy` does.
fn checked_optional(array: Option<PyReadonlyArray1<'_, f64>>) -> PyResult<Option<Array1<f64>>> {
    array.map(|array| crate::checked_copy(&array)).transpose()
}

/// The array an optional input must hold when its pathway is selected.
fn required<'a>(argument: &str, array: &'a Option<Array1<f64>>) -> PyResult<ArrayView1<'a, f64>> {
    array.as_ref().map(|array| array.view()).ok_or_else(|| {
        PyValueError::new_err(format!("{argument} is required for the selected pathway"))
    })
}

/// Thornthwaite (1948) monthly PET: (years, 12, cells) in, same shape out.
#[pyfunction]
fn thornthwaite<'py>(
    py: Python<'py>,
    monthly_temps_celsius: PyReadonlyArray3<'py, f64>,
    latitude_radians: PyReadonlyArray1<'py, f64>,
    leap_years: PyReadonlyArray1<'py, bool>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let monthly_temps_celsius = crate::checked_copy(&monthly_temps_celsius)?;
    let latitude_radians = crate::checked_copy(&latitude_radians)?;
    let leap_years = leap_years.as_array().to_owned();
    py.detach(|| {
        climate_core::eto::thornthwaite(
            monthly_temps_celsius.view(),
            latitude_radians.view(),
            leap_years.view(),
        )
    })
    .map(|pet| pet.into_pyarray(py))
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Hargreaves (1985) daily PET: (time, cells) in, same shape out.
#[pyfunction]
fn hargreaves<'py>(
    py: Python<'py>,
    daily_tmin_celsius: PyReadonlyArray2<'py, f64>,
    daily_tmax_celsius: PyReadonlyArray2<'py, f64>,
    daily_tmean_celsius: PyReadonlyArray2<'py, f64>,
    latitude_radians: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let daily_tmin_celsius = crate::checked_copy(&daily_tmin_celsius)?;
    let daily_tmax_celsius = crate::checked_copy(&daily_tmax_celsius)?;
    let daily_tmean_celsius = crate::checked_copy(&daily_tmean_celsius)?;
    let latitude_radians = crate::checked_copy(&latitude_radians)?;
    py.detach(|| {
        climate_core::eto::hargreaves(
            daily_tmin_celsius.view(),
            daily_tmax_celsius.view(),
            daily_tmean_celsius.view(),
            latitude_radians.view(),
        )
    })
    .map(|pet| pet.into_pyarray(py))
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// FAO-56 Eq 6: Penman-Monteith ETo from prepared, already-broadcast terms.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn pm_eto<'py>(
    py: Python<'py>,
    net_radiation: PyReadonlyArray1<'py, f64>,
    soil_heat_flux: PyReadonlyArray1<'py, f64>,
    temperature_celsius: PyReadonlyArray1<'py, f64>,
    wind_speed_2m: PyReadonlyArray1<'py, f64>,
    saturation_vp: PyReadonlyArray1<'py, f64>,
    actual_vp: PyReadonlyArray1<'py, f64>,
    delta: PyReadonlyArray1<'py, f64>,
    gamma: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let net_radiation = crate::checked_copy(&net_radiation)?;
    let soil_heat_flux = crate::checked_copy(&soil_heat_flux)?;
    let temperature_celsius = crate::checked_copy(&temperature_celsius)?;
    let wind_speed_2m = crate::checked_copy(&wind_speed_2m)?;
    let saturation_vp = crate::checked_copy(&saturation_vp)?;
    let actual_vp = crate::checked_copy(&actual_vp)?;
    let delta = crate::checked_copy(&delta)?;
    let gamma = crate::checked_copy(&gamma)?;
    let inputs = PmEtoInputs {
        net_radiation: net_radiation.view(),
        soil_heat_flux: soil_heat_flux.view(),
        temperature_celsius: temperature_celsius.view(),
        wind_speed_2m: wind_speed_2m.view(),
        saturation_vp: saturation_vp.view(),
        actual_vp: actual_vp.view(),
        delta: delta.view(),
        gamma: gamma.view(),
    };
    py.detach(|| climate_core::pm_eto::pm_eto(&inputs))
        .map(|eto| eto.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// FAO-56 Penman-Monteith ETo from meteorology, with the humidity and radiation
/// pathways resolved by the caller.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn fao56_eto<'py>(
    py: Python<'py>,
    daily_tmin_celsius: PyReadonlyArray1<'py, f64>,
    daily_tmax_celsius: PyReadonlyArray1<'py, f64>,
    latitude_degrees: PyReadonlyArray1<'py, f64>,
    elevation_m: PyReadonlyArray1<'py, f64>,
    wind_speed_m_s: PyReadonlyArray1<'py, f64>,
    wind_speed_height_m: PyReadonlyArray1<'py, f64>,
    day_of_year: PyReadonlyArray1<'py, f64>,
    soil_heat_flux_mj_m2_day: PyReadonlyArray1<'py, f64>,
    albedo: PyReadonlyArray1<'py, f64>,
    humidity_variant: u8,
    tdew_celsius: Option<PyReadonlyArray1<'py, f64>>,
    rh_min: Option<PyReadonlyArray1<'py, f64>>,
    rh_max: Option<PyReadonlyArray1<'py, f64>>,
    rh_mean: Option<PyReadonlyArray1<'py, f64>>,
    radiation_variant: u8,
    solar_radiation_mj_m2_day: Option<PyReadonlyArray1<'py, f64>>,
    sunshine_hours: Option<PyReadonlyArray1<'py, f64>>,
    coastal: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let daily_tmin_celsius = crate::checked_copy(&daily_tmin_celsius)?;
    let daily_tmax_celsius = crate::checked_copy(&daily_tmax_celsius)?;
    let latitude_degrees = crate::checked_copy(&latitude_degrees)?;
    let elevation_m = crate::checked_copy(&elevation_m)?;
    let wind_speed_m_s = crate::checked_copy(&wind_speed_m_s)?;
    let wind_speed_height_m = crate::checked_copy(&wind_speed_height_m)?;
    let day_of_year = crate::checked_copy(&day_of_year)?;
    let soil_heat_flux_mj_m2_day = crate::checked_copy(&soil_heat_flux_mj_m2_day)?;
    let albedo = crate::checked_copy(&albedo)?;
    let (tdew_celsius, rh_min, rh_max, rh_mean) = (
        checked_optional(tdew_celsius)?,
        checked_optional(rh_min)?,
        checked_optional(rh_max)?,
        checked_optional(rh_mean)?,
    );
    let (solar_radiation_mj_m2_day, sunshine_hours) = (
        checked_optional(solar_radiation_mj_m2_day)?,
        checked_optional(sunshine_hours)?,
    );

    let humidity = match humidity_variant {
        HUMIDITY_DEWPOINT => HumidityPathway::Dewpoint {
            tdew_celsius: required("tdew_celsius", &tdew_celsius)?,
        },
        HUMIDITY_RH_MIN_MAX => HumidityPathway::MinimumAndMaximumRelativeHumidity {
            rh_min: required("rh_min", &rh_min)?,
            rh_max: required("rh_max", &rh_max)?,
        },
        HUMIDITY_RH_MAX => HumidityPathway::MaximumRelativeHumidity {
            rh_max: required("rh_max", &rh_max)?,
        },
        HUMIDITY_RH_MEAN => HumidityPathway::MeanRelativeHumidity {
            rh_mean: required("rh_mean", &rh_mean)?,
        },
        HUMIDITY_TMIN => HumidityPathway::MinimumTemperature,
        _ => return Err(PyValueError::new_err("unknown humidity pathway")),
    };
    let radiation = match radiation_variant {
        RADIATION_SUPPLIED => RadiationPathway::Supplied {
            solar_radiation_mj_m2_day: required(
                "solar_radiation_mj_m2_day",
                &solar_radiation_mj_m2_day,
            )?,
        },
        RADIATION_SUNSHINE => RadiationPathway::SunshineHours {
            sunshine_hours: required("sunshine_hours", &sunshine_hours)?,
        },
        RADIATION_TEMPERATURE_RANGE => RadiationPathway::TemperatureRange { coastal },
        _ => return Err(PyValueError::new_err("unknown radiation pathway")),
    };

    let inputs = MetInputs {
        daily_tmin_celsius: daily_tmin_celsius.view(),
        daily_tmax_celsius: daily_tmax_celsius.view(),
        latitude_degrees: latitude_degrees.view(),
        elevation_m: elevation_m.view(),
        wind_speed_m_s: wind_speed_m_s.view(),
        wind_speed_height_m: wind_speed_height_m.view(),
        day_of_year: day_of_year.view(),
        soil_heat_flux_mj_m2_day: soil_heat_flux_mj_m2_day.view(),
        albedo: albedo.view(),
        humidity,
        radiation,
    };
    py.detach(|| climate_core::pm_eto::penman_monteith_eto(&inputs))
        .map(|eto| eto.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Register the PET bindings in the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(thornthwaite, m)?)?;
    m.add_function(wrap_pyfunction!(hargreaves, m)?)?;
    m.add_function(wrap_pyfunction!(pm_eto, m)?)?;
    m.add_function(wrap_pyfunction!(fao56_eto, m)?)?;
    Ok(())
}
