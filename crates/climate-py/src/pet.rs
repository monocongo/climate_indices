//! Array conversion for the PET kernels; the equations live in `climate-core`.
//!
//! Every array argument must be a float64 ndarray in the layout the matching
//! kernel documents. The kernels receive one element per broadcast position, so
//! the caller flattens the broadcast inputs and reshapes the result.

use numpy::ndarray::Array1;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use climate_core::pm_eto::{HumidityPathway, MetInputs, Operand, PmEtoInputs, RadiationPathway};

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

#[derive(FromPyObject)]
enum PyOperand<'py> {
    Array(PyReadonlyArray1<'py, f64>),
    Scalar(f64),
}

// Only actual arrays are copied; no caller-owned storage survives detach.
enum OwnedOperand {
    Array(Array1<f64>),
    Scalar(f64),
}

impl PyOperand<'_> {
    fn owned(self) -> PyResult<OwnedOperand> {
        match self {
            Self::Array(array) => crate::checked_copy(&array).map(OwnedOperand::Array),
            Self::Scalar(value) => Ok(OwnedOperand::Scalar(value)),
        }
    }
}

impl OwnedOperand {
    fn view(&self) -> Operand<'_> {
        match self {
            Self::Array(array) => Operand::Array(array.view()),
            Self::Scalar(value) => Operand::Scalar(*value),
        }
    }
}

fn checked_optional(value: Option<PyOperand<'_>>) -> PyResult<Option<OwnedOperand>> {
    value.map(PyOperand::owned).transpose()
}

fn required<'a>(argument: &str, value: &'a Option<OwnedOperand>) -> PyResult<Operand<'a>> {
    value.as_ref().map(OwnedOperand::view).ok_or_else(|| {
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
    // `checked_copy`'s empty-array guard: rust-numpy shifts the data pointer for
    // negative strides even on empty axes, so avoid building that view.
    let leap_years = if leap_years.is_empty() {
        Array1::<bool>::from_vec(Vec::new())
    } else {
        leap_years.as_array().to_owned()
    };
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
    net_radiation: PyOperand<'py>,
    soil_heat_flux: PyOperand<'py>,
    temperature_celsius: PyOperand<'py>,
    wind_speed_2m: PyOperand<'py>,
    saturation_vp: PyOperand<'py>,
    actual_vp: PyOperand<'py>,
    delta: PyOperand<'py>,
    gamma: PyOperand<'py>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let net_radiation = net_radiation.owned()?;
    let soil_heat_flux = soil_heat_flux.owned()?;
    let temperature_celsius = temperature_celsius.owned()?;
    let wind_speed_2m = wind_speed_2m.owned()?;
    let saturation_vp = saturation_vp.owned()?;
    let actual_vp = actual_vp.owned()?;
    let delta = delta.owned()?;
    let gamma = gamma.owned()?;
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
    daily_tmin_celsius: PyOperand<'py>,
    daily_tmax_celsius: PyOperand<'py>,
    latitude_degrees: PyOperand<'py>,
    elevation_m: PyOperand<'py>,
    wind_speed_m_s: PyOperand<'py>,
    wind_speed_height_m: PyOperand<'py>,
    day_of_year: PyOperand<'py>,
    soil_heat_flux_mj_m2_day: PyOperand<'py>,
    albedo: PyOperand<'py>,
    humidity_variant: u8,
    tdew_celsius: Option<PyOperand<'py>>,
    rh_min: Option<PyOperand<'py>>,
    rh_max: Option<PyOperand<'py>>,
    rh_mean: Option<PyOperand<'py>>,
    radiation_variant: u8,
    solar_radiation_mj_m2_day: Option<PyOperand<'py>>,
    sunshine_hours: Option<PyOperand<'py>>,
    coastal: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let daily_tmin_celsius = daily_tmin_celsius.owned()?;
    let daily_tmax_celsius = daily_tmax_celsius.owned()?;
    let latitude_degrees = latitude_degrees.owned()?;
    let elevation_m = elevation_m.owned()?;
    let wind_speed_m_s = wind_speed_m_s.owned()?;
    let wind_speed_height_m = wind_speed_height_m.owned()?;
    let day_of_year = day_of_year.owned()?;
    let soil_heat_flux_mj_m2_day = soil_heat_flux_mj_m2_day.owned()?;
    let albedo = albedo.owned()?;
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
