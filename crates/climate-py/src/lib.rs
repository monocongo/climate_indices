//! PyO3/NumPy bindings that expose `climate-core` as `climate_indices._native`.
//!
//! This is the only crate that knows about Python. It converts arrays and
//! errors at the boundary and holds no climate algorithm of its own; the
//! Python package decides when to call it and falls back to its pure-Python
//! implementation when the extension is not installed. Inputs must already be
//! float64 arrays: extraction fails with `TypeError` rather than casting.

use numpy::ndarray::{Array, Array1, Array2, Dimension};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use climate_core::fire::{DayLength, KbdiCell};
use climate_core::recurrence::RecurrenceInputs;

type ParameterArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
type PearsonArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
);
type LogLogisticArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<bool>>,
);

// A recurrence step produced a non-finite value from finite inputs.
//
// Distinct from `PyValueError` so the Python dispatch can raise the
// `InvalidArgumentError` the pure-Python driver raises, without inspecting a
// message to tell the two apart.
create_exception!(_native, NonFiniteResultError, PyValueError);

fn climate_error(error: climate_core::ClimateError) -> PyErr {
    let message = error.to_string();
    match error {
        climate_core::ClimateError::NonFinite { .. } => NonFiniteResultError::new_err(message),
        _ => PyValueError::new_err(message),
    }
}

/// Copy one input array, rejecting the layouts rust-numpy cannot view safely.
///
/// `element` names the dtype in the error, so a rejected float64 kernel input
/// keeps the message it has always raised.
fn copy_input<T: numpy::Element + Copy, D: Dimension>(
    array: &PyReadonlyArray<'_, T, D>,
    element: &str,
) -> PyResult<Array<T, D>> {
    if !array.is_aligned() || !array.data().is_aligned() {
        return Err(PyValueError::new_err(format!("unaligned {element} array")));
    }
    // rust-numpy normalizes negative strides by shifting the data pointer, even
    // on empty axes. Avoid creating a possibly unaligned/out-of-bounds view.
    if array.is_empty() {
        return Array::from_shape_vec(array.dims(), Vec::new())
            .map_err(|error| PyValueError::new_err(error.to_string()));
    }
    // Copy before `detach`: another Python thread may mutate caller-owned storage.
    Ok(array.as_array().to_owned())
}

/// A float64 kernel input: the dtype every `checked_copy` caller used to reach.
fn checked_copy<D: Dimension>(array: &PyReadonlyArray<'_, f64, D>) -> PyResult<Array<f64, D>> {
    copy_input(array, "float64")
}

/// A validity mask.
fn checked_copy_flags<D: Dimension>(
    array: &PyReadonlyArray<'_, bool, D>,
) -> PyResult<Array<bool, D>> {
    copy_input(array, "bool")
}

/// A calendar or gap-count array.
fn checked_copy_counts<D: Dimension>(
    array: &PyReadonlyArray<'_, i64, D>,
) -> PyResult<Array<i64, D>> {
    copy_input(array, "int64")
}

/// Gamma shape and scale per column of a (years, columns) calibration block.
#[pyfunction]
fn gamma_parameters<'py>(
    py: Python<'py>,
    calibration: PyReadonlyArray2<'py, f64>,
) -> PyResult<ParameterArrays<'py>> {
    let calibration = checked_copy(&calibration)?;
    let (alphas, betas) = py.detach(|| climate_core::gamma::gamma_parameters(calibration.view()));
    Ok((alphas.into_pyarray(py), betas.into_pyarray(py)))
}

/// Zero-inflated gamma CDF of a (years, columns) block, one parameter per column.
#[pyfunction]
fn gamma_probabilities<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    alphas: PyReadonlyArray1<'py, f64>,
    betas: PyReadonlyArray1<'py, f64>,
    probabilities_of_zero: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (values, alphas, betas, probabilities_of_zero) = (
        checked_copy(&values)?,
        checked_copy(&alphas)?,
        checked_copy(&betas)?,
        checked_copy(&probabilities_of_zero)?,
    );
    py.detach(|| {
        climate_core::gamma::gamma_probabilities(
            values.view(),
            alphas.view(),
            betas.view(),
            probabilities_of_zero.view(),
        )
    })
    .map(|probabilities| probabilities.into_pyarray(py))
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Pearson Type III probability of zero, loc, scale, and skew per column of a
/// (years, columns) calibration block, plus which columns could be fitted.
#[pyfunction]
fn pearson_parameters<'py>(
    py: Python<'py>,
    calibration: PyReadonlyArray2<'py, f64>,
) -> PyResult<PearsonArrays<'py>> {
    let calibration = checked_copy(&calibration)?;
    let fit = py.detach(|| climate_core::pearson::pearson_parameters(calibration.view()));
    Ok((
        fit.probabilities_of_zero.into_pyarray(py),
        fit.locs.into_pyarray(py),
        fit.scales.into_pyarray(py),
        fit.skews.into_pyarray(py),
        fit.valid.into_pyarray(py),
    ))
}

/// `scipy.stats.pearson3.cdf` of a (years, columns) block, one parameter set per column.
#[pyfunction]
fn pearson_cdf<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    skews: PyReadonlyArray1<'py, f64>,
    locs: PyReadonlyArray1<'py, f64>,
    scales: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (values, skews, locs, scales) = (
        checked_copy(&values)?,
        checked_copy(&skews)?,
        checked_copy(&locs)?,
        checked_copy(&scales)?,
    );
    py.detach(|| {
        climate_core::pearson::pearson_cdf_block(
            values.view(),
            skews.view(),
            locs.view(),
            scales.view(),
        )
    })
    .map(|cdf| cdf.into_pyarray(py))
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Generalized logistic loc, scale, and shape per column of a (years, columns)
/// calibration block, plus which columns could be fitted.
#[pyfunction]
fn loglogistic_parameters<'py>(
    py: Python<'py>,
    calibration: PyReadonlyArray2<'py, f64>,
) -> PyResult<LogLogisticArrays<'py>> {
    let calibration = checked_copy(&calibration)?;
    let fit = py.detach(|| climate_core::loglogistic::loglogistic_parameters(calibration.view()));
    Ok((
        fit.locs.into_pyarray(py),
        fit.scales.into_pyarray(py),
        fit.shapes.into_pyarray(py),
        fit.valid.into_pyarray(py),
    ))
}

/// Generalized logistic CDF of a (years, columns) block, one parameter set per column.
#[pyfunction]
fn loglogistic_cdf<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    locs: PyReadonlyArray1<'py, f64>,
    scales: PyReadonlyArray1<'py, f64>,
    shapes: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (values, locs, scales, shapes) = (
        checked_copy(&values)?,
        checked_copy(&locs)?,
        checked_copy(&scales)?,
        checked_copy(&shapes)?,
    );
    py.detach(|| {
        climate_core::loglogistic::loglogistic_cdf_block(
            values.view(),
            locs.view(),
            scales.view(),
            shapes.view(),
        )
    })
    .map(|cdf| cdf.into_pyarray(py))
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// `scipy.stats.norm.ppf` applied element-wise to an array of any shape.
#[pyfunction]
fn norm_ppf<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    elementwise(py, probabilities, climate_core::special::norm_ppf)
}

/// The Hastings inverse-normal approximation applied element-wise to an array of any shape.
#[pyfunction]
fn hastings_inverse_normal<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    elementwise(
        py,
        probabilities,
        climate_core::eddi::hastings_inverse_normal,
    )
}

/// Empirical rank count and Tukey plotting position for one calendar period.
#[pyfunction]
fn tukey_probabilities<'py>(
    py: Python<'py>,
    climatology: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    pads: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (climatology, values, pads) = (
        checked_copy(&climatology)?,
        checked_copy(&values)?,
        checked_copy(&pads)?,
    );
    py.detach(|| {
        climate_core::eddi::tukey_probabilities(climatology.view(), values.view(), pads.view())
    })
    .map(|probabilities| probabilities.into_pyarray(py))
    .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Apply a scalar kernel element-wise, preserving the input's shape.
fn elementwise<'py>(
    py: Python<'py>,
    values: PyReadonlyArrayDyn<'py, f64>,
    kernel: fn(f64) -> f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if !values.is_aligned() || !values.data().is_aligned() {
        return Err(PyValueError::new_err("unaligned float64 array"));
    }
    // Flatten first: rust-numpy's ndarray conversion is limited to 32 dimensions.
    let shape = values.shape().to_vec();
    let flat = values
        .reshape_with_order([values.len()], numpy::npyffi::NPY_ORDER::NPY_CORDER)?
        .readonly();
    let flat = checked_copy(&flat)?;
    py.detach(|| flat.mapv(kernel))
        .into_pyarray(py)
        .reshape(shape)
}

/// The recurrence bookkeeping every fire kernel takes, copied at the boundary.
struct RecurrenceArgs<'py> {
    weather_valid: PyReadonlyArray2<'py, bool>,
    static_valid: PyReadonlyArray1<'py, bool>,
    in_season: Option<PyReadonlyArray2<'py, bool>>,
    trailing_gap_days: PyReadonlyArray1<'py, i64>,
    spin_up: usize,
    nan_policy: String,
    max_gap_days: i64,
    record: bool,
}

/// The copied, owned form of [`RecurrenceArgs`], borrowed by the kernel call.
struct RecurrenceArrays {
    weather_valid: Array2<bool>,
    static_valid: Array1<bool>,
    in_season: Option<Array2<bool>>,
    trailing_gap_days: Array1<i64>,
    spin_up: usize,
    nan_policy: String,
    max_gap_days: i64,
    record: bool,
}

impl<'py> RecurrenceArgs<'py> {
    fn copy(&self) -> PyResult<RecurrenceArrays> {
        Ok(RecurrenceArrays {
            weather_valid: checked_copy_flags(&self.weather_valid)?,
            static_valid: checked_copy_flags(&self.static_valid)?,
            in_season: self
                .in_season
                .as_ref()
                .map(checked_copy_flags)
                .transpose()?,
            trailing_gap_days: checked_copy_counts(&self.trailing_gap_days)?,
            spin_up: self.spin_up,
            nan_policy: self.nan_policy.clone(),
            max_gap_days: self.max_gap_days,
            record: self.record,
        })
    }
}

impl RecurrenceArrays {
    fn inputs(&self) -> PyResult<RecurrenceInputs<'_>> {
        climate_core::fire::recurrence_inputs(
            self.weather_valid.view(),
            self.static_valid.view(),
            self.in_season.as_ref().map(|season| season.view()),
            self.trailing_gap_days.view(),
            self.spin_up,
            &self.nan_policy,
            self.max_gap_days,
        )
        .map_err(climate_error)
    }
}

/// One moisture code's run: its recorded history, final value, and gap counts.
type CodeRunArrays<'py> = (
    Option<Bound<'py, PyArray2<f64>>>,
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray1<i64>>>,
);

/// A KBDI run: its recorded history, final index and wet spell, and gap counts.
type KbdiRunArrays<'py> = (
    Option<Bound<'py, PyArray2<f64>>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Option<Bound<'py, PyArray1<i64>>>,
);

fn code_run_arrays<'py>(py: Python<'py>, run: climate_core::fire::CodeRun) -> CodeRunArrays<'py> {
    (
        run.values.map(|values| values.into_pyarray(py)),
        run.code.into_pyarray(py),
        run.trailing_gap_days.map(|gaps| gaps.into_pyarray(py)),
    )
}

/// The Fine Fuel Moisture Code over the whole time axis.
#[pyfunction]
#[pyo3(signature = (temperature_celsius, relative_humidity_percent, wind_speed_kilometers_per_hour, precipitation_mm, initial_ffmc, weather_valid, static_valid, in_season, trailing_gap_days, spin_up, nan_policy, max_gap_days, record))]
#[allow(clippy::too_many_arguments)]
fn ffmc<'py>(
    py: Python<'py>,
    temperature_celsius: PyReadonlyArray2<'py, f64>,
    relative_humidity_percent: PyReadonlyArray2<'py, f64>,
    wind_speed_kilometers_per_hour: PyReadonlyArray2<'py, f64>,
    precipitation_mm: PyReadonlyArray2<'py, f64>,
    initial_ffmc: PyReadonlyArray1<'py, f64>,
    weather_valid: PyReadonlyArray2<'py, bool>,
    static_valid: PyReadonlyArray1<'py, bool>,
    in_season: Option<PyReadonlyArray2<'py, bool>>,
    trailing_gap_days: PyReadonlyArray1<'py, i64>,
    spin_up: usize,
    nan_policy: String,
    max_gap_days: i64,
    record: bool,
) -> PyResult<CodeRunArrays<'py>> {
    let recurrence = RecurrenceArgs {
        weather_valid,
        static_valid,
        in_season,
        trailing_gap_days,
        spin_up,
        nan_policy,
        max_gap_days,
        record,
    }
    .copy()?;
    let temperature_celsius = checked_copy(&temperature_celsius)?;
    let relative_humidity_percent = checked_copy(&relative_humidity_percent)?;
    let wind = checked_copy(&wind_speed_kilometers_per_hour)?;
    let precipitation = checked_copy(&precipitation_mm)?;
    let initial_ffmc = checked_copy(&initial_ffmc)?;
    let inputs = recurrence.inputs()?;
    let run = py
        .detach(|| {
            climate_core::fire::ffmc(
                temperature_celsius.view(),
                relative_humidity_percent.view(),
                wind.view(),
                precipitation.view(),
                initial_ffmc.view(),
                &inputs,
                recurrence.record,
            )
        })
        .map_err(climate_error)?;
    Ok(code_run_arrays(py, run))
}

/// The Duff Moisture Code over the whole time axis.
#[pyfunction]
#[pyo3(signature = (temperature_celsius, relative_humidity_percent, precipitation_mm, day_length_table, months, day_length_band, initial_dmc, weather_valid, static_valid, in_season, trailing_gap_days, spin_up, nan_policy, max_gap_days, record))]
#[allow(clippy::too_many_arguments)]
fn duff_moisture_code<'py>(
    py: Python<'py>,
    temperature_celsius: PyReadonlyArray2<'py, f64>,
    relative_humidity_percent: PyReadonlyArray2<'py, f64>,
    precipitation_mm: PyReadonlyArray2<'py, f64>,
    day_length_table: PyReadonlyArray2<'py, f64>,
    months: PyReadonlyArray2<'py, i64>,
    day_length_band: PyReadonlyArray1<'py, i64>,
    initial_dmc: PyReadonlyArray1<'py, f64>,
    weather_valid: PyReadonlyArray2<'py, bool>,
    static_valid: PyReadonlyArray1<'py, bool>,
    in_season: Option<PyReadonlyArray2<'py, bool>>,
    trailing_gap_days: PyReadonlyArray1<'py, i64>,
    spin_up: usize,
    nan_policy: String,
    max_gap_days: i64,
    record: bool,
) -> PyResult<CodeRunArrays<'py>> {
    let recurrence = RecurrenceArgs {
        weather_valid,
        static_valid,
        in_season,
        trailing_gap_days,
        spin_up,
        nan_policy,
        max_gap_days,
        record,
    }
    .copy()?;
    let temperature_celsius = checked_copy(&temperature_celsius)?;
    let relative_humidity_percent = checked_copy(&relative_humidity_percent)?;
    let precipitation = checked_copy(&precipitation_mm)?;
    let day_length_table = checked_copy(&day_length_table)?;
    let months = checked_copy_counts(&months)?;
    let day_length_band = checked_copy_counts(&day_length_band)?;
    let day_length = DayLength {
        table: day_length_table.view(),
        band: day_length_band.view(),
        months: months.view(),
    };
    let initial_dmc = checked_copy(&initial_dmc)?;
    let inputs = recurrence.inputs()?;
    let run = py
        .detach(|| {
            climate_core::fire::dmc(
                temperature_celsius.view(),
                relative_humidity_percent.view(),
                precipitation.view(),
                &day_length,
                initial_dmc.view(),
                &inputs,
                recurrence.record,
            )
        })
        .map_err(climate_error)?;
    Ok(code_run_arrays(py, run))
}

/// The Drought Code over the whole time axis.
#[pyfunction]
#[pyo3(signature = (temperature_celsius, precipitation_mm, day_length_table, months, day_length_band, initial_dc, weather_valid, static_valid, in_season, trailing_gap_days, spin_up, nan_policy, max_gap_days, record))]
#[allow(clippy::too_many_arguments)]
fn drought_code<'py>(
    py: Python<'py>,
    temperature_celsius: PyReadonlyArray2<'py, f64>,
    precipitation_mm: PyReadonlyArray2<'py, f64>,
    day_length_table: PyReadonlyArray2<'py, f64>,
    months: PyReadonlyArray2<'py, i64>,
    day_length_band: PyReadonlyArray1<'py, i64>,
    initial_dc: PyReadonlyArray1<'py, f64>,
    weather_valid: PyReadonlyArray2<'py, bool>,
    static_valid: PyReadonlyArray1<'py, bool>,
    in_season: Option<PyReadonlyArray2<'py, bool>>,
    trailing_gap_days: PyReadonlyArray1<'py, i64>,
    spin_up: usize,
    nan_policy: String,
    max_gap_days: i64,
    record: bool,
) -> PyResult<CodeRunArrays<'py>> {
    let recurrence = RecurrenceArgs {
        weather_valid,
        static_valid,
        in_season,
        trailing_gap_days,
        spin_up,
        nan_policy,
        max_gap_days,
        record,
    }
    .copy()?;
    let temperature_celsius = checked_copy(&temperature_celsius)?;
    let precipitation = checked_copy(&precipitation_mm)?;
    let day_length_table = checked_copy(&day_length_table)?;
    let months = checked_copy_counts(&months)?;
    let day_length_band = checked_copy_counts(&day_length_band)?;
    let day_length = DayLength {
        table: day_length_table.view(),
        band: day_length_band.view(),
        months: months.view(),
    };
    let initial_dc = checked_copy(&initial_dc)?;
    let inputs = recurrence.inputs()?;
    let run = py
        .detach(|| {
            climate_core::fire::dc(
                temperature_celsius.view(),
                precipitation.view(),
                &day_length,
                initial_dc.view(),
                &inputs,
                recurrence.record,
            )
        })
        .map_err(climate_error)?;
    Ok(code_run_arrays(py, run))
}

/// KBDI over the whole time axis: history, final index and wet spell, gap counts.
#[pyfunction]
#[pyo3(signature = (precipitation_mm, maximum_temperature_celsius, mean_annual_precipitation_mm, initial_kbdi, initial_wet_spell_precipitation, weather_valid, static_valid, in_season, trailing_gap_days, spin_up, nan_policy, max_gap_days, record))]
#[allow(clippy::too_many_arguments)]
fn kbdi<'py>(
    py: Python<'py>,
    precipitation_mm: PyReadonlyArray2<'py, f64>,
    maximum_temperature_celsius: PyReadonlyArray2<'py, f64>,
    mean_annual_precipitation_mm: PyReadonlyArray1<'py, f64>,
    initial_kbdi: PyReadonlyArray1<'py, f64>,
    initial_wet_spell_precipitation: PyReadonlyArray1<'py, f64>,
    weather_valid: PyReadonlyArray2<'py, bool>,
    static_valid: PyReadonlyArray1<'py, bool>,
    in_season: Option<PyReadonlyArray2<'py, bool>>,
    trailing_gap_days: PyReadonlyArray1<'py, i64>,
    spin_up: usize,
    nan_policy: String,
    max_gap_days: i64,
    record: bool,
) -> PyResult<KbdiRunArrays<'py>> {
    let recurrence = RecurrenceArgs {
        weather_valid,
        static_valid,
        in_season,
        trailing_gap_days,
        spin_up,
        nan_policy,
        max_gap_days,
        record,
    }
    .copy()?;
    let precipitation = checked_copy(&precipitation_mm)?;
    let temperature = checked_copy(&maximum_temperature_celsius)?;
    let mean_annual_precipitation = checked_copy(&mean_annual_precipitation_mm)?;
    let initial_kbdi = checked_copy(&initial_kbdi)?;
    let initial_wet_spell_precipitation = checked_copy(&initial_wet_spell_precipitation)?;
    let initial_state: Vec<KbdiCell> = initial_kbdi
        .iter()
        .zip(&initial_wet_spell_precipitation)
        .map(|(&kbdi, &wet_spell_precipitation)| KbdiCell {
            kbdi,
            wet_spell_precipitation,
        })
        .collect();
    let inputs = recurrence.inputs()?;
    let run = py
        .detach(|| {
            climate_core::fire::kbdi(
                precipitation.view(),
                temperature.view(),
                mean_annual_precipitation.view(),
                &initial_state,
                &inputs,
                recurrence.record,
            )
        })
        .map_err(climate_error)?;
    Ok((
        run.values.map(|values| values.into_pyarray(py)),
        run.state
            .iter()
            .map(|cell| cell.kbdi)
            .collect::<Array1<f64>>()
            .into_pyarray(py),
        run.state
            .iter()
            .map(|cell| cell.wet_spell_precipitation)
            .collect::<Array1<f64>>()
            .into_pyarray(py),
        run.trailing_gap_days.map(|gaps| gaps.into_pyarray(py)),
    ))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", climate_core::VERSION)?;
    m.add(
        "NonFiniteResultError",
        m.py().get_type::<NonFiniteResultError>(),
    )?;
    m.add_function(wrap_pyfunction!(gamma_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(loglogistic_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(loglogistic_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(tukey_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(hastings_inverse_normal, m)?)?;
    m.add_function(wrap_pyfunction!(ffmc, m)?)?;
    m.add_function(wrap_pyfunction!(duff_moisture_code, m)?)?;
    m.add_function(wrap_pyfunction!(drought_code, m)?)?;
    m.add_function(wrap_pyfunction!(kbdi, m)?)?;
    Ok(())
}
