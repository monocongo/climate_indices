//! PyO3/NumPy bindings that expose `climate-core` as `climate_indices._native`.
//!
//! This is the only crate that knows about Python. It converts arrays and
//! errors at the boundary and holds no climate algorithm of its own; the
//! Python package decides when to call it and falls back to its pure-Python
//! implementation when the extension is not installed. Inputs must already be
//! float64 arrays: extraction fails with `TypeError` rather than casting.

use numpy::ndarray::{Array, CowArray, Dimension, ShapeBuilder};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

fn checked_view<'a, D: Dimension>(
    array: &'a PyReadonlyArray<'_, f64, D>,
) -> PyResult<CowArray<'a, f64, D>> {
    if !array.is_aligned() || !array.data().is_aligned() {
        return Err(PyValueError::new_err("unaligned float64 array"));
    }
    // rust-numpy normalizes negative strides by shifting the data pointer, even
    // on empty axes. Avoid creating a possibly unaligned/out-of-bounds view.
    if array.is_empty() {
        return Array::from_shape_vec(array.dims(), Vec::new())
            .map(CowArray::from)
            .map_err(|error| PyValueError::new_err(error.to_string()));
    }
    Ok(array.as_array().into())
}

fn checked_copy<D: Dimension>(array: &PyReadonlyArray<'_, f64, D>) -> PyResult<Array<f64, D>> {
    // Copy before `detach`: another Python thread may mutate caller-owned storage.
    Ok(checked_view(array)?.into_owned())
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

/// Calibration normals, one per column of a (years, periods*columns) block.
#[pyfunction]
fn pnp_normals<'py>(
    py: Python<'py>,
    calibration: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // np.nansum replaces NaNs in a K-order copy. Preserve its unit-stride year
    // axis even when ndarray's copy of a noncontiguous input is C-ordered.
    let strides = calibration.strides();
    let fortran = strides[0] != 0 && strides[0].unsigned_abs() < strides[1].unsigned_abs();
    let calibration = checked_copy(&calibration)?;
    let calibration = if fortran && calibration.strides()[0] != 1 {
        Array::from_shape_fn(calibration.raw_dim().f(), |index| calibration[index])
    } else {
        calibration
    };
    Ok(py
        .detach(|| climate_core::pnp::pnp_normals(calibration.view()))
        .into_pyarray(py))
}

/// Percentage of normal, element-wise or per cell of a (time, columns) block.
#[pyfunction]
fn pnp_percentages<'py>(
    py: Python<'py>,
    scale_sums: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    // A division per element costs about what copying `scale_sums` to release
    // the GIL would, and the copy would add an input-sized buffer: keep the GIL.
    let scale_sums = checked_view(&scale_sums)?;
    let normals = checked_view(&normals)?;
    climate_core::pnp::pnp_percentages(scale_sums.view(), normals.view())
        .map(|percentages| percentages.into_pyarray(py))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Precipitation Concentration Index of one year of daily rainfall.
#[pyfunction]
fn pci(rainfall: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let rainfall = checked_copy(&rainfall)?;
    climate_core::pci::pci(rainfall.view())
        .ok_or_else(|| PyValueError::new_err("pci requires a 365- or 366-day year"))
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", climate_core::VERSION)?;
    m.add_function(wrap_pyfunction!(gamma_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(pnp_normals, m)?)?;
    m.add_function(wrap_pyfunction!(pnp_percentages, m)?)?;
    m.add_function(wrap_pyfunction!(pci, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(loglogistic_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(loglogistic_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(norm_ppf, m)?)?;
    m.add_function(wrap_pyfunction!(tukey_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(hastings_inverse_normal, m)?)?;
    Ok(())
}
