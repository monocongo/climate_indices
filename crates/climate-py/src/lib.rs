//! PyO3/NumPy bindings that expose `climate-core` as `climate_indices._native`.
//!
//! This is the only crate that knows about Python. It converts arrays and
//! errors at the boundary and holds no climate algorithm of its own; the
//! Python package decides when to call it and falls back to its pure-Python
//! implementation when the extension is not installed. Inputs must already be
//! float64 arrays: extraction fails with `TypeError` rather than casting.

use numpy::ndarray::{Array, Dimension};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1,
    PyReadonlyArray2, PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type ParameterArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

fn checked_copy<D: Dimension>(array: &PyReadonlyArray<'_, f64, D>) -> PyResult<Array<f64, D>> {
    if !array.is_aligned() || !array.data().is_aligned() {
        return Err(PyValueError::new_err("unaligned float64 array"));
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
    let calibration = checked_copy(&calibration)?;
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
    let scale_sums = checked_copy(&scale_sums)?;
    let normals = checked_copy(&normals)?;
    py.detach(|| climate_core::pnp::pnp_percentages(scale_sums.view(), normals.view()))
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

/// `scipy.stats.norm.ppf` applied element-wise to an array of any shape.
#[pyfunction]
fn norm_ppf<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    if !probabilities.is_aligned() || !probabilities.data().is_aligned() {
        return Err(PyValueError::new_err("unaligned float64 array"));
    }
    // Flatten first: rust-numpy's ndarray conversion is limited to 32 dimensions.
    let shape = probabilities.shape().to_vec();
    let flat = probabilities
        .reshape_with_order([probabilities.len()], numpy::npyffi::NPY_ORDER::NPY_CORDER)?
        .readonly();
    let flat = checked_copy(&flat)?;
    py.detach(|| flat.mapv(climate_core::special::norm_ppf))
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
    m.add_function(wrap_pyfunction!(norm_ppf, m)?)?;
    Ok(())
}
