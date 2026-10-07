//! PyO3/NumPy bindings that expose `climate-core` as `climate_indices._native`.
//!
//! This is the only crate that knows about Python. It converts arrays and
//! errors at the boundary and holds no climate algorithm of its own; the
//! Python package decides when to call it and falls back to its pure-Python
//! implementation when the extension is not installed. Inputs must already be
//! float64 arrays: extraction fails with `TypeError` rather than casting.

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Gamma shape and scale per column of a (years, columns) calibration block.
#[pyfunction]
fn gamma_parameters<'py>(
    py: Python<'py>,
    calibration: PyReadonlyArray2<'py, f64>,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    // Copy before `detach`: the kernel must not read caller-owned storage once the
    // GIL is released, or another Python thread could write it mid-read.
    let calibration = calibration.as_array().to_owned();
    let (alphas, betas) = py.detach(|| climate_core::gamma::gamma_parameters(calibration.view()));
    (alphas.into_pyarray(py), betas.into_pyarray(py))
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
    // Copy before `detach`: the kernel must not read caller-owned storage once the
    // GIL is released, or another Python thread could write it mid-read.
    let (values, alphas, betas, probabilities_of_zero) = (
        values.as_array().to_owned(),
        alphas.as_array().to_owned(),
        betas.as_array().to_owned(),
        probabilities_of_zero.as_array().to_owned(),
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

/// `scipy.stats.norm.ppf` applied element-wise to an array of any shape.
#[pyfunction]
fn norm_ppf<'py>(
    py: Python<'py>,
    probabilities: PyReadonlyArrayDyn<'py, f64>,
) -> Bound<'py, PyArrayDyn<f64>> {
    // Copy before `detach`: the kernel must not read caller-owned storage once the
    // GIL is released, or another Python thread could write it mid-read.
    let probabilities = probabilities.as_array().to_owned();
    py.detach(|| probabilities.mapv(climate_core::special::norm_ppf))
        .into_pyarray(py)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", climate_core::VERSION)?;
    m.add_function(wrap_pyfunction!(gamma_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(gamma_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(norm_ppf, m)?)?;
    Ok(())
}
