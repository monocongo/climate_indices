//! Array conversion for the Palmer-family kernels; the algorithms live in `climate-core`.
//!
//! Every array argument must be a float64 ndarray in the layout the matching
//! kernel documents: monthly arrays `(n_years, 12, n_cells)`, monthly
//! coefficients `(12, n_cells)`. `palmer.py` reshapes to and from these.

use numpy::ndarray::Array3;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use climate_core::palmer_pdi::PdiDurationFactors;
use climate_core::palmer_wells::WellsFactors;
use climate_core::palmer_zindex::CafecInputs;
use climate_core::self_calibration::Spell;

use crate::{checked_copy, climate_error};

type Monthly<'py> = Bound<'py, PyArray3<f64>>;
type Sums<'py> = Bound<'py, PyArray2<f64>>;
type WaterBalanceArrays<'py> = (
    (
        Monthly<'py>,
        Monthly<'py>,
        Monthly<'py>,
        Monthly<'py>,
        Monthly<'py>,
        Monthly<'py>,
        Monthly<'py>,
        Monthly<'py>,
        Monthly<'py>,
    ),
    (
        Sums<'py>,
        Sums<'py>,
        Sums<'py>,
        Sums<'py>,
        Sums<'py>,
        Sums<'py>,
        Sums<'py>,
        Sums<'py>,
        Sums<'py>,
    ),
);
type IndexArrays<'py, A> = (A, A, A);

/// The Palmer water balance: the nine monthly arrays
/// `(spdat, pldat, prdat, rdat, tldat, etdat, rodat, sssdat, ssudat)` and the
/// nine calibration-period sums
/// `(psum, spsum, petsum, plsum, prsum, rsum, tlsum, etsum, rosum)`.
#[pyfunction]
fn palmer_water_balance<'py>(
    py: Python<'py>,
    precips: PyReadonlyArray3<'py, f64>,
    pet: PyReadonlyArray3<'py, f64>,
    awc: PyReadonlyArray1<'py, f64>,
    calibration_year_initial_idx: usize,
    calibration_year_final_idx: usize,
) -> PyResult<WaterBalanceArrays<'py>> {
    let (precips, pet, awc) = (
        checked_copy(&precips)?,
        checked_copy(&pet)?,
        checked_copy(&awc)?,
    );
    let out = py
        .detach(|| {
            climate_core::palmer::water_balance(
                precips.view(),
                pet.view(),
                awc.view(),
                calibration_year_initial_idx,
                calibration_year_final_idx,
            )
        })
        .map_err(climate_error)?;
    Ok((
        (
            out.spdat.into_pyarray(py),
            out.pldat.into_pyarray(py),
            out.prdat.into_pyarray(py),
            out.rdat.into_pyarray(py),
            out.tldat.into_pyarray(py),
            out.etdat.into_pyarray(py),
            out.rodat.into_pyarray(py),
            out.sssdat.into_pyarray(py),
            out.ssudat.into_pyarray(py),
        ),
        (
            out.psum.into_pyarray(py),
            out.spsum.into_pyarray(py),
            out.petsum.into_pyarray(py),
            out.plsum.into_pyarray(py),
            out.prsum.into_pyarray(py),
            out.rsum.into_pyarray(py),
            out.tlsum.into_pyarray(py),
            out.etsum.into_pyarray(py),
            out.rosum.into_pyarray(py),
        ),
    ))
}

/// Owned copies of the arrays a month's CAFEC precipitation reads.
struct CafecArrays {
    monthly: [Array3<f64>; 5],
    coefficients: [numpy::ndarray::Array2<f64>; 4],
}

impl CafecArrays {
    fn copy(
        monthly: [&PyReadonlyArray3<'_, f64>; 5],
        coefficients: [&PyReadonlyArray2<'_, f64>; 4],
    ) -> PyResult<Self> {
        Ok(Self {
            monthly: [
                checked_copy(monthly[0])?,
                checked_copy(monthly[1])?,
                checked_copy(monthly[2])?,
                checked_copy(monthly[3])?,
                checked_copy(monthly[4])?,
            ],
            coefficients: [
                checked_copy(coefficients[0])?,
                checked_copy(coefficients[1])?,
                checked_copy(coefficients[2])?,
                checked_copy(coefficients[3])?,
            ],
        })
    }

    fn inputs(&self) -> CafecInputs<'_> {
        let [precips, pet, prdat, spdat, pldat] = &self.monthly;
        let [alpha, beta, gamma, delta] = &self.coefficients;
        CafecInputs {
            precips: precips.view(),
            pet: pet.view(),
            prdat: prdat.view(),
            spdat: spdat.view(),
            pldat: pldat.view(),
            alpha: alpha.view(),
            beta: beta.view(),
            gamma: gamma.view(),
            delta: delta.view(),
        }
    }
}

/// Monthly mean absolute departure (`dbar`) and raw K-prime factors.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn palmer_k_prime<'py>(
    py: Python<'py>,
    precips: PyReadonlyArray3<'py, f64>,
    pet: PyReadonlyArray3<'py, f64>,
    prdat: PyReadonlyArray3<'py, f64>,
    spdat: PyReadonlyArray3<'py, f64>,
    pldat: PyReadonlyArray3<'py, f64>,
    alpha: PyReadonlyArray2<'py, f64>,
    beta: PyReadonlyArray2<'py, f64>,
    gamma: PyReadonlyArray2<'py, f64>,
    delta: PyReadonlyArray2<'py, f64>,
    trat: PyReadonlyArray2<'py, f64>,
    calibration_year_initial_idx: usize,
    calibration_year_final_idx: usize,
) -> PyResult<(Sums<'py>, Sums<'py>)> {
    let arrays = CafecArrays::copy(
        [&precips, &pet, &prdat, &spdat, &pldat],
        [&alpha, &beta, &gamma, &delta],
    )?;
    let trat = checked_copy(&trat)?;
    let (dbar, k_prime) = py
        .detach(|| {
            climate_core::palmer_zindex::k_prime_and_dbar(
                &arrays.inputs(),
                trat.view(),
                calibration_year_initial_idx,
                calibration_year_final_idx,
            )
        })
        .map_err(climate_error)?;
    Ok((dbar.into_pyarray(py), k_prime.into_pyarray(py)))
}

/// The K-weighted Z-index for every month of the record.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn palmer_raw_zindex<'py>(
    py: Python<'py>,
    precips: PyReadonlyArray3<'py, f64>,
    pet: PyReadonlyArray3<'py, f64>,
    prdat: PyReadonlyArray3<'py, f64>,
    spdat: PyReadonlyArray3<'py, f64>,
    pldat: PyReadonlyArray3<'py, f64>,
    alpha: PyReadonlyArray2<'py, f64>,
    beta: PyReadonlyArray2<'py, f64>,
    gamma: PyReadonlyArray2<'py, f64>,
    delta: PyReadonlyArray2<'py, f64>,
    ak: PyReadonlyArray2<'py, f64>,
) -> PyResult<Monthly<'py>> {
    let arrays = CafecArrays::copy(
        [&precips, &pet, &prdat, &spdat, &pldat],
        [&alpha, &beta, &gamma, &delta],
    )?;
    let ak = checked_copy(&ak)?;
    py.detach(|| climate_core::palmer_zindex::raw_zindex(&arrays.inputs(), ak.view()))
        .map(|z| z.into_pyarray(py))
        .map_err(climate_error)
}

/// The `pdi.f` recursion over a `(n_months, n_cells)` Z-index block.
#[pyfunction]
fn palmer_pdi<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    wetm: f64,
    wetb: f64,
    drym: f64,
    dryb: f64,
) -> PyResult<IndexArrays<'py, Sums<'py>>> {
    let z = checked_copy(&z)?;
    let factors = PdiDurationFactors {
        wetm,
        wetb,
        drym,
        dryb,
    };
    let result = py.detach(|| climate_core::palmer_pdi::calculate(z.view(), factors));
    Ok((
        result.pdsi.into_pyarray(py),
        result.phdi.into_pyarray(py),
        result.pmdi.into_pyarray(py),
    ))
}

/// The Wells recursion over a Z-index series, with Python-derived coefficients.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn palmer_wells<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    wetm: f64,
    wetb: f64,
    drym: f64,
    dryb: f64,
    wet_denominator: f64,
    dry_denominator: f64,
    wetc: f64,
    dryc: f64,
    dry_spell_c: f64,
) -> PyResult<IndexArrays<'py, Bound<'py, PyArray1<f64>>>> {
    let z = checked_copy(&z)?;
    let factors = WellsFactors {
        wetm,
        wetb,
        drym,
        dryb,
        wet_denominator,
        dry_denominator,
        wetc,
        dryc,
        dry_spell_c,
    };
    let result = py
        .detach(|| climate_core::palmer_wells::calculate(z.view(), &factors))
        .map_err(climate_error)?;
    Ok((
        result.pdsi.into_pyarray(py),
        result.phdi.into_pyarray(py),
        result.pmdi.into_pyarray(py),
    ))
}

/// scPDSI duration factors `(m, b)` for a calibration Z series; `sign` is
/// `self_calibration.WET_SIGN` (1) or `DRY_SIGN` (-1).
#[pyfunction]
fn scpdsi_duration_factors(
    py: Python<'_>,
    z: PyReadonlyArray1<'_, f64>,
    sign: i64,
) -> PyResult<(f64, f64)> {
    let spell = match sign {
        1 => Spell::Wet,
        -1 => Spell::Dry,
        _ => return Err(PyValueError::new_err(format!("invalid spell sign: {sign}"))),
    };
    let z = checked_copy(&z)?;
    let z = z.to_vec();
    py.detach(|| climate_core::self_calibration::duration_factors(&z, spell))
        .map_err(climate_error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(palmer_water_balance, m)?)?;
    m.add_function(wrap_pyfunction!(palmer_k_prime, m)?)?;
    m.add_function(wrap_pyfunction!(palmer_raw_zindex, m)?)?;
    m.add_function(wrap_pyfunction!(palmer_pdi, m)?)?;
    m.add_function(wrap_pyfunction!(palmer_wells, m)?)?;
    m.add_function(wrap_pyfunction!(scpdsi_duration_factors, m)?)?;
    Ok(())
}
