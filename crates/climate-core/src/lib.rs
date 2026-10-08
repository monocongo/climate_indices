//! Pure-Rust numerical kernels for `climate_indices`.
//!
//! This crate knows nothing about Python: no PyO3, no NumPy bindings, no
//! Python exceptions. It receives already-validated numerical inputs and
//! returns numerical outputs. Validation, xarray/CF metadata, logging,
//! warnings, and I/O stay in the Python package; `climate-py` is the only
//! crate that converts between Python objects and the types used here.
//!
//! Every kernel is a port of an existing Python implementation, which stays
//! the reference oracle for parity tests. See `docs/architecture.md`.

use std::fmt;

pub mod eto;
pub mod gamma;
pub mod pm_eto;
pub mod special;

/// Version of this crate, re-exported by the Python extension as `__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// An input a kernel cannot compute with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClimateError {
    /// A per-column argument whose length is not the number of columns.
    ShapeMismatch {
        argument: &'static str,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for ClimateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                argument,
                expected,
                actual,
            } => write!(f, "{argument} has length {actual}, expected {expected}"),
        }
    }
}

impl std::error::Error for ClimateError {}
