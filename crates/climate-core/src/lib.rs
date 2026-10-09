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

pub mod eddi;
pub mod eto;
pub mod fire;
pub mod flood;
pub mod gamma;
pub mod lmoments;
pub mod loglogistic;
pub mod pci;
pub mod pearson;
pub mod pm_eto;
pub mod pnp;
pub mod recurrence;
mod reduction;
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
    /// A calendar-period parameter with no calendar steps.
    EmptyPeriod { argument: &'static str },
    /// A recurrence step produced a non-finite value from finite inputs.
    ///
    /// The message is the Python one (`_advance_component`), so a caller can
    /// raise the same error it would have raised on the Python path.
    NonFinite { index_type: &'static str },
    /// A per-day or per-cell index outside the table it selects from.
    IndexOutOfRange {
        argument: &'static str,
        value: i64,
        minimum: i64,
        maximum: i64,
    },
    /// A missing-day policy name a kernel does not implement.
    UnknownNanPolicy { value: String },
    /// A row window (a Calibration Period) that does not lie inside its axis.
    RowsOutOfRange {
        argument: &'static str,
        start: usize,
        end: usize,
        length: usize,
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
            Self::EmptyPeriod { argument } => {
                write!(f, "{argument} must contain at least one calendar step")
            }
            Self::NonFinite { index_type } => {
                write!(
                    f,
                    "{index_type} produced a non-finite value from finite inputs."
                )
            }
            Self::IndexOutOfRange {
                argument,
                value,
                minimum,
                maximum,
            } => write!(
                f,
                "{argument} value {value} is outside the table rows [{minimum}, {maximum}]"
            ),
            Self::UnknownNanPolicy { value } => {
                write!(f, "unknown missing-day policy {value:?}")
            }
            Self::RowsOutOfRange {
                argument,
                start,
                end,
                length,
            } => write!(
                f,
                "{argument} rows [{start}, {end}) are outside the {length} available rows"
            ),
        }
    }
}

impl std::error::Error for ClimateError {}
