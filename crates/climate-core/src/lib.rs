//! Pure-Rust numerical kernels for `climate_indices`.
//!
//! The public modules expose numerical kernels, not a complete SPI/SPEI workflow.
//! Callers prepare calendars, select the Calibration Period, validate physical
//! ranges, and apply any required aggregation and final index transformation.
//! Kernels check the structural constraints documented on each function.
//! This crate has no Python, I/O, logging, or metadata dependency.
//!
//! # Arrays and compatibility
//!
//! Array inputs borrow `ndarray` views; array outputs own their data.
//! Other kernels accept scalars or slices and return their documented result types.
//! Strided views are supported unless a function documents a restriction.
//! `ndarray` is a public dependency, not a re-export: callers use a compatible
//! `ndarray` dependency.
//! An incompatible `ndarray` upgrade requires a breaking crate release.
//! The crate has no feature flags and no optional runtime dependency.
//!
//! # Shapes, calendars, and units
//!
//! All array indices are zero-based. A daily block has shape `(days, cells)`.
//! One series uses one cell; a grid uses one column per independent cell.
//! Distribution fits use `(years, columns)`, with one column per calendar step
//! and cell. A caller can use `column = step * cells + cell` consistently for
//! values and fit parameters. Monthly records have 12 calendar steps per year;
//! the prepared all-leap daily layout has 366, including a February 29 position
//! in every year. Kernels do not infer calendar dates or insert missing days.
//!
//! Precipitation and PET kernels use mm, temperatures use degrees C, and daily
//! window lengths count calendar days. PET latitude inputs use radians in
//! [`eto`] and degrees in [`pm_eto`]. Palmer water-balance inputs instead use
//! inches. Function documentation specifies other units and preconditions.
//! NaN handling is kernel-specific; NaN is not a substitute for shape validation.
//!
//! # One series and a grid
//!
//! [`flood::effective_precipitation`] accepts both layouts without a wrapper or
//! a copy of the input. Its output has the same shape as the input.
//!
//! ```
//! use climate_core::{ClimateError, flood::effective_precipitation};
//! use ndarray::{Axis, array};
//!
//! let series_mm = array![1.0, 2.0, 3.0];
//! let series = effective_precipitation(series_mm.view().insert_axis(Axis(1)), 2)?;
//! assert_eq!(series.dim(), (3, 1));
//! assert!(series[[0, 0]].is_nan());
//! assert_eq!(series[[1, 0]], 3.5);
//!
//! let grid_mm = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
//! let grid = effective_precipitation(grid_mm.view(), 2)?;
//! assert_eq!(grid.dim(), (3, 2));
//! assert_eq!(grid[[1, 0]], series[[1, 0]]);
//! assert_eq!(grid[[1, 1]], 35.0);
//! # Ok::<(), ClimateError>(())
//! ```
//!
//! # Calendar-column fitting
//!
//! [`gamma::gamma_parameters`] fits each column independently. This small
//! synthetic sample illustrates the layout, not a sufficient Calibration Period.
//! Gamma shape is dimensionless; gamma scale has the input's units (here mm).
//! The result below is a probability, not SPI.
//!
//! ```
//! use climate_core::{ClimateError, gamma::{gamma_parameters, gamma_probabilities}};
//! use ndarray::{Array1, Array2};
//!
//! let (years, steps, cells) = (3, 12, 2);
//! let rain_mm = Array2::from_shape_fn((years, steps * cells), |(year, column)| {
//!     10.0 * (year + column + 1) as f64
//! });
//! let (shape, scale_mm) = gamma_parameters(rain_mm.view());
//! let probability_of_zero = Array1::zeros(steps * cells);
//! let probabilities = gamma_probabilities(
//!     rain_mm.view(), shape.view(), scale_mm.view(), probability_of_zero.view(),
//! )?;
//! let (year, step, cell) = (1, 5, 1);
//! assert!((0.0..=1.0).contains(&probabilities[[year, step * cells + cell]]));
//! # Ok::<(), ClimateError>(())
//! ```
//!
//! # Errors and numerical behavior
//!
//! Match [`ClimateError`] variants and fields, not diagnostic text. Public enums
//! are non-exhaustive, so downstream matches need a wildcard arm. Display text
//! and diagnostic strings are not stable API and carry no Python-message promise.
//! `climate-py` owns Python exception translation. The Python implementations
//! remain the numerical parity oracles. Each public kernel documents its numerical
//! contract; the measured evidence below distinguishes workflow and scalar errors.
//! No kernel's operation order changes with this surface.

#![doc = include_str!("../NUMERICS.md")]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::fmt;

pub mod eddi;
pub mod eto;
pub mod fire;
pub mod flood;
pub mod gamma;
pub mod lmoments;
pub mod loglogistic;
pub mod palmer;
pub mod palmer_pdi;
pub mod palmer_wells;
pub mod palmer_zindex;
pub mod pci;
pub mod pearson;
pub mod pm_eto;
pub mod pnp;
pub mod recurrence;
mod reduction;
pub mod self_calibration;
pub mod special;

/// Version of this crate, re-exported by the Python extension as `__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A structural input error or a numerical failure reported by a kernel.
///
/// Variants and their typed fields form the error contract. Diagnostic strings,
/// including [`Display`](fmt::Display) output, are not stable identifiers.
///
/// ```
/// use climate_core::{ClimateError, flood::effective_precipitation};
/// use ndarray::array;
///
/// let rain = array![[1.0]];
/// match effective_precipitation(rain.view(), 0) {
///     Err(ClimateError::EmptyPeriod { argument }) => assert_eq!(argument, "duration"),
///     Err(_) => panic!("unexpected error"),
///     Ok(_) => panic!("a zero-day window must fail"),
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClimateError {
    /// An argument whose shape does not match the kernel's expected shape.
    ShapeMismatch {
        /// The argument that has the wrong shape.
        argument: &'static str,
        /// The expected axis length or total element count, as the kernel specifies.
        expected: usize,
        /// The supplied axis length or total element count.
        actual: usize,
    },
    /// A calendar-period parameter with no calendar steps.
    EmptyPeriod {
        /// The empty period argument.
        argument: &'static str,
    },
    /// A recurrence step produced a non-finite value from finite inputs.
    NonFinite {
        /// The recurrence that failed.
        index_type: &'static str,
    },
    /// A per-day or per-cell index outside the table it selects from.
    IndexOutOfRange {
        /// The table-index argument.
        argument: &'static str,
        /// The supplied index.
        value: i64,
        /// The inclusive lower bound.
        minimum: i64,
        /// The inclusive upper bound.
        maximum: i64,
    },
    /// A missing-day policy name a kernel does not implement.
    UnknownNanPolicy {
        /// The unrecognized policy name.
        value: String,
    },
    /// A row window (a Calibration Period) that does not lie inside its axis.
    RowsOutOfRange {
        /// The row-window argument.
        argument: &'static str,
        /// The inclusive first row.
        start: usize,
        /// The exclusive last row.
        end: usize,
        /// The number of available rows.
        length: usize,
    },
    /// A calibration or recurrence stage that cannot produce a usable value.
    NoConvergence {
        /// A diagnostic description, not a stable failure identifier.
        message: &'static str,
    },
    /// A series too short (or too sparse) to fill a rolling window.
    InsufficientData {
        /// The minimum number of periods the window needs.
        required: usize,
        /// The number of non-missing periods supplied.
        available: usize,
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
            Self::NoConvergence { message } => f.write_str(message),
            Self::InsufficientData {
                required,
                available,
            } => write!(
                f,
                "no complete {required}-period rolling window: only {available} non-missing periods were available"
            ),
        }
    }
}

impl std::error::Error for ClimateError {}
