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

/// Version of this crate, re-exported by the Python extension as `__version__`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
