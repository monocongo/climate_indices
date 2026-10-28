# The Rust API retains borrowed ndarray views and typed, extensible errors

## Status

Amended.
This record implements
[CRATE-03 — Design the public Rust API surface](https://github.com/monocongo/climate_indices/issues/1307).
The maintainer approved retaining borrowed ndarray views in the public Rust API.
{doc}`0021-climate-indices-crate-identity` defines the package identity.
CRATE-05 ([#1309](https://github.com/monocongo/climate_indices/issues/1309)) renamed the
package from `climate-core` to `climate-indices` and moved it to
`crates/climate-indices`, so the import path below is `climate_indices`.

[RUSTUP-03 (#1433)](https://github.com/monocongo/climate_indices/issues/1433)
amends decisions 2–3: Timescale Aggregation is approved for implementation;
complete index workflows remain outside the crate for 3.1.1.
The amendment itself added no kernel.
[RUSTUP-13 (#1485)](https://github.com/monocongo/climate_indices/issues/1485)
implements `aggregation::trailing_sum` and `aggregation::trailing_mean`.
Callers still select the Timescale and prepare calendars; complete workflows remain outside the crate.

## Context

The block kernels already accept borrowed arrays and return owned numerical results.
Other public kernels use scalars or slices.
Their public modules include distribution fitting, PET, Palmer, fire, flood,
and the shared daily recurrence engine.
Python supplies calendar preparation, physical-range validation, and index orchestration through `climate-py`.
Rust callers need these responsibilities and the array conventions stated without requiring Python knowledge.

Hiding ndarray would require new types or conversions at every kernel boundary.
Keeping it exposes a dependency version in the public API.
The approved choice accepts that compatibility cost and preserves the existing kernels.

## Decision

1. **Retain the existing public kernel modules and signatures.**
   Block-kernel inputs use borrowed ndarray views; outputs use owned arrays and existing result structs.
   Existing scalar and slice-based APIs retain their signatures, including scalar and `Option<f64>` results.
   The crate does not re-export ndarray or add array wrapper types.
   Callers depend on a compatible ndarray version, currently `0.17.2`.
   An incompatible ndarray upgrade requires a breaking crate release.
   On the independent `0.x` track, that means changing the minor version, not the patch version.

2. **Use the same block kernels for one series and a grid.**
   Daily kernels such as `flood::effective_precipitation` accept `(days, cells)`.
   A series uses one cell, obtained from a borrowed 1-D view with `insert_axis(Axis(1))`.
   A grid assigns one independent cell to each column.
   Distribution kernels accept `(years, columns)` and fit each column independently.
   Each column represents one calendar step of one cell.
   `column = step * cells + cell` is one supported packing convention, not a required memory layout.
   Callers must use the same packing for values and fit parameters.
   Strided views remain supported where the function does not document a restriction.
   Standard Palmer recursion accepts `(months, cells)`; Wells recursion remains per-series.
   Thornthwaite and Palmer water-balance blocks retain `(years, 12, cells)`.
   Complete SPI/SPEI/standardized-index workflows remain outside the crate for 3.1.1.
   Callers compose the kernels; the crate adds no duplicate per-series wrappers.

3. **State units and calendar responsibilities explicitly.**
   Precipitation and PET kernels use mm, and temperature inputs use degrees C.
   Palmer water-balance quantities instead use inches.
   PET latitude uses radians in `eto` and degrees in `pm_eto`.
   Function documentation defines other units, including radiation and wind speed.
   Array indices are zero-based; day-of-year and calendar-month inputs are one-based where documented.
   Monthly calendar layouts contain 12 steps per year.
   Prepared all-leap daily layouts contain 366 positions per year, including February 29 in every year.
   Callers select the Timescale and Calibration Period and supply calendar padding, clipping, and the final index transformation.
   The implemented exception is the Timescale Aggregation module over `(steps, cells)`, added by RUSTUP-13 (#1485).
   It computes trailing window sums or means; a mean is the window sum divided by `scale`.
   The output retains the input shape, with the leading `scale - 1` positions set to NaN.
   A window containing a missing value is NaN; missing values are never skipped.
   Kernels retain their current structural checks and kernel-specific NaN behavior.
   They do not perform general physical-range validation.

4. **Keep `ClimateError` as the public error type and make it non-exhaustive.**
   Existing variants and typed fields define error categories:
   `ShapeMismatch`, `EmptyPeriod`, `NonFinite`, `IndexOutOfRange`,
   `UnknownNanPolicy`, `RowsOutOfRange`, `NoConvergence`, and `InsufficientData`.
   Removing a variant or changing a typed field incompatibly requires a breaking release.
   Callers match variants and fields, not `Display` output or diagnostic strings.
   In particular, `NoConvergence.message` is diagnostic context, not a stable failure identifier.
   Diagnostic text has no Rust compatibility promise and no Python-message requirement.
   This change preserves existing messages and Python exception translation.
   `climate-py` selects exception classes by variant, without parsing message text.
   A future diagnostic change must separately preserve the Python contract at that boundary.

5. **Make every public enum non-exhaustive.**
   Besides `ClimateError`, this covers `recurrence::MissingDayPolicy`,
   `self_calibration::Spell`, and `pm_eto::{Operand, HumidityPathway, RadiationPathway}`.
   Downstream matches need a wildcard arm, while callers can construct the existing variants.
   Rust callers can build `RecurrenceInputs` with a typed `MissingDayPolicy` directly.
   The string-based `recurrence_inputs` helper remains available for bindings.
   Existing public structs retain their fields and remain directly constructible.

6. **Enforce safety and documentation at the crate boundary.**
   `climate-indices` enables `#![forbid(unsafe_code)]` and `#![deny(missing_docs)]`.
   The unsafe-code restriction applies to this crate, not its dependencies.
   ndarray's `s!` macro contains an `allow(unsafe_code)` attribute incompatible with `forbid`.
   Equivalent safe axis-slicing methods replace those macro calls, including in tests.
   The replacement preserves the selected elements, strides, and arithmetic order.
   Public field documentation states the existing meanings rather than changing the fields.

7. **Add no feature flags.**
   All kernels remain available with no default or optional features.
   There is no demonstrated need for serde, parallel execution, or another optional dependency here.
   Python bindings remain in the separate, unpublished `climate-py` crate.

## Timescale Aggregation and workflow decision (RUSTUP-03)

**Q1: accept Timescale Aggregation as a numerical kernel.**
`compute.sum_to_scale` currently uses `np.convolve` for a series and a strided
sliding-window reduction for a Spatial Block.
`compute.prepare_scaled` divides the sum by `scale` for a mean.
The SPI and SPEI Rust examples each implemented their own `trailing_sum`.
The shared block kernel now removes that duplication without taking ownership of calendars or index policy.

The [#1177 hourly benchmark](https://github.com/monocongo/climate_indices/blob/main/benchmarks/README.md#hourly-standardized-energy-indices-1177)
reports about 2.2 s and 150 MiB peak allocation for a 25-cell, 41-year block at a 720-hour Timescale.
It attributes the cost to re-summing each window; the work grows with the Timescale.
This is existing benchmark evidence, not a measurement or speedup from this amendment.

The separate port must preserve {doc}`0017-rust-core-acceleration-backend` decisions 4–6.
Its parity checks must establish the summation order for both NumPy paths before selecting an optimization.
A running sum or prefix subtraction is not equivalent under cancellation and can retain errors after a missing value leaves the window.
Signed, extreme-magnitude windows need measured absolute and relative errors, not an assumed `1e-10` bound.
Any tolerance exception requires the measurements and justification that ADR-0017 specifies.

**Q2: decline complete index workflows for 3.1.1.**
Calibration Period resolution, Pearson-to-gamma fallback, Zero Handling, data-quality warnings, and clipping remain caller responsibilities.
Moving them into `spi()`/`spei()`/standardized-index crate functions would duplicate Python orchestration and conflict with ADR-0017 decision 3.
The runnable Rust examples remain the workflow guidance.

## Compiling sketch and validation

The crate-level rustdoc in `crates/climate-indices/src/lib.rs` demonstrates a single-series call,
a grid call, calendar-column fitting, and typed error handling.
The `MissingDayPolicy` rustdoc demonstrates downstream matching with a wildcard arm.
These examples use the `climate_indices` import path and a direct ndarray dependency.
`cargo test -p climate-indices --doc` compiles and executes them as downstream callers.
The synthetic fit sample illustrates shape conventions, not a sufficient scientific Calibration Period.

The workspace build and native Python parity tests verify that `climate-py` still consumes the surface.
No kernel changes its numerical expressions or operation order.
This record makes no new measured accuracy claim.
The RUSTUP-03 amendment changed no numerical code.
RUSTUP-13 adds independent-window reductions with the existing operation orders, not a running sum.
Its dispatch retains longer series convolutions and non-C-contiguous Python blocks rather than changing their reduction order.
The parity registry, numerical contract, and `benchmarks/README.md` record its separate accuracy and performance measurements.

## Alternatives considered

- **Hide ndarray behind crate-owned types.** This adds conversions or a parallel array API without a demonstrated requirement.
- **Re-export ndarray.** This changes import convenience, but does not remove the exposed dependency's compatibility cost.
- **Add separate series and grid wrappers.** Borrowed singleton-axis views already support one series without copying its data.
- **Stabilize diagnostic text.** Typed variants support control flow without freezing prose or binding Rust users to Python messages.
- **Add feature flags speculatively.** This increases the supported build combinations without a current user requirement.

## Consequences and follow-up ownership

Adding `non_exhaustive` intentionally requires wildcard arms in downstream exhaustive matches.
The existing Python bindings already use a wildcard for `ClimateError` and still construct the other enums normally.
The package is now `climate-indices`; CRATE-05 performed the rename and added the
packaging metadata, and the first real crate release stays with
[CRATE-11](https://github.com/monocongo/climate_indices/issues/1315).

- [CRATE-04 — Numerical contract](https://github.com/monocongo/climate_indices/issues/1308) owns measured per-kernel accuracy contracts.
- [CRATE-05 — Packaging](https://github.com/monocongo/climate_indices/issues/1309) performed the package, directory, and import rename to `climate-indices`,
  and updated the compiling rustdoc imports with it.
- [CRATE-06 — Standalone validation](https://github.com/monocongo/climate_indices/issues/1310) owns standalone reference fixtures.
- [CRATE-07 — Rust-user documentation](https://github.com/monocongo/climate_indices/issues/1311) owns full examples and guides beyond this API sketch.
- [RUSTUP-03](https://github.com/monocongo/climate_indices/issues/1433) owns the Aggregation decision and the follow-up port issue under
  [RUSTUP-00](https://github.com/monocongo/climate_indices/issues/1430).
  The port must include the kernel module, binding, stub, plain-float64 dispatch in `compute.sum_to_scale`, and parity registry entries.
  It must repeat the #1177 hourly benchmark and update the numerical contract and both CHANGELOGs.
  The port depends on RUSTUP-03 and must block [RUSTUP-12](https://github.com/monocongo/climate_indices/issues/1442).
