# The Rust API retains borrowed ndarray views and typed, extensible errors

## Status

Accepted. This record implements
[CRATE-03 — Design the public Rust API surface](https://github.com/monocongo/climate_indices/issues/1307).
The maintainer approved retaining borrowed ndarray views in the public Rust API.
{doc}`0021-climate-indices-crate-identity` defines the planned package identity.
The current package remains `climate-core`; CRATE-05 owns its rename.

## Context

The kernels already accept borrowed arrays and return owned numerical results.
Their public modules include distribution fitting, PET, Palmer, fire, flood,
and the shared daily recurrence engine.
Python supplies calendar preparation, physical-range validation, and index orchestration through `climate-py`.
Rust callers need these responsibilities and the array conventions stated without requiring Python knowledge.

Hiding ndarray would require new types or conversions at every kernel boundary.
Keeping it exposes a dependency version in the public API.
The approved choice accepts that compatibility cost and preserves the existing kernels.

## Decision

1. **Retain the existing public kernel modules and signatures.**
   Inputs use borrowed ndarray views; outputs use owned arrays and existing result structs.
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
   This task adds no complete SPI/SPEI workflow or duplicate per-series wrappers.

3. **State units and calendar responsibilities explicitly.**
   Precipitation and PET kernels use mm, and temperature inputs use degrees C.
   Palmer water-balance quantities instead use inches.
   PET latitude uses radians in `eto` and degrees in `pm_eto`.
   Function documentation defines other units, including radiation and wind speed.
   Array indices are zero-based; day-of-year and calendar-month inputs are one-based where documented.
   Monthly calendar layouts contain 12 steps per year.
   Prepared all-leap daily layouts contain 366 positions per year, including February 29 in every year.
   Callers select the Calibration Period and supply any calendar padding, aggregation, clipping, or final index transformation.
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
   `climate-core` enables `#![forbid(unsafe_code)]` and `#![deny(missing_docs)]`.
   The unsafe-code restriction applies to this crate, not its dependencies.
   ndarray's `s!` macro contains an `allow(unsafe_code)` attribute incompatible with `forbid`.
   Equivalent safe axis-slicing methods replace those macro calls, including in tests.
   The replacement preserves the selected elements, strides, and arithmetic order.
   Public field documentation states the existing meanings rather than changing the fields.

7. **Add no feature flags.**
   All kernels remain available with no default or optional features.
   There is no demonstrated need for serde, parallel execution, or another optional dependency here.
   Python bindings remain in the separate, unpublished `climate-py` crate.

## Compiling sketch and validation

The crate-level rustdoc in `crates/climate-core/src/lib.rs` demonstrates a single-series call,
a grid call, calendar-column fitting, and typed error handling.
The `MissingDayPolicy` rustdoc demonstrates downstream matching with a wildcard arm.
These examples use the current `climate_core` import path and a direct ndarray dependency.
`cargo test -p climate-core --doc` compiles and executes them as downstream callers.
The synthetic fit sample illustrates shape conventions, not a sufficient scientific Calibration Period.

The workspace build and native Python parity tests verify that `climate-py` still consumes the surface.
No kernel changes its numerical expressions or operation order.
This record makes no new measured accuracy claim.

## Alternatives considered

- **Hide ndarray behind crate-owned types.** This adds conversions or a parallel array API without a demonstrated requirement.
- **Re-export ndarray.** This changes import convenience, but does not remove the exposed dependency's compatibility cost.
- **Add separate series and grid wrappers.** Borrowed singleton-axis views already support one series without copying its data.
- **Stabilize diagnostic text.** Typed variants support control flow without freezing prose or binding Rust users to Python messages.
- **Add feature flags speculatively.** This increases the supported build combinations without a current user requirement.

## Consequences and follow-up ownership

Adding `non_exhaustive` intentionally requires wildcard arms in downstream exhaustive matches.
The existing Python bindings already use a wildcard for `ClimateError` and still construct the other enums normally.
The current package remains unpublished, and this task does not change package versions or release automation.

- [CRATE-04 — Numerical contract](https://github.com/monocongo/climate_indices/issues/1308) owns measured per-kernel accuracy contracts.
- [CRATE-05 — Packaging](https://github.com/monocongo/climate_indices/issues/1309) owns the package, directory, and import rename to `climate-indices`.
  It must update the compiling rustdoc imports as part of that rename.
- [CRATE-06 — Standalone validation](https://github.com/monocongo/climate_indices/issues/1310) owns standalone reference fixtures.
- [CRATE-07 — Rust-user documentation](https://github.com/monocongo/climate_indices/issues/1311) owns full examples and guides beyond this API sketch.
