# API Changes

This page records feature-level deprecation and migration notes, including
breaking changes that ship without a deprecation period.

## Breaking changes in 3.0.0

3.0.0 ships four breaking changes that users hit without a deprecation period.
Each one below states what a user sees, how to detect it, and what to change.

### Daily xarray calendar alignment (3.0.0)

**What a user sees:** {func}`~climate_indices.typed_public_api.spi`,
{func}`~climate_indices.typed_public_api.spei`,
{func}`~climate_indices.typed_public_api.eddi`,
{func}`~climate_indices.typed_public_api.percentage_of_normal`, and
{func}`climate_indices.xarray_adapter.pet_hargreaves` may return **different,
corrected** daily values for any input spanning a non-leap year. Before 3.0.0
the adapter passed daily Gregorian values straight into the NumPy core's
366-day-per-year layout, silently shifting every value after February 28. Inputs
on a supported Gregorian calendar (`standard`, `gregorian`, or
`proleptic_gregorian`) whose daily coordinates begin on January 1 still succeed
— with different numbers, and no error raised. Inputs that previously succeeded
on an unsupported calendar, with daily coordinates that do not begin on
January 1, or with monthly coordinates that do not begin in January, now raise
`CoordinateValidationError` instead of being silently misinterpreted. The NumPy
array API is unaffected.

**How to detect it:** rerun any daily xarray calculation whose time coordinate
spans a non-leap year and compare against a cached result. A run that completes
with shifted numbers is this change; a run that now raises is the new
calendar-origin validation.

**What to change:** nothing is required for supported calendars — the
corrected values are the fix. Confirm the time coordinate uses `standard`,
`gregorian`, or `proleptic_gregorian` and that daily input begins on January 1
(monthly input begins in January), then recompute. See the calendar-alignment
warning and the "Unsupported calendar or calendar origin" pitfall in {doc}`the
xarray migration guide </xarray_migration>`, and
[ADR-0004](../adr/0004-xarray-calendar-semantics.md) for the calendar
semantics behind the correction.

### NumPy gridded input shape guard (3.0.0)

**What a user sees:** {func}`climate_indices.indices.spi`,
{func}`climate_indices.indices.spei`, and
{func}`climate_indices.compute.prepare_scaled` now read a
three-or-more-dimensional NumPy array as a time-major `(time, *cells)` block.
One shape cannot be told apart from a `(years, periods, *cells)` array — a block
whose first cell axis is a calendar period length — and is rejected unless
declared:

```text
Invalid shape of input array: ... -- a (time, *cells) block whose first cell
axis is a calendar period length is ambiguous with a (years, periods, *cells)
array; declare it with spatial_time_major=True
```

The behavior it replaces is not uniform: in 2.4.0 `indices.spei` flattened a
3-D input into one series and returned numbers from the wrong layout, while
`indices.spi` already rejected any 3-D input with a generic shape error, and
`compute.prepare_scaled` did not exist. 3.0.0 rejects the ambiguous shape for
all three, and `indices.spi` accepts declared blocks it used to reject.

**How to detect it:** the run raises `ValueError` naming the shape. Callers
moving from `indices.spei` see an error where a flattened result used to come
back; callers moving from `indices.spi` see a shape-specific error and a
declared-block path instead of the generic 1-D/2-D rejection.

**What to change:** reorder the cell axes so the first one is not a calendar
period length, or pass `spatial_time_major=True` when the array really is a
time-major `(time, *cells)` block. The keyword exists on `indices.spi`,
`indices.spei`, and `compute.prepare_scaled`; the package-root `spi()` and
`spei()` wrappers do not forward it. The xarray adapter declares the keyword for
every block it packs, so only direct NumPy callers are affected. See
[ADR-0009](../adr/0009-spatial-block-declaration.md).

### PCI February correction (3.0.0)

**What a user sees:** {func}`~climate_indices.typed_public_api.pci` returns
different values. The cumulative day-of-year month-end boundaries had February
written as a month length (28 or 29) instead of its cumulative index: the
February slice was empty, and March absorbed February while starting three days
early in a non-leap year (two in a leap year), re-counting those late-January
days. The direction of the change depends on the rainfall distribution — for
uniform 1 mm/day rainfall the value moves from `9.754549` to `8.337066` for a
366-day input and `8.340026` for a 365-day input, and a series with rain only on
January 29–31 moves from `50.0` to `100.0`.

**How to detect it:** compare a recomputed `pci()` value against a cached one.
Any 365- or 366-day input can move: the total changes when any of the final
three January days (final two in a leap year) is non-zero, or when February and
March both carry rainfall. A February-only check is not enough.

**What to change:** recompute PCI values, and recalibrate any downstream
thresholds tuned against the previous values. The fix landed in
[#846](https://github.com/monocongo/climate_indices/pull/846).

### Periodicity validation type (3.0.0)

**What a user sees:** the periodicity check shared by
{func}`climate_indices.compute.prepare_scaled`,
`compute.transform_fitted_gamma`, `compute.transform_fitted_pearson`,
`compute.gamma_parameters`, and `compute.reshape_values` now raises
{class}`PeriodicityError <climate_indices.exceptions.PeriodicityError>` where
those paths previously raised a bare `ValueError` for an invalid periodicity
argument. Error messages are unchanged. The same change lands on
{func}`climate_indices.indices.spi`, {func}`climate_indices.indices.spei`, and
the other index functions, which previously raised the parent
`InvalidArgumentError` instead of the specialization.

**How to detect it:** look for `except ValueError` around a call that passes a
`periodicity` argument. A run that used to handle the error locally now
propagates it instead, since `PeriodicityError` derives from
`InvalidArgumentError`, not from `ValueError`.

**What to change:** catch `PeriodicityError` — or its parent
`InvalidArgumentError`, the type the troubleshooting guide documents for
`Invalid periodicity argument` — imported from `climate_indices.exceptions`.
Where a `ValueError` handler has to keep working across the upgrade, catch both.
The exception's `valid_values` attribute now reads
`"Periodicity.monthly, Periodicity.daily"`; the message text is unchanged.

A related but non-breaking change: a missing required named dimension (for
example a time dimension absent from an xarray input) now raises
{class}`DimensionMismatchError <climate_indices.exceptions.DimensionMismatchError>`
instead of `CoordinateValidationError`. `DimensionMismatchError` derives from
`CoordinateValidationError`, so existing handlers keep catching it.

## `spi` console script (removed in 3.0.0)

The `spi` console script (`climate_indices.__spi__:main`) was deprecated
in 2.4.0 and is removed in 3.0.0 (#919): 2.4.0 is the last release that ships
the script and emits `ClimateIndicesDeprecationWarning` on invocation. From
3.0.0 on the `climate_indices.__spi__` module is gone, so imports of it and
invocations of the `spi` command raise the usual import and shell errors.

Use `climate_indices --index spi` instead, with two caveats:

- The `--save_params` and `--load_params` options, which cache fitted SPI
  distribution parameters in a NetCDF file, are retired along with the script
  rather than migrated to `climate_indices` (#957). Fitting parameters remain
  available at the library level: fit the scaled values once with
  `compute.gamma_parameters()` or `compute.pearson_parameters()`, then pass
  the parameters as a dict — `{"alpha": ..., "beta": ...}` for gamma,
  `{"prob_zero": ..., "loc": ..., "scale": ..., "skew": ...}` for Pearson —
  as the `fitting_params` argument of `indices.spi()`. The SPI
  section of the documentation index shows the gridded workflow. For SPEI, fit
  the series SPEI itself prepares -- precipitation clipped at zero, minus PET,
  plus the 1000 mm offset, then scaled -- and pass those parameters to
  `indices.spei()`; the SPI workflow's precipitation series fits a different
  distribution and its parameters are accepted without validation.
- For multi-scale runs, `climate_indices --index spi` reopens and stages the
  precipitation input once per scale, while the `spi` script stages it once
  for all scales. Large multi-scale batches may therefore need more time and
  memory after migrating.
