# API Changes

This page records feature-level deprecation and migration notes, including
breaking changes that ship without a deprecation period.

## Breaking changes in 3.0.0

3.0.0 ships three breaking changes that users hit without a deprecation period.
Each one below states what a user sees, how to detect it, and what to change.

### Daily xarray calendar alignment (3.0.0)

**What a user sees:** {func}`climate_indices.spi`, {func}`climate_indices.spei`,
{func}`climate_indices.eddi`, {func}`climate_indices.percentage_of_normal`, and
{func}`climate_indices.xarray_adapter.pet_hargreaves` return **different,
corrected** daily values for any input spanning a non-leap year. Before 3.0.0
the adapter passed daily Gregorian values straight into the NumPy core's
366-day-per-year layout, silently shifting every value after February 28. Inputs
on a supported Gregorian calendar (`standard`, `gregorian`, or
`proleptic_gregorian`) whose daily coordinates begin on January 1 still succeed
— with different numbers, and no error raised. Inputs that previously succeeded
on an unsupported calendar, or with daily coordinates that do not begin on
January 1, now raise `CoordinateValidationError` instead of being silently
misinterpreted. The NumPy array API is unaffected.

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
{func}`climate_indices.compute.prepare_scaled` now raise `ValueError` for a
three-or-more-dimensional array shaped `(time, 12, *cells)` or
`(time, 366, *cells)`, because that shape is equally readable as a
`(years, periods, *cells)` array:

```text
Invalid shape of input array: ... -- a (time, *cells) block whose first cell
axis is a calendar period length is ambiguous with a (years, periods, *cells)
array; declare it with spatial_time_major=True
```

Previously the ambiguous array was read one way without complaint, which could
return plausible-looking numbers from the wrong axis.

**How to detect it:** the run previously completed and returned numbers; it now
raises `ValueError` naming the shape.

**What to change:** reorder the cell axes so the first one is not a calendar
period length, or pass `spatial_time_major=True` when the array really is a
time-major `(time, *cells)` block. The xarray adapter declares the keyword for
every block it packs, so only direct NumPy callers are affected. See
[ADR-0009](../adr/0009-spatial-block-declaration.md).

### PCI February correction (3.0.0)

**What a user sees:** {func}`climate_indices.pci` returns different values. The
cumulative day-of-year month-end boundaries had February written as a month
length (28 or 29) instead of its cumulative index, leaving the February slice
empty, so March absorbed those days and PCI was overstated. For uniform
1 mm/day rainfall the value moves from `9.754549` to `8.337066` for a 366-day
input and `8.340026` for a 365-day input.

**How to detect it:** compare a recomputed `pci()` value against a cached one.
Every 365- or 366-day input whose February rainfall is non-zero is affected.

**What to change:** recompute PCI values, and recalibrate any downstream
thresholds tuned against the previous, overstated values. The fix landed in
[#846](https://github.com/monocongo/climate_indices/pull/846).

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
  the result as the `fitting_params` argument of `indices.spi()`. The SPI
  section of the documentation index shows the gridded workflow. For SPEI, fit
  the series SPEI itself prepares -- precipitation clipped at zero, minus PET,
  plus the 1000 mm offset, then scaled -- and pass those parameters to
  `indices.spei()`; the SPI workflow's precipitation series fits a different
  distribution and its parameters are accepted without validation.
- For multi-scale runs, `climate_indices --index spi` reopens and stages the
  precipitation input once per scale, while the `spi` script stages it once
  for all scales. Large multi-scale batches may therefore need more time and
  memory after migrating.
