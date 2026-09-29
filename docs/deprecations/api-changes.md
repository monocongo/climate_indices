# API Changes

This page records feature-level deprecation and migration notes, including
breaking changes that ship without a deprecation period.

## Deprecations

### PNP `calibration_start_year`/`calibration_end_year` (deprecated in 3.1.0)

`percentage_of_normal` used `calibration_start_year`/`calibration_end_year`
while every other Python index API used
`calibration_year_initial`/`calibration_year_final`. The canonical spelling is now
`calibration_year_initial`/`calibration_year_final` for the whole Python API.
Existing calls using the old names keep working and emit
`ClimateIndicesDeprecationWarning`; the aliases are removed in 4.0.0. The CLI
keeps `--calibration_start_year`/`--calibration_end_year` and adds
`--calibration_year_initial`/`--calibration_year_final` aliases. PNP xarray
outputs retain the existing `calibration_start_year`/`calibration_end_year`
attributes alongside the canonical names; their history entry still records the
timescale, not calibration years. No numerical behavior changes.

## Breaking changes in 3.1.0

### `compute.scale_values` removed (3.1.0)

**What a user sees:** `from climate_indices.compute import scale_values` or
`compute.scale_values(...)` now raises `ImportError`/`AttributeError`.

**How to detect it:** grep for `scale_values` in code that imports from
`climate_indices.compute`.

**What to change:** call
`compute.prepare_scaled(values, scale, periodicity)` instead. `scale_values` was
a thin pass-through over `prepare_scaled`, which owns the shared preparation
pipeline.

## Breaking changes in 3.0.0

3.0.0 ships six breaking changes that users hit without a deprecation period.
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

Two further NumPy entry points pin their dimension errors to
{class}`DataShapeError <climate_indices.exceptions.DataShapeError>` rather than
to `ValueError`: {func}`climate_indices.indices.eddi` and
{func}`climate_indices.indices.percentage_of_normal` reject **every** undeclared
three-or-more-dimensional input, not only the ambiguous shape. `indices.eddi`
already raised `DataShapeError` in 2.4.0, so only its declared-block acceptance
path is new. `indices.percentage_of_normal` did not: it reached `np.convolve`
with the 3-D array and failed there with numpy's
`ValueError: object too deep for desired array`. In 3.0.0 that call raises
`DataShapeError` instead, carrying `expected_shape` and `actual_shape`, so a
handler catching `ValueError` around it no longer matches.

**How to detect it:** the run raises `ValueError` naming the shape. Callers
moving from `indices.spei` see an error where a flattened result used to come
back; callers moving from `indices.spi` see a shape-specific error and a
declared-block path instead of the generic 1-D/2-D rejection.
`indices.eddi` and `indices.percentage_of_normal` raise `DataShapeError` — a
`ClimateIndicesError`, and not a `ValueError` — for any undeclared 3-D input.

**What to change:** reorder the cell axes so the first one is not a calendar
period length, or pass `spatial_time_major=True` when the array really is a
time-major `(time, *cells)` block. The keyword exists on `indices.spi`,
`indices.spei`, `compute.prepare_scaled`, `indices.eddi`,
`indices.percentage_of_normal`, `indices.pet`, and `palmer.pdsi`; the
package-root `spi()` and `spei()` wrappers accept it and pass it through to
those functions. The xarray adapter declares the keyword for every block it
packs, so only direct NumPy callers are affected. Where a `ValueError` handler
has to keep working across the upgrade, catch `DataShapeError` as well. See
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

### Gamma probability of zero from the calibration period (3.0.0)

**What a user sees:** gamma {func}`~climate_indices.typed_public_api.spi`,
`indices.standardized_index()`, {func}`~climate_indices.typed_public_api.spei`,
and `compute.transform_fitted_gamma()` can return different values. The gamma
probability of zero, the mass placed below the fitted distribution, was the
zero count over every year of the record, with missing years counted in the
denominator, while `alpha` and `beta` came from the calibration period. It is
now the calibration period's zero count over its non-missing values, as the
Pearson Type III fit already computed it. A gamma `fitting_params` dictionary
may supply it as `prob_zero`, and `compute.transform_fitted_gamma()` takes it
as `probabilities_of_zero`; either must lie in `[0, 1]`, with NaN for a step
whose zero mass is undefined. `fit_diagnostics()` reports the same value for
gamma and returns it in `parameters`.

**How to detect it:** a full-record calibration of complete years, at scale 1,
with no missing value and no `prob_zero` key is bit-identical. Outside those
conditions an output moves when:

- its input holds a zero (for SPEI, an exact zero in the offset P − PET series)
  at a calendar step whose calibration zero fraction differs from the whole
  record's. That happens when the calibration period is shorter than the
  record, or holds a missing value, which includes the NaN padding of a record
  that ends partway through a year and the leading `scale − 1` values of a
  scaled series. Every value at that step moves, not only its zeros;
- a gamma `fitting_params` already carried `prob_zero` or
  `probabilities_of_zero`. Gamma ignored the key, which now sets the zero mass,
  so every value can move even without zeros;
- a step has no calibration data. Its zeros are now NaN rather than an extreme
  drought.

SPI whose Pearson Type III fit falls back to gamma moves the same way, and a
direct `compute.transform_fitted_gamma()` call now reads a masked entry as
missing.

**What to change:** recompute affected outputs. To reproduce a previous gamma
result, pass the zero fraction of the scaled values over every year of the
record, the NaN-padded final year included, as `fitting_params["prob_zero"]`
(or as `probabilities_of_zero` to `compute.transform_fitted_gamma()`), and
drop a `prob_zero` key that the previous release ignored. To keep a reused
gamma fit's zero mass fixed across datasets, save the calibration period's zero
fraction as `prob_zero` alongside `alpha` and `beta`, as `fit_diagnostics()`
returns it. See
[ADR-0015](../adr/0015-zero-handling-in-standardized-indices.md).

### Calibration Period windows the record cannot represent (3.0.0)

**What a user sees:** a calibration window whose first year is after its last
(for example `calibration_year_initial=2010, calibration_year_final=2000`) now
raises `CalibrationPeriodError` from every index. So does
{func}`climate_indices.indices.percentage_of_normal` for any window its record
does not cover: one that starts before the data, ends after the last year the
data reaches, or starts after the data. `CalibrationPeriodError` is both an
`InvalidArgumentError` and a `ValueError`, so a handler for either still catches
it. Before, a reversed window gave all-NaN from the fitted indices (or, when it
ended before the record, a fit on some other rows of it), and
`percentage_of_normal` gave all-NaN for a reversed window or one that starts
after the record, or averaged only the years that exist for one that ends past
the data. Its window is now measured in years rather than 12 steps per year, so
a window longer than a daily record is rejected, and a window ending in the
record's partial final year is accepted.

The window is now checked before anything else can return early. An all-missing
input with a reversed window raises from `spi`, `spei`, `standardized_index`,
`eddi` and `percentage_of_normal`, where it used to come back as missing values;
`eddi` and `percentage_of_normal` also reject a window the record does not cover.
A window that passes still returns the missing input unchanged. Palmer,
`fit_diagnostics` and the flood indices already checked it. `spi`, `spei` and
`standardized_index` also reject a reversed window when a complete
`fitting_params` set is supplied, where the supplied parameters used to skip the
check and transform the values anyway.

**How to detect it:** look for `percentage_of_normal` calls with a fixed window,
such as 1981-2010, on records that may end earlier, and for code that expects
all-NaN output instead of an exception from a reversed or uncovered window.
The logged `calculation_failed` event for `percentage_of_normal` now carries
`error_type="CalibrationPeriodError"`.

**What to change:** pass a window the record covers, for example
`min(calibration_year_final, last_data_year)`. A reversed window is a swapped
argument pair. The fitted indices (`spi`, `spei`, `standardized_index`,
`fit_diagnostics`) keep clamping a window that is not reversed to the record, so
no change is needed there. A window the record covers returns the same numbers.

## `spi` console script (removed in 3.0.0)

The `spi` console script (`climate_indices.__spi__:main`) is removed in 3.0.0
(#919). Its deprecation warning was added during the 3.0.0 cycle and shipped in
no release, so no released version warned before the removal: 2.4.0 is the last
release that ships the script, and it emits no warning when the script is run.
From 3.0.0 on the `climate_indices.__spi__` module is gone, so imports of it and
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
