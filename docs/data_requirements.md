# Input Data Requirements

This page is the contract for values passed to `climate_indices`: which inputs
each index needs, the units and time axis it expects, and which checks the
library performs versus which the caller must guarantee. The NumPy, typed, and
xarray APIs share the same numerics and the same expectations; they differ in
how much of the contract is inferred and validated for you.

## Variables, units, and temporal layout

| Index | Required values | Units | Time axis |
|---|---|---|---|
| SPI (`spi`) | Precipitation | Any consistent units (the index is scale-invariant); millimeters conventional | Monthly or daily |
| SPEI (`spei`) | Precipitation and PET | Both in millimeters (the implementation adds a fixed 1000 mm offset to `precipitation - PET`, so matching non-millimeter units are not equivalent) | Monthly or daily |
| Thornthwaite PET (`pet_thornthwaite`) | Mean temperature, latitude | Degrees Celsius, degrees north | Monthly |
| Hargreaves PET (`pet_hargreaves`) | Daily minimum and maximum temperature, latitude | Degrees Celsius, degrees north | Daily |
| EDDI (`eddi`) | PET | Any consistent units (values are ranked); millimeters conventional | Monthly or daily |
| PNP (`percentage_of_normal`) | Precipitation | Same units throughout (the index is a ratio) | Monthly or daily |
| PCI (`pci`) | Rainfall for one calendar year | Millimeters | Daily, single year |
| Palmer (`palmer.pdsi` and related) | Precipitation, PET, available water capacity | Inches in the NumPy API | Monthly, 12 values per year; PDSI has an xarray adapter, scPDSI remains NumPy/CLI |

The NumPy API consumes the supplied values without converting units. Some
xarray adapters convert recognized CF unit declarations; follow the individual
index guides rather than assuming all adapters do so. Editing an attribute is
not itself a conversion. The command line normalizes recognized precipitation,
PET, and temperature units, including conversion to inches for Palmer, and
rejects unrecognized declarations.

## Time axis

- Monthly series must contain complete, chronological months and begin in
  January. The xarray API rejects skipped months, duplicate timestamps, and
  unsupported frequencies; the NumPy API assumes this layout positionally.
- Daily series must begin on January 1 and step one calendar day at a time.
  The NumPy API expects the internal 366-day layout; `utils.transform_to_366day`
  converts Gregorian daily arrays to that layout, requiring every year but the
  last to be complete and padding a partial final year with NaN. The xarray API
  accepts ordinary Gregorian coordinates and also accepts a partial final year.
- Supported calendars are `standard`, `gregorian`, and
  `proleptic_gregorian` `datetime64` coordinates. `cftime` calendars are
  rejected; see `docs/adr/0004-xarray-calendar-semantics.md`.
- The xarray API infers `data_start_year`, periodicity, and the calibration
  period from the `time` coordinate, which needs at least three timestamps to
  infer periodicity. Pass the arguments explicitly to override inference, and
  always pass them with the NumPy API.
- The first `scale - 1` outputs of an index are NaN because no full window
  exists before them. That is expected, not a sign of invalid input.

## Calibration period

- The calibration period is the baseline the index is fitted or ranked
  against, and it must fall inside the input's year coverage. A window the
  record does not cover is clamped by the gamma, Pearson Type III and
  log-logistic fits, which emit `CalibrationPeriodClampedWarning`, and the
  xarray result's `calibration_year_initial` and `calibration_year_final`
  attributes name the years the fit used.
- 30 years is the documented minimum. A shorter period emits
  `ShortCalibrationWarning`, and more than 20% missing values inside the
  period emits a warning about fitting reliability.
- When no calibration period is given, the xarray API uses the full input
  range. For a single-series (1-D) in-memory input that contains NaNs and takes
  `calibration_year_initial`/`calibration_year_final` (the fitting-based
  indices), it raises `InsufficientDataError` when fewer than 30 effective
  non-NaN years fall inside that range; a complete input shorter than 30 years proceeds with
  only `ShortCalibrationWarning`.
- That effective-year check reads input values and samples a single series, so
  it does not run on gridded (more than one dimension) or Dask-backed inputs:
  neither path enforces a per-cell 30-year non-NaN minimum, and a grid whose
  cells are sparse inside the calibration window may proceed into fitting
  without raising `InsufficientDataError`.
- Daily single-series sufficiency is measured on Gregorian observations with
  a 365-day divisor, while fitting uses 366-slot years with synthetic February
  29 values. This is not a per-calendar-slot minimum of observed samples;
  [issue #760](https://github.com/monocongo/climate_indices/issues/760) tracks
  the unresolved observed-versus-synthetic sufficiency policy.
- Pearson fitting requires enough non-zero values per calendar period. SPI
  and the default `standardized_index()` enable gamma fallback when a Pearson
  fit raises a fitting error or its transform loses more than half of the
  input's valid values. The decision applies to the whole input block, not
  each insufficient calendar period or cell; a failed period alone does not
  guarantee fallback. SPEI does not enable that fallback. Use gamma for
  strongly zero-inflated precipitation, and inspect output and fit diagnostics.
- SPI, SPEI, `standardized_index()`, EDDI, Palmer, and
  `percentage_of_normal()` reject a reversed Calibration Period with
  `CalibrationPeriodError`. Gamma, Pearson Type III, and log-logistic fits
  clamp a non-reversed window the record does not cover; EDDI, Palmer, and
  `percentage_of_normal()` reject uncovered windows with
  `CalibrationPeriodError`. Supplied fitting parameters do not bypass the
  reversed-window check. The flood-family EDI and Flood Index instead raise
  `InvalidArgumentError` for reversed or uncovered windows and require at least
  two complete calibration years. See the [migration guide](deprecations/api-changes.md)
  for changes from 2.4.0.

## Missing values and zeros

- NaN marks missing data and propagates through affected timescale windows.
  Daily xarray calendar conversion does synthesize a non-leap February 29
  by interpolation; these are calendar slots, not new observed samples.
  Calibration estimation omits NaNs rather than consuming them — `percentage_of_normal` averages each calendar
  step with `np.nanmean`, so a NaN inside the calibration period does not block
  the normal, non-NaN cells can still receive finite percentages from the
  remaining values, and the NaN cell's own output stays NaN.
- Zero precipitation is data, meaning a dry period, not a missing value, and is
  preserved. Zero-inflated series are a distribution-fitting concern, not a
  data-cleaning one.
- SPI and SPEI clip negative precipitation to zero with a warning; EDDI does
  the same for PET. `percentage_of_normal` and PCI use their supplied
  rainfall values unmodified, so fix sign conventions upstream instead of
  relying on a clip.
- Dask-backed xarray input to the index adapters must keep the full `time`
  dimension in a single chunk; spatial chunks remain free to parallelize.
  Those adapters never rechunk implicitly and raise
  `CoordinateValidationError` with the exact fix
  (`data = data.chunk({"time": -1})`) when `time` is split. The PET
  adapters (`pet_thornthwaite`, `pet_hargreaves`) are the exception: they
  pass `allow_rechunk=True` and silently consolidate a split `time`
  dimension, with a potentially large memory cost.
- `pci` is a separate case: its xarray wrapper passes the input values
  directly to `indices.pci` without the chunk validation, so a split `time`
  dimension is not rejected and accessing `.values` may eagerly materialize
  the rainfall data.

## Multiple variables and grids

- Paired inputs (SPEI precipitation and PET, Hargreaves minimum and maximum
  temperature) must share coordinates. The xarray API aligns them with an inner
  join; an empty intersection raises `CoordinateValidationError`. The generic
  SPEI adapter warns with `InputAlignmentWarning` only when the primary
  precipitation input loses timesteps, so extra PET timesteps are dropped
  silently. The Hargreaves and Penman-Monteith adapters warn when either
  temperature input, or a Penman-Monteith time-series input, loses timesteps.
  Only time is trimmed: for SPEI, Hargreaves, and Penman-Monteith, a shared
  cell dimension with differing coordinates raises
  `CoordinateValidationError` instead of being intersected.
- When the inputs must match exactly, align them before calculation with
  `xr.align(..., join="exact")`, which raises instead of intersecting. The
  end-to-end sample does this in `scripts/prepare_e2e_inputs.py`.
- Regridding, reprojection, and resampling happen outside the library. Inputs
  on different grids, calendars, or periods are not reconciled for you.

## What is validated where

| Check | NumPy / typed API | xarray API | CLI |
|---|---|---|---|
| Time coordinate completeness, periodicity, and start month | Caller | Enforced | Caller |
| Timesteps at least `scale` | Not enforced | Enforced | Not enforced |
| Calibration length and missing-data warnings | Warned | Warned | Warned |
| Calibration non-NaN sample size | Caller | Enforced when NaNs are present (in-memory 1-D fitting-based input) | Caller |
| Calibration years inside data coverage | Partial (see note) | Partial (see note) | Partial (see note) |
| Multi-variable alignment | Caller, by array size | Inner join with warning | Caller |
| Units | Caller | Index-specific CF conversion | Converted and validated |
| Latitude range | Enforced | Enforced | Enforced |
| Dask `time` chunk | Not applicable | Enforced, except the PET adapters and `pci` | Not applicable |

The coverage checks are index-specific: EDDI, Palmer, and percentage of normal
reject uncovered windows. SPI and SPEI fits clamp them and emit
`CalibrationPeriodClampedWarning`. Indices that accept a Calibration Period
reject reversed windows; exception classes differ as described above.

## See also

- `docs/xarray_compatibility.md` for coverage status of the beta xarray API.
- {doc}`xarray_migration` for moving a workflow from NumPy to xarray.
- `docs/adr/0003-dask-time-dimension-single-chunk.md` and
  `docs/adr/0004-xarray-calendar-semantics.md` for chunking and calendar
  decisions.
- {doc}`troubleshooting` for symptoms and fixes.
- `notebooks/zarr_dask_spi_spei.ipynb` with
  `scripts/prepare_e2e_inputs.py` for a worked preparation and computation
  example.
