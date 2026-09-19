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
| Palmer (`palmer.pdsi` and related) | Precipitation, PET, available water capacity | Inches | Monthly, 12 values per year; CLI and NumPy API, no direct xarray API |

The value array matters, not the `units` attribute. The NumPy and xarray APIs
never convert units, and editing an attribute is not conversion. The command
line converts recognized unit declarations to millimeters (precipitation, PET)
and degrees Celsius (temperature) before calculation, and rejects unrecognized
ones.

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
  against, and it must fall inside the input's year coverage.
- 30 years is the documented minimum. A shorter period emits
  `ShortCalibrationWarning`, and more than 20% missing values inside the
  period emits a warning about fitting reliability.
- When no calibration period is given, the xarray API uses the full input
  range. For in-memory inputs that contain NaNs, it raises
  `InsufficientDataError` when fewer than 30 effective non-NaN years fall
  inside that range; a complete input shorter than 30 years proceeds with only
  `ShortCalibrationWarning`.
- That effective-year check reads input values, so it does not run on
  Dask-backed inputs: a Dask input with fewer than 30 effective non-NaN years
  may proceed into fitting without raising `InsufficientDataError`.
- Pearson fitting requires enough non-zero values per calendar period. SPI
  falls back to gamma fitting when the Pearson data is insufficient, while
  SPEI raises `InsufficientDataError`; use the gamma distribution for strongly
  zero-inflated precipitation.
- SPI and SPEI do not reject out-of-range calibration requests: when either
  bound falls outside the input's year coverage, the implementation replaces
  both bounds with the full available record before fitting. Matching the
  calibration years to the actual data coverage is therefore the caller's
  responsibility. `eddi` validates both bounds and raises
  `InvalidArgumentError` when they fall outside the data.
  `percentage_of_normal` raises for a start year before the data and for a
  calibration span larger than the input, but not for an end year beyond the
  data's final year.

## Missing values and zeros

- NaN marks missing data and propagates per cell: an input NaN produces a NaN
  output. Nothing is interpolated or filled. Calibration estimation omits NaNs
  rather than consuming them — `percentage_of_normal` averages each calendar
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
  silently. The Hargreaves adapter compares the aligned length with both the
  minimum- and maximum-temperature inputs and warns when either loses
  timesteps.
- When the inputs must match exactly, align them before calculation with
  `xr.align(..., join="exact")`, which raises instead of intersecting. The
  end-to-end sample does this in `scripts/prepare_e2e_inputs.py`.
- Regridding, reprojection, resampling, and unit conversion happen outside the
  library. Inputs on different grids, calendars, or periods are not reconciled
  for you.

## What is validated where

| Check | NumPy / typed API | xarray API | CLI |
|---|---|---|---|
| Time coordinate completeness, periodicity, and start month | Caller | Enforced | Caller |
| Timesteps at least `scale` | Not enforced | Enforced | Not enforced |
| Calibration length and missing-data warnings | Warned | Warned | Warned |
| Calibration non-NaN sample size | Caller | Enforced when NaNs are present (in-memory inputs) | Caller |
| Calibration years inside data coverage | Partial (see note) | Partial (see note) | Partial (see note) |
| Multi-variable alignment | Caller, by array size | Inner join with warning | Caller |
| Units | Caller | Caller | Converted and validated |
| Latitude range | Enforced | Enforced | Enforced |
| Dask `time` chunk | Not applicable | Enforced, except the PET adapters and `pci` | Not applicable |

The coverage checks are index-specific: `eddi` validates both calibration
bounds, `percentage_of_normal` validates the start year and the calibration
span only, and SPI and SPEI silently replace an out-of-range calibration
request with the full available record.

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
