# Effective Drought Index (EDI)

## Definition

`flood.edi(pe, data_start_year, calibration_year_initial, calibration_year_final)`
standardizes daily effective precipitation (PE) against the explicitly selected
Calibration Period. For each of 366 positional calendar days, it subtracts the
mean PE of that day in the calibration years and divides by the population
standard deviation of that day's PE. This is Byun & Wilhite (1999), Eq. 9,
with a fixed 365-day PE window: the harmonic factor in `PRN = DEP / H_D`
cancels in `PRN / SD(PRN)`. Positive values indicate wetter conditions;
negative values indicate drier conditions. These indices describe flood
*potential*, not observed flooding.

Input starts January 1 and uses a 366-day all-leap layout for every year.
Convert Gregorian precipitation *before* computing PE: use
`utils.transform_to_366day` for 1-D input or the equivalent
`DailyCalendarPlan.to_all_leap` for Spatial Blocks. Both fill February 29
in non-leap years with the mean of February 28 and March 1; leaving it NaN
would make every PE window spanning that day NaN. The first `duration - 1`
PE values are NaN. An incomplete final year may be
computed but cannot belong to the Calibration Period. Missing observations
remain NaN; days with fewer than two finite calibration values or zero
variance yield NaN. The `duration` argument is accepted for symmetry with
`effective_precipitation()` but does not affect fixed-window EDI after PE
has been computed.

The published variable-duration dry-spell extension and five-day running-mean
smoothing of the per-calendar-day climatology are **not implemented**. The
default PE window is 365 days; other windows are caller experiments, not the
recorded EDI form ([ADR-0014](../adr/0014-flood-family-scientific-conventions.md)).
Only the NumPy route `flood.edi()` is available; the root and xarray routes
follow with [#1108](https://github.com/monocongo/climate_indices/issues/1108).

## Validation

`tests/test_flood_edi.py` checks the source-backed fixed-window algebra,
missing data, daily baselines, and shape/calibration contracts. No numeric
external oracle is available for this fixed-window form; the published Table 5
uses variable-duration EDI and requires the unavailable Hickman daily record
(see `tests/fixture/flood/README.md`). These tests are specification checks,
not external scientific validation.

## Reference

Byun, H.-R., and Wilhite, D. A. (1999). Objective quantification of drought
severity and duration. *Journal of Climate*, 12, 2747–2756.
