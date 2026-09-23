# Flood subsystem

## Decision

Flood-potential and wet-extreme indices live in one namespaced package:

```text
src/climate_indices/flood/
    __init__.py       public facade
    _pe.py            effective precipitation (PE) kernel
    _edi.py           Effective Drought Index (EDI)
    _if.py            Flood Index (I_F)
    _antecedent.py    Antecedent Precipitation Index (API)
from climate_indices import flood
```

The facade is the stable NumPy layer for this family, an intentional
family-level exception to the drought-oriented `indices.py`/`compute.py`
placement in [ADR-0001](../adr/0001-dual-numpy-xarray-api.md), recorded in
[ADR-0013](../adr/0013-flood-module-api-and-naming.md). Implementation modules
carry a leading underscore so `flood.edi` stays bound to the function rather
than the module.

**PE is the single front door.** EDI and I_F take effective precipitation, not
raw precipitation, and do not recompute the kernel behind the caller's back.
One PE array can therefore feed both indices, which matters on gridded input
where the kernel is the expensive step. The cost is that nothing can verify a
caller's `pe` was produced with the same `duration` passed to `edi()`; that
dependency is stated in the signature table below and in the docstring.

## Scope and boundary

In scope: indices computed from meteorological and climatological inputs, plus
input-agnostic standardization that can accept runoff or streamflow. Out of
scope: hydrologic and hydraulic modeling, inundation mapping, flood-frequency
analysis of discharge, snowmelt-dependent indices, and terrain-, soil-,
land-cover-, or routing-dependent quantities. A sibling repository takes
anything needing a calibrated hydrologic model, terrain or land-cover data, or
routed discharge. The indices describe flood *potential*, not flooding, and
every docstring and doc page has to say so.

[ADR-0014](../adr/0014-flood-family-scientific-conventions.md) records the
scientific conventions these signatures serve, including the deliberate
deviations from the published definitions.

## API tiers

The NumPy API is the stable tier and lands first (FLOOD-08 through FLOOD-10 and
FLOOD-12).
Xarray adapters, CF metadata, and Dask support follow (FLOOD-11, FLOOD-13) and
stay on their `flood.<name>` route as beta paths under
[ADR-0012](../adr/0012-xarray-api-stays-beta-through-3.0.0.md). Family API
additions do not go into `typed_public_api.py`; `edi` is the single exception and
reaches the package root with its xarray adapter, for the reasons recorded in
ADR-0013.

## Names and input contracts

New configuration is keyword-only, and the conventions follow the two
established spellings rather than the three drifting ones in `indices.py`:
inputs are named for their physical quantity as the fire package does
(`precipitation`, not `values` or `precips_mm`), and calibration bounds are
`calibration_year_initial`/`calibration_year_final` as in `spi`, `spei`, and
`eddi`. The family declares **no** `units=` parameter: none of these indices has
a unit-dependent constant or threshold, PE and API are millimeters by
construction, and EDI and I_F are dimensionless and scale-invariant. Xarray
adapters convert at their boundary, as `fire/_units.py` does. There is no
`periodicity=` parameter — these indices are daily only, under the calendar
contract of [ADR-0004](../adr/0004-xarray-calendar-semantics.md). The
`duration` arguments default to the 365-day window
[ADR-0014](../adr/0014-flood-family-scientific-conventions.md) fixes as the
convention; other windows are caller experiments, not the recorded form.

| Public name | Inputs | Output / accepted alternative |
| --- | --- | --- |
| `effective_precipitation(precipitation, *, duration=365, spatial_time_major=False)` | daily precipitation in mm | effective precipitation in mm; the leading `duration - 1` days are NaN for want of a full window |
| `edi(pe, data_start_year, calibration_year_initial, calibration_year_final, *, duration=365, spatial_time_major=False)` | effective precipitation in mm from the row above, and the calendar years bounding the Calibration Period | dimensionless EDI; `duration` must match the kernel that produced `pe`, because `H_duration` is the denominator of PRN, and a mismatch is not detectable from the array |
| `flood_index(pe, data_start_year, calibration_year_initial, calibration_year_final, *, year_start_month, spatial_time_major=False)` | effective precipitation in mm, the Calibration Period years, and the caller's year boundary as a calendar month | dimensionless I_F, standardized against the mean and standard deviation of the annual maxima of PE; `year_start_month` is required and has no default, and partial leading and trailing periods are excluded from the maxima |
| `antecedent_precipitation_index(precipitation, k, *, initial_state=None, return_state=False, spin_up=0, nan_policy="propagate", max_gap_days=0, spatial_time_major=False)` | daily precipitation in mm, and a decay constant `0 < k < 1` | antecedent precipitation in mm; for the non-lagged recursion, constant input converges to the closed form `P / (1 − k)`, which is the regression test FLOOD-12 specifies |

`spatial_time_major` is the keyword-only block declaration of
[ADR-0009](../adr/0009-spatial-block-declaration.md) and behaves as it does for
the other NumPy index functions.

## Stateful recurrence contract

API is a daily recursion, and it follows [ADR-0006](../adr/0006-fire-recursive-state-and-execution.md)
and [ADR-0007](../adr/0007-fire-missing-data-policy.md) exactly as KBDI does:
`spin_up`, `nan_policy`, `max_gap_days`, `initial_state`, and `return_state`
mean what they mean there, and no flood-specific variant is introduced. The
recursion's published form is the non-lagged one — Kohler & Linsley (1951)
Eq. (3) gives `I₁ = k · I₀`, and the text states that "if rain occurs on any
day, the amount of rain observed is added to the index" — so
`API_t = k · API_{t−1} + P_t`. Typical `k` is 0.85–0.90 for the eastern and
central United States; the source carries the index forward through the whole
record, which is what the ADR-0006/0007 contract provides, and treats an assumed
initial value as converging "within several weeks", which is the `spin_up`
convention. The lagged variant `k · (API_{t−1} + P_{t−1})` that some sources use
is documented in the docstring as the alternative and is not implemented.

## Deferred

- **Variable-duration EDI** (the Byun & Wilhite extension) and reproduction of
  their Table 5, both blocked on the Hickman 1995–1996 daily record: the dry-day
  threshold and the `DS` definition are settled by ADR-0014 decision 1, and the
  paper names its data source (193 High Plains stations, 37 years 1960–96,
  reduced to 113). Anyone building a Table 5 fixture should note that its
  "Minimum of CNS" row cannot be a consecutive-day count under the paper's own
  definition of CNS, and that the Fig. 2 legend plots ANES in those panels — the
  row appears mislabelled, so it is not a CNS fixture.
- **A numeric I_F oracle**, blocked on the Deo et al. (2015) full text; I_F is
  specification-level until then, and the exponential kernel its abstracts
  describe is not implemented (ADR-0014).
- **A numeric API oracle**, blocked on Kohler & Linsley (1951).
- **Flood-event helpers** (onset, duration, severity as runs of `I_F > 0`).
- **WAP and SWAP**, deferred as a separate lineage; **SMRI** and snowmelt
  indices, out of scope.
- **The public names of the precipitation-extreme indices** (Rx1day, Rx5day,
  R95pTOT), which are in scope per ADR-0014 decision 6 and land in this package
  with FLOOD-14 ([#1111](https://github.com/monocongo/climate_indices/issues/1111));
  they are absent from the table above because their names are not yet fixed.
