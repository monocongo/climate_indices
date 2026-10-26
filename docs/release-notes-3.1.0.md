# climate_indices 3.1.0 release notes

**Draft:** 3.1.0 is not yet tagged or published, and no release date is set. The
candidate freeze replaces this notice with the release date. The
[release runbook](release-process.md) records the candidate and tag mechanics.

Version 3.1.0 adds compound-hazard analysis and energy-sector indices, promotes
the DataArray index APIs to Stable, and adds optional Icechunk I/O. Review the
[full changelog](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md) and
[API migration guide](deprecations/api-changes.md) before upgrading from 3.0.0.

## Highlights

- **Compound hazards:** The new `climate_indices.compound` module aligns declared
  hazards onto one calendar and returns joint counts and frequencies, the likelihood
  multiplication factor (LMF), conditional and forward-window probabilities,
  concurrent and sequential event matches, and monthly block-bootstrap intervals and
  independence tests. The NumPy routes are stable; the xarray Dataset routes and the
  eager event catalog `compound_events_xarray()` are Beta. The statistics describe
  co-occurrence, not causation or significance. A reproduction of published LMF cases
  and a bootstrap calibration study are scoped evidence, not unrestricted scientific
  validation. See the [compound-hazards guide](compound-hazards.md) and
  [ADR-0022](adr/0022-compound-analysis-contracts.md).
- **Energy indices:** The new `climate_indices.energy` module provides heating,
  cooling, and energy degree days; hub-height wind speed and the wind-shear exponent;
  wind and temperature-derated PV capacity factors (NumPy-only); SREPI, a
  standardized demand index, and SRLI for residual load; and run-theory
  energy-drought events. The outputs describe the climate-driven component of energy
  supply and demand, not actual generation or load. Evidence differs by index:
  SREPI, the demand index, and SRLI are compared with R `SEI` 0.2.0 fixtures, while
  the wind capacity factor and event statistics are regression-only. See
  [energy applications](energy_applications.md) and
  [VALIDATION.md](https://github.com/monocongo/climate_indices/blob/main/VALIDATION.md).
- **Stable DataArray APIs:** The DataArray index entry points, including the
  Dataset-returning index and diagnostic functions, and the public `xarray_adapter`
  interface are Stable: minor versions make no breaking interface change. The
  promotion changes no signature, numerical algorithm, calendar rule, or Dask chunk
  requirement. The compound Dataset routes, the eager event catalogs, and the
  `energy_drought_events()` DataArray route remain Beta. See the
  [compatibility matrix](xarray_compatibility.md).
- **Optional Icechunk I/O:** `climate_indices.icechunk_io.open_icechunk()` and
  `write_icechunk()` read and write transactional Icechunk snapshots. The
  `icechunk` extra needs Python 3.11 or later; the core install and its Python 3.10
  support do not change. Writes reject an existing group by default, and a write that
  fails before its commit leaves the branch unchanged. A commit error, such as a
  timeout, can leave the outcome unknown, so check the branch tip before you retry.
  See [Read and write Icechunk snapshots](writing-outputs.md#read-and-write-icechunk-snapshots).
- **Standardized-index options:** `indices.standardized_index()` adds `mean`
  aggregation, a rank-based empirical distribution, a beta distribution for bounded
  `[0, 1]` series, and the `clip_negatives` keyword. `compute.Periodicity.hourly`
  standardizes hourly series by hour of day and day of year with the empirical
  distribution only. `climate_indices.standardized_index` is the xarray entry point.
  `spi()` and `spei()` do not accept the new distributions.
- **Flood event summaries:** `flood.flood_event_summary()` summarizes
  flood-potential events by onset period or hydrological year. `flood_events()` now
  uses the shared run layer and returns the same events as before.

## Command-line interface

`climate_indices --index` adds `hdd`, `cdd`, `edd`, `wind_profile`, `srepi`,
`demand_index`, and `srli`. `--output_scale` applies to the three standardized
energy indices, and `--periodicity hourly` is accepted only for `srepi`,
`demand_index`, and `srli`. The `all` and `scaled` selections do not change. See
the [workflow examples](workflow-examples.md).

## Rust crate

The kernel crate in this repository is now the standalone `climate-indices` crate
on its own `0.x` release track; version 0.1.0 is published on crates.io. It does not
change the Python package or the optional `climate_indices._native` extension. See
the [crate changelog](https://github.com/monocongo/climate_indices/blob/main/crates/climate-indices/CHANGELOG.md).
The native KBDI kernel and the shared daily recurrence driver are faster, with
bit-identical output.

## Upgrade considerations

3.1.0 removes no public name. Two narrow behavior changes can reject calls that
3.0.0 accepted:

1. A gridded NumPy array whose first cell axis has exactly 8784 values, the hourly
   period length, must now declare `spatial_time_major=True` or reorder its cell
   axes.
2. A function wrapped with `@xarray_adapter()` raises `TypeError` for an unknown
   keyword argument instead of ignoring it.

Percentage-of-normal values do not change, but their metadata now describes them
as ratios: `units` is `"1"`, and the CLI writes `valid_min: 0.0` with no
`valid_max`. Update a reader or threshold that expected percent units.

The lockfile pins security updates for `certifi`, `nbconvert`, and `Pillow`;
published dependency ranges do not change. Pillow 13.0.0 requires Python 3.11, so
the Python 3.10 lock stays on Pillow 12.3.0.

If you upgrade from 2.4.0, also read the 3.0.0 changes. The migration guide now
records the removal of `compute.scale_values()` under 3.0.0, the release that
removed it.

See the [changelog](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md) for exact conditions, affected entry
points, and references, and the [migration guide](deprecations/api-changes.md)
for detection and remedies.
