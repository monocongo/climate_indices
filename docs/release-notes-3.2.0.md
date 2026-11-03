# climate_indices 3.2.0 release notes

Version 3.2.0 is unreleased. The release date is not selected.
The {doc}`release runbook <release-process>` describes the date, candidate freeze,
and rehearsal requirements.

Version 3.2.0 adds the power-system reliability indices, Major Event Day detection,
and the Standardized Precipitation Anomaly Index (SPAI). It also tightens the optional
Rust backend and clarifies the licensing files. The standalone `climate-indices` Rust
crate published version `0.1.1` on 2026-11-03, on its own track. Review the
{doc}`migration guidance <deprecations/api-changes>` before upgrading from 3.1.1.
The [full changelog](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md)
records the selected changes.

## Reliability indices

The new `climate_indices.reliability` module computes distribution-system reliability
indices from interruption event records. Import it with `from climate_indices import reliability`.

- `classify_interruptions()` validates event records and classifies each as sustained
  or momentary. The default threshold is 5 minutes, and exactly 5:00 is momentary.
- `reliability_indices()` returns a frozen `ReliabilityResult` with SAIFI, SAIDI, CAIDI,
  MAIFI_E, MAIFI, ASAI, the customer-interruption (`CI`) and customer-minute (`CMI`)
  accumulators, and separate sustained and momentary outage counts.
- `group_momentary_events()` collapses raw momentary operations into events.
- `simulate_reliability()` runs a seeded Monte Carlo simulation of a series or parallel
  system and feeds its event log to the same metric path.

A metric with no defined value is NaN or `None` and appears in `undefined_metrics`.
The library never replaces it with zero. Durations are in minutes, and a partial period is
never annualized. Planned outages count unless you pass `exclude_planned=True`.
Overlapping events stay valid, and one `OverlappingEventsWarning` reports them.

`major_event_day_threshold()` fits the IEEE 1366 Major Event Day threshold to a daily-SAIDI
history. `reliability_indices(..., major_event_threshold=...)` then removes every event that
starts on a major day. The default applies no filter. The rules were verified against
IEEE Std 1366-2012; the 2022 edition was not available.

The evidence is specification and regression level, plus 20 EIA aggregate rows that check
relationships and bounds. No event-level oracle exists for the sustained metrics or ASAI.
`VALIDATION.md` states these limits. See the {doc}`reliability reference <reference>`,
the {doc}`algorithm reference <algorithm_refs/reliability>`,
the {doc}`application guide <reliability_applications>`, and
{doc}`ADR-0026 <adr/0026-reliability-module-api-and-conventions>`.

## SPAI

`indices.spai()` and `climate_indices.spai()` compute the Standardized Precipitation
Anomaly Index of Chanda and Maity (2015). SPAI subtracts each calendar period's mean over
the Calibration Period and fits one distribution to all anomalies pooled. SPI fits one
distribution for each calendar period.

The accepted distributions are `Distribution.t_location_scale` (the reference choice),
`Distribution.pearson`, and `Distribution.normal`. The first and last are new `Distribution`
members that only `spai()` accepts. Both the NumPy and DataArray routes are **Beta**, and each
call emits one `BetaFeatureWarning`. SPAI has no command-line flag.

The evidence is regression and specification level, with an estimator cross-check. The article
body was not readable, so the anomaly definition rests on the authors' published closure.
See the {doc}`algorithm reference <algorithm-reference>` and `VALIDATION.md`.

## Optional Rust backend

Three changes affect which calls reach Rust. None changes a public signature, and the parity
contract remains `rtol = atol = 1e-10` with matching missing-value positions.

- The empirical transform uses less peak memory. On a 47,185,920-value block, the Rust peak
  fell by one values buffer (360 MiB) and the median time did not rise.
- A one-dimensional series keeps `np.convolve` for trailing sums at Timescales 2 to 7,
  because the kernel measured slower at every tested length. A Spatial Block still dispatches.
- The CLI worker pool now applies the caller's NumPy error policy in each worker. A CLI run
  under an all-ignore policy therefore reaches the kernels, as an in-process call does.

`benchmarks/native_dispatch.py` now records which kernels ran on each route and names the
first gate that made a call decline. See the {doc}`numerical contract <rust-numerical-contract>`
and the {doc}`architecture notes <architecture>`.

## Rust crate 0.1.1

The `climate-indices` crate published version `0.1.1` to
[crates.io](https://crates.io/crates/climate-indices/0.1.1) on 2026-11-03 from the tag
`climate-indices-v0.1.1`. Its version track is independent of this Python release
(ADR-0021). The release contains the
kernels that the Python extension gained in 3.1.1, and it adds
`empirical::plotting_positions_in_place`. It keeps the `0.1.0` public API and its Rust 1.85
minimum. See the
[crate changelog](https://github.com/monocongo/climate_indices/blob/main/crates/climate-indices/CHANGELOG.md).

This Python release neither installs nor depends on the crates.io package. The wheels build
from the workspace crate.

## Licensing and package metadata

The root `LICENSE` now contains only the BSD-3-Clause text. The third-party notices moved
to `THIRD_PARTY_NOTICES.md`, and the historical NOAA public-domain notice stays verbatim in
{doc}`the licensing history <licensing>`. The package declares the SPDX expression
`BSD-3-Clause AND BSL-1.0`, the sdist and both wheels carry all three documents, and the
`License :: OSI Approved :: BSD License` classifier is removed. The licensing terms do
not change.

## Validation records and CI

`VALIDATION.md` now classifies the SPEI log-logistic comparison as a cross-implementation
check that shares the R `SPEI` estimator lineage. It also adds a row for
`standardized_index()` with `Distribution.empirical`, anchored on R `SEI` output for monthly
data only.

`.github/required-status-checks.txt` and `scripts/sync_required_checks.py` keep the required
status checks in step with the workflow matrix, and a release-integrity test fails on drift.

For older upgrades, also read the {doc}`3.1.1 release notes <release-notes-3.1.1>`
and the {doc}`API migration guide <deprecations/api-changes>`.
