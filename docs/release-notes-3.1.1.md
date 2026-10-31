# climate_indices 3.1.1 release notes

Version 3.1.1 is unreleased. The release date is not selected.
The {doc}`release runbook <release-process>` describes the date, candidate freeze,
and rehearsal requirements.

Version 3.1.1 adds Output Provenance, expands optional Rust acceleration, and
requires Python 3.11–3.14. Review the {doc}`migration guidance <deprecations/api-changes>`
before upgrading from 3.1.0.
The [full changelog](https://github.com/monocongo/climate_indices/blob/main/CHANGELOG.md)
records the selected changes.

## Python support

The package requires `>=3.11,<3.15`. Python 3.10 is no longer supported.
Upgrade Python before installing 3.1.1. If you must retain Python 3.10, use
`climate-indices==3.1.0`.

## Output Provenance

Xarray and CLI outputs carry a strict-JSON `climate_indices_provenance` record
with software identity, backend availability, effective parameters, input metadata,
and upstream lineage. The record remains attached through NetCDF and Zarr round trips.
Compound Dataset routes store it at Dataset level, beside their source manifest.

`climate_indices.provenance.with_source()` labels inputs with source identifiers.
`read_provenance()` decodes an output record. Builds embed the source commit,
so installed packages can report it without Git. Metadata collection does not
compute lazy input data. Existing attributes and the `history` format remain unchanged.

The record supports reproducibility but does not guarantee identical numbers.
It records native-extension availability, not the kernel that ran.
Parameter and input detail depend on the route, and size limits can omit sections.
The record removes recognized credentials and local paths, but it does not guarantee anonymity.
Review free text before sharing outputs. See {doc}`Dataset provenance <provenance>`
for the schema, limits, and privacy rules.

## Optional Rust backend

Eligible float64 calls can now use Rust for these operations:

- Gamma goodness-of-fit D statistics.
- Empirical plotting positions.
- Run Theory identification and per-run metrics.
- Compound bootstrap resample metrics.
- CFFWIS behavior indices: ISI, BUI, FWI, and DSR.
- Overwinter Drought Code.
- Trailing Timescale sums.

The parity contract remains `rtol = atol = 1e-10`, with matching missing-value positions.
Complete-input KBDI uses a tolerance-preserving fast path rather than bitwise identity.
See {doc}`ADR-0025 <adr/0025-kbdi-complete-input-fast-path>` for its eligibility rules.

Python retains longer series convolutions, non-C-contiguous aggregation blocks,
Fosberg FFWI, HDW, Haines, and the beta distribution.
Calls that cannot use native empirical ranking retain the Python fallback loop.
Without the optional extension, calculations continue to use Python.
A runtime error from the extension propagates; it does not silently retry in Python.
See the {doc}`numerical contract <rust-numerical-contract>` and
{doc}`architecture <architecture>` for route-specific conditions.

The standalone `climate-indices` crate prepares version `0.1.1` on its independent release track.
It also exposes trailing means to Rust callers.
This Python release does not publish the crate.
See the [crate changelog](https://github.com/monocongo/climate_indices/blob/main/crates/climate-indices/CHANGELOG.md).

## Execution and error handling

Distribution transforms, native dispatch, CLI index adapters, Spatial Kernel packing,
and Calibration Period inference now use shared implementations.
Public signatures, CLI selections, calendar rules, and Dask chunk requirements remain unchanged.

`climate_indices.vocabulary` owns the shared types, including `Distribution` and `Periodicity`.
Existing paths, such as `indices.Distribution` and `compute.Periodicity`, remain aliases of the same objects.
Enum pickle compatibility remains unchanged. Existing callers need no import changes.

Pearson fitting and diagnostics share one fallback decision and poor-fit reporting policy.
Post-fit transform and loss-measurement errors propagate instead of triggering gamma fallback.
Console logging uses plain tracebacks even when `rich` is installed.
These tracebacks do not include frame-local variables. JSON logging remains unchanged.

## Dependency and documentation updates

The lockfile pins security updates for `msgpack` 1.2.3, `filelock` 4.2.1, and `tornado` 6.6.
Published runtime dependency ranges remain unchanged. Development tooling updates Ruff and GitHub Actions.

The homepage derives its Python support range from package metadata.
The benchmark documentation records reproducible empirical-ranking comparisons and the reasons for retaining the Python fallback loop.
The release runbook adds a consumer smoke test for the independently published Rust crate.

For older upgrades, also read the {doc}`3.1.0 release notes <release-notes-3.1.0>`
and the {doc}`API migration guide <deprecations/api-changes>`.
