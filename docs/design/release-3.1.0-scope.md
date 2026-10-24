# Release 3.1.0 scope and content inventory

This internal record supports [RELEASE31-1, #1458](https://github.com/monocongo/climate_indices/issues/1458).
It separates merged content from pending release commitments.
It does not freeze a candidate, set a release date, or authorize publication.
The [release runbook](../release-process.md) governs those actions.

## Evidence boundary

The inventory covers `v3.0.0..f69285becbddc70f2bf861b5a45c62be717fd078`.
The tag resolves to `85ce853a7b7edbc87c59dbbef4ff2102a5a3224c`.
This range contains **413 commits and 69 merged PRs**.

The issue's initial count of 72 includes three PRs already present at the tag:
[#1329](https://github.com/monocongo/climate_indices/pull/1329),
[#1331](https://github.com/monocongo/climate_indices/pull/1331), and
[#1332](https://github.com/monocongo/climate_indices/pull/1332).
A merge-date search includes these October 12 merges, but a commit-range inventory excludes them.
GitHub records both #1445 and #1447 against merge commit `ab9c80ef`; both PRs belong in the inventory.
The Git log contains 68 distinct PR merge commits.

Reproduce the commit boundary with:

```bash
git rev-parse 'v3.0.0^{commit}'
git rev-list --count v3.0.0..f69285becbddc70f2bf861b5a45c62be717fd078
git log v3.0.0..f69285becbddc70f2bf861b5a45c62be717fd078 --merges --format='%H %s'
```

To reconcile PR membership, retrieve the merged PR records with their `mergeCommit` values.
Include a PR only when its merge commit belongs to the range above.
Do not use the release date alone as the lower boundary.

## Proposed milestone 8 description

Release scope for v3.1.0: the compound module, the energy family, the Rust crate, and the flood foundation.

**Compound module** — CMP-00 (#1364).
The release includes declared-input harmonization, joint counts and frequencies, the likelihood multiplication factor, and conditional and forward-window probabilities.
It also includes concurrent and sequential event matching, monthly block-bootstrap inference, eager numeric event catalogs, and Beta xarray Dataset routes.
The merged guide and published-method reproduction (#1374) and bootstrap calibration (#1398) provide scoped evidence, not unrestricted scientific validation.

**Energy family** — #1157.
The release includes degree days, hub-height wind and shear diagnostics, wind and PV capacity-factor transforms, and standardized production, demand, and residual-load indices.
It includes energy-drought event statistics, supported DataArray routes, and monthly, daily, and empirical-only hourly standardization with CLI integration.
These outputs describe climate-driven resource and demand proxies, not actual generation or load.

**Rust crate** — #1305.
The standalone package is `climate-indices`, with Rust library path `climate_indices` and first functional release version `0.1.0`.
The merged work includes public kernel contracts, standalone verification, package metadata, examples, benchmarks, quality gates, and release automation.
The first functional publication and adoption checks remain tracked by #1315; merged automation does not establish publication.

**Flood foundation** — FLOOD-01 (#1098).
The existing module design, scope boundary, oracle research, documentation, and CF metadata remain part of the release scope.
The 3.0.0 release already shipped the flood index family.
The new flood work consolidates events onto the shared Run layer and adds per-hydrological-year summaries (#1258, PR #1425).

**Other included content.**
The release includes optional Icechunk I/O and its notebook, adapter keyword validation, corrected PNP ratio metadata, security updates, and performance improvements.
The [content inventory](https://github.com/monocongo/climate_indices/blob/main/docs/design/release-3.1.0-scope.md#merged-content-inventory) records all 69 merged PRs and 413 commits through `f69285be`.
It includes documentation, test, dependency, and CI changes.

**Pending commitments.**
DataArray index API promotion to Stable (#1061, PR #1456) remains pending at this evidence boundary.
Compound Dataset routes and eager event catalogs retain Beta status.
The `compute.scale_values` removal is already documented in `docs/deprecations/api-changes.md`.
The remaining release-preparation, validation, rehearsal, and maintainer checks remain release gates, not new feature scope.

**Late-change admission rule.**
A late change enters only if it fixes correctness, security, or data loss; a regression from this release's own changes; or release mechanics.
A planned release task does not authorize unrelated features or refactoring.
After the candidate freeze, any admitted change requires a new freeze and rehearsal of the exact candidate SHA.

**Exclusions.**
The IMPROVE2 round (#1416–#1423), the RUSTUP round (#1430–#1442), and the broad documentation refresh (#986) remain in v3.1.1, milestone 11.
The release-specific documentation and integrity tasks remain in scope; they do not admit the entire #986 refresh.
No release date is assigned by this scope record.

## Merged content inventory

Each PR appears once below.
The groups cover user-visible changes and supporting work, rather than treating test-only and dependency updates as new features.
The links identify merged PRs, not their implementation issues.

| Content | Merged PRs | Release effect |
| --- | --- | --- |
| Compound contracts and harmonization | [#1375](https://github.com/monocongo/climate_indices/pull/1375), [#1379](https://github.com/monocongo/climate_indices/pull/1379), [#1380](https://github.com/monocongo/climate_indices/pull/1380) | The module defines declared hazards, calendar alignment, validity, and a public drought–heatwave reference fixture. |
| Compound statistics and events | [#1386](https://github.com/monocongo/climate_indices/pull/1386), [#1388](https://github.com/monocongo/climate_indices/pull/1388), [#1392](https://github.com/monocongo/climate_indices/pull/1392), [#1397](https://github.com/monocongo/climate_indices/pull/1397) | The module adds joint statistics, conditional probabilities, event matching, and monthly bootstrap inference. |
| Compound adapters, catalogs, and evidence | [#1400](https://github.com/monocongo/climate_indices/pull/1400), [#1402](https://github.com/monocongo/climate_indices/pull/1402), [#1405](https://github.com/monocongo/climate_indices/pull/1405), [#1410](https://github.com/monocongo/climate_indices/pull/1410), [#1412](https://github.com/monocongo/climate_indices/pull/1412), [#1424](https://github.com/monocongo/climate_indices/pull/1424) | Beta Dataset routes and eager catalogs have bounded execution, allocation guards, a published-method reproduction, and characterized inference limits. |
| Standardization extensions | [#1346](https://github.com/monocongo/climate_indices/pull/1346), [#1354](https://github.com/monocongo/climate_indices/pull/1354), [#1385](https://github.com/monocongo/climate_indices/pull/1385), [#1389](https://github.com/monocongo/climate_indices/pull/1389) | The shared standardization API gains explicit Aggregation, empirical and beta distributions, and documented public contracts. |
| Energy foundation and evidence | [#1357](https://github.com/monocongo/climate_indices/pull/1357), [#1359](https://github.com/monocongo/climate_indices/pull/1359), [#1360](https://github.com/monocongo/climate_indices/pull/1360), [#1362](https://github.com/monocongo/climate_indices/pull/1362), [#1363](https://github.com/monocongo/climate_indices/pull/1363), [#1383](https://github.com/monocongo/climate_indices/pull/1383), [#1390](https://github.com/monocongo/climate_indices/pull/1390) | Scope, CF metadata, applications, the hydropower precipitation-proxy recipe, and reference fixtures support the energy family. |
| Energy physical transforms | [#1376](https://github.com/monocongo/climate_indices/pull/1376), [#1378](https://github.com/monocongo/climate_indices/pull/1378), [#1382](https://github.com/monocongo/climate_indices/pull/1382), [#1395](https://github.com/monocongo/climate_indices/pull/1395), [#1396](https://github.com/monocongo/climate_indices/pull/1396) | Degree days and wind profiles gain DataArray support; wind and PV capacity factors remain NumPy-only transforms. |
| Energy standardized indices and events | [#1391](https://github.com/monocongo/climate_indices/pull/1391), [#1393](https://github.com/monocongo/climate_indices/pull/1393), [#1394](https://github.com/monocongo/climate_indices/pull/1394), [#1399](https://github.com/monocongo/climate_indices/pull/1399) | The family adds standardized production, demand, residual load, energy-drought events, and supported standardization adapters. |
| Energy CLI, hourly support, and documentation | [#1401](https://github.com/monocongo/climate_indices/pull/1401), [#1404](https://github.com/monocongo/climate_indices/pull/1404), [#1408](https://github.com/monocongo/climate_indices/pull/1408), [#1413](https://github.com/monocongo/climate_indices/pull/1413), [#1455](https://github.com/monocongo/climate_indices/pull/1455) | The CLI exposes energy indices and hourly empirical standardization; the documentation includes evidence limits and a cached ERA5 demonstration. |
| Rust public package and verification | [#1361](https://github.com/monocongo/climate_indices/pull/1361), [#1377](https://github.com/monocongo/climate_indices/pull/1377), [#1381](https://github.com/monocongo/climate_indices/pull/1381), [#1384](https://github.com/monocongo/climate_indices/pull/1384), [#1403](https://github.com/monocongo/climate_indices/pull/1403), [#1409](https://github.com/monocongo/climate_indices/pull/1409) | The renamed standalone crate has public kernel contracts, numerical documentation, Python-independent verification, package metadata, and runnable examples. |
| Rust performance and release gates | [#1411](https://github.com/monocongo/climate_indices/pull/1411), [#1414](https://github.com/monocongo/climate_indices/pull/1414), [#1415](https://github.com/monocongo/climate_indices/pull/1415), [#1426](https://github.com/monocongo/climate_indices/pull/1426) | Criterion benchmarks, compatibility and minimum-Rust checks, packaged documentation, trusted publishing, and a pre-release rehearsal support crate delivery. |
| Flood event summaries | [#1425](https://github.com/monocongo/climate_indices/pull/1425) | Flood events reuse Run statistics and add per-hydrological-year summaries; this is not a new hydrologic model. |
| Icechunk I/O | [#1445](https://github.com/monocongo/climate_indices/pull/1445), [#1447](https://github.com/monocongo/climate_indices/pull/1447), [#1450](https://github.com/monocongo/climate_indices/pull/1450) | Optional input/output support, storage and session tests, and an SPI/SPEI notebook add an Icechunk workflow without changing the core install. |
| Correctness and user guidance | [#1333](https://github.com/monocongo/climate_indices/pull/1333), [#1429](https://github.com/monocongo/climate_indices/pull/1429), [#1449](https://github.com/monocongo/climate_indices/pull/1449), [#1453](https://github.com/monocongo/climate_indices/pull/1453) | The README reflects published 3.0.0; Pearson fallback guidance, adapter keyword rejection, and PNP ratio metadata correct user-facing behavior and documentation. |
| Recurrence performance | [#1355](https://github.com/monocongo/climate_indices/pull/1355), [#1356](https://github.com/monocongo/climate_indices/pull/1356) | KBDI computes its climatology denominator once per cell; daily recurrence execution uses one pass. |
| Dependency, security, and script maintenance | [#1334](https://github.com/monocongo/climate_indices/pull/1334), [#1347](https://github.com/monocongo/climate_indices/pull/1347), [#1348](https://github.com/monocongo/climate_indices/pull/1348), [#1349](https://github.com/monocongo/climate_indices/pull/1349), [#1350](https://github.com/monocongo/climate_indices/pull/1350), [#1351](https://github.com/monocongo/climate_indices/pull/1351), [#1352](https://github.com/monocongo/climate_indices/pull/1352), [#1427](https://github.com/monocongo/climate_indices/pull/1427), [#1428](https://github.com/monocongo/climate_indices/pull/1428), [#1443](https://github.com/monocongo/climate_indices/pull/1443), [#1451](https://github.com/monocongo/climate_indices/pull/1451) | Workflow actions and locked dependencies receive updates, including certifi and Pillow security repairs; a comparison script receives cleanup. |

## Tracker reconciliation and remaining gates

At this review, both [PR #1449](https://github.com/monocongo/climate_indices/pull/1449)
and [#1315](https://github.com/monocongo/climate_indices/issues/1315) already carry milestone 8.
No reassignment is necessary.
The exclusion tickets already carry [v3.1.1, milestone 11](https://github.com/monocongo/climate_indices/milestone/11).

- [#1459](https://github.com/monocongo/climate_indices/issues/1459) reconciles API records after PR #1456 merges.
- [#1460](https://github.com/monocongo/climate_indices/issues/1460) refreshes the release-facing documents and derived bundles.
- [#1461](https://github.com/monocongo/climate_indices/issues/1461) completes the Icechunk release documentation.
- [#1462](https://github.com/monocongo/climate_indices/issues/1462) extends release-integrity and wheel coverage.
- [#1463](https://github.com/monocongo/climate_indices/issues/1463) prepares the version, lockfile, changelog, and curated release notes.
- [#1464](https://github.com/monocongo/climate_indices/issues/1464) records the Python rehearsal on the frozen candidate.
- [#1465](https://github.com/monocongo/climate_indices/issues/1465) records maintainer-only release hygiene.
- [#1466](https://github.com/monocongo/climate_indices/issues/1466) records post-release verification.

The crate's first functional publication remains subject to #1314 and #1315.
The current `0.0.0` name reservation is not the `0.1.0` functional release.
These gates do not authorize an agent to merge, push release tags, or publish either package.
