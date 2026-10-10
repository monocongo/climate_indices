# climate\_indices 3.0 — release story notes

Oct 10, 2026 · James Adams

Draft communications notes, not a publication record. As of this snapshot, 3.0.0 is not tagged or published; wheel coverage below describes the release workflow's planned artifacts. Use the [release runbook](release-process.md) for the final cut and the [release notes](release-notes-3.0.0.md) for upgrade guidance.

## The headline

In six months, climate\_indices grew from a drought-index library into a multi-hazard climate index toolkit, with an optional Rust backend that accelerates eligible numerical calls. Palmer PDSI measured about 194x faster on a station series and 2.77x on a full-grid eager call on the documented macOS host; neither is a portable speed guarantee.

Version 3.0 adds wildfire and flood index families beside the drought indices the package is known for. It adds a self-calibrated Palmer index, a public validation layer backed by external reference datasets, and gridded xarray paths that replace per-cell loops. The Rust kernels are optional: Python stays the public API, and the retained Python implementations remain as parity oracles at `rtol = atol = 1e-10` with matching NaN positions.

## By the numbers

The 3.0 cycle ran six months, merged 366 pull requests, and more than doubled both the Python code and the test suite.

| Measure | 2.4.0 | 3.0 (main, Oct 9 2026) |
| --- | --- | --- |
| Release window | tagged Apr 6, 2026 | work Apr 8 – Oct 9, 2026 |
| Commits since 2.4.0 | — | 2,218 |
| Pull requests merged | — | 366 |
| Python source lines | 12,017 | 28,900 |
| Rust source lines | 0 | 8,784 (2 crates) |
| Test files | 35 | 107 |
| Test functions | 881 | 2,384 |
| Release wheel plan | 1 pure-Python | pure-Python + 5 Rust platform wheels (Linux, macOS, Windows; Python 3.10–3.14) |

Snapshot: `74fef017f10913b135b194ee1ce8cd294ef04184` (main, Oct 9, 2026), compared with tag `v2.4.0`. Counts use tracked `src/climate_indices/**/*.py`, `crates/**/*.rs`, test files named `test_*.py`, and AST test-function definitions, not collected parametrized cases. The 366 PR count is reachable merge commits whose subjects start with `Merge pull request #`, not a live GitHub query. Refresh against the final tagged commit before using these figures as release totals.

Human contributors since 2.4.0, by commits: James Adams (2,006), McLain Adams (124), marekl11 (4), and mihailchkik (2), counted from git history at this snapshot.

## From drought to multi-hazard

3.0 adds two new hazard families, fire weather and flood potential, to the drought indices, so one package now covers dry, fire-prone and flood-prone conditions.

| Family | What's in it | How you can call it |
| --- | --- | --- |
| Drought (extended) | Self-calibrated PDSI (`scpdsi`), log-logistic SPEI, generic `standardized_index()` for runoff and streamflow (SRI/SSI), FAO-56 Penman-Monteith PET, threshold-run analysis (`runs`) | NumPy; xarray for PDSI, SPI, SPEI, EDDI, PET; CLI |
| Fire weather (new) | Keetch-Byram Drought Index, Canadian FWI system (FFMC, DMC, DC, ISI, BUI, FWI, DSR) with overwintering, Fosberg FFWI, Hot-Dry-Windy Index, Haines Index | NumPy; xarray for KBDI, CFFWIS, HDW, Haines; CLI for KBDI |
| Flood potential (new) | Byun-Wilhite effective precipitation, Effective Drought Index (EDI), Flood Index, Antecedent Precipitation Index, flood event extraction | NumPy (stable); xarray (beta); CLI |

Two phrasing notes for posts: the flood indices measure flood *potential*, not observed flooding, and EDI (Effective Drought Index) is not EDDI (Evaporative Demand Drought Index). The release notes make both points explicitly.

## Rust backend

Use the committed measurements merged in [PR #1322](https://github.com/monocongo/climate_indices/pull/1322), not unmerged AWS follow-up results. The macOS arm64 station-series benchmark measured PDSI at 194.06x and scPDSI at 62.45x, best of five after warm-up. Ratios are Python wall time / Rust-enabled wall time for whole calls, including Python orchestration and binding costs.

The separate Apple M5 CONUS measurement covers 469,758 land cells × 528 months. Its single eager-call PDSI ratio is 2.77x; SPI gamma is 1.33x and SPEI gamma 1.57x. Eight spatial blocks on one thread bring SPI gamma to 0.98x: effectively parity. Station-series gains must not be extrapolated to full-grid runtimes or costs. These prepared-input timings exclude loading and land-cell packing.

- **What moved to Rust:** numerical kernels for SPI, SPEI, the standardized index, EDDI, PNP, PCI, Thornthwaite, Hargreaves and Penman-Monteith PET, Palmer and scPDSI, the fire moisture codes and KBDI, and the flood family. Validation, xarray/Dask, metadata, the CLI and I/O stay in Python.
- **Parity contract:** `rtol = atol = 1e-10`, with NaNs in the same positions. Backend parity is regression evidence, not external scientific validation.
- **API compatibility:** adding the backend does not change public signatures, return types or exceptions. Version 3.0.0 itself does have breaking changes; read the [upgrade considerations](release-notes-3.0.0.md#upgrade-considerations).
- **Installation:** five planned `cp310-abi3` wheels cover manylinux_2_28 Linux x86-64/aarch64, macOS arm64/x86-64, and Windows x86-64 on Python 3.10–3.14. Other platforms and source installs use pure Python without a Rust toolchain.
- **Limits:** some inputs and NumPy/warning configurations keep the Python path even when the extension is installed. Penman-Monteith is slower in Rust on the committed station samples; PCI and diagnostics are near parity. Native runtime errors propagate rather than silently retrying Python.

Sources: [benchmark methodology and committed artifacts](https://github.com/monocongo/climate_indices/blob/main/benchmarks/README.md#rust-kernels-vs-the-python-reference-across-the-parity-registry-rust-011), [release notes](release-notes-3.0.0.md#optional-rust-backend). AWS timing evidence remains follow-up work under [issue #1324](https://github.com/monocongo/climate_indices/issues/1324).

## Trust and validation

3.0 checks its indices against published reference data from NOAA, NRCan, SPEIbase and FAO-56, and records the remaining gaps openly in `VALIDATION.md`.

| Index | Reference | Agreement |
| --- | --- | --- |
| EDDI | NOAA PSL reference series | max error 2.43e-6 |
| SPI | NOAA NCEI climate-divisional SPI | characterized across all 344 divisions |
| PDSI | NOAA nClimDiv standard Palmer | median absolute difference 0.0127 |
| scPDSI | Wells-lineage oracle fixtures | within 5e-5 |
| CFFWIS (Canadian FWI) | NRCan reference | within 1e-9 |
| SPEI | SPEIbase v2.11 / R SPEI | input-matched cross-implementation checks; separate gamma/Thornthwaite plausibility floors |
| KBDI | Keetch & Byram (1968), Figure 1 | reproduces the published record |
| Hot-Dry-Windy | Srock et al. (2018) case study | reproduces fire-day peak timing; magnitudes characterization only |
| Penman-Monteith PET | FAO-56 Examples 10–16 and 18 | matches worked examples |

The flood family has no external numeric oracle yet; the release notes call this out as a validation gap. Anyone can rerun the external checks with `pytest -m validation`.

## Usability and API

Gridded work got much faster even before Rust: replacing the xarray per-cell loop with one NumPy call per grid block sped up the 38x87-cell, 40-year reference grid by 4.4x (SPI) to 342x (EDDI).

- **Gridded execution:** SPI, SPEI, EDDI, percentage of normal, PET and PDSI run whole `(time, *cells)` blocks in one pass. Gridded PDSI went from 135.4 s to 1.2 s (111.3x).
- **Zero handling for SPI:** choose classic, center-of-mass (Stagge et al., 2015) or mean-zero (Allen and Otero, 2024) placement of dry periods.
- **Output scales:** SPI, SPEI and `standardized_index()` can return normal scores, probabilities, or bounded values in \[-1, 1\].
- **Fit diagnostics:** per-period fitted parameters, probability of zero, and Kolmogorov-Smirnov goodness of fit.
- **Public validation namespace:** the input checks used inside the package are now public API.
- **Docs:** MyST Markdown with runnable xarray, Zarr and Dask examples; Python 3.10–3.14 supported.

The NumPy API is the stable surface; the xarray DataArray API stays Beta through 3.0. Before upgrading from 2.4.0, review corrected daily calendar alignment, calibration and probability-of-zero behavior, changed exceptions, and removal of the legacy `spi` console script in the [migration guide](deprecations/api-changes.md).
