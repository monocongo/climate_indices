# Intercomparison harness: layout, environments, adapter contract, result store

**Ticket:** REFIMPL-08, [#1610](https://github.com/monocongo/climate_indices/issues/1610). **Status:** the location decision (1) was approved by the maintainer on 2026-11-22; decisions 2 to 5 are proposed with the slice below.

**Question.** What is the smallest harness that runs every comparator on identical inputs, records exactly what each one did, and can be rerun by a reviewer?

**Answer in one line.** A top-level `intercomparison/` directory in this repository: a driver that calls each comparator as a subprocess in its own locked environment through a JSON-and-CSV contract, writes one Zarr store and one JSON manifest per study, and commits only the manifest and derived metrics.

## 1. Location: `intercomparison/` in this repository

| Option | CI cost | `uv.lock` isolation | One frozen archive (REFIMPL-39, [#1641](https://github.com/monocongo/climate_indices/issues/1641)) | Verdict |
| --- | --- | --- | --- | --- |
| Top-level `intercomparison/` (chosen) | No new job: the offline tests add a few seconds to `pytest` | Each comparator is a nested uv project with `package = false`; it is not a member of any workspace, so the root `uv.lock` is unchanged | The directory, `tests/fixture/` and `docs/research/reference-papers/` can be archived with `git archive <tag>` and cited by the tag; the fixtures are already in this repository | Chosen |
| `docs/research/reference-papers/harness/` | None | Same | Same | Rejected: executable code and locked environments under `docs/`, which Sphinx and the reader both treat as prose |
| Sibling repository | Needs its own CI and a pin of this repository by commit | Full isolation | Cleanest citation, but the fixtures must be copied or fetched, and a change to `spi()` and its harness test land in two repositories | Rejected: the cost is paid on every change, the benefit once |

What this changed in the repository, and what was checked:

- `pyproject.toml`: `intercomparison/` added to the sdist `exclude` list. The wheel needs no entry: hatch packages only `src/climate_indices` for this layout. `tests/test_build_provenance.py` builds the sdist (and the wheel from it) and, separately, a wheel straight from the source tree, each with `intercomparison/contract.py` present, and asserts that no member contains it. Removing the sdist entry makes the first assertion fail.
- `.gitignore`: `intercomparison/**/*.zarr/` and `intercomparison/studies/*/work/`.
- `tests/test_release_integrity.py` has no check of top-level directories. Its only `git ls-files` check is for Git LFS pointers, and `.gitattributes` marks only `*.nc` as LFS, so committed JSON and CSV are safe.
- The core package gains no dependency and no public API. `zarr` and `jsonschema`, used by the driver and the tests, are already in the `test` extra.

## 2. Environment per comparator

- **Python comparators:** one uv project each, `intercomparison/adapters/<name>/`, with its own `pyproject.toml` and committed `uv.lock`, run as `uv run --locked --project <dir> python <adapter>`. The `xclim` project pins `xclim==1.0.0` and resolves the same NumPy, pandas and xarray as the survey (`comparators.lock.yaml`: numpy 2.5.5, pandas 3.1.1, xarray 2026.9.0). Its Python is limited to 3.12, the survey's version; the repository environment runs the library on its own Python.
- **R and Octave comparators:** a container per comparator, as the existing recipes in `docs/research/reference-papers/containers/` do. Those pin the base image by digest only for `ndmc-spi`; `linux-smoke` and `ncl` use tags, so a container adopted into the harness must be pinned by digest first. These are not built in this slice. The adapter tickets (REFIMPL-12 to REFIMPL-15, [#1614](https://github.com/monocongo/climate_indices/issues/1614) to [#1617](https://github.com/monocongo/climate_indices/issues/1617)) and the Palmer comparator adapters own them. Open: `comparators.md` notes that CRAN dependencies are not pinned in `containers/linux-smoke/`; the first R adapter must choose a dated package snapshot or an `renv.lock`. That is not decided here.
- GPL comparators are executed, never vendored: an adapter holds a call and a pin, not the comparator's source. Only numeric outputs and provenance are committed.

## 3. Adapter contract

An adapter is a subprocess, `<adapter> run <request.json> <out_dir>`, so environments never mix and a non-Python adapter implements the same two documents. `intercomparison/contract.py` is standard-library only, so every Python adapter can import it.

- **Request:** `index`, `unit`, `positions` (the method positions to apply), and `inputs`, each a series file named by path (relative to the request) and SHA-256. The adapter verifies the hashes before computing; a mismatch is an error.
- **Series file:** `time,value` CSV, values as `repr(float)` (the shortest string that round-trips a binary64 exactly), missing as `nan` (`NaN` is read too). Tested for exact round trip of `0.1 + 0.2`, a subnormal, `-0.0` and NaN. A `path` may not leave the directory of the document that names it, and the bytes parsed are the bytes hashed.
- **Result** (`result.json`): comparator, version, `effective_method`, software versions, platform, wall time, warnings, and each output series by path and SHA-256, verified when read.
- **Positions vocabulary.** The slice uses `accumulation`, `scale`, `distribution`, `estimator`, `zero_handling`, `calendar_grouping` and `calibration_period`: six axis ids of `method_spec.schema.json` plus the accumulation scale, with short tokens as values. Reconciling the tokens with the free-text axis values of `method-matrix.json` is left to the first adapter tickets (REFIMPL-12 to REFIMPL-15).
- **Effective method.** The driver compares each adapter's reported `effective_method` with the requested `positions` and records one `method:` warning per difference; a position the adapter does not report counts as a difference. Values are compared with their type, so `3`, `3.0` and `True` differ. **Limit:** the effective method is what the adapter says it passed. A position that is an argument of the call (`scale`, `distribution`, the calibration period, `estimator` for xclim) is reported from the value passed. A position fixed inside the comparator (for example xclim's rolling mean and zero mass) is a literal in the adapter, with the source it was read from, so it is only as current as that reading of the pinned version. The check catches an adapter that is misconfigured or asked for something it cannot do. It does not catch a comparator that ignores an argument. Detecting that needs a test of the comparator's behaviour, not a property of the contract.

The proposal in the ticket, `run(index, method_spec, inputs) -> Result`, is kept in shape. The request carries `positions`, not a full `method_spec` document, because the `method_spec` schema describes what implementations can do, not what a study asked for.

## 4. Result store: Zarr plus a manifest

- **Store.** One Zarr group per study, array `values` of dtype float64 and dims `(implementation, unit, time)`, written only by the driver in the repository environment. `attrs` carry the implementation order, the units and the time labels. Chosen over Parquet because the studies are arrays and the gridded demonstration (REFIMPL-36, [#1638](https://github.com/monocongo/climate_indices/issues/1638)) is an array; Parquet is the better format for the derived metrics, and those are committed as JSON or CSV.
- **Manifest.** `manifest.json`, validated by `intercomparison/manifest.schema.json`. It reuses the provenance field names of `tests/fixture/provenance_schema.json` (`source`, `url`, `download_date`, `subset_description`, `checksum_sha256`, `fixture_version`, `license`, `notes`). It omits `validation_tolerance`: tolerances belong to each study's pre-registration under the protocol (`protocol.md`). It adds the harness commit and dirty flag, the positions, each input's source path and both hashes, and per implementation the version, effective method, software, platform, wall time, warnings and output SHA-256.
- **Reproducibility.** A rerun reproduces every `output_sha256` and `checksum_sha256`; wall time (adapters import their comparator before the timer starts), platform, the creation date and the harness commit and dirty flag differ. Tested offline (`test_rerun_gives_identical_output_hashes`), and by running the documented commands in a fresh clone of the branch, where every output hash matched the committed manifest. `git_dirty` counts untracked files.

## 5. What is committed

| Committed | Not committed |
| --- | --- |
| `manifest.json` per study, derived metrics (REFIMPL-16), small paired fixtures, the adapters, the contract, the schema, each adapter's `uv.lock` | `store.zarr`, `work/` (requests, input CSVs, per-adapter results), each adapter's `.venv` |

Bulk size, for planning: 344 divisions x about 15 implementations x 1,536 months x 8 bytes is about 63 MB per index and configuration, which stays in git-ignored stores. Anyone can regenerate a store from the manifest's inputs, pins and one command.

## The slice

Gamma SPI-3 for division 0101 of `tests/fixture/palmer/` (precipitation, inches, 1895-01 to 2022-12), through `climate_indices` 3.2.0 and `xclim` 1.0.0 in the aligned configuration `spi-gamma-thom`, to `intercomparison/studies/refimpl-08-slice/manifest.json`. Rerun: [`intercomparison/README.md`](../../../intercomparison/README.md).

Observation, not a result (metrics are REFIMPL-16): 1,534 of 1,536 months are paired (the first two have no three-month sum in either). 1,533 differ by at most 7.4e-14. One month (1965-12, index 851) differs by 0.0627: the library clips SPI at -3.09 (its output bound) and xclim's bound is ±8.21, so it returns -3.1527. The three are logged in the [discrepancy ledger](discrepancy-ledger.md) as P08-N, P08-B and P08-A. The `climate_indices` adapter reported no method difference. The `xclim` adapter reported one: it applies a rolling mean where the request says sum, which gives the same gamma SPI. Division 0101 has no zero-precipitation month, so the `zero_handling` position is declared but not exercised in this slice.

## Not done here

- Adapters beyond the slice (REFIMPL-12 to REFIMPL-15), and moving the survey containers.
- Metrics, the results schema and figures (REFIMPL-16, [#1618](https://github.com/monocongo/climate_indices/issues/1618); REFIMPL-40, [#1642](https://github.com/monocongo/climate_indices/issues/1642)).
- A CI job that runs a comparator. Heavy runs stay manual, as for the Earth Engine pilot (`scripts/earthengine_climate_engine_pilot.py`).
- A native x86_64 run of the slice, and a CI run of the Linux smoke in `containers/linux-smoke/` (both assigned to REFIMPL-08 in `comparators.md`). The slice ran on macOS arm64 only.
- A zero-heavy division, to exercise `zero_handling`.
