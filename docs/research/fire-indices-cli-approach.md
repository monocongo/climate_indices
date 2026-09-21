# Exposing fire-weather indices (KBDI first) via the climate_indices CLI

- **Research date:** 2026-09-16
- **Scope:** Read-only recon of the working tree at HEAD `d28825c4` on `main`: CLI source, fire API, tests, docs/ADRs, and the GitHub issue trail for KBDI CLI exposure. No files changed; no builds run.

## Recorded conflict

#802 ("FIRE-10 — KBDI CLI integration", milestone `v2.6 — Fire subsystem foundation`, depends on closed #801) says: *"Expose KBDI through the existing `__main__.py` command-line processing entry point"* and tasks *"Add `kbdi` to the index choices"*, `--netcdf_tmax`, `--kbdi_units`, `--kbdi_initial`, *"Reject scale-based arguments (`--scales`)"*, CF-compliant output, fail-fast invalid combos. The maintainer comment on #802 records: *"`docs/design/fire-subsystem.md:22` states 'No fire CLI is part of this subsystem.' … Resolving this needs a maintainer scope call — either amend `fire-subsystem.md:22` … or close this out as declined."* It also flags the two hard conflicts: daily 366-day reshape vs. KBDI cross-year recursion, and unconditional temperature→Celsius vs. `kbdi(..., units="imperial")`.

Earlier wayfinder map #786 (2026-09-09) explicitly notes: *"Include a standalone `--index kbdi` CLI/NetCDF path. Do not add KBDI to the existing `--index all` aggregate."* Its child #791 ("Design the standalone KBDI CLI and NetCDF contract", open, blocked by #790, blocks #792) asks how `--index kbdi` should accept inputs and *"Prefer the smallest extension of the existing CLI path."* No answer is recorded on #791 yet.

## Observed CLI anatomy

`src/climate_indices/__main__.py` (1675 lines) is a flat argparse CLI, no subparsers:

- `main()` at :1417 builds the parser at :1439; `--index` choices `["spi","spei","pnp","scaled","pet","palmers","all"]` at :1443; `--netcdf_temp`/`--var_name_temp` (not `--netcdf_tmax`) at :1451-1458; `process_climate_indices()` at :1491 dispatches per-index branches at :1514 (spi/scaled/all), :1543 (pet/spei/scaled/palmers/all), :1570 (spei), :1604 (pnp), :1631 (palmers). `--index all` is a superset selector, so KBDI must be excluded from all branches (#786 requirement).
- Shared arguments in `src/climate_indices/_cli.py:71` (`_add_common_spi_arguments`): `--periodicity` required, choices monthly/daily (:84-90); `--scales` optional `nargs="*"` (:91-96); `--output_file_base` required (:111). `_prepare_file` (`_cli.py:16`) validates dimension names only — its docstring says *"returned unchanged; no dimensions are reordered"* (:20), contradicting the callers' "rearrange" comments.
- `_validate_args()` at :48 requires precip for every index except `pet` (:69), validates grid/division/timeseries dims, and requires `--scales` only for `["spi","spei","scaled","pnp"]` (:407-419). PET is the precedent for a periodicity restriction: daily rejected with a clear message at :126-129.
- `_compute_write_index()` at :654 is the drought/scaled engine. Daily gridded input is reshaped into 366-day years in `_drop_data_into_shared_arrays_grid` at :566-576 (`utils.transform_to_366day`, applied to every variable before index dispatch); results are transformed back at :1012-1022 (`transform_to_gregorian`). Units are coerced before dispatch: precipitation inches→mm at :725-742, temperature Fahrenheit/Kelvin→Celsius at :740-762. Output files are `output_file_base + "_" + var_name + ".nc"` (:1042; Palmer variants :884-938); attributes come from `_get_variable_attributes` (:495-533), which raises for any index outside spi/spei/pnp/pet (:528). `_build_arguments` raises at :490 and index dispatch raises at :1003 for unknown indices.
- Multiprocessing scaffolding: `_parallel_process` (:1116) splits work on `shape[0]` (:1138-1141); `_apply_along_axis` (:1256) maps grid→axis 2 (time), divisions→axis 1, timeseries→axis 0 (:1281-1288). For gridded input this is structurally compatible with KBDI (split lat, full time per worker); for 1-D timeseries it splits the time axis, breaking the recurrence. ADR-0002 records the CLI-multiprocessing vs. API-Dask split.
- Output metadata is hardcoded in `__main__.py`/`__spi__.py`, and open issue #661 says `cf_metadata_registry.py` is the source of truth for CF metadata.

## Precedent for a second console script

`pyproject.toml:76-79`: `climate_indices` and `process_climate_indices` both point at `climate_indices.__main__:main`; `spi = "climate_indices.__spi__:main"`. `__spi__.py` (1337 lines) duplicates `InputType` (:37), its own `_validate_args` (:47), `_get_variable_attributes` (:185), shared-array/multiprocessing code, and its own `_compute_write_index` (:459); it shares from `_cli.py` only `_add_common_spi_arguments` and `_prepare_file` (imports at `__spi__.py:14-15`). So the "second console script" precedent exists, but it demonstrates duplicated plumbing rather than a thin wrapper.

## Fire API relevant to CLI

`src/climate_indices/fire.py`:

- `kbdi(precipitation, maximum_temperature, mean_annual_precipitation=None, *, units="metric", initial_kbdi=None, initial_state=None, return_state=False, spin_up=0, nan_policy="propagate", max_gap_days=0, time_dim="time")` — overloads :314/:331, implementation :347, full docstring :364-450. NumPy and xarray inputs; array inputs are time-first and must be consecutive daily observations. Metric: mm/day, °C, mm/year; imperial: inches/day, °F, inches/year. `initial_kbdi=None` selects 0 (:399); `KBDIState`/`KBDIResult` at :118/:135; `_validate_kbdi_configuration` at :254 enforces units/nan_policy/max_gap_days/spin_up/initial exclusivity.
- Omitting `mean_annual_precipitation` derives it from each cell's record if it has at least `30*365 = 10950` finite daily values (`_KBDI_MINIMUM_MEAN_ANNUAL_RECORD_DAYS` :112; check :517), else raises.
- xarray route `_kbdi_xarray` :822 validates daily cadence/monotonicity, aligns inputs, enforces a single Dask time chunk, dispatches the NumPy core per spatial chunk, and attaches CF attrs at :1037-1044 resolving `"kbdi"` vs. `"kbdi_imperial"` from `units` at call time. It returns an `xr.DataArray`; there is no NetCDF writer.
- `fosberg_ffwi(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, cap_at_100=True)` :1148 and `hot_dry_windy(temperature_celsius, relative_humidity_percent, wind_speed_meters_per_second, height_agl_meters, *, level_axis=-1)` :1265 are NumPy-only, elementwise; no xarray adapters, so no CLI plumbing can reuse them directly. CFFWIS is unimplemented (#803, milestone v2.7).
- `cf_metadata_registry.py:165-225`: entries `kbdi` (mm, variant metric), `kbdi_imperial` (0.01 in, variant imperial), `ffwi` (dimensionless), `hdw` (hPa m s-1); comment records that CFFWIS and Haines entries are deferred; no fire index has an official CF `standard_name`.

## Options

**(A) Extend the existing CLI — `--index kbdi` in `__main__.py`.** Matches #802's wording/tasks and #786's mapped decision; one command, one install surface. Reuses argparse, `_prepare_file`, output naming, logging. But `_compute_write_index` cannot be used as-is: the 366-day transform (:566-576) pads synthetic Feb 29s before the recurrence, and the temperature coercion (:740-762) converts °F→°C exactly when `units="imperial"` needs °F — additionally precipitation inches→mm (:725-742) is converted before KBDI would read the array as inches, so imperial values would be wrong by 25.4×. Concrete failure modes if naively extended: monthly `--periodicity` accepted (only PET rejects at :126-129) and would silently compute the wrong cadence; `--scales` is optional and silently ignored for non-scaled indices, so #802's rejection requirement needs an explicit check; 1-D timeseries input would be split along time by `_parallel_process` (:1138-1141) + `_apply_along_axis` axis=0, corrupting the recurrence; 366-day reshaping changes the state trajectory even when values otherwise line up. Smallest safe form: an isolated `index == "kbdi"` branch in `process_climate_indices` that bypasses `_compute_write_index`, calls the beta xarray `fire.kbdi` on the two DataArrays, and writes the returned CF-annotated DataArray. That reuses `_prepare_file` and conventions without touching generic helpers. New code: parser flags, validation block, ~40-60 line branch; one test. Extends later to FFWI/HDW only after their xarray adapters exist; CFFWIS after #803/#804 plus the multi-output adapter (design doc, API tiers).

**(B) Separate fire console script (e.g. `fire_indices = climate_indices.__fire__:main`).** Precedent exists (`spi` at `pyproject.toml:79`), and a KBDI script could be thin because `fire.kbdi`'s xarray path already owns units, cadence validation, and CF metadata — no 366-day or Celsius conflicts, no `--scales`. But it duplicates argument parsing/IO (see `__spi__.py`'s 1337 lines sharing only two helpers) and splits the user-facing workflow across commands; conflicts more strongly with `fire-subsystem.md:22` (a whole new script, not a branch), so it needs at least the same doc amendment plus a recorded entry-point decision. Extensibility is the open cost: FFWI/HDW/CFFWIS have no xarray adapters, so the script would need its own NumPy IO layer or gated subcommands.

**(C) Fire subcommand within the existing CLI.** No subparsers exist (`__main__.py:1439`, `__spi__.py:1222`); introducing them changes the invocation grammar for every current user, and `--index` already is the index selector. This is (A) with extra churn and no observed precedent. Reject unless the CLI is being restructured for other reasons.

**(D) No CLI (library-only).** Literally consistent with `fire-subsystem.md:22` and `ADR-0005:31`, and matches the maintainer's fallback ("close this out as declined"). Does not meet #802's acceptance criteria ("End-to-end run over an example NetCDF produces a CF-compliant output file") or #786's mapped destination; would require closing #802 as declined.

## Recommendation

**(A), an isolated KBDI branch, `--index kbdi` in `__main__.py`** — the smallest surface that satisfies #802 and #786's explicit "smallest extension of the existing CLI path", provided the branch does not route through `_compute_write_index`'s daily path.

Concrete minimum:

1. Add `kbdi` to `--index` choices (:1443) and exclude it from every `--index all` branch (:1514-1631).
2. Reuse `--netcdf_precip`/`--var_name_precip` and `--netcdf_temp`/`--var_name_temp` for daily precipitation and daily maximum temperature (the issue says `--netcdf_tmax`; existing CLI naming is `--netcdf_temp` — resolve before implementing).
3. Add `--kbdi_units {metric,imperial}` (default metric) and `--kbdi_initial` (float, default 0.0). Omit mean annual precipitation in v1: `fire.kbdi` derives it from ≥30 years (:112, :517) — callers with short records get the library's clear error.
4. In `_validate_args` add a KBDI block: require `--periodicity daily` (mirror the PET monthly-only check at :126-129); reject `--scales`, `--calibration_start_year/end_year`, `--netcdf_pet`, `--netcdf_awc` when `index == "kbdi"`; the existing precip-required (:69) and dimension checks cover the rest.
5. Compute via the xarray route `fire.kbdi(precip_da, tmax_da, units=..., initial_kbdi=...)`; write `values.to_netcdf(f"{output_file_base}_kbdi.nc")` (or follow the `base_var.nc` convention at :1042). CF metadata comes from `_kbdi_xarray`/:1037-1044, satisfying "CF-compliant output" without hand-written attrs (cf. open #661).
6. Smoke test following `tests/test_main_palmers.py`'s in-memory `argparse.Namespace` + monkeypatched `xr.open_dataset` pattern; no committed daily NetCDF is available (see below).

Must be amended before/with the change: `docs/design/fire-subsystem.md:22` ("No fire CLI is part of this subsystem.") needs scoping — it predates the KBDI xarray adapter. `ADR-0005:31` is self-scoped ("introduced by this decision") and needs no edit. Record the CLI entry-point/units decision in #791 (and optionally a new ADR), then let #792 produce the executable blueprint. Docs: add a `--index kbdi` example in `docs/index.md` (CLI examples start :171) and a line in `docs/wildfire_applications.md` (:109-122 roadmap).

Follow-ups: `--netcdf_mean_annual` (or climatology variable) once derivation is deemed insufficient; `nan_policy`/`max_gap_days` flags if gridded gaps are expected (default `propagate` poisons every subsequent value after one missing day, per ADR-0007); FFWI/HDW CLI after their xarray adapters and CF wiring; CFFWIS after #803/#804 and the multi-output adapter.

## Test surface

- `tests/test_cli_common.py` (82 lines) covers `_prepare_file` dimension accept/reject (:46, :62) and `_add_common_spi_arguments` parsing (:69). Nothing invokes `main()` end-to-end.
- `tests/test_main_palmers.py` builds an `argparse.Namespace` and monkeypatches `cli_main.xr.open_dataset` with in-memory `xr.Dataset`s (from :55) — the reusable template for a KBDI CLI smoke test.
- No daily NetCDF fixture is checked out: `example_data/example_nclimgrid_lowres.nc` is a 130-byte Git LFS pointer (`.gitattributes:1` `*.nc filter=lfs`), and `data/e2e/` is gitignored (`.gitignore:109`). KBDI fixtures are CSV/npy: `tests/fixture/kbdi_ghcn/fresno_1991_2020.csv`, `tests/fixture/kbdi_se38_figure1/figure1.csv`. Xarray KBDI tests live in `tests/test_fire_kbdi.py` from :827; registry assertions in `tests/test_cf_metadata.py:188`.

## Unknowns / open questions for the maintainer

- `example_nclimgrid_lowres.nc` contents could not be verified locally (LFS pointer only). Is it monthly? If so, where should the "trimmed `example_data` file" for the #802 smoke test come from — a new committed daily fixture, or the in-memory pattern?
- Flag naming: keep existing `--netcdf_temp`/`--var_name_temp` or add `--netcdf_tmax`/`--var_name_tmax` as #802 literally writes?
- Output variable and file naming for imperial: variable `kbdi` with `0.01 in` units vs. `kbdi_imperial`; file suffix `_kbdi.nc`?
- Should the CLI expose `nan_policy`/`max_gap_days` at all in v1, or is default `propagate` acceptable for gridded inputs?
- Is a new ADR required for "fire indices are exposed through the existing CLI", or is the `fire-subsystem.md:22` edit plus #791 resolution enough?
- Who resolves #791/#792 before #802 implementation? #791 is blocked by #790 (prototype), which is open.

## Sources

- `src/climate_indices/__main__.py`, `src/climate_indices/_cli.py`, `src/climate_indices/__spi__.py`, `src/climate_indices/fire.py`, `src/climate_indices/cf_metadata_registry.py`
- `pyproject.toml:76-79`; `tests/test_cli_common.py`; `tests/test_main_palmers.py`; `tests/test_fire_kbdi.py`; `tests/test_cf_metadata.py`
- `docs/design/fire-subsystem.md` (esp. :22); `docs/adr/0001-dual-numpy-xarray-api.md`; `docs/adr/0002-multiprocessing-cli-dask-xarray.md`; `docs/adr/0005-fire-module-api.md` (its Decision section's flat-module deferral); `docs/adr/0006-fire-recursive-state-and-execution.md`; `docs/adr/0007-fire-missing-data-policy.md`
- `docs/index.md:171`; `docs/wildfire_applications.md`; `docs/research/interactive-climate-explorer-landscape.md` (format)
- Issues: [#802](https://github.com/monocongo/climate_indices/issues/802), [#801](https://github.com/monocongo/climate_indices/issues/801), [#793](https://github.com/monocongo/climate_indices/issues/793), [#786](https://github.com/monocongo/climate_indices/issues/786), [#791](https://github.com/monocongo/climate_indices/issues/791), [#792](https://github.com/monocongo/climate_indices/issues/792), [#785](https://github.com/monocongo/climate_indices/issues/785), [#803](https://github.com/monocongo/climate_indices/issues/803), [#798](https://github.com/monocongo/climate_indices/issues/798), [#661](https://github.com/monocongo/climate_indices/issues/661)
