# Palmer Rust kernels: parity, branch stability, and cross-platform reproducibility

**Ticket:** REFIMPL-34, [#1636](https://github.com/monocongo/climate_indices/issues/1636). **Map:** [#1602](https://github.com/monocongo/climate_indices/issues/1602).
**Pinned version:** `380aa01f` (`origin/main`, 2026-11-21).
**Status:** pre-registered. Everything down to "Thresholds" is committed before any measurement is run. "Deviations" and "Results" are added afterwards, in later commits.

## Question

Can the manuscript state that the Rust and Python Palmer paths give the same results, on which platforms, and to what tolerance? The Palmer recursions branch on exact comparisons with zero and on fixed thresholds, so a one-ulp difference upstream of a sentinel could flip a branch and produce a large difference. The registry tolerance of `1e-10` does not rule that out on inputs outside the registry.

## Claims being tested

None of these is a finding yet. Each is a lead to confirm or refute at the pinned version.

| Lead | Source |
| --- | --- |
| Rust and Python `pdsi()` and `scpdsi()` agree within `rtol = atol = 1e-10` on all 344 divisions, with matching signs | `tests/test_native_parity_palmer.py`, `test_every_division_matches_for_pdsi` and `..._for_scpdsi` (asserts a tolerance, records no maxima) |
| The Palmer kernels contain no fused multiply-add | `crates/climate-indices/src/palmer*.rs` has no `mul_add`; `docs/architecture.md` limits FMA to `special::mul_add` on aarch64 |
| The release build keeps IEEE semantics and sets no `target-cpu` | `Cargo.toml` `[profile.release]`; no `.cargo/config` in the tree |
| The scPDSI rescale and percentile run in NumPy, not Rust; only the recursion and the duration factors are Rust | `src/climate_indices/drought/palmer.py`, `_rescale_scpdsi_zindex` |
| Native Palmer is tested only on Linux x86-64 and macOS arm64 | `.github/workflows/unit-tests-workflow.yml`, `test-native` job |

## Sample

- All 344 climate divisions, 1,536 months each (1895-01 to 2022-12), from `tests/fixture/palmer/<division>/` (`precips.npy`, `pet.npy`) and `tests/fixture/palmer_awc.json`. Calibration 1931-1990, as in `tests/conftest.py`.
- Entry points: `palmer.pdsi` and `palmer.scpdsi`, run twice per division, once with `palmer._native` set to the extension and once to `None`.
- Outputs compared: every array in the return value (PDSI, PHDI, PMDI, Z-Index, and for scPDSI the same set), plus the fitted parameter dictionary.
- Not in the sample: the T4 CONUS grid (tier not yet assembled, owned by REFIMPL-10, [#1612](https://github.com/monocongo/climate_indices/issues/1612)); `pdsi()` on that grid is deferred to it.

## Hypotheses

| Id | Hypothesis | Refuted if |
| --- | --- | --- |
| H1 | Maximum absolute difference between the Rust and Python result is at most `1e-10`, for every output, on all 344 divisions, for both `pdsi()` and `scpdsi()` | any output exceeds `1e-10` |
| H2 | No month has a different spell state between the two paths: the sign of every output element matches exactly, and NaN positions match | any month differs in sign, zero-ness, or NaN position |
| H3 | Engineered series that sit on each sentinel and threshold, and one ulp either side, take the same branch on both paths | any engineered series gives different signs, or a difference above `1e-10`, between paths |
| H4 | The Rust results are bit-identical across the five wheel targets of `release.yml` (Linux x86-64 and aarch64, macOS arm64 and x86-64, Windows x86-64) | any output of any division differs in its float64 bytes between two platforms |
| H5 | The pure-Python results are also bit-identical across the same platforms | same, for the Python path |

H5 is a control: if H4 holds and H5 does not, the difference lies in NumPy or the platform math library, not in the Rust kernels.

## Method

1. **Full-fixture parity (H1, H2).** `scripts/measure_palmer_parity.py` runs both paths on each division and records, per output, the maximum absolute and relative difference, the count of months whose sign differs, and the count of NaN-position mismatches. It also records the toolchain: `rustc -vV`, `rustc --print cfg` target features, the release profile, and the NumPy and platform versions.
2. **Engineered boundary series (H3).** A deterministic generator in the test module builds Z series, run through `palmer._pdi_recursion` and `palmer._wells_recursion` on both paths. Each series places a value on a boundary and then on the neighbours at `np.nextafter` distance. The boundaries come from the branch inventory in "Branch inventory" below. The committed divergence sequences in `palmer-recursion-divergence-sequences.json` are also run through both Rust and Python.
3. **Gaps and zeros.** NaN gaps and zero precipitation are added to the division inputs, because the parity registry's input builders clip precipitation to [5, 300] and fill NaN, so they never produce either.
4. **Cross-platform (H4, H5).** `.github/workflows/palmer-reproducibility.yml`, `workflow_dispatch` only, runs the measurement with `--digest` on each target and compares the SHA-256 of the raw float64 bytes per division and output across platforms. A platform not yet run is reported as not run, not as passing.

## Branch inventory

Boundaries the engineered series target, from `_palmer_wells.py` and `_palmer_pdi.py` and their Rust mirrors `palmer_wells.rs` and `palmer_pdi.rs`:

- Spell thresholds: `|z| >= 0.5 * slope` in the Wells continuation; `x1 >= 0.5` and `x2 <= -0.5` in the Wells establishment; `|x3| <= 0.5`, `x3 > 0.5`, `x3 < -0.5` and `z >= 0.15`, `z <= -0.15` in the `pdi.f` month advance.
- Probability bounds: `prob <= 0`, `prob >= 100`, and `prob >= 100 - 1e-5` (`_TOLERANCE`) in the Wells abatement.
- Exact zeros: `x1`, `x2`, `x3` equal to `0.0` in `_pmdi`, `_continue_spell`, `_establish_spell`, the backtrack, and the PHDI fallback; `px1 >= 1` and `px2 <= -1` with `px3 == 0` in `_statement_200`.
- Clamps: `max(0, ...)` and `min(0, ...)` in `_candidate_values`.

## Thresholds

- **Accept parity** (H1, H3): `|rust - python| <= 1e-10` absolute and relative, and exact sign and NaN-position equality. These are the registry's values; they are not tuned to the result.
- **Pinned maxima:** the tests pin the measured maximum per output, never above `1e-10`. If the measured maximum is `0.0`, `0.0` is pinned.
- **Cross-platform** (H4, H5): equality of the float64 bytes, no tolerance. A difference is a finding to attribute (fused multiply-add, the math library, NumPy), not a failure to hide.
- **Branch flip:** any engineered or fixture case where the paths take a different branch, recorded in the REFIMPL-24 ledger ([#1626](https://github.com/monocongo/climate_indices/issues/1626)) with its cause and owner.

## Out of scope

Performance ([#1637](https://github.com/monocongo/climate_indices/issues/1637)); the T4 grid ([#1612](https://github.com/monocongo/climate_indices/issues/1612)); changing any kernel, tolerance, or dispatch rule.
