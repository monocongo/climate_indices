# Palmer Rust kernels: parity, branch stability, and cross-platform reproducibility

**Ticket:** REFIMPL-34, [#1636](https://github.com/monocongo/climate_indices/issues/1636). **Map:** [#1602](https://github.com/monocongo/climate_indices/issues/1602).
**Pinned version:** `380aa01f` (`origin/main`, 2026-11-21).
**Status:** measured on macOS arm64; the other four platforms are not run (see "Results"). Everything down to "Out of scope" was committed (`eb710e28`) before any measurement ran; "Deviations" and "Results" were written afterwards. Reproduce with `uv run python scripts/measure_palmer_parity.py measure --out parity.json` (about 45 seconds on four workers) and `uv run pytest tests/test_native_parity_palmer.py`.

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

## Deviations

All recorded after the first measurement run, because each was found while building the tests.

- **D1.** The plan said a `--digest` flag would add SHA-256 digests. The script records them always, for both the Rust and the Python path, because H5 needs the Python digests and a measurement without them cannot be compared later.
- **D2.** The plan said to pin the measured maxima. Only macOS arm64 is measured, so only that platform pins `0.0` (`PINNED_RTOL`, `PINNED_ATOL` in `tests/test_native_parity_palmer.py`); every other platform keeps the registry's `1e-10` until the workflow reports it. A `0.0` pin on an unmeasured platform would be a guess.
- **D3.** The first boundary tests put one boundary value in a series and compared the two paths. They passed, but a probe showed they were vacuous: a lone `z` on a threshold gives `z / 3` whichever branch is taken, and an injected Rust mutant (`pv > 0.0` in place of `pv >= 0.0`) was not caught. They were replaced by corpora of (one ulp below, on, one ulp above) triples with a floor on how many triples change the answer, and the mutants were then caught (see "Results").
- **D4.** The plan listed NaN gaps as a new case. `test_missing_months_match` already covers them, so only zero precipitation was added.

## Results

Pinned version `380aa01f`, rustc 1.99.0 (LLVM 23.1.1), `aarch64-apple-darwin`, Python 3.14.7, NumPy 2.4.6, release profile `lto = "thin"`, `codegen-units = 1`, no `target-cpu` flag.

### H1, H2: 344 divisions, macOS arm64

For `pdsi()` and `scpdsi()` alike, for each of PDSI, PHDI, PMDI, the Z-Index and the fitted-parameter dictionary: maximum absolute difference **0.0**, maximum relative difference **0.0**, **0** months with a differing sign and **0** NaN-position mismatches. All 344 divisions give a byte-identical float64 array on the two paths, for every output. H1 and H2 hold, at the strongest level (bit equality, not the `1e-10` tolerance).

### H3: engineered boundary series, macOS arm64

| Corpus | Series run on both paths | Series where one ulp changes the answer | Disagreements |
| --- | --- | --- | --- |
| Random Z with 60% boundary entries, one entry moved one ulp either way, `pdi.f` recursion | 1,500 triples | 28 (PDSI), 14 (PHDI), 6 (PMDI) | 0 |
| Same, Wells recursion | 600 triples | 4 | 0 |
| Established spell, then a threshold value, one ulp either side, `pdi.f` | 384 triples | 26 | 0 |
| Same, Wells | 384 triples | 30 | 0 |
| The twelve minimal divergence sequences of REFIMPL-29 | 12, both recursions | n/a | 0 |
| Division 4002 with 8% of months and one whole year set to zero precipitation | `pdsi()`, `scpdsi()` | n/a | 0 |

The boundary values are 0, -0.0, +-0.15, +-0.5, +-1 and one ulp either side, plus the fitted-factor thresholds `0.5 * wetm`, `0.5 * drym`, `0.5 * (m + b)` and their multiples.

No branch flip is known. H3 holds on this platform.

**Which boundaries are live.** Injected Rust mutants, each built, run against the new tests, and reverted:

| Mutant | Caught |
| --- | --- |
| `palmer_pdi.rs` statement 170: `pv >= 0.0` to `pv > 0.0` | yes, by the random straddling corpus |
| `palmer_pdi.rs` statement 180: `pv <= 0.0` to `pv < 0.0` | yes, by the random straddling corpus |
| `palmer_pdi.rs` `advance_month`: `z >= 0.15` to `z > 0.15`, and to `z >= 10.0` | no |
| `palmer_wells.rs` `_continue_spell`: `direction * z >= abatement_z` to `>` | no |

The three uncaught mutants are equivalent mutants, not test gaps. The entry test (`z >= 0.15`; `direction * z >= 0.5 * slope`) is redundant with the sign test that the abatement statement then applies: in an established spell `v` is zero, so `pv = (z - 0.15) + min(v, 0)` is non-negative exactly when `z >= 0.15`, and the Wells abatement path computes `v = 0` at the threshold and continues the spell the same way. The same edit in the Python source (`z >= 10.0` in `_advance_month`; `>` in `_continue_spell`) leaves the output unchanged on 300 random series and on the 1,152-series spell corpus. So the branch that decides the spell state at exactly `z == 0.15` is the `pv` sign test, and the Z-Index sentinels that matter are the exact-zero state variables already tabulated in `docs/algorithm_refs/palmer.md`.

### H4, H5: cross-platform

| Platform | Rust vs Python max abs difference | Rust digests equal to macOS arm64 | Python digests equal to macOS arm64 |
| --- | --- | --- | --- |
| macOS arm64 (`aarch64-apple-darwin`) | 0.0 | reference | reference |
| Linux x86-64 | not run | not run | not run |
| Linux aarch64 | not run | not run | not run |
| macOS x86-64 | not run | not run | not run |
| Windows x86-64 | not run | not run | not run |

H4 and H5 are open. `.github/workflows/palmer-reproducibility.yml` runs all five targets and compares the digests; it has not been dispatched. What can be said without it: the Palmer kernels contain no `mul_add`, the build sets no `target-cpu` and no fast-math flag, and the scPDSI rescale and percentile run in NumPy on both paths, so the only floating-point work specific to the Rust path is the water balance, the CAFEC coefficients, the Z-Index, and the two recursions. A difference on another platform would most plausibly come from the math library (`log10` in the K' factor) or from NumPy's own vectorized kernels, which is what the Python control separates.

### Sentence for the manuscript

On macOS arm64, the Rust and Python Palmer implementations return byte-identical PDSI, PHDI, PMDI, Z-Index and fitted parameters for `pdsi()` and `scpdsi()` on all 344 climate divisions, and agree on 1,500 + 600 + 384 + 384 engineered boundary series; no branch flip was found. The other four release targets are not yet measured.
