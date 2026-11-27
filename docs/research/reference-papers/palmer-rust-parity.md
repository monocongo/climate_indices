# Palmer Rust kernels: parity, branch stability, and cross-platform reproducibility

**Ticket:** REFIMPL-34, [#1636](https://github.com/monocongo/climate_indices/issues/1636). **Map:** [#1602](https://github.com/monocongo/climate_indices/issues/1602).
**Pinned version:** `380aa01f` (`origin/main`, 2026-11-21) for H1–H3; `b3854aaf` (`origin/main`, 2026-11-26; run 2026-11-27) for the five-platform run.
**Status:** H1–H3 were measured on macOS arm64. On all five release targets, the [five-platform run](https://github.com/monocongo/climate_indices/actions/runs/46069019548) measured H1, H2, H4 and H5, and the H3 boundary tests passed on output agreement only. H4 and H5 are refuted. The split between glibc and Apple's libm is traced to a one-ulp difference in `log10`; the smaller differences on the other targets are attributed to `log10` by inference (see "H4, H5"). The T4 grid is still deferred to [#1612](https://github.com/monocongo/climate_indices/issues/1612). Commit `eb710e28` records the original plan. See D1–D8 for differences between that plan and this record. Reproduce with `uv run python scripts/measure_palmer_parity.py measure --out parity.json` (about 25 seconds on four workers) and `CLIMATE_INDICES_REQUIRE_NATIVE=1 uv run pytest tests/test_native_parity_palmer.py`. `uv run maturin develop --release` builds the extension that the tests need, and the tests run whenever it is present. `CLIMATE_INDICES_REQUIRE_NATIVE=1` makes a missing extension fail instead of skip the module.

## Question

Can the manuscript state that the Rust and Python Palmer paths give the same results, on which platforms, and to what tolerance? The Palmer recursions branch on exact comparisons with zero and on fixed thresholds, so a one-ulp difference upstream of a sentinel could flip a branch and produce a large difference. The registry tolerance of `1e-10` does not rule that out on inputs outside the registry.

## Claims being tested

None of these is a finding yet. Each is a lead to confirm or refute at the pinned version.

| Lead | Source |
| --- | --- |
| Rust and Python `pdsi()` and `scpdsi()` agree within `rtol = atol = 1e-10` on all 344 divisions, with matching signs | `tests/test_native_parity_palmer.py`, `test_every_division_matches_for_pdsi` and `..._for_scpdsi` (asserts a tolerance, records no maxima) |
| The Palmer kernels contain no fused multiply-add | `crates/climate-indices/src/palmer*.rs` has no `mul_add`; `docs/architecture.md` limits FMA to `special::mul_add` on aarch64 |
| The release build keeps IEEE semantics and sets no `target-cpu` | `Cargo.toml` `[profile.release]`; no `.cargo/config` in the tree |
| The scPDSI rescale, the percentile and the CAFEC coefficients run in NumPy on both paths; the Rust kernels are the water balance, K-prime and dbar, the raw Z-Index, and `palmer_pdi` or `palmer_wells` with `scpdsi_duration_factors` | `src/climate_indices/drought/palmer.py`, `_rescale_scpdsi_zindex`, `_calc_cafec_coefficients`; `_PDSI_KERNELS` and `_SCPDSI_KERNELS` in `tests/test_native_parity_palmer.py` |
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

H2 and H3 state the original hypotheses. The tests observe the returned values, not the internal spell state or branch decisions. Output agreement alone cannot confirm the internal claims.

H5 is a control: if H4 holds and H5 does not, the Python outputs differ end to end across platforms. This does not identify which Python-path operation caused the difference. It also does not show that the Rust intermediate values are platform-invariant. Attribution needs further isolation; "H4, H5" does it by substituting the `log10` values.

## Method

1. **Full-fixture parity (H1, H2).** `scripts/measure_palmer_parity.py` runs both paths on each division and records, per output, the maximum absolute and relative difference, the count of months whose sign differs, and the count of NaN-position mismatches. It also records the toolchain: `rustc -vV`, `rustc --print cfg` (the host target features, not the flags the extension was built with), and the NumPy and platform versions. The release profile is read from `Cargo.toml`, not measured.
2. **Engineered boundary series (H3).** A deterministic generator in the test module builds Z series, run through `palmer._pdi_recursion` and `palmer._wells_recursion` on both paths. Each series places a value on a boundary and then on the neighbours at `np.nextafter` distance. The boundaries come from the branch inventory in "Branch inventory" below. The committed divergence sequences in `palmer-recursion-divergence-sequences.json` are also run through both Rust and Python.
3. **Gaps and zeros.** NaN gaps and zero precipitation are added to the division inputs, because the parity registry's input builders clip precipitation to [5, 300] and fill NaN, so they never produce either.
4. **Cross-platform (H4, H5).** `.github/workflows/palmer-reproducibility.yml`, `workflow_dispatch` only, runs the measurement (digests are always recorded, see D1) on each target and compares the SHA-256 of the raw float64 bytes per division and output across platforms. A platform not yet run is reported as not run, not as passing.

## Branch inventory

From `_palmer_wells.py` and `_palmer_pdi.py` and their Rust mirrors `palmer_wells.rs` and `palmer_pdi.rs`. The engineered series target the spell thresholds and the exact zeros directly; the probability bounds and clamps are state the series reach only through those:

- Spell thresholds: `direction * z >= 0.5 * slope` (signed; `slope` is `wetm` or `drym`) in the Wells continuation, taken only when `probability` is 0 or 100; `x1 >= 0.5` and `x2 <= -0.5` in the Wells establishment; `|x3| <= 0.5`, `x3 > 0.5`, `x3 < -0.5` and `z >= 0.15`, `z <= -0.15` in the `pdi.f` month advance.
- Probability bounds: `prob <= 0` and `prob >= 100` in `_pmdi`, the `abatement_underway` test (`probability` not 0 or 100) in `_continue_spell`, and `prob >= 100 - 1e-5` (`_TOLERANCE`) in `_abatement_transition`.
- Exact zeros: `x1`, `x2`, `x3` equal to `0.0` in `_pmdi`, `_continue_spell`, `_establish_spell`, the backtrack, and the PHDI fallback; `px1 >= 1` and `px2 <= -1` with `px3 == 0` in `_statement_200`.
- Clamps: `max(0, ...)` and `min(0, ...)` in `_candidate_values`.

## Thresholds

- **Measured maximum (H1):** The measurement records the maximum absolute difference for each output and its sign and NaN-position mismatch counts; `tests/test_native_parity_palmer.py` enforces a combined `assert_allclose` tolerance (`0.0` on macOS arm64, `1e-10` elsewhere), not the H1 absolute bound alone. On non-macOS platforms, `python = 10` and `rust = 10 + 5e-10` pass the test tolerance of `1.1e-9`, but the absolute difference exceeds the H1 bound of `1e-10`. The recorded maximum absolute difference is the H1 evidence. A measurement-only run exits 0 even above `1e-10`.
- **Test tolerance:** The tests use `assert_allclose`, which checks `|rust - python| <= atol + rtol * |python|`, with matching signs and NaN positions. The full-division tests use a single platform-wide tolerance: `rtol = atol = 0.0` on macOS arm64 and `1e-10` elsewhere. Boundary tests use `rtol = atol = 1e-10` on all platforms. The measurement reports separate maxima per output; these are not separate test thresholds.
- **Cross-platform** (H4, H5): equality of the float64 bytes, no tolerance. A difference is a finding to attribute (fused multiply-add, the math library, NumPy), not a failure to hide.
- **Branch flip:** any engineered or fixture case where the paths take a different branch, recorded in the REFIMPL-24 ledger ([#1626](https://github.com/monocongo/climate_indices/issues/1626)) with its cause and owner.

## Out of scope

Performance ([#1637](https://github.com/monocongo/climate_indices/issues/1637)); the T4 grid ([#1612](https://github.com/monocongo/climate_indices/issues/1612)); changing any kernel, tolerance, or dispatch rule.

## Deviations

The study recorded these deviations after the first measurement run.

- **D1.** The plan said a `--digest` flag would add SHA-256 digests. The script records them always, for both the Rust and the Python path, because H5 needs the Python digests and a measurement without them cannot be compared later.
- **D2.** The plan said to pin the measured maxima. When the pin was set, only macOS arm64 had been measured, so only that platform pins `0.0` (`PINNED_RTOL`, `PINNED_ATOL` in `tests/test_native_parity_palmer.py`). Every other platform keeps the registry's `1e-10`; D8 says why the five-platform run left this unchanged. A `0.0` pin on an unmeasured platform would be a guess.
- **D3.** The first boundary tests put one boundary value in a series and compared the two paths. They passed, but a probe showed they were vacuous: a lone `z` on a threshold gives `z / 3` whichever branch is taken, and an injected Rust mutant (`pv > 0.0` in place of `pv >= 0.0`) was not caught. They were replaced by corpora of (one ulp below, on, one ulp above) triples with a floor on how many triples change the answer, and the mutants were then caught (see "Results").
- **D4.** The plan listed NaN gaps as a new case. `test_missing_months_match` already covers them, so only zero precipitation was added.
- **D5.** The first macOS arm64 measurement was invalid and was discarded. The script took the extension from `palmer._native`, which each call left switched off, so every division after a pool worker's first compared Python with Python and reported identical. Found in pre-PR review, fixed in `0508b366` (the extension now comes from the backend module and is restored in a `finally`), and the full measurement re-run; the numbers below are from the re-run. The per-division pytest tests were never affected.
- **D6.** An earlier draft called the Wells entry-threshold mutant equivalent, on the strength of identical output on 300 random series and on the spell corpus. Review produced a counterexample (see "Which boundaries are live"), a deterministic test now pins it, and the Rust mutant fails it.
- **D7.** After the first measurement, the Claims table clarified which stages run in Rust or NumPy. The Method clarified which toolchain details the script records and removed the planned `--digest` option (D1). The Branch inventory corrected the Wells signed continuation threshold and the conditions for its probability checks. The Thresholds section now distinguishes the measured maximum from the test tolerance (D2).
- **D8.** H4 was expected to hold. The five-platform run refuted it, and H5 with it. The workflow's compare step stays failing, because it is the hypothesis test, and the difference was traced instead. The trace added steps the plan did not have: two local reproductions checked against the CI digests, a `log10` substitution, and a correct-rounding check against `decimal`. Their scripts are not committed; "H4, H5" gives the procedure. D2 is unchanged: the run reports `0.0` on three more targets, but the pin stays on macOS arm64 only, because those targets run only in the dispatch workflow and the June-cut test is outside the measurement.

## Results

H1–H3: pinned version `380aa01f`, rustc 1.99.0 (LLVM 23.1.1), `aarch64-apple-darwin`, Python 3.14.7, NumPy 2.4.6, release profile `lto = "thin"`, `codegen-units = 1`, no `target-cpu` flag.

### H1, H2: 344 divisions, macOS arm64

For `pdsi()` and `scpdsi()`, all four index arrays had maximum absolute and relative differences of **0.0**, with no sign or NaN-position mismatches. The fitted parameters also had a maximum absolute difference of **0.0**. Each output had matching float64 digests on both paths in all 344 divisions. The review measurement reproduced these results and verified the expected Rust kernel calls for every division. H1 and the output criteria for H2 held. The measurement did not compare internal spell states.

### H3: engineered boundary series, macOS arm64

| Corpus | Cases run on both paths | Triples with a numerical response to a one-ulp input change | Disagreements |
| --- | --- | --- | --- |
| Random Z with 60% boundary entries, one entry moved one ulp either way, `pdi.f` recursion | 1,500 triples | 28 (PDSI), 14 (PHDI), 6 (PMDI) | 0 |
| Same, Wells recursion | 600 triples | 4 | 0 |
| Established spell, then a threshold value, one ulp either side, `pdi.f` | 384 triples | 26 | 0 |
| Same, Wells | 384 triples | 30 | 0 |
| The twelve minimal divergence sequences of REFIMPL-29 ([#1631](https://github.com/monocongo/climate_indices/issues/1631)) | 12, both recursions | n/a | 0 |
| Wells, a spell terminated and a new one established in the same month, then `z` one ulp below, on, and one ulp above `0.5 * wetm` (wet) or `0.5 * drym` (dry) | 2 triples | 2 | 0 |
| Division 4002 with 8% of months and one whole year set to zero precipitation | `pdsi()`, `scpdsi()` | n/a | 0 |

The random corpora use -0.0 (equal to 0), +-0.15, +-0.5, +-1 and one ulp either side. The spell corpora add thresholds computed from the default duration factors (`wetm = drym = 0.309`, `wetb = dryb = 2.691`): `0.5 * wetm`, `0.5 * (m + b)` and their multiples, so the wet and dry sides of those corpora are not independent. The two spell-termination triples are exact negatives of each other for the same reason. No Palmer-fitted factors with `wetm != drym` are exercised.

The outputs agreed within the test tolerance. In the Wells counts, all three runs of each counted triple returned results. These tests did not observe which branches ran, so H3 remains unverified.

After both flush fixes, revision `632282e7` produced 27 PDSI, 13 PHDI, and 0 PMDI responses on the random corpus. The established-spell corpus produced 14 PMDI responses. The table above retains the results from `380aa01f`.

**Which boundaries are live.** Injected Rust mutants, each built, run against the tests, and reverted:

| Mutant | Caught |
| --- | --- |
| `palmer_pdi.rs` `statement_170`: `pv >= 0.0` to `pv > 0.0` | yes, by the random straddling corpus |
| `palmer_pdi.rs` `statement_180`: `pv <= 0.0` to `pv < 0.0` | yes, by the random straddling corpus |
| `palmer_pdi.rs` `advance_month`: `z >= 0.15` to `z > 0.15`, and to `z >= 10.0` | not by the corpora |
| `palmer_wells.rs` `continue_spell`: `direction * z >= abatement_z` to `>` | not by the corpora; yes by the termination test added after review (D6) |

The corpora did not detect the two `pdi.f` mutants. The `z >= 10.0` change in Python also left the output unchanged on 300 random series. This does not prove equivalence. A month can end one spell and establish the opposite spell, leaving `pro == 100` and nonzero `x3` and `v`. Statement 170 uses `min(v, 0)` for wet-spell carry, and statement 180 uses `max(v, 0)` for dry-spell carry. The earlier explanation incorrectly assumed that `v` or `x3` must be zero after a spell ends.

The Wells entry test is **not** redundant. It is equivalent to the abatement path only when `probability == 0`, where `v == 0`. When a spell terminates and a new one is established in the same month, the state carries `probability == 100` and a nonzero `v`; a `z` exactly on `0.5 * slope` then restarts the spell (`v = 0`, `probability = 0`), while one ulp below continues the abatement. For `wetm = drym = 0.897`, `wetb = dryb = 0.333` and `z = [-1, 3, 0, t, t, t]` with `t = 0.4485`, the PDSI at index 4 is 0.51175 on the threshold and 0.36463 one ulp below. Random draws reach that state rarely, which is why the corpora missed it, and the previous version of this paragraph, which called the Wells mutant equivalent from a null result on the random corpus, was wrong.

### H4, H5: cross-platform

[Run 46069019548](https://github.com/monocongo/climate_indices/actions/runs/46069019548) at `b3854aaf`, 2026-11-27. On every target the measurement and the Palmer parity tests passed (789 tests each). The compare job failed on the Rust comparison. The Python control does not gate the job, and it also listed differing outputs. Counts are out of 3,440 outputs: 344 divisions, 2 indices, and 5 outputs (PDSI, PHDI, PMDI, Z-Index, fitted parameters).

| Platform | Runner | Rust vs Python max abs (max rel) | Rust outputs differing from macOS arm64 | Python outputs differing from macOS arm64 | Rust max abs difference from macOS arm64 |
| --- | --- | --- | --- | --- | --- |
| macOS arm64 (`aarch64-apple-darwin`) | `macos-latest`, macOS 26.7.2 | 0.0 | reference | reference | reference |
| Linux aarch64 (`aarch64-unknown-linux-gnu`) | `ubuntu-24.04-arm`, glibc 2.39 | 0.0 | 1,980 in 233 divisions | 1,980 in 233 divisions | 7.38e-14 |
| Linux x86-64 (`x86_64-unknown-linux-gnu`) | `ubuntu-latest`, glibc 2.43 | 7.38e-14 (8.1e-10) | 1,980 in 233 divisions, bytes equal to Linux aarch64 | 23 in 3 divisions (0805, 1109, 4505) | 7.38e-14 |
| macOS x86-64 (`x86_64-apple-darwin`) | `macos-15-intel`, macOS 15.7.9 | 0.0 | 9 in division 0505 | the same 9 | not measured |
| Windows x86-64 (`x86_64-pc-windows-msvc`) | `windows-latest`, Windows Server 2025 | 0.0 | 14 in divisions 1409 and 4505 | the same 14 | not measured |

No platform had a sign, NaN-position or infinity mismatch between paths, so H1 and the output criteria of H2 hold on all five. On Linux x86-64 the Rust-versus-Python maxima are 4.4e-15 (PDSI, PHDI, PMDI) and 5.3e-15 (Z-Index) for `pdsi()`, and 5.2e-14, 6.2e-14, 7.4e-14 and 3.9e-14 for `scpdsi()`, with 3.5e-14 in the scPDSI parameters. The 8.1e-10 relative maximum is on a near-zero PMDI value. The last column comes from the local reproductions described below. It is nearly the Linux x86-64 Rust-versus-Python comparison, because Linux x86-64 NumPy matches macOS arm64 in all but 23 outputs, and the per-output maxima of the two comparisons agree exactly.

All targets use rustc 1.99.0 (LLVM 23.1.1) with the release profile `lto = "thin"` and `codegen-units = 1`, no `target-cpu` and no fast-math flag; the Palmer kernels contain no `mul_add`. Default target features from `rustc --print cfg` (compile-time defaults, not the runner CPU): macOS arm64 28 features including `neon`, `fp16`, `dotprod`, `fcma` and `jsconv`; Linux aarch64 `neon`; Linux x86-64 `fxsr`, `sse`, `sse2`; macOS x86-64 up to `sse4.1`, `ssse3`, `cmpxchg16b`; Windows up to `sse3`, `cmpxchg16b`. Python 3.14.10 (3.14.7 on macOS x86-64) and NumPy 2.4.6 everywhere. NumPy picks its SIMD kernels at run time, and the report does not record which kernels ran.

**Cause.** The split between glibc and Apple's libm is traced to `log10` in the K′ factor, `1.5 * log10((trat + 2.8) / dbar) + 0.5`. In Rust this is `palmer_k_prime` (`palmer_zindex.rs`), and in NumPy it is `_calc_k_prime_and_dbar`. The remaining differences are attributed to it by inference (below). `log10` is the only transcendental function in either path. It is evaluated on the same 12 calendar-month arguments in each `pdsi()` and `scpdsi()` call. The rest is correctly rounded IEEE arithmetic (including `sqrt`), selection, and short reductions that gave identical bytes on both glibc architectures. The evidence:

1. Two local builds of `b3854aaf` returned float64 bytes equal to the CI reports on all 3,440 outputs, for both paths. One build ran natively on macOS arm64, the other on Linux aarch64 in a Debian trixie container (glibc 2.41, CPython 3.14.3). Their raw arrays therefore stand for the CI ones.
2. The K′ `log10` arguments are bit-identical on the two platforms. The results differ by exactly one ulp in 663 of the 4,128 values (344 divisions by 12 months; 16%), in 301 divisions. The 233 divisions with an output difference are exactly those whose K′ differs. In the other 68 of the 301, the one-ulp change rounds away in `1.5 * log10(...) + 0.5`, so K′ is bit-identical.
3. The macOS arm64 Python path was run with glibc's `log10` results substituted in all 688 K′ calls (344 divisions, 2 indices). It reproduced the Linux aarch64 digests on all 3,440 outputs. So, on the Python path, the one-ulp `log10` difference accounts for the whole glibc–Apple split. The same holds for the Rust path, because Rust and Python are byte-identical on both platforms.
4. Against an 80-digit `decimal` reference, Apple's arm64 `log10` is correctly rounded on 4,127 of the 4,128 arguments. glibc's is correctly rounded on 3,464: it is one ulp off on 664, including the one where both libraries return the same wrong value.

With glibc 2.39, 2.41 and 2.43 on two architectures, the Rust path gave identical Palmer outputs, which is why Linux x86-64 Rust equals Linux aarch64. The `log10` values themselves were captured only for 2.41. On Linux x86-64, the NumPy path differs from the Rust (glibc) path in 1,981 outputs but from macOS arm64 in only 23, so there NumPy's `log10` does not follow glibc. NumPy's Linux x86-64 builds can dispatch `log10` to their own SIMD implementation at run time, which would explain this. The report does not record the dispatch, so this is an inference.

The remaining differences among the platforms outside glibc are attributed by inference, because their arrays were not recovered: macOS x86-64 in 0505, Windows in 1409 and 4505, and Linux x86-64 NumPy in 0805, 1109 and 4505. Rank the 344 divisions by how close their nearest K′ `log10` value lies to a rounding midpoint. Divisions 1409 and 4505 rank first and second (1.97e-6 and 1.58e-5 ulp from the midpoint), and 1109, 0805 and 0505 rank 7th, 10th and 17th. Math libraries that are accurate but not correctly rounded disagree first on exactly such arguments, so the same mechanism is the likely cause. The magnitude of these differences is not measured.

Between macOS arm64 and the glibc outputs (both paths on Linux aarch64, and the Rust path on Linux x86-64), the largest difference is 7.38e-14 (scPDSI PMDI, division 2001), and 5.3e-15 for `pdsi()`. No output month changed sign or NaN position; as in H2, internal spell states were not compared. Neither path is at fault: the difference enters through `log10`, which both paths take from outside their own code.

### Sentence for the manuscript

On each of the five release targets (Linux x86-64 and aarch64, macOS arm64 and x86-64, Windows x86-64), Rust and Python `pdsi()` and `scpdsi()` agreed across 344 climate divisions. They were byte-identical on four targets and within 7.4e-14 (absolute) on Linux x86-64, and no month differed in sign or NaN position. Results are not bit-reproducible across platforms. The `log10` in the K′ factor comes from the platform math library, and glibc's differs from Apple's by one ulp in 16% of the K′ values. That difference alone changes 233 of the 344 divisions between macOS arm64 and the glibc builds (Linux aarch64, and the Rust path on Linux x86-64), by at most 7.4e-14, with no output changing sign. macOS x86-64 and Windows differ from macOS arm64 in one and two divisions. These differences are attributed to the same `log10` by inference, and their size was not measured. The boundary tests (`rtol = atol = 1e-10`) passed on all five targets across 2,868 engineered boundary triples, 12 divergence sequences, two mirrored Wells spell-termination triples and the zero-precipitation case. They did not compare internal branch decisions.
