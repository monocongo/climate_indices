# Rust numerical contract

Each public Rust kernel documents its algorithm, units, shapes, calendar assumptions,
and missing-value behavior in its `Numerical contract` section.
Build the API reference with `cargo doc -p climate-indices --no-deps`.
The crate documentation includes the same generated accuracy page as this reference.

```{include} ../crates/climate-indices/NUMERICS.md
:heading-offset: 1
```

## Palmer branch stability

The Palmer recursions branch on exact comparisons with zero and on fixed thresholds
(`docs/algorithm_refs/palmer.md`, "Exact-zero comparisons"), so a one-ulp difference
upstream of a sentinel could, in principle, move a month to a different branch and change
a Palmer index by whole units, which the `1e-10` parity tolerance alone would not
prevent. Three pieces of evidence bound that risk. None of it is a bound for inputs
outside what is listed.

- **Full fixture.** `scripts/measure_palmer_parity.py` runs `pdsi()` and `scpdsi()` on the
  344 climate divisions with and without the Rust extension. On macOS arm64 every output
  (PDSI, PHDI, PMDI, Z-Index, fitted parameters) is byte-identical on both paths: maximum
  absolute difference 0.0, no month with a differing sign, no NaN-position mismatch.
  `tests/test_native_parity_palmer.py` pins that maximum for that platform and keeps the
  `1e-10` contract elsewhere.
- **Engineered boundaries.** The same test module runs series that carry 0, -0.0, +-0.15,
  +-0.5, +-1, thresholds computed from the default duration factors and one ulp either side, through the `pdi.f`
  and Wells recursions on both paths, each series repeated with one boundary entry moved
  one ulp down and up. Both paths agree on every series, and a floor on how many series
  change under a one-ulp move keeps the check from passing vacuously. Deliberately
  flipping a `pv` sign test in the Rust source makes these tests fail.
- **No fused multiply-add.** The Palmer kernels use no `mul_add`, the release profile
  keeps IEEE semantics, and the build sets no `target-cpu`. FMA appears in the crate only
  in `special::mul_add` on aarch64 (`crates/climate-indices/src/special/mod.rs`).

In the `pdi.f` recursion the entry test `z >= 0.15` is not equivalent to the `pv` sign test
that the abatement statement applies next: statement 170 clamps the spell carry with
`min(v, 0)`, so with a negative carry at exactly `z == 0.15` it continues the abatement where
the entry test would reset prob(end) to 0. The study record therefore claims no equivalence
for the two `pdi.f` threshold mutants, and its corpora did not catch them (see "Which
boundaries are live"). The Wells continuation test `direction * z >= 0.5 * slope` is not
redundant either: right after a spell ends, with `probability == 100`, a `z` exactly on the
threshold restarts the spell and one ulp below continues the abatement, and a test pins
that case.

Cross-platform results come from `.github/workflows/palmer-reproducibility.yml` (manual
dispatch; first run 2026-11-27 at `b3854aaf`). On all five release targets the Rust and
Python paths agree: byte-identical on four, and within 7.4e-14 on Linux x86-64, with no
sign or NaN-position mismatch. The outputs are not bit-identical across platforms, and
bit-for-bit cross-platform reproducibility is not part of this contract. The only
transcendental in the Palmer paths, `log10` in the K' factor, comes from the platform math
library on the Rust path; on every target except Linux x86-64 the NumPy-path outputs
match the Rust path byte for byte. glibc's
differs from Apple's by one ulp in 16% of the K' values. Between macOS arm64 and the glibc
builds (Linux aarch64, and the Rust path on Linux x86-64), that moves 233 of the 344
divisions by at most 7.4e-14. On Linux x86-64 the two paths differ in 1,981 outputs. NumPy's
`log10` probably does not return glibc's values there, perhaps because NumPy dispatches it
to its own SIMD implementation; the dispatch was not recorded.
The study record is `docs/research/reference-papers/palmer-rust-parity.md`.
