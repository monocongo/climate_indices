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

The [five-platform run](https://github.com/monocongo/climate_indices/actions/runs/46069019548)
measured the 344-division fixture at `b3854aaf` on 2026-11-27. On the same platform,
Rust and Python outputs were byte-identical on four targets and within 7.4e-14 (absolute)
on Linux x86-64. No index month differed in sign or NaN position. In contrast, the
cross-platform Rust digest comparison failed because the bytes differed between targets.
This contract does not require bit-for-bit agreement across platforms.

In local arm64 reproductions, 16% of the `log10` results used to compute K′ differed
by one ulp between glibc and Apple's math library. Substituting the glibc values on the
macOS arm64 Python path reproduced the Linux aarch64 Python output digests. The outputs
of these two platforms differed in 233 divisions, by at most 7.4e-14. Rust uses the
platform math library for `log10`. Its output digests match Python's on both platforms,
which supports the same explanation for the Rust results.

On Linux x86-64, the two paths differ in 1,981 outputs. NumPy may use its own SIMD
implementation of `log10` there. The run did not identify which implementation ran.
The study record is `docs/research/reference-papers/palmer-rust-parity.md`.
