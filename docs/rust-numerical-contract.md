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

In the `pdi.f` recursion the entry test `z >= 0.15` duplicates the `pv` sign test that the
abatement statement applies next, so at exactly `z == 0.15` the `pv` sign test decides the
spell state. The Wells continuation test `direction * z >= 0.5 * slope` is not redundant:
right after a spell ends, with `probability == 100`, a `z` exactly on the threshold restarts
the spell and one ulp below continues the abatement, and a test pins that case.

Cross-platform results for Linux (x86-64 and
aarch64), macOS x86-64 and Windows come from `.github/workflows/palmer-reproducibility.yml`
(manual dispatch); until it has been run, only macOS arm64 is measured. The study record
is `docs/research/reference-papers/palmer-rust-parity.md`.
