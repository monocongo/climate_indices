# Beta distribution for a bounded series

## Status

Accepted. The implementation lands in
[#1169](https://github.com/monocongo/climate_indices/issues/1169), the first
distribution option of the energy family's Phase 2
([#1157](https://github.com/monocongo/climate_indices/issues/1157)). The
evidence classification below is what the oracle survey
([#1160](https://github.com/monocongo/climate_indices/issues/1160)) left open
for this ticket; the empirical option that shares the seam is
[#1168](https://github.com/monocongo/climate_indices/issues/1168), and the
indices that consume beta are
[#1171](https://github.com/monocongo/climate_indices/issues/1171) and
[#1172](https://github.com/monocongo/climate_indices/issues/1172).

## Context

A capacity factor and a bounded [0, 1] demand proxy are not gamma- or
Pearson-distributed: their support is the unit interval, and a strictly
non-negative family with unbounded support is the wrong model for them even
though it can be fitted to the numbers. ADR-0019 decision 3 states the support
rule — a bounded [0, 1] series takes beta or the rank-based empirical option —
and the energy subsystem note pairs beta with `aggregation="mean"`, which is
what keeps a windowed value inside the support.

The oracle survey ([#1160](https://github.com/monocongo/climate_indices/issues/1160),
"Standardized energy indices: SEI oracle contract") found that the R `SEI`
package of Allen & Otero (2023) **rejects `dist="beta"`**: its bounded-data
censoring is neither a beta estimator nor a canonical zero- and one-inflated
beta policy. No independent implementation of a beta fit is available to this
project, so the survey's remaining decision 2 asks #1169 to select the
estimator, the endpoint convention, the minimum sample count, and the evidence
classification instead of borrowing gamma's tolerance.

## Decision

1. **The estimator is the method of moments over the interior values, per
   calendar step.** The interior is the calibration period's values strictly
   between 0 and 1; the shape parameters are `α = m·c` and `β = (1 − m)·c` with
   `c = m(1 − m)/v − 1`, where `m` and `v` are the interior mean and sample
   variance (`ddof=1`). A step needs at least two interior values; a step with
   fewer has no shape to fit, reports NaN parameters, and any interior value
   evaluated at that step is missing. The estimator is the classic MoM
   estimator, chosen over L-moments because the method of moments is the
   convention this family's literature uses for beta capacity-factor fits and
   because the L-moment machinery in `lmoments.py` is a GLO/GEV port, not a
   beta one. The fitted keys are `alpha`, `beta`, `prob_zero`, and `prob_one`.

2. **Both boundaries are point masses, and probability is
   `p0 + (1 − p0 − p1)·F(x)`.** `p0` and `p1` are the fraction of each step's
   non-missing calibration values equal to 0 and to 1, counted exactly as
   ADR-0015 counts a zero. `F` is the fitted beta CDF over the open interval.
   A boundary value never consults the shape parameters, so exact 0s and 1s
   standardize even when the interior fit fails. A step whose whole
   calibration mass sits at one boundary has no interior fit and scores that
   boundary's extreme end: `−inf` for a step that is always 0 and `+inf` for a
   step that is always 1. This is ADR-0015's convention for an all-zero step,
   which reads a completely dry record as the dry extreme rather than the wet
   one, applied to both boundaries.

3. **`zero_handling` applies to both masses, mirrored.** The three modes keep
   their ADR-0015 meanings at the lower boundary. At the upper boundary each
   placement is the mirror about 1/2: `classic` leaves a one at `1 − p1`,
   `center_of_mass` places it at `1 − p1/2`, and `mean_zero` at
   `+φ(Φ⁻¹(p1))/p1`, the normal-scale mean of the upper tail. A mass outside
   `(0, 1)` is not moved, exactly as ADR-0015 decision 5 requires.

4. **The support is enforced, not clipped.** A value outside [0, 1] raises
   `InvalidArgumentError`, and so do supplied masses whose sum exceeds 1, which
   would leave the interior a negative probability. A tolerance of `1e-12`
   absorbs the rounding of a windowed mean over bounded values, which can land
   a few ulps outside the range; the tolerance is not a licence to clip real
   data, because clipping would standardize a different series (ADR-0019
   decision 3, the same reason the parametric families reject a signed series).

5. **Coverage is `standardized_index()` only.** `spi()`, `spei()`, and
   `fit_diagnostics()` reject `Distribution.beta`; the CLI's SPI and SPEI loops
   name their distributions explicitly rather than iterating the enum, so no CLI
   output changes. The xarray and CLI distribution surfaces belong to the
   energy adapters ([#1174](https://github.com/monocongo/climate_indices/issues/1174))
   and CLI integration ([#1178](https://github.com/monocongo/climate_indices/issues/1178)).

6. **The evidence is specification and regression only.** With no independent
   beta implementation available, `tests/test_beta_distribution.py` is
   classified as unit and regression evidence: an exact PIT check against the
   closed form with supplied parameters, an estimator-recovery check on a
   synthetic sample of known shape, the boundary-mass and degenerate-step
   contracts, and the round-trip through `fitting_params`. Nothing here is an
   external validation, and `VALIDATION.md` gained no beta row for that reason.
   The classification is upgraded only if an independent beta estimator, or a
   published beta standardization, is matched against these conventions first.

## Consequences

- `standardized_index()` accepts one more distribution; SPI and SPEI output is
  untouched, because neither surface admits beta and the parity fixtures pin
  their existing results.
- `beta_parameters()` and `transform_fitted_beta()` join the public `compute`
  surface as the NumPy reference functions for this family, mirroring
  `gamma_parameters()` and `transform_fitted_gamma()`.
- `_place_zeros()` gained an `upper` keyword rather than a second near-duplicate
  helper, so the two boundary masses share one closed-form placement
  implementation; the lower-boundary call sites behave exactly as before.
- `fit_diagnostics()` reports no beta fit: its Kolmogorov–Smirnov surface is
  written for the gamma and Pearson CDFs, and a caller holding a beta
  `fitting_params` set can still reuse it through the round-trip test's route.
- The no-oracle limitation is a live reopen condition, not settled science. A
  future independent beta implementation, or a published MoM beta
  standardization over the same boundary convention, becomes the comparison
  that promotes this evidence class, and #1170 is where the SEI fixtures for
  the neighbouring empirical path land.

## References

- Allen, S. & Otero, N. (2023). Standardised indices to monitor energy droughts.
  *Renewable Energy* 217, 119206. <https://doi.org/10.1016/j.renene.2023.119206>
- [ADR-0015](0015-zero-handling-in-standardized-indices.md) — zero handling in
  standardized indices.
- [ADR-0019](0019-energy-module-api-and-aggregation.md) — the energy module's
  API, the aggregation keyword, and the signed and bounded series rules.
- `docs/research/energy-oracle-survey.md` — the SEI oracle contract and the beta
  evidence gap: <https://github.com/monocongo/climate_indices/blob/main/docs/research/energy-oracle-survey.md>.
