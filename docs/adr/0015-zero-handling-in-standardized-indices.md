# Zero handling in standardized indices

## Status

Amended: the NumPy implementation landed in
[#1186](https://github.com/monocongo/climate_indices/issues/1186)
([#1204](https://github.com/monocongo/climate_indices/pull/1204)).
`zero_handling` is accepted by `spi()`, `standardized_index()`, the package-root
`spi()`, and the `compute` gamma and Pearson transforms, and gamma `p0` is
counted over the calibration period. Decision 4 now records that
implementation, including the undefined zero mass of a step without calibration
data, and the Consequences name when classic output stays unchanged. The xarray adapter records the mode and correct output bounds in CF metadata;
the CLI accepts `--zero_handling` and writes the mode and method citation in
[#1187](https://github.com/monocongo/climate_indices/issues/1187). The
closed-form and property tests and the guidance landed in
[#1188](https://github.com/monocongo/climate_indices/issues/1188); the
cross-implementation fixtures against the R `SEI`/`SCI` packages remain
outstanding in
[#1209](https://github.com/monocongo/climate_indices/issues/1209). This record
is amended as each one lands.

SPI and `indices.standardized_index()` treat zero accumulations as a point mass
of probability `p0` below the fitted gamma or Pearson Type III distribution:
the transform computes `p0 + (1 − p0)·F(x)` and then `Φ⁻¹` of that. A zero
normally scores `Φ⁻¹(p0)`, the **top** of the zero mass (Pearson's existing
support-limit mask can override this score). Where `p0 ≥ 0.5` (arid
cells, dry seasons, daily and short-timescale SPI) a completely dry period
scores zero or higher. The series can never reach a drought threshold, and the
index mean is biased high. Stagge et al. (2015) documented this and proposed
the centre of the zero mass; Allen and Otero (2024, appendix) derive the value
that makes the normal-scale mean exactly zero. This record settles the option
the zero-inflated standardization epic
([#1184](https://github.com/monocongo/climate_indices/issues/1184)) adds, as
asked by [#1185](https://github.com/monocongo/climate_indices/issues/1185).

## Decision

1. **Name and values.** The option is the keyword-only `zero_handling`, spelt
   the same way in `spi()`, `standardized_index()`, the xarray adapter, and the
   CLI (`--zero_handling`, matching the CLI's existing underscore flags). It
   takes one of three strings:

   | Value | A zero scores | Property |
   |---|---|---|
   | `"classic"` (default) | `Φ⁻¹(p0)` | Today's behaviour; NOAA/NCEI and SPEIbase convention |
   | `"center_of_mass"` | `Φ⁻¹(p0 / 2)` | Stagge et al. (2015); ideal probability-scale mean is 1/2 |
   | `"mean_zero"` | `−φ(Φ⁻¹(p0)) / p0` | Allen and Otero (2024); `E[Z ∣ Z < Φ⁻¹(p0)]`, giving an ideal normal-scale mean of 0 |

   The mean properties assume a correctly fitted *nonzero-component* CDF and
   matching zero mass, before clipping. Pearson currently fits its CDF to all
   calibration values, including zeros, and counts exact zeros while assigning
   traces below 0.0005 the zero score. Neither mean is guaranteed for that
   path; even an ideal fit loses exactness after clipping. Any other value
   raises `ValueError` naming the three accepted values. At `p0 = 0.5` the
   three modes give 0.00, −0.67, and −0.80.

2. **What counts as a zero.** A non-classic mode moves the transform's
   zero/trace-mask positions, and nothing else. For gamma these are exact zeros
   after negatives are clipped. For Pearson Type III they
   are values below the existing 0.0005 trace threshold where `p0 > 0`, even
   though `p0` counts only exact zeros. The classic support-limit masks retain
   their existing precedence; a non-classic mode overrides those masks at the
   trace/zero positions so they receive the chosen score. All other values
   retain their classic transform.

3. **Coverage.** Gamma and Pearson Type III SPI across its NumPy, spatial-block,
   xarray, and CLI surfaces, and the NumPy-only `standardized_index()` (see
   [ADR-0001](./0001-dual-numpy-xarray-api.md)). When a Pearson fit falls back
   to gamma, the gamma transform applies the same mode. SPEI is excluded: its
   P − PET series is offset before fitting and has no physical zero mass.
   EDDI is excluded: it is non-parametric and has no `p0`. `spei()` and
   `eddi()` do not take the parameter; an unknown keyword retains the existing
   `TypeError` contract. The CLI rejects a non-classic mode with SPEI via
   `ValueError`, rather than ignoring it; EDDI has no CLI route.

4. **`p0` comes from the calibration period, for both distributions.** Pearson
   already computed `p0` over the non-missing calibration values and returned
   its array from `pearson_parameters()`; callers pass it in `fitting_params` as
   `prob_zero`. Gamma did neither. It counted zeros over all years, including
   missing ones in the denominator, in `compute.transform_fitted_gamma()` and
   kept only `alpha` and `beta`, so its `p0` disagreed with the window its shape
   and scale were fitted on. The modes make `p0` decide where every zero lands,
   so this epic makes gamma match Pearson: divide calibration-window zero counts
   by non-missing calibration counts. A step with no non-missing calibration
   value has no defined zero mass: its `p0` is NaN and its zeros transform to
   NaN, as its non-zero values do without supplied parameters, rather than
   scoring as an extreme drought (amended in #1186, which first gave such a
   step no mass). `p0` is computed over the calibration period and read from
   `fitting_params["prob_zero"]` when supplied; a supplied value must lie in
   `[0, 1]` (NaN marks an undefined mass) and match the values' cell
   dimensions, as Pearson parameters must; a supplied `alpha` or `beta` must
   broadcast to the values, which keeps shapes such as `(periods, 1, 1)` valid. Pearson keeps its existing handling of a step without
   calibration data rather than this rule, so its classic output does not move
   relative to the previous release: the minimum-non-zero guard gives the step
   `p0 = 0` and zeroed parameters, so its non-zero values are NaN while its zeros
   still take the trace floor's score, clipped to −3.09. The two distributions
   therefore differ at such a step's zeros, which is accepted here. `gamma_parameters()` still returns
   the existing `(alpha, beta)` tuple, while `fit_diagnostics()` returns gamma
   `prob_zero` with `alpha` and `beta`, so its parameters reproduce a fit.
   Callers saving a reusable gamma `fitting_params` dictionary must include
   calibration-window `prob_zero` to preserve the zero mass across datasets.
   Dictionaries without `prob_zero` remain valid: the transform computes it
   from the calibration window of the values being transformed.

5. **Edge cases.** A mode acts on the *effective* `p0`, the value left after the
   invalid-fit handling below. With `p0 == 0` there are no calibration zeros, but the
   transformed record may contain zeros outside that period. Every mode gives
   those zeros the same `"classic"` result. Pearson's minimum-non-zero guard
   also sets `p0` to 0 where a step has fewer than 4 non-zero calibration
   values, so in exactly the arid steps this record targets its zeros score
   −3.09 in every mode. With `p0 == 1` (every calibration
   value zero), no continuous distribution can be fitted. Keep the existing
   invalid-fit handling rather than inventing a zero score: gamma resets `p0`
   (a supplied one only when it is exactly 1) to 0, so zeros transform to `−∞`
   and are clipped to −3.09; nonzero values
   outside the calibration window may remain NaN. Pearson's minimum-non-zero
   guard zeroes that step's parameters; its trace floor can give zeros a finite
   score, and nonzero values outside the window may trigger the gamma fallback.
   In particular, the result need not equal the old whole-record calculation
   when only the calibration window is all zero.

6. **Clipping.** The existing `[−3.09, 3.09]` clip applies to every mode,
   including the zeros a mode moves. The CLI bounds remain valid; the xarray
   SPI output must replace any inherited input `valid_min`/`valid_max` with
   these bounds, wired in #1187. Clipping either tail can
   move the mean away from zero, even when the zero score itself is inside
   the bounds. The `"mean_zero"` zero score passes −3.09 below
   `p0 ≈ 0.0026` (for example `p0 = 0.001` gives −3.37); the
   `"center_of_mass"` zero score passes it below `p0 ≈ 0.002` (`p0 = 0.001`
   gives −3.29). Neither mode exempts zeros from the clip.

7. **Probability scale.** When the probability-scale (PIT) output
   ([#1192](https://github.com/monocongo/climate_indices/issues/1192)) is
   requested, a gamma zero (or an unmasked Pearson zero/trace value) maps to
   the effective `p0` under `"classic"` and to `p0 / 2` under both non-classic
   modes where `0 < p0 < 1`. Gamma resets an all-zero calibration step's
   empirical `p0 == 1` to 0 before transformation, so its zero PIT is 0.
   At `p0 == 0`, every mode keeps the classic probability;
   Pearson's existing trace floor or support-limit masks may then apply. A
   classic Pearson zero overridden by a support-limit mask instead exposes the
   actual masked probability, not `p0`. `p0 / 2` is the centre of the zero
   mass on the probability scale; under an ideal mixed fit the *whole
   probability-index series* has mean 1/2, not the zeros alone. It is not `Φ`
   of the `"mean_zero"` normal-scale value. That mode's property is defined on
   the normal scale only; #1192's inverse-normal parity test applies to
   classic/centre outputs only before clipping, not to `"mean_zero"` zeros.
   The PIT output documents these exceptions.

8. **Fitting parameters and round-trip.** `zero_handling` is a transform
   choice, not a fit result. It is not stored in `fitting_params`, and the
   fitted parameters (including `prob_zero`) are identical under every mode.
   A saved parameter set can be applied under any mode, and a caller who
   reuses parameters passes `zero_handling` again. There is no precedence rule
   because there is only one source.

9. **CF metadata.** Every SPI output from the xarray adapter and the CLI
   carries a `zero_handling` attribute naming the mode, `"classic"` included,
   so an output file says how its zeros were placed.
   The non-classic modes append their reference (Stagge et al., 2015, or Allen
   and Otero, 2024) to the output's `references` attribute.

Rejected alternatives. `zero_placement` and `zero_method` as the parameter
name: `zero_handling` is the name the epic and its sub-issues already use.
Exempting moved zeros from the clip: that keeps the exact mean, but the CF
`valid_min` becomes false and the clip has to know which values were zeros.
Storing the mode in `fitting_params`: that couples a transform choice to
the fit, and a conflicting explicit argument then needs a precedence rule.
Silently ignoring the mode for SPEI and EDDI: a user who asked for zero
handling on SPEI would get classic output with nothing to say so.

## Consequences

- **Backward compatibility is narrower than "unchanged".** The `"classic"`
  default leaves Pearson-transformed SPI outputs unchanged, but Pearson SPI
  that falls back to gamma can also change. Classic gamma SPI and gamma
  `standardized_index()` outputs are unchanged only for a full-record
  calibration of complete years, at scale 1, with no missing value and no
  `prob_zero` in a gamma `fitting_params`. They change at a step with a zero
  when the calibration window is shorter than the record and the in-window and
  whole-record zero fractions differ, or when a calibration value is missing,
  which includes the NaN padding of a partial final year and the leading
  `scale − 1` values of a scaled series. A gamma `fitting_params` that already
  carried `prob_zero` (or `probabilities_of_zero`), which gamma ignored, now
  sets the zero mass and can move every value. A step with no calibration data
  now gives its zeros NaN. The shared gamma correction can also change gamma
  SPEI through any of these, although SPEI has no `zero_handling` option; the
  offset P − PET series needs an exact zero for the denominator to matter.
  That is decision 4, and it is a correction. The committed NOAA and SPEIbase
  comparisons pass unchanged, and the CHANGELOG entry names the change.
- `gamma_parameters()` keeps its two-array return value. Callers that want a
  fixed zero mass across datasets must add calibration-window `prob_zero` to
  their own gamma `fitting_params` dictionary, or take the one
  `fit_diagnostics()` returns.
- The NumPy transforms took `zero_handling` alongside `p0` in
  [#1186](https://github.com/monocongo/climate_indices/issues/1186), whose
  blanket "classic output is unchanged" acceptance was narrowed to the
  conditions above and covers the changed gamma `p0` denominator, including
  SPEI and Pearson-to-gamma fallback. Surface wiring, unsupported-mode
  rejection, and the CF attribute/bounds followed in
  [#1187](https://github.com/monocongo/climate_indices/issues/1187), including
  CLI rejection for non-classic SPEI and output metadata for every SPI mode.
  The closed-form and property tests, the docs (including the gamma
  parameter-reuse guidance in `choosing-parameters.md`), and the
  `VALIDATION.md` entry landed in
  [#1188](https://github.com/monocongo/climate_indices/issues/1188).
  Cross-implementation fixtures against the SEI R package remain outstanding
  in
  [#1209](https://github.com/monocongo/climate_indices/issues/1209).
- The Zero Handling term joined `src/climate_indices/CONTEXT.md` with the
  NumPy implementation in #1186.

## References

- Stagge, J. H., Tallaksen, L. M., Gudmundsson, L., Van Loon, A. F., &
  Stahl, K. (2015). Candidate distributions for climatological drought indices
  (SPI and SPEI). *International Journal of Climatology*, 35(13), 4027–4040.
  <https://doi.org/10.1002/joc.4267>
- Allen, S., & Otero, N. (2024). Calculating standardised indices using SEI.
  *The R Journal*, 16(4), 102–122, Appendix.
  <https://doi.org/10.32614/RJ-2024-038>
