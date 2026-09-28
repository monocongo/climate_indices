# Zero handling in standardized indices

## Status

Accepted. Implementation is tracked by
[#1186](https://github.com/monocongo/climate_indices/issues/1186) through
[#1188](https://github.com/monocongo/climate_indices/issues/1188). Until they
land, `zero_handling` does not exist in the code, and gamma `p0` is still
counted over the whole record. This record is amended as each one lands.

SPI and `indices.standardized_index()` treat zero accumulations as a point mass
of probability `p0` below the fitted gamma or Pearson Type III distribution:
the transform computes `p0 + (1 − p0)·F(x)` and then `Φ⁻¹` of that. A zero
therefore scores `Φ⁻¹(p0)`, the **top** of the zero mass. Where `p0 ≥ 0.5` (arid
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
   | `"center_of_mass"` | `Φ⁻¹(p0 / 2)` | Stagge et al. (2015); the probability-scale mean is 1/2 |
   | `"mean_zero"` | `−φ(Φ⁻¹(p0)) / p0` | Allen and Otero (2024); `E[Z ∣ Z < Φ⁻¹(p0)]`, so the normal-scale mean is 0 |

   Any other value raises `ValueError` that names the three accepted values.
   At `p0 = 0.5` the three modes give 0.00, −0.67, and −0.80.

2. **What counts as a zero.** A mode moves exactly the values the classic
   transform places at the top of the zero mass, and nothing else. For gamma
   these are exact zeros after negatives are clipped. For Pearson Type III they
   are values below the existing 0.0005 trace threshold where `p0 > 0`. Nonzero
   values keep `p0 + (1 − p0)·F(x)` in every mode, so the modes differ only on
   zeros.

3. **Coverage.** Gamma and Pearson Type III SPI, `standardized_index()`, and
   their spatial-block, xarray, and CLI surfaces. When a Pearson fit falls back
   to gamma, the gamma transform applies the same mode. SPEI is excluded: its
   P − PET series is offset before fitting and has no physical zero mass.
   EDDI is excluded: it is non-parametric and has no `p0`. `spei()` and
   `eddi()` do not take the parameter. The xarray adapter and the CLI raise
   `ValueError` when a non-classic mode is combined with SPEI or EDDI rather
   than ignoring it.

4. **`p0` comes from the calibration period, for both distributions.** Pearson
   already computes `p0` over the calibration window and stores it in
   `fitting_params` as `prob_zero`. Gamma does neither. It counts zeros over the
   whole record in `compute.transform_fitted_gamma()` and keeps only `alpha` and
   `beta`, so its `p0` disagrees with the window its shape and scale were fitted
   on. The modes make `p0` decide where every zero lands, so this epic makes gamma
   match Pearson. `p0` is computed over the calibration period, returned in the
   gamma `fitting_params` as `prob_zero`, and read back from it when supplied.
   Gamma `fitting_params` without `prob_zero` (every dictionary saved before
   this change) stay valid, and `p0` is then computed from the calibration
   window of the values being transformed.

5. **Edge cases.** With `p0 == 0` there are no zeros, so every mode equals
   `"classic"`. With `p0 == 1` (every calibration value zero) every mode keeps
   today's classic behaviour. Gamma resets `p0` to 0, so the zeros transform to
   `−∞` and are clipped to −3.09. Pearson's minimum-non-zero guard zeroes that
   step's parameters, which triggers the existing fallback. No mode defines its
   own all-zero semantics, because there is no fitted distribution to place the
   zeros against.

6. **Clipping.** The existing `[−3.09, 3.09]` clip applies to every mode,
   including the zeros a mode moves. `valid_min`/`valid_max` in the CF metadata
   stay true for all outputs. The cost is that `"mean_zero"` loses its exact
   mean-zero property once its zero value passes −3.09. That happens below
   `p0 ≈ 0.0026` (for example `p0 = 0.001` gives −3.37). With
   `"center_of_mass"` it happens below `p0 ≈ 0.002` (`p0 = 0.001` gives
   −3.29). Either case means at most one zero in roughly 400 calibration years
   at that time step, so the bias is negligible in practice. The docs say so
   rather than exempting zeros from the clip.

7. **Probability scale.** When the probability-scale (PIT) output
   ([#1192](https://github.com/monocongo/climate_indices/issues/1192)) is
   requested, a zero maps to `p0` under `"classic"` and to `p0 / 2` under both
   non-classic modes. `p0 / 2` is the centre of the zero mass on the probability
   scale, and it gives zeros a PIT mean of 1/2. It is not `Φ` of the
   `"mean_zero"` normal-scale value. That mode's property is defined on the
   normal scale only, and the PIT output documents this.

8. **Fitting parameters and round-trip.** `zero_handling` is a transform
   choice, not a fit result. It is not stored in `fitting_params`, and the
   fitted parameters (including `prob_zero`) are identical under every mode.
   A saved parameter set can be applied under any mode, and a caller who
   reuses parameters passes `zero_handling` again. There is no precedence rule
   because there is only one source.

9. **CF metadata.** Every SPI and standardized-index output from the xarray
   adapter and the CLI carries a `zero_handling` attribute naming the mode,
   `"classic"` included, so an output file says how its zeros were placed.
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
  default leaves Pearson SPI outputs unchanged. Gamma SPI and gamma
  `standardized_index()` outputs change wherever the calibration window is
  shorter than the record and the in-window and whole-record zero fractions
  differ. That is decision 4, and it is a correction. When the calibration
  period covers the whole record, which is the configuration of the committed
  NOAA and SPEIbase comparisons, the outputs are unchanged. The fixtures and
  their tolerances are re-checked, and the CHANGELOG entry names the change.
- Gamma `fitting_params` returned by the library gain a `prob_zero` key. Code
  that asserts the exact key set of a gamma dictionary sees a new key.
- The NumPy transforms take `zero_handling` alongside `p0`
  ([#1186](https://github.com/monocongo/climate_indices/issues/1186)). Surface
  wiring, the SPEI/EDDI rejection, and the CF attribute follow in
  [#1187](https://github.com/monocongo/climate_indices/issues/1187).
  Cross-implementation fixtures against the SEI R package, docs, and the
  `VALIDATION.md` entry follow in
  [#1188](https://github.com/monocongo/climate_indices/issues/1188).
- The Zero Handling term joins `src/climate_indices/CONTEXT.md` with the
  NumPy implementation, once the parameter exists.

## References

- Stagge, J. H., Tallaksen, L. M., Gudmundsson, L., Van Loon, A. F., &
  Stahl, K. (2015). Candidate distributions for climatological drought indices
  (SPI and SPEI). *International Journal of Climatology*, 35(13), 4027–4040.
  <https://doi.org/10.1002/joc.4267>
- Allen, S., & Otero, N. (2024). Calculating standardised indices using SEI.
  *The R Journal*, 16(4), 102–122, Appendix.
  <https://doi.org/10.32614/RJ-2024-038>
