# Log-logistic distribution for SPEI

## Status

Accepted. The implementation lands in
[#106](https://github.com/monocongo/climate_indices/issues/106) as the
implementation sub-issue of the log-logistic epic
([#1194](https://github.com/monocongo/climate_indices/issues/1194)).
Cross-implementation fixtures and the SPEIbase evidence upgrade follow in
[#1195](https://github.com/monocongo/climate_indices/issues/1195) and
[#1196](https://github.com/monocongo/climate_indices/issues/1196). This record
is amended as each one lands; #1195 and #1196 have landed, and the input-matched
log-logistic SPEIbase comparison in `tests/test_speibase_like_for_like.py`
upgraded the SPEI evidence in `VALIDATION.md`.

## Context

The three-parameter log-logistic distribution is the reference distribution for
SPEI (Vicente-Serrano et al., 2010; Beguería et al., 2014) and is what SPEIbase
uses. `climate_indices` supported only gamma and Pearson Type III, so the
SPEIbase comparison in `VALIDATION.md` could only be classified as plausibility,
not validation.

Three different families are all called "log-logistic" in this space: Hosking's
generalized logistic (GLO), which R's `SPEI` package fits under the name
`"log-Logistic"`; the 3-parameter Fisk distribution implemented by SciPy
`fisk`; and SciPy's `genlogistic`, a different family again. R `SPEI`
(`R/spei.R`) calls `lmom`'s `pelglo`/`parglo` and `lmom::cdfglo` with `ub-pwm`
L-moments by default, and SPEIbase v2.11 is built with that code.

SciPy `fisk` is the same family only on the GLO's lower-bounded shape branch
(`k < 0`, where Fisk's `c = 1/|k|`, `scale = α/|k|`, and `loc = ξ + α/k`); it
cannot represent the upper-bounded `k > 0` branch. Its estimator `fisk.fit` is
MLE only, so fitting with it would diverge from the SPEI literature and from
SPEIbase exactly in the tails a drought index cares about.

## Decision

1. **The distribution is Hosking's GLO.** `Distribution.loglogistic` is the
   generalized logistic that R `SPEI` fits as `"log-Logistic"`, not
   `scipy.stats.genlogistic` and not a Fisk fit. The public enum name is
   `loglogistic` because that is the name used by the SPEI literature and
   documentation; `glo` and `fisk` were the alternatives.

2. **Parameter estimation is L-moments (`ub-pwm`).** `lmoments.fit_glo()`
   estimates the first three sample L-moments with the existing SAMLMR
   translation and converts them to GLO parameters with a translation of
   `lmom`'s PELGLO subroutine. The fitted parameters are `loc`, `scale`, and
   `shape`, and they are the `fitting_params` keys for this distribution. The
   transform is a translation of `lmom::cdfglo`.

3. **Coverage is SPEI only.** `spei()` accepts `Distribution.loglogistic` on the
   NumPy, xarray/Dask, and CLI surfaces. `spi()`, `standardized_index()`, and
   `fit_diagnostics()` reject it with `InvalidArgumentError`: a precipitation or
   generic non-negative series has a physical zero mass that this distribution's
   zero-placement treatment does not yet provide. The CLI's `--index spi`
   therefore skips log-logistic while `--index spei` includes it.

4. **There is no zero mass and no `zero_handling` for GLO.** SPEI standardizes
   P − PET, which `spei()` offsets to stay positive; that series has no physical
   zero mass, so log-logistic (like R `SPEI`) standardizes every value through
   the fitted CDF and `zero_handling` does not apply. The `+1000` offset shifts
   only `loc`, leaving the scale, shape, and standardized values unchanged.

## Consequences

- Adding the enum member would otherwise make `--index spi` compute
  log-logistic SPI, so `_run_spi` iterates a filtered `_SPI_DISTRIBUTIONS`
  while `_run_spei` keeps iterating all members. `--index spei` and
  `--index all` therefore write one more output per scale
  (`spei_loglogistic_NN.nc`) than they did before; that is a user-visible
  CLI output change, not a silent one.
- `spi()` and `standardized_index()` carry an explicit rejection rather than
  silently standardizing zeros as if they were a fitted value.
- `VALIDATION.md` gained an input-matched log-logistic SPEIbase comparison in
  #1196 (`tests/test_speibase_like_for_like.py`), a cross-implementation check
  of this port against the grids SPEIbase v2.11 was computed from; the earlier
  gamma/Thornthwaite comparison remains classified as plausibility. The
  "log-logistic is not implemented" availability caveats were removed when the
  implementation landed.
- A later SPI/`standardized_index()` log-logistic surface, if wanted, has to add
  a zero-placement mode for the GLO first.

## References

- Hosking, J. R. M. & Wallis, J. R. (1997). *Regional Frequency Analysis: An
  Approach Based on L-Moments.* Cambridge University Press.
- Vicente-Serrano, S. M., Beguería, S. & López-Moreno, J. I. (2010). A
  Multiscalar Drought Index Sensitive to Global Warming: The Standardized
  Precipitation Evapotranspiration Index. *Journal of Climate* 23, 1696–1718.
  <https://doi.org/10.1175/2009JCLI2909.1>
- Beguería, S., Vicente-Serrano, S. M., Reig, F. & Latorre, B. (2014).
  Standardized precipitation evapotranspiration index (SPEI) revisited:
  parameter fitting, evapotranspiration models, tools, datasets and drought
  monitoring. *International Journal of Climatology* 34, 3001–3023.
  <https://doi.org/10.1002/joc.3887>
