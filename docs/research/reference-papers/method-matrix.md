# Method axes and per-package method matrix (REFIMPL-09)

Issue: [REFIMPL-09](https://github.com/monocongo/climate_indices/issues/1611), part of the [reference-implementation map](https://github.com/monocongo/climate_indices/issues/1602).
Date: 2026-11-21.

This record answers one question: along which axes can two implementations of a standardized index legitimately differ, and where does each package sit on each axis, by default and by option.
It runs no comparison and reports no numerical result.
Its files are:

- [`method_spec.schema.json`](method_spec.schema.json): the schema (JSON Schema 2020-12).
- [`method-matrix.json`](method-matrix.json): the matrix, one entry per package entry point, validated against the schema. Adapters join on its implementation ids, which are those of `lineage.json` ([REFIMPL-04](https://github.com/monocongo/climate_indices/issues/1606), pull request 1667, not yet merged).
- This file: the axes, the leads checked, the aligned configurations, and, between the markers, tables generated from the JSON by [`render_method_matrix.py`](render_method_matrix.py). Run `uv run python docs/research/reference-papers/render_method_matrix.py --check` to confirm the tables match the data.

Use [`VALIDATION.md`](../../../VALIDATION.md)'s evidence vocabulary: a method matrix is not validation, and a shared position on an axis upgrades no status.

## Axes

The ticket named eleven axes. A twelfth, the inverse-normal transform, is added because two comparators and the library's EDDI use an approximation whose size (A&S give 4.5e-4) is above the "equivalent" class (at most 1e-6) of [REFIMPL-02](https://github.com/monocongo/climate_indices/issues/1604)'s difference classes (see [`protocol.md`](protocol.md)), and because no other axis accounts for it.

| Axis | What it fixes | Positions seen |
| --- | --- | --- |
| accumulation | How the window is combined, and the first `scale - 1` steps | Sum or mean; calendar months or fixed days; NaN propagates, or a missing value counts as 0 |
| calendar grouping | The sample a distribution is fit to | Calendar month; ISO week; day of year (366-column calendar with an interpolated leap day, raw day of year, or a leap-day bin); caller-supplied group |
| calibration period | Which values define "normal" | Whole record; fixed window of years; fit and apply split; none |
| distribution | The family | Gamma, Pearson Type III, log-logistic (generalized logistic or scipy `fisk`), GEV, empirical |
| estimator | How parameters are found | Thom closed form; numerical maximum likelihood; unbiased-PWM or plotting-position L-moments; method of moments |
| location or offset | A fitted or imposed origin | None; fitted; additive offset to the water balance |
| zero handling | Mass at zero for precipitation | `p0 = n0/n`; `n0/(n+1)`; zero scored at the top of the mass, its centre, or its conditional mean; censoring; none |
| plotting position | Rank to probability (non-parametric) | `(1+k)/(n+2)`; Tukey with exact or rounded constants; Hazen |
| inverse normal | Probability to z | Exact quantile; Abramowitz and Stegun 26.2.23 (Hastings) approximation |
| output bound | Limits on the index | None; +-3.09; +-8.21; +-8.2 from a probability clip |
| missing data and minimum sample | What a gap or a short record does | NaN propagation; 3, 4, 10 or 30-year minimums; silent zero |
| fit failure | What an unfittable period returns | NaN with warning; silent NaN; fallback family; degenerate parameters; error |

Each cell in the JSON has a status:

- **verified**: the position was read in source at the pinned version, and each citation names the file and the lines. Citations were resolved by searching for the cited text, so a line number cannot be a typo.
- **unverified**: the position rests on documentation or an observed output, and `basis` says which. Nothing was read in source.
- **not_applicable**: the axis does not apply (for example, an estimator for a non-parametric method).

"Default" is the value when the argument is not given. Options are other positions the package can be set to, with the call that sets them.

## Coverage

<!-- BEGIN GENERATED: counts -->

| Family | Entries | Verified | Unverified | Not applicable |
| --- | --- | --- | --- | --- |
| SPI | 13 | 110 | 31 | 15 |
| SPEI | 10 | 77 | 30 | 13 |
| Non-parametric standardization (empirical SPI/SPEI, EDDI) | 7 | 36 | 29 | 19 |

<!-- END GENERATED: counts -->

Fifteen implementations appear. Ten were read in source: `climate_indices` at `98578c87`, `xclim` 1.0.0, `spei` 0.8.3, `standard_precip` 2.0.0 (with `lmoments3` 1.0.8), R `SPEI` 1.8.1, `SCI` 1.0-3, `SEI` 0.2.0, `climatekit` 0.2.3, the NDMC SPI Generator 1.7.6.0 (its source ships in the zip), and NCL 6.6.2 (source of `dim_spi_n` at the tag). Each download was checked against the sha256 in `comparators.lock.yaml`. The lock's NCL entry hashes the conda-forge binary, not the source tarball read here. `lmoments3` is not in that lock; its sdist sha256 is recorded in the matrix. Climate Engine, `GRIDMET/DROUGHT`, NCEI climdiv SPI, SPEIbase, and NOAA PSL EDDI have no source here (none was found for the first two), and every cell for them is unverified or not applicable.

## Leads checked

The "known so far" lines of the ticket, and the leads in `research-notes.md`, checked at the pins.

| Lead | At the pin | Result |
| --- | --- | --- |
| `climate_indices`: gamma by Thom closed form | `compute.gamma_parameters` | Confirmed, on the non-zero values |
| `climate_indices`: Pearson III by L-moments on non-zero values, gamma fallback | `compute.pearson_parameters`, `spi()` | Confirmed (zeros left out of the sample since #1562) |
| `climate_indices`: log-logistic by unbiased-PWM L-moments on `(P - PET) + 1000` | `compute.loglogistic_parameters`, `_spei_water_balance` | Confirmed. The 1000 is added to each step before accumulating, so the window holds `1000 x scale`; the fit is location-equivariant, so this does not change the result |
| `climate_indices`: GEV by Hosking L-moments on all values, no fallback | `compute.gev_parameters` | Confirmed |
| `climate_indices`: empirical by the SEI position; EDDI by Tukey with NOAA's rounded constants; clip +-3.09 | `compute._empirical_fit`, `eddi` | Confirmed. The empirical path needs 10 valid values and ranks ties at the maximum; EDDI ranks ties at the minimum and needs 2 |
| `xclim`: `gamma` or `fisk` | `standardized_precipitation_*` | Imprecise. The default is `gamma` for SPEI as well as SPI; `fisk` is an option. `genextreme` and `lognorm` are also accepted |
| `xclim`: `ML` default, `APP`, and `PWM` (gamma only) | `dist_methods`, `stats.fit` | Partly confirmed. `ML` is the default and `APP` needs `floc`. `PWM` is not allowed for a named distribution; it needs an `lmoments3` distribution object, and only the gamma accepts `floc` |
| `xclim`: zero inflation always on for SPI, off for SPEI | `_agro.py` | Confirmed |
| `xclim`: SPEI `offset` deprecated | SPEI signature | The parameter is gone in 1.0.0 |
| `xclim`: bound +-8.21 | `standardized_index` | Confirmed |
| `xclim`: `APP` with `floc=0` reproduces `climate_indices` | `_agro.py` docstring | The claim is in the docstring. It is consistent with the code: `APP` gamma is the Thom form on the values with `x - floc > 0`; the others are dropped from the fit but still scored |
| R `SPEI`: unbiased PWM default, plotting-position PWM, maximum likelihood, no clipping | `spei.R` | Confirmed. Maximum likelihood is an option for the log-logistic only |
| R `SCI`: Weibull `p0` with centre-of-mass placement | `sci.r` | Imprecise. `p0` is a required logical with no default; with `p0 = TRUE` and the default `p0.center.mass = FALSE` it is `n0/n`. `n0/(n+1)` and centre-of-mass scoring apply only with `p0.center.mass = TRUE` |
| R `SEI`: maximum likelihood, several families, empirical option | `std_index.R` | The default `dist` is `"empirical"`, with `method = "mle"` for the parametric families. Positive-support families stop on a zero (zeros are handled only by censoring with `lower = 0`), and `lmme` excludes the log-logistic |
| Climate Engine and `GRIDMET/DROUGHT`: non-parametric over fixed trailing days, plotting position unconfirmed | pilot document | Unchanged: no source, still unverified |

## Findings the leads did not cover

Each is a position a comparison could mistake for a difference in implementation quality.
Entries for the ones that disagree with a package's own documentation are logged in the [REFIMPL-24](https://github.com/monocongo/climate_indices/issues/1626) ledger.

1. **R `SPEI` 1.8.1 ignores the zero probability for the Gamma and PearsonIII distributions.** `spei.R` computes `pze` and removes non-positive values from the fit, then applies `qnorm(pze + (1 - pze) * pnorm(spei[ff, s]))` to `spei[ff, s]`, which is still `NA` (initialised as `acu * NA`), and overwrites it with `qnorm(cdf_res)`, the CDF of the positive-only fit, on the next statement. So `pze` never reaches the output: for Gamma a zero accumulation has a CDF of 0 and returns `-Inf`, for PearsonIII it gets the fitted CDF at 0, and non-zero values omit the zero mass. `NEWS.md` says zero handling was fixed in 1.6. This was read from source, not run; [REFIMPL-13](https://github.com/monocongo/climate_indices/issues/1615) should confirm it by running the adapter. SPEIbase's log-logistic path is not affected.
2. **The inverse-normal transform is not the same function everywhere.** NCL's gamma path (`anvnrmd`), the NDMC program (`inv_normal`), and the library's EDDI use the Abramowitz and Stegun 26.2.23 rational approximation; every other implementation read in source uses an exact quantile, including NCL's Pearson III path (AS241). NOAA PSL EDDI is reported, not read, to use the approximation, and Climate Engine, `GRIDMET/DROUGHT` and NCEI climdiv SPI do not document theirs. The bounds differ too: NCL (gamma) and NDMC do not clip (sentinels +-1e37 and +-9999), NCL's Pearson path clamps at +-3.09, and `climatekit` replaces only a CDF of exactly 0 or 1.
3. **`spei` (Vonk) keeps zeros in the maximum-likelihood fit.** It applies `p0` to the CDF but fits a three-parameter distribution, location free, to all values including zeros. `xclim` (SPI only; its SPEI keeps zeros), `standard_precip` (`gam` and `pe3`), `climate_indices` (gamma and Pearson), `climatekit` (gamma), and NCL remove zeros first.
4. **`xclim` averages, the others sum.** Its window is a mean (`skipna=False`), and so is R `SCI`'s. For a gamma or log-logistic fit this does not change the index; it does matter for a non-zero `floc` or offset, and when comparing fitted parameters directly.
5. **`standard_precip` day-of-year grouping is the raw day of year.** After 28 February a leap year's dates are one day later than a common year's. `xclim`'s daily grouping is also the raw day of year, with fitted parameters re-mapped to the target calendar afterwards. The library interpolates a 366th day, and `spei` (Vonk) gives 29 February its own bin.
6. **`standard_precip` documents `glo`, `gno` and `kap` as maximum-likelihood only, but registers L-moment fits for them.** Its SPEI docstring recommends `glo` with `lmom`. For these three the two fit types use different distribution families (`lmoments3` versus scipy's `genlogistic`, `gennorm`, `kappa4`).
7. **`climatekit` treats a missing day as zero** in the monthly totals, and its trailing window spans rows of the months present in `dates`, not calendar months, if the dates have gaps. Its gamma fit is the Thom form, which its docstring calls maximum likelihood "consistent with" R `SPEI` (R `SPEI` uses L-moments). The log-logistic data are shifted by `-min + 1`, a data-dependent offset.
8. **NCL thins the sample for Pearson III when `nrun` exceeds 12.** The fit and `pzero` use every `nstep`-th accumulation (`nstep` is `nrun` rounded up to a multiple of 12, at least 24, and is not set above 72), and every value is then scored.
9. **R `SEI` cannot fit a log-logistic to a signed water balance.** Its `llogis` is the two-parameter form with positive support, so a negative value is an error.
10. **The NDMC DLL project compiles two files.** `spimonthlyonly.cs` ships but is not in `spi_dll.csproj`; the accumulation and the month grouping are in the VB.NET front end.

## Aligned configurations

A configuration is listed when several implementations can be set to the same position on every axis they expose.
"Independent" means every pair is classed *independent* in `lineage.json`; that file is pull request 1667's and unmerged, so the triples are provisional until it merges. An implementation missing from a family's lineage table cannot form a triple in that family. Classes are per pair and per index, not per configuration: for gamma with the Thom form, the `climate_indices`/`ncl` and `xclim`/`ncl` links come from the Pearson and PWM fits, so triples with NCL are not excluded by any gamma-specific relation, and the listed triples are conservative. `lineage.json` pins `climate_indices` at `a7235bf7` and marks SPEIbase `source_read` true; this matrix reads the library at `98578c87` and read no SPEIbase source.

Summary against the ticket's done condition, "at least one fully aligned configuration across at least three independent implementations, or its absence stated":

- **SPI: partially aligned, so the condition is not met.** Gamma with the Thom closed form and a zero mass of `n0/n` is reachable in the library (calibration set to the record), `xclim` (`APP`, `floc=0`), `climatekit`, the NDMC program, and NCL. The independent triples are {`climate_indices`, `ndmc_spi`, `r_climatekit`} and {`ndmc_spi`, `r_climatekit`, `xclim`}, but neither triple is fully aligned: the residual is not zero (the NDMC and NCL inverse normal, two-decimal output, and the bounds). Numerical maximum-likelihood gamma, Pearson III by L-moments, and gamma by L-moments have no independent triple.
- **SPEI: absent.** The log-logistic by unbiased PWM is reachable in the library, R `SPEI`, `climatekit`, `standard_precip`, and (through an `lmoments3` object) `xclim`, but the library is classed *validated against* R `SPEI`, the other L-moment paths descend from Hosking's code, and `climatekit` has no SPEI class yet. The scipy `fisk` maximum-likelihood fit has two implementations that share scipy.
- **Non-parametric: absent.** Only the library and R `SEI` expose the `(1+k)/(n+2)` position, and the library validates against SEI.

<!-- BEGIN GENERATED: aligned -->

### `spi-gamma-thom`: Gamma, Thom closed form, zeros as a mass of n0/n

| Axis | Position |
| --- | --- |
| distribution | gamma |
| estimator | Thom (1958) closed form on the non-zero values |
| zero handling | p0 = zeros / n; a zero scores Phi^-1(p0) |
| calendar grouping | calendar month |
| calibration period | whole record |
| accumulation | sum over `scale` months (mean gives the same index for gamma) |

Reachable in:

- `climate_indices`: spi(distribution=gamma, zero_handling='classic', calibration years = the record)
- `xclim`: standardized_precipitation_index(dist='gamma', method='APP', fitkwargs={'floc': 0})
- `r_climatekit`: ck_spi(distribution='gamma'), fixed
- `ndmc_spi`: fixed: gamma, Thom, whole record
- `ncl`: dim_spi_n(spi_type=0), fixed

Mutually independent triples (lineage.json classes): {`climate_indices`, `ndmc_spi`, `r_climatekit`}; {`ndmc_spi`, `r_climatekit`, `xclim`}.

Residual differences after alignment:

- inverse_normal: NCL (gamma path) and NDMC use the A&S 26.2.23 approximation, the others exact (A&S bound 4.5e-4)
- output_bound: the library clips at +-3.09; climatekit replaces a CDF of exactly 0 or 1 by 0.001 or 0.999; xclim clips at +-8.21; NCL (gamma) and NDMC do not clip (sentinels +-1e37 and +-9999); apply one common clip in the harness
- NDMC prints two decimals, so a 1e-6 difference class cannot be resolved against it
- climatekit takes daily input, treats a missing day as 0, and needs 3 accumulations per month; the library wants 30 calibration years

### `spi-gamma-ml`: Gamma, numerical maximum likelihood with location fixed at 0, zeros stripped

| Axis | Position |
| --- | --- |
| distribution | gamma |
| estimator | maximum likelihood, loc = 0 |
| zero handling | p0 = zeros / n, zeros removed from the fit |

Reachable in:

- `xclim`: dist='gamma', method='ML', fitkwargs={'floc': 0}
- `standard_precip`: calculate(dist_type='gam', fit_type='mle'), floc=0 is its default
- `r_sci`: fitSCI(distr='gamma', p0=TRUE)

**No independent triple.** xclim and standard_precip share a numerical routine (both call scipy's gamma fit), so the three are not mutually independent.

Residual differences after alignment:

- R SCI adds a 0.01 x min(positive) pseudo-observation when zeros are present
- optimizer start values and tolerances differ (scipy versus fitdistrplus)

### `spi-pearson-lmoments`: Pearson Type III by L-moments on the non-zero values

| Axis | Position |
| --- | --- |
| distribution | Pearson Type III |
| estimator | unbiased-PWM L-moments (Hosking) |
| zero handling | p0 = zeros / n, zeros removed from the fit |

Reachable in:

- `climate_indices`: spi(distribution=pearson)
- `ncl`: dim_spi_n(spi_type=3)
- `standard_precip`: calculate(dist_type='pe3', fit_type='lmom')
- `r_spei`: spi(distribution='PearsonIII', fit='ub-pwm'); its zero adjustment has no effect (see the matrix)

**No independent triple.** All four descend from Hosking's L-moment Fortran or call it; lineage.json classes their pairs shared numerical routine.

Residual differences after alignment:

- NCL's Pearson path thins the fit sample for nrun above 12, requires a series length that is a multiple of 12, clamps the probability to [0.001, 0.999], and inverts with AS241 (not the A&S approximation its gamma path uses)
- R SPEI never applies p0

### `spi-gamma-lmoments`: Gamma by L-moments (Hosking PELGAM)

| Axis | Position |
| --- | --- |
| distribution | gamma |
| estimator | L-moments (PELGAM) |

Reachable in:

- `standard_precip`: calculate(dist_type='gam', fit_type='lmom'), the default
- `r_spei`: spi(distribution='Gamma', fit='ub-pwm'), the default
- `r_sei`: std_index(dist='gamma', method='lmme'), without gr_new (the per-group call reverts method to 'mle')

**No independent triple.** All three call Hosking's PELGAM (directly or through lmoments3 or lmom), so lineage.json classes their pairs shared numerical routine.

Residual differences after alignment:

- R SPEI's zero adjustment has no effect; R SEI stops on zeros

### `spei-glo-pwm`: Three-parameter log-logistic (generalized logistic) by unbiased-PWM L-moments

| Axis | Position |
| --- | --- |
| distribution | log-logistic (generalized logistic) |
| estimator | unbiased-PWM L-moments (Hosking PELGLO) |
| location offset | fitted |
| zero handling | none |
| calendar grouping | calendar month |
| calibration period | whole record |

Reachable in:

- `climate_indices`: spei(distribution=loglogistic)
- `r_spei`: spei(distribution='log-Logistic', fit='ub-pwm'), the default
- `r_climatekit`: ck_spei(distribution='log-logistic'), fixed
- `standard_precip`: calculate(dist_type='glo', fit_type='lmom') on the balance
- `xclim`: dist=&lt;lmoments3 glo object>, method='PWM'; not exercised in this ticket

**No independent triple.** The library is classed 'validated against' R SPEI, and every L-moment path descends from Hosking's code. climatekit, the one implementation here with its own estimator code, has no SPEI independence class yet.

Residual differences after alignment:

- climatekit shifts the data by -min + 1 first (harmless for a location-equivariant fit) and is absent from the lineage SPEI table
- the library adds 1000 per step before accumulating (also harmless)
- inputs: the water balance and the PET behind it must be supplied identically

### `spei-fisk-ml`: Three-parameter log-logistic (scipy fisk) by maximum likelihood

| Axis | Position |
| --- | --- |
| distribution | log-logistic (scipy fisk) |
| estimator | maximum likelihood |
| location offset | fitted |

Reachable in:

- `spei`: spei(dist=fisk), the default
- `xclim`: dist='fisk', method='ML'

**No independent triple.** Only two implementations; both call scipy's maximum-likelihood fit (lineage.json: shared numerical routine).

Residual differences after alignment:

- start values: xclim uses its own _fit_start, spei scale=std(data)

### `np-sei-position`: Empirical ranking with the SEI plotting position (1 + k) / (n + 2)

| Axis | Position |
| --- | --- |
| distribution | empirical |
| plotting position | (1 + k) / (n + 2), ties at the maximum rank |
| inverse normal | exact |

Reachable in:

- `climate_indices`: standardized_index(distribution=empirical)
- `r_sei`: std_index(dist='empirical')

**No independent triple.** Only the library and SEI expose this position, and the library is validated against SEI (tests/test_energy_sei_reference.py). Vonk's spei uses (i - 1/2)/n, EDDI uses Tukey with rounded constants, and the providers' formulas are unconfirmed.

Residual differences after alignment:

- calendar grouping: the library groups internally, SEI needs a factor
- SEI has no calibration-window argument other than x_ref

<!-- END GENERATED: aligned -->

### Mapping to the REFIMPL-02 ladder

[`protocol.md`](protocol.md) (REFIMPL-02, version 1.0.0, proposed until the maintainer approves its text) defines the ladder. L0 uses the defaults and aligns no axes. The table maps this matrix's axes to the other rungs.

| Rung | Axes in this matrix |
| --- | --- |
| L1 | accumulation, calendar grouping, calibration period (and the inputs, which this matrix does not cover) |
| L2 | distribution |
| L3 | estimator, location or offset |
| L4 | zero handling, plotting position, output bound, missing data and minimum sample, fit failure, and inverse normal |

## Not done

- No code from any comparator was run. Findings 1 to 10 are from reading source; finding 1 especially needs a run.
- Climate Engine, `GRIDMET/DROUGHT`, NCEI climdiv SPI, SPEIbase, and NOAA PSL EDDI are unverified: their source was not read, and no source was found for the first two.
- NCL was read at the 6.6.2 tag; the conda-forge binary run in REFIMPL-03 was not compared with it.
- `climatekit` is a provisional include and its `ck_spei` is not in the lineage table.
- PET (REFIMPL-21) and Palmer (REFIMPL-26) are out of scope.

## Per-package tables

Generated from `method-matrix.json`. A citation is a path inside the implementation's source tree at its pin, or a path in this repository for documentation.

### SPI

<!-- BEGIN GENERATED: spi -->

#### Defaults at a glance

| Axis | climate_indices<br>`drought.indices.spi` | xclim<br>`standardized_precipitation_index` | spei (Vonk)<br>`spei.spi` | standard_precip<br>`standard_precip.spi.SPI.calculate` | R SPEI<br>`SPEI::spi` | R SCI<br>`SCI::fitSCI` | R SEI<br>`SEI::std_index` | R climatekit<br>`climatekit::ck_spi` | NDMC SPI Generator<br>`SPIGenerator.exe` | NCL dim_spi_n<br>`dim_spi_n` | Climate Engine API<br>`Climate` | Earth Engine GRIDMET/DROUGHT<br>`Earth` | NCEI nClimDiv SPI<br>`NCEI` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accumulation | Sum over `scale` steps; a window holding a NaN is NaN; first scale-1 steps NaN | Input resampled to `freq` by mean only when its frequency differs (so daily to monthly gives a mean daily r... | timescale=0 by default: no accumulation. For timescale > 0: rolling SUM over that many series steps (not mo... | Rolling SUM over `scale` consecutive ROWS (positional, not time-aware; warns on irregular spacing); NaN in... | Kernel-weighted rolling sum, align right, first scale-1 NA; the default rectangular kernel is a plain sum | MEAN over time.scale (backward convolution filter); NA propagates | None unless agg_period is set (default NULL). With agg_period, agg_fun 'sum' by default over a time-based w... | Daily values summed to the months present in `dates` with sum(na.rm=TRUE) (a missing day counts as 0, a mon... | Sum of `timescale` consecutive monthly (or weekly) values; any missing value makes the sum missing (-99); f... | Sum of nrun consecutive values; any missing value makes the sum missing; first nrun-1 missing | Fixed trailing days (30, 90, 180, 365), not calendar months (unverified) | Fixed trailing days (30, 90, 180, 365), not calendar months (unverified) | Not established (unverified) |
| calendar grouping | Calendar month (12) or day of year on a 366-column calendar, every year a leap year; Feb 29 interpolated in... | Month (freq MS), ISO week (W) or day of year (D), by the input frequency. Day of year is the raw time.dayof... | Series folded onto a year-2000 frame and grouped by pandas Grouper(freq=fit_freq): calendar month, or day w... | freq 'M' month, 'W' ISO week, 'D' raw day of year (after Feb 28 a leap year's days sit one later than a com... | cycle() of the ts frequency; non-ts input is assumed monthly (12) | Calendar month, from first.mon | None built in: the caller passes a grouping factor gr_new (for example month) and one distribution is fit p... | Calendar month | Calendar month (12 groups) for monthly input; day of year groups for weekly input, with timescale x 4 aggre... | Twelve calendar months, one fit per month | Unresolved (unverified) | Unresolved (unverified) | Not established (unverified) |
| calibration period | Required calibration_year_initial/final; parameters fit on that window, every year standardized against it;... | cal_start/cal_end slice (default whole input); fit parameters from an earlier fit passed as `params` overri... | None: the distribution is fit on the whole series. SI.predict applies fitted parameters to another series | baseline_start/baseline_end (integer years inclusive, or dates); default whole record; all rows are standar... | ref.start/ref.end as c(year, period); default whole series; parameters can be passed in `params` | No window argument: fitSCI fits the series it is given, transformSCI standardizes any series with that fit | x_ref fits the distribution (default x_new); x_new is standardized; moving_window uses the preceding values... | No argument: the whole record | No argument: every non-missing value of the series enters the fit | None: the whole input record (the Pearson routine's comments say to edit the loop limits to use another per... | 1981-2016 set in the request (unverified) | 1981-2016 documented by the catalog (unverified) | Documented 1931-1990; observed behaviour reproduced by the full period of record (1895-2022), as Baldwin an... (unverified) |
| distribution | Chosen by required `distribution`: gamma or Pearson Type III; Pearson falls back to gamma on a failed fit o... | gamma by default | gamma by default; any scipy.stats continuous distribution | 'gam' (gamma) by default | spi(): Gamma by default | `distr` is required: gamma, genlog, gev, gumbel, lnorm, norm, pe3 or weibull by L-moment start values, othe... | Default is 'empirical' (see the non-parametric family); parametric: norm, lnorm, logis, llogis, exp, gamma,... | gamma by default | Gamma only | spi_type 0: gamma (default); spi_type 3: Pearson III; others are an error | Non-parametric (rank based) (unverified) | Non-parametric (rank based) (unverified) | Pearson Type III, the product's documented choice (unverified) |
| estimator | Gamma: Thom (1958) closed-form approximation to maximum likelihood on non-zero calibration values. Pearson:... | method='ML': scipy maximum likelihood started from _fit_start values | scipy dist.fit with method 'MLE' (shape, loc and scale free), started at scale=std(data) | fit_type='lmom' (default): lmoments3 L-moment fit; for gamma, Hosking's rational approximation from L1 and L2 | fit='ub-pwm': unbiased PWM to L-moments, then lmom::pelgam, with lmomco::pargam as the fallback when pelgam... | Maximum likelihood by fitdistrplus::mledist from L-moment or moment start values | method 'mle' (fitdistrplus::fitdist) | Gamma: Thom closed form on the positive accumulations (documented as MLE); Pearson III: method of moments (... | Thom (1958) closed form on the positive values | Gamma: Thom closed form on the positive values. Pearson III (spi_type=3): Hosking SAMLMR L-moments of the n... | Non-parametric: no parameters are fit | Non-parametric: no parameters are fit | Not established (unverified) |
| location offset | Gamma has no location (scale and shape only); Pearson fits a location; no offset added; negative precipitat... | ML fits a free location unless `floc` is passed in fitkwargs; APP requires floc | Location fitted (three-parameter fit); no offset | Gamma: location fixed at 0 (L-moments, and MLE via the floc=0 default); other distributions fit a location | Gamma: two parameters, no location; PearsonIII and log-Logistic: three | Set by the chosen distribution (pe3 and genlog carry a location) | Two-parameter families, no location | Gamma: none; Pearson III: location xi = mu - alpha * sign(skew) * beta | None (two-parameter gamma) | Gamma: none; Pearson III: fitted | Non-parametric | Non-parametric | Not established (unverified) |
| zero handling | Mixed CDF p0 + (1-p0)F(x), p0 = zeros / non-missing calibration values per calendar step; a zero scores Phi... | zero_inflated=True: zeros removed from the fit; p0 = zeros / non-NaN observations, zeros included ('ecdf');... | prob_zero=True: p0 = zeros / valid observations in the fit group mixes the CDF and sets zeros to p0 (classi... | gam and pe3: zeros removed from the fit, p0 = zeros / observations, mixed CDF p0 + (1-p0)F(x); zeros get p0... | Gamma and PearsonIII: p0 = zeros / n is computed and non-positive values are removed from the fit, but the... | `p0` is a required logical. With p0=TRUE and the default p0.center.mass=FALSE, p0 = zeros / n; when p0 > 0... | Positive-support families stop on zeros; zeros are handled only by censoring, which needs lower=0 (default... | Gamma: q = zeros / n; zeros score the CDF q and positives q + (1-q)F (classic). Pearson III: no zero mass | pzero = zeros / n; zeros score pzero, positives pzero + (1-pzero)P (classic) | pzero = zeros / n; zeros score pzero, positives pzero + (1-pzero)P (classic) | Not documented; ordinary ranks presumed (unverified) | Not documented; ordinary ranks presumed (unverified) | Not established (unverified) |
| plotting position | Parametric fit; the empirical path is in the non-parametric family | Parametric fit; the zero plotting position is under zero_handling | Parametric by default; the normal-scores option is in the non-parametric family | Parametric fit only | Parametric fit only (plotting positions appear only in the pp-pwm estimator) | Parametric fit only | Parametric fit; the empirical option is in the non-parametric family | Parametric fit only | Parametric fit only | Parametric fit only | Tukey family documented or inferred; formula and tie rule unconfirmed (unverified) | Tukey family documented or inferred; formula and tie rule unconfirmed (unverified) | Parametric fit |
| inverse normal | Exact: scipy.stats.norm.ppf (or the Rust norm_ppf kernel) | Exact: scipy norm ppf | Exact: scipy norm.ppf | Exact: scipy norm.ppf | Exact: qnorm | Exact: qnorm | Exact: qnorm, for index_type='normal' (the default) | Exact: stats::qnorm | Abramowitz and Stegun 26.2.23 rational approximation (c0 = 2.515517); A&S give a maximum error of 4.5e-4, a... | Gamma path: Abramowitz and Stegun 26.2.23 rational approximation (anvnrmd); A&S give a maximum error of 4.5... | Not documented (unverified) | Not documented (unverified) | Not established (unverified) |
| output bound | Clipped to [-3.09, 3.09] on the normal scale; output_scale='probability' is unclipped, 'bounded' is 2p-1 | Clipped to [-8.21, 8.21], the float64 limit of the inversion | None; a CDF of 0 or 1 gives an infinite value | CDF clipped to [1e-16, 1-1e-16], so about +-8.2 | None; a CDF of 0 or 1 gives -Inf or Inf | None by default (sci.limit = Inf); optional truncation at +-sci.limit | None; a parametric CDF of 0 or 1 gives an infinite value | A CDF of exactly 0 or below becomes 0.001 and of 1 or above becomes 0.999 (z = -3.09, +3.09); values in bet... | No clip; a CDF of 0 or 1 returns the sentinel +-9999.0, which equals the missing code. Output is rounded to... | Gamma path: none; a probability of 0 returns the sentinel +-1e37 | Not documented (unverified) | Not documented (unverified) | Not established (unverified) |
| missing data minimum sample | NaN in a window gives NaN; fit uses the non-missing calibration values; Pearson L-moments need at least 4 n... | NaN propagates through the window; NaN dropped from the fit; ML and APP: fewer than 2 values gives NaN para... | NaN dropped before the fit; no minimum sample | Rows with NaN dropped per group before the fit; L-moment fits need max(4, numargs + 3) values, MLE 4, count... | na.rm=FALSE: any NA in the input is an error; na.rm=TRUE: NA dropped from the fit and the window sum stays... | Non-finite values dropped from the fit; an empty or constant month has NA parameters with a warning; a wind... | NA removed; fewer than n_thres (10) values gives NA with a warning | At least 3 accumulations per calendar month, and 2 positive values for gamma | Missing codes -99 and -9999 are excluded from the fit and propagate through the window sum; no minimum sample | A calendar month whose accumulations are all equal is not fit and stays missing; any missing value in a win... | Not documented (unverified) | Not documented (unverified) | Not established (unverified) |
| fit failure | Pearson falls back to gamma with a warning. Gamma: a step with no fittable sample has NaN parameters and it... | Any NaN parameter makes all parameters NaN, so the step is NaN, silently for ML and APP; the PWM branch war... | Not handled: no try/except around the scipy fit, so its errors propagate and its result is used as returned | Params None, index NaN for that group, one UserWarning per column; TypeError from bad fit arguments propagates | Invalid L-moments or a constant series leave that period NA, silently | Month parameters NA, with a warning, unless start.fun.fix=TRUE, which keeps the start values | try() failure gives a warning and NA | Gamma: A &lt;= 0 returns NA without a warning. Pearson III: invalid sd or shape returns NA without a warning,... | One positive value: degenerate parameters (alpha 0, gamma 1, beta the mean), no error. All zeros: pzero = 1... | Gamma: exactly one positive value gives degenerate parameters (alpha 0, gamma 1, beta the mean), no error;... | Not documented (unverified) | Not documented (unverified) | Not established (unverified) |

#### Per-package evidence

##### climate_indices: drought.indices.spi (distribution= gamma | pearson, required)

Read at: monocongo/climate_indices@98578c87.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Sum over `scale` steps; a window holding a NaN is NaN; first scale-1 steps NaN<br>Option: Mean of the window (`standardized_index(aggregation='mean')`) [`src/climate_indices/CONTEXT.md:50`] | `src/climate_indices/compute.py:243-245`; `src/climate_indices/compute.py:309` |
| calendar grouping | Calendar month (12) or day of year on a 366-column calendar, every year a leap year; Feb 29 interpolated in non-leap years | `src/climate_indices/CONTEXT.md:27-28`; `src/climate_indices/compute.py:724-725` |
| calibration period | Required calibration_year_initial/final; parameters fit on that window, every year standardized against it; a window the record does not cover is clamped to the record (a window ending one year past it keeps its start) with CalibrationPeriodClampedWarning. The 30-year minimum raises InsufficientDataError only in the xarray adapter, for an in-memory single series containing NaN; the NumPy functions only warn (ShortCalibrationWarning) | `src/climate_indices/compute.py:699`; `src/climate_indices/_calibration_period.py:24`; `src/climate_indices/xarray_adapter.py:2056`; `src/climate_indices/compute.py:1249` |
| distribution | Chosen by required `distribution`: gamma or Pearson Type III; Pearson falls back to gamma on a failed fit or more than 50% lost values; log-logistic and GEV rejected by spi()<br>Option: gamma (`distribution=Distribution.gamma`)<br>Option: Pearson Type III (`distribution=Distribution.pearson`) | `src/climate_indices/drought/indices.py:991-992`; `src/climate_indices/compute.py:116` |
| estimator | Gamma: Thom (1958) closed-form approximation to maximum likelihood on non-zero calibration values. Pearson: Hosking SAMLMR sample L-moments (unbiased PWM) on non-zero values | `src/climate_indices/compute.py:2000-2001`; `src/climate_indices/lmoments.py:279` |
| location offset | Gamma has no location (scale and shape only); Pearson fits a location; no offset added; negative precipitation clipped to zero before fitting | `src/climate_indices/compute.py:2535`; `src/climate_indices/compute.py:2299` |
| zero handling | Mixed CDF p0 + (1-p0)F(x), p0 = zeros / non-missing calibration values per calendar step; a zero scores Phi^-1(p0) ('classic', the top of the zero mass). For Pearson, a support-limit mask can override that score and values below 0.0005 are placed with the zeros<br>Option: Zero scores Phi^-1(p0/2) (`zero_handling='center_of_mass'`) [`src/climate_indices/drought/indices.py:1022`]<br>Option: Zero scores -phi(Phi^-1(p0))/p0 (`zero_handling='mean_zero'`) [`src/climate_indices/drought/indices.py:1023`] | `src/climate_indices/compute.py:2557`; `src/climate_indices/drought/indices.py:531` |
| plotting position | n/a: Parametric fit; the empirical path is in the non-parametric family |  |
| inverse normal | Exact: scipy.stats.norm.ppf (or the Rust norm_ppf kernel) | `src/climate_indices/compute.py:2580` |
| output bound | Clipped to [-3.09, 3.09] on the normal scale; output_scale='probability' is unclipped, 'bounded' is 2p-1 | `src/climate_indices/drought/indices.py:50`; `src/climate_indices/drought/indices.py:186-190` |
| missing data minimum sample | NaN in a window gives NaN; fit uses the non-missing calibration values; Pearson L-moments need at least 4 non-zero values; no explicit minimum for gamma | `src/climate_indices/lmoments.py:19`; `src/climate_indices/_calibration_period.py:23` |
| fit failure | Pearson falls back to gamma with a warning. Gamma: a step with no fittable sample has NaN parameters and its non-zero values score NaN with no fit warning (the goodness-of-fit check skips invalid parameters); an all-zero calibration step scores its zeros -inf, clipped to -3.09; a step with no calibration data has undefined zero mass and its zeros are NaN | `src/climate_indices/drought/indices.py:534`; `src/climate_indices/compute.py:2511`; `src/climate_indices/compute.py:1420`; `src/climate_indices/compute.py:2524` |

##### xclim: standardized_precipitation_index (xclim.compute._agro)

Read at: xclim 1.0.0 sdist (sha256 e06477de).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Input resampled to `freq` by mean only when its frequency differs (so daily to monthly gives a mean daily rate, not a total), then rolling MEAN over `window` (default 1, no rolling) with skipna=False; NaN propagates; first window-1 steps NaN | `src/xclim/compute/stats.py:828`; `src/xclim/compute/stats.py:841` |
| calendar grouping | Month (freq MS), ISO week (W) or day of year (D), by the input frequency. Day of year is the raw time.dayofyear (a leap year's dates after 28 February sit one later), and fitted parameters are re-mapped to the target calendar with resample_doy | `src/xclim/compute/stats.py:810-816`; `src/xclim/compute/stats.py:1136` |
| calibration period | cal_start/cal_end slice (default whole input); fit parameters from an earlier fit passed as `params` override it | `src/xclim/compute/stats.py:1113`; `src/xclim/compute/stats.py:1079` |
| distribution | gamma by default<br>Option: fisk (log-logistic) (`dist='fisk'; ML or APP`)<br>Option: genextreme (`dist='genextreme'; ML only`)<br>Option: lognorm (`dist='lognorm'; ML or APP`)<br>Option: any scipy or lmoments3 distribution object (`dist=&lt;rv_continuous>; PWM with lmoments3`) | `src/xclim/compute/_agro.py:1025`; `src/xclim/compute/_agro.py:1149-1154` |
| estimator | method='ML': scipy maximum likelihood started from _fit_start values<br>Option: APP: Thom closed form for gamma on the values with x - floc > 0 (others are dropped from the fit but still scored), requires floc (`method='APP', fitkwargs={'floc': 0}`) [`src/xclim/compute/stats.py:652`; `src/xclim/compute/stats.py:657`; `src/xclim/compute/stats.py:921`]<br>Option: PWM: lmoments3 distribution object (`method='PWM', dist=lmoments3 object`) [`src/xclim/compute/stats.py:67`] | `src/xclim/compute/_agro.py:1026`; `src/xclim/compute/stats.py:56` |
| location offset | ML fits a free location unless `floc` is passed in fitkwargs; APP requires floc<br>Option: Location fixed (`fitkwargs={'floc': 0}`) | `src/xclim/compute/stats.py:921` |
| zero handling | zero_inflated=True: zeros removed from the fit; p0 = zeros / non-NaN observations, zeros included ('ecdf'); zeros placed at the upper bound of the zero mass, as 'classic'<br>Option: Zero at the midpoint of its rank interval (`prob_zero_interpolation='center' or a float in [0, 1]`)<br>Option: p0 = zeros / (n + 1) (`plotting_position_zero='weibull' or a (alpha, beta) tuple`) [`src/xclim/compute/stats.py:1103`]<br>Note: 'center' is the midpoint of the zero ranks, not p0/2 as the library's center_of_mass. | `src/xclim/compute/_agro.py:1163`; `src/xclim/compute/stats.py:1162`; `src/xclim/compute/stats.py:1096` |
| plotting position | n/a: Parametric fit; the zero plotting position is under zero_handling |  |
| inverse normal | Exact: scipy norm ppf | `src/xclim/compute/stats.py:1176` |
| output bound | Clipped to [-8.21, 8.21], the float64 limit of the inversion | `src/xclim/compute/stats.py:1184` |
| missing data minimum sample | NaN propagates through the window; NaN dropped from the fit; ML and APP: fewer than 2 values gives NaN parameters; PWM: numargs + 3 values needed (4 for gamma), else NaN | `src/xclim/compute/stats.py:50`; `src/xclim/compute/stats.py:69` |
| fit failure | Any NaN parameter makes all parameters NaN, so the step is NaN, silently for ML and APP; the PWM branch warns when there are too few unique values | `src/xclim/compute/stats.py:109-110`; `src/xclim/compute/stats.py:94-95` |

##### spei (Vonk): spei.spi (src/spei/si.py)

Read at: spei 0.8.3 sdist (sha256 50221408).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | timescale=0 by default: no accumulation. For timescale > 0: rolling SUM over that many series steps (not months), min_periods=timescale, then NaN rows dropped<br>Option: Mean of the window (`SI(agg_func='mean')`) [`src/spei/si.py:393`] | `src/spei/si.py:26`; `src/spei/si.py:408-411` |
| calendar grouping | Series folded onto a year-2000 frame and grouped by pandas Grouper(freq=fit_freq): calendar month, or day with Feb 29 in its own bin; fit_freq inferred from the (rolled, NaN-dropped) index; if pandas cannot infer a frequency (for example gaps) it silently falls back to monthly even for daily data<br>Option: Distribution fit over a window of neighbouring days (daily or weekly input only) (`fit_window=&lt;odd int>`) [`src/spei/si.py:449`] | `src/spei/si.py:477-478`; `src/spei/utils.py:118`; `src/spei/utils.py:84` |
| calibration period | None: the distribution is fit on the whole series. SI.predict applies fitted parameters to another series | `src/spei/si.py:589` |
| distribution | gamma by default; any scipy.stats continuous distribution | `src/spei/si.py:76` |
| estimator | scipy dist.fit with method 'MLE' (shape, loc and scale free), started at scale=std(data)<br>Option: Method of moments (`SI(fit_method='MM')`) [`src/spei/si.py:390`] | `src/spei/dist.py:95-99`; `src/spei/si.py:66` |
| location offset | Location fitted (three-parameter fit); no offset | `src/spei/dist.py:104` |
| zero handling | prob_zero=True: p0 = zeros / valid observations in the fit group mixes the CDF and sets zeros to p0 (classic placement), but zeros are NOT removed from the fit data<br>Option: No zero mass (`prob_zero=False`) | `src/spei/dist.py:70`; `src/spei/dist.py:117`; `src/spei/dist.py:61` |
| plotting position | n/a: Parametric by default; the normal-scores option is in the non-parametric family |  |
| inverse normal | Exact: scipy norm.ppf | `src/spei/si.py:576` |
| output bound | None; a CDF of 0 or 1 gives an infinite value | `src/spei/si.py:576` |
| missing data minimum sample | NaN dropped before the fit; no minimum sample | `src/spei/utils.py:143` |
| fit failure | Not handled: no try/except around the scipy fit, so its errors propagate and its result is used as returned | `src/spei/dist.py:95-99` |

##### standard_precip: standard_precip.spi.SPI.calculate (base_sp.BaseStandardIndex)

Read at: standard-precip 2.0.0 sdist (sha256 6a1f65c0); L-moment fits read in lmoments3 1.0.8 sdist (sha256 bb58bfc2, not pinned in comparators.lock.yaml).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Rolling SUM over `scale` consecutive ROWS (positional, not time-aware; warns on irregular spacing); NaN in the window gives NaN | `standard_precip/base_sp.py:50`; `standard_precip/base_sp.py:149` |
| calendar grouping | freq 'M' month, 'W' ISO week, 'D' raw day of year (after Feb 28 a leap year's days sit one later than a common year's), None one group; or a caller integer column (freq_col) | `standard_precip/base_sp.py:308-314` |
| calibration period | baseline_start/baseline_end (integer years inclusive, or dates); default whole record; all rows are standardized | `standard_precip/base_sp.py:330`; `standard_precip/base_sp.py:165` |
| distribution | 'gam' (gamma) by default<br>Option: pe3, exp, gev, gpa, gum, nor, wei, glo, gno, kap, wak (`dist_type=&lt;code>`)<br>Note: The docstring lists glo, gno and kap as MLE only and wak as L-moments only, but the registry maps glo, gno and kap to L-moment distributions too (and the SPEI docstring recommends glo with lmom). | `standard_precip/base_sp.py:185`; `standard_precip/_distributions.py:52-66` |
| estimator | fit_type='lmom' (default): lmoments3 L-moment fit; for gamma, Hosking's rational approximation from L1 and L2<br>Option: scipy maximum likelihood (`fit_type='mle'; gamma with floc=0 unless the caller passes loc or floc`) [`standard_precip/_distributions.py:50`] | `standard_precip/base_sp.py:73`; `lmoments3-1.0.8/src/lmoments3/distr.py:1198`; `lmoments3-1.0.8/src/lmoments3/distr.py:68` |
| location offset | Gamma: location fixed at 0 (L-moments, and MLE via the floc=0 default); other distributions fit a location | `standard_precip/_distributions.py:50`; `lmoments3-1.0.8/src/lmoments3/distr.py:1218` |
| zero handling | gam and pe3: zeros removed from the fit, p0 = zeros / observations, mixed CDF p0 + (1-p0)F(x); zeros get p0 (classic). Other distributions: no zero mass | `standard_precip/base_sp.py:103`; `standard_precip/base_sp.py:138`; `standard_precip/_distributions.py:53-55` |
| plotting position | n/a: Parametric fit only |  |
| inverse normal | Exact: scipy norm.ppf | `standard_precip/base_sp.py:146` |
| output bound | CDF clipped to [1e-16, 1-1e-16], so about +-8.2 | `standard_precip/base_sp.py:144` |
| missing data minimum sample | Rows with NaN dropped per group before the fit; L-moment fits need max(4, numargs + 3) values, MLE 4, counted after zeros are stripped for gam and pe3; a group of all zeros is not fit | `standard_precip/_distributions.py:103-109`; `standard_precip/base_sp.py:345` |
| fit failure | Params None, index NaN for that group, one UserWarning per column; TypeError from bad fit arguments propagates | `standard_precip/base_sp.py:382-386` |

##### R SPEI: SPEI::spi (calls spei())

Read at: SPEI 1.8.1 CRAN tarball (sha256 56cfcf6c).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Kernel-weighted rolling sum, align right, first scale-1 NA; the default rectangular kernel is a plain sum<br>Option: triangular, circular or gaussian kernel, with shift (`kernel=list(type=..., shift=...)`) [`R/kern.R:74`] | `R/spei.R:597`; `R/spei.R:598-601`; `R/kern.R:93`; `R/kern.R:101` |
| calendar grouping | cycle() of the ts frequency; non-ts input is assumed monthly (12) | `R/spei.R:627`; `R/spei.R:523` |
| calibration period | ref.start/ref.end as c(year, period); default whole series; parameters can be passed in `params` | `R/spei.R:611` |
| distribution | spi(): Gamma by default<br>Option: PearsonIII (`distribution='PearsonIII'`)<br>Option: log-Logistic (`distribution='log-Logistic'`) | `R/spi.R:9` |
| estimator | fit='ub-pwm': unbiased PWM to L-moments, then lmom::pelgam, with lmomco::pargam as the fallback when pelgam errors<br>Option: Plotting-position PWM, a = -0.35 (`fit='pp-pwm'`) [`R/spei.R:666`]<br>Option: Maximum likelihood, log-Logistic only (`fit='max-lik'`) [`R/spei.R:701`] | `R/spei.R:667`; `R/spei.R:688-691` |
| location offset | Gamma: two parameters, no location; PearsonIII and log-Logistic: three | `R/spei.R:557` |
| zero handling | Gamma and PearsonIII: p0 = zeros / n is computed and non-positive values are removed from the fit, but the p0 adjustment is applied to spei[ff, s] (still NA, initialised as acu * NA) and overwritten by qnorm(cdf_res), the CDF of the positive-only fit, on the next statement, so p0 never reaches the output. For Gamma a zero accumulation (CDF 0) becomes -Inf; for PearsonIII a zero gets the fitted CDF at 0<br>Note: Read from source at the pin, not executed. NEWS.md says zero-precipitation handling was fixed in 1.6. The adapter ticket (#1615) should confirm by running it; logged in the discrepancy ledger. | `R/spei.R:641`; `R/spei.R:642`; `R/spei.R:617`; `R/spei.R:723`; `R/spei.R:727` |
| plotting position | n/a: Parametric fit only (plotting positions appear only in the pp-pwm estimator) |  |
| inverse normal | Exact: qnorm | `R/spei.R:727` |
| output bound | None; a CDF of 0 or 1 gives -Inf or Inf | `R/spei.R:727` |
| missing data minimum sample | na.rm=FALSE: any NA in the input is an error; na.rm=TRUE: NA dropped from the fit and the window sum stays NA. A fit needs at least 4 values (after zero removal for Gamma and PearsonIII) and a non-zero standard deviation | `R/spei.R:467`; `R/spei.R:657` |
| fit failure | Invalid L-moments or a constant series leave that period NA, silently | `R/spei.R:672-674`; `R/spei.R:653` |

##### R SCI: SCI::fitSCI + transformSCI

Read at: SCI 1.0-3 CRAN tarball (sha256 1ec42a6c).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | MEAN over time.scale (backward convolution filter); NA propagates | `R/sci.r:44` |
| calendar grouping | Calendar month, from first.mon | `R/sci.r:41` |
| calibration period | No window argument: fitSCI fits the series it is given, transformSCI standardizes any series with that fit | `R/sci.r:166` |
| distribution | `distr` is required: gamma, genlog, gev, gumbel, lnorm, norm, pe3 or weibull by L-moment start values, others by moment start values | `R/distrstart.r:1-2` |
| estimator | Maximum likelihood by fitdistrplus::mledist from L-moment or moment start values | `R/sci.r:117` |
| location offset | Set by the chosen distribution (pe3 and genlog carry a location) | `R/sci_distributions.r:65` |
| zero handling | `p0` is a required logical. With p0=TRUE and the default p0.center.mass=FALSE, p0 = zeros / n; when p0 > 0 the zeros are removed and one pseudo-observation 0.01 x min(positive) is added before the fit; mixed CDF p0 + (1-p0)F(x), zero at the top of the mass. With p0=FALSE zeros stay in the fit<br>Option: p0 = zeros / (n + 1); zeros scored (n0 + 1) / (2(n + 1)) (`p0.center.mass=TRUE`) [`R/sci.r:90`; `R/sci.r:212`]<br>Note: p0 has no default; the Weibull form n0/(n+1) and centre-of-mass scoring apply only with p0.center.mass=TRUE. | `R/sci.r:88`; `R/sci.r:97`; `R/sci.r:210` |
| plotting position | n/a: Parametric fit only |  |
| inverse normal | Exact: qnorm | `R/sci.r:228` |
| output bound | None by default (sci.limit = Inf); optional truncation at +-sci.limit | `R/sci.r:230` |
| missing data minimum sample | Non-finite values dropped from the fit; an empty or constant month has NA parameters with a warning; a window containing NA is NA | `R/sci.r:69`; `R/sci.r:80` |
| fit failure | Month parameters NA, with a warning, unless start.fun.fix=TRUE, which keeps the start values<br>Option: Start values used on failure (`start.fun.fix=TRUE`) | `R/sci.r:103-105` |

##### R SEI: SEI::std_index (dist='gamma' or other parametric)

Read at: SEI 0.2.0 CRAN tarball (sha256 55ac099c).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | None unless agg_period is set (default NULL). With agg_period, agg_fun 'sum' by default over a time-based window; a window with more than na_thres (10) percent NA is NA, otherwise NAs are dropped and the function applies to the rest without rescaling, so a sum is biased low | `R/std_index.R:249`; `R/std_index.R:251`; `R/aggregate_xts.R:128-130` |
| calendar grouping | None built in: the caller passes a grouping factor gr_new (for example month) and one distribution is fit per group. The per-group call forwards only some arguments (method, n_thres, moving_window and rescale_fun are dropped, so method reverts to 'mle'), and aggregation then runs inside each group, so aggregate beforehand | `R/std_index.R:267-271` |
| calibration period | x_ref fits the distribution (default x_new); x_new is standardized; moving_window uses the preceding values only | `R/std_index.R:237`; `R/std_index.R:338` |
| distribution | Default is 'empirical' (see the non-parametric family); parametric: norm, lnorm, logis, llogis, exp, gamma, weibull, or kde | `R/std_index.R:238`; `R/fit_dist.R:374-375` |
| estimator | method 'mle' (fitdistrplus::fitdist)<br>Option: mme, qme, mge, mse (fitdistrplus) (`method=...`)<br>Option: L-moments via lmom, not for llogis (`method='lmme'`) [`R/fit_dist.R:290`; `R/fit_dist.R:303`] | `R/std_index.R:241`; `R/fit_dist.R:267` |
| location offset | Two-parameter families, no location | `R/fit_dist.R:269` |
| zero handling | Positive-support families stop on zeros; zeros are handled only by censoring, which needs lower=0 (default -Inf), with p_l = share at or below it, and censored values are excluded from the fit with no pseudo-observation. cens 'normal' (default) places them at pnorm(-dnorm(qnorm(p_l))/p_l), the conditional mean; 'prob' at p_l/2; 'none' at p_l | `R/fit_dist.R:387`; `R/get_pit.R:297-305`; `R/std_index.R:259` |
| plotting position | n/a: Parametric fit; the empirical option is in the non-parametric family |  |
| inverse normal | Exact: qnorm, for index_type='normal' (the default)<br>Option: The probability itself, or 2p - 1, instead of a z-score (`index_type='prob01' \| 'prob11'`) [`R/std_index.R:397`] | `R/std_index.R:395` |
| output bound | None; a parametric CDF of 0 or 1 gives an infinite value | `R/std_index.R:395` |
| missing data minimum sample | NA removed; fewer than n_thres (10) values gives NA with a warning | `R/fit_dist.R:232-235` |
| fit failure | try() failure gives a warning and NA | `R/fit_dist.R:263` |

##### R climatekit: climatekit::ck_spi (daily precipitation and dates)

Read at: climatekit 0.2.3 CRAN tarball (sha256 f3bdc9f8); provisional include.
Absent from the lineage SPEI table, so its SPEI pairs have no independence class yet.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Daily values summed to the months present in `dates` with sum(na.rm=TRUE) (a missing day counts as 0, a month of all-NA days sums to 0, absent months are not filled), then a trailing sum over `scale` rows of those totals, so the window spans calendar months only if the dates have no gaps | `R/drought.R:432`; `R/drought.R:52` |
| calendar grouping | Calendar month | `R/drought.R:55` |
| calibration period | No argument: the whole record | `R/drought.R:33` |
| distribution | gamma by default<br>Option: Pearson III (`distribution='pearsonIII'`) | `R/drought.R:34` |
| estimator | Gamma: Thom closed form on the positive accumulations (documented as MLE); Pearson III: method of moments (mean, sd, skewness) | `R/drought.R:466`; `R/drought.R:505` |
| location offset | Gamma: none; Pearson III: location xi = mu - alpha * sign(skew) * beta | `R/drought.R:513` |
| zero handling | Gamma: q = zeros / n; zeros score the CDF q and positives q + (1-q)F (classic). Pearson III: no zero mass | `R/drought.R:456`; `R/drought.R:475` |
| plotting position | n/a: Parametric fit only |  |
| inverse normal | Exact: stats::qnorm | `R/drought.R:482` |
| output bound | A CDF of exactly 0 or below becomes 0.001 and of 1 or above becomes 0.999 (z = -3.09, +3.09); values in between are not clamped, so \|z\| can exceed 3.09 | `R/drought.R:479-480` |
| missing data minimum sample | At least 3 accumulations per calendar month, and 2 positive values for gamma | `R/drought.R:63`; `R/drought.R:459` |
| fit failure | Gamma: A &lt;= 0 returns NA without a warning. Pearson III: invalid sd or shape returns NA without a warning, and an out-of-support point (NA CDF) scores z = 0 | `R/drought.R:464`; `R/drought.R:502`; `R/drought.R:526` |

##### NDMC SPI Generator: SPIGenerator.exe (VB.NET) calling StandardPrecipitationIndex.dll (C#)

Read at: SPIgenerator.zip sha256 ce8f37bc (1.7.6.0); source under code/ (VB.NET front end, C# DLL).
spimonthlyonly.cs ships but is not in the DLL project (spi_dll.csproj compiles two files).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Sum of `timescale` consecutive monthly (or weekly) values; any missing value makes the sum missing (-99); first timescale-1 values missing. Daily input is aggregated first in the front end: a month is missing when more than MaxMissingDays (8) days are absent, missing days are skipped without rescaling, and a week is missing only when all 7 days are | `code/spi_standalone/spi.vb:215-237`; `code/spi_standalone/spicollection.vb:898` |
| calendar grouping | Calendar month (12 groups) for monthly input; day of year groups for weekly input, with timescale x 4 aggregation | `code/spi_standalone/spi.vb:108`; `code/spi_standalone/spi.vb:152` |
| calibration period | No argument: every non-missing value of the series enters the fit | `code/spi_dll/standardprecipitationindex.cs:34` |
| distribution | Gamma only | `code/spi_dll/gammadistributionestimatedprobabilities.cs:76` |
| estimator | Thom (1958) closed form on the positive values | `code/spi_dll/gammadistributionestimatedprobabilities.cs:142` |
| location offset | None (two-parameter gamma) | `code/spi_dll/gammadistributionestimatedprobabilities.cs:143` |
| zero handling | pzero = zeros / n; zeros score pzero, positives pzero + (1-pzero)P (classic) | `code/spi_dll/gammadistributionestimatedprobabilities.cs:121`; `code/spi_dll/gammadistributionestimatedprobabilities.cs:170` |
| plotting position | n/a: Parametric fit only |  |
| inverse normal | Abramowitz and Stegun 26.2.23 rational approximation (c0 = 2.515517); A&S give a maximum error of 4.5e-4, a bound not stated in the source | `code/spi_dll/gammadistributionestimatedprobabilities.cs:41` |
| output bound | No clip; a CDF of 0 or 1 returns the sentinel +-9999.0, which equals the missing code. Output is rounded to two decimals in the writer | `code/spi_dll/gammadistributionestimatedprobabilities.cs:66`; `code/spi_standalone/spicollection.vb:328` |
| missing data minimum sample | Missing codes -99 and -9999 are excluded from the fit and propagate through the window sum; no minimum sample | `code/spi_dll/standardprecipitationindex.cs:15` |
| fit failure | One positive value: degenerate parameters (alpha 0, gamma 1, beta the mean), no error. All zeros: pzero = 1, a CDF of 1, and an output of +9999 | `code/spi_dll/gammadistributionestimatedprobabilities.cs:124-138`; `code/spi_dll/gammadistributionestimatedprobabilities.cs:132-138` |

##### NCL dim_spi_n: dim_spi_n(p, nrun, opt, dim): spi_type 0 gamma (spid.f), 3 Pearson III (spi3.f)

Read at: NCAR/ncl tag 6.6.2 (06f54867); the conda-forge 6.6.2 binary was run in REFIMPL-03, its build is not compared with this source.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Sum of nrun consecutive values; any missing value makes the sum missing; first nrun-1 missing | `ni/src/lib/nfpfort/spid.f:57` |
| calendar grouping | Twelve calendar months, one fit per month | `ni/src/lib/nfpfort/spid.f:72` |
| calibration period | None: the whole input record (the Pearson routine's comments say to edit the loop limits to use another period) | `ni/src/lib/nfpfort/spi3.f:132` |
| distribution | spi_type 0: gamma (default); spi_type 3: Pearson III; others are an error | `ni/src/lib/nfp/spiW.c:164` |
| estimator | Gamma: Thom closed form on the positive values. Pearson III (spi_type=3): Hosking SAMLMR L-moments of the non-zero values; for nrun above 12, the fit and pzero use every nstep-th accumulation (nstep = nrun rounded up to a multiple of 12, at least 24, undefined above 72) and all values are then scored | `ni/src/lib/nfpfort/spid.f:219`; `ni/src/lib/nfpfort/spi3.f:208`; `ni/src/lib/nfpfort/spi3.f:160` |
| location offset | Gamma: none; Pearson III: fitted | `ni/src/lib/nfpfort/spid.f:220` |
| zero handling | pzero = zeros / n; zeros score pzero, positives pzero + (1-pzero)P (classic) | `ni/src/lib/nfpfort/spid.f:194`; `ni/src/lib/nfpfort/spid.f:245` |
| plotting position | n/a: Parametric fit only |  |
| inverse normal | Gamma path: Abramowitz and Stegun 26.2.23 rational approximation (anvnrmd); A&S give a maximum error of 4.5e-4, a bound not stated in the source<br>Option: Hosking QUASTN, algorithm AS241 (exact to about 1e-16) (`spi_type=3 (Pearson III)`) [`ni/src/lib/nfpfort/spi3.f:843`] | `ni/src/lib/nfpfort/spid.f:131` |
| output bound | Gamma path: none; a probability of 0 returns the sentinel +-1e37<br>Option: Probability clamped to [0.001, 0.999], so +-3.09 (`spi_type=3 (Pearson III)`) [`ni/src/lib/nfpfort/spi3.f:261-262`] | `ni/src/lib/nfpfort/spid.f:146` |
| missing data minimum sample | A calendar month whose accumulations are all equal is not fit and stays missing; any missing value in a window makes it missing. Pearson III requires the series length to be a multiple of 12 | `ni/src/lib/nfpfort/spid.f:90`; `ni/src/lib/nfpfort/spi3.f:27` |
| fit failure | Gamma: exactly one positive value gives degenerate parameters (alpha 0, gamma 1, beta the mean), no error; an all-equal month, all-zero included, is skipped before the fit. Pearson III: fewer than 3 non-zero values or l2 = 0 leaves the value missing | `ni/src/lib/nfpfort/spid.f:205`; `ni/src/lib/nfpfort/spid.f:198` |

##### Climate Engine API: Climate Engine API standard_index, distribution=nonparametric

Read at: live service, accessed 2026-11-18 (comparators.md). **Source not read.**
No source; rows rest on the pilot's documentation and observations.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Fixed trailing days (30, 90, 180, 365), not calendar months | `docs/research/earthengine-climate-engine-pilot.md:58`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calendar grouping | **unverified.** Unresolved | `docs/research/earthengine-climate-engine-pilot.md:57`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calibration period | **unverified.** 1981-2016 set in the request | `docs/research/earthengine-climate-engine-pilot.md:32`<br>Basis: pilot request configuration, docs/research/earthengine-climate-engine-pilot.md |
| distribution | **unverified.** Non-parametric (rank based) | `docs/research/earthengine-climate-engine-pilot.md:53`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| estimator | n/a: Non-parametric: no parameters are fit |  |
| location offset | n/a: Non-parametric |  |
| zero handling | **unverified.** Not documented; ordinary ranks presumed | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| plotting position | **unverified.** Tukey family documented or inferred; formula and tie rule unconfirmed | `docs/research/earthengine-climate-engine-pilot.md:54`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| inverse normal | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| output bound | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| missing data minimum sample | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| fit failure | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |

##### Earth Engine GRIDMET/DROUGHT: Earth Engine GRIDMET/DROUGHT spi30d..spi1y bands

Read at: live collection, accessed 2026-11-18 (comparators.md). **Source not read.**
No source; rows rest on the pilot's documentation and observations.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Fixed trailing days (30, 90, 180, 365), not calendar months | `docs/research/earthengine-climate-engine-pilot.md:58`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calendar grouping | **unverified.** Unresolved | `docs/research/earthengine-climate-engine-pilot.md:57`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calibration period | **unverified.** 1981-2016 documented by the catalog | `docs/research/earthengine-climate-engine-pilot.md:32`<br>Basis: Earth Engine catalog as quoted in the pilot |
| distribution | **unverified.** Non-parametric (rank based) | `docs/research/earthengine-climate-engine-pilot.md:53`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| estimator | n/a: Non-parametric: no parameters are fit |  |
| location offset | n/a: Non-parametric |  |
| zero handling | **unverified.** Not documented; ordinary ranks presumed | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| plotting position | **unverified.** Tukey family documented or inferred; formula and tie rule unconfirmed | `docs/research/earthengine-climate-engine-pilot.md:54`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| inverse normal | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| output bound | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| missing data minimum sample | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| fit failure | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |

##### NCEI nClimDiv SPI: NCEI nClimDiv climdiv-sp*dv product (344 divisions)

Read at: fixture downloaded 2026-09-12 (tests/fixture/ncei_spi/provenance.json). **Source not read.**
Operational code is not public.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| calendar grouping | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| calibration period | **unverified.** Documented 1931-1990; observed behaviour reproduced by the full period of record (1895-2022), as Baldwin and Chen report | `tests/fixture/ncei_spi/provenance.json:70`<br>Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| distribution | **unverified.** Pearson Type III, the product's documented choice | `tests/fixture/ncei_spi/provenance.json:70`<br>Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| estimator | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| location offset | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| zero handling | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| plotting position | n/a: Parametric fit |  |
| inverse normal | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| output bound | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| missing data minimum sample | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |
| fit failure | **unverified.** Not established | Basis: tests/fixture/ncei_spi/provenance.json notes and Baldwin and Chen (2020) as cited there; operational code is not public |

<!-- END GENERATED: spi -->

### SPEI

<!-- BEGIN GENERATED: spei -->

#### Defaults at a glance

| Axis | climate_indices<br>`drought.indices.spei` | xclim<br>`standardized_precipitation_evapotranspiration_index` | spei (Vonk)<br>`spei.spei` | standard_precip<br>`standard_precip.spei.SPEI.calculate` | R SPEI<br>`SPEI::spei` | R SEI<br>`SEI::std_index` | R climatekit<br>`climatekit::ck_spei` | Climate Engine API<br>`Climate` | Earth Engine GRIDMET/DROUGHT<br>`Earth` | SPEIbase v2.11<br>`SPEIbase` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accumulation | Water balance (P - PET + 1000) formed per step, then summed over `scale` steps; NaN in a window gives NaN | The caller supplies the water budget (precipitation minus PET); resampled to `freq` by mean, then rolling M... | timescale=0 by default: no accumulation. For timescale > 0: rolling SUM over that many series steps (not mo... | The caller supplies the water balance as the precipitation column; rolling SUM over `scale` rows | Kernel-weighted rolling sum, align right, first scale-1 NA; the default rectangular kernel is a plain sum | None unless agg_period is set (default NULL). With agg_period, agg_fun 'sum' by default over a time-based w... | Daily P - PET summed to the months present in `dates` with na.rm=TRUE (a missing day counts as 0, absent mo... | Fixed trailing days (30, 90, 180, 365), not calendar months (unverified) | Fixed trailing days (30, 90, 180, 365), not calendar months (unverified) | Sum over scale months (R SPEI default), per the build version (unverified) |
| calendar grouping | Calendar month (12) or day of year on a 366-column calendar, every year a leap year; Feb 29 interpolated in... | Month (freq MS), ISO week (W) or day of year (D), by the input frequency. Day of year is the raw time.dayof... | Series folded onto a year-2000 frame and grouped by pandas Grouper(freq=fit_freq): calendar month, or day w... | freq 'M' month, 'W' ISO week, 'D' raw day of year (after Feb 28 a leap year's days sit one later than a com... | cycle() of the ts frequency; non-ts input is assumed monthly (12) | None built in: the caller passes a grouping factor gr_new (for example month) and one distribution is fit p... | Calendar month | Unresolved (unverified) | Unresolved (unverified) | Calendar month (unverified) |
| calibration period | Required calibration_year_initial/final; parameters fit on that window, every year standardized against it;... | cal_start/cal_end slice (default whole input); fit parameters from an earlier fit passed as `params` overri... | None: the distribution is fit on the whole series. SI.predict applies fitted parameters to another series | baseline_start/baseline_end (integer years inclusive, or dates); default whole record; all rows are standar... | ref.start/ref.end as c(year, period); default whole series; parameters can be passed in `params` | x_ref fits the distribution (default x_new); x_new is standardized; moving_window uses the preceding values... | No argument: the whole record | 1981-2016 set in the request (unverified) | 1981-2016 documented by the catalog (unverified) | Not established (whole record presumed) (unverified) |
| distribution | Required `distribution`: gamma, Pearson Type III, loglogistic (the generalized logistic SPEIbase uses, ADR-... | gamma by default (not log-logistic) | fisk (scipy log-logistic) by default; any scipy continuous distribution | Inherits the SPI default dist_type='gam' with fit_type='lmom', which strips zeros, applies p0 and fits a ga... | log-Logistic by default | Default 'empirical'; 'llogis' is the two-parameter log-logistic, and every positive-support family stops on... | log-logistic by default | Non-parametric (rank based) (unverified) | Non-parametric (rank based) (unverified) | log-Logistic (unverified) |
| estimator | Log-logistic: Hosking PELGLO on unbiased-PWM sample L-moments. GEV: Hosking L-moment estimator. Gamma and P... | method='ML': scipy maximum likelihood started from _fit_start values | scipy dist.fit with method 'MLE' (shape, loc and scale free), started at scale=std(data) | 'glo' with fit_type='lmom': lmoments3 Hosking generalized logistic L-moment fit; with 'mle': scipy genlogis... | fit='ub-pwm': unbiased PWM to L-moments, then lmom::pelglo, with lmomco::parglo as the fallback when pelglo... | method 'mle' (fitdistrplus::fitdist) | Unbiased PWM sample L-moments, then the generalized logistic form (covers -1 &lt; tau3 &lt; 1); GEV by Hosking's... | Non-parametric: no parameters are fit | Non-parametric: no parameters are fit | Unbiased PWM, per R SPEI 1.8.1 defaults (unverified) |
| location offset | A constant 1000 is added to P - PET at each step before accumulation (so 1000 x scale after summing); the l... | No offset parameter in 1.0.0; ML fits a free location unless floc is passed; APP requires floc | Location fitted (three-parameter fit); no offset | glo: location fitted (Hosking GLO has a location); gam: fixed at 0 | Three-parameter log-logistic (xi, alpha, kappa): a fitted location; no offset | Two-parameter families, no location | Data are shifted by -min + 1 before the PWM fit (a data-dependent offset); the fit has its own location | Non-parametric | Non-parametric | Fitted three-parameter location (unverified) |
| zero handling | spei() takes no zero_handling. Log-logistic and GEV use every value, zeros included, with no zero mass; the... | None: zero_inflated=False, so zeros are ordinary values | None by default: prob_zero=False | Zeros stripped and a p0 mass added (the inherited gam default); no stripping with dist_type='glo', and pe3... | None for log-Logistic: the zero-probability branch applies only to the other distributions | Not applicable to a signed series; censoring at lower/upper remains available | None: zeros are ordinary values | Not documented; ordinary ranks presumed (unverified) | Not documented; ordinary ranks presumed (unverified) | Signed water balance |
| plotting position | Parametric fit; the empirical path is in the non-parametric family | Parametric fit; the zero plotting position is under zero_handling | Parametric by default; the normal-scores option is in the non-parametric family | Parametric fit only | Parametric fit only (plotting positions appear only in the pp-pwm estimator) | Parametric fit; the empirical option is in the non-parametric family | Parametric fit only | Tukey family documented or inferred; formula and tie rule unconfirmed (unverified) | Tukey family documented or inferred; formula and tie rule unconfirmed (unverified) | Parametric fit |
| inverse normal | Exact: scipy.stats.norm.ppf (or the Rust norm_ppf kernel) | Exact: scipy norm ppf | Exact: scipy norm.ppf | Exact: scipy norm.ppf | Exact: qnorm | Exact: qnorm, for index_type='normal' (the default) | Exact: stats::qnorm | Not documented (unverified) | Not documented (unverified) | Exact qnorm, per R SPEI 1.8.1 (unverified) |
| output bound | Clipped to [-3.09, 3.09] on the normal scale; output_scale='probability' is unclipped, 'bounded' is 2p-1 | Clipped to [-8.21, 8.21], the float64 limit of the inversion | None; a CDF of 0 or 1 gives an infinite value | CDF clipped to [1e-16, 1-1e-16], so about +-8.2 | None; a CDF of 0 or 1 gives -Inf or Inf | None; a parametric CDF of 0 or 1 gives an infinite value | A CDF of exactly 0 or below becomes 0.001 and of 1 or above becomes 0.999 (z = -3.09, +3.09); values in bet... | Not documented (unverified) | Not documented (unverified) | None, per R SPEI 1.8.1; the product may truncate (unverified) |
| missing data minimum sample | NaN in a window gives NaN; log-logistic and GEV need at least 4 non-NaN calibration values per step, fewer... | NaN propagates through the window; NaN dropped from the fit; ML and APP: fewer than 2 values gives NaN para... | NaN dropped before the fit; no minimum sample | Rows with NaN dropped per group before the fit; L-moment fits need max(4, numargs + 3) values, MLE 4, count... | na.rm=FALSE: any NA in the input is an error; na.rm=TRUE: NA dropped from the fit and the window sum stays... | NA removed; fewer than n_thres (10) values gives NA with a warning | At least 3 accumulations per calendar month for log-logistic (4 for GEV) | Not documented (unverified) | Not documented (unverified) | Not established (unverified) |
| fit failure | A step that cannot be fit has its scale zeroed and transforms to NaN, with a warning; no fallback to anothe... | Any NaN parameter makes all parameters NaN, so the step is NaN, silently for ML and APP; the PWM branch war... | Not handled: no try/except around the scipy fit, so its errors propagate and its result is used as returned | Params None, index NaN for that group, one UserWarning per column; TypeError from bad fit arguments propagates | Invalid L-moments or a constant series leave that period NA, silently | try() failure gives a warning and NA | L-moment l2 &lt;= 0 or \|tau3\| >= 1: NA with a cli warning | Not documented (unverified) | Not documented (unverified) | Not established (unverified) |

#### Per-package evidence

##### climate_indices: drought.indices.spei (distribution required)

Read at: monocongo/climate_indices@98578c87.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Water balance (P - PET + 1000) formed per step, then summed over `scale` steps; NaN in a window gives NaN | `src/climate_indices/drought/indices.py:617`; `src/climate_indices/drought/indices.py:1091`; `src/climate_indices/compute.py:309` |
| calendar grouping | Calendar month (12) or day of year on a 366-column calendar, every year a leap year; Feb 29 interpolated in non-leap years | `src/climate_indices/CONTEXT.md:27-28`; `src/climate_indices/compute.py:724-725` |
| calibration period | Required calibration_year_initial/final; parameters fit on that window, every year standardized against it; a window the record does not cover is clamped to the record (a window ending one year past it keeps its start) with CalibrationPeriodClampedWarning. The 30-year minimum raises InsufficientDataError only in the xarray adapter, for an in-memory single series containing NaN; the NumPy functions only warn (ShortCalibrationWarning) | `src/climate_indices/compute.py:699`; `src/climate_indices/_calibration_period.py:24`; `src/climate_indices/xarray_adapter.py:2056`; `src/climate_indices/compute.py:1249` |
| distribution | Required `distribution`: gamma, Pearson Type III, loglogistic (the generalized logistic SPEIbase uses, ADR-0016) or gev; no fallback to gamma<br>Option: loglogistic (generalized logistic) (`distribution=Distribution.loglogistic`)<br>Option: gev (`distribution=Distribution.gev`)<br>Option: gamma or Pearson Type III (`distribution=Distribution.gamma \| pearson`) | `src/climate_indices/drought/indices.py:1135`; `src/climate_indices/drought/indices.py:1199`; `src/climate_indices/drought/indices.py:1192` |
| estimator | Log-logistic: Hosking PELGLO on unbiased-PWM sample L-moments. GEV: Hosking L-moment estimator. Gamma and Pearson as for SPI | `src/climate_indices/lmoments.py:364`; `src/climate_indices/compute.py:2980` |
| location offset | A constant 1000 is added to P - PET at each step before accumulation (so 1000 x scale after summing); the log-logistic fit has its own location; negative precipitation clipped to zero<br>Note: The fit is translation-equivariant in location, so the offset does not change the log-logistic or GEV result. | `src/climate_indices/drought/indices.py:1091`; `src/climate_indices/drought/indices.py:1086` |
| zero handling | spei() takes no zero_handling. Log-logistic and GEV use every value, zeros included, with no zero mass; the gamma and Pearson options apply the classic p0 mixture and exclude exact zeros from the fit | `src/climate_indices/compute.py:2982-2984`; `docs/adr/0015-zero-handling-in-standardized-indices.md:75` |
| plotting position | n/a: Parametric fit; the empirical path is in the non-parametric family |  |
| inverse normal | Exact: scipy.stats.norm.ppf (or the Rust norm_ppf kernel) | `src/climate_indices/compute.py:2580` |
| output bound | Clipped to [-3.09, 3.09] on the normal scale; output_scale='probability' is unclipped, 'bounded' is 2p-1 | `src/climate_indices/drought/indices.py:50`; `src/climate_indices/drought/indices.py:186-190` |
| missing data minimum sample | NaN in a window gives NaN; log-logistic and GEV need at least 4 non-NaN calibration values per step, fewer gives a failed step (NaN); gamma and Pearson as for SPI | `src/climate_indices/lmoments.py:19`; `src/climate_indices/lmoments.py:297` |
| fit failure | A step that cannot be fit has its scale zeroed and transforms to NaN, with a warning; no fallback to another family | `src/climate_indices/compute.py:2840`; `src/climate_indices/drought/indices.py:1192` |

##### xclim: standardized_precipitation_evapotranspiration_index (xclim.compute._agro)

Read at: xclim 1.0.0 sdist (sha256 e06477de).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | The caller supplies the water budget (precipitation minus PET); resampled to `freq` by mean, then rolling MEAN over `window`, skipna=False | `src/xclim/compute/_agro.py:1188`; `src/xclim/compute/stats.py:841` |
| calendar grouping | Month (freq MS), ISO week (W) or day of year (D), by the input frequency. Day of year is the raw time.dayofyear (a leap year's dates after 28 February sit one later), and fitted parameters are re-mapped to the target calendar with resample_doy | `src/xclim/compute/stats.py:810-816`; `src/xclim/compute/stats.py:1136` |
| calibration period | cal_start/cal_end slice (default whole input); fit parameters from an earlier fit passed as `params` override it | `src/xclim/compute/stats.py:1113`; `src/xclim/compute/stats.py:1079` |
| distribution | gamma by default (not log-logistic)<br>Option: fisk (log-logistic) (`dist='fisk'; ML or APP`)<br>Option: genextreme, lognorm, or a distribution object (`dist=...`) | `src/xclim/compute/_agro.py:1191` |
| estimator | method='ML': scipy maximum likelihood started from _fit_start values<br>Option: APP: Thom closed form for gamma on the values with x - floc > 0 (others are dropped from the fit but still scored), requires floc (`method='APP', fitkwargs={'floc': 0}`) [`src/xclim/compute/stats.py:652`; `src/xclim/compute/stats.py:657`; `src/xclim/compute/stats.py:921`]<br>Option: PWM: lmoments3 distribution object (`method='PWM', dist=lmoments3 object`) [`src/xclim/compute/stats.py:67`] | `src/xclim/compute/_agro.py:1026`; `src/xclim/compute/stats.py:56` |
| location offset | No offset parameter in 1.0.0; ML fits a free location unless floc is passed; APP requires floc | `src/xclim/compute/_agro.py:1187-1198`; `src/xclim/compute/stats.py:921` |
| zero handling | None: zero_inflated=False, so zeros are ordinary values | `src/xclim/compute/_agro.py:1269` |
| plotting position | n/a: Parametric fit; the zero plotting position is under zero_handling |  |
| inverse normal | Exact: scipy norm ppf | `src/xclim/compute/stats.py:1176` |
| output bound | Clipped to [-8.21, 8.21], the float64 limit of the inversion | `src/xclim/compute/stats.py:1184` |
| missing data minimum sample | NaN propagates through the window; NaN dropped from the fit; ML and APP: fewer than 2 values gives NaN parameters; PWM: numargs + 3 values needed (4 for gamma), else NaN | `src/xclim/compute/stats.py:50`; `src/xclim/compute/stats.py:69` |
| fit failure | Any NaN parameter makes all parameters NaN, so the step is NaN, silently for ML and APP; the PWM branch warns when there are too few unique values | `src/xclim/compute/stats.py:109-110`; `src/xclim/compute/stats.py:94-95` |

##### spei (Vonk): spei.spei (src/spei/si.py); the input series is P - PET

Read at: spei 0.8.3 sdist (sha256 50221408).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | timescale=0 by default: no accumulation. For timescale > 0: rolling SUM over that many series steps (not months), min_periods=timescale, then NaN rows dropped<br>Option: Mean of the window (`SI(agg_func='mean')`) [`src/spei/si.py:393`] | `src/spei/si.py:26`; `src/spei/si.py:408-411` |
| calendar grouping | Series folded onto a year-2000 frame and grouped by pandas Grouper(freq=fit_freq): calendar month, or day with Feb 29 in its own bin; fit_freq inferred from the (rolled, NaN-dropped) index; if pandas cannot infer a frequency (for example gaps) it silently falls back to monthly even for daily data<br>Option: Distribution fit over a window of neighbouring days (daily or weekly input only) (`fit_window=&lt;odd int>`) [`src/spei/si.py:449`] | `src/spei/si.py:477-478`; `src/spei/utils.py:118`; `src/spei/utils.py:84` |
| calibration period | None: the distribution is fit on the whole series. SI.predict applies fitted parameters to another series | `src/spei/si.py:589` |
| distribution | fisk (scipy log-logistic) by default; any scipy continuous distribution | `src/spei/si.py:142` |
| estimator | scipy dist.fit with method 'MLE' (shape, loc and scale free), started at scale=std(data)<br>Option: Method of moments (`SI(fit_method='MM')`) [`src/spei/si.py:390`] | `src/spei/dist.py:95-99`; `src/spei/si.py:66` |
| location offset | Location fitted (three-parameter fit); no offset | `src/spei/dist.py:104` |
| zero handling | None by default: prob_zero=False<br>Option: p0 = zeros / window length (`prob_zero=True`) [`src/spei/dist.py:70`] | `src/spei/si.py:146` |
| plotting position | n/a: Parametric by default; the normal-scores option is in the non-parametric family |  |
| inverse normal | Exact: scipy norm.ppf | `src/spei/si.py:576` |
| output bound | None; a CDF of 0 or 1 gives an infinite value | `src/spei/si.py:576` |
| missing data minimum sample | NaN dropped before the fit; no minimum sample | `src/spei/utils.py:143` |
| fit failure | Not handled: no try/except around the scipy fit, so its errors propagate and its result is used as returned | `src/spei/dist.py:95-99` |

##### standard_precip: standard_precip.spei.SPEI.calculate

Read at: standard-precip 2.0.0 sdist (sha256 6a1f65c0); L-moment fits read in lmoments3 1.0.8 sdist (sha256 bb58bfc2, not pinned in comparators.lock.yaml).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | The caller supplies the water balance as the precipitation column; rolling SUM over `scale` rows | `standard_precip/spei.py:11`; `standard_precip/base_sp.py:50` |
| calendar grouping | freq 'M' month, 'W' ISO week, 'D' raw day of year (after Feb 28 a leap year's days sit one later than a common year's), None one group; or a caller integer column (freq_col) | `standard_precip/base_sp.py:308-314` |
| calibration period | baseline_start/baseline_end (integer years inclusive, or dates); default whole record; all rows are standardized | `standard_precip/base_sp.py:330`; `standard_precip/base_sp.py:165` |
| distribution | Inherits the SPI default dist_type='gam' with fit_type='lmom', which strips zeros, applies p0 and fits a gamma, and the docstring calls inappropriate for a signed balance; it recommends dist_type='glo' with fit_type='lmom' | `standard_precip/spei.py:22`; `standard_precip/base_sp.py:185` |
| estimator | 'glo' with fit_type='lmom': lmoments3 Hosking generalized logistic L-moment fit; with 'mle': scipy genlogistic, a different family implementation | `lmoments3-1.0.8/src/lmoments3/distr.py:153`; `standard_precip/_distributions.py:64` |
| location offset | glo: location fitted (Hosking GLO has a location); gam: fixed at 0 | `lmoments3-1.0.8/src/lmoments3/distr.py:153`; `standard_precip/_distributions.py:50` |
| zero handling | Zeros stripped and a p0 mass added (the inherited gam default); no stripping with dist_type='glo', and pe3 would strip zeros | `standard_precip/_distributions.py:64` |
| plotting position | n/a: Parametric fit only |  |
| inverse normal | Exact: scipy norm.ppf | `standard_precip/base_sp.py:146` |
| output bound | CDF clipped to [1e-16, 1-1e-16], so about +-8.2 | `standard_precip/base_sp.py:144` |
| missing data minimum sample | Rows with NaN dropped per group before the fit; L-moment fits need max(4, numargs + 3) values, MLE 4, counted after zeros are stripped for gam and pe3; a group of all zeros is not fit | `standard_precip/_distributions.py:103-109`; `standard_precip/base_sp.py:345` |
| fit failure | Params None, index NaN for that group, one UserWarning per column; TypeError from bad fit arguments propagates | `standard_precip/base_sp.py:382-386` |

##### R SPEI: SPEI::spei (input is P - PET)

Read at: SPEI 1.8.1 CRAN tarball (sha256 56cfcf6c).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Kernel-weighted rolling sum, align right, first scale-1 NA; the default rectangular kernel is a plain sum<br>Option: triangular, circular or gaussian kernel, with shift (`kernel=list(type=..., shift=...)`) [`R/kern.R:74`] | `R/spei.R:597`; `R/spei.R:598-601`; `R/kern.R:93`; `R/kern.R:101` |
| calendar grouping | cycle() of the ts frequency; non-ts input is assumed monthly (12) | `R/spei.R:627`; `R/spei.R:523` |
| calibration period | ref.start/ref.end as c(year, period); default whole series; parameters can be passed in `params` | `R/spei.R:611` |
| distribution | log-Logistic by default<br>Option: Gamma, PearsonIII (`distribution=...`) | `R/spei.R:336` |
| estimator | fit='ub-pwm': unbiased PWM to L-moments, then lmom::pelglo, with lmomco::parglo as the fallback when pelglo errors<br>Option: Plotting-position PWM, a = -0.35 (`fit='pp-pwm'`) [`R/spei.R:666`]<br>Option: Maximum likelihood refinement of the L-moment fit (`fit='max-lik'`) [`R/spei.R:702`] | `R/spei.R:667`; `R/spei.R:683-686` |
| location offset | Three-parameter log-logistic (xi, alpha, kappa): a fitted location; no offset | `R/spei.R:571` |
| zero handling | None for log-Logistic: the zero-probability branch applies only to the other distributions | `R/spei.R:640` |
| plotting position | n/a: Parametric fit only (plotting positions appear only in the pp-pwm estimator) |  |
| inverse normal | Exact: qnorm | `R/spei.R:727` |
| output bound | None; a CDF of 0 or 1 gives -Inf or Inf | `R/spei.R:727` |
| missing data minimum sample | na.rm=FALSE: any NA in the input is an error; na.rm=TRUE: NA dropped from the fit and the window sum stays NA. A fit needs at least 4 values (after zero removal for Gamma and PearsonIII) and a non-zero standard deviation | `R/spei.R:467`; `R/spei.R:657` |
| fit failure | Invalid L-moments or a constant series leave that period NA, silently | `R/spei.R:672-674`; `R/spei.R:653` |

##### R SEI: SEI::std_index (input is P - PET)

Read at: SEI 0.2.0 CRAN tarball (sha256 55ac099c).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | None unless agg_period is set (default NULL). With agg_period, agg_fun 'sum' by default over a time-based window; a window with more than na_thres (10) percent NA is NA, otherwise NAs are dropped and the function applies to the rest without rescaling, so the error has no fixed direction and depends on the omitted values | `R/std_index.R:249`; `R/std_index.R:251`; `R/aggregate_xts.R:128-130` |
| calendar grouping | None built in: the caller passes a grouping factor gr_new (for example month) and one distribution is fit per group. The per-group call forwards only some arguments (method, n_thres, moving_window and rescale_fun are dropped, so method reverts to 'mle'), and aggregation then runs inside each group, so aggregate beforehand | `R/std_index.R:267-271` |
| calibration period | x_ref fits the distribution (default x_new); x_new is standardized; moving_window uses the preceding values only | `R/std_index.R:237`; `R/std_index.R:338` |
| distribution | Default 'empirical'; 'llogis' is the two-parameter log-logistic, and every positive-support family stops on negative values, so a signed P - PET cannot be fit by it<br>Option: norm, logis, kde (`dist=... (families that accept a signed series)`) | `R/std_index.R:238`; `R/fit_dist.R:385` |
| estimator | method 'mle' (fitdistrplus::fitdist)<br>Option: mme, qme, mge, mse (fitdistrplus) (`method=...`)<br>Option: L-moments via lmom, not for llogis (`method='lmme'`) [`R/fit_dist.R:290`; `R/fit_dist.R:303`] | `R/std_index.R:241`; `R/fit_dist.R:267` |
| location offset | Two-parameter families, no location | `R/fit_dist.R:269` |
| zero handling | Not applicable to a signed series; censoring at lower/upper remains available | `R/std_index.R:257-258` |
| plotting position | n/a: Parametric fit; the empirical option is in the non-parametric family |  |
| inverse normal | Exact: qnorm, for index_type='normal' (the default)<br>Option: The probability itself, or 2p - 1, instead of a z-score (`index_type='prob01' \| 'prob11'`) [`R/std_index.R:397`] | `R/std_index.R:395` |
| output bound | None; a parametric CDF of 0 or 1 gives an infinite value | `R/std_index.R:395` |
| missing data minimum sample | NA removed; fewer than n_thres (10) values gives NA with a warning | `R/fit_dist.R:232-235` |
| fit failure | try() failure gives a warning and NA | `R/fit_dist.R:263` |

##### R climatekit: climatekit::ck_spei (daily precipitation and PET; provisional, not in the lineage SPEI table)

Read at: climatekit 0.2.3 CRAN tarball (sha256 f3bdc9f8); provisional include.
Absent from the lineage SPEI table, so its SPEI pairs have no independence class yet.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Daily P - PET summed to the months present in `dates` with na.rm=TRUE (a missing day counts as 0, absent months are not filled), then a trailing sum over `scale` rows of those totals | `R/drought.R:129`; `R/drought.R:139` |
| calendar grouping | Calendar month | `R/drought.R:55` |
| calibration period | No argument: the whole record | `R/drought.R:33` |
| distribution | log-logistic by default<br>Option: GEV (`distribution='gev'`) | `R/drought.R:116` |
| estimator | Unbiased PWM sample L-moments, then the generalized logistic form (covers -1 &lt; tau3 &lt; 1); GEV by Hosking's approximation | `R/drought.R:617`; `R/drought.R:644`; `R/drought.R:571` |
| location offset | Data are shifted by -min + 1 before the PWM fit (a data-dependent offset); the fit has its own location | `R/drought.R:612` |
| zero handling | None: zeros are ordinary values | `R/drought.R:612` |
| plotting position | n/a: Parametric fit only |  |
| inverse normal | Exact: stats::qnorm | `R/drought.R:662` |
| output bound | A CDF of exactly 0 or below becomes 0.001 and of 1 or above becomes 0.999 (z = -3.09, +3.09); values in between are not clamped, so \|z\| can exceed 3.09. GEV: an out-of-support point scores z = 0 | `R/drought.R:660-661`; `R/drought.R:590` |
| missing data minimum sample | At least 3 accumulations per calendar month for log-logistic (4 for GEV) | `R/drought.R:150`; `R/drought.R:546` |
| fit failure | L-moment l2 &lt;= 0 or \|tau3\| >= 1: NA with a cli warning | `R/drought.R:629`; `R/drought.R:641` |

##### Climate Engine API: Climate Engine API standard_index, distribution=nonparametric (SPEI)

Read at: live service, accessed 2026-11-18 (comparators.md). **Source not read.**
No source; rows rest on the pilot's documentation and observations.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Fixed trailing days (30, 90, 180, 365), not calendar months | `docs/research/earthengine-climate-engine-pilot.md:58`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calendar grouping | **unverified.** Unresolved | `docs/research/earthengine-climate-engine-pilot.md:57`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calibration period | **unverified.** 1981-2016 set in the request | `docs/research/earthengine-climate-engine-pilot.md:32`<br>Basis: pilot request configuration, docs/research/earthengine-climate-engine-pilot.md |
| distribution | **unverified.** Non-parametric (rank based) | `docs/research/earthengine-climate-engine-pilot.md:53`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| estimator | n/a: Non-parametric: no parameters are fit |  |
| location offset | n/a: Non-parametric |  |
| zero handling | **unverified.** Not documented; ordinary ranks presumed | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| plotting position | **unverified.** Tukey family documented or inferred; formula and tie rule unconfirmed | `docs/research/earthengine-climate-engine-pilot.md:54`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| inverse normal | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| output bound | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| missing data minimum sample | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| fit failure | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |

##### Earth Engine GRIDMET/DROUGHT: Earth Engine GRIDMET/DROUGHT spei30d..spei1y bands (SPEI)

Read at: live collection, accessed 2026-11-18 (comparators.md). **Source not read.**
No source; rows rest on the pilot's documentation and observations.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Fixed trailing days (30, 90, 180, 365), not calendar months | `docs/research/earthengine-climate-engine-pilot.md:58`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calendar grouping | **unverified.** Unresolved | `docs/research/earthengine-climate-engine-pilot.md:57`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calibration period | **unverified.** 1981-2016 documented by the catalog | `docs/research/earthengine-climate-engine-pilot.md:32`<br>Basis: Earth Engine catalog as quoted in the pilot |
| distribution | **unverified.** Non-parametric (rank based) | `docs/research/earthengine-climate-engine-pilot.md:53`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| estimator | n/a: Non-parametric: no parameters are fit |  |
| location offset | n/a: Non-parametric |  |
| zero handling | **unverified.** Not documented; ordinary ranks presumed | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| plotting position | **unverified.** Tukey family documented or inferred; formula and tie rule unconfirmed | `docs/research/earthengine-climate-engine-pilot.md:54`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| inverse normal | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| output bound | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| missing data minimum sample | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| fit failure | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |

##### SPEIbase v2.11: SPEIbase v2.11 (CRU TS 4.09 inputs, Penman-Monteith PET)

Read at: fixture downloaded 2026-09-17 (tests/fixture/speibase/provenance.json). **Source not read.**
Described as built with R SPEI 1.8.1; not read here (lineage.json marks source_read true for it; here no source of the product was read).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Sum over scale months (R SPEI default), per the build version | `docs/research/reference-papers/comparators.md:186`<br>Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| calendar grouping | **unverified.** Calendar month | `docs/research/reference-papers/comparators.md:186`<br>Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| calibration period | **unverified.** Not established (whole record presumed) | `docs/research/reference-papers/comparators.md:186`<br>Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| distribution | **unverified.** log-Logistic | `docs/research/reference-papers/comparators.md:186`<br>Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| estimator | **unverified.** Unbiased PWM, per R SPEI 1.8.1 defaults | `docs/research/reference-papers/comparators.md:186`<br>Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| location offset | **unverified.** Fitted three-parameter location | `docs/research/reference-papers/comparators.md:186`<br>Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| zero handling | n/a: Signed water balance |  |
| plotting position | n/a: Parametric fit |  |
| inverse normal | **unverified.** Exact qnorm, per R SPEI 1.8.1 | `docs/research/reference-papers/comparators.md:186`<br>Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| output bound | **unverified.** None, per R SPEI 1.8.1; the product may truncate | `docs/research/reference-papers/comparators.md:186`<br>Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| missing data minimum sample | **unverified.** Not established | Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |
| fit failure | **unverified.** Not established | Basis: comparators.md and tests/fixture/speibase/provenance.json; the product's build is described on spei.csic.es and is not read here |

<!-- END GENERATED: spei -->

### Non-parametric standardization

<!-- BEGIN GENERATED: non_parametric -->

#### Defaults at a glance

| Axis | climate_indices<br>`drought.indices.standardized_index` | climate_indices<br>`drought.indices.eddi` | R SEI<br>`SEI::std_index` | spei (Vonk)<br>`spei.si.SI` | NOAA PSL EDDI<br>`NOAA` | Climate Engine API<br>`Climate` | Earth Engine GRIDMET/DROUGHT<br>`Earth` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| accumulation | Sum over `scale` steps; a window holding a NaN is NaN; first scale-1 steps NaN | PET summed over `scale` steps; NaN in a window gives NaN | None unless agg_period is set (default NULL). With agg_period, agg_fun 'sum' by default over a time-based w... | timescale=0 by default: no accumulation. For timescale > 0: rolling SUM over that many series steps (not mo... | Not established from source (unverified) | Fixed trailing days (30, 90, 180, 365), not calendar months (unverified) | Fixed trailing days (30, 90, 180, 365), not calendar months (unverified) |
| calendar grouping | Calendar month (12) or day of year on a 366-column calendar, every year a leap year; Feb 29 interpolated in... | Calendar month (12) or day of year on a 366-column calendar, every year a leap year; Feb 29 interpolated in... | None built in: the caller passes a grouping factor gr_new (for example month) and one distribution is fit p... | Series folded onto a year-2000 frame and grouped by pandas Grouper(freq=fit_freq): calendar month, or day w... | Not established from source (unverified) | Unresolved (unverified) | Unresolved (unverified) |
| calibration period | Required calibration_year_initial/final; parameters fit on that window, every year standardized against it;... | calibration_year_initial/final, required; each calendar step ranked within that window; a window the record... | x_ref fits the distribution (default x_new); x_new is standardized; moving_window uses the preceding values... | None: the distribution is fit on the whole series. SI.predict applies fitted parameters to another series | Not established from source (unverified) | 1981-2016 set in the request (unverified) | 1981-2016 documented by the catalog (unverified) |
| distribution | Empirical (rank based); no distribution is fit | Empirical ranking (NOAA PSL method) | 'empirical' is the default | Normal-scores transform: SI(normal_scores_transform=True), or the sgi() wrapper, which also sets agg_func='... | Empirical ranking (unverified) | Non-parametric (rank based) (unverified) | Non-parametric (rank based) (unverified) |
| estimator | Non-parametric | Non-parametric | Non-parametric | Non-parametric | Non-parametric | Non-parametric: no parameters are fit | Non-parametric: no parameters are fit |
| location offset | Non-parametric; accepts a signed series with clip_negatives=False | Non-parametric | Non-parametric; accepts a signed series | Non-parametric | Non-parametric | Non-parametric | Non-parametric |
| zero handling | Zeros are ordinary ranks; no separate zero mass | Ordinary ranks after preparation | Ordinary ranks; zeros can be censored with lower=0 | Ordinary ranks | Not established from source (unverified) | Not documented; ordinary ranks presumed (unverified) | Not documented; ordinary ranks presumed (unverified) |
| plotting position | (1 + k) / (n + 2), k = calibration values at or below the value (ties take the maximum rank); n = valid cal... | Tukey with NOAA's rounded constants: (pads + below + 0.66) / (n + pads + 0.33); below = strict count (ties... | (1 + ecdf(x) n) / (n + 2) = (1 + k) / (n + 2), k = values at or below (ties take the maximum rank) | (i - 1/2) / n for the i-th smallest value in each group (Hazen); ties get distinct positions by sort order | Tukey with rounded constants 0.66 and 0.33, per the library's account of it (unverified) | Tukey family documented or inferred; formula and tie rule unconfirmed (unverified) | Tukey family documented or inferred; formula and tie rule unconfirmed (unverified) |
| inverse normal | Exact: scipy.special.ndtri | Hastings (Abramowitz and Stegun 26.2.23) approximation, as the NOAA PSL Fortran (that Fortran is not read h... | Exact: qnorm | Exact: scipy norm.ppf | Hastings approximation, per the library's account of it (unverified) | Not documented (unverified) | Not documented (unverified) |
| output bound | Positions lie in (0, 1); then clipped to [-3.09, 3.09] on the normal scale | Probability clipped to [1e-10, 1-1e-10], z clipped to [-3.09, 3.09] | None; positions lie inside (0, 1), so values are finite | Positions lie inside (0, 1), so values are finite | Not established from source (unverified) | Not documented (unverified) | Not documented (unverified) |
| missing data minimum sample | Fewer than 10 valid calibration values at a step gives NaN | Fewer than 2 valid climatology values gives NaN; a missing value stays missing | Fewer than n_thres (10) values gives NA with a warning; fewer than 100 warns that the empirical distributio... | NaN dropped; no minimum sample | Not established from source (unverified) | Not documented (unverified) | Not documented (unverified) |
| fit failure | Nothing is fit | Nothing is fit | Nothing is fit | Nothing is fit | Nothing is fit | Not documented (unverified) | Not documented (unverified) |

#### Per-package evidence

##### climate_indices: drought.indices.standardized_index(distribution=empirical)

Read at: monocongo/climate_indices@98578c87.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | Sum over `scale` steps; a window holding a NaN is NaN; first scale-1 steps NaN<br>Option: Mean of the window (`standardized_index(aggregation='mean')`) [`src/climate_indices/CONTEXT.md:50`] | `src/climate_indices/compute.py:243-245`; `src/climate_indices/compute.py:309` |
| calendar grouping | Calendar month (12) or day of year on a 366-column calendar, every year a leap year; Feb 29 interpolated in non-leap years | `src/climate_indices/CONTEXT.md:27-28`; `src/climate_indices/compute.py:724-725` |
| calibration period | Required calibration_year_initial/final; parameters fit on that window, every year standardized against it; a window the record does not cover is clamped to the record (a window ending one year past it keeps its start) with CalibrationPeriodClampedWarning. The 30-year minimum raises InsufficientDataError only in the xarray adapter, for an in-memory single series containing NaN; the NumPy functions only warn (ShortCalibrationWarning) | `src/climate_indices/compute.py:699`; `src/climate_indices/_calibration_period.py:24`; `src/climate_indices/xarray_adapter.py:2056`; `src/climate_indices/compute.py:1249` |
| distribution | Empirical (rank based); no distribution is fit | `src/climate_indices/compute.py:3255`; `src/climate_indices/compute.py:3405-3409` |
| estimator | n/a: Non-parametric |  |
| location offset | n/a: Non-parametric; accepts a signed series with clip_negatives=False |  |
| zero handling | Zeros are ordinary ranks; no separate zero mass | `src/climate_indices/compute.py:3275-3277` |
| plotting position | (1 + k) / (n + 2), k = calibration values at or below the value (ties take the maximum rank); n = valid calibration values at that calendar step | `src/climate_indices/compute.py:3277`; `src/climate_indices/compute.py:3287` |
| inverse normal | Exact: scipy.special.ndtri | `src/climate_indices/compute.py:3291` |
| output bound | Positions lie in (0, 1); then clipped to [-3.09, 3.09] on the normal scale | `src/climate_indices/drought/indices.py:681` |
| missing data minimum sample | Fewer than 10 valid calibration values at a step gives NaN | `src/climate_indices/compute.py:99` |
| fit failure | n/a: Nothing is fit |  |

##### climate_indices: drought.indices.eddi

Read at: monocongo/climate_indices@98578c87.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | PET summed over `scale` steps; NaN in a window gives NaN | `src/climate_indices/drought/indices.py:407` |
| calendar grouping | Calendar month (12) or day of year on a 366-column calendar, every year a leap year; Feb 29 interpolated in non-leap years | `src/climate_indices/CONTEXT.md:27-28`; `src/climate_indices/compute.py:724-725` |
| calibration period | calibration_year_initial/final, required; each calendar step ranked within that window; a window the record does not cover raises CalibrationPeriodError (rejected, not clamped as SPI, SPEI and the empirical path do) | `src/climate_indices/drought/indices.py:311`; `src/climate_indices/drought/indices.py:396` |
| distribution | Empirical ranking (NOAA PSL method) | `src/climate_indices/drought/indices.py:319` |
| estimator | n/a: Non-parametric |  |
| location offset | n/a: Non-parametric |  |
| zero handling | Ordinary ranks after preparation | `src/climate_indices/drought/indices.py:450-456`; `src/climate_indices/compute.py:2383-2385` |
| plotting position | Tukey with NOAA's rounded constants: (pads + below + 0.66) / (n + pads + 0.33); below = strict count (ties take the minimum rank); leading scale pads count as lower than every value | `src/climate_indices/drought/indices.py:456` |
| inverse normal | Hastings (Abramowitz and Stegun 26.2.23) approximation, as the NOAA PSL Fortran (that Fortran is not read here) | `src/climate_indices/drought/indices.py:62`; `src/climate_indices/drought/indices.py:277` |
| output bound | Probability clipped to [1e-10, 1-1e-10], z clipped to [-3.09, 3.09] | `src/climate_indices/drought/indices.py:472`; `src/climate_indices/drought/indices.py:477` |
| missing data minimum sample | Fewer than 2 valid climatology values gives NaN; a missing value stays missing | `src/climate_indices/drought/indices.py:461` |
| fit failure | n/a: Nothing is fit |  |

##### R SEI: SEI::std_index(dist='empirical')

Read at: SEI 0.2.0 CRAN tarball (sha256 55ac099c).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | None unless agg_period is set (default NULL). With agg_period, agg_fun 'sum' by default over a time-based window; a window with more than na_thres (10) percent NA is NA, otherwise NAs are dropped and the function applies to the rest without rescaling, so the error has no fixed direction and depends on the omitted values | `R/std_index.R:249`; `R/std_index.R:251`; `R/aggregate_xts.R:128-130` |
| calendar grouping | None built in: the caller passes a grouping factor gr_new (for example month) and one distribution is fit per group. The per-group call forwards only some arguments (method, n_thres, moving_window and rescale_fun are dropped, so method reverts to 'mle'), and aggregation then runs inside each group, so aggregate beforehand | `R/std_index.R:267-271` |
| calibration period | x_ref fits the distribution (default x_new); x_new is standardized; moving_window uses the preceding values only | `R/std_index.R:237`; `R/std_index.R:338` |
| distribution | 'empirical' is the default | `R/std_index.R:238` |
| estimator | n/a: Non-parametric |  |
| location offset | n/a: Non-parametric; accepts a signed series |  |
| zero handling | Ordinary ranks; zeros can be censored with lower=0 | `R/std_index.R:257-258` |
| plotting position | (1 + ecdf(x) n) / (n + 2) = (1 + k) / (n + 2), k = values at or below (ties take the maximum rank) | `R/fit_dist.R:241` |
| inverse normal | Exact: qnorm | `R/std_index.R:395` |
| output bound | None; positions lie inside (0, 1), so values are finite | `R/std_index.R:395` |
| missing data minimum sample | Fewer than n_thres (10) values gives NA with a warning; fewer than 100 warns that the empirical distribution is not recommended | `R/fit_dist.R:232-235`; `R/fit_dist.R:379` |
| fit failure | n/a: Nothing is fit |  |

##### spei (Vonk): spei.si.SI(normal_scores_transform=True)

Read at: spei 0.8.3 sdist (sha256 50221408).

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | timescale=0 by default: no accumulation. For timescale > 0: rolling SUM over that many series steps (not months), min_periods=timescale, then NaN rows dropped<br>Option: Mean of the window (`SI(agg_func='mean')`) [`src/spei/si.py:393`] | `src/spei/si.py:26`; `src/spei/si.py:408-411` |
| calendar grouping | Series folded onto a year-2000 frame and grouped by pandas Grouper(freq=fit_freq): calendar month, or day with Feb 29 in its own bin; fit_freq inferred from the (rolled, NaN-dropped) index; if pandas cannot infer a frequency (for example gaps) it silently falls back to monthly even for daily data<br>Option: Distribution fit over a window of neighbouring days (daily or weekly input only) (`fit_window=&lt;odd int>`) [`src/spei/si.py:449`] | `src/spei/si.py:477-478`; `src/spei/utils.py:118`; `src/spei/utils.py:84` |
| calibration period | None: the distribution is fit on the whole series. SI.predict applies fitted parameters to another series | `src/spei/si.py:589` |
| distribution | Normal-scores transform: SI(normal_scores_transform=True), or the sgi() wrapper, which also sets agg_func='mean'. spi(), spei(), ssfi() and ssmi() hard-code False | `src/spei/si.py:133`; `src/spei/si.py:68`; `src/spei/si.py:513` |
| estimator | n/a: Non-parametric |  |
| location offset | n/a: Non-parametric |  |
| zero handling | Ordinary ranks | `src/spei/si.py:513` |
| plotting position | (i - 1/2) / n for the i-th smallest value in each group (Hazen); ties get distinct positions by sort order | `src/spei/si.py:526` |
| inverse normal | Exact: scipy norm.ppf | `src/spei/si.py:576` |
| output bound | Positions lie inside (0, 1), so values are finite | `src/spei/si.py:526` |
| missing data minimum sample | NaN dropped; no minimum sample | `src/spei/utils.py:143` |
| fit failure | n/a: Nothing is fit |  |

##### NOAA PSL EDDI: NOAA PSL EDDI product and Fortran

Read at: fixtures under tests/fixture (tests/test_noaa_eddi_reference.py). **Source not read.**
The NOAA Fortran is not read here.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Not established from source | Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| calendar grouping | **unverified.** Not established from source | Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| calibration period | **unverified.** Not established from source | Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| distribution | **unverified.** Empirical ranking | `src/climate_indices/drought/indices.py:319-321`<br>Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| estimator | n/a: Non-parametric |  |
| location offset | n/a: Non-parametric |  |
| zero handling | **unverified.** Not established from source | Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| plotting position | **unverified.** Tukey with rounded constants 0.66 and 0.33, per the library's account of it | `src/climate_indices/drought/indices.py:323`; `src/climate_indices/drought/indices.py:455-456`<br>Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| inverse normal | **unverified.** Hastings approximation, per the library's account of it | `src/climate_indices/drought/indices.py:277`<br>Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| output bound | **unverified.** Not established from source | Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| missing data minimum sample | **unverified.** Not established from source | Basis: src/climate_indices/drought/indices.py docstrings say the library follows the NOAA PSL Fortran; the Fortran is not read here (tests/test_noaa_eddi_reference.py compares to the fixtures) |
| fit failure | n/a: Nothing is fit |  |

##### Climate Engine API: Climate Engine API standard_index, distribution=nonparametric

Read at: live service, accessed 2026-11-18 (comparators.md). **Source not read.**
No source; rows rest on the pilot's documentation and observations.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Fixed trailing days (30, 90, 180, 365), not calendar months | `docs/research/earthengine-climate-engine-pilot.md:58`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calendar grouping | **unverified.** Unresolved | `docs/research/earthengine-climate-engine-pilot.md:57`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calibration period | **unverified.** 1981-2016 set in the request | `docs/research/earthengine-climate-engine-pilot.md:32`<br>Basis: pilot request configuration, docs/research/earthengine-climate-engine-pilot.md |
| distribution | **unverified.** Non-parametric (rank based) | `docs/research/earthengine-climate-engine-pilot.md:53`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| estimator | n/a: Non-parametric: no parameters are fit |  |
| location offset | n/a: Non-parametric |  |
| zero handling | **unverified.** Not documented; ordinary ranks presumed | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| plotting position | **unverified.** Tukey family documented or inferred; formula and tie rule unconfirmed | `docs/research/earthengine-climate-engine-pilot.md:54`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| inverse normal | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| output bound | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| missing data minimum sample | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| fit failure | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |

##### Earth Engine GRIDMET/DROUGHT: Earth Engine GRIDMET/DROUGHT SPI, SPEI and EDDI bands

Read at: live collection, accessed 2026-11-18 (comparators.md). **Source not read.**
No source; rows rest on the pilot's documentation and observations.

| Axis | Position (default; options follow) | Evidence |
| --- | --- | --- |
| accumulation | **unverified.** Fixed trailing days (30, 90, 180, 365), not calendar months | `docs/research/earthengine-climate-engine-pilot.md:58`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calendar grouping | **unverified.** Unresolved | `docs/research/earthengine-climate-engine-pilot.md:57`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| calibration period | **unverified.** 1981-2016 documented by the catalog | `docs/research/earthengine-climate-engine-pilot.md:32`<br>Basis: Earth Engine catalog as quoted in the pilot |
| distribution | **unverified.** Non-parametric (rank based) | `docs/research/earthengine-climate-engine-pilot.md:53`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| estimator | n/a: Non-parametric: no parameters are fit |  |
| location offset | n/a: Non-parametric |  |
| zero handling | **unverified.** Not documented; ordinary ranks presumed | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| plotting position | **unverified.** Tukey family documented or inferred; formula and tie rule unconfirmed | `docs/research/earthengine-climate-engine-pilot.md:54`<br>Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| inverse normal | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| output bound | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| missing data minimum sample | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |
| fit failure | **unverified.** Not documented | Basis: docs/research/earthengine-climate-engine-pilot.md method-alignment table; the product source is not available and the pilot records its exact methods as unconfirmed |

<!-- END GENERATED: non_parametric -->
