# Earth Engine and Climate Engine cross-validation pilot

Status: step 1 of a proposed collaboration with the Climate Engine team (DRI).
Reference A has run. Reference B is pending an API key. This page is an internal
working note; the user-facing setup instructions live in
[the Earth Engine and Climate Engine guide](../earthengine-climate-engine.md).

## Purpose

Answer one question with numbers: when `climate_indices` is given the same input
data that a public provider used, does it produce the same drought index values,
and where does it differ? The answer decides whether a joint dataset, a shared
method note, or a contributed implementation is worth proposing.

Two references were chosen because they are independent of this library and of
each other:

- **A. `GRIDMET/DROUGHT`**, the precomputed drought collection in the Earth
  Engine Data Catalog.
- **B. The Climate Engine API** `timeseries/standard_index` endpoint.

They are reported separately. A result against one is never presented as a
result against the other.

## What ran

| Item | Value |
| --- | --- |
| Library | `climate_indices` 3.2.0, branch `feature/earthengine-integration` |
| Reader | `xee` 0.1.2, `earthengine-api` 1.7.52, Python 3.14.7 |
| Inputs | `IDAHO_EPSCOR/GRIDMET`, bands `pr`, `eto`, `etr` (mm/day), 1980-01 to 2024-12 |
| Reference A | `GRIDMET/DROUGHT`, 5-day pentad, 528 images read, 1981-01 to 2024-12 |
| Reference B | `POST https://api.climateengine.org/timeseries/standard_index/coordinates`, not run (no key) |
| Region | Apalachicola–Chattahoochee–Flint basin bounding box, 30.0–35.0 N, 85.7–83.3 W |
| Grid | Native gridMET lattice, cropped to the box: 58 by 133 pixels, 7,714 cells, 4,638 m |
| Cells with data | 7,303 (the masked remainder lies outside gridMET's land coverage) |
| Calibration | 1981–2016, the baseline the catalog documents for reference A |
| Comparison window | 1981–2024, 528 months |
| Command | `uv run --extra earthengine scripts/earthengine_climate_engine_pilot.py` |

Both collections report the identical native grid, and the script stops if they
ever differ. The crop snaps to the source pixel edges, so no value is resampled
before comparison.

The exact version, commit and dirty flag of every run are written to
`docs/research/earthengine-pilot-results/pilot-metrics.json`.

## Method alignment

| Choice | `climate_indices` | Reference A | Match |
| --- | --- | --- | --- |
| Distribution | rank based | non-parametric, plotting positions | Yes, family |
| Plotting position | `(k + 1) / (n + 2)` for SPI and SPEI (SEI convention) | consistent with Tukey `(k − 1/3) / (n + 1/3)` | **No** for SPI and SPEI |
| Plotting position, EDDI | Tukey `(k − 1/3) / (n + 1/3)` | Tukey | Yes |
| Calibration | 1981–2016, explicit | 1981–2016, documented | Yes |
| Accumulation window | calendar month | trailing fixed days, 5-day cadence | **No** |
| Sampling date | month end | last pentad at or before month end | Offset 0–4 days, mean 1.83 |
| Precipitations input | gridMET `pr` | gridMET `pr` | Yes |
| Evaporative demand, SPEI | gridMET `eto` (grass) | "reference evapotranspiration", crop unnamed | Assumed |
| Evaporative demand, EDDI | gridMET `eto` primary, `etr` sensitivity | crop unnamed | Assumed |
| Time scales | 1, 3, 6, 12 calendar months | 30d, 90d, 180d, 1y | Nearest equivalent only |

Two mismatches are structural and cannot be removed without changing what this
library computes:

1. **Window shape.** `climate_indices` accumulates whole calendar months.
   Reference A accumulates fixed-length trailing windows sampled every five days.
   A 12-month index therefore covers 365 or 366 days in the reference and 365 or
   366 days for us, but a 3-month index covers 90 days there and 89 to 92 days
   here.
2. **Plotting position.** This library's `Distribution.empirical` uses the SEI
   convention `(k + 1) / (n + 2)`, which is pinned by an external oracle
   (`tests/test_zero_handling_reference.py`). On a 36-value calibration sample
   that convention produces z from −1.620 to +1.938 with a mean of **+0.0538**.
   The Tukey convention that reference A's endpoints imply is symmetric about
   zero, with z from −2.089 to +2.089.

   `eddi()` already uses the Tukey convention, so EDDI is comparable and
   SPI/SPEI are not, in this specific respect. That asymmetry is in this
   library, not in the reference.

The number +0.0538 is the most important figure in this report. It is a constant
that the plotting-position difference adds to every SPI and SPEI value, and it
is computable in closed form:

```bash
uv run python -c "
import numpy as np; from scipy.stats import norm
k = np.arange(1, 37)
print(norm.ppf((k + 1) / 38).mean())"
```

## Results, reference A

Pooled over 7,303 cells and 528 months, so `n` is about 3.85 million pairs per
index and scale. `bias` is `mean(ours − reference)`. Category agreement uses the
catalog's documented SPI/SPEI/EDDI classes.

| Index | Scale | n | bias | MAE | RMSE | r | max abs | same class | within one class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPI | 1 mo (30d) | 3,855,984 | +0.061 | 0.230 | 0.369 | 0.928 | 3.50 | 0.638 | 0.906 |
| SPI | 3 mo (90d) | 3,855,984 | +0.054 | 0.157 | 0.235 | 0.974 | 2.81 | 0.699 | 0.961 |
| SPI | 6 mo (180d) | 3,855,984 | +0.048 | 0.133 | 0.194 | 0.984 | 2.72 | 0.730 | 0.975 |
| SPI | 12 mo (1y) | 3,855,984 | +0.049 | 0.119 | 0.171 | 0.990 | 1.82 | 0.743 | 0.984 |
| SPEI | 1 mo (30d) | 3,848,681 | +0.070 | 0.226 | 0.354 | 0.934 | 3.21 | 0.637 | 0.909 |
| SPEI | 3 mo (90d) | 3,848,681 | +0.062 | 0.153 | 0.224 | 0.978 | 2.57 | 0.703 | 0.962 |
| SPEI | 6 mo (180d) | 3,848,681 | +0.058 | 0.130 | 0.188 | 0.986 | 1.96 | 0.737 | 0.975 |
| SPEI | 12 mo (1y) | 3,855,984 | +0.062 | 0.115 | 0.164 | 0.991 | 1.65 | 0.755 | 0.988 |
| EDDI | 1 mo (30d) | 3,848,681 | −0.023 | 0.175 | 0.277 | 0.958 | 4.18 | 0.712 | 0.944 |
| EDDI | 3 mo (90d) | 3,848,681 | −0.021 | 0.108 | 0.171 | 0.984 | 4.18 | 0.801 | 0.985 |
| EDDI | 6 mo (180d) | 3,855,984 | −0.014 | 0.087 | 0.145 | 0.988 | 3.37 | 0.845 | 0.995 |
| EDDI | 12 mo (1y) | 3,848,681 | −0.026 | 0.072 | 0.116 | 0.993 | 2.38 | 0.861 | 0.997 |

Agreement improves monotonically with scale for all three indices. Correlation
of 0.93 at one month is what a different accumulation window, a five-day
sampling offset, and a different plotting position produce on the same
underlying weather; correlation of 0.99 at twelve months means the annual
accumulation removes most of that noise.

Sixteen difference maps are in `docs/research/earthengine-pilot-results/`
(`difference-<index>_<scale>.png`). The one-month maps show a weak gradient
across the basin rather than isolated artifacts; the twelve-month maps show
near-zero structure, which is consistent with the window mismatch being the
dominant cause.

### The bias decomposes

Subtract the closed-form +0.0538 plotting-position offset from each bias:

| Index | bias at 1 mo | residual after −0.0538 | bias at 12 mo | residual after −0.0538 |
| --- | --- | --- | --- | --- |
| SPI | +0.061 | +0.008 | +0.049 | −0.005 |
| SPEI | +0.070 | +0.016 | +0.062 | +0.008 |
| EDDI | −0.023 | **−0.076** | −0.026 | **−0.080** |

For SPI and SPEI the residual is at most 0.016, i.e. the data agree and the
convention difference explains the bias. For EDDI, where both sides already use
the Tukey convention, the residual of about −0.08 is a genuine difference in
method or input, not an artifact of ranks.

### The EDDI residual is not a reference-crop problem

The catalog does not say whether its EDDI uses grass (`eto`) or alfalfa (`etr`)
reference evapotranspiration. Reading both costs one extra band, so the run
computes both:

| EDDI input | bias at 1 mo | MAE at 1 mo | r at 1 mo | MAE at 12 mo | r at 12 mo |
| --- | --- | --- | --- | --- | --- |
| `eto`, grass | −0.023 | 0.175 | 0.958 | 0.072 | 0.993 |
| `etr`, alfalfa | −0.043 | 0.211 | 0.948 | 0.140 | 0.978 |

Alfalfa is worse at every scale, so the hypothesis that the gap comes from the
reference crop is not supported. Grass reference ET matches better. The
remaining −0.08 is unexplained by anything measured here and is listed as an
open question.

## Results, reference B

Not run. `CE_API_KEY` is unset on the machine that produced these numbers, so
the script records `reference_b.status = "pending"`. The request it will send is
fixed in `climate_engine_series()` and covered by an offline test: dataset
`GRIDMET`, `distribution=nonparametric`, `start_year=1981`, `end_year=2016`,
`accumulation` 30/90/180/365 days, three points in the basin, key in the
`Authorization` header only.

Reference B will be the more informative comparison once the key exists,
because its `distribution=nonparametric` option is the same method family as
ours, and its `accumulation` is expressed in days like reference A's windows.

## Interpretation

1. **The implementations agree on the underlying data.** On identical inputs,
   SPI and SPEI correlate at 0.93 to 0.99 with the catalog product across four
   scales, and the bias is explained entirely by one documented convention
   difference plus a window difference.
2. **The library carries two empirical conventions.** `eddi()` uses Tukey's
   symmetric plotting position; `Distribution.empirical` uses the SEI
   convention. Both are documented and tested, and they differ by a constant.
   A user comparing EDDI against SPI from this library is comparing across two
   conventions without being told.
3. **Category agreement is the number users care about and the weakest one.**
   Same-class agreement at one month is 0.64, and 0.91 within one class. A
   one-class disagreement is normal for two products that differ in window and
   plotting position; a two-class disagreement is not, and it occurs for about
   9% of one-month pairs.
4. **Nothing here needs a plausible-sounding excuse.** The residual biases are
   small enough to attribute, or, for EDDI, small enough to list as open rather
   than explain away.

## Limitations

- **Reference A is not an independent method.** It is UC Merced's product, and
  this library's EDDI follows the same NOAA PSL lineage. Agreement on EDDI is
  expected and is not evidence of correctness.
- **One region, one period.** The ACF basin is humid and subtropical. A dry
  basin would weight the tails differently, and the empirical ceiling binds
  harder where values are extreme.
- **Pooled metrics hide spatial structure.** `n` of 3.85 million is a large
  sample of a small, correlated field. The effective sample size is closer to
  the number of independent months than to the number of cell-months, so
  treat the metrics as descriptive, not as significance tests.
- **The empirical ceiling is load-bearing.** With a 36-value sample, this
  library's SPI and SPEI cannot exceed +1.94 or fall below −1.62 inside the
  calibration sample, while reference A reaches ±2.09. Extreme droughts are
  therefore compared through a compressed scale, which inflates apparent
  category disagreement exactly where users look most closely.
- **The plotting-position inference is inference.** The ±2.090 endpoint of
  reference A matches Tukey at n = 36 to four decimals, and Climate Engine
  documents Tukey plotting positions, but the catalog page does not name the
  formula. Treated as an open question, not a fact.
- **`eto` versus `etr` remains unresolved for the reference.** The sensitivity
  run only shows which of the two matches better; it does not identify what the
  producer used.

## Open questions for the Climate Engine team

1. Which plotting position does `GRIDMET/DROUGHT` and the Climate Engine
   `standard_index` endpoint use: `(k + 1) / (n + 2)`, `k / (n + 1)`, or Tukey
   `(k − 1/3) / (n + 1/3)`? The three differ by a constant offset of up to 0.054
   in z for a 36-year baseline, which is the size of the bias measured here.
2. Is standardization applied within each calendar month (12 strata, as this
   library does), within each pentad of the year (73 strata), or pooled?
3. Does the pentad labelled 30 January cover the five days ending that day or
   starting it? This report assumes ending.
4. Which reference crop does the catalog's EDDI use, grass or alfalfa? The
   sensitivity here says grass, but it cannot confirm.
5. Is there a documented reason for a 1981–2016 baseline rather than the full
   record? It fixes the sample at 36 values per stratum, which caps the
   achievable index magnitude.
6. Would the team accept a contribution of `climate_indices` code as a
   verification path for either product, and if so, which plotting position
   should a shared method note standardize on?

## Reproduction

```bash
# once per machine
uv run --extra earthengine earthengine authenticate
export EE_PROJECT=<a project registered for Earth Engine>

# every time; writes the JSON and the sixteen maps
uv run --extra earthengine scripts/earthengine_climate_engine_pilot.py

# reference B, once a key exists
CE_API_KEY=<key> uv run --extra earthengine scripts/earthengine_climate_engine_pilot.py
```

The run takes about two minutes and reads 540 monthly images plus 528 pentads.
Offline tests cover everything that does not need a network:
`uv run pytest tests/test_earthengine_pilot_glue.py`.

## Proposed `VALIDATION.md` entry

Evidence level: **characterization comparison**, in the vocabulary
`VALIDATION.md` already uses. It is not reference reproduction: the reference is
a different implementation with a different accumulation window and a different
plotting position, and no existing status is upgraded by this pilot.

The proposed text is in the "Cross-provider characterization" section of
[`VALIDATION.md`](../../VALIDATION.md).
