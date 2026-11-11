# Earth Engine and Climate Engine cross-validation pilot

Both references have run. This internal report records a characterization comparison, not reference reproduction.
The [setup guide](../earthengine-climate-engine.md) gives the authentication instructions.
No existing validation status changes.

## Purpose

Measure agreement between `climate_indices` and two provider products that use gridMET inputs.
Keep the references separate:

- **A:** the Earth Engine catalog collection `GRIDMET/DROUGHT`.
- **B:** the Climate Engine API `timeseries/standard_index/coordinates` endpoint.

These are separate products, not proven independent implementations.
The pilot does not establish that their input series or methods are identical.

## What ran

| Item | Value |
| --- | --- |
| Library | `climate_indices` 3.2.0, branch `feature/earthengine-integration` |
| Reader | `xee` 0.1.2, `earthengine-api` 1.7.52, Python 3.14.7 |
| Inputs | `IDAHO_EPSCOR/GRIDMET`, bands `pr`, `eto`, `etr`, daily totals in mm |
| Input years | 1980–2024; 1980 supplies the initial aggregation windows |
| Reference A | `GRIDMET/DROUGHT`, one selected pentad per month, 528 images |
| Reference B | `POST https://api.climateengine.org/timeseries/standard_index/coordinates` |
| API points, longitude/latitude | (-84.9, 30.7), (-84.4, 33.0), (-84.0, 34.5) |
| Region | ACF basin bounding rectangle, 29.5–35.0 N, 85.7–83.3 W; not a basin polygon |
| Grid | Native gridMET lattice: 58 by 133 pixels, 7,714 cells, approximately 4,638 m |
| Cells with data | 7,303 |
| Calibration Period | 1981–2016 |
| Comparison years | 1981–2024, 528 months |
| Acquisition date | 2026-11-10 |

The script checks that both Earth Engine collections have the same native grid.
It crops to pixel edges without changing the pixel spacing.
For B, it selects the nearest grid cell to each requested point.
The JSON records those cell coordinates and each API request body.
The provider's spatial reducer is `mean`; pixel-selection equivalence remains unverified.

`docs/research/earthengine-pilot-results/pilot-metrics.json` records the library version, source commit, and dirty flag.
The final run records source commit `544e98b6cc0772db9f7650209e978ac4c1be2046` and `git_dirty=false`.
The corrected Earth Engine filter includes 2024-12-31; earlier runs omitted that final input day.
The collection IDs identify live products, not immutable snapshots.
The JSON therefore records each asset's Earth Engine revision and a SHA-256 over the exact time steps and values read, so a later provider update is detectable.

## Method alignment

| Choice | `climate_indices` | Reference A | Reference B |
| --- | --- | --- | --- |
| Input family | gridMET `pr`; `eto` for primary SPEI/EDDI | gridMET; exact ET crop unresolved | `dataset=GRIDMET`; exact input equivalence unresolved |
| Distribution | empirical/rank based | nonparametric plotting positions | `distribution=nonparametric` |
| SPI/SPEI position | SEI `(k + 1) / (n + 2)` | endpoints consistent with Tukey; formula unconfirmed | documentation names Tukey; exact implementation unverified |
| EDDI position | Tukey `(k − 1/3) / (n + 1/3)` | formula unconfirmed | exact implementation unverified |
| Calibration Period | 1981–2016 | catalog documents 1981–2016 | request sets 1981–2016 |
| Calendar grouping | each calendar month | unresolved | unresolved |
| Aggregation | 1/3/6/12 calendar months | 30d/90d/180d/1y | 30/90/180/365 days |
| Sampling | month end | last pentad on or before month end | last daily record at most four days before month end |
| Spatial support | every cropped cell for A; three nearest cells for B | cropped field | three requested points |

The API's `accumulation` parameter uses days for the daily GRIDMET dataset.
Daily output does not remove the fixed-day versus calendar-month window difference.
For example, three calendar months can contain 89–92 days rather than exactly 90.
Reference A's chosen timestamps precede month end by 0–4 days, with a mean of 1.83 days.
The B response observed during notebook execution ends on 2024-12-30 despite the requested end date of 2024-12-31.
Thus the sampling rule can retain a date offset; it does not guarantee exact month-end observations.

### Rank-convention illustration

For 36 distinct calibration ranks, SEI normal quantiles have a discrete mean of approximately +0.0538.
Tukey normal quantiles have a discrete mean of zero.
The SEI endpoints are approximately −1.620 and +1.938; Tukey's are approximately ±2.089.
This calculation reproduces those figures:

```bash
uv run python -c '
import numpy as np
from scipy.stats import norm
k = np.arange(1, 37)
for p in ((k + 1) / 38, (k - 1/3) / (36 + 1/3)):
    z = norm.ppf(p)
    print(z.min(), z.max(), z.mean())
'
```

These are exact-normal calculations for distinct ranks, not a measured decomposition of the provider differences.
The difference depends on rank, so +0.0538 is not a constant correction for individual observations.
Ties, incomplete calibration samples, inverse-normal approximations, and observations outside the Calibration Period can alter the comparison.
No SEI correction applies to EDDI, which already uses Tukey's position here.

## Results, reference A

Bias is `mean(ours − reference)` over finite pairs.
The category metrics use the catalog's eleven drought-to-wet classes, including the D0–D4 thresholds.
They are index-category comparisons, not agreement with an observed USDM map.

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
| EDDI | 1 mo (30d) | 3,848,681 | −0.022 | 0.175 | 0.277 | 0.958 | 4.18 | 0.712 | 0.944 |
| EDDI | 3 mo (90d) | 3,848,681 | −0.021 | 0.108 | 0.170 | 0.984 | 4.18 | 0.802 | 0.985 |
| EDDI | 6 mo (180d) | 3,855,984 | −0.014 | 0.087 | 0.145 | 0.988 | 3.37 | 0.845 | 0.994 |
| EDDI | 12 mo (1y) | 3,848,681 | −0.026 | 0.072 | 0.116 | 0.993 | 2.38 | 0.861 | 0.997 |

Correlation increases and MAE decreases with timescale for this sample.
The pilot does not isolate the causes of that pattern.
The sixteen `difference-<index>_<scale>.png` files show mean signed differences, including the alfalfa-ET sensitivity.
Positive and negative monthly errors can cancel in these maps; they do not show maximum monthly errors.
One comparison month is missing for SPEI 1, 3 and 6 months and for EDDI 1, 3 and 12 months.
Those rows pair 7,303 cells with 527 months, not 528; the saved metrics do not record which month.

### Reference-crop sensitivity

| EDDI input | bias at 1 mo | MAE at 1 mo | r at 1 mo | MAE at 12 mo | r at 12 mo |
| --- | --- | --- | --- | --- | --- |
| `eto`, grass | −0.022 | 0.175 | 0.958 | 0.072 | 0.993 |
| `etr`, alfalfa | −0.042 | 0.211 | 0.948 | 0.140 | 0.978 |

Grass reference ET matches more closely in this pilot.
This sensitivity does not identify the reference producer's crop or explain the remaining bias.

## Results, reference B

The API returns a daily series for each point.
The script samples those series at month ends under the bounded rule above.
Each row contains 1,584 finite pairs: three points and 528 months.

| Index | Scale | n | bias | MAE | RMSE | r | max abs | same class | within one class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPI | 1 mo (30d) | 1,584 | −0.055 | 0.148 | 0.223 | 0.978 | 1.61 | 0.697 | 0.971 |
| SPI | 3 mo (90d) | 1,584 | −0.063 | 0.130 | 0.195 | 0.984 | 1.41 | 0.723 | 0.973 |
| SPI | 6 mo (180d) | 1,584 | −0.066 | 0.137 | 0.193 | 0.984 | 1.05 | 0.715 | 0.972 |
| SPI | 12 mo (365d) | 1,584 | −0.069 | 0.095 | 0.146 | 0.994 | 0.84 | 0.771 | 0.993 |
| SPEI | 1 mo (30d) | 1,584 | −0.052 | 0.146 | 0.221 | 0.978 | 1.44 | 0.699 | 0.968 |
| SPEI | 3 mo (90d) | 1,584 | −0.055 | 0.125 | 0.184 | 0.985 | 1.14 | 0.747 | 0.979 |
| SPEI | 6 mo (180d) | 1,584 | −0.056 | 0.125 | 0.177 | 0.986 | 0.84 | 0.737 | 0.984 |
| SPEI | 12 mo (365d) | 1,584 | −0.055 | 0.091 | 0.139 | 0.994 | 0.68 | 0.782 | 0.994 |
| EDDI | 1 mo (30d) | 1,584 | −0.084 | 0.137 | 0.200 | 0.982 | 1.40 | 0.717 | 0.983 |
| EDDI | 3 mo (90d) | 1,584 | −0.085 | 0.126 | 0.180 | 0.986 | 1.00 | 0.732 | 0.986 |
| EDDI | 6 mo (180d) | 1,584 | −0.083 | 0.125 | 0.180 | 0.987 | 1.00 | 0.739 | 0.990 |
| EDDI | 12 mo (365d) | 1,584 | −0.087 | 0.111 | 0.157 | 0.991 | 1.00 | 0.754 | 0.996 |

SPI-3 per-point correlations are approximately 0.976, 0.984, and 0.991.
The B biases are negative for every index and timescale.
SPI and SPEI biases therefore have the opposite sign from the pooled A results.
The cause remains unresolved.

A and B use different spatial samples here.
Their pooled metrics cannot establish a provider ranking or a direct disagreement between the references.
The fixed-day windows, calendar grouping, plotting positions, and spatial sampling require further alignment.
The script does not tune any configuration to reduce these differences.

## Interpretation and limitations

- This is a characterization comparison, not a correctness proof or reference reproduction.
- The region is one humid southeastern rectangle, not the basin polygon or CONUS.
- The references share gridMET inputs and may share method lineage; their independence is not established.
- Approximately 3.85 million cell-month pairs are spatially and temporally correlated, not independent samples.
- The B comparison covers three points and does not measure a spatial error field.
- A rank-convention difference is plausible, but this pilot does not prove a causal bias decomposition.
- The documented Tukey family does not establish identical tie handling, calibration grouping, or inverse-normal implementation.
- Data-source identity does not establish byte-identical inputs, especially across live services.
- The notebook's gamma SPI comparison demonstrates data access, not method-matched nonparametric reproduction.
- The data terms remain provider-specific. No raw Climate Engine response or credential is committed.

## Open questions for the Climate Engine team

1. Which exact plotting position and tie rule do the catalog and API products use?
2. Do they standardize within calendar days, pentads, months, or another grouping?
3. Does each catalog timestamp mark the start or end of its pentad window?
4. Which reference crop supplies SPEI and EDDI evaporative demand?
5. Do the API's calibration years select the same calibration samples as the catalog's 1981–2016 window?
6. Why does the API return no 2024-12-31 standard-index observation for this requested end date?
7. Which grid cell or spatial footprint does the point reducer select?
8. What explains the negative B bias after these settings are aligned?
9. Can the team provide immutable asset versions, citation requirements, and permission to redistribute a small paired fixture?
10. Would a shared method note or verification script help the collaboration?

## Reproduction

```bash
# Authenticate once on the machine.
uv run --extra earthengine earthengine authenticate
export EE_PROJECT=<a project registered for Earth Engine>

# Load CE_API_KEY from your local environment without printing it.
uv run --extra earthengine scripts/earthengine_climate_engine_pilot.py

# Run the offline tests without credentials.
uv run pytest tests/test_earthengine_pilot_glue.py
```

The pilot writes the metrics JSON and sixteen maps.
Without `CE_API_KEY`, it records B as pending and runs only A.
Runtime depends on provider latency and quotas; earlier complete runs took several minutes.
The API key travels only in the authorization header.

## Validation entry

The [cross-provider section in `VALIDATION.md`](../../VALIDATION.md) records characterization evidence only.
The script, offline tests, notebooks, and saved metrics provide the reproduction path.
No core-library API changes or validation-status upgrades are part of this pilot.
