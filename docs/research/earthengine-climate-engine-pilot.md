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
Asset revision dates and input checksums were not captured, so later provider updates can change the results.

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

## Follow-up 1: SPI and SPEI with Tukey plotting positions

### Pre-registration

[Issue #1571](https://github.com/monocongo/climate_indices/issues/1571) records the predictions before the experiment.
The maintainer approved the plan before implementation.
The hypothesis was that the SEI-versus-Tukey difference explains most of the SPI/SPEI bias against reference A.
Matching a convention that the provider documents is method alignment, not tuning.

The registered predictions and rejection criteria were:

- **A, SPI/SPEI, all four timescales:** bias falls by approximately 0.05 and finishes within ±0.02 of zero.
  Correlation barely changes. MAE and RMSE fall modestly.
- **A, 1981–2016:** the reduction approaches the exact distinct-rank mean difference for the calibration sample size.
  For 36 values, the formula predicts **0.053831430857**, not an estimate from the climate data.
- **A, 2017–2024, secondary prediction:** the reduction can differ or reverse because endpoint behavior differs.
  Exact Tukey endpoints are approximately ±2.089, versus SEI's ±1.938 outside the calibration range.
- **B, SPI/SPEI:** bias moves further negative, to approximately −0.10 to −0.12.
  That outcome would not reject the hypothesis; it would strengthen the separate concern about B's negative bias.
- **EDDI:** every metric remains unchanged. Any change requires a stop and an investigation.
- **Reject the hypothesis if** A SPI/SPEI bias remains above +0.03 at most timescales, or the reduction is much smaller than predicted.

For sample size `n`, the expected reduction is the mean of
`norm.ppf((k + 1) / (n + 2)) - norm.ppf((k - 1/3) / (n + 1/3))`, for `k = 1, ..., n`.
The metrics preserve this formula-based expectation for sample sizes 10–36.
It is a distinct-rank calibration expectation, not a constant correction for individual values.

### Convention audit and implementation

A plotting position converts a rank into a cumulative probability.
The inverse-normal transform converts that probability into a standard-normal quantile.
The pilot uses exact thirds for Tukey and the same exact inverse-normal transform as the library's empirical SEI path.
No library behavior, default, signature, or public API changed.

| Rule | Library SEI path | Pilot Tukey path | Library EDDI path |
| --- | --- | --- | --- |
| Rank count | `count(calibration <= value)` | `count(calibration < value) + 1`, capped at `n` | Strict-below count, plus leading scale pads |
| Ties | Maximum rank | Minimum rank | Minimum rank |
| Position | `(k + 1) / (n + 2)` | `(k - 1/3) / (n + 1/3)` | `(pads + below + 0.66) / (valid + pads + 0.33)` |
| Below minimum | `1 / (n + 2)` | Lowest in-range position | Lowest position, with pad adjustment |
| Above maximum | `(n + 1) / (n + 2)` | Highest in-range position | Probability can exceed one before clipping |
| Missing calibration values | Excluded; fewer than ten valid values gives NaN | Same ten-value rule | Excluded; fewer than two gives NaN |
| Zeros | Ordinary ranks; no separate zero mass | Same | Ordinary ranks after preparation |
| Negatives before aggregation | SPI clips; this SPEI path preserves the sign | Same | Clips to zero |
| Calendar grouping | Each calendar month and cell separately | Same | Same |
| Inverse normal | `scipy.special.ndtri` | Same | Hastings approximation; probability and output clipping |

EDDI's leading scale pads enter its rank count if the Calibration Period contains those pads.
Here, the pads occur in 1980, outside the 1981–2016 Calibration Period.
All three paths preserve missing observations; an aggregation window containing a missing value remains missing.
The pilot helper reuses `compute.prepare_scaled` for aggregation and grouping.
It cannot reuse EDDI unchanged because of the signed SPEI input, rounded constants, minimum sample size, and endpoint behavior.

EDDI uses NOAA's rounded `0.66` and `0.33`, not exact thirds.
For 36 distinct ranks, those rounded constants give an exact-normal mean of −0.000682, rather than zero.
Offline tests compare shared calibration inputs with the library's EDDI path within 0.005 z-score units.
That tolerance includes the rounded constants and the Hastings approximation.
The native extension was absent, so the live run and equivalence check used the Python implementation.

### Acquisition and controls

The completed reference A run used source commit `abafe444b92daa18de4964ecbb83253e377eed75`, with `git_dirty=false`, on 2026-11-11.
It kept the original inputs, region, grid, periods, sampling rules, and metrics.
The script loaded each input array once and computed both conventions from those arrays.
It also loaded the reference array once.
Every original A metric, including both EDDI variants, matched #1569 within an absolute tolerance of `1e-10`.

The new results occupy `tukey_followup_1571` in `earthengine-pilot-results/pilot-metrics.json`.
Every original top-level result remains unchanged.
The original difference maps still describe the SEI run; this follow-up does not replace them.
The JSON also records input-array SHA-256 hashes and the calibration rank audit.

`earthengine-pilot-results/tukey-image-manifest.json.gz` records 16,437 daily input images and 528 selected drought images.
Every image exposed both `system:index` and `system:version`.
The collection ID and image index identify each asset; the manifest also preserves the returned image ID.
The metrics record the compressed manifest's checksum.
These acquisition-time records do not establish immutable snapshots or byte identity with the earlier run, which captured no input checksums.

The first paired attempt timed out during the Climate Engine read, whose configured HTTP timeout was 180 seconds.
The completed rerun used `--skip-climate-engine` and retained A only.
Both credentials were present; neither credential nor a raw Climate Engine response entered the outputs.
No new B metric or B control check is available.

### Reading the comparison tables

Each metric cell contains **SEI / Tukey / change**, where change is Tukey minus SEI.
The full-period SEI metrics reproduce #1569; the period splits come from this run's shared arrays.
Both conventions use the same finite pairs within each row.
Class agreement remains a fraction, not a percentage or agreement with an observed USDM map.

### Reference A, 1981-2024

| Index | Timescale | n | Bias | MAE | RMSE | r | Max abs | Same class | Within one class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPI | 1 mo | 3,855,984 | +0.061 / +0.033 / -0.0287 | 0.230 / 0.224 / -0.0057 | 0.369 / 0.375 / +0.0064 | 0.928 / 0.928 / -0.0001 | 3.502 / 3.776 / +0.2741 | 0.638 / 0.660 / +0.0224 | 0.906 / 0.903 / -0.0034 |
| SPI | 3 mo | 3,855,984 | +0.054 / +0.026 / -0.0273 | 0.157 / 0.137 / -0.0193 | 0.235 / 0.223 / -0.0114 | 0.974 / 0.974 / -0.0000 | 2.806 / 2.801 / -0.0055 | 0.699 / 0.751 / +0.0519 | 0.961 / 0.963 / +0.0025 |
| SPI | 6 mo | 3,855,984 | +0.048 / +0.023 / -0.0249 | 0.133 / 0.109 / -0.0239 | 0.194 / 0.177 / -0.0176 | 0.984 / 0.984 / -0.0006 | 2.724 / 2.715 / -0.0089 | 0.730 / 0.790 / +0.0607 | 0.975 / 0.978 / +0.0036 |
| SPI | 12 mo | 3,855,984 | +0.049 / +0.026 / -0.0222 | 0.119 / 0.086 / -0.0325 | 0.171 / 0.151 / -0.0201 | 0.990 / 0.989 / -0.0011 | 1.823 / 1.919 / +0.0961 | 0.743 / 0.831 / +0.0882 | 0.984 / 0.985 / +0.0004 |
| SPEI | 1 mo | 3,848,681 | +0.070 / +0.040 / -0.0299 | 0.226 / 0.215 / -0.0105 | 0.354 / 0.356 / +0.0016 | 0.934 / 0.934 / -0.0000 | 3.209 / 3.369 / +0.1602 | 0.637 / 0.669 / +0.0316 | 0.909 / 0.909 / -0.0002 |
| SPEI | 3 mo | 3,848,681 | +0.062 / +0.032 / -0.0298 | 0.153 / 0.130 / -0.0234 | 0.224 / 0.207 / -0.0174 | 0.978 / 0.978 / -0.0001 | 2.570 / 2.555 / -0.0149 | 0.703 / 0.762 / +0.0596 | 0.962 / 0.969 / +0.0067 |
| SPEI | 6 mo | 3,848,681 | +0.058 / +0.029 / -0.0284 | 0.130 / 0.103 / -0.0271 | 0.188 / 0.165 / -0.0231 | 0.986 / 0.986 / -0.0005 | 1.958 / 1.917 / -0.0410 | 0.737 / 0.802 / +0.0652 | 0.975 / 0.982 / +0.0068 |
| SPEI | 12 mo | 3,855,984 | +0.062 / +0.035 / -0.0274 | 0.115 / 0.080 / -0.0348 | 0.164 / 0.138 / -0.0257 | 0.991 / 0.990 / -0.0008 | 1.650 / 2.119 / +0.4693 | 0.755 / 0.847 / +0.0917 | 0.988 / 0.989 / +0.0012 |
| EDDI | 1 mo | 3,848,681 | -0.022 / -0.022 / +0.0000 | 0.175 / 0.175 / +0.0000 | 0.277 / 0.277 / +0.0000 | 0.958 / 0.958 / +0.0000 | 4.184 / 4.184 / +0.0000 | 0.712 / 0.712 / +0.0000 | 0.944 / 0.944 / +0.0000 |
| EDDI | 3 mo | 3,848,681 | -0.021 / -0.021 / +0.0000 | 0.108 / 0.108 / +0.0000 | 0.170 / 0.170 / +0.0000 | 0.984 / 0.984 / +0.0000 | 4.184 / 4.184 / +0.0000 | 0.802 / 0.802 / +0.0000 | 0.985 / 0.985 / +0.0000 |
| EDDI | 6 mo | 3,855,984 | -0.014 / -0.014 / +0.0000 | 0.087 / 0.087 / +0.0000 | 0.145 / 0.145 / +0.0000 | 0.988 / 0.988 / +0.0000 | 3.368 / 3.368 / +0.0000 | 0.845 / 0.845 / +0.0000 | 0.994 / 0.994 / +0.0000 |
| EDDI | 12 mo | 3,848,681 | -0.026 / -0.026 / +0.0000 | 0.072 / 0.072 / +0.0000 | 0.116 / 0.116 / +0.0000 | 0.993 / 0.993 / +0.0000 | 2.380 / 2.380 / +0.0000 | 0.861 / 0.861 / +0.0000 | 0.997 / 0.997 / +0.0000 |
| EDDI (alfalfa) | 1 mo | 3,848,681 | -0.042 / -0.042 / +0.0000 | 0.211 / 0.211 / +0.0000 | 0.309 / 0.309 / +0.0000 | 0.948 / 0.948 / +0.0000 | 4.184 / 4.184 / +0.0000 | 0.658 / 0.658 / +0.0000 | 0.925 / 0.925 / +0.0000 |
| EDDI (alfalfa) | 3 mo | 3,848,681 | -0.047 / -0.047 / +0.0000 | 0.151 / 0.151 / +0.0000 | 0.217 / 0.217 / +0.0000 | 0.974 / 0.974 / +0.0000 | 4.184 / 4.184 / +0.0000 | 0.730 / 0.730 / +0.0000 | 0.969 / 0.969 / +0.0000 |
| EDDI (alfalfa) | 6 mo | 3,855,984 | -0.045 / -0.045 / +0.0000 | 0.140 / 0.140 / +0.0000 | 0.206 / 0.206 / +0.0000 | 0.977 / 0.977 / +0.0000 | 3.543 / 3.543 / +0.0000 | 0.756 / 0.756 / +0.0000 | 0.980 / 0.980 / +0.0000 |
| EDDI (alfalfa) | 12 mo | 3,848,681 | -0.064 / -0.064 / +0.0000 | 0.140 / 0.140 / +0.0000 | 0.205 / 0.205 / +0.0000 | 0.978 / 0.978 / +0.0000 | 2.489 / 2.489 / +0.0000 | 0.754 / 0.754 / +0.0000 | 0.977 / 0.977 / +0.0000 |

### Reference A, 1981-2016

| Index | Timescale | n | Bias | MAE | RMSE | r | Max abs | Same class | Within one class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPI | 1 mo | 3,154,896 | +0.068 / +0.014 / -0.0538 | 0.226 / 0.212 / -0.0135 | 0.366 / 0.364 / -0.0017 | 0.927 / 0.928 / +0.0008 | 3.209 / 3.366 / +0.1574 | 0.648 / 0.681 / +0.0328 | 0.911 / 0.913 / +0.0021 |
| SPI | 3 mo | 3,154,896 | +0.065 / +0.011 / -0.0538 | 0.156 / 0.129 / -0.0278 | 0.235 / 0.214 / -0.0211 | 0.974 / 0.975 / +0.0009 | 2.806 / 2.801 / -0.0055 | 0.704 / 0.768 / +0.0641 | 0.961 / 0.968 / +0.0073 |
| SPI | 6 mo | 3,154,896 | +0.058 / +0.004 / -0.0538 | 0.134 / 0.100 / -0.0340 | 0.197 / 0.165 / -0.0316 | 0.985 / 0.985 / +0.0007 | 2.724 / 2.715 / -0.0089 | 0.732 / 0.810 / +0.0784 | 0.973 / 0.981 / +0.0081 |
| SPI | 12 mo | 3,154,896 | +0.057 / +0.003 / -0.0538 | 0.119 / 0.071 / -0.0478 | 0.171 / 0.130 / -0.0412 | 0.990 / 0.991 / +0.0009 | 1.823 / 1.774 / -0.0486 | 0.746 / 0.865 / +0.1184 | 0.984 / 0.990 / +0.0054 |
| SPEI | 1 mo | 3,154,896 | +0.074 / +0.020 / -0.0538 | 0.221 / 0.202 / -0.0189 | 0.351 / 0.344 / -0.0068 | 0.934 / 0.936 / +0.0013 | 3.209 / 3.366 / +0.1574 | 0.646 / 0.690 / +0.0437 | 0.913 / 0.919 / +0.0058 |
| SPEI | 3 mo | 3,154,896 | +0.071 / +0.017 / -0.0538 | 0.152 / 0.120 / -0.0324 | 0.223 / 0.196 / -0.0278 | 0.978 / 0.979 / +0.0009 | 2.570 / 2.555 / -0.0149 | 0.705 / 0.778 / +0.0735 | 0.962 / 0.973 / +0.0115 |
| SPEI | 6 mo | 3,154,896 | +0.064 / +0.010 / -0.0538 | 0.132 / 0.093 / -0.0387 | 0.191 / 0.153 / -0.0381 | 0.987 / 0.987 / +0.0009 | 1.958 / 1.917 / -0.0410 | 0.734 / 0.821 / +0.0876 | 0.974 / 0.985 / +0.0114 |
| SPEI | 12 mo | 3,154,896 | +0.066 / +0.012 / -0.0538 | 0.115 / 0.064 / -0.0517 | 0.164 / 0.115 / -0.0482 | 0.992 / 0.993 / +0.0012 | 1.535 / 1.549 / +0.0140 | 0.752 / 0.880 / +0.1278 | 0.988 / 0.993 / +0.0052 |
| EDDI | 1 mo | 3,154,896 | -0.039 / -0.039 / +0.0000 | 0.159 / 0.159 / +0.0000 | 0.250 / 0.250 / +0.0000 | 0.967 / 0.967 / +0.0000 | 2.056 / 2.056 / +0.0000 | 0.727 / 0.727 / +0.0000 | 0.953 / 0.953 / +0.0000 |
| EDDI | 3 mo | 3,154,896 | -0.035 / -0.035 / +0.0000 | 0.096 / 0.096 / +0.0000 | 0.145 / 0.145 / +0.0000 | 0.989 / 0.989 / +0.0000 | 1.379 / 1.379 / +0.0000 | 0.815 / 0.815 / +0.0000 | 0.989 / 0.989 / +0.0000 |
| EDDI | 6 mo | 3,154,896 | -0.028 / -0.028 / +0.0000 | 0.074 / 0.074 / +0.0000 | 0.116 / 0.116 / +0.0000 | 0.993 / 0.993 / +0.0000 | 1.187 / 1.187 / +0.0000 | 0.860 / 0.860 / +0.0000 | 0.996 / 0.996 / +0.0000 |
| EDDI | 12 mo | 3,154,896 | -0.037 / -0.037 / +0.0000 | 0.062 / 0.062 / +0.0000 | 0.095 / 0.095 / +0.0000 | 0.996 / 0.996 / +0.0000 | 1.084 / 1.084 / +0.0000 | 0.879 / 0.879 / +0.0000 | 0.998 / 0.998 / +0.0000 |
| EDDI (alfalfa) | 1 mo | 3,154,896 | -0.039 / -0.039 / +0.0000 | 0.197 / 0.197 / +0.0000 | 0.284 / 0.284 / +0.0000 | 0.957 / 0.957 / +0.0000 | 2.194 / 2.194 / +0.0000 | 0.667 / 0.667 / +0.0000 | 0.933 / 0.933 / +0.0000 |
| EDDI (alfalfa) | 3 mo | 3,154,896 | -0.035 / -0.035 / +0.0000 | 0.138 / 0.138 / +0.0000 | 0.198 / 0.198 / +0.0000 | 0.980 / 0.980 / +0.0000 | 1.384 / 1.384 / +0.0000 | 0.741 / 0.741 / +0.0000 | 0.973 / 0.973 / +0.0000 |
| EDDI (alfalfa) | 6 mo | 3,154,896 | -0.028 / -0.028 / +0.0000 | 0.123 / 0.123 / +0.0000 | 0.179 / 0.179 / +0.0000 | 0.983 / 0.983 / +0.0000 | 1.449 / 1.449 / +0.0000 | 0.773 / 0.773 / +0.0000 | 0.983 / 0.983 / +0.0000 |
| EDDI (alfalfa) | 12 mo | 3,154,896 | -0.037 / -0.037 / +0.0000 | 0.119 / 0.119 / +0.0000 | 0.174 / 0.174 / +0.0000 | 0.984 / 0.984 / +0.0000 | 1.294 / 1.294 / +0.0000 | 0.780 / 0.780 / +0.0000 | 0.983 / 0.983 / +0.0000 |

### Reference A, 2017-2024

| Index | Timescale | n | Bias | MAE | RMSE | r | Max abs | Same class | Within one class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPI | 1 mo | 701,088 | +0.032 / +0.117 / +0.0847 | 0.247 / 0.276 / +0.0294 | 0.381 / 0.420 / +0.0395 | 0.931 / 0.929 / -0.0018 | 3.502 / 3.776 / +0.2741 | 0.593 / 0.568 / -0.0241 | 0.887 / 0.859 / -0.0283 |
| SPI | 3 mo | 701,088 | +0.004 / +0.097 / +0.0922 | 0.157 / 0.176 / +0.0189 | 0.234 / 0.262 / +0.0278 | 0.970 / 0.968 / -0.0023 | 2.724 / 2.801 / +0.0774 | 0.676 / 0.674 / -0.0029 | 0.961 / 0.942 / -0.0192 |
| SPI | 6 mo | 701,088 | +0.004 / +0.110 / +0.1054 | 0.127 / 0.149 / +0.0216 | 0.183 / 0.222 / +0.0380 | 0.979 / 0.976 / -0.0030 | 2.010 / 2.479 / +0.4693 | 0.720 / 0.700 / -0.0192 | 0.981 / 0.964 / -0.0169 |
| SPI | 12 mo | 701,088 | +0.010 / +0.130 / +0.1201 | 0.118 / 0.154 / +0.0364 | 0.168 / 0.220 / +0.0525 | 0.984 / 0.979 / -0.0052 | 1.698 / 1.919 / +0.2212 | 0.726 / 0.678 / -0.0477 | 0.984 / 0.961 / -0.0221 |
| SPEI | 1 mo | 693,785 | +0.052 / +0.131 / +0.0790 | 0.247 / 0.274 / +0.0273 | 0.370 / 0.406 / +0.0359 | 0.934 / 0.932 / -0.0020 | 2.989 / 3.369 / +0.3797 | 0.594 / 0.570 / -0.0238 | 0.889 / 0.861 / -0.0276 |
| SPEI | 3 mo | 693,785 | +0.023 / +0.102 / +0.0793 | 0.156 / 0.173 / +0.0173 | 0.227 / 0.251 / +0.0240 | 0.972 / 0.969 / -0.0023 | 2.090 / 2.125 / +0.0345 | 0.694 / 0.691 / -0.0035 | 0.963 / 0.948 / -0.0150 |
| SPEI | 6 mo | 693,785 | +0.028 / +0.116 / +0.0871 | 0.122 / 0.147 / +0.0258 | 0.174 / 0.212 / +0.0375 | 0.981 / 0.979 / -0.0026 | 1.754 / 1.774 / +0.0204 | 0.750 / 0.713 / -0.0369 | 0.983 / 0.968 / -0.0144 |
| SPEI | 12 mo | 701,088 | +0.046 / +0.138 / +0.0916 | 0.115 / 0.156 / +0.0413 | 0.163 / 0.211 / +0.0479 | 0.986 / 0.983 / -0.0037 | 1.650 / 2.119 / +0.4693 | 0.767 / 0.697 / -0.0704 | 0.986 / 0.970 / -0.0166 |
| EDDI | 1 mo | 693,785 | +0.055 / +0.055 / +0.0000 | 0.249 / 0.249 / +0.0000 | 0.376 / 0.376 / +0.0000 | 0.906 / 0.906 / +0.0000 | 4.184 / 4.184 / +0.0000 | 0.642 / 0.642 / +0.0000 | 0.906 / 0.906 / +0.0000 |
| EDDI | 3 mo | 693,785 | +0.046 / +0.046 / +0.0000 | 0.163 / 0.163 / +0.0000 | 0.256 / 0.256 / +0.0000 | 0.945 / 0.945 / +0.0000 | 4.184 / 4.184 / +0.0000 | 0.740 / 0.740 / +0.0000 | 0.965 / 0.965 / +0.0000 |
| EDDI | 6 mo | 701,088 | +0.049 / +0.049 / +0.0000 | 0.144 / 0.144 / +0.0000 | 0.235 / 0.235 / +0.0000 | 0.957 / 0.957 / +0.0000 | 3.368 / 3.368 / +0.0000 | 0.775 / 0.775 / +0.0000 | 0.987 / 0.987 / +0.0000 |
| EDDI | 12 mo | 693,785 | +0.024 / +0.024 / +0.0000 | 0.116 / 0.116 / +0.0000 | 0.183 / 0.183 / +0.0000 | 0.965 / 0.965 / +0.0000 | 2.380 / 2.380 / +0.0000 | 0.777 / 0.777 / +0.0000 | 0.990 / 0.990 / +0.0000 |
| EDDI (alfalfa) | 1 mo | 693,785 | -0.054 / -0.054 / +0.0000 | 0.277 / 0.277 / +0.0000 | 0.403 / 0.403 / +0.0000 | 0.888 / 0.888 / +0.0000 | 4.184 / 4.184 / +0.0000 | 0.616 / 0.616 / +0.0000 | 0.890 / 0.890 / +0.0000 |
| EDDI (alfalfa) | 3 mo | 693,785 | -0.099 / -0.099 / +0.0000 | 0.207 / 0.207 / +0.0000 | 0.289 / 0.289 / +0.0000 | 0.931 / 0.931 / +0.0000 | 4.184 / 4.184 / +0.0000 | 0.677 / 0.677 / +0.0000 | 0.950 / 0.950 / +0.0000 |
| EDDI (alfalfa) | 6 mo | 701,088 | -0.125 / -0.125 / +0.0000 | 0.215 / 0.215 / +0.0000 | 0.298 / 0.298 / +0.0000 | 0.939 / 0.939 / +0.0000 | 3.543 / 3.543 / +0.0000 | 0.679 / 0.679 / +0.0000 | 0.962 / 0.962 / +0.0000 |
| EDDI (alfalfa) | 12 mo | 693,785 | -0.185 / -0.185 / +0.0000 | 0.238 / 0.238 / +0.0000 | 0.310 / 0.310 / +0.0000 | 0.942 / 0.942 / +0.0000 | 2.489 / 2.489 / +0.0000 | 0.634 / 0.634 / +0.0000 | 0.950 / 0.950 / +0.0000 |

### Reference A bias by calendar month

The table gives Tukey bias for 1981–2024. The JSON also preserves the twelve SEI biases for each row.

| Index | Timescale | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SPI | 1 mo | +0.033 | +0.035 | +0.070 | +0.029 | +0.065 | +0.030 | +0.014 | +0.020 | -0.016 | +0.153 | -0.017 | -0.023 |
| SPI | 3 mo | -0.029 | -0.040 | +0.062 | +0.052 | +0.070 | +0.039 | +0.041 | +0.014 | -0.009 | +0.080 | +0.029 | +0.008 |
| SPI | 6 mo | -0.000 | -0.022 | -0.006 | +0.002 | +0.027 | +0.064 | +0.051 | +0.060 | +0.014 | +0.058 | +0.020 | +0.014 |
| SPI | 12 mo | +0.028 | +0.027 | +0.032 | +0.026 | +0.030 | +0.028 | +0.018 | +0.015 | +0.001 | +0.035 | +0.027 | +0.051 |
| SPEI | 1 mo | +0.037 | +0.050 | +0.072 | +0.028 | +0.067 | +0.036 | +0.016 | +0.028 | -0.007 | +0.168 | +0.005 | -0.022 |
| SPEI | 3 mo | -0.024 | -0.034 | +0.067 | +0.056 | +0.070 | +0.043 | +0.045 | +0.018 | -0.001 | +0.080 | +0.039 | +0.025 |
| SPEI | 6 mo | +0.005 | -0.015 | +0.001 | +0.009 | +0.033 | +0.066 | +0.053 | +0.062 | +0.022 | +0.061 | +0.026 | +0.026 |
| SPEI | 12 mo | +0.035 | +0.036 | +0.040 | +0.038 | +0.040 | +0.039 | +0.026 | +0.022 | +0.011 | +0.042 | +0.037 | +0.055 |
| EDDI | 1 mo | -0.008 | -0.069 | -0.038 | +0.006 | -0.020 | -0.016 | +0.005 | +0.022 | +0.004 | -0.126 | -0.029 | +0.005 |
| EDDI | 3 mo | +0.001 | -0.012 | -0.027 | -0.030 | -0.033 | -0.023 | -0.013 | +0.007 | +0.012 | -0.028 | -0.046 | -0.055 |
| EDDI | 6 mo | +0.003 | +0.008 | +0.003 | -0.015 | -0.033 | -0.033 | -0.022 | -0.012 | -0.006 | -0.014 | -0.017 | -0.028 |
| EDDI | 12 mo | -0.020 | -0.036 | -0.030 | -0.030 | -0.030 | -0.028 | -0.021 | -0.013 | -0.019 | -0.023 | -0.027 | -0.033 |
| EDDI (alfalfa) | 1 mo | -0.022 | -0.107 | -0.055 | -0.005 | -0.037 | -0.031 | -0.027 | -0.014 | -0.013 | -0.148 | -0.036 | -0.006 |
| EDDI (alfalfa) | 3 mo | -0.010 | -0.042 | -0.054 | -0.051 | -0.055 | -0.040 | -0.044 | -0.029 | -0.035 | -0.058 | -0.069 | -0.071 |
| EDDI (alfalfa) | 6 mo | -0.035 | -0.025 | -0.021 | -0.035 | -0.061 | -0.058 | -0.054 | -0.048 | -0.043 | -0.047 | -0.051 | -0.065 |
| EDDI (alfalfa) | 12 mo | -0.062 | -0.072 | -0.068 | -0.070 | -0.070 | -0.063 | -0.055 | -0.050 | -0.056 | -0.061 | -0.067 | -0.071 |

### Expected versus observed bias reduction

Reduction is SEI bias minus Tukey bias. A positive number means that Tukey reduced the bias.

| Index | Timescale | Expected, n = 36 | Observed, 1981–2016 | Observed, 1981–2024 | Observed, 2017–2024 |
| --- | --- | --- | --- | --- | --- |
| SPI | 1 mo | 0.053831 | 0.053837 | 0.028654 | -0.084671 |
| SPI | 3 mo | 0.053831 | 0.053831 | 0.027289 | -0.092152 |
| SPI | 6 mo | 0.053831 | 0.053831 | 0.024880 | -0.105403 |
| SPI | 12 mo | 0.053831 | 0.053831 | 0.022202 | -0.120129 |
| SPEI | 1 mo | 0.053831 | 0.053831 | 0.029880 | -0.079036 |
| SPEI | 3 mo | 0.053831 | 0.053831 | 0.029830 | -0.079311 |
| SPEI | 6 mo | 0.053831 | 0.053831 | 0.028423 | -0.087120 |
| SPEI | 12 mo | 0.053831 | 0.053831 | 0.027389 | -0.091604 |

Each populated calendar-month calibration sample contains 36 values.
For each index and timescale, there are 87,636 populated month/cell samples and 4,932 all-missing samples.
SPI-1 has 30 tied samples; SPI-3 and SPEI-1 each have one. The other five SPI/SPEI variants have none.
The largest difference from the distinct-rank expectation is approximately 0.000006, for SPI-1.
Thus ties do not explain the full-period shortfall.

The 2017–2024 bias instead increases by 0.079–0.120 under the registered strict-rank rule.
This rule affects observations between calibration ranks as well as observations outside the calibration range.
At an untied calibration value, `count(< value)` is one less than `count(<= value)`.
Between calibration values, those counts are equal.
Consequently, adding one to the strict-below count changes the calibration/post-calibration relation; an endpoint-only explanation is incomplete.
This is a mechanical property of the tested convention, not evidence of either provider's exact rank implementation.

### Reference B: pending

The B acquisition failed with a read timeout.
No new full-period, split-period, or change table is available for B.
Its negative-bias prediction and EDDI control remain untested in this follow-up.
The original B results above remain unchanged and must not be presented as a successful new run.

### Outcome against the pre-registration

| Prediction | Outcome |
| --- | --- |
| A full-period bias falls approximately 0.05 and finishes within ±0.02 | Failed. Reductions are 0.0222–0.0299; all eight residual biases exceed +0.02. |
| A calibration-period reduction matches the formula | Held. Reductions are 0.053831–0.053837. |
| Correlation barely changes | Held for the full period. Absolute changes are at most 0.00111. |
| MAE and RMSE fall modestly | Partly held. MAE falls for all eight variants. RMSE rises at one month for both indices and falls at longer timescales. |
| Post-calibration behavior can differ or reverse | Held. Bias, MAE, and RMSE increase for all eight variants in 2017–2024. |
| B bias becomes approximately −0.10 to −0.12 | Pending because acquisition failed. |
| EDDI stays unchanged | Held for A, including the alfalfa sensitivity, every period, and every saved metric. B remains pending. |

The residual bias is seasonal, not a uniform offset.
At one month, October bias is +0.153 for SPI and +0.168 for SPEI, while December bias is approximately −0.023 and −0.022.
At three months, January and February are negative, and October is approximately +0.080.
At six months, the largest positive bias occurs in June; at twelve months, it occurs in December.
These descriptive patterns do not identify their physical cause.

**Conclusion:** reject the full-period hypothesis under the pre-registered criteria.
The reduction is approximately half the predicted amount, and SPEI remains above +0.03 at three of four timescales.
The calibration-only explanation holds, but it does not transfer unchanged to the full comparison period.
No setting was adjusted after observing the results.
The conditional library-option issue was not opened because the full-period hypothesis did not hold.
This remains characterization evidence; no validation status changes.

### Reproduction of the follow-up

```bash
# This command runs both references and records B as pending if its transport fails.
uv run --extra earthengine scripts/earthengine_climate_engine_pilot.py --spi-spei-position tukey

# This is the command used for the completed reference A rerun.
uv run --extra earthengine scripts/earthengine_climate_engine_pilot.py --spi-spei-position tukey --skip-climate-engine

uv run pytest tests/test_earthengine_pilot_glue.py tests/test_earthengine_tukey_followup.py
```

The script keeps `sei` as its default and reserves `tukey` for the paired follow-up.
B's successful completion, its missing tables, and independent review remain pending.
The PR must remain a draft until the maintainer reviews the results.
