# Algorithm Reference

Formulas, computation steps, and implementation details for the indices this
library implements. For what each index is, when to use it, and how to choose
parameters, see {doc}`algorithms`.

```{contents} On this page
:backlinks: none
:local: true
```

## SPI computation

The SPI computation follows these steps:

1. **Input validation and preprocessing**

   - Accept 1-D or 2-D arrays of precipitation values (any units)
   - Clip negative precipitation values to zero
   - Handle missing data appropriately

2. **Temporal aggregation**

   - Compute sliding sums over the specified timescale (e.g., 1, 3, 6, 12, 24 months)
   - For monthly data: reshape to (years, 12)
   - For daily data: reshape to (years, 366) assuming leap year format

3. **Distribution fitting**

   - Fit scaled precipitation to a probability distribution (gamma or Pearson Type III)
   - Compute distribution parameters separately for each calendar month/day
   - Use only calibration period data for parameter estimation

   **Gamma distribution** (default):

   - Two-parameter gamma distribution with shape (α) and scale (β) parameters
   - Parameters estimated using maximum likelihood estimation
   - Suitable for most precipitation distributions

   **Pearson Type III distribution** (alternative):

   - Three-parameter distribution with location, scale, and skewness
   - Parameters estimated using L-moments method (Hosking, 1997)
   - Better for highly skewed precipitation distributions
   - Automatic fallback to gamma if fitting fails

4. **Cumulative probability transformation**

   - Transform fitted values to cumulative probabilities using the distribution's CDF
   - Account for zero precipitation values in probability calculations. A zero
     accumulation cannot be transformed by the fitted continuous distribution, so it
     carries the probability mass `p0` and the fitted value is `p = p0 + (1 − p0)·F(x)`;
     where within that mass a zero is placed is set by `zero_handling` (see below).

5. **Inverse normal transformation**

   - Apply inverse normal (Gaussian) CDF to obtain standardized values
   - This transforms probabilities to standard normal deviates
   - `output_scale="probability"` stops after step 4 and returns the fitted
     cumulative probability in [0, 1] (the probability integral transform, or
     PIT, value); `output_scale="bounded"` returns its `2p - 1` mapping in
     [-1, 1]. Neither is passed through the inverse normal transform

6. **Clipping to valid range**

   - Constrain final SPI values to [-3.09, 3.09] on the default
     `output_scale="normal"` only
   - This range represents probabilities from 0.001 to 0.999

### Zero handling

`spi()` and `standardized_index()` take a `zero_handling` argument that decides where
within the zero mass a zero accumulation scores (ADR-0015). Normal scale:

| Value | A zero scores | Property |
| --- | --- | --- |
| `"classic"` (default) | `Φ⁻¹(p0)` | The top of the zero mass; NOAA/NCEI and SPEIbase convention |
| `"center_of_mass"` | `Φ⁻¹(p0 / 2)` | Ideal probability-scale mean of 1/2 (Stagge et al., 2015) |
| `"mean_zero"` | `−φ(Φ⁻¹(p0)) / p0` | Ideal normal-scale mean of 0; the mean of `Z` conditional on `Z < Φ⁻¹(p0)` (Allen & Otero, 2024) |

`p0` is the zero fraction of the calibration period's non-missing values. Only the
zero positions move: a value with a positive accumulation keeps its classic
transform, and for Pearson Type III a trace value below 0.0005 shares a zero's
position where `0 < p0 < 1`. On `output_scale="probability"` and `"bounded"` both
non-classic modes place a zero at the centre of the mass, `p0 / 2`, mapped to the
requested scale. A mode applies only where `0 < p0 < 1`: where the calibration
period holds no zero or is entirely zero, every mode keeps the classic result.
SPEI is excluded, because its `P − PET` series is offset before fitting and has no
physical zero mass. `standardized_index()` with `Distribution.beta` applies the mode
to both of its boundary masses, placing the mass at 1 at the mirror of the mass at
0 ([ADR-0024](adr/0024-beta-distribution-for-bounded-series.md)).

## SPEI computation

The SPEI computation follows these steps:

1. **Water balance computation**

   - Compute climatic water balance: D = P - PET
   - Where P is precipitation and PET is potential evapotranspiration
   - Add constant offset (+1000 mm) to ensure all values are positive

2. **Apply SPI methodology**

   - Apply the same steps as SPI (aggregation, fitting, transformation)
   - Use water balance (D) instead of precipitation (P) as input
   - Support both gamma and Pearson Type III distributions

3. **Same constraints as SPI**

   - Clip to valid range [-3.09, 3.09] on the default `output_scale="normal"`
     only; `"probability"` and `"bounded"` are not z-clipped
   - Return dimensionless standardized values

**Key differences from SPI:**

- Incorporates temperature effects through PET
- Requires both precipitation and temperature data
- More sensitive to warming trends
- Better represents agricultural drought (crop water stress)

## PET computation

The two temperature-based methods this library implements, and their equations;
for when to choose each, see {doc}`algorithms`.

### Thornthwaite Method

The Thornthwaite method estimates monthly PET using mean temperature and latitude.

**Equation:**

$$
PET = 1.6 \left( \frac{L}{12} \right) \left( \frac{N}{30} \right) \left( \frac{10 T_a}{I} \right)^a
$$

Where:

- *PET* = potential evapotranspiration (mm/month)
- *L* = mean day length for the month (hours)
- *N* = number of days in the month
- *T*{sub}`a` = mean daily air temperature (°C, clipped to ≥0)
- *I* = annual heat index = Σ(*T*{sub}`ai`/5){sup}`1.514` for all 12 months
- *a* = (6.75×10{sup}`-7`)\*I\*{sup}`3` - (7.71×10{sup}`-5`)\*I\*{sup}`2` + (1.792×10{sup}`-2`)\*I\* + 0.49239

**Implementation details:**

- Automatically adjusts for leap years
- Negative temperatures are clipped to 0°C (no evaporation below freezing)
- Day length computed from latitude using solar geometry
- Returns monthly PET values in mm/month

### Hargreaves Method

The Hargreaves method estimates daily PET using temperature range as a proxy for solar radiation.

**Equation (based on FAO-56 equation 52):**

$$
ET_o = 0.0023 (T_{mean} + 17.8) (T_{max} - T_{min})^{0.5} \times 0.408 \times R_a
$$

Where:

- *ET*{sub}`o` = reference evapotranspiration over grass (mm/day)
- *T*{sub}`mean` = mean daily temperature (°C)
- *T*{sub}`max` = maximum daily temperature (°C)
- *T*{sub}`min` = minimum daily temperature (°C)
- *R*{sub}`a` = extraterrestrial radiation (MJ m{sup}`-2` day{sup}`-1`)

**Implementation details:**

- Computes extraterrestrial radiation from day of year and latitude
- Accounts for Earth-Sun distance variation
- Returns daily PET values in mm/day
- Validates temperature relationships (T{sub}`min` ≤ T{sub}`mean` ≤ T{sub}`max`)

## Palmer index computation

The Palmer drought indices use a two-layer soil moisture accounting model:

1. **Water balance model**

   - Compute potential recharge, runoff, and loss for each time step
   - Track moisture in surface and subsurface soil layers
   - Calculate moisture departure from climatically appropriate conditions

2. **CAFEC procedure**

   - Derive Climatically Appropriate For Existing Conditions (CAFEC) coefficients
   - Calibrate moisture departure to local climate

3. **Index calculation**

   - PDSI: Cumulative moisture anomaly with persistence
   - Z-Index: Monthly moisture departure (no persistence)
   - PHDI: Emphasizes long-term moisture deficits
   - PMDI: Responds quickly to short-term changes

## PCI computation

**Equation** (Oliver, 1980):

$$
PCI = \frac{\sum_{i=1}^{12} P_i^2}{\left(\sum_{i=1}^{12} P_i\right)^2} \times 100
$$

Where *P*{sub}`i` is the precipitation in month *i*.

**Implementation details:**

- Requires complete annual cycle (365 or 366 daily values)
- Rejects incomplete years or years with missing data
- Returns single PCI value per year

## PNP computation

**Equation:**

$$
PNP = \frac{P_{observed}}{P_{normal}} \times 100
$$

Where:

- *P*{sub}`observed` = observed precipitation for the period
- *P*{sub}`normal` = long-term average precipitation for the same calendar period

**Implementation details:**

- Supports multi-step scales (e.g., 3-month, 6-month, or daily equivalents such as 90-day)
- Computes normals separately for each calendar time step
- Standard calibration period: 1981-2010 (U.S. climate normals)
- Can use any calibration period ≥30 years

## Statistical Methods

### Distribution Fitting

**Gamma distribution**:

The two-parameter gamma distribution is the default for SPI and SPEI:

$$
f(x; \alpha, \beta) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\beta}
$$

Where:

- α = shape parameter
- β = scale parameter
- Γ(α) = gamma function

Parameters are estimated using maximum likelihood estimation (MLE) on the calibration period data for each calendar month independently.

**Pearson Type III distribution**:

The three-parameter Pearson Type III distribution is an alternative for skewed data:

$$
f(x; \mu, \sigma, \gamma) = \frac{1}{\sigma \Gamma(\alpha) \beta^\alpha} (x - \xi)^{\alpha-1} e^{-(x-\xi)/\beta}
$$

Where:

- μ = location parameter
- σ = scale parameter
- γ = skewness parameter

Parameters are estimated using L-moments method (see below) on the calibration period data.

### L-moments Estimation

L-moments (Linear moments) are used to estimate Pearson Type III distribution parameters. L-moments have advantages over conventional moments:

- More robust to outliers
- Better for small sample sizes
- Unbiased estimators
- Exist even when conventional moments do not

**Implementation**:

This library implements the L-moments estimation procedures from Hosking (1997):

1. Compute sample L-moments (λ₁, λ₂, τ₃) from calibration data
2. Estimate Pearson Type III parameters from L-moments using analytical relationships
3. Validate parameter estimates

**Reference**: Hosking, J. R. M. (1997). *FORTRAN Routines for Use with the Method of L-Moments, Version 3*. IBM Research Report RC20525. IBM Research Division, T. J. Watson Research Center, Yorktown Heights, NY.

**Minimum data requirement**: At least 4 non-zero values per calendar month for stable L-moments estimation.

### Goodness-of-Fit Validation

The library performs goodness-of-fit testing to validate that the fitted distribution adequately represents the observed data.

**Kolmogorov-Smirnov (K-S) test**:

- Tests the null hypothesis that data follows the fitted distribution
- Applied separately for each calendar month
- Significance level: α = 0.05
- Warning issued if p < 0.05 (poor fit)

**Quality checks**:

1. **Calibration period length**: Warning if < 30 years

   - Constant: `MIN_CALIBRATION_YEARS = 30`

2. **Missing data threshold**: Warning if > 20% missing

   - Constant: `MISSING_DATA_THRESHOLD = 0.20`

3. **Goodness-of-fit threshold**: Warning if p-value < 0.05

   - Constant: `GOODNESS_OF_FIT_P_VALUE_THRESHOLD = 0.05`

**Fallback strategy**:

If Pearson Type III fitting fails (due to insufficient data or numerical issues), the library automatically falls back to gamma distribution with warning messages in the log.

### Run Theory (Event Identification)

Run theory characterizes drought and wet events as *runs*: maximal contiguous
periods on one side of a threshold (Yevjevich, 1967). `climate_indices.runs`
reports detection, event metrics, and period summaries. Severity classes and
spatial aggregation stay downstream.

For a run of elapsed duration $d$ with values $x_i$, of which the set $A$ lies on
the chosen side of a threshold $T$ (all $d$ values when `max_gap=0`):

$$
\text{magnitude} = \sum_{i \in A} |x_i - T|
\qquad
\text{intensity} = \frac{\text{magnitude}}{d}
$$

The **peak** is the minimum value in the run when below the threshold, the
maximum when above. **Interarrival** is the number of time steps from one run's
start to the next run's start, so the final run has none (NaN).

Conventions:

- Direction is explicit (`direction="below"` or `"above"`) and never inferred
  from the threshold's sign, so a threshold such as "above -0.5" is expressible
  and wet runs take their peak on the maximum side.
- A NaN value is not in any run and terminates the run it would otherwise
  continue, so gaps split runs.
- A value exactly equal to the threshold is not in a run.
- `max_gap=0` disables pooling. A positive `max_gap` joins successive runs when
  their intervening gap is no longer than that many input steps.
- Pooling never crosses NaN or masked values. A threshold-equal value can form
  part of a pooled gap, but it contributes no magnitude.
- Duration includes pooled gaps. Magnitude sums only the deviations on the
  selected side of the threshold. Intensity divides that magnitude by elapsed
  duration, including the gaps.
- `min_duration` filters runs after pooling. Interarrival is computed between
  surviving runs only.

The primitive is index-agnostic: a dry SPI spell, a wet SPEI spell, and a KBDI
drought spell (``direction="above"``, since KBDI rises in drought) are the same
run concept on a different series.

#### Period summaries

`runs.summarize_runs()` returns a `RunStatistics` containing a `RunSet` of
retained events and summary arrays. Supply one `periods` label per input step,
such as its year or calendar month. Repeated labels form one group, and the
summary arrays follow sorted unique labels. Without labels, one summary covers
the whole input, including an empty input.

Each complete event belongs to its onset period. The summary reports event
count, total elapsed duration, maximum event magnitude, and valid input count.
An event that crosses a period boundary contributes all its duration and
magnitude to its onset period, so total duration can exceed that period's length.
Frequency is event count divided by the period's non-NaN, unmasked input count.
It is events per valid input step, not the fraction of steps in an event, an
annual rate, or a return period. Infinities retain the original run comparisons
and count as observations; callers must mask them if they represent missing data.

Periods with no events have zero count, duration, and maximum magnitude.
Frequency is NaN when a period has no valid observations. Missing, masked, or
infinite period labels are rejected rather than silently dropped. Missing means
NaN or infinity in a float array, NaT in a datetime64 or timedelta64 array, or
a masked element. NumPy converts mixed string and NaN labels to a string array
that holds the text `"nan"`, so such input is an ordinary label, not missing.
These conventions also apply to
{func}`climate_indices.energy.energy_drought_events`.

```{testcode}
import numpy as np
from climate_indices.runs import summarize_runs

result = summarize_runs(
    [-2, 0, -3, np.nan, -4],
    periods=[2020, 2020, 2021, 2021, 2021],
    max_gap=1,
)
assert result.events.start_index.tolist() == [0, 4]
assert result.count.tolist() == [1, 1]
assert result.total_duration.tolist() == [3, 1]
assert result.max_magnitude.tolist() == [3.0, 3.0]
assert result.frequency.tolist() == [0.5, 0.5]
```

A probability-scale series (`output_scale="probability"`) reports a lower-tail
probability, so its drought runs also use `direction="below"`. The 10%, 5%, and
2.5% one-sided levels (0.1, 0.05, 0.025) correspond to z-scores of about -1.28,
-1.64, and -1.96. Note this is a different ladder from the McKee/WMO drought
classes of -1.0, -1.5, and -2.0, whose probability-scale equivalents are about
0.159, 0.067, and 0.023. Probability-scale outputs are not clipped, so a run can
be counted all the way into the tail.

**Reference**: Yevjevich, V. (1967). *An objective approach to definitions and
investigations of continental hydrologic droughts*. Hydrology Papers 23,
Colorado State University. Drought classes: McKee et al. (1993) and WMO (2012),
both in the bibliography below.

### Compound Event Matching (Concurrent and Sequential)

`climate_indices.compound.compound_events()` matches two run sets that were
identified on the same time axis, for example a drought run set and a heat run
set. Both sources are given as [`RunSet`](#run-theory-event-identification)
objects from `identify_runs()`, plus the shared time axis, its declared cadence,
and one Boolean observation-validity mask. A row names one source run from each
set and never merges or scores them: the result holds indices, durations, lags,
and censoring flags only, so no drought magnitude, fire severity, or heat
intensity is combined into an unrequested severity score.

The time axis must be a complete cadence grid. An absent date is an unknown
observation, not a step to close, so a gap is rejected instead of being
bridged. Runs are validated before matching: each run must hold positive
durations equal to `end_index - start_index + 1`, lie inside the record, be
sorted and nonoverlapping, and contain no unknown observation. A run asserts a
contiguous observed spell, so a run that crosses an unknown step is rejected
rather than silently truncated.

**Concurrent matching** keeps every pair whose inclusive intervals intersect
and reports the intersection, with no one-to-one assignment:

$$
\text{start} = \max(a_{\text{start}}, b_{\text{start}}), \qquad
\text{end} = \min(a_{\text{end}}, b_{\text{end}}), \qquad
\text{overlap\_duration} = \text{end} - \text{start} + 1
$$

Touching endpoints count, so a pair sharing one time step has overlap duration
one. Adjacent pair records are never merged, and a pair is never discarded
because another run already matched, so one long run can match several short
ones and the reverse.

**Sequential matching** keeps every pair with $b_{\text{start}} >
a_{\text{end}}$, so an overlapping pair, or one starting exactly on the first
run's end, is not a follow-on. The lag is the separating distance:

$$
\text{lag} = b_{\text{start}} - a_{\text{end}}
$$

Adjacent runs therefore have lag one. A caller-supplied inclusive window
`lag_window=(low, high)` keeps every pair with `low <= lag <= high`; `None`
matches every lag at or above one. Every step from $a_{\text{end}}$ through
$b_{\text{start}}$ must be observed. A pair separated by an unknown step is not
reported, because a cascade is never inferred across a gap, and the rejected
count is recorded as `n_unknown_bridges` in the result metadata. The reported
interval is the first run's onset through the second run's end, and the lag is
reported separately: the enclosing interval is not an invented joint duration,
and `overlap_duration` is zero because sequential runs are disjoint by
definition.

**Censoring.** A run boundary that touches the record boundary or an unknown
neighbouring step is censored: the physical event may extend beyond the
observed record or across the unknown step. Every row records four flags —
`first_left_censored`, `first_right_censored`, `second_left_censored`, and
`second_right_censored` — so an observed duration is never presented as a
complete physical event duration. Censoring does not remove a run.

Rows appear in first source-run order and then in second source-run order, and
`first_run_id` and `second_run_id` are the caller's own positions, so
`first.start_index[first_run_id]` recovers the source run and its metrics. Every
result array is read-only, and an empty catalog keeps its integer, float, and
Boolean dtypes. The manifest records the calendar, the matching, censoring, and
missingness rules, the effective lag window, the observed and unknown step
counts, the run and pair counts, the rejected unknown-bridge count, and each
source's declared threshold, direction, and `min_duration`. Those three declarations are not retained by `RunSet`, so
any that the caller leaves out is recorded as `"unknown"` and is never
reconstructed from run metrics.

**Limits.** Pair counts are not independent observations and must not feed a
binomial significance test. Matching inherits the run definition of the source
indices and adds no onset probability or causal claim: a lagged association is
a descriptive co-occurrence, not a validated cascade. The catalog has no primary
index CF key, because it is an event catalog and not an index field.

**Reference**: the event contract and its censoring, lag, and all-pairs rules
are recorded in
[ADR-0022](https://github.com/monocongo/climate_indices/blob/main/docs/adr/0022-compound-analysis-contracts.md).

## Complete Bibliography

**SPI and Drought Indices:**

- McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *Proceedings of the 8th Conference on Applied Climatology*, 17-22 January, Anaheim, CA. American Meteorological Society, Boston, MA, 179-184.
- Vicente-Serrano, S. M., Begueria, S., & Lopez-Moreno, J. I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. *Journal of Climate*, 23(7), 1696-1718. <https://doi.org/10.1175/2009JCLI2909.1>

**Potential Evapotranspiration:**

- Thornthwaite, C. W. (1948). An approach toward a rational classification of climate. *Geographical Review*, 38(1), 55-94. <https://doi.org/10.2307/210739>
- Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. *Applied Engineering in Agriculture*, 1(2), 96-99. <https://doi.org/10.13031/2013.26773>
- Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56. Food and Agriculture Organization of the United Nations, Rome. ISBN 92-5-104219-5. Available at: <http://www.fao.org/3/x0490e/x0490e00.htm>

**Palmer Drought Indices:**

- Palmer, W. C. (1965). *Meteorological Drought*. U.S. Weather Bureau Research Paper No. 45. Washington, D.C.
- Wells, N., Goddard, S., & Hayes, M. J. (2004). A Self-Calibrating Palmer Drought Severity Index. *Journal of Climate*, 17(12), 2335-2351. <https://doi.org/10.1175/1520-0442(2004)017%3C2335:ASPDSI%3E2.0.CO;2>

**Precipitation Concentration Index:**

- Oliver, J. E. (1980). Monthly precipitation distribution: A comparative index. *The Professional Geographer*, 32(3), 300-309. <https://doi.org/10.1111/j.0033-0124.1980.00300.x>

**Statistical Methods:**

- Hosking, J. R. M. (1997). *FORTRAN Routines for Use with the Method of L-Moments, Version 3*. IBM Research Report RC20525. IBM Research Division, T. J. Watson Research Center, Yorktown Heights, NY. (Note: Available through various online archives and the R package `lmomco`)

**Additional Resources:**

- World Meteorological Organization (WMO). (2012). *Standardized Precipitation Index User Guide* (WMO-No. 1090). Geneva, Switzerland. Available at: <https://library.wmo.int/doc_num.php?explnum_id=7768>
- American Meteorological Society. (2020). Drought. *Glossary of Meteorology*. <https://glossary.ametsoc.org/wiki/Drought>
