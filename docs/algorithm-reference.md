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
   - Account for zero precipitation values in probability calculations

5. **Inverse normal transformation**

   - Apply inverse normal (Gaussian) CDF to obtain standardized values
   - This transforms probabilities to standard normal deviates

6. **Clipping to valid range**

   - Constrain final SPI values to [-3.09, 3.09]
   - This range represents probabilities from 0.001 to 0.999

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

   - Clip to valid range [-3.09, 3.09]
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

- Supports multi-month scales (e.g., 3-month, 6-month)
- Computes normals separately for each calendar month
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
