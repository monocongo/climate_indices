====================================================
Algorithm Documentation and Scientific References
====================================================

This page provides comprehensive documentation of the climate index algorithms implemented in this library, including scientific references, parameter selection guidance, and implementation details.

.. contents:: Table of Contents
   :local:
   :backlinks: none

Overview
========

The ``climate_indices`` library implements scientifically validated algorithms for computing drought and climate variability indices. All implementations follow peer-reviewed methodologies and have been validated against reference datasets. The algorithms are designed for operational climate monitoring, research applications, and climate impact assessments.

Standardized Precipitation Index (SPI)
======================================

Overview
--------

The Standardized Precipitation Index (SPI) is a widely used indicator for characterizing meteorological drought on multiple timescales. Developed by McKee, Doesken, and Kleist (1993), SPI transforms precipitation data into standardized units that represent the probability of occurrence relative to the long-term climatological distribution.

SPI values are dimensionless and follow a normal distribution with mean 0 and standard deviation 1, making them comparable across different locations and climate regimes.

Algorithm Description
---------------------

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

When to Use SPI
---------------

**Ideal applications:**

- Precipitation-only drought monitoring
- Locations where temperature data is unavailable or unreliable
- Comparing drought conditions across different climate zones
- Operational drought monitoring systems
- Agricultural drought assessment (short timescales: 1-3 months)
- Hydrological drought assessment (long timescales: 12-24 months)

**Limitations:**

- Does not account for evapotranspiration losses
- May underestimate drought severity in warming climates
- Not suitable for studying temperature-driven drought
- Limited applicability in arid regions with many zero-precipitation months

Parameter Selection
-------------------

Scale (Timescale)
~~~~~~~~~~~~~~~~~

The scale parameter determines the temporal aggregation period (1-72 months supported).

**Common scales and applications:**

- **1-month**: Short-term precipitation anomalies, agricultural impacts
- **3-month**: Seasonal drought, soil moisture deficits
- **6-month**: Medium-term drought, streamflow impacts
- **12-month**: Annual drought, reservoir management
- **24-month**: Long-term drought, groundwater depletion

**Selection guidance:**

- Match scale to the impact domain of interest
- Agricultural applications: 1-6 months
- Hydrological applications: 6-24 months
- Water resource management: 12-48 months

Distribution Choice
~~~~~~~~~~~~~~~~~~~

**Gamma distribution** (recommended default):

- Computationally efficient
- Stable parameter estimation
- Adequate for most precipitation distributions
- Recommended by McKee et al. (1993)

**Pearson Type III distribution** (alternative):

- Better fit for highly skewed data
- More parameters = more flexible
- Requires more data for stable estimation
- Recommended when gamma fit is poor (Kolmogorov-Smirnov test p < 0.05)

Calibration Period
~~~~~~~~~~~~~~~~~~

The calibration period defines the climatological baseline for "normal" precipitation.

**Requirements:**

- **Minimum length**: 30 years (≥30 years strongly recommended)
- **Maximum missing data**: <20% of calibration period values
- **Goodness-of-fit validation**: Kolmogorov-Smirnov test at p = 0.05 significance level

**Selection guidance:**

- Use WMO-recommended 30-year climate normals (e.g., 1991-2020)
- Ensure calibration period is representative of current climate
- Avoid periods with major data gaps or quality issues
- Consider updating calibration period periodically for climate change studies

Standardized Precipitation Evapotranspiration Index (SPEI)
===========================================================

Overview
--------

The Standardized Precipitation Evapotranspiration Index (SPEI) is an extension of SPI that incorporates the effect of temperature on drought through potential evapotranspiration (PET). Developed by Vicente-Serrano, Begueria, and Lopez-Moreno (2010), SPEI provides a more complete picture of water balance by accounting for both water supply (precipitation) and atmospheric water demand (PET).

SPEI is particularly valuable in warming climates where increased temperatures amplify drought severity through enhanced evaporative demand.

Algorithm Description
---------------------

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

When to Use SPEI
----------------

**Ideal applications:**

- Drought monitoring in warming climates
- Agricultural drought assessment (crop water stress)
- Climate change impact studies
- Regions with significant temperature variability
- Studies requiring both water supply and demand

**Advantages over SPI:**

- Captures temperature-driven drought intensification
- More realistic representation of agricultural water stress
- Better correlation with soil moisture and crop yields
- Suitable for climate change studies

**Limitations:**

- Requires quality temperature data in addition to precipitation
- PET estimation adds uncertainty (method-dependent)
- More complex computation than SPI
- PET methods may not be equally valid in all climates

SPI vs SPEI Decision Guide
---------------------------

**Choose SPI when:**

- Only precipitation data is available
- Focus is on meteorological drought
- Comparing historical periods with consistent climate
- Operational monitoring with limited data sources
- Temperature data quality is questionable

**Choose SPEI when:**

- Both precipitation and temperature data are available
- Focus is on agricultural or ecological drought
- Studying climate change impacts
- Temperature trends are significant
- Crop water stress is the primary concern

**Performance comparison:**

==================  ====================  =====================
Characteristic      SPI                   SPEI
==================  ====================  =====================
Data requirements   Precipitation only    Precipitation + temp
Drought type        Meteorological        Agricultural/ecological
Climate change      Less sensitive        More sensitive
Computation         Simpler               More complex
Agricultural use    Good                  Better
Historical studies  Good                  Better for recent decades
==================  ====================  =====================

Parameter Selection
-------------------

SPEI uses the same parameter selection guidelines as SPI:

- **Scale**: Same timescale considerations (1-72 months)
- **Distribution**: Gamma (default) or Pearson Type III
- **Calibration period**: ≥30 years, <20% missing data, K-S test p=0.05

**Additional consideration - PET method:**

See the Potential Evapotranspiration section below for guidance on choosing between Thornthwaite and Hargreaves methods.

Potential Evapotranspiration (PET)
==================================

Overview
--------

Potential evapotranspiration (PET) represents the atmospheric water demand—the amount of water that would evaporate and transpire from a reference surface if sufficient water were available. PET is a key input for SPEI and water balance calculations.

This library implements two temperature-based PET methods:

1. **Thornthwaite (1948)**: Monthly timestep, temperature-only
2. **Hargreaves (1985)**: Daily timestep, temperature and radiation

Thornthwaite Method
-------------------

The Thornthwaite method estimates monthly PET using mean temperature and latitude.

**Equation:**

.. math::

   PET = 1.6 \left( \frac{L}{12} \right) \left( \frac{N}{30} \right) \left( \frac{10 T_a}{I} \right)^a

Where:

- *PET* = potential evapotranspiration (mm/month)
- *L* = mean day length for the month (hours)
- *N* = number of days in the month
- *T*\ :sub:`a` = mean daily air temperature (°C, clipped to ≥0)
- *I* = annual heat index = Σ(*T*\ :sub:`ai`/5)\ :sup:`1.514` for all 12 months
- *a* = (6.75×10\ :sup:`-7`)*I*\ :sup:`3` - (7.71×10\ :sup:`-5`)*I*\ :sup:`2` + (1.792×10\ :sup:`-2`)*I* + 0.49239

**Implementation details:**

- Automatically adjusts for leap years
- Negative temperatures are clipped to 0°C (no evaporation below freezing)
- Day length computed from latitude using solar geometry
- Returns monthly PET values in mm/month

**Strengths:**

- Simple, requires only temperature and latitude
- Well-tested for temperate climates
- Computationally efficient
- Suitable when only temperature data is available

**Limitations:**

- Overestimates PET in arid/windy regions
- Underestimates PET in humid regions
- Does not account for wind speed, humidity, or solar radiation
- Monthly timestep only (not suitable for daily analysis)

Hargreaves Method
-----------------

The Hargreaves method estimates daily PET using temperature range as a proxy for solar radiation.

**Equation (based on FAO-56 equation 52):**

.. math::

   ET_o = 0.0023 (T_{mean} + 17.8) (T_{max} - T_{min})^{0.5} \times 0.408 \times R_a

Where:

- *ET*\ :sub:`o` = reference evapotranspiration over grass (mm/day)
- *T*\ :sub:`mean` = mean daily temperature (°C)
- *T*\ :sub:`max` = maximum daily temperature (°C)
- *T*\ :sub:`min` = minimum daily temperature (°C)
- *R*\ :sub:`a` = extraterrestrial radiation (MJ m\ :sup:`-2` day\ :sup:`-1`)

**Implementation details:**

- Computes extraterrestrial radiation from day of year and latitude
- Accounts for Earth-Sun distance variation
- Returns daily PET values in mm/day
- Validates temperature relationships (T\ :sub:`min` ≤ T\ :sub:`mean` ≤ T\ :sub:`max`)

**Strengths:**

- More accurate than Thornthwaite in many climates
- Daily timestep suitable for high-resolution analysis
- Temperature range is a reasonable proxy for radiation
- Recommended by FAO for data-limited situations

**Limitations:**

- Requires daily min/max temperature (more data than Thornthwaite)
- Assumes temperature range correlates with radiation (not always true)
- May be less accurate in cloudy/humid climates
- Not suitable when only mean temperature is available

Thornthwaite vs Hargreaves Decision Guide
------------------------------------------

**Choose Thornthwaite when:**

- Only monthly mean temperature is available
- Working with historical datasets (common format)
- Computational efficiency is critical
- Consistency with legacy analyses is required
- Operating in temperate climates

**Choose Hargreaves when:**

- Daily temperature data (min/max) is available
- Higher temporal resolution is needed
- Operating in semi-arid to arid climates
- Following FAO guidelines for crop water requirements
- More accurate PET estimates are required

**Performance comparison:**

======================  =======================  =====================
Characteristic          Thornthwaite             Hargreaves
======================  =======================  =====================
Temporal resolution     Monthly                  Daily
Temperature inputs      Mean only                Min, max, mean
Additional inputs       Latitude                 Latitude
Accuracy (general)      Moderate                 Good
Accuracy (arid)         Poor (overestimates)     Good
Accuracy (humid)        Poor (underestimates)    Moderate
Data requirements       Minimal                  Moderate
Computational cost      Low                      Low
Climate suitability     Temperate                Semi-arid to arid
======================  =======================  =====================

.. note::
   For the most accurate PET estimates, consider using more sophisticated methods
   like Penman-Monteith (FAO-56) if wind speed, humidity, and solar radiation data
   are available. However, temperature-based methods remain the most practical choice
   for large-scale or historical analyses.

Palmer Drought Indices
======================

Overview
--------

The Palmer Drought Severity Index (PDSI) and related indices were developed by Wayne Palmer (1965) for the U.S. National Weather Service. The Palmer system includes five related indices:

- **PDSI**: Palmer Drought Severity Index (long-term moisture conditions)
- **PHDI**: Palmer Hydrological Drought Index (groundwater, reservoir levels)
- **PMDI**: Palmer Modified Drought Index (short-term agricultural drought)
- **Z-Index**: Palmer Z-Index (monthly moisture anomaly)
- **SCPDSI**: Self-Calibrating PDSI (Wells et al., 2004) - improved spatial comparability

These indices are based on a water balance model that accounts for precipitation, evapotranspiration, soil moisture, and runoff.

When to Use Palmer Indices
---------------------------

**Ideal applications:**

- Operational drought monitoring (NOAA, USDA)
- Agricultural drought assessment
- Long-term water resource planning (PHDI)
- Comparing drought across different U.S. regions (SCPDSI)

**Strengths:**

- Comprehensive water balance approach
- Well-established in operational use
- Multiple indices for different applications
- Self-calibrating version improves spatial comparability

**Limitations:**

- Complex calibration requiring soil data
- Fixed temporal scale (not multi-scalar like SPI/SPEI)
- Slow response to emerging drought
- Original PDSI not directly comparable across climates
- Designed for U.S. climates (may need adjustment elsewhere)

Algorithm Overview
------------------

The Palmer drought indices use a two-layer soil moisture accounting model:

1. **Water balance model**

   - Compute potential recharge, runoff, and loss for each time step
   - Track moisture in surface and subsurface soil layers
   - Calculate moisture departure from climatically appropriate conditions

2. **CAFEC procedure**

   - Derive Climatically Appropriate For Existing Conditions (CAFEC) coefficients
   - Calibrate moisture departure to local climate
   - Normalize indices for spatial comparability (SCPDSI)

3. **Index calculation**

   - PDSI: Cumulative moisture anomaly with persistence
   - Z-Index: Monthly moisture departure (no persistence)
   - PHDI: Emphasizes long-term moisture deficits
   - PMDI: Responds quickly to short-term changes

.. note::
   The Palmer indices require significant calibration data including soil water holding
   capacity, which may not be available for all locations. SPI and SPEI are often
   preferred for global applications due to their simpler data requirements and
   multi-scalar nature.

Evaporative Demand Drought Index (EDDI)
=======================================

Overview
--------

The Evaporative Demand Drought Index (EDDI) is a drought monitoring tool
developed by the NOAA Physical Sciences Laboratory (PSL) that uses
atmospheric evaporative demand as its sole input. Unlike precipitation-based
indices (SPI, SPEI), EDDI detects drought onset by measuring the "thirst of
the atmosphere" -- increases in evaporative demand that precede soil moisture
depletion and reduced streamflow.

EDDI was designed to provide early warning of agricultural and hydrological
drought, often signaling drought conditions weeks to months before
precipitation-based indicators.

Algorithm Description
---------------------

The EDDI computation follows these steps:

1. **Input validation and preprocessing**

   - Accept 1-D or 2-D arrays of PET (potential evapotranspiration) values
   - Clip negative PET values to zero
   - Handle missing data (NaN propagation)

2. **Temporal aggregation**

   - Compute sliding sums over the specified timescale (e.g., 1, 2, 3, 6, 12 months)
   - For monthly data: reshape to (years, 12)
   - For daily data: reshape to (years, 366)

3. **Empirical ranking (non-parametric)**

   Unlike SPI/SPEI which fit parametric distributions, EDDI uses empirical
   ranking within each calendar period (month or day-of-year):

   - For each time step, rank the current PET value against the calibration
     climatology for that calendar period
   - Ranking uses ``rank = 1 + count(current > climatology)`` matching the
     NOAA Fortran implementation

4. **Tukey plotting position**

   Convert ranks to cumulative probabilities:

   .. math::

      P = \frac{\text{rank} - 0.33}{N + 0.33}

   Where *N* is the number of valid climatology values. The Tukey plotting
   position is unbiased for a wide range of distributions.

5. **Hastings inverse-normal approximation**

   Transform probabilities to z-scores using the Abramowitz & Stegun (1964)
   rational approximation (equation 26.2.23):

   .. math::

      z = \text{sign}(P - 0.5) \left( t - \frac{c_0 + c_1 t + c_2 t^2}{1 + d_1 t + d_2 t^2 + d_3 t^3} \right)

   Where :math:`t = \sqrt{-2 \ln(\min(P, 1-P))}` and the constants are:

   - :math:`c_0 = 2.515517`, :math:`c_1 = 0.802853`, :math:`c_2 = 0.010328`
   - :math:`d_1 = 1.432788`, :math:`d_2 = 0.189269`, :math:`d_3 = 0.001308`

   This approximation achieves accuracy within :math:`4.5 \times 10^{-4}` of
   the exact inverse normal CDF.

6. **Output clipping**

   - Final z-scores are clipped to [-3.09, 3.09]
   - Minimum climatology requirement: 2 valid values per calendar period

PET Method Sensitivity
----------------------

.. important::
   **EDDI is most accurate when using Penman-Monteith FAO56 reference
   evapotranspiration (ETo) as the PET input.**

The choice of PET estimation method significantly affects EDDI's drought
detection accuracy. Hobbins et al. (2016) demonstrated that:

- **Penman-Monteith FAO56** (recommended): Uses radiation, temperature,
  humidity, and wind speed. Captures the full energy balance driving
  evaporative demand. Produces the most physically consistent EDDI signals.

- **Hargreaves**: Uses temperature and extraterrestrial radiation. Acceptable
  when full meteorological data is unavailable, but misses humidity and wind
  contributions to evaporative demand.

- **Thornthwaite** (not recommended for EDDI): Uses temperature only. May
  produce misleading drought signals because temperature alone is a poor
  proxy for evaporative demand, particularly in:

  - Arid regions where advection dominates
  - Climate change scenarios where temperature trends diverge from
    energy-balance trends
  - Continental interiors with strong seasonal wind patterns

**Recommendation:** When computing EDDI for operational drought monitoring,
always use Penman-Monteith FAO56 PET. If the required meteorological
variables (radiation, humidity, wind) are unavailable, consider using
Hargreaves as a fallback, but document this limitation in any analysis.

Interpretation
--------------

EDDI values are standardized z-scores with the following interpretation:

.. list-table:: EDDI Drought Categories
   :header-rows: 1
   :widths: 25 25 50

   * - EDDI Range
     - Category
     - Interpretation
   * - EDDI >= 2.0
     - ED4 (Exceptional)
     - Extremely high evaporative demand
   * - 1.6 <= EDDI < 2.0
     - ED3 (Extreme)
     - Very high evaporative demand
   * - 1.3 <= EDDI < 1.6
     - ED2 (Severe)
     - High evaporative demand
   * - 0.8 <= EDDI < 1.3
     - ED1 (Moderate)
     - Moderately high evaporative demand
   * - -0.5 <= EDDI < 0.8
     - Normal
     - Near-normal evaporative demand
   * - EDDI < -0.5
     - Wet signal
     - Below-normal evaporative demand

.. note::
   Positive EDDI indicates higher-than-normal atmospheric drying potential,
   which is associated with drought conditions. This is the opposite sign
   convention from SPI/SPEI, where negative values indicate drought.

References
----------

- Hobbins, M. T., A. Wood, D. McEvoy, J. Huntington, C. Morton, M. Anderson,
  and C. Hain, 2016: The Evaporative Demand Drought Index. Part I: Linking
  Drought Evolution to Variations in Evaporative Demand. *J. Hydrometeor.*,
  **17**, 1745-1761. https://doi.org/10.1175/JHM-D-15-0121.1

- McEvoy, D. J., J. Huntington, M. T. Hobbins, A. Wood, C. Morton,
  M. Anderson, and C. Hain, 2016: The Evaporative Demand Drought Index.
  Part II: CONUS-Wide Assessment Against Common Drought Indicators.
  *J. Hydrometeor.*, **17**, 1763-1779.
  https://doi.org/10.1175/JHM-D-15-0122.1

- Abramowitz, M. and I. A. Stegun, 1964: *Handbook of Mathematical
  Functions*. National Bureau of Standards Applied Mathematics Series, 55.

Additional Indices
==================

Precipitation Concentration Index (PCI)
----------------------------------------

The Precipitation Concentration Index (PCI) quantifies the temporal distribution of precipitation throughout the year (Oliver, 1980).

**Equation:**

.. math::

   PCI = \frac{\sum_{i=1}^{12} P_i^2}{\left(\sum_{i=1}^{12} P_i\right)^2} \times 100

Where *P*\ :sub:`i` is the precipitation in month *i*.

**Interpretation:**

- PCI < 10: Uniform precipitation distribution
- 10 ≤ PCI < 15: Moderate precipitation concentration
- 15 ≤ PCI < 20: Irregular precipitation distribution
- PCI ≥ 20: Strong precipitation concentration

**Use cases:**

- Characterizing precipitation seasonality
- Climate classification
- Assessing temporal variability of water supply
- Agricultural planning (growing season water availability)

**Implementation details:**

- Requires complete annual cycle (365 or 366 daily values)
- Rejects incomplete years or years with missing data
- Returns single PCI value per year

Percentage of Normal Precipitation (PNP)
-----------------------------------------

The Percentage of Normal Precipitation (PNP) expresses precipitation as a percentage of the long-term average for a given location and time period.

**Equation:**

.. math::

   PNP = \frac{P_{observed}}{P_{normal}} \times 100

Where:

- *P*\ :sub:`observed` = observed precipitation for the period
- *P*\ :sub:`normal` = long-term average precipitation for the same calendar period

**Implementation details:**

- Supports multi-month scales (e.g., 3-month, 6-month)
- Computes normals separately for each calendar month
- Standard calibration period: 1981-2010 (U.S. climate normals)
- Can use any calibration period ≥30 years

**Interpretation:**

- PNP > 100%: Above-normal precipitation
- PNP = 100%: Normal precipitation
- PNP < 100%: Below-normal precipitation
- PNP < 70%: Drought conditions (rule of thumb)

**Use cases:**

- Simple, intuitive drought indicator
- Public communication (easily understood)
- Operational monitoring
- Agricultural extension services

**Limitations:**

- Not standardized across different climates
- Skewed distribution (not suitable for statistical analysis)
- Less sophisticated than SPI/SPEI
- Sensitive to calibration period selection

Statistical Methods
===================

Distribution Fitting
--------------------

**Gamma distribution**:

The two-parameter gamma distribution is the default for SPI and SPEI:

.. math::

   f(x; \alpha, \beta) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\beta}

Where:

- α = shape parameter
- β = scale parameter
- Γ(α) = gamma function

Parameters are estimated using maximum likelihood estimation (MLE) on the calibration period data for each calendar month independently.

**Pearson Type III distribution**:

The three-parameter Pearson Type III distribution is an alternative for skewed data:

.. math::

   f(x; \mu, \sigma, \gamma) = \frac{1}{\sigma \Gamma(\alpha) \beta^\alpha} (x - \xi)^{\alpha-1} e^{-(x-\xi)/\beta}

Where:

- μ = location parameter
- σ = scale parameter
- γ = skewness parameter

Parameters are estimated using L-moments method (see below) on the calibration period data.

L-moments Estimation
--------------------

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

Goodness-of-Fit Validation
---------------------------

The library performs goodness-of-fit testing to validate that the fitted distribution adequately represents the observed data.

**Kolmogorov-Smirnov (K-S) test**:

- Tests the null hypothesis that data follows the fitted distribution
- Applied separately for each calendar month
- Significance level: α = 0.05
- Warning issued if p < 0.05 (poor fit)

**Quality checks**:

1. **Calibration period length**: Warning if < 30 years

   - Constant: ``MIN_CALIBRATION_YEARS = 30``

2. **Missing data threshold**: Warning if > 20% missing

   - Constant: ``MISSING_DATA_THRESHOLD = 0.20``

3. **Goodness-of-fit threshold**: Warning if p-value < 0.05

   - Constant: ``GOODNESS_OF_FIT_P_VALUE_THRESHOLD = 0.05``

**Fallback strategy**:

If Pearson Type III fitting fails (due to insufficient data or numerical issues), the library automatically falls back to gamma distribution with warning messages in the log.

Validation Datasets and Methods
================================

Reference Datasets
------------------

The climate_indices library has been validated against several authoritative sources:

**NCAR Climate Data Guide**:

- SPI reference implementations
- Multi-scale validation datasets
- URL: https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi

**NOAA/NCEI Climate Monitoring**:

- Operational SPI, SPEI, PDSI values for U.S. climate divisions
- Monthly updates for verification
- URL: https://www.ncdc.noaa.gov/temp-and-precip/drought/

**Global SPEI Database**:

- Vicente-Serrano et al. global SPEI dataset
- 0.5° resolution, 1901-present
- URL: https://spei.csic.es/database.html

Validation Approach
-------------------

**Test methodology**:

1. **Unit tests**: Test individual functions with known inputs/outputs
2. **Integration tests**: Test complete index calculations
3. **Regression tests**: Compare against reference implementations
4. **Property-based tests**: Validate mathematical properties using Hypothesis
5. **Benchmark tests**: Verify computational efficiency and memory usage

**Validation metrics**:

- Pearson correlation coefficient (r > 0.99 for reference datasets)
- Root mean square error (RMSE)
- Maximum absolute error
- Computation time and memory usage

External Validation Resources
------------------------------

**NCAR Command Language (NCL)**:

- Mature SPI/SPEI implementations
- URL: https://www.ncl.ucar.edu/

**Climate Indices in Python (other libraries)**:

- ``climate-indices`` (NCAR/UCAR)
- ``spei`` (CRAN R package, Python wrapper)

.. tip::
   When validating this library's output against other implementations, ensure
   consistent parameter choices (distribution, calibration period, scale) and
   input data preprocessing (missing value handling, time alignment).

Complete Bibliography
=====================

**SPI and Drought Indices:**

- McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *Proceedings of the 8th Conference on Applied Climatology*, 17-22 January, Anaheim, CA. American Meteorological Society, Boston, MA, 179-184.

- Vicente-Serrano, S. M., Begueria, S., & Lopez-Moreno, J. I. (2010). A Multiscalar Drought Index Sensitive to Global Warming: The Standardized Precipitation Evapotranspiration Index. *Journal of Climate*, 23(7), 1696-1718. https://doi.org/10.1175/2009JCLI2909.1

**EDDI (Evaporative Demand Drought Index):**

- Hobbins, M. T., Wood, A., McEvoy, D., Huntington, J., Morton, C., Anderson, M., & Hain, C. (2016). The Evaporative Demand Drought Index. Part I: Linking Drought Evolution to Variations in Evaporative Demand. *Journal of Hydrometeorology*, 17(6), 1745-1761. https://doi.org/10.1175/JHM-D-15-0121.1

- McEvoy, D. J., Huntington, J., Hobbins, M. T., Wood, A., Morton, C., Anderson, M., & Hain, C. (2016). The Evaporative Demand Drought Index. Part II: CONUS-Wide Assessment Against Common Drought Indicators. *Journal of Hydrometeorology*, 17(6), 1763-1779. https://doi.org/10.1175/JHM-D-15-0122.1

**Potential Evapotranspiration:**

- Thornthwaite, C. W. (1948). An approach toward a rational classification of climate. *Geographical Review*, 38(1), 55-94. https://doi.org/10.2307/210739

- Hargreaves, G. H., & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. *Applied Engineering in Agriculture*, 1(2), 96-99. https://doi.org/10.13031/2013.26773

- Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56. Food and Agriculture Organization of the United Nations, Rome. ISBN 92-5-104219-5. Available at: http://www.fao.org/3/x0490e/x0490e00.htm

**Palmer Drought Indices:**

- Palmer, W. C. (1965). *Meteorological Drought*. U.S. Weather Bureau Research Paper No. 45. Washington, D.C.

- Wells, N., Goddard, S., & Hayes, M. J. (2004). A Self-Calibrating Palmer Drought Severity Index. *Journal of Climate*, 17(12), 2335-2351. https://doi.org/10.1175/1520-0442(2004)017<2335:ASPDSI>2.0.CO;2

**Precipitation Concentration Index:**

- Oliver, J. E. (1980). Monthly precipitation distribution: A comparative index. *The Professional Geographer*, 32(3), 300-309. https://doi.org/10.1111/j.0033-0124.1980.00300.x

**Statistical Methods:**

- Hosking, J. R. M. (1997). *FORTRAN Routines for Use with the Method of L-Moments, Version 3*. IBM Research Report RC20525. IBM Research Division, T. J. Watson Research Center, Yorktown Heights, NY. (Note: Available through various online archives and the R package ``lmomco``)

**Additional Resources:**

- World Meteorological Organization (WMO). (2012). *Standardized Precipitation Index User Guide* (WMO-No. 1090). Geneva, Switzerland. Available at: https://library.wmo.int/doc_num.php?explnum_id=7768

- American Meteorological Society. (2020). Drought. *Glossary of Meteorology*. https://glossary.ametsoc.org/wiki/Drought
