# Climate Index Background and Selection Guidance

What each index measures, when to use it, and how to choose parameters. For the
computation steps, formulas, and implementation details, see
{doc}`algorithm-reference`; the scientific literature is listed in that page's
bibliography.

```{contents} Table of Contents
:backlinks: none
:local: true
```

## Overview

The `climate_indices` library implements scientifically validated algorithms for computing drought and climate variability indices. All implementations follow peer-reviewed methodologies and have been validated against reference datasets. The algorithms are designed for operational climate monitoring, research applications, and climate impact assessments.

## Standardized Precipitation Index (SPI)

### Overview

The Standardized Precipitation Index (SPI) is a widely used indicator for characterizing meteorological drought on multiple timescales. Developed by McKee, Doesken, and Kleist (1993), SPI transforms precipitation data into standardized units that represent the probability of occurrence relative to the long-term climatological distribution.

SPI values are dimensionless and follow a normal distribution with mean 0 and standard deviation 1, making them comparable across different locations and climate regimes.

### When to Use SPI

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

See {doc}`algorithm-reference` for the SPI computation steps and distribution
parameters.

### Parameter Selection

#### Scale (Timescale)

The scale parameter determines the temporal aggregation period: 1-72 months for
monthly data, 1-2196 days for daily data.

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

#### Distribution Choice

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

#### Calibration Period

The calibration period defines the climatological baseline for "normal" precipitation.

**Requirements:**

- **Minimum length**: 30 years (≥30 years strongly recommended)
- **Maximum missing data**: \<20% of calibration period values
- **Goodness-of-fit validation**: Kolmogorov-Smirnov test at p = 0.05 significance level

**Selection guidance:**

- Use WMO-recommended 30-year climate normals (e.g., 1991-2020)
- Ensure calibration period is representative of current climate
- Avoid periods with major data gaps or quality issues
- Consider updating calibration period periodically for climate change studies

## Standardized Precipitation Evapotranspiration Index (SPEI)

### Overview

The Standardized Precipitation Evapotranspiration Index (SPEI) is an extension of SPI that incorporates the effect of temperature on drought through potential evapotranspiration (PET). Developed by Vicente-Serrano, Begueria, and Lopez-Moreno (2010), SPEI provides a more complete picture of water balance by accounting for both water supply (precipitation) and atmospheric water demand (PET).

SPEI is particularly valuable in warming climates where increased temperatures amplify drought severity through enhanced evaporative demand.

### When to Use SPEI

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

### SPI vs SPEI Decision Guide

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

| Characteristic     | SPI                | SPEI                      |
| ------------------ | ------------------ | ------------------------- |
| Data requirements  | Precipitation only | Precipitation + temp      |
| Drought type       | Meteorological     | Agricultural/ecological   |
| Climate change     | Less sensitive     | More sensitive            |
| Computation        | Simpler            | More complex              |
| Agricultural use   | Good               | Better                    |
| Historical studies | Good               | Better for recent decades |

### Parameter Selection

SPEI uses the same parameter selection guidelines as SPI:

- **Scale**: Same timescale considerations (1-72 months for monthly data, 1-2196 days for daily data)
- **Distribution**: Gamma (default) or Pearson Type III
- **Calibration period**: ≥30 years, \<20% missing data, K-S test p=0.05

**Additional consideration - PET method:**

See the Potential Evapotranspiration section below for guidance on choosing between Thornthwaite and Hargreaves methods.

See {doc}`algorithm-reference` for the SPEI computation steps.

## Potential Evapotranspiration (PET)

### Overview

Potential evapotranspiration (PET) represents the atmospheric water demand—the amount of water that would evaporate and transpire from a reference surface if sufficient water were available. PET is a key input for SPEI and water balance calculations.

This library implements two temperature-based PET methods:

1. **Thornthwaite (1948)**: Monthly timestep, temperature-only
2. **Hargreaves (1985)**: Daily timestep, temperature and radiation

The equations and implementation details are in {doc}`algorithm-reference`.

### Thornthwaite strengths and limitations

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

### Hargreaves strengths and limitations

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

### Thornthwaite vs Hargreaves Decision Guide

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

| Characteristic      | Thornthwaite          | Hargreaves        |
| ------------------- | --------------------- | ----------------- |
| Temporal resolution | Monthly               | Daily             |
| Temperature inputs  | Mean only             | Min, max, mean    |
| Additional inputs   | Latitude              | Latitude          |
| Accuracy (general)  | Moderate              | Good              |
| Accuracy (arid)     | Poor (overestimates)  | Good              |
| Accuracy (humid)    | Poor (underestimates) | Moderate          |
| Data requirements   | Minimal               | Moderate          |
| Computational cost  | Low                   | Low               |
| Climate suitability | Temperate             | Semi-arid to arid |

:::{note}
For the most accurate PET estimates, consider using more sophisticated methods
like Penman-Monteith (FAO-56) if wind speed, humidity, and solar radiation data
are available. However, temperature-based methods remain the most practical choice
for large-scale or historical analyses.
:::

## Palmer Drought Indices

### Overview

The Palmer Drought Severity Index (PDSI) and related indices were developed by Wayne Palmer (1965) for the U.S. National Weather Service. The Palmer system includes five related indices:

- **PDSI**: Palmer Drought Severity Index (long-term moisture conditions)
- **PHDI**: Palmer Hydrological Drought Index (groundwater, reservoir levels)
- **PMDI**: Palmer Modified Drought Index (short-term agricultural drought)
- **Z-Index**: Palmer Z-Index (monthly moisture anomaly)
- **scPDSI**: Self-Calibrated PDSI (Wells et al., 2004), available through
  {func}`climate_indices.palmer.scpdsi`, improves spatial comparability by
  calibrating to each location's climate.

These indices are based on a water balance model that accounts for precipitation, evapotranspiration, soil moisture, and runoff.

### When to Use Palmer Indices

**Ideal applications:**

- Operational drought monitoring (NOAA, USDA)
- Agricultural drought assessment
- Long-term water resource planning (PHDI)
- Comparing drought across different U.S. regions (scPDSI)

**Strengths:**

- Comprehensive water balance approach
- Well-established in operational use
- Multiple indices for different applications

**Limitations:**

- Complex calibration requiring soil data
- Fixed temporal scale (not multi-scalar like SPI/SPEI)
- Slow response to emerging drought
- Original PDSI not directly comparable across climates
- Designed for U.S. climates (may need adjustment elsewhere)

:::{note}
The Palmer indices require significant calibration data including soil water holding
capacity, which may not be available for all locations. SPI and SPEI are often
preferred for global applications due to their simpler data requirements and
multi-scalar nature.
:::

The soil-moisture accounting model those indices build on is described in
{doc}`algorithm-reference`.

## Additional Indices

### Precipitation Concentration Index (PCI)

The Precipitation Concentration Index (PCI) quantifies the temporal distribution of precipitation throughout the year (Oliver, 1980).

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

The PCI equation and implementation details are in {doc}`algorithm-reference`.

### Percentage of Normal Precipitation (PNP)

The Percentage of Normal Precipitation (PNP) expresses precipitation as a percentage of the long-term average for a given location and time period.

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

The PNP equation and implementation details are in {doc}`algorithm-reference`.

## Validation Datasets and Methods

### Reference Datasets

The climate_indices library has been validated against several authoritative sources:

**NCAR Climate Data Guide**:

- SPI reference implementations
- Multi-scale validation datasets
- URL: <https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-index-spi>

**NOAA/NCEI Climate Monitoring**:

- Operational SPI, SPEI, PDSI values for U.S. climate divisions
- Monthly updates for verification
- URL: <https://www.ncdc.noaa.gov/temp-and-precip/drought/>

**Global SPEI Database**:

- Vicente-Serrano et al. global SPEI dataset
- 0.5° resolution, 1901-present
- URL: <https://spei.csic.es/database.html>

### Validation Approach

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

### External Validation Resources

**NCAR Command Language (NCL)**:

- Mature SPI/SPEI implementations
- URL: <https://www.ncl.ucar.edu/>

**Climate Indices in Python (other libraries)**:

- `climate-indices` (NCAR/UCAR)
- `spei` (CRAN R package, Python wrapper)

:::{tip}
When validating this library's output against other implementations, ensure
consistent parameter choices (distribution, calibration period, scale) and
input data preprocessing (missing value handling, time alignment).
:::
