# Climate Indices Reference

## Standardized Precipitation Index (SPI)

### Definition
Probabilistic index representing the number of standard deviations a precipitation observation departs from a fitted probability distribution.

### Scales
| Scale | Description | Use Case |
|-------|-------------|----------|
| 1-month | Short-term conditions | Agricultural drought |
| 3-month | Seasonal conditions | Soil moisture |
| 6-month | Medium-term | Streamflow, reservoir |
| 12-month | Long-term | Groundwater, reservoir |
| 24-month | Extended | Long-term water supply |

### Interpretation
| SPI Value | Category |
|-----------|----------|
| >= 2.0 | Extremely wet |
| 1.5 to 1.99 | Very wet |
| 1.0 to 1.49 | Moderately wet |
| -0.99 to 0.99 | Near normal |
| -1.0 to -1.49 | Moderately dry |
| -1.5 to -1.99 | Severely dry |
| <= -2.0 | Extremely dry |

### Implementation
- Distribution: Gamma (2-parameter)
- Rolling sum over scale months
- CDF transform to uniform, then inverse normal

---

## Standardized Precipitation Evapotranspiration Index (SPEI)

### Definition
Like SPI but uses water balance (precipitation minus PET) instead of precipitation alone.

### Advantages Over SPI
- Accounts for temperature effects on drought
- Better for climate change studies
- More sensitive to evaporative demand

### Implementation
- Distribution: Pearson Type III (3-parameter, handles negative values)
- Uses log-logistic distribution in some implementations
- Fallback to Gamma if Pearson fitting fails

---

## Potential Evapotranspiration (PET)

### Thornthwaite Method
- Input: Monthly mean temperature
- Simple, temperature-based
- Less accurate in arid regions

### Hargreaves Method
- Input: Temperature (min, max, mean), latitude
- Uses temperature range as radiation proxy
- Better in data-sparse regions

### FAO Penman-Monteith (Reference)
- Input: Temperature, humidity, wind, radiation
- Most physically-based
- Not implemented in this library

---

## Palmer Drought Indices

### Components
| Index | Name | Description |
|-------|------|-------------|
| PDSI | Palmer Drought Severity Index | Self-calibrating soil moisture |
| PHDI | Palmer Hydrological Drought Index | Long-term hydrological |
| PMDI | Palmer Modified Drought Index | Operational version |
| Z-Index | Palmer Z-Index | Short-term moisture anomaly |
| scPDSI | Self-Calibrating PDSI | Spatially comparable |

### Requirements
- Precipitation
- Temperature or PET
- Available Water Capacity (AWC) of soil

---

## Percentage of Normal Precipitation (PNP)

### Definition
Simple ratio of observed precipitation to long-term average.

```
PNP = (observed / normal) * 100
```

### Interpretation
- 100% = Normal
- > 100% = Above normal
- < 100% = Below normal

### Limitations
- Does not account for distribution
- Not comparable across different climates
