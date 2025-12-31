# Project Brief

A Python library for computing climate indices useful for drought monitoring and climate research.

## Purpose
Provide reference implementations of standardized climate indices that can be used by researchers, operational drought monitors, and climate scientists.

## Key Indices
- **SPI** (Standardized Precipitation Index): Probabilistic drought index based on precipitation
- **SPEI** (Standardized Precipitation Evapotranspiration Index): Like SPI but accounts for temperature via PET
- **PET** (Potential Evapotranspiration): Thornthwaite and Hargreaves methods
- **PNP** (Percentage of Normal Precipitation): Simple ratio to normal conditions
- **Palmer** indices: PDSI, PHDI, PMDI, Z-Index, scPDSI

## Two API Styles
1. **NumPy-based**: Direct functions operating on 1D/2D arrays
2. **xarray-native**: DataArray accessor and functions with optional Dask parallelism

## Target Users
- Climate researchers
- Drought monitoring agencies (NOAA, NIDIS)
- Agricultural forecasters
- Water resource managers
