# Architecture

## Module Structure

```
climate_indices/
├── src/climate_indices/
│   ├── __init__.py          # Package init, accessor registration
│   ├── __main__.py          # CLI entry point (argparse)
│   ├── __spi__.py           # SPI-specific CLI (xarray/Dask pipeline)
│   ├── accessors.py         # xarray DataArray accessor (@xr.register_dataarray_accessor)
│   ├── compute.py           # Core computations (Numba-accelerated)
│   ├── indices.py           # High-level API (SPI, SPEI, PET, etc.)
│   ├── eto.py               # Evapotranspiration (Thornthwaite, Hargreaves)
│   ├── palmer.py            # Palmer drought indices
│   ├── lmoments.py          # L-moments statistical computations
│   └── utils.py             # Utilities, logging, data transforms
├── tests/                   # pytest test suite
├── notebooks/               # Jupyter notebooks for demos
└── docs/                    # Sphinx documentation
```

## Design Patterns

### Layered Architecture
1. **Core Layer** (`compute.py`): Numba-accelerated numerical functions
2. **Algorithm Layer** (`indices.py`, `palmer.py`, `eto.py`): Climate index implementations
3. **API Layer** (`accessors.py`, `__main__.py`): User-facing interfaces

### xarray/Dask Processing Pipeline
- NetCDF files opened with `xr.open_dataset(chunks={"time": -1})`
- Time dimension kept as single chunk (rolling windows must not cross boundaries)
- `xr.apply_ufunc()` with `dask="parallelized"` for vectorized operations
- Dask distributed via `--multiprocessing all` CLI flag

### Distribution Fitting Strategy
1. Attempt primary distribution (Pearson Type III or Gamma)
2. On excessive NaN results, fallback to alternative distribution
3. Controlled via `DistributionFallbackStrategy` enum

### Data Flow (SPI Example)
```
Input Data → Validation → Rolling Sum → Distribution Fitting → CDF Transform → Standardize → Output
     ↓            ↓            ↓               ↓                    ↓              ↓
  NetCDF     _validate    apply_ufunc      gamma.fit()         gamma.cdf()    norm.ppf()
```

## Key Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `Periodicity` | `monthly`, `daily` | Time series frequency |
| `Distribution` | `gamma`, `pearson` | Statistical distribution |
| `InputType` | `grid`, `divisions`, `timeseries` | Data structure type |

## CLI Entry Points

| Command | Module | Description |
|---------|--------|-------------|
| `climate_indices` | `__main__.py` | Full CLI with all indices |
| `spi` | `__spi__.py` | SPI-specific xarray pipeline |
