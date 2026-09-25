# Workflow examples

Complete invocations to copy and adapt. Option names and valid values are in the
[command-line interface reference](reference.md#command-line-interface); the
distribution-fitting parameters used below are described in
{doc}`algorithm-reference`.

## Example input and output datasets

Example NetCDF datasets that are valid input to the `climate_indices` command
are available from the associated project
[example_climate_indices](https://github.com/monocongo/example_climate_indices/).
The input NetCDF files used in the examples below (`nclimdiv.nc`,
`nclimgrid_lowres_prcp.nc`, etc.) can be fetched from this repository, as well
as associated output NetCDF datasets that can be used to validate result of the
below examples.

## Example command line invocations

### US Climate Divisions (all indices)

```bash
process_climate_indices --index all --periodicity monthly --scales 3 6 \
--netcdf_precip /data/nclimdiv.nc \
--netcdf_temp /data/nclimdiv.nc \
--netcdf_awc /data/nclimdiv.nc \
--output_file_base /data/nclimdiv \
--var_name_precip prcp --var_name_temp tavg --var_name_awc awc \
--calibration_start_year 1951 --calibration_end_year 2010
```

The above command will compute all indices from an input NetCDF dataset containing
precipitation, temperature, and available water capacity variables (in this case,
the US Climate Divisions NetCDF dataset provided in the example inputs directory).
The input dataset is monthly data and the calibration period used will be
Jan. 1951 through Dec. 2010. The indices will be computed at 3-month and 6-month scales.
Upon completion the individual NetCDF files will contain variables for all computed indices:
`/data/nclimdiv_pet.nc`, `/data/nclimdiv_pnp_03.nc`, `/data/nclimdiv_pnp_06.nc`,
`/data/nclimdiv_spi_gamma_03.nc`, `/data/nclimdiv_spi_gamma_06.nc`,
`/data/nclimdiv_spi_pearson_03.nc`, `/data/nclimdiv_spi_pearson_06.nc`,
`/data/nclimdiv_spei_gamma_03.nc`, `/data/nclimdiv_spei_gamma_06.nc`,
`/data/nclimdiv_spei_pearson_03.nc`, `/data/nclimdiv_spei_pearson_06.nc`,
`/data/nclimdiv_pdsi.nc`, `/data/nclimdiv_phdi.nc`, `/data/nclimdiv_pmdi.nc`,
`/data/nclimdiv_zindex.nc`, and `/data/nclimdiv_scpdsi.nc`.
Parallelization will occur utilizing all but one of the available CPUs
(default since the `--multiprocessing` option is omitted).

### PET monthly

```bash
process_climate_indices --index pet --periodicity monthly --netcdf_temp \
/data/nclimgrid_lowres_tavg.nc --var_name_temp tavg --output_file_base \
<out_dir>/nclimgrid_lowres --multiprocessing all_but_one
```

The above command will compute PET (potential evapotranspiration) using the
Thornthwaite method from an input temperature dataset (in this case, the reduced
resolution nClimGrid temperature dataset provided in the example inputs directory).
The input dataset is monthly data and the calibration period used will be Jan. 1951
through Dec. 2010. The output file will be `<out_dir>/nclimgrid_lowres_pet.nc`.
Parallelization will occur utilizing all but one of the available CPUs.

### SPI daily

```bash
process_climate_indices --index spi  --periodicity daily --netcdf_precip \
/data/cmorph_lowres_daily_conus_prcp.nc --var_name_precip \
prcp --output_file_base <out_dir>/cmorph_lowres_daily_conus --scales 30 90 \
--calibration_start_year 1998 --calibration_end_year 2016 \
--multiprocessing all
```

The above command will compute SPI (standardized precipitation index, both gamma
and Pearson Type III distributions) from an input precipitation dataset (in this case,
the reduced resolution CMORPH precipitation dataset provided in the example inputs
directory). The input dataset is daily data and the calibration period used will be
Jan. 1st, 1998 through Dec. 31st, 2016. The index will be computed at 30-day and
90-day timescales. The output files will be `<out_dir>/cmorph_lowres_daily_conus_spi_gamma_30.nc`,
`<out_dir>/cmorph_lowres_daily_conus_spi_gamma_90.nc`,
`<out_dir>/cmorph_lowres_daily_conus_spi_pearson_30.nc`, and
`<out_dir>/cmorph_lowres_daily_conus_spi_pearson_90.nc`. Parallelization will occur utilizing
all CPUs.

### SPI monthly

```bash
process_climate_indices --index spi --periodicity monthly --netcdf_precip \
/data/nclimgrid_lowres_prcp.nc --var_name_precip prcp \
--output_file_base <out_dir>/nclimgrid_lowres --scales 6 12 \
--calibration_start_year 1951 --calibration_end_year 2010 \
--multiprocessing all
```

The above command will compute SPI (standardized precipitation index, both gamma and
Pearson Type III distributions) from an input precipitation dataset (in this case,
the reduced resolution nClimGrid precipitation dataset provided in the example inputs directory).
The input dataset is monthly data and the calibration period used will be
Jan. 1951 through Dec. 2010. The index will be computed at 6-month and 12-month timescales.
The output files will be `<out_dir>/nclimgrid_lowres_spi_gamma_06.nc`,
`<out_dir>/nclimgrid_lowres_spi_gamma_12.nc`, `<out_dir>/nclimgrid_lowres_spi_pearson_06.nc`,
and `<out_dir>/nclimgrid_lowres_spi_pearson_12.nc`. Parallelization will occur utilizing
all CPUs.

### KBDI daily

```bash
process_climate_indices --index kbdi --periodicity daily \
--netcdf_precip /data/cmorph_lowres_daily_conus_prcp.nc --var_name_precip prcp \
--netcdf_temp /data/daily_tmax.nc --var_name_temp tmax \
--output_file_base <out_dir>/kbdi_example
```

The above command will compute KBDI (Keetch-Byram Drought Index) from daily
precipitation and daily maximum temperature datasets. The inputs must cover at
least 30 years of daily record so that KBDI's mean annual precipitation can be
derived. Input values are interpreted as metric (mm and degrees Celsius) by
default; use `--kbdi_units imperial` for inputs in inches and degrees
Fahrenheit, which also selects output in hundredths of an inch. The output
file will be `<out_dir>/kbdi_example_kbdi.nc` (`<out_dir>/kbdi_example_kbdi_imperial.nc`
with `--kbdi_units imperial`). Unlike the other indices, KBDI is not computed
by the `--index all` selection.

### Flood indices daily

```bash
process_climate_indices --index edi --periodicity daily \
--netcdf_precip /data/daily_prcp.nc --var_name_precip prcp \
--calibration_start_year 1991 --calibration_end_year 2020 \
--output_file_base <out_dir>/flood_example

process_climate_indices --index flood_index --periodicity daily \
--netcdf_pe <out_dir>/flood_example_pe.nc --var_name_pe pe --year_start_month 10 \
--calibration_start_year 1991 --calibration_end_year 2020 \
--output_file_base <out_dir>/flood_example

process_climate_indices --index api --periodicity daily --api_k 0.9 \
--netcdf_precip /data/daily_prcp.nc --var_name_precip prcp \
--output_file_base <out_dir>/flood_example
```

The above commands compute the flood-potential indices (which show flood
potential, not flooding) from a daily precipitation dataset, all with
`--periodicity daily`. EDI and the Flood Index (I_F) are computed from effective
precipitation (PE): given `--netcdf_precip` they first compute PE and write it to
`<out_dir>/flood_example_pe.nc`, and given `--netcdf_pe` and `--var_name_pe` in
its place, as in the second command, they reuse an existing PE file, so one PE
file can feed both indices. Provide one of the two, not both. The calibration
years are optional and inferred from the input when omitted. I_F requires
`--year_start_month`, the calendar month on which each year of annual maxima
starts, and API requires `--api_k`, its daily decay constant between 0 and 1.
`--index pe` writes only PE. The output files are `<out_dir>/flood_example_edi.nc`,
`<out_dir>/flood_example_flood_index.nc`, and `<out_dir>/flood_example_api.nc`. Like
KBDI, these indices are not computed by the `--index all` selection.

### SPEI monthly

```bash
process_climate_indices --index spei --periodicity monthly --netcdf_precip \
/data/nclimgrid_lowres_prcp.nc --var_name_precip prcp --netcdf_pet \
/data/nclimgrid_lowres_pet.nc --var_name_pet pet --output_file_base \
<out_dir>/nclimgrid_lowres --scales 9 18 --calibration_start_year 1951 \
--calibration_end_year 2010 --multiprocessing all
```

The above command will compute SPEI (standardized precipitation evapotranspiration index,
both gamma and Pearson Type III distributions) from input precipitation and potential evapotranspiration datasets
(in this case, the reduced resolution nClimGrid precipitation and PET datasets provided in the example inputs directory).
The input datasets are monthly data and the calibration period used will be Jan. 1951 through Dec. 2010. The index
datasets will be computed at 9-month and 18-month timescales. The output files will be
`<out_dir>/nclimgrid_lowres_spei_gamma_09.nc`, `<out_dir>/nclimgrid_lowres_spei_gamma_18.nc`,
`<out_dir>/nclimgrid_lowres_spei_pearson_09.nc`, and `<out_dir>/nclimgrid_lowres_spei_pearson_18.nc`.
Parallelization will occur utilizing all CPUs.

### Palmers monthly

```bash
process_climate_indices --index palmers --periodicity monthly --netcdf_precip \
/data/nclimgrid_lowres_prcp.nc --var_name_precip prcp --netcdf_pet \
/data/nclimgrid_lowres_pet.nc --var_name_pet pet --netcdf_awc \
/data/nclimgrid_lowres_soil.nc  --var_name_awc awc --output_file_base \
<out_dir>/nclimgrid_lowres --calibration_start_year 1951 --calibration_end_year 2010 \
--multiprocessing all
```

The above command will compute the Palmer drought indices: PDSI (original Palmer Drought Severity Index),
PHDI (Palmer Hydrological Drought Index), PMDI (Palmer Modified Drought Index), Z-Index (Palmer
Z-Index), and scPDSI (Self-calibrated Palmer Drought Severity Index) from input precipitation,
potential evapotranspiration, and available water capacity datasets
(in this case, the reduced resolution nClimGrid precipitation, PET, and AWC datasets provided in the
example inputs directory). The input datasets are monthly data and the calibration period used will be
Jan. 1951 through Dec. 2010. The output files will be
`<out_dir>/nclimgrid_lowres_pdsi.nc`, `<out_dir>/nclimgrid_lowres_phdi.nc`,
`<out_dir>/nclimgrid_lowres_pmdi.nc`, `<out_dir>/nclimgrid_lowres_zindex.nc`,
and `<out_dir>/nclimgrid_lowres_scpdsi.nc`.

:::{note}
The Palmer routines ({func}`climate_indices.palmer.pdsi`) take precipitation,
PET, and available water capacity in inches. The command line normalizes
precipitation and PET to millimeters for the other indices, so these inputs
may declare either unit: values labeled `inches`/`inch` or
`mm`/`millimeters` are converted to inches before computing. A
precipitation rate (`mm/dy`) is not accepted for Palmers -- it is a daily
rate, not the monthly accumulated depth `palmer.pdsi()` requires. An AWC
variable without a `units` attribute is assumed to be inches, and one that
declares any other unit is rejected.
:::

:::{note}
Self-calibrated PDSI (scPDSI) is available through the NumPy API as
{func}`climate_indices.palmer.scpdsi` and through
`process_climate_indices --index palmers`, which writes it as
`<output_file_base>_scpdsi.nc` alongside PDSI, PHDI, PMDI, and Z-Index.
:::

Parallelization will occur utilizing all CPUs.

## Pre-compute SPI distribution fitting variables

The distribution fitting parameters of a calibration period can be computed once
and then supplied to subsequent SPI calculations, so that later values are
standardized against the same fitted climatology.

The fitting cache is a library-level feature. The legacy `spi` script's
`--save_params` and `--load_params` options, which cached the same parameters
in a NetCDF file, were removed with the script in 3.0.0 -- see
{doc}`deprecations/api-changes`.

```python
import xarray as xr

from climate_indices import compute, indices

periodicity = compute.Periodicity.monthly
precipitation = xr.open_dataset("nclimgrid_prcp.nc")["prcp"].transpose("time", "lat", "lon")
values = precipitation.values

# fit the 3-month scaled precipitation once, over the calibration period
scaled = compute.prepare_scaled(values, scale=3, periodicity=periodicity, spatial_time_major=True)
alphas, betas = compute.gamma_parameters(
    scaled,
    data_start_year=1975,
    calibration_start_year=1998,
    calibration_end_year=2016,
    periodicity=periodicity,
)

# standardize against that fitted climatology
spi = indices.spi(
    values,
    scale=3,
    distribution=indices.Distribution.gamma,
    data_start_year=1975,
    calibration_year_initial=1998,
    calibration_year_final=2016,
    periodicity=periodicity,
    fitting_params={"alpha": alphas, "beta": betas},
    spatial_time_major=True,
)
```

Perform the fitting over the calibration period only, and keep the parameters for
as long as that climatology is the one to standardize against -- for example with
`np.savez("nclimgrid_fitting.npz", alpha=alphas, beta=betas)` and `np.load()` in
later runs. The parameters are then used exactly as supplied: the calibration years
still passed to `indices.spi()` no longer take part in any fit, so they have to
match the period the parameters were fitted over.

They also have to come from the same fitting scale and the same series the
consuming call standardizes. The parameter arrays record neither, and
`indices.spi()` does not validate them, so parameters fitted from 3-month scaled
precipitation are accepted by a 6-month call and silently standardize against the
wrong distribution. Keep a cached fit next to the scale it was fitted at, and reuse
it only for that same scale.

Pearson Type III parameters are computed per series from the scaled values with
`compute.pearson_parameters()` and supplied as `prob_zero`, `loc`, `scale`,
and `skew`. The parameters of a single series are period-only arrays, shape (12,)
for monthly or (366,) for daily input. For the gamma fit, the parameters of a
time-major grid carry the cell dimensions after the period axis, shape
(12, lat, lon) for monthly gridded input, as in the example above; a grid has to be
time-major with three or more dimensions, since two-dimensional input is still read
as the legacy (years, periods) layout of one series. Note that the command line reads
gridded input in the opposite dimension order, (lat, lon, time).

The example above is a whole-grid NumPy path: `.values` materializes the entire
input, and the scaling, fit, and transform then add full-grid arrays of their own, so
the grid has to fit in memory alongside them. For a grid that does not, pass the
xarray data instead -- {func}`climate_indices.spi` and
{func}`climate_indices.spei` accept a Dask-backed DataArray chunked over the spatial
dimensions with time as a single chunk -- see {doc}`xarray_migration`. For a
complete parallel example and the measured speedups, see {doc}`performance`.
