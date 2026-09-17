# Climate Indices for Wildfire Applications

Climate indices describe environmental conditions associated with wildfire,
not fires themselves. Drought's relationship with fire is complex: temperature,
soil moisture, humidity, wind, and available vegetation interact, and prolonged
drought can even reduce fire occurrence by limiting fuel growth
([Drought.gov][drought-fire]).

**None of these indices predicts whether, where, or when an ignition or fire
will occur** ([Drought.gov][drought-fire];
[Natural Resources Canada][nrcan-fwi]). Use them for hydroclimate context and
fire-weather situational awareness, not as ignition, occurrence, spread, or
fire-behavior forecasts. Operational systems consider additional information;
for example, Canada's
Fire Weather Index System separates weather-based fire danger from assessments
of human ignition activity ([Natural Resources Canada][nrcan-fwi]).

## From drought to fire-weather potential

A useful conceptual chain is:

```text
Antecedent meteorological drought       SPI / SPEI / Palmer family
                    ↓
Anomalous atmospheric evaporative demand             EDDI
                    ↓
Drier vegetation and fuels                 KBDI / CFFWIS
                    ↓
Fire-weather potential                 Fosberg / HDW / Haines
```

Read each arrow as a possible physical pathway, never a deterministic
transition. Drought can dry vegetation and increase flammability, while fuel
amount, humidity, temperature, wind, and ecosystem response can strengthen,
weaken, or reverse the relationship ([Drought.gov][drought-fire]).

EDDI isolates one part of this chain: how unusual atmospheric evaporative
demand is for a location and Timescale. NOAA computes reference
evapotranspiration from temperature, humidity, wind, and solar radiation, then
standardizes accumulated demand against the same window in the historical
record. NOAA publishes EDDI at 1–12-week and 1–12-month Timescales
([NOAA PSL][noaa-eddi]); the peer-reviewed method is described by
[Hobbins et al. (2016)][hobbins-2016].

Higher EDDI values indicate anomalously high evaporative demand, which can
provide early warning of rapidly developing drought and fire-weather risk
([NOAA PSL][noaa-eddi]). EDDI does not directly measure soil moisture,
vegetation moisture, fuel loading, ignition, or fire occurrence, so it has no
universal fire threshold ([Hobbins et al. (2016)][hobbins-2016];
[Drought.gov][drought-fire]).

## EDDI example: Central California, summer 2020

NOAA's 27 August 2020 drought update placed Central California among the driest
areas in its 2-week EDDI map for 21 August and reported 100 active large fires
across the United States on 26 August. This documented co-occurrence makes the
season useful for retrospective exploration; it does not establish prediction
or attribution ([NOAA/NIDIS drought update][drought-update-2020]).

Given a continuous daily xarray `DataArray` of reference ET covering the
calibration period and event, compute a 14-day EDDI series for a Central
California point. This uses the xarray `DataArray` route of
`climate_indices.eddi`, which is **beta**: numerical results match the stable
NumPy API, but parameter inference, metadata, and coordinate handling may change
in a future minor release ([xarray_compatibility.md](./xarray_compatibility.md)).
Use the NumPy API when an integration cannot absorb beta interface changes.

```python
import matplotlib.pyplot as plt
import xarray as xr

from climate_indices import eddi

reference_et = xr.open_dataarray("daily_reference_et_1980_2020.nc")
central_california_et = reference_et.sel(lat=37.5, lon=-120.0, method="nearest")

eddi_14day = eddi(
    pet_values=central_california_et,
    scale=14,
    calibration_year_initial=1980,
    calibration_year_final=2019,
)

eddi_14day.sel(time=slice("2020-06-01", "2020-09-30")).plot()
plt.axhline(0, color="black", linewidth=0.8)
plt.title("Central California 14-day EDDI, June–September 2020")
plt.show()
```

Adapt coordinate names and longitude convention to the source dataset. The time
coordinate must be `datetime64` in the standard, Gregorian, or
proleptic_gregorian calendar — a `calendar` attribute naming one of those, or
no `calendar` attribute at all — and daily input must begin on January 1;
`cftime` calendars and other start dates are rejected. Given those constraints,
the xarray path infers daily Periodicity and the data start year from the time
coordinate. Keep the event outside the
1980–2019 Calibration Period, then interpret positive excursions as unusually
high demand relative to that period ([NOAA PSL][noaa-eddi]).

This example is not a numerical reproduction of NOAA's map. NOAA's product
uses FAO-56 Penman–Monteith reference evapotranspiration driven by NLDAS-2 and a
1979–present climatology; matching it requires matching those inputs and
methods ([NOAA PSL][noaa-eddi]). Compare EDDI with observed fuels, weather, and
fire records rather than treating temporal overlap as predictive skill
([Drought.gov][drought-fire]).

## Fire-index roadmap

`climate_indices.fire` currently provides Fosberg FFWI, a surface-weather
index; HDW, which combines vapor-pressure deficit and wind in the lowest
500 m above ground ([NCEP][ncep-fire]; [Srock et al. (2018)][srock-2018]);
KBDI, the cumulative moisture-deficit index for forest-fire control
([Keetch and Byram, 1968][kbdi]); the Haines Index, a lower-atmosphere
stability and moisture diagnostic for potential large-fire growth
([National Weather Service][nws-haines]); and the CFFWIS moisture codes and
behavior indices FFMC, DMC, DC, ISI, BUI, FWI, and DSR, the latter available
together through the `fire.cffwis()` orchestrator ([Natural Resources
Canada][nrcan-fwi]). KBDI is also exposed through the existing command line as
`process_climate_indices --index kbdi`, from daily precipitation and maximum
temperature inputs (see :doc:`index`).

Haines and HDW are both lower-atmosphere diagnostics rather than
fuel-moisture or drought measures, but only HDW includes wind, which is why
HDW was developed: where the two disagree, the difference is usually what the
wind is doing. Both are computed from standard reanalysis fields, HDW from a
vertical profile and Haines from the pressure levels its variant names.

The [fire-family epic #793][fire-epic] tracks the rest of the family: an ERA5
CONUS demonstration notebook (#811) and the scope decision on where the package
stops relative to NFDRS and operational fire-behavior modeling (#813). The
performance benchmarks for the recursive indices are published below.

These remain meteorological and climatological indices. Epic #793
explicitly excludes ignition probability, operational fire behavior, and fire
occurrence modeling.

## Performance and sizing

The recursive fire indices share one execution shape: a daily loop carried
forward in time, with every spatial cell updated in the same vectorized step
(`docs/adr/0006-fire-recursive-state-and-execution.md`). Time is therefore not a
chunkable dimension, and cost is best measured in cell-days: grid cells times
days of record.

Measured throughput on a development machine (Apple silicon, macOS, CPython
3.14.7, single process, pure NumPy path, best of three runs):

| Index | Grid | Record | Throughput |
| --- | --- | --- | --- |
| CFFWIS, all seven outputs | 256 x 256 cells | 365 days | 7.4 M cell-days/s |
| CFFWIS, all seven outputs | 1000 x 1000 cells | 30 days | 7.6 M cell-days/s |
| KBDI | 256 x 256 cells | 365 days | 42 M cell-days/s |
| KBDI | 1000 x 1000 cells | 30 days | 43 M cell-days/s |

Small grids run slower per cell-day, from 1.2 M cell-days/s at 8 x 8 cells to
5.7 M at 32 x 32 cells, because the per-day vectorized work drops toward NumPy's
call overhead. The table above is the regime a gridded run lives in.

Reproduce the throughput rows with `pytest tests/test_benchmark_fire.py -m
benchmark --benchmark-enable`, setting `BENCHMARK_FIRE_GRID_SIDES` and
`BENCHMARK_FIRE_RECORD_DAYS` to the grid and record of each row.

At a CONUS extent on a 0.25 degree grid, roughly 25 to 50 degrees north by 125
to 66 degrees west, about 100 x 237 cells, and 14,610 days of record over 40
years, CFFWIS covers about 346 M cell-days. At the measured throughput that is
**approximately one minute** of recurrence compute, with KBDI at about 8
seconds. Those two figures are linear extrapolations from the table above, not
measurements: throughput is close to flat between the two large grid sizes, so a
linear model in cell-days is what the measurements support.

Memory, not time, is the binding constraint at that scale. Every retained output
keeps a full daily history per cell, so 23,700 cells over 14,610 days is 2.7 GB
per field, and the default seven outputs plus four inputs come to about 30 GB of
retained data. Peak RSS for one CFFWIS xarray run in a fresh process, 128 x 128
cells (29.9 M cell-days), same machine:

| Record | Spatial chunk | Peak RSS | Modeled retained history |
| --- | --- | --- | --- |
| 1825 days | 32 x 32 | 5.3 GB | 2.6 GB |
| 1825 days | 64 x 64 | 6.8 GB | 2.6 GB |
| 1825 days | single 128 x 128 | 5.5 GB | 2.6 GB |
| 3650 days | 64 x 64 | 11.4-12.2 GB | 5.3 GB |

The modeled column is cells x days x 8 bytes per field, counting the seven output
histories and the four input histories. Reproduce the memory rows with the same
command and `BENCHMARK_FIRE_MEMORY_GRID_SIDE`, `BENCHMARK_FIRE_MEMORY_RECORD_DAYS`,
and `BENCHMARK_FIRE_CHUNK_SIDES` set to the row's grid, record, and chunk side;
the output-selection comparison runs at the last configured chunk side. Measured
peak runs about 2 to 2.5 times
that model, because the path holds copies of the inputs and outputs alongside the
histories, and peak RSS is not monotone in the spatial chunk size. Three durable
facts follow. Peak memory grows with the record length, and no chunk size changes
that. `outputs=` cuts the retained data without a matching cut in peak RSS at
these scales: 6.8 GB with all seven outputs against 6.3 GB with `fwi` alone at
1825 days, so treat it as a reduction in retained volume rather than a
demonstrated peak-memory lever. And a multi-decade CONUS run should be sized from
the modeled column times that factor, working one time-continuous spatial block
at a time.

`numba` is not an optional dependency of this package. ADR-0006 requires
representative benchmark evidence before one is added, and the throughput table
above is that evidence as measured today: no agreed runtime target is missed, so
the pure NumPy recurrence remains the implementation.

The budget guards behind these numbers live in `tests/test_benchmark_fire.py`,
run by the benchmarks workflow on every pull request. They fail once the
recurrence costs about 1.7 times more per cell-day, once the orchestrator costs
more than 1.1 times the chained code calls it replaces, or once peak RSS leaves
the coarse bound around the modeled footprint. The default test suite covers
those guards' failure paths deterministically.

## Sources

- [Drought.gov: Drought and Wildfire][drought-fire]
- [NOAA Physical Sciences Laboratory: Evaporative Demand Drought Index][noaa-eddi]
- [Hobbins et al. (2016), *The Evaporative Demand Drought Index. Part I*][hobbins-2016]
- [NOAA/NIDIS: 2020 Drought Update][drought-update-2020]
- [Keetch and Byram (1968), *A Drought Index for Forest Fire Control*][kbdi]
- [Natural Resources Canada: Canada's Fire Weather Index System][nrcan-fwi]
- [NCEP GRIB2 fire-weather parameters][ncep-fire]
- [Srock et al. (2018), *The Hot-Dry-Windy Index*][srock-2018]
- [National Weather Service: Haines Index][nws-haines]

[drought-fire]: https://www.drought.gov/topics/fire
[noaa-eddi]: https://psl.noaa.gov/eddi/
[hobbins-2016]: https://doi.org/10.1175/JHM-D-15-0121.1
[drought-update-2020]: https://www.drought.gov/news/2020-drought-update-look-drought-across-united-states-15-maps
[kbdi]: https://research.fs.usda.gov/treesearch/40
[nrcan-fwi]: https://natural-resources.canada.ca/forests-forestry/wildland-fires/canada-fire-weather-index-system
[ncep-fire]: https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-2-4.shtml
[srock-2018]: https://doi.org/10.3390/atmos9070279
[nws-haines]: https://forecast.weather.gov/glossary.php?word=haines+index
[fire-epic]: https://github.com/monocongo/climate_indices/issues/793
