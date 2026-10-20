# ERA5 cache for the energy tutorial

`energy_era5_1991_2020.nc` is an input cache, not an independent numerical
reference or an energy-generation dataset. The adjacent `.provenance.json`
records its SHA-256, retrieval date, source URL, size, variables, and selection.
`notebooks/energy_demo.ipynb` verifies the checksum before reading the file.

## Source and modifications

The producer is ECMWF. The source is the anonymously readable
[Google Research ARCO-ERA5 redistribution](https://github.com/google-research/arco-era5):

```text
https://storage.googleapis.com/gcp-public-data-arco-era5/ar/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr
```

The store name is an identifier, not a guarantee of its time coverage. Its time
coordinate actually ends in December 2021; this cache uses only 1991–2020.
The underlying product is
[ERA5 single-level reanalysis](https://doi.org/10.24381/cds.adbb2d47).

- Select six-hourly samples at 00, 06, 12, and 18 UTC over 1991–2020.
- Select latitudes 48, 51, and 54°N and longitudes 3, 6, 9, and 12°E.
- Keep every second point of the upstream 1.5° conservatively regridded field.
  This is spatial subsampling, not area averaging or country aggregation.
- Round coordinate labels to exact degrees after selection with a 0.01° tolerance.
- Rename `2m_temperature` to `temperature_k` (K) and `10m_wind_speed` to
  `wind_speed_10m` (m s-1). Preserve the float32 meteorological values.
- Store dimensions as `(time, latitude, longitude)` with lossless compression.
  There is no temporal averaging or capacity-factor calculation in this cache.

ARCO supplies wind speed, not the magnitude of daily mean wind components.
This store omits the wind unit attribute; the source variable is interpreted
as m s-1. Preparation rejects conflicting source units and requires K and
m s-1 in annual caches before reuse.
The notebook applies the wind profile and nonlinear curve at each six-hourly
sample, then averages capacity factors to daily values. Temperature becomes a
UTC daily mean before the mean-based degree-day calculation. Six-hour samples
cannot recover hourly variability. Upstream spatial regridding also limits the
nonlinear transform. No bias correction, terrain correction, observed load,
installed capacity, or plant losses are supplied.

## Data terms and attribution

The data terms are separate from the repository's software licence. The
[ERA5 catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
lists CC-BY-4.0. The
[ECMWF attribution guidance](https://confluence.ecmwf.int/pages/viewpage.action?pageId=673339614)
permits copying and adaptation with credit, identification of the delivery
service, a licence reference, and a statement of modification.
The [ARCO data-use FAQ](https://github.com/google-research/arco-era5#can-i-use-the-data-for-researchcommercial-purposes)
also directs users to the Copernicus data terms and acknowledgement.
Its Apache software licence does not replace the data terms.

Modified ECMWF ERA5 reanalysis data, © European Centre for Medium-Range Weather
Forecasts (ECMWF), accessed through Google Research's ARCO-ERA5 redistribution.
Licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
Contains modified Copernicus Climate Change Service information 2022.
Neither the European Commission nor ECMWF is responsible for any use that may
be made of the Copernicus information or data it contains.

Cite Hersbach et al. (2020), *The ERA5 global reanalysis*,
[doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803), the
[single-level dataset DOI](https://doi.org/10.24381/cds.adbb2d47), and ARCO-ERA5.
Retain this attribution and the modification record when redistributing the cache.

## Regenerate

Run from the repository root:

```bash
uv sync --group dev
uv run --group dev scripts/prepare_energy_demo_inputs.py
```

The first run reads roughly 2 GB of global time chunks, despite the small
selected output. The script caches validated annual subsets under the ignored
`data/energy-demo/arco-cache/` directory. Later runs reuse them. CI does not run
the preparation script and needs no account or token. Preparation stages
complete files before replacement and rolls back a failed provenance replacement.
The two output files are not one filesystem transaction. After abrupt process
termination, a mismatched checksum prevents notebook execution; regenerate the
pair before use.

To request finer hourly ERA5 through CDS, follow the notebook's separate CDS API instructions; those
requests do not reproduce this coarse ARCO cache byte for byte.
