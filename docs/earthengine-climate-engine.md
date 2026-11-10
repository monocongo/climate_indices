# Use Earth Engine and Climate Engine data

This guide shows how to read Google Earth Engine collections and Climate Engine
API time series and pass them to `climate_indices`. It covers installation,
authentication, the unit and calendar traps, quota discipline, and the terms of
use you accept by reading either service.

The optional extra is not a dependency of the package. Nothing in
`climate_indices` imports `xee`, `earthengine-api`, or the Climate Engine API.
Your code does, and it hands the arrays to this library.

## Before you start

- A Google account, and a Google Cloud project **registered** for Earth Engine.
  An unregistered project fails with a 403 naming the project; see
  [Troubleshooting](#troubleshooting).
- A decision on the plan: Earth Engine is free for noncommercial activities and
  paid for commercial use. Read the terms before you build on it; the
  [Terms of Use](https://earthengine.google.com/terms/) distinguish
  noncommercial activities from selling a product or charging a fee for
  applications or data created with the service, and the
  [Additional API Terms](https://developers.google.com/earth-engine/reference/Additional.API.Terms)
  restrict the API to development, research, and educational purposes.
- For the Climate Engine API, a key. There is no self-service sign-up: keys are
  issued by the Climate Engine team through
  [their request form](https://forms.gle/ekVsexF6qU4cqWYv6), and requests are
  usually answered within a few days. A key is valid for 60 days and is renewed
  by filling out the same form again.

## Install

```bash
uv sync --extra earthengine
# or
pip install "climate_indices[earthengine]"
```

This installs `xee` and `earthengine-api`. `xee` is the xarray backend; the
Earth Engine client it drives needs OAuth2 credentials, not an API key.

## Authenticate

Choose one route.

**Interactive, once per machine.** This opens a browser and writes
`~/.config/earthengine/credentials`:

```bash
uv run --extra earthengine earthengine authenticate
```

**Service account, for unattended runs.** Create a key for a service account in
the registered project, then point two variables at it:

```bash
export EE_SERVICE_ACCOUNT="pilot@my-project.iam.gserviceaccount.com"
export EE_PRIVATE_KEY_FILE="$HOME/.config/earthengine/service-account.json"
```

**Then name the project.** `EE_PROJECT` is this repository's convention, not a
Google variable: the client takes the project as an argument.

```bash
export EE_PROJECT="my-registered-project"
```

Verify before reading any data:

```bash
uv run --extra earthengine python -c "
import ee, os
ee.Initialize(project=os.environ['EE_PROJECT'],
              url='https://earthengine-highvolume.googleapis.com')
print(ee.Number(1).getInfo())"
```

`1` means both the credential and the project work. Never commit a credential,
print one in a notebook, or paste one into an issue.

## Read a collection with xee

`xr.open_dataset(..., engine="ee")` needs a grid, and the grid decision matters
more than it looks. The native grid of a collection covers its whole footprint,
not your area, and reading it whole is wasteful:

```python
import ee
import xarray as xr
from xee import helpers

ee.Initialize(project="my-registered-project",
              url="https://earthengine-highvolume.googleapis.com")

collection = (
    ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
    .filterDate("2020-01-01", "2021-01-01")
    .filterBounds(ee.Geometry.Rectangle([-85.7, 29.5, -83.3, 35.0]))
    .select(["pr", "eto"])
)

# The native grid, then crop it to the area instead of resampling.
grid = dict(helpers.extract_grid_params(collection))
scale_x, _, origin_x, _, scale_y, origin_y = grid["crs_transform"]
dataset = xr.open_dataset(collection, engine="ee", **grid)
```

`scripts/earthengine_climate_engine_pilot.py` shows the full crop:
`crop_grid_to_bbox()` keeps the source pixel edges and returns the smallest
aligned window, which is what lets you compare two collections cell by cell
without resampling either one.

## Aggregate on the server, not on the client

A daily collection over 45 years holds about 16,000 images. Opening all of them
in xarray means 16,000 requests, and a monthly index does not need them. Sum
each calendar month inside Earth Engine and read the result:

```python
def monthly_sum(collection, year, month):
    start = ee.Date.fromYMD(year, month, 1)
    last_day = start.advance(1, "month").advance(-1, "day")
    return (
        collection.filterDate(start, last_day.advance(1, "day"))
        .sum()
        .set("system:time_start", last_day.millis())
    )
```

That turns 16,000 images into about 540. Xee's own guidance is the same:
consolidate work server-side before opening the collection.

Quota discipline, in order of effect:

1. Crop the grid to the area you need.
2. Aggregate time steps server-side.
3. Select only the bands you use; a drought collection can carry 26.
4. Read one band at a time rather than materializing all of them.
5. Start with modest Dask parallelism, and only raise it if you see no 429s.

## Units and calendars

The catalog states the units; the client does not carry them into the xarray
attributes. Read them from the catalog and assert them in your code.

| Collection | Band | Catalog unit | Trap |
| --- | --- | --- | --- |
| `IDAHO_EPSCOR/GRIDMET` | `pr` | mm, daily total | monthly sums are mm per month |
| `IDAHO_EPSCOR/GRIDMET` | `eto` | mm, daily grass reference ET | a monthly sum is mm per month |
| `IDAHO_EPSCOR/GRIDMET` | `etr` | mm, daily alfalfa reference ET | not the same crop as `eto` |
| `IDAHO_EPSCOR/GRIDMET` | `tmmn`, `tmmx` | **kelvin** | convert before any temperature index |
| `IDAHO_EPSCOR/GRIDMET` | `vpd` | kPa | not hPa |
| `GRIDMET/DROUGHT` | `spi*`, `spei*`, `eddi*` | standard deviations | pentad cadence, not monthly |

Two checks catch most silent mistakes:

- Assert the domain-mean monthly total is plausible in millimetres. A
  metres-versus-millimetres error is three orders of magnitude low, and a
  per-day value that was never summed is roughly thirty times low.
- Assert the time coordinate has one value per month and no gaps. A calendar
  month sum built server-side can silently return a duplicate or a missing step.

`climate_indices` takes values in millimetres per month for SPI and SPEI, and
millimetres per month of reference ET for EDDI. Both gridMET bands are already
in millimetres per day, so a monthly sum is the input the library expects.

## Climate Engine API, standard index series

The API returns a time series for a point or polygon. Request a key with the
form linked in [Before you start](#before-you-start), then read it from the
environment: `export CE_API_KEY=...`. It goes in the `Authorization` header,
never in the URL, never in a notebook output, and never in a file you commit.

A key is valid for 60 days. When it expires the API returns an authorisation
error, not a data error; renew it with the same form.

```bash
curl -X POST "https://api.climateengine.org/timeseries/standard_index/coordinates" \
  -H "Authorization: $CE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "coordinates": "[[-84.9, 30.7]]",
        "area_reducer": "mean",
        "dataset": "GRIDMET",
        "variable": "spi",
        "distribution": "nonparametric",
        "accumulation": "90",
        "start_year": 1981,
        "end_year": 2016,
        "start_date": "1981-01-01",
        "end_date": "2024-12-31"
      }'
```

| Parameter | Meaning |
| --- | --- |
| `variable` | `spi`, `spei`, `eddi`, `speih`, `eddih` |
| `distribution` | `gamma` (maximum likelihood), `loglogistic` (L-moments), `nonparametric` (Tukey plotting positions) |
| `accumulation` | the accumulation period **in the dataset's own unit**; days for the daily `GRIDMET` dataset, months for `PRISM_MONTHLY_4km` |
| `start_year`, `end_year` | the calibration window |
| `start_date`, `end_date` | the returned window, independent of the calibration |

The endpoint refuses a precomputed drought product as its input: use
`GRIDMET`, not `GRIDMET_DROUGHT`, as the dataset.

Pass the returned values to the NumPy API. `climate_indices.spi()` and
`climate_indices.spei()` accept a 1-D array; there is no wrapper that reads the
API for you, and none is planned until a pilot shows one is worth having.

## Terms, licences and citation

Reading a collection makes you responsible for its terms. The ones this guide
touches:

- **Earth Engine** is free for noncommercial activities and paid for commercial
  use. See the [Terms of Use](https://earthengine.google.com/terms/) and the
  [noncommercial](https://earthengine.google.com/noncommercial/) page. A project
  registered as noncommercial must re-verify periodically.
- **gridMET**, and the `GRIDMET/DROUGHT` indices derived from it, are in the
  public domain per their catalog pages. Cite Abatzoglou, J. T. (2012),
  *Development of gridded surface meteorological data for ecological
  applications and modelling*, International Journal of Climatology,
  [doi:10.1002/joc.3413](https://doi.org/10.1002/joc.3413), and note the date you
  acquired the data.
- **Climate Engine** output is governed by its own terms and citation guidance.
  Review them on [climateengine.org](https://climateengine.org/) before you
  redistribute a derived product.
- The **example scripts and notebooks in this repository** are covered by the
  repository licence, not by the data licences above. A derived product carries
  both.

## Troubleshooting

**`Extra earthengine is not defined in the optional-dependencies table`**
You are in a checkout without the extra. Run the command from the repository
root of the branch that defines it.

**`Project <id> is not registered to use Earth Engine`**
The credential is fine and the project is not. Register it at
`https://code.earthengine.google.com/register?project=<id>` or
`https://console.cloud.google.com/earth-engine/configuration?project=<id>`,
and enable the Earth Engine API for it.

**`API keys are not supported by this API. Expected OAuth2 access token`**
A Google Cloud API key cannot authenticate Earth Engine. Run
`earthengine authenticate`, or use a service account.

**`Please authorize access to your Earth Engine account`**
No stored credential was found. Run `earthengine authenticate` as the same user
that runs the script.

**`Connection pool is full, discarding connection`**
A urllib3 warning from parallel reads, not an error. Lower Dask parallelism if
it appears with 429 responses.

**A monthly series is missing a month, or holds two**
Check the count before computing anything. The pilot asserts one step per month
and fails loudly, because a silent shift there changes every index value.

## Related

- {doc}`standardized-hydrologic-indices` for turning a series into an index.
- {doc}`xarray_migration` for the xarray API's argument handling.
- {doc}`troubleshooting` for errors raised by the library itself.
- `docs/research/earthengine-climate-engine-pilot.md` for the measured
  agreement with the precomputed Earth Engine drought collection.
