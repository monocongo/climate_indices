# Interactive climate-index explorer landscape

**Research date:** 2026-08-08  
**Revised:** 2026-09-10 — Climate Engine added; the executive conclusion and recommendation are superseded, see [Decision](#decision-2026-09-10-shelve-the-explorer).  
**Scope:** Primary sources only: official documentation, first-party repositories/source, project-owned demos, and first-party service terms. Repository links are pinned where practical.

## Executive conclusion

The statement **“this exploration is unavailable in other tools” is not accurate**.

The exploration pattern already exists for precomputed cubes:

- **xcube Viewer** shows a selected geographic time slice, reports the current value on hover, supports time navigation, and extracts a time series when a point place is added or selected. [XV-ANALYSE]
- **ncWMS2/Godiva3** shows a selected time slice and, on map click, reports the coordinate/value and links directly to a time-series plot. [NC-USAGE]
- **Lexcube** is a close, non-map 3-D alternative: hovering reports a pixel and clicking the front cube face plots that location's time series, although only over the currently visible time extent. [LEX-README]
- **Google Earth Engine** supplies map-click callbacks, image-collection time-series charts, deferred server execution, and hosted Apps, so the interaction can be assembled there with custom code. [EE-CLICK] [EE-CHART] [EE-DEFERRED] [EE-APPS]

The original review then claimed that no turnkey application exposes the exact `climate_indices` scientific controls—SPI, scale 6, gamma distribution, and explicit calibration years—over a computed gridded result with linked map/time-series exploration.

**That claim was wrong, and the 2026-08-08 review missed the tool that disproves it.** Climate Engine's public API computes SPI on demand with a caller-supplied `distribution` and a caller-supplied `start_year`/`end_year` standardization period, distinct from the display window, and its viewer provides the linked map and point series. [CE-API-DIST] [CE-API-SPEC] The reviewed candidate list omitted Climate Engine entirely, so the "not found" finding reflected the gap in the survey rather than a gap in the field.

What Climate Engine does **not** offer is narrower and concrete: Pearson Type III, user-supplied datasets and grids, self-calibrating Palmer, and the fire-weather family. See [Climate Engine](#climate-engine) below.

The 2026-08-08 review concluded from the above that the project should not build another generic cube viewer, and
proposed the three options below. **Those options are superseded** — see
[Decision (2026-09-10)](#decision-2026-09-10-shelve-the-explorer). They are retained to show what was considered.

1. Test a **thin Panel/HoloViz reference application** first, because it can call the existing typed xarray API directly and preserve numerical identity with `climate_indices`. This is the most direct MVP hypothesis, not a proven low-effort production path: job control, caching, persistence, and deployment remain application work. The current xarray path accepts the index parameters and parallelizes spatial cells through lazy Dask-backed `xarray.apply_ufunc`; it requires the full time dimension in one chunk and is explicitly marked beta. [CI-SPI] [CI-DASK] [CI-XARRAY-GUIDE]
2. In parallel, run a bounded **xcube integration/upstream spike**. xcube Viewer already owns nearly all of the exploration UX, while xcube Server has dynamic xarray datasets and parameter-schema-based compute jobs whose results become datasets. [XC-WEBAPI] [XC-DYNAMIC] [XC-COMPUTE-ROUTES] [XC-COMPUTE-CONTEXT] The likely upstream contribution is a generic Viewer form for compute-operation schemas plus a documented external operation-registration seam—not climate-specific controls in Viewer core.
3. Use **Xpublish** if a standards-oriented backend or multiple clients become requirements. It is a strong xarray/Dask/FastAPI plugin substrate, with tile/WMS and EDR position endpoints, but it is not a finished viewer. [XP-README] [XP-PLUGINS] [XP-TILES] [XP-EDR]

## What counts as the proposed workflow

The proposed product combines three layers that should not be conflated:

1. **Scientific derivation:** fit and transform an index using parameters such as scale, distribution, and calibration period at every grid cell.
2. **Whole-cube orchestration:** execute that computation across all spatial cells, with progress, cancellation, cache/persistence, provenance, and failure handling.
3. **Exploration:** map one time slice, navigate time, hover/click a location, and plot the selected point's full time series.

`climate_indices` already provides the scientific kernel and a beta xarray/Dask path for layer 1 and the computational core of layer 2. [CI-SPI] [CI-DASK] Existing viewers predominantly solve layer 3 for a precomputed or logically derived cube. This distinction explains why the map interaction is already common while the exact end-to-end workflow is not turnkey.

## Capability matrix

Legend: **Yes** = built in; **Custom** = supported building blocks but application/plugin code is required; **Precompute** = derivation must happen before viewing; **Partial** = material limitation, including a lazy logical cube where durable full materialization is unverified; **No evidence** = not established in the reviewed first-party material; **API only** = backend primitive without the linked UI. Compound labels such as **Custom/Partial** and **Yes/Partial** retain both meanings.

### Target-workflow capabilities

| Candidate | Exact parameterized `climate_indices` computation | On-demand whole-grid computation / durable result | Geographic slice + time navigation | Hover/click → full point series | Generic viewer or scientific engine? |
|---|---|---|---|---|---|
| **Panel + HoloViz** | **Custom**: app calls the typed API [CI-SPI] | **Custom/Partial**: lazy spatial Dask is available; durable full materialization, persistence, and cache remain app work [CI-DASK] [CI-XARRAY-GUIDE] | **Custom** with xarray/hvPlot/GeoViews [HVPLOT-GRID] [HVPLOT-GEO] [GEOVIEWS] | **Custom** with HoloViews streams [HV-TAP] | Application framework and visualization stack, not an index engine [PANEL] [DATASHADER] |
| **xcube + xcube Viewer** | **Custom**: operation required; no SPI operation supplied [XC-COMPUTE-OPS] | **Custom/Partial**: compute jobs can return xarray datasets; durable materialization is not supplied by the job context [XC-COMPUTE-CONTEXT] | **Yes** [XV-ANALYSE] | **Yes** [XV-ANALYSE] | Cube server plus finished viewer; closest reuse target [XC-WEBAPI] [XV-INDEX] |
| **Climate Engine** | **Yes/Partial**: on-demand SPI/SPEI/EDDI with caller-chosen `distribution` (`gamma`, `loglogistic`, `nonparametric`) and `start_year`/`end_year` standardization period; no Pearson III and no user datasets [CE-API-DIST] [CE-API-SPEC] | **Yes for its own catalog**: Earth Engine executes and serves the result; the caller does not own or persist a cube [CE-API-SPEC] | **Yes** in the hosted app and API [CE-APP] | **Yes** via the app and point/statistics endpoints [CE-API-SPEC] | Hosted drought-index service with a finished viewer [CE-HOME] |
| **NASA Giovanni** | **No evidence** of a user algorithm path; official UI offers predefined analyses [GIO-EARTHDATA] [GIO-MANUAL] | **Partial**: whole-grid outputs exist for supported NASA analyses, not arbitrary `climate_indices` [GIO-MANUAL] | **Partial**: maps and animation are supported [GIO-MANUAL] | **No evidence** of linked cell series; documented time series spatially average the selected area [GIO-MANUAL] | Hosted Earth-science analysis service |
| **Google Earth Engine** | **Custom**: reimplementation rather than direct reuse of local Python `climate_indices`; no first-party built-in equivalent was found [EE-MAP] [EE-MATH] [EE-ERFINV] | **Custom/Partial**: deferred server collections and export primitives are available, but durable full-cube execution was not validated [EE-DEFERRED] [EE-EXPORT] [EE-TERMS] | **Custom** app [EE-APPS] | **Custom** app using first-party click and chart APIs [EE-CLICK] [EE-CHART] | Programmable hosted geospatial compute service |
| **ncWMS2 + Godiva3** | **Precompute**; no general scientific derivation API [NC-CONFIG] [NC-DEVELOP] | **No evidence** of scientific cube computation [NC-CONFIG] [NC-DEVELOP] | **Yes** [NC-USAGE] | **Yes** [NC-USAGE] | WMS server/viewer for existing CF data [NC-README] |
| **Lexcube** | **Precompute** [LEX-README] | **No evidence** of scientific cube computation [LEX-README] | **Partial**: 3-D cube faces and range controls, not a conventional projected map [LEX-README] | **Partial**: click gives a series over the visible time range [LEX-README] | Jupyter 3-D cube widget |
| **Pan3D** | **Precompute**; VTK calculator expressions are not calibration-period fitting [PAN-COMPUTED] | **No evidence** of a climate-index whole-cube workflow [PAN-COMPUTED] [PAN-ANALYTICS] | **Yes/Partial**: globe and time controls, oriented toward 3-D exploration [PAN-EXPLORERS] [PAN-VIEWER] | **No evidence** of direct point linking; Analytics plots are zonal/global/temporal aggregates over the current slice selection [PAN-ANALYTICS] | xarray-to-VTK/trame viewers and explorers |
| **Xpublish + plugins** | **Custom**: plugin can call `climate_indices` [XP-PLUGINS] | **Custom/Partial**: server-side Dask is supported, but whole-result jobs/materialization must be designed [XP-README] | **API only**: tiles/WMS with dimension selection; frontend required [XP-TILES] | **API only**: EDR position accepts point and datetime selections; frontend and full-series behavior require validation [XP-EDR] | Extensible xarray/FastAPI backend, not a viewer [XP-README] |

### Engineering and reuse fit

| Candidate | License / maintenance evidence | Python/xarray and lazy fit | Local / hosted modes | Dynamic-variable or plugin path | Deployment and realistic reuse path |
|---|---|---|---|---|---|
| **Panel/HoloViz** | BSD-style licenses; the stack's cited repositories had 2026 activity at review time [PANEL-LICENSE] [HVPLOT-LICENSE] [GEOVIEWS-LICENSE] [HV-LICENSE] [DATASHADER-LICENSE] [HOLOVIZ-MAINT] [HVPLOT-MAINT] [GEOVIEWS-MAINT] [HV-MAINT] [DATASHADER-MAINT] | Excellent Python/xarray fit; hvPlot supports xarray and Dask, Datashader handles scalable rasterization [HVPLOT] [DATASHADER] | Notebook/local Python and self-hosted server [PANEL] | Arbitrary Python callbacks; app owns validation, jobs, cache, and provenance [PANEL] | `panel serve`; most direct exact-MVP candidate, but effort must be validated and climate_indices would own the application |
| **xcube/Viewer** | MIT; both repositories had 2026 activity [XC-LICENSE] [XV-LICENSE] [XC-MAINT] [XV-MAINT] | Excellent xarray/Zarr/Dask fit [XC-README] | Local or self-hosted xcube Server plus browser SPA [XC-SERVE] [XV-INDEX] | User expressions, configured dynamic datasets, and compute-operation registry/jobs [XV-USER-VARS] [XC-DYNAMIC] [XC-COMPUTE-ROUTES] | Piggyback on Viewer; contribute generic operation UI/registration seam upstream; keep climate operation external |
| **Giovanni** | NASA service is public; the public source repository's README says pull requests are not maintained, and no reuse license was identified in its root [GIO-README] [GIO-REPO] | Poor fit: service implementation is not a Python/xarray/Dask extension surface [GIO-README] | NASA-hosted service; practical local integration path is not documented [GIO-EARTHDATA] | Fixed service/algorithm catalog; no reviewed user plugin/upload path [GIO-MANUAL] | Use as precedent or external service, not as the base for this application |
| **Earth Engine** | Hosted proprietary service terms; client libraries and Xee are Apache-2.0, and their repositories had 2026 activity [EE-TERMS] [EE-CLIENT-LICENSE] [XEE-LICENSE] [EE-MAINT] [XEE-MAINT] | Server-side deferred graph, not Dask; Xee can expose Earth Engine collections as lazy xarray and work with Dask [EE-DEFERRED] [XEE] | Hosted compute and hosted Apps; Python clients can run locally but call the service [EE-APPS] [EE-CLIENT] | Arbitrary server expressions and collection mapping, but index semantics must be implemented separately [EE-MAP] [EE-MATH] | Viable only when data/service dependency and numerical reimplementation are acceptable |
| **Climate Engine** | Hosted service operated by DRI/UCSB on Google Earth Engine; no reuse license identified for the application itself [CE-HOME] [CE-ABOUT] | Service-side, not a Python/xarray/Dask extension surface; results arrive through the HTTP API [CE-API-SPEC] | Hosted only; no local deployment path identified | Fixed dataset catalog and fixed index list; no user data, expression, or plugin path found [CE-API-SPEC] | Use as an external service or as evidence of prior art, not as a base to build on |
| **ncWMS2/Godiva3** | Custom BSD-like terms require retaining the Reading e-Science Centre logo when using Godiva; the latest cited release is a 2024 snapshot after the 2022 stable release [NC-README] [NC-RELEASES] [NC-STABLE-RELEASE] | Java/NetCDF/OPeNDAP, not Python/xarray/Dask [NC-README] [NC-CONFIG] | Standalone local JAR or Java web application [NC-INSTALL] | Java data readers and styles; “dynamic services” means unindexed files, not arbitrary derived science [NC-CONFIG] [NC-DEVELOP] | Good precomputed-CF reuse where WMS is already required; unattractive computation host |
| **Lexcube** | The official README identifies the distribution as GPLv3+; version 2.0.1 was released in 2026 [LEX-README] [LEX-LICENSE] [LEX-RELEASE] | Direct 3-D xarray input; examples use chunked Zarr/NetCDF and the widget has its own aggressive touched-chunk cache [LEX-README] | Jupyter widget plus project-owned hosted demo [LEX-README] | No scientific plugin path documented; derive the DataArray externally [LEX-README] | Low-effort optional notebook view; not the conventional map application |
| **Pan3D** | Apache-2.0; version 1.3.1 was released in 2026 [PAN-LICENSE] [PAN-RELEASE] | Strong xarray input; sources open lazily, but selected arrays are converted to NumPy/VTK for rendering [PAN-VIEWER] [PAN-MATERIALIZE] | Local/self-hosted trame server, Jupyter, Binder, and repository Docker assets [PAN-README] [PAN-COMMAND] [PAN-DOCKER] | Modular explorers and simple VTK calculated fields [PAN-README] [PAN-COMPUTED] | Reuse as a 3-D component only if that UX is required; point-link feature would still be new work |
| **Xpublish** | Apache-2.0 core; core, tiles, and EDR repositories had 2026 activity [XP-LICENSE] [XP-MAINT] [XP-TILES-MAINT] [XP-EDR-MAINT] | Excellent xarray/DataTree/FastAPI fit with server-side Dask [XP-README] | Local or self-hosted API [XP-README] | First-class local and entry-point plugins; dynamic dataset providers are documented [XP-PLUGINS] | Best reusable backend/plugin path; still requires Panel/JavaScript/other frontend and production job controls |

### Compute orchestration and operational ownership

| Candidate | Trigger, status, cancellation, and failures | Persistence, cache, and provenance | Practical ownership |
|---|---|---|---|
| **Panel/HoloViz app** | **Custom**: widgets/callbacks can trigger computation, but the application must define job status, progress, cancellation, and error presentation [PANEL] | **Custom**: no domain result store or provenance policy comes from the visualization framework [PANEL] | `climate_indices` deployment owns the complete operational layer |
| **xcube** | **Partial**: REST routes schedule/list/get/cancel jobs; the context records scheduled/started/completed/failed/cancelled states and traceback data [XC-COMPUTE-ROUTES] [XC-COMPUTE-CONTEXT] | **Partial**: the result is registered as a dataset, but jobs are in-process dictionaries executed by a local thread pool; durable queue/result storage is not supplied there [XC-COMPUTE-CONTEXT] | Strongest existing orchestration skeleton, but production durability remains deployment/extension work |
| **Giovanni** | **Yes for its hosted workflows**: Plot Data shows processing steps, and failures can be captured for feedback [GIO-MANUAL] | **Partial**: the service exposes result history and data lineage, but no reviewed extension path lets `climate_indices` own or customize those facilities [GIO-MANUAL] | NASA operates the workflow; unsuitable as a local orchestration component |
| **Earth Engine** | **Yes/Partial for the service**: interactive requests are deferred; asynchronous batch tasks have lifecycle, monitoring, cancellation, and failure states [EE-DEFERRED] [EE-PROCESSING] | **Partial**: exports and service caching/quotas exist, while application-specific parameter keys and scientific provenance remain custom [EE-EXPORT] [EE-TERMS] | Google operates compute; the application owns index semantics and provenance |
| **Climate Engine** | **Yes for its hosted workflows**: the service schedules and returns results; the caller has no job model to own [CE-API-SPEC] | **Service-owned**: caching, quotas, and dataset versioning belong to the operator; no caller-side provenance contract | The operator owns computation entirely; unsuitable as a component, useful as prior art |
| **ncWMS2, Lexcube, Pan3D** | **Precompute**: reviewed viewer workflows do not orchestrate the target index computation [NC-USAGE] [LEX-README] [PAN-VIEWER] | Derived-output persistence, caching, provenance, and failures belong to the external precompute pipeline | Use only after another component has completed and stored the index cube |
| **Xpublish** | **Custom/Partial**: a plugin or route can trigger work, but no domain job/progress/cancellation system is documented in core [XP-PLUGINS] | Core exposes a configurable in-process cache; durable result storage and scientific provenance remain plugin/deployment work [XP-REST] | Good API substrate, not a complete computation service |

## Per-candidate findings

### Panel, HoloViews, hvPlot, GeoViews, and Datashader

Panel is an application framework: it combines Python widgets, reactive/callback APIs, plots, and server deployment. [PANEL] hvPlot directly supports xarray and Dask; multidimensional grouping can generate widgets, while its geographic support uses GeoViews for projected elements and related mapping features. [HVPLOT] [HVPLOT-GRID] [HVPLOT-GEO] [GEOVIEWS] HoloViews `Tap` streams expose selected plot coordinates to a Python callback, which is sufficient to select the nearest xarray cell and redraw a line plot. [HV-TAP] Datashader rasterizes/resamples large data for display; its own documentation describes it as a rendering stage requiring another plotting library for axes and interactivity, so it is not an index-computation engine. [DATASHADER]

This stack can implement the exact workflow without translating the scientific algorithm. The important design rule is to separate an explicit **Compute** action from map-time and location interactions: parameter changes build/persist one derived cube; moving the time slider or selecting a point only slices that result. The application must add cache keys, progress/cancellation, persistence, concurrency limits, and provenance beyond what the plotting stack provides.

**Fit:** best route to a scientifically faithful MVP and reference UX; weak argument for a large bespoke production viewer before xcube is tested.

### xcube and xcube Viewer

xcube is built around xarray, Zarr, and Dask. [XC-README] Its Web API publishes cube metadata/data and tiles, extracts time-series statistics for a geometry, and supports compute operations whose on-demand results appear as new datasets. [XC-WEBAPI] Separately, server configuration can augment a dataset with Python-computed variables or define a dynamic xarray dataset from input datasets and parameters. [XC-DYNAMIC]

Viewer already covers the target exploration well: current-value hover, time controls/player, point/polygon/circle places, automatic time-series extraction, multiple variables/places, and export. [XV-ANALYSE] Viewer “user variables” are Python-like algebra over dataset variables, NumPy constants/ufuncs, and `where`; that documented expression surface is appropriate for elementwise indices such as NDVI, not a temporal gamma fit with calibration semantics. [XV-USER-VARS]

The compute API exposes operation metadata/parameter schemas, job schedule/status/cancel endpoints, and registers the returned xarray dataset. [XC-COMPUTE-ROUTES] The current context stores jobs in process and uses a local `ThreadPoolExecutor`; it does not itself provide a durable queue or persistent result catalog. [XC-COMPUTE-CONTEXT] The reviewed Viewer source at the pinned revision contains no client for the `/compute/operations` or `/compute/jobs` endpoints, so a generic operation-submission UI is the principal missing product seam. [XV-SOURCE]

**Fit:** closest finished open-source product. Propose upstream a schema-driven compute form and documented external operation registration. A separate climate-indices extension should own SPI/SPEI controls and result persistence.

### Climate Engine

Climate Engine is a hosted service from the Desert Research Institute and UC Santa Barbara that runs Earth-observation
and climate analysis on Google Earth Engine, with a web app, an HTTP API, and automated reports. [CE-HOME] [CE-ABOUT]

Its `standard_index` endpoints compute SPI, SPEI and EDDI on demand. The API rejects an invalid distribution with the
enumeration `['gamma', 'loglogistic', 'nonparametric']`, and reports that `gamma` is valid for `spi`, while
`loglogistic` and `nonparametric` additionally serve `spei`, `eddi` and related variables. [CE-API-DIST] Those
endpoints take `start_year` and `end_year` separately from `start_date` and `end_date`, so the standardization period
is caller-supplied rather than fixed. [CE-API-SPEC] Precipitation sources include GridMET, CHIRPS, ERA5 and PRISM, and
SPEI accepts an ASCE grass-reference or Hargreaves PET method. [CE-SPEI]

This is the direct refutation of the original executive conclusion. Parameterized, on-demand gamma SPI with an
explicit calibration period and linked map/point exploration is an existing, hosted, free-to-use product.

The separate precomputed **gridMET DROUGHT** product is a different thing and should not be confused with the above:
it is CONUS-only at 4 km, standardized over a fixed 1981–2016 window for SPI/SPEI/EDDI and 1979–2018 for PDSI and the
Palmer Z index, at fixed 14-day to 5-year scales, using the non-parametric plotting-position method. [CE-GRIDMET-DROUGHT]
Reading that product's fixed window as a property of Climate Engine as a whole is the specific error this revision corrects.

What remains outside Climate Engine, from the reviewed material:

- **Pearson Type III.** The distribution enumeration contains no `pearson3`. [CE-API-DIST] Pearson III is the
  distribution NOAA NCEI uses operationally, and `climate_indices` implements it.
- **User-supplied data and grids.** The catalog is fixed; no upload, expression, or plugin path was found. [CE-API-SPEC]
- **Self-calibrating Palmer.** PDSI is served from the precomputed product on a fixed baseline, not computed from
  caller parameters. [CE-GRIDMET-DROUGHT]
- **Fire-weather indices.** KBDI, CFFWIS, Fosberg, Hot-Dry-Windy and Haines do not appear in the index list. [CE-API-DIST]
- **Library and offline use.** It is a hosted service under its own terms, not a package that runs on a laptop, in a
  notebook, or inside another pipeline.

**Fit:** the closest competitor to the proposed Explorer, and the reason to stop building one. Cite it as prior art;
position `climate_indices` on the reference-implementation, own-data, and index-coverage axes instead.

### NASA Giovanni

NASA describes Giovanni as an online service for accessing, visualizing, and analyzing Earth-science remote-sensing data without first downloading it; first-party material lists time-averaged maps, comparisons, vertical plots, and animations. [GIO-EARTHDATA] Its user guide documents real server-side analyses, including time-slice animation and area-averaged time series. The standard time series computes a spatial average over the selected area at every time step, rather than documenting a map-cell click linked to a point series. [GIO-MANUAL]

No first-party path was found for uploading an arbitrary local xarray cube, installing `climate_indices`, or registering a user SPI operation. The public source README says the software will be updated but pull requests are not maintained; the repository does not state an open-source license in the reviewed root. [GIO-README] [GIO-REPO]

**Fit:** useful evidence that hosted on-demand geoscience analysis is established, but not a realistic reuse or upstream-contribution base for this task.

### Google Earth Engine

Earth Engine scripts construct deferred server-side computation graphs; processing occurs when a result is requested. [EE-DEFERRED] `ImageCollection.map` can create a derived collection, and the image API exposes mathematical primitives including regularized incomplete gamma and inverse error functions. [EE-MAP] [EE-MATH] [EE-ERFINV] These are ingredients, not evidence of a complete SPI implementation: the reviewed first-party sources did not establish fitting and calibration behavior equivalent to `climate_indices`. Zero precipitation, missing values, fitting rules, calibration grouping, and failure behavior would all need a separate implementation and parity tests.

The exact interaction can be assembled with `ui.Map.onClick`, which passes clicked longitude/latitude to a callback, and `ui.Chart.image.series`, which plots reduced band values across an `ImageCollection` for a region. [EE-CLICK] [EE-CHART] Earth Engine Apps publish Code Editor analyses as hosted interactive applications. [EE-APPS]

The Python and JavaScript clients are Apache-2.0, and Xee is an Apache-2.0 xarray backend offering lazy, parallel pixel retrieval and Dask interoperability. [EE-CLIENT] [XEE] The compute service itself is governed by Earth Engine terms and quotas; commercial or ongoing operational use may require paid terms. [EE-TERMS]

**Fit:** strong hosted option when source data already lives in Earth Engine and service coupling is acceptable; poor choice when reusing and preserving `climate_indices` behavior is central.

### ncWMS2 and Godiva3

ncWMS2 is a Java WMS for CF-compliant NetCDF, with local files and OPeNDAP as normal sources; Godiva3 is its included browser client. [NC-README] [NC-CONFIG] Godiva supports map/time selection and explicitly documents clicking the map to obtain the point value, coordinate, and links for time-series/vertical-profile plots. The `GetTimeseries` extension accepts a `GetFeatureInfo` location and a time range and can return an image, CSV, JSON, or CoverageJSON. [NC-USAGE]

It does not supply a Python/xarray or Dask scientific execution layer. Extension points focus on palettes, SLD styles, and Java `DatasetFactory` readers. [NC-DEVELOP] Its “dynamic services” expose files without pre-indexing; they are not dynamic derived-variable computations. [NC-CONFIG]

**Fit:** precompute a CF NetCDF cube with `climate_indices` and serve it here when WMS/Godiva is already an organizational standard. Otherwise xcube is a closer Python fit.

### Lexcube

Lexcube accepts exactly three-dimensional NumPy arrays or rectangular xarray DataArrays, including local/remote NetCDF and Zarr and Xee-backed Earth Engine data. It renders the data as a 3-D cube in Jupyter and offers a project-owned web demo. [LEX-README]

Version 2's interaction is especially close: hover reports the pixel; left-clicking the front face plots that location's time series. The documented series is synchronized to the currently visible time selection, so showing the full source history requires expanding that range. [LEX-README] Computation remains external, and the front cube face is not a conventional projected 2-D map.

**Fit:** compelling supplementary notebook visualization, not the primary map application. GPLv3+ matters if code is embedded or adapted rather than merely used as an external tool. [LEX-LICENSE]

### Pan3D

Pan3D is an Apache-2.0 Python/trame package for xarray-compatible multidimensional data. XArray Viewer can load local or remote data, subset it, convert a selected time slice to VTK, and navigate time. [PAN-README] [PAN-VIEWER] Globe Explorer projects latitude/longitude data on a globe; Analytics Explorer supplies zonal, temporal, and global plots through xCDAT. [PAN-EXPLORERS]

Pan3D's calculated-field API applies VTK calculator expressions to loaded scalars/vectors; this is not a temporal distribution-fitting plugin. [PAN-COMPUTED] The current Analytics implementation computes zonal/global/time aggregates from slice ranges and does not establish a direct geographic point-pick-to-full-series path. [PAN-ANALYTICS] Lazy xarray loading helps source access, but conversion calls `to_numpy()` for selected arrays, so this is not a whole-cube Dask computation host. [PAN-MATERIALIZE]

**Fit:** consider only if 3-D globe/volume exploration is a requirement. Contributing point-probe linking upstream could be useful to Pan3D generally, but it would not solve parameterized climate-index orchestration.

### Xpublish and mapping/EDR plugins

Xpublish publishes xarray Datasets/DataTrees through FastAPI, supports server-side Dask for on-demand delivery, and explicitly advertises pluggable derived products. [XP-README] Its plugin API supports dataset routers, app routers, dataset providers, local registration, and installed entry points. [XP-PLUGINS]

The current `xpublish-tiles` project provides OGC Tiles and WMS routes for xarray datasets, including dimension/time selection. [XP-TILES] `xpublish-edr` exposes position, area, and cube queries with time selection and several response formats; a position request is a suitable backend primitive for the selected point's temporal data. [XP-EDR] Neither Xpublish core nor these plugins provide the linked browser viewer.

**Fit:** a credible `xpublish-climate-indices` plugin could validate parameters, compute/cache a derived dataset, expose map tiles, and answer point-series queries. This is the best backend contribution path if API reuse matters, but more moving parts than a Panel-only MVP.

## Build-versus-integrate options

### Option A — Thin Panel/HoloViz application

**Choose when:** validating user value and scientific controls quickly.

- Reuses the exact typed `climate_indices` API and Dask path. [CI-SPI] [CI-DASK]
- Provides complete UX freedom using Python callbacks and geographic xarray plots. [PANEL] [HVPLOT] [HV-TAP]
- Requires ownership of application state, compute jobs, progress, cache/materialization, authentication, and deployment.

**Assessment:** recommended first slice, but keep it deliberately thin and separable from computation.

### Option B — Piggyback on xcube and contribute upstream

**Choose when:** a maintained, finished cube viewer is more valuable than bespoke UX.

- Viewer already supplies the expensive exploration surface. [XV-ANALYSE]
- Server already models dynamic datasets and parameterized compute jobs. [XC-DYNAMIC] [XC-COMPUTE-ROUTES]
- Missing seams appear generic: operation discovery/submission UI, documented third-party operation registration, durable jobs/results, and Viewer refresh after result creation.

**Assessment:** preferred medium-term product direction if a spike confirms Dask execution, result persistence, and Viewer refresh. Contribute generic machinery upstream; publish climate-specific computation separately.

### Option C — Panel frontend plus Xpublish backend/plugin

**Choose when:** multiple frontends, HTTP APIs, OGC tiles/EDR, or independent scaling are required.

- Xpublish is a natural xarray/Dask plugin host. [XP-README] [XP-PLUGINS]
- Tiles and point-query primitives already exist. [XP-TILES] [XP-EDR]
- A frontend, job system, cache/provenance policy, and operational hardening are still required.

**Assessment:** stronger reusable architecture than Panel-only, but premature for an MVP unless API reuse is an explicit requirement.

### Option D — Precompute, then use an existing viewer

Compute a parameterized cube with `climate_indices`, persist it as Zarr/NetCDF, then open it in xcube Viewer, ncWMS/Godiva, Lexcube, or Pan3D. Each supports existing multidimensional datasets but has different UX. [XV-INDEX] [NC-README] [LEX-README] [PAN-README]

**Assessment:** lowest-risk operational route when parameter changes are infrequent; it is not interactive on-demand derivation.

### Option E — Earth Engine application

Implement an Earth Engine-native index collection and link the map and chart APIs. [EE-MAP] [EE-CLICK] [EE-CHART]

**Assessment:** use only for Earth Engine-resident inputs and acceptable service terms. It duplicates scientific logic and therefore needs a numerical parity program against `climate_indices`.

### Option F — Contribute only, with no climate_indices application

A pure upstream contribution could add xcube's generic operation UI or Pan3D's generic point probe, but no reviewed upstream project currently owns the `climate_indices` parameter semantics and whole-grid orchestration together.

**Assessment:** useful parallel work, not a substitute for a small integration artifact that proves the science-to-exploration path.

## Decision (2026-09-10): shelve the Explorer

The recommended direction below is superseded. It was written on the premise that parameterized, `climate_indices`-backed
whole-grid computation joined to map/point exploration was unavailable elsewhere. Climate Engine provides exactly that,
on demand, with a caller-chosen distribution and calibration period. [CE-API-DIST] [CE-API-SPEC]

Two findings closed the question:

1. **The differentiator was not real.** Gamma SPI with explicit calibration years is an existing hosted product.
2. **The cost premise was also not real, in the opposite direction.** The Explorer architecture assumed whole-cube
   computation was expensive enough to need durable materialization, a computation key, provenance manifests, job
   lifecycle and eviction. Measured here, SPI-6 over 480 monthly steps cost 2.07 ms per cell, of which 93% was an
   unconditional per-cell Kolmogorov-Smirnov goodness-of-fit diagnostic. With that check computed directly rather than
   through `scipy.stats.kstest`, a 144 × 144 cube takes about 4.6 s and CONUS nClimGrid about 3 minutes. Nothing at
   that scale needs a publication registry.

The library keeps the performance fix, which benefits the CLI, the pipeline and the test suite regardless. The Explorer
backlog is closed.

Where `climate_indices` is still unmatched, and where effort should go instead:

- Pearson Type III and the NCEI-conventional lineage;
- arbitrary user datasets and grids, offline and in-library;
- self-calibrating Palmer/scPDSI;
- the fire-weather family, which no reviewed tool offers.

## Recommended direction (superseded 2026-09-10)

1. **Correct the product claim.** Say: “Existing tools explore precomputed cubes; the proposed value is parameterized, `climate_indices`-backed whole-grid computation integrated with that exploration.” Do not say the exploration itself is unavailable.
2. **Build a small Panel/HoloViz reference application, not a new general viewer.** Support one tracer-bullet workflow—such as SPI-6/gamma/calibration bounds—with an explicit Compute action, one map slice, time navigation, and nearest-cell full series. Treat it as experimental while the xarray API remains beta. [CI-SPI]
3. **Normalize and persist by computation key.** At minimum include source identity/version, variable, index, scale, distribution, calibration period, periodicity, library version, chunking-relevant grid identity, and missing-data policy. Map and point interactions must never refit the index.
4. **Run an xcube spike before committing to a production frontend.** Verify external operation registration, schema discovery, job cancellation, Dask scheduler behavior, durable Zarr output, cache reuse, Viewer discovery of a completed dataset, and time-series performance.
5. **Open an xcube design discussion.** Propose a generic schema-driven compute form and stable operation/plugin API. Keep `climate_indices` controls in an external package or deployment, so upstream core remains domain-neutral.
6. **Adopt Xpublish only when backend/API requirements justify it.** If chosen, contribute a standalone climate-index plugin and use EDR/tiles as reusable delivery interfaces.
7. **Offer precompute-and-open immediately.** Document xcube Viewer and ncWMS/Godiva as existing exploration options for users who do not need parameter changes in the same session.

## Uncertainty and open validation questions

These questions require prototypes, maintainer confirmation, or load tests; the reviewed sources do not settle them.

### Product semantics

- Does “compute the entire cube” mean a lazy logical DataArray, a fully computed in-memory result, or a durable Zarr/NetCDF artifact?
- Must hover return a cell's exact value, while click returns the series, or is click-only acceptable?
- Should coordinate selection use nearest cell, interpolation, or a pixel-footprint/area average?
- Are grids always rectilinear longitude/latitude, or must projected, curvilinear, rotated-pole, and antimeridian-crossing grids work?

### Scientific and Dask behavior

- What are expected time length, grid size, chunk sizes, concurrency, latency, and memory limits?
- The index fit needs the full temporal series in one Dask chunk. Can representative global cubes meet per-worker memory limits, and what spatial chunk gives acceptable scheduling overhead? [CI-XARRAY-GUIDE]
- Should computed fitting parameters be retained and exposed for provenance/reuse?
- What invalid-cell and partial-failure reporting is required for sparse calibration periods?
- Is the beta xarray API stable enough for a hosted application contract, or should the app pin a library version? [CI-SPI]

### xcube spike

- Is there a supported third-party startup/plugin hook for compute-operation registration, rather than importing deployment code directly?
- Can a compute operation materialize to durable Zarr and register that output atomically, rather than only registering a lazy/in-memory dataset?
- Will the current Viewer discover a newly registered result without a full reload?
- Does xcube's local thread-based job context cooperate with a configured distributed Dask scheduler, and what survives process restart? [XC-COMPUTE-CONTEXT]
- How should authentication, per-user datasets, quotas, cache eviction, and cancellation propagate through the compute API?

### Xpublish and alternatives

- Does EDR position preserve calendars, masks, coordinate metadata, and full temporal ordering required by climate indices?
- Are the current tiles and EDR plugins mature enough for the required grid types and concurrent load?
- Is Lexcube's GPLv3+ acceptable if it is embedded or adapted? [LEX-LICENSE]
- Is ncWMS2's Godiva logo condition and maintenance cadence acceptable for deployment? [NC-README] [NC-RELEASES]
- Would Earth Engine reproduce `climate_indices` exactly for zero precipitation, missing values, fitting failures, and calibration grouping, and are service terms suitable for the intended audience? [EE-TERMS]

## Sources

All sources below are owned by the relevant project or service. “Maintenance” links point to official releases or pinned repository commits, not third-party activity summaries.

### Climate Engine

[CE-HOME]: https://www.climateengine.org/
[CE-ABOUT]: https://climateengine.org/get_started/get-started/
[CE-APP]: https://app.climateengine.org/
[CE-SPEI]: https://climateengine.org/datasets/drought/standardized-precipitation-evapotranspiration-index/
[CE-GRIDMET-DROUGHT]: https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT
[CE-API-SPEC]: https://api.climateengine.org/openapi.json
[CE-API-DIST]: Climate Engine API v1, `GET /raster/mapid/standard_index`, retrieved 2026-09-10. An invalid `distribution` returns `Invalid distribution: zzz. Should be one of ['gamma', 'loglogistic', 'nonparametric'].`; an invalid `variable` returns the valid index list for that distribution, `['spi']` for `gamma`.

### climate_indices

[CI-SPI]: https://github.com/monocongo/climate_indices/blob/a1b18ac0f8551c6d2919d36787ae8061ad71a06e/src/climate_indices/typed_public_api.py#L68-L151
[CI-DASK]: https://github.com/monocongo/climate_indices/blob/a1b18ac0f8551c6d2919d36787ae8061ad71a06e/src/climate_indices/xarray_adapter.py#L1360-L1445
[CI-XARRAY-GUIDE]: https://github.com/monocongo/climate_indices/blob/a1b18ac0f8551c6d2919d36787ae8061ad71a06e/docs/xarray_migration.rst#L488-L649

### HoloViz

[PANEL]: https://github.com/holoviz/panel/blob/b6b1d99fb8bfd84e55c3499a475cbaa4b9f36fba/README.md
[PANEL-LICENSE]: https://github.com/holoviz/panel/blob/b6b1d99fb8bfd84e55c3499a475cbaa4b9f36fba/LICENSE.txt
[HVPLOT]: https://github.com/holoviz/hvplot/blob/51b663497655b085f5a54fa9818f935dab5069ad/README.md
[HVPLOT-GRID]: https://github.com/holoviz/hvplot/blob/51b663497655b085f5a54fa9818f935dab5069ad/doc/user_guide/Gridded_Data.ipynb
[HVPLOT-GEO]: https://github.com/holoviz/hvplot/blob/51b663497655b085f5a54fa9818f935dab5069ad/doc/user_guide/Geographic_Data.ipynb
[HVPLOT-LICENSE]: https://github.com/holoviz/hvplot/blob/51b663497655b085f5a54fa9818f935dab5069ad/LICENSE
[HVPLOT-MAINT]: https://github.com/holoviz/hvplot/commit/51b663497655b085f5a54fa9818f935dab5069ad
[GEOVIEWS]: https://github.com/holoviz/geoviews/blob/a1a732580cc33ab77ec6539ffe2047ff5b7ed0e1/README.md
[GEOVIEWS-LICENSE]: https://github.com/holoviz/geoviews/blob/a1a732580cc33ab77ec6539ffe2047ff5b7ed0e1/LICENSE
[GEOVIEWS-MAINT]: https://github.com/holoviz/geoviews/commit/a1a732580cc33ab77ec6539ffe2047ff5b7ed0e1
[HV-TAP]: https://github.com/holoviz/holoviews/blob/ad5a0887f64185928b36de16235a3499682a10be/examples/reference/streams/bokeh/Tap.ipynb
[HV-LICENSE]: https://github.com/holoviz/holoviews/blob/ad5a0887f64185928b36de16235a3499682a10be/LICENSE.txt
[HOLOVIZ-MAINT]: https://github.com/holoviz/panel/commit/b6b1d99fb8bfd84e55c3499a475cbaa4b9f36fba
[HV-MAINT]: https://github.com/holoviz/holoviews/commit/ad5a0887f64185928b36de16235a3499682a10be
[DATASHADER]: https://github.com/holoviz/datashader/blob/4cf418e880dfd90d58de70d48c8a02d97f8b066e/examples/getting_started/3_Interactivity.ipynb
[DATASHADER-LICENSE]: https://github.com/holoviz/datashader/blob/4cf418e880dfd90d58de70d48c8a02d97f8b066e/LICENSE.txt
[DATASHADER-MAINT]: https://github.com/holoviz/datashader/commit/4cf418e880dfd90d58de70d48c8a02d97f8b066e

### xcube and xcube Viewer

[XC-README]: https://github.com/xcube-dev/xcube/blob/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8/README.md
[XC-WEBAPI]: https://github.com/xcube-dev/xcube/blob/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8/docs/source/webapi.rst
[XC-SERVE]: https://github.com/xcube-dev/xcube/blob/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8/docs/source/cli/xcube_serve.rst
[XC-DYNAMIC]: https://github.com/xcube-dev/xcube/blob/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8/docs/source/cli/xcube_serve.rst#L430-L535
[XC-COMPUTE-OPS]: https://github.com/xcube-dev/xcube/blob/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8/xcube/webapi/compute/operations.py
[XC-COMPUTE-ROUTES]: https://github.com/xcube-dev/xcube/blob/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8/xcube/webapi/compute/routes.py#L172-L286
[XC-COMPUTE-CONTEXT]: https://github.com/xcube-dev/xcube/blob/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8/xcube/webapi/compute/context.py#L59-L202
[XC-LICENSE]: https://github.com/xcube-dev/xcube/blob/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8/LICENSE
[XC-MAINT]: https://github.com/xcube-dev/xcube/commit/e3f5a2dda090f1508e0fd2c56e0c5ecec2bcc2c8
[XV-INDEX]: https://github.com/xcube-dev/xcube-viewer/blob/12ee493104a98834e42144c4758553e1ef5d1828/docs/index.md
[XV-ANALYSE]: https://github.com/xcube-dev/xcube-viewer/blob/12ee493104a98834e42144c4758553e1ef5d1828/docs/user_guide/analyse.md
[XV-USER-VARS]: https://github.com/xcube-dev/xcube-viewer/blob/12ee493104a98834e42144c4758553e1ef5d1828/public/docs/user-variables.en.md
[XV-SOURCE]: https://github.com/xcube-dev/xcube-viewer/tree/12ee493104a98834e42144c4758553e1ef5d1828/src
[XV-LICENSE]: https://github.com/xcube-dev/xcube-viewer/blob/12ee493104a98834e42144c4758553e1ef5d1828/LICENSE
[XV-MAINT]: https://github.com/xcube-dev/xcube-viewer/commit/12ee493104a98834e42144c4758553e1ef5d1828

### NASA Giovanni

[GIO-EARTHDATA]: https://www.earthdata.nasa.gov/data/tools/giovanni
[GIO-MANUAL]: https://giovanni.gsfc.nasa.gov/giovanni/doc/UsersManualworkingdocument.docx.html
[GIO-README]: https://github.com/nasa/Giovanni/blob/992c27e6ddbf35ad17620839842801ca28334bbc/README.md
[GIO-REPO]: https://github.com/nasa/Giovanni/tree/992c27e6ddbf35ad17620839842801ca28334bbc

### Google Earth Engine and Xee

[EE-DEFERRED]: https://developers.google.com/earth-engine/guides/deferred_execution
[EE-PROCESSING]: https://developers.google.com/earth-engine/guides/processing_environments
[EE-EXPORT]: https://developers.google.com/earth-engine/guides/exporting_images
[EE-MAP]: https://developers.google.com/earth-engine/apidocs/ee-imagecollection-map
[EE-CLICK]: https://developers.google.com/earth-engine/apidocs/ui-map-onclick
[EE-CHART]: https://developers.google.com/earth-engine/apidocs/ui-chart-image-series
[EE-APPS]: https://developers.google.com/earth-engine/guides/apps
[EE-MATH]: https://developers.google.com/earth-engine/apidocs/ee-image-gammainc
[EE-ERFINV]: https://developers.google.com/earth-engine/apidocs/ee-image-erfinv
[EE-TERMS]: https://earthengine.google.com/terms/
[EE-CLIENT]: https://github.com/google/earthengine-api/blob/229706143faf75a57de78eaed04d1c3ed76db938/README.md
[EE-CLIENT-LICENSE]: https://github.com/google/earthengine-api/blob/229706143faf75a57de78eaed04d1c3ed76db938/LICENSE
[EE-MAINT]: https://github.com/google/earthengine-api/commit/229706143faf75a57de78eaed04d1c3ed76db938
[XEE]: https://github.com/google/Xee/blob/17dfdd9e492be422e94a8d07059fd7fabf3e1664/README.md
[XEE-LICENSE]: https://github.com/google/Xee/blob/17dfdd9e492be422e94a8d07059fd7fabf3e1664/LICENSE
[XEE-MAINT]: https://github.com/google/Xee/commit/17dfdd9e492be422e94a8d07059fd7fabf3e1664

### ncWMS2 and Godiva3

[NC-README]: https://github.com/Reading-eScience-Centre/ncwms/blob/b8934570af57c9f18590002411ef621a42d71e54/README.md
[NC-USAGE]: https://github.com/Reading-eScience-Centre/ncwms/blob/b8934570af57c9f18590002411ef621a42d71e54/docs/04-usage.md
[NC-CONFIG]: https://github.com/Reading-eScience-Centre/ncwms/blob/b8934570af57c9f18590002411ef621a42d71e54/docs/03-config.md
[NC-DEVELOP]: https://github.com/Reading-eScience-Centre/ncwms/blob/b8934570af57c9f18590002411ef621a42d71e54/docs/06-development.md
[NC-INSTALL]: https://github.com/Reading-eScience-Centre/ncwms/blob/b8934570af57c9f18590002411ef621a42d71e54/docs/02-installation.md
[NC-RELEASES]: https://github.com/Reading-eScience-Centre/ncwms/releases/tag/ncwms-2.5.3-SNAPSHOT2
[NC-STABLE-RELEASE]: https://github.com/Reading-eScience-Centre/ncwms/releases/tag/ncwms-2.5.2

### Lexcube

[LEX-README]: https://github.com/msoechting/lexcube/blob/2db2324d14641d8dab63cf86335978c7acb65609/README.md
[LEX-LICENSE]: https://github.com/msoechting/lexcube/blob/2db2324d14641d8dab63cf86335978c7acb65609/COPYING
[LEX-RELEASE]: https://github.com/msoechting/lexcube/releases/tag/v2.0.1

### Pan3D

[PAN-README]: https://github.com/Kitware/pan3d/blob/29737bd2ac4151008b0d4576f4d60f431dbbeec3/docs/README.md
[PAN-VIEWER]: https://github.com/Kitware/pan3d/blob/29737bd2ac4151008b0d4576f4d60f431dbbeec3/docs/tutorials/dataset_viewer.md
[PAN-COMMAND]: https://github.com/Kitware/pan3d/blob/29737bd2ac4151008b0d4576f4d60f431dbbeec3/docs/tutorials/command_line.md
[PAN-EXPLORERS]: https://github.com/Kitware/pan3d/blob/29737bd2ac4151008b0d4576f4d60f431dbbeec3/docs/tutorials/explorers.md
[PAN-COMPUTED]: https://github.com/Kitware/pan3d/blob/29737bd2ac4151008b0d4576f4d60f431dbbeec3/src/pan3d/xarray/algorithm.py#L430-L481
[PAN-ANALYTICS]: https://github.com/Kitware/pan3d/blob/29737bd2ac4151008b0d4576f4d60f431dbbeec3/src/pan3d/ui/analytics.py#L183-L371
[PAN-MATERIALIZE]: https://github.com/Kitware/pan3d/blob/29737bd2ac4151008b0d4576f4d60f431dbbeec3/src/pan3d/xarray/algorithm.py#L580-L606
[PAN-DOCKER]: https://github.com/Kitware/pan3d/tree/29737bd2ac4151008b0d4576f4d60f431dbbeec3/docker
[PAN-LICENSE]: https://github.com/Kitware/pan3d/blob/29737bd2ac4151008b0d4576f4d60f431dbbeec3/LICENSE
[PAN-RELEASE]: https://github.com/Kitware/pan3d/releases/tag/v1.3.1

### Xpublish ecosystem

[XP-README]: https://github.com/xpublish-community/xpublish/blob/00abd9d8f9a30c0ae3709ff68472f993aa57b45d/README.md
[XP-PLUGINS]: https://github.com/xpublish-community/xpublish/blob/00abd9d8f9a30c0ae3709ff68472f993aa57b45d/docs/source/user-guide/plugins.md
[XP-REST]: https://github.com/xpublish-community/xpublish/blob/00abd9d8f9a30c0ae3709ff68472f993aa57b45d/xpublish/rest.py#L316-L343
[XP-LICENSE]: https://github.com/xpublish-community/xpublish/blob/00abd9d8f9a30c0ae3709ff68472f993aa57b45d/LICENSE
[XP-MAINT]: https://github.com/xpublish-community/xpublish/commit/00abd9d8f9a30c0ae3709ff68472f993aa57b45d
[XP-TILES]: https://github.com/earth-mover/xpublish-tiles/blob/9584fea5f8bb53226f832c67e502f51e7637b299/README.md
[XP-TILES-MAINT]: https://github.com/earth-mover/xpublish-tiles/commit/9584fea5f8bb53226f832c67e502f51e7637b299
[XP-EDR]: https://github.com/xpublish-community/xpublish-edr/blob/f085f70aa1fe95511ae31a805330c208ac8499b1/README.md
[XP-EDR-MAINT]: https://github.com/xpublish-community/xpublish-edr/commit/f085f70aa1fe95511ae31a805330c208ac8499b1
