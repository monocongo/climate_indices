# climate_indices (core library)

A Python scientific computing library that turns raw climate observations (precipitation, temperature) into standardized drought and moisture indices — SPI, SPEI, PNP, EDDI, and the Palmer family — via distribution fitting and statistical transformation. It also computes fire-weather indices, namespaced under the `fire` module.

## Language

### Core concepts

**Timescale**:
The number of consecutive time steps (months or days) accumulated before an index is computed — e.g. a 6-month accumulation window produces "SPI-6." Public function parameters are spelled `scale` in code (`indices.py`, `compute.py`, `typed_public_api.py`); use "Timescale" in prose and documentation.
_Avoid_: Scale (fine as a code identifier, not as prose)

**Periodicity**:
The calendar granularity of an input time series — `monthly` (12 values/year) or `daily` (366 values/year, every year treated as a leap year). Determines how values are reshaped and grouped by calendar position before fitting.

**Calibration Period**:
The fixed span of years whose data defines "normal" — the distribution parameters (or, for PNP, the simple average) are fit only on this window, and all years (including years outside it) are then standardized against it. Enforced minimum is 30 years. The public API spells the boundary years two different ways depending on which function you call: `calibration_year_initial`/`calibration_year_final` (`indices.spi()`, `indices.spei()`, legacy CLI) vs. `calibration_start_year`/`calibration_end_year` (`compute.py` internals, `palmer.py`, `indices.percentage_of_normal()`). Both spellings refer to the same concept — treat this as a known inconsistency, not two different things.
_Avoid_: Baseline period, reference period

**Distribution / Distribution Fitting**:
The statistical model — Gamma or Pearson Type III — chosen to represent a scaled variable's underlying distribution, so a raw value can be converted to a cumulative probability and then a standardized (z-score-like) index value.
_Avoid_: pearson3 (this spelling only appears in test-fixture filenames, not in API/domain language)

**Probability of Zero**:
The empirical fraction of zero-valued observations at a given calendar time step, tracked separately because precipitation (and P−PET series for SPEI) can be exactly zero, which the continuous Gamma/Pearson distributions can't represent directly. Mixed into the fitted CDF so zero-precipitation periods still get a well-defined standardized value.

### Indices

**SPI (Standardized Precipitation Index)**:
A meteorological drought indicator that standardizes accumulated precipitation, at a chosen timescale, against its long-term calibration-period distribution.
_Avoid_: Precipitation index

**SPEI (Standardized Precipitation Evapotranspiration Index)**:
The same standardization methodology as SPI, but applied to accumulated (precipitation − PET) instead of precipitation alone — captures both moisture supply and atmospheric demand.

**PNP (Percentage of Normal Precipitation)**:
Each timescale-accumulated value expressed as a percentage of the average ("normal") value for that same calendar time step over the calibration period. Not distribution-fitted or standardized — a simpler ratio-to-normal measure, distinct from SPI/SPEI.
_Avoid_: Percent of Normal Precipitation (this wording appears in the CF metadata registry's `long_name`; prefer "Percentage of Normal Precipitation" / PNP for consistency with the public API and function name)

**EDDI (Evaporative Demand Drought Index)**:
A non-parametric drought index (NOAA PSL methodology) built from accumulated PET: values are ranked within each calendar period of the calibration window, ranks become cumulative probabilities, and probabilities become z-scores.

### Palmer family

**PDSI (Palmer Drought Severity Index)**:
The primary Palmer water-balance drought index — a recursive monthly severity measure derived from the Z-Index.

**PHDI (Palmer Hydrological Drought Index)**:
A Palmer-family index tracking established ("backed-out") drought or wet-spell severity, distinct from PDSI's more immediately responsive value; falls back to the PDSI value when no spell is established.

**PMDI (Palmer Modified Drought Index)**:
A Palmer-family index that is a probability-weighted blend of the incipient wet/dry indices and the established severity index, meant to respond faster than PHDI while being less erratic than PDSI alone.

**Palmer Z-Index**:
A monthly moisture-anomaly index — the weighted difference between actual precipitation and CAFEC precipitation — that drives the recursive PDSI/PHDI/PMDI calculations.
_Avoid_: Moisture anomaly index (informal synonym used in one comment; prefer Z-Index)

**scPDSI (Self-calibrated Palmer Drought Severity Index)**:
The Wells et al. (2004) self-calibrating variant of PDSI, which recalibrates duration factors and the K-prime (K′) climate characteristic per location instead of using fixed national constants. K-prime is distinct from the standard K-Factor defined below. It is available through the NumPy API as `palmer.scpdsi()` and through the CLI's `--index palmers` path, which writes it as `<output_file_base>_scpdsi.nc` alongside the four standard outputs.

**CAFEC (Climatically Appropriate For Existing Conditions)**:
Per-calendar-month calibration coefficients (alpha, beta, gamma, delta) computed from calibration-period water-balance sums, representing the precipitation/moisture terms "appropriate" for that location's climate — actual conditions are compared against CAFEC to produce the Z-Index.

**Available Water Capacity (AWC)**:
A location's total soil moisture-holding capacity, in inches, split into a fixed top-layer capacity and a variable underlying-layer capacity. Supplied as an external input per location (e.g. per climate division), not computed.

**K-Factor (Climate Characteristic)**:
Monthly weighting factors that convert the raw CAFEC moisture departure into the Z-Index, calibrated to make Z-Index values comparable in severity across different climates.

**Duration Factors (m, b)** and the **duration-factor weighting fraction (c)**:
The slope and intercept of the line that maps a spell's accumulated Z-Index onto PDSI severity: Palmer's (1965) national defaults for standard PDSI, overridable through `pdsi()`'s `fitting_params`, and fitted per location by scPDSI. The same pair implies the duration-factor weighting fraction `c`, the share of previously accumulated severity each recursion carries forward (`c = b / (m + b)`, equivalently `1 - m / (m + b)`), while the current period's Z-Index enters divided by `m + b`. The pdi.f and Wells recursions derive `c` with those two different expressions and neither is guaranteed bit-identical to the other, so each keeps its own. Distinct from the K-Factor weighting factors above, which weight the Z-Index rather than the accumulated severity.

### PET (Potential Evapotranspiration)

**PET (Potential Evapotranspiration)**:
The atmospheric moisture-demand quantity — how much water would evaporate/transpire under unlimited moisture availability — used as an input to SPEI and EDDI.

**Thornthwaite Method**:
Monthly PET estimated from mean air temperature and day length, via a temperature-derived heat index (Thornthwaite, 1948).

**Hargreaves Method**:
Daily PET estimated from min/max/mean temperature and extraterrestrial radiation (Hargreaves, 1985; FAO-56 eq. 52).

### Fire family

Fire-weather and fuel-dryness indices live in the namespaced `fire` package (`from climate_indices import fire`), never as unqualified package functions — see [ADR-0005](../../docs/adr/0005-fire-module-api.md) and the [fire subsystem design](../../docs/design/fire-subsystem.md).

**FFWI (Fosberg Fire Weather Index)**:
A dimensionless, weather-only, elementwise fire-weather index from temperature, relative humidity, and wind speed; computed by `fire.fosberg_ffwi()`.

**HDW (Hot-Dry-Windy Index)**:
The vapor pressure deficit times wind speed, maximized over the lowest 500 m above ground level of a vertical profile; computed by `fire.hot_dry_windy()` in hPa m s⁻¹.

**Haines Index (Lower Atmosphere Severity Index)**:
A dimensionless, state-free score in [2, 6] of one lower-atmosphere layer's stability and moisture — its lapse rate plus its dewpoint depression — computed by `fire.haines_index()` from the pressure levels its `variant` (`"low"`, `"mid"`, or `"high"`) names, or by `fire.haines_index_from_profile()`, which interpolates profiles to those levels and selects the variant from a terrain elevation field. Carries no wind term; HDW was developed in part to supply it.

**KBDI (Keetch-Byram Drought Index)**:
A daily recursive measure of cumulative moisture deficiency in deep duff and upper soil layers, on a 0–800 scale of hundredths of an inch (0–203.2 mm metric), from precipitation, daily maximum temperature, and mean annual precipitation; computed by `fire.kbdi()`. Moisture loss reverses only through net rain: consecutive rainy days form one wet spell, and only rain above its first 5.08 mm (0.20 in) reduces the index. A fire-danger index, not a standardized drought index like SPI or SPEI.

**FFMC (Fine Fuel Moisture Code)**:
The daily recursive moisture content of fine surface litter and other fine fuels, from noon temperature, relative humidity, 10 m wind speed, and 24-hour rain; computed by `fire.ffmc()`. A dimensionless code in [0, 101] and the base of the CFFWIS moisture codes.

**DMC (Duff Moisture Code)**:
The daily recursive moisture content of loosely compacted organic layers of moderate depth, from noon temperature, relative humidity, and 24-hour rain; computed by `fire.duff_moisture_code()`. Drying scales with the month- and latitude-dependent effective day length, and the dimensionless code is floored at zero with no upper bound.

**DC (Drought Code)**:
The daily recursive moisture content of deep, compact organic layers, from noon temperature and 24-hour rain; computed by `fire.drought_code()`. Potential evapotranspiration scales with the month- and latitude-dependent day length, and the dimensionless code is floored at zero with no upper bound. The CFFWIS component only, distinct from SPI, SPEI, PDSI, and the other drought indices.

**CFFWIS (Canadian Forest Fire Weather Index System)**:
The Canadian Forest Service's fire-weather system, computed by `fire.cffwis()`: the three moisture codes `fire.ffmc()` (FFMC), `fire.duff_moisture_code()` (DMC), and `fire.drought_code()` (DC), then the behavior indices ISI, BUI, FWI, and DSR. `fire.cffwis()` threads all three moisture codes through one daily pass and can return any requested subset of the seven outputs.

**ISI (Initial Spread Index)**:
A dimensionless, state-free CFFWIS behavior index of the expected rate of fire spread immediately after ignition, from FFMC and 10 m wind speed; computed by `fire.initial_spread_index()`.

**BUI (Buildup Index)**:
A dimensionless, state-free CFFWIS behavior index of the fuel available for spreading, from DMC and DC; computed by `fire.buildup_index()`.

**FWI (Canadian Fire Weather Index)**:
A dimensionless, state-free CFFWIS behavior index combining ISI and BUI; computed by `fire.cffwis_fwi()`. Distinct from the Fosberg Fire Weather Index (FFWI).

**DSR (Daily Severity Rating)**:
The `0.0272 * FWI ** 1.77` transform of the Canadian FWI that makes seasonal averaging meaningful; computed by `fire.daily_severity_rating()`.

### Statistics

**L-Moments**:
Linear-combination-of-order-statistics summary measures of a sample's location, scale, and skew — used here as a more robust alternative to conventional moments for fitting the Pearson Type III distribution.

### Gridded execution

**Spatial Block** (spelled time-major in code):
A gridded input array shaped `(time, *cells)` — the time axis first, every trailing axis an independent cell — that the fitting-based indices scale, fit, and transform in one pass. Any array with three or more dimensions is read this way; 1-D input stays a series and 2-D input stays the legacy `(years, periods)` layout. The one ambiguous shape is a block whose first cell axis is a calendar period length (12 or 366), which is indistinguishable from a `(years, periods, *cells)` array; that shape has to be declared with `spatial_time_major=True`, which `xarray_adapter` sets for every block it packs. See [ADR-0009](../../docs/adr/0009-spatial-block-declaration.md).
_Avoid_: time-major block (the code spelling, not the prose term)

**Spatial Kernel**:
An index whose NumPy core accepts a Spatial Block, declared per index with `spatial_kernel=True` at its adapter call site, or by an entry point that owns its `xr.apply_ufunc` call (`pet_thornthwaite`/`pet_hargreaves`, `pdsi`). Such an index runs one `xr.apply_ufunc` call per non-core block instead of one per grid cell; indices whose cores still loop over cells keep the Per-Cell Path.

**Per-Cell Path**:
The alternative dispatch, `xr.apply_ufunc(..., vectorize=True)`, which calls the kernel once per grid cell over 1-D time series. Still used for inputs with a single non-core dimension. scPDSI has no adapter entry point at all and stays on the per-location NumPy path ([ADR-0011](../../docs/adr/0011-palmer-spatial-block-and-per-location-scpdsi.md)).

### Input validation

**Input Type**:
The array backend a computation's inputs arrive in — NumPy-coercible (ndarray, list, tuple, scalars) or `xarray.DataArray` — classified by `validation.detect_input_type()` for adapter dispatch. Says nothing about how a dataset is stored.

**Dataset Layout**:
The storage layout a NetCDF dataset's data variables follow — `grid` (lat/lon), `divisions` (US climate division IDs), or `timeseries` — together with the dimension orders each one accepts: `validation.detect_dataset_layout()` classifies it and `validation.expected_dimensions()` reports the orders, including the time-free ones a per-location companion such as available water capacity uses. The CLI validates its inputs against this contract. Distinct from Input Type: a grid dataset read into NumPy arrays is still `InputType.NUMPY`. The CLI's shared-array transport copies values in storage order and its kernels index the time axis at a fixed position, so a `grid` data variable has to be stored `(lat, lon, time)` and a `divisions` data variable `(division, time)`. The layout contract is wider than the transport: it also accepts a time-major `grid` or `divisions` variable for the xarray-backed KBDI path, which never enters the transport.
_Avoid_: Input type (that term names the array backend above)

### Metadata & provenance

**Fixture Provenance**:
A schema-validated JSON record (source, URL, download date, checksum, tolerance) accompanying each external reference/validation dataset in `tests/fixture/`, proving the data hasn't silently changed since it was downloaded. A testing/QA concept, not something attached to library outputs.
_Avoid_: Provenance (ambiguous alone — see also Output Provenance)

**Output Provenance**:
A version string and CF-style `history` attribute stamped onto computed index outputs (xarray DataArrays), recording that climate_indices produced them and with what version. A domain concept about the results themselves, distinct from Fixture Provenance.
_Avoid_: Provenance (ambiguous alone — see also Fixture Provenance)

**CF Standard Name Omission**:
Policy: the CF Conventions' `standard_name` attribute is deliberately left unset on computed outputs (SPI, SPEI, PDSI, etc.) because none of these indices has an officially CF-registered standard name. This is intentional, not a gap to fill in.
