# climate_indices (core library)

A Python scientific computing library that turns raw climate observations (precipitation, temperature) into standardized drought and moisture indices — SPI, SPEI, PNP, EDDI, and the Palmer family — via distribution fitting and statistical transformation.

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
The Wells et al. (2004) self-calibrating variant of PDSI, which recalibrates duration factors and the K-prime (K′) climate characteristic per location instead of using fixed national constants. K-prime is distinct from the standard K-Factor defined below. It is available through the NumPy API as `palmer.scpdsi()`. The CLI's `--index palmers` path continues to produce only PDSI, PHDI, PMDI, and Z-Index until CLI support is added separately.

**CAFEC (Climatically Appropriate For Existing Conditions)**:
Per-calendar-month calibration coefficients (alpha, beta, gamma, delta) computed from calibration-period water-balance sums, representing the precipitation/moisture terms "appropriate" for that location's climate — actual conditions are compared against CAFEC to produce the Z-Index.

**Available Water Capacity (AWC)**:
A location's total soil moisture-holding capacity, in inches, split into a fixed top-layer capacity and a variable underlying-layer capacity. Supplied as an external input per location (e.g. per climate division), not computed.

**K-Factor (Climate Characteristic)**:
Monthly weighting factors that convert the raw CAFEC moisture departure into the Z-Index, calibrated to make Z-Index values comparable in severity across different climates.

### PET (Potential Evapotranspiration)

**PET (Potential Evapotranspiration)**:
The atmospheric moisture-demand quantity — how much water would evaporate/transpire under unlimited moisture availability — used as an input to SPEI and EDDI.

**Thornthwaite Method**:
Monthly PET estimated from mean air temperature and day length, via a temperature-derived heat index (Thornthwaite, 1948).

**Hargreaves Method**:
Daily PET estimated from min/max/mean temperature and extraterrestrial radiation (Hargreaves, 1985; FAO-56 eq. 52).

### Statistics

**L-Moments**:
Linear-combination-of-order-statistics summary measures of a sample's location, scale, and skew — used here as a more robust alternative to conventional moments for fitting the Pearson Type III distribution.

### Metadata & provenance

**Fixture Provenance**:
A schema-validated JSON record (source, URL, download date, checksum, tolerance) accompanying each external reference/validation dataset in `tests/fixture/`, proving the data hasn't silently changed since it was downloaded. A testing/QA concept, not something attached to library outputs.
_Avoid_: Provenance (ambiguous alone — see also Output Provenance)

**Output Provenance**:
A version string and CF-style `history` attribute stamped onto computed index outputs (xarray DataArrays), recording that climate_indices produced them and with what version. A domain concept about the results themselves, distinct from Fixture Provenance.
_Avoid_: Provenance (ambiguous alone — see also Fixture Provenance)

**CF Standard Name Omission**:
Policy: the CF Conventions' `standard_name` attribute is deliberately left unset on computed outputs (SPI, SPEI, PDSI, etc.) because none of these indices has an officially CF-registered standard name. This is intentional, not a gap to fill in.
