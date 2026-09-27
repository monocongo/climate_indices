# Potential Evapotranspiration (PET)

## Scope

`climate_indices` implements three PET methods:

- **Thornthwaite (1948)** — monthly PET from mean monthly air temperature,
  latitude, and calendar-derived daylight hours.
- **Hargreaves-Samani (1985)** — daily PET (ETo) from minimum, maximum, and
  mean daily air temperature, latitude, and day of year (via
  extraterrestrial radiation), following Equation 52 in Allen et al. (1998)
  FAO Irrigation and Drainage Paper 56.
- **FAO-56 Penman-Monteith (1998)** — daily PET (ETo) from minimum and maximum
  air temperature, humidity, wind speed, solar radiation, and elevation,
  following Equation 6 in Allen et al. (1998) FAO Irrigation and Drainage
  Paper 56.

## Public API

Use `climate_indices.pet_thornthwaite()`, `climate_indices.pet_hargreaves()`,
and `climate_indices.pet_penman_monteith()`. All three accept NumPy arrays and
xarray DataArrays; the NumPy path is stable and the xarray path is beta.

## FAO-56 Penman-Monteith

`climate_indices.pet_penman_monteith()` implements the full FAO-56 reference
crop evapotranspiration equation (Eq 6):

$$
ET_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T+273} u_2 (e_s - e_a)}
{\Delta + \gamma (1 + 0.34 u_2)}
$$

The FAO-56 intermediate variables are derived internally from the supplied
meteorology rather than passed in:

- atmospheric pressure (Eq 7) and the psychrometric constant (Eq 8) from
  elevation;
- saturation vapour pressure (Eq 12) and its slope (Eq 13) from daily minimum
  and maximum temperature;
- actual vapour pressure (Eq 14-19) from the best available humidity pathway;
- wind speed at the 2 m standard height (Eq 47) from the measurement height;
- net radiation (Eq 40) from extraterrestrial radiation (Eq 21), clear-sky
  radiation (Eq 37), net shortwave radiation (Eq 38), and net longwave
  radiation (Eq 39) with the cloudiness term.

**Units and return value.** All temperatures are degrees Celsius, wind is
metres per second, radiation is MJ m-2 day-1, elevation is metres, and the
returned PET is in millimetres per day — the same convention as
`pet_hargreaves`.

**Required versus optional inputs.** The required inputs are daily minimum and
maximum temperature, latitude, elevation, wind speed, and (for NumPy input) the
day of year. Humidity and radiation inputs are optional. Humidity pathway
precedence is dewpoint, then RHmin/RHmax, then RHmax, then RHmean, and finally
the arid-region `e0(Tmin - 2)` estimate. Radiation precedence is supplied solar
radiation, then sunshine hours, then the temperature-range estimate (Eq 50,
with the interior `kRs = 0.16` or coastal `kRs = 0.19` coefficient); the
temperature-range estimate is limited to the clear-sky radiation. A NumPy call
requires an explicit `day_of_year`; an xarray call infers it from the time
coordinate.

**Soil heat flux.** For daily steps `G` is assumed to be zero, matching the
FAO-56 daily convention; a caller may override it via
`soil_heat_flux_mj_m2_day`.

**Missing-data behaviour.** NaN inputs propagate to the output. An xarray input
must be a daily coordinate beginning on January 1, consistent with the other
daily PET adapter; optional time-series inputs are aligned with an inner join
and dropped timesteps emit `InputAlignmentWarning`.

**SPEI use.** SPEI accepts a caller-supplied PET array, so the output of
`pet_penman_monteith()` can be passed directly to `climate_indices.spei()` as
`pet_mm`.

## Validation

PET has no single authoritative external gridded dataset: comparing against
a reanalysis-driven reference-ET product (e.g. gridMET) confounds algorithm
correctness with genuine climate-dependent bias between PET formulas, so no
defensible numerical tolerance exists (see the decision on
[Identify a gridded reference-ET dataset for PET stretch validation](https://github.com/monocongo/climate_indices/issues/775)).
Literature worked examples are therefore the primary external output-reference
evidence for PET. The sources provide daylight hours or extraterrestrial
radiation rather than a latitude/date pair, so the tests reconstruct those
coordinates with the library's solar geometry; they do not independently
validate that reconstruction.

`tests/test_pm_eto.py` covers the FAO-56 Chapter 3 radiation and wind helpers
against worked Examples 10-16 (Allen et al., 1998) and the full daily chain
against Example 18 (Uccle, 6 July, ETo = 3.88 mm/day, at ±0.05 mm/day).
`tests/test_pm_eto_xarray.py` checks that the xarray path reproduces the NumPy
path, keeps Dask input lazy, lands the CF metadata, and enforces the daily
calendar contract.

`tests/test_eto.py` covers the temperature-based methods against synthetic
regression fixtures (`tests/fixture/temp_celsius.npy`,
`tests/fixture/pet_thornthwaite.npy`, and the `hargreaves_*` fixtures generated
in `tests/conftest.py`) and, since
issue [#774](https://github.com/monocongo/climate_indices/issues/774),
against literature-derived worked examples in
`tests/fixture/pet_literature/`:

- `test_eto_thornthwaite_literature_watson` — Thornthwaite worked example
  from Watson & Burnett (1995).
- `test_eto_hargreaves_literature_mehta` — Hargreaves-Samani worked example
  from Mehta (2006).

Neither the original Thornthwaite (1948) *Geographical Review* article nor
the original Hargreaves & Samani (1985) *Applied Engineering in Agriculture*
article is freely accessible from this environment (JSTOR paywall for the
former; the latter's hosts were unreachable). Both worked examples are
instead taken from secondary sources that reproduce the original equations
with full inputs and outputs — the same two worked examples used by PyETo
(https://github.com/woodcrafty/PyETo) to validate its own Thornthwaite and
Hargreaves implementations, which `climate_indices.eto` credits in its
module docstring as the code it was derived from. Full citations, and the
derivation of the (latitude, day-of-year) inputs each worked example
implies, are documented in `tests/fixture/pet_literature/metadata.json`.

## Tolerance

- Thornthwaite: `atol=4.0` mm/month, matching the tolerance PyETo's own test
  suite established for this exact source (its rounded coefficients and
  intermediate values, and no day-count adjustment, introduce a few mm of
  imprecision independent of any implementation difference). The worked
  example's monthly daylight hours are matched to an approximate
  back-solved latitude (43.0°N) since the source states daylight hours
  directly rather than a latitude; see metadata.json.
- Hargreaves: `atol=0.05` mm/day, matching the source's 1-decimal-place
  precision. The worked example's extraterrestrial radiation input is
  reproduced exactly via a back-solved (latitude, day-of-year) pair; see
  metadata.json.
- FAO-56 Penman-Monteith: `atol=0.05` mm/day, matching the rounded
  intermediate values printed in FAO-56 Example 18. The radiation and wind
  helper tests use a `0.05`-`0.1` absolute tolerance against the 1-decimal
  values printed in Examples 10-16.

## References

- Thornthwaite, C.W. (1948). An Approach toward a Rational Classification
  of Climate. Geographical Review, 38, 55-94.
  https://doi.org/10.2307/210739
- Hargreaves, G.H. and Samani, Z.A. (1985). Reference Crop Evapotranspiration
  from Temperature. Applied Engineering in Agriculture, 1(2), 96-99.
- Allen, R.G., Pereira, L.S., Raes, D., and Smith, M. (1998). Crop
  Evapotranspiration - Guidelines for Computing Crop Water Requirements.
  FAO Irrigation and Drainage Paper 56. https://www.fao.org/4/x0490e/x0490e00.htm
- Watson, I. and Burnett, A.D. (1995). Hydrology: An Environmental Approach.
  Lewis Publishers/CRC Press. ISBN 978-1-56670-087-0.
- Mehta, V.K. (2006). Estimating Evapotranspiration from Weather Data.
  Arghyam / Cornell University.
- Richards, M. PyETo. https://github.com/woodcrafty/PyETo
