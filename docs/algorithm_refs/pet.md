# Potential Evapotranspiration (PET)

## Scope

`climate_indices.eto` implements two PET methods:

- **Thornthwaite (1948)** — monthly PET from mean monthly air temperature,
  latitude, and calendar-derived daylight hours.
- **Hargreaves-Samani (1985)** — daily PET (ETo) from minimum, maximum, and
  mean daily air temperature, latitude, and day of year (via
  extraterrestrial radiation), following Equation 52 in Allen et al. (1998)
  FAO Irrigation and Drainage Paper 56.

## Public API

Use `climate_indices.eto.eto_thornthwaite()` and
`climate_indices.eto.eto_hargreaves()`. Both accept NumPy arrays; xarray
support is wired through `xarray_adapter.py` where applicable.

## Validation

PET has no single authoritative external gridded dataset: comparing against
a reanalysis-driven reference-ET product (e.g. gridMET) confounds algorithm
correctness with genuine climate-dependent bias between PET formulas, so no
defensible numerical tolerance exists (see the decision on
[Identify a gridded reference-ET dataset for PET stretch validation](https://github.com/monocongo/climate_indices/issues/775)).
Literature worked examples are therefore the primary independent validation
evidence for PET.

`tests/test_eto.py` covers both methods against synthetic regression
fixtures (`tests/fixture/temp_celsius.npy`, `tests/fixture/pet_thornthwaite.npy`,
and the `hargreaves_*` fixtures generated in `tests/conftest.py`) and, since
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
