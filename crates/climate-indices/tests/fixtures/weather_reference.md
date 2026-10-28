# Standalone weather references

`weather_reference.rs` reads these committed files with Rust's standard-library
text parser. The tests use the existing `ndarray` dependency for kernel inputs.
Cargo does not start Python, load an extension, access the network, or generate
fixtures.

## Reproduction and source identity

The source revision is `f4cb938eb622a3ddf8b96c129337e0ebecd4121e` of
`monocongo/climate_indices`. The exact generation command, run from the repository
root, is:

```bash
PYTHONPATH=src uv run python scripts/prepare_rust_weather_fixtures.py
```

Each numeric file records the Python and NumPy versions used for generation.

The generator blocks `climate_indices._native` imports through `sys.modules`
before it imports the package. It also sets all four relevant dispatch handles
to `None` and asserts that the import block remains active. Thus, an installed
extension cannot supply expected values. The generator works without an extension.

The generator checks the source files against the recorded revision before it
writes fixtures. It refuses changed implementations or external input fixtures.
To refresh from another revision, review the source changes and update
`SOURCE_REVISION` explicitly.

The Python regression sources are:

- `src/climate_indices/eto.py` supplies `eto_thornthwaite` and `eto_hargreaves`.
- `src/climate_indices/pm_eto.py` supplies `pm_eto` and `penman_monteith_eto`.
- `src/climate_indices/fire/_cffwis.py` and `_cffwis_codes.py` supply the three
  moisture-code recurrences and day-length tables.
- `src/climate_indices/fire/_kbdi.py` supplies `kbdi`.
- `src/climate_indices/flood/_pe.py`, `_edi.py`, `_if.py`, and `_antecedent.py`
  supply the four flood calculations.
- `src/climate_indices/_recurrence.py` supplies the shared missing-day rules.

No Rust result enters any fixture. Synthetic inputs use small integer formulas,
not random-number generators. Python-generated outputs are regression evidence,
not independent scientific validation. The external references below retain
their separate classification and source limitations, as recorded in
`VALIDATION.md`.

## Format and comparison rules

A record starts with `NAME ROWS COLUMNS`. Its next `ROWS` lines contain `COLUMNS`
whitespace-separated float64 values in C order. Comment lines start with `#`.
Vectors use one row. Scalars use one row and one column. The generator writes
17 significant digits for round-trip precision and spells missing values `nan`.

All Python regression comparisons use
`abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected)`.
Every comparison checks the complete NaN mask exactly before comparing numbers.
The parser checks field uniqueness, and the tests check history shapes and value
counts. Gap counts use exact equality. No fixture contains infinity.

## Python regression files

### `weather_eto.txt`

`monthly_temperature` and `thornthwaite` have shape `(24, 4)`. Rust reshapes the
monthly temperatures to `(2, 12, 4)`: year, month, cell. Python receives a
`(24, 2, 2)` Spatial Block. Both layouts retain the same C-order cell sequence.
The years are 1999 and 2000; `leap_years` is `[0, 1]`.

`latitude_radians` contains the converted latitudes `[35, -35, 90, -90]` degrees.
Monthly input temperatures use degrees Celsius and are already clamped at zero,
as the Rust kernel requires. Thornthwaite output uses mm/month. There is one
missing monthly observation, but no completely missing calendar month.

`tmin`, `tmax`, `tmean`, and `hargreaves` have shape `(370, 4)`. Temperatures use
degrees Celsius; output uses mm/day. The rows follow the 366-day positional year.
The last four rows exercise a partial second year. One temperature range is
inverted, another is zero, and one input observation is missing. Both hemispheres
and both poles exercise the solar geometry.

### `weather_pm.txt`

The eight equation inputs have shape `(1, 6)`. Radiation and soil heat flux use
MJ m-2 day-1. Temperature uses degrees Celsius. Wind at 2 m uses m/s. Vapor
pressures use kPa, and `delta` and `gamma` use kPa per degree Celsius.

`core_0` contains six all-array results. `core_1` contains six results with only
net radiation varying; all other operands use their first input value.
`core_2` contains one all-scalar result. All outputs use mm/day. The tests also
exercise `Operand::at` with a scalar at `usize::MAX` and a strided array,
including its NaN element.

The meteorological inputs and each `met_H_R` output have shape `(1, 8)`.
Temperatures use degrees Celsius, latitude uses degrees north, wind uses m/s,
and day of year is one-based. Elevation is 100 m, wind measurement height is
10 m, soil heat flux is zero, and dimensionless albedo is 0.23.

`H` selects dewpoint (12.05 degrees Celsius), minimum/maximum relative humidity
(63/84 percent), maximum humidity (84 percent), mean humidity (70 percent), or
the minimum-temperature estimate, in order from zero through four.

`R` selects supplied solar radiation (22 MJ m-2 day-1), sunshine duration
(9.25 hours), the coastal temperature-range estimate, or the interior estimate,
in order from zero through three. The 20 combinations include cold weather,
calm wind, missing observations, zero temperature range, and polar night.

### `weather_fire.txt`

Weather, months, and season flags have shape `(18, 5)`. Each history has shape
`(16, 5)` because the first two days advance the state without recording output.
Seeds, final states, latitude bands, and final gap counts have shape `(1, 5)`.
The five latitudes are `[55, 20, 0, -20, -40]` degrees north. All 12 months occur.

Temperature uses degrees Celsius. Relative humidity uses percent. Stored wind
uses km/h; the generator passes m/s to Python, which converts it to km/h.
Rain, mean annual rain, KBDI, and wet-spell precipitation use mm. FFMC, DMC,
and DC are dimensionless. The DMC table has shape `(5, 12)` and contains hours.
The DC table has shape `(3, 12)` and contains the source equation's adjustment
term. Bands are zero-based and months are one-based.

The fixtures contain rain-threshold boundaries, wet spells, cold days, and
non-default seeds across the DMC rain-response ranges. Each code runs with both
`propagate` and `bridge` missing-day policies. The bridge allowance is two days.
Gaps include a leading missing day, a short gap, a three-day gap that exceeds
the allowance, and a trailing missing day. DC's season mask disables days eight
and nine, using zero-based positions; these days carry its current value.

Every run records its history, final code, and final gap counts. KBDI also
records its final wet-spell precipitation. Initial wet-spell precipitation is
zero, initial gap counts are -1, and all static inputs are valid.

The behavior indices are computed from those recorded histories with the public
Python functions, so `isi`, `bui`, `fwi`, and `dsr` have the `(16, 5)` shape of
the histories and are dimensionless. The overwinter start-up DC uses the same
final DC state and `overwinter_precipitation`
`[0, 200, 110, 50, 250]` mm at the reference fractions 0.75 and 0.75, giving
`overwinter_dc_propagate` and `overwinter_dc_bridge`, each shaped `(1, 5)` and
dimensionless.

### `weather_flood.txt`

`rain`, `pe_1`, `pe_2`, `pe_7`, and `flood_index` have shape `(1115, 3)`.
This is three complete 366-day years followed by 17 days. Python receives
`(1115, 3, 1)` Spatial Blocks. Rain and effective precipitation (PE) use mm.
The Flood Index is dimensionless.

The first cell varies between years and contains a missing observation. The
second has constant rain. The third has observations only in the first year.
The PE durations are one, two, and seven days. Warm-up rows and windows with
a missing observation remain NaN. The first cell subsequently recovers.

`pe_padded` and `edi` have shape `(4, 1098)`: year, then 366 calendar days times
three cells. The incomplete final year is padded with NaN. Calibration uses
2000 through 2002, represented by Rust rows `0..3`. EDI is dimensionless.
The Flood Index uses the first three complete January-start years, with Rust
`first_start=0` and `calibration_years=3`. Constant or insufficient calibration
samples cannot define a standard deviation and remain NaN.

`api_rain` has shape `(18, 3)`. Each API history has shape `(16, 3)`; states and
gap counts have shape `(1, 3)`. API uses mm, decay `k=0.9`, a zero initial state,
and a two-day spin-up. Both missing-day policies run with the same gap patterns
as the fire tests. The bridge allowance is two days.

These flood values are regression only. No external flood numeric reference is
claimed; see the evidence classification in `VALIDATION.md`.

## External reference files

### `weather_literature.txt`

The generator copies the PET arrays unchanged from
`tests/fixture/pet_literature/` at the source revision. That directory's
`metadata.json` and `provenance.json` contain the original citations and
coordinate-reconstruction details.

- `watson_temperature` and `watson_pet` each contain 12 monthly values.
  Watson and Burnett (1995), *Hydrology: An Environmental Approach*, Example
  19-2, reproduces the Thornthwaite calculation. Temperature uses degrees
  Celsius; PET uses mm/month. The absolute tolerance is 4.0 mm/month, with
  zero relative tolerance. The latitude of 43 degrees north is reconstructed,
  not supplied by the source; 2001 is an arbitrary non-leap year.
- `mehta_temperature` contains `[tmin, tmax, tmean]` in degrees Celsius.
  `mehta_pet` contains the Mehta (2006) worked-example value, 6.1 mm/day.
  The test uses day 58 and latitude -12.3 degrees to reproduce the source's
  extraterrestrial radiation. The absolute tolerance is 0.05 mm/day, with
  zero relative tolerance. The source does not independently validate the
  reconstructed coordinates.
- The one-element meteorological records and `fao56_pet` reproduce Allen et
  al. (1998), FAO Irrigation and Drainage Paper 56, Example 18, Uccle.
  The expected output is the printed 3.88 mm/day, not a Python calculation.
  Inputs match `tests/test_native_pet_parity.py` and `tests/test_pm_eto.py`:
  Tmin/Tmax 12.3/21.5 degrees Celsius, latitude 50.8 degrees north,
  elevation 100 m, day 187, wind 2.78 m/s measured at 10 m, RHmin/RHmax
  63/84 percent, and sunshine 9.25 hours. Soil heat flux is zero and albedo
  is 0.23. The absolute tolerance is 0.05 mm/day, with zero relative tolerance.

The immediate PET literature source is the upstream PyETo test suite:
<https://github.com/woodcrafty/PyETo/blob/master/tests/test_thornthwaite.py>
and <https://github.com/woodcrafty/PyETo/blob/master/tests/test_fao.py>.
The FAO source is <https://www.fao.org/4/x0490e/x0490e00.htm>.
These examples are external literature checks, not independent validation of
coordinate reconstruction or unrelated meteorological conditions.

### `weather_nrcan.txt`

The generator copies the 48-day weather record and FFMC/DMC/DC output columns
from `tests/fixture/cffwis_vwp1985/reference.csv` unchanged. These outputs come
from the NRCan `cffdrs_py` reference at commit
`0f57fcca2a6a84b69fe8d50f29d947f0be34d5f6`, not from `climate_indices`.

The source is Van Wagner and Pickett (1985), Canadian Forestry Service,
Forest Technical Report 33: <https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19973.pdf>.
The reference implementation's input record is
<https://github.com/cffdrs/cffdrs_py/blob/0f57fcca2a6a84b69fe8d50f29d947f0be34d5f6/cffdrs/tests/data/fwi_test_data.csv>.
The existing provenance file records attribution, licensing, and source history.

Weather and outputs have shape `(48, 1)`. Units match `weather_fire.txt`.
The latitude is 40 degrees north; all lookup bands use row zero. Seeds are
FFMC=85, DMC=6, and DC=15. There is no spin-up, missing observation, or seasonal
shutdown. The tests use the existing absolute tolerance of `1e-9`, with zero
relative tolerance and exact NaN masks.

This is an external reference-implementation comparison with shared NRCan
algorithm lineage. It is not an algorithmically independent validation. It
covers one northern calibration sample; Python regressions cover other bands.

## Public function coverage

The integration tests exercise every public numerical function in the four
requested modules:

- `eto::thornthwaite` and `eto::hargreaves` run against Python and literature values.
- `pm_eto::pm_eto`, `pm_eto::penman_monteith_eto`, and `pm_eto::Operand::at` run directly.
- `fire::ffmc`, `fire::dmc`, `fire::dc`, and `fire::kbdi` run against Python values.
  The first three also run against NRCan values.
- `fire::initial_spread_index`, `fire::buildup_index`, `fire::fwi`,
  `fire::daily_severity_rating`, and `fire::overwinter_dc` run against Python
  values derived from the recorded moisture-code histories.
- `flood::effective_precipitation`, `flood::edi`, `flood::flood_index`, and
  `flood::antecedent_precipitation_index` run against Python values.

No public numerical function in these modules is omitted. The fixtures do not
provide external numerical validation for KBDI or the flood family.
