# Validation Status

This document records the validation state for the 3.0.0 release artifacts. It
separates external scientific validation from regression coverage so users can
see which results are backed by independent reference data and which are still
guarded only by internal fixtures.

## Test Commands

| Scope | Command | Expected result |
| --- | --- | --- |
| Core suite | `uv run pytest -m "not benchmark and not validation"` | Unit, property, xarray, release guardrail, and regression tests pass. |
| Validation marker suite | `uv run pytest -m validation` | External validation tests, including committed NOAA EDDI fixtures, the SPEIbase v2.11 SPEI plausibility floors, and the standard-Palmer nClimDiv comparison, pass; scPDSI oracle cross-validation and all regression coverage pass. |
| Lint | `uv run ruff check src/ tests/` | No lint findings. |
| Format | `uv run ruff format --check src/ tests/` | No formatting changes needed. |
| Notebooks | `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/xarray_getting_started.ipynb notebooks/palmer_indices_xarray.ipynb notebooks/eddi_xarray.ipynb` | All 3.0.0 notebooks execute from a clean checkout. |
| Notebook smoke (Zarr/Dask tutorial, local only) | `bash scripts/smoke_e2e_notebook.sh` | `notebooks/zarr_dask_spi_spei.ipynb` executes end to end from a fresh kernel against the prepared sample inputs. |

## Per-Index Evidence

| Index | Status | Tolerance | Evidence | Known gap |
| --- | --- | --- | --- | --- |
| SPI | Validated (internal) + qualified external comparison (Pearson III) | Gamma `atol=1e-8`, Pearson `atol=1e-5` in xarray equivalence tests; NOAA NCEI climdiv characterization ceilings documented in `tests/test_ncei_spi_reference.py` | Legacy NumPy fixtures and xarray equivalence tests compare gamma and Pearson outputs across scales. `tests/test_ncei_spi_reference.py` characterizes `indices.spi()` (Pearson III, full-period-of-record calibration) against NOAA NCEI's operational climdiv SPI product for all 344 climate divisions and all 7 published timescales; marked `validation`. | `tests/fixture/ncei_spi/provenance.json` records that NCEI's climate-divisional SPI product's independence from this codebase's lineage could not be confirmed as airtight from public documentation alone (see `docs/research/spi-dataset-survey.md`), and that NCEI's actual calibration behavior is a full/expanding period-of-record window rather than its documented fixed 1931-1990 baseline (Baldwin & Chen 2020). WMO SPI User Guide worked-example extraction remains outstanding (issue #778). |
| SPEI | Validated (internal) + external plausibility check (gamma) | Gamma `atol=1e-8`, Pearson `atol=1e-5` in xarray equivalence tests; SPEIbase v2.11 agreement floors documented in `tests/test_speibase_reference.py` | Legacy NumPy fixtures and xarray equivalence tests compare gamma and Pearson outputs across scales. `tests/test_speibase_reference.py` compares `indices.spei()` (gamma distribution, Thornthwaite PET, full-period-of-record calibration 1901-2022) against areal-average CSIC SPEIbase v2.11 series for three CONUS climate divisions and timescales 1/3/6/12; marked `validation`. | Plausibility only, not numerical validation: SPEIbase uses FAO-56 Penman-Monteith PET and the log-logistic distribution while the compared series uses Thornthwaite PET and gamma, and the two series use different precipitation inputs and spatial support. See "SPEI Plausibility Classification" below. |
| PET Thornthwaite | Validated | Synthetic-fixture `atol=0.001`; literature worked example `atol=4.0` mm/month | `tests/test_eto.py::test_eto_thornthwaite` covers synthetic regression fixtures; `test_eto_thornthwaite_literature_watson` compares against a Thornthwaite (1948) worked example (Watson & Burnett, 1995) in `tests/fixture/pet_literature/`. See `docs/algorithm_refs/pet.md`. | The source supplies daylight hours, not latitude; coordinate reconstruction is not independently validated. |
| PET Hargreaves | Validated | Synthetic-input sanity checks; literature worked example `atol=0.05` mm/day | `tests/test_eto.py::test_eto_hargreaves_with_fixtures` checks synthetic daily-input output shape and bounds; `test_eto_hargreaves_literature_mehta` compares against a Hargreaves-Samani (1985) worked example (Mehta, 2006) in `tests/fixture/pet_literature/`. See `docs/algorithm_refs/pet.md`. | The source supplies extraterrestrial radiation, not latitude/day of year; coordinate reconstruction is not independently validated. |
| PNP | Validated | Existing fixture tolerances in `tests/test_indices.py` and xarray PNP tests | Percent-of-normal fixture and xarray wrapper tests cover output shape and metadata. | None blocking 3.0.0. |
| PCI | Validated | Existing fixture tolerances in `tests/test_indices.py` and xarray PCI tests | Daily rainfall fixture and scalar xarray wrapper tests cover PCI. | None blocking 3.0.0. |
| EDDI | Validated | NOAA reference tests use `rtol=1e-5`, `atol=1e-5` | Committed paired NOAA PSL monthly reference ET/EDDI fixtures for 1-, 3-, and 6-month Timescales, 1979–2023, pass in `tests/test_noaa_eddi_reference.py`; maximum observed error is `2.43e-6`. | This is one fixed latitude/longitude subset, not a CONUS-wide assessment. |
| KBDI | Reference-reproduced, not independently validated | Published Figure 1 series reproduced with `atol=4.0` hundredths of an inch (table-vs-equation discretization, measured maximum 2.91); GHCN station regression `rtol=1e-12`, `atol=1e-9` mm | `tests/test_fire_kbdi_reference.py` reproduces the Keetch & Byram (1968) Figure 1 worked example from a committed fixture, audits the published net-rain wet-spell rule, and runs a complete 30-year GHCN-Daily PRCP/TMAX record for Fresno, California against frozen values from an independent plain-Python calculator; the xclim (Finkele variant) and WFAS cross-checks skip when their optional inputs are absent. Recurrence, gap, and unit contracts are covered by `tests/test_fire_kbdi.py`. | Figure 1 is an exact oracle only for the 1968 integer table workflow, not the corrected continuous Equation 18 the library implements; xclim is a later Australian/FFDI variant, and no reproducible WFAS point oracle was obtainable. |
| CFFWIS (FFMC, DMC, DC, ISI, BUI, FWI, DSR) | Validated against the NRCan reference implementation | All seven components `atol=1e-9` against full-precision `cffdrs_py` reference outputs; xclim sanity check at `atol=1e-9` for DC/DMC/BUI and `atol=0.5` for the FFMC lineage | `tests/test_fire_cffwis_reference.py` runs the 48-day Van Wagner & Pickett (1985) calibration sample committed as `tests/fixture/cffwis_vwp1985/` against reference values generated at `cffdrs_py` commit `0f57fcc`; the xclim cross-check skips when xclim is absent. Recurrence, gap, unit, and seasonal-carry contracts remain covered by `tests/test_fire_cffwis_moisture.py`, `tests/test_fire_cffwis_behavior.py`, and `tests/test_fire_dc_overwintering.py`. | The committed record is one calibration sample at latitude 40 N, so the external check exercises the northern DMC/DC day-length band only; the other latitude bands are covered by frozen regression vectors, and no independent (non-NRCan-lineage) CFFWIS oracle exists. |
| PDSI, PHDI, PMDI, Z-Index (standard Palmer) | Qualified independent external-product validation | nClimDiv comparison ceilings documented in `tests/test_nclimdiv_reference.py`; library fixtures `atol=5e-5`, `rtol=0` | `tests/test_nclimdiv_reference.py::test_pdsi_vs_noaa_nclimdiv_qualified_external_validation` compares `pdsi()` against the operational NOAA NCEI nClimDiv arrays for all 344 climate divisions, January 1895 through December 2022 (measured median absolute difference 0.0127; 86.2% of months within 0.05). `tests/test_palmer.py` retains the library-generated standard-Palmer fixtures as regression coverage. Both are marked `validation`. | Qualified rather than absolute: NCEI publishes two decimal places, and the committed `precips.npy`/`pet.npy` inputs are not byte-identical to NCEI's operational inputs, so the practical agreement floor is ~0.013 rather than 0.005. See "Palmer Validation Classification" below. |
| scPDSI | Independent implementation cross-validation; nClimDiv comparison is characterization only | Wells-lineage oracle fixtures `atol=5e-5`, `rtol=0` in `tests/test_scpdsi.py`; nClimDiv characterization ceilings in `tests/test_nclimdiv_reference.py` | `tests/test_scpdsi.py` compares the four self-calibrating outputs and fitted duration factors against Wells-lineage reference fixtures for all 344 climate divisions. `tests/test_nclimdiv_reference.py::test_scpdsi_vs_noaa_nclimdiv_characterization` characterizes aggregate agreement against nClimDiv, which applies standard Palmer's fixed national calibration rather than per-division self-calibration. Both are marked `validation`. | No qualified external-product reference yet; assessing DRI/WRCC scPDSI for that role remains open (issue #780). The NCEI Fortran and Wells C++ recursion lineages disagree on roughly 6.3% of months by up to 8 PDSI units, and on those months the NCEI-lineage values sit closer to NOAA operational data 88% of the time (`tests/fixture/palmer/provenance.json`). |

## EDDI Fixture Policy

The paired NOAA PSL EDDI fixtures are committed and run in CI. They contain
monthly reference ET and 1-, 3-, and 6-month EDDI for latitude 39.75–39.875
and longitude -105.0–-104.875, with the 1979–2023 table baseline.

Required fixture layout:

```text
tests/fixture/noaa-eddi-1month/
tests/fixture/noaa-eddi-3month/
tests/fixture/noaa-eddi-6month/
```

Each directory must contain:

- `provenance.json`
- `metadata.json`
- `pet_input.npy`
- `eddi_reference.npy`

The provenance file follows `tests/fixture/provenance_schema.json`, cites the
NOAA PSL EDDI time-series source, records checksums, and states the calibration
period. The validation tests compare non-NaN reference values and separately
verify NaN placement. `scripts/prepare_noaa_eddi_fixtures.py` refuses an
unexpected table baseline so fixture refreshes are intentional.

## SPEI Plausibility Classification

`indices.spei()` (gamma distribution, Thornthwaite PET) is compared against
areal-average CSIC SPEIbase v2.11 series in `tests/test_speibase_reference.py`,
for three CONUS climate divisions spanning an aridity gradient (humid Alabama
0101, semi-arid Oklahoma 3405, arid southwest Arizona 0205) and timescales 1,
3, 6 and 12 months.

This comparison is a **plausibility check, not external numerical
validation**, and is deliberately not enforced with `atol`/`rtol`. Three
confounds separate the two products:

- SPEIbase v2.11 uses FAO-56 Penman-Monteith PET from CRU TS 4.09, while the
  compared series uses Thornthwaite PET computed from the committed nClimDiv
  temperatures; the two PET families diverge with climate aridity (van der
  Schrier et al. 2011).
- SPEIbase standardizes with the log-logistic distribution; climate_indices
  has no log-logistic implementation (issue #106), so the compared series uses
  gamma.
- The reference is a 0.5-degree grid average inside a climate division polygon
  while the compared series is the division's station-derived areal average,
  from different precipitation inputs (CRU TS vs. nClimDiv).

The tests therefore assert per-division, per-timescale floors on correlation,
sign agreement, and drought-category agreement, derived from the measurements
recorded in `tests/fixture/speibase/provenance.json`;
`scripts/prepare_speibase_fixtures.py` re-measures those values from the arrays
it just built, refusing to publish a refresh that drifts beyond `0.001` of the
recorded expectations, and regenerates the floors from the new measurements;
`test_floors_keep_documented_slack` pins every floor to within a documented
band below its measurement so a floor cannot be widened to hide a regression. Measured correlation is strongest in the semi-arid Oklahoma
division (0.95-0.97 across timescales) and weakest in the arid Arizona
division (0.82-0.86), the pattern the PET-method divergence predicts; a tight
numerical gate would either fail on that legitimate climate-dependent bias or
have to be widened until it no longer tested anything.

The compared climate_indices series is calibrated against the full 1901-2022
period of record, matching the committed inputs, while SPEIbase v2.11 is
standardized on its own 1901-2024 record; the fixtures retain the 1901-2022
overlap. The dataset survey behind this comparison lives at
`docs/research/spei-dataset-survey.md` on branch `research/spei-dataset-survey`.

## Palmer Validation Classification

The Palmer evidence splits by product family, because the two consume different
reference artefacts.

**Standard Palmer (`pdsi()`: PDSI, PHDI, PMDI, Z-Index)** is qualified
independent external-product validation against the operational NOAA NCEI
nClimDiv arrays in `tests/fixture/nclimdiv/`, enforced by
`tests/test_nclimdiv_reference.py::test_pdsi_vs_noaa_nclimdiv_qualified_external_validation`.
Measured over 528,384 division-months, `pdsi()` agrees with nClimDiv to a median
absolute difference of 0.0127, with 86.2% of months within 0.05. The
qualification is deliberate and recorded rather than averaged away: NCEI
publishes two decimal places, and the committed `precips.npy`/`pet.npy` inputs
are not byte-identical to NCEI's operational inputs, so the practical agreement
floor is near 0.013 rather than the 0.005 that published precision alone would
permit. The library-generated standard-Palmer fixtures (`tests/fixture/palmer/`)
remain in the suite as regression coverage only -- their provenance identifies
the source as "climate_indices library reference output" -- and they are no
longer the family's only evidence.

**scPDSI (`scpdsi()`)** has independent implementation cross-validation, not
external-product validation: `tests/test_scpdsi.py` compares the four
self-calibrating outputs and fitted duration factors against the Wells-lineage
C++ oracle fixtures at `atol=5e-5`, `rtol=0`. The nClimDiv comparison in
`tests/test_nclimdiv_reference.py::test_scpdsi_vs_noaa_nclimdiv_characterization`
stays explicitly classified as characterization, because nClimDiv uses standard
Palmer's fixed national K-factors while `scpdsi()` self-calibrates per division.
The two recursion lineages disagree on roughly 6.3% of months by up to 8 PDSI
units (the NCEI-lineage values are closer to NOAA operational data on 88% of
those months; see `tests/fixture/palmer/provenance.json`), so neither series can
serve as the other's tolerance oracle.

No further standard-Palmer external reference is required for this validation
contract. scPDSI external-product validation remains open pending the DRI/WRCC
assessment in issue #780.

## scPDSI Calibration-Anchor Measurement

Wells' self-calibration methodology tunes each climate division's duration
and K-prime factors so that, by construction, the calibration-period 2nd and
98th percentiles of the resulting scPDSI series land at exactly -4.0000 and
+4.0000. (`scpdsi()` takes those percentiles of the PDSI output and rescales
the Z-index to match; the Z-index is the quantity adjusted, not the quantity
anchored.) Measuring how closely `scpdsi()` reproduces that anchor across all
344 climate divisions (calibration period 1931-1990) verifies the
self-calibration percentile-scaling step independently of the NOAA nClimDiv
comparison in `tests/test_nclimdiv_reference.py` (see the "Per-Index Evidence"
table above), which instead measures agreement with an external product that
uses different (fixed, national) K-factors.

| Percentile anchor | Median | Mean | Median deviation | Max deviation | Within +/-0.05 | Within +/-0.25 |
| --- | --- | --- | --- | --- | --- | --- |
| 2nd (target -4.0000) | -4.0000 | -4.0142 | 0.0102 | 0.9730 | 74% | 93% |
| 98th (target +4.0000) | +4.0000 | +3.9829 | 0.0130 | 1.5421 | 70% | 91% |

The medians land on the target exactly while the maxima do not, because
`scpdsi()` applies a fixed three rescaling passes rather than iterating to a
fixed point. Divisions whose recursion has not settled within three passes
retain the residual deviations shown above; this is the reference behaviour,
not a defect. These figures are regenerated and bounded by
`test_scpdsi_calibration_anchor_lands_on_target` in
`tests/test_nclimdiv_reference.py`, so they fail a test rather than rotting
silently if the rescaling changes.

This is regression coverage over the committed fixture set, not independent
scientific validation: it confirms the self-calibration percentile-scaling
arithmetic behaves as Wells describes for these 344 divisions, not that the
resulting scPDSI values themselves match an external authoritative source.
See "Palmer Validation Classification" above for the latter.

## References

- Beguería, S., Vicente-Serrano, S. M., Reig, F., and Latorre, B. (2014).
  Standardized precipitation evapotranspiration index (SPEI) revisited:
  parameter fitting, evapotranspiration models, tools, datasets and drought
  monitoring. International Journal of Climatology, 34(10), 3001-3023.
  https://doi.org/10.1002/joc.3887
- Hobbins, M. T., Wood, A., McEvoy, D. J., Huntington, J. L., Morton, C.,
  Anderson, M., and Hain, C. (2016). The Evaporative Demand Drought Index.
  Part I. Journal of Hydrometeorology, 17, 1745-1761.
  https://doi.org/10.1175/JHM-D-15-0121.1
- Palmer, W. C. (1965). Meteorological Drought. U.S. Weather Bureau Research
  Paper No. 45. https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf
- NOAA Physical Sciences Laboratory EDDI archive:
  https://downloads.psl.noaa.gov/Projects/EDDI/
- van der Schrier, G., Jones, P. D., and Briffa, K. R. (2011). The sensitivity
  of the PDSI to the Thornthwaite and Penman-Monteith parameterizations for
  potential evapotranspiration. Journal of Geophysical Research: Atmospheres,
  116, D03106. https://doi.org/10.1029/2010JD015001
- Vicente-Serrano, S. M., Beguería, S., and López-Moreno, J. I. (2010). A
  multiscalar drought index sensitive to global warming: the Standardized
  Precipitation Evapotranspiration Index. Journal of Climate, 23(7),
  1696-1718. https://doi.org/10.1175/2009JCLI2909.1
- Van Wagner, C. E., and Pickett, T. L. (1985). Equations and FORTRAN
  program for the Canadian Forest Fire Weather Index System. Canadian
  Forestry Service, Forest Technical Report 33.
  https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19973.pdf
