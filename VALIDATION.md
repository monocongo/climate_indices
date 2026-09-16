# Validation Status

This document records the validation state for the v2.5 release artifacts. It
separates external scientific validation from regression coverage so users can
see which results are backed by independent reference data and which are still
guarded only by internal fixtures.

## Test Commands

| Scope | Command | Expected result |
| --- | --- | --- |
| Core suite | `uv run pytest -m "not benchmark and not validation"` | Unit, property, xarray, release guardrail, and regression tests pass. |
| Validation marker suite | `uv run pytest -m validation` | External validation tests, including committed NOAA EDDI fixtures, pass; Palmer and scPDSI regression coverage passes. |
| Lint | `uv run ruff check src/ tests/` | No lint findings. |
| Format | `uv run ruff format --check src/ tests/` | No formatting changes needed. |
| Notebooks | `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/xarray_getting_started.ipynb notebooks/palmer_indices_xarray.ipynb notebooks/eddi_xarray.ipynb` | All v2.5 notebooks execute from a clean checkout. |
| Notebook smoke (Zarr/Dask tutorial) | `bash scripts/smoke_e2e_notebook.sh` | `notebooks/zarr_dask_spi_spei.ipynb` executes end to end from a fresh kernel against the prepared sample inputs. |

## Per-Index Evidence

| Index | Status | Tolerance | Evidence | Known gap |
| --- | --- | --- | --- | --- |
| SPI | Validated (internal) + qualified external comparison (Pearson III) | Gamma `atol=1e-8`, Pearson `atol=1e-5` in xarray equivalence tests; NOAA NCEI climdiv characterization ceilings documented in `tests/test_ncei_spi_reference.py` | Legacy NumPy fixtures and xarray equivalence tests compare gamma and Pearson outputs across scales. `tests/test_ncei_spi_reference.py` characterizes `indices.spi()` (Pearson III, full-period-of-record calibration) against NOAA NCEI's operational climdiv SPI product for all 344 climate divisions and all 7 published timescales; marked `validation`. | `tests/fixture/ncei_spi/provenance.json` records that NCEI's climate-divisional SPI product's independence from this codebase's lineage could not be confirmed as airtight from public documentation alone (see `docs/research/spi-dataset-survey.md`), and that NCEI's actual calibration behavior is a full/expanding period-of-record window rather than its documented fixed 1931-1990 baseline (Baldwin & Chen 2020). WMO SPI User Guide worked-example extraction remains outstanding (issue #778). |
| SPEI | Validated | Gamma `atol=1e-8`, Pearson `atol=1e-5` in xarray equivalence tests | Legacy NumPy fixtures and xarray equivalence tests compare gamma and Pearson outputs across scales. | Existing fixtures are historical project references rather than newly extracted literature tables. |
| PET Thornthwaite | Validated | Synthetic-fixture `atol=0.001`; literature worked example `atol=4.0` mm/month | `tests/test_eto.py::test_eto_thornthwaite` covers synthetic regression fixtures; `test_eto_thornthwaite_literature_watson` compares against a Thornthwaite (1948) worked example (Watson & Burnett, 1995) in `tests/fixture/pet_literature/`. See `docs/algorithm_refs/pet.md`. | The source supplies daylight hours, not latitude; coordinate reconstruction is not independently validated. |
| PET Hargreaves | Validated | Synthetic-input sanity checks; literature worked example `atol=0.05` mm/day | `tests/test_eto.py::test_eto_hargreaves_with_fixtures` checks synthetic daily-input output shape and bounds; `test_eto_hargreaves_literature_mehta` compares against a Hargreaves-Samani (1985) worked example (Mehta, 2006) in `tests/fixture/pet_literature/`. See `docs/algorithm_refs/pet.md`. | The source supplies extraterrestrial radiation, not latitude/day of year; coordinate reconstruction is not independently validated. |
| PNP | Validated | Existing fixture tolerances in `tests/test_indices.py` and xarray PNP tests | Percent-of-normal fixture and xarray wrapper tests cover output shape and metadata. | None blocking v2.5. |
| PCI | Validated | Existing fixture tolerances in `tests/test_indices.py` and xarray PCI tests | Daily rainfall fixture and scalar xarray wrapper tests cover PCI. | None blocking v2.5. |
| EDDI | Validated | NOAA reference tests use `rtol=1e-5`, `atol=1e-5` | Committed paired NOAA PSL monthly reference ET/EDDI fixtures for 1-, 3-, and 6-month Timescales, 1979–2023, pass in `tests/test_noaa_eddi_reference.py`; maximum observed error is `2.43e-6`. | This is one fixed latitude/longitude subset, not a CONUS-wide assessment. |
| KBDI | Reference-reproduced, not independently validated | Published Figure 1 series reproduced with `atol=4.0` hundredths of an inch (table-vs-equation discretization, measured maximum 2.91); GHCN station regression `rtol=1e-12`, `atol=1e-9` mm | `tests/test_fire_kbdi_reference.py` reproduces the Keetch & Byram (1968) Figure 1 worked example from a committed fixture, audits the published net-rain wet-spell rule, and runs a complete 30-year GHCN-Daily PRCP/TMAX record for Fresno, California against frozen values from an independent plain-Python calculator; the xclim (Finkele variant) and WFAS cross-checks skip when their optional inputs are absent. Recurrence, gap, and unit contracts are covered by `tests/test_fire_kbdi.py`. | Figure 1 is an exact oracle only for the 1968 integer table workflow, not the corrected continuous Equation 18 the library implements; xclim is a later Australian/FFDI variant, and no reproducible WFAS point oracle was obtainable. |
| CFFWIS (FFMC, DMC, DC, ISI, BUI, FWI, DSR) | Validated against the NRCan reference implementation | All seven components `atol=1e-9` against full-precision `cffdrs_py` reference outputs; xclim sanity check at `atol=1e-9` for DC/DMC/BUI and `atol=0.5` for the FFMC lineage | `tests/test_fire_cffwis_reference.py` runs the 48-day Van Wagner & Pickett (1985) calibration sample committed as `tests/fixture/cffwis_vwp1985/` against reference values generated at `cffdrs_py` commit `0f57fcc`; the xclim cross-check skips when xclim is absent. Recurrence, gap, unit, and seasonal-carry contracts remain covered by `tests/test_fire_cffwis_moisture.py`, `tests/test_fire_cffwis_behavior.py`, and `tests/test_fire_dc_overwintering.py`. | The committed record is one calibration sample at latitude 40 N, so the external check exercises the northern DMC/DC day-length band only; the other latitude bands are covered by frozen regression vectors, and no independent (non-NRCan-lineage) CFFWIS oracle exists. |
| Palmer PDSI/PHDI/PMDI/Z-Index and scPDSI family | Regression covered, not independently validated | Palmer and scPDSI regression tests use `atol=5e-5`, `rtol=0`; nClimDiv characterization ceilings documented in `tests/test_nclimdiv_reference.py` | `tests/test_palmer.py` exercises the standard Palmer fixtures; `tests/test_scpdsi.py` compares four self-calibrating outputs and fitted duration factors against Wells-lineage reference fixtures for all 344 climate divisions; `tests/test_nclimdiv_reference.py` characterizes aggregate `pdsi()`/`scpdsi()` agreement against the NOAA NCEI nClimDiv reference arrays. All three are marked `validation`. | `tests/fixture/palmer/provenance.json` distinguishes library-generated standard Palmer outputs from Wells-lineage scPDSI outputs. These fixture sets protect against regressions but are not treated as independent authoritative scientific validation. |

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

## Palmer Authoritative-Reference Decision

The committed Palmer fixture set remains valuable as a regression suite because
it covers hundreds of climate divisions and all four Palmer outputs. It is not
treated as independent validation for v2.5 because its provenance identifies the
source as "climate_indices library reference output."

Before closing the v2.5 scientific validation checklist, one of these must be
completed:

1. Replace or supplement the committed Palmer outputs with independently
   generated reference values from a documented NOAA/NCEI operational source.
2. Add a small published numerical example extracted from Palmer (1965) or a
   later authoritative implementation note, with provenance and tolerances.
3. Explicitly defer independent Palmer validation in the release issue and keep
   the current tests labeled as regression coverage only.

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
See "Palmer Authoritative-Reference Decision" above for the latter.

## References

- Hobbins, M. T., Wood, A., McEvoy, D. J., Huntington, J. L., Morton, C.,
  Anderson, M., and Hain, C. (2016). The Evaporative Demand Drought Index.
  Part I. Journal of Hydrometeorology, 17, 1745-1761.
  https://doi.org/10.1175/JHM-D-15-0121.1
- Palmer, W. C. (1965). Meteorological Drought. U.S. Weather Bureau Research
  Paper No. 45. https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf
- NOAA Physical Sciences Laboratory EDDI archive:
  https://downloads.psl.noaa.gov/Projects/EDDI/
- Van Wagner, C. E., and Pickett, T. L. (1985). Equations and FORTRAN
  program for the Canadian Forest Fire Weather Index System. Canadian
  Forestry Service, Forest Technical Report 33.
  https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19973.pdf
