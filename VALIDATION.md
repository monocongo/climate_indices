# Validation Status

This document records the validation state for the v2.5 release artifacts. It
separates external scientific validation from regression coverage so users can
see which results are backed by independent reference data and which are still
guarded only by internal fixtures.

## Test Commands

| Scope | Command | Expected result |
| --- | --- | --- |
| Core suite | `uv run pytest -m "not benchmark and not validation"` | Unit, property, xarray, release guardrail, and regression tests pass. |
| Validation suite | `uv run pytest -m validation` | External validation tests pass where fixtures are present; missing external data is skipped with an explicit reason. |
| Lint | `uv run ruff check src/ tests/` | No lint findings. |
| Format | `uv run ruff format --check src/ tests/` | No formatting changes needed. |
| Notebooks | `uv run jupyter nbconvert --execute --to notebook --inplace notebooks/xarray_getting_started.ipynb notebooks/palmer_indices_xarray.ipynb notebooks/eddi_xarray.ipynb` | All v2.5 notebooks execute from a clean checkout. |

## Per-Index Evidence

| Index | Status | Tolerance | Evidence | Known gap |
| --- | --- | --- | --- | --- |
| SPI | Validated | Gamma `atol=1e-8`, Pearson `atol=1e-5` in xarray equivalence tests | Legacy NumPy fixtures and xarray equivalence tests compare gamma and Pearson outputs across scales. | Existing fixtures are historical project references rather than newly extracted literature tables. |
| SPEI | Validated | Gamma `atol=1e-8`, Pearson `atol=1e-5` in xarray equivalence tests | Legacy NumPy fixtures and xarray equivalence tests compare gamma and Pearson outputs across scales. | Existing fixtures are historical project references rather than newly extracted literature tables. |
| PET Thornthwaite | Validated | Existing fixture tolerances in `tests/test_eto.py` | Thornthwaite fixtures are exercised by PET and Palmer tests. | None blocking v2.5. |
| PET Hargreaves | Validated | Existing fixture tolerances in `tests/test_eto.py` | Hargreaves tests cover daily temperature inputs. | None blocking v2.5. |
| PNP | Validated | Existing fixture tolerances in `tests/test_indices.py` and xarray PNP tests | Percent-of-normal fixture and xarray wrapper tests cover output shape and metadata. | None blocking v2.5. |
| PCI | Validated | Existing fixture tolerances in `tests/test_indices.py` and xarray PCI tests | Daily rainfall fixture and scalar xarray wrapper tests cover PCI. | None blocking v2.5. |
| EDDI | Partially validated | NOAA reference tests use `rtol=1e-5`, `atol=1e-5` when fixtures exist | Algorithm property tests always run; NOAA PSL reference tests live in `tests/test_noaa_eddi_reference.py` and are marked `validation`. | `tests/fixture/noaa-eddi-{1,3,6}month/` is not committed. The validation tests skip until independently prepared PET/input and NOAA reference outputs are available. |
| Palmer PDSI/PHDI/PMDI/Z-Index | Regression covered, not independently validated | Palmer regression tests use `atol=5e-5`, `rtol=0` | `tests/test_palmer.py` exercises all committed climate-division fixtures and is marked `validation`. | `tests/fixture/palmer/provenance.json` states the outputs were generated from this library. These fixtures protect against regressions but do not satisfy independent authoritative-output validation. |

## EDDI Fixture Policy

The NOAA EDDI validation tests intentionally skip when the reference fixture
directories are absent. This keeps normal CI reproducible without committing a
large external dataset, while still making the validation contract executable
for release candidates.

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

The provenance file must follow `tests/fixture/provenance_schema.json`, cite the
NOAA PSL EDDI source, record checksums, and state the calibration period used.
The validation tests compare only non-NaN reference values and separately verify
NaN placement.

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

## References

- Hobbins, M. T., Wood, A., McEvoy, D. J., Huntington, J. L., Morton, C.,
  Anderson, M., and Hain, C. (2016). The Evaporative Demand Drought Index.
  Part I. Journal of Hydrometeorology, 17, 1745-1761.
  https://doi.org/10.1175/JHM-D-15-0121.1
- Palmer, W. C. (1965). Meteorological Drought. U.S. Weather Bureau Research
  Paper No. 45. https://www.droughtmanagement.info/literature/USWB_Meteorological_Drought_1965.pdf
- NOAA Physical Sciences Laboratory EDDI archive:
  https://downloads.psl.noaa.gov/Projects/EDDI/
