# Reference-implementation papers: research notes

**Research date:** 2026-11-11
**Repository state read:** `monocongo/climate_indices` at commit `f21d3d9` (version 3.2.0)
**Purpose:** The sources behind the "known so far" statements in the REFIMPL tickets, and a list of what was not verified. The issue numbers for the tickets are in [`issue-map.json`](issue-map.json).

## Repository facts

Each statement was read in the file named.

| Statement | File |
| --- | --- |
| Per-index evidence classes, tolerances, and known gaps | `VALIDATION.md` |
| The earlier validation map excluded manuscript drafting; its open items are the scPDSI external product, reproducibility packaging, and the SPEI Global Drought Monitor stretch | Issue 769 body |
| Gamma fit is the closed form `alpha = (1 + sqrt(1 + 4A/3)) / (4A)` | `src/climate_indices/compute.py`, `gamma_parameters` |
| The same fit is called maximum likelihood in one document and method of moments in another | `docs/algorithm-reference.md` lines 37 and 341; `VALIDATION.md` SPI zero-placement row |
| Standard Palmer follows NCEI `pdi.f`; scPDSI follows the Wells recursion; they disagree on about 6.3% of months by up to 8 units | `drought/_palmer_pdi.py`, `drought/_palmer_wells.py`, `tests/fixture/palmer/provenance.json` |
| scPDSI oracle arrays came from GPL-3 `Sibada/scPDSI` run outside the repository | `tests/fixture/palmer/provenance.json` |
| scPDSI applies three rescaling passes; anchor deviations up to 0.97 and 1.54 | `VALIDATION.md`, "scPDSI Calibration-Anchor Measurement" |
| scPDSI is per-location only, with no xarray or CLI route | ADR-0011, `docs/research/dri-wrcc-scpdsi-assessment.md` |
| Per-series Rust speedups (PDSI 194x, scPDSI 62x) and nClimGrid timings (2.77x eager, 6.18x-9.38x pooled) | `benchmarks/README.md` |
| Earth Engine and Climate Engine pilot results and ten open provider questions | `docs/research/earthengine-climate-engine-pilot.md` |
| WestWide Drought Tracker scPDSI assessed and not adopted | `docs/research/dri-wrcc-scpdsi-assessment.md` |
| The committed AWC grid fixture is a 3 by 3 sample from POLARIS, CC BY-NC 4.0 | `tests/fixture/nclimgrid_awc/provenance.json` |
| nClimGrid period-of-record files are mutable; modified derivatives may be redistributed with attribution | `docs/research/nclimgrid-acquisition-and-redistribution.md` |
| `CITATION.cff` and `pyproject.toml` carry the same version (3.2.0); a release-time test compares them, and a Zenodo DOI is still pending | `CITATION.cff`, `pyproject.toml`, `tests/test_release_integrity.py` |
| Issue and label conventions, wayfinder map and child tickets | `docs/agent/issue-tracker.md` |
| Project policy prohibits attribution of work to an AI model or tool in commits, PRs, and issue text | `AGENTS.md` |

## External facts

| Statement | Source |
| --- | --- |
| `xclim` 1.0.0 SPI and SPEI accept the named distributions `gamma`, `fisk`, `genextreme`, and `lognorm`. Named `gamma` and `fisk` support `ML` and `APP`, while `PWM` requires a compatible distribution object. When fitting, SPI enables zero inflation and SPEI disables it. Both clip to ±8.21. The SPEI signature drops the former `offset` parameter. The SPI docstring describes matching `climate_indices` with `method="APP"` and `fitkwargs={"floc": 0}`, subject to different clipping bounds. The module has no Palmer function. | [SPI and SPEI source](https://xclim.readthedocs.io/en/v1.0.0/_modules/xclim/compute/_agro.html), [standardization source](https://xclim.readthedocs.io/en/v1.0.0/_modules/xclim/compute/stats.html#standardized_index) |
| `xclim` 0.40.0 history: SPEI offset changed "to better reproduce results in the reference library monocongo/climate_indices" | https://pypi.org/project/xclim/0.40.0 |
| `xclim` PET methods and default `BR65`; Hargreaves method modified in its pull request 1723 | https://github.com/Ouranosinc/xclim/pull/1723 |
| R `SPEI` fits by unbiased PWM by default; plotting-position PWM and maximum likelihood also available | https://rdrr.io/pkg/SPEI/man/Drought-indices.html |
| Beguería et al. (2014) recommend unbiased PWM; doi:10.1002/joc.3887 | Publisher abstract |
| `spei` (Vonk) published in JOSS 10(111), 8454, doi:10.21105/joss.08454; its notebook compares against `standard_precip`, `climate_indices`, `pastas`, and R `SPEI` | https://joss.theoj.org/papers/10.21105/joss.08454 and https://martinvonk.github.io/SPEI/examples/example04_package_comparison.html |
| `PyEt` v1.3.1 published in *GMD* 17, 7083-7103, 2024, as a model description paper; about twenty PET methods | https://gmd.copernicus.org/articles/17/7083/2024/ |
| R `SEI` in *The R Journal*, article RJ-2024-038; names `climate-indices`, `spei`, `SCI`, `SPEI` as existing tools | https://journal.r-project.org/articles/RJ-2024-038/ |
| Laimighofer and Laaha (2022): distribution and observation window dominate SPI and SPEI uncertainty; estimation method and goodness-of-fit test matter little | *Journal of Hydrology* 613, 128385 (abstract) |
| Jacobi et al. (2013): MATLAB Palmer tool checked against the NCDC Fortran, discrepancies documented; doi:10.1002/wrcr.20342 | Publisher abstract |
| Penman-Monteith extension of the Jacobi tool, with code and example data | https://github.com/dficklin/CAGEO-paper |
| `JonKing93/pdsi` MATLAB function | https://github.com/jonking93/pdsi |
| `Sibada/scPDSI`: R, GPL-3, version 0.1.3, based on the Wells, Goddard and Hayes C++ | https://rdrr.io/github/Sibada/scPDSI |
| `GRIDMET/DROUGHT`: non-parametric standardization over 1981-2016; trailing windows from 14 days to 5 years; PDSI and Z from a modified formulation, baseline 1979-2018, STATSGO AWC | https://developers.google.com/earth-engine/datasets/catalog/GRIDMET_DROUGHT |
| gridMET pentad PDSI follows Rhee and Carbone (2007) with modified coefficients and is labelled experimental | https://www.drought.gov/data-maps-tools/us-gridded-palmer-drought-severity-index-pdsi-gridmet |
| Climate Engine distribution types: non-parametric (legacy default), log-logistic, gamma | https://cpaess.ucar.edu/sites/default/files/2022-09/3-4-Climate-Engine.pdf |
| drought.gov describes `climate_indices` as a fork of code originally developed by NIDIS and NCEI | https://www.drought.gov/data-maps-tools/source-code-climate-and-drought-indices-python-spi-spei-pet |
| *GMD* manuscript types and the code and data policy (frozen archive with a persistent identifier; a GitHub site is not an archive) | https://www.geoscientific-model-development.net/about/manuscript_types.html and https://www.geoscientific-model-development.net/policies/code_and_data_policy.html |
| JOSS paper sections, including "Research impact statement" and "AI usage disclosure"; scope now weighs research impact | https://joss.readthedocs.io/en/latest/paper.html and https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future |

## Not verified

These appear in the tickets as items to check, not as findings.

- Authorship and year of the R `SEI` article (given in the tickets as Allen and Otero, 2024).
- Bibliographic details of Bourgault et al. (2023), Stagge et al. (2015), Guttman (1999), Pieper et al. (2020), Dai (2011), van der Schrier et al. (2013), Ficklin et al. (2015), Huntington et al. (2017), Abatzoglou (2013), Vose et al. (2014), Hobbins et al. (2016), and Farahmand and AghaKouchak (2015). REFIMPL-05 verifies each.
- Which tables in Palmer (1965) are sufficient for stage-by-stage fixtures. REFIMPL-28 determines this.
- Whether the Wells C++ also applies a fixed number of rescaling passes. REFIMPL-33 checks.
- Availability and licence of the original Wells C++. REFIMPL-06. (REFIMPL-30 found the NCEI `pdi.f` public at `www1.ncdc.noaa.gov/pub/data/nidis/software/ncei/palmer/` with no licence stated; see `palmer-comparators.md`.)
- Whether the Jacobi MATLAB tool runs under GNU Octave. REFIMPL-30 could not obtain it. (`JonKing93/pdsi` does run under Octave 9.2.0; see `palmer-comparators.md`.)
- Whether `precip-index` and other small packages derive from `climate_indices`. REFIMPL-04.
- Current AMS and *GMD* policy wording on AI-assisted work. REFIMPL-07.
- That no prior systematic cross-package comparison exists. The closest found are the `spei` comparison notebook and Jacobi et al. (2013). REFIMPL-05 tests the novelty claim.

## Drift check, 2026-11-18, at `cb658a3`

The tickets were drafted at `f21d3d9`. Before creation, `origin/main` was compared against them.

- No issue or pull request matched `REFIMPL` or "Reference-implementation papers". All 21 labels exist. Every repository path the tickets cite still resolves.
- Issue 1571 closed on 2026-11-17 through pull requests 1576 and 1594. Tukey plotting positions change the SPI and SPEI bias by exactly -0.053831 over the 1981-2016 Calibration Period, as the rank formulas predict. Over 1981-2024 the change is only 0.022 to 0.030, and the bias rises over 2017-2024. The full-period hypothesis failed. This is characterization only; see `VALIDATION.md` and `docs/research/earthengine-climate-engine-pilot.md`.
- Issue 1570 is still open. Its comments record that Climate Engine `timeseries/standard_index` accepts at most two coordinates and one variable and accumulation per request, at about 93 to 129 seconds each.
- Since the tickets were drafted, SPEI with the GEV distribution (`compute.gev_parameters`, pull request 1597), `suitability.compare_distributions` (pull request 1596), and ADR-0028 (pull request 1595) have merged under the open distribution-suitability epic, issue 1581. REFIMPL-09, REFIMPL-11, REFIMPL-15, REFIMPL-19, and REFIMPL-20 were edited to match.
- `gh` 2.104.0 provides `issue create --parent`, `issue edit --add-sub-issue`, and `issue edit --add-blocked-by`.
