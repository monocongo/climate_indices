# Scientific review opportunities

*Public draft for maintainer review · 21 November 2026*

A short written assessment can help establish which scientific claims the drought methods in `climate_indices` support. Choose one question below. The maintainer can supply the inputs, outputs, and source passages for review by email. If you want to reproduce a result, we can also provide a pinned recipe. We record the scope of each review, preserve disagreements, and ask for consent before publishing comments or naming contributors. [GitHub Project #40](https://github.com/users/monocongo/projects/40) · [Library validation record](https://github.com/monocongo/climate_indices/blob/380aa01f/VALIDATION.md)

## 1. When does a Palmer drought spell become established?

**Drought climatology or historical Palmer methods · 60–90 minutes.** The merged study includes a reduced example with five active months of Palmer Z-Index input and identical fixed duration factors. The two recursions use different thresholds for declaring an established spell. Review the source passages and one month-by-month trace. Identify what each published method supports, what remains ambiguous at the boundary, and what the project can say about this case. The study documents an internal comparison, and outside scientific review is pending. [Issue #1631](https://github.com/monocongo/climate_indices/issues/1631) · [Merged study](https://github.com/monocongo/climate_indices/pull/1658) · [Reduced sequences](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/palmer-recursion-divergence-sequences.json) · [Source trace, row 14](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/algorithm_refs/palmer-provenance.md#3-duration-factors-and-the-three-candidate-indices)

## 2. What should the standard Palmer recursion do when a record ends?

**Operational drought indices or hydrology · 60–90 minutes.** An open issue reports agreement with public `pdi.f` on full-year records under the tested setup and flags a legacy end-of-record behavior for review. Review one case where the final month's Palmer Modified Drought Index is assigned to several earlier months still awaiting resolution. Say whether the sources support that choice, describe it as a legacy convention, or leave it unresolved. [Issue #1661](https://github.com/monocongo/climate_indices/issues/1661) · [Reported behavior and proposed change](https://github.com/monocongo/climate_indices/issues/1666) · [Provenance, finding D9](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/algorithm_refs/palmer-provenance.md#differences-found-while-tracing)

## 3. Which calibration rule should self-calibrating PDSI use?

**Drought climatology or statistical hydrology · 45–60 minutes.** The library follows a three-pass rule from the Sibada R port. The Wells paper and original C++ program describe or apply one rescaling pass. The convergence policy and a proposed external-product comparison remain open. Review the conventions and advise which comparison would justify retaining three passes, offering another mode, or narrowing the claim. [Issue #1635](https://github.com/monocongo/climate_indices/issues/1635) · [Provenance, finding D7](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/algorithm_refs/palmer-provenance.md#differences-found-while-tracing) · [Wells et al.](https://journals.ametsoc.org/view/journals/clim/17/12/1520-0442_2004_017_2335_aspdsi_2.0.co_2.xml)

## 4. How should we compare SPI in a dry precipitation record?

**Hydroclimate statistics · 45–60 minutes.** The cross-package Standardized Precipitation Index (SPI) study is planned. Existing tests compare selected treatments of zero precipitation against R implementations. Review a proposed dry-station case before the run. Specify which calendar, fitting, and zero-handling choices must match and how we should judge differences in the dry tail. [Protocol issue #1604](https://github.com/monocongo/climate_indices/issues/1604) · [SPI study issue #1619](https://github.com/monocongo/climate_indices/issues/1619) · [Current zero-handling comparisons](https://github.com/monocongo/climate_indices/blob/380aa01f/tests/test_zero_handling_reference.py)

## 5. Where does a difference between two SPEI estimates arise?

**Drought statistics or evapotranspiration · 45–60 minutes.** An existing input-matched comparison with SPEIbase covers three climate divisions and shares an estimator lineage. The broader Standardized Precipitation Evapotranspiration Index (SPEI) study is planned. Review one proposed paired test and identify which settings would isolate the effects of distribution fitting from the convention used to convert daily evapotranspiration rates to monthly totals. [Study issue #1621](https://github.com/monocongo/climate_indices/issues/1621) · [Source clarification #1608](https://github.com/monocongo/climate_indices/issues/1608) · [Existing SPEIbase comparison](https://github.com/monocongo/climate_indices/blob/380aa01f/tests/test_speibase_like_for_like.py)

## 6. What would make a monthly evapotranspiration example a fair comparison?

**Agricultural meteorology or FAO-56 practice · 30–60 minutes.** Current Penman–Monteith tests include a published daily example. A monthly worked-example check is proposed. Review the interpretation of FAO-56 Example 17 for one site and month. Identify the inputs and assumptions each package would need to share and suggest a tolerance consistent with the example's rounding. [Issue #1623](https://github.com/monocongo/climate_indices/issues/1623) · [FAO-56, Chapter 4](https://www.fao.org/4/x0490e/x0490e08.htm) · [Current tests](https://github.com/monocongo/climate_indices/blob/380aa01f/tests/test_pm_eto.py)

## 7. Do two comparable implementations provide independent evidence?

**Research software or numerical provenance · 30–45 minutes.** The lineage map is merged and the method matrix is under review. Choose one pair of implementations, check the cited sources for shared code or declared compatibility, and propose an evidence label or a way to resolve an unknown. [Lineage issue #1606](https://github.com/monocongo/climate_indices/issues/1606) · [Merged lineage map](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/lineage.md) · [Method matrix issue #1611](https://github.com/monocongo/climate_indices/issues/1611) · [Method matrix review](https://github.com/monocongo/climate_indices/pull/1671)

## How to participate

Send the issue number, the question you want to review, and your preferred level of involvement. The maintainer can prepare the case and discuss credit before work begins. [Email the maintainers](mailto:hello@earthflux.org) · [GitHub Discussions](https://github.com/monocongo/climate_indices/discussions)
