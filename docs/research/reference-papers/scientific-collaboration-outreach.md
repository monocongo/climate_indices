# Scientific collaboration outreach package

**Drafts for maintainer review**

Replace the bracketed details and verify the contact route before sharing. Send one invitation with one scientific question and cite the work that makes the request relevant to that reader.

## Short introduction

I maintain `climate_indices`, an open-source library that implements published drought and other environmental indicators. I’m looking for scientific collaborators who can help settle specific questions about how those methods are interpreted and evaluated. I’ll prepare the evidence, run the code and document the results. An initial contribution can be a short written judgment about one case, including why the evidence falls short. [Project #40](https://github.com/users/monocongo/projects/40) · [Validation record](https://github.com/monocongo/climate_indices/blob/380aa01f/VALIDATION.md)

## Email 1 — Palmer drought methods

**Subject — A short review of one Palmer spell-establishment case**

Dear [Name],

Your work on [specific Palmer or hydroclimate study] is directly relevant to a question I’m trying to resolve in `climate_indices`. Our standard and Wells-lineage calculations can identify different spell histories even when they start with the same monthly moisture anomalies and use the same duration factors. The source record traces them to different magnitudes for establishing a spell, with a remaining question about how to describe equality at the boundary. [Palmer provenance](https://github.com/monocongo/climate_indices/pull/1659) · [Reduced-case study](https://github.com/monocongo/climate_indices/pull/1658)

Would you be willing to spend 60–90 minutes on one five-active-month example? I can send a short evidence sheet with the input series, two output traces and the passages we relied on. I’d value your judgment on whether we have explained the discrepancy faithfully, what should be left unresolved and which check could clarify it. A reply by email is enough.

I would do any code or data work and return the resulting note for your review. If you’re willing to have any feedback shared, we can agree on how it is described and credited.

Best,

James Adams · [Verified reply address]

## Email 2 — Hydroclimate statistics and SPI

**Subject — Which comparison would isolate the source of a dry-station SPI disagreement?**

Dear [Name],

I’m designing a small comparison for the Standardized Precipitation Index. Your study of [specific work on distribution, calibration or short records] is relevant to how we phrase the question. Two tools can disagree because their fitting periods, estimators or treatment of zero precipitation differ. I’d like to pin one dry-station example and agree on how to distinguish those choices before running the comparison. [Comparison protocol](https://github.com/monocongo/climate_indices/issues/1604) · [SPI study](https://github.com/monocongo/climate_indices/issues/1619)

Would you have 45–60 minutes to review a one-page recipe and suggest one change? I’d ask whether the proposed controls separate the effects, and which tail error or change in category is worth measuring. I can supply the data, document the settings and run the comparisons. If the example cannot support the intended claim, knowing that before we run the study would help.

I’d share the revised recipe and credit any contribution in the form we agree on. A written reply would work well.

Best,

James Adams · [Verified reply address]

## Email 3 — Agrometeorology and reference evapotranspiration

**Subject — Review of one monthly FAO-56 worked example**

Dear [Name],

Your work on [specific study or software for reference evapotranspiration] prompted a question about our FAO-56 Penman–Monteith checks in `climate_indices`. Our existing test uses the published daily example. We are considering FAO-56 Example 17, the April Bangkok monthly case, as a follow-up and want to make sure the averaging, soil heat flux and printed rounding are handled consistently. [FAO-56 Chapter 4](https://www.fao.org/4/x0490e/x0490e08.htm) · [PET comparison issue](https://github.com/monocongo/climate_indices/issues/1623) · [Existing checks](https://github.com/monocongo/climate_indices/blob/380aa01f/VALIDATION.md)

Would you spend 30–60 minutes checking an annotated worksheet against the source? I’d value a judgment on which inputs are fully specified and which expected values can support a useful conformance check. I’ll assemble the worksheet and carry out any follow-up computation. The result could be a small reference example that others can reuse, subject to the data and source terms.

A short reply is enough, and I would confirm with you before sharing or crediting any feedback.

Best,

James Adams · [Verified reply address]

## LinkedIn draft

When two drought calculations disagree on the same inputs, which differences can we explain from the published methods?

I maintain `climate_indices`, an open-source library that is a fork of drought-index work originally developed by NIDIS/NCEI/NOAA. The library’s public validation project now includes a study comparing two Palmer spell-tracking rules and an equation-to-code provenance record. The study reports a failed hypothesis and records how we revised part of the analysis. Those details matter to anyone interpreting the resulting series. [Project and evidence](https://github.com/users/monocongo/projects/40) · [Palmer comparison](https://github.com/monocongo/climate_indices/pull/1658)

I’d like to invite one Palmer or hydroclimate researcher to look at a five-active-month case and tell me which explanation the sources and outputs support, and what remains uncertain. I’ll prepare the evidence and handle follow-up computation. A written response or a short call works. If this is close to your work, please get in touch. The library is a scientific product and is not official communication of NOAA or the U.S. Department of Commerce.

## GitHub scientific review issue template

**Suggested issue title**

Scientific review of [one method choice] under [specific case]

### Question and claim

[State the question in plain language. Give the exact proposed claim and where it would appear.]

### Evidence and limits

[Link one primary publication, the version or commit, the input data and the expected outputs. State the units, temporal and spatial scope, source lineage, and one material limitation. Record what the technical comparison established and who performed it.]

### Feedback requested

- Which interpretation does the evidence support?
- Which assumptions or wording need qualification?
- What result or missing reference would change your conclusion?

[Estimated reading time and optional rerun. Give a route for written feedback outside GitHub.]

### Review record

[Summarize the scope and outcome as supported, qualified, contradicted or unresolved. Record any independent numerical rerun separately. Preserve disagreements and ask for consent before sharing feedback or naming a contributor.]

## Suggested EarthFlux collaboration landing page

### Help interpret a drought calculation

A method choice can change when a drought appears in a historical series. EarthFlux’s public Scenario Lab lets you examine selected stored series and their input and method choices. The open-source library behind those calculations, `climate_indices`, publishes its numerical checks and unresolved questions. [Explore Scenario Lab](https://earthflux.org/lab/) · [Review the library evidence](https://github.com/monocongo/climate_indices/blob/380aa01f/VALIDATION.md)

**First review opportunity**

Help interpret one case where two Palmer spell-tracking rules return different results from the same inputs. We’ll provide a short evidence sheet, source passages and both outputs. Allow 60–90 minutes. You can respond in plain text, and we’ll take care of follow-up calculations. [Read the review question](https://github.com/monocongo/climate_indices/issues/1631)

Use [verified collaboration contact] to ask for the evidence sheet. We’ll agree on the scope and how to record your contribution before sharing it.
