# Scientific collaboration assessment

**GitHub Project #40 · 21 November 2026 · Maintainer working document**

## 1. Executive summary

Project #40 is a credible **scientific validation roadmap with specific opportunities for open review**. Its clearest accomplishment is making disagreements inspectable. The project traces Palmer drought calculations to publications and legacy code, maps shared ancestry among comparison tools, and reports a tested expectation that failed. It has yet to complete the planned multi-package studies, freeze an independently reproduced research package, or document outside scientific review. [Project #40](https://github.com/users/monocongo/projects/40) · [Research objectives](https://github.com/monocongo/climate_indices/issues/1602) · [Palmer provenance](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/algorithm_refs/palmer-provenance.md)

Start with one Palmer example. Ask an independent domain expert to judge how a fixed sequence of moisture anomalies produces different reported spells under two documented recursions. Supply the outputs and published passages, budget 60–90 minutes, accept an email reply, and record exactly which interpretation survives review. Neither a continental comparison nor an EarthFlux product demo is a prerequisite. [Issue #1631](https://github.com/monocongo/climate_indices/issues/1631) · [Merged research PR #1658](https://github.com/monocongo/climate_indices/pull/1658)

## 2. Project #40 inventory

**Source and scope.** The public project is titled **Reference Implementation**, has an empty README, and describes its purpose as investigating reference-implementation claims. [#1602](https://github.com/monocongo/climate_indices/issues/1602) targets two reproducible manuscripts, one on comparing SPI, SPEI and potential evapotranspiration (PET) implementations and one on Palmer provenance, validation and performance. A JOSS software paper is conditional. The umbrella explicitly makes “reference implementation” a possible research outcome and accepts results favoring other packages.

Authentication worked with the existing CLI credential. `gh project view 40 --owner monocongo` and `gh project item-list 40 --owner monocongo --limit 500 --format json` were cross-checked through GitHub GraphQL `user.projectV2`. The item and field connections returned all rows, with `hasNextPage: false`. Issue comments, closing-PR references, and relationship/timeline connections were checked for pagination. The earlier epic timeline used a second page. Current-state fields and recent PRs were refreshed at approximately 21:43 UTC. The board reports its last update at 20:15 UTC. Source pins are library main `380aa01f07d37572cbf6b8d5e1000b9d50382e03`, prior report source `98578c87` and release `v3.2.0`. No permission change was needed. GitHub documents `read:project` for API queries if read access is needed in the future. [Project](https://github.com/users/monocongo/projects/40) · [Read-only API scope](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/using-the-api-to-manage-projects) · [Current source pin](https://github.com/monocongo/climate_indices/commit/380aa01f07d37572cbf6b8d5e1000b9d50382e03)

| Measure | Observed |
| --- | --- |
| Items | 47 issues, all `monocongo/climate_indices`. No EarthFlux item, draft or PR card |
| Issue lifecycle | 44 open, three closed as Completed, #1605, #1627 and #1631 |
| Board Status | 39 Backlog, five In review (#1606, #1611, #1628, #1632, #1640), three Done. Ready and In progress empty at refresh |
| Other project fields | 18 fields. No issue milestones or iterations in use. Priority, Size, Estimate, Start date and Target date were unset. Existing views include Backlog, Priority board, Team items, Roadmap and My items |
| Labels | Examples include `validation`, `epic:validation`, `epic:scpdsi`, `type:literature`, `type:testing`, `type:infrastructure`, `ready-for-human`, and workflow labels. None of these means reviewed by an outside scientist |
| Comments and links | 34 board-issue comments, all from the maintainer account. Structured closing-PR queries returned zero links. PRs were identified via issue bodies, comments, PRs and history. Selected PR review records contained automated reviews, with no completed external domain review located |
| Primary orientation | Analyst classification by main deliverable, counting each item once. 18 scientific design, methods or comparator tasks, 17 engineering or research infrastructure tasks, 12 documentation, coordination, provenance or publication tasks. Many mix more than one purpose |
| Explicit requests | [#1608](https://github.com/monocongo/climate_indices/issues/1608) seeks provider/comparator author input. [#1604](https://github.com/monocongo/climate_indices/issues/1604) calls for adapter feedback via that issue. [#1647](https://github.com/monocongo/climate_indices/issues/1647) requests a verifier outside the initial study work and permits a separate session or a person. No agreement to participate was found |

The full inventory below groups all 47 issues. Each number links to the issue body, labels, scientific dependencies and discussion.

| Scientific initiative | Board issues | Work observed |
| --- | --- | --- |
| Research questions, methods review, literature, outreach and disclosure | [#1602][i1602], [#1603][i1603], [#1604][i1604], [#1607][i1607], [#1608][i1608], [#1609][i1609] | Paper portfolio, protocol and literature scope proposed. Maintainer comments capture questions for providers. Public evidence of answers or confirmed participants is absent |
| Comparator selection, lineage, method axes and benchmark inputs | [#1605][i1605], [#1606][i1606], [#1611][i1611], [#1612][i1612], [#1627][i1627] | Survey, provisional lineage map and gamma-estimator wording merged. Lineage issue remains In review. Method matrix in review in [#1671][p1671]. New dataset tiers planned |
| Harness, adapters, stress tests, metrics and discrepancy tracking | [#1610][i1610], [#1613][i1613], [#1614][i1614], [#1615][i1615], [#1616][i1616], [#1617][i1617], [#1618][i1618], [#1624][i1624], [#1625][i1625], [#1626][i1626] | Most engineering planned. Initial Palmer discrepancy ledger merged with [#1658][p1658]. Linux smoke-check follow-up [#1670][p1670] open |
| SPI/SPEI/PET and empirical provider studies | [#1619][i1619], [#1620][i1620], [#1621][i1621], [#1622][i1622], [#1623][i1623] | Prior scoped references exist. Main studies planned, with the provider comparison explicitly blocked |
| Palmer publications, sources, fidelity and comparator routes | [#1628][i1628], [#1629][i1629], [#1630][i1630], [#1631][i1631], [#1632][i1632], [#1640][i1640], [#1661][i1661] | Source record, same-input recursion analysis and legacy-report tests merged. Comparator recipes in review. Public `pdi.f` fidelity comparison has issue-reported results and a local draft, with full supporting report pending |
| Palmer comparative study, calibration, sensitivity and scaled example | [#1633][i1633], [#1634][i1634], [#1635][i1635], [#1636][i1636], [#1637][i1637], [#1638][i1638], [#1639][i1639] | These study and demonstration objectives remain Backlog. Existing Rust parity and prior product comparisons support selected checks |
| Archived artifacts, manuscripts and claim review | [#1641][i1641], [#1642][i1642], [#1643][i1643], [#1644][i1644], [#1645][i1645], [#1646][i1646], [#1647][i1647] | Archive, figures/ledger pipeline, manuscripts, updated claims and pre-submission verification planned |

**Dependencies and activity.** #1602 tracks 45 child issues (#1603–#1647). #1661 is also on the board. #1604 and #1611 govern what can be compared. #1610 precedes adapters and shared metrics, and #1633 depends on #1618, #1631 and #1632. #1622 depends on #1617 and #1618 and off-board [#1570](https://github.com/monocongo/climate_indices/issues/1570) for provider timing and recipe questions. #1626 says the ledger remains open until #1647 completes, while #1647 needs #1626 closed. A frozen, dated ledger with dispositions would enable a review before closure. Recent comments concern the [method matrix](https://github.com/monocongo/climate_indices/issues/1611#issuecomment-7068613352), [discrepancy ledger](https://github.com/monocongo/climate_indices/issues/1626#issuecomment-7068613138) and [upstream questions](https://github.com/monocongo/climate_indices/issues/1608#issuecomment-7043047658). The intervening merges of #1658, #1668 and #1667 are substantive outcomes since the earlier report. #1659 and #1657 merged while their issues remain In review. [Board map][i1602] · [Provenance PR][p1659] · [Legacy-report tests][p1657] · [Source manifest PR][p1668]

For reproducible orientation counts, the **scientific/design** group is #1604, #1606, #1607, #1611, #1612, #1619–#1624, #1630–#1635, #1661. **Engineering/infrastructure** is #1605, #1610, #1613–#1618, #1625, #1626, #1636–#1642. **Documentation/coordination** is #1602, #1603, #1608, #1609, #1627–#1629, #1643–#1647. A different definition of “scientific” could reassign comparator curation, reporting and parity work. The classification does not measure evidence strength.

## 3. Scientific rigor assessment

Evidence stages distinguish plans, artifacts, internal experiments, external numeric comparisons, independently executed reproductions, and human scientific judgments. A match to external code can test compatibility even when implementation lineages are shared. External review can assess interpretation without attempting a fresh numerical run.

| Scientific quality category | Evidence and maturity | What remains |
| --- | --- | --- |
| Independent numerical validation | Before #40, the library compared standard Palmer with nClimDiv, log-logistic SPEI with matched SPEIbase inputs, and monthly EDDI at one NOAA PSL location. The scope and ancestry differ across these comparisons. [Library evidence][validation] | No #40 independent end-to-end reproduction or new broad independent validation located |
| Comparison with established implementations | Merged #1605 pins smoke-tested candidates and excludes circular nClimGrid SPI. #1661 reports a same-lineage public Fortran comparison. #1632 recipes are still in [review][p1660]. [Comparator survey][survey] | Alignment of stages and variants, independently executed subset, explicit applicable reference conditions, and permission to redistribute artifacts |
| Literature review | Merged research notes and Palmer trace include primary-source passages and unresolved readings. The #1607 bibliography and novelty review remain planned. [Palmer trace][palmer] · [Alley (1984)](https://doi.org/10.1175/1520-0450(1984)023%3C1100:TPDSIL%3E2.0.CO;2) · [Jacobi et al. (2013)](https://doi.org/10.1002/wrcr.20342) | Verify missing material such as the primary PMDI source. Novelty and author interpretation are review questions |
| Reference dataset selection | Earlier fixtures have provenance and rejected candidates. #778 found the WMO example unsuitable for a precision test. #780 completed a WestWide scPDSI assessment without adopting it. #1612 proposes broader input tiers. [WMO decision](https://github.com/monocongo/climate_indices/issues/778#issuecomment-5655289592) · [WestWide decision](https://github.com/monocongo/climate_indices/issues/780#issuecomment-5724558375) | Choose representable data regimes, preserve input edition and scope, and agree reuse terms before staging comparators |
| Reproducibility | Merged version pins, sequences, numeric outputs and scripts support selected repeats. [Recursion report][recursion] · [Pinned versions][survey] | #1641 research archive and independent repeat remain open. Open #1670 documents Linux installation/output smoke checks and excludes live provider calls |
| Scientific provenance | #1659 traces publications and reference code to Python/Rust. #1667 maps pairwise lineage with alternatives and unknowns. #1668 merges a source manifest. [Palmer trace][palmer] · [Lineage map][lineage] · [Sources][p1668] | Resolve pivotal source version and rights questions. Label compatibility comparisons and provider unknowns explicitly |
| Uncertainty quantification | The merged Palmer study reports descriptive case/fixture differences. #1604 proposes dependence-aware rules and #1634 plans factor effects with uncertainty. [Protocol][i1604] · [Sensitivity][i1634] · [Laimighofer and Laaha](https://doi.org/10.1016/j.jhydrol.2022.128385) | Define sampling unit and geography/season, distinguish fit and data uncertainty, and estimate uncertainty with dependence. Division-months are dependent observations |
| Methodological transparency | #1604 proposes alignment from defaults through input, calibration, estimator and zero/missing rules. #1658 reports exploratory attribution changes and its failed H5 lookback criterion. [Comparison protocol][i1604] · [Results and amendments][recursion] | Prospectively define a small new test and report hidden conventions alongside residuals. Unexplained residuals may arise from implementation, rounding, or an unrecorded configuration |
| Scientific limitations | Known rounding, unknown operational lineage, common routines, boundary differences and source gaps are recorded. [Validation][validation] · [Palmer limits][palmer] | Bring limitations to the reader before results. Application fitness and accuracy in a different region do not follow from legacy compatibility |
| External scientific review | #1608 lists targeted queries, and #1647 plans verification. The issue and selected review samples show no documented outside scientific adjudication. [Upstream engagement][i1608] · [Verification][i1647] | Add explicit qualified human review. A different coding session may supply a useful technical repeat |

A qualified scientist would see that the team understands the distinction. #1661 distinguishes following the literature, reproducing a legacy edge, and differing from it. #1604 separates known-truth simulation, software equivalence, and agreement with defined external products. #1658 reports that 4.03% disagreed after the stipulated three-month same-sign established-spell lookback, against a 0.5% threshold. The test does not establish disagreement within a continuing spell, because its measured month can end or change the spell. The report explicitly discloses this scope and the attribution changes. These choices make scrutiny possible. The board title and terms such as “best” and “independent” still invite readers to infer a maturity claim before they reach that evidence. [Reported failure][recursion] · [Source fidelity][i1661]

## 4. Progress-report cross-reference and traceability

The prior suite is preserved in local commit `7f78635df935bc8bc8915b4bffe8aeab6623f1bb`, branch `docs/research-engineering-progress-2026-11`. The original temporary checkout is absent. The full report, executive brief, claims, strategy, and LinkedIn draft were accessible via `git show`. The earlier report appropriately distinguishes upstream validation, EarthFlux's public ACF preview and proposed independent replication. The strongest new public rationale for recruiting scientists is the already-executed Palmer analysis. A separate local EarthFlux opportunity-index draft was consulted for scope and audience. The concise index here uses the refreshed library evidence and serves as the publication candidate for this review. [Opportunity index](scientific-review-opportunities.md)

| Scientific objective | Issue → implementation | Validation evidence → maturity | Defensible claim or qualification |
| --- | --- | --- | --- |
| Choose candidates and check independence | [#1605][i1605] → [merged #1656][p1656] and [#1606][i1606] → [merged #1667][p1667] | Pinned smoke tests and source-based pair categories across 177 pairs. Unknowns and alternative interpretations explicit | “We published a comparator and lineage survey.” This advances the previous report's still-open lineage record. It does not count comparisons passed |
| Use accurate estimator language | [#1627][i1627] → [merged #1649][p1649] | Terminology corrected to Thom's closed-form approximation to maximum likelihood. Documentation complete | “We corrected the estimator description.” Its wording does not show broader numerical accuracy |
| Explain Palmer recursion differences | [#1628][i1628] → [merged #1659][p1659] and [#1631][i1631] → [merged #1658][p1658] | Source map, matched-Z experiment with fixed factors, reduced sequences and tests. The merged report gives 33,373 of 528,384 fixture division-months beyond its threshold, a maximum difference of 8.135, and the H5 failure | “We documented different published Palmer rules and measured differences on a fixed fixture.” Internal results and month-level dependence do not support geographic prevalence or a ranking of drought validity |
| Test fidelity to compiled public Fortran | [#1661][i1661] → D9 in [Palmer provenance][palmer] and local research draft | Issue reports agreement within `1.2e-14` on the specified complete-year fixtures versus the double-precision public `pdi.f` build. The uncommitted `pdi-f-fidelity.md` draft at `../climate_indices-pdi-f-fidelity/docs/research/reference-papers/pdi-f-fidelity.md` pins `46481c48` and documents precision, process isolation, supplied forcing and soil storage mapping. No full report merged at this snapshot | “The team reported a scoped legacy-program comparison and documented edge cases.” This is additional internal fidelity evidence. Reproduction, rights review, and exact operational source lineage remain open |
| Clarify scPDSI passes and comparability | [#1635][i1635] → [D7][palmer] and [existing fixtures][validation] | Wells paper and original C++ apply one rescale. The tested Sibada R port and library use three and retain percentile residuals | “The library follows the tested three-pass port.” Refine the older report's generic “Wells reference” wording. No external product or convergence guarantee |
| Resolve statistical or physical convention differences | [#1604][i1604], [#1619][i1619], [#1621][i1621], [#1623][i1623] → [existing scoped tests][validation], [method matrix PR][p1671] | Prior evidence and planned controlled comparisons. Independent reference applicability depends on forcing and method choices | “Selected drought methods have documented reference checks.” Keep same-estimator SPEI and qualified SPI/product evidence separate. ACF basin-first computations and the released Pearson exact-zero fix boundary are separate issues |
| Make claims independently reviewable | [#1626][i1626], [#1641][i1641], [#1647][i1647] → [Palmer ledger][ledger], proposed archive/verification | The internal ledger has started. Scientific review and archival reproduction still pending | “We are preparing reproducible evidence and inviting review of a specific question.” Establish the review before announcing external validation |

The earlier report's cautions about the ACF recipe, Rust parity, limited method validation and operational use still apply at its release and deployment pins. Add the now-merged source and lineage artifacts, the recorded negative Palmer result, and the port-specific calibration distinction at the next revision. [Pearson exact-zero correction](https://github.com/monocongo/climate_indices/pull/1563) · [EarthFlux Lab](https://earthflux.org/lab/)

## 5. Scientific collaboration opportunities

These are initial asks, with reviewer time separate from engineering time. The [public opportunity index](scientific-review-opportunities.md) presents the same seven options for sharing. Infrastructure, performance and drafting issues can give a collaboration practical support once the scientific scope is chosen.

| Rank and domain | Question, current evidence and unresolved risk | No-code contribution and effort | Engineering offer and possible longer output |
| --- | --- | --- | --- |
| **1. Palmer spell establishment** · [#1631][i1631] and [#1628][i1628] | Matched moisture anomalies and duration factors still give different spell interpretations. The [merged case study][recursion] provides a short example. A reduced fixture isolates a mechanism and cannot estimate field prevalence | **60–90 min.** A Palmer or hydroclimate scientist annotates one case, judges what the literature supports about when a spell starts, and qualifies the claim | Supply the exact sequence, outputs, state transition, relevant pages and a one-command rerun. Produce a citable methods note and fixture. Expand to matched external implementations only if useful |
| **2. Public Fortran fidelity** · [#1661][i1661] | Is one end-of-record PMDI behavior scientifically intended, a legacy compatibility choice, or unresolved? The issue gives internal results and published [D9][palmer] describes exceptions. The full report is pending. Public source version and operational product may differ | **60–90 min.** A Palmer/operational climatologist examines one terminal or partial-year case and names the warranted interpretation. A source owner can clarify code lineage | Supply parallel traces, citations, inputs and a boundary case. A reviewed discrepancy/fixture could inform a source-fidelity report. Same-lineage comparison establishes fidelity under its conditions |
| **3. Self-calibration policy** · [#1635][i1635], [#1630][i1630] | Which one-pass, three-pass or convergence claim fits the source and declared comparison? Prior anchor tests and [D7][palmer] narrow the question. External product and calibration policy open | **45–60 min.** A Palmer methods expert interprets one case or identifies a fully specified worked example, including what cannot be inferred | Provide paired outputs and calibration residuals. Produce a reusable example and decision note. A later paired study can evaluate robustness without requiring a default change |
| **4. Dry-series SPI protocol** · [#1604][i1604], [#1619][i1619] | Does a comparison distinguish calibration, distribution, zero convention and implementation effects? Protocol and study pending. Prior [zero tests][zero] show why matching matters. Tails and autocorrelation constrain the analysis | **45–60 min.** A drought statistician critiques one dry-station recipe, threshold and uncertainty summary before new runs | Convert their recommendations into the pinned comparison and share results. A prospective protocol is useful for the larger intercomparison |
| **5. SPEI source conventions** · [#1621][i1621], [#1608][i1608] | Does PET month length, estimator or transform explain one residual? [SPEIbase check][spei] matches selected inputs and shares a fitting routine. Full comparison pending | **45–60 min.** A drought-index/PET researcher checks the recipe and one residual. A producer can confirm an exact convention in writing | Provide matched inputs and compact paired outputs. Record a citable correction or ledger disposition. Producer input and independent scientific interpretation have distinct roles |
| **6. Monthly reference evapotranspiration** · [#1623][i1623] | Does FAO-56 Example 17 fully specify inputs and expected output for a same-method check? Example 18 and intermediates have [tests][pet], with the monthly example open | **30–60 min.** An agrometeorologist checks reference surface, units, radiation and month conventions without programming | Transcribe permitted inputs and build a compact teaching/reference case. Worked-example agreement alone does not validate field ET |
| **7. Source lineage and comparator configurations** · [#1606][i1606], [#1611][i1611] | Is one method genuinely separate, validated against the library, or built from the same routine? [Merged source map][lineage] and [open method matrix][p1671] provide evidence, with provider internals unresolved | **30–45 min.** An RSE or upstream maintainer checks a single cited pair and method row | Provide pinned configurations and a small replay. Extend to a shared fixture or geoscience cookbook with independent methods review |

### Researcher perspectives

| Audience | Why they might care, and how EarthFlux could help | Skepticism and evidence needed | First contribution and longer-term return |
| --- | --- | --- | --- |
| NOAA/NIDIS drought scientist | Documenting version, inputs and outputs can make recurrent operational-method questions easier to answer. EarthFlux can contextualize one ACF comparison. [#1608][i1608] · [#1619][i1619] · [#1661][i1661] | Show product vintage, input/calibration support and rounding. Separate test recipes and thresholds from official USDM classifications | Correct a definition or source pointer by email. A verified cross-reference or co-defined analyst case could merit further work, subject to agency procedures |
| University hydrologist | Soil storage, PET, initial state and recursion choices are useful methods and teaching questions. EarthFlux could publish one inspectable place-based example. [#1630][i1630] · [#1633][i1633] · [#1634][i1634] · [#1635][i1635] | Same-ancestry agreement and aggregate errors can hide important exceptions. Show matched forcing/intermediates, physical assumptions and how index values differ from streamflow or soil-moisture observations | Interpret a fixture or identify a relevant station. A student exercise, co-authored methods study or funded comparison can justify ongoing effort |
| Fire-weather researcher | Meteorologically derived drought inputs can support research and a future EarthFlux recipe comparison. Fire validation is outside #40. Lineage [#1606][i1606] and edge-case design [#1624][i1624] are transferable patterns | Separate evaporative demand, fuel condition, ignition and spread. Establish inputs, seasonal initialization, rain and latitude conventions with the relevant observations | Point to one suitable case for a separate study. Consider deeper work when there is a named fire-weather question and credible access to data |
| Climate-data/RSE researcher | Open adapters, immutable inputs and shared reference cases avoid repeated implementation work. EarthFlux could host a downloadable manifest and demonstration. [#1611][i1611] · [#1614][i1614] · [#1641][i1641] · [#1647][i1647] | Verify source ancestry, data rights, pinned environment and expected outputs. Reproduce outside the maintainer's environment | Review an environment or run one case. A teaching cookbook, auditable corpus, software citation or stewardship effort can be professionally useful |
| ML climate-feature researcher | Versioned indicators may make interpretable candidate covariates if stable, available on time and portable. EarthFlux could export a recipe with those facts. [#1612][i1612] · [#1621][i1621] · [#1634][i1634] · [#1640][i1640] | Future calibration and retrospective Palmer backtracking can leak information. Require availability timestamps, blocked time/place tests and baseline skill. #40 contains no ML evaluation | Critique one as-of feature definition. A target-specific forecasting or interpretation study could follow where a real user and data exist |

### Prospective collaborators

The research matches and contact routes come from public institutional or author sources. Proposed invitations, benefits and availability are unconfirmed. Source owners' clarifications carry their own evidentiary role. Seek scientific adjudication from someone outside the implementation and any selected comparator where practical.

| Candidate and public contact | Verified specialization | Bounded ask and reciprocal value |
| --- | --- | --- |
| **NOAA NCEI Climate Monitoring Branch** `ncei.monitoring.info@noaa.gov` and **NIDIS Drought Portal** `drought.portal@noaa.gov` | NClimDiv production and drought communication. [NCEI catalog](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00005%3Bview%3Diso) · [NIDIS portal](https://www.drought.gov/about/us-drought-portal) | #1608/#1661. Ask NCEI about one public Fortran/product vintage or rounding relationship. Ask NIDIS separately whether an ACF explanation sets the right boundary. Offer a corrected reuse note |
| **Michael Hayes, University of Nebraska–Lincoln/NDMC** `mhayes2@unl.edu` | scPDSI co-author. [University profile](https://snr.unl.edu/aboutus/who/people/faculty-member.aspx?pid=47) · [Wells et al.](https://journals.ametsoc.org/view/journals/clim/17/12/1520-0442_2004_017_2335_aspdsi_2.0.co_2.xml) | #1608/#1630/#1635. Ask for a source pointer on spell rules or one calibration example. Offer a citable reconstruction, with separate independent review |
| **Darren Ficklin, Indiana University** `dficklin@iu.edu` and **Aiguo Dai, University at Albany** `adai@albany.edu` | Ficklin's Palmer extension supplies Penman–Monteith PET and examples. Dai studies Palmer variants and soil moisture. [Ficklin profile](https://geography.indiana.edu/about/faculty/ficklin-darren.html) · [Author's code](https://github.com/dficklin/CAGEO-paper) · [Dai profile](https://www.atmos.albany.edu/facstaff/adai/myself-dai.html) · [Dai study](https://doi.org/10.1029/2010JD015541) | #1631/#1632/#1634/#1635. Invite one independent interpretation of the reduced example, or one matched-PET/soil-capacity contrast. Offer an annotated figure or shared fixture. Reassess independence if the expert's own output is selected |
| **Johannes Laimighofer or Gregor Laaha, BOKU** `johannes.laimighofer@boku.ac.at`, `gregor.laaha@boku.ac.at` | Statistical hydrology, SPI/SPEI uncertainty, and low-flow forecast evaluation. [Paper](https://doi.org/10.1016/j.jhydrol.2022.128385) · [Laimighofer profile](https://forschung.boku.ac.at/en/researcher/Johannes-Laimighofer-02E0A36B8D0A5994) · [Laaha project contact](https://forschung.boku.ac.at/en/projects/15681) | #1604/#1620/#1621. Review one controlled comparison and its tail/season uncertainty. Offer a reusable prospective method card and engineered experiment |
| **Sergio Vicente-Serrano or Santiago Beguería, CSIC** `svicen@ipe.csic.es` or the public SPEI project route | SPEI developers and methods authors. [CSIC contact](https://momentum.csic.es/wp-content/uploads/2024/07/MMT24-PTI-CLIMA-01-01-FICHA-MOMENTUM-ESP.pdf) · [SPEI revisited](https://doi.org/10.1002/joc.3887) · [SPEI source](https://github.com/sbegueria/SPEI) | #1608/#1621. Ask for one version-specific PET month-length or estimator detail. Offer a corrected recipe that can be reused across packages |
| **Ouranos xclim team** `bourgault.pascal@ouranos.ca` | xarray-native standardization and PET. The source documents a compatibility configuration. [Credits](https://xclim.readthedocs.io/en/stable/authors.html) · [Implementation](https://xclim.readthedocs.io/en/stable/_modules/xclim/indices/_agro.html) | #1606/#1611/#1614. Verify a two-row default vs aligned SPI configuration. Offer a documented interoperability fixture |
| **Dan McEvoy, DRI** `Daniel.McEvoy@dri.edu` and **John Abatzoglou, UC Merced** `jabatzoglou@ucmerced.edu` | Evaporative demand and gridMET. [DRI profile](https://www.dri.edu/directory/dan-mcevoy/) · [Producer methodology](https://www.climatologylab.org/gridmet.html) · [UC Merced contact](https://snri.ucmerced.edu/people/inftrees) | #1608/#1622 and #1570 outside the board. Ask one provider for one point's window, reference crop or immutable vintage. Offer a shareable case, subject to terms |
| **Canadian Forest Service/CFFDRS** and **Project Pythia/Pangeo** | Adjacent methods and open geoscience workflows. [CWFIS public route](https://directory.ciffc.ca/Home/Organization/27) · [Pythia guide and discussion route](https://projectpythia.org/cookbook-guide/) | Fire has no direct #40 item, so reserve that contact for a defined fire-weather study. For #1641/#1647, a Pythia reader could test a public one-case package and improve its teaching utility |

## 6. Top five recommendations

1. **#1631, one Palmer spell comparison.** Existing inputs and outputs make interpretation possible immediately.
2. **#1661, one Fortran edge.** Ask what distinguishes fidelity from a legacy behavior worth changing.
3. **#1635, calibration passes.** Clarify the scientific meaning and compatibility policy before proposing a new default.
4. **#1604/#1619, one dry-station protocol.** Invite scientific criticism before computing results.
5. **#1621/#1608, one SPEI recipe detail.** Ask a producer to resolve a specific convention and a separate reviewer to evaluate the resulting claim.

## 7. Credibility strengths and weaknesses

Strong evidence includes a source-by-source audit, reproducible bounded examples, an explicit lineage map and a reported failed expectation. Material gaps are external judgment, independently executed reproductions, input rights and source version clarity, and uncertainty or application evidence beyond bounded comparisons. [#1661][i1661] addresses fidelity to public `pdi.f`. It explicitly exposes cases where reproducing that source can preserve behavior unsupported by opened scientific references. The full progress report's scoped external Palmer product comparison remains a separate result. The project has a basis for a review invitation today and a substantial research workload before making broader claims.

## 8. Board improvements and public framing

The current board works for an engineer coordinating study tasks. A scientist can follow a well-framed issue, but prefixes, dependencies, terms such as CAFEC, “oracle” and “method axes,” and a missing entry guide impose avoidable work. Supporting literature, exact evidence and unresolved questions are strongest in selected Palmer records. The typical issue gives engineering deliverables, with limited guidance for a no-code reader.

**Immediate.** Send the [public prospectus](scientific-collaboration-prospectus.md) and [opportunity index](scientific-review-opportunities.md) instead of asking a scientist to browse 47 tickets. For the selected case, provide one scientific question, the current evidence stage, page citations, an accessible output table and a time-bounded reply option. Explain terms only where they affect the conclusion.

**After agreement.** Add a short board introduction and, if scientists would use it, a saved view or one label for the existing strongest issues. Keep the engineering statuses and issue dependencies. Use a dated ledger snapshot as the verification input and reserve claim approval for the relevant scientific methods reviewer. No board or issue edits were made.

Present #40 as a **scientific validation roadmap with an open scientific review invitation**. The roadmap is demonstrated. Treat a broader program label as a future outcome once named or consented review records exist.

## 9. Lightweight scientific review workflow

1. **Prepare the case.** One page with the question, one proposed claim and source passages. Supply a result table and optional replay with exact inputs, versions, assumptions, selection criteria and a reasonable tolerance. Give a readable artifact to people who prefer email.
2. **Invite judgment.** Ask whether the cited source and outputs support the claim, what remains ambiguous, and which test would change the answer. Record whether a source owner or independent reviewer provided each statement.
3. **Keep two short records.** Technical reproduction reports who ran what and what matched. Scientific review reports the exact conclusion reviewed, assumptions, judgment (supported, qualified, contradicted, or unresolved), remaining disagreements and follow-up. Credit the contributor by agreement. Preserve disagreements and negative outcomes.
4. **Return wording for approval before publication.** Confirm attribution, quotation and scope. A particular method review carries no product-wide endorsement. Offer a shareable, citable result and discuss funded scientific or student time for sustained work. Co-authorship should reflect substantial intellectual contribution and manuscript approval, with expectations agreed up front.

The maintainer provides code, tests, reruns and data preparation. Qualified experts should decide whether a setting represents the intended method, what is a valid comparator or uncertainty model, and which application claims the evidence can bear. Describe AI use as assistance with implementation, testing and evidence assembly under maintainer responsibility. The exact inputs, scope, limitations and review record matter more than the tools used to produce code. [Disclosure decision][i1609] · [Verification plan][i1647]

## 10. 30-day collaboration pilot

**Question.** With the same monthly moisture-anomaly (Z-Index) sequence and fixed duration factors, do the published standard Palmer and Wells spell-establishment rules explain the difference in the reported spell? Look at the existing five-active-month reduced `E` case and the specified padding. Standard and Wells source traces identify threshold magnitudes 1.0 and 0.5 respectively, with equality and backtracking semantics worth reviewing. Use [#1631][i1631], the [merged sequences][sequences] and [#1628][i1628]. This is a scoped methods check, with no claim of real-world prevalence or forecasting skill.

| Time | Activity and deliverable |
| --- | --- |
| Days 1–5 | Use 1–2 engineering days to pin one sequence, parameters and initialization, show both outputs and the differing decision, and add an optional one-command replay with known output. Supply Palmer/Alley/Wells passages and any uncertainty about threshold equality. Attach the exact claim proposed for review |
| Days 6–12 | Invite one independent Palmer or hydroclimate scientist for **60–90 minutes**. Ask whether the cited rules explain the example, what is ambiguous, and which qualification or new check matters. Accept an email reply. Technical rerun optional |
| Days 13–23 | Record feedback and any rerun separately. Send the revised claim, disagreement and credit to the reviewer for correction or consent |
| Days 24–30 | Produce a dated scope-and-findings note, even for an unresolved conclusion. With consent, attach it and the shareable case to the existing public research record. Link any subsequent EarthFlux display to the note. The previous ACF replay remains a separate application demonstration |

**Success.** A qualified external expert gives a specific judgment or criticism and the maintainer shows how it changed the public claim or proposed experiment. A second reader can locate the source and repeat the scoped test. Independence of execution is recorded only if someone outside the initial work actually repeats it.

**Fallback.** After about one working week, use one alternate qualified contact with the same packet. If neither responds, record review pending and have a colleague check that the packet can be followed without a local setup. Preserve the technical feedback and keep the scientific judgment open.

## 11. Conclusions and priorities

**Package one Palmer case, invite one qualified independent reviewer, and document the result next to the claim.** The board gives scientists meaningful decisions on interpretation, acceptable comparisons and disclosure. Engineering support, shareable output, editorial control over their contribution, and a clear route to funded follow-on work offer more than recognition alone.

For funding, the progress report summarizes verified releases, earlier evidence and EarthFlux boundaries. The claim matrix limits public language. Project #40 exposes questions, methods and evidence trails. A consented case review and its reproducible package would show how those pieces work together and make a specific funding request for scientific time, comparator curation and long-term stewardship more credible.

Project #40 demonstrates an emerging culture of rigorous scientific software development. [#1628][i1628] produced traceable source documentation, [#1631][i1631] and [#1658][p1658] preserve an unfavorable outcome and methodological amendments, and [#1661][i1661] distinguishes compatibility from intended behavior. Most tickets are still the work needed to earn broader confidence. The next meaningful test is whether an independent scientist can use the evidence to challenge one conclusion and see that challenge reflected in the record.

[validation]: https://github.com/monocongo/climate_indices/blob/380aa01f/VALIDATION.md
[palmer]: https://github.com/monocongo/climate_indices/blob/380aa01f/docs/algorithm_refs/palmer-provenance.md
[lineage]: https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/lineage.md
[ledger]: https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/discrepancy-ledger.md
[survey]: https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/comparators.md
[recursion]: https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/palmer-recursion-divergence.md
[sequences]: https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/palmer-recursion-divergence-sequences.json
[zero]: https://github.com/monocongo/climate_indices/blob/380aa01f/tests/test_zero_handling_reference.py
[spei]: https://github.com/monocongo/climate_indices/blob/380aa01f/tests/test_speibase_like_for_like.py
[pet]: https://github.com/monocongo/climate_indices/blob/380aa01f/tests/test_pm_eto.py
[i1602]: https://github.com/monocongo/climate_indices/issues/1602
[i1603]: https://github.com/monocongo/climate_indices/issues/1603
[i1604]: https://github.com/monocongo/climate_indices/issues/1604
[i1605]: https://github.com/monocongo/climate_indices/issues/1605
[i1606]: https://github.com/monocongo/climate_indices/issues/1606
[i1607]: https://github.com/monocongo/climate_indices/issues/1607
[i1608]: https://github.com/monocongo/climate_indices/issues/1608
[i1609]: https://github.com/monocongo/climate_indices/issues/1609
[i1610]: https://github.com/monocongo/climate_indices/issues/1610
[i1611]: https://github.com/monocongo/climate_indices/issues/1611
[i1612]: https://github.com/monocongo/climate_indices/issues/1612
[i1613]: https://github.com/monocongo/climate_indices/issues/1613
[i1614]: https://github.com/monocongo/climate_indices/issues/1614
[i1615]: https://github.com/monocongo/climate_indices/issues/1615
[i1616]: https://github.com/monocongo/climate_indices/issues/1616
[i1617]: https://github.com/monocongo/climate_indices/issues/1617
[i1618]: https://github.com/monocongo/climate_indices/issues/1618
[i1619]: https://github.com/monocongo/climate_indices/issues/1619
[i1620]: https://github.com/monocongo/climate_indices/issues/1620
[i1621]: https://github.com/monocongo/climate_indices/issues/1621
[i1622]: https://github.com/monocongo/climate_indices/issues/1622
[i1623]: https://github.com/monocongo/climate_indices/issues/1623
[i1624]: https://github.com/monocongo/climate_indices/issues/1624
[i1625]: https://github.com/monocongo/climate_indices/issues/1625
[i1626]: https://github.com/monocongo/climate_indices/issues/1626
[i1627]: https://github.com/monocongo/climate_indices/issues/1627
[i1628]: https://github.com/monocongo/climate_indices/issues/1628
[i1629]: https://github.com/monocongo/climate_indices/issues/1629
[i1630]: https://github.com/monocongo/climate_indices/issues/1630
[i1631]: https://github.com/monocongo/climate_indices/issues/1631
[i1632]: https://github.com/monocongo/climate_indices/issues/1632
[i1633]: https://github.com/monocongo/climate_indices/issues/1633
[i1634]: https://github.com/monocongo/climate_indices/issues/1634
[i1635]: https://github.com/monocongo/climate_indices/issues/1635
[i1636]: https://github.com/monocongo/climate_indices/issues/1636
[i1637]: https://github.com/monocongo/climate_indices/issues/1637
[i1638]: https://github.com/monocongo/climate_indices/issues/1638
[i1639]: https://github.com/monocongo/climate_indices/issues/1639
[i1640]: https://github.com/monocongo/climate_indices/issues/1640
[i1641]: https://github.com/monocongo/climate_indices/issues/1641
[i1642]: https://github.com/monocongo/climate_indices/issues/1642
[i1643]: https://github.com/monocongo/climate_indices/issues/1643
[i1644]: https://github.com/monocongo/climate_indices/issues/1644
[i1645]: https://github.com/monocongo/climate_indices/issues/1645
[i1646]: https://github.com/monocongo/climate_indices/issues/1646
[i1647]: https://github.com/monocongo/climate_indices/issues/1647
[i1661]: https://github.com/monocongo/climate_indices/issues/1661
[p1649]: https://github.com/monocongo/climate_indices/pull/1649
[p1656]: https://github.com/monocongo/climate_indices/pull/1656
[p1657]: https://github.com/monocongo/climate_indices/pull/1657
[p1658]: https://github.com/monocongo/climate_indices/pull/1658
[p1659]: https://github.com/monocongo/climate_indices/pull/1659
[p1660]: https://github.com/monocongo/climate_indices/pull/1660
[p1667]: https://github.com/monocongo/climate_indices/pull/1667
[p1668]: https://github.com/monocongo/climate_indices/pull/1668
[p1670]: https://github.com/monocongo/climate_indices/pull/1670
[p1671]: https://github.com/monocongo/climate_indices/pull/1671
