# Palmer recursion divergence: `pdi.f` versus Wells

**Ticket:** REFIMPL-29, [#1631](https://github.com/monocongo/climate_indices/issues/1631). **Map:** [#1602](https://github.com/monocongo/climate_indices/issues/1602).
**Pinned version:** `46481c48c79e878eb50ce3398ba062fd3a1ff38c` (`origin/main`, 2026-11-18).
**Status:** complete. Everything down to "Deviations" was committed before any analysis; the deviations were committed before the runs they affect; "Results" was written afterwards. Reproduce with `uv run python scripts/characterize_palmer_recursion_divergence.py` (about 15 minutes) and `uv run python scripts/find_palmer_minimal_sequences.py` (about 2 minutes); the committed JSON files are their outputs.

## Question

Given an identical Z-Index series and identical duration factors, exactly when and why do `climate_indices.drought._palmer_pdi.calculate` and `climate_indices.drought._palmer_wells.calculate` produce different PDSI values, and which behavior does Palmer (1965) support?

## Claims being tested

None of these is a finding yet. Each is a lead to confirm or refute at the pinned version.

| Lead | Source |
| --- | --- |
| The two recursions disagree on about 6.3% of months by up to 8 PDSI units | `tests/fixture/palmer/provenance.json`, measured with the Wells C++ in standard mode, not with `_palmer_wells.py` |
| On the disputed months the `pdi.f` values are closer to NOAA operational data 88% of the time | same file, against `tests/fixture/nclimdiv` |
| The two weighting-fraction forms agree in exact arithmetic but not bitwise | docstrings of `_palmer_pdi.py` and `_palmer_duration.py` |

The provenance figures describe the Wells **C++**. This study runs the Python port `_palmer_wells.py`, which `tests/test_scpdsi.py` matches to the C++ at `atol=5e-5` on the fitted-factor series. The first result to report is whether the Python port reproduces the 6.3% and 8-unit figures.

## Sample

- All 344 climate divisions, 1,536 months each (1895-01 to 2022-12), no missing values in the committed Z series.
- Input: `tests/fixture/palmer/<division>/zindex.npy`, passed unchanged. It is the library's own Z-Index, which `VALIDATION.md` already records as matching nClimDiv. It is not NOAA data (see the fixture provenance).
- `pdi.f` side: `_palmer_pdi.calculate`, with the series reshaped to `(128, 12, 1)`.
- Wells side: `_palmer_wells.calculate` on the flat series.
- Reference for H4: `tests/fixture/nclimdiv/pdsi.npy`, shape `(344, 1536)`, rows in `divisions.json` order.

## Configurations

| ID | Duration factors | Purpose |
| --- | --- | --- |
| **P** (primary) | Palmer's national constants, `DurationFactors.from_defaults()` and a `PdiDurationFactors` with the same `wetm, wetb, drym, dryb` | The configuration standard `pdsi()` uses |
| **F** (secondary) | Each division's fitted `scdurfact.npy`, passed to both | Exposes any difference that needs `wetb != dryb`, such as the Wells `dryc` cross coefficient |
| **P-fp** | P, with `_palmer_pdi._weighting_fraction` replaced by the Wells complement form `1 - m/(m+b)` | Isolates floating-point effects |
| **P-fp2** | P, with the Wells coefficients replaced by the division form `b/(m+b)` | The same isolation in the other direction |

Hypotheses are tested on **P**. **F** is reported alongside it and does not enter any pass/fail threshold.

## Definitions

- **Disagreement.** `|PDSI_pdi - PDSI_wells| > 1e-9`. Counts at 0.01, 0.5, and 1.0 are also reported.
- **Episode.** A maximal run of consecutive disagreeing months in one division.
- **First divergence.** The first month of an episode where the two machines' state differs. Because the two recursions backtrack, the *output* month can precede the *state* month that caused it. Classification therefore looks at state, not at output position.
- **Class.** Assigned to each episode by the first-divergence state pair. The classes are fixed here, in this order of precedence:
  1. **floating-point.** The episode disappears under P-fp or P-fp2.
  2. **establishment.** The two machines differ on whether an incipient index (X1 or X2) is promoted to X3 this month, or on the threshold used.
  3. **termination.** The two machines differ on whether an established spell ends this month.
  4. **abatement.** An established spell is mid-abatement (probability strictly between 0 and 100) and the two machines differ on the probability or the abatement trigger.
  5. **tie between candidates.** X1 and X2 are both candidates and the machines choose differently, or one hits an exact-zero candidate.
  6. **backtracking.** The state agrees at the first divergence, but the deferred months are resolved to different candidate series.
  7. **unclassified.** Anything else. Any unclassified episode means Done-when is not met, and is reported as such.
- **Logic difference.** An episode that survives both P-fp and P-fp2.
- **Palmer category.** The 11-class table: extremely wet (>= 4.00), very wet (3.00 to 3.99), moderately wet (2.00 to 2.99), slightly wet (1.00 to 1.99), incipient wet (0.50 to 0.99), near normal (-0.49 to 0.49), incipient dry (-0.50 to -0.99), mild (-1.00 to -1.99), moderate (-2.00 to -2.99), severe (-3.00 to -3.99), extreme (<= -4.00). The boundaries are Palmer's as reproduced by NCEI. They have **not** been checked against the 1965 report's own table, and the report states that before using any of them. `suitability.drought_category` is not used: it classifies SPI/SPEI on the U.S. Drought Monitor scale.

## Hypotheses and thresholds

All on configuration P.

| ID | Hypothesis | Pass if |
| --- | --- | --- |
| H1 | The Python recursions reproduce the provenance figures | Disagreement rate in [5.3%, 7.3%] and maximum \|difference\| in [7.0, 9.0] |
| H2 | Every episode falls in a named class | Unclassified episodes = 0 |
| H3 | Floating-point effects are negligible | Episodes classed floating-point account for < 0.1% of disagreeing months |
| H4 | `pdi.f` is the closer lineage to nClimDiv on disputed months | `pdi.f` closer in >= 80% of disagreeing months, ties excluded and counted. The claim being checked is 88% |
| H5 | The two lineages agree wherever neither is near a spell boundary | Months where both machines hold an established spell with \|X3\| > 1.5 and the same sign for the previous 3 months disagree in < 0.5% of cases |

A failed hypothesis is reported as failed. No threshold is changed after the first run; a change is a deviation recorded below.

## Descriptive outputs (no thresholds)

- Disagreement rate by division, by decade, and by season.
- Episode-length distribution.
- Fraction of disagreeing months whose Palmer category differs.
- Largest disagreement per class, and the division and month.

## Procedure

1. This commit.
2. `scripts/characterize_palmer_recursion_divergence.py` runs both recursions without modifying either module. It writes `palmer-recursion-divergence-results.json`, deterministically, with no random numbers.
3. For each observed class, a greedy-deletion shrink of a real episode finds the shortest synthetic Z sequence that still first diverges in that class. Each is traced to a line in both modules.
4. `tests/test_palmer_recursion_divergence.py` pins both outputs on each minimal sequence.
5. The textual analysis compares each class with Palmer (1965) and Wells, Goddard, and Hayes (2004).

## Out of scope

Changing either recursion, and deciding which one `pdsi()` or `scpdsi()` should use. Disagreements are logged in the [REFIMPL-24 ledger](discrepancy-ledger.md), whose entries cite the tests; any default change would need its own issue and ADR.

## Deviations

### D1. Classification by harmonization, not by first state divergence (before any attribution run)

The pre-registered rule classified an episode by the first month in which the two machines' state differed. A probe of that rule on configuration P (signature table only; no class counts, hypotheses H1 to H5 untouched) gave 3,832 output episodes but 83,014 maximal runs of state mismatch. The abatement probability differs between the machines in most abatement months without changing PDSI, so "first state difference" is usually not output-relevant. The rule is dropped.

Replacement. Four code-level differences were read from the two modules before any run:

| Mechanism | `pdi.f` (`_palmer_pdi.py`) | Wells (`_palmer_wells.py`) | Maps to class |
| --- | --- | --- | --- |
| **E** establishment threshold | promote X1/X2 to X3 at \|X\| >= 1 | at \|X\| >= 0.5 | establishment |
| **T** termination | a spell with \|X3\| <= 0.5 is treated as ended | a spell persists until the end probability reaches 100 | termination |
| **A** abatement trigger | fixed 0.15 | half the fitted slope (0.1545 at Palmer's factors) | abatement |
| **Tol** tolerances | exact comparisons | 1e-5 in the carry term and the 100% test | abatement |

Each is neutralized by editing a copy of `_palmer_pdi.py` *in memory* so that it uses the Wells rule; neither module on disk changes. The 16 subsets of {E, T, A, Tol} are run on configuration P, and each P episode (an output run from the base comparison) is attributed as follows:

- **E / T / A / Tol**: exactly one single-mechanism variant removes every disagreeing month of the episode.
- **overdetermined**: more than one single-mechanism variant removes it. The set is listed.
- **interaction**: no single mechanism removes it, but a subset of two or more does. The smallest subset is listed.
- **residual**: the full harmonization still leaves it. The tie and backtracking classes live here: they are the exact-zero tie-breaks and the deferred-month resolution, which this study does not harmonize. A residual episode is then examined by hand, and anything that cannot be named is **unclassified**.
- **floating-point**: unchanged from the pre-registration (P-fp, P-fp2).

H2 passes if there are no unclassified episodes; H3, H4 and H5 are unchanged. "Logic difference" now means an episode with a mechanism, interaction or residual attribution that survives P-fp and P-fp2. The synthetic minimal sequences are found per mechanism by greedy deletion on a real episode that the mechanism alone removes.

### D2. Two further mechanisms and a leave-one-out attribution rule (after the four-mechanism factorial)

The 16-subset factorial over {E, T, A, Tol} ran once, before this entry. It left 1,098 disagreeing months in 111 episodes on configuration P after all four were harmonized. Reading the shortest residual episodes by hand found two more code-level differences, each tested for whether harmonizing it clears what is left; the six-mechanism set then leaves 0 months:

| Mechanism | `pdi.f` (`_palmer_pdi.py`) | Wells (`_palmer_wells.py`) | Maps to class |
| --- | --- | --- | --- |
| **C** candidate carried at establishment | `_statement_200` recomputes `px2` only where X1 was not promoted (`active & ~branch1`), so `px2` restarts from zero after an X1 promotion | `_candidate_values` recomputes X2 every month, and `_establish_spell` zeroes only the promoted candidate | tie between candidates |
| **F** end-of-record flush | `_finish_up` rewrites the slots `0..k8max-1` (the longest deferral window ever) from the tentative `x` array | no equivalent step | backtracking |

The 64-subset factorial over six mechanisms would take about 40 minutes, so the attribution rule is changed to *counterfactual necessity*:

- Run: the base, the full harmonization, each mechanism alone (6 runs), and the full harmonization minus each one mechanism (6 runs).
- An episode's **label** is the set of mechanisms whose omission from the full harmonization leaves the episode's span disagreeing (its *necessary set*), for example `E`, `E+C`, or `F`.
- If the necessary set is empty (each mechanism is individually dispensable), the label is **overdetermined**, with the mechanisms that remove it alone listed.
- If the full harmonization leaves the episode, the label is **residual**. The study is complete for configuration P only if no episode is residual.

The four-mechanism factorial is kept in the report as an exploratory result, labelled as such. The pre-registered hypotheses H1 to H5 and their thresholds are unchanged. Configuration F (fitted factors) is characterized by its headline rates only: its attribution needs a further mechanism (Wells' `dryc` cross coefficient) and is not run here.

### Other change before the first commit

The plan named `suitability.drought_category` for category changes. It was replaced by the Palmer table above because that function classifies SPI/SPEI.

## Results

Everything below is for configuration **P** (Palmer's national factors) unless stated. Counts come from `palmer-recursion-divergence-results.json`.

### Hypotheses

| ID | Pass? | Observed |
| --- | --- | --- |
| H1 reproduces the provenance figures | **pass** | 33,373 of 528,384 months disagree (6.316%), maximum \|difference\| 8.135 PDSI units. The provenance figures were measured with the Wells C++; the Python port reproduces them |
| H2 every episode in a named class | **pass** | 3,832 episodes, 0 residual after harmonizing E, T, A, C, F, Tol (D2) |
| H3 floating point negligible | **pass** | 0 episodes disappear under P-fp or P-fp2. The two weighting-fraction forms differ by 1 ulp (`0.8970000000000001` against `0.897`) and move PDSI by up to 5e-15, but flip no branch on any division |
| H4 `pdi.f` closer to nClimDiv | **pass** | closer on 29,249 of 33,373 disagreeing months (87.6%), no ties. The provenance file says 88% for the C++ |
| H5 agreement inside established spells | **fail** | 9,990 of 248,033 months (4.03%) disagree even after three months of the same established spell with \|X3\| > 1.5 in both machines, against a threshold of 0.5%. Disagreement is not confined to spell boundaries. Which classes produce it was not attributed |

The largest disagreement (8.135 units, division 0501, October 1915) belongs to class A, the smallest-looking difference of all: a threshold of 0.15 against 0.1545.

### Classes

An episode's label is the set of mechanisms that must all be harmonized for it to vanish (D2). No episode needed Tol, so the 1e-5 tolerances in the Wells recursion have no effect on this fixture. All classes below are logic differences; the floating-point class is empty.

| Class | Episodes | Months | Largest \|difference\| (division, month) | Shortest sequence (active months) |
| --- | ---: | ---: | --- | ---: |
| E | 3,415 | 30,212 | 3.516 (3009, 1899-12) | 5 |
| E+T | 85 | 1,008 | 1.704 (0304, 1943-05) | 4 |
| E+A | 50 | 410 | 3.534 (2001, 1986-01) | 2 |
| E+C | 40 | 350 | 3.952 (3204, 1959-09) | 16 |
| A | 163 | 806 | **8.135** (0501, 1915-10) | 7 |
| C | 21 | 162 | 0.102 (2902, 1968-09) | 15 |
| F | 38 | 133 | 3.417 (1502, 2022-03) | 6 |
| E+F | 4 | 75 | 2.498 (1207, 2022-08) | 4 |
| E+A+C | 4 | 56 | 2.864 (0603, 1982-07) | 31 |
| A+C | 1 | 22 | 3.133 (0907, 1989-04) | 20 |
| E+T+A | 2 | 20 | 1.917 (3310, 1943-06) | 14 |
| E+T+C | 3 | 99 | 0.673 (1901, 1971-12) | 21 |
| overdetermined (E+A, E+F, A+F) | 6 | 20 | n/a | not sought |

Months in any class containing a mechanism: E 96.6%, A 3.9%, T 3.4%, C 2.1%, F 0.6% (overdetermined episodes excluded). Alone, E removes 90.5% of all disagreeing months; the other mechanisms matter mostly in combination with it. Every sequence is pinned in `tests/test_palmer_recursion_divergence.py`, one test per class. Sequences for `E+A`, `E+A+C` and `E+T+A` hold unrounded values because rounding them loses the class: they sit on a branch boundary.

### What each mechanism is, and what Palmer (1965) says

Line numbers are for `src/climate_indices/drought/` at the pinned commit. Palmer quotations are from an OCR of the scanned report. Page numbers are the page of the PDF file; the printed numbers were not checked.

| Mechanism | `pdi.f` port | Wells port | Palmer (1965) | Determined by the text? |
| --- | --- | --- | --- | --- |
| **E** establishment threshold | promote at \|X\| >= 1 (`_palmer_pdi.py:375,386`) | at \|X\| >= 0.5 (`_palmer_wells.py:217,227`) | X3 is "reserved for indicating the severity of any wet spell or any drought which has become definitely established; i.e., X3 > 1.00 or X3 < -1.00" (PDF p. 38) | **Yes, for `pdi.f`.** The text supports 1.0. Whether Wells intends 0.5 could not be checked |
| **T** termination | a spell with \|X3\| <= 0.5 has ended (`_palmer_pdi.py:512`) | persists until the end probability reaches 100 (`:149-164`) | "as soon as the severity index reaches the 'near normal' category, which lies between -0.50 and +0.50, no drought exists" (p. 35) | **Yes, for `pdi.f`**, on that passage. The text does not say what to do when the probability and the \|X3\| <= 0.5 rule disagree |
| **A** abatement trigger | fixed 0.15 (`_palmer_pdi.py:450,471,518`) | half the fitted slope, 0.1545 at national factors (`_palmer_wells.py:187`) | Setting X = 0.50 and ΔX = 0 in the index recurrence "one finds that Z = -0.15" and "Ue = Z + 0.15" (p. 35) | **No.** The printed 0.15 is the rounding of 0.309 × 0.5 = 0.1545. Both ports are consistent with the text; it does not say which to use. The 0.0045 gap produces the largest disagreement in the fixture |
| **C** candidate carried | `px2` restarts from zero after an X1 promotion (`_palmer_pdi.py:384`) | X2 recomputed every month (`_palmer_wells.py:74-77`) | X1 and X2 "must be computed at every opportunity" while a spell runs; the worked example computes the opposite-sign index in the month a spell is established (p. 38) | **Open.** Weakly favours Wells: the text does not say what happens to X2 in the establishing month itself |
| **F** end-of-record flush | `_finish_up` rewrites `0..k8max-1` stale slots (`_palmer_pdi.py:557-569`) | no such step | no passage on undecided months at the end of the record was found in pp. 36-41; the zero-candidate rule for deciding earlier months is on p. 39 ("we assign X = X2 in February and also for the preceding January and December") | **No.** Possible defect in `climate_indices` (see below) |
| **Tol** tolerances | exact comparisons | 1e-5 | not stated | **Not needed.** No episode depends on it |

Palmer's Table 11 (p. 34) prints the class boundaries with gaps (for example "-.49 to .49 near normal" and "-.50 to -.99 incipient drought"), so inclusivity at the boundaries is not stated. The study used left-closed magnitudes (0.50 to 0.99, 1.00 to 1.99, ...), as pre-registered.

**What could not be checked.** Wells, Goddard, and Hayes (2004) was not read: the AMS host refused the fetch (HTTP 403). Nothing in this report states what Wells says; every Wells statement above is about the Python port or the provenance record. The NCEI `pdi.f` source and the Wells C++ were not consulted either, so "follows `pdi.f`" means "follows the module docstring". A copy of the Wells paper would let the E, A and C rows be settled.

### The end-of-record flush (class F)

Class F covers 38 episodes alone, plus 9 more combined with E or A (47 in all): on these the `pdi.f` port overwrites months that backtracking had already resolved. `_finish_up` iterates over `k8max`, the longest deferral window anywhere in the record, but only the current window `k8` can still be open. The slots beyond `k8` hold months from an earlier window, and the loop rewrites them with their tentative value. The 48 episodes that remain when only F is left unharmonized all start between months 1515 and 1535 of the 1,536-month record (the last 21 months). Whether the Fortran does the same is not known, and whether it affects the nClimDiv comparison was not tested. This is logged in the ledger as a possible defect for the maintainer to decide; this study changes nothing.

### Persistence and category

- Episodes last 1 month (270), 2 (477), 3 (499), 4 to 6 (1,040), 7 to 12 (832), 13 to 24 (458), and over 24 (256).
- 10,010 disagreeing months (30.0%) fall in a different Palmer class under the two recursions.
- Division rates range from 0.52% (3207) to 16.0% (3906), median 6.1%; no division is free of disagreement. By decade the rate runs from 4.9% (1990s) to 8.7% (1890s); by calendar month it is flat (2,720 to 2,844 disagreeing months per month).

### Configuration F (fitted duration factors)

Reported by headline only. With each division's fitted factors the recursions disagree on 259,544 of 528,384 months (49.1%), by up to 13.27 units, and `pdi.f` is closer to nClimDiv on 58.8% of them. The difference from P is expected from the code: A applies half the *fitted* slope in the Wells port against a fixed 0.15, and the Wells port uses a cross coefficient (`dryc`) built from the wet intercept. Attribution for F was not run, so its classes are unknown. Neither recursion is wrong for F: `pdsi()` does not use fitted factors unless asked, and `scpdsi()` is defined by Wells.

### Figure specifications

Not rendered.

1. **Disagreement rate by division.** One value per division, `per_division_rate_P` in the results file (range 0.5% to 16%). Ranked horizontal bars with the median (6.1%) marked; a choropleth over nClimDiv boundaries would need a geometry source that is not in the repository. Colour by the dominant class only if the figure is kept to one panel.
2. **Annotated example series.** Division 0101, months 288 to 309 (1919-01 to 1920-10): Z-Index as bars, `pdi.f` and Wells PDSI as two lines, the establishment month (`x3 = 0.704` in the Wells port, below `pdi.f`'s threshold of 1) marked with the label E, and the later month where the two series rejoin marked. Source values are reproducible from the fixture with the commands above.

## Limitations

- Wells et al. (2004) text unread; `pdi.f` and Wells C++ sources not consulted.
- Palmer page references come from OCR of a scan; printed page numbers unchecked.
- Configuration F not attributed. H5's failure not attributed.
- One fixture (344 divisions, 1895 to 2022) and one parameter set for the hypotheses.
- The harmonizing edits are string replacements on a copy of `_palmer_pdi.py`; `build_pdi_variant` fails loudly if the source changes, but a refactor would need the replacements updated.

## Opportunity

Palmer's Table 12 (a 40-month western Kansas example with every intermediate column) is a candidate stage-by-stage literature fixture for REFIMPL-28 (#1630) and REFIMPL-26 (#1628). It is a scan, so it would need the transcription route used for earlier fixtures.
