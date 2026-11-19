# Palmer recursion divergence: `pdi.f` versus Wells

**Ticket:** REFIMPL-29, [#1631](https://github.com/monocongo/climate_indices/issues/1631). **Map:** [#1602](https://github.com/monocongo/climate_indices/issues/1602).
**Pinned version:** `46481c48c79e878eb50ce3398ba062fd3a1ff38c` (`origin/main`, 2026-11-18).
**Status:** pre-registration. No analysis code has been run against the fixture. Results, minimal sequences, and the textual analysis are added in later commits; this section is not edited afterwards except to record a deviation under "Deviations".

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

### Other change before the first commit

The plan named `suitability.drought_category` for category changes. It was replaced by the Palmer table above because that function classifies SPI/SPEI.
