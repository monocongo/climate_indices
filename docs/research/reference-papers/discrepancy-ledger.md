# Discrepancy ledger

**Ticket:** REFIMPL-24, [#1626](https://github.com/monocongo/climate_indices/issues/1626).
**Status:** seeded by REFIMPL-29 ([#1631](https://github.com/monocongo/climate_indices/issues/1631)) with that ticket's entries only. REFIMPL-24 owns the format, the machine-readable copy, and the summary tables, and may restructure this file.

Dispositions follow the categories in #1626: method choice, undocumented convention, defect in `climate_indices`, defect in a comparator, numerical noise, unexplained. **Every disposition below is a proposal for the maintainer**; none has been decided. A default change is never decided here.

## Entries from REFIMPL-29

All entries: study [#1631](https://github.com/monocongo/climate_indices/issues/1631), index PDSI, comparison `_palmer_pdi.calculate` (NCEI `pdi.f` lineage) against `_palmer_wells.calculate` (Wells lineage), same Z-Index and same duration factors. Pinned commit `46481c48`. Configuration P is Palmer's national factors on the 344-division fixture. The pinned input is the test case of the same name in `tests/test_palmer_recursion_divergence.py`. Months are of 33,373 disagreeing months out of 528,384. Method: [palmer-recursion-divergence.md](palmer-recursion-divergence.md).

| ID | Class | Configuration | Size | Cause | Proposed disposition |
| --- | --- | --- | --- | --- | --- |
| P29-E | E | P | 3,415 episodes, 30,212 months (90.5% of all disagreeing months); up to 3.52 | Establishment threshold: `pdi.f` promotes X1/X2 to X3 at \|X\| >= 1, Wells at >= 0.5. Palmer (1965) PDF p. 38 supports 1.0 | **Method choice** (documented lineage difference, in `docs/algorithm_refs/palmer.md` and `VALIDATION.md`); add the threshold to the REFIMPL-24 method matrix, which does not exist yet. Palmer's text favours the `pdi.f` side |
| P29-T | E+T, E+T+A, E+T+C | P | 90 episodes, 1,127 months; up to 1.92 | Termination: `pdi.f` ends a spell at \|X3\| <= 0.5 (Palmer p. 35); Wells ends it only when the end probability reaches 100. Always together with E | **Method choice**; same row in the REFIMPL-24 method matrix as E |
| P29-A | A | P | 163 episodes, 806 months; **up to 8.14** (0501, 1915-10) | Abatement trigger: `pdi.f` uses 0.15; Wells uses half the fitted slope, 0.1545 at national factors. Palmer prints 0.15 as the rounding of 0.309 x 0.5 (p. 35) | **Undocumented convention.** The 0.0045 gap is not mentioned in the user documentation yet gives the largest disagreement in the fixture. Documentation issue; a note for any other `pdi.f` port |
| P29-C | C, E+C, A+C, E+A+C, E+T+C | P | 69 episodes, 689 months; up to 3.95 | Candidate carried at establishment: `pdi.f` restarts `px2` from zero after an X1 promotion (`_palmer_pdi.py:384`); Wells recomputes X2 every month. Palmer p. 38 says both indices are computed "at every opportunity" but is silent on the establishing month | **Unexplained** until the Fortran source or the Wells text decides it. If the Fortran restarts `px2`, **method choice**; if not, a defect in the `pdi.f` port |
| P29-F | F, E+F, overdetermined with F | P | 47 episodes, 223 months (including 15 months in overdetermined episodes); up to 3.42 (1502, 2022-03). The 48 episodes left when only F is unharmonized start in the last 21 months of the record (months 1515-1535); E+F episodes can start earlier (division 1207 at month 1473). Those start months were observed during the run; the results JSON does not carry them | End-of-record flush: `_palmer_pdi._finish_up` loops over `k8max` rather than the open window `k8` and rewrites stale slots with tentative values, undoing backtracking. Palmer's text has no passage on it | **Possible defect in `climate_indices`.** Needs the `pdi.f` source to say whether the Fortran does the same. Maintainer to decide on a bug issue with a failing test; not changed here |
| P29-A2 | E+A, overdetermined E+A | P | 51 episodes, 415 months; up to 3.53 | E+A (50 episodes): E and A both required. Overdetermined E+A (1 episode): E alone or A alone removes it | **Method choice** / undocumented convention, as the two rows above |
| P29-FP | floating point | P | 0 episodes | The `b/(m+b)` and `1 - m/(m+b)` forms differ by 1 ulp and move PDSI by up to 5e-15 but flip no branch; both figures were observed during the run and are not in the results JSON | **Numerical noise**; none observed on this fixture |
| P29-TOL | tolerances | P | 0 episodes | No episode needs the Wells port's 1e-5 tolerances to be harmonized (all but Tol leaves 0 months at the 1e-9 threshold); equality of output values below that threshold was not tested | **Numerical noise**; none observed |
| P29-H5 | H5 lookback failure | P | 9,990 of 248,033 months (4.03%) disagree after three same-sign established months with \|X3\| > 1.5 in both machines | The measured month can end or change the spell. This test does not locate disagreement relative to spell boundaries (D4). Which classes cause it was not attributed | **Unexplained** (not attributed) |
| P29-FIT | fitted factors | F | 259,544 of 528,384 months (49.1%); up to 13.27 | The recursions diverge far more under each division's fitted duration factors. Likely sources: A (half the fitted slope against 0.15) and the Wells `dryc` cross coefficient; not tested | **Unexplained** (not attributed) |

The `pdi.f`-lineage values are closer to nClimDiv on 87.6% of the P disagreements (29,249 of 33,373, no ties). That is characterization, not validation: the Z-Index is the library's own, and it does not upgrade any evidence status in `VALIDATION.md`.

## Not logged here

- The 6.3% and 8-unit provenance figures reproduce (6.316%, 8.135), so no change to `tests/fixture/palmer/provenance.json`, `VALIDATION.md`, or `docs/algorithm_refs/palmer.md` is proposed.
- Whether Wells et al. (2004) fixes E, A, or C could not be checked: the full text was not retrievable (HTTP 403).
