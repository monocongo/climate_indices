# Palmer (1965) literature fixtures

Numbers printed in Palmer, W. C. (1965), *Meteorological Drought*, U.S. Weather
Bureau Research Paper No. 45, transcribed for a stage-by-stage check of the
library's Palmer computation against a source that predates every software
lineage. Ticket: [#1630](https://github.com/monocongo/climate_indices/issues/1630)
(REFIMPL-28). The scan is not committed (see `provenance.json`); only transcribed
numbers are.

Source PDF used: `https://ncei.noaa.gov/monitoring-content/temp-and-precip/drought/docs/palmer.pdf`,
65 pages, SHA-256 `9b52259f8732de597f4c8c34982b196a03a5dc2668fdca0d0a5b22cf60d1e65f`.
Page numbers below are the printed page numbers; the PDF page is printed page + 6.

## Files

All quantities are in inches (precipitation, PE, soil moisture) and degrees
Fahrenheit, exactly as printed. An empty field is a dashed or blank cell in the
print; `T` (trace) is kept as a token.

| File | Table (printed page) | Content |
| --- | --- | --- |
| `table1_water_balance_central_iowa.csv` | Table 1 (p. 10) | Hydrologic accounting, central Iowa, 1933-1935, AWC 1.00 + 9.00 in. 36 months |
| `table2_long_term_means.csv` | Table 2 (p. 12) | Long-term monthly means, western Kansas (1887-1957, AWC 1 + 5) and central Iowa (1931-57, AWC 1 + 9) |
| `table3_coefficients.csv` | Table 3 (p. 12) | CAFEC coefficients alpha, beta, gamma, delta, and the weighting columns `k` and `K`, both regions |
| `table4_moisture_departures_western_kansas.csv` | Table 4 (p. 15) | CAFEC computation, western Kansas, June 1932 - May 1935 (36 months) |
| `table9_k_factors.csv` | Table 9 (p. 25) | D-bar, K', D-bar K' and K for western Kansas and central Iowa (the two regions that Tables 2 and 3 also cover), with the printed sums |
| `table12_severity_western_kansas.csv` | Table 12 (p. 31) | X1/X2/X3 spell recursion, western Kansas, April 1888 - July 1891 (40 months), every printed column |

Not transcribed: Tables 5-8 and 10-11 (departure series or worked months with no
inputs), Table 13 onward (X without inputs), and the other seven regions of
Table 9 (no matching means).

## Transcription protocol

1. Pass 1: read each table from 300 dpi page images, cell by cell.
2. Arithmetic audits on the printed numbers alone, no library involved (tests in
   `tests/test_palmer_literature.py`): every column sum, `S = Ss + Su`,
   `P = ET + RO + dS` and `PR = AWC - S'` in Table 1, `P-hat = col6 + col7 + col8 - col9`,
   `d = P - P-hat`, `z = d k`, `Z = d K` and `alpha PE`, `beta PR`, ... in Table 4,
   `D-bar K'`, `K = 17.67 K' / sum` and Equation 26 in Table 9, and the recurrences
   `Z/3`, `-0.103 X(i-1)`, `dX = Z/3 - 0.103 X(i-1)`, `X = X(i-1) + dX` and
   `Pe = 100 V / Q` in Table 12.
3. Pass 2: an independent re-transcription from the images, made without access
   to pass 1, then compared cell by cell. Every disagreement was settled against a
   600 dpi view of the page. See "Verification result" below.

A cell that fails an audit is either a transcription error (fixed) or a printing
error in the paper. Printing errors are **not corrected** in the CSVs; they are
listed below and handled in the tests.

## Printing inconsistencies found

All were re-read at 600 dpi: the CSV matches the page.

| Where | Printed | Arithmetic from the neighbouring printed cells | Consequence |
| --- | --- | --- | --- |
| Table 4, April 1934, columns 8 and 9 | gamma PRO = 0.26, delta PL = 0.06 | gamma PRO = 0.0359 x 1.65 = 0.06; delta PL = 0.2688 x 0.95 = 0.26. `P-hat = 2.43` only holds with the two values swapped | Columns 8 and 9 are transposed in the print for that row. `P-hat`, `d`, `z`, `Z` are consistent. |
| Table 9, western Kansas, April and May, row K | 1.38 (April), 1.54 (May) | `17.67 K' / 15.41` gives 1.54 (April), 1.38 (May), which is also what Table 3 column 7 prints | The K values for April and May are transposed in Table 9. Table 3 is used as the reference. Table 4's Z values agree with Table 3. |
| Table 9, western Kansas, February, row D-bar K' | blank | `0.43 x 1.92 = 0.83` | One illegible/omitted cell, left empty. |
| Table 12, June 1889, X1 | 1.10 | X1 (May) 0.95 + dX1 0.25 = 1.20 | X1 and X3 carry 1.10 forward, so the printed X3 from June 1889 onward is 0.10 below the printed recursion's own arithmetic (decaying by 0.897 a month). |
| Table 12, June 1890, Z | -2.92 | Z/3 prints -0.94, i.e. Z = -2.82; the printed dX2 (-0.84) is consistent with -0.94 | The Z column and Z/3 column disagree. |
| Table 12, July 1891, Z | 4.22 | Z/3 prints 1.74, i.e. Z = 5.22; printed dX3 (1.46) and X3 (4.21) are consistent with 1.74 | The Z column and Z/3 column disagree. |

## Hypotheses and tolerances (committed before any library comparison was run)

A tolerance is set by print precision, never by what makes a test pass. The
half-unit of a printed quantity is 0.005 for two decimals and 0.00005 for four.
Where a library result is built from several rounded printed inputs, the bound is
the sum of the half-units of each input weighted by its coefficient (worst-case
propagation, computed in the test).

| Stage | Library code | Hypothesis | Tolerance |
| --- | --- | --- | --- |
| H1 water balance | `palmer._calc_water_balances` (Python path) | Every printed Table 1 column (`Ss`, `Su`, `S`, `PR`, `R`, `PL`, `L`, `ET`, `RO`) is reproduced for all 36 months from the printed P and PE, starting with both layers full | 0.01 in (one print unit: rounded inputs and a rounded output) per cell |
| H2 CAFEC coefficients | `palmer._calc_cafec_ratio` | The printed Table 3 alpha-delta lie inside the interval implied by rounding the Table 2 means (Table 3's footnote says they were computed from sums, so exact agreement is not expected) | Interval arithmetic on the 0.005 half-unit of each mean; the number of months where the interval is narrower than 0.02 is reported |
| H3 K factors | `palmer._calc_kfactors` | Given the printed D-bar and K', the library normalisation reproduces Table 3 column 7 K | Propagated from the half-units of K' and the printed sum |
| H4 d, z, Z | `palmer._calc_cafec_zindex` | From the printed PE, PR, PRO, PL, P and the printed Table 3 coefficients and K, the library reproduces `P-hat`, `d` and `Z` for all 36 months | Propagated from the half-units of each printed input and coefficient |
| H5 X recursion (pdi.f lineage) | `_palmer_pdi.calculate` with Palmer's p = 0.897, q = 1/3 | Fed the printed Z, it reproduces printed X for all 40 months | 0.02 |
| H6 X recursion (Wells lineage) | `_palmer_wells.calculate`, same factors | Differs from the printed X, because Wells establishes a spell at \|X\| >= 0.5 where Palmer prints 1.0 (mechanism E in `docs/research/reference-papers/palmer-recursion-divergence.md`) | 0.02, as for H5 |

Table 12 is a mid-series excerpt. Palmer states that March 1888 had Z = +0.81 and
X = +0.27. The test therefore starts the series in January 1888 with Z = 0, 0, 0.81;
that prefix is modelled, not printed, and only April 1888 onward is compared.
A month is excluded from the H5 count only if it is downstream of one of the
three printed inconsistencies above, and every excluded month is named in the test.

## Verification result

Filled in with the test results (see the tests and the discrepancy ledger).
