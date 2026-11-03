# Reliability benchmark relationships

This fixture supports #1345. It supplements the analytic records in `../reliability/` without changing them.
`reference.json` contains three separate evidence types. None supplies an operational outage-log oracle for the sustained metrics or ASAI.

## EIA aggregate panel

Source: U.S. Energy Information Administration, *Electric Power Annual*,
[Table 11.1, Reliability Metrics of U.S. Distribution System](https://www.eia.gov/electricity/annual/html/epa_11_01.html).
The underlying source is Form EIA-861, *Annual Electric Power Industry Report*.

The panel retains every year from 2015 through 2024 for two national IEEE-method categories:

- `all_events` includes Major Event Days.
- `without_major_event_days` excludes Major Event Days according to the reporting utilities.

The panel does not mix these categories with the loss-of-supply or Any Method columns.
EIA identifies the IEEE reporting methods as the 2003 or 2012 editions, not the library's 2022 target.
SAIDI and CAIDI use minutes. SAIFI uses interruptions per customer per year.
The table supplies aggregate values, not durations, customer counts, or event identifiers.
The values are public domain under [EIA's reuse policy](https://www.eia.gov/about/copyrights_reuse.php).

The fixture normalizes the numerical values to an inferred column precision of 0.1 minute for SAIDI and CAIDI and 0.001 for SAIFI.
Cells such as SAIDI 268.4 and SAIFI 1.275 support this inference.
We interpret shorter values, such as SAIFI 1.42, as values with omitted trailing zeros.
This is an assumption, not a documented EIA precision guarantee.
Under this assumption, the rounding intervals have half-widths of 0.05 minute for SAIDI and CAIDI, and 0.0005 for SAIFI.
For published `s = SAIDI`, `f = SAIFI`, and `c = CAIDI`, the test requires:

```text
abs(c - s/f) <= 0.05 + max((s+0.05)/(f-0.0005) - s/f,
                           s/f - (s-0.05)/(f+0.0005))
```

This bound accounts for rounding in all three published values under the inferred precision convention. It is not a fitted tolerance.
The test checks every row, rejects a one-minute CAIDI corruption in every row, and verifies finite values and bounds.
The test also constructs equivalent two-row logs using 1,000 synthetic served customers.
Those logs check public-API accumulator consistency and ASAI bounds, not EIA's original event arithmetic or Major Event Day filtering.
The CMI tolerance is `atol=1e-9` customer-minutes because it scales the arithmetic by 1,000 customers.
Other arithmetic uses `rtol=0, atol=1e-12`; ASAI uses `atol=1e-15`.

## EPRI-style scenario

Source framing: EPRI, *Planning for Climate Resilience in the Power System*,
[Appendix A, Customer Outage Risk (CORi)](https://apps.epri.com/climate-readi-modeling-guide/en/appendix.html#customer-outage-risk-cori).
The Distribution Operations page links this appendix.
The appendix describes customer-normalized outage hours and their reliability counterparts, SAIDI and CAIDI.

All scenario numbers are constructed, not published EPRI benchmark values.
One hazard event interrupts two disjoint customer cohorts simultaneously.
Each cohort has 500 customers; restoration takes 60 minutes for one cohort and 180 minutes for the other.
The served population is 1,000. The total is 120,000 customer-minutes, or two outage hours per served customer per event.
At a supplied frequency of 0.25 events per year, the expected contribution is 0.5 outage hours per customer per year.
This frequency is an assumption, not a frequency estimator or an annualization performed by the library.

A second case halves the affected customer counts but keeps the served population and durations unchanged.
The per-event served-customer average becomes one hour, while CAIDI remains two hours per interruption.
Thus the general relationship is `frequency * SAIFI * CAIDI / 60`, not `frequency * CAIDI / 60` alone.
The latter applies here only when the event interrupts every served customer once.
The two restoration rows are not two hazard events, even though `outage_count` counts both rows.
The test expects the overlap warning and retains both cohorts.

This is a thin caller-side relationship. It does not implement CORi, LORi, DORi, fragility curves, or event probabilities.
It does not validate resilience forecasts or the Monte Carlo path.

## Published momentary cross-check

Source: IEEE Std 1366-2012, section 4.3.1, equations 47–49, printed page 17, PDF page 28.
The public copy is [Indiana docket 45647, Public's Exhibit CX-8](https://iurc.portal.in.gov/_entity/sharepointdocumentlocation/def9f7d7-91b6-ec11-983e-001dd80335bf/bb9c6bba-fd52-45ad-8e64-a444aef13c39?file=45647%2C+Public%27s+Exhibit+No.+CX-8.pdf).
The source describes 2,000 served customers, 750 momentary customers with two operations, and 250 sustained customers.
It prints SAIFI 0.125, MAIFI 0.75, and MAIFI_E 0.375.
The test reproduces these three numerical facts exactly; all three have exact binary representations.

The timestamps and durations in the JSON are constructed to satisfy the event schema.
They are not source observations. The sustained duration does not supply an external SAIDI or CAIDI reference.
The already-assembled momentary event does not test raw-operation grouping.
This is a published worked-example reproduction, not IEEE Std 1366-2022 conformance or operational event validation.
IEEE retains copyright in the source. This fixture does not redistribute the document, its figures, or its prose.

## Retrieval and integrity

All sources were retrieved on 2026-11-03.
The SHA-256 in `provenance.json` covers the exact `reference.json` bytes.
The following retrieval hashes identify the source versions inspected; the source documents are not committed.

| Source | SHA-256 |
| --- | --- |
| EIA Table 11.1 HTML | `d3fd52a698e72366cef4142d70c924a988adb3a336d337432d11e88d58c38410` |
| IEEE public docket PDF | `1834d6915399315b7e6fe20dbcf8ad3a80f871737a27f8341984465c2b29d158` |
| EPRI Appendix A content asset | `9304519bd083fa45fa5a2ac2ba3e6545a1e87f20efeb5f4dfbed983d75d0f27f` |

The EPRI page requires JavaScript. Its inspected content asset was
`https://apps.epri.com/climate-readi-modeling-guide/assets/appendix.html-Bag6C9kp.js`.
The EIA values were checked against the HTML table cells, not search-result summaries.
The IEEE numerical facts were checked against the retrieved PDF text.
The PG&E 2024 reliability PDF exceeded the web fetcher's 20 MB limit; it supplies no fixture here.
No operational event-level oracle was obtained. The IEEE worked example does not remove that limitation.

Run the offline checks with:

```bash
uv run pytest -m validation tests/test_reliability_reference.py
```

The default suite deselects every test in this module, but collection still imports the module and reads its fixtures.
Missing files cause failures, not skips or network downloads.
`provenance.json` follows `tests/fixture/provenance_schema.json` and records the licence and tolerances.
