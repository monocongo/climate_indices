# Reliability event fixture

These records are synthetic. They provide specification and regression checks, not external validation.
No external event dataset was downloaded.

The record schema is `src/climate_indices/reliability/event_schema.json`.
`provenance.json` follows `tests/fixture/provenance_schema.json`.
Its SHA-256 checksum covers the exact bytes of `events.json`.

Use `total_customers=1000` and `period_hours=24` for the analytic cases in ADR-0026, decision 10.
The two sustained records have 300 customer interruptions and 24,000 customer-minutes.
The two momentary records have 300 customer events and 750 customer device operations.
The final record has zero duration and zero customers. It remains in the normalized record.

`reliability_indices(events, total_customers=1000)` computes SAIFI 0.3 from the
two sustained records, with 300 customer interruptions and two outages. The
remaining accumulators are checks only until their metric tickets land.
