# Flood reference status

`api.csv` compares the API recurrence against R `ahrapi` on synthetic inputs.
No exact numeric reference is committed for PE, EDI, or I_F. The Hickman, Nebraska
characterization of PE/EDI against Byun and Wilhite (1999) Table 5 lives in
`tests/fixture/flood_hickman/` with its own `provenance.json`.
`tests/test_flood_reference.py` asserts the selected Eq. (2) identity,
its endpoint weights, and the API closed form. The paper's candidate identities
are recorded here to make the ADR-0014 selection explicit:

- Byun and Wilhite's (1999) exact two-day identities: Eq. (1)
  `EP₂ = P₁ exp(-1/2) + P₂ exp(-1)`, Eq. (2)
  `EP₂ = P₁ + (P₁ + P₂) / 2`, and Eq. (3)
  `EP₂ = (2 P₁ + P₂) / 3`. ADR-0014 selects Eq. (2).
- The selected kernel's endpoint weights: `w₁ = H_D` and `w_D = 1 / D`.
- Kohler and Linsley (1951) Eq. (3), for constant precipitation:
  `API → P / (1 - k)`.

These are specification-level contract checks, not external scientific
validation or implementation-generated regression vectors.

## API vector comparison

The eight daily inputs in mm come from the example in the pinned `R/compute_api.R`
linked in `provenance.json`. The days are consecutive and synthetic, so there
are no calendar dates. The initial state is zero. There is no spin-up,
calibration, or missing precipitation. Both implementations use `k=0.95`.
Use R 4.6.2 and save the pinned file as `compute_api.R`. Run:

```sh
Rscript --vanilla -e 'source("compute_api.R"); p <- c(0, 0, 5, 2, 0, 0, 1, 0); a <- compute_api(p, k=0.95, initial=0); cat("precipitation_mm,api_mm\n"); cat(sprintf("%.17g,%.17g\n", p, a), sep="")' > api.csv
```

The source SHA-256 is `709d5ba42393130b2e899b9ee9ae4ca57204591114bc7f201ac3a678e4d1a86e`.
The 17 significant digits preserve binary64 values on read-back. The `1e-14` mm
absolute tolerance allows about eleven ulps at the maximum API of about 6.79 mm.
This allows rounding differences from eight recurrence steps, not a method change.
The fixture contains only numbers, not the MIT-licensed R code. The test does not
require R or network access. The R grid route treats missing rain as zero and
is not a reference for this library's gap policy.

## Deferred numeric oracles

| Index | Needed before committing an oracle |
| --- | --- |
| PE / EDI | Characterized, not exactly reproduced, in `tests/fixture/flood_hickman/` (single station, calendar-day-mean fill). For an exact reference, reconstruct the paper's 113-station High Plains subset and 100 km substitution rule; pin the 30-year MEP window within the 1960–1996 record; resolve AMS reuse terms. The published Table 5 uses variable-duration EDI, not the fixed-window public contract. |
| I_F | The Deo et al. (2015) full text confirms the kernel (`docs/research/deo-2015-flood-index-notes.md`); the Brisbane values need the paper's adjusted BoM 40094 rainfall for 1915–2012, which is not published, and its 365-day calendar, then reuse terms. The 2014 USQ conference paper is another possible source for the inputs. |
| API | Retrieve the remaining Kohler and Linsley (1951) pages to check for published numeric values. The R comparison does not establish agreement on observed inputs or validate flood prediction. |

Any future artifact must add schema-valid `provenance.json` metadata and must
not be generated from `climate_indices` output.
