# Flood reference status

No numeric oracle artifact is committed here, so this directory intentionally
has no `provenance.json`. The schema applies when an external reference dataset
is added. `tests/test_flood_reference.py` instead records exact, source-backed
algebraic identities:

- Byun and Wilhite (1999) Eq. (2), selected by ADR-0014:
  `EP₂ = P₁ + (P₁ + P₂) / 2`.
- The same kernel's endpoint weights: `w₁ = H_D` and `w_D = 1 / D`.
- Kohler and Linsley (1951) Eq. (3), for constant precipitation:
  `API → P / (1 - k)`.

These are specification-level contract checks, not external scientific
validation or implementation-generated regression vectors.

## Deferred numeric oracles

| Index | Needed before committing an oracle |
| --- | --- |
| PE / EDI | Reconstruct the paper's 113-station High Plains subset and 100 km substitution rule; pin the 30-year MEP window within the 1960–1996 record; resolve AMS reuse terms. The published Table 5 uses variable-duration EDI, not the fixed-window public contract. |
| I_F | Retrieve Deo et al. (2015) or Byun and Jung (1998), obtain reproducible rainfall inputs, and confirm reuse terms. The 2014 USQ conference paper is another retrieval path. |
| API | Retrieve the remaining Kohler and Linsley (1951) pages and confirm whether they contain reproducible numeric values. The pages currently held are definitional only. |

Any future artifact must add schema-valid `provenance.json` metadata and must
not be generated from `climate_indices` output.
