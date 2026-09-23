# Flood reference status

No numeric oracle artifact is committed here, so this directory intentionally
has no `provenance.json`. The schema applies when an external reference dataset
is added. `tests/test_flood_reference.py` asserts the selected Eq. (2) identity,
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

## Deferred numeric oracles

| Index | Needed before committing an oracle |
| --- | --- |
| PE / EDI | Reconstruct the paper's 113-station High Plains subset and 100 km substitution rule; pin the 30-year MEP window within the 1960–1996 record; resolve AMS reuse terms. The published Table 5 uses variable-duration EDI, not the fixed-window public contract. |
| I_F | Retrieve Deo et al. (2015) or Byun and Jung (1998), obtain reproducible rainfall inputs, and confirm reuse terms. The 2014 USQ conference paper is another retrieval path. |
| API | Retrieve the remaining Kohler and Linsley (1951) pages and confirm whether they contain reproducible numeric values. The pages currently held are definitional only. |

Any future artifact must add schema-valid `provenance.json` metadata and must
not be generated from `climate_indices` output.
