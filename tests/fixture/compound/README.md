# Public drought–heatwave LMF reference fixture

This fixture supports CMP-02 (#1366). It contains public study inputs and the
same-author deterministic likelihood multiplication factor (LMF) calculation.
It does not validate a compound API, which is not implemented yet.

## Source and evidence scope

Vieira Passos, M., Kan, J.-C., Destouni, G., Barquet, K., and Kalantari, Z. (2024).
*Identifying regional hotspots of heatwaves, droughts, floods, and their co-occurrences*.
*Stochastic Environmental Research and Risk Assessment* 38, 3875–3893.
<https://doi.org/10.1007/s00477-024-02783-3>.

The public inputs are Vieira Passos, M. (2024), *Climate Indices for heat waves,
droughts and floods in Sweden (1922-2021)*, version 2, Figshare.
<https://doi.org/10.6084/m9.figshare.25189118.v2>.
The dataset is CC BY 4.0. This subset retains the original values, including missing values;
its only data reduction selects three spatial cells.

The author calculation is `Compound/Compound_patterns.ipynb` at commit
`6c88f255ee3034279dcebe2a2fc40c954dbe9575` in
<https://github.com/m-passos/hydroclimatic-hazards-hotspots>.
The preparation script adapts only the monthly-to-daily preparation (cell `63ae9174`) and the deterministic
LMF arithmetic (cell `93ef7024`), not the random significance test.
The notebook is checksum-verified as a source artifact and is never executed.
The adapted code's MIT notice is in `NOTICE`.

Matching this fixture later can establish scoped **reference-reproduced, not independently validated** evidence.
This ticket supplies no production comparison and makes no new claim in `VALIDATION.md`.
The nine cases do not establish a full reproduction of Figure 8, its significance mask, or upstream SPEI correctness.
CMP-10 (#1374) owns the compound API comparison and final evidence classification.

## Files and coordinates

- `inputs.npz` stores monthly dates, monthly `SPEI`, daily dates, and daily `heatwave index` values as numeric arrays.
- `reference.json` records expected counts, probabilities, LMF, source hashes/file IDs, generation versions, and processing choices.
- `provenance.json` follows the repository schema and hashes canonical `reference.json`.
- `reference.json` hashes `inputs.npz`, so both expected results and raw inputs have integrity checks.

| Cell | Selection target (latitude, longitude) | Source indices (lat, lon) | Actual coordinate, rounded |
| --- | --- | --- | --- |
| Southern | 56, 14 | 5, 23 | 56.03652424, 14.04645455 |
| Central | 62, 15 | 48, 30 | 62.04727273, 14.97363636 |
| Northern | 67, 20 | 83, 68 | 66.93974242, 20.00690909 |

Selection takes the nearest latitude and longitude independently; an exact tie takes the first array index.
The targets were fixed before calculating LMF and were not selected for significant results.
Full-precision coordinates are in `reference.json`.

The monthly source has 1,200 month-end observations from January 1922 through December 2021.
The daily source has 36,525 observed Gregorian dates over those years, including real leap days.
Each monthly series contains 11 missing source values; their dates remain in `reference.json` and their NaNs remain in `inputs.npz`.
The supplied NetCDF attributes do not identify the Calibration Period. The fixture records that fact rather than assuming one.

## Convention audit

The source notebook defines the processing sequence:

1. Forward-fill monthly SPEI onto daily dates, from 31 January 1922 through 31 December 2021.
2. Interpolate remaining NaNs linearly along the daily time coordinate.
3. Select June, July, and August separately across all 100 years.
4. Classify heat with `HWI >= 0` and drought with `SPEI <= -1`.
5. Count each occupied day once and divide joint frequency by the product of the two observed marginal frequencies.

Padding happens before interpolation. Finite monthly values therefore remain constant until the next month-end label.
Only the gaps created by missing monthly values receive linear interpolation.
All prepared summer inputs at the selected cells are finite.
Each June case has 3,000 days; July and August each have 3,100 days.

These are the author's published-code conventions, not recommendations for the new module:

- Monthly-to-daily padding does not create independently observed daily drought resolution.
- The inclusive source thresholds differ from the strict probability comparisons in ADR-0022.
  A later compound comparison must use explicitly declared Boolean indicators and preserve these original comparators in its manifest.
- No additional minimum-duration filter is introduced into the author's daily threshold arrays.
- Original source NaNs remain auditable even though the author's preparation fills their corresponding daily gaps.
- The author returns NaN when no joint occurrence exists. JSON represents that value as `null`.
  ADR-0022 instead specifies zero LMF when both marginals are positive and the joint count is zero.
  A small explicit test preserves this intentional difference; all nine empirical cases have positive joint counts.
- The author's IID daily bootstrap and random tie noise are excluded. This fixture contains no p-values.
- Suppression means `0 <= LMF < 1`. Negative LMF values are impossible for the ratio, despite the paper's prose.

The offline checks use a separate date-index lookup, gap interpolation, contingency table, and rational arithmetic.
They do not call the fixture generator's LMF routine or import `climate_indices.compound`.
They check counts exactly and ratios with `rtol=0`, `atol=1e-12` for floating-point arithmetic only.

## Regeneration

Keep the large source files outside the repository.
Download only the two pinned data files and the pinned notebook:

```bash
curl --fail --location --output SPEI12.nc https://ndownloader.figshare.com/files/44481290
curl --fail --location --output heatwave_SMHI.nc https://ndownloader.figshare.com/files/44481308
curl --fail --location --output Compound_patterns.ipynb https://raw.githubusercontent.com/m-passos/hydroclimatic-hazards-hotspots/6c88f255ee3034279dcebe2a2fc40c954dbe9575/Compound/Compound_patterns.ipynb
```

Record the actual retrieval date in `--download-date`.
`--output-dir` is required and must name a new directory below this checkout's `tmp/compound-reference/`.
The script resolves the path and rejects traversal or symlinks that escape that directory.
It also refuses existing outputs, including dangling symlinks, so regeneration cannot overwrite committed evidence:

```bash
uv run python scripts/prepare_compound_reference_fixture.py --source-dir /path/to/sources --download-date YYYY-MM-DD --output-dir tmp/compound-reference/regenerated
uv run python -c "
import json, numpy as np
old, new = 'tests/fixture/compound', 'tmp/compound-reference/regenerated'
a, b = (json.load(open(f'{d}/reference.json')) for d in (old, new))
assert a['cells'] == b['cells'] and a['cases'] == b['cases']
with np.load(f'{old}/inputs.npz') as x, np.load(f'{new}/inputs.npz') as y:
    assert x.files == y.files
    [np.testing.assert_array_equal(x[k], y[k]) for k in x.files]
print('regenerated content matches the committed fixture')"
```

The repository tests read only the committed directory, so they cannot check a regenerated one.
The comparison above checks the meaningful content instead.
The `environment` block, `inputs_sha256`, and the provenance checksum can differ between platforms,
because compressed-archive bytes depend on the zlib build and `reference.json` records library versions.
Those differences are not evidence of changed data.
Replacing the committed files needs a deliberate review of the resulting diff.

The script verifies the full SHA-256 of each source before reading data or creating output.
It stages the three numeric/JSON files beside the target and publishes them together, so a failed run leaves nothing behind.
The source pair totals about 3.02 GB, but extraction loads only three spatial series.
Retain this README and the notices when publishing the fixture.
A changed source hash is a retrieval blocker, not permission to update expected values.
