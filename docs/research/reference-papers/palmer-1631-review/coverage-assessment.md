# Coverage of the Palmer spell-establishment comparison

[Case E](evidence-sheet.md) isolates one dry-spell decision in five active months. This supplement checks the same threshold rule for wet and dry spells over the complete fixed-factor fixture used in [#1631](https://github.com/monocongo/climate_indices/issues/1631). The five-month case remains the starting point for independent source review.

## Method

We replayed both pinned Python recursions over the full records for all 344 climate divisions from January 1895 through December 2022. Each pair received the same monthly Z-Index from Palmer fixture version 1.1.1 and the same fixed national duration factors. We also changed the standard recurrence's establishment threshold from 1.0 to 0.5 in memory, then compared final PDSI with the Wells output. A material difference exceeds `1e-9` PDSI units. A disagreement episode is a maximal run of such months within a division. [Original study](../palmer-recursion-divergence.md#sample)

For the direct decision check, we included months when both runs carried no established spell (`X3 = 0`) into the month and their previous wet and dry candidates matched within `1e-9`. We advanced the common recurrence for each candidate using that state and the current Z-Index. The dominant magnitude had to exceed the other by more than `1e-9` and lie more than `1e-9` from either boundary. The other candidate had to lie below 0.5 by more than `1e-9`. We compared three predictions with each implementation's recorded post-month `X3`.

| Dominant candidate magnitude | Standard Palmer | Wells |
| --- | --- | --- |
| Below 0.5 | No establishment | No establishment |
| Between 0.5 and 1.0 | No establishment | Establishment |
| Above 1.0 | Establishment | Establishment |

We grouped the matched decisions by sign and by three periods of nearly equal length. For each sign and period, we ranked the between-threshold cases by SHA-256 of the division identifier and month, then selected the first two from different state codes. This selection uses neither the observed PDSI difference nor its size.

## Results

**All 3,084 matched decisions agreed with the published threshold magnitudes.** The Wells lineage alone established a spell in 1,326 division-months across 336 divisions, including 772 dry and 554 wet cases. The other 1,758 months checked agreement below both thresholds or above both.

| Period | Candidate sign | Below 0.5 | Between 0.5 and 1.0 | Above 1.0 |
| --- | --- | ---: | ---: | ---: |
| 1895–1936 | Dry | 285 | 348 | 93 |
| 1895–1936 | Wet | 303 | 271 | 154 |
| 1937–1979 | Dry | 197 | 247 | 66 |
| 1937–1979 | Wet | 188 | 166 | 106 |
| 1980–2022 | Dry | 140 | 177 | 45 |
| 1980–2022 | Wet | 105 | 117 | 76 |
| **Total** | | **1,218** | **1,326** | **540** |

Final PDSI differed in the decision month in just one of the 1,326 split decisions. An incipient candidate and an established spell can yield the same reported PDSI for that month. Case E shows how their later paths can produce a different final value.

Changing the standard threshold alone cleared **30,992** of the 33,373 originally differing months and produced **631** differences in months that previously agreed. The resulting count was 3,012, a net reduction of 91.0%. Of the 3,832 baseline disagreement episodes, 3,421 cleared completely, 140 cleared in part, and 271 still differed throughout their original spans.

### Twelve examples for inspection

In each example, standard Palmer's post-month `X3` is zero and Wells's `X3` equals the signed candidate shown. Values are rounded here to six decimal places.

| Period | Candidate sign | Division | Month | Wells X3 |
| --- | --- | --- | --- | ---: |
| 1895–1936 | Dry | 0802 | 1915-08 | −0.677981 |
| 1895–1936 | Dry | 0304 | 1895-02 | −0.764742 |
| 1895–1936 | Wet | 3309 | 1895-01 | 0.731275 |
| 1895–1936 | Wet | 3610 | 1895-01 | 0.668159 |
| 1937–1979 | Dry | 3403 | 1978-01 | −0.532317 |
| 1937–1979 | Dry | 0102 | 1940-10 | −0.636921 |
| 1937–1979 | Wet | 2002 | 1957-11 | 0.891277 |
| 1937–1979 | Wet | 4604 | 1970-12 | 0.767284 |
| 1980–2022 | Dry | 4605 | 1982-05 | −0.650083 |
| 1980–2022 | Dry | 3209 | 1992-03 | −0.507022 |
| 1980–2022 | Wet | 3007 | 2003-07 | 0.873441 |
| 1980–2022 | Wet | 4804 | 2011-01 | 0.520991 |

## Scope

These are descriptive counts for the same library-derived Z-Index fixture. Adjacent months and nearby divisions can share the same weather and recurrence history. The matches increase coverage of the implementation check, and the illustrative cases give the reviewer examples for both signs. Scientific interpretation still depends on the cited publications. Exact-boundary semantics are addressed in the [case E evidence sheet](evidence-sheet.md).

## Replay

From the repository root, run:

```sh
uv run python scripts/audit_palmer_establishment_coverage.py
```

The command verifies the original study's recursion pin `46481c48` and checks the baseline and threshold-only output totals against the committed study. The fixture and division list match the review packet's checkout `380aa01f`. The command prints counts by period, sign and threshold band, along with the selected cases at full floating-point precision.
