# Original Keetch–Byram Drought Index formulation

**Research date:** 2026-09-09  
**Scope:** The normative scientific formulation for [Establish the normative original KBDI formulation][issue]. This note separates the 1968 source from later corrections, variants, and modern API policy.

## Answer

Use Keetch and Byram's original English-unit operational procedure with the documented correction of Equation 18's final numerator constant from the misprinted `0.830` to **`8.30`**. The original publication defines two related but numerically different workflows:

1. **1968 table workflow:** integer lookup Tables 1–5 plus the daily field instructions. This is the only fully specified original operational workflow and the workflow exercised by the published Figure 1 series.
2. **Corrected Equation 18 workflow:** direct evaluation at the supplied temperature, mean annual rainfall, and rain-reduced KBDI. This supports continuous values but requires explicit project decisions for temperature range, rounding, bounds, and missing data.

Do not implement the literal printed `0.830`, which conflicts with the publication's tables and published corrections. Do not silently substitute an Australian/Finkele or other regional KBDI variant.

## Corrected Equation 18

Keetch and Byram print this equation on p. 31:

```text
dQ = ((800 - Q) * (0.968 * exp(0.0486 * T) - 0.830)
      / (1 + 10.88 * exp(-0.0441 * R))) * dτ * 10^-3
```

Alexander documents that `0.830` is a typographical error. The intended English-unit form is:

```text
dQ = ((800 - Q) * (0.968 * exp(0.0486 * T) - 8.30)
      / (1 + 10.88 * exp(-0.0441 * R))) * dτ * 10^-3
```

| Symbol | Meaning | English unit |
|---|---|---|
| `dQ` | daily drought factor/increment | hundredths of an inch |
| `Q` | previous KBDI after any same-day net-rain reduction | hundredths of an inch |
| `T` | daily maximum air temperature | °F |
| `R` | long-term mean annual precipitation | inches |
| `dτ` | time increment | one day |

Sources: Keetch and Byram (1968), p. 31, Equation 18; Alexander (1990), p. 23, Figure 1 and symbol definitions; Alexander (1992), p. 61.

The correction is numerically material. For Figure 1's June 10 example (`Q = 190`, `T = 70 °F`, `R = 50 in`), corrected Equation 18 gives `dQ = 5.758343…`, which rounds to Table 4's factor `6`; literal `0.830` gives `7.830017…`, inconsistent with the original table. Alexander also corrects Equation 15's derivational constant to `0.2113` from `2.113`; Equation 15 is not needed when Equation 18 is implemented directly.

Alexander (1990), p. 23, supplies this corrected SI form:

```text
dQ = ((203.2 - Q) * (0.968 * exp(0.0875 * T + 1.5552) - 8.30)
      / (1 + 10.88 * exp(-0.001736 * R))) * dτ * 10^-3
```

Here `Q` and `dQ` are millimetres, `T` is °C, and `R` is millimetres. This is a published metric conversion/correction, not the original English-unit field and table contract.

## Original daily operation

Keetch and Byram's instructions establish this order (pp. 10–13):

1. Record precipitation for the preceding 24 hours to `0.01 in` and determine net rain using the event procedure below.
2. Reduce the previous KBDI by net rain expressed in hundredths of an inch.
3. Use today's rounded maximum temperature and the **reduced** KBDI to obtain the drought factor from the selected table.
4. Add the drought factor to obtain today's KBDI.

Thus same-day rain reduction happens before drought-factor calculation and addition. The June 3 worked row demonstrates `182 - 46 = 136`, followed by a Table 4 drought factor of `6`, giving `142` (pp. 11–13).

## Effective-rain event state

The source's explicit rules are (p. 12):

- An isolated daily rain **greater than** `0.20 in` has `0.20 in` removed; the remainder is net rain.
- Rain at or below `0.20 in` has no effect on KBDI.
- During consecutive rainy days with no canopy drying between showers, subtract `0.20 in` only once, on the day cumulative rain first **exceeds** the threshold.
- After the threshold-crossing day, all rain on each subsequent rainy day is net rain until the wet spell ends.
- The first 24-hour period with no measurable rain ends the wet spell.
- When snow blankets fuels, assume no drying and transfer all measured water equivalent as net rain.

A stateful interpretation is:

```text
dry -> accumulating: positive rain, event total <= 0.20 in
accumulating -> effective: event total > 0.20 in
effective -> effective: subsequent rain while the wet spell continues
any state -> dry: first 24-hour period with no measurable rain
```

On the crossing day, `net_rain = event_total - 0.20 in`; before crossing it is zero; after crossing, `net_rain = daily_rain`. Consequently, exactly `0.20 in` is not effective rain, while an event of `0.20 + 0.01 in` yields `0.01 in` effective rain on day two. Figure 1 confirms both threshold crossing (`0.16 + 0.09 -> 0.05`) and continuation (`0.25 -> 0.05`, then `0.16 -> 0.16`).

The phrase "no drying of tree canopy between showers" cannot be inferred exactly from daily totals. With only daily precipitation, treating adjacent positive-rain records as one wet spell is a required implementation approximation, not an explicit source equivalence. The special snow rule likewise cannot be implemented without extra snow state; ordinary precipitation inputs alone do not identify it.

## Tables, temperature, and mean annual rainfall

### Table semantics

| Table | Printed mean-annual-rainfall range | Rainfall used to generate the table |
|---|---:|---:|
| 1 | 10–19 in | 15 in |
| 2 | 20–29 in | 25 in |
| 3 | 30–39 in | 35 in |
| 4 | 40–59 in | 50 in |
| 5 | 60 in or more | 70 in |

The factors were tabulated from Equation 18 at 3 °F increments, computed to tenths, and rounded to whole numbers with `0.5` rounded upward (p. 4; Tables 1–5, pp. 5–9). The source instructs observers at a table boundary to use the next-higher table, giving `19.50` and `39.50 in` as examples (p. 13). It does not formally define every fractional boundary or a table below 10 inches.

For direct Equation 18, `R` is the caller-supplied long-term mean annual precipitation in inches. It is not a rolling annual total, is not inferred from the calculation period, and is not quantized to a table category. The report says KBDI may be computed for any desired mean annual rainfall (p. 4).

### Temperature semantics

The field workflow records daily maximum air temperature—or dry-bulb temperature at the basic observation—to the nearest whole °F, with fractions of `0.5` and above rounded upward (p. 12). Tables use rows `50–52`, `53–55`, through `104–106` and `107+` (pp. 5–9). The derivation fits the empirical temperature curve over `50–110 °F` (p. 30), and the summary states drought development occurs at daily maximum temperatures of `50 °F` or higher (p. 22).

Exact table compatibility therefore uses no drought factor below 50 °F and the `107+` row at rounded temperatures of at least 107 °F. The source does not define how a continuous Equation 18 implementation should behave below 50 °F or above the fitted 110 °F range. It also gives no separate temperature-rounding rule for continuous evaluation.

### Rounding semantics

| Quantity | Source rule |
|---|---|
| Precipitation observation | nearest `0.01 in`; tie handling unstated |
| Temperature observation | nearest whole °F; `0.5` rounds upward |
| Table drought factor | nearest integer; `0.5` rounds upward |
| Table-workflow KBDI | integer hundredths of an inch, implied by observations, table state, and integer factors |
| Continuous Equation 18 output | no final-rounding rule stated |

Binary floating-point's ties-to-even rounding is not an exact implementation of the table workflow's half-up rule.

## Bounds and initialization

KBDI is explicitly a `0–800` scale: zero represents saturation/no moisture deficiency and 800 represents maximum drought (pp. 13–14). The tables have no negative state and their drought factor reaches zero at 800. Preserving that stated physical range requires:

```text
q_after_rain = max(0, q_previous - net_rain_hundredths)
q_today = min(800, q_after_rain + drought_factor)
```

The publication does not print these clamps as pseudocode; they are necessary mathematical interpretations for rain greater than the current deficit and possible near-800 overshoot.

The source does **not** prescribe automatic zero initialization. It directs observers to backtrack to a date when saturation is reasonably certain and calculate forward. Suggested anchors include spring snowmelt in heavy-snow areas or roughly 6–8 inches of rain in one week in snow-free areas, after which KBDI should be very low, if not zero (p. 10). Requiring or accepting a previous-day KBDI is faithful; defaulting it to zero is a modern API convenience that must be documented as such.

## Contracts the source leaves open

| Topic | Source status | Required project decision |
|---|---|---|
| Missing precipitation or temperature | Unspecified | Reject or propagate unknown through the cumulative recurrence; never silently delete or zero-fill a day. |
| Missing, duplicate, or out-of-order dates | Daily progression implied; behavior unspecified | Require contiguous, ordered, unique daily observations or an explicit restart state after a gap. |
| Reporting boundary/time zone | Only preceding 24 hours/today's temperature specified | Require aligned daily records and document the observation-day convention. |
| Leap days/calendar | `dτ = 1 day`; no calendar-year state | Treat each contiguous 24-hour record equally; do not invent all-leap reshaping. |
| Climatology period | "long-term mean annual rainfall" without a normal-period definition | Require caller-supplied climatology; do not derive it from the run by default. |
| Nonphysical values | Unspecified | Validate nonnegative precipitation and positive mean annual precipitation. |
| Snow-covered fuels | Special all-water-equivalent rule, but no input contract | Either add explicit snow state or document snow handling as unsupported. |
| Continuous output rounding | Unspecified | Preserve continuous values unless a separately declared table-compatibility mode is selected. |
| Continuous temperature outside 50–110 °F | Unspecified | Choose and document rejection, clipping, or zero/extended-factor behavior. |

## Table workflow versus corrected continuous equation

The workflows cannot be expected to agree exactly because the tables discretize temperature into 3 °F bins, KBDI into 50-point columns, mean annual rainfall into five reference values, and drought factors into integers. Figure 1 is an exact oracle for the table workflow, not generally for direct continuous Equation 18.

A table-compatible implementation would require English-unit observation quantization, table selection, integer state/factors, and exact reproduction of Tables 1–5. A corrected-equation implementation can accept exact climatology and retain continuous values, but it must own and document the modern decisions listed above.

## Later formulations not normative here

- Alexander's corrected SI equation is a direct metric expression of corrected Equation 18, but it is not the original English field/table workflow.
- The current USFS/WFAS description corroborates the inputs, greater-than-0.20-inch threshold, and reduce-before-increase order, but does not specify the complete wet-spell, rounding, initialization, or direct-equation contract.
- xclim's pinned KBDI implementation explicitly follows the later Finkele Australian/FFDI method. It uses a 5.0 mm allowance rather than 5.08 mm, continuous metric values, default-zero initialization, and different recurrence details. It is not an oracle for the 1968 procedure.

## Published worked-series anchor

Figure 1 starts from a previous-day KBDI of 164 and uses Table 4. Its 30 daily results are:

```text
[174, 182, 142, 151, 159, 173, 177, 176, 190, 196,
 200, 204, 212, 215, 219, 226, 235, 248, 263, 271,
 276, 283, 295, 311, 328, 345, 365, 378, 380, 374]
```

It covers isolated rain, threshold-crossing multi-day rain, continued effective rain, and same-day reduce-then-factor ordering. It does not settle saturation initialization, missing values, calendars, or continuous-equation semantics.

## Sources

- [Keetch, J. J.; Byram, G. M. (1968), *A Drought Index for Forest Fire Control*, Research Paper SE-38, U.S. Forest Service][se38-record]. [PDF][se38-pdf]. Relevant printed pages: 4–9 (table construction and tables), 10–13 (initialization and daily operation), 22 (temperature summary), 30–31 (derivation and Equation 18).
- [Alexander, M. E. (1990), *Computer Calculation of the Keetch-Byram Drought Index—Programmers Beware!*, *Fire Management Notes* 51(4):23–25][alexander-1990]. Corrected equations and units appear on p. 23; revised-reprint/table notes appear on p. 24. See also the [FRAMES catalog record][alexander-frames].
- [Alexander, M. E. (1992), *The Keetch–Byram Drought Index: A Corrigendum*, *Bulletin of the American Meteorological Society* 73(1):61–62][alexander-1992].
- [Drought.gov, USFS/WFAS KBDI description][drought-gov].
- [xclim v0.62.0 Finkele/FFDI implementation at commit `6328035`][xclim-source].

[issue]: https://github.com/monocongo/climate_indices/issues/787
[se38-record]: https://research.fs.usda.gov/treesearch/40
[se38-pdf]: https://research.fs.usda.gov/download/treesearch/40.pdf
[alexander-1990]: https://library.ignfa.gov.in/wp-content/uploads/2026/01/computer-calculation-of-the-keetch-byram-drought-index-programmers-beware-30112021.pdf
[alexander-frames]: https://www.frames.gov/catalog/10946
[alexander-1992]: https://doi.org/10.1175/1520-0477-73.1.61
[drought-gov]: https://www.drought.gov/data-maps-tools/keetch-byram-drought-index
[xclim-source]: https://github.com/Ouranosinc/xclim/blob/6328035cce3661409acae5a11d7f00d45a1bc82f/src/xclim/indices/fire/_ffdi.py#L44-L266
