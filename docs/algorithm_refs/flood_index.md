# Flood Index (I_F)

`flood.flood_index(pe, data_start_year, calibration_year_initial,
calibration_year_final, year_start_month=...)` describes **flood potential, not
flooding**. It accepts effective precipitation (PE) computed separately with
`flood.effective_precipitation()` and returns
`I_F = (PE − mean(PE_max)) / SD(PE_max)`. The mean and population standard
deviation (`ddof=0`) are calculated from annual maxima of PE for complete
annual periods whose **start years** fall within the inclusive Calibration
Period. Each period starts on the first day of the caller-supplied
`year_start_month`. Partial leading or trailing periods do not contribute
maxima; a period with no finite PE has no maximum. At least two finite maxima
and nonzero variance are required per cell. Missing PE remains missing.

Daily PE begins on January 1 in a 366-day all-leap positional calendar.
Convert Gregorian precipitation to that layout before computing PE;
`utils.transform_to_366day` or `DailyCalendarPlan.to_all_leap` fills
non-leap February 29. No hydrological-year start is inferred.

The PE kernel is the harmonic double sum of Byun and Wilhite (1999), Eq. (2),
selected over their Eq. (3) by [ADR-0014](../adr/0014-flood-family-scientific-conventions.md).
The Deo et al. (2015, 2019) abstracts describe an exponentially decaying PE;
whether the 2015 implementation uses that kernel remains unverified. The
exponential form and flood-event detection helpers are **not implemented**.
Moishin et al. (2021), co-authored with Deo, supplies an openly licensed
same-lineage statement of the double sum, not a numeric oracle for this API.

`tests/test_flood_index.py` checks the declared normalization and year-boundary
contract; no numeric external oracle is available yet (see
`tests/fixture/flood/README.md`). Only the NumPy route is implemented;
xarray/CF integration follows in [#1108](https://github.com/monocongo/climate_indices/issues/1108).

## References

- Deo, R. C., Byun, H.-R., Adamowski, J. F., and Kim, D.-W. (2015). A real-time flood monitoring index based on daily effective precipitation and its application to Brisbane and Lockyer Valley flood events. *Water Resources Management*, 29, 4075–4093. https://doi.org/10.1007/s11269-015-1046-3
- Byun, H.-R., and Wilhite, D. A. (1999). Objective quantification of drought severity and duration. *Journal of Climate*, 12, 2747–2756.
- Moishin, M., Deo, R. C., Prasad, R., Raj, N., and Abdulla, S. (2021). *IEEE Access* 9 (CC BY 4.0).
