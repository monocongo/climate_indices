# The xarray seam owns calendar conversion, and requires Gregorian January-origin input

The NumPy index implementations group monthly values into 12 positional slots per year and daily
values into 366, treating every year as a leap year and every series as beginning in January.
`xarray_adapter.py` converts Gregorian daily input to that all-leap calendar before computation and
restores Gregorian positions afterward — synthesizing February 29 in non-leap years as the mean of
February 28 and March 1, the same conversion the CLI already performed with
`utils.transform_to_366day()` and `utils.transform_to_gregorian()`. Input is therefore restricted to
`standard`, `gregorian`, or `proleptic_gregorian` `datetime64` coordinates beginning in January
(January 1 for daily), and anything else raises `CoordinateValidationError`. A caller who hands the
xarray API an ordinary Gregorian daily series would reasonably expect the labeled coordinates to be
honored rather than reinterpreted positionally, and before this change they were not: results were
silently shifted for every calendar day after February 28 of a non-leap year, then relabeled with
the original coordinates, which hid the drift. Owning conversion at the seam keeps the NumPy
implementations untouched, as [ADR-0001](./0001-dual-numpy-xarray-api.md) requires, and leaves the
separate execution paths of [ADR-0002](./0002-multiprocessing-cli-dask-xarray.md) intact.

Rejecting rather than normalizing is deliberate. A partial *final* year is supported, because the
positional layout of every preceding year is still determined. A partial *first* year is not, because
padding one would require guessing the caller's intent about where the calendar begins, and guessing
wrong reintroduces exactly the silent drift this decision removes. `cftime` calendars are rejected
outright rather than approximated, since `noleap`, `360_day`, and `all_leap` have no faithful mapping
onto the 366-day layout.

## Consequences

Callers must pass Gregorian datetime coordinates starting in January; monthly series beginning in
another month, daily series beginning after January 1, and cftime-backed coordinates now raise
instead of returning quietly wrong numbers. Daily results from the xarray API changed numerically for
any series containing a non-leap year — they were previously drifted and are now calendar-aligned.
Validation costs one `xr.infer_freq` call per invocation, which is fixed per call and negligible for
gridded workloads but measurable against the cheapest NumPy paths (see
`_PET_THORNTHWAITE_OVERHEAD_THRESHOLD` in `tests/test_benchmark_overhead.py`).

The policy lives in one place — `_build_daily_calendar_plan()` — and is applied both by the
`@xarray_adapter` decorator (SPI, SPEI, EDDI, PNP) and by the hand-written `pet_thornthwaite` and
`pet_hargreaves` paths. `pci()` is not covered: it is a manual single-year wrapper requiring exactly
365 or 366 values, and its contract needs a separate decision. Palmer has no direct xarray API.
