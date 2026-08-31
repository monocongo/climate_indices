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
outright rather than approximated. `noleap` and `360_day` have no faithful mapping onto the 366-day
layout. `all_leap` matches that layout exactly, but is rejected for a different reason: the interface
accepts only `datetime64` coordinates, and cftime cannot supply one.

## Consequences

Callers must pass Gregorian datetime coordinates starting in January; monthly series beginning in
another month, daily series beginning after January 1, and cftime-backed coordinates now raise
instead of returning quietly wrong numbers. Daily results from the xarray API changed numerically for
any series containing a non-leap year — they were previously drifted and are now calendar-aligned.
Validation costs one periodicity check per invocation. `xr.infer_freq` proved too expensive for that
— it dominated the check at roughly 97% of its cost, and because it scales with series length the
overhead did not amortize as a 1-D series grew. `_match_supported_periodicity()` therefore recognizes
the two supported layouts with vectorized `datetime64` arithmetic, falling back to `xr.infer_freq`
only for input it does not match, where the exact frequency string is still wanted for the error
message. That keeps the check fixed per call and negligible for gridded workloads, and it is why
`pet_thornthwaite` needs no overhead budget of its own in `tests/test_benchmark_overhead.py`. The
looser budget that remains on `pet_hargreaves` predates this decision and covers unrelated adapter
cost — `xr.align` on the tmin/tmax pair — tracked in issue #740.

The policy lives in one place — `_build_daily_calendar_plan()` — and is applied both by the
`@xarray_adapter` decorator (SPI, SPEI, EDDI, PNP) and by the hand-written `pet_thornthwaite` and
`pet_hargreaves` paths.

Two paths remain uncovered, deliberately. `@xarray_adapter(infer_params=False)` disables coordinate
inference entirely, and with it the calendar contract, so a function decorated that way and given an
explicit `Periodicity.daily` still computes positionally against Gregorian input. That escape hatch
exists so a caller can opt out of every inference this adapter performs, and no shipped wrapper uses
it — but it is the one remaining way to get silently drifted output, and anything adopting it takes
on the conversion itself. `pci()` is also uncovered: it is a manual single-year wrapper requiring
exactly 365 or 366 values, and its contract needs a separate decision. Palmer has no direct xarray
API.
