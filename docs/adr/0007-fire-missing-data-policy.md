# Recursive fire indices never bridge missing days by default

## Status

Amended: the state contract this record describes is the scalar one. The stateful
fire implementations also carry per-cell state for spatial input, where
`trailing_gap_days` is an `int64` array whose `-1` entries mark cells whose
recurrence has not started (`fire/_cffwis.py`, `fire/_kbdi.py`; #799–#807). The
policy itself is unchanged.

A daily recurrence turns every missing observation into a fork in the state:
treat the day as no change, poison the rest of the series, or invent an
interpolated value. The package's existing xarray contract already propagates
NaN inputs to NaN outputs, and station data is gappy, so the fire family needs
one explicit rule before KBDI and CFFWIS are public. This also interacts with
the seasonal start-up and overwintering work in
[#806](https://github.com/monocongo/climate_indices/issues/806), which must
not be encoded as missing data.

## Decision

Every stateful NumPy fire API takes keyword-only
`nan_policy: Literal["propagate", "bridge"] = "propagate"` and
`max_gap_days: int = 0`. Validation: `nan_policy="bridge"` requires
`max_gap_days >= 1`; `nan_policy="propagate"` requires the default
`max_gap_days=0`. Anything else raises `InvalidArgumentError`. A missing day
is a day where any time-varying weather input is NaN.

**propagate (default).** A missing day always yields a NaN output and never
updates state. The first valid day after an interior missing run resumes with
a NaN state, so every output from that day onward is NaN. This is the
conservative default: no state is fabricated across a gap. A missing run that
touches the end of a started recurrence poisons the final state the same way.
Leading missing days never poison before the recurrence has started: the
recurrence begins at the first valid day with the seed, just as if the input
started there. Once a state reports a started recurrence, a leading missing
run is interior to the series and poisons. If every input day is missing, the
output is all NaN; the returned state is the seed or a supplied not-started
`initial_state`, while a supplied started state poisons.

**bridge.** Interior and trailing missing runs no longer than `max_gap_days`
are skipped: their outputs are NaN, their state is "no change", and the next
valid day resumes from the last valid state. The first missing run longer
than `max_gap_days` poisons: its days are NaN outputs and the state turns NaN
as soon as the run passes `max_gap_days`, so every later output is NaN. A
trailing run within the limit leaves `return_state` at the last valid state,
so a caller can explicitly append across it. `bridge` never invents weather
values.

The limit applies to the continuous series, not to one call. Each per-index
state dataclass ([ADR-0006](./0006-fire-recursive-state-and-execution.md))
therefore carries `trailing_gap_days: int | None`: the number of missing days
immediately before the return point, `0` when the last input day was valid,
and `None` while no valid day has started the recurrence. A resumed call
measures its leading missing run against the state: `None` means the run is
still pre-start, so it is unbounded and never poisons; otherwise a run that
pushes `trailing_gap_days + run_length` past `max_gap_days` poisons, exactly
as the one-shot series would. The count keeps accumulating while no valid day
resumes the recurrence, so an all-missing continuation of a started state
still poisons once it exceeds the allowance, while an all-missing
continuation of a not-started state stays `None`. A call without
`initial_state` has no run to continue, so its leading missing days are
unbounded and never poison: the recurrence simply starts at the first valid
day.

Interpolation is not a policy. Callers that want filled weather data must do
it upstream, where the fill is explicit, testable, and visible in the input
series. This matches the no-hidden-materialization stance of the Dask
time-chunk rule ([ADR-0003](./0003-dask-time-dimension-single-chunk.md)) and
the state contract
([ADR-0006](./0006-fire-recursive-state-and-execution.md)).

Sub-freezing days and other physically valid but inactive days defined by an
index, such as the CFFWIS temperature floors and no-accumulation rules, are
observations, not missing data. Gap policy never overrides index-specific
handling. Off-season periods are also not missing data: seasonal shutdown and
overwintering (#806) define their own explicit state carry, and encoding the
off-season as NaN would trigger the gap policy instead.

Static inputs are not gap-managed. A NaN mean annual precipitation or
latitude means the affected cell has no valid recurrence; per-index
validation decides whether that raises or produces an all-NaN cell. `spin_up`
does not change the policy: the same rule applies across the whole input,
including days omitted from the output. `initial_state` carries the boundary
bookkeeping described above but never changes the rule itself.

## Consequences

KBDI (#799) and the CFFWIS moisture codes (#803) implement the identical
signature, validation, and resume semantics, and each carries the
parametrized gap matrix: all-NaN input; leading and trailing blocks; interior
single-day gaps; interior and trailing runs of exactly `max_gap_days` and
`max_gap_days + 1`; a bitwise check that a bridged run equals running the
recurrence over the valid days alone; a bridged run split across an append
boundary, resumed from the mid-run state, bitwise-equals the one-shot run,
including a run that exceeds the limit only once the pieces are joined; and
`return_state` after a trailing gap, NaN under `propagate` and the last valid
state plus its `trailing_gap_days` count under a bridged `bridge` run; and the
all-NaN call, all-NaN output with the returned state left not started
(`trailing_gap_days is None`) unless the supplied state was already started,
which poisons. Xarray adapters forward both arguments so the policy applies
per cell (#801, #807).
