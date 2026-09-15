# Recursive fire indices never bridge missing days by default

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
touches the end of the input poisons the final state the same way. Leading
missing days do not poison: the recurrence begins at the first valid day with
the seed or the supplied `initial_state`, just as if the input started there.
If every input day is missing, the output is all NaN and the returned state
is the initial state.

**bridge.** Interior and trailing missing runs no longer than `max_gap_days`
are skipped: their outputs are NaN, their state is "no change", and the next
valid day resumes from the last valid state. The first missing run longer
than `max_gap_days` poisons: outputs from the run's first missing day onward
are NaN and the state from that point is NaN. A trailing run within the limit
leaves `return_state` at the last valid state, so a caller can explicitly
append across it. `bridge` never invents weather values.

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
and `initial_state` do not change the policy: the same rule applies across
the whole input, including days omitted from the output.

## Consequences

KBDI (#799) and the CFFWIS moisture codes (#803) implement the identical
signature, validation, and resume semantics, and each carries the
parametrized gap matrix: all-NaN input; leading and trailing blocks; interior
single-day gaps; interior and trailing runs of exactly `max_gap_days` and
`max_gap_days + 1`; a bitwise check that a bridged run equals running the
recurrence over the valid days alone; and `return_state` after a trailing
gap, NaN under `propagate` and the last valid state under a bridged `bridge`
run. Xarray adapters forward both arguments so the policy applies per cell
(#801, #807).
