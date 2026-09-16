# Seasonal carry is an explicit in-season mask

The Drought Code is the only CFFWIS moisture code carried across the winter:
the FFMC and DMC are assumed to reach saturation from overwinter precipitation,
while the DC's long response time means a wrong spring start-up biases a large
part of the season. Operational practice shuts the DC down at the end of the
fire season and restarts it in the spring from the overwintering equation
(Lawson and Armitage, 2008). This package has no calendar, no snow input, and
no region-independent threshold for when a fire season begins or ends, so the
season boundary is a caller policy rather than a property of the weather data.
This decision is the seasonal half of
[ADR-0007](./0007-fire-missing-data-policy.md), which requires off-season
periods to carry state explicitly instead of being encoded as missing days.

## Decision

`drought_code()` takes the keyword-only boolean `in_season` mask, time-first
and broadcast against the weather inputs under the same left-aligned rule.
`None` — the default — treats every day as in-season and leaves the recurrence
unchanged, so overwintering is opt-in.

An off-season day is neither an observation nor a missing day. The recurrence
state is frozen, the day never counts against `max_gap_days`, and it can
neither poison nor bridge a gap. Off-season weather has no effect on the code.
The output for such a day is the carried DC, not NaN, so NaN keeps its
ADR-0007 meaning of missing or poisoned and a frozen day stays distinguishable
from a gap. A cell whose recurrence has not started has no carried value and
stays NaN, exactly as any day before its first valid observation does.

The mask is the caller's rule: a fixed date window, a temperature threshold,
a snow-cover rule, or anything else the caller can compute. The library does
not infer season boundaries and does not ship a season detector. Inference
would need inputs the DC recurrence does not take, a calendar it does not
carry, and a threshold policy that is region-specific; the NRCan `fire_season`
WF93/LA08 methods belong with the inputs they require, not buried in a
moisture-code kernel.

Start-up is separate from the recurrence. `overwinter_drought_code()` is a
pure function of the final autumn DC and the overwinter precipitation total,
implementing Lawson and Armitage's Eqs. 2-4 with the carry-over fraction and
wetting efficiency as keyword-only tunables whose defaults are the NRCan
reference values (0.75 and 0.75). The caller chains seasons by passing its
result back as `initial_dc` (or through the returned state's `dc`), which is
the append contract of
[ADR-0006](./0006-fire-recursive-state-and-execution.md).

Rejected alternatives: a fixed-date season column, which hides a
region-specific policy in the library and cannot express a hemisphere-agnostic
or snow-driven season; a built-in temperature-threshold detector (WF93), which
needs a maximum-temperature series and a threshold policy the DC recurrence
has no use for; and NaN off-season outputs, which collide with ADR-0007's
missing-day meaning.

## Consequences

A continuous call with `in_season` freezes the DC over the off-season but does
not apply the overwintering equation, so the next season resumes from the
autumn DC. That is the correct carry for a series that simply should not
respond to off-season weather; it is not overwintering, and callers who want
the published spring start-up must chain seasons through
`overwinter_drought_code()`.

`trailing_gap_days` is carried across the off-season rather than reset: the
count is the missing days immediately before the return point, and an
off-season day neither closes nor extends the run. A missing last in-season
day followed by a missing first day of the next season therefore counts as a
two-day run, whether the off-season is spanned by one masked call or by a
resumed call.

The `cffwis()` orchestrator does not take the mask yet: seasonal carry is
implemented for the DC alone, and threading an off-season policy through the
shared three-code loop is separate work. The overwintering tunables are
scalars, not spatial fields, so per-cell fractions await a caller that needs
them.
