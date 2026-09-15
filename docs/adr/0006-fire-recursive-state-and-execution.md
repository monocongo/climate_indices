# Recursive fire indices use explicit state and a pure-NumPy baseline

KBDI and the CFFWIS moisture codes are daily nonlinear recurrences. Unlike the
existing weather-only Fosberg calculation, they must retain both the published
code seed and auxiliary bookkeeping across an append boundary. They also need
all prior days for every spatial cell, which makes arbitrary Dask time chunks
incorrect. These contracts will be hard to change after KBDI and CFFWIS are
public.

## Decision

Stateful fire APIs use time-first NumPy arrays. Each algorithm has its own
frozen state dataclass: `KBDIState`, `FFMCState`, `DMCState`, `DCState`, or
`CFFWISState`. It contains the output code plus every auxiliary recurrence
value needed to resume exactly, such as KBDI's cumulative wet-spell
precipitation. A generic tuple would conceal that required state; an xarray
`Dataset` would make the stable NumPy core depend on the beta xarray layer.

Each single-output API accepts keyword-only `initial_<code>: float | None`,
`initial_state`, `return_state=False`, and `spin_up=0`. `None` selects the
published seed (KBDI 0, FFMC 85, DMC 6, DC 15); an explicit `initial_state`
replaces the seed and cannot be combined with one. `return_state=True` returns
a named `{Index}Result(values, state)`. `CFFWISResult` retains its named
outputs and adds the state under the same flag. `spin_up` runs and omits that
many leading input days; no universal nonzero default is scientifically
justified, so users choose a study-specific transient length.

Implementations vectorize each daily step across spatial cells, loop only over
time, and copy the final state before returning it. Each implementation's
round-trip test must resume a run with the state returned at an intermediate
day and bitwise-compare the concatenation against one continuous run. The
tuple-based prototype engine that first demonstrated this contract was removed
as dead code in #850; #799 and #803 apply the same execution pattern to their
frozen state dataclasses.

NumPy is the required correct baseline. `numba` is not an optional dependency:
it adds a compiler/runtime support surface and the time loop is short relative
to each vectorized spatial update. Reconsider only after a representative
benchmark shows that a pure-NumPy spatial tile misses an agreed runtime target;
an accelerator must preserve this state and bitwise append contract.

Fire xarray adapters must call the existing shared
`xarray_adapter._validate_dask_chunks()` for every time-varying input and raise
`CoordinateValidationError` when `time` has multiple chunks. They do not
rechunk automatically: hidden rechunking can materialize an impractically large
daily history. Spatial chunks remain unrestricted.

## Consequences

KBDI (#799) and CFFWIS moisture-code work (#803) implement the documented
state classes and API signatures rather than inventing local conventions. The
CFFWIS xarray adapter (#807) inherits a clear, testable time-chunk error.
Benchmark measurements and the NumPy-versus-numba decision record are kept in
[#795](https://github.com/monocongo/climate_indices/issues/795).
