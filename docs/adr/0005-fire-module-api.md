# Fire APIs are namespaced in `fire.py`

[ADR-0001](./0001-dual-numpy-xarray-api.md) places new non-Palmer calculations
in `compute.py` and exposes them through the top-level typed/xarray API. Fire
weather has a distinct, growing family of elementwise and recursive daily
indices, including a multi-output CFFWIS system. Putting it in those
drought-oriented modules would create ambiguous top-level names such as FWI
and couple unrelated API evolution.

## Decision

Fire computations live in `climate_indices.fire`, a stable NumPy module. It is
publicly imported as `from climate_indices import fire`; its functions are not
re-exported from the package root. The Canadian FWI component is
`fire.cffwis_fwi()`, never `fire.fwi()`, and Fosberg remains
`fire.fosberg_ffwi()`.

Beta xarray paths retain the same `fire.<name>` route, use the established
xarray adapter and CF metadata registry, and do not add unqualified entries to
`typed_public_api.py`. CFFWIS returns named outputs: a result object for NumPy
and an `xarray.Dataset` for xarray.

Keep `fire.py` flat until it exceeds roughly 1,500 lines or CFFWIS state
handling needs separated modules. Promotion to a `fire` package must preserve
the `from climate_indices import fire` import and every public function name.
`fire.py` passed that line count with the CFFWIS moisture codes (#803);
promotion is deferred to a dedicated mechanical refactor tracked against the
remaining CFFWIS work (#804), so the flat module is a deliberate, recorded
deferral rather than a silent departure from the trigger above.

## Consequences

This amends ADR-0001 only for the fire family. Fire’s naming, unit, state, and
error contracts are recorded in [the fire subsystem design](https://github.com/monocongo/climate_indices/blob/main/docs/design/fire-subsystem.md).
No fire CLI or fire-specific exception hierarchy is introduced by this
decision.
