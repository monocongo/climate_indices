# Fire APIs are namespaced in the `fire` package

## Status

Amended: the flat-module deferral recorded in the Decision section is complete.
`cec29737` ("refactor(fire): promote the flat module to a fire package")
promoted `fire.py` to `src/climate_indices/fire/`, so the trigger is met and
nothing remains deferred. The naming decision itself — `from climate_indices
import fire`, `fire.cffwis_fwi()`, no unqualified `typed_public_api.py` entries —
is unchanged.

[ADR-0001](./0001-dual-numpy-xarray-api.md) places new non-Palmer calculations
in `compute.py` and exposes them through the top-level typed/xarray API. Fire
weather has a distinct, growing family of elementwise and recursive daily
indices, including a multi-output CFFWIS system. Putting it in those
drought-oriented modules would create ambiguous top-level names such as FWI
and couple unrelated API evolution.

## Decision

Fire computations live in `climate_indices.fire`, a stable NumPy subpackage. It is
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
`fire.py` passed that line count with the CFFWIS moisture codes (#803).
Promotion was deferred to a dedicated mechanical refactor tracked against the
remaining CFFWIS work (#804) and landed in `cec29737`: the flat module was a
deliberate, recorded deferral rather than a silent departure from the trigger
above, and that deferral is now complete.

## Consequences

This amends ADR-0001 only for the fire family. Fire’s naming, unit, state, and
error contracts are recorded in the internal fire subsystem design note
(`docs/design/fire-subsystem.md`), which stays in the repository rather than
this site because it also plans unimplemented fire indices.
No fire CLI or fire-specific exception hierarchy is introduced by this
decision.
