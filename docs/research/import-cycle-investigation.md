# Import-cycle investigation (#1238)

## Decision

No refactor required for the cycles reported in
[issue #1238](https://github.com/monocongo/climate_indices/issues/1238).
They combine typing-only imports, function-local imports, and package/submodule
resolution through `__init__.py`; none demonstrated a partially initialized
module failure. This is not a claim that the package imports only its NumPy
core: its public re-exports eagerly load fire, flood, and the xarray adapter.

Evidence snapshots:

- `4b5bde51bf516449d25afde4efe8cfb9911bce03`: the commit recorded in the
  graphify worktree's `graphify-out/graph.json`; its report lists 20 cycles.
- `198a2404`: `origin/main` when this investigation started. Subsequent module
  splits changed some edges, so both snapshots were checked independently.

Classification keys: **(a)** module-level runtime dependency; **(b)**
`TYPE_CHECKING`/annotation only; **(c)** function-local import; **(d)** package
re-export or submodule convenience import. A (d) import can execute at runtime;
it is not equivalent to (b) or (c).

## Classification of the reported chains

All paths below refer to the graph's `4b5bde51` snapshot under
`src/climate_indices/`.

| Reported chain (number of variants) | Classification and evidence |
| --- | --- |
| `compute -> indices -> eto -> compute` (1) | `compute -> indices` is **(b)** at `compute.py:29–32` and **(c)** inside `fit_diagnostics` at line 2910. `indices -> eto` and `eto -> compute` are runtime submodule imports **(a/d)**. The cycle is not traversed during module initialization. |
| `__init__ -> fire/__init__ -> fire/_hdw -> __init__` (1) | The first two edges are public re-exports **(d)**. `_hdw.py:12` imports the `pm_eto` submodule, not a late-bound public function or `__version__` from the partially initialized root. Mapping `from climate_indices import pm_eto` back to the root file creates a package-resolution cycle **(d)**, not a dependency on completion of the root's initialization. |
| `__init__ -> fire/__init__ -> fire/{_cffwis,_haines,_kbdi,_hdw} -> xarray_adapter -> __init__` (4) | Re-exports **(d)** lead to actual runtime `build_output_attrs` imports **(a)** (`_cffwis.py:39`, `_haines.py:18`, `_kbdi.py:43`, `_hdw.py:28`). The adapter's root import of `__version__` is function-local **(c)**, at lines 1153, 2007, 2301, and 2394. Its top-level `from climate_indices import compute, eto, indices, palmer, pm_eto, utils` resolves submodules **(a/d)**, not root re-exported callables. |
| `__init__ -> fire/__init__ -> fire/{_cffwis,_haines,_kbdi} -> xarray_adapter -> {compute,eto,indices,palmer} -> __init__` (12) | Adapter-to-core imports are real runtime edges **(a)**. The terminal imports are submodule convenience imports **(d)**: `compute.py:16` requests `lmoments, utils`; `eto.py:30` requests `compute, utils`; `indices.py:14` requests `compute, eto`; `palmer.py:11` requests `_palmer_wells, compute, self_calibration, utils`. None requests a late-bound root re-export. Resolving each of these to the root file rather than the named submodules produces the reported cycles. |
| `__init__ -> fire/__init__ -> fire/_hdw -> xarray_adapter -> {compute,eto} -> __init__` (2) | Same **(a/d)** classification and terminal imports as the preceding row. These are the remaining two five-file variants in the report. |

Relevant immutable source links:
[compute](https://github.com/monocongo/climate_indices/blob/4b5bde51/src/climate_indices/compute.py#L29-L32),
[HDW](https://github.com/monocongo/climate_indices/blob/4b5bde51/src/climate_indices/fire/_hdw.py#L12-L28),
[adapter imports](https://github.com/monocongo/climate_indices/blob/4b5bde51/src/climate_indices/xarray_adapter.py#L40),
[deferred version lookup](https://github.com/monocongo/climate_indices/blob/4b5bde51/src/climate_indices/xarray_adapter.py#L1151-L1155).

On `198a2404`, `compute -> indices` remains typing-only/function-local, and
`xarray_adapter -> __version__` remains function-local. The split CFFWIS code
adds `fire/_cffwis -> fire/_cffwis_xarray -> fire/_cffwis`; its forward edge is
function-local (`_cffwis.py:973`), so this is also **(c)**, not an import-time
cycle. `typed_public_api` likewise defers its root version lookup.

## Fire and core dependency boundaries

No inspected `fire/*` module needs a top-level import of the root's public
re-exports or version. HDW's `from climate_indices import pm_eto` is an actual
submodule dependency, safe while root initialization is in progress. Rewriting
that spelling would not change which modules load or fix an observed failure.

The dependency direction is adapter **to** core: `compute`, `eto`, `indices`,
and `palmer` have no direct imports of `xarray_adapter`. Numerical execution
can remain independent of adapter dispatch, but ordinary
`import climate_indices.compute` first executes the package initializer,
which eagerly imports the adapter through the fire/public API. Import-time
separation or making xarray optional would require a separately scoped public
API/startup change; it is not a circular-import repair justified by this issue.

## Extraction and runtime checks

A standard-library `ast` scan covered all Python modules in each snapshot
(36 at `4b5bde51`, 45 at `198a2404`). It separated imports under `TYPE_CHECKING`
from imports inside functions and module-level imports; for
`from climate_indices import name`, it resolved `name` to an existing
submodule when applicable. Tarjan strongly connected components over the
module-level edges found **zero multi-module SCCs** at both commits. Implicit
parent-package initialization is not represented by those direct edges; the
fresh-process checks below address that limitation. Source inspection also
checked that the deferred back edges are not invoked during initialization.

Twelve entry modules were imported in separate fresh Python processes per
snapshot: root, `compute`, `indices`, `eto`, `palmer`, `pm_eto`,
`xarray_adapter`, `fire`, and `fire.{_hdw,_cffwis,_haines,_kbdi}`. Each check
verified the imported file belonged to the intended source snapshot, module
initialization had completed, and public `spi`/`fire.hot_dry_windy` callables
and `__version__` were available. **24/24 passed**, using Python 3.14.7 and
the dependencies installed from `198a2404`'s `uv.lock`.

The import checks can be repeated from a dependency-synced checkout (set
`SRC` to an archived snapshot's `src` directory to check that snapshot):

```bash
SRC="$PWD/src"
for module in climate_indices climate_indices.compute climate_indices.indices \
  climate_indices.eto climate_indices.palmer climate_indices.pm_eto \
  climate_indices.xarray_adapter climate_indices.fire \
  climate_indices.fire._hdw climate_indices.fire._cffwis \
  climate_indices.fire._haines climate_indices.fire._kbdi; do
  PYTHONPATH="$SRC" uv run --no-sync python -c '
import importlib, os, sys
m = importlib.import_module(sys.argv[1])
assert m.__file__.startswith(os.environ["PYTHONPATH"] + os.sep), m.__file__
assert not getattr(m.__spec__, "_initializing", False)
import climate_indices as c
assert isinstance(c.__version__, str)
assert callable(c.spi) and callable(c.fire.hot_dry_windy)
' "$module" || exit 1
done
PYTHONPATH="$SRC" uv run --no-sync python -X importtime \
  -c 'import climate_indices.compute' 2> importtime.log
```

`-X importtime` was run for both snapshots. Both traces show the eager
`climate_indices.fire`/`climate_indices.xarray_adapter` initialization, without
an import error. Timing itself is not evidence of acyclicity; the classified
source edges and successful fresh-process imports provide that evidence.

Untouched `198a2404` baseline:

- `uv run --no-sync pytest -n 4`: **3,374 passed**, 166 runtime warnings;
  default marker selection excludes benchmark and validation tests.
- `uv run --no-sync ruff check src/ tests/`: clean.
- `uv run --no-sync ruff format --check src/ tests/`: 143 files already formatted.
- `uv run --no-sync mypy src/ tests/test_type_checking.py`: no issues in 46 files.

No imports removed, no numerical behavior changed, and no follow-up refactor
needed for the reported cycles. Evidence is snapshot-specific, not a guarantee
against future import-order regressions.
