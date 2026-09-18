# Palmer adopts the spatial-block contract; scPDSI stays per-location

ADR-0001 named Palmer the documented exception to the modern xarray API and required
a separate, explicit decision before it could grow one: "Adding xarray support for
Palmer indices requires a separate, explicit architecture decision rather than
unsupported wiring through `indices.py` or `xarray_adapter.py`." This is that decision
for `palmer.pdsi()`'s NumPy layer; the xarray adapter registration itself
(`spatial_kernel=True`, the no-loop guard test) was left to a follow-up ticket, so this
ADR grants the permission without spending it. That registration landed in #1016:
`climate_indices.pdsi()` owns its `xr.apply_ufunc` call and returns a Dataset of the
four indices.

## Decision

`palmer.pdsi()` adopts ADR-0009's time-major block contract: a three-or-more-dimensional
`precips`/`pet` is read as `(time, *cells)`, ambiguous shapes (a first cell axis of 12 or
366) must declare `spatial_time_major=True`, and `awc` accepts a scalar or an array
broadcastable to the cell shape. Internally every input -- a single location included --
carries a trailing cell axis (`n_cells == 1` for the legacy 1-D/2-D contract), so the
whole recursion has one code path rather than a duplicated scalar and vectorized pair.

`palmer.scpdsi()` does **not** grow this contract and explicitly rejects a 3+-D
`precips`/`pet` (`ValueError`, naming ADR-0011). scPDSI stays on the per-location path:

- `_calculate_scpdsi_prepared` runs a full Wells backtracking recursion
  (`_palmer_wells.calculate`) **four times** per location -- once raw, three more after
  percentile rescaling -- not once.
- Each of those runs is preceded by a duration-factor fit
  (`self_calibration.duration_factors`) that itself makes twenty rolling-window passes
  over the calibration Z-index (ten window lengths, wet and dry sides) plus a
  correlation-adaptive least-squares fit with a trailing-point-dropping loop. None of
  this is bulk array arithmetic; it is per-location control flow at least as intricate
  as the PDSI spell recursion, times four.
- `_palmer_wells.calculate` raises `ConvergenceError` on a non-contracting duration
  factor or a zero denominator. In a blocked kernel, one cell's `ConvergenceError` would
  fail the whole block; PDSI's masked recursion has no such per-cell exception path to
  guard against, because it has no exception-raising stage in the hot loop at all.

Vectorizing scPDSI would mean re-deriving all of the above as a masked, per-cell state
machine -- roughly the effort spent on this ticket again, for an index with a smaller
user base than standard PDSI and no existing performance complaint driving it. Given
that, and the `ConvergenceError` problem specifically, scPDSI's parallelism stays at the
block level: multiple locations run concurrently (CLI multiprocessing per ADR-0002, or a
future per-location `apply_ufunc` with `vectorize=True`), not within one location's
recursion.

## Vectorizing the PDSI recursion

The standard PDSI recursion has two halves. The water balance, CAFEC coefficients, and
Z-index are pure per-month arithmetic with only elementwise branches (`min`/`max`,
threshold comparisons) -- these convert to `np.where` mechanically, with cells as a
trailing broadcast axis. The spell recursion (`_statement_170`-`_statement_220`,
`_assign`, the K8 backtracking window) is genuine per-cell control flow: which
statement runs, and for how many months a decision defers, differs by cell. That half
is rewritten so every statement function takes an `active: np.ndarray` mask and writes
only where it holds, with the K8 window preallocated to the record length (removing the
scalar recursion's runtime `np.append` growth) instead of the historical `K8_SIZE = 40`
bound.

Python's `min`/`max` builtins have NaN-comparison semantics that `np.minimum`/
`np.maximum` do not reproduce (`np.maximum` propagates NaN from either operand; Python's
`max(a, b)` returns `b` only if `b > a`, so a NaN in the first position loses
regardless). The recursion calls the builtins with data-dependent NaN in either
position, so a straight `np.maximum`/`np.minimum` substitution would not have been
bit-for-bit; `_py_max`/`_py_min` replicate the exact per-call argument order instead.

## Equivalence and the K-factor ULP

Equivalence against the per-location path is bit-for-bit (`assert_array_equal`), not a
tolerance: the operation order is unchanged, only scalar arithmetic becoming array
arithmetic, which is bit-identical under IEEE 754. One stage needed a specific fix to
reach that bar. `_calc_kfactors`'s `swtd` normalizer sums twelve months' weighted
departures; reducing that sum along axis 0 of a `(12, n_cells)` array uses NumPy's
strided-reduction path, which can associate differently -- by up to 1 ULP -- from the
pairwise summation NumPy uses for a standalone contiguous `(12,)` array (the
per-location shape). The spell recursion branches on exact float comparisons
(`x3 == 0`, `px3 == 0`), so that ULP was observed to cascade into a fully different
sequence of branch decisions for roughly 15% of a 48-division stress grid.
`_calc_kfactors` reduces a C-contiguous `(n_cells, 12)` transpose instead, putting the
sum back on NumPy's fast, contiguous axis and reproducing the per-location value
exactly; this was verified empirically against the divergent cases, not assumed.

## Memory footprint

The vectorized recursion trades the per-location path's small per-cell working set for
block-sized state: the spell window (`indexj`/`indexm`/`sx`/`sx1`/`sx2`/`sx3`) is
preallocated to the record length for every cell, and the prepared, recursion, and output
arrays all carry `(years, 12, cells)`. Peak memory is therefore a function of
`record_months x cells` -- a few tens of float64 buffers, not one -- so a large enough
legal block can exhaust memory before producing output. As with the fitting-based kernels
(ADR-0009), the block size is the memory lever: a direct caller chunks spatially rather
than handing `pdsi()` a dense continental grid, and the CLI splits the grid along latitude
across one process per worker, so per-worker memory is the chunk's share of the grid (the
sum across workers stays proportional to the grid).

## A block cell that is entirely missing

A standalone call on an all-missing series short-circuits before the recursion runs and
returns NaN for all four outputs. Inside a block that is not entirely missing, that
shortcut never fires for one merely-all-missing cell: the recursion runs, and
`_statement_200`'s `max(0, ...)`/`min(0.0, ...)` calls -- reproduced exactly per the NaN
semantics above -- turn a NaN Z-index into `0`, not NaN. Left unmasked, a real grid's
ocean or no-data cells would read back as an ordinary near-zero PDSI instead of missing
data. `palmer.pdsi()` therefore detects per-cell full missingness in a block
(`np.all(np.isnan(precips), axis=0)`) after the recursion completes and overwrites just
those cells' four outputs with NaN, reproducing the standalone shortcut's result without
touching any other cell. Masked input is converted once at the shared calculation entry
(`_fill_masked_with_nan`), so a masked element is the same missing marker as a NaN and the
backing value under a mask is never read or published.

## Consequences

- `docs/adr/0009-spatial-block-declaration.md`'s closing sentence, "Palmer has no
  adapter layer at all (#937)", is now stale for the NumPy layer; an xarray adapter
  registration landed in #1016 (`climate_indices.pdsi()`).
- The stale `# TODO(v2.5.0): implement palmer_xarray() wrapper using Pattern C` at the
  end of `palmer.py` is removed: "Pattern C" was defined nowhere in the repository and
  is superseded by the ADR-0009 contract this ADR adopts.
- `scpdsi()`'s signature intentionally diverges from `pdsi()` by exactly one parameter
  (`spatial_time_major`); `tests/test_scpdsi.py::test_public_signature_matches_pdsi_and_is_exported`
  pins that as the only allowed difference.
