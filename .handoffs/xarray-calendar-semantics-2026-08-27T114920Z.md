# Handoff: xarray calendar semantics

## Objective

Continue the top architecture recommendation on a dedicated implementation branch: make calendar semantics explicit at the xarray seam, with an eventual pull request after the architecture findings PR is merged or used as the stacked base.

## Completed

- Created the clean worktree and `chore/architecture-cleanup` branch from `origin/main`.
- Ran the `improve-codebase-architecture` exploration against the domain context and all ADRs.
- Published the durable findings in [`docs/architecture-deepening-review-2026-08-27.md`](../docs/architecture-deepening-review-2026-08-27.md).
- Opened [PR #753](https://github.com/monocongo/climate_indices/pull/753) for the findings.
- Selected “Make calendar semantics explicit at the xarray seam” as the implementation focus.

Do not duplicate the findings in this handoff; read the review document for evidence, candidate details, and secondary observations.

## Branch strategy

- Findings PR branch: `chore/architecture-cleanup`
- Implementation branch: `fix/xarray-calendar-semantics`
- The implementation branch is intentionally created from the findings branch so this handoff and review remain available.
- After PR #753 merges, rebase the implementation branch onto updated `origin/main` before opening its pull request. If work must be reviewed earlier, use PR #753 as the temporary stacked base.
- Do not modify the separate original worktree, which contains unrelated uncommitted work.

## Required reading

1. [`docs/architecture-deepening-review-2026-08-27.md`](../docs/architecture-deepening-review-2026-08-27.md), especially candidate 1.
2. [`src/climate_indices/CONTEXT.md`](../src/climate_indices/CONTEXT.md).
3. [`docs/adr/0001-dual-numpy-xarray-api.md`](../docs/adr/0001-dual-numpy-xarray-api.md).
4. [`docs/adr/0002-multiprocessing-cli-dask-xarray.md`](../docs/adr/0002-multiprocessing-cli-dask-xarray.md).
5. [`docs/adr/0003-dask-time-dimension-single-chunk.md`](../docs/adr/0003-dask-time-dimension-single-chunk.md).
6. Relevant implementations in `xarray_adapter.py`, `utils.py`, `indices.py`, and `eto.py`.

## Current technical state

- `_infer_periodicity()` accepts ordinary daily Gregorian coordinates.
- Generic xarray adaptation then sends raw values to NumPy implementations.
- Daily NumPy computation groups values into fixed 366-day positional years, so non-leap years shift later calendar positions.
- Monthly NumPy computation assumes the first value is January, while the xarray seam currently accepts regular monthly coordinates beginning in another month.
- The CLI already uses `utils.transform_to_366day()` before computation and `utils.transform_to_gregorian()` afterward.
- `transform_to_366day()` assumes a January 1 start and full years except a possible partial final year; for non-leap years it synthesizes February 29 as the mean of February 28 and March 1.
- The xarray result restores original coordinates after computation, which can hide positional drift.
- No source implementation has been changed yet.

## Design questions to resolve in the grilling loop

Do not propose or implement an interface until these choices are explicit:

1. Is the first implementation limited to standard/proleptic Gregorian datetime coordinates, or must it support cftime calendars now?
2. Should monthly input beginning after January be rejected or internally padded to a January origin and restored afterward?
3. Should daily input beginning after January 1 be rejected or normalized as a partial first year?
4. Must partial final years remain supported exactly as today?
5. Should the xarray module reuse the current NumPy calendar transformations or own coordinate-aware transformations that remain lazy for Dask?
6. How will synthetic February 29 values interact with missing-data propagation and Timescale windows?
7. Which public computations enter the first vertical slice: SPI only, all generic index adapters, or generic indices plus both PET methods?

## Suggested first vertical slice

After grilling, start with a daily SPI regression through the public xarray interface:

- Use at least one non-leap year followed by a leap year.
- Compare against explicit `transform_to_366day()` → stable NumPy SPI → `transform_to_gregorian()` behavior.
- Assert original time coordinates and dimension order are preserved.
- Repeat for eager and Dask-backed inputs while retaining the one-time-chunk invariant.
- Add monthly non-January-origin behavior as a separate test once reject-versus-normalize is decided.

Then generalize only through the shared xarray seam so SPEI, EDDI, and PNP gain leverage without editing each caller. Treat PET as a follow-up slice if its calendar needs differ.

## Validation gate

For source or test changes run:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
uv run pytest
```

Run focused tests throughout development, but complete the full gate before the eventual pull request.

## Suggested skills

- **improve-codebase-architecture** — continue its required grilling loop and preserve the module/interface/seam vocabulary.
- **tdd** — establish daily Gregorian and monthly-origin regressions before changing adaptation.
- **diagnose** — minimize and instrument any mismatch between explicit CLI-style conversion and xarray results.
- **code-review** — review the final branch against repository standards, ADRs, and the selected recommendation before opening the eventual pull request.
