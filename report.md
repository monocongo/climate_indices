## Resolutions

- id: `4076003087`
  - anchor: https://github.com/monocongo/climate_indices/pull/1119#discussion_r4076003087
  - reviewer: `coderabbitai[bot]`
  - finding: Bound the 1.3 s Dask-vs-eager gap to what the benchmark actually measures — the full process-scheduler call, not isolated process start-up — in `benchmarks/README.md#L473-474`, `docs/performance.md#L157-158`, and the generated `llms-full.txt#L3515-3516`.
  - resolution: `SKIPPED` (already fixed on branch)
  - commit: `null` (pre-existing fix; no new commit made this run)
  - evidence: `benchmarks/README.md:474-476` reads "that difference is the measured total process-scheduler overhead (pool start-up, scheduling, serialization, result transfer), not pool start-up alone"; `docs/performance.md:157-158` reads "Interpretation: that 1.32 s is the total process-scheduler overhead"; `llms-full.txt:3515-3518` carries the identical regenerated text. All three were updated together in commit `aa73f0b1b267407b74b75c37e662ba2274b47cb9` ("docs(benchmarks): report the Dask overhead as measured, not as pool start-up"), already on `perf/1097-spi-dask-benchmark` and pushed to `origin` before this run started.
  - validation: N/A — no diff produced this run.
  - thread_resolved: `true`

## Skipped

None (the sole actionable finding is covered above as already-fixed).

## Validation Summary

- commands: none run — no code/doc changes made this run.
- outcome: `pass` (nothing to validate; pre-existing fix already lives on the pushed branch)
- notes: `validation.yaml` was not found at the repo root, but no validation was needed since this run made no changes.
