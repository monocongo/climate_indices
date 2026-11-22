# Optional numerical replay of Palmer case E

Run this from a checkout of `380aa01f07d37572cbf6b8d5e1000b9d50382e03` with Git and uv available. Use a separate worktree if you already have a checkout in use.

```sh
git worktree add --detach ../climate-indices-palmer-1631-replay 380aa01f07d37572cbf6b8d5e1000b9d50382e03
cd ../climate-indices-palmer-1631-replay
uv run pytest 'tests/test_palmer_recursion_divergence.py::test_reduced_sequence_pins_both_recursions[E]'
```

Expect **1 passed**. A difference greater than `1e-9` occurs only at zero-based index 9, where the test checks standard PDSI near `-0.567203` and Wells-lineage PDSI at `0.0`, with absolute tolerance `1e-8`. The complete arrays and signed-zero input are stored under key `E`. [Test](https://github.com/monocongo/climate_indices/blob/380aa01f/tests/test_palmer_recursion_divergence.py#L204-L224) · [Committed sequence](https://github.com/monocongo/climate_indices/blob/380aa01f/docs/research/reference-papers/palmer-recursion-divergence-sequences.json#L213-L258)

The packet preparation reran the targeted test at the pin and checked both complete calculated arrays against the stored values. The state trace and threshold-only comparison reused the study’s tracer and in-memory `E` variant. This is an internal numerical check. A reviewer can judge the source interpretation from the evidence sheet alone.
