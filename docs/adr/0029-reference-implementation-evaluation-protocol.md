# Reference-implementation evaluation protocol is versioned and studies are pre-registered

## Status

Approved for implementation. The maintainer made these decisions in
[REFIMPL-02 (#1604)](https://github.com/monocongo/climate_indices/issues/1604) under the
[reference-implementation map (#1602)](https://github.com/monocongo/climate_indices/issues/1602).
This record covers how the protocol binds later work, not its content: the protocol is
[`protocol.md`](https://github.com/monocongo/climate_indices/blob/main/docs/research/reference-papers/protocol.md),
and it takes effect when the maintainer approves its text. The decisions below are approved;
the work they govern is not yet done:

- the metrics library that implements every measure, in
  [REFIMPL-16 (#1618)](https://github.com/monocongo/climate_indices/issues/1618), whose issue text predates the protocol;
- the harness and adapters, in [REFIMPL-08 (#1610)](https://github.com/monocongo/climate_indices/issues/1610);
- the studies that pre-register against it, from [REFIMPL-17 (#1619)](https://github.com/monocongo/climate_indices/issues/1619)
  to [REFIMPL-23 (#1625)](https://github.com/monocongo/climate_indices/issues/1625).

## Context

The project maintains one of the packages the studies compare, so a reader can fairly ask
whether the criteria, thresholds, and samples were chosen after the results were seen.
A protocol that lives only in an issue can be reworded, and a study that cites no version
cannot be checked against it.

## Decision

1. **The protocol is one versioned file.** Every change to a threshold, a difference
   class, or the definition of a measure is a major version, and does not apply to a study
   already pre-registered. Wording and links are patch changes.
2. **Every study is pre-registered against a protocol version.** The pre-registration
   (hypotheses, sample, seeds, thresholds, adapter configurations, analysis commit) is
   committed and tagged `prereg/<ticket>-<n>` before the study runs, and the result
   cites the tag. A result file with no ancestor pre-registration is not a result under
   the protocol.
3. **A change after the tag is an amendment**: dated, listed in the result, and marked
   *post hoc* wherever it affects a number.
4. **No composite score.** Criteria are reported separately and no ranking is computed.
5. **Every disagreement is ledgered.** Residuals after the last ladder rung, failed
   hypotheses, and non-correct robustness outcomes go to the REFIMPL-24 ledger before a
   result is final.

## Consequences

- Study tickets (REFIMPL-17 to REFIMPL-23, REFIMPL-31, REFIMPL-32) cite the protocol
  version and the pre-registration tag in their outcome.
- Changing a threshold means a new protocol version and a new pre-registration, not a
  quiet edit to a running study.
- Pushing a pre-registration tag is a push to a shared resource and needs the
  maintainer's approval each time.
- The record does not make any evidence class stronger: `VALIDATION.md` still decides
  what a result is called.
