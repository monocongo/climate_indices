# Cross-cutting index patterns are verified by behavior, not source greps

v2.4.0 shipped NFR-PATTERN-COVERAGE — 42 compliance points, seven indices × six
canonical patterns — checked by `tests/test_pattern_compliance.py`. That suite
asserted on source text (`"calculation_started" in source`), AST node counts
(`@overload` decorator counts), and the contents of the test suite itself
(`"test_spi_" in test_property_based.py`). Those are lint/process assertions, not
behavior: they fail for more than one reason, they break on any rename that
preserves behavior, and they ran in every version/distro job. The six patterns
remain the package's cross-cutting contract; how it is verified changed.

## Decision

`tests/test_pattern_compliance.py` is deleted as a pytest suite. Nothing replaces
the source-text, AST-count, or test-file-content assertions: "the source contains
this string" is not evidence about the public API, and the epic's testing
decisions forbid asserting log prose and private call order.

The 42-point NFR list is preserved here for audit. Seven indices, six patterns:

| Index | Source function | CF registry key | `typed_public_api` name |
| --- | --- | --- | --- |
| SPI | `indices.spi` | `spi` | `spi` |
| SPEI | `indices.spei` | `spei` | `spei` |
| PET Thornthwaite | `eto.eto_thornthwaite` | `pet_thornthwaite` | `pet_thornthwaite` |
| PET Hargreaves | `eto.eto_hargreaves` | `pet_hargreaves` | `pet_hargreaves` |
| PNP | `indices.percentage_of_normal` | `percentage_of_normal` | `percentage_of_normal` |
| PCI | `indices.pci` | `pci` | `pci` |
| Palmer | `palmer.pdsi` | per-output keys (`pdsi`, `phdi`, `pmdi`, `z_index`) | manual `palmer_xarray` wrapper |

Patterns: xarray adapter, `typed_public_api` overloads, CF metadata registry
entry, structlog lifecycle logging, structured exceptions, property-based tests.

Each pattern is now verified as follows:

| Pattern | Verification |
| --- | --- |
| xarray adapter | `tests/test_typed_public_api.py`, `tests/test_xarray_adapter.py`, `tests/test_xarray_equivalence.py`, `tests/test_pci_xarray.py`, `tests/test_pnp_xarray.py` — DataArray in, DataArray out, values equal the NumPy path |
| `typed_public_api` overloads | `tests/test_typed_public_api.py` exercises the NumPy and DataArray overloads; `uv run mypy src/` rejects an overload set whose implementation is inconsistent |
| CF metadata registry | `tests/test_cf_metadata.py` (keys and required fields); `tests/test_xarray_adapter.py` asserts registry metadata lands on output DataArrays |
| structlog lifecycle logging | none, deliberately. Event names are operational prose, not API, so no test asserts them. If an event name becomes a supported contract (dashboards, log-based alerts), it gets an explicit behavioral test then; #915 owns observability test consolidation |
| structured exceptions | `tests/test_exceptions.py` (hierarchy and catchability) plus the per-index `InvalidArgumentError` assertions in `tests/test_input_validation.py`, `tests/test_indices.py`, and `tests/test_eto.py` |
| property-based tests | `tests/test_property_based.py` (Hypothesis) is itself the evidence; the suite is run, not grepped |

Two deviations were recorded in the deleted suite's data and remain intentional:
PET lifecycle logging is emitted by the `indices.pet()` wrapper rather than
`eto.eto_thornthwaite()` itself, and Palmer reaches xarray through a manual
wrapper rather than `typed_public_api` overloads (ADR-0001).

## Consequences

- A behavior-preserving rename or refactor no longer fails `uv run pytest`.
- Pattern coverage is asserted where it matters: mypy for overload consistency,
  the xarray and CF-metadata suites for the API surface, Hypothesis for numeric
  properties.
- No test fails when a new index omits a pattern. The CF-metadata contract gets a
  single table-driven owner in #912/#913; that is the mechanism intended to keep
  new indices honest, not a cross-product tripwire.
- The `INDICES` mapping above is the audit reference for the 42 points. The
  v2.4.0 CHANGELOG's pointer to `tests/test_pattern_compliance.py` is a historical
  record of what shipped in that release, not a live reference.
