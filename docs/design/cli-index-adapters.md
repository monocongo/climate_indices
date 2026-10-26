# CLI index adapters

## Scope

This internal design note records the implementation choice for issue #1422.
The change replaces `_IndexRegistration` in `src/climate_indices/__main__.py`.
It preserves CLI arguments, numerical kernels, output names, attributes, files,
and the consumed-flags contract.

## Alternatives

A validated callback table could reject incomplete registrations during construction.
It would require fewer structural changes, but each index would still depend on
separate runner, preparation, computation, and writing functions.
It would also retain optional callbacks for the two execution paths.

Concrete adapters put each index's declarations and index-specific methods together.
Abstract base classes require complete implementations before an adapter can be
instantiated. Shared methods preserve common behavior without duplicating it.
This implementation uses concrete adapters.

The adapters remain in `__main__.py` because their shared request types and input
helpers already live there. Separate adapter modules would require another module
boundary or circular imports. This refactor does not introduce either.

## Interfaces

`_IndexAdapter` defines `run(arguments, input_type, transport)`,
`validate_arguments(args)`, and `validate_inputs(args, context)`.
The validation methods do nothing unless an index overrides them.
Each concrete adapter declares its input fields and consumed flags.
The existing pipeline table still controls aggregate order and prerequisite PET
computation. The PE-dependent adapters still compute PE when necessary.

`_SharedArrayAdapter` adds mandatory kernel, worker, input-name, kernel-argument,
computation, and writing methods. Input preparation and coordinate preparation
have explicit no-op defaults. `_SingleOutputAdapter` shares output allocation
and atomic writing, but requires each index to define its output metadata.
The Palmer adapter instead owns its five output definitions and preparation,
computation, and writing methods.

The xarray-backed adapters implement the common interface without unused
shared-array methods. `_EnergyAdapter` also declares an output name; the
existing energy-name lookup derives its entries from the registered adapters.
Shared helpers still handle input opening, chunking, validation, and writing.

## Preserved boundaries

`Transport` and `_cli_transport.py` do not change.
The multiprocessing kernels remain module-level functions so workers can
serialize them. The xarray-backed indices retain their existing Dask execution.
This preserves the execution split in ADR-0002.
All output paths retain the existing atomic NetCDF writer.

## Checks

The adapter tests reject an incomplete shared-array implementation.
A direct PNP adapter test writes a real NetCDF output and checks its values
against independently calculated monthly ratios.
The existing flag-contract, pipeline-order, output, and CLI end-to-end tests
check compatibility. The CLI end-to-end tests remain unchanged.
