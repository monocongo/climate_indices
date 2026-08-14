# Climate Index Explorer

This context defines the shared language for planning and operating the interactive climate-index explorer.

## Language

**Trusted Local Mode**:
Single-user explorer operation within the operator's trust boundary, with access to local or remote data sources selected by that operator.
_Avoid_: Desktop mode, developer mode

**Curated Hosted Demo Mode**:
Public explorer operation for untrusted visitors using only fixed, allowlisted, read-only data sources, without visitor-supplied data, durable identity, or persistent sessions.
_Avoid_: Hosted mode, cloud mode, production mode

**Computation Specification**:
A complete declaration of a climate-index derivation, including its source identity, input variable, index parameters, calibration period, periodicity, and missing-data policy.
_Avoid_: Pipeline spec, dashboard parameters, CLI options

**Source Descriptor**:
A non-secret identity for a source dataset at a specific, verifiable revision.
_Avoid_: Source URL, source path, connection string

**Source Catalog**:
The set of available Source Descriptors and the rules for resolving them to accessible datasets without embedding credentials.
_Avoid_: Dataset list, URL list

**Demonstration Sample**:
A provenance-verified, authentically sourced, reduced nClimGrid precipitation dataset used for the explorer demonstration and its performance acceptance. It is distinct from the synthetic deterministic fixture used by automated tests.
_Avoid_: Example fixture, generated benchmark data

**Index Capability**:
A supported combination of climate index, required input roles, periodicity, configurable parameters, and validation constraints exposed by the explorer.
_Avoid_: Dashboard form, algorithm option

**Computation Job**:
A tracked effort to materialize one Computation Specification, including its lifecycle, progress, diagnostics, and execution-attempt history.
_Avoid_: Pipeline run, Dask task, request

**Computation Attempt**:
One execution try within a Computation Job; a retry creates a new attempt without replacing earlier history.
_Avoid_: Retry, rerun

**Derived Index Cube**:
An immutable, fully materialized gridded climate-index result that has passed validation and been atomically registered. It is shared by Computation Key rather than owned by a UI session or visitor; partial output is not a Derived Index Cube.
_Avoid_: Output file, result dataset, cache entry, session result

**Computation Key**:
The deterministic identity of a requested Derived Index Cube, based on the complete Computation Specification, source and grid revisions, and `climate_indices` version.
_Avoid_: Job ID, cache key, filename

**Pinned Derived Index Cube**:
A Derived Index Cube that an operator has explicitly protected from automated retention policies. Active use can delay eviction but does not itself pin the cube.
_Avoid_: Visitor-owned result, session-owned result

**Derived Index Cube Record**:
The durable account of a Derived Index Cube's derivation, publication, availability, and eviction state. It remains after artifact eviction so the result's prior existence and reproducibility remain explainable.
_Avoid_: Cache metadata, tombstone

**Exploration View**:
A read-only presentation of one Derived Index Cube at one Display Time, optionally linked to a Selected Grid Cell. Changing an Exploration View never changes the Computation Specification or triggers computation.
_Avoid_: Dashboard session, result editor

**Display Time**:
The time coordinate whose slice is currently shown on the map and highlighted in the linked series.
_Avoid_: Computation time, selected date

**Selected Grid Cell**:
The source-grid cell committed for linked exploration of its full Derived Index Cube series. It remains selected while Display Time changes.
_Avoid_: Clicked point, interpolated location
