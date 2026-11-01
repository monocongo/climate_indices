# Dataset provenance

Output variables that `climate_indices` computes through its xarray API or
its command line carry a provenance record, except for the compound-hazard Dataset routes.
The record names the software
that produced the variable, the computational backend, the effective
scientific parameters, and each input with the metadata it supplied. The
record is stored as variable attributes, so it survives NetCDF and Zarr write
and reopen cycles and stays with its variable when outputs from separate runs
are merged into one Dataset.

The record supports reproducibility: it tells you what to rerun. It does not by
itself guarantee identical numbers in another environment. NumPy, SciPy, the
platform's math libraries, and the optional Rust backend can differ within the
documented tolerances (see {doc}`rust-numerical-contract`).

## Attributes

Each output variable gets these project-defined attributes, except for the compound-hazard Dataset routes described below.
They are not CF or ACDD attributes, and they do not change `Conventions`.

| Attribute | Value |
| --- | --- |
| `climate_indices_version` | The installed package version, e.g. `3.1.0`. |
| `climate_indices_git_commit` | The full source commit (40 or 64 hexadecimal characters), or `unknown`. |
| `climate_indices_provenance_schema_version` | The record schema version, `1.0.0`. |
| `climate_indices_provenance` | The complete record, as one strict-JSON string. |
| `history` | The input's `history`, with one new UTC line appended. |

`climate_indices_git_commit` names a commit only. Read `software.git_dirty` in
the record before you treat the commit as the exact code: a modified working
tree keeps its base commit and reports `git_dirty: true`.

The compound-hazard Dataset routes keep their record only as a Dataset attribute, next
to the `compound_manifest` attribute that names their sources. Their individual variables have no record.
Use `read_provenance(result)` for these routes; `read_provenance(result[variable])` returns `None`.

## Record contents

```json
{
  "schema": "climate_indices.dataset-provenance",
  "schema_version": "1.0.0",
  "record_id": "sha256:...",
  "created": "2026-10-25T14:03:11Z",
  "operation": {"index": "SPI", "interface": "python", "implementation": "climate_indices.indices.spi"},
  "software": {
    "distribution": "climate_indices", "version": "3.1.0",
    "repository_url": "https://github.com/monocongo/climate_indices",
    "git_commit": "9d5d929ba4c7a92d0da185e3a80ee44f8b9f8237",
    "git_commit_source": "build-metadata", "git_dirty": false
  },
  "backend": {"native_extension": "available", "native_version": "3.1.0", "kernel_selection": "per call at execution time; not recorded"},
  "parameters": {"scale": 3, "distribution": "gamma", "periodicity": "monthly", "data_start_year": 1981,
                 "calibration_year_initial": 1991, "calibration_year_final": 2020,
                 "output_scale": "normal", "zero_handling": "classic", "fitting_params": null},
  "inputs": [{"role": "values", "variable": "prcp", "source": {"kind": "local-file", "name": "nclimgrid_prcp.nc"},
              "dims": ["time"], "shape": [480], "dtype": "float64", "chunked": false,
              "coordinates": {"time": {"size": 480, "first": "1981-01-01T00:00:00", "last": "2020-12-01T00:00:00"}},
              "attributes": {"units": "mm"}, "upstream": []}]
}
```

- `operation`: the index display name (as in `history`), the interface that
  built the record, and the computing function when it is known. The interface is
  `cli` for the command line's shared-memory indices (SPI, SPEI, PNP, PET, and the
  Palmer family) and `python` for everything computed through the xarray API,
  including the command line's KBDI, flood, and energy indices.
- `software`: see [Software identity](#software-identity).
- `backend`: whether the optional Rust extension is installed and its version.
  The library chooses a kernel per call, at execution time, from the operand
  layout and the floating-point policy. A lazy result is computed after its
  metadata is attached, so the record does not say which kernel ran.
- `parameters`: the values the computation used. For SPI, SPEI, PNP, EDDI, and
  `standardized_index`, this is every parameter of the computing function,
  including defaults and the `periodicity`, `data_start_year`, and calibration
  years inferred from the time coordinate. The calibration years are the ones the
  fit used after any clamping. The other adapters record the parameters they
  report as calculation attributes, plus their scientific options.
  Supplied `fitting_params` are recorded by identity: small arrays inline, larger
  ones by shape and dtype, never their full values.
- `inputs`: one entry per input, by role (the function parameter name, or the
  CLI role `precipitation`, `temperature`, `pet`, or `awc`). Energy elementwise
  transforms use `primary`, `companion_1`, `companion_2`, and subsequent numbered
  roles in positional input order, not parameter names. Compound Dataset routes
  record no inputs; their `compound_manifest` attribute identifies the sources.
  See [Input metadata](#input-metadata).
- `record_id`: a SHA-256 of the record without this field. Upstream references
  use it to remove duplicate steps.
- `truncated`: present only when a size limit dropped sections.

`created` records UTC time with whole-second precision. `record_id` identifies the
record contents, including that timestamp, not a unique execution. Identical
computations within one second can have the same record ID. To compare two
results, compare `software`, `backend`, `parameters`, and the inputs' identifiers.

Unavailable values are explicit: `null`, with a `*_source` field or a note that
says why.

## Software identity

The commit is resolved once per process, the first time a record is built. It is
never resolved at import time or per chunk, and it never uses the network.

| Order | Condition | `git_commit_source` |
| --- | --- | --- |
| 1 | The package is imported from `<root>/src/climate_indices` and `<root>` is its own Git checkout. Git reports `HEAD` and whether tracked files are modified. | `source-checkout` |
| 2 | The installed package carries `climate_indices/_build_info.json`, written at build time. | `build-metadata` |
| 3 | The installed distribution is the imported package, and its `direct_url.json` names a Git `commit_id` (a `pip install git+https://...` install). | `direct-url` |
| 4 | Nothing applies. The version is still recorded. | `unavailable` |

Rules:

- The working directory and the `GIT_*` environment variables never contribute,
  so running from an unrelated repository cannot change the commit.
- In a source checkout, embedded build metadata is ignored, because the checkout
  can have moved on since a build. If Git is unavailable there, the commit is
  `null`.
- If the build metadata and `direct_url.json` disagree, the commit is `null` and
  both candidates are listed under `software.conflicts`.
- The version string is never parsed for a commit. A release tag, a package
  version, an artifact checksum, and a source commit are different identifiers,
  and only the commit is recorded as a commit.

How builds embed the commit: the Hatchling build hook `hatch_build.py` writes
`_build_info.json` (`git_commit`, `git_dirty`) into the sdist and the wheel when
it builds from a Git checkout. It never writes into `src/`. A wheel built from
that sdist, which has no `.git`, ships the sdist's file, so the sdist-to-wheel
path keeps the commit. Editable installs embed nothing. maturin does not run
Hatchling hooks, so the release workflow runs `python hatch_build.py` before it
builds the platform wheels, and `[tool.maturin] include` ships the file. An
artifact never records its own checksum.

## Input metadata

Each input entry is built from metadata only: attributes, encodings, and the
in-memory indexes of its coordinates. Collecting it never reads data values and
never triggers a Dask computation.

- `variable`, `dims`, `shape`, `dtype`, and `chunked`.
- `coordinates`: for each dimension, its size, its first and last index label,
  and the `units` and `calendar` from the coordinate encoding or attributes.
- `attributes`: the input variable's attributes, without its own provenance
  attributes. They are recorded under the input, not copied onto the output.
- `source`: the file the input was opened from (xarray's `encoding["source"]`),
  or the locator a caller supplied.
- `upstream`: when the input is itself a `climate_indices` output, a digest of
  its record (`record_id`, `created`, `operation`, `parameters`, the software
  version and commit, and its inputs' identifiers) followed by the digests of its
  own upstream steps. The chain is flattened and deduplicated by `record_id`, so
  it does not nest and grows by one digest per step, to at most 16.

A variable selected from a Dataset does not carry the Dataset's global
attributes. Use {func}`climate_indices.provenance.with_source` to add them, or to
name the source of in-memory data:

```python
from climate_indices.provenance import with_source

dataset = xr.open_dataset("nclimgrid_prcp.nc")
precipitation = with_source(
    dataset["prcp"],
    uri="https://www.ncei.noaa.gov/data/nclimgrid-monthly/access/nclimgrid_prcp.nc",
    doi="10.7289/V5SX6B56",
    dataset_attributes=dataset.attrs,
)
```

`with_source` returns a shallow copy, so the data stays lazy. Only the identity,
licensing, and lineage keys of `dataset_attributes` are kept, such as `title`,
`source`, `institution`, `references`, `license`, `id`, `doi`, `version`, and
`history`. A `checksum` must name its `algorithm`, `value`, and the `object` it
identifies; nothing is computed, and an object-store ETag is not a content
checksum.

The command line labels each input with its file. The xarray routes also record
each file's global attributes. The shared-memory routes retain global attributes
only for inputs from the first opened file. Secondary files' global attributes
are unavailable after the files are combined. These routes describe each input
before unit normalization, so the record keeps the source's own `units`. The one
exception is the flood family (`pe`, `edi`, and `flood_index`): it converts
precipitation to millimeters first, so its record shows `units` as `mm`.

### Normalization and limits

- Enums become their name, NumPy scalars become Python numbers, bytes are decoded
  as UTF-8 with replacement, and dates become ISO 8601 strings.
- `NaN`, `Infinity`, and `-Infinity` become those strings, because strict JSON
  has no such numbers.
- Arrays and sequences with at most 16 elements are inlined; longer ones are
  recorded by type, shape, and dtype.
- Strings are truncated at 1024 characters, mappings at 64 keys, and nesting at
  depth 4, each with an explicit marker. Other objects become
  `{"unsupported_type": "<name>"}`. Supported fire-state objects record their
  fields, including units and gap bookkeeping. Small state arrays include their
  values; larger arrays retain only their shape and dtype.
- A record is at most 32 KiB, far below HDF5's 64 KiB attribute limit. Above
  that, input attributes, then dataset attributes, then upstream digests are
  replaced by `{"omitted": "record size limit"}`, and `truncated` lists them.
  If necessary, parameters and complete input descriptions are omitted next.
  The limit includes `record_id` and the truncation markers.
- The JSON is ASCII with sorted keys, so NetCDF-3, NetCDF-4, and Zarr store it
  unchanged.

### Privacy

- A local source locator is reduced to its file name, without its directories.
- A URL loses its user information, query string, and fragment. Signed-URL
  tokens are in the query string, so they are removed.
- In recorded text, URL credentials and query strings are removed. Recognizable
  absolute paths, including `file://` URLs, are reduced to their file names.
  This is not a general anonymization guarantee. Review free text before sharing
  outputs, particularly paths with spaces or names written outside paths.
- Attributes whose names look like secrets (`password`, `token`, `secret`,
  `api_key`, `credential`, `signature`, `authorization`) are recorded as
  `[redacted]`.

## Inspect provenance after reopening

The same calibration-period choice that changes the index values is recorded in
the record, so two otherwise identical results stay distinguishable:

```{testsetup} provenance
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

workdir = tempfile.TemporaryDirectory()
time = xr.date_range("1981-01-01", periods=480, freq="MS")
precipitation = xr.DataArray(
    np.random.default_rng(0).gamma(2.0, 30.0, time.size),
    dims="time",
    coords={"time": time},
    name="prcp",
    attrs={"units": "mm"},
)
```

```{doctest} provenance
>>> import climate_indices
>>> from climate_indices.indices import Distribution
>>> from climate_indices.provenance import read_provenance
>>>
>>> spi_1981 = climate_indices.spi(
...     precipitation, scale=3, distribution=Distribution.gamma,
...     calibration_year_initial=1981, calibration_year_final=2010,
... )
>>> spi_1991 = climate_indices.spi(
...     precipitation, scale=3, distribution=Distribution.gamma,
...     calibration_year_initial=1991, calibration_year_final=2020,
... )
>>> combined = xr.Dataset({"spi_1981": spi_1981, "spi_1991": spi_1991})
>>> path = Path(workdir.name) / "spi.nc"
>>> combined.to_netcdf(path, engine="h5netcdf")
>>> with xr.open_dataset(path, engine="h5netcdf") as reopened:
...     records = {name: read_provenance(reopened[name]) for name in reopened.data_vars}
>>> [records[name]["parameters"]["calibration_year_initial"] for name in ("spi_1981", "spi_1991")]
[1981, 1991]
>>> records["spi_1981"]["record_id"] != records["spi_1991"]["record_id"]
True
>>> records["spi_1981"]["software"]["version"] == climate_indices.__version__
True
```

A Zarr store returns the same record:

```{doctest} provenance
:skipif: not __import__("importlib.util", fromlist=["util"]).find_spec("zarr")

>>> store = Path(workdir.name) / "spi.zarr"
>>> _ = combined.to_zarr(store, mode="w", consolidated=False)
>>> with xr.open_zarr(store, consolidated=False) as reopened:
...     read_provenance(reopened["spi_1991"]) == read_provenance(spi_1991)
True
```

```{testcleanup} provenance
workdir.cleanup()
```

`read_provenance()` returns `None` for a variable without a record. From a shell,
`ncdump -h` shows the attributes of a NetCDF output, and the JSON string can be
pretty-printed with any JSON tool.

## Behavior across steps and runs

- **Repeated attachment.** A record describes one variable and replaces any
  earlier `climate_indices_*` provenance attributes. An output never inherits an
  input's provenance attributes; the input's record becomes an `upstream`
  digest instead.
- **Derived inputs.** Computing SPEI from a PET result records the PET record as
  the upstream step of the `pet_mm` input, with its own inputs' identifiers.
- **Merged runs.** Except for compound-hazard Dataset routes, records are per variable, not global attributes.
  Merging outputs from separate runs keeps each variable's lineage.
  Compound records are Dataset attributes and do not stay with variables selected from those Datasets.
- **History.** The output keeps its primary input's `history` and appends one
  line. Each input's own `history` stays in its entry of the record.

## Limitations

- Zarr append (`append_dim`) and region writes are not supported paths. xarray
  keeps the attributes of the write that created each variable (checked with
  xarray 2026.1.0 and zarr 3.3.0), so the record does not describe values
  appended or overwritten later.
- The CLI's shared-memory outputs still inherit their first input file's global
  attributes, as before; the record describes the inputs separately.
- Adapters that do not use the `xarray_adapter` decorator record the parameters
  listed in their own calculation attributes and options, not their full
  signatures.
- The NumPy API returns plain arrays and records nothing.
