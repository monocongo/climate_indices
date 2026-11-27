# Intercomparison harness

Runs comparator implementations of one index on identical inputs, each in its own environment,
and records what each one did. Design and decisions: [`docs/research/reference-papers/harness-design.md`](../docs/research/reference-papers/harness-design.md)
(REFIMPL-08, [#1610](https://github.com/monocongo/climate_indices/issues/1610)).

This directory is excluded from the sdist and the wheel and adds no dependency to the package.
CI runs only the offline tests in `tests/test_intercomparison_harness.py` and `tests/test_intercomparison_metrics.py`; the runs below are manual.

## Rerun the slice (gamma SPI-3, division 0101, `climate_indices` and `xclim`)

```bash
uv sync --extra test        # the repository environment: climate_indices, zarr, jsonschema
uv run python intercomparison/run.py --division 0101 --study intercomparison/studies/refimpl-08-slice
```

Use an empty study directory or one with existing harness output. The driver replaces the previous
manifest and bulk output. Do not use the same study directory for concurrent runs.

One command per environment, as the driver runs them (each reads `<request.json>` and writes `<out_dir>/result.json`):

| Environment | Command |
| --- | --- |
| `climate_indices` (repository env) | `uv run python intercomparison/adapters/climate_indices_adapter.py run <request.json> <out_dir>` |
| `xclim` 1.0.0 (own locked env) | `uv run --locked --project intercomparison/adapters/xclim python intercomparison/adapters/xclim/adapter.py run <request.json> <out_dir>` |

The first `xclim` run builds `intercomparison/adapters/xclim/.venv` from its `uv.lock` (Python 3.12, network needed).

## Outputs

| Path | Committed | What |
| --- | --- | --- |
| `studies/<study>/manifest.json` | yes | inputs, positions requested, and per implementation the version, effective method, software, platform, wall time, warnings and output SHA-256 |
| `studies/<study>/store.zarr` | no | array `values`, dims (implementation, unit, time) |
| `studies/<study>/work/` | no | the request, the input CSV, and each adapter's `result.json` and series |

For this committed slice, a rerun on the same inputs, at the `harness` commit the manifest records, from a clean tree (`git_dirty` false), on the platform and with the Python and package versions it records, reproduces every `output_sha256` and `checksum_sha256`. Other platforms or NumPy and SciPy versions can change the last bits of a float, and so a hash; compare with a tolerance there. Wall time, platform and `created` are recorded metadata that may differ. A dirty manifest cannot be reproduced from its metadata.
Validate a manifest with `manifest.schema.json`.

## Metrics

`metrics.py` implements the measures of the [evaluation protocol](../docs/research/reference-papers/protocol.md)
(REFIMPL-16, [#1618](https://github.com/monocongo/climate_indices/issues/1618)): one function per row of its Measure reference,
each naming its row. It needs NumPy and SciPy only. `write_metrics` validates a results document against `metrics.schema.json`
and needs `jsonschema`; JSON `null` stands for an undefined (NaN) value. Load it the way `run.py` loads `contract.py`:

```python
import sys

sys.path.insert(0, "intercomparison")
import metrics

metrics.difference_class(a, b)  # {"class": <one of the protocol's difference classes>, "d": max|a - b|}
```

To recompute the Earth Engine pilot's reference A figures through these measures (pooled pairs, the eleven-class scale and
the pilot's `lower_inclusive` rule), with Earth Engine credentials as the pilot documents:

```bash
uv run --extra earthengine scripts/earthengine_climate_engine_pilot.py --check-metrics-library
```

It writes nothing. It fails if the library and the pilot's own `comparison_metrics` disagree on the same arrays, and prints
each figure's difference from the committed `pilot-metrics.json` (a new collection revision moves those, not the library).
