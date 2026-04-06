# Test Fixture Reference Data

This directory contains reference datasets used for validating the
climate_indices library against published scientific results.

## Provenance Protocol

Every reference dataset directory **must** include a `provenance.json` file
that records the origin, integrity, and version of the data. This enables
reproducible validation and guards against silent data corruption.

### Required Fields

| Field                  | Type   | Description                                                  |
|------------------------|--------|--------------------------------------------------------------|
| `source`               | string | Name of the providing institution (e.g., "NOAA PSL")         |
| `url`                  | string | Download URL or landing page for the dataset                 |
| `download_date`        | string | ISO 8601 date when the data was retrieved (`YYYY-MM-DD`)     |
| `subset_description`   | string | Human-readable description of the subset stored here         |
| `checksum_sha256`      | string | SHA-256 hash of the canonical data file(s) in the directory  |
| `fixture_version`      | string | Semantic version of the fixture layout (start at `"1.0.0"`)  |
| `validation_tolerance` | object | Keys are tolerance names; values are numeric tolerances       |

### Optional Fields

| Field         | Type   | Description                                             |
|---------------|--------|---------------------------------------------------------|
| `citation`    | string | Bibliographic reference for the methodology or dataset  |
| `doi`         | string | DOI of the dataset or methodology paper                 |
| `license`     | string | License or terms of use for the reference data          |
| `notes`       | string | Free-text notes about the subset or known limitations   |

### Example `provenance.json`

```json
{
  "source": "NOAA Physical Sciences Laboratory",
  "url": "https://downloads.psl.noaa.gov/Projects/EDDI/CONUS_archive/",
  "download_date": "2026-02-17",
  "subset_description": "EDDI 1-month scale, CONUS, 1980-2020 subset for library validation",
  "checksum_sha256": "abc123...",
  "fixture_version": "1.0.0",
  "validation_tolerance": {
    "rtol": 1e-5,
    "atol": 1e-5
  },
  "citation": "Hobbins, M. T., A. Wood, D. McEvoy, J. Huntington, C. Morton, M. Anderson, and C. Hain, 2016: The Evaporative Demand Drought Index.",
  "doi": "10.1175/JHM-D-15-0121.1"
}
```

### Directory Layout

```
tests/fixture/
    README.md                       # this file
    provenance_schema.json          # JSON Schema for provenance.json validation
    <dataset-name>/
        provenance.json             # required metadata
        *.npy / *.nc / *.csv        # reference data files
```

### Checksum Computation

To compute the checksum for a directory of `.npy` files, concatenate the
sorted file contents and hash the result:

```bash
cat $(ls *.npy | sort) | shasum -a 256
```

For single-file fixtures, hash the file directly:

```bash
shasum -a 256 reference_data.npy
```

### CI Validation

The test suite includes `test_provenance_protocol.py` which:

1. Scans `tests/fixture/` for directories containing a `provenance.json`
2. Validates each file against the JSON Schema in `provenance_schema.json`
3. Verifies that the `checksum_sha256` matches the actual data files

This runs as part of the standard `pytest` suite and will fail if any
reference dataset has been modified without updating its provenance metadata.
