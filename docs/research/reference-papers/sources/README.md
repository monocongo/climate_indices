# Reference sources

One place for the papers and reference code behind the REFIMPL work (Palmer first,
other index families later). Each source has a stable id, a manifest entry, and, where
the licence allows, a file and a page-cited notes file. This folder is excluded from
Sphinx (`docs/conf.py`) and from the sdist and wheel (`docs/` is excluded in `pyproject.toml`).

## Layout

```
sources.yaml        manifest, one entry per source (field meanings in its header)
NOTES-TEMPLATE.md   template for a per-source notes file
<family>/pdf/       papers cleared for commit
<family>/notes/     page-cited notes, <id>.md
<family>/code/      reference code cleared for commit
local/              gitignored: files not cleared for commit, <id>.<ext>
external/           gitignored: clones and downloads of third-party code and data, one <id>/ directory each,
                    pinned by commit or sha256
```

`palmer/` is the first family workspace. Create other family folders only when their files are needed.
Keep cleared data and data documentation directly under `<family>/`.

Do not use Git LFS here: `.gitattributes` applies it to `*.nc` only, and anonymous
agent clones cannot fetch LFS objects.

## Ids

Use the id in the manifest, `palmer-provenance.md`, and ticket comments.
The id is the file stem under `local/` and `<family>/`.
Under `external/<id>/`, keep upstream file names unchanged. Papers: `<surnames>-<year>`, all surnames for one or two authors
(`palmer-1965`, `heddinghaus-sabol-1991`), the first author's for three or more
(`wells-2004`). Code: a short slug (`ncei-pdi-f`, `sibada-scpdsi-0.1.3`).

## What may be committed

Do not commit a source file under `<family>/` until its manifest entry has
`committable: 'yes'` with a recorded `license_evidence`. Quote `'yes'` and `'no'` to keep them YAML strings.
Until REFIMPL-27 (#1629)
decides, every entry is `pending` or `no`, and files stay under `local/` or `external/`.

- Publisher-copyright papers are not committed.
- GPL-3 code is read only; nothing is copied or vendored. Comparator code is never
  vendored (`comparators.lock.yaml`).
- Downloaded files are untrusted: one new directory each, hash it, never execute or
  compile it, and pass paths as arguments.

Default pytest discovery searches only `tests/` to avoid importing downloaded reference code.

## Add a source

1. Add an entry to `sources.yaml`. Record only verified facts and their evidence; leave unknown fields `null`.
2. Place the file under `local/` or `external/` (or `<family>/` once cleared).
3. Record `local_path` and `sha256` (`shasum -a 256 <path>`).
4. Write `<family>/notes/<id>.md` from `NOTES-TEMPLATE.md`.
5. Update the tickets the source unblocks and the status table below.

## Prompt against a source

Cite the id, point at `<family>/notes/<id>.md` when it exists, and name the provenance
rows or D-items the work is for. Example: "Using `heddinghaus-sabol-1991`, check
provenance row 22 and D2; record the page in its notes file."

`docs/algorithm_refs/palmer-provenance.md` (rows and D-items) is added by PR #1659; until it
merges, those references point at a file not yet on `main`.

## Status

<!-- Regenerate from sources.yaml: id, kind, status, committable. -->

| id | kind | status | committable |
| --- | --- | --- | --- |
| `palmer-1965` | paper | opened | pending |
| `alley-1984` | paper | opened | pending |
| `karl-1986` | paper | opened | pending |
| `wells-2004` | paper | opened | pending |
| `heim-2002` | paper | opened | pending |
| `ncei-divisional-readme` | data-doc | opened | pending |
| `sibada-scpdsi-0.1.3` | code | opened | no |
| `heddinghaus-sabol-1991` | paper | not-opened | pending |
| `karl-knight-1985` | paper | not-opened | pending |
| `jacobi-2013` | paper | not-opened | pending |
| `ncei-pdi-f` | code | located | pending |
| `ncei-spi-f` | code | located | pending |
| `ncei-soilcnst-asc` | data | located | pending |
| `wells-cpp-original` | code | not-opened | pending |
| `cran-scpdsi-0.1.3` | code | located | no |
| `spade-2020` | paper | located | pending |
| `dficklin-cageo-paper` | code | located | pending |
| `jonking93-pdsi` | code | located | pending |
