# The crate publishes to crates.io as `climate-indices` on an independent 0.x track

## Status

Amended. This record implements [#1306](https://github.com/monocongo/climate_indices/issues/1306)
(CRATE-02), a child of the CRATE-01 epic
([#1305](https://github.com/monocongo/climate_indices/issues/1305)). It supersedes
decision 2 of {doc}`0017-rust-core-acceleration-backend` for the clause that says the
crate is not published to crates.io, and the RUST-000 non-goal *no crates.io
publication* recorded on [#1270](https://github.com/monocongo/climate_indices/issues/1270).
The identity is decided here; CRATE-05
([#1309](https://github.com/monocongo/climate_indices/issues/1309)) performed the rename,
which is why the present tense below describes the crate as `climate-indices` while the
context and alternatives keep the `climate-core` name they were written with.
CRATE-10 ([#1314](https://github.com/monocongo/climate_indices/issues/1314)) and CRATE-11
([#1315](https://github.com/monocongo/climate_indices/issues/1315)) amended the release
facts: the published `0.1.0-rc.1` rehearsal, the bindings range, trusted publishing, and
the `0.1.0` manifest version.

## Context

{doc}`0017-rust-core-acceleration-backend` decision 2 and the RUST-000 non-goal closed
crates.io publication while the Rust code existed only to accelerate `climate_indices`.
The crate was `crates/climate-core`, `publish = false`, with no crate README and an API
shaped around the Python seam. CRATE-01 ([#1305](https://github.com/monocongo/climate_indices/issues/1305))
reopens that: the crate becomes first-order, so a Rust user can `cargo add` it and use it
with no knowledge of the Python package, without changing the Python contract.

The name had to be settled first, because crates.io has no reservation. A name is claimed
only by publishing, and a published crate is not freely removable: crate owners can delete
a crate, but only under narrow conditions, so a name cannot be relinquished at will. The
maintainer therefore decided the name on #1306 and published a `0.0.0`
placeholder on 2026-10-08: single version, owner `monocongo`, not yanked, carrying a
`VERSION` constant, a README, and the BSD-3-Clause license, and no algorithms. That
placeholder is the bridge; the first real release supersedes it.

Four questions stayed open for this record: the version scheme, the `climate-py`
dependency direction, whether the crate README couples to the Python project, and release
ownership and cadence.

## Decision

1. **The crate is `climate-indices` everywhere.** The crates.io package is
   `climate-indices`, for continuity with the PyPI distribution it came from; the library
   path is `climate_indices`, so Rust code writes `use climate_indices::...`; and the
   directory is `crates/climate-indices`. The `0.0.0` placeholder is never yanked — the
   first real release supersedes it. The rename lands under CRATE-05
   ([#1309](https://github.com/monocongo/climate_indices/issues/1309)), as its own change,
   because it touches the workspace, the CI paths, and `[tool.maturin]`. The historical
   records keep the old name: {doc}`0017-rust-core-acceleration-backend`,
   {doc}`0018-optional-rust-packaging`, and the 3.0.0 release notes describe what was true
   when they were written.

2. **The crate runs an independent `0.x` version track.** The workspace already declares
   `0.1.0`, and the first real release is `0.1.0`. The crate version does not follow
   `pyproject.toml`, which is `3.0.0`. The Python distribution's major version states the
   stability of the Python API; the crate's API is not designed until CRATE-03
   ([#1307](https://github.com/monocongo/climate_indices/issues/1307)), so a `3.x` crate
   would promise a stability the crate has not earned.

3. **`climate-py` keeps the workspace path dependency, and stays `publish = false`.**
   `climate-py` is the PyO3 seam and is published only as the wheel
   `climate_indices._native`. Depending on a *published* `climate-core` version instead
   would force a crates.io release before every wheel change, and would couple a wheel
   build to registry state and the network. The published wheel embeds the compiled crate,
   so no wheel pins a crate version and the two artifacts release independently. The
   requirement is the range `>=0.1.0, <1.0.0`, so it admits every stable `0.x`
   release.

4. **The crate README stands alone.** It cites `climate_indices` as its origin and as the
   numerical parity oracle, and links to this repository and the Python project, but a
   reader needs no Python knowledge to use the crate. A first-order crate documents
   itself, not the application that happened to spawn it.

5. **Only the maintainer releases the crate, from a separate tag namespace, on demand.**
   Crate tags are `climate-indices-vX.Y.Z`. The Python release process in
   `docs/release-process.md` keeps exact `vX.Y.Z` tags, so the two namespaces cannot
   collide: the trigger in `.github/workflows/release.yml` is `v*.*.*` and its guard
   requires `^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$`, so a crate tag starts no PyPI release,
   and a Python tag publishes no crate. The crate releases when the crate changes, not
   with PyPI. CRATE-10 ([#1314](https://github.com/monocongo/climate_indices/issues/1314))
   moved publishing to crates.io trusted publishing; a second owner remains the
   maintainer's decision.

6. **No further name reservations.** crates.io treats `climate_indices` and
   `climate-indices` as one name, so the obvious variant is already blocked, and
   reserving more variants would be the squatting the crates.io policy prohibits.

## Alternatives considered

- **Keep `climate-core` as the published name.** Rejected: it diverges from the PyPI name
  and from the identity the maintainer chose, so a user meets two names for one product.
- **Sync the crate version to the Python distribution.** Rejected: it claims a stable
  `3.x` API before CRATE-03 designs one, and it forces a crate release on every Python
  patch release.
- **Depend on the published crate from `climate-py`.** Rejected: every wheel build would
  need a crates.io release first, and the build would depend on registry availability.
- **A crate README that requires the Python project.** Rejected: a Rust user must be able
  to read it without Python knowledge, which is what first-order means here.
- **Do the rename in this record's change.** Rejected: it mixes a decision with build and
  CI changes, needs the full Rust gate, and makes the review harder. CRATE-05 owns it.
- **Reserve typosquat variants.** Rejected: hyphen and underscore are already one name,
  and more reservations without a real release are squatting.

## Consequences

- CRATE-05 performed the rename: the directory, the package, the default library path,
  every `use climate_core::` in `crates/climate-py`, the `climate-core` path dependency
  in `crates/climate-py/Cargo.toml`, `benchmarks/README.md`, `tests/parity_registry.py`,
  `docs/architecture.md`, `docs/development-guide.md`, the docstrings in
  `src/climate_indices/flood/_native.py` and `src/climate_indices/fire/_native.py`, the
  `LICENSE`, and the generated `llms-full.txt`. It ran the Rust gate, because the
  workspace members and CI paths changed.
- Consequently the current crate carries its own version, README, keywords, categories,
  `rust-version`, and docs.rs metadata, and `cargo publish --dry-run` succeeds from a
  clean checkout. `climate-py` stays `publish = false` through the workspace.
- The Python contract does not change. The extension module stays
  `climate_indices._native`, and the Rust library name is a Rust path, not a Python import,
  so `climate_indices` names both without a collision.
- Before CRATE-05 both crates inherited `version.workspace = true` and
  `publish = false` from the workspace; that split is what CRATE-05 changed. The crate
  now carries `version = "0.1.0"` for its first stable release, which CRATE-11
  published, and `publish = true`; CRATE-10's rehearsal published `0.1.0-rc.1` first. `climate-py` keeps
  `publish = false` through the workspace.
- Crate releases and PyPI releases are independent in both directions, and neither tag
  namespace can start the other workflow.
- CRATE-10 ([#1314](https://github.com/monocongo/climate_indices/issues/1314)) publishes
  published the `0.1.0-rc.1` pre-release to rehearse the release path before the first
  stable version, and the bindings requirement became a range that admits a pre-release. A caret requirement such as `"0"` does not match one, and
  `cargo package` cannot resolve the workspace for a pre-release version. The range still
  admits every stable `0.x` release, so decision 3's property holds for a stable release:
  a later crate release needs no further bindings manifest change. Cargo matches a
  pre-release only against a requirement that names that same version, so each new
  version's pre-release rehearsal moves the range's floor. CRATE-11 moved the floor to
  `0.1.0` after that version was published. The rehearsal is documented in
  {doc}`../release-process`.
- {doc}`0017-rust-core-acceleration-backend` keeps the historical name, and
  `docs/architecture.md` states the crate as `climate-indices` with its independent
  version track. The published `0.0.0` placeholder contains no algorithms.
- The numerics become reachable under a public name before CRATE-04
  ([#1308](https://github.com/monocongo/climate_indices/issues/1308)) states the numerical
  contract and CRATE-06 ([#1310](https://github.com/monocongo/climate_indices/issues/1310))
  makes the crate verifiable without Python. The `0.0.0` placeholder carries no algorithm,
  so nothing was exposed until CRATE-10 published the `0.1.0-rc.1` rehearsal; CRATE-11
  ([#1315](https://github.com/monocongo/climate_indices/issues/1315)) published the first
  stable release, `0.1.0`.

## References

- [#1306](https://github.com/monocongo/climate_indices/issues/1306) — CRATE-02, this
  record's ticket; the maintainer's name decision and the placeholder notes.
- [#1305](https://github.com/monocongo/climate_indices/issues/1305) — CRATE-01 epic, the
  child tickets, and the definition of done.
- {doc}`0017-rust-core-acceleration-backend` — the crate split, the pure-Rust boundary,
  and the publication clause this record supersedes.
- {doc}`0018-optional-rust-packaging` — how the crate reaches a wheel, which fixes the
  boundary this record's dependency direction keeps.
- [#1314](https://github.com/monocongo/climate_indices/issues/1314) — CRATE-10, owner and
  release automation; [#1315](https://github.com/monocongo/climate_indices/issues/1315) —
  CRATE-11, the first real release.
