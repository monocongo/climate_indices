# Optional-Rust packaging: hatchling artifacts beside maturin binary wheels

## Status

Accepted. Implements [#1283](https://github.com/monocongo/climate_indices/issues/1283)
(RUST-013), the packaging questions the RUST-012 record
([#1282](https://github.com/monocongo/climate_indices/issues/1282)) defers to this ticket
in its decision 8 and in its "maturin as the PEP 517 backend now" alternative. The
implementation is `.github/workflows/release.yml`, `crates/climate-py/Cargo.toml`, and
`tests/test_release_integrity.py`.

## Context

The Rust backend is optional: `climate_indices` must import, pass its whole suite, and
produce identical results whether or not `climate_indices._native` is installed (the
RUST-012 record). The published artifacts therefore have to answer two independent
questions — what a runtime without the extension does, and how a user installs the
package from source without a Rust toolchain.

The runtime half was already answered: dispatch imports `_native` inside
`try`/`except ImportError` and every computation falls back to Python. The packaging
half was not. Until now hatchling was the PEP 517 backend and published an sdist plus a
`py3-none-any` wheel, neither of which carries the extension, so PyPI users never got
it; `[tool.maturin]` served only `maturin develop`.

Three facts constrain the choice:

- PyPI has no way to mark a compiled dependency optional. A wheel either installs with
  its extension or the install fails, so a published binary wheel is a commitment about
  the platform's ABI, not a hint.
- pip prefers the most specific compatible wheel for a version over a more generic one,
  and any wheel over the sdist (`CandidateEvaluator._sort_key` ranks by
  `wheel.find_most_preferred_tag` and gives an sdist the lowest rank). A `py3-none-any`
  wheel published at the same version is therefore a real fallback, not a decoy.
- `rust-numpy` targets NumPy's C-API ABI v2 with runtime version checks, so a wheel built
  against a recent NumPy also runs against the `numpy>=1.24` floor: the published
  `cp310-abi3` wheel imports and dispatches SPI under NumPy 1.26.4 on Python 3.12.

## Decision

1. **Hatchling stays the PEP 517 backend.** The sdist and the `py3-none-any` wheel remain
   pure Python. `uv sync`, editable installs, `uv.lock`, and `python -m build` are
   unchanged, and a source install never needs Rust.
2. **maturin builds binary wheels that carry `_native`, published at the same version as
   the pure artifacts.** `release.yml` builds them in `build-wheels` and publishes them
   with the sdist and the pure wheel from the same tag, so the fallback rule in the
   Context applies: a matching platform wheel wins, everything else gets the pure wheel.
3. **abi3 (`abi3-py310`) rather than one wheel per interpreter.** `rust-numpy` works with
   PyO3's limited API, so a single `cp310-abi3` wheel per platform validates on every
   supported Python (3.10–3.14). The matrix has no interpreter axis, and the same wheel
   file is installed on the oldest and newest supported versions in CI.
4. **The published matrix is five wheels**: `x86_64` and `aarch64` manylinux_2_28, macOS
   arm64, macOS `x86_64`, and Windows `x86_64`. Every one is built and imported on its own
   runner. **musllinux is deliberately not published** — musl users (Alpine) fall back to
   the pure wheel. Adding it needs a musl container build and a musl import check; the
   demand that would justify it has not appeared.
5. **The two installation paths are tested as artifacts, not as intent.** `wheel-check`
   installs each platform wheel outside the checkout, asserts pip resolves the platform
   wheel over `py3-none-any` when only the release artifacts are visible, and asserts SPI
   reaches its Rust kernels. `no-rust-install` removes `cargo` and `rustc` from `PATH`,
   installs the pure wheel, rebuilds the shipped sdist into a `py3-none-any` wheel with no
   compiled file in it, and runs the core suite against the installed package on both
   boundary Pythons. `publish` waits for all of them.
6. **Failure modes.** An sdist build without Rust cannot fail, because hatchling never
   invokes `cargo`; there is no error message to document, and the CI job above pins that
   with the toolchain removed rather than by trusting it. A stale or mismatched
   `_native` — a `.so` built for another interpreter or platform — raises `ImportError`
   on import, which dispatch catches and the Python path serves. Any other error from the
   extension (including a PyO3 panic) propagates and is never retried, as the RUST-012
   record requires. A developer's build artifact cannot leak into a pure wheel: hatchling
   omits `*.so`/`*.pyd`, which `.gitignore` already covers.

## Alternatives considered

**(a) maturin as the PEP 517 backend, sdist requires Rust.**

- Platforms: the same five, one wheel each, since abi3 is a property of the crate.
- abi3: works; not the deciding factor.
- `pip install climate_indices` on a platform without a wheel: builds from the sdist and
  **fails without a Rust toolchain**, which is the failure this ticket exists to avoid.
- Dev workflow: every `uv sync` of the project builds the extension, so `uv.lock`
  consumers and Read the Docs need a toolchain, and the fast pure-Python inner loop is
  gone unless the extension build is made conditional by hand.
- Downstream: conda-forge and distro packagers get one build path but must carry Rust in
  bootstrap and offline builds, and cannot ship the pure-Python fallback at all.

Rejected: it trades a compile-free install for a smaller workflow file, and it makes the
extension mandatory for users who do not want it.

**(b) hatchling for the sdist and pure wheel, maturin for binary wheels at the same
version — chosen.**

- Platforms: the five above; musllinux and any other target fall back to the pure wheel.
- abi3: works, and is what keeps the matrix at five wheels.
- `pip install climate_indices` without a matching wheel: installs the pure wheel and runs
  the Python implementations. No compiler, no build step, no failure path.
- Dev workflow: unchanged. `uv sync`, editable installs, and `uv.lock` see a pure-Python
  project; `maturin develop` stays opt-in.
- Downstream: conda-forge and distro packagers keep building the pure Python package and
  may add the extension when their infrastructure can; the two artifacts share a version
  and a public API, so they can also ship only one.

**(c) an optional build step that falls back to pure Python** (setuptools-rust
`optional=True`, or a maturin/hatch hook).

- Platforms: whatever a user's machine can build; nothing is published platform-specific
  unless wheels are still built separately.
- abi3: irrelevant, since every user builds their own.
- `pip install climate_indices` without a wheel: builds the extension when a toolchain is
  present and silently installs without it when the toolchain is missing or the build
  fails. Two users on the same platform end up with different numerical backends from the
  same command, and a broken Rust build is indistinguishable from an absent compiler.
- Dev workflow: a new build-backend dependency and a second build path to keep green.
- Downstream: the package's build becomes non-deterministic for packagers, who must then
  pin the outcome.

Rejected: it buys "the extension when possible" at the cost of reproducibility, and the
runtime fallback already covers the machines that cannot build.

**(d) a separate accelerator distribution** (for example `climate-indices-native`).

- Platforms: the same five, as a second project to build, version, and publish.
- abi3: works, with the same effect on the matrix.
- `pip install climate_indices` without a wheel: unchanged, but `pip install
  climate-indices-native` on a platform without that wheel fails loudly, and nothing tells
  a user it is optional.
- Dev workflow: two versions to keep in step and two artifacts to test together, so the
  parity suite spans a distribution boundary.
- Downstream: packagers must decide whether to ship a second package; a version skew
  between the two is a new failure mode with no upstream gate.

Rejected: it splits one versioned contract across two artifacts and hides the fallback
where users cannot see it. #1282 considered and deferred this; nothing has changed since.

## Consequences

- `release.yml` gains `build-wheels` (five legs) and `no-rust-install` (both boundaries),
  and `wheel-check` becomes a six-leg matrix; `publish` waits for all of them, and the
  GitHub Release attaches the merged artifact set. Like the pure-Python build, the wheel
  builds wait for the test matrix, so a red tree publishes nothing; release wall time
  grows by the wheel builds and the two installation checks.
- The published wheel set is one sdist, one `py3-none-any` wheel, and five `abi3` wheels.
  A platform without a matching wheel installs the pure wheel and runs Python; that is the
  supported no-Rust path and it is the fallback, not an error.
- README's install section, `docs/architecture.md`, `docs/deployment-guide.md`, and
  `docs/release-process.md` state which installs carry the extension, how to check
  (`python -c "import climate_indices._native"`), and how to build from source.
- The macOS `x86_64` wheel depends on the `macos-15-intel` runner label, which GitHub
  retires in August 2027; the aarch64 Linux wheel depends on the `ubuntu-24.04-arm` label.
  When either goes, drop that leg to the pure wheel or move it to a cross-build.
- `tests/test_release_integrity.py` pins the matrix, the abi3 feature, and both install
  paths, so a change that quietly stops publishing the extension fails in review.
- No CI leg installs a binary wheel against the NumPy 1.x floor: `test-minimum-deps`
  resolves old dependencies in pure Python, and the wheel checks run the stack a fresh
  install resolves. The NumPy 1.26 import above was measured by hand. A leg that installs
  the abi3 wheel with `--resolution lowest-direct` would close that gap if the floor ever
  needs to be depended on.
