# Rust core as an acceleration backend

## Status

Accepted. The decision is implemented by the RUST ticket series under the epic
[#1270](https://github.com/monocongo/climate_indices/issues/1270): the workspace
scaffold and SPI gamma port (RUST-001, RUST-002), the CI jobs (RUST-003), and the
per-kernel ports that followed. Packaging of binary wheels is deferred to RUST-013
([#1283](https://github.com/monocongo/climate_indices/issues/1283)) and may amend
the packaging parts of this record.

## Context

The per-index computations behind `climate_indices` — distribution fits, inverse
normal transforms, empirical ranking, the recursive and backtracking state machines
of the Palmer family, and the fire-weather recurrences — are dominated by scalar
loops over long time series and grid cells that Python and NumPy do not express
efficiently. A native backend could remove that cost.

The risk is not the Rust code; it is the contract. `climate_indices` is cited in
published work and used as a reference implementation (see `VALIDATION.md`), so the
public Python API, the raised exceptions, the warnings, and above all the numerical
results are the product. A rewrite that changes a result in the fourth decimal, or
that makes an installed extension silently alter behaviour, breaks the citation
contract more thoroughly than leaving the performance on the table.

## Decision

1. **Python remains the public API and citation contract.** Users keep importing and
   calling `climate_indices` (`indices.spi`, `indices.spei`, `palmer.pdsi`,
   `climate_indices.fire.*`, the xarray adapter, the CLI) exactly as before. Public
   signatures, return values, raised exceptions, and `climate_indices` warnings do
   not change, and nothing observable depends on whether the extension is installed.

2. **The Rust code is split into `crates/climate-core` and `crates/climate-py`.**
   `climate-core` is pure numerical Rust (`ndarray`): no PyO3, no NumPy bindings, no
   Python exceptions or CPython assumptions. It receives already-validated numbers
   and returns numbers, and is not published to crates.io. `climate-py` is the only
   crate that knows about Python; it builds the `cdylib` imported as
   `climate_indices._native` and holds conversion and binding code only, no
   algorithm. `src/climate_indices/_native.pyi` is the extension's type stub.

3. **Orchestration stays in Python.** Validation at the public API boundary,
   calibration-period resolution, data-quality and goodness-of-fit warnings,
   provenance, structlog events, xarray/Dask orchestration, CF metadata, the CLI,
   and all I/O run in Python, on both backends. A port replaces a numerical kernel,
   not the application around it.

4. **The Python implementations stay and remain the parity oracle.** They are kept
   alive and directly testable after a port and are what the Rust kernels are
   compared against; retiring them needs a separate, explicit decision. Tests pin
   the Python path with the `python_backend` fixture in `tests/conftest.py`. Parity
   is asserted at `rtol = atol = 1e-10` with identical NaN positions, and a port
   reproduces the Python numerics — operation order, and the NaN, zero, and
   edge-case semantics — rather than improving them.

5. **A tolerance exception has to be measured, not asserted.** Loosening the
   `1e-10` contract for a kernel requires a written justification in its ticket:
   the maximum absolute and relative error, where it occurs, which primitive
   diverges, and whether reproducing the Python numerical method closes the gap.
   Fixtures and existing reference tests are never edited to match Rust.

6. **Dispatch routes by input, never by fallback on failure.** The Python module
   that owns a computation imports `_native` inside `try`/`except ImportError` and
   routes to Rust only for a plain, aligned float64 `ndarray` in a supported layout
   whose prepared arguments the kernel accepts; anything else — unaligned or masked
   arrays, other dtypes, parameters that vary by year, NumPy floating-point error
   policies that report, Python 3.14 context-aware warnings — keeps the Python path.
   A runtime error raised by the extension propagates and is never silently retried
   in Python. Dispatch therefore cannot turn "the extension is broken" into "the
   answer changed", and the two paths cannot diverge without a parity test failing.

7. **Where SciPy evaluates a special function, port the routine SciPy evaluates.**
   The Cephes `igam`, `ndtri`, `ndtr`, and `lgam` behind `scipy.special` and
   `scipy.stats` are ported line by line into `climate-core/src/special/`, not
   replaced with a generic crate implementation, because the generic versions do not
   hold the parity contract in the transformed tails. SciPy's own build-dependent
   multiply-add fusion is reproduced by `special::mul_add` on aarch64.

8. **Hatchling stays the PEP 517 backend; maturin builds the extension for
   development.** The published sdist and wheel remain pure Python, install without
   a Rust toolchain, and run the Python implementations; developers with a Rust
   toolchain run `uv run maturin develop --release`. Whether released binary wheels
   carry `_native`, and what the supported install path without Rust is, is RUST-013's
   decision and is deliberately not made here.

## Consequences

- `docs/architecture.md` § *Optional Rust Backend* carries the kernel/seam table and
  the current dispatch rules; the porting checklist is in
  [`docs/development-guide.md`](../development-guide.md#porting-a-kernel-to-rust).
  Both are kept in step with the code as ports land.
- CI adds three jobs to the pure-Python legs: `rust` (fmt, clippy `-D warnings`,
  `cargo test --workspace`), `test-native` (extension built, pytest with
  `CLIMATE_INDICES_REQUIRE_NATIVE=1`, so a missing extension fails instead of
  skipping the parity suites), and `native-wheel` (build, install into a fresh venv,
  import from outside the checkout). `rust-toolchain.toml` pins the compiler so a
  new stable clippy lint cannot fail an unrelated change; the workspace is edition
  2024, so the minimum supported Rust is 1.85.
- Every kernel needs its own documentation block stating the Python source it ports,
  inputs, outputs, and its zero, NaN, invalid-input, and degenerate-column semantics
  (`crates/climate-core/src/gamma.rs` is the reference). The parity suites prove the
  documented semantics rather than restating them in prose.
- The Python implementations are now maintained in two places for every ported
  kernel: the oracle and its Rust counterpart. A numerical bug fix has to be applied
  to both, and a fix to only the Python side fails the parity tests.
- The extension only exists in a checkout that was built. Nothing user-facing may
  require it, so the pure-Python fallback is exercised by the default test legs on
  every event, and `docs/llms*.txt` regeneration is part of any change to a bundled
  document rather than a separate step.
- `maturin develop` copies `_native.*.so` into `src/climate_indices/`, where it
  survives `uv sync`; deleting it returns the checkout to pure Python. Contributors
  can therefore be running either backend without intending to.

## Alternatives considered

- **Fused kernels.** Merging the fit and transform into one Rust entry point would
  cut array conversions, but it moves calibration-period resolution, warnings, and
  output scaling into Rust, and it makes the port of each kernel's edge cases
  harder to review against its Python oracle. Rejected: the seam is where parity is
  checkable.
- **A `statrs`-based port.** Depending on a general statistics crate for the
  distribution fits would be far less Rust code, but `statrs` does not reproduce the
  algorithms SciPy evaluates, so the parity contract could not hold in the tails
  without either loosening it or reimplementing the same special functions anyway.
- **Maturin as the PEP 517 backend now.** Making the extension part of the default
  build would simplify the development story, at the cost of requiring a Rust
  toolchain for every `pip install` and of publishing binaries before the packaging
  policy is decided. Deferred to RUST-013.
- **A separate distribution for the optional extension.** Keeping the acceleration
  out of `climate_indices` entirely would keep the main distribution provably pure
  Python and make the extension opt-in by install, but it splits the parity tests
  and the version contract across two artifacts and creates an import fallback that
  users cannot see. Not chosen; RUST-013 may still revisit it.

## References

- [#1270](https://github.com/monocongo/climate_indices/issues/1270) — RUST-000 epic;
  the child tickets and definition of done.
- `docs/architecture.md`, § *Optional Rust Backend* — crate layout, the Python
  seam/Rust kernel table, dispatch rules, and the CI jobs.
- `docs/development-guide.md`, § *Porting a Kernel to Rust* — the porting checklist.
- SciPy / Cephes: `scipy.special` and `scipy.stats`; the ported routines are Cephes
  `igam`, `igamc`, `ndtr`, `ndtri`, and `lgam` (Mosig & Cephes, *Math Library*).
