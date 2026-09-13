# scPDSI PR3: Reference Oracle and Fixture Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate scPDSI, scPHDI, scPMDI, self-calibrated Z-index and fitted duration factors for all 344 climate divisions in `tests/fixture/palmer/`, from the GPLv3 reference C++ implementation built and run entirely outside this repository.

**Architecture:** A standalone `clang++` binary wraps the unmodified `Sibada/scPDSI` C++ sources, made compilable by four shim symbols installed through include shadowing rather than source edits. A Python driver converts `.npy` inputs to text, invokes the binary once per division per mode, and converts text output back to `.npy`. A blocking validation gate runs the whole corpus in standard-PDSI mode against the already-committed fixtures before any scPDSI array is written.

**Tech Stack:** `clang++` (C++11), Python 3.13 via `uv run`, numpy.

**Design doc:** [`docs/superpowers/specs/2026-08-08-scpdsi-pr3-oracle-harness-design.md`](../specs/2026-08-08-scpdsi-pr3-oracle-harness-design.md)

## Global Constraints

- **GPLv3 boundary (non-negotiable).** No GPL source, headers, shim code, build scripts, or `main.cpp` may enter the git repository. The oracle is built and run only under `$SCRATCH`. Only `.npy` numeric arrays and a `provenance.json` edit are committed. Nothing under `tests/` may reference the harness.
- `$SCRATCH` = `/private/tmp/claude-501/-Users-jadams-git-climate-indices--claude-worktrees-feature-scpdsi-2-selfcal-stats/350ed1e3-78ff-4221-b1c3-b32b954ba82c/scratchpad`
- `$REF` = `/private/tmp/claude-501/-Users-jadams-git-climate-indices/0ae5a498-414e-4e92-9405-12f585253f67/scratchpad` (downloaded reference source: `pdsi.cpp`, `pdsi.h`, `pdsi_ext.cpp`)
- **No `src/` changes. No new or modified tests.** Fixture data only.
- Fixed run parameters for every division: `data_start_year=1895`, `data_end_year=2022`, `calibration_year_initial=1931`, `calibration_year_final=1990`, 1536 monthly values, AWC in **inches** from `tests/fixture/palmer_awc.json`.
- All floating-point text serialization uses `%.17g` (C++) / `repr`-equivalent `"%.17g"` (Python) so the text round-trip is bit-exact.
- Validation tolerance is `atol=5e-5, rtol=0`, matching `tests/test_palmer.py`.
- `MISSING` sentinel is `-999.00`; it maps to `NaN` and is treated as a bug signal, not expected data.
- Run Python through `uv run`. Run `uv run ruff check --fix` on any Python file that ends up in the repo — note that no Python file is expected to.

## File Structure

**Scratchpad (never committed):**

| Path | Responsibility |
|---|---|
| `$SCRATCH/oracle/pdsi.{h,cpp}`, `pdsi_ext.cpp` | Reference source, copied byte-identical from `$REF`, plus one added accessor in `pdsi.h` (Task 1) |
| `$SCRATCH/oracle/Rcpp.h` | Shim: `Rcpp::NumericVector`, `Rcpp::NumericMatrix` |
| `$SCRATCH/oracle/R.h` | Shim: `Rf_error`, `Rf_warning` |
| `$SCRATCH/oracle/main.cpp` | CLI: parse `in.txt`, run, write `out.txt` |
| `$SCRATCH/oracle/build.sh` | `clang++` invocation |
| `$SCRATCH/drive.py` | `.npy` ⇄ text, orchestration, comparison, fixture writing |
| `$SCRATCH/work/<div>/{in.txt,std.txt,sc.txt}` | Per-division intermediates |
| `$SCRATCH/gate_report.md` | Validation gate results (Task 4) |

**Repository (committed):**

| Path | Responsibility |
|---|---|
| `tests/fixture/palmer/<div>/scpdsi.npy` | 1536 float64 |
| `tests/fixture/palmer/<div>/scphdi.npy` | 1536 float64 |
| `tests/fixture/palmer/<div>/scpmdi.npy` | 1536 float64 |
| `tests/fixture/palmer/<div>/sczindex.npy` | 1536 float64 |
| `tests/fixture/palmer/<div>/scdurfact.npy` | 4 float64: `[wetm, wetb, drym, dryb]` |
| `tests/fixture/palmer/provenance.json` | Updated provenance entry |

---

### Task 1: Build a working oracle binary

**Files:**
- Create: `$SCRATCH/oracle/Rcpp.h`, `$SCRATCH/oracle/R.h`, `$SCRATCH/oracle/main.cpp`, `$SCRATCH/oracle/build.sh`
- Copy: `$REF/{pdsi.h,pdsi.cpp,pdsi_ext.cpp}` → `$SCRATCH/oracle/`
- Modify: `$SCRATCH/oracle/pdsi.h` — add one public method, change no existing line

**Interfaces:**
- Consumes: nothing.
- Produces: an executable `$SCRATCH/oracle/oracle` with CLI `oracle <in.txt> <out.txt> --std|--sc`, and the `in.txt`/`out.txt` text formats specified in Step 4.

- [ ] **Step 1: Copy the reference source unmodified and record its hashes**

```bash
mkdir -p "$SCRATCH/oracle"
cp "$REF/pdsi.h" "$REF/pdsi.cpp" "$REF/pdsi_ext.cpp" "$SCRATCH/oracle/"
cd "$SCRATCH/oracle" && shasum -a 256 pdsi.h pdsi.cpp pdsi_ext.cpp | tee reference-hashes.txt
```

Keep `reference-hashes.txt`. At the end of Task 1 the hashes of `pdsi.cpp` and `pdsi_ext.cpp` must be unchanged; only `pdsi.h` may differ.

- [ ] **Step 2: Write the shim headers**

Write `$SCRATCH/oracle/Rcpp.h` and `$SCRATCH/oracle/R.h` to this exact contract. `pdsi.h` does `#include <Rcpp.h>`, `#include <R.h>`, `using namespace Rcpp;`, so compiling with `-I.` makes these shadow the real headers and leaves the reference source untouched.

`Rcpp.h` declares, inside `namespace Rcpp`:

- `class NumericVector`, backed by `std::vector<double>`:
  - `NumericVector()` — empty
  - `explicit NumericVector(int n)` — `n` elements, zero-initialized
  - `int length() const`
  - `double& operator[](int)` and `double operator[](int) const`
  - default copy-construct and copy-assign (`P_vec = P` in `Rext_init` relies on assignment)
- `class NumericMatrix`, backed by a single `std::vector<double>` in row-major order:
  - `NumericMatrix()` — 0×0
  - `NumericMatrix(int rows, int cols)` — zero-initialized
  - `int nrow() const`, `int ncol() const`
  - `double& operator()(int i, int j)` and `double operator()(int i, int j) const`
  - default copy-assign
- Nothing else. `Rcpp::List` is only used by `scpdsi.cpp`, which is not compiled.

`R.h` declares two printf-style variadic functions at global scope:

- `void Rf_error(const char* fmt, ...)` — format with `vsnprintf` into a fixed 1024-byte buffer, then `throw std::runtime_error(buf)`. Must be marked so the compiler knows it does not return is *not* required; `pdsi.cpp` does not depend on `noreturn`.
- `void Rf_warning(const char* fmt, ...)` — format with `vsnprintf`, then `fprintf(stderr, "warning: %s\n", buf)`.

Both headers need include guards and must be self-contained (`<vector>`, `<stdexcept>`, `<cstdio>`, `<cstdarg>`).

- [ ] **Step 3: Add the inch-input accessor to `pdsi.h`**

`Rext_init` hardcodes `metric = 1` and `AWC = o_AWC / 25.4`; `metric`, `AWC`, `Ss`, `Su` are private (declared after the `private:` at `pdsi.h:188`). Add this method to the **public** section of `class pdsi` (i.e. before line 188), adding lines only:

Add one public inline method, `void Rext_use_inches(number awc_inches)`, which must:

1. clear the `metric` flag, so `Rext_get_Rvec` stops dividing P/PE by 25.4;
2. assign `AWC` the inch value directly, bypassing `Rext_init`'s `o_AWC / 25.4`;
3. repeat `Rext_init`'s own surface/underlying soil-layer initialization verbatim, so the split
   is identical to what the reference would have produced.

The literal source is deliberately not reproduced in this plan: the reference is GPLv3 and this
repository is BSD-3-Clause, and the Global Constraints forbid shim code from entering the repo.
Copy step 3's lines from `Rext_init` in the out-of-tree working copy.

Verify with a plain diff that no existing line changed:

```bash
diff <(cat "$REF/pdsi.h") "$SCRATCH/oracle/pdsi.h"
```

Expected: a single `>`-only added hunk, no `<` lines.

- [ ] **Step 4: Write `main.cpp`**

`main.cpp` implements exactly this behavior:

1. `argv[1]` = input path, `argv[2]` = output path, `argv[3]` = `--std` or `--sc`. Any other argument count or mode string → print usage to stderr, `return 2`.
2. Read the input file:
   - Line 1: `start_year end_year calib_start_year calib_end_year awc_inches n_values` (4 ints, 1 double, 1 int).
   - Next `n_values` lines: `P PE` (two doubles each).
   - Any parse shortfall → stderr message, `return 3`.
3. Fill `Rcpp::NumericVector P(n_values)` and `PE(n_values)`.
4. Construct `pdsi PDSI;`, then in order:
   - `PDSI.Rext_init(P, PE, awc_inches * 25.4, start_year, end_year, calib_start_year, calib_end_year);`
     The `* 25.4` only feeds `Rext_init`'s internal `/25.4`; the next call overwrites `AWC` exactly, so this argument's rounding is discarded.
   - `PDSI.Rext_use_inches(awc_inches);`
   - `PDSI.Rext_PDSI_mon(mode == "--sc");`
   Wrap all of this in `try { ... } catch (const std::exception& e) { fprintf(stderr, ...); return 4; }` so a shimmed `Rf_error` becomes a nonzero exit rather than an abort.
5. Write the output file:
   - Line 1: `DURFACT <wetm> <wetb> <drym> <dryb>` — from `PDSI.Rext_out_params()`, whose indices are `0=wetm, 1=drym, 2=wetb, 3=dryb`. **Reorder** to `wetm, wetb, drym, dryb` on output.
   - Next `n_values` lines: `<Z> <PDSI> <PHDI> <PMDI>` from `vals_mat(n, 8)`, `(n, 13)`, `(n, 14)`, `(n, 15)`.
   - Every double printed with `%.17g`, space-separated.
6. `return 0`.

Note that `vals_mat` and `Rext_out_params()` are public members of `pdsi`, so no further accessor is needed.

- [ ] **Step 5: Write `build.sh` and compile**

```bash
#!/bin/sh
set -eu
cd "$(dirname "$0")"
clang++ -std=c++11 -O2 -I. -o oracle main.cpp pdsi.cpp pdsi_ext.cpp
```

Run: `sh "$SCRATCH/oracle/build.sh"`
Expected: compiles to an executable. Warnings from the 1990s-vintage reference source are acceptable; errors are not. Do **not** silence errors by editing `pdsi.cpp` — if it will not compile, the shim contract in Step 2 is wrong and that is what to fix.

- [ ] **Step 6: Confirm the reference source is still byte-identical**

```bash
cd "$SCRATCH/oracle" && shasum -a 256 -c reference-hashes.txt
```

Expected: `pdsi.cpp: OK` and `pdsi_ext.cpp: OK`. `pdsi.h` is expected to FAIL — that is the one intended edit from Step 3.

- [ ] **Step 7: Smoke-test on one division**

Hand-build an `in.txt` for division `0101` (AWC 6.0) with a throwaway Python one-liner, run both modes, and eyeball the output:

Run: `"$SCRATCH/oracle/oracle" /tmp/in0101.txt /tmp/std0101.txt --std && head -3 /tmp/std0101.txt`
Expected: exit 0, a `DURFACT` line, then 1536 rows of four finite numbers. In `--std` mode the duration factors are whatever the uncalibrated defaults are and are not meaningful; in `--sc` mode they must be finite and non-zero. PDSI values should land roughly in [-7, 7]. No `-999` anywhere.

- [ ] **Step 8: No commit**

Nothing from this task enters the repository. Confirm with `git status --short` — expected: clean.

---

### Task 2: Python driver with a bit-exactness proof

**Files:**
- Create: `$SCRATCH/drive.py`

**Interfaces:**
- Consumes: the `oracle` binary and text formats from Task 1.
- Produces, in `$SCRATCH/drive.py`:
  - `DIVISIONS: list[str]` — sorted numeric-named dirs under `tests/fixture/palmer/`
  - `write_input(div: str) -> Path` — writes `$SCRATCH/work/<div>/in.txt`, returns its path
  - `run_oracle(div: str, mode: str) -> Path` — `mode` in `{"std", "sc"}`; invokes the binary, raises on nonzero exit, returns the output path
  - `parse_output(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]` — returns `(durfact, {"zindex", "pdsi", "phdi", "pmdi"})`, each 1536 float64, with `-999.0` replaced by `NaN`
  - `REPO: Path`, `FIX: Path` — repository root and `tests/fixture/palmer`

- [ ] **Step 1: Write the failing round-trip check**

Add to `drive.py` a `check_roundtrip()` that proves the text layer loses nothing — this is the claim the whole harness's precision rests on:

```python
def check_roundtrip() -> None:
    """Assert the .npy -> text -> parse path reproduces inputs bit-exactly."""
    bad = 0
    for div in DIVISIONS:
        p = np.load(FIX / div / "precips.npy")
        pe = np.load(FIX / div / "pet.npy")
        path = write_input(div)
        lines = path.read_text().splitlines()[1:]
        rt_p = np.array([float(ln.split()[0]) for ln in lines])
        rt_pe = np.array([float(ln.split()[1]) for ln in lines])
        bad += int((rt_p != p).sum() + (rt_pe != pe).sum())
    if bad:
        raise SystemExit(f"text round-trip is lossy: {bad} values differ")
    print(f"round-trip bit-exact across {len(DIVISIONS)} divisions")
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run python "$SCRATCH/drive.py" roundtrip`
Expected: FAIL — `write_input` is not defined yet.

- [ ] **Step 3: Implement `write_input`, `run_oracle`, `parse_output`**

`write_input` writes the Task 1 line-1 header (`1895 2022 1931 1990 <awc> 1536`) with `awc` from `tests/fixture/palmer_awc.json`, then one `"%.17g %.17g"` line per month. `run_oracle` shells out with `subprocess.run(..., check=True, capture_output=True)` and surfaces stderr on failure. `parse_output` reads the `DURFACT` line into a 4-element array, the rest into a `(1536, 4)` block, splits the columns, and applies `arr[arr == -999.0] = np.nan`.

- [ ] **Step 4: Run the round-trip check and verify it passes**

Run: `uv run python "$SCRATCH/drive.py" roundtrip`
Expected: `round-trip bit-exact across 344 divisions`. If any value differs, the format string is wrong — fix it before continuing. Every downstream tolerance claim depends on this passing.

- [ ] **Step 5: No commit**

`git status --short` — expected: clean.

---

### Task 3: Blocking validation gate against the committed standard-PDSI fixtures

**Files:**
- Modify: `$SCRATCH/drive.py`
- Create: `$SCRATCH/gate_report.md`

**Interfaces:**
- Consumes: `run_oracle`, `parse_output` from Task 2.
- Produces: `run_gate() -> int` — returns the number of failing (division, series) pairs, and writes `$SCRATCH/gate_report.md`.

- [ ] **Step 1: Implement the gate**

```python
SERIES = {"pdsi": "pdsi.npy", "phdi": "phdi.npy", "pmdi": "pmdi.npy", "zindex": "zindex.npy"}
ATOL, RTOL = 5e-5, 0.0

def run_gate() -> int:
    rows, failures = [], 0
    for div in DIVISIONS:
        write_input(div)
        _, got = parse_output(run_oracle(div, "std"))
        for key, fname in SERIES.items():
            want = np.load(FIX / div / fname)
            ok = np.allclose(got[key], want, atol=ATOL, rtol=RTOL, equal_nan=True)
            dev = float(np.nanmax(np.abs(got[key] - want)))
            if not ok:
                failures += 1
                rows.append((div, key, dev))
    ...  # write gate_report.md: total compared, failures, worst deviations
    return failures
```

Report the max absolute deviation across all 344 × 4 comparisons even when everything passes — a gate that passes at 4.9e-5 is a very different signal from one that passes at 1e-12, and the difference matters for choosing PR4's tolerance.

- [ ] **Step 2: Run the gate**

Run: `uv run python "$SCRATCH/drive.py" gate`
Expected: `0 failures across 1376 comparisons`, plus a reported worst-case deviation.

- [ ] **Step 3: Halt if the gate fails**

If `run_gate()` returns nonzero: **stop. Do not proceed to Task 4.** Write up which divisions and series failed, the max deviation, and whether failures cluster (by state prefix, by AWC value, by series), and bring it to the user. A water-balance mismatch here would silently corrupt all 344 scPDSI fixtures.

- [ ] **Step 4: No commit**

`git status --short` — expected: clean.

---

### Task 4: Generate and commit the 344 scPDSI fixtures

**Files:**
- Modify: `$SCRATCH/drive.py`
- Create: `tests/fixture/palmer/<div>/{scpdsi,scphdi,scpmdi,sczindex,scdurfact}.npy` × 344

**Interfaces:**
- Consumes: `run_oracle`, `parse_output` from Task 2; a passing gate from Task 3.
- Produces: `generate() -> None` — writes all fixture arrays.

- [ ] **Step 1: Implement generation with sanity checks**

For each division: run `--sc`, parse, then **before writing anything**, check:

- every one of the four series is all-finite (`np.isfinite(...).all()`) — non-finite means `m + b` went to zero or negative, per the PR2 finding, and that division must be reported rather than written
- `durfact` is all-finite and `wetm + wetb != 0`, `drym + dryb != 0`
- no value equals `-999.0` after the `NaN` mapping
- `|scpdsi|` max is within a sane band (flag anything above 20 for inspection rather than silently committing it)

Accumulate violations and, if any exist, write nothing and report. Only when all 344 divisions pass every check does the function write the `.npy` files, with `np.save(..., arr.astype(np.float64))` and `scdurfact` ordered `[wetm, wetb, drym, dryb]`.

- [ ] **Step 2: Run generation**

Run: `uv run python "$SCRATCH/drive.py" generate`
Expected: `344 divisions written, 1720 files`, and a printed summary of the scPDSI value range and the duration-factor ranges across all divisions.

- [ ] **Step 3: Verify what landed in the working tree**

```bash
git status --short | wc -l          # expect 1720
ls tests/fixture/palmer/0101/       # expect the 5 new sc* files alongside the originals
uv run python -c "
import numpy as np
a = np.load('tests/fixture/palmer/0101/scpdsi.npy')
d = np.load('tests/fixture/palmer/0101/scdurfact.npy')
print(a.shape, a.dtype, np.isfinite(a).all(), a.min(), a.max())
print('durfact', d)
"
```

Expected: `(1536,) float64 True` with a plausible range, and four finite duration factors.

- [ ] **Step 4: Confirm the existing fixtures were not disturbed**

```bash
git status --short | grep -v '^??' | wc -l
```

Expected: `0` — every change is a new untracked file. This task must not modify `pdsi.npy` or any other existing array.

- [ ] **Step 5: Confirm no GPL artifact leaked into the repo**

```bash
git status --short | grep -Ei '\.(cpp|h|sh)$' | wc -l
```

Expected: `0`.

- [ ] **Step 6: Commit**

```bash
git add tests/fixture/palmer
git commit -m "test: add scPDSI reference fixtures for 344 climate divisions"
```

---

### Task 5: Provenance entry

**Files:**
- Modify: `tests/fixture/palmer/provenance.json`

**Interfaces:**
- Consumes: the generated fixtures and the gate report.
- Produces: nothing downstream in this PR; PR4 reads the recorded tolerance.

- [ ] **Step 1: Read the schema and the existing entry**

`tests/fixture/provenance_schema.json` is `additionalProperties: false`. Permitted keys are exactly: `source`, `url`, `download_date`, `subset_description`, `checksum_sha256` (must match `^[a-f0-9]{64}$`), `fixture_version` (must match `^\d+\.\d+\.\d+$`), `validation_tolerance`, `citation`, `doi`, `license`, `notes`. Anything that does not fit a dedicated key goes in `notes`.

- [ ] **Step 2: Update the entry**

Keep the file a single object matching the existing shape. Required content:

- `subset_description` extended to cover the new scPDSI arrays alongside the existing standard-Palmer ones.
- `citation` — Wells, N., Goddard, S., and Hayes, M. J., 2004: A Self-Calibrating Palmer Drought Severity Index. *Journal of Climate*, 17(12), 2335–2351.
- `validation_tolerance` — `{"rtol": 0, "atol": 5e-5}`.
- `fixture_version` bumped (minor, layout gained files without changing existing ones).
- `notes` must record, honestly:
  - generated from the `Sibada/scPDSI` GPLv3 reference C++, built and run entirely outside this repository, with only numeric outputs committed
  - run parameters: 1895–2022, calibration 1931–1990, inch units, per-division AWC from `palmer_awc.json`
  - that the harness was cross-validated by running the same binary in standard-PDSI mode against the committed `pdsi`/`phdi`/`pmdi`/`zindex` arrays for all 344 divisions at `atol=5e-5`, with the observed worst-case deviation from Task 3
  - **that this cross-check is against `climate_indices`' own prior output, not an independent NOAA reference** — the existing entry describes the baseline as "climate_indices library reference output", and the new note must not overclaim it as external validation
  - that `scdurfact.npy` holds `[wetm, wetb, drym, dryb]`

- [ ] **Step 3: Validate against the schema**

```bash
uv run python -c "
import json, jsonschema
s = json.load(open('tests/fixture/provenance_schema.json'))
d = json.load(open('tests/fixture/palmer/provenance.json'))
jsonschema.validate(d, s)
print('provenance valid')
"
```

Expected: `provenance valid`. If `jsonschema` is unavailable, check the constraints by hand: no extra keys, `checksum_sha256` 64 lowercase hex, `fixture_version` three dot-separated integers, `download_date` `YYYY-MM-DD`.

- [ ] **Step 4: Run the existing test suite**

Run: `uv run pytest tests/test_palmer.py -q`
Expected: all pass, unchanged. This PR adds files that nothing reads yet, so a failure here means something was disturbed.

Then run the full suite: `uv run pytest -q`
Expected: same pass count as `main` (1101 at PR2's baseline, adjusted for whatever `main` is at).

- [ ] **Step 5: Commit**

```bash
git add tests/fixture/palmer/provenance.json
git commit -m "docs: record scPDSI fixture provenance and validation methodology"
```

---

## Final review

After Task 5, before opening the PR:

- [ ] Confirm the branch diff contains only `.npy` files, `provenance.json`, and the two `docs/superpowers/` markdown files: `git diff --stat origin/main...HEAD | tail -5` and `git diff --name-only origin/main...HEAD | grep -vE '\.npy$' `
- [ ] Confirm no `src/` file and no `tests/*.py` file was touched.
- [ ] Dispatch an Opus whole-branch review. **Ask explicitly for adversarial checks on the numeric output**, not just the diff shape — spot-check a few divisions by re-deriving expected behavior independently, confirm the sc and std runs actually differ (a harness bug that ignored the `--sc` flag would produce a clean-looking but worthless fixture set), and confirm the scPDSI series is not accidentally identical to the committed `pdsi.npy`.
- [ ] Address findings, then use `superpowers:finishing-a-development-branch`.

## Self-review notes

- **Spec coverage:** license boundary → Global Constraints + Task 4 Step 5; shim → Task 1 Steps 2–3; inch exactness → Task 1 Step 3 + Task 2 Step 4; output mapping → Task 1 Step 4; data flow → Task 2; gate → Task 3; deliverables → Task 4; provenance → Task 5; `m+b` risk → Task 4 Step 1; skipped-init risk → covered by the Task 3 gate.
- **Deliberate deviation from writing-plans:** the C++ in Tasks 1–2 is specified as an exact contract rather than literal source, because the Global Constraints forbid shim code and build scripts from entering the repository and this plan is committed. The literal source is written only under `$SCRATCH/oracle/`.
