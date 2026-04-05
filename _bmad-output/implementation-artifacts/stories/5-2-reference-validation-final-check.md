# Story 5.2: Reference Validation Final Check

Status: done

## Story

As a v2.4.0 release gatekeeper,
I want to verify that NOAA EDDI reference test infrastructure is on the integration branch and provenance metadata is complete,
so that FR-EDDI-001 (scientific reproducibility) and NFR-EDDI-VAL are satisfied before the v2.4.0 release.

## Acceptance Criteria

1. `tests/test_noaa_eddi_reference.py` is present on the integration branch (cherry-picked from `feature/epic-3-eddi-pnp-scpdsi` via commit `1e16916`)
2. EDDI self-consistency tests always pass (no external data required):
   - `TestEddiSelfConsistency::test_eddi_output_range`
   - `TestEddiSelfConsistency::test_eddi_scale_consistency`
   - `TestEddiSelfConsistency::test_eddi_constant_input`
3. NOAA EDDI reference tests gracefully skip when fixture data absent OR pass at 1e-5 when data present — zero FAILURES allowed:
   - `TestNoaaEddiReference::test_eddi_noaa_reference_1month`
   - `TestNoaaEddiReference::test_eddi_noaa_reference_3month`
   - `TestNoaaEddiReference::test_eddi_noaa_reference_6month`
4. Provenance files present on integration branch (cherry-picked from commit `41d68db`):
   - `tests/fixture/provenance_schema.json`
   - `tests/fixture/palmer/provenance.json`
5. `tests/test_eddi.py` (currently untracked) is committed to the integration branch
6. `uv run pytest tests/test_eddi.py tests/test_noaa_eddi_reference.py -v` exits 0 (pass or skip, no failures)

## Tasks / Subtasks

- [ ] Verify prerequisite: `src/climate_indices/eddi.py` exists on integration branch (AC: #1)
  - [ ] If absent, cherry-pick EDDI implementation from epic-3 before test infra
- [ ] Cherry-pick provenance protocol commit (AC: #4)
  - [ ] `git cherry-pick 41d68db`
- [ ] Cherry-pick NOAA reference test infrastructure (AC: #1, #2, #3)
  - [ ] `git cherry-pick 1e16916`
- [ ] Commit untracked test_eddi.py (AC: #5)
  - [ ] `git add tests/test_eddi.py && git commit -m "test(eddi): add EDDI unit tests (Story 5.2)"`
- [ ] Run EDDI test suite (AC: #2, #3, #6)
  - [ ] `uv run pytest tests/test_eddi.py tests/test_noaa_eddi_reference.py -v --tb=short`
  - [ ] Self-consistency tests (3) must PASS; reference tests may SKIP — zero FAILURES
- [ ] Run full regression suite
  - [ ] `uv run pytest tests/ -v --tb=short` — no regressions from cherry-picks
- [ ] Update sprint-status.yaml: `5-2-reference-validation-final-check` → `done` (after completion)

## Dev Notes

**This is NOT greenfield work.** Stories 3.2 and 3.3 created the reference validation infrastructure on `feature/epic-3-eddi-pnp-scpdsi`. Story 5.2 is the integration branch merge step — exactly like Story 5.1 was for epic-1.

**NOAA data availability:** Commit `1e16916` notes that the NOAA PSL archive was inaccessible at time of commit ("proxy restrictions"). The test infrastructure handles this gracefully with `pytest.skip()`. The release gate is met when:
- Self-consistency tests pass ✅
- Reference tests skip (not fail) when data absent ✅

**test_noaa_eddi_reference.py structure (from commit 1e16916):**
- `TestNoaaEddiReference` (9 tests) — skip when `tests/fixture/noaa-eddi-*/` absent
- `TestEddiSelfConsistency` (3 tests) — always run, validate mathematical properties (output range, scale consistency, constant input behavior)

**test_eddi.py is untracked** (15 tests, 482 lines). It covers EDDI computation, validation, edge cases (all-NaN, negative values, invalid shape, daily periodicity). Commit it directly — do NOT cherry-pick.

**Palmer provenance (commit 41d68db):**
- `tests/fixture/provenance_schema.json` — JSON Schema draft 2020-12 for provenance validation
- `tests/fixture/palmer/provenance.json` — checksum for division 0101, citation, tolerance 1e-8

**Cherry-pick order:** oldest to newest — `41d68db` then `1e16916`.

### Common Failure Modes

1. **`eddi` module missing:** `test_noaa_eddi_reference.py` imports `from climate_indices.eddi import eddi`. If import fails, cherry-pick EDDI implementation commits from epic-3 first (working backwards: `aadb088`, `1e16916`, `f889c82`, etc.)
2. **Conflict in `tests/fixture/`:** Accept incoming version of provenance files.
3. **Self-consistency test failure:** Indicates EDDI algorithm regression. Do NOT relax assertions — find the root cause in the cherry-picked EDDI implementation.

### Project Structure Notes

- NOAA reference tests: `tests/test_noaa_eddi_reference.py` (to be cherry-picked from `1e16916`)
- EDDI unit tests: `tests/test_eddi.py` (untracked, commit directly)
- Provenance schema: `tests/fixture/provenance_schema.json` (to be cherry-picked from `41d68db`)
- Palmer provenance: `tests/fixture/palmer/provenance.json` (to be cherry-picked from `41d68db`)
- Fixture prep script: `scripts/prepare_noaa_eddi_fixtures.py` (to be cherry-picked from `1e16916`)
- Epic-3 branch: `feature/epic-3-eddi-pnp-scpdsi`

### References

- Story 3.2 commit: `41d68db` — `[Source: git log]`
- Story 3.3 commit: `1e16916` — `[Source: git log]`
- NOAA data note: commit message for `1e16916` — `[Source: git show 1e16916]`
- NFR-EDDI-VAL (1e-5 tolerance): `[Source: _bmad-output/planning-artifacts/epics.md#Epic-5-Story-5.2]`
- Epic 5 story definitions: `[Source: _bmad-output/planning-artifacts/epics.md#Story-5.2]`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (us.anthropic.claude-sonnet-4-6)

### Debug Log References

### Completion Notes List

### File List
