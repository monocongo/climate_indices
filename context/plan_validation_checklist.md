# Solution Verification Checklist

Verify your plan against this checklist before implementation.

## Root Cause & Research

- [ ] Identified root cause, not symptoms
- [ ] Researched scientific/numerical best practices
- [ ] Analyzed existing codebase patterns
- [ ] Reviewed related climate index literature if applicable

## Architecture & Design

- [ ] Evaluated current architecture fit
- [ ] Recommended changes if beneficial
- [ ] Identified technical debt impact
- [ ] Challenged suboptimal patterns
- [ ] Honest assessment (not a yes-man)

## Solution Quality

- [ ] AGENTS.md compliant
- [ ] Simple, streamlined, no redundancy
- [ ] 100% complete (not 99%)
- [ ] Best solution with trade-offs explained
- [ ] Prioritized long-term maintainability

## Scientific Accuracy

- [ ] Numerical stability verified (NaN, Inf handling)
- [ ] Edge cases handled (empty arrays, single values)
- [ ] Unit conversions correct (if applicable)
- [ ] Distribution fitting follows established methods
- [ ] Results validated against reference implementations

## Security & Safety

- [ ] No security vulnerabilities introduced
- [ ] Input validation added for user-facing functions
- [ ] File paths sanitized (no path traversal)
- [ ] No sensitive data logged

## Integration & Testing

- [ ] All upstream/downstream impacts handled
- [ ] All affected files updated
- [ ] Consistent with existing patterns
- [ ] Fully integrated, no silos
- [ ] Tests with edge cases added
- [ ] Type hints complete

## Technical Completeness

- [ ] Dependencies updated if needed (pyproject.toml)
- [ ] Documentation updated if API changed
- [ ] CLI arguments documented if modified
- [ ] Performance analyzed for large datasets

---

## ANALYZE ALL ITEMS IN THIS CHECKLIST ONE BY ONE. ACHIEVE 100% COVERAGE.

## Process: READ -> RESEARCH -> ANALYZE ROOT CAUSE -> CHALLENGE -> THINK -> RESPOND
