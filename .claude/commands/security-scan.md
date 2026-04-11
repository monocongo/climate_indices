Perform a security-focused review of: $ARGUMENTS

If no arguments are given, scan all Python files in `src/` and `tests/`.

## Automated checks

Run: `uv run ruff check --select S src/` (bandit-equivalent security rules)

## Manual review checklist

Search for and flag:

1. **Logged data values** — structlog calls that include array contents, coordinate values, or user-provided data
2. **Hardcoded credentials or API keys** — any string that looks like a token, password, or secret
3. **Unsafe deserialization** — `pickle.loads()` called on untrusted or externally-sourced input
4. **Path traversal** — file paths constructed via string interpolation or concatenation without validation
5. **Code execution on untrusted input** — `eval()`, `exec()`, `subprocess` with shell=True on user input

## Output format

Report findings grouped by severity:

**Critical** (exploitable, fix immediately):
**Warning** (potential risk, review carefully):
**Info** (informational, low risk):

If no findings, report: **No security issues found.**
