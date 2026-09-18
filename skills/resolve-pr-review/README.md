# resolve-pr-review

Resolve GitHub PR review comments and emit an auditable report. Portable across
repos: repo-specific commands live in `validation.yaml`, never in the skill.

## Install

1. Copy `skills/resolve-pr-review/` into your harness skill directory, e.g.
   `~/.pi/agent/skills/resolve-pr-review/` or a project-local
   `.agents/skills/resolve-pr-review/`.
2. Register `manifest.yaml` with the harness loader if it requires registration;
   the entrypoint is `prompt.md`.
3. Authenticate `gh`: `gh auth status` must succeed.

## Configure

- Copy [`validation.example.yaml`](validation.example.yaml) to the repo root as
  `validation.yaml` and adjust the profiles to the repo's commands.
- Select a profile per invocation with the `validation_profile` input; absent or
  unknown → `default`, and validation reports `partial` if the file is missing.
- `mode`: `fix` (default) edits, commits, and pushes to the PR head branch;
  `dry-run` reports planned fixes without editing; `report-only` classifies only.

## Usage

Local CLI (harness-specific syntax; inputs from the manifest):

```bash
agent run --skill resolve-pr-review \
  --input review_url="https://github.com/example-org/example-repo/pull/412#pullrequestreview-3312045678" \
  --input repo="example-org/example-repo" \
  --input mode=fix \
  --input validation_profile=default
```

CI trigger (GitHub Actions), on each submitted review:

```yaml
on:
  pull_request_review:
    types: [submitted]

jobs:
  resolve-review:
    if: github.event.review.user.login == 'review-bot'
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.ref }}
      - run: gh auth status
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      - run: |
          agent run --skill resolve-pr-review \
            --input review_url="https://github.com/${{ github.repository }}/pull/${{ github.event.pull_request.number }}#pullrequestreview-${{ github.event.review.id }}" \
            --input repo="${{ github.repository }}" \
            --input mode=fix
```

Scheduler mode: scan assigned PRs with `gh pr list --search "review-requested:@me"`,
fetch each PR's latest unresolved review, and invoke once per PR.

The skill never resolves GitHub threads automatically. Post the report with
`gh pr comment` when a human-visible record on the PR is wanted.

## Extend

- **Skip rules** — append to `SKILL.md` §8, reusing existing reason strings.
- **Validators** — add a named profile under `profiles:` in `validation.yaml`.
- **Output rendering** — consume `report.json` (schema-conformant) instead of
  `report.md`, or supply a custom template path in harness config.

## Files

| File | Purpose |
| --- | --- |
| `SKILL.md` | Human-readable contract and workflow |
| `manifest.yaml` | Harness metadata and input/output declarations |
| `prompt.md` | Runtime prompt injected by the harness |
| `schema.json` | JSON Schema for the machine-readable report |
| `validation.example.yaml` | Template for a repo's `validation.yaml` |
| `examples/` | Sample invocation and report |
