# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues (`monocongo/climate_indices`). Use the `gh` CLI for all operations.

Write actions below (close, label, assign) are for issues/PRs the current task actually concerns — don't close, relabel, or reassign unrelated tickets without maintainer direction.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."` (use a heredoc for multi-line bodies). `--template` cannot be combined with `--body`, so it only works interactively; for a scripted create, mirror the headings in `.github/ISSUE_TEMPLATE/Bug_report.md` / `Feature_request.md` in the body instead.
- **Read an issue**: `gh issue view <number> --json number,title,body,state,labels,comments --jq '{number, title, body, state, labels: [.labels[].name], comments: [.comments[].body]}'` (`--comments` is a separate human-readable flag and cannot be combined with `--json`/`--jq`).
- **List issues**: `gh issue list --limit 200 --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`, narrowing with `--label`/`--state`/`--search` (`--limit` defaults to 30, which silently truncates a backlog this repo's size).
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`. Run `gh label list` for the current set; beyond GitHub's defaults this repo uses `epic:*`, `type:*`, `status:*` (`status:blocked`/`in-progress`/`in-review`/`pending-data`), and `ready-for-agent`/`ready-for-human`.
- **Close**: `gh issue close <number> --comment "..."`

`gh` infers the target repo from the clone's git remotes automatically; pass `-R owner/repo` (or run `gh repo set-default`) only when a clone has several remotes and the default is ambiguous. GitHub shares one number space across issues and PRs, so a bare `#42` may be either — resolve with `gh issue view 42` and fall back to `gh pr view 42`.

## Pull requests as a triage surface

**PRs as a triage surface: no.** _(A maintainer edits this line in this file to `yes` if this repo starts treating external PRs as feature requests; until then, the rest of this section does not apply.)_

When set to `yes`, PRs run through the same labels and states as issues, using the `gh pr` equivalents:

- **Read a PR**: `gh pr view <number> --comments` and `gh pr diff <number>` for the diff.
- **List external PRs for triage**: `gh pr list --json number,title,body,labels,author,comments` does not expose author association (`Unknown JSON field: "authorAssociation"`) — use `gh api repos/{owner}/{repo}/pulls --jq '.[] | select(.author_association | IN("CONTRIBUTOR","FIRST_TIME_CONTRIBUTOR","FIRST_TIMER","NONE"))'` (drop `OWNER`/`MEMBER`/`COLLABORATOR`).
- **Comment / label / close**: `gh pr comment`, `gh pr edit --add-label`/`--remove-label`, `gh pr close`.

## Skill vocabulary (phrasing from optional agent skills, not repository policy)

- **"publish to the issue tracker"** → create a GitHub issue.
- **"fetch the relevant ticket"** → run `gh issue view <number> --json number,title,body,state,labels,comments` (see Conventions above for the `--jq` filter).

## Wayfinding operations (optional — only when using the `/wayfinder` skill)

Used by `/wayfinder`. The **map** is a single issue with **child** issues as tickets.

- **Map**: a single issue labelled `wayfinder:map`, holding the Notes / Decisions-so-far / Fog body. `gh issue create --title "..." --body "..." --label wayfinder:map`.
- **Child ticket**: an issue linked to the map as a GitHub sub-issue with `gh issue create --parent <map> --title "..." --body "..." --label wayfinder:<type>` (or `gh issue edit <map> --add-sub-issue <child>` for an existing issue). Where sub-issues aren't enabled, add the child to a task list in the map body and put `Part of #<map>` at the top of the child body. Labels: `wayfinder:<type>` (`research`/`prototype`/`grilling`/`task`). Once claimed, the ticket is assigned to the driving dev.
- **Blocking**: GitHub's **native issue dependencies** — the canonical, UI-visible representation. Add an edge with `gh issue edit <child> --add-blocked-by <blocker-number>` (a bare issue number, e.g. `42`; no database-id lookup needed). `gh issue view <child> --json blockedBy --jq '[.blockedBy.nodes[] | select(.state=="OPEN")] | length'` reports open blockers — `blockedBy.totalCount` alone also counts closed ones, so it is not the live gate. Where dependencies aren't available, fall back to a `Blocked by: #<n>, #<n>` line at the top of the child body. A ticket is unblocked when every blocker is closed.
- **Frontier query**: list the map's children with `gh issue view <map> --json subIssues --jq '.subIssues.nodes[].number'`, drop any with an open blocker (`gh issue view <child> --json blockedBy,assignees --jq '([.blockedBy.nodes[] | select(.state=="OPEN")] | length), (.assignees | length)'` — either non-zero), or fall back to `Blocked by` line parsing where dependencies aren't available; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me` — the session's first write.
- **Resolve**: `gh issue comment <n> --body "<answer>"`, then `gh issue close <n>`, then append a context pointer (gist + link) to the map's Decisions-so-far.
