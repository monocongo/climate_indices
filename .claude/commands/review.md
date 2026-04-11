Review the code changes specified by: $ARGUMENTS

If no arguments are given, review all uncommitted changes (`git diff HEAD`).

Steps:
1. Get the diff: run `git diff HEAD` (or `git diff $ARGUMENTS` if a commit/branch is specified)
2. Invoke the `code-reviewer` agent with the diff as context
3. Output the findings using the Blocking / Advisory / Nitpick format

End with **LGTM** if there are no blocking issues.
