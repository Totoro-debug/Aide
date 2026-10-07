# Issue tracker: GitHub

[GitHub Issues](https://github.com/Totoro-debug/Aide/issues) are the authoritative source for product requirements and accepted discussion decisions. Local documents describe current vocabulary, usage, and consequential architectural trade-offs; Git and GitHub retain implementation history.

Use `gh` inside the clone; it infers the repository from the Git remote.

- Read requirements and decisions: `gh issue view <number> --comments`.
- List relevant work: `gh issue list --state open --json number,title,body,labels`, adding label/state filters as needed.
- Create a requested issue or PRD: `gh issue create --title "..." --body-file <utf8-file>`.
- Update labels: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Publish an authorized result: `gh issue comment <number> --body-file <utf8-file>`.
- Close completed work: `gh issue close <number>`.

For multiline bodies, write the exact text to a UTF-8 temporary file and use `--body-file`. Read the issue's accepted discussion as well as its opening body. Issues and PRs share a number space; resolve ambiguous references with `gh pr view <number>` and then `gh issue view <number>`. Read implementation changes with `gh pr diff <number>`.
