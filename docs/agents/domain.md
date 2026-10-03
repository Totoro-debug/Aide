# Domain Documentation

MyClaw has one bounded context. Before exploring a feature, read the relevant definitions in [CONTEXT.md](../../CONTEXT.md) and current decisions in [docs/adr/](../adr/).

- [GitHub Issues](https://github.com/Totoro-debug/OmniAgent/issues) are the authoritative product requirements and discussion history. Follow [issue-tracker.md](issue-tracker.md) to retrieve the relevant issue and its accepted decisions.
- [CONTEXT.md](../../CONTEXT.md) defines domain vocabulary, without API inventories or implementation plans.
- `docs/adr/` records current architectural decisions and their consequences. Consolidate still-valid decisions when replacing an older design; use Git and GitHub for historical versions.
- [README.md](../../README.md) is the installation, configuration, and usage guide.

Current architecture starts with [shared service ownership](../adr/0029-host-cli-and-web-through-one-local-service.md), [Agent Home and Workspace storage](../adr/0001-file-first-local-persistence.md), and [run-local Tool authorization](../adr/0026-tool-permission-levels-and-foreground-snapshots.md). CLI and Web are service clients; each Workspace has shared Memory and Schedule resources and independently claimed Sessions. Read the focused ADR for the feature being changed rather than reconstructing current ownership from historical issue descriptions.

Use glossary terms consistently in issues, proposals, code, and tests. Trace uncertain behavior through the implementation and relevant issue before resolving a conflict. Surface unresolved design choices to the user instead of silently selecting an obsolete document as authority.

Keep local documents focused on current information. Do not retain duplicate PRDs, completed implementation plans, migration ledgers, or historical release counts as active contracts. Tests should verify behavior, architecture, persistence, and documentation links rather than require old prose or deleted documents.
