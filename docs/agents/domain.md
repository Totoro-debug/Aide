# Domain Documentation

Omni has one bounded context. Use these sources:

- [CONTEXT.md](../../CONTEXT.md): domain vocabulary.
- [README.md](../../README.md): installation, configuration, and current user behavior.
- [GitHub Issues](https://github.com/Totoro-debug/OmniAgent/issues): product requirements, accepted discussions, and delivery history; retrieve them using [issue-tracker.md](issue-tracker.md).
- [ADR](../adr/): architectural trade-offs that still constrain changes.

Read only the decisions relevant to the change: [service ownership](../adr/0029-host-cli-and-web-through-one-local-service.md), [local storage](../adr/0001-file-first-local-persistence.md), [Tool invocation and exposure](../adr/0020-expose-configured-mcp-tools-through-tool-gateway.md), [context budgeting](../adr/0023-manage-agent-run-context-by-projected-token-budget.md), [authorization](../adr/0026-tool-permission-levels-and-foreground-snapshots.md), or [Restore transactions](../adr/0028-session-restore-architecture.md). Trace uncertain behavior through current code and the relevant issue rather than treating an old design as authority.

Keep one home for each fact. An implemented feature does not need its own ADR when README or CONTEXT already explains it. Retain an ADR only for a consequential trade-off whose rationale would otherwise be unclear. Remove replaced designs, completed plans, duplicate requirements, API inventories, and migration ledgers; Git and GitHub retain history. Tests should check behavior and valid document links rather than freeze prose or require deleted documents.
