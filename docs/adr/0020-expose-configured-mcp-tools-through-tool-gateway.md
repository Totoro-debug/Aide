---
status: accepted
---

# Use One Tool Gateway with Run-Local Exposure

Built-in and configured MCP Tools share one invocation and result boundary. MCP extends the available catalog without adding an authorization path. Complete schemas are exposed progressively to reduce request size and avoid accumulating schemas across a long Session.

## Invocation and exposure

`ToolGateway.call()` parses Provider arguments, resolves a Tool, prepares detached invocation facts, obtains call-local authorization, executes, and normalizes the result. Built-in Tools retain local casting and validation. MCP Tools forward the complete argument object using the discovered schema, without silently filtering remote parameters. Authorization follows [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md).

Each Agent Run owns its catalog view and activation state. Initial exposure includes File Tools, Exec, and Tool Search; search activates deferred capabilities for subsequent requests in that Run. Activation expires at the terminal boundary and is never restored from Session history. Concurrent Runs may share Tool implementations without sharing activation or mutable Workspace context.

Exposure is a context choice, not an execution gate. A catalogued Tool remains callable through the normal Gateway even before its schema is exposed; calling it does not implicitly activate it. User Schedule catalogs exclude the Schedule Tool entirely, preventing search or direct invocation from managing Jobs. Dream uses its private `edit_file` capability.

Tool Search queries the existing catalog using English BM25 keywords. It performs no MCP discovery. MCP keyword examples are in the [configuration template](../../aide/templates/default-config.md). Missing keywords are generated through the existing chat route and persisted best effort; generation failure uses the remote Tool name in memory without blocking startup. A changed exposure revises the complete request and is checked against [ADR-0023](0023-manage-agent-run-context-by-projected-token-budget.md) before the next model call.

## MCP lifecycle and results

Service-global HTTP and Workspace-scoped stdio connections follow [ADR-0029](0029-host-cli-and-web-through-one-local-service.md). Discovery produces immutable Tool snapshots; there is no live list subscription or standalone refresh. Unusable Servers or Tools are omitted with sanitized diagnostics so one failure does not block other capabilities.

MCP uses the official Python SDK's stdio and Streamable HTTP transports. Only `CallToolResult.content` enters the text Tool boundary; structured content and output-schema validation are excluded. Remote error text remains model/session data and is not treated as a sanitized operational diagnostic. Nullable normalization changes schema nodes without rewriting literal data.

There is no generic Tool retry or Gateway-wide execution lock. MCP timeouts and ordinary execution failures become Tool Errors; confirmed transport closure affects later connection preparation rather than mutating the active catalog. Cancellation propagates without undoing completed side effects. Oversized successful results become Workspace-owned Tool Artifacts; errors and refusals stay inline, and artifact-write failure preserves the successful outcome with a bounded marker.
