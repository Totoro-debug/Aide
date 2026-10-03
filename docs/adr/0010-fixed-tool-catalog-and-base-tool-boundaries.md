---
status: accepted
---

# Fix the Built-in Tool Catalog and BaseTool Boundaries

The Runtime Generation Tool Catalog contains these ten Built-in Tools in fixed order: Read File, Write File, Edit File, List Dir, Glob, Grep, Exec, Web Search, Web Fetch, and Schedule. Configured MCP Tools follow them as defined by [ADR-0020](0020-expose-configured-mcp-tools-through-tool-gateway.md). Foreground and User Schedule Agent Run Tool Catalogs add the run-local Built-in Tool Search; the User Schedule Catalog also excludes Schedule, as defined by [ADR-0021](0021-defer-tool-schema-exposure-per-agent-run.md). User Configuration cannot enable, disable, register, or replace any Built-in capability.

`ToolGateway.call()` is the sole public invocation boundary. It parses raw Provider arguments, resolves a Tool, calls the final `BaseTool.prepare()` pipeline, obtains any one-shot confirmation, and invokes `execute_authorized(arguments, authorization)` before normalizing the result; the default authorized seam delegates to `execute_prepared(arguments)`. Built-in preparation casts, defaults, filters, and validates with a temporary restricted Schema; MCP preparation preserves the complete argument object without those transformations. Each Tool exposes a `parameters` dictionary, and `to_schema()` returns a detached projection whenever that Tool is exposed in a Model request.

`BaseTool.prepare()` returns one detached `ToolInvocationFacts` value after argument normalization, validation, and capability-specific fact collection. This structured contract supersedes the former tuple and free-form safety-reason return contract. `ToolPermissionPolicy` is the only authorization decision point; Tools do not return free-form confirmation reasons or open an alternate authorization path. The Gateway retains the call-local authorization session through execution so Host-backed Tools can authorize each audited hop with the same typed state.

File Tools use normal Workspace path resolution, including Workspace State. File, PowerShell, Git, MCP, Schedule, and Web authorization are defined together in [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md).

Exec remains a fixed catalogued capability backed by a Workspace-generation Host. Windows `auto` selects available PowerShell 7 (`pwsh`) and otherwise Windows PowerShell 5.1; explicit `powershell` and `pwsh` never cross-fallback. Inspection and execution share the resolved executable, canonical cwd, minimal environment, timeout/cancellation handling, and `-NoLogo -NoProfile -NonInteractive` flags. The Host returns raw outcomes and typed assessments; only the Exec Tool/Gateway boundary creates Tool Results. A missing selected shell produces a sanitized diagnostic, remains catalogued, and returns a stable capability error at invocation. Inspector failure retains conservative confirmation semantics. Host boundaries follow [ADR-0007](0007-use-host-adapters.md), and resource ownership follows [ADR-0029](0029-host-cli-and-web-through-one-local-service.md). MyClaw does not claim process-tree ownership or OS-level filesystem, network, or process isolation.

`BaseTool` externalizes oversized successful results beneath `.myclaw/artifacts/<session_id>/<tool_call_id>.txt`, using a UUID fallback for an invalid call ID. Artifacts have no separate module, commit, rollback, cleanup, or ownership lifecycle.

ADR-0016 defines the Skill Loader's internal confirmation-free filesystem boundary; model-issued File accesses follow [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md). This does not add or dynamically register a Tool.

Tool execution has no generic retry or Gateway-wide lock. Successful results exceeding the configured character limit (default 4096) are externalized; errors and refusals remain inline. An artifact write failure retains success with a bounded failure marker. Tool Result content is model/session data, not a sanitized terminal diagnostic: trusted MCP `isError` text is retained, while foreground Tool activity carries status only.
