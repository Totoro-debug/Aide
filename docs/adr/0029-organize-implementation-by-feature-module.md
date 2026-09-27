---
status: accepted
---

# Organize Implementation by Feature Module

MyClaw places each implementation in the Python package that owns its feature.
Internal callers import symbols from their defining modules, not from aggregate
package exports. Each implementation has one canonical module path. The old
paths for moved implementations under `myclaw.agent` and at
`myclaw.utils.scheduler` have no forwarding modules, duplicate definitions, or
`sys.modules` aliases. Callers using those internal paths must update imports.

The definition paths below are relative to `myclaw`.

| Package | Responsibility and canonical definitions |
| --- | --- |
| `myclaw.agent` | Foreground orchestration, ReAct execution, Message Bus, and Task Framing: `agent.loop.AgentLoop`, `agent.runner.AgentRunner`, and `agent.blackboard.Blackboard`. |
| `myclaw.context` | Model request construction, token budgeting, and run-local preparation: `context.builder.ContextBuilder`, `context.budget.ContextBudget`, and `context.controller.AgentRunContextController`. |
| `myclaw.session` | Conversation Session state, Restore Anchors, File Backup, and Session Restore: `session.session.Session`, `session.backup_store.FileMutationRecorder`, and `session.restore.RestoreManager`. |
| `myclaw.memory` | Workspace memory records, persistence, snapshots, and Dream: `memory.manager.MemoryManager` and `memory.dream.Dream`. |
| `myclaw.workspace` | Workspace-owned state paths and validation: `workspace.state.WorkspaceState`. |
| `myclaw.permission` | Runtime permission state, Tool authorization policy, and confirmation coordination: `permission.state.RuntimePermissionControl`, `permission.policy.ToolPermissionPolicy`, and `permission.confirmation.ToolConfirmationCoordinator`. |
| `myclaw.tools` | Tool invocation, preparation, Schema, and the Schedule model Tool adapter: `tools.gateway.ToolGateway`, `tools.base.BaseTool`, `tools.schema`, and `tools.schedule`. |
| `myclaw.schedule` | Schedule Job models, persistence, service, and scheduler clock: `schedule.service.ScheduleService` and `schedule.clock.AsyncioSchedulerClock`. |

Within `myclaw.tools`, `files` owns File Tools and shared file-path helpers;
`exec` owns the Exec Tool, Host, and command policy; `web` owns Web Search, Web
Fetch, and network safety; `mcp` owns MCP Tool adaptation, runtime connections,
and keywords; and `discovery` owns deferred Tool Search and activation. These
are defining packages, not compatibility facades. The Schedule Tool remains in
`myclaw.tools.schedule` because it adapts a model invocation to the Schedule
Service; Schedule Job behavior remains in `myclaw.schedule`.

The existing `config`, `provider`, `management`, `terminal`, `logging`, `skills`,
`templates`, and `utils` packages retain their feature or shared-helper roles.
In particular, `logging.session` owns technical Session Log output, while
Conversation Session state belongs to `myclaw.session`. Package `__init__.py`
files do not aggregate business symbols; the root package version and the
existing `templates` resource-loading entry point remain exceptions.

Package ownership does not change object lifetime or call boundaries. The CLI
remains the sole composition root, and each Agent Loop remains a Session-scoped
Runtime Generation under [ADR-0017](0017-use-cli-composition-root-and-session-scoped-agent-loop.md).
Tool Gateway remains the sole public Tool invocation boundary under
[ADR-0010](0010-fixed-tool-catalog-and-base-tool-boundaries.md), and the CLI
continues to own one confirmation coordinator for the Runtime Lifetime under
[ADR-0027](0027-runtime-lifetime-tool-confirmation-coordinator.md). Existing
local delayed imports retain their initialization role; package grouping does
not introduce a new runtime aggregate.

This source-layout decision does not change `.myclaw/` or Agent Home paths,
stored fields, persistence order, or recovery behavior. It requires no data
migration. Implementation: [module-layout issue #272](https://github.com/Totoro-debug/MyClaw/issues/272).
