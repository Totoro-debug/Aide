---
status: accepted
---

# Ignore Unknown User Configuration Fields

User Configuration loading and `/config` interpret the document as a projection of fields understood by the running Omni version. Unknown fields and static tables, including removed settings such as `compaction_message_threshold`, are ignored so configuration can survive upgrades, downgrades, and stale keys; the raw file is not rewritten merely because such a field is present. Known fields still apply their declared validation; a missing required field uses its declared default or fails when no default exists. Malformed TOML still fails, every declared Provider remains eagerly validated, and an invalid MCP Server retains its existing isolated diagnostic behavior.

Default-value fallback is one explicit contract for every defaultable leaf: `runtime.max_tool_result_chars` (`4096`), `runtime.max_iterations` (`50`), `runtime.enable_skill_always_load` (`false`), `runtime.compact_ratio` (`0.9`), `runtime.permission_level` (`workspace-write`), `runtime.exec_shell` (`auto`), `memory.batch_size` (`10`), `memory.schedule` (`0 * * * *`), and each `models.routes.*.reasoning_effort` (`mid`). A missing value silently uses its default. An explicitly invalid value uses its default and produces exactly one sanitized diagnostic containing only the field and effective default value; the invalid source value is never copied into that diagnostic. This supersedes the earlier compact-ratio-only fallback wording. `/config` exposes the effective defaultable runtime values before the redacted raw content, and startup reports the same diagnostics. No environment-variable overlay or Route-driven lazy configuration parsing is introduced. This trades typo detection for forward and backward compatibility; effective configuration and status views expose the value actually in use.

Requirements: [Multi-level Tool permissions](https://github.com/Totoro-debug/OmniAgent/issues/245), Phase 1 tracked by [T03](https://github.com/Totoro-debug/OmniAgent/issues/247).

## Structured Web editing and startup configuration

The tolerant loading policy does not permit an invalid Web edit to replace the configuration. The structured editor validates the complete candidate strictly, checks its source revision, and atomically publishes under the same Agent Home configuration lock used by Reasoning Effort and MCP keyword writes. Invalid candidates or conflicting revisions preserve the original bytes. Comments, unrelated fields, and secrets survive ordinary edits; saved credentials are projected only as presence state and can be changed through explicit replace or clear actions.

The local service tracks saved and startup revisions and reports restart-required state. Its startup configuration remains active throughout the service lifetime, including Workspaces activated later. Reading external edits, saving, repairing, and explicit chat Reasoning Effort persistence do not drain work, rebuild resources, or schedule activation. Missing or malformed startup configuration leaves Agent work disabled while Web setup or repair remains available; repair requires a subsequent service startup. Repair backs up an existing invalid document before replacement. Client Permission Level and global chat Reasoning Effort retain their runtime capture boundaries. See [ADR-0030](0030-service-owned-shared-resources-and-session-execution.md) and [T1](https://github.com/Totoro-debug/OmniAgent/issues/307).

Requirements: [Runtime and Memory settings](https://github.com/Totoro-debug/OmniAgent/issues/301), [Model and MCP settings](https://github.com/Totoro-debug/OmniAgent/issues/302), [first use and repair](https://github.com/Totoro-debug/OmniAgent/issues/303).
