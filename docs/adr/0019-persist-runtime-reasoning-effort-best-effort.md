---
status: accepted
---

# Persist Runtime Reasoning Effort Best-Effort After Memory Commit

Reasoning Effort has five levels: `low`, `mid`, `high`, `xhigh`, and `max`, with `mid` as the configuration default. Runtime selection applies to `chat` and `default` requests and survives Session replacement; explicitly configured `memory` and `schedule` routes retain their own values.

CLI and Web display these lowercase values directly in every interface language. Configuration uses its existing default-value policy for unsupported values. Persisted Session metadata (including Restore Anchor snapshots) and browser recovery records accept only the current five levels. Reads do not migrate or rewrite unsupported values. Anthropic and OpenAI-compatible adapters map `mid` to the existing Provider wire value `medium`; other levels retain their existing mappings.

The Workspace runtime's shared `ModelRouter` is the immediate authority for its current Reasoning Effort. A successful
`/effort` update publishes that in-memory value before it performs any User Configuration I/O. The Management
Port owns this ordering, so a configuration failure cannot prevent the next logical model request from using the
committed value or cancel an active Agent Run.

`ConfigLoader.update_reasoning_effort()` is a narrow domain operation. It rereads the latest `config.toml`, uses a
`tomlkit` round-trip to preserve comments, table order, credentials, and unrelated settings, updates the
`default` route, and updates `chat` only when that table is explicitly present. It validates the complete candidate
configuration before making exactly one same-directory atomic replacement. It never materializes a missing `chat`
route and does not use the startup `UserConfiguration` snapshot as a write source.

All Omni operations that replace `config.toml` serialize through the stable `.config.toml.lock` sidecar in Agent
Home. Each operation acquires the OS-backed process lock before rereading the latest document and holds it through
validation and atomic replacement, so cooperating Omni writers cannot publish from the same stale source. Lock
acquisition has a fixed one-second timeout. A timeout or lock failure follows the operation's existing best-effort
failure path.

Persistence is deliberately best effort. Parse, validation, and replacement failures leave the published runtime
value intact; Management records one safe diagnostic containing only the stable operation and exception type, then
returns the normal successful selection result. It does not log configuration contents, credentials, or a traceback.
Temporary divergence between runtime status and the on-disk User Configuration is accepted until a later
successful update or runtime restart. The local service coordinates successful persistence with global configuration activation under [ADR-0029](0029-host-cli-and-web-through-one-local-service.md).

The serialization guarantee applies only to Omni writers that use this lock. An ordinary external editor does not
participate and must not save `config.toml` during a Omni write transaction; atomic replacement prevents partial
files but cannot merge an arbitrary concurrent external write. This decision does not add a general mutation
framework, rollback transactions, or a global mutable User Configuration aggregate. Provider retry/fallback
behavior, Session state, and Agent Loop ownership are unchanged.

Requirements: [Reasoning Effort selection](https://github.com/Totoro-debug/OmniAgent/issues/213), [best-effort persistence](https://github.com/Totoro-debug/OmniAgent/issues/216).
