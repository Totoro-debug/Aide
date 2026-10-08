---
status: accepted
---

# Host CLI and Web through One Local Service

One on-demand Agent Service owns user operations, shared capabilities, execution scheduling, and exclusive Session Claims. Independent runtimes sharing files were rejected because atomic writes cannot coordinate live state, scheduling, or confirmations. An always-on daemon was rejected because work should stop after the final client leaves.

## Responsibility and data flow

CLI and Web submit named operations and consume authoritative state/events. They own unsent drafts, forms, focus, and presentation; business validation, input classification, operation ordering, and persistence belong to the service. Client adapters handle transport, while a reusable launcher handles discovery and bootstrap. Tool execution remains behind [ADR-0020](0020-expose-configured-mcp-tools-through-tool-gateway.md), without direct user Tool endpoints.

Accepted input enters the claimed Session's FIFO. An on-demand processor runs each input through its terminal persistence boundary before starting the next, then retires when empty. Distinct Sessions run concurrently, without a global Run limit or Workspace-wide execution lock. Queue recall and starting work share a coordination boundary: recall returns only unstarted inputs in FIFO order, and retries cannot recall newer input.

Each Run owns context preparation, permission and Skill snapshots, Tool exposure, cancellation, output identity, and file mutation recording. Model and Tool output returns through broker events scoped by Workspace, Session, and Run; clients do not compete for an output queue. Shared Tool implementations receive execution context explicitly.

## Resource ownership

| Lifetime | Resources |
| --- | --- |
| Service | Skills, Built-in Tool catalog, Model Router/Providers, Exec Host, HTTP MCP, event broker, confirmation coordinator, configuration snapshots and resource leases |
| Workspace association | Stores, Memory, Dream, Schedule state and coordination, stdio MCP |
| Session | Authoritative history/metadata, FIFO, Claim coordination, optional active processor |
| Agent Run | Captured projections, Gateway view, authorization, cancellation, transient execution state |

Opening another Session creates no Provider or MCP connection. HTTP MCP initializes globally; stdio MCP initializes once per active Workspace. Loaded Session histories stay resident until deletion, target Restore replacement, or service exit; releasing a Claim does not discard history. This reduces duplicated resources and idle tasks without bounding total memory or active work.

One global Skill reload publishes a validated snapshot for later Runs across Sessions. Active Runs retain their captured snapshot, and failed reload retains the previous one. Valid configuration saves, repairs, and external edits become eligible for each Session's next Run; queued input captures configuration when execution starts. Resource preparation shares unchanged Provider clients and MCP connections while retaining changed versions for existing Runs, title work, and already-registered SubAgents. Rebuilding Session histories or draining accepted work would couple a settings edit to unrelated execution, so configuration resources have an independent lifetime. Retired resources close after their Workspace and execution leases are released. Invalid configuration cannot replace a working resource snapshot and blocks new execution until repaired. Runtime permission selection affects a client's later Runs and overrides the configured default. Global chat Reasoning Effort updates memory before best-effort configuration persistence; cooperating configuration writers serialize reread and atomic publication under the Agent Home lock.

An explicit Session model/effort combination is captured when its Run starts, before the first wait, and drives foreground requests and budgeting. Later selection changes affect later Runs; auxiliary routes keep their purposes. Restore preserves the current selection. An unavailable selection leaves history readable but prevents new work until replaced.

## Claims and lifecycle

The service admits one Web client and multiple CLI clients. Session Claims, UI selection, resident history, and execution have independent lifetimes. Switching away can leave accepted work running. Reconnect grace retains Claims, accepted inputs, output, and confirmation identity; expiry cancels abandoned work and releases Claims. User-visible timing and launch/stop commands are in [README](../../README.md).

Registered Projects can schedule while the service has online clients. An unregistered Workspace admits Schedule work only while an online client uses it. Project removal closes admission, cancels and drains its work, resolves confirmations, and removes registration while retaining user files and saved state. Re-registering saved user Jobs requires explicit Schedule resume.

Shutdown stops admission, drains confirmation aborts and terminal outcomes while Stores remain writable, flushes Sessions, then closes shared MCP and Providers. Cleanup continues after individual failures. Failed terminal persistence prevents a drain from being reported as successful; cancellation never rolls back completed effects.

## Confirmation coordination

One active confirmation slot serves the service. Foreground and background queues are FIFO; foreground has priority when the slot becomes free without preempting an active request. Runtime envelopes bind normalized invocation and owner identity, and authorized clients receive an opaque token. Only the first valid decision is accepted; duplicates and late decisions cannot authorize another call.

Presentation is restricted to the Workspace audience. Reconnect retains the same request and token. Connected requests have no user-response timeout; cancellation or disconnect expiry aborts their owners. Unavailable presentation fails closed. Schedule owns its terminal outcome persistence, while Dream and other System Jobs remain outside this confirmation path. Permission policy is defined in [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md).
