---
status: accepted
---

# Host CLI and Web through One Local Service

One on-demand local service owns active Workspace runtimes and exclusive Session Claims. CLI and Web are clients of that service; static Web assets, versioned JSON management endpoints, and the WebSocket event channel share `127.0.0.1:8765`. Launchers verify service identity before attaching, and an unrelated process occupying the port is an error. Sharing files between independent runtimes was rejected because atomic writes do not coordinate in-memory state, Schedule dispatch, or confirmation decisions. An always-on daemon was rejected because work should stop when no CLI or Web client remains.

## Runtime ownership and conversation flow

`LocalService` owns client identity, Project registrations, Workspace admission, the event broker, configuration activation, and one `ToolConfirmationCoordinator`. Each `WorkspaceServiceRuntime` owns a shared `WorkspaceRuntime`, resolved Exec Host, Session Claims, and independent Session Agent Loops. A Workspace runtime owns its Model Router, MCP Runtime Manager, Memory Manager, Dream, and single Schedule Service. Session Loops receive those shared resources and own Session-bound context, Skill state, Tool Gateway, and run preparation.

CLI input crosses `ServiceClient` and the authenticated service transport before reaching a Session Loop. Each Loop has its own Message Bus with one service output consumer; the broker emits scoped events to clients. The terminal has its own presentation adapter. CLI and Web never compete for the same Outbound queue. Distinct Sessions in one Workspace may run concurrently, but a Session has only one loading client. Empty drafts stay in memory and disappear when abandoned.

Foreground and User Schedule work use independent run-local Agent Runner and Gateway views. User Jobs have dedicated Schedule Sessions and Loops and exclude the Schedule Tool; they do not dispatch through whichever foreground Session a client happens to view. Dream calls the shared Router directly and owns only its one-shot Memory request and restricted edits. Execution contracts follow [ADR-0014](0014-use-message-bus-agent-loop-and-agent-runner.md), [ADR-0021](0021-defer-tool-schema-exposure-per-agent-run.md), and [ADR-0024](0024-use-one-shot-dream-model-request.md).

Each client owns its current foreground Tool Permission Level, shared across that client's Session and Workspace switches and retained during reconnect grace. A new client starts from the active configuration. User Schedule occurrences instead capture the configured Workspace-generation level and resolved Exec Shell at admission. These snapshots are runtime-only; neither a Schedule Job nor Session persistence stores a permission selection. Authorization follows [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md).

## Project admission, reconnection, and shutdown

Each registered Project is a durable reference to an existing Workspace directory. Registration does not copy data or automatically enroll CLI Workspaces. Available registered Projects run Schedule Jobs while the service has online clients. An unregistered CLI Workspace admits new Schedule work only while a connected client uses it; its last disconnection pauses admission immediately, and expiry drains and releases that runtime without deleting Jobs or user files.

A client may switch away while an accepted run continues. Its 30-second reconnect grace retains Session Claims, accepted input, live output, and pending confirmation; replay or a current snapshot restores presentation. Expiry cancels abandoned work and releases claims. After the final online client disconnects, the service stops new Schedule admission immediately, allows the same 30-second reconnect period, then drains work and exits. `omni service stop` begins draining immediately.

Project removal closes admission durably, cancels its foreground and Schedule work, aborts pending confirmations, drains terminal outcomes, releases claims, and notifies clients before deleting the registration. A failed drain retains a retryable removal record. The Workspace directory and state remain on disk. Re-registering a directory with saved user Jobs requires explicit Schedule resume; overdue work never resumes merely because the path reappears.

Session switching operates on service claims and Session Loops without replacing the whole Workspace runtime. Restore rebuilds the affected Session Loop and rotates its claim after the durable transaction; the Workspace idle barriers and recovery contract follow [ADR-0028](0028-session-restore-architecture.md). Global configuration activation prepares replacement Workspace generations after accepted work finishes; a preparation failure retains the old active generation and permits retry.

Shutdown closes confirmation admission and drains typed confirmation aborts while Schedule stores remain writable. Workspace closure pauses and drains Schedule, closes Session work, then closes MCP, Dream, and Model Router resources. Cleanup continues after individual failures. Accepted Tool effects, Memory writes, artifacts, and persisted Jobs are not rolled back by cancellation or shutdown.

## One confirmation decision across clients

The service binds one stable `ServiceConfirmationPresenter` to its coordinator. Runtime-only immutable `ConfirmationEnvelope` values carry the exact normalized request, foreground or background origin, and an owner identifying the generation and run or Job occurrence. Internal envelopes are not durable records; the presenter emits a JSON projection with an opaque wire token and source identifiers to authorized clients.

There is one active confirmation slot across the service. Foreground and background queues are individually FIFO; foreground is selected first when the slot becomes free, without preempting the active item. Authorized clients can display the same request, but only the first valid decision is accepted. Duplicate, late, and unknown decisions cannot authorize another call. Connected requests have no user-response timeout; disconnect expiry and runtime cancellation abort their owners instead of implying user decline.

The presenter restricts requests to the Workspace audience, including Web clients able to inspect a registered Project. Reconnection snapshots retain the original pending request and token. Resolution or lifecycle dismissal removes it from every recipient. CLI and Web keep Decline as the safe default and identify background Job context.

Owner and generation cancellation propagate typed `ConfirmationAborted` through Gateway, Runner, and Loop to Schedule Service, which owns terminal Job persistence. Unavailable presentation fails closed with `ConfirmationUnavailable`. A failed terminal Store write fails the drain rather than being treated as successful cleanup. Job deletion persists absence first, then cancels and drains its exact active occurrence and confirmation owner; a failed delete does not cancel the Job. Dream and other System Schedule work remain outside this confirmation path.

Requirements: [Local Web interface and shared multi-session service (#281)](https://github.com/Totoro-debug/OmniAgent/issues/281).
