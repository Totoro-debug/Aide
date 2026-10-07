---
status: accepted
---

# Host CLI and Web through One Local Service

One on-demand local service owns shared capabilities, Workspace state associations, and exclusive Session Claims. CLI and Web are clients of that service; static Web assets, versioned JSON management endpoints, and the WebSocket event channel share `127.0.0.1:8765`. Launchers verify service identity before attaching, and an unrelated process occupying the port is an error. Sharing files between independent runtimes was rejected because atomic writes do not coordinate in-memory state, Schedule dispatch, or confirmation decisions. An always-on daemon was rejected because work should stop when no CLI or Web client remains.

## Runtime ownership and conversation flow

`AgentService` owns client identity, Projects, admission, the event broker, startup configuration, one confirmation coordinator, and the shared Skills, Built-in Tool catalog, Router and Providers. Workspace records associate Stores, Memory, Dream, Schedule state and stdio MCP; the service owns their lifecycle. HTTP MCP is shared globally. [ADR-0030](0030-service-owned-shared-resources-and-session-execution.md) defines shared ownership and Session scheduling.

CLI input crosses `ServiceClient` and the authenticated transport into the claimed Session's FIFO. A processor runs accepted inputs serially and retires when empty. Distinct Sessions execute concurrently. Output is forwarded only during a Run with explicit Workspace, Session and Run identity. CLI and Web receive scoped broker events; they never compete for an Outbound queue. Every new foreground draft records an explicit `chat` or `project` creation scope. Project operations use `project`; ordinary Web conversations use `chat`; CLI creation uses `project` in a currently registered Project directory and `chat` elsewhere. Lists filter this stored scope without deriving it from later registration changes. Records without scope are excluded from scoped lists and are never rewritten during reads. Restore retains the original scope. Empty drafts disappear when abandoned; loaded conversation history remains resident until service exit, explicit deletion or target Restore replacement.

Foreground and user Schedule work create independent run-local execution collaborators, Agent Runners and Gateway views. User Jobs retain their dedicated Schedule Sessions and exclude the Schedule Tool. Dream retains its one-shot Memory model request and restricted edits. Execution contracts follow [ADR-0014](0014-use-message-bus-agent-loop-and-agent-runner.md), [ADR-0021](0021-defer-tool-schema-exposure-per-agent-run.md), and [ADR-0024](0024-use-one-shot-dream-model-request.md).

Each client owns its current foreground Tool Permission Level, shared across that client's Session and Workspace switches and retained during reconnect grace. A new client starts from the active configuration. User Schedule occurrences instead capture the configured startup level and resolved Exec Shell at admission. These snapshots are runtime-only; neither a Schedule Job nor Session persistence stores a permission selection. Authorization follows [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md).

## Project admission, reconnection, and shutdown

Each registered Project is a durable reference to an existing Workspace directory. Registration does not copy data or automatically enroll CLI Workspaces. Available registered Projects run Schedule Jobs while the service has online clients. An unregistered CLI Workspace admits new Schedule work only while a connected client uses it; its last disconnection pauses admission immediately, and expiry cancels and drains its abandoned work while preserving resident Session history, saved Jobs and user files.

A client may switch away while an accepted run continues. Its 30-second reconnect grace retains Session Claims, accepted input, live output, and pending confirmation; replay or a current snapshot restores presentation. Expiry cancels abandoned work and releases claims. After the final online client disconnects, the service stops new Schedule admission immediately, allows the same 30-second reconnect period, then drains work and exits. `omni service stop` begins draining immediately.

Project removal closes admission durably, cancels its foreground and Schedule work, aborts pending confirmations, drains terminal outcomes, releases claims, and notifies clients before deleting the registration. A failed drain retains a retryable removal record. The Workspace directory and state remain on disk. Re-registering a directory with saved user Jobs requires explicit Schedule resume; overdue work never resumes merely because the path reappears.

Session switching changes selection and Claims independently of execution. Restore reloads only the target Session and rotates its Claim after the durable transaction; the Workspace idle barriers and recovery contract follow [ADR-0028](0028-session-restore-architecture.md). Configuration read, save, and repair preserve startup resources and accepted work. Saved revisions require a subsequent service startup, including first-use repair. The service publishes one global chat Reasoning Effort override to current and subsequently activated Workspaces without rebuilding Providers; persistence remains best effort.

Shutdown closes confirmation admission and drains typed confirmation aborts while Schedule stores remain writable. Workspace cleanup drains Schedule terminal outcomes while its Store remains writable, flushes every Session, then closes Dream and its stdio association. Service shutdown closes global HTTP MCP and Providers after all Workspace cleanup. Cleanup continues after individual failures. Accepted Tool effects, Memory writes, artifacts, and persisted Jobs are not rolled back by cancellation or shutdown.

## One confirmation decision across clients

The service binds one stable `ServiceConfirmationPresenter` to its coordinator. Runtime-only immutable `ConfirmationEnvelope` values carry the exact normalized request, foreground or background origin, and an owner identifying the generation and run or Job occurrence. Internal envelopes are not durable records; the presenter emits a JSON projection with an opaque wire token and source identifiers to authorized clients.

There is one active confirmation slot across the service. Foreground and background queues are individually FIFO; foreground is selected first when the slot becomes free, without preempting the active item. Authorized clients can display the same request, but only the first valid decision is accepted. Duplicate, late, and unknown decisions cannot authorize another call. Connected requests have no user-response timeout; disconnect expiry and runtime cancellation abort their owners instead of implying user decline.

The presenter restricts requests to the Workspace audience, including Web clients able to inspect a registered Project. Reconnection snapshots retain the original pending request and token. Resolution or lifecycle dismissal removes it from every recipient. CLI and Web keep Decline as the safe default and identify background Job context.

Owner and generation cancellation propagate typed `ConfirmationAborted` through Gateway, Runner, and Run execution to Schedule Service, which owns terminal Job persistence. Unavailable presentation fails closed with `ConfirmationUnavailable`. A failed terminal Store write fails the drain rather than being treated as successful cleanup. Job deletion persists absence first, then cancels and drains its exact active occurrence and confirmation owner; a failed delete does not cancel the Job. Dream and other System Schedule work remain outside this confirmation path.

Requirements: [Local Web interface and shared multi-session service (#281)](https://github.com/Totoro-debug/OmniAgent/issues/281).
