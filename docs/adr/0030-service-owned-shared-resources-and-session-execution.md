---
status: accepted
---

# Agent Service-owned Shared Resources and Session Execution

AgentService is the single composition and scheduling authority for CLI and Web.
It owns shared capabilities across Workspaces and creates execution collaborators
for each Agent Run. Resident Session history, exclusive Claims, and active work
have independent lifetimes, reducing duplicate resources and idle tasks while
preserving asynchronous Session concurrency.

The authoritative requirements and quantified delivery criteria are tracked in
[#306](https://github.com/Totoro-debug/OmniAgent/issues/306).

## Confirmed decisions

- AgentService owns Skills, Tools, Model Providers, and other shared resources
  and schedules Agent Runs. There is no persistent execution object per Session
  or intermediate Workspace Runtime ownership layer. Workspace records associate
  data and coordination state; resource creation and closure belong to the service.
- The optimization targets repeated resource construction across Workspaces
  and the number of persistent per-Session Agent Loop instances. Resource
  counts and memory observations are recorded in the [T7 issue discussion](https://github.com/Totoro-debug/OmniAgent/issues/313).
- Accepted inputs within one Conversation Session execute serially. Distinct
  Conversation Sessions may execute concurrently.
- There is no global Agent Run concurrency limit.
- Busy Sessions retain the existing FIFO input behavior. A Session's next Run
  begins after its previous Run reaches its terminal persistence boundary.
- Switching the displayed Session changes the UI selection without interrupting
  execution in the previously displayed Session.
- Loaded Sessions remain resident for the service lifetime, without idle
  eviction or a bounded history cache. Session residency does not require a
  persistent execution or input-consumer task.
- Session Claims are managed independently from retained Session objects.
  Existing exclusive Client claims, release after switched work completes,
  and disconnect expiry behavior are retained.
- Memory, Dream, and Schedule are managed globally while their data and
  Workspace associations remain isolated.
- MCP Servers use one global configuration. Each configured HTTP MCP Server
  has one shared instance across the service. Each configured stdio MCP Server
  has one shared instance per Workspace, reused by its Sessions.
- HTTP MCP connections initialize when the service activates. Stdio MCP
  connections initialize when their Workspace first activates.
- Skills may use immutable snapshots with deferred refresh. Standalone MCP
  refresh and live Tool discovery refresh are deferred.
- One global Skill reload publishes a validated snapshot for subsequent Runs
  across every Session; in-flight Runs retain their captured snapshots. A failed
  reload keeps the previous snapshot.
- AgentService uses the configuration loaded at startup for its entire lifetime,
  including Workspaces activated later. Agent and Web edits may persist settings
  for the next startup. Reading or saving configuration does not drain work or
  replace resources. Automatic configuration-generation replacement is removed.
- Explicit runtime controls retain their current semantics: Permission Level
  selection changes a Client's later Run snapshots; global chat Reasoning Effort
  selection changes later logical model requests and retains its existing
  best-effort persistence. These controls do not rebuild shared resources.
- A Conversation Session may explicitly select an Available Model and Reasoning
  Effort together. The combination has an independent metadata version and is
  captured before the foreground Run's first wait, including title coordination.
  Later selection changes affect subsequent Runs. The captured combination and
  its model capacity drive requests, retries, continuation and context budgeting;
  auxiliary Model Routes retain their configured purposes. Sessions without an
  explicit combination keep the existing CLI and Schedule defaults. Session
  Restore preserves the current selection, while an unavailable selection keeps
  history readable and must be replaced before new input is admitted.

## Execution structure

AgentService is the composition root and scheduling authority. Its internal
modules retain the existing ReAct engine, Tool Gateway authorization boundary,
event broker, and Workspace storage behavior. Consolidating ownership does not
require placing their implementations into a single class or source file.

Each retained Session entry contains its authoritative Session object, FIFO,
coordination state, and optional active work. Admission updates this entry and
starts a processor if none exists. The processor runs one input through its
terminal commit before taking the next input and exits when the queue is empty.
Starting and retiring processors must be coordinated with admission so accepted
input cannot be stranded. Different Session processors may run concurrently.

Each Run owns its execution context, permission snapshot, Skill snapshot,
context preparation state, cancellation, Tool exposure, output identity, and
file mutation recorder. Shared Tool implementations receive Workspace and
execution context explicitly rather than changing an instance's active
Workspace. Output retains Workspace, Session, and Run identity through the
existing transport contracts.

Memory and Schedule execution logic is managed globally, with stores, cursors,
Job reservations, Dream serialization, and required locks keyed by Workspace.
No Workspace-wide execution lock serializes all foreground Sessions. Existing
Restore readiness, File Restore behavior, confirmation decisions, and terminal
persistence remain part of the integration contract.

## Resource and lifecycle consequences

For P configured Providers used by Runs, H enabled HTTP MCP Servers, S enabled
stdio MCP Servers, and W active Workspaces, successful stable initialization
creates P Provider clients, H HTTP connections, and W * S stdio connections.
MCP managers select exactly one transport explicitly: service-global HTTP accepts no Workspace, and stdio requires one Workspace. Their connection factories receive `Path | None` under that same contract. Opening more Sessions does not create additional clients or connections. There
is one current global Skill snapshot and one shared Built-in Tool catalog;
workspace-specific stdio discovery remains scoped to its Workspace.

Claims, cached Session authorities, and active work have independent lifetimes.
Releasing a Claim does not close a retained Session. Explicit Session deletion
invalidates its cached authority. Restore replaces only the affected Session
authority after the durable transaction and rotates its Claim; it does not
rebuild shared resources. Foreground and User Schedule Runs both use this
execution structure, while Dream retains its one-shot model path.

Configuration responses distinguish saved settings from startup settings and
report when restart is required. The former pending-application and retry UI,
GET-triggered activation, and configuration repair activation must be revised
together. Ordinary Permission and Reasoning Effort selections remain immediate
at their established capture boundaries.

Service shutdown stops admission, resolves confirmation aborts, drains terminal
Run and Schedule outcomes while stores remain writable, flushes retained Session
state, and then closes shared MCP and Model Provider resources. Project removal
retains its existing Workspace-scoped cancellation and persistence behavior
without closing service-global resources used by other Workspaces.

Retained conversation history and unconstrained active Runs can still increase
memory use. This decision removes duplicate capability resources and idle
Session execution tasks; it does not impose a total-memory bound.

[ADR-0029](0029-host-cli-and-web-through-one-local-service.md) records client,
transport, confirmation, and Project lifecycle decisions. The implementation requirements and verification history are recorded in
[#306](https://github.com/Totoro-debug/OmniAgent/issues/306) and its child Issues.
