---
status: accepted
---

# Use the CLI as Composition Root and Agent Loop as Runtime Generation

The CLI asynchronous root is the sole composition root. It owns the Workspace path/state, lifetime-scoped Message Bus, Model Router, MCP Runtime Manager, resolved Exec Host, Memory Manager, Dream, Schedule Service, Runtime Lifetime confirmation coordinator, Management services, Terminal application, and current Agent Loop reference. Each Agent Loop is one Session-scoped Runtime Generation and constructs its own Session, Skill Loader, Context Builder, Agent Run Context Controller, Tool Gateway, and Agent Runner. Blackboard owns Task Framing generation. No container or differently named Runtime aggregate proxies this ownership; the Exec Host is the narrow Host Adapter contract defined by ADR-0010 rather than an ownership proxy.

The MCP Runtime Manager connects configured Servers before the first Agent Loop is created and supplies an immutable MCP Tool Snapshot for each generation. The Skill Loader instead publishes atomically replaceable frozen Skill state within a generation. Complete request preflight includes the candidate Skill projection and the Runtime Generation's corresponding initial Tool Exposure; their separate refresh rules follow [ADR-0016](0016-use-agent-home-skill-catalog-and-progressive-loading.md) and [ADR-0020](0020-expose-configured-mcp-tools-through-tool-gateway.md).

As the composition root, `myclaw/terminal/cli.py` may import MCP Runtime Manager and snapshot types from `myclaw.tools.mcp.runtime`, `MCPKeywordPreparer` from `myclaw.tools.mcp.keywords`, `BUILT_IN_TOOL_NAMES` from `myclaw.tools.gateway`, and Exec Host selection functions and capability errors from `myclaw.tools.exec.host`. This permission does not extend to other Terminal presentation modules. There are no separate top-level `myclaw.mcp_runtime` or `myclaw.exec_host` re-export modules.

Agent Runner is the sole bounded ReAct implementation. Foreground and User Schedule work share the current Agent Loop's generation-owned Tool instances, but each uses a run-local Runner bound to its own context collaborators and an independent Agent Run Gateway view with isolated Catalog and Tool Activation state. Dream is a single-completion responsibility that calls Model Router directly and applies only returned Long-term Memory edits, as recorded in [ADR-0024](0024-use-one-shot-dream-model-request.md). Context construction follows [ADR-0018](0018-centralize-model-request-context-construction.md).

Foreground model-issued Schedule calls use the foreground Run snapshot through that run's Gateway. `list` is direct at every foreground level; `add` and `remove` require one confirmation in Read-Only and are direct in the other levels. A foreground `add` also requires confirmation when the immutable configured Schedule level is higher than the captured current level; that reason is merged with the CRUD reason when both apply. The confirmation is bound to normalized call details, and no permission level or snapshot is written to `ScheduleJob` or its public JSON. `/permission` changes only the current level for later foreground Runs.

At User Schedule occurrence admission, Schedule Service creates an immutable runtime occurrence with a UUID and a snapshot of the startup configured Tool Permission Level plus the process-lifetime resolved Exec Shell. The snapshot is passed unchanged to the Schedule Runtime Context and run-local Gateway; a background confirmation uses the occurrence's `BackgroundConfirmationOwner` and the canonical `job_id` and title only for presentation. Read-Only and Workspace-Write apply the existing File, Exec, Web Fetch, and per-call MCP confirmation rules; Full-Access removes ordinary permission prompts while retaining hard errors. Dream and other System Schedule Jobs remain internal direct operations. The occurrence snapshot is never added to `ScheduleJob`, Session, Tool Result, or public JSON.

Session replacement first prepares a candidate MCP generation, pauses old foreground admission, waits for required Session persistence, and constructs and synchronously preflights an unstarted target Agent Loop. Active foreground work requires the existing force confirmation before destructive replacement. Once destructive replacement begins, the CLI closes the old generation's Schedule confirmation admission, cancels all of its confirmation items, and drains the resulting Schedule terminal commits while the Store is writable before quiescing its presenter. The successful replacement order is:

`old generation confirmation cancel -> Schedule confirmation drain -> Terminal quiesce -> Schedule pause_and_drain -> current unavailable -> old Loop abort/drain -> bus.reset -> Terminal rebind -> target.start -> MCP candidate activation -> publish current -> release admission barrier -> Schedule resume`

The target is started before it becomes current, and Schedule dispatch resumes only afterward. Target construction or preflight failure terminates the Terminal Conversation with a safe error; the CLI still owns and closes all prepared components. Failure after destructive replacement begins leaves Management unavailable and triggers shutdown rather than restoring a partially aborted generation.

After the Terminal returns and restores terminal state, shutdown first closes the Runtime Lifetime confirmation coordinator and drains Schedule confirmation-aborted occurrences while the Store remains writable, so active and queued waiters receive typed lifecycle cancellation and terminal state is committed before broad cancellation. It then awaits `Management deactivate -> Schedule pause_and_drain + close -> pending/active Loop abort or close -> MCP close -> Dream close -> Model Router close`. Cleanup continues after individual failures and preserves the primary error. Normal Loop closure persists and drains; forced abort may lose unpersisted state and does not undo accepted Tool, Artifact, Memory, or Schedule side effects.

Schedule deletion uses the Store mutation as its linearization point. A
successful delete then cancels the exact active occurrence and its coordinator
owner, if present, and drains that task before returning; a failed Store delete
does not cancel it. Lifecycle `ConfirmationAborted` remains typed through the
Gateway, Runner, and Agent Loop so Schedule Service alone owns the terminal Job
update. Its generation admission gate remains closed until Schedule pause, and
the pause barrier cancels ordinary tasks while retaining any late gate-aborted
occurrence until its terminal Store write completes. A terminal Store failure
faults the Service and fails the confirmation drain rather than permitting
replacement or shutdown to treat persistence as complete.

The Runtime Lifetime coordinator and its stable modal presenter are defined by [ADR-0027](0027-runtime-lifetime-tool-confirmation-coordinator.md). The CLI binds every generation's foreground and User Schedule confirmation requester to that coordinator while the Terminal keeps one presenter and stable display/input references across replacement. Schedule-owned background confirmation uses the same coordinator, creates no second presenter, and persists no Runtime-only confirmation envelope.

Schedule pause cancels and drains dispatcher work, Agent Runs, and terminal commits. An uncommitted one-shot Job remains pending for later dispatch; recurring schedules keep their accepted cursor and do not replay the interrupted occurrence. Direct Schedule Service closure waits for already-started terminal commits, while CLI replacement and shutdown first invoke the cancelling pause operation.

Memory Manager owns Summary, Summary Cursor, Long-term Memory persistence, and the live memory snapshot, without model work. Dream consumes Summary batches through its dedicated `memory` lane, advances the Summary Cursor before model execution without rollback, supplies the complete Long-term Memory to one logical Model Router request, and sequentially applies only returned edits to that exact file without a confirmation request. Model Router retains its own automatic Provider retries, and Dream refreshes the live memory view after edits. Schedule Service owns its Store and dispatches the persisted `job_id="dream", source="system"` Job directly to Dream without a Schedule Session. User Jobs reach the current Agent Loop through a CLI-owned callable.

Requirements: [CLI ownership](https://github.com/Totoro-debug/myclaw/issues/188), [Schedule terminal commits](https://github.com/Totoro-debug/myclaw/issues/195).
