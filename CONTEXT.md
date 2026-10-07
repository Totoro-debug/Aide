# Personal Agent Runtime

This context defines the language for an independently designed, local-first, single-user Personal Agent runtime.

## Language

**Personal Agent**:
A local-first, single-user Agent runtime that works continuously for one person through conversation and explicit management actions.
_Avoid_: Bot platform, multi-tenant assistant, channel-first agent, agent platform

**Agent Home**:
The fixed account-global home of one Personal Agent installation, separate from every Workspace and Conversation Session.
_Avoid_: Project workspace, session directory, install directory, configurable data root

**Workspace**:
The user-selected directory that scopes one Personal Agent interaction and owns its non-global state and file capabilities.
_Avoid_: Agent Home, install directory, session directory, project ID

**Project**:
A directory persistently registered in the Web Interface for reuse; it is the Workspace of Conversation Sessions created under it.
_Avoid_: a copy of Workspace data, Workspace State, backup, project ID

**Default Conversation Workspace**:
The user-configurable Workspace used for Web Conversation Sessions started without selecting a Project. These Sessions belong to that Workspace and share its Workspace-owned resources.
_Avoid_: no Workspace, Agent Home, default Project, global Session storage

**Workspace State**:
Persistent Personal Agent state owned by exactly one Workspace rather than by the installation or operating-system account.
_Avoid_: Agent Home, project source, global state, cache

**Message Bus**:
The transient Inbound and Outbound queues for one foreground conversation lane, connecting a Command-line Conversation to the Agent Service.
_Avoid_: persistent event log, broadcast bus, Schedule queue

**Inbound Message**:
One ordinary user input waiting in the Message Bus for serial foreground processing.
_Avoid_: Management Command, Tool Confirmation, cancellation command, Agent Event

**Outbound Message**:
One transient presentation message emitted by the Agent Service for a Command-line Conversation.
_Avoid_: Agent Event, Session message, diagnostic log, broadcast event

**Agent Run Activity Group**:
A user-visible group of non-final model output and Tool activity belonging to one foreground Agent Run, distinct from that run's final output.
_Avoid_: Conversation, Agent Run, event log, transcript

**Command-line Conversation**:
The user-facing conversation with the Personal Agent presented as a full-screen terminal experience.
_Avoid_: Terminal session, shell command, chat channel, one-shot command, plain REPL

**Web Interface**:
The local-browser entry for one person to converse with and manage the Personal Agent.
_Avoid_: remote service, multi-user platform, read-only dashboard

**Agent Service**:
The sole user-facing capability boundary for the Personal Agent, owning all conversation and management operations, their business rules, configuration validation, input admission and queueing, and shared runtime coordination. CLI and Web clients selectively invoke its operations and present their state.
_Avoid_: Local Service, Agent Loop, Workspace Runtime, always-on daemon, remote service, separate CLI runtime

**Client**:
One CLI or Web participant in the Agent Service, with its own foreground permission selection and exclusive claims on Conversation Sessions. One Agent Service admits at most one Web Client and multiple CLI Clients; releasing control does not end Session residency.
_Avoid_: operating-system account, Conversation Session, Project, Model Provider client

**Management Command**:
An explicit user command for inspecting or changing runtime-managed state without relying on natural-language conversation.
_Avoid_: Tool call, chat instruction, task management, one-shot conversation

**Management Port**:
The management portion of the Agent Service capability boundary, through which Management Commands use runtime capabilities without knowing their storage or implementation details.
_Avoid_: Message Bus, direct file access, admin API

**Runtime Lifetime**:
The lifetime of one Agent Service instance and the Clients, shared capabilities, Workspace-owned state, and Conversation Sessions it coordinates.
_Avoid_: Detached mode, daemon mode, persistent background process, one-shot command

**Session Log**:
Workspace-owned technical diagnostics associated with one Conversation Session rather than with the whole installation.
_Avoid_: Runtime Log, Conversation log, chat transcript, audit log, activity feed

**Agent Runner**:
A reusable, Session-independent engine that performs one bounded model-and-Tool ReAct loop for any request that requires ReAct.
_Avoid_: Agent Service, Conversation Session, Runtime Lifetime, Provider retry loop, single model call

**Agent Run**:
One complete Agent execution for one input against one Conversation Session, from input acceptance through its final outcome and persistence request.
_Avoid_: Agent Turn, Runtime, Model call, Tool call

**ReAct Cycle**:
One assistant response that requests Tools together with every corresponding Tool result completed before the next model request; a terminal assistant response without Tool calls ends the Agent Run instead of starting another cycle.
_Avoid_: Agent Run, Model call, individual Tool call, Provider retry

**Blackboard**:
A hidden task definition attached to a Conversation Session and its foreground Agent Runs, containing one current goal and one completion boundary. It supports interpretation without controlling execution or exposing a task-management surface.
_Avoid_: Task list, plan, workflow state, progress tracker

**Task Framing**:
The interpretation of an eligible new foreground user input against the current Blackboard and latest assistant response to keep, replace, or clear the current task definition. It is mutually exclusive with a Manual Skill Invocation in the same Agent Run.
_Avoid_: Task decomposition, planning, orchestration, progress update

**Runtime Context**:
Dynamic facts about the current runtime and Agent Run supplied to a model call but not authored by the user.
_Avoid_: User instruction, Long-term Memory, Session message

**System Prompt**:
The stable system-level context that establishes Personal Agent identity, memory, and capability guidance for a model call.
_Avoid_: Runtime Context, user message, Conversation Summary

**Model Request Context**:
The provider-neutral ordered messages assembled for one model call, including its System Prompt, Runtime Context, projected conversational messages, and any model-visible execution continuation.
_Avoid_: Prompt, Conversation Session, raw Session transcript, Provider request payload

**Reported Model Usage**:
The token usage returned by a Model Provider for one completed model request and response, used as the primary measurement basis for later context budgeting when available.
_Avoid_: Session token total, local token estimate, cost total

**Projected Next-request Usage**:
The expected token occupancy of the next Model Request Context, based first on the latest applicable Reported Model Usage plus subsequent context changes, with a complete local estimate as fallback.
_Avoid_: Reported Model Usage, cumulative Session usage, output token limit

**Available Context**:
The maximum input-token capacity of one resolved Model Route after reserving that route's configured maximum output, calculated as `context_window - max_output`.
_Avoid_: Context window, Compaction Context Window, current context usage

**Compaction Context Window**:
The proactive compression threshold for a Model Route, calculated as `ceil(Available Context * Runtime compact_ratio)`; it triggers compaction before the hard input capacity is exhausted.
_Avoid_: Available Context, model context limit, output token limit

**Skill**:
A named, discoverable instruction package that guides an Agent Run through existing capabilities without registering Tools or expanding permissions.
_Avoid_: Tool, Plugin, Management Command, capability extension

**Skill Catalog**:
The ordered set of valid Skill metadata presented for discovery without exposing the corresponding instructions in that catalog.
_Avoid_: Tool Catalog, command list, loaded Skill content

**Skill Snapshot**:
The immutable set of validated Skill metadata and complete instructions published by one successful global load. An Agent Run retains its captured snapshot while a later successful reload supplies the snapshot for subsequent Runs.
_Avoid_: live Skill directory, Tool Catalog, Runtime Lifetime cache

**Skill Invocation**:
The selection and application of one Skill's instructions to a foreground Agent Run, initiated explicitly by the user or autonomously by the model.
_Avoid_: Management Command, Tool capability, Skill discovery

**Manual Skill Invocation**:
A Skill Invocation initiated by an exact available Skill slash name at the beginning of foreground input, optionally followed by whitespace and a request.
_Avoid_: Autonomous Skill Invocation, always-loaded Skill, Management Command

**Conversation Session**:
A durable conversational thread owned by one Workspace, with one authoritative in-memory Session retained after loading for the Agent Service lifetime.
_Avoid_: Chat ID, terminal session, Workspace, runtime checkpoint, background task

**Session Claim**:
The exclusive right of one Web or Command-line client to load and operate one Conversation Session while other clients cannot load it.
_Avoid_: Workspace lock, shared viewer, permission grant, permanent ownership

**Restore Anchor**:
A persisted foreground user message in a Conversation Session that identifies the state immediately before that message and has a Session-scoped monotonically increasing number that is never reused.
_Avoid_: queued input, active Agent Run, timestamp, backup

**Session Restore**:
A Management Command action that truncates one Conversation Session to a Restore Anchor, with an optional File Restore.
_Avoid_: Session resume, Conversation Summary rollback, conversation branch

**File Restore**:
The Session Restore mode that attempts to return eligible files changed by that Conversation Session to their state before the Restore Anchor, while reporting any file it cannot restore.
_Avoid_: Workspace reset, Exec rollback, MCP side-effect rollback

**File Backup**:
The byte-for-byte state of a file captured immediately before an eligible Built-in Tool attempts to modify it, associated with the foreground Restore Anchor that owns the Tool call.
_Avoid_: Workspace snapshot, file version, metadata backup

**Backup Gap**:
A known eligible Built-in Tool modification that continued without a usable File Backup and makes File Restore unavailable for every restore range containing it.
_Avoid_: file conflict, Tool error, missing legacy backup

**Memory System**:
The three-layer memory structure owned by a Workspace: Short-term Memory, Conversation Summary, and Long-term Memory.
_Avoid_: Single memory store, vector memory, raw transcript archive

**Short-term Memory**:
The uncompacted suffix of a Conversation Session used to continue its current thread.
_Avoid_: Chat log, transcript, prompt history, full Session file

**`last_compacted`**:
The position in a Conversation Session separating messages already represented by Conversation Summary from Short-term Memory.
_Avoid_: checkpoint, bookmark, Session ID

**Conversation Summary**:
A Workspace-owned ordered stream of compact summaries derived from earlier Conversation Session messages.
_Avoid_: Long-term Memory, raw history, manual note, Session memory

**Action Summary**:
A Conversation Session-owned rolling summary of neutral, completed small tasks represented by compacted messages; each successful compaction replaces or clears it for later Agent Runs without adding it to the Workspace Conversation Summary stream.
_Avoid_: Long-term Memory, progress status, task tracker, Conversation Summary

**Long-term Memory**:
A Workspace-level durable memory of stable information intended to influence later Agent behavior across Conversation Sessions.
_Avoid_: Raw history, Session archive, manual notes, vector database, Conversation Summary

**Dream**:
A background or manually triggered one-shot memory process that turns new Conversation Summary entries into Long-term Memory through at most one logical Memory Model request before applying any returned edits.
_Avoid_: Memory Task, Chat turn, Conversation Compaction, full Agent Run

**Summary Cursor**:
The Workspace-owned position through which Dream has consumed the Conversation Summary stream.
_Avoid_: Session position, checkpoint

**Lesson**:
A reusable experience that should change future Agent behavior or design judgment.
_Avoid_: Conversation Summary, activity log, task note

**Tool Gateway**:
The sole public boundary for resolving, validating, authorizing, executing, and normalizing Tool calls.
_Avoid_: Direct Tool registry, plugin executor, shell wrapper

**Tool Confirmation**:
A host-mediated, one-shot user decision bound to one validated Tool call in one live Agent Run.
_Avoid_: Permission Policy, model approval, chat reply, persistent approval

**Background Tool Confirmation**:
A Tool Confirmation originating from a user Schedule Agent Run rather than the current foreground Agent Run. It remains bound to exactly one Tool call and is visibly identified as background work.
_Avoid_: Background permission, Schedule permission, persistent approval

**Tool Catalog**:
The ordered set of concrete Tool capabilities available for name lookup and invocation through a Tool Gateway. Membership is independent of Tool Exposure.
_Avoid_: Plugin list, command list, model tools, subagent registry, MCP registry

**Tool Exposure**:
The inclusion of a Tool's complete invocation definition among the capabilities presented to the model for one model call. It neither changes Tool Catalog membership nor gates invocation.
_Avoid_: Tool Catalog membership, Tool Activation, Tool execution, Permission Policy

**Tool Activation**:
The Agent Run-scoped selection of a deferred Tool for exposure in subsequent model calls. It expires with the Agent Run and neither authorizes nor invokes the Tool.
_Avoid_: MCP Server enablement, Tool Confirmation, execution permission, persistent Session capability, Tool execution

**Tool Search Keywords**:
The English terms associated with a Tool for capability discovery through Tool Search.
_Avoid_: Tool description, invocation parameters, user instruction, Skill metadata

**Tool Search**:
The Agent Run-scoped discovery of deferred Tools from that Run's available Tool Catalog using English query keywords. Returned Tools are activated for subsequent model calls in that Run.
_Avoid_: Web Search, MCP discovery, Tool execution, permission grant

**Built-in Tool**:
A Tool capability shipped as part of the Personal Agent runtime rather than discovered from an MCP Server.
_Avoid_: MCP Tool, Plugin, Management Command

**MCP Server**:
An external capability provider explicitly selected by the user through User Configuration and accessed by the Personal Agent through the Model Context Protocol.
_Avoid_: Model Provider, Plugin, Tool Catalog, MCP endpoint

**MCP Server Configuration**:
The single User Configuration item keyed by `mcp_name` that declares an MCP Server's enablement, transport, and connection settings.
_Avoid_: MCP endpoint, MCP profile, Server Tool

**MCP Runtime Manager**:
The Agent Service-owned manager of MCP Server connections and discovered Tools, with one HTTP connection per configured Server across the service and one stdio connection per configured Server within each Workspace.
_Avoid_: Tool Gateway, MCP registry, Workspace Runtime

**MCP Tool**:
A Tool capability discovered from an MCP Server and included in a Tool Catalog with the same invocation semantics as a Built-in Tool, while retaining its external origin.
_Avoid_: Built-in Tool, Plugin Tool, direct MCP call

**MCP Tool Snapshot**:
The immutable ordered set of successfully discovered MCP Tools published together.
_Avoid_: live MCP registry, mutable Tool Catalog, MCP Server list

**Tool Artifact**:
A durable external representation of an oversized successful Tool result associated with one Conversation Session.
_Avoid_: Tool result, attachment, memory entry

**Tool Result Micro-compression**:
An Agent Run model-context projection that omits oversized results from eligible earlier ReAct Tool messages after that run has made more than ten such Tool calls, without changing the returned Tool messages or persisted Session history.
_Avoid_: Tool Artifact, Tool result deletion, Tool execution truncation, permission filtering

**Schedule**:
The domain encompassing persistent scheduled tasks and the service that manages and runs them.
_Avoid_: Scheduled Work, a single scheduled task, the scheduling service

**Schedule Job**:
One Workspace-owned persistent task in the Schedule domain with its own execution timing. A user-created Schedule Job has its own Conversation Session; a System Schedule Job may use a dedicated internal executor.
_Avoid_: Schedule, Scheduled Work, shell cron job, reminder

**Schedule Job Title**:
The stable user-facing label owned by one Schedule Job, distinct from both its instruction message and its dedicated Conversation Session title.
_Avoid_: Job message, Schedule Session title, display-only alias

**System Schedule Job**:
A Schedule Job created and maintained by the Personal Agent for internal Runtime work rather than by the user. It is hidden from user Schedule listing and mutation.
_Avoid_: User Schedule Job, public Schedule, shell cron job

**Dream Schedule Job**:
The unique System Schedule Job for one Workspace that invokes Dream through its dedicated execution path without creating a Schedule Session or a foreground Agent Run.
_Avoid_: User Schedule Job, Memory Task scheduler, scheduled Agent Run

**Schedule Service**:
The Agent Service-owned authority for managing and executing Schedule Jobs while preserving each Job's Workspace ownership.
_Avoid_: Schedule, Schedule Job, detached background process

**Tool Permission Level**:
One of Read-Only, Workspace-Write, or Full-Access, selecting how much Tool autonomy the user grants. Lower levels may use exact-call Tool Confirmation to cross their normal boundary; the level governs only Tool capabilities and not runtime-owned persistence.
_Avoid_: Access Mode, process permission, Tool Exposure, Tool Activation, persistent approval

**Catastrophic Exec Operation**:
An Exec invocation recognized as capable of erasing broad workspace, repository, user, filesystem, or device state, or disrupting the host system. It always requires Tool Confirmation, including under Full-Access.
_Avoid_: Full-delete command, ordinary recursive deletion, invalid command

**Permission Policy**:
The Tool-specific authorization rules that determine whether one normalized invocation can run directly or requires Tool Confirmation. Validation, business refusal, missing capabilities, and execution failures are separate hard outcomes rather than permission levels.
_Avoid_: Tool switch, safety flag, enablement

**Model Route**:
A named model purpose that resolves a model request without exposing Provider selection to its caller.
_Avoid_: Model string, provider selection, backend, ad hoc route

**Available Model**:
A model owned by a configured Model Provider, with complete context, output, sampling, reasoning, and request-timeout defaults shared by the Model Routes that select it.
_Avoid_: Model Route, Model Provider, active Session model

**Session Model Configuration**:
The Available Model and Reasoning Effort selected together for one Conversation Session and used by its subsequent Agent Runs. Changes do not alter an Agent Run already in progress.
_Avoid_: global Model Route change, Model Provider configuration, mid-run model switch

**Reasoning Effort**:
A five-level intent that asks a Model Provider to trade response capability and thoroughness against latency and cost for a model request.
_Avoid_: Thinking level, token budget, model identity

**Model Provider**:
A configured backend that implements model calls for one or more Model Routes.
_Avoid_: Model Route, model string, gateway

**User Configuration**:
The single account-global persisted configuration that selects runtime, model, and memory behavior at Agent Service startup. Editing it saves settings for the next startup without replacing the current service's active configuration. The Default Conversation Workspace preference is read from the latest saved configuration when Web starts a new conversation.
_Avoid_: Agent profile, Session override, per-chat settings, identity prompt, repair mode
