---
status: accepted
---

# Expose All User Capabilities through Agent Service

Evolve the existing Agent Service into the sole boundary for every capability the Personal Agent offers its user. CLI and Web selectively consume these capabilities through their own interfaces; capability coverage in the service does not require identical feature coverage in both interfaces.

The change must also simplify CLI and Web by moving user-operation business rules and orchestration into the service. Merely wrapping existing service methods while retaining business workflows in the interfaces would not satisfy this decision. The existing shared service is chosen over an additional application facade or a separate process because it already owns execution, authorization, shared resources, and lifecycle coordination.

## Confirmed responsibility boundary

Agent Service owns complete user operations, including configuration loading and validation, message submission and FIFO admission, Session creation and claiming, model selection, cancellation, confirmation, and management operations. It owns operation ordering, business validation, and failure handling. CLI and Web invoke named operations, subscribe to their state, and handle UI presentation; neither implements a user capability or coordinates a business workflow.

Thin Python and TypeScript client adapters contain connection mechanics, protocol encoding and decoding, event delivery, and transport failure reporting. These adapters expose the service's named operations and state subscriptions. They do not own business rules, message admission, an independent Agent input queue, configuration validation, or multi-step user-operation orchestration.

Service discovery and process bootstrap belong to a reusable launcher adapter. CLI supplies launch options and displays the resulting service status or error. Startup configuration loading, diagnostics, validation, and repair eligibility belong to Agent Service, including when configuration is missing or invalid.

## User capabilities and UI state

User capabilities are operations the user can directly request, including sending messages, managing queued input, invoking Management Commands, and manually invoking Skills. Tools are Agent execution capabilities: Agent code invokes them through Tool Gateway. This decision does not add direct user-facing Tool execution endpoints.

UI retains unsent message drafts, configuration form drafts, focus, scrolling, disclosure, and other presentation state. It submits configuration edits through the service's Config interface. The service validates the complete candidate, coordinates saving and configuration conflicts, and owns persistence; invalid or unfinished form contents remain visible in UI without becoming saved configuration. This does not introduce service-owned configuration editor drafts.

Management Command recognition, Manual Skill Invocation recognition, input classification, permission checks, and queue admission are service-owned operations. A client may present command or Skill completion from published capability metadata, but it cannot be the authority that decides how an input executes.

## Atomic queued-input recall

The service provides one operation that atomically recalls all ordinary inputs in the current Conversation Session that have not started executing. It returns their original text in FIFO order for UI editing. The active Agent Run continues; Management Commands and Tool Confirmation never enter this queue.

Recall and starting a queued Run share the service's queue coordination boundary. A client displays the actual recalled set returned by the service rather than inferring it from a local cache. Request idempotency preserves the original result and cannot recall newer inputs on retry. Recalled inputs must be removed from execution and their live queue state must be resolved consistently.

## Existing behavior to carry through the boundary change

User operations remain individually identifiable. Session model selection is saved independently from input submission; a rejected input does not undo a successfully saved selection. FIFO inputs capture the model combination when their Agent Run starts, rather than when input is accepted. Centralizing these operations must preserve those execution and persistence boundaries.

The service already supports startup with missing or invalid configuration while exposing configuration repair and disabling Agent work. A thin CLI must obtain configuration status and sanitized diagnostics from this service path rather than parse configuration independently before connecting.

This decision records the accepted architectural direction and responsibility boundary. The authoritative user requirements, independently deliverable Tasks, and quantified acceptance criteria are tracked in [Unified user capabilities and simplified CLI/Web (#325)](https://github.com/Totoro-debug/OmniAgent/issues/325).
