---
status: accepted
---

# Retain Execution Outcomes Without Protocol Payloads

Execution Trace focuses on model outputs, requested Tool invocations, outcomes, failure reasons, timing, and reported token usage. Model records link to the Tool records they request. Retain protocol type and endpoint metadata, but stop retaining complete model request contexts, Tool definitions, opaque provider continuation state, and transport payloads such as HTTP bodies, SSE, and MCP messages. Tool records retain invocation arguments and the final returned result; raw process output and content before conversion are not additional Trace copies.

Keep the compact Trace independently queryable in the global SQLite store rather than resolving its output through Conversation Session files. Some output therefore remains duplicated in Session history, in exchange for preserving inspection after Session Restore and covering auxiliary work whose output is absent from that history. Oversized Tool results may use existing Tool Artifact references instead of copying the referenced content into Trace.

Retain Trace for 30 days. Statistics cover the records still retained; deleting a Session clears its associated Trace, while Session Restore retains historical Trace with its existing restored marker. Design for a fresh Trace store: existing Trace is outside this change's scope, and no legacy compatibility, record migration, or automatic database-reset logic is added.

If a model stream fails or is cancelled after semantic output has reached the consumer, retain that partial output with a failed outcome, its failure reason, and an explicit incomplete-output marker. Partial output covers text and reasoning already delivered through the existing semantic stream; Tool calls are recorded once a complete semantic Tool call is available, without reconstructing executable calls from unfinished protocol fragments. This does not require retaining protocol chunks or changing retry, fallback, cancellation, or stream delivery behavior. Unreported usage remains unknown.

The default detail view emphasizes name, status, call time, duration, and model input/output token usage, followed by model output and linked Tools or Tool arguments and the final result. Failure reasons are prominent. Protocol, endpoint, purpose, retries, cache usage, and reasoning effort belong in an expandable section; internal identity fields remain available for behavior rather than being displayed as a field inventory.

This revises the payload-retention and one-year-retention trade-offs in [ADR-0031](0031-store-execution-trace-in-sqlite.md).
