---
status: accepted
---

# Store Execution Trace in SQLite

Payload retention and the expiry period below are superseded by [ADR-0033](0033-retain-execution-outcomes-without-protocol-payloads.md); the global SQLite ownership and read boundary remain in effect.

Execution Trace uses SQLite for persisted records and raw captured data, supporting global statistics and detailed Session inspection. This is a deliberate exception to [file-first local persistence](0001-file-first-local-persistence.md): keeping JSONL authoritative while maintaining a separate query index would require coordinating two representations of the same captured work.

The Agent Service owns one global database at `~/.aide/traces.sqlite3`, with each record associated with its originating Workspace and Session when applicable. A global store supports historical statistics without relying on directory discovery: the current persistent directory catalogs do not enumerate every Workspace used by CLI Clients. This gives Trace an Agent Home lifetime rather than making it travel with a Workspace directory.

Capture and persistence are best effort and must not block normal execution or replace a Model or Tool outcome with a Trace failure. Bounded buffering and background writes may absorb temporary failures; exhausted buffers, process interruption, or unavailable storage may leave missing Trace data.

For a model call that eventually succeeds, retain the final successful attempt's raw data and the retry count. Earlier failed attempts' payloads are discarded. This trades the ability to inspect individual recovered failures for a smaller, simpler retained history.

Execution status has only success and failure. An ultimately failed model call records a failed status and the relevant error content; cancellation and refusal also count as failure, with their reasons in the error content.

Capture raw data at the available transport or tool boundary. Web Search deliberately retains the complete results returned by DDGS rather than its internal HTTP traffic, avoiding coupling Trace to private dependency internals. Captured records identify their actual source layer.

Deleting a Session deletes its associated Trace data; otherwise retained records expire after one year.

The Web Interface reads persisted, finalized records through a dedicated authenticated read boundary, including records from Sessions claimed by CLI Clients. Trace inspection requires no Session Claim and presents no in-progress calls or live event stream.
