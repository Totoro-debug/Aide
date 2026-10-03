---
status: accepted
---

# Use a Workspace-Owned Session Log

Technical diagnostics belong to one Workspace and Conversation Session and are stored lazily at `.omni/logs/<session_id>.log`. A validated Session context owns a Loguru WARNING-level UTF-8 sink with enqueue enabled, 10 MiB rotation, and retention of one historical file; setup is fail-open and retried on the next context, while context exit removes the sink and allows its queue to drain without a custom deadline.

Same-Session concurrency is unsupported. The design accepts an unbounded queue, infinite drain, no per-record fsync, no active redaction, no control escaping, and per-Session retention without a Workspace-wide size bound. Legacy Agent Home Runtime Log files remain untouched.
