---
status: accepted
---

# Use an Active In-Memory Session with Ordered Snapshots

Each active `Session` is the authority for one Conversation Session. Foreground Sessions have exclusive client claims; Schedule Sessions belong to their dedicated Job Loops. It owns JSON-native messages, metadata, `last_compacted`, identity, timestamps, and the strict compact JSONL snapshot boundary; it does not own model calls, Tool execution, foreground presentation, or runtime lifecycle. Distinct Sessions may be active concurrently under [ADR-0029](0029-host-cli-and-web-through-one-local-service.md).

`commit_agent_run()` validates and deep-copies a complete terminal Agent Run increment, then publishes messages, `last_compacted`, Action Summary, usage, and metadata through one in-memory state replacement before scheduling one persisted snapshot. New empty Sessions remain memory-only. A persisted Session contains one exact header followed by user, assistant, and Tool message dictionaries; schema-versioned or malformed histories are rejected without migration, repair, or version dispatch.

`persist()` captures a complete deep-copied snapshot and schedules ordered atomic replacement with at most three asynchronous attempts and 100 ms then 200 ms backoff. Exhausted ordinary failures are silent and do not change the Agent Run outcome. `close()` performs the same bounded three-attempt final save synchronously, while `abandon()` cancels pending snapshots, rejects later mutation, and performs no final save.

Conversation Summary updates `last_compacted` without a cross-file transaction, so summary and Session state may diverge after a crash. CLI and Web processes coordinate through the service rather than independently loading the same Session. Forced Session abort may discard state that was not already persisted. Restore's strict durable writes follow [ADR-0028](0028-session-restore-architecture.md).
