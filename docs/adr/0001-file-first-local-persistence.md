---
status: accepted
---

# Use File-First Local Persistence

Omni stores inspectable local files instead of a database. One shared service coordinates concurrent clients; atomic replacement alone cannot coordinate independent in-memory runtimes. Runtime ownership is defined in [ADR-0029](0029-host-cli-and-web-through-one-local-service.md).

## Ownership

Agent Home (`~/.omni/`) owns configuration, user-authored Skills, Project registrations, conversation-directory indexes, and service discovery and locks. Directory catalogs hold references, never copies of Workspace data.

Each Workspace owns its state under `<workspace>/.omni/`:

```text
.omni/
  .gitignore
  memory/
    memory.md
    summary.jsonl
    .cursor
  sessions/<session_id>.jsonl
  schedule-sessions/schedule_<job_id>.jsonl
  artifacts/<session_id>/<tool_call_id>.txt
  logs/<session_id>.log
  restore/<session_id>/
  session-deletions/
  schedule.json
```

CLI uses the startup directory; Web uses a Project or the configured default conversation directory. Directory identity is canonicalized. Omni does not infer a Git root or redirect failed Workspace initialization to another store. Inside Agent Home, only the direct `chat` directory may hold Workspace State; links and redirected state paths cannot widen that exception.

## Publication and consistency

An active Session is authoritative for its messages and metadata. A terminal Agent Run publishes one validated in-memory increment before scheduling an ordered atomic JSONL snapshot. Empty drafts stay in memory. Ordinary Session saves use bounded retries and may fail without changing the Run outcome; strict Restore writes instead block further conversation until recovery succeeds.

Stores validate their own formats. Atomic publication is a per-store guarantee: Session state, Conversation Summary, Memory, and Schedule have no common transaction. Cancellation cannot undo accepted Tool effects or earlier Memory writes. Tool Artifacts are direct writes; Session Logs are best-effort diagnostics, with no durability or redaction guarantee.

Built-in File Tools may access Workspace State under [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md). Direct writes to `.omni/restore/` are rejected so Tools cannot corrupt their own backup journal. Restore ownership and recovery are defined in [ADR-0028](0028-session-restore-architecture.md).
