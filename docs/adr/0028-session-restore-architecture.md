---
status: accepted
---

# Make Session Restore a Durable Transaction

Session Restore uses a recorded-change journal rather than a filesystem snapshot. Ordinary File Tool writes remain usable when backup fails; a confirmed Session rollback instead requires durable intent and startup recovery. The user-facing rollback scope is documented in [README](../../README.md#使用须知).

## Anchors and capture

A committed foreground User message receives a Session-scoped monotonically increasing anchor and saved pre-input Session state. A failed or cancelled Run still receives an anchor if its terminal increment commits; queued or abandoned inputs do not. Restore truncates the anchor and later messages while retaining Session identity and the current model selection.

A Run-local recorder captures authorized, actual `write_file` and `edit_file` mutations immediately before execution. It records exact pre-write bytes or nonexistence, canonical target identity, and observable post-write state. Repeated writes group by canonical target, using the earliest pre-write state in the selected range. A retargeted link is never followed during replay. Shared Tool instances carry no mutable anchor state.

Records and blobs live under `.aide/restore/<session_id>/`. Backup failure does not replace the original Tool result. When possible it records a Backup Gap, disabling File Restore for that range. A total storage failure and crash can leave missing coverage unknowable. Journal entries whose Run never committed remain orphaned rather than becoming tracked anchor mutations.

## Transaction

1. Validate the client's Session Claim and establish Workspace Restore readiness, waiting for title work, Session persistence, and Schedule natural idle. Block new admission without cancelling accepted Schedule work.
2. Inspect and revalidate an immutable plan against the serialized Session and journal revision. A stale plan fails before mutation; a known Backup Gap prevents file mode.
3. After final confirmation, capture a complete safety snapshot and durably publish `pending.json` with mode, targets, phase, and progress. If required safety data or intent cannot be written, abort before changing files or Session state. The confirmed transaction cannot be cancelled.
4. Replay unique targets independently. Atomically restore regular-file bytes or delete a file whose recorded state was absent. Unsafe targets fail individually; replay continues. Conflicting current bytes may be overwritten and are reported.
5. Strictly persist the truncated Session and finalize the journal branch even when some files failed. A Session write failure keeps the pending transaction authoritative and blocks normal conversation until recovery succeeds.

The latest safety snapshot is published only when complete and retains one checkpoint, including external target bytes when applicable. Those bytes never enter model prompts or diagnostics. Failure acknowledgement is durable so restart can finish presenting a partial result.

## Recovery and isolation

Startup validates restore records and resumes pending work before loading Workspace Memory, Schedule, or foreground Sessions. Replay is idempotent: a target already matching the desired state is not rewritten. Recovery converges on the same Session result after interruption at any durable phase.

Restore replaces only the target Session authority and rotates its Claim; shared resources remain owned by [ADR-0029](0029-host-cli-and-web-through-one-local-service.md). It does not rewind Workspace Summary, Long-term Memory, Schedule, Exec/MCP effects, Artifacts, or Logs. A later change by one of those actors to a tracked file may still be overwritten during file replay.

Direct Built-in File Tool writes to the protected restore subtree are rejected. Other live runtime-owned targets may be replayed only when safely reloadable. File replay restores bytes and existence, not directories, timestamps, ACLs, or other metadata. Backups have no quota; best-effort capture cannot guarantee complete coverage after storage failure.
