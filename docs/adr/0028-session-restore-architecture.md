---
status: accepted
---

# Define the Session Restore Architecture

This ADR records the durable architecture for [Add `/restore` for Session
Restore with optional File Restore (#262)](https://github.com/Totoro-debug/MyClaw/issues/262).
It is the ADR deliverable for [#263](https://github.com/Totoro-debug/MyClaw/issues/263)
and is a constraint for the later implementation tasks. It changes no
production behavior by itself.

## Scope and vocabulary

Session Restore is a Management Command for the current foreground
Conversation Session. A persisted foreground User message is a Restore Anchor.
The anchor stores a Session-scoped monotonically increasing integer that is
never reused, an internal run token, and the complete Session-owned state from
immediately before that User message. Only a committed User message receives
the numeric anchor. A failed or canceled run that reaches the terminal Session
commit still creates an anchor; a queued input or an abandoned run that never
commits its User message does not.

Selecting an anchor truncates that User message and every later Session
message. The same Session ID remains active, including when the result is an
empty Session. Session-owned state restored from the anchor includes the title,
Blackboard, Action Summary, `last_compacted`, token usage, and other validated
Session metadata. The restore sets `updated_at` to the restore time. Anchor
fields and the saved pre-input state are persistence details: they do not enter
the Model Request Context, provider-visible message projection, or terminal
conversation projection.

File Restore is the optional Session Restore mode. It covers only authorized,
actual mutations attempted by the foreground Built-in `write_file` and
`edit_file` Tools. It may include targets outside the Workspace when those
Tools were authorized to reach them. Exec, MCP, Schedule, Dream, manual edits,
Conversation Summary, Long-term Memory, Tool Artifacts, and Session Log are not
independent rollback targets.

## Capture and durable records

The foreground Agent Run captures its pre-input Session state before title
work or other asynchronous foreground changes. It creates a run token for
File Backup association. The numeric Restore Anchor is assigned atomically
when the User message is committed. A foreground run-local recorder is passed
through the Tool execution path; mutable anchor state is not attached to
shared Tool instances.

For an authorized `write_file` or `edit_file` mutation, the Tool Gateway has
already completed preparation and permission handling before the recorder
resolves the canonical target and attempts a File Backup immediately before
the actual mutation. The backup is either the exact pre-write bytes or an
explicit nonexistence state. The recorder observes the post-write existence and
hash when possible. Repeated writes to one canonical target are grouped, and
File Restore uses the earliest pre-write state in the selected active range.
Aliases are grouped to their canonical target; a retargeted link is never
followed to a different target during replay.

Restore records are Workspace State under one Session-specific directory:

```text
.myclaw/restore/<session_id>/
  state.json                  # next operation number, coverage, active branch
  entries/<operation>.json    # run token, canonical target, before/after state
  blobs/<opaque-name>.bin
  latest-safety/
  pending.json
```

The recorder persists the blob and journal entry before allowing the Tool
mutation to continue. A failed backup never raises into the original Tool
execution contract and produces no immediate user warning. When possible, the
recorder writes a Backup Gap marker instead. A Backup Gap means that an
eligible Built-in Tool modification continued without a usable File Backup.
If both backup persistence and Gap-marker persistence fail and the process
crashes, the missing coverage can remain unknowable; this is an accepted
limitation.

Journal records retain the run token. A record whose run never commits a User
message is orphaned after a crash and is not mapped to a Restore Anchor. It is
therefore not silently treated as a tracked branch mutation.

## Inspection and transaction order

`RestoreManager.inspect()` produces an immutable plan containing the Session
ID, anchor, removal counts, a digest of the current serialized Session, the
active journal revision, eligible targets, latest observed post-write state,
external target counts, and any Backup Gap in the selected active range.
`revalidate()` runs after the runtime idle barriers and before final
confirmation; execution revalidates again before its first mutation. A stale
plan fails without mutation.

File mode is unavailable when the selected active range contains a known
Backup Gap. A Gap outside that range does not disable File Restore. When the
range has no tracked writes, the workflow goes directly to conversation-only
confirmation. A file conflict, including changed bytes or changed existence,
does not disable File Restore; it is attempted and reported.

After final confirmation, the operation cannot be canceled. The transaction
first captures a complete internal safety snapshot of the pre-restore Session
bytes, active journal state, and current bytes or nonexistence for every target
that may be changed. It then durably writes `pending.json` containing the
anchor, mode, immutable target list, operation phase, per-target progress, and
failure-modal acknowledgement state. If required safety data or the pending
intent cannot be written, the operation aborts before changing any file or
Session state.

The `latest-safety` directory retains exactly one complete snapshot per
Session as a recovery checkpoint, not as a user-visible branch or history
feature. It remains after its transaction completes until a newer complete
snapshot replaces it. A new snapshot is published only after it is complete;
only then does it replace the previous snapshot. The snapshot may contain
external file bytes. Those bytes never enter prompts, logs, Tool Results, or
terminal previews. The design introduces no size cap or quota.

File replay processes each unique target independently. A replaced link,
directory, device, unsafe path, or runtime-owned target that cannot be safely
reloaded is a per-file failure and is left alone. If current bytes already
match the recorded target, replay reports success without writing. Otherwise,
it atomically replaces the regular file from the File Backup or deletes a file
whose recorded state was nonexistence. Parent directories and filesystem
metadata are not restored.

A successfully restored conflict is listed in the ordinary Session Restore
result. A target that cannot be safely restored is listed by path in the
failure result, while replay continues for every other target. The Session is
strictly durably truncated even when one or more files fail, so a partial File
Restore never prevents the Conversation Session rollback. A failure modal is
shown only when at least one file fails and lists failed paths plus successful
conflicts; its acknowledgement is durable so an interrupted restart can show
it again. Conversation-only restore leaves files unchanged and removes the
discarded turns' journal entries from the active branch, while the safety
snapshot retains the pre-restore state for recovery.

The current Session JSONL is written by Session Restore itself, not by the
generic file replay loop. If the strict Session write fails, `pending.json`
remains authoritative and normal conversation is blocked until startup
recovery completes.

## Runtime boundaries

### Conversation Session and Memory System

Session Restore owns only the current Conversation Session, its Short-term
Memory projection, Session messages, Session metadata, and the active restore
journal branch. It restores the Session-owned Blackboard, Action Summary,
`last_compacted`, token usage, title, and related metadata from the anchor.

The Workspace Memory System remains outside the restore boundary. Conversation
Summary, Long-term Memory, the Summary Cursor, and Dream results are not rolled
back. Tool Artifacts and Session Log diagnostics remain available. A later
File Restore may overwrite a file that one of these out-of-scope actors changed
if that file has an eligible foreground File Backup, but those actors' state is
never itself rewound.

### Schedule

Session Restore does not roll back Schedule Jobs or Schedule Sessions. Before
the irreversible confirmation, Schedule Service uses a separate natural idle
barrier: it prevents new admission and waits for active occurrences,
Background Tool Confirmations, Dream work, and terminal commits to finish
naturally. The existing canceling `pause_and_drain()` behavior remains a
separate operation. After a successful restore or cancellation, Schedule
admission resumes.

### Exec and MCP

Exec and MCP mutations are outside File Backup capture and are never replayed
or undone as independent effects. The same rule applies to MCP connection
state and external side effects. If Exec or MCP later changes a target that has
an eligible foreground File Backup, File Restore may overwrite that target as
part of best-effort restoration; this does not imply rollback of the Exec or
MCP operation itself.

### Protected restore state

ADR-0005 permits the fixed File Tools to access Workspace State through normal
Workspace path resolution. The protected `.myclaw/restore/` subtree is an
explicit write-side exception: Built-in File Tools must reject direct writes
there, so a recorder cannot journal its own journal or corrupt a pending
transaction. The restore recorder and transaction manager own the internal
restore records. Other `.myclaw` runtime-owned files remain subject to their
existing rules; a target owned by a live runtime store is restored only when it
can be safely reloaded, otherwise it is reported as a file failure.

## Startup recovery

Pending restore recovery runs in this order:

1. Validate Workspace State and the protected restore records.
2. Invoke `RestoreManager.recover_pending()` before loading Memory Manager,
   Schedule Store, or any foreground Conversation Session.
3. Resume the durable target list and replay it idempotently. Do not rewrite a
   target that already has the desired bytes or desired absence.
4. Complete the strict Session write, finalize the active journal branch, and
   persist the final pending state and failure acknowledgement.
5. Only after recovery succeeds, load the Workspace Memory System, Schedule,
   and the foreground Session, then rebuild the Runtime Generation for the
   retained Session ID.

Recovery uses the durable target list and per-target results, so interruption
after any durable phase converges on one final Session state. If the Session
cannot be persisted, startup reports a blocking Workspace error and retries on
the next startup. Recovery does not modify Conversation Summary, Long-term
Memory, Schedule state, Exec/MCP state, or diagnostic records.

## Consequences and limits

File Restore is a recorded-change mechanism rather than a filesystem snapshot.
An external edit made before the first eligible Tool mutation is part of that
Tool's captured pre-write state. Restore intentionally covers file bytes and
existence only; permissions, ACLs, timestamps, and directories are not
restored. A complete storage failure followed by a crash can leave one Tool
write indistinguishable from an external change. These limits preserve the
normal Tool result and permission behavior while making confirmed Session
restoration recoverable.
