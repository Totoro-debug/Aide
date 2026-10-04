---
status: accepted
---

# Use File-First Local Persistence

Omni stores configuration, Project registrations, and Workspace-owned state as inspectable local files instead of using a database or mixed storage model. The shared local service is the runtime authority for concurrent CLI and Web clients; atomic file replacement alone does not coordinate independent runtimes. Service ownership follows [ADR-0029](0029-host-cli-and-web-through-one-local-service.md).

## Agent Home and Workspace boundaries

Agent Home is fixed at `~/.omni/` for the current operating-system account, without profiles or configurable data roots. It owns global `config.toml`, user-authored `skills/`, the durable `projects.json` catalog, the `conversation-workspaces.json` directory index, and service discovery and locking state. Both catalogs contain normalized directory references, not copies of Workspace data or service-lifetime Workspace IDs. Legacy Agent Home Runtime Log files remain untouched.

Each Workspace owns its non-global persistent state under `<workspace>/.omni/`. CLI startup selects the current directory; Web selects a registered Project or the Default Conversation Workspace, initially `~/.omni/chat`. The saved default-directory preference applies to the next non-Project conversation immediately; old Sessions remain in their original directories. Directory identity is normalized and resolved for shared runtime ownership. Omni does not infer a Git root, search ancestors, or fall back to Agent Home or temporary storage when Workspace State cannot be initialized safely.

The dedicated Web entry may create a missing default directory. Workspace State inside Agent Home is permitted only for the direct `chat` directory; symlinks, Windows junctions and redirected state paths cannot widen this exception to other Agent Home management directories. File Tool authorization and protected Restore paths retain their existing rules. History discovery reads Session headers without activating Workspace resources or returning conversation bodies. New Session metadata records immutable `creation_scope` (`chat` or `project`), while legacy Sessions are classified by their normalized directory's current Project registration. Empty drafts stay in memory; Restore preserves creation scope.

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

Startup creates the root, internal Git ignore rule, `memory/`, `sessions/`, and a missing Long-term Memory template. Dream System Job registration creates or reconciles `schedule.json`; other paths are created on demand. Known records validate their own formats, while unknown entries remain untouched.

## Publication and access

Each store defines its publication guarantees: Conversation Sessions and other declared stores use atomic replacement where specified; Tool Artifacts use direct writes. Each active in-memory `Session` is authoritative for its own Agent Runs. Foreground Sessions use exclusive client claims; Schedule Sessions belong to their dedicated Job Loops. Conversation Summary and Session snapshots have no cross-file transaction and may diverge after a crash. Session persistence is defined by [ADR-0009](0009-active-session-snapshot-persistence.md), and diagnostics by [ADR-0008](0008-use-workspace-session-log.md).

Each persisted Schedule Job has a strict canonical object shape with a required `title`. The decoder accepts a document containing only the exact pre-title shape and derives titles using Session title normalization. The in-memory Jobs become canonical immediately; the next successful Store mutation rewrites the file. Mixed versions, partial hybrids, and unknown fields are rejected; a failed write leaves the previous document authoritative.

Fixed File Tools can access Workspace State through normal path resolution and [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md) authorization. Direct writes to the protected `.omni/restore/` subtree are rejected under [ADR-0028](0028-session-restore-architecture.md). The Skill Loader's internal reads follow [ADR-0016](0016-use-agent-home-skill-catalog-and-progressive-loading.md); Agent Home as a whole grants no exemption from external-path Tool Confirmation.
