---
status: accepted
---

# Use Run-Local Tool Permission Levels and Foreground Snapshots

MyClaw exposes three configured Tool Permission Levels through
`[runtime].permission_level`:

- `read-only` permits internal Workspace File reads directly. Every File write
  and every external File access requires one confirmation for that call.
- `workspace-write` permits internal Workspace File reads and writes directly.
  Every external File access requires one confirmation for that call.
- `full-access` permits valid File accesses directly. It removes permission
  prompts, not argument validation, business refusals, capability errors, or
  execution errors.

The default remains `workspace-write`. Each service client owns one
`RuntimePermissionControl` containing the configured level and current
selection. Its Session Loops receive the same control object, so Session or
Workspace switching does not reset the current value. A foreground Agent
Run captures one immutable `PermissionSnapshot` before Session title work,
manual Skill resolution, or Task Framing. The snapshot contains the current
Tool Permission Level and the Workspace generation's resolved Exec Shell. Changing
the control during a run affects only a later run. Selection and capture are
synchronous operations confined to the service event-loop thread, so there is no
await boundary at which a caller can observe a partially updated selection.

## Authorization Boundary

`ToolGateway.call()` remains the only public Tool invocation seam. The
Gateway parses and prepares normalized arguments, performs validation and
business refusal checks, collects detached invocation facts, opens one
authorization session, requests at most one confirmation for that normalized
call, and only then executes it. Confirmation state is run-local and
call-local; it is never cached in a shared Tool instance.

The foreground Runtime Context and the foreground run Gateway receive the
same snapshot object. Runtime Context reports the permission level, resolved
Exec Shell family, and that permission checks may require confirmation. The
Gateway uses the snapshot for File policy and the strict PowerShell
Exec policies described below; Web Search remains direct and Web Fetch uses
the execution-time per-hop authorization described below.

## Exec Permission Mapping

Exec inspection is a detached Host fact collection that happens after
argument normalization and before authorization. A missing selected shell is
a capability error at every permission level. When the selected shell exists,
an unavailable, failed, timed-out, malformed, or inconsistent inspector
requires one confirmation at every level. Catastrophic Exec matches also
require one confirmation at every level.

For PowerShell 5.1 and 7, Read-Only and Workspace-Write direct execution is
limited to fixed candidate command and parameter grammars. Read candidates are
`Get-ChildItem`, `Get-Content`, `Get-Item`, `Get-Location`, `Get-FileHash`,
`Measure-Object`, `Select-Object`, `Sort-Object`, `Select-String`, `Test-Path`,
`Resolve-Path`, `Format-List`, `Format-Table`, and `Out-String`. Workspace-Write
also permits `New-Item`, `Set-Content`, `Add-Content`, `Clear-Content`,
`Copy-Item`, `Move-Item`, `Rename-Item`, `Remove-Item`, and `Out-File`.
The Host returns canonical command identity, expected Microsoft
module, resolution count, and static path-role facts. The policy canonicalizes
those paths using the Exec cwd and Workspace root, accepts only the FileSystem
Provider, and applies Read-Only or Workspace-Write path rules. Unknown or
dynamic syntax, untrusted identity, pipeline-fed paths, unknown parameters,
and external paths require one confirmation. Full-Access directly executes
parseable non-catastrophic dynamic commands, while still enforcing normal
argument validation, capability checks, business refusals, and Tool errors.

Approved Git read forms are `status`, `diff`, `log`, `show`, `branch --list`,
`rev-parse`, and `ls-files`, using a unique native executable outside the Workspace.
The Host fixes Git configuration, pager, external-diff/textconv, fsmonitor,
hook, prompt, and optional-lock behavior through the process environment and
adds `--no-ext-diff --no-textconv` to `diff` and `show`. Static inspection does
not launch Git. After Workspace-aware canonicalization proves the executable
is external and every repository directory is internal, the exact resolved
Git executable audits effective local/worktree include and clean/process
filter configuration. Positive, failed, or ineligible audits, unlisted Git
forms, and Workspace-resident Git executables require confirmation at the
lower levels.

## Foreground Management and Client Lifetime

The service retains one `RuntimePermissionControl` per client and reuses it
for that client's foreground Sessions. The `/permission` command
selects one of the three levels for later foreground Agent Runs only. The
selection is not persisted to User Configuration, Conversation Session, or
Schedule. Session switching and reconnect grace retain it; expiry ends the
client selection, and a new client starts from the active configured value.
Global configuration activation updates the configured default while
preserving an explicit client selection; clients without an explicit selection
adopt the new default. Ownership follows [ADR-0029](0029-host-cli-and-web-through-one-local-service.md).

The selector reports the current level and leaves it unchanged when the same
level is submitted. An upgrade to `full-access` opens a warning with Cancel
focused by default on every attempt. Full-Access removes ordinary permission
confirmation for foreground File Tools, parseable non-catastrophic
PowerShell Exec calls, and MCP Tool invocations; it is not an
operating-system sandbox and does not bypass validation, capability checks,
business refusals, catastrophic or uncertain Exec confirmation, or Tool
errors. Web Search is direct at every level. Web Fetch uses the per-hop
network authorization below. Foreground Read-Only and Workspace-Write MCP
calls each request one confirmation bound to the normalized server/tool
identity and complete arguments; approval is never cached. User Schedule
Agent Runs use the admission snapshot through their run-local Gateway and
enter the background confirmation coordinator at the lower levels. Full-Access
calls directly. A
CLI startup notice is shown at most once when the configured level is
Full-Access. `/config` reports the configured level, and `/status` reports both
configured and current foreground levels.

## Foreground Schedule Management

The fixed Schedule Tool remains catalogued, exposed, searchable, and schema-
identical at every permission level. Its `list` action is direct for every
foreground Run. Its `add` and `remove` actions require one confirmation in
Read-Only and are direct in Workspace-Write and Full-Access. The confirmation
is still required for a mutation even when the configured level is lower than
the current foreground level.

The generation-owned Gateway retains its immutable configured Schedule
level. A foreground Run captures the selected current level in its
`PermissionSnapshot`. For `add` only, a configured level strictly above the
captured current level adds an escalation reason. If the Run is Read-Only,
that reason is merged with the CRUD reason into one stable, duplicate-free
confirmation. `remove` never uses the configured-versus-current escalation
rule. The total order is Read-Only < Workspace-Write < Full-Access.

Schedule Tool facts are collected after argument normalization, validation,
business refusal, and the existing missing-Job preflight. Confirmation details
contain only the canonical invocation: `action` plus normalized `message`,
`title`, and `schedule` for `add`; `action` plus `job_id` for `remove`; and
`action` for `list`. A declined request performs no Store or Service mutation,
and approval is limited to that call. Invalid input, missing or nonexistent
Jobs, Store/Service failures, and other hard Tool errors remain errors at every
level; Full-Access does not bypass them. Schedule Job persistence and public
JSON contain no permission level or snapshot.

At User Schedule occurrence admission, Schedule Service captures the immutable
active generation's configured level and resolved Exec Shell in a runtime
`PermissionSnapshot`. The exact snapshot is projected into the Schedule Runtime
Context and the run-local Gateway. Read-Only and Workspace-Write apply the same
structured File, Exec, Web Fetch, and per-call MCP rules as the corresponding
foreground level; Full-Access removes ordinary permission prompts but retains
hard errors and catastrophic/uncertain Exec confirmation. Each low-level MCP
call therefore enters the shared background confirmation coordinator, with no
confirmation cache. Dream and other System Schedule Jobs remain outside this
model and keep their internal direct behavior. The snapshot, owner, and
presentation envelope remain runtime-only and never enter `ScheduleJob`,
Session, Tool Result, or public JSON.

## File Facts and Host Paths

Read File, List Dir, Glob, and Grep emit canonical host `FileAccess` facts with
role `read`. Write File emits a `write` fact, while Edit File emits both `read`
and `write` facts for its target. Canonicalization uses the declared base path,
host path case and drive/UNC rules, symlink/junction/reparse resolution, and the
nearest existing ancestor for a missing write target. The policy compares
canonical paths with the canonical Workspace root; string prefixes are not
used.

Foreground model File Tools do not inherit the Skill Root exemption used by
internal Skill loading. A foreground model-issued File access beneath
`~/.myclaw/skills` is an ordinary external access unless it is also beneath
the Workspace. User Schedule Agent Runs use their admission snapshot for the
same canonical File policy. Dream's private memory Tool and Runtime persistence
writes remain outside this model File policy.

Permission classification happens after preparation and business refusal, so
invalid arguments, hard errors, capability errors, and execution errors do
not become permission prompts. A declined confirmation never reaches the
execution boundary.

## Web Fetch Network Authorization

Web Fetch validates and normalizes the URL during preparation. The normalized
target records the lowercase HTTP scheme, IDNA host, effective port, and
request URL; URL credentials and unsupported schemes remain invalid. Execution
creates no shared Tool state. One per-call authorization session audits the
initial target and every redirect hop.

For each hop, the controlled asynchronous resolver is called once and its
complete address set is retained. A target is public only when the set is
non-empty and every IPv4 or IPv6 address is globally routable. Private,
loopback, link-local, reserved, unspecified, multicast, IPv4-mapped
non-global, mixed, empty, DNS-failed, and DNS-timeout results are unsafe or
uncertain. Read-Only and Workspace-Write request confirmation for an unsafe
hop; Full-Access skips only that permission prompt. DNS, connection, TLS,
HTTP status, body, decoding, and timeout failures remain ordinary Web Fetch
Tool Errors at every level.

The session permits at most one confirmation for the normalized invocation.
Approval covers only that call and does not create a host, IP, or network
grant. A declined or unavailable confirmation prevents the connector from
opening the unsafe hop. If DNS fails, a low-permission approval still returns
the recorded DNS Tool Error and does not retry resolution.

Redirects are handled by Web Fetch with automatic client redirect following
disabled. Each `Location` is resolved relative to the current URL, normalized
and validated, resolved, authorized, and bound before its connection is
opened. The HTTP adapter receives only the address set audited for the current
hop; its custom resolver disables a second unconstrained DNS lookup, and
`trust_env=False` prevents environment proxy settings from bypassing the
audit. Target retrieval is not delegated to an opaque remote URL reader,
because such a service could hide target-side redirects from this boundary.
Requests carry the normalized URL's Host and TLS certificate hostname so
direct IPv4/IPv6 dialing preserves port, Host, SNI, and certificate validation
semantics. Web Fetch sends no cross-origin sensitive headers.

## Scope Boundaries

This decision does not alter the fixed Tool Catalog, Tool Exposure, Tool
Activation, Tool Search, MCP discovery, transport, or schema projection.
It does not provide an operating-system sandbox. The service's confirmation
queues, client audiences, reconnect behavior, and cancellation ownership are
defined in [ADR-0029](0029-host-cli-and-web-through-one-local-service.md).

The existing Skill Loader remains responsible for its own internal reads, and
ADR-0016 no longer defines a confirmation-free boundary for model-issued
`read_file` calls. Exec uses the same structured, detached authorization facts
for its PowerShell policies. Calls without a foreground permission
snapshot use explicit origin and fact-based policy decisions; they do not enter
an alternate Tool authorization path.

Consequences: every Gateway authorization decision is expressed with structured,
detached facts and one call-local authorization session. No Tool supplies a
free-form authorization reason or bypasses the policy boundary. The foreground
run has a stable authorization view even if runtime control state changes while
the model is working.
