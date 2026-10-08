---
status: accepted
---

# Use Run-Local Tool Permission Snapshots

Aide authorizes detached invocation facts against immutable Run snapshots, keeping reusable Tools free of mutable authorization state. Permission selection is described in [README](../../README.md#使用须知). Full-Access does not provide an OS sandbox.

| Level | Workspace File reads | Workspace File writes | External Files / MCP |
| --- | --- | --- | --- |
| `read-only` | Direct | Confirm each call | Confirm each call |
| `workspace-write` | Direct | Direct | Confirm each call |
| `full-access` | Direct | Direct | Direct |

## Capture and authorization

Each client owns its foreground permission selection across Session switches and reconnect grace. A foreground Run captures the selection and resolved Exec Shell before its first asynchronous preparation. Configuration updates change the default for clients without an explicit selection. User Schedule Runs capture the latest configured permission and Shell when execution begins, independently of a client's temporary selection; occurrence admission facts and confirmation identity remain with the scheduler. Snapshots and approvals are runtime-only.

The Gateway prepares normalized arguments, validates capabilities and business rules, then opens one authorization session. Invalid arguments and hard failures remain errors at every level. Approval applies to one normalized invocation and is never cached. Decline or unavailable presentation prevents execution; confirmation ownership and lifecycle follow [ADR-0029](0029-host-cli-and-web-through-one-local-service.md).

File classification uses canonical host paths, resolving links, junctions, drive/UNC rules, and the nearest existing ancestor for missing targets. Workspace containment is never a string-prefix check. Internal Skill Loader reads are outside model-issued File authorization; reading a Skill through a model Tool remains an ordinary File access.

## Exec

Inspection and execution use the same resolved PowerShell executable and canonical cwd with profiles disabled. Lower permission levels directly execute only fixed command/parameter grammars whose identity and FileSystem paths satisfy policy. Dynamic syntax, unknown parameters, pipeline-fed paths, untrusted command identity, and external paths require confirmation.

Approved Git reads additionally constrain executable identity, repository location, and effective include/filter configuration. The Host fixes configuration, hooks, pager, prompt, and external-diff behavior; an uncertain audit requires confirmation.

A missing Shell is a capability error. Failed or inconsistent inspection and catastrophic operations require confirmation at every level. Full-Access permits ordinary parseable dynamic commands while retaining those checks and normal Tool failures.

## Schedule

Foreground Schedule listing is direct. Mutations require confirmation in Read-Only. Adding a Job also requires confirmation when the configured Schedule level exceeds the Run's captured level; combined reasons produce one confirmation. Removal has no escalation rule. Permission selection never enters Job persistence.

User Schedule Runs use the same File, Exec, Web, and MCP policies through their admission snapshots and background confirmation envelopes. Dream and other System Jobs retain their private internal execution path.

## Web Fetch

Preparation normalizes the URL and rejects credentials or unsupported schemes. Execution audits the initial target and every redirect hop using one controlled DNS resolution per hop. A target is public only when its nonempty complete address set is globally routable. Unsafe or uncertain hops require confirmation at the lower levels; Full-Access skips that permission prompt while preserving DNS, TLS, HTTP, and other execution failures.

One invocation can request at most one confirmation. Approval grants no persistent network access and does not retry failed DNS. Redirect following is explicit: each normalized hop is resolved, authorized, and bound before connecting. The HTTP adapter uses the audited addresses without a second unrestricted lookup or environment proxy, retaining Host, SNI, and certificate validation. An opaque remote URL reader cannot enforce this boundary and is therefore excluded.
