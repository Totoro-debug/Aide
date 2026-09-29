---
status: accepted
---

# Host CLI and Web through One Local Service

The CLI currently owns one in-process foreground Session, Message Bus, Schedule Service, Memory Manager, and Tool Confirmation presenter. Independent CLI and Web processes cannot safely coordinate concurrent Sessions in the same Workspace through the existing files: Schedule and Memory state can diverge, and multiple consumers would split Message Bus output. To support concurrent Sessions from both interfaces, one local service will own active Workspace runtimes and exclusive Session Claims. CLI and Web are clients of that service; the Web app, management API, and live event channel share one loopback port.

Each registered Project remains a reference to its existing Workspace directory. The service runs Schedule Jobs for registered Projects while it is active. Removing a Project cancels its foreground and Schedule work, releases its Session Claims, and removes only the registration; re-registering a directory with saved Jobs requires an explicit decision before scheduling resumes. Distinct Sessions in a Workspace may run concurrently, while one Session may be loaded by only one client at a time. After the last client disconnects, the service allows a 30-second reconnect period, then drains active work and exits.

This replaces the CLI-only composition-root and presenter ownership described in [ADR-0017](0017-use-cli-composition-root-and-session-scoped-agent-loop.md) and [ADR-0027](0027-runtime-lifetime-tool-confirmation-coordinator.md). The implementation must preserve existing CLI interaction and Session persistence behavior while adding a service-owned event broker, cross-client confirmation presentation, and a single owner for each Workspace's Schedule and Memory state. Sharing files between independent runtimes was rejected because per-file atomic writes do not coordinate in-memory state, dispatch, or confirmation decisions. An always-on daemon was rejected because the service should stop when no CLI or Web client remains.

Requirements: [Local Web interface and shared multi-session service (#281)](https://github.com/Totoro-debug/MyClaw/issues/281).
