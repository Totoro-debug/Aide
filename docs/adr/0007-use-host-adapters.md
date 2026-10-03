---
status: accepted
---

# Use Windows Runtime and Filesystem Boundaries

Omni runs only on Windows. Application, service, and validation entry points reject other operating systems before creating runtime state. The check occurs at execution time, so module discovery, static analysis, and package builds can import modules without starting the application.

The host filesystem module retains its public persistence interface and always uses the Windows adapter. Native path conversion, UNC and extended paths, case-insensitive containment, reserved names, alternate streams, hard links, junction/reparse checks, atomic replacement, cross-process locks, ACL protection, and handle-pinned deletion remain concentrated behind this boundary. Session Restore uses the same synchronization boundary and retains existing Windows persistence behavior and schemas.

Exec uses the Workspace-generation PowerShell Host defined in [ADR-0010](0010-fixed-tool-catalog-and-base-tool-boundaries.md), retaining Windows PowerShell 5.1 and PowerShell 7. The package is pure Python and emits one `py3-none-any` wheel; that packaging tag does not widen the runtime support contract. Windows x64 is currently validated; this decision adds no CPU architecture commitment. Release acceptance requires both real PowerShell hosts and the complete Windows filesystem and application scenarios.
