"""Runtime-Lifetime state for configured MCP Server connections."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from loguru import logger

from aide.agent.tools.mcp import MCPServerConnection, MCPTool
from aide.config.config import MCPServerConfiguration, MCPTransport
from aide.utils.async_tasks import await_task_preserving_cancellation

_MAX_MODEL_TOOL_NAME_LENGTH = 64
_MODEL_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

type MCPToolSnapshot = tuple[MCPTool, ...]


@dataclass(frozen=True, slots=True)
class MCPServerFailure:
    """Safe metadata for one failed MCP Server lifecycle attempt."""

    mcp_name: str
    phase: str
    exception_type: str


class MCPConnectionAdapter(Protocol):
    """The connection seam owned by the MCP Runtime Manager."""

    @property
    def unavailable(self) -> bool: ...

    async def connect(self) -> tuple[MCPTool, ...]:
        """Return discovered Tools already named with allocate_mcp_tool_name()."""
        ...

    async def close(self) -> None: ...


type MCPConnectionFactory = Callable[
    [MCPServerConfiguration, Path | None], MCPConnectionAdapter
]


@dataclass(frozen=True, slots=True)
class _ConnectionAttempt:
    tools: tuple[MCPTool, ...] | None
    failure: MCPServerFailure | None


@dataclass(frozen=True, slots=True)
class MCPStartupReport:
    """The startup MCP Tool Snapshot and failure metadata."""

    snapshot: MCPToolSnapshot
    failed_servers: tuple[str, ...]
    failures: tuple[MCPServerFailure, ...] = ()
    skipped_tool_counts: tuple[tuple[str, int], ...] = ()


def allocate_mcp_tool_name(mcp_name: str, remote_name: str) -> str | None:
    """Allocate the documented provider-safe name for one remote Tool.

    The fallback is considered only when the preferred name exceeds the provider's
    maximum length. Invalid candidates are rejected without character replacement
    or truncation; collisions are handled by the Runtime Manager.
    """
    if not isinstance(mcp_name, str) or not mcp_name:
        return None
    if not isinstance(remote_name, str) or not remote_name:
        return None

    preferred = f"mcp_{mcp_name}_{remote_name}"
    candidate = f"mcp_{remote_name}" if len(preferred) > _MAX_MODEL_TOOL_NAME_LENGTH else preferred
    return candidate if _MODEL_TOOL_NAME_PATTERN.fullmatch(candidate) else None


def _record_server_failure(
    mcp_name: str,
    *,
    phase: str,
    error: Exception,
) -> MCPServerFailure:
    failure = MCPServerFailure(
        mcp_name=mcp_name,
        phase=phase,
        exception_type=type(error).__name__,
    )
    logger.opt(exception=error).error(
        "MCP Server failure mcp_name={} phase={} type={}",
        failure.mcp_name,
        failure.phase,
        failure.exception_type,
    )
    return failure


def _log_skipped_tool(
    mcp_name: str,
    *,
    phase: str,
    exception_type: str,
) -> None:
    logger.error(
        "MCP Tool skipped mcp_name={} phase={} type={}",
        mcp_name,
        phase,
        exception_type,
    )


def _skipped_tool_count(connection: MCPConnectionAdapter) -> int:
    count = getattr(connection, "skipped_tool_count", 0)
    if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
        return count
    return 0


@dataclass(slots=True)
class _SharedConnection:
    configuration: MCPServerConfiguration
    connection: MCPConnectionAdapter
    users: int = 0
    connecting: asyncio.Task[tuple[MCPTool, ...]] | None = None


class _ConnectionPool:
    """Retain a connection until all configuration snapshots stop using it."""

    def __init__(self, factory: MCPConnectionFactory, workspace: Path | None) -> None:
        self._factory = factory
        self._workspace = workspace
        self._entries: list[_SharedConnection] = []

    def acquire(self, configuration: MCPServerConfiguration) -> _ConnectionLease:
        identity = replace(configuration, tool_keywords={})
        entry = next((item for item in self._entries if item.configuration == identity), None)
        if entry is None:
            entry = _SharedConnection(identity, self._factory(configuration, self._workspace))
            self._entries.append(entry)
        entry.users += 1
        return _ConnectionLease(self, entry)

    async def release(self, entry: _SharedConnection) -> None:
        entry.users -= 1
        if entry.users:
            return
        self._entries.remove(entry)
        if entry.connecting is not None and not entry.connecting.done():
            entry.connecting.cancel()
            await asyncio.gather(entry.connecting, return_exceptions=True)
        await entry.connection.close()


class _ConnectionLease:
    def __init__(self, pool: _ConnectionPool, entry: _SharedConnection) -> None:
        self._pool = pool
        self._entry = entry
        self._closed = False

    @property
    def unavailable(self) -> bool:
        return self._closed or self._entry.connection.unavailable

    @property
    def skipped_tool_count(self) -> int:
        return _skipped_tool_count(self._entry.connection)

    async def connect(self) -> tuple[MCPTool, ...]:
        if self._entry.connecting is None:
            self._entry.connecting = asyncio.create_task(self._entry.connection.connect())
        return await asyncio.shield(self._entry.connecting)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._pool.release(self._entry)


class MCPRuntimeManager:
    """Own MCP connections for one runtime scope and prepare tool snapshots.

    HTTP connections belong to the service and have no Workspace. Stdio
    connections belong to one explicit Workspace.
    """

    def __init__(
        self,
        workspace: Path | None,
        *,
        connection_factory: MCPConnectionFactory | None = None,
        built_in_names: Iterable[str] = (),
        transport: MCPTransport,
    ) -> None:
        if workspace is not None and not isinstance(workspace, Path):
            raise TypeError("MCP Runtime Manager requires a Path workspace or None")
        self._workspace = workspace
        if transport not in {"stdio", "streamable-http"}:
            raise ValueError("MCP Runtime Manager transport selection is invalid")
        if (transport == "stdio") != (workspace is not None):
            raise ValueError("Stdio requires a Workspace; HTTP requires no Workspace")
        self._connection_factory = connection_factory or _default_connection_factory
        self._connection_pool = _ConnectionPool(self._connection_factory, workspace)
        self._transport = transport
        names = list(built_in_names)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Built-in Tool names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("Built-in Tool names must be unique")
        self._built_in_names = frozenset(names)
        self._configuration: dict[str, MCPServerConfiguration] = {}
        self._connections: dict[str, MCPConnectionAdapter] = {}
        self._discovered_tools: dict[str, tuple[MCPTool, ...]] = {}
        self._failed_servers: set[str] = set()
        self._failures: dict[str, MCPServerFailure] = {}
        self._skipped_tool_counts: dict[str, int] = {}
        self._snapshot: MCPToolSnapshot = ()
        self._startup_report: MCPStartupReport | None = None
        self._started = False

    def fork(self) -> MCPRuntimeManager:
        """Create a new scope snapshot sharing unchanged Server connections."""
        manager = MCPRuntimeManager(
            self._workspace, connection_factory=self._connection_factory,
            built_in_names=self._built_in_names, transport=self._transport,
        )
        manager._connection_pool = self._connection_pool
        return manager

    async def start(
        self,
        configuration: Mapping[str, MCPServerConfiguration],
    ) -> MCPStartupReport:
        """Connect enabled Servers and publish the startup snapshot."""
        normalized = _select_transport(_normalize_configuration(configuration), self._transport)
        if self._started or self._connections:
            await self.close()

        connections: dict[str, MCPConnectionAdapter] = {}
        failed: set[str] = set()
        failures: dict[str, MCPServerFailure] = {}
        for mcp_name, server_configuration in normalized.items():
            if not server_configuration.enabled:
                continue
            try:
                connections[mcp_name] = self._connection_pool.acquire(server_configuration)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failed.add(mcp_name)
                failures[mcp_name] = _record_server_failure(
                    mcp_name,
                    phase="factory",
                    error=error,
                )

        try:
            attempts = await self._connect_many(connections)
        except BaseException:
            await _close_connections(connections.values())
            raise

        discovered: dict[str, tuple[MCPTool, ...]] = {}
        for mcp_name, attempt in attempts.items():
            if attempt.tools is None:
                failed.add(mcp_name)
                if attempt.failure is not None:
                    failures[mcp_name] = attempt.failure
                continue
            discovered[mcp_name] = attempt.tools

        self._configuration = normalized
        self._connections = connections
        self._discovered_tools = discovered
        self._failed_servers = failed
        self._failures = failures
        self._started = True
        self._snapshot = self._build_snapshot()
        report = MCPStartupReport(
            snapshot=self._snapshot,
            failed_servers=tuple(sorted(self._failed_servers)),
            failures=self._failure_report(),
            skipped_tool_counts=self._skipped_tool_report(),
        )
        self._startup_report = report
        return report

    async def close(self) -> None:
        """Close all Runtime-Lifetime MCP connections and clear Manager state."""
        connections = tuple(self._connections.values())
        self._configuration = {}
        self._connections = {}
        self._discovered_tools = {}
        self._failed_servers = set()
        self._failures = {}
        self._skipped_tool_counts = {}
        self._snapshot = ()
        self._startup_report = None
        self._started = False
        await _close_connections(connections)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def snapshot(self) -> MCPToolSnapshot:
        if not self._started:
            raise RuntimeError("MCP Runtime Manager has not been started")
        return self._snapshot

    @property
    def startup_report(self) -> MCPStartupReport:
        if self._startup_report is None:
            raise RuntimeError("MCP Runtime Manager has not been started")
        return self._startup_report

    async def _connect_many(
        self,
        connections: Mapping[str, MCPConnectionAdapter],
    ) -> dict[str, _ConnectionAttempt]:
        async def connect_one(
            mcp_name: str,
            connection: MCPConnectionAdapter,
        ) -> tuple[str, _ConnectionAttempt]:
            try:
                tools = await connection.connect()
                if not isinstance(tools, (tuple, list)):
                    raise TypeError("MCP Server connection returned an invalid Tool collection")
                if connection.unavailable:
                    raise RuntimeError("MCP Server connection became unavailable during connect")
                return mcp_name, _ConnectionAttempt(tools=tuple(tools), failure=None)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                return mcp_name, _ConnectionAttempt(
                    tools=None,
                    failure=_record_server_failure(
                        mcp_name,
                        phase="connect",
                        error=error,
                    ),
                )

        tasks = tuple(
            asyncio.create_task(connect_one(mcp_name, connections[mcp_name]))
            for mcp_name in sorted(connections)
        )
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return dict(results)

    def _build_snapshot(self) -> MCPToolSnapshot:
        used_names = set(self._built_in_names)
        snapshot: list[MCPTool] = []
        skipped_tool_counts: dict[str, int] = {}
        for mcp_name in sorted(self._configuration):
            configuration = self._configuration[mcp_name]
            if not configuration.enabled or mcp_name in self._failed_servers:
                continue
            connection = self._connections.get(mcp_name)
            if connection is None or connection.unavailable:
                continue
            named_tools: list[MCPTool] = []
            skipped_count = _skipped_tool_count(connection)
            for tool in _sorted_tools(self._discovered_tools.get(mcp_name, ())):
                if tool.unavailable:
                    skipped_count += 1
                    _log_skipped_tool(mcp_name, phase="tool", exception_type="MCPConnectionError")
                    continue
                named_tools.append(tool)
                if tool.name in used_names:
                    skipped_count += 1
                    _log_skipped_tool(
                        mcp_name,
                        phase="tool_name",
                        exception_type="ToolNameCollision",
                    )
                    continue
                used_names.add(tool.name)
                snapshot.append(tool)
            self._discovered_tools[mcp_name] = tuple(named_tools)
            if skipped_count:
                skipped_tool_counts[mcp_name] = skipped_count
        self._skipped_tool_counts = skipped_tool_counts
        return tuple(snapshot)

    def _failure_report(self) -> tuple[MCPServerFailure, ...]:
        return tuple(
            self._failures[mcp_name]
            for mcp_name in sorted(self._failed_servers)
            if mcp_name in self._failures
        )

    def _skipped_tool_report(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (mcp_name, self._skipped_tool_counts[mcp_name])
            for mcp_name in sorted(self._skipped_tool_counts)
        )


class MCPWorkspaceRuntimeManager:
    """Own one Workspace's Stdio MCP connections and compose its Tool view."""

    def __init__(
        self,
        workspace: Path,
        *,
        shared_runtime: MCPRuntimeManager,
        connection_factory: MCPConnectionFactory | None = None,
        built_in_names: Iterable[str] = (),
    ) -> None:
        if not isinstance(workspace, Path):
            raise TypeError("Workspace MCP Runtime Manager requires a Path workspace")
        if not isinstance(shared_runtime, MCPRuntimeManager):
            raise TypeError("Workspace MCP Runtime Manager requires a shared MCP Runtime Manager")
        self._shared_runtime = shared_runtime
        self._stdio_runtime = MCPRuntimeManager(
            workspace,
            connection_factory=connection_factory,
            built_in_names=built_in_names,
            transport="stdio",
        )
        self._snapshot: MCPToolSnapshot = ()
        self._startup_report: MCPStartupReport | None = None
        self._started = False

    def fork(self, shared_runtime: MCPRuntimeManager) -> MCPWorkspaceRuntimeManager:
        """Keep Stdio connection ownership while changing the shared HTTP snapshot."""
        workspace = self._stdio_runtime._workspace
        assert workspace is not None
        manager = MCPWorkspaceRuntimeManager(
            workspace,
            shared_runtime=shared_runtime,
            built_in_names=self._stdio_runtime._built_in_names,
        )
        manager._stdio_runtime = self._stdio_runtime.fork()
        return manager

    async def start(
        self,
        configuration: Mapping[str, MCPServerConfiguration],
    ) -> MCPStartupReport:
        """Start only this Workspace's Stdio connections and publish its view."""
        if not self._shared_runtime.started:
            raise RuntimeError("Shared MCP Runtime Manager has not been started")
        local_report = await self._stdio_runtime.start(configuration)
        report = self._merge_report(local_report)
        self._snapshot = report.snapshot
        self._startup_report = report
        self._started = True
        return report

    async def close(self) -> None:
        """Close this Workspace's Stdio connections without touching HTTP."""
        await self._stdio_runtime.close()
        self._snapshot = ()
        self._startup_report = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def snapshot(self) -> MCPToolSnapshot:
        if not self._started:
            raise RuntimeError("Workspace MCP Runtime Manager has not been started")
        return self._snapshot

    @property
    def startup_report(self) -> MCPStartupReport:
        if self._startup_report is None:
            raise RuntimeError("Workspace MCP Runtime Manager has not been started")
        return self._startup_report

    def _merge_report(
        self, local_report: MCPStartupReport
    ) -> MCPStartupReport:
        global_report = self._shared_runtime.startup_report
        snapshot, collision_counts = _compose_workspace_snapshot(
            global_report.snapshot,
            local_report.snapshot,
            self._stdio_runtime._built_in_names,
        )
        skipped_tool_counts = _merge_skipped_tool_counts(
            global_report.skipped_tool_counts,
            local_report.skipped_tool_counts,
            collision_counts,
        )
        failures = tuple(
            sorted(
                (*global_report.failures, *local_report.failures),
                key=lambda failure: (failure.mcp_name, failure.phase, failure.exception_type),
            )
        )
        return MCPStartupReport(
            snapshot=snapshot,
            failed_servers=tuple(
                sorted(set(global_report.failed_servers) | set(local_report.failed_servers))
            ),
            failures=failures,
            skipped_tool_counts=skipped_tool_counts,
        )


def _default_connection_factory(
    configuration: MCPServerConfiguration,
    workspace: Path | None,
) -> MCPServerConnection:
    return MCPServerConnection(
        configuration,
        Path(".") if workspace is None else workspace,
        model_name_for=lambda remote_name: (
            allocate_mcp_tool_name(
                configuration.mcp_name,
                remote_name,
            )
            or ""
        ),
    )


def _normalize_configuration(
    configuration: Mapping[str, MCPServerConfiguration],
) -> dict[str, MCPServerConfiguration]:
    if not isinstance(configuration, Mapping):
        raise TypeError("MCP Runtime Manager configuration must be a mapping")
    normalized: dict[str, MCPServerConfiguration] = {}
    for server_configuration in configuration.values():
        if not isinstance(server_configuration, MCPServerConfiguration):
            raise TypeError(
                "MCP Runtime Manager configuration values must be MCPServerConfiguration"
            )
        mcp_name = server_configuration.mcp_name
        if mcp_name in normalized:
            raise ValueError(f"Duplicate MCP Server name: {mcp_name}")
        normalized[mcp_name] = server_configuration
    return normalized


def _select_transport(
    configuration: Mapping[str, MCPServerConfiguration],
    transport: MCPTransport,
) -> dict[str, MCPServerConfiguration]:
    return {
        mcp_name: server_configuration
        for mcp_name, server_configuration in configuration.items()
        if server_configuration.transport == transport
    }


def _compose_workspace_snapshot(
    global_snapshot: Sequence[MCPTool],
    local_snapshot: Sequence[MCPTool],
    built_in_names: frozenset[str],
) -> tuple[MCPToolSnapshot, dict[str, int]]:
    """Compose global and local Tools with the original deterministic ordering."""
    used_names = set(built_in_names)
    snapshot: list[MCPTool] = []
    collision_counts: dict[str, int] = {}
    candidates = [(tool, False) for tool in global_snapshot]
    candidates.extend((tool, True) for tool in local_snapshot)
    for tool, is_local in sorted(candidates, key=lambda item: _tool_sort_key(item[0])):
        if tool.name in used_names:
            collision_counts[tool.server_name] = collision_counts.get(tool.server_name, 0) + 1
            _log_skipped_tool(
                tool.server_name,
                phase="tool_name" if is_local else "workspace_tool_name",
                exception_type="ToolNameCollision",
            )
            continue
        used_names.add(tool.name)
        snapshot.append(tool)
    return tuple(snapshot), collision_counts


def _tool_sort_key(tool: MCPTool) -> tuple[str, str, str, str, str]:
    return (
        tool.server_name,
        tool.remote_name,
        tool.description,
        tool.name,
        json.dumps(tool.parameters, sort_keys=True, separators=(",", ":")),
    )


def _merge_skipped_tool_counts(
    *reports: tuple[tuple[str, int], ...] | Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for report in reports:
        entries = report.items() if isinstance(report, Mapping) else report
        for mcp_name, count in entries:
            counts[mcp_name] = counts.get(mcp_name, 0) + count
    return tuple((mcp_name, counts[mcp_name]) for mcp_name in sorted(counts) if counts[mcp_name])


def _sorted_tools(tools: Sequence[MCPTool]) -> tuple[MCPTool, ...]:
    valid_tools = tuple(tool for tool in tools if isinstance(tool, MCPTool))
    return tuple(sorted(valid_tools, key=_tool_sort_key))


async def _close_connections(connections: Iterable[MCPConnectionAdapter]) -> None:
    async def close_all() -> None:
        results = await asyncio.gather(
            *(connection.close() for connection in connections),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("MCP connection cleanup failed", errors)

    cleanup_task = asyncio.create_task(close_all())
    await await_task_preserving_cancellation(cleanup_task)


__all__ = [
    "MCPConnectionAdapter",
    "MCPConnectionFactory",
    "MCPRuntimeManager",
    "MCPServerFailure",
    "MCPStartupReport",
    "MCPToolSnapshot",
    "MCPWorkspaceRuntimeManager",
    "allocate_mcp_tool_name",
]
