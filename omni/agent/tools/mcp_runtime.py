"""Runtime-Lifetime state for configured MCP Server connections."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from loguru import logger

from omni.agent.tools.mcp import MCPServerConnection, MCPTool
from omni.config.config import MCPServerConfiguration
from omni.utils.async_tasks import await_task_preserving_cancellation

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


type MCPConnectionFactory = Callable[[MCPServerConfiguration, Path], MCPConnectionAdapter]


@dataclass(frozen=True, slots=True)
class _ConnectionAttempt:
    tools: tuple[MCPTool, ...] | None
    failure: MCPServerFailure | None


@dataclass(frozen=True, slots=True)
class MCPStartupReport:
    """The initial generation's MCP Tool Snapshot and failure metadata."""

    snapshot: MCPToolSnapshot
    failed_servers: tuple[str, ...]
    failures: tuple[MCPServerFailure, ...] = ()
    skipped_tool_counts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class MCPSnapshotReport:
    """A candidate MCP Tool Snapshot prepared for a later generation."""

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


class MCPRuntimeManager:
    """Connect configured MCP Servers and prepare immutable generation snapshots."""

    def __init__(
        self,
        workspace: Path,
        *,
        connection_factory: MCPConnectionFactory | None = None,
        built_in_names: Iterable[str] = (),
    ) -> None:
        if not isinstance(workspace, Path):
            raise TypeError("MCP Runtime Manager requires a Path workspace")
        self._workspace = workspace
        self._connection_factory = connection_factory or _default_connection_factory
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
        self._pending_report: MCPSnapshotReport | None = None
        self._started = False

    async def start(
        self,
        configuration: Mapping[str, MCPServerConfiguration],
    ) -> MCPStartupReport:
        """Connect enabled Servers and activate the initial generation snapshot."""
        normalized = _normalize_configuration(configuration)
        if self._started or self._connections:
            await self.close()
        self._pending_report = None

        connections: dict[str, MCPConnectionAdapter] = {}
        failed: set[str] = set()
        failures: dict[str, MCPServerFailure] = {}
        for mcp_name, server_configuration in normalized.items():
            if not server_configuration.enabled:
                continue
            try:
                connections[mcp_name] = self._connection_factory(
                    server_configuration,
                    self._workspace,
                )
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
        return MCPStartupReport(
            snapshot=self._snapshot,
            failed_servers=tuple(sorted(self._failed_servers)),
            failures=self._failure_report(),
            skipped_tool_counts=self._skipped_tool_report(),
        )

    async def prepare_generation(self) -> MCPSnapshotReport:
        """Prepare a candidate snapshot while retaining the active snapshot.

        Healthy connections and their discovered Tool definitions are reused. Only
        Servers marked unavailable, or whose previous connection attempt failed,
        are connected again. The caller chooses when to activate the returned
        candidate through :meth:`activate_generation`.
        """
        if not self._started:
            raise RuntimeError("MCP Runtime Manager has not been started")
        self._pending_report = None

        self._sync_unavailable_servers()
        retry_names = tuple(sorted(self._failed_servers))
        retry_connections: dict[str, MCPConnectionAdapter] = {}
        retry_failures: dict[str, MCPServerFailure] = {}
        for mcp_name in retry_names:
            connection = self._connections.get(mcp_name)
            if connection is not None:
                retry_connections[mcp_name] = connection
                continue
            server_configuration = self._configuration[mcp_name]
            try:
                connection = self._connection_factory(server_configuration, self._workspace)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                retry_failures[mcp_name] = _record_server_failure(
                    mcp_name,
                    phase="factory",
                    error=error,
                )
                continue
            self._connections[mcp_name] = connection
            retry_connections[mcp_name] = connection

        attempts = await self._connect_many(retry_connections)

        for mcp_name in retry_names:
            attempt = attempts.get(mcp_name)
            if attempt is None or attempt.tools is None:
                self._failed_servers.add(mcp_name)
                failure = retry_failures.get(mcp_name)
                if attempt is not None:
                    failure = attempt.failure or failure
                if failure is not None:
                    self._failures[mcp_name] = failure
                continue
            self._failed_servers.discard(mcp_name)
            self._failures.pop(mcp_name, None)
            self._discovered_tools[mcp_name] = attempt.tools

        candidate = self._build_snapshot()
        report = MCPSnapshotReport(
            snapshot=candidate,
            failed_servers=tuple(sorted(self._failed_servers)),
            failures=self._failure_report(),
            skipped_tool_counts=tuple(
                (mcp_name, count)
                for mcp_name, count in self._skipped_tool_report()
                if mcp_name in retry_names
            ),
        )
        self._pending_report = report
        return report

    def activate_generation(
        self,
        report: MCPSnapshotReport,
    ) -> MCPToolSnapshot:
        """Publish a previously prepared candidate as the active snapshot."""
        if not isinstance(report, MCPSnapshotReport):
            raise TypeError("MCP generation activation requires an MCP snapshot report")
        if not self._started:
            raise RuntimeError("MCP Runtime Manager has not been started")
        if report is not self._pending_report:
            raise ValueError("MCP snapshot report is not the current prepared candidate")
        self._snapshot = report.snapshot
        self._pending_report = None
        return self._snapshot

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
        self._pending_report = None
        self._started = False
        await _close_connections(connections)

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

    def _sync_unavailable_servers(self) -> None:
        for mcp_name, connection in self._connections.items():
            if connection.unavailable:
                self._failed_servers.add(mcp_name)

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


def _default_connection_factory(
    configuration: MCPServerConfiguration,
    workspace: Path,
) -> MCPServerConnection:
    return MCPServerConnection(
        configuration,
        workspace,
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


def _sorted_tools(tools: Sequence[MCPTool]) -> tuple[MCPTool, ...]:
    valid_tools = tuple(tool for tool in tools if isinstance(tool, MCPTool))
    return tuple(
        sorted(
            valid_tools,
            key=lambda tool: (
                tool.remote_name,
                tool.description,
                tool.name,
                json.dumps(tool.parameters, sort_keys=True, separators=(",", ":")),
            ),
        )
    )


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
    "MCPSnapshotReport",
    "MCPStartupReport",
    "MCPToolSnapshot",
    "allocate_mcp_tool_name",
]
