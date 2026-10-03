"""Background process entry point for the per-Agent-Home local service."""

from __future__ import annotations

import argparse
import asyncio
import errno
import os
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from aiohttp import web

from omni.config.agent_home import AgentHome
from omni.config.config import ConfigError
from omni.service.discovery import (
    DEFAULT_SERVICE_HOST,
    DEFAULT_SERVICE_PORT,
    ServiceDiscovery,
    create_credential,
    read_credential,
    read_discovery,
    remove_credential,
    remove_discovery,
    startup_lock,
    write_discovery,
)
from omni.service.errors import ServiceError
from omni.service.runtime import LocalService
from omni.service.transport import create_app
from omni.utils.platform import WINDOWS_REQUIRED_ERROR, is_windows_host


async def serve_service(
    agent_home: AgentHome,
    *,
    host: str = DEFAULT_SERVICE_HOST,
    port: int = DEFAULT_SERVICE_PORT,
    reconnect_timeout: float = 30.0,
) -> ServiceDiscovery:
    """Run one owned service until its lifecycle reaches ``stopped``."""
    if host != DEFAULT_SERVICE_HOST:
        raise ServiceError(
            "validation_error", "The local service must bind IPv4 loopback.", status=422
        )
    service = LocalService(
        agent_home,
        reconnect_timeout=reconnect_timeout,
    )
    runner = web.AppRunner(create_app(service), access_log=None)
    discovery: ServiceDiscovery | None = None
    credential_created = False
    try:
        with startup_lock(agent_home):
            try:
                existing = read_discovery(agent_home)
                existing_token = read_credential(agent_home)
            except (OSError, ValueError):
                existing = None
                existing_token = None
            if existing is not None and existing_token is not None:
                from omni.service.client import ServiceClient

                if await ServiceClient._probe(existing, existing_token) is not None:
                    raise ServiceError(
                        "service_already_running",
                        "An authenticated local service already owns this Agent Home.",
                    )
            await runner.setup()
            site = web.TCPSite(runner, host=host, port=port)
            try:
                await site.start()
            except OSError as error:
                if error.errno in {errno.EADDRINUSE, 10048}:
                    raise ServiceError(
                        "service_port_occupied",
                        "The local service port is occupied by another process.",
                    ) from error
                raise
            await service.start()
            bound_port = _bound_port(site)
            create_credential(agent_home)
            credential_created = True
            discovery = ServiceDiscovery(
                service_instance_id=service.service_instance_id,
                protocol_version=service.protocol_version,
                host=host,
                port=bound_port,
                pid=os.getpid(),
            )
            write_discovery(agent_home, discovery)
        await service.wait_closed()
        assert discovery is not None
        return discovery
    except BaseException:
        if service.state != "stopped":
            with suppress(BaseException):
                await service.stop()
        raise
    finally:
        if discovery is not None:
            remove_discovery(agent_home, instance_id=discovery.service_instance_id)
        if credential_created:
            remove_credential(agent_home)
        with suppress(BaseException):
            await runner.cleanup()


def _bound_port(site: web.TCPSite) -> int:
    server = cast(Any, site._server)
    if server is None or not server.sockets:
        raise RuntimeError("Local service did not expose a listening socket")
    address = server.sockets[0].getsockname()
    port = address[1]
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise RuntimeError("Local service listening port is invalid")
    return port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MyClaw local service.")
    parser.add_argument("--agent-home", type=Path, required=True)
    parser.add_argument("--host", default=DEFAULT_SERVICE_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_SERVICE_PORT)
    parser.add_argument("--reconnect-timeout", type=float, default=30.0)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    if not is_windows_host():
        print(WINDOWS_REQUIRED_ERROR, file=sys.stderr)
        return 1
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(
            serve_service(
                AgentHome(args.agent_home),
                host=args.host,
                port=args.port,
                reconnect_timeout=args.reconnect_timeout,
            )
        )
    except ConfigError as error:
        print(f"{error.error.code}: {error.error.message}")
        return 2
    except ServiceError as error:
        print(f"{error.code}: {error.message}")
        return 2
    except OSError:
        print("service_start_failed: The local service could not start.")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(run())


__all__ = ["build_parser", "run", "serve_service"]
