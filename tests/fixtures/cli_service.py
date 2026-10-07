"""Serve the real local service for CLI and Textual adapter tests."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aiohttp.test_utils import TestServer

from aide.config.agent_home import AgentHome
from aide.config.config import ConfigLoader
from aide.service.discovery import ServiceDiscovery, create_credential, write_discovery
from aide.service.runtime import AgentService
from aide.service.transport import create_app


@asynccontextmanager
async def cli_service(home: AgentHome) -> AsyncIterator[AgentService]:
    service = AgentService(home, ConfigLoader(home).load_for_startup(), reconnect_timeout=3600)
    server = TestServer(create_app(service), host="127.0.0.1")
    create_credential(home)
    try:
        await service.start()
        await server.start_server()
        assert server.port is not None
        write_discovery(
            home,
            ServiceDiscovery(
                service.service_instance_id,
                service.protocol_version,
                "127.0.0.1",
                server.port,
                os.getpid(),
            ),
        )
        yield service
    finally:
        await server.close()
        await service.stop()
