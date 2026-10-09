"""Test observation and HTTP conveniences around the real ServiceClient."""

from collections.abc import Callable, Mapping
from contextlib import suppress

import aiohttp

from aide.config.agent_home import AgentHome
from aide.service.client import ServiceClient, ServiceStartupError
from aide.service.discovery import ServiceDiscovery


class ObservedServiceClient(ServiceClient):
    """Observe events after the production client accepts and processes them."""

    def add_state_listener(
        self, listener: Callable[[Mapping[str, object]], None]
    ) -> Callable[[], None]:
        self._test_listeners.add(listener)
        return lambda: self._test_listeners.discard(listener)

    async def _handle_event(
        self, event: Mapping[str, object], *, recover_display: bool = False,
    ) -> None:
        await super()._handle_event(event, recover_display=recover_display)
        for listener in tuple(self._test_listeners):
            with suppress(Exception):
                listener(event)

    def __init__(
        self, *, agent_home: AgentHome, discovery: ServiceDiscovery, token: str,
        http: aiohttp.ClientSession, client_id: str, reconnect_credential: str,
    ) -> None:
        self._test_listeners: set[Callable[[Mapping[str, object]], None]] = set()
        super().__init__(agent_home=agent_home, discovery=discovery, token=token, http=http,
                         client_id=client_id, reconnect_credential=reconnect_credential)

    async def get_runtime_memory(self) -> dict[str, object]:
        """Read the current Workspace's Long-term Memory through its named operation."""
        response = await self._named_session_operation("memory/read")
        if not isinstance(response.get("content"), str):
            raise ServiceStartupError("service_protocol_error", "Memory response is invalid.")
        return response

    async def reload_runtime_skills(self) -> dict[str, object]:
        """Reload the shared Skill catalog and return published metadata."""
        response = await self._named_session_operation("skills/reload")
        skills = response.get("skills")
        if not isinstance(skills, list) or any(
            not isinstance(item, dict) for item in skills
        ):
            raise ServiceStartupError("service_protocol_error", "Skill response is invalid.")
        return response
