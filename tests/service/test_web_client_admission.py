"""One Web participant coexists with CLI clients without weakening Session Claims."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio
from aiohttp.test_utils import TestServer

from aide.config.config import ConfigLoader
from aide.service.discovery import create_credential
from aide.service.errors import ServiceError
from aide.service.runtime import AgentService, ClientState
from aide.service.transport import AgentServiceTransport
from tests.fixtures import FakeClock
from tests.service.test_service_concurrency import _CollectingSink
from tests.service.test_service_transport import _prepare_agent_home


@pytest_asyncio.fixture
async def admission_service(
    tmp_path: Path,
) -> AsyncIterator[tuple[AgentService, FakeClock, asyncio.Event]]:
    home = _prepare_agent_home(tmp_path / "agent-home")
    clock = FakeClock(datetime(2026, 10, 7, tzinfo=UTC))
    timer = asyncio.Event()

    async def sleep(_seconds: float) -> None:
        await timer.wait()

    service = AgentService(
        home, ConfigLoader(home).load_for_startup(), monotonic_now=clock.monotonic, sleep=sleep
    )
    await service.start()
    try:
        yield service, clock, timer
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_ten_web_registrations_admit_one_alongside_three_cli_clients(
    admission_service: tuple[AgentService, FakeClock, asyncio.Event],
) -> None:
    service, _clock, _timer = admission_service
    attempts = await asyncio.gather(
        *(service.register_client("web") for _ in range(10)), return_exceptions=True
    )
    winners = [result for result in attempts if isinstance(result, ClientState)]
    assert len(winners) == 1
    failures = [result for result in attempts if isinstance(result, ServiceError)]
    assert len(failures) == 9
    assert all(error.code == "web_client_exists" and error.status == 409 for error in failures)
    assert all(not error.retryable for error in failures)
    clients = [winners[0], *[await service.register_client("cli") for _ in range(3)]]
    for client in clients:
        await service.connect_client(client.client_id, _CollectingSink())
    assert sum(client.connected for client in service._clients.values()) == 4


@pytest.mark.asyncio
async def test_unconnected_web_reservation_expires_without_deadline_extension(
    admission_service: tuple[AgentService, FakeClock, asyncio.Event],
) -> None:
    service, clock, timer = admission_service
    client = await service.register_client("web")
    expiry = client.disconnect_task
    assert expiry is not None
    clock.advance(29)
    recovered = await service.register_client("web", client.reconnect_credential)
    assert recovered is client
    clock.advance(1)
    with pytest.raises(ServiceError, match="expired"):
        await service.register_client("web", client.reconnect_credential)
    timer.set()
    await asyncio.wait_for(expiry, timeout=2)
    replacement = await service.register_client("web")
    assert replacement.client_id != client.client_id


@pytest.mark.asyncio
async def test_pending_web_launch_survives_last_connected_cli_departure(
    admission_service: tuple[AgentService, FakeClock, asyncio.Event],
) -> None:
    service, _clock, _timer = admission_service
    cli = await service.register_client("cli")
    await service.connect_client(cli.client_id, _CollectingSink())
    web = await service.register_client("web")
    await service.disconnect_client(cli.client_id)
    await asyncio.sleep(0)
    assert service.state == "reconnecting"
    await service.connect_client(web.client_id, _CollectingSink())
    assert service.state == "ready"


@pytest.mark.asyncio
async def test_web_reconnect_preserves_claim_and_cli_exclusion(
    admission_service: tuple[AgentService, FakeClock, asyncio.Event], tmp_path: Path,
) -> None:
    service, clock, _timer = admission_service
    web_client = await service.register_client("web")
    cli = await service.register_client("cli")
    await service.connect_client(web_client.client_id, _CollectingSink())
    await service.connect_client(cli.client_id, _CollectingSink())
    directory = tmp_path / "workspace"
    directory.mkdir()
    workspace = await service.attach_workspace(web_client.client_id, directory)
    session = await workspace.create_draft(web_client.client_id, creation_scope="chat")
    await service.claim(web_client.client_id, workspace.workspace_id, session)
    claim = workspace._claims[session]
    await service.disconnect_client(web_client.client_id)
    clock.advance(29)
    with pytest.raises(ServiceError) as occupied:
        await service.register_client("web")
    assert occupied.value.code == "web_client_exists"
    recovered = await service.register_client("web", web_client.reconnect_credential)
    await service.connect_client(recovered.client_id, _CollectingSink())
    assert workspace._claims[session] is claim
    assert claim.status == "claimed"
    with pytest.raises(ServiceError) as claimed:
        await workspace.claim(cli.client_id, session)
    assert claimed.value.code == "session_claimed"


@pytest.mark.asyncio
async def test_web_place_remains_reserved_until_claim_cleanup_finishes(
    admission_service: tuple[AgentService, FakeClock, asyncio.Event], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, clock, timer = admission_service
    cli = await service.register_client("cli")
    await service.connect_client(cli.client_id, _CollectingSink())
    client = await service.register_client("web")
    await service.connect_client(client.client_id, _CollectingSink())
    directory = tmp_path / "workspace"
    directory.mkdir()
    workspace = await service.attach_workspace(client.client_id, directory)
    entered, finish = asyncio.Event(), asyncio.Event()
    original = workspace.expire_client

    async def cleanup(client_id: str) -> None:
        entered.set()
        await finish.wait()
        await original(client_id)

    monkeypatch.setattr(workspace, "expire_client", cleanup)
    await service.disconnect_client(client.client_id)
    expiry = client.disconnect_task
    assert expiry is not None
    clock.advance(30)
    timer.set()
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        with pytest.raises(ServiceError) as busy:
            await service.register_client("web")
        assert busy.value.code == "web_client_exists"
    finally:
        finish.set()
        await asyncio.wait_for(expiry, timeout=2)
    replacement = await service.register_client("web")
    await service.connect_client(replacement.client_id, _CollectingSink())


@pytest.mark.asyncio
async def test_duplicate_browser_and_ticket_preserve_original_control(
    admission_service: tuple[AgentService, FakeClock, asyncio.Event],
) -> None:
    service, _clock, _timer = admission_service
    create_credential(service.agent_home)
    transport = AgentServiceTransport(service)
    cli = await service.register_client("cli")
    await service.connect_client(cli.client_id, _CollectingSink())
    async with TestServer(transport.create_app()) as server:
        origin = str(server.make_url("")).rstrip("/")
        async with aiohttp.ClientSession(cookie_jar=aiohttp.CookieJar(unsafe=True)) as browser:
            ticket = service.open_web_interface(cli.client_id, "launch")["ticket"]
            transport._web_tickets[ticket] = float("inf")
            response = await browser.post(
                server.make_url("/api/v1/web/ticket"), headers={"Origin": origin},
                json={"ticket": ticket},
            )
            assert response.status == 200
            csrf = (await response.json())["csrf_token"]
            cookies = list(browser.cookie_jar)
            response = await browser.post(
                server.make_url("/api/v1/clients"),
                headers={"Origin": origin, "X-Aide-CSRF": csrf},
                json={"request_id": "register", "kind": "web"},
            )
            assert response.status == 200
            registered = await response.json()
            socket = await browser.ws_connect(
                server.make_url("/api/v1/events"), headers={"Origin": origin},
                protocols=("aide-v1", registered["web_control_credential"]),
            )
            try:
                for _ in range(2):
                    response = await browser.post(
                        server.make_url("/api/v1/clients"),
                        headers={"Origin": origin, "X-Aide-CSRF": csrf},
                        json={"request_id": "duplicate-tab", "kind": "web"},
                    )
                    assert response.status == 409
                    assert (await response.json())["code"] == "web_client_exists"
                async with aiohttp.ClientSession(
                    cookie_jar=aiohttp.CookieJar(unsafe=True)
                ) as foreign:
                    for target in (browser, foreign):
                        next_ticket = service.open_web_interface(cli.client_id, "again")["ticket"]
                        transport._web_tickets[next_ticket] = float("inf")
                        response = await target.post(
                            server.make_url("/api/v1/web/ticket"),
                            headers={"Origin": origin}, json={"ticket": next_ticket},
                        )
                        assert response.status == 409
                        assert (await response.json())["code"] == "web_client_exists"
                        assert "Set-Cookie" not in response.headers
                assert list(browser.cookie_jar) == cookies
                response = await browser.get(
                    server.make_url("/api/v1/config"),
                    headers={"Origin": origin, "X-Aide-Control": registered["web_control_credential"]},
                )
                assert response.status == 200
                assert not socket.closed
            finally:
                await socket.close()
