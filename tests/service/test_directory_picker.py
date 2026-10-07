"""Folder selection keeps HTTP responsive and owns its dialog lifetime."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from aiohttp.test_utils import TestClient, TestServer

from omni.config.agent_home import AgentHome
from omni.config.config import ConfigLoader
from omni.service.directory_picker import DirectoryPicker
from omni.service.discovery import create_credential
from omni.service.errors import ServiceError
from omni.service.runtime import AgentService
from omni.service.transport import AgentServiceTransport
from tests.configuration.test_config import MINIMAL_VALID_CONFIG


@pytest_asyncio.fixture
async def picker_http(tmp_path: Path) -> AsyncIterator[TestClient[Any, Any]]:
    home = AgentHome(tmp_path / "agent-home")
    home.initialize()
    (home.path / "config.toml").write_text(MINIMAL_VALID_CONFIG, encoding="utf-8")
    token = create_credential(home)
    service = AgentService(home, ConfigLoader(home).load_for_startup())
    await service.start()
    transport = AgentServiceTransport(service)
    client = await service.register_client("web")
    http = TestClient(
        TestServer(transport.create_app()),
        headers={
            "Authorization": f"Bearer {token}",
            "X-Omni-CSRF": token,
            "X-Omni-Client": client.client_id,
        },
    )
    await http.start_server()
    try:
        yield http
    finally:
        await http.close()
        await service.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [None, "D:\\空 文件夹"])
async def test_selection_returns_directory_or_cancellation_without_registering(
    picker_http: TestClient[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    path: str | None,
) -> None:
    calls = 0

    async def select(self: DirectoryPicker) -> str | None:
        nonlocal calls
        calls += 1
        return path

    monkeypatch.setattr(DirectoryPicker, "_run", select)
    response = await picker_http.post(
        "/api/v1/projects/directory-picker", json={"request_id": "pick"}
    )
    assert response.status == 200
    assert await response.json() == {"request_id": "pick", "path": path}
    assert calls == 1
    projects = await picker_http.get("/api/v1/projects")
    assert await projects.json() == {"projects": []}


@pytest.mark.asyncio
async def test_rejected_requests_never_open_a_dialog(
    picker_http: TestClient[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def select(self: DirectoryPicker) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(DirectoryPicker, "_run", select)
    cases = [
        ({"Authorization": "Bearer wrong"}, {"request_id": "pick"}, 401),
        ({"X-Omni-CSRF": "wrong"}, {"request_id": "pick"}, 403),
        ({"X-Omni-Client": "unknown"}, {"request_id": "pick"}, 401),
        ({"Origin": "https://example.com"}, {"request_id": "pick"}, 403),
        ({}, {}, 422),
    ]
    for headers, body, status in cases:
        response = await picker_http.post(
            "/api/v1/projects/directory-picker",
            headers=headers,
            json=body,
        )
        assert response.status == status, await response.text()
    assert calls == 0


@pytest.mark.asyncio
async def test_open_dialog_allows_status_requests_and_rejects_a_second_dialog(
    picker_http: TestClient[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def select(self: DirectoryPicker) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setattr(DirectoryPicker, "_run", select)
    first = asyncio.create_task(
        picker_http.post(
            "/api/v1/projects/directory-picker",
            json={"request_id": "first"},
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        status = await picker_http.get("/api/v1/service")
        assert status.status == 200
        duplicate = await picker_http.post(
            "/api/v1/projects/directory-picker",
            json={"request_id": "second"},
        )
        assert duplicate.status == 409
        assert (await duplicate.json())["code"] == "directory_picker_busy"
    finally:
        release.set()
        response = await first
        assert response.status == 200


@pytest.mark.asyncio
async def test_dialog_failure_is_safe_and_can_be_retried(
    picker_http: TestClient[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def select(self: DirectoryPicker) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ServiceError("directory_picker_unavailable", "Folder dialog failed.", status=500)

    monkeypatch.setattr(DirectoryPicker, "_run", select)
    first = await picker_http.post(
        "/api/v1/projects/directory-picker", json={"request_id": "first"}
    )
    assert first.status == 500
    second = await picker_http.post(
        "/api/v1/projects/directory-picker", json={"request_id": "second"}
    )
    assert second.status == 200
    assert (await second.json())["path"] is None


@pytest.mark.asyncio
async def test_disconnected_browser_cancels_selection_and_allows_another_request(
    picker_http: TestClient[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    stopped = asyncio.Event()

    async def select(self: DirectoryPicker) -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(DirectoryPicker, "_run", select)
    request = asyncio.create_task(
        picker_http.post(
            "/api/v1/projects/directory-picker",
            json={"request_id": "disconnect"},
        )
    )
    await asyncio.wait_for(entered.wait(), 3)
    request.cancel()
    await asyncio.gather(request, return_exceptions=True)
    await asyncio.wait_for(stopped.wait(), 3)

    async def cancel(self: DirectoryPicker) -> None:
        return None

    monkeypatch.setattr(DirectoryPicker, "_run", cancel)
    retry = await picker_http.post(
        "/api/v1/projects/directory-picker",
        json={"request_id": "retry"},
    )
    assert retry.status == 200


@pytest.mark.asyncio
async def test_cancellation_during_launch_reaps_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_process = asyncio.create_subprocess_exec
    started = asyncio.Event()
    release = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []

    async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await create_process(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )
        processes.append(process)
        started.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    selection = asyncio.create_task(DirectoryPicker().pick())
    await asyncio.wait_for(started.wait(), 3)
    selection.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await selection
    assert processes[0].returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["cancel", "shutdown"])
async def test_dialog_process_is_reaped_on_request_cancellation_or_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    finish: str,
) -> None:
    create_process = asyncio.create_subprocess_exec
    started = asyncio.Event()
    processes: list[asyncio.subprocess.Process] = []

    async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await create_process(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    picker = DirectoryPicker()
    selection = asyncio.create_task(picker.pick())
    await asyncio.wait_for(started.wait(), 3)
    if finish == "cancel":
        selection.cancel()
    else:
        await picker.close()
    with pytest.raises(asyncio.CancelledError):
        await selection
    assert len(processes) == 1
    assert processes[0].returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [json.dumps({"path": "D:\\中文 空目录"}), "bad json", "{}"])
async def test_helper_output_and_unicode_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    output: str,
) -> None:
    create_process = asyncio.create_subprocess_exec

    async def spawn(*args: Any, **kwargs: Any) -> asyncio.subprocess.Process:
        return await create_process(sys.executable, "-c", f"print({output!r})", **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    picker = DirectoryPicker()
    if output.startswith('{"path"'):
        assert await picker.pick() == "D:\\中文 空目录"
    else:
        with pytest.raises(ServiceError, match="folder selection window"):
            await picker.pick()
