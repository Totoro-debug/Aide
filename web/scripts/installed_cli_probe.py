"""Exercise an installed console entry with Textual's headless terminal adapter."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import shutil
import socket
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from importlib.metadata import entry_points
from pathlib import Path
from time import monotonic
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import aiohttp

import aide
import aide.terminal.cli as cli
from aide.agent.message_bus import InboundMessage
from aide.agent.workspace_state import WorkspaceState
from aide.config.agent_home import AgentHome
from aide.schedule.store import WorkspaceScheduleStore
from aide.service.client import ServiceClient
from aide.service.discovery import read_credential, read_discovery
from aide.service.errors import ServiceError
from aide.terminal.conversation import TerminalConversationApp


class _ProcessExitWitness:
    """Observe the original process identity, independently of discovery and its socket."""

    def __init__(self, pid: int) -> None:
        self.closed = False
        self._check: Callable[[], bool]
        self._release: Callable[[], None]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle.restype = ctypes.c_int
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            raise OSError(ctypes.get_last_error(), "Could not observe the service process")

        def check_handle() -> bool:
            result = int(kernel.WaitForSingleObject(handle, 0))
            if result not in {0, 258}:  # WAIT_OBJECT_0 / WAIT_TIMEOUT
                raise OSError(ctypes.get_last_error(), "Service process wait failed")
            return result == 0

        def release_handle() -> None:
            if not kernel.CloseHandle(handle):
                raise OSError(ctypes.get_last_error(), "Service process handle close failed")

        self._check = check_handle
        self._release = release_handle
        self.method = "windows-process-handle"
        try:
            assert not self.exited(), "Original service process already exited before departure"
        except BaseException:
            self.close()
            raise

    def exited(self) -> bool:
        assert not self.closed
        return self._check()

    def close(self) -> None:
        if not self.closed:
            self._release()
            self.closed = True


async def _wait_for_session_result(path: Path, prompt: str) -> bool:
    deadline = asyncio.get_running_loop().time() + 60
    while asyncio.get_running_loop().time() < deadline:
        try:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            records = []
        if any(
            record.get("role") == "user" and record.get("content") == prompt
            for record in records
            if isinstance(record, dict)
        ) and any(
            record.get("role") == "assistant"
            and isinstance(record.get("content"), str)
            and record["content"].startswith("# Streamed answer")
            and record.get("error") is None
            for record in records
            if isinstance(record, dict)
        ):
            return True
        await asyncio.sleep(0.05)
    return False


async def _wait_for_file(path: Path, timeout: float = 90.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if path.exists():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"installed CLI scenario did not receive its release signal: {path}")


async def _wait_for_observation(path: Path, prompt: str, count: int = 1) -> None:
    deadline = asyncio.get_running_loop().time() + 90
    while asyncio.get_running_loop().time() < deadline:
        try:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            records = []
        observed = sum(
            isinstance(record, dict)
            and isinstance(record.get("prompt"), str)
            and prompt in record["prompt"]
            and bool(record.get("tools"))
            for record in records
        )
        if observed >= count:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"fixture provider did not observe {count} requests for {prompt!r}")


async def _wait_for_prompt_completion(path: Path, prompt: str) -> None:
    deadline = asyncio.get_running_loop().time() + 90
    while asyncio.get_running_loop().time() < deadline:
        try:
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            records = []
        user_index = next(
            (
                index
                for index, record in enumerate(records)
                if isinstance(record, dict)
                and record.get("role") == "user"
                and record.get("content") == prompt
            ),
            None,
        )
        if user_index is not None and any(
            isinstance(record, dict)
            and record.get("role") == "assistant"
            and record.get("status") == "completed"
            and record.get("error") is None
            for record in records[user_index + 1 :]
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"installed CLI prompt did not reach a terminal result: {prompt!r}")


async def _wait_for_cancelled_session(path: Path, prompt: str) -> None:
    deadline = asyncio.get_running_loop().time() + 90
    while asyncio.get_running_loop().time() < deadline:
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        except (FileNotFoundError, json.JSONDecodeError):
            records = []
        if any(
            record.get("role") == "user" and record.get("content") == prompt for record in records
        ) and any(
            record.get("role") == "assistant"
            and isinstance(record.get("error"), dict)
            and record["error"].get("code") == "turn_cancelled"
            for record in records
        ):
            return
        await asyncio.sleep(0.05)
    raise AssertionError("cancelled CLI run did not persist its terminal result")


async def _authenticated_json(
    home: AgentHome,
    url: str,
    *,
    client_id: str | None = None,
) -> dict[str, object]:
    headers = {"Authorization": f"Bearer {read_credential(home)}"}
    if client_id is not None:
        headers["X-Aide-Client"] = client_id
    async with aiohttp.ClientSession() as http:
        async with http.get(url, headers=headers) as response:
            body = await response.json()
            assert response.status == 200, body
            assert isinstance(body, dict), body
            return body


async def _schedule_snapshot(client: ServiceClient) -> dict[str, object]:
    result = await client._http_request(
        "GET",
        f"/api/v1/workspaces/{client.workspace_id}/schedule/jobs",
    )
    assert isinstance(result.get("jobs"), list), result
    assert isinstance(result.get("status"), dict), result
    return result


async def _run_competition_scenario(
    *,
    client: ServiceClient,
    ready_path: Path,
    done_path: Path,
    prompt: str,
    entry_value: str,
) -> None:
    draft = await client.create_session(client.workspace_id)
    session_id = draft.get("session_id")
    assert isinstance(session_id, str), draft
    await client.open_conversation(session_id=session_id)
    discovery = read_discovery(AgentHome.production())
    assert discovery is not None
    session_path = Path.cwd() / ".aide" / "sessions" / f"{session_id}.jsonl"
    ready_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "adapter": "installed console entry with Textual run_test; not a TTY",
                "console_entry": entry_value,
                "service_instance_id": discovery.service_instance_id,
                "service_pid": discovery.pid,
                "workspace_id": client.workspace_id,
                "session_id": session_id,
            }
        ),
        encoding="utf-8",
    )
    try:
        await client.bus.put_inbound(InboundMessage(prompt))
        assert await _wait_for_session_result(session_path, prompt)
        done_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "marker": "INSTALLED_CLI_COMPETITION_OK",
                    "adapter": "installed console entry with Textual run_test; not a TTY",
                    "console_entry": entry_value,
                    "service_instance_id": discovery.service_instance_id,
                    "service_pid": discovery.pid,
                    "workspace_id": client.workspace_id,
                    "session_id": session_id,
                    "input_accepted": True,
                    "assistant_persisted": True,
                    "jsonl_path": str(session_path),
                }
            ),
            encoding="utf-8",
        )
    except BaseException as error:
        done_path.write_text(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
            encoding="utf-8",
        )
        raise


async def _run_last_client_grace_scenario(
    *,
    client: ServiceClient,
    ready_path: Path,
    done_path: Path,
    release_path: Path,
    entry_value: str,
) -> None:
    home = AgentHome.production()
    discovery = read_discovery(home)
    assert discovery is not None
    reconnect_credential = client.reconnect_credential
    original_client_id = client.client_id
    original_instance_id = discovery.service_instance_id
    original_pid = discovery.pid
    project = await client._http_request(
        "POST",
        "/api/v1/projects",
        payload={"request_id": "installed-last-client-grace-project", "path": str(Path.cwd())},
        mutation=True,
    )
    assert project.get("workspace_id") == client.workspace_id, project
    ready_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "adapter": "installed console entry with Textual run_test; not a TTY",
                "console_entry": entry_value,
                "service_instance_id": original_instance_id,
                "service_pid": original_pid,
                "client_id": original_client_id,
                "workspace_id": client.workspace_id,
                "session_id": client.session_id,
            }
        ),
        encoding="utf-8",
    )
    try:
        await _wait_for_file(release_path)
        due_time = datetime.now(UTC) + timedelta(seconds=10)
        job = await client._http_request(
            "POST",
            f"/api/v1/workspaces/{client.workspace_id}/schedule/jobs",
            payload={
                "request_id": "installed-last-client-grace-job",
                "message": "installed last-client grace must not run this Job",
                "title": "installed last-client grace Job",
                "at_time": due_time.isoformat(),
            },
            mutation=True,
        )
        job_value = job.get("job")
        assert isinstance(job_value, dict), job
        job_id = job_value.get("job_id")
        assert isinstance(job_id, str), job
        disconnected_at = monotonic()
        await client.close()
        await asyncio.sleep(29.0)
        elapsed = monotonic() - disconnected_at
        during_grace = read_discovery(home)
        assert during_grace is not None
        assert during_grace.service_instance_id == original_instance_id
        assert during_grace.pid == original_pid
        with socket.create_connection((during_grace.host, during_grace.port), timeout=2):
            pass
        service_info = await _authenticated_json(
            home,
            f"http://{during_grace.host}:{during_grace.port}/api/v1/service",
        )
        assert service_info["state"] == "reconnecting", service_info
        schedule_during_grace = await _authenticated_json(
            home,
            f"http://{during_grace.host}:{during_grace.port}/api/v1/workspaces/"
            f"{client.workspace_id}/schedule/jobs",
            client_id=original_client_id,
        )
        status_during_grace = schedule_during_grace.get("status")
        assert isinstance(status_during_grace, dict), schedule_during_grace
        assert status_during_grace["admitted"] is False, schedule_during_grace
        jobs_during_grace = schedule_during_grace.get("jobs")
        assert isinstance(jobs_during_grace, list), schedule_during_grace
        observed_job = next(
            item
            for item in jobs_during_grace
            if isinstance(item, dict) and item.get("job_id") == job_id
        )
        assert observed_job["state"]["last_status"] is None, observed_job
        assert datetime.now(UTC) > due_time
        assert not (
            Path.cwd() / ".aide" / "schedule-sessions" / f"schedule_{job_id}.jsonl"
        ).exists()

        reconnected = await ServiceClient.connect_or_start(
            home,
            Path.cwd(),
            reconnect_credential=reconnect_credential,
        )
        process_witness: _ProcessExitWitness | None = None
        try:
            reconnected_discovery = read_discovery(home)
            assert reconnected_discovery is not None
            assert reconnected_discovery.service_instance_id == original_instance_id
            assert reconnected_discovery.pid == original_pid
            assert reconnected.client_id == original_client_id
            schedule_after_reconnect = await _schedule_snapshot(reconnected)
            status_after_reconnect = schedule_after_reconnect["status"]
            assert isinstance(status_after_reconnect, dict), schedule_after_reconnect
            assert status_after_reconnect["admitted"] is True, schedule_after_reconnect
            expiry_prompt = "installed expiry barrier"
            expiry_session_path = (
                Path.cwd() / ".aide" / "sessions" / f"{reconnected.session_id}.jsonl"
            )
            await reconnected.bus.put_inbound(InboundMessage(expiry_prompt))
            await _wait_for_observation(
                Path(os.environ["AIDE_PROVIDER_OBSERVATION_PATH"]), expiry_prompt
            )
            expiry_job_prompt = f"installed expiry grace Job must not run {uuid4()}"
            expiry_job_due = datetime.now(UTC) + timedelta(seconds=10)
            expiry_job_response = await reconnected._http_request(
                "POST",
                f"/api/v1/workspaces/{reconnected.workspace_id}/schedule/jobs",
                payload={
                    "request_id": str(uuid4()),
                    "message": expiry_job_prompt,
                    "title": "installed expiry grace Job",
                    "at_time": expiry_job_due.isoformat(),
                },
                mutation=True,
            )
            expiry_job = expiry_job_response.get("job")
            assert isinstance(expiry_job, dict), expiry_job_response
            expiry_job_id = expiry_job.get("job_id")
            assert isinstance(expiry_job_id, str), expiry_job_response
            process_witness = _ProcessExitWitness(original_pid)
            witness_captured_at = monotonic()
            expiry_due_margin = (expiry_job_due - datetime.now(UTC)).total_seconds()
            assert expiry_due_margin > 0, "Expiry Job became due before the last departure"
            expiry_disconnect = monotonic()
            await reconnected.close()
            deadline = asyncio.get_running_loop().time() + 60
            while asyncio.get_running_loop().time() < deadline:
                if process_witness.exited() and read_discovery(home) is None:
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                            probe.bind((discovery.host, discovery.port))
                    except OSError:
                        pass
                    else:
                        break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError(
                    "last client departure did not autonomously exit the original service process"
                )
            shutdown_elapsed = monotonic() - expiry_disconnect
            assert shutdown_elapsed >= 30.0, shutdown_elapsed
            assert process_witness.exited()
            process_witness.close()
            assert datetime.now(UTC) > expiry_job_due
            await _wait_for_cancelled_session(expiry_session_path, expiry_prompt)
            workspace_state = WorkspaceState(Path.cwd())
            persisted_jobs = await WorkspaceScheduleStore(workspace_state).snapshot()
            persisted_job = next(job for job in persisted_jobs if job.job_id == expiry_job_id)
            assert persisted_job.message == expiry_job_prompt
            assert persisted_job.state.last_status is None
            assert persisted_job.state.last_finished_at_ms is None
            assert persisted_job.state.last_error is None
            expiry_schedule_path = (
                workspace_state.schedule_sessions_directory / f"{persisted_job.session_id}.jsonl"
            )
            assert not expiry_schedule_path.exists()
            provider_records = [
                json.loads(line)
                for line in Path(os.environ["AIDE_PROVIDER_OBSERVATION_PATH"])
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            assert not any(
                expiry_job_prompt in str(record.get("prompt", "")) for record in provider_records
            )
            done_path.write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "marker": "INSTALLED_CLI_LAST_CLIENT_GRACE_OK",
                        "adapter": "installed console entry with Textual run_test; not a TTY",
                        "console_entry": entry_value,
                        "service_instance_id": original_instance_id,
                        "service_pid": original_pid,
                        "client_id": original_client_id,
                        "same_client_reconnected": True,
                        "reconnect_elapsed_seconds": elapsed,
                        "grace_seconds_observed": 29,
                        "state_during_grace": service_info["state"],
                        "discovery_preserved_during_grace": True,
                        "port_open_during_grace": True,
                        "schedule_admitted_during_grace": False,
                        "schedule_admitted_after_reconnect": True,
                        "no_new_job_during_grace": True,
                        "job_id": job_id,
                        "project_registered": True,
                        "job_due_during_grace": due_time.isoformat(),
                        "schedule_session_absent_during_grace": True,
                        "autonomous_shutdown": True,
                        "shutdown_elapsed_seconds": shutdown_elapsed,
                        "discovery_removed": True,
                        "port_released": True,
                        "foreground_cancelled_persisted": True,
                        "explicit_stop_used": False,
                        "original_process_exit_observed": True,
                        "process_witness_method": process_witness.method,
                        "process_witness_captured_before_departure": (
                            witness_captured_at < expiry_disconnect
                        ),
                        "process_witness_closed": process_witness.closed,
                        "reconnect_after_expiry_departure": False,
                        "expiry_job_id": expiry_job_id,
                        "expiry_job_prompt": expiry_job_prompt,
                        "expiry_job_due_at": expiry_job_due.isoformat(),
                        "expiry_job_due_margin_before_departure_seconds": expiry_due_margin,
                        "expiry_job_due_before_shutdown": True,
                        "expiry_job_schedule_session_absent": True,
                        "expiry_job_provider_request_absent": True,
                        "expiry_job_persisted_unexecuted": True,
                        "expiry_job_persisted_state": persisted_job.state.to_dict(),
                    }
                ),
                encoding="utf-8",
            )
        finally:
            try:
                await reconnected.close()
            finally:
                if process_witness is not None:
                    process_witness.close()
    except BaseException as error:
        done_path.write_text(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
            encoding="utf-8",
        )
        raise


async def _run_joint_scenario(
    *,
    client: ServiceClient,
    ready_path: Path,
    foreground_ready_path: Path,
    removal_done_path: Path,
    settings_ready_path: Path,
    settings_done_path: Path,
    settings_start_path: Path,
    observation_path: Path,
    entry_value: str,
    notices: list[str],
) -> None:
    discovery = read_discovery(AgentHome.production())
    assert discovery is not None
    draft = await client.create_session(client.workspace_id)
    session_id = draft.get("session_id")
    assert isinstance(session_id, str), draft
    await client.open_conversation(session_id=session_id)
    initial_workspace_id = client.workspace_id
    initial_session_id = client.session_id
    removal_prompt = "project removal barrier"
    settings_prompt = "settings generation barrier"
    confirmation_path = Path(os.environ["AIDE_JOINT_BROWSER_READY"] + ".confirmation")

    async def compete_for_confirmation() -> None:
        await _wait_for_file(confirmation_path)
        request = json.loads(confirmation_path.read_text(encoding="utf-8"))
        try:
            await client.decide_confirmation(request["token"], "approved")
            result = {"accepted": True}
        except ServiceError as error:
            result = error.to_dict(request["request_id"])
        confirmation_path.with_suffix(".result").write_text(
            json.dumps(result), encoding="utf-8"
        )

    confirmation_competitor = asyncio.create_task(compete_for_confirmation())
    ready_path.write_text(
        json.dumps(
            {
                "status": "ready",
                "adapter": "installed console entry with Textual run_test; not a TTY",
                "console_entry": entry_value,
                "service_instance_id": discovery.service_instance_id,
                "service_pid": discovery.pid,
                "workspace_id": initial_workspace_id,
                "session_id": initial_session_id,
            }
        ),
        encoding="utf-8",
    )
    try:
        await client.bus.put_inbound(InboundMessage(removal_prompt))
        await _wait_for_observation(observation_path, removal_prompt)
        foreground_ready_path.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "prompt": removal_prompt,
                    "service_instance_id": discovery.service_instance_id,
                    "service_pid": discovery.pid,
                    "workspace_id": initial_workspace_id,
                    "session_id": initial_session_id,
                    "claim_version": client.claim_version,
                }
            ),
            encoding="utf-8",
        )
        deadline = asyncio.get_running_loop().time() + 90
        while asyncio.get_running_loop().time() < deadline:
            if (
                client.workspace_id == ""
                and client.session_id == ""
                and client.claim_version == 0
                and not client.control.foreground_input_admitted()
            ):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("project removal did not detach the installed CLI Claim")
        notice = "Project registration was removed; its work has stopped."
        assert notice in notices, notices
        await _wait_for_cancelled_session(
            Path.cwd() / ".aide" / "sessions" / f"{initial_session_id}.jsonl", removal_prompt
        )
        removal_done_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "terminal_state": "cancelled",
                    "notification_received": True,
                    "foreground_cancelled_persisted": True,
                    "claim_released": True,
                    "workspace_id_after_removal": client.workspace_id,
                    "session_id_after_removal": client.session_id,
                    "claim_version_after_removal": client.claim_version,
                    "service_instance_id": discovery.service_instance_id,
                    "service_pid": discovery.pid,
                }
            ),
            encoding="utf-8",
        )
        await _wait_for_file(settings_start_path)
        reentered = await client._http_request(
            "POST",
            "/api/v1/projects",
            payload={
                "request_id": "installed-joint-settings-reentry",
                "path": str(Path.cwd()),
            },
            mutation=True,
        )
        new_workspace_id = reentered.get("workspace_id")
        assert isinstance(new_workspace_id, str), reentered
        client.workspace_id = new_workspace_id
        new_draft = await client.create_session(new_workspace_id)
        new_session_id = new_draft.get("session_id")
        assert isinstance(new_session_id, str), new_draft
        await client.open_conversation(session_id=new_session_id)
        settings_session_path = Path.cwd() / ".aide" / "sessions" / f"{new_session_id}.jsonl"
        settings_ready_path.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "service_instance_id": discovery.service_instance_id,
                    "service_pid": discovery.pid,
                    "workspace_id": new_workspace_id,
                    "session_id": new_session_id,
                }
            ),
            encoding="utf-8",
        )
        await client.bus.put_inbound(InboundMessage(settings_prompt))
        await _wait_for_observation(observation_path, settings_prompt)
        await _wait_for_file(Path(os.environ["AIDE_CLI_SETTINGS_RELEASE"]))
        await _wait_for_prompt_completion(settings_session_path, settings_prompt)
        config = await client._http_request("GET", "/api/v1/config")
        application = config.get("application")
        assert isinstance(application, dict)
        assert application["status"] == "restart-required"
        assert application["active_revision"] != application["saved_revision"]
        new_prompt = "installed new generation response"
        await client.bus.put_inbound(InboundMessage(new_prompt))
        await _wait_for_prompt_completion(settings_session_path, new_prompt)
        records = [
            json.loads(line)
            for line in settings_session_path.read_text(encoding="utf-8").splitlines()
        ]
        new_input = next(
            index
            for index, record in enumerate(records)
            if record.get("role") == "user" and record.get("content") == new_prompt
        )
        result = next(
            record for record in records[new_input + 1 :] if record.get("role") == "assistant"
        )
        assert result["context_usage"]["model"] == "small-model", result
        await _wait_for_observation(observation_path, new_prompt)
        observations = [
            json.loads(line) for line in observation_path.read_text(encoding="utf-8").splitlines()
        ]
        assert any(
            new_prompt in str(observation.get("prompt", ""))
            and observation.get("model") == "small-model"
            and observation.get("tools")
            for observation in observations
        ), "saved settings changed the startup model before restart"
        settings_done_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "service_instance_id": discovery.service_instance_id,
                    "service_pid": discovery.pid,
                    "workspace_id": new_workspace_id,
                    "session_id": new_session_id,
                    "same_service_instance": read_discovery(AgentHome.production()) == discovery,
                    "foreground_terminal": True,
                    "new_generation_model": result["context_usage"]["model"],
                }
            ),
            encoding="utf-8",
        )
        done_path = Path(os.environ["AIDE_CLI_DONE"])
        done_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "marker": "INSTALLED_CLI_JOINT_OK",
                    "console_entry": entry_value,
                    "service_instance_id": discovery.service_instance_id,
                    "service_pid": discovery.pid,
                    "initial_workspace_id": initial_workspace_id,
                    "initial_session_id": initial_session_id,
                    "settings_workspace_id": new_workspace_id,
                    "settings_session_id": new_session_id,
                    "project_removal_terminal": True,
                    "claim_released": True,
                    "settings_generation_completed": True,
                }
            ),
            encoding="utf-8",
        )
    except BaseException as error:
        Path(os.environ["AIDE_CLI_DONE"]).write_text(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
            encoding="utf-8",
        )
        raise

    finally:
        confirmation_competitor.cancel()
        await asyncio.gather(confirmation_competitor, return_exceptions=True)


async def headless_terminal(self: Any, **_kwargs: object) -> None:
    async with self.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert self.query_one("#conversation-input")
        scenario = os.environ.get("AIDE_CLI_SCENARIO")
        current_discovery = read_discovery(AgentHome.production())
        if scenario not in {"competition", "last-client-grace"}:
            assert current_discovery == before
        else:
            assert current_discovery is not None
        bus = self._bus
        client = getattr(bus, "client", None)
        assert client is not None
        if scenario == "competition":
            await _run_competition_scenario(
                client=client,
                ready_path=Path(os.environ["AIDE_CLI_READY"]),
                done_path=Path(os.environ["AIDE_CLI_DONE"]),
                prompt=os.environ["AIDE_CLI_PROMPT"],
                entry_value=console_entry.value,
            )
        elif scenario == "last-client-grace":
            await _run_last_client_grace_scenario(
                client=client,
                ready_path=Path(os.environ["AIDE_CLI_READY"]),
                done_path=Path(os.environ["AIDE_CLI_DONE"]),
                release_path=Path(os.environ["AIDE_CLI_RELEASE"]),
                entry_value=console_entry.value,
            )
        elif scenario == "joint":
            notices: list[str] = []
            original_output = bus.put_remote_output

            async def observe_output(message: dict[str, object]) -> None:
                if message.get("type") == "system_control" and isinstance(
                    message.get("content"), str
                ):
                    notices.append(str(message["content"]))
                await original_output(message)

            with patch.object(bus, "put_remote_output", observe_output):
                await _run_joint_scenario(
                    client=client,
                    ready_path=Path(os.environ["AIDE_CLI_READY"]),
                    foreground_ready_path=Path(os.environ["AIDE_CLI_FOREGROUND_READY"]),
                    removal_done_path=Path(os.environ["AIDE_CLI_REMOVAL_DONE"]),
                    settings_ready_path=Path(os.environ["AIDE_CLI_SETTINGS_READY"]),
                    settings_done_path=Path(os.environ["AIDE_CLI_SETTINGS_DONE"]),
                    settings_start_path=Path(os.environ["AIDE_CLI_SETTINGS_START"]),
                    observation_path=Path(os.environ["AIDE_PROVIDER_OBSERVATION_PATH"]),
                    entry_value=console_entry.value,
                    notices=notices,
                )
        elif scenario == "cross-client":
            ready_path = Path(os.environ["AIDE_CLI_READY"])
            done_path = Path(os.environ["AIDE_CLI_DONE"])
            prompt = os.environ["AIDE_CLI_PROMPT"]
            contested_session_id = os.environ["AIDE_CLI_CONTESTED_SESSION"]
            try:
                try:
                    await client.open_conversation(session_id=contested_session_id)
                except ServiceError as error:
                    claim_error_code = error.code
                    claim_error_text = str(error)
                else:
                    raise AssertionError("The installed CLI claimed the browser Session")
                assert claim_error_code == "session_claimed"
                assert os.environ["AIDE_CLI_PRIVATE_MARKER"] not in claim_error_text
                try:
                    await client._http_request(
                        "GET",
                        f"/api/v1/workspaces/{client.workspace_id}/sessions/{contested_session_id}"
                        f"?claim_version={client.claim_version}",
                        extra_headers={"X-Aide-Claim": client.claim_credential},
                    )
                except ServiceError as error:
                    assert error.code in {"stale_claim", "session_claimed"}, error.code
                    assert os.environ["AIDE_CLI_PRIVATE_MARKER"] not in str(error)
                else:
                    raise AssertionError("Claim loser could read the browser Session body")
                await bus.put_inbound(InboundMessage(prompt))
                await _wait_for_observation(
                    Path(os.environ["AIDE_PROVIDER_OBSERVATION_PATH"]), prompt
                )
                session_path = Path.cwd() / ".aide" / "sessions" / f"{client.session_id}.jsonl"
                ready_path.write_text(
                    json.dumps(
                        {
                            "status": "ready",
                            "adapter": "installed console entry with Textual run_test; not a TTY",
                            "workspace_id": client.workspace_id,
                            "session_id": client.session_id,
                            "contested_session_id": contested_session_id,
                            "claim_error_code": claim_error_code,
                            "claim_error_contains_private_marker": False,
                            "body_read_denied": True,
                        }
                    ),
                    encoding="utf-8",
                )
                persisted = await _wait_for_session_result(session_path, prompt)
                assert persisted
                done_path.write_text(
                    json.dumps(
                        {
                            "status": "passed",
                            "marker": "INSTALLED_CLI_CROSS_CLIENT_OK",
                            "adapter": "installed console entry with Textual run_test; not a TTY",
                            "workspace_id": client.workspace_id,
                            "session_id": client.session_id,
                            "contested_session_id": contested_session_id,
                            "distinct_sessions": client.session_id != contested_session_id,
                            "claim_denied_code": claim_error_code,
                            "claim_error_contains_private_marker": False,
                            "body_read_denied": True,
                            "input_accepted": True,
                            "assistant_persisted": True,
                            "jsonl_path": str(session_path),
                        }
                    ),
                    encoding="utf-8",
                )
            except BaseException as error:
                done_path.write_text(
                    json.dumps({"status": "failed", "error": f"{type(error).__name__}: {error}"}),
                    encoding="utf-8",
                )
                raise
        self.exit()


assert Path(aide.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert shutil.which("node") is None and shutil.which("npm") is None
scenario = os.environ.get("AIDE_CLI_SCENARIO")
before = read_discovery(AgentHome.production())
if scenario not in {"competition", "last-client-grace", "joint"}:
    assert before is not None
    assert before.service_instance_id == sys.argv[1]
    assert before.pid == int(sys.argv[2])
# Replace only terminal I/O; retain the installed entry, CLI composition and client.
console_entry = next(iter(entry_points(group="console_scripts", name="aide")))
entry = console_entry
sys.argv = ["aide"]
with (
    patch.object(cli, "is_interactive_terminal", return_value=True),
    patch.object(TerminalConversationApp, "run_async", headless_terminal),
):
    try:
        entry.load()()
    except SystemExit as error:
        assert error.code in (None, 0), error.code
if scenario not in {"competition", "last-client-grace"}:
    assert before is not None
    assert read_discovery(AgentHome.production()) == before
    print(json.dumps({"marker": "INSTALLED_CLI_CONNECT_OK", **before.to_dict()}))
