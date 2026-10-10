"""Validate normal installed Web/CLI lifecycles, separate from the Node controller.

Run with python -m scripts.installed_web_validation --output <new external directory>.
Requires the existing web/ Playwright development dependencies on the controller.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import sys
from hashlib import sha256
from pathlib import Path

import aiohttp

import aide.client.web.frontend.scripts.e2e_service as fixture_service
from aide.client.web.frontend.scripts.e2e_service import _start_fixture_provider
from aide.config.agent_home import AgentHome
from aide.service.discovery import read_credential, read_discovery
from aide.utils.platform import WINDOWS_REQUIRED_ERROR, is_windows_host
from scripts.release_validation import (
    ROOT,
    _artifact_environment,
    _assert_sdist_web_assets,
    _assert_wheel_web_assets,
    _exception_payload,
    _extract_sdist,
    _redact_report_text,
    _run_command,
    _smoke_installed_wheel,
    _source_identity,
    _source_web_asset_bytes,
    build_acceptance_matrix,
)


def build_distributions(output: Path) -> tuple[Path, Path]:
    expected = _source_web_asset_bytes()
    direct = output / "distribution"
    direct.mkdir()
    environment = _artifact_environment()
    _run_command(
        [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(direct)],
        env=environment,
    )
    (wheel,) = direct.glob("*.whl")
    (sdist,) = direct.glob("*.tar.gz")
    _assert_wheel_web_assets(wheel, expected)
    _assert_sdist_web_assets(sdist, expected)
    source = _extract_sdist(sdist, output / "extracted")
    rebuilt = output / "rebuilt"
    rebuilt.mkdir()
    _run_command(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(rebuilt)],
        cwd=source,
        env=environment,
    )
    (rebuilt_wheel,) = rebuilt.glob("*.whl")
    _assert_wheel_web_assets(rebuilt_wheel, expected)
    return wheel, rebuilt_wheel


async def _wait_for_json_file(
    path: Path,
    timeout: float = 90.0,
    *,
    process: asyncio.subprocess.Process | None = None,
    failure_path: Path | None = None,
    secrets: tuple[str, ...] = (),
) -> dict[str, object]:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if failure_path is not None and failure_path.exists():
            raise RuntimeError(
                _redact_report_text(failure_path.read_text(encoding="utf-8"), secrets)
            )
        if process is not None and process.returncode is not None:
            stdout, stderr = await process.communicate()
            detail = _redact_report_text(
                (stdout + stderr).decode("utf-8", errors="replace"), secrets
            )
            raise RuntimeError(
                f"installed browser exited before readiness ({process.returncode}): {detail}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            await asyncio.sleep(0.05)
            continue
        if isinstance(value, dict):
            return value
        raise RuntimeError(f"installed validation evidence is not an object: {path}")
    raise RuntimeError(f"installed validation evidence did not appear: {path}")


async def _wait_for_path(path: Path, timeout: float = 90.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if path.exists():
            return
        await asyncio.sleep(0.05)
    raise RuntimeError(f"installed validation signal did not appear: {path}")


def _prepare_cli_scenario(
    root: Path,
    config: str,
    environment: dict[str, str],
) -> tuple[AgentHome, Path, dict[str, str]]:
    profile = root / "用户配置"
    home_path = profile / ".aide"
    workspace = root / "workspace"
    home_path.mkdir(parents=True)
    workspace.mkdir()
    (home_path / "config.toml").write_text(config, encoding="utf-8")
    return (
        AgentHome(home_path),
        workspace,
        {
            **environment,
            "USERPROFILE": str(profile),
            "HOME": str(profile),
        },
    )


async def _service_json(
    home: AgentHome,
    path: str,
    *,
    client_id: str | None = None,
) -> dict[str, object]:
    discovery = read_discovery(home)
    if discovery is None:
        raise RuntimeError(f"installed service discovery is missing: {home.path}")
    headers = {"Authorization": f"Bearer {read_credential(home)}"}
    if client_id is not None:
        headers["X-Aide-Client"] = client_id
    async with aiohttp.ClientSession() as http:
        async with http.get(
            f"http://{discovery.host}:{discovery.port}{path}", headers=headers
        ) as response:
            body = await response.json()
            if response.status != 200 or not isinstance(body, dict):
                raise RuntimeError(f"installed service request failed: {response.status}: {body}")
            return body


async def _run_installed_cli_competition(
    root: Path,
    *,
    entry: str,
    python: Path,
    config: str,
    environment: dict[str, str],
) -> dict[str, object]:
    scenario_root = root / "two-cli-competition"
    home, workspace, scenario_environment = _prepare_cli_scenario(
        scenario_root, config, environment
    )
    discovery_path = home.path / "service.json"
    ready_paths = [scenario_root / f"cli-{index}-ready.json" for index in (1, 2)]
    done_paths = [scenario_root / f"cli-{index}-done.json" for index in (1, 2)]
    prompt = "installed CLI competition streaming markdown"
    listener_bytes: list[bytes] = []

    async def record_listener(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            listener_bytes.append(await reader.read(4096))
        finally:
            writer.close()
            await writer.wait_closed()

    listener = await asyncio.start_server(record_listener, "127.0.0.1", 8765)
    processes: list[asyncio.subprocess.Process] = []
    try:
        collision = await asyncio.create_subprocess_exec(
            entry,
            "web",
            cwd=workspace,
            env=scenario_environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(collision)
        collision_out, collision_error = await asyncio.wait_for(collision.communicate(), timeout=60)
        if collision.returncode == 0 or discovery_path.exists():
            raise RuntimeError("installed CLI accepted an unrelated fixed-port listener")
        collision_message = (collision_out + collision_error).decode("utf-8", errors="replace")
        if (
            "Traceback" in collision_message
            or "code must be a stable error code" in collision_message
        ):
            raise RuntimeError(
                "installed port collision exposed a traceback instead of a CLI error"
            )
        if (
            "port" not in collision_message.casefold()
            and "start" not in collision_message.casefold()
        ):
            raise RuntimeError("installed fixed-port collision did not report a startup error")
        if not listener.is_serving():
            raise RuntimeError("installed CLI stopped the unrelated fixed-port listener")
        try:
            credential = read_credential(home)
        except FileNotFoundError:
            credential = ""
        if credential and credential.encode("ascii") in b"".join(listener_bytes):
            raise RuntimeError("unrelated listener received the service credential")
        listener.close()
        await listener.wait_closed()
        absent_stop = await asyncio.create_subprocess_exec(
            entry,
            "service",
            "stop",
            cwd=workspace,
            env=scenario_environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        processes.append(absent_stop)
        absent_out, absent_error = await asyncio.wait_for(absent_stop.communicate(), timeout=60)
        absent_message = (absent_out + absent_error).decode("utf-8", errors="replace")
        if (
            absent_stop.returncode != 1
            or "service_not_running: No active local service was found." not in absent_message
            or "Traceback" in absent_message
        ):
            raise RuntimeError(
                "installed service stop did not report the absence of a service clearly"
            )
        commands: list[dict[str, object]] = []
        for ready_path, done_path in zip(ready_paths, done_paths, strict=True):
            commands.append(
                {
                    "command": [
                        str(python),
                        "-I",
                        str(ROOT / "aide/client/web/frontend/scripts/installed_cli_probe.py"),
                    ],
                    "environment": {
                        **scenario_environment,
                        "AIDE_CLI_SCENARIO": "competition",
                        "AIDE_CLI_READY": str(ready_path),
                        "AIDE_CLI_DONE": str(done_path),
                        "AIDE_CLI_PROMPT": prompt,
                    },
                }
            )

        async def spawn(spec: dict[str, object]) -> asyncio.subprocess.Process:
            command = spec["command"]
            process_environment = spec["environment"]
            assert isinstance(command, list)
            assert isinstance(process_environment, dict)
            process = await asyncio.create_subprocess_exec(
                *(str(part) for part in command),
                cwd=workspace,
                env={str(key): str(value) for key, value in process_environment.items()},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            processes.append(process)
            return process

        spawned = await asyncio.gather(*(spawn(spec) for spec in commands), return_exceptions=True)
        failures = [result for result in spawned if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("installed CLI competition spawn failed", failures)
        cli_processes = [
            result for result in spawned if isinstance(result, asyncio.subprocess.Process)
        ]
        ready = await asyncio.gather(*(_wait_for_json_file(path) for path in ready_paths))
        output = await asyncio.wait_for(
            asyncio.gather(*(process.communicate() for process in cli_processes)),
            timeout=180,
        )
        for process, (stdout, stderr) in zip(cli_processes, output, strict=True):
            if process.returncode != 0:
                text = (stdout + stderr).decode("utf-8", errors="replace")
                raise RuntimeError(f"installed CLI competition process failed: {text}")
        done = [json.loads(path.read_text(encoding="utf-8")) for path in done_paths]
        if any(not isinstance(value, dict) or value.get("status") != "passed" for value in done):
            raise RuntimeError(f"installed CLI competition returned failed evidence: {done}")
        service_instance_ids = {value.get("service_instance_id") for value in ready}
        service_pids = {value.get("service_pid") for value in ready}
        session_ids = {value.get("session_id") for value in ready}
        workspace_ids = {value.get("workspace_id") for value in ready}
        if len(service_instance_ids) != 1 or len(service_pids) != 1:
            raise RuntimeError(f"CLI competition did not share one service: {ready}")
        if len(session_ids) != 2 or len(workspace_ids) != 1:
            raise RuntimeError(f"CLI competition did not create two Sessions: {ready}")
        return {
            "status": "passed",
            "commands": [spec["command"] for spec in commands],
            "ready": ready,
            "done": done,
            "same_service_instance": True,
            "same_service_pid": True,
            "distinct_sessions": True,
            "same_workspace": True,
            "unrelated_listener_alive": True,
            "service_credential_not_observed": True,
            "collision_without_traceback": True,
            "absent_service_stop": {"expected_exit_code": 1, "exit_code": absent_stop.returncode},
            "fixed_port_collision_rejected": True,
            "collision_command": [entry, "web"],
            "adapter": "installed console metadata entry with headless Textual I/O; real executable TTY not validated",
            "service_instance_id": next(iter(service_instance_ids)),
            "service_pid": next(iter(service_pids)),
        }
    finally:
        primary = sys.exception()
        cleanup_errors: list[BaseException] = []
        for process in processes:
            if process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except BaseException as error:
                    cleanup_errors.append(error)
        try:
            listener.close()
            await listener.wait_closed()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            if discovery_path.exists():
                await stop_install(entry, workspace, scenario_environment, discovery_path)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "installed CLI competition cleanup failed",
                ([primary] if primary is not None else []) + cleanup_errors,
            )


async def _run_installed_last_client_exit(
    root: Path,
    *,
    entry: str,
    python: Path,
    config: str,
    environment: dict[str, str],
) -> dict[str, object]:
    scenario_root = root / "last-client-exit"
    home, workspace, scenario_environment = _prepare_cli_scenario(
        scenario_root, config, environment
    )
    discovery_path = home.path / "service.json"
    keep_ready = scenario_root / "keepalive-ready.json"
    observation_path = scenario_root / "provider-observations.jsonl"
    observation_path.write_text("", encoding="utf-8")
    fixture_service.PROVIDER_OBSERVATION_PATH = observation_path
    fixture_service.INSTALLED_EXPIRY_RELEASE.clear()
    keep_done = scenario_root / "keepalive-done.json"
    keep_release = scenario_root / "keepalive-release"
    companion_ready = scenario_root / "companion-ready.json"
    companion_done = scenario_root / "companion-done.json"
    keep_environment = {
        **scenario_environment,
        "AIDE_CLI_SCENARIO": "last-client-exit",
        "AIDE_CLI_READY": str(keep_ready),
        "AIDE_CLI_DONE": str(keep_done),
        "AIDE_CLI_RELEASE": str(keep_release),
        "AIDE_PROVIDER_OBSERVATION_PATH": str(observation_path),
    }
    companion_environment = {
        **scenario_environment,
        "AIDE_CLI_SCENARIO": "competition",
        "AIDE_CLI_READY": str(companion_ready),
        "AIDE_CLI_DONE": str(companion_done),
        "AIDE_CLI_PROMPT": "installed CLI companion streaming markdown",
    }
    keep_command = [str(python), "-I", str(ROOT / "aide/client/web/frontend/scripts/installed_cli_probe.py")]
    companion_command = [str(python), "-I", str(ROOT / "aide/client/web/frontend/scripts/installed_cli_probe.py")]
    keep_process = await asyncio.create_subprocess_exec(
        *keep_command,
        cwd=workspace,
        env=keep_environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    companion_process: asyncio.subprocess.Process | None = None
    try:
        keep_ready_evidence = await _wait_for_json_file(keep_ready)
        companion_process = await asyncio.create_subprocess_exec(
            *companion_command,
            cwd=workspace,
            env=companion_environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        companion_output, companion_error = await asyncio.wait_for(
            companion_process.communicate(), timeout=180
        )
        if companion_process.returncode != 0:
            text = (companion_output + companion_error).decode("utf-8", errors="replace")
            raise RuntimeError(f"installed exit companion failed: {text}")
        companion_evidence = json.loads(companion_done.read_text(encoding="utf-8"))
        if not isinstance(companion_evidence, dict) or companion_evidence.get("status") != "passed":
            raise RuntimeError(
                f"installed exit companion returned failed evidence: {companion_evidence}"
            )
        discovery_after_companion = read_discovery(home)
        if discovery_after_companion is None:
            raise RuntimeError("last-client exit service stopped while another client was online")
        service_before_release = await _service_json(home, "/api/v1/service")
        if service_before_release.get("state") != "ready":
            raise RuntimeError(
                f"last-client exit service was not ready with keepalive client: {service_before_release}"
            )
        await asyncio.sleep(2.0)
        service_after_wait = await _service_json(home, "/api/v1/service")
        if service_after_wait.get("state") != "ready":
            raise RuntimeError(
                f"last-client exit service stopped despite keepalive client: {service_after_wait}"
            )
        keep_release.write_text("release\n", encoding="ascii")
        keep_output, keep_error = await asyncio.wait_for(keep_process.communicate(), timeout=150)
        if keep_process.returncode != 0:
            text = (keep_output + keep_error).decode("utf-8", errors="replace")
            raise RuntimeError(f"installed last-client exit probe failed: {text}")
        keep_done_evidence = json.loads(keep_done.read_text(encoding="utf-8"))
        if not isinstance(keep_done_evidence, dict) or keep_done_evidence.get("status") != "passed":
            raise RuntimeError(
                f"installed last-client exit returned failed evidence: {keep_done_evidence}"
            )
        return {
            "status": "passed",
            "keepalive_command": keep_command,
            "companion_command": companion_command,
            "keepalive_ready": keep_ready_evidence,
            "companion": companion_evidence,
            "keepalive": keep_done_evidence,
            "same_service_instance": (
                keep_ready_evidence.get("service_instance_id")
                == companion_evidence.get("service_instance_id")
                == keep_done_evidence.get("service_instance_id")
            ),
            "same_service_pid": (
                keep_ready_evidence.get("service_pid")
                == companion_evidence.get("service_pid")
                == keep_done_evidence.get("service_pid")
            ),
            "companion_online_prevented_stop": True,
            "service_ready_after_companion_exit": True,
            "service_ready_after_two_second_wait": True,
            "autonomous_shutdown": keep_done_evidence.get("autonomous_shutdown") is True,
            "explicit_stop_after_scenario": False,
        }
    finally:
        primary = sys.exception()
        cleanup_errors: list[BaseException] = []
        fixture_service.INSTALLED_EXPIRY_RELEASE.set()
        for process in (companion_process, keep_process):
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except BaseException as error:
                    cleanup_errors.append(error)
        try:
            if discovery_path.exists():
                await stop_install(entry, workspace, scenario_environment, discovery_path)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "installed last-client exit cleanup failed",
                ([primary] if primary is not None else []) + cleanup_errors,
            )


async def _run_installed_joint(
    root: Path,
    *,
    entry: str,
    python: Path,
    config: str,
    environment: dict[str, str],
    node: str,
) -> dict[str, object]:
    scenario_root = root / "joint"
    home, workspace, scenario_environment = _prepare_cli_scenario(
        scenario_root, config, environment
    )
    discovery_path = home.path / "service.json"
    capture = scenario_root / "capture-browser.cmd"
    url_file = scenario_root / "browser-url.txt"
    capture.write_text(f'@echo off\n>"{url_file}" echo %*\n', encoding="utf-8")
    observation_path = scenario_root / "provider-observations.jsonl"
    observation_path.write_text("", encoding="utf-8")
    fixture_service.PROVIDER_OBSERVATION_PATH = observation_path
    fixture_service.SETTINGS_ENTERED.clear()
    fixture_service.SETTINGS_RELEASE.clear()
    fixture_service.PROJECT_REMOVAL_ENTERED.clear()
    fixture_service.PROJECT_REMOVAL_RELEASE.clear()
    browser_ready = scenario_root / "browser-ready.json"
    cli_ready = scenario_root / "cli-ready.json"
    cli_foreground_ready = scenario_root / "cli-foreground-ready.json"
    cli_removal_done = scenario_root / "cli-removal-done.json"
    cli_settings_start = scenario_root / "cli-settings-start"
    cli_settings_ready = scenario_root / "cli-settings-ready.json"
    cli_settings_done = scenario_root / "cli-settings-done.json"
    cli_settings_release = scenario_root / "cli-settings-release"
    cli_done = scenario_root / "cli-done.json"
    fixture_service.SETTINGS_RELEASE_PATH = cli_settings_release
    scenario_environment = {
        **scenario_environment,
        "BROWSER": str(capture),
        "AIDE_CAPTURE_URL": str(url_file),
    }
    browser_process: asyncio.subprocess.Process | None = None
    cli_process: asyncio.subprocess.Process | None = None
    release_task: asyncio.Task[None] | None = None
    launch_ticket = ""
    try:
        await asyncio.to_thread(
            _run_command,
            [entry, "web"],
            cwd=workspace,
            env=scenario_environment,
            timeout=60,
        )
        discovery = read_discovery(home)
        if discovery is None:
            raise RuntimeError("installed joint scenario did not publish service discovery")
        launch_url = url_file.read_text(encoding="utf-8").strip().strip('"')
        if not launch_url.startswith("http://127.0.0.1:8765/#ticket="):
            raise RuntimeError("installed joint scenario received an invalid Web launch URL")
        launch_ticket = launch_url.split("#ticket=", 1)[1]
        await asyncio.to_thread(
            _run_command, [entry, "web"], cwd=workspace, env=scenario_environment, timeout=60
        )
        second_launch_url = url_file.read_text(encoding="utf-8").strip().strip('"')
        if not second_launch_url.startswith("http://127.0.0.1:8765/#ticket="):
            raise RuntimeError("installed joint scenario received an invalid second Web launch URL")
        second_launch_ticket = second_launch_url.split("#ticket=", 1)[1]
        if second_launch_ticket == launch_ticket:
            raise RuntimeError("installed CLI reused a single-use Web launch ticket")
        browser_environment = {
            **os.environ,
            "AIDE_E2E_URL": "http://127.0.0.1:8765",
            "AIDE_E2E_TICKET": launch_url.split("#ticket=", 1)[1],
            "AIDE_E2E_SECOND_TICKET": second_launch_ticket,
            "AIDE_E2E_WORKSPACE": str(workspace),
            "AIDE_E2E_OUTPUT": str(scenario_root),
            "AIDE_E2E_INSTANCE": discovery.service_instance_id,
            "AIDE_PROVIDER_OBSERVATION_PATH": str(observation_path),
            "AIDE_JOINT_BROWSER_READY": str(browser_ready),
            "AIDE_CLI_READY": str(cli_ready),
            "AIDE_CLI_FOREGROUND_READY": str(cli_foreground_ready),
            "AIDE_CLI_REMOVAL_DONE": str(cli_removal_done),
            "AIDE_CLI_SETTINGS_START": str(cli_settings_start),
            "AIDE_CLI_SETTINGS_READY": str(cli_settings_ready),
            "AIDE_CLI_SETTINGS_DONE": str(cli_settings_done),
            "AIDE_CLI_SETTINGS_RELEASE": str(cli_settings_release),
            "AIDE_CLI_DONE": str(cli_done),
        }
        browser_process = await asyncio.create_subprocess_exec(
            node,
            str(ROOT / "aide/client/web/frontend/scripts/installed-joint-e2e.mjs"),
            cwd=workspace,
            env=browser_environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def release_settings_generation() -> None:
            await _wait_for_json_file(cli_removal_done, timeout=180)
            fixture_service.PROJECT_REMOVAL_RELEASE.set()
            await _wait_for_path(cli_settings_release, timeout=180)
            fixture_service.SETTINGS_RELEASE.set()

        release_task = asyncio.create_task(release_settings_generation())
        await _wait_for_json_file(
            browser_ready,
            process=browser_process,
            failure_path=scenario_root / "joint-failure.json",
            secrets=(
                launch_ticket,
                second_launch_ticket,
                read_credential(home),
                "installed-fixture-key",
            ),
        )
        cli_environment = {
            **scenario_environment,
            "AIDE_CLI_SCENARIO": "joint",
            "AIDE_CLI_READY": str(cli_ready),
            "AIDE_CLI_FOREGROUND_READY": str(cli_foreground_ready),
            "AIDE_CLI_REMOVAL_DONE": str(cli_removal_done),
            "AIDE_CLI_SETTINGS_START": str(cli_settings_start),
            "AIDE_CLI_SETTINGS_READY": str(cli_settings_ready),
            "AIDE_CLI_SETTINGS_DONE": str(cli_settings_done),
            "AIDE_CLI_SETTINGS_RELEASE": str(cli_settings_release),
            "AIDE_CLI_DONE": str(cli_done),
            "AIDE_PROVIDER_OBSERVATION_PATH": str(observation_path),
        }
        cli_process = await asyncio.create_subprocess_exec(
            str(python),
            "-I",
            str(ROOT / "aide/client/web/frontend/scripts/installed_cli_probe.py"),
            cwd=workspace,
            env=cli_environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        (browser_output, browser_error), (cli_output, cli_error) = await asyncio.wait_for(
            asyncio.gather(browser_process.communicate(), cli_process.communicate()),
            timeout=240,
        )
        if browser_process.returncode != 0:
            text = (browser_output + browser_error).decode("utf-8", errors="replace")
            raise RuntimeError(
                "installed joint browser failed: "
                + _redact_report_text(
                    text, (launch_ticket, read_credential(home), "installed-fixture-key")
                )
            )
        if cli_process.returncode != 0:
            text = (cli_output + cli_error).decode("utf-8", errors="replace")
            raise RuntimeError(
                "installed joint console failed: "
                + _redact_report_text(
                    text, (launch_ticket, read_credential(home), "installed-fixture-key")
                )
            )
        result = json.loads((scenario_root / "joint.json").read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("marker") != "INSTALLED_JOINT_E2E_OK":
            raise RuntimeError(f"installed joint evidence is invalid: {result}")
        return {
            **result,
            "browser_command": [node, str(ROOT / "aide/client/web/frontend/scripts/installed-joint-e2e.mjs")],
            "cli_command": [
                str(python),
                "-I",
                str(ROOT / "aide/client/web/frontend/scripts/installed_cli_probe.py"),
            ],
        }
    finally:
        primary = sys.exception()
        cleanup_errors: list[BaseException] = []
        fixture_service.SETTINGS_RELEASE.set()
        fixture_service.SETTINGS_RELEASE_PATH = None
        fixture_service.PROJECT_REMOVAL_RELEASE.set()
        if release_task is not None:
            release_task.cancel()
            try:
                await release_task
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                cleanup_errors.append(error)
        for process in (cli_process, browser_process):
            if process is not None and process.returncode is None:
                try:
                    process.kill()
                    await process.wait()
                except BaseException as error:
                    cleanup_errors.append(error)
        try:
            if discovery_path.exists():
                await stop_install(entry, workspace, scenario_environment, discovery_path)
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            url_file.unlink(missing_ok=True)
        except OSError as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "installed joint cleanup failed",
                ([primary] if primary is not None else []) + cleanup_errors,
            )


async def validate_install(wheel: Path, root: Path, node: str) -> dict[str, object]:
    installed = await asyncio.to_thread(_smoke_installed_wheel, wheel, root)
    entry = str(installed["entry_point"])
    python = Path(entry).parent / "python.exe"
    profile = root / "用户配置"
    home = profile / ".aide"
    home.mkdir(parents=True)
    workspace = root / "workspace"
    workspace.mkdir()
    capture = root / "capture-browser.cmd"
    url_file = root / "browser-url.txt"
    capture.write_text(f'@echo off\n>"{url_file}" echo %*\n', encoding="utf-8")
    environment = {
        **_artifact_environment(),
        "USERPROFILE": str(profile),
        "HOME": str(profile),
        "BROWSER": str(capture),
        "AIDE_CAPTURE_URL": str(url_file),
    }
    provider, base_url = await _start_fixture_provider()
    config = f'''[models.providers.fixture]
protocol = "openai-compatible"
base_url = "{base_url}"
api_key = "installed-fixture-key"
[models.providers.fixture.models.small-model]
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 120
[models.providers.fixture.models.installed-new-model]
context_window = 8192
max_output = 1024
temperature = 0
reasoning_effort = "mid"
timeout = 120

[runtime]
permission_level = "workspace-write"

[models.routes.chat]
provider_id = "fixture"
model = "small-model"
'''
    discovery_path = home / "service.json"
    cross_ready_path = root / "cross-browser-ready.json"
    cross_cli_ready_path = root / "cross-cli-ready.json"
    cross_cli_done_path = root / "cross-cli-done.json"
    cross_release_path = root / "cross-release"
    cross_observation_path = root / "cross-provider-observations.jsonl"
    cross_release_task: asyncio.Task[None] | None = None
    private_marker = "installed-browser-private-session-marker"
    try:
        cli_competition = await _run_installed_cli_competition(
            root,
            entry=entry,
            python=python,
            config=config,
            environment=environment,
        )
        last_client_exit = await _run_installed_last_client_exit(
            root,
            entry=entry,
            python=python,
            config=config,
            environment=environment,
        )
        confirmation_path = root / "joint-confirmation-outside.txt"
        confirmation_path.write_text("confirmation fixture content\n", encoding="utf-8")
        fixture_service.CONFIRMATION_PATH = str(confirmation_path)
        joint = await _run_installed_joint(
            root,
            entry=entry,
            python=python,
            config=config,
            environment=environment,
            node=node,
        )
        cross_observation_path.write_text("", encoding="utf-8")
        fixture_service.PROVIDER_OBSERVATION_PATH = cross_observation_path
        fixture_service.INSTALLED_CONCURRENCY_RELEASE.clear()
        fixture_service.INSTALLED_EXPIRY_RELEASE.clear()

        async def release_concurrent_runs() -> None:
            await _wait_for_path(cross_release_path, timeout=180)
            fixture_service.INSTALLED_CONCURRENCY_RELEASE.set()

        cross_release_task = asyncio.create_task(release_concurrent_runs())
        (home / "config.toml").write_text(config, encoding="utf-8")
        # The production entry owns a fixed port. Refuse an occupied port without stopping it.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 8765))
        await asyncio.to_thread(
            _run_command, [entry, "web"], cwd=workspace, env=environment, timeout=60
        )
        launch_url = url_file.read_text(encoding="utf-8").strip().strip('"')
        assert launch_url.startswith("http://127.0.0.1:8765/#ticket=")
        discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
        browser_environment = {
            **os.environ,
            "AIDE_E2E_URL": "http://127.0.0.1:8765",
            "AIDE_E2E_TICKET": launch_url.split("#ticket=", 1)[1],
            "AIDE_E2E_WORKSPACE": str(workspace),
            "AIDE_E2E_OUTPUT": str(root),
            "AIDE_E2E_INSTANCE": discovery["service_instance_id"],
            "AIDE_CROSS_CLIENT_READY": str(cross_ready_path),
            "AIDE_CROSS_CLIENT_CLI_READY": str(cross_cli_ready_path),
            "AIDE_CROSS_CLIENT_CLI_DONE": str(cross_cli_done_path),
            "AIDE_CLI_PRIVATE_MARKER": private_marker,
            "AIDE_PROVIDER_OBSERVATION_PATH": str(cross_observation_path),
            "AIDE_CONCURRENCY_RELEASE": str(cross_release_path),
        }
        browser_process = await asyncio.create_subprocess_exec(
            node,
            str(ROOT / "aide/client/web/frontend/scripts/installed-web-e2e.mjs"),
            cwd=workspace,
            env=browser_environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        cli_process: asyncio.subprocess.Process | None = None
        try:
            browser_ready = await _wait_for_json_file(cross_ready_path)
            cli_environment = {
                **environment,
                "AIDE_CLI_SCENARIO": "cross-client",
                "AIDE_CLI_READY": str(cross_cli_ready_path),
                "AIDE_CLI_DONE": str(cross_cli_done_path),
                "AIDE_CLI_PROMPT": "installed CLI concurrent session streaming markdown",
                "AIDE_CLI_CONTESTED_SESSION": str(browser_ready["browser_session_id"]),
                "AIDE_CLI_PRIVATE_MARKER": private_marker,
                "AIDE_PROVIDER_OBSERVATION_PATH": str(cross_observation_path),
            }
            cli_process = await asyncio.create_subprocess_exec(
                str(python),
                "-I",
                str(ROOT / "aide/client/web/frontend/scripts/installed_cli_probe.py"),
                discovery["service_instance_id"],
                str(discovery["pid"]),
                cwd=workspace,
                env=cli_environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            (browser_output, browser_error), (cli_output, cli_error) = await asyncio.wait_for(
                asyncio.gather(browser_process.communicate(), cli_process.communicate()),
                timeout=180,
            )
        finally:
            for process in (cli_process, browser_process):
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
        if browser_process.returncode != 0:
            browser_text = (browser_output + browser_error).decode("utf-8", errors="replace")
            raise RuntimeError(
                browser_text.replace(launch_url.split("#ticket=", 1)[1], "[redacted]")
            )
        if cli_process is None or cli_process.returncode != 0:
            cli_text = (cli_output + cli_error).decode("utf-8", errors="replace")
            raise RuntimeError(f"installed CLI cross-client probe failed: {cli_text}")
        cross_cli = json.loads(cross_cli_done_path.read_text(encoding="utf-8"))
        if cross_cli.get("status") != "passed":
            raise RuntimeError(f"installed CLI cross-client probe returned: {cross_cli}")
        cli_result = await asyncio.to_thread(
            _run_command,
            [
                str(python),
                "-I",
                str(ROOT / "aide/client/web/frontend/scripts/installed_cli_probe.py"),
                discovery["service_instance_id"],
                str(discovery["pid"]),
            ],
            cwd=workspace,
            env=environment,
            timeout=60,
        )
        cli_evidence = json.loads(cli_result.stdout.strip().splitlines()[-1])
        assert cli_evidence["marker"] == "INSTALLED_CLI_CONNECT_OK"
        assert json.loads(discovery_path.read_text(encoding="utf-8")) == discovery
        await stop_install(entry, workspace, environment, discovery_path)
        transcripts = list((workspace / ".aide/sessions").glob("*.jsonl"))
        messages = [
            json.loads(line)
            for path in transcripts
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if cross_cli:
            assert (
                sum(
                    message.get("role") == "user"
                    and message.get("content") == "browser concurrent session streaming markdown"
                    for message in messages
                )
                == 1
            )
            assert (
                sum(
                    message.get("role") == "user"
                    and message.get("content")
                    == "installed CLI concurrent session streaming markdown"
                    for message in messages
                )
                == 1
            )
        else:
            assert (
                sum(
                    message.get("role") == "user"
                    and message.get("content")
                    == "installed package conversation\nstreaming markdown"
                    for message in messages
                )
                == 1
            )
        assert any(
            message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
            and message["content"].startswith("# Streamed answer")
            and message.get("error") is None
            for message in messages
        )
        return {
            **installed,
            "wheel_path": str(wheel),
            "wheel_sha256": sha256(wheel.read_bytes()).hexdigest(),
            "cli_competition": cli_competition,
            "last_client_exit": last_client_exit,
            "joint": joint,
            "web": json.loads((root / "browser.json").read_text(encoding="utf-8")),
            "cross_client": {
                "browser_ready": browser_ready,
                "cli": cross_cli,
                "browser_command": [node, str(ROOT / "aide/client/web/frontend/scripts/installed-web-e2e.mjs")],
                "cli_command": [
                    str(python),
                    "-I",
                    str(ROOT / "aide/client/web/frontend/scripts/installed_cli_probe.py"),
                ],
            },
            "cli": cli_evidence,
            "stop": "passed",
            "persisted_conversation": "passed",
        }
    finally:
        primary = sys.exception()
        cleanup_errors: list[BaseException] = []
        fixture_service.INSTALLED_CONCURRENCY_RELEASE.set()
        fixture_service.INSTALLED_EXPIRY_RELEASE.set()
        fixture_service.SETTINGS_RELEASE.set()
        fixture_service.PROJECT_REMOVAL_RELEASE.set()
        fixture_service.PROVIDER_OBSERVATION_PATH = None
        if cross_release_task is not None:
            cross_release_task.cancel()
            try:
                await cross_release_task
            except asyncio.CancelledError:
                pass
            except Exception as error:
                cleanup_errors.append(error)
        try:
            if discovery_path.exists():
                await stop_install(entry, workspace, environment, discovery_path)
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            await provider.cleanup()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            url_file.unlink(missing_ok=True)
        except OSError as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            raise BaseExceptionGroup(
                "installed validation cleanup failed",
                ([primary] if primary is not None else []) + cleanup_errors,
            )


async def stop_install(
    entry: str, workspace: Path, environment: dict[str, str], discovery: Path
) -> None:
    try:
        await asyncio.to_thread(
            _run_command, [entry, "service", "stop"], cwd=workspace, env=environment, timeout=60
        )
    except RuntimeError as error:
        message = str(error)
        if not (
            message.startswith("command failed with exit 1: ")
            and "\nservice_not_running: No active local service was found." in message
            and not discovery.exists()
        ):
            raise
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 8765))
        except OSError as port_error:
            raise error from port_error
        (workspace.parent / "stop-cleanup.json").write_text(
            json.dumps(
                {
                    "status": "already-stopped",
                    "command": [entry, "service", "stop"],
                    "exit_code": 1,
                    "error_code": "service_not_running",
                    "discovery_removed": True,
                    "port_released": True,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    deadline = asyncio.get_running_loop().time() + 30
    while True:
        if not discovery.exists():
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                    probe.bind(("127.0.0.1", 8765))
                return
            except OSError:
                pass
        if asyncio.get_running_loop().time() >= deadline:
            raise RuntimeError("installed service did not remove discovery and release its port")
        await asyncio.sleep(0.05)


async def validate(wheels: tuple[Path, Path], output: Path, node: str) -> None:
    source = _source_identity()
    results = {}
    for kind, wheel in zip(("direct", "rebuilt"), wheels, strict=True):
        results[kind] = await validate_install(wheel, output / kind, node)
    revision = _run_command(["git", "rev-parse", "HEAD"]).stdout.strip()
    dirty = bool(_run_command(["git", "status", "--porcelain"]).stdout.strip())
    payload = {
        "marker": "INSTALLED_WEB_LIFECYCLE_OK",
        "installations": results,
        "acceptance_matrix": build_acceptance_matrix(
            (),
            artifact_status="passed",
            browser_statuses={"R17": "passed"},
            installed_statuses={
                "R01": "passed",
                "R02": "passed",
                "R05": "passed",
                "R06": "passed",
                "R08": "passed",
                "R13": "passed",
                "R15": "passed",
                "R17": "passed",
            },
        ),
        "source_root": str(ROOT),
        "source_head": revision,
        "source_dirty": dirty,
        "source": source,
        "probe_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "scripts/installed_web_validation.py",
                "aide/client/web/frontend/scripts/installed_cli_probe.py",
                "aide/client/web/frontend/scripts/installed-web-e2e.mjs",
                "aide/client/web/frontend/scripts/installed-joint-e2e.mjs",
                "aide/client/web/frontend/scripts/e2e_service.py",
            )
        },
        "terminal_validation": {
            "adapter": "installed console metadata entry with Textual run_test",
            "real_tty": "not-run",
            "no_node_in_application_path": True,
        },
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


def main() -> None:
    if not is_windows_host():
        print(WINDOWS_REQUIRED_ERROR, file=sys.stderr)
        raise SystemExit(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("output must be outside the source repository")
    node = shutil.which("node")
    if node is None:
        parser.error("the Playwright controller requires Node; installed applications do not")
    output.mkdir(parents=True, exist_ok=False)
    try:
        wheels = build_distributions(output)
        asyncio.run(validate(wheels, output, node))
    except BaseException as error:
        payload = {
            "status": "failed",
            "source": _source_identity(),
            "failure": _exception_payload(error, secrets=("installed-fixture-key",)),
        }
        (output / "failure-report.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
