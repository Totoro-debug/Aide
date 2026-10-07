from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import cast

import pytest

from aide.agent.tools import base as tool_base_module
from aide.agent.tools.core.exec import ExecTool
from aide.agent.tools.core.exec_host import (
    ExecProcessSpec,
    PowerShellExecHost,
    resolve_exec_shell,
)
from aide.agent.tools.core.exec_policy import ExecAssessment, ExecOutcome, assess_command
from aide.agent.tools.network_safety import DNSResolver
from aide.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ModelToolCall,
)
from tests.fixtures import SingleToolGateway
from tests.fixtures.gateway import contextual_tool


def _call(
    arguments: dict[str, object],
    *,
    call_id: str = "call_exec",
) -> ModelToolCall:
    return ModelToolCall(id=call_id, name="exec", arguments=json.dumps(arguments))


class FakeResolver:
    def __init__(self, answers: tuple[str, ...] = ()) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []
        self.failure: Exception | None = None

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        self.calls.append((hostname, port))
        if self.failure is not None:
            raise self.failure
        return self.answers


class FakeProcess:
    def __init__(
        self,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        returncode: int | None = 0,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.killed = asyncio.Event()
        self.reaped = asyncio.Event()
        self.started = asyncio.Event()
        self.released = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        self.started.set()
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.killed.set()
        self.returncode = -9
        self.released.set()

    async def wait(self) -> int:
        self.reaped.set()
        return -9 if self.returncode is None else self.returncode


class BlockingProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__(returncode=None)

    async def communicate(self) -> tuple[bytes, bytes]:
        self.started.set()
        await self.released.wait()
        return b"partial stdout\n", b"partial stderr\n"


class TimeoutThenSlowReapProcess(FakeProcess):
    def __init__(self) -> None:
        super().__init__(returncode=None)
        self.reap_started = asyncio.Event()
        self.allow_reap = asyncio.Event()

    async def communicate(self) -> tuple[bytes, bytes]:
        self.started.set()
        raise TimeoutError

    async def wait(self) -> int:
        self.reap_started.set()
        await self.allow_reap.wait()
        self.reaped.set()
        return -9 if self.returncode is None else self.returncode


def _gateway(
    workspace: Path,
    *,
    resolver: DNSResolver | None = None,
    confirmation: ConfirmationRequester | None = None,
) -> SingleToolGateway:
    tool = contextual_tool(ExecTool, workspace=workspace, resolver=resolver, host=_powershell_host())
    return SingleToolGateway((tool,), confirmation=confirmation)


class InspectedPowerShellHost(PowerShellExecHost):
    """Supply parser facts while exercising real process cleanup and output handling."""

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        return assess_command(command, family="pwsh", syntax_confidence="high", syntax_uncertain=False)


def _powershell_host() -> PowerShellExecHost:
    return InspectedPowerShellHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=lambda name: r"C:\PowerShell\pwsh.exe" if name == "pwsh" else None,
            version_probe=lambda *_: (7, 5),
        )
    )

def _fake_process_factory(
    monkeypatch: pytest.MonkeyPatch,
    process: FakeProcess,
) -> list[tuple[tuple[str, ...], dict[str, object]]]:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    async def create_process(*command: str, **kwargs: object) -> FakeProcess:
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    return calls


def test_exec_schema_declares_host_shell_command_cwd_and_timeout(workspace: Path) -> None:
    schema = contextual_tool(ExecTool, workspace=workspace).to_schema()

    assert schema == {
        "type": "function",
        "function": {
            "name": "exec",
            "description": ExecTool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                        "minLength": 1,
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Working directory.",
                        "minLength": 1,
                        "default": ".",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds.",
                        "minimum": 1,
                        "maximum": 600,
                        "default": 60,
                    },
                },
                "required": ["command"],
            },
        },
    }


@pytest.mark.asyncio
async def test_exec_starts_a_no_profile_powershell_with_minimal_environment_and_captured_streams(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    nested = workspace / "nested"
    nested.mkdir()
    process = FakeProcess(stdout=b"out\n", stderr=b"err\n")
    calls = _fake_process_factory(monkeypatch, process)
    gateway = _gateway(workspace)
    command = "Write-Output output"

    result = await gateway.call(_call({"command": command, "cwd": "nested", "timeout": 1}))
    default_cwd_result = await gateway.call(_call({"command": "Get-Location"}, call_id="default-cwd"))

    assert result.status == "success"
    assert default_cwd_result.status == "success"
    assert "Exit code: 0" in result.content
    assert "stdout:\nout\n" in result.content
    assert "stderr:\nerr\n" in result.content
    argv, options = calls[0]
    assert argv[:5] == (r"C:\PowerShell\pwsh.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command")
    assert command in argv[5]
    assert options["cwd"] == os.fspath(nested.resolve())
    assert options["stdin"] is asyncio.subprocess.DEVNULL
    assert options["stdout"] is asyncio.subprocess.PIPE
    assert options["stderr"] is asyncio.subprocess.PIPE
    environment = cast(dict[str, str], options["env"])
    expected_environment = {
        name: next(value for key, value in os.environ.items() if key.upper() == name)
        for name in ("HOME", "LANG", "TERM", "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE")
        if any(key.upper() == name for key in os.environ)
    }
    assert environment == expected_environment
    assert calls[1][1]["cwd"] == os.fspath(workspace.resolve())


@pytest.mark.asyncio
async def test_exec_keeps_nonzero_exit_success_and_replaces_invalid_utf8(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = FakeProcess(stdout=b"ok\xff\n", stderr=b"failed\xfe\n", returncode=23)
    _fake_process_factory(monkeypatch, process)

    result = await _gateway(workspace).call(_call({"command": "Write-Output output"}))

    assert result.status == "success"
    assert "Exit code: 23" in result.content
    assert "ok\ufffd" in result.content
    assert "failed\ufffd" in result.content


@pytest.mark.asyncio
async def test_exec_freezes_canonical_cwd_before_inspection_and_execution(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    tmp_path: Path,
) -> None:
    inspected: list[Path] = []
    executed: list[Path] = []
    inside = workspace / "inside"
    outside = tmp_path / "outside"
    inside.mkdir()
    outside.mkdir()
    dot_resolutions = 0

    def drifting_resolution(root: Path, requested: str | Path) -> Path:
        nonlocal dot_resolutions
        del root
        if requested == ".":
            dot_resolutions += 1
            return inside if dot_resolutions < 4 else outside
        return Path(requested)

    class Host:
        resolved_shell = _powershell_host().resolved_shell

        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del command
            inspected.append(cwd)
            return ExecAssessment(syntax_confidence="high", syntax_uncertain=False)

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del command, timeout
            executed.append(cwd)
            return ExecOutcome(exit_code=0, stdout=b"", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            raise AssertionError(f"unexpected process spec request for {cwd}")

    monkeypatch.setattr(tool_base_module, "resolve_tool_path", drifting_resolution)
    gateway = SingleToolGateway((contextual_tool(ExecTool, workspace=workspace, host=Host()),))

    result = await gateway.call(_call({"command": "Get-Location"}))

    assert result.status == "success"
    assert inspected == executed == [inside]
    assert dot_resolutions == 1


@pytest.mark.asyncio
async def test_exec_rejects_blank_or_out_of_range_arguments_before_dns_or_spawn(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    resolver = FakeResolver(("10.0.0.1",))
    calls: list[object] = []

    async def must_not_spawn(*command: str, **kwargs: object) -> FakeProcess:
        del command, kwargs
        calls.append(True)
        raise AssertionError("Exec spawned for invalid arguments")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    gateway = _gateway(workspace, resolver=resolver)

    blank = await gateway.call(_call({"command": "  "}, call_id="blank"))
    low = await gateway.call(_call({"command": "Get-Location", "timeout": 0}, call_id="low"))
    high = await gateway.call(_call({"command": "Get-Location", "timeout": 601}, call_id="high"))

    assert blank.status == "error"
    assert low.status == "error"
    assert high.status == "error"
    assert resolver.calls == []
    assert calls == []


@pytest.mark.asyncio
async def test_exec_requests_confirmation_for_destructive_commands(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = FakeProcess(stdout=b"approved\n")
    _fake_process_factory(monkeypatch, process)
    requests: list[ConfirmationRequest] = []

    async def confirm(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(workspace, confirmation=confirm).call(
        _call({"command": "rm -rf ./build"}, call_id="destructive")
    )

    assert result.status == "success"
    assert len(requests) == 1
    assert requests[0].tool_call_id == "destructive"
    assert requests[0].tool_name == "exec"
    assert "destructive" in requests[0].reason.lower()


@pytest.mark.asyncio
async def test_exec_refuses_an_unsafe_call_without_a_confirmation_channel(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = FakeProcess(stdout=b"must not run")
    calls = _fake_process_factory(monkeypatch, process)

    result = await _gateway(workspace).call(_call({"command": "rm -rf ./build"}))

    assert result.status == "refused"
    assert "confirmation" in result.content.lower()
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "rm --recursive ./build",
        "rm -v -rf ./build",
        "del /f secret.txt",
        "del /q /f secret.txt",
        "rmdir /s /q build",
        "rmdir /q /s build",
        "format c:",
        "mkfs.ext4 /dev/sda",
        "diskpart",
        "dd if=/dev/zero of=/dev/sda",
        "printf x > /dev/sda",
        r"dd if=x of=\\.\PhysicalDrive0",
        r"printf x > \\.\PhysicalDrive0",
        "shutdown now",
        "reboot",
        "poweroff",
        ":(){ :|:& };:",
    ),
)
async def test_exec_requests_confirmation_for_each_known_destructive_pattern(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    command: str,
) -> None:
    process = FakeProcess(stdout=b"approved\n")
    _fake_process_factory(monkeypatch, process)
    requests: list[ConfirmationRequest] = []

    async def confirm(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(workspace, confirmation=confirm).call(_call({"command": command}))

    assert result.status == "success"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "answers", "failure"),
    (
        ("http://private.example", ("10.0.0.1",), None),
        ("http://mapped.example", ("::ffff:10.0.0.1",), None),
        ("http://dns-failure.example", (), OSError("DNS unavailable")),
    ),
)
async def test_exec_requests_confirmation_for_private_or_unverifiable_urls(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    url: str,
    answers: tuple[str, ...],
    failure: Exception | None,
) -> None:
    process = FakeProcess(stdout=b"approved\n")
    _fake_process_factory(monkeypatch, process)
    resolver = FakeResolver(answers)
    resolver.failure = failure
    requests: list[ConfirmationRequest] = []

    async def confirm(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(
        workspace,
        resolver=resolver,
        confirmation=confirm,
    ).call(_call({"command": f"curl {url}"}, call_id="url"))

    assert result.status == "success"
    assert len(requests) == 1
    assert resolver.calls == [(url.split("://", maxsplit=1)[1], 80)]
    assert "requires confirmation" in requests[0].reason


@pytest.mark.asyncio
async def test_exec_resolves_public_url_addresses_without_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = FakeProcess(stdout=b"public\n")
    _fake_process_factory(monkeypatch, process)
    resolver = FakeResolver(("8.8.8.8", "2001:4860:4860::8888"))
    gateway = _gateway(workspace, resolver=resolver)

    result = await gateway.call(_call({"command": "curl https://example.com/path"}))

    assert result.status == "success"
    assert resolver.calls == [("example.com", 443)]
    assert result.confirmation is None


@pytest.mark.asyncio
async def test_exec_checks_all_urls_in_one_command(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = FakeProcess(stdout=b"approved\n")
    _fake_process_factory(monkeypatch, process)
    resolver = FakeResolver(("8.8.8.8",))
    requests: list[ConfirmationRequest] = []

    async def confirm(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(workspace, resolver=resolver, confirmation=confirm).call(
        _call({"command": "curl https://public.example http://10.0.0.1"})
    )

    assert result.status == "success"
    assert resolver.calls == [("public.example", 443)]
    assert len(requests) == 1
    assert "private" in requests[0].reason


@pytest.mark.asyncio
async def test_exec_requests_confirmation_for_literal_private_url(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = FakeProcess(stdout=b"approved\n")
    _fake_process_factory(monkeypatch, process)
    resolver = FakeResolver(("8.8.8.8",))
    requests: list[ConfirmationRequest] = []

    async def confirm(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(workspace, resolver=resolver, confirmation=confirm).call(
        _call({"command": "curl http://127.0.0.1:8080"})
    )

    assert result.status == "success"
    assert resolver.calls == []
    assert len(requests) == 1
    assert "private" in requests[0].reason


@pytest.mark.asyncio
async def test_exec_requests_confirmation_for_external_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    process = FakeProcess(stdout=b"external\n")
    calls = _fake_process_factory(monkeypatch, process)
    requests: list[ConfirmationRequest] = []

    async def confirm(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(workspace, confirmation=confirm).call(
        _call({"command": "Get-Location", "cwd": str(outside)}, call_id="external-cwd")
    )

    assert result.status == "success"
    assert len(requests) == 1
    assert "outside the Workspace" in requests[0].reason
    assert calls[0][1]["cwd"] == os.fspath(outside.resolve())


@pytest.mark.asyncio
async def test_exec_timeout_kills_direct_powershell_and_returns_partial_streams(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = BlockingProcess()
    _fake_process_factory(monkeypatch, process)

    result = await _gateway(workspace).call(
        _call({"command": "Start-Sleep -Seconds 60", "timeout": 1}, call_id="timeout")
    )

    assert result.status == "error"
    assert "timed out" in result.content
    assert "partial stdout" in result.content
    assert "partial stderr" in result.content
    assert "Exit code:" not in result.content
    assert process.killed.is_set()
    assert process.reaped.is_set()


@pytest.mark.asyncio
async def test_exec_cancellation_kills_and_reaps_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = BlockingProcess()
    _fake_process_factory(monkeypatch, process)
    execution = asyncio.create_task(
        _gateway(workspace).call(_call({"command": "Start-Sleep -Seconds 60"}, call_id="cancel"))
    )
    await process.started.wait()

    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert process.killed.is_set()
    assert process.reaped.is_set()


@pytest.mark.asyncio
async def test_exec_cancellation_during_spawn_cleans_up_the_created_process(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = BlockingProcess()
    spawn_started = asyncio.Event()
    release_spawn = asyncio.Event()

    async def create_process(*command: str, **kwargs: object) -> BlockingProcess:
        del command, kwargs
        spawn_started.set()
        await release_spawn.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    execution = asyncio.create_task(
        _gateway(workspace).call(_call({"command": "Start-Sleep -Seconds 60"}, call_id="spawn-cancel"))
    )
    await spawn_started.wait()
    execution.cancel()
    release_spawn.set()

    with pytest.raises(asyncio.CancelledError):
        await execution
    assert process.killed.is_set()
    assert process.reaped.is_set()


@pytest.mark.asyncio
async def test_exec_cancellation_during_spawn_propagates_with_bounded_late_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = BlockingProcess()
    spawn_started = asyncio.Event()
    spawn_cancelled = asyncio.Event()
    release_spawn = asyncio.Event()

    async def create_process(*command: str, **kwargs: object) -> BlockingProcess:
        del command, kwargs
        spawn_started.set()
        try:
            await release_spawn.wait()
        except asyncio.CancelledError:
            spawn_cancelled.set()
            await release_spawn.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    execution = asyncio.create_task(
        _gateway(workspace).call(_call({"command": "Start-Sleep -Seconds 60"}, call_id="bounded-cancel"))
    )
    await spawn_started.wait()

    execution.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=6.0)
    assert spawn_cancelled.is_set()

    release_spawn.set()
    await asyncio.wait_for(process.reaped.wait(), timeout=0.5)
    assert process.killed.is_set()


@pytest.mark.asyncio
async def test_exec_cancellation_during_timeout_cleanup_finishes_cleanup_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = TimeoutThenSlowReapProcess()
    _fake_process_factory(monkeypatch, process)
    execution = asyncio.create_task(
        _gateway(workspace).call(_call({"command": "Start-Sleep -Seconds 60"}, call_id="cleanup-cancel"))
    )
    await process.reap_started.wait()

    execution.cancel()
    await asyncio.sleep(0)

    assert not execution.done()
    process.allow_reap.set()
    with pytest.raises(asyncio.CancelledError):
        await execution
    assert process.killed.is_set()
    assert process.reaped.is_set()


@pytest.mark.asyncio
async def test_exec_output_uses_prefix_truncation_at_4000_characters(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    process = FakeProcess(stdout=b"x" * 5000)
    _fake_process_factory(monkeypatch, process)

    result = await _gateway(workspace).call(_call({"command": "Write-Output many"}))

    assert result.status == "success"
    assert len(result.content) <= 4000
    assert result.content.startswith("Exit code: 0\nstdout:\n")
    assert result.content.endswith("\n\n...[truncated]")


@pytest.mark.asyncio
async def test_exec_launch_failure_is_a_tool_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    async def fail_to_spawn(*command: str, **kwargs: object) -> FakeProcess:
        del command, kwargs
        raise OSError("POWERSHELL_START_FAILURE")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_to_spawn)

    result = await _gateway(workspace).call(_call({"command": "Get-Location"}))

    assert result.status == "error"
    assert "POWERSHELL_START_FAILURE" in result.content
