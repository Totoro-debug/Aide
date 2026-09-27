from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from myclaw.tools.exec.host import (
    EXEC_CAPABILITY_ERROR,
    BashExecHost,
    ExecProcessSpec,
    PowerShellExecHost,
    create_exec_host,
    resolve_exec_shell,
)
from myclaw.tools.exec.policy import (
    ExecAssessment,
    ExecCommandIdentity,
    ExecOutcome,
    ExecShellSelector,
    catastrophic_matches,
)
from myclaw.tools.exec.tool import ExecTool
from myclaw.tools.gateway import ModelToolCall, ToolResult
from tests.fixtures import SingleToolGateway


def _which(values: dict[str, str]) -> Callable[[str], str | None]:
    def which(name: str, path: str | None = None) -> str | None:
        del path
        return values.get(name)

    return which


def _version(
    versions: dict[str, tuple[int, ...]],
) -> Callable[[str, dict[str, str]], tuple[int, ...] | None]:
    def version(executable: str, environment: dict[str, str]) -> tuple[int, ...] | None:
        del environment
        return versions.get(executable)

    return version


@pytest.mark.parametrize(
    ("platform", "selector", "paths", "versions", "family", "executable", "available"),
    (
        (
            "windows",
            "auto",
            {"pwsh": r"C:\PowerShell\pwsh.exe", "powershell": r"C:\Windows\powershell.exe"},
            {r"C:\PowerShell\pwsh.exe": (7, 5)},
            "pwsh",
            r"C:\PowerShell\pwsh.exe",
            True,
        ),
        (
            "windows",
            "auto",
            {"pwsh": r"C:\PowerShell\pwsh.exe", "powershell": r"C:\Windows\powershell.exe"},
            {
                r"C:\PowerShell\pwsh.exe": (5, 1),
                r"C:\Windows\powershell.exe": (5, 1),
            },
            "powershell",
            r"C:\Windows\powershell.exe",
            True,
        ),
        (
            "windows",
            "auto",
            {"pwsh": r"C:\PowerShell\pwsh.exe", "powershell": r"C:\Windows\powershell.exe"},
            {r"C:\Windows\powershell.exe": (5, 1)},
            "powershell",
            r"C:\Windows\powershell.exe",
            True,
        ),
        (
            "windows",
            "powershell",
            {"powershell": r"C:\Windows\powershell.exe", "pwsh": r"C:\PowerShell\pwsh.exe"},
            {r"C:\Windows\powershell.exe": (5, 1)},
            "powershell",
            r"C:\Windows\powershell.exe",
            True,
        ),
        (
            "windows",
            "pwsh",
            {"pwsh": r"C:\PowerShell\pwsh.exe", "powershell": r"C:\Windows\powershell.exe"},
            {r"C:\PowerShell\pwsh.exe": (7, 5)},
            "pwsh",
            r"C:\PowerShell\pwsh.exe",
            True,
        ),
        (
            "windows",
            "pwsh",
            {"powershell": r"C:\Windows\powershell.exe"},
            {},
            "pwsh",
            None,
            False,
        ),
        (
            "windows",
            "pwsh",
            {"pwsh": r"C:\PowerShell\pwsh.exe"},
            {r"C:\PowerShell\pwsh.exe": (6, 2)},
            "pwsh",
            None,
            False,
        ),
        (
            "windows",
            "powershell",
            {"pwsh": r"C:\PowerShell\pwsh.exe"},
            {r"C:\PowerShell\pwsh.exe": (7, 5)},
            "powershell",
            None,
            False,
        ),
        (
            "windows",
            "powershell",
            {"powershell": r"C:\Windows\powershell.exe"},
            {},
            "powershell",
            None,
            False,
        ),
        (
            "windows",
            "auto",
            {"pwsh": r"C:\PowerShell\pwsh.exe"},
            {r"C:\PowerShell\pwsh.exe": (5, 1)},
            "powershell",
            None,
            False,
        ),
        (
            "posix",
            "pwsh",
            {"bash": "/usr/bin/bash"},
            {},
            "bash",
            "/usr/bin/bash",
            True,
        ),
    ),
)
def test_resolve_exec_shell_covers_platform_and_fallback_contract(
    platform: str,
    selector: ExecShellSelector,
    paths: dict[str, str],
    versions: dict[str, tuple[int, ...]],
    family: str,
    executable: str | None,
    available: bool,
) -> None:
    resolved = resolve_exec_shell(
        selector,
        platform=platform,
        which=_which(paths),
        version_probe=_version(versions),
        environment={"PATH": r"C:\secret\path"} if platform == "windows" else {"PATH": "/safe"},
    )

    assert resolved.family == family
    assert resolved.executable == executable
    assert resolved.available is available
    if not available:
        assert resolved.diagnostic == EXEC_CAPABILITY_ERROR
        assert "secret" not in (resolved.diagnostic or "")


@pytest.mark.parametrize(
    ("selector", "paths", "versions"),
    (
        ("pwsh", {"pwsh": "pwsh.exe"}, {"pwsh.exe": (7, 5)}),
        (
            "pwsh",
            {"pwsh": r"C:\Tools\not-powershell.exe"},
            {r"C:\Tools\not-powershell.exe": (7, 5)},
        ),
        (
            "powershell",
            {"powershell": r"C:\Windows\powershell.exe"},
            {r"C:\Windows\powershell.exe": (7, 5)},
        ),
    ),
)
def test_resolve_exec_shell_rejects_ambiguous_or_wrong_shell_identity(
    selector: ExecShellSelector,
    paths: dict[str, str],
    versions: dict[str, tuple[int, ...]],
) -> None:
    resolved = resolve_exec_shell(
        selector,
        platform="windows",
        which=_which(paths),
        version_probe=_version(versions),
        environment={"PATH": r"C:\safe"},
    )

    assert resolved.available is False
    assert resolved.executable is None


@pytest.mark.parametrize(
    ("returncode", "stdout"),
    (
        (0, "7\n"),
        (1, "MYCLAW_PS_VERSION:7.5.0\n"),
        (0, "noise\nMYCLAW_PS_VERSION:7.5.0\n"),
    ),
)
def test_real_version_probe_rejects_failed_or_spoofed_output(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=returncode, stdout=stdout),
    )

    resolved = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": r"C:\PowerShell\pwsh.exe"}),
        environment={"PATH": r"C:\safe"},
    )

    assert resolved.available is False


def test_minimal_environment_is_platform_aware() -> None:
    source = {
        "HOME": r"C:\Users\person",
        "PATH": r"C:\Tools",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "SystemRoot": r"C:\Windows",
        "USERPROFILE": r"C:\Users\person",
        "PSModulePath": r"C:\untrusted-profile-modules",
    }

    windows = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": r"C:\PowerShell\pwsh.exe"}),
        version_probe=_version({r"C:\PowerShell\pwsh.exe": (7, 5)}),
        environment=source,
    )
    posix = resolve_exec_shell(
        "auto",
        platform="posix",
        which=_which({"bash": "/usr/bin/bash"}),
        environment=source,
    )

    assert windows.env["PATHEXT"] == ".COM;.EXE;.BAT;.CMD"
    assert windows.env["SYSTEMROOT"] == r"C:\Windows"
    assert windows.env["USERPROFILE"] == r"C:\Users\person"
    assert "PSMODULEPATH" not in windows.env
    assert posix.env == {"HOME": r"C:\Users\person", "PATH": r"C:\Tools"}


def test_unavailable_exec_host_is_cataloguable_but_fails_at_tool_call(
    tmp_path: Path,
) -> None:
    resolved = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({}),
        version_probe=_version({}),
        environment={"PATH": r"C:\private"},
    )
    host = create_exec_host(resolved)
    tool = ExecTool(workspace=tmp_path, host=host)

    assert tool.name == "exec"
    assert host.resolved_shell.available is False


@pytest.mark.asyncio
async def test_missing_shell_is_a_capability_error_without_confirmation(tmp_path: Path) -> None:
    resolved = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({}),
        version_probe=_version({}),
        environment={"PATH": r"C:\private"},
    )
    result = await SingleToolGateway(
        (ExecTool(workspace=tmp_path, host=create_exec_host(resolved)),)
    ).call(
        ModelToolCall(id="missing-shell", name="exec", arguments='{"command":"pwd"}')
    )

    assert result.status == "error"
    assert result.content == EXEC_CAPABILITY_ERROR
    assert result.confirmation is None


@pytest.mark.asyncio
async def test_bash_parse_only_inspection_does_not_execute_input(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel.txt"
    host = BashExecHost(
        resolve_exec_shell(
            "auto",
            platform="posix",
            which=_which({"bash": "/usr/bin/bash"}),
            version_probe=_version({}),
            environment={"PATH": "/safe"},
        )
    )

    assessment = await host.inspect(
        f"value=$(touch {sentinel}); echo $value; touch {sentinel}",
        tmp_path,
    )

    assert assessment.syntax_uncertain is False
    assert assessment.dynamic_constructs
    assert not sentinel.exists()


@pytest.mark.asyncio
async def test_powershell_inspection_uses_static_parser_without_running_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'{"syntax_ok":true,"command_names":["Set-Content"]}', b""

        def kill(self) -> None:
            raise AssertionError("parser process should not be killed")

        async def wait(self) -> int:
            return 0

    async def create_process(*command: str, **kwargs: object) -> Process:
        calls.append((command, dict(kwargs)))
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    executable = r"C:\PowerShell\pwsh.exe"
    resolved = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    host = PowerShellExecHost(resolved)

    assessment = await host.inspect("Set-Content sentinel.txt changed", tmp_path)

    assert assessment.syntax_uncertain is False
    assert assessment.command_identities[0].requested == "Set-Content"
    assert calls[0][0][:4] == (
        executable,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
    )
    assert len(calls[0][0]) == 6
    assert "ParseInput" in calls[0][0][5]
    assert "Set-Content sentinel.txt changed" not in calls[0][0][5]
    assert calls[0][1]["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_powershell_scriptblock_is_parse_only_and_parser_failure_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"not-json", b""

        def kill(self) -> None:
            raise AssertionError("a completed parser process should not be killed")

        async def wait(self) -> int:
            return 0

    async def create_process(*command: str, **kwargs: object) -> Process:
        del command, kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    executable = r"C:\PowerShell\pwsh.exe"
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect(
        "& { Set-Content sentinel.txt changed; $value = $(Get-Date) }",
        tmp_path,
    )

    assert assessment.syntax_uncertain is True
    assert assessment.inspector_status == "uncertain"
    assert assessment.diagnostics
    assert not (tmp_path / "sentinel.txt").exists()


@pytest.mark.asyncio
async def test_powershell_inspector_crash_is_uncertain_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def create_process(*command: str, **kwargs: object) -> Any:
        del command, kwargs
        raise RuntimeError("parser crash")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    executable = r"C:\PowerShell\pwsh.exe"
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect("Set-Content sentinel.txt changed", tmp_path)

    assert assessment.syntax_uncertain is True
    assert assessment.inspector_status == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(("returncode", "stderr"), ((9, b""), (0, b"parser warning")))
async def test_powershell_inspector_inconsistent_process_result_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    returncode: int,
    stderr: bytes,
) -> None:
    class Process:
        def __init__(self) -> None:
            self.returncode = returncode

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'{"syntax_ok":true,"command_names":["Get-Location"]}', stderr

        def kill(self) -> None:
            raise AssertionError("a completed parser process should not be killed")

        async def wait(self) -> int:
            return self.returncode

    async def create_process(*command: str, **kwargs: object) -> Process:
        del command, kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    executable = r"C:\PowerShell\pwsh.exe"
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect("Get-Location", tmp_path)

    assert assessment.syntax_uncertain is True
    assert assessment.inspector_status == "failed"


@pytest.mark.asyncio
async def test_pwsh_scriptblock_is_reported_as_dynamic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'{"syntax_ok":true,"command_names":["Get-Location"]}', b""

        def kill(self) -> None:
            raise AssertionError("a completed parser process should not be killed")

        async def wait(self) -> int:
            return 0

    async def create_process(*command: str, **kwargs: object) -> Process:
        del command, kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    executable = r"C:\PowerShell\pwsh.exe"
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect("& { Get-Location }", tmp_path)

    assert [construct.kind for construct in assessment.dynamic_constructs] == ["scriptblock"]


@pytest.mark.asyncio
async def test_powershell_inspection_and_execution_share_process_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    class Process:
        returncode = 0

        def __init__(self, output: bytes) -> None:
            self.output = output

        async def communicate(self) -> tuple[bytes, bytes]:
            return self.output, b""

        def kill(self) -> None:
            raise AssertionError("a completed process should not be killed")

        async def wait(self) -> int:
            return 0

    outputs = iter((b'{"syntax_ok":true,"command_names":["Get-Location"]}', b"raw"))

    async def create_process(*command: str, **kwargs: object) -> Process:
        calls.append((command, dict(kwargs)))
        return Process(next(outputs))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    executable = r"C:\PowerShell\pwsh.exe"
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect("Get-Location", tmp_path)
    outcome = await host.execute("Get-Location", tmp_path, timeout=3)

    assert assessment.syntax_uncertain is False
    assert outcome.stdout == b"raw"
    assert calls[0][0][:4] == calls[1][0][:4]
    assert calls[0][1]["cwd"] == calls[1][1]["cwd"] == str(tmp_path)
    assert calls[0][1]["env"] == calls[1][1]["env"]


@pytest.mark.parametrize(
    "command",
    (
        "rm -rf /",
        "rm -rf /home/person",
        "rm -rf ..",
        "rm -rf $HOME",
        "rm -rf .",
        "Remove-Item -Recurse -Force $env:USERPROFILE",
        "dd if=/dev/zero of=/dev/sda",
        "printf x > /dev/nvme0n1",
        r"Set-Content \\.\PhysicalDrive0 data",
        "diskpart",
        "format c:",
        "Remove-Partition -DiskNumber 0 -PartitionNumber 1",
        "git clean -fdx",
        "git -C . clean -xfd",
        "git clean --force --directories -x",
        "git reset --hard",
        "git checkout -f main",
        "git restore .",
        "rm -rf $(cat target)",
        'RM "--recursive" "--force" "$HOME"',
        "shutdown /r /t 0",
        "Restart-Computer -Force",
        "systemctl reboot",
        ":(){ :|:& };:",
    ),
)
def test_catastrophic_matcher_returns_structured_matches(command: str) -> None:
    matches = catastrophic_matches(command)

    assert matches
    assert all(match.rule and match.evidence for match in matches)


@pytest.mark.parametrize(
    "command",
    (
        "echo shutdown",
        "Write-Output 'format c:'",
        "cat /dev/sda",
        "printf '%s' 'rm -rf /'",
        "rm -rf ./build",
        "git clean -ndx",
        "git checkout -f -- README.md",
        "echo 'git reset --hard'",
        "echo ':(){ :|:& };:'",
    ),
)
def test_catastrophic_matcher_ignores_safe_or_scoped_lookalikes(command: str) -> None:
    assert catastrophic_matches(command) == ()


def test_assessment_has_typed_cross_host_boundary() -> None:
    assessment = ExecAssessment(
        syntax_confidence="high",
        syntax_uncertain=False,
        command_identities=(ExecCommandIdentity(requested="pwd", resolved="/usr/bin/pwd"),),
    )

    assert assessment.command_identities[0].resolved == "/usr/bin/pwd"
    assert assessment.file_accesses == ()
    assert assessment.network_targets == ()
    assert assessment.catastrophic_matches == ()


@pytest.mark.asyncio
async def test_host_returns_raw_outcome_and_gateway_constructs_tool_result(tmp_path: Path) -> None:
    class Host:
        resolved_shell = resolve_exec_shell(
            "auto",
            platform="posix",
            which=_which({"bash": "/usr/bin/bash"}),
            version_probe=_version({}),
            environment={"PATH": "/safe"},
        )

        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del command, cwd
            return ExecAssessment(syntax_confidence="high", syntax_uncertain=False)

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del command, cwd, timeout
            return ExecOutcome(exit_code=7, stdout=b"raw stdout", stderr=b"raw stderr")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the Gateway must use the raw Host methods")

    result = await SingleToolGateway((ExecTool(workspace=tmp_path, host=Host()),)).call(
        ModelToolCall(
            id="raw-boundary",
            name="exec",
            arguments='{"command":"printf raw"}',
        )
    )

    assert isinstance(result, ToolResult)
    assert result.status == "success"
    assert "Exit code: 7" in result.content
    assert "raw stdout" in result.content


def test_exec_process_spec_is_shared_between_inspection_and_execution(tmp_path: Path) -> None:
    resolved = resolve_exec_shell(
        "auto",
        platform="posix",
        which=_which({"bash": "/usr/bin/bash"}),
        version_probe=_version({}),
        environment={"PATH": "/safe"},
    )
    host = BashExecHost(resolved)

    inspect_spec = host.process_spec(tmp_path)
    execute_spec = host.process_spec(tmp_path)

    assert isinstance(inspect_spec, ExecProcessSpec)
    assert inspect_spec == execute_spec
    assert inspect_spec.executable == "/usr/bin/bash"
    assert inspect_spec.flags == ("--noprofile", "--norc")
