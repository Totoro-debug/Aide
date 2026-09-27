from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from myclaw.agent.tools.core.exec import ExecTool
from myclaw.agent.tools.core.exec_host import (
    ExecProcessSpec,
    PowerShellExecHost,
    create_exec_host,
    resolve_exec_shell,
)
from myclaw.agent.tools.core.exec_policy import (
    POWERSHELL_READ_CANDIDATES,
    POWERSHELL_WRITE_CANDIDATES,
    ExecAssessment,
    ExecCommandIdentity,
    ExecOutcome,
    catastrophic_matches,
)
from myclaw.agent.tools.tool_gateway import (
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
)
from myclaw.permission.policy import PermissionContext, PermissionSnapshot

_POWERSHELL_CANDIDATE_FIXTURES = (
    ("Get-ChildItem -LiteralPath .", "read-only"),
    ("Get-Content -LiteralPath .\\inside.txt", "read-only"),
    ("Get-Content -LiteralPath .\\inside.txt -Tail 1", "read-only"),
    ("Get-Item -LiteralPath .", "read-only"),
    ("Get-Location", "read-only"),
    ("Get-Location -PSDrive C", "read-only"),
    ("Get-FileHash -LiteralPath .\\inside.txt", "read-only"),
    ("Measure-Object", "read-only"),
    ("Select-Object -Property Name", "read-only"),
    ("Sort-Object -Property Name", "read-only"),
    ("Select-String -Pattern content -LiteralPath .\\inside.txt", "read-only"),
    ("Test-Path -LiteralPath .\\inside.txt", "read-only"),
    ("Resolve-Path -LiteralPath .\\inside.txt", "read-only"),
    ("Format-List -Property Name", "read-only"),
    ("Format-Table -Property Name", "read-only"),
    ("Out-String -Width 80", "read-only"),
    ("New-Item -ItemType File -Path .\\new.txt", "workspace-write"),
    ("Set-Content -LiteralPath .\\write.txt -Value content", "workspace-write"),
    ("Add-Content -LiteralPath .\\write.txt -Value content", "workspace-write"),
    ("Clear-Content -LiteralPath .\\write.txt", "workspace-write"),
    ("Copy-Item -LiteralPath .\\inside.txt -Destination .\\copy.txt", "workspace-write"),
    ("Move-Item -LiteralPath .\\inside.txt -Destination .\\moved.txt", "workspace-write"),
    ("Rename-Item -LiteralPath .\\inside.txt -NewName renamed.txt", "workspace-write"),
    ("Remove-Item -LiteralPath .\\inside.txt", "workspace-write"),
    ("Remove-Item -Recurse -Force .\\inside.txt", "workspace-write"),
    ("Out-File -FilePath .\\out.txt", "workspace-write"),
)


def _module_for_powershell_candidate(command_name: str) -> str:
    if command_name in {
        "Get-FileHash",
        "Measure-Object",
        "Select-Object",
        "Sort-Object",
        "Select-String",
        "Format-List",
        "Format-Table",
        "Out-String",
        "Out-File",
    }:
        return "Microsoft.PowerShell.Utility"
    return "Microsoft.PowerShell.Management"


def _which(values: dict[str, str]) -> Callable[[str], str | None]:
    def which(name: str, path: str | None = None) -> str | None:
        del path
        return values.get(name)

    return which


def _version(
    values: dict[str, tuple[int, ...]],
) -> Callable[[str, dict[str, str]], tuple[int, ...] | None]:
    def probe(executable: str, environment: dict[str, str]) -> tuple[int, ...] | None:
        del environment
        return values.get(executable)

    return probe


def _create_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except (OSError, NotImplementedError) as error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks unavailable: {error}")
    created = subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(link), str(target)),
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"directory junctions unavailable: {created.stderr.strip()}")


@pytest.mark.asyncio
async def test_powershell_inspection_returns_canonical_identity_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    payload = {
        "syntax_ok": True,
        "command_names": ["Get-Content"],
        "identities": [
            {
                "requested": "Get-Content",
                "canonical": "Get-Content",
                "kind": "cmdlet",
                "resolved": "Microsoft.PowerShell.Management",
                "module": "Microsoft.PowerShell.Management",
                "resolution_count": 1,
            }
        ],
    }

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return json.dumps(payload).encode("utf-8"), b""

        def kill(self) -> None:
            raise AssertionError("a completed parser process should not be killed")

        async def wait(self) -> int:
            return 0

    async def create_process(*command: str, **kwargs: object) -> Process:
        del command, kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect("Get-Content .\u005cfile.txt", tmp_path)

    assert assessment.syntax_uncertain is False
    assert assessment.command_identities == (
        ExecCommandIdentity(
            requested="Get-Content",
            canonical="Get-Content",
            resolved="Microsoft.PowerShell.Management",
            module="Microsoft.PowerShell.Management",
            kind="cmdlet",
            resolution_count=1,
        ),
    )


@pytest.mark.asyncio
async def test_powershell_alias_requires_confirmation_and_never_executes_on_decline(
    tmp_path: Path,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )

    class Host:
        resolved_shell = shell

        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del command, cwd
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(
                    ExecCommandIdentity(
                        requested="gci",
                        canonical="Get-ChildItem",
                        resolved="Get-ChildItem",
                        module="Microsoft.PowerShell.Management",
                        resolution_count=1,
                        kind="alias",
                    ),
                ),
            )

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del command, cwd, timeout
            raise AssertionError("an alias must not execute after a declined confirmation")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the policy test must not use process_spec")

    snapshot = PermissionSnapshot(level="read-only", exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=tmp_path,
        ),
    )
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id="powershell-alias",
            name="exec",
            arguments=json.dumps({"command": "gci .", "cwd": ".", "timeout": 7}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details == {
        "command": "gci .",
        "cwd": str(tmp_path.resolve()),
        "timeout": 7,
    }


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="requires native Windows PowerShell paths")
async def test_windows_powershell_51_canonical_workspace_read_executes_directly(
    tmp_path: Path,
) -> None:
    executable = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    shell = resolve_exec_shell(
        "powershell",
        platform="windows",
        which=_which({"powershell": executable}),
        version_probe=_version({executable: (5, 1)}),
        environment={"PATH": r"C:\Windows\System32"},
    )
    target = tmp_path / "inside.txt"
    target.write_text("content", encoding="utf-8")
    inspected: list[Path] = []
    executed: list[tuple[str, Path, int]] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del command
            inspected.append(cwd)
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(
                    ExecCommandIdentity(
                        requested="Get-Content",
                        canonical="Get-Content",
                        resolved="Microsoft.PowerShell.Management",
                        module="Microsoft.PowerShell.Management",
                        resolution_count=1,
                        kind="cmdlet",
                    ),
                ),
            )

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            executed.append((command, cwd, timeout))
            return ExecOutcome(exit_code=0, stdout=b"content", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the policy test must use Host methods")

    snapshot = PermissionSnapshot(level="read-only", exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=tmp_path,
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id="powershell-51-read",
            name="exec",
            arguments=json.dumps(
                {
                    "command": "Get-Content -LiteralPath .\\inside.txt",
                    "cwd": ".",
                    "timeout": 11,
                }
            ),
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert inspected == [tmp_path.resolve()]
    assert executed == [
        ("Get-Content -LiteralPath .\\inside.txt", tmp_path.resolve(), 11)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("command, level", _POWERSHELL_CANDIDATE_FIXTURES)
async def test_every_approved_powershell_candidate_has_a_direct_fixture(
    tmp_path: Path,
    command: str,
    level: str,
) -> None:
    if os.name != "nt" and command in {
        "Get-Content -LiteralPath .\\inside.txt",
        "Get-Content -LiteralPath .\\inside.txt -Tail 1",
        "Get-FileHash -LiteralPath .\\inside.txt",
        "Select-String -Pattern content -LiteralPath .\\inside.txt",
        "Test-Path -LiteralPath .\\inside.txt",
        "Resolve-Path -LiteralPath .\\inside.txt",
        "Copy-Item -LiteralPath .\\inside.txt -Destination .\\copy.txt",
        "Move-Item -LiteralPath .\\inside.txt -Destination .\\moved.txt",
    }:
        pytest.skip("requires native Windows PowerShell paths")
    inside = tmp_path / "inside.txt"
    inside.write_text("content", encoding="utf-8")
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    command_name = command.split(maxsplit=1)[0]
    assert command_name in POWERSHELL_READ_CANDIDATES | POWERSHELL_WRITE_CANDIDATES
    assessment = ExecAssessment(
        syntax_confidence="high",
        syntax_uncertain=False,
        command_identities=(
            ExecCommandIdentity(
                requested=command_name,
                canonical=command_name,
                resolved=_module_for_powershell_candidate(command_name),
                module=_module_for_powershell_candidate(command_name),
                resolution_count=1,
                kind="cmdlet",
            ),
        ),
    )
    executions: list[tuple[str, Path, int]] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, inspected_command: str, cwd: Path) -> ExecAssessment:
            assert inspected_command == command
            assert cwd == tmp_path.resolve()
            return assessment

        async def execute(self, executed_command: str, cwd: Path, timeout: int) -> ExecOutcome:
            executions.append((executed_command, cwd, timeout))
            return ExecOutcome(exit_code=0, stdout=b"ok", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the policy fixture must use Host methods")

    snapshot = PermissionSnapshot(level=level, exec_shell=shell)  # type: ignore[arg-type]
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=tmp_path,
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id=f"candidate-{command_name}",
            name="exec",
            arguments=json.dumps({"command": command}),
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert executions == [(command, tmp_path.resolve(), 60)]


@pytest.mark.asyncio
async def test_static_data_pipeline_can_write_to_an_explicit_workspace_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inside.txt"
    target = tmp_path / "copy.txt"
    source.write_text("inside", encoding="utf-8")
    command = (
        f"Get-Content -LiteralPath '{source}' | "
        f"Set-Content -LiteralPath '{target}'"
    )
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    executions: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, inspected_command: str, cwd: Path) -> ExecAssessment:
            del cwd
            assert inspected_command == command
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(
                    ExecCommandIdentity(
                        requested="Get-Content",
                        canonical="Get-Content",
                        resolved="Microsoft.PowerShell.Management",
                        module="Microsoft.PowerShell.Management",
                        resolution_count=1,
                        kind="cmdlet",
                    ),
                    ExecCommandIdentity(
                        requested="Set-Content",
                        canonical="Set-Content",
                        resolved="Microsoft.PowerShell.Management",
                        module="Microsoft.PowerShell.Management",
                        resolution_count=1,
                        kind="cmdlet",
                    ),
                ),
            )

        async def execute(self, executed_command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            executions.append(executed_command)
            return ExecOutcome(exit_code=0, stdout=b"ok", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the pipeline fixture must use Host methods")

    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            PermissionSnapshot(level="workspace-write", exec_shell=shell),
            workspace_root=tmp_path,
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id="static-data-pipeline",
            name="exec",
            arguments=json.dumps({"command": command}),
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert executions == [command]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "$path = 'inside.txt'; Get-Content $path",
        "Get-Content $(Get-Location)",
        "& { Get-Location }",
        "foreach ($item in (Get-ChildItem)) { Get-Content $item }",
        "Get-Content .\\inside.txt > .\\copy.txt",
        "Get-Content .\\*.txt",
        "Get-Content $env:USERPROFILE",
        "Invoke-Expression 'Get-Location'",
        "Get-Content .\\inside.txt | Remove-Item",
        "Get-Content -UnknownSwitch .\\inside.txt",
        "Get-Content -LiteralPath .\\inside.txt -Encoding $encoding",
        "Get-Content -LiteralPath .\\inside.txt -Delimiter $(Get-Location)",
        "Get-ChildItem -Path . -Depth ($depth)",
        "Get-Item -LiteralPath (Join-Path . inside.txt)",
        "Test-Path -LiteralPath .\\inside.txt -PathType $type",
        "Resolve-Path -LiteralPath $pwd",
        "Get-FileHash -LiteralPath .\\inside.txt -Algorithm $algorithm",
        "Select-String -Pattern $pattern -LiteralPath .\\inside.txt",
        "Measure-Object -Property $property",
        "Select-Object -Property $property",
        "Sort-Object -Property (Get-Location)",
        "Format-Table -Property @{Name='x'; Expression={$_.x}}",
        "Out-String -Width $width",
        "New-Item -Path .\\new.txt -ItemType $type",
        "Set-Content -LiteralPath .\\write.txt -Value (Get-Location)",
        "Add-Content -LiteralPath .\\write.txt -Value $value",
        "Clear-Content -LiteralPath .\\write.txt -Force:$force",
        "Copy-Item -LiteralPath $source -Destination .\\copy.txt",
        "Move-Item -LiteralPath .\\inside.txt -Destination $destination",
        "Rename-Item -LiteralPath .\\inside.txt -NewName $name",
        "Out-File -FilePath .\\out.txt -Encoding $encoding",
        "Get-Location |",
        "Get-Location || Get-Location",
        "Get-Item -Path=.\\inside.txt",
    ),
)
async def test_low_permission_powershell_dynamic_or_unknown_calls_confirm_once(
    tmp_path: Path,
    command: str,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    command_name = command.split(maxsplit=1)[0]
    calls: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, inspected_command: str, cwd: Path) -> ExecAssessment:
            del cwd
            assert inspected_command == command
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(
                    ExecCommandIdentity(
                        requested=command_name,
                        canonical=command_name,
                        resolved="Microsoft.PowerShell.Management",
                        module="Microsoft.PowerShell.Management",
                        resolution_count=1,
                        kind="cmdlet",
                    ),
                ),
            )

        async def execute(self, executed_command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            calls.append(executed_command)
            return ExecOutcome(exit_code=0, stdout=b"unexpected", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the low-permission dynamic fixture must not spawn")

    snapshot = PermissionSnapshot(level="read-only", exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(snapshot, workspace_root=tmp_path),
    )
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id=f"dynamic-{abs(hash(command))}",
            name="exec",
            arguments=json.dumps({"command": command}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details["command"] == command
    assert calls == []


def test_exec_identity_positional_kind_remains_compatible() -> None:
    identity = ExecCommandIdentity("pwd", "/usr/bin/pwd", "native")

    assert identity.kind == "native"
    assert identity.canonical is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "$value = Get-Location; $value",
        "& { Set-Content .\\dynamic.txt value }",
        "Invoke-Expression 'Get-Location'",
    ),
)
async def test_full_access_executes_parseable_noncatastrophic_powershell_dynamic_code(
    tmp_path: Path,
    command: str,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    executed: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, inspected_command: str, cwd: Path) -> ExecAssessment:
            del cwd
            assert inspected_command == command
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                dynamic_constructs=(),
                command_identities=(),
            )

        async def execute(self, executed_command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            executed.append(executed_command)
            return ExecOutcome(exit_code=0, stdout=b"dynamic", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the Full-Access fixture must use Host methods")

    snapshot = PermissionSnapshot(level="full-access", exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(snapshot, workspace_root=tmp_path),
    )

    result = await gateway.call(
        ModelToolCall(
            id=f"full-dynamic-{abs(hash(command))}",
            name="exec",
            arguments=json.dumps({"command": command}),
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert executed == [command]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "command_template", "cwd", "expected_confirmation"),
    (
        ("read-only", "Get-Content -LiteralPath {outside}", ".", True),
        ("workspace-write", "Get-Content -LiteralPath {outside}", ".", True),
        ("full-access", "Get-Content -LiteralPath {outside}", ".", False),
        ("read-only", "Set-Content -LiteralPath {inside} -Value x", ".", True),
        ("workspace-write", "Set-Content -LiteralPath {inside} -Value x", ".", False),
        ("workspace-write", "Set-Content -LiteralPath {outside} -Value x", ".", True),
        ("full-access", "Set-Content -LiteralPath {outside} -Value x", ".", False),
        ("workspace-write", "Get-Content -LiteralPath {inside}", ".", False),
        ("workspace-write", "Get-Content -LiteralPath {missing}\\new.txt", ".", True),
        ("read-only", "Get-Content -LiteralPath .\\inside.txt", ".", False),
        ("read-only", "Get-Content -LiteralPath ..\\outside\\outside.txt", ".", True),
        ("read-only", 'Get-Content -LiteralPath "{spaced}"', ".", False),
        ("workspace-write", "Set-Content -LiteralPath {missing}\\new.txt -Value x", ".", False),
        ("read-only", "Get-Content -LiteralPath {drive_root}", ".", True),
        ("read-only", "Get-Content -LiteralPath \\\\server\\share\\file.txt", ".", True),
        ("read-only", "Get-Content -LiteralPath {inside_case}", ".", False),
        (
            "workspace-write",
            "Move-Item -LiteralPath {outside} -Destination {workspace}\\moved.txt",
            ".",
            True,
        ),
        ("read-only", "Get-Content -LiteralPath Registry::HKLM\\Software", ".", True),
        ("read-only", "Get-Content -LiteralPath HKLM:\\Software", ".", True),
        ("read-only", "Get-Content -LiteralPath Env:\\USERPROFILE", ".", True),
        ("read-only", "Get-Content -LiteralPath C:relative", ".", True),
        ("workspace-write", "New-Item -Path {inside} -Name ..\\outside", ".", True),
        ("workspace-write", "Rename-Item -LiteralPath {inside} -NewName ..\\outside.txt", ".", True),
        ("read-only", "Get-Location", "..", True),
        ("full-access", "Get-Location", "..", False),
    ),
)
async def test_powershell_path_roles_follow_level_and_canonical_containment(
    tmp_path: Path,
    level: str,
    command_template: str,
    cwd: str,
    expected_confirmation: bool,
) -> None:
    if os.name != "nt" and command_template in {
        "Get-Content -LiteralPath .\\inside.txt",
        "Get-Content -LiteralPath {inside_case}",
        "New-Item -Path {inside} -Name ..\\outside",
        "Rename-Item -LiteralPath {inside} -NewName ..\\outside.txt",
    }:
        pytest.skip("requires native Windows PowerShell paths")
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "inside.txt").write_text("inside", encoding="utf-8")
    (workspace / "space name.txt").write_text("inside", encoding="utf-8")
    (outside / "outside.txt").write_text("outside", encoding="utf-8")
    command = command_template.format(
        workspace=str(workspace),
        inside=str(workspace / "inside.txt"),
        inside_case=str(workspace / "inside.txt").swapcase(),
        spaced=str(workspace / "space name.txt"),
        outside=str(outside / "outside.txt"),
        missing=str(workspace / "missing"),
        drive_root=workspace.anchor,
    )
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    command_name = command.split(maxsplit=1)[0]
    module = _module_for_powershell_candidate(command_name)
    executions: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, inspected_command: str, inspected_cwd: Path) -> ExecAssessment:
            del inspected_cwd
            assert inspected_command == command
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(
                    ExecCommandIdentity(
                        requested=command_name,
                        canonical=command_name,
                        resolved=module,
                        module=module,
                        resolution_count=1,
                        kind="cmdlet",
                    ),
                ),
            )

        async def execute(self, executed_command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            executions.append(executed_command)
            return ExecOutcome(exit_code=0, stdout=b"ok", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the policy fixture must use Host methods")

    snapshot = PermissionSnapshot(level=level, exec_shell=shell)  # type: ignore[arg-type]
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=workspace, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=workspace,
        ),
    )
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id=f"path-{abs(hash((level, command, cwd)))}",
            name="exec",
            arguments=json.dumps({"command": command, "cwd": cwd}),
        ),
        confirmation=decline,
    )

    assert len(requests) == int(expected_confirmation)
    assert result.status == ("refused" if expected_confirmation else "success")
    assert len(executions) == int(not expected_confirmation)
    if expected_confirmation:
        assert requests[0].details["cwd"] == str(
            (workspace / cwd).resolve() if cwd != "." else workspace.resolve()
        )


@pytest.mark.asyncio
async def test_powershell_read_through_workspace_reparse_point_confirms(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside", encoding="utf-8")
    link = workspace / "linked'path"
    _create_directory_link(link, outside)
    escaped_target = str(link / "secret.txt").replace("'", "''")
    command = f"Get-Content -LiteralPath '{escaped_target}'"
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    requests: list[ConfirmationRequest] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, inspected_command: str, cwd: Path) -> ExecAssessment:
            del cwd
            assert inspected_command == command
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(
                    ExecCommandIdentity(
                        requested="Get-Content",
                        canonical="Get-Content",
                        resolved="Microsoft.PowerShell.Management",
                        module="Microsoft.PowerShell.Management",
                        resolution_count=1,
                        kind="cmdlet",
                    ),
                ),
            )

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del command, cwd, timeout
            raise AssertionError("an external reparse target must not execute after decline")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the reparse fixture must use Host methods")

    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=workspace, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            PermissionSnapshot(level="read-only", exec_shell=shell),
            workspace_root=workspace,
        ),
    )

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id="reparse-read",
            name="exec",
            arguments=json.dumps({"command": command}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="requires native Windows PowerShell paths")
@pytest.mark.parametrize(
    "command",
    (
        "git status",
        "git.exe status",
        "git diff --stat",
        "git log --oneline -n 5",
        "git show HEAD",
        "git branch --list",
        "git rev-parse --show-toplevel",
        "git ls-files --stage",
    ),
)
async def test_every_cross_host_git_read_form_has_a_direct_powershell_fixture(
    tmp_path: Path,
    command: str,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    git_executable = r"C:\Program Files\Git\cmd\git.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    executed: list[str] = []
    audited: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, inspected_command: str, cwd: Path) -> ExecAssessment:
            del cwd
            assert inspected_command == command
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                git_delegation_safe=True,
                command_identities=(
                    ExecCommandIdentity(
                        requested=command.split(maxsplit=1)[0],
                        canonical="git.exe",
                        resolved=git_executable,
                        resolution_count=1,
                        kind="native",
                    ),
                ),
            )

        async def audit_git_delegation(
            self,
            audited_command: str,
            cwd: Path,
            workspace_root: Path,
            assessment: ExecAssessment,
        ) -> ExecAssessment:
            assert audited_command == command
            assert cwd == tmp_path.resolve()
            assert workspace_root == tmp_path
            audited.append(audited_command)
            return replace(assessment, git_delegation_safe=True)

        async def execute(self, executed_command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            executed.append(executed_command)
            return ExecOutcome(exit_code=0, stdout=b"git", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the Git fixture must use Host methods")

    snapshot = PermissionSnapshot(level="read-only", exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(snapshot, workspace_root=tmp_path),
    )

    result = await gateway.call(
        ModelToolCall(
            id=f"git-{abs(hash(command))}",
            name="exec",
            arguments=json.dumps({"command": command}),
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert audited == [command]
    assert executed == [command]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "git commit -m message",
        "git status --unknown",
        "git branch",
        "git -c core.pager=cat status",
        "git log --format=$format",
        "git status --untracked-files=$mode",
        "git -C $repo status",
        "git show :/needle",
        "git rev-parse 'HEAD^{tree}'",
        "git status -- ':!inside.txt'",
        "git --config-env=core.pager=PAGER status",
        "Get-Location | git status",
    ),
)
async def test_unlisted_git_forms_require_one_confirmation(
    tmp_path: Path,
    command: str,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    git_executable = r"C:\Program Files\Git\cmd\git.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    executed: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, inspected_command: str, cwd: Path) -> ExecAssessment:
            del cwd
            assert inspected_command == command
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(
                    ExecCommandIdentity(
                        requested="git",
                        canonical="git.exe",
                        resolved=git_executable,
                        resolution_count=1,
                        kind="native",
                    ),
                ),
            )

        async def execute(self, executed_command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            executed.append(executed_command)
            return ExecOutcome(exit_code=0, stdout=b"unexpected", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the unlisted Git fixture must not spawn")

    snapshot = PermissionSnapshot(level="workspace-write", exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(snapshot, workspace_root=tmp_path),
    )
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id=f"git-unknown-{abs(hash(command))}",
            name="exec",
            arguments=json.dumps({"command": command}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details["command"] == command
    assert executed == []


def test_git_environment_is_fixed_only_for_git_processes() -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    resolved = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe", "GIT_PAGER": "untrusted"},
    )
    host = PowerShellExecHost(resolved)

    plain = host.environment_for_command("Get-Location")
    git = host.environment_for_command("git status")

    assert plain == resolved.env
    assert git["GIT_CONFIG_NOSYSTEM"] == "1"
    assert git["GIT_CONFIG_GLOBAL"] == "NUL"
    assert git["GIT_PAGER"] == "cat"
    assert git["GIT_EXTERNAL_DIFF"] == ""
    assert git["GIT_OPTIONAL_LOCKS"] == "0"
    assert git["GIT_CONFIG_VALUE_1"] == "false"
    assert git["GIT_CONFIG_VALUE_3"] == "NUL"
    assert host.command_for_execution("git diff --stat") == (
        "git diff --no-ext-diff --no-textconv --stat"
    )
    assert host.command_for_execution('git -C "repo root" show HEAD') == (
        'git -C "repo root" show --no-ext-diff --no-textconv HEAD'
    )
    assert host.command_for_execution("git status") == "git status"


def test_bash_command_and_environment_are_hardened_for_git() -> None:
    resolved = resolve_exec_shell(
        "auto",
        platform="posix",
        which=_which({"bash": "/bin/bash"}),
        environment={"PATH": "/bin", "HOME": "/home/test"},
    )
    host = create_exec_host(resolved)

    environment = host.environment_for_command("git diff --stat")  # type: ignore[attr-defined]
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_PAGER"] == "cat"
    assert environment["GIT_EXTERNAL_DIFF"] == ""
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_CONFIG_VALUE_1"] == "false"
    assert environment["GIT_CONFIG_VALUE_3"] == "/dev/null"
    assert host.command_for_execution("git diff --stat") == (  # type: ignore[attr-defined]
        "git diff --no-ext-diff --no-textconv --stat"
    )


@pytest.mark.parametrize(
    "command",
    (
        "Remove-Item -Recurse -Force C:\\",
        "Remove-Item -Re -Fo C:\\",
        "Remove-Item -Recurse:$true -Force:$true C:\\",
        "ri -r -fo $env:USERPROFILE",
        "Remove-Item -Recurse -Force \\\\server\\share",
        "Clear-Content -LiteralPath C:\\",
        "Clear-Content *",
        "Clear-Content @('C:\\*','D:\\*')",
        "Clear-Content \\\\server\\share\\*",
        "Clear-Content .\\.\\*",
        "Clear-Content C:\\dir\\..\\*",
        "Clear-Content \\\\server\\share\\.\\*",
        "Clear-Content C:\\**",
        "Clear-Content ([array]('C:\\*'))",
        "Clear-Content $('C:\\*')",
        "Clear-Content FileSystem::C:\\*",
        "Clear-Content Microsoft.PowerShell.Core\\FileSystem::C:\\*",
        "Clear-Content ${env:USERPROFILE}\\*",
        "clc -LiteralPath:C:\\",
        "Set-Content C:\\* erased",
        "Set-Content -Path \\\\.\\PhysicalDrive0 value",
        "Add-Content -Path C:\\ value",
        "Out-File -FilePath \\\\.\\PhysicalDrive0",
        "Format-Volume -DriveLetter C",
        "Clear-Disk -Number 0 -RemoveData",
        "Remove-Partition -DiskNumber 0 -PartitionNumber 1",
        "git clean -fdx",
        "git reset --hard",
        "git checkout -f main",
        "git checkout -f -- :/",
        "git checkout -f HEAD -- :!does-not-exist",
        "git restore .",
        "git restore :/",
        "git restore :/**",
        "git restore :(top)",
        "git restore :(glob)**",
        "git restore :(top,glob,icase)**",
        "git restore :(exclude)does-not-exist",
        "git restore :!does-not-exist",
        "git restore --source HEAD :(exclude)does-not-exist",
        'git -c "alias.wipe=!git clean -fdx" wipe',
        "Stop-Computer -Force",
        "Restart-Computer -Force",
        "shutdown /r /t 0",
        ":(){ :|:& };:",
    ),
)
def test_powershell_and_shared_catastrophic_variants_are_structured(
    command: str,
) -> None:
    matches = catastrophic_matches(command)

    assert matches
    assert all(match.rule and match.evidence for match in matches)


def test_disabled_powershell_switch_does_not_create_catastrophic_false_positive() -> None:
    assert catastrophic_matches("Remove-Item -Recurse:$false -Force C:\\") == ()


_CATASTROPHIC_CONFIRM_COMMANDS = (
    "Remove-Item -Recurse -Force C:\\",
    "Remove-Item -Re -Fo C:\\",
    "Remove-Item -Recurse -Force C:\\,D:\\",
    "Remove-Item -Recurse -Force @('C:\\','D:\\')",
    "Clear-Content -LiteralPath C:\\",
    "Clear-Content *",
    "Clear-Content @('C:\\*','D:\\*')",
    "Clear-Content \\\\server\\share\\*",
    "Clear-Content .\\.\\*",
    "Clear-Content C:\\dir\\..\\*",
    "Clear-Content \\\\server\\share\\.\\*",
    "Clear-Content C:\\**",
    "Clear-Content ([array]('C:\\*'))",
    "Clear-Content $('C:\\*')",
    "Clear-Content FileSystem::C:\\*",
    "Clear-Content Microsoft.PowerShell.Core\\FileSystem::C:\\*",
    "Clear-Content ${env:USERPROFILE}\\*",
    "Set-Content C:\\* erased",
    "Set-Content -Path \\\\.\\PhysicalDrive0 value",
    "Format-Volume -DriveLetter C",
    "Clear-Disk -Number 0 -RemoveData",
    "Remove-Partition -DiskNumber 0 -PartitionNumber 1",
    "git clean -fdx",
    "git reset --hard",
    "git checkout -f main",
    "git checkout -f -- :/",
    "git checkout -f HEAD -- :!does-not-exist",
    "git restore .",
    "git restore :/",
    "git restore :/**",
    "git restore :(top)",
    "git restore :(glob)**",
    "git restore :(top,glob,icase)**",
    "git restore :(exclude)does-not-exist",
    "git restore :!does-not-exist",
    "git restore --source HEAD :(exclude)does-not-exist",
    "Restart-Computer -Force",
    "shutdown /r /t 0",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ("read-only", "workspace-write", "full-access"))
@pytest.mark.parametrize("command", _CATASTROPHIC_CONFIRM_COMMANDS)
async def test_catastrophic_powershell_calls_confirm_once_at_every_permission_level(
    tmp_path: Path,
    level: Literal["read-only", "workspace-write", "full-access"],
    command: str,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    requests: list[ConfirmationRequest] = []
    executions: list[str] = []

    matches = catastrophic_matches(command)
    assert matches
    assessment = ExecAssessment(
        syntax_confidence="high",
        syntax_uncertain=False,
        catastrophic_matches=matches,
    )

    class Host:
        resolved_shell = shell

        async def inspect(
            self,
            inspected_command: str,
            cwd: Path,
        ) -> ExecAssessment:
            del cwd
            assert inspected_command == command
            return assessment

        async def execute(
            self,
            executed_command: str,
            cwd: Path,
            timeout: int,
        ) -> ExecOutcome:
            del cwd, timeout
            executions.append(executed_command)
            return ExecOutcome(exit_code=0, stdout=b"unexpected", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("catastrophic commands must confirm before spawn")

    snapshot = PermissionSnapshot(level=level, exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=tmp_path,
        ),
    )

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id=f"catastrophic-{level}-{abs(hash(command))}",
            name="exec",
            arguments=json.dumps({"command": command}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_case", "status"),
    (
        ("unavailable", "uncertain"),
        ("crash", "failed"),
        ("nonzero-exit", "failed"),
        ("stderr", "failed"),
        ("timeout", "timeout"),
        ("inconsistent-output", "uncertain"),
    ),
)
@pytest.mark.parametrize("level", ("read-only", "workspace-write", "full-access"))
async def test_shell_present_inspector_uncertainty_confirms_at_every_level(
    tmp_path: Path,
    failure_case: str,
    status: str,
    level: Literal["read-only", "workspace-write", "full-access"],
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    requests: list[ConfirmationRequest] = []
    executions: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del command, cwd
            return ExecAssessment.uncertain_result(
                "fixture inspector failure",
                status=status,  # type: ignore[arg-type]
            )

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            executions.append(command)
            return ExecOutcome(exit_code=0, stdout=b"unexpected", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("uncertain inspection must confirm before spawn")

    snapshot = PermissionSnapshot(level=level, exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=tmp_path,
        ),
    )

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id=f"inspector-{failure_case}-{level}",
            name="exec",
            arguments=json.dumps({"command": "Get-Location"}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert executions == []


@pytest.mark.asyncio
async def test_unavailable_selected_powershell_is_a_hard_error_at_every_level(
    tmp_path: Path,
) -> None:
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({}),
        version_probe=_version({}),
        environment={"PATH": r"C:\safe"},
    )
    assert shell.available is False
    requests: list[ConfirmationRequest] = []

    for level in ("read-only", "workspace-write", "full-access"):
        snapshot = PermissionSnapshot(level=level, exec_shell=shell)
        gateway = ToolGateway._for_memory(
            (ExecTool(workspace=tmp_path, host=create_exec_host(shell)),),
            permission_context=PermissionContext.from_snapshot(
                snapshot,
                workspace_root=tmp_path,
            ),
        )

        async def unexpected_confirmation(
            request: ConfirmationRequest,
        ) -> Literal["approved"]:
            requests.append(request)
            return "approved"

        result = await gateway.call(
            ModelToolCall(
                id=f"unavailable-{level}",
                name="exec",
                arguments=json.dumps({"command": "Get-Location"}),
            ),
            confirmation=unexpected_confirmation,
        )
        assert result.status == "error"
        assert "capability" in result.content.lower()

    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ("Get-Location ;", "Get-Location \ud800", "Get-Location " + "x" * 1_048_577),
    ids=("malformed", "encoding", "large"),
)
async def test_powershell_inspection_boundaries_fail_closed_without_running_user_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: str,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    calls: list[tuple[str, ...]] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b'{"syntax_ok":false,"command_names":["Get-Location"]}', b""

        def kill(self) -> None:
            raise AssertionError("a completed parser process should not be killed")

        async def wait(self) -> int:
            return 0

    async def create_process(*argv: str, **kwargs: object) -> Process:
        del kwargs
        calls.append(argv)
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect(command, tmp_path)

    assert assessment.syntax_uncertain is True
    assert assessment.inspector_status == "uncertain"
    if "\ud800" in command or len(command) > 1_048_576:
        assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "resolution_count"),
    (
        ("alias", 1),
        ("function", 1),
        ("script", 1),
        ("shim", 1),
        ("workspace", 1),
        ("ambiguous", 2),
        ("unknown", 0),
        ("native", 1),
    ),
)
async def test_noncanonical_powershell_identity_categories_confirm(
    tmp_path: Path,
    kind: str,
    resolution_count: int,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    calls: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del cwd
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                git_delegation_safe=True,
                command_identities=(
                    ExecCommandIdentity(
                        requested="Get-Location",
                        canonical="Get-Location",
                        resolved="C:\\Tools\\Get-Location.exe",
                        module="Microsoft.PowerShell.Management",
                        resolution_count=resolution_count,
                        kind=kind,  # type: ignore[arg-type]
                    ),
                ),
            )

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            calls.append(command)
            return ExecOutcome(exit_code=0, stdout=b"unexpected", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("untrusted command identity must confirm before spawn")

    snapshot = PermissionSnapshot(level="read-only", exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(snapshot, workspace_root=tmp_path),
    )
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id=f"identity-{kind}",
            name="exec",
            arguments=json.dumps({"command": "Get-Location"}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        {"syntax_ok": True, "command_names": ["Get-Location"]},
        {
            "syntax_ok": True,
            "command_names": ["Get-Location"],
            "identities": [
                {
                    "requested": "Get-Item",
                    "canonical": "Get-Item",
                    "kind": "cmdlet",
                    "resolved": "Microsoft.PowerShell.Management",
                    "module": "Microsoft.PowerShell.Management",
                    "resolution_count": 1,
                }
            ],
        },
        {
            "syntax_ok": True,
            "command_names": ["Get-Location"],
            "identities": [
                {
                    "requested": "Get-Location",
                    "canonical": None,
                    "kind": "cmdlet",
                    "resolved": None,
                    "module": None,
                    "resolution_count": 1,
                }
            ],
        },
        {
            "syntax_ok": True,
            "command_names": ["Get-Location"],
            "identities": [
                {
                    "requested": "Get-Location",
                    "canonical": "Get-Location.exe",
                    "kind": "native",
                    "resolved": ".\\git.exe",
                    "module": None,
                    "resolution_count": 1,
                }
            ],
        },
    ),
    ids=(
        "missing-identities",
        "mismatched-identity",
        "inconsistent-cmdlet",
        "relative-native",
    ),
)
async def test_incomplete_or_inconsistent_identity_payload_is_typed_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return json.dumps(payload).encode("utf-8"), b""

        def kill(self) -> None:
            raise AssertionError("a completed inspector process should not be killed")

        async def wait(self) -> int:
            return 0

    async def create_process(*argv: str, **kwargs: object) -> Process:
        del argv, kwargs
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
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

    assert assessment.uncertain is True
    assert assessment.inspector_status == "uncertain"


@pytest.mark.asyncio
async def test_git_repository_execution_delegation_requires_confirmation_fact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    git_executable = r"C:\Program Files\Git\cmd\git.exe"
    payload = {
        "syntax_ok": True,
        "command_names": ["git"],
        "identities": [
            {
                "requested": "git",
                "canonical": "git.exe",
                "kind": "native",
                "resolved": git_executable,
                "module": git_executable,
                "resolution_count": 1,
            }
        ],
    }
    responses = [
        (0, json.dumps(payload).encode("utf-8"), b""),
        (0, b"filter.untrusted.process\n", b""),
    ]
    calls: list[tuple[str, ...]] = []

    class Process:
        def __init__(self, response: tuple[int, bytes, bytes]) -> None:
            self.returncode, self._stdout, self._stderr = response

        async def communicate(self) -> tuple[bytes, bytes]:
            return self._stdout, self._stderr

        def kill(self) -> None:
            raise AssertionError("a completed Git audit should not be killed")

        async def wait(self) -> int:
            return self.returncode

    async def create_process(*argv: str, **kwargs: object) -> Process:
        del kwargs
        calls.append(argv)
        return Process(responses.pop(0))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect("git status", tmp_path)
    assert assessment.git_delegation_safe is None
    assert len(calls) == 1
    assessment = await host.audit_git_delegation(
        "git status",
        tmp_path,
        tmp_path,
        assessment,
    )

    assert assessment.syntax_uncertain is False
    assert assessment.git_delegation_safe is False
    assert len(calls) == 2
    assert calls[1][0] == git_executable
    assert calls[1][1:4] == ("-C", str(tmp_path), "config")
    assert "--no-includes" in calls[1]


@pytest.mark.asyncio
async def test_git_repository_audit_error_is_typed_inspector_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    git_executable = r"C:\Program Files\Git\cmd\git.exe"
    payload = {
        "syntax_ok": True,
        "command_names": ["git"],
        "identities": [
            {
                "requested": "git",
                "canonical": "git.exe",
                "kind": "native",
                "resolved": git_executable,
                "module": git_executable,
                "resolution_count": 1,
            }
        ],
    }
    responses = [
        (0, json.dumps(payload).encode("utf-8"), b""),
        (128, b"", b"fatal: bad config"),
    ]

    class Process:
        def __init__(self, response: tuple[int, bytes, bytes]) -> None:
            self.returncode, self._stdout, self._stderr = response

        async def communicate(self) -> tuple[bytes, bytes]:
            return self._stdout, self._stderr

        def kill(self) -> None:
            raise AssertionError("a completed Git audit should not be killed")

        async def wait(self) -> int:
            return self.returncode

    async def create_process(*argv: str, **kwargs: object) -> Process:
        del argv, kwargs
        return Process(responses.pop(0))

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )

    assessment = await host.inspect("git status", tmp_path)
    assessment = await host.audit_git_delegation(
        "git status",
        tmp_path,
        tmp_path,
        assessment,
    )

    assert assessment.uncertain is True
    assert assessment.inspector_status == "failed"


@pytest.mark.asyncio
async def test_git_audit_never_launches_workspace_identity_or_touches_external_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    cases = (
        ("git status", str(tmp_path / "bin" / "git.exe")),
        (r"git -C \\attacker\share status", r"C:\Program Files\Git\cmd\git.exe"),
    )

    for command, git_executable in cases:
        payload = {
            "syntax_ok": True,
            "command_names": ["git"],
            "identities": [
                {
                    "requested": "git",
                    "canonical": "git.exe",
                    "kind": "native",
                    "resolved": git_executable,
                    "module": git_executable,
                    "resolution_count": 1,
                }
            ],
        }
        calls: list[tuple[str, ...]] = []

        class Process:
            returncode = 0

            def __init__(self, response_payload: dict[str, object]) -> None:
                self._response_payload = response_payload

            async def communicate(self) -> tuple[bytes, bytes]:
                return json.dumps(self._response_payload).encode("utf-8"), b""

            def kill(self) -> None:
                raise AssertionError("a completed inspector should not be killed")

            async def wait(self) -> int:
                return 0

        async def create_process(
            *argv: str,
            call_log: list[tuple[str, ...]] = calls,
            response_payload: dict[str, object] = payload,
            **kwargs: object,
        ) -> Process:
            del kwargs
            call_log.append(argv)
            return Process(response_payload)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
        host = PowerShellExecHost(
            resolve_exec_shell(
                "pwsh",
                platform="windows",
                which=_which({"pwsh": executable}),
                version_probe=_version({executable: (7, 5)}),
                environment={"PATH": r"C:\safe"},
            )
        )

        assessment = await host.inspect(command, tmp_path)
        assessment = await host.audit_git_delegation(
            command,
            tmp_path,
            tmp_path,
            assessment,
        )

        assert assessment.git_delegation_safe is None
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_git_execution_uses_the_assessed_canonical_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    git_executable = r"C:\Program Files\Git\cmd\git.exe"
    calls: list[tuple[str, ...]] = []

    class Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"diff", b""

        def kill(self) -> None:
            raise AssertionError("a completed Git process should not be killed")

        async def wait(self) -> int:
            return 0

    async def create_process(*argv: str, **kwargs: object) -> Process:
        del kwargs
        calls.append(argv)
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    host = PowerShellExecHost(
        resolve_exec_shell(
            "pwsh",
            platform="windows",
            which=_which({"pwsh": executable}),
            version_probe=_version({executable: (7, 5)}),
            environment={"PATH": r"C:\safe"},
        )
    )
    assessment = ExecAssessment(
        syntax_confidence="high",
        syntax_uncertain=False,
        git_delegation_safe=True,
        command_identities=(
            ExecCommandIdentity(
                requested="git",
                canonical="git.exe",
                resolved=git_executable,
                resolution_count=1,
                kind="native",
            ),
        ),
    )

    outcome = await host.execute_assessed(
        "git diff --stat",
        tmp_path,
        5,
        assessment=assessment,
    )

    assert outcome.exit_code == 0
    assert len(calls) == 1
    execution_script = calls[0][-1]
    assert f"& '{git_executable}' diff --no-ext-diff --no-textconv --stat" in execution_script


@pytest.mark.asyncio
async def test_workspace_git_executable_requires_confirmation(
    tmp_path: Path,
) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    git_path = tmp_path / "bin" / "git.exe"
    requests: list[ConfirmationRequest] = []
    calls: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del cwd
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(
                    ExecCommandIdentity(
                        requested="git",
                        canonical="git.exe",
                        resolved=str(git_path),
                        resolution_count=1,
                        kind="native",
                    ),
                ),
            )

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            calls.append(command)
            return ExecOutcome(exit_code=0, stdout=b"unexpected", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("a Workspace Git executable must confirm before spawn")

    snapshot = PermissionSnapshot(level="read-only", exec_shell=shell)
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(snapshot, workspace_root=tmp_path),
    )

    async def decline(request: ConfirmationRequest) -> Literal["declined"]:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id="workspace-git",
            name="exec",
            arguments=json.dumps({"command": "git status"}),
        ),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert calls == []


@pytest.mark.asyncio
async def test_schedule_exec_keeps_legacy_authorization_behavior(tmp_path: Path) -> None:
    executable = r"C:\PowerShell\pwsh.exe"
    shell = resolve_exec_shell(
        "pwsh",
        platform="windows",
        which=_which({"pwsh": executable}),
        version_probe=_version({executable: (7, 5)}),
        environment={"PATH": r"C:\safe"},
    )
    executions: list[str] = []

    class Host:
        resolved_shell = shell

        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del cwd
            return ExecAssessment(
                syntax_confidence="high",
                syntax_uncertain=False,
                command_identities=(ExecCommandIdentity(requested=command, kind="unknown"),),
            )

        async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
            del cwd, timeout
            executions.append(command)
            return ExecOutcome(exit_code=0, stdout=b"ok", stderr=b"")

        def process_spec(self, cwd: Path) -> ExecProcessSpec:
            del cwd
            raise AssertionError("the schedule fixture must use Host methods")

    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=Host()),),
        permission_context=PermissionContext.from_snapshot(
            PermissionSnapshot(level="read-only", exec_shell=shell),
            workspace_root=tmp_path,
            origin="schedule",
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id="schedule-legacy-exec",
            name="exec",
            arguments=json.dumps({"command": "Get-Date"}),
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert executions == ["Get-Date"]


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ("pwsh", "powershell"))
async def test_real_powershell_host_inspects_and_executes_canonical_cmdlet(
    tmp_path: Path,
    selector: str,
) -> None:
    shell = resolve_exec_shell(selector, platform="windows")  # type: ignore[arg-type]
    if not shell.available:
        pytest.skip(f"{selector} is not installed")
    host = PowerShellExecHost(shell)

    assessment = await host.inspect("Get-Location", tmp_path)
    dynamic_assessment = await host.inspect("& { Get-Location }", tmp_path)
    outcome = await host.execute_assessed(
        "Get-Location",
        tmp_path,
        10,
        assessment=assessment,
    )
    gateway = ToolGateway._for_memory(
        (ExecTool(workspace=tmp_path, host=host),),
        permission_context=PermissionContext.from_snapshot(
            PermissionSnapshot(level="full-access", exec_shell=shell),
            workspace_root=tmp_path,
        ),
    )
    dynamic_result = await gateway.call(
        ModelToolCall(
            id=f"real-dynamic-{selector}",
            name="exec",
            arguments=json.dumps({"command": "& { Get-Location }"}),
        )
    )

    assert assessment.syntax_uncertain is False
    assert assessment.command_identities == (
        ExecCommandIdentity(
            requested="Get-Location",
            canonical="Get-Location",
            resolved="Microsoft.PowerShell.Management",
            module="Microsoft.PowerShell.Management",
            resolution_count=1,
            kind="cmdlet",
        ),
    )
    assert dynamic_assessment.syntax_uncertain is False
    assert dynamic_assessment.inspector_status == "available"
    assert dynamic_result.status == "success"
    assert outcome.exit_code == 0
    assert outcome.timed_out is False


@pytest.mark.asyncio
async def test_real_pwsh_inspection_does_not_autoload_an_untrusted_module(
    tmp_path: Path,
) -> None:
    shell = resolve_exec_shell("pwsh", platform="windows")
    if not shell.available:
        pytest.skip("pwsh is not installed")
    module_root = tmp_path / "modules"
    module_dir = module_root / "UntrustedModule"
    module_dir.mkdir(parents=True)
    sentinel = tmp_path / "module-loaded.txt"
    escaped_sentinel = str(sentinel).replace("'", "''")
    (module_dir / "UntrustedModule.psm1").write_text(
        f"[IO.File]::WriteAllText('{escaped_sentinel}', 'loaded')\n"
        "function Invoke-UntrustedCommand {}\n"
        "Export-ModuleMember -Function Invoke-UntrustedCommand\n",
        encoding="utf-8",
    )
    (module_dir / "UntrustedModule.psd1").write_text(
        "@{\n"
        "RootModule = 'UntrustedModule.psm1'\n"
        "ModuleVersion = '1.0.0'\n"
        "FunctionsToExport = @('Invoke-UntrustedCommand')\n"
        "}\n",
        encoding="utf-8",
    )
    environment = shell.env
    environment["PSModulePath"] = str(module_root)
    host = PowerShellExecHost(replace(shell, environment=tuple(environment.items())))

    assessment = await host.inspect("Invoke-UntrustedCommand", tmp_path)

    assert assessment.syntax_uncertain is False
    assert assessment.command_identities[0].kind == "unknown"
    assert assessment.command_identities[0].resolution_count == 0
    assert not sentinel.exists()
