from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Literal

import pytest

from myclaw.permission.policy import PermissionContext, PermissionSnapshot
from myclaw.tools.exec.host import (
    BashExecHost,
    ExecProcessSpec,
    ResolvedExecShell,
    resolve_exec_shell,
)
from myclaw.tools.exec.policy import (
    CatastrophicMatch,
    ExecAssessment,
    ExecCommandIdentity,
    ExecDynamicConstruct,
    ExecOutcome,
    ExecPathAccess,
)
from myclaw.tools.exec.tool import ExecTool
from myclaw.tools.gateway import ConfirmationRequest, ModelToolCall, ToolGateway


def _shell() -> ResolvedExecShell:
    return resolve_exec_shell(
        "auto",
        platform="posix",
        which=lambda name: "/usr/bin/bash" if name == "bash" else None,
        environment={"PATH": "/usr/bin", "HOME": "/home/test"},
    )


def _call(command: str, *, cwd: str = ".", timeout: int = 7) -> ModelToolCall:
    return ModelToolCall(
        id=f"bash-{abs(hash(command))}",
        name="exec",
        arguments=json.dumps({"command": command, "cwd": cwd, "timeout": timeout}),
    )


class _Host:
    def __init__(self, assessment: ExecAssessment) -> None:
        self.resolved_shell = _shell()
        self.assessment = assessment
        self.executions: list[tuple[str, Path, int]] = []

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        del command, cwd
        return self.assessment

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
        self.executions.append((command, cwd, timeout))
        return ExecOutcome(exit_code=0, stdout=b"ok", stderr=b"")

    def process_spec(self, cwd: Path) -> ExecProcessSpec:
        del cwd
        raise AssertionError("the policy fixture must use Host methods")


class _RecordingBashHost(BashExecHost):
    def __init__(self, resolved_shell: ResolvedExecShell) -> None:
        super().__init__(resolved_shell)
        self.executions: list[tuple[tuple[str, ...], Path, int, dict[str, str]]] = []

    async def _run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout: int,
        environment: dict[str, str] | None = None,
    ) -> ExecOutcome:
        if "config" in argv and argv[0] != self.resolved_shell.executable:
            return ExecOutcome(exit_code=1, stdout=b"", stderr=b"")
        self.executions.append(
            (argv, cwd, timeout, {} if environment is None else environment)
        )
        return ExecOutcome(exit_code=0, stdout=b"synthetic", stderr=b"")


def _synthetic_bash_host(
    tmp_path: Path,
    command_names: set[str],
    *,
    path_entries: tuple[Path, ...] | None = None,
) -> tuple[Path, _RecordingBashHost]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in command_names:
        executable = bin_dir / name
        executable.write_bytes(b"\x7fELF")
        executable.chmod(0o755)
    entries = (bin_dir,) if path_entries is None else path_entries
    shell = ResolvedExecShell(
        selector="auto",
        platform="posix",
        family="bash",
        executable="/usr/bin/bash",
        flags=("--noprofile", "--norc"),
        environment=(("PATH", os.pathsep.join(str(entry) for entry in entries)),),
        available=True,
    )
    return workspace, _RecordingBashHost(shell)


def _gateway(
    workspace: Path,
    host: _Host | BashExecHost,
    *,
    level: Literal["read-only", "workspace-write", "full-access"],
) -> ToolGateway:
    snapshot = PermissionSnapshot(level=level, exec_shell=host.resolved_shell)
    return ToolGateway._for_memory(
        (ExecTool(workspace=workspace, host=host),),
        permission_context=PermissionContext.from_snapshot(
            snapshot,
            workspace_root=workspace,
        ),
    )


async def _decline(requests: list[ConfirmationRequest], request: ConfirmationRequest) -> Literal["declined"]:
    requests.append(request)
    return "declined"


def _identity(
    requested: str,
    *,
    kind: Literal[
        "builtin",
        "native",
        "script",
        "shim",
        "workspace",
        "ambiguous",
        "unknown",
        "alias",
        "function",
    ] = "native",
    resolved: str | None = None,
    canonical: str | None = None,
    resolution_count: int | None = 1,
) -> ExecCommandIdentity:
    return ExecCommandIdentity(
        requested=requested,
        canonical=canonical if canonical is not None else requested,
        resolved=(
            f"/usr/bin/{requested}"
            if resolved is None and kind == "native"
            else resolved
        ),
        resolution_count=resolution_count,
        kind=kind,
    )


def _assessment(
    *identities: ExecCommandIdentity,
    file_accesses: tuple[ExecPathAccess, ...] = (),
    dynamic_constructs: tuple[ExecDynamicConstruct, ...] = (),
    catastrophic_matches: tuple[CatastrophicMatch, ...] = (),
    git_delegation_safe: bool | None = None,
) -> ExecAssessment:
    return ExecAssessment(
        syntax_confidence="high",
        syntax_uncertain=False,
        command_identities=identities,
        file_accesses=file_accesses,
        dynamic_constructs=dynamic_constructs,
        catastrophic_matches=catastrophic_matches,
        git_delegation_safe=git_delegation_safe,
    )


@pytest.mark.asyncio
async def test_bash_unlisted_command_requires_one_confirmation_and_never_executes(
    tmp_path: Path,
) -> None:
    host = _Host(
        ExecAssessment(
            syntax_confidence="high",
            syntax_uncertain=False,
            command_identities=(
                _identity("printf", kind="builtin"),
            ),
        )
    )
    gateway = _gateway(tmp_path, host, level="read-only")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(_call("printf hello"), confirmation=lambda request: _decline(requests, request))

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details == {
        "command": "printf hello",
        "cwd": str(tmp_path.resolve()),
        "timeout": 7,
    }
    assert host.executions == []


@pytest.mark.asyncio
async def test_bash_dynamic_construct_requires_one_confirmation(
    tmp_path: Path,
) -> None:
    command = "value=$(pwd); cat \"$value\""
    host = _Host(
        ExecAssessment(
            syntax_confidence="high",
            syntax_uncertain=False,
            command_identities=(_identity("cat"),),
            dynamic_constructs=(
                ExecDynamicConstruct("assignment", "variable assignment"),
                ExecDynamicConstruct("substitution", "command or process substitution"),
            ),
        )
    )
    gateway = _gateway(tmp_path, host, level="workspace-write")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(_call(command), confirmation=lambda request: _decline(requests, request))

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    ("alias", "function", "script", "shim", "workspace", "ambiguous", "unknown"),
)
async def test_bash_untrusted_identity_requires_one_confirmation(
    tmp_path: Path,
    kind: Literal["alias", "function", "script", "shim", "workspace", "ambiguous", "unknown"],
) -> None:
    command = "cat ./inside.txt"
    host = _Host(
        ExecAssessment(
            syntax_confidence="high",
            syntax_uncertain=False,
            command_identities=(
                _identity(
                    "cat",
                    kind=kind,
                    resolved="/workspace/bin/cat" if kind not in {"ambiguous", "unknown"} else None,
                    canonical="cat" if kind not in {"ambiguous", "unknown"} else None,
                    resolution_count=2 if kind == "ambiguous" else 0 if kind == "unknown" else 1,
                ),
            ),
        )
    )
    (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    gateway = _gateway(tmp_path, host, level="read-only")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(_call(command), confirmation=lambda request: _decline(requests, request))

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "role", "path", "expected_confirmation"),
    (
        ("read-only", "read", "../outside.txt", True),
        ("workspace-write", "read", "../outside.txt", True),
        ("full-access", "read", "../outside.txt", False),
        ("read-only", "write", "./new.txt", True),
        ("workspace-write", "write", "./new.txt", False),
        ("workspace-write", "write", "../new.txt", True),
    ),
)
async def test_bash_static_path_roles_follow_level_and_workspace_containment(
    tmp_path: Path,
    level: Literal["read-only", "workspace-write", "full-access"],
    role: Literal["read", "write"],
    path: str,
    expected_confirmation: bool,
) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    if role == "read":
        (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    command = "cat " + path if role == "read" else "touch " + path
    host = _Host(
        ExecAssessment(
            syntax_confidence="high",
            syntax_uncertain=False,
            command_identities=(_identity(command.split(maxsplit=1)[0]),),
            file_accesses=(ExecPathAccess(path=path, role=role),),
        )
    )
    gateway = _gateway(tmp_path, host, level=level)
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(_call(command), confirmation=lambda request: _decline(requests, request))

    assert len(requests) == int(expected_confirmation)
    assert result.status == ("refused" if expected_confirmation else "success")
    assert len(host.executions) == int(not expected_confirmation)


@pytest.mark.asyncio
async def test_bash_full_access_runs_parseable_dynamic_noncatastrophic_command(
    tmp_path: Path,
) -> None:
    command = "value=$(pwd); echo \"$value\""
    host = _Host(
        ExecAssessment(
            syntax_confidence="high",
            syntax_uncertain=False,
            command_identities=(_identity("echo"),),
            dynamic_constructs=(ExecDynamicConstruct("substitution", "command substitution"),),
        )
    )
    gateway = _gateway(tmp_path, host, level="full-access")

    result = await gateway.call(_call(command))

    assert result.status == "success"
    assert result.confirmation is None
    assert host.executions == [(command, tmp_path.resolve(), 7)]


@pytest.mark.asyncio
async def test_bash_catastrophic_operation_confirms_at_full_access(
    tmp_path: Path,
) -> None:
    command = "rm -rf /"
    host = _Host(
        ExecAssessment(
            syntax_confidence="high",
            syntax_uncertain=False,
            command_identities=(_identity("rm", kind="native", resolved="/usr/bin/rm"),),
            catastrophic_matches=(CatastrophicMatch("broad-recursive-force-delete", command),),
        )
    )
    gateway = _gateway(tmp_path, host, level="full-access")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(_call(command), confirmation=lambda request: _decline(requests, request))

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
async def test_bash_inspector_resolves_builtin_and_marks_dynamic_ast(tmp_path: Path) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    shell = resolve_exec_shell(
        "auto",
        platform="posix",
        which=lambda name: "/usr/bin/bash" if name == "bash" else None,
        environment={"PATH": str(empty_bin)},
    )
    host = BashExecHost(shell)

    assessment = await host.inspect("value=$(pwd); cat \"$value\"", tmp_path)

    assert assessment.syntax_uncertain is False
    identities = {
        identity.requested: identity for identity in assessment.command_identities
    }
    assert identities["pwd"].kind == "builtin"
    assert identities["cat"].kind == "unknown"
    assert {construct.kind for construct in assessment.dynamic_constructs} >= {
        "assignment",
        "substitution",
        "variable",
    }


@pytest.mark.asyncio
async def test_bash_inspector_rejects_invalid_encoding_without_spawning(tmp_path: Path) -> None:
    host = BashExecHost(_shell())

    assessment = await host.inspect("pwd \ud800", tmp_path)

    assert assessment.uncertain is True
    assert assessment.inspector_status == "uncertain"


def test_bash_git_execution_is_hardened_and_uses_assessed_executable() -> None:
    host = BashExecHost(_shell())

    environment = host.environment_for_command("git diff --stat")
    command = host.command_for_execution(
        "git diff --stat",
        git_executable="/usr/bin/git",
    )

    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_PAGER"] == "cat"
    assert environment["GIT_EXTERNAL_DIFF"] == ""
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_CONFIG_VALUE_1"] == "false"
    assert environment["GIT_CONFIG_VALUE_3"] == "/dev/null"
    assert command == "/usr/bin/git diff --no-ext-diff --no-textconv --stat"


@pytest.mark.asyncio
async def test_bash_inspector_failure_confirms_at_full_access(tmp_path: Path) -> None:
    class FailingHost(_Host):
        async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
            del command, cwd
            return ExecAssessment.uncertain_result("fixture failure", status="failed")

    host = FailingHost(ExecAssessment.uncertain_result("unused", status="failed"))
    gateway = _gateway(tmp_path, host, level="full-access")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(
        _call("pwd"),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "command_name"),
    (
        ("pwd", "pwd"),
        ("ls .", "ls"),
        ("cat ./inside.txt", "cat"),
        ("head -n 1 ./inside.txt", "head"),
        ("tail -n 1 ./inside.txt", "tail"),
        ("wc -l ./inside.txt", "wc"),
        ("stat ./inside.txt", "stat"),
        ("file ./inside.txt", "file"),
        ("grep needle ./inside.txt", "grep"),
        ("rg needle ./inside.txt", "rg"),
        ("find . -type f -name '*.txt' -print", "find"),
        ("sort ./inside.txt", "sort"),
        ("uniq ./inside.txt", "uniq"),
        ("cut -f 1 ./inside.txt", "cut"),
        ("diff ./inside.txt ./other.txt", "diff"),
    ),
)
async def test_bash_fixed_read_candidates_run_inside_workspace(
    tmp_path: Path,
    command: str,
    command_name: str,
) -> None:
    (tmp_path / "inside.txt").write_text("needle\nneedle\n", encoding="utf-8")
    (tmp_path / "other.txt").write_text("other\n", encoding="utf-8")
    identity = (
        _identity("pwd", kind="builtin", resolved=None)
        if command_name == "pwd"
        else _identity(command_name)
    )
    host = _Host(_assessment(identity))
    gateway = _gateway(tmp_path, host, level="read-only")

    result = await gateway.call(_call(command))

    assert result.status == "success"
    assert host.executions == [(command, tmp_path.resolve(), 7)]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ("mkdir ./new-dir", "touch ./new.txt", "rm ./inside.txt"))
async def test_bash_workspace_write_candidates_run_in_workspace(
    tmp_path: Path,
    command: str,
) -> None:
    (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    command_name = command.split(maxsplit=1)[0]
    host = _Host(_assessment(_identity(command_name)))
    gateway = _gateway(tmp_path, host, level="workspace-write")

    result = await gateway.call(_call(command))

    assert result.status == "success"
    assert host.executions == [(command, tmp_path.resolve(), 7)]


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ("cp ./inside.txt ./copy.txt", "mv ./inside.txt ./moved.txt"))
async def test_bash_copy_and_move_classify_source_and_destination_roles(
    tmp_path: Path,
    command: str,
) -> None:
    (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    command_name = command.split(maxsplit=1)[0]
    host = _Host(_assessment(_identity(command_name)))
    gateway = _gateway(tmp_path, host, level="workspace-write")

    result = await gateway.call(_call(command))

    assert result.status == "success"
    assert host.executions == [(command, tmp_path.resolve(), 7)]


@pytest.mark.asyncio
async def test_bash_write_candidate_requires_confirmation_in_read_only_mode(
    tmp_path: Path,
) -> None:
    command = "touch ./new.txt"
    host = _Host(_assessment(_identity("touch")))
    gateway = _gateway(tmp_path, host, level="read-only")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "git status",
        "git diff --stat",
        "git log --oneline",
        "git show --stat",
        "git branch --list",
        "git rev-parse --show-toplevel",
        "git ls-files --cached",
        "git -C . status",
    ),
)
async def test_bash_fixed_git_read_forms_run_when_delegation_is_safe(
    tmp_path: Path,
    command: str,
) -> None:
    (tmp_path / ".git").mkdir()
    host = _Host(
        _assessment(
            _identity("git"),
            git_delegation_safe=True,
        )
    )
    gateway = _gateway(tmp_path, host, level="read-only")

    result = await gateway.call(_call(command))

    assert result.status == "success"
    assert host.executions == [(command, tmp_path.resolve(), 7)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "git clean -fd",
        "git branch",
        "git diff --ext-diff",
        "git status --config-env=x=y",
        "git show ':/needle'",
        "git status -- ':(exclude)inside.txt'",
        "git status -- ':!inside.txt'",
    ),
)
async def test_bash_non_read_git_forms_require_confirmation(
    tmp_path: Path,
    command: str,
) -> None:
    host = _Host(_assessment(_identity("git"), git_delegation_safe=True))
    gateway = _gateway(tmp_path, host, level="read-only")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
async def test_bash_simple_pipeline_retains_each_command_identity(
    tmp_path: Path,
) -> None:
    (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    command = "cat ./inside.txt | sort"
    host = _Host(_assessment(_identity("cat"), _identity("sort")))
    gateway = _gateway(tmp_path, host, level="read-only")

    result = await gateway.call(_call(command))

    assert result.status == "success"
    assert host.executions == [(command, tmp_path.resolve(), 7)]


@pytest.mark.asyncio
async def test_bash_inspector_retains_repeated_pipeline_commands(tmp_path: Path) -> None:
    host = BashExecHost(_shell())

    assessment = await host.inspect("pwd | pwd", tmp_path)

    assert [identity.requested for identity in assessment.command_identities] == [
        "pwd",
        "pwd",
    ]
    assert assessment.file_accesses == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected_reason"),
    (
        ("cp ../outside.txt ./copy.txt", "outside"),
        ("cp ./inside.txt ../copy.txt", "outside"),
        ("mv ../outside.txt ./moved.txt", "outside"),
        ("mv ./inside.txt ../moved.txt", "outside"),
    ),
)
async def test_bash_copy_and_move_external_roles_require_confirmation(
    tmp_path: Path,
    command: str,
    expected_reason: str,
) -> None:
    (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    (tmp_path.parent / "outside.txt").write_text("outside", encoding="utf-8")
    command_name = command.split(maxsplit=1)[0]
    host = _Host(_assessment(_identity(command_name)))
    gateway = _gateway(tmp_path, host, level="workspace-write")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert expected_reason in requests[0].reason.lower()
    assert host.executions == []


@pytest.mark.asyncio
async def test_bash_execute_assessed_uses_assessed_git_executable_and_hardening(
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], Path, dict[str, str]]] = []

    class Host(BashExecHost):
        async def _run(
            self,
            argv: tuple[str, ...],
            *,
            cwd: Path,
            timeout: int,
            environment: dict[str, str] | None = None,
        ) -> ExecOutcome:
            del timeout
            calls.append((argv, cwd, {} if environment is None else environment))
            return ExecOutcome(exit_code=0, stdout=b"git", stderr=b"")

    host = Host(_shell())
    assessment = _assessment(
        _identity("git", resolved="/opt/git/bin/git"),
        git_delegation_safe=True,
    )

    result = await host.execute_assessed(
        "git diff --stat",
        tmp_path,
        5,
        assessment=assessment,
    )

    assert result.stdout == b"git"
    assert calls[0][0][-1] == "/opt/git/bin/git diff --no-ext-diff --no-textconv --stat"
    assert calls[0][1] == tmp_path
    assert calls[0][2]["GIT_CONFIG_NOSYSTEM"] == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "identity", "construct"),
    (
        ("value=1; cat ./inside.txt", "cat", "assignment"),
        ("cat \"$(pwd)\"", "cat", "substitution"),
        ("cat \"$HOME/file\"", "cat", "variable"),
        ("cat *.txt", "cat", "glob"),
        ("cat <(pwd)", "cat", "substitution"),
        ("cat ./inside.txt > ./out.txt", "cat", "redirection"),
        ("function f { cat ./inside.txt; }", "cat", "control-flow"),
    ),
)
async def test_bash_dynamic_constructs_require_confirmation_below_full_access(
    tmp_path: Path,
    command: str,
    identity: str,
    construct: str,
) -> None:
    host = _Host(
        _assessment(
            _identity(identity),
            dynamic_constructs=(ExecDynamicConstruct(construct, construct),),
        )
    )
    gateway = _gateway(tmp_path, host, level="workspace-write")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "value=$(pwd); cat ./inside.txt",
        "cat ./inside.txt > ./out.txt",
        "for item in a; do cat ./inside.txt; done",
    ),
)
async def test_bash_full_access_runs_parseable_noncatastrophic_dynamic_commands(
    tmp_path: Path,
    command: str,
) -> None:
    host = _Host(
        _assessment(
            _identity("cat"),
            dynamic_constructs=(ExecDynamicConstruct("dynamic", "dynamic"),),
        )
    )
    gateway = _gateway(tmp_path, host, level="full-access")

    result = await gateway.call(_call(command))

    assert result.status == "success"
    assert result.confirmation is None
    assert host.executions == [(command, tmp_path.resolve(), 7)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "rm -rf /",
        "git reset --hard",
        ":(){ :|:&};:",
    ),
)
async def test_bash_catastrophic_forms_confirm_even_in_full_access(
    tmp_path: Path,
    command: str,
) -> None:
    host = _Host(
        _assessment(
            _identity(command.split(maxsplit=1)[0]),
            catastrophic_matches=(CatastrophicMatch("catastrophic", command),),
        )
    )
    gateway = _gateway(tmp_path, host, level="full-access")
    requests: list[ConfirmationRequest] = []

    result = await gateway.call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "command_name"),
    (
        ("pwd", "pwd"),
        ("ls .", "ls"),
        ("cat ./inside.txt", "cat"),
        ("head -n 1 ./inside.txt", "head"),
        ("tail -n 1 ./inside.txt", "tail"),
        ("wc -l ./inside.txt", "wc"),
        ("stat ./inside.txt", "stat"),
        ("file ./inside.txt", "file"),
        ("grep needle ./inside.txt", "grep"),
        ("rg needle ./inside.txt", "rg"),
        ("find . -type f -name '*.txt' -print", "find"),
        ("sort ./inside.txt", "sort"),
        ("uniq ./inside.txt", "uniq"),
        ("cut -f 1 ./inside.txt", "cut"),
        ("diff ./inside.txt ./other.txt", "diff"),
    ),
)
async def test_bash_read_candidates_cross_real_inspection_and_policy(
    tmp_path: Path,
    command: str,
    command_name: str,
) -> None:
    names = set() if command_name == "pwd" else {command_name}
    workspace, host = _synthetic_bash_host(tmp_path, names)
    (workspace / "inside.txt").write_text("needle\n", encoding="utf-8")
    (workspace / "other.txt").write_text("other\n", encoding="utf-8")

    result = await _gateway(workspace, host, level="read-only").call(_call(command))

    assert result.status == "success"
    assert result.confirmation is None
    assert len(host.executions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "command_name"),
    (
        ("mkdir ./new-dir", "mkdir"),
        ("touch ./new.txt", "touch"),
        ("cp ./inside.txt ./copy.txt", "cp"),
        ("mv ./inside.txt ./moved.txt", "mv"),
        ("rm ./inside.txt", "rm"),
    ),
)
async def test_bash_write_candidates_cross_real_inspection_and_policy(
    tmp_path: Path,
    command: str,
    command_name: str,
) -> None:
    workspace, host = _synthetic_bash_host(tmp_path, {command_name})
    (workspace / "inside.txt").write_text("inside\n", encoding="utf-8")

    result = await _gateway(workspace, host, level="workspace-write").call(_call(command))

    assert result.status == "success"
    assert result.confirmation is None
    assert len(host.executions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "git status",
        "git diff --stat",
        "git log --oneline",
        "git show HEAD",
        "git branch --list",
        "git rev-parse --show-toplevel",
        "git ls-files --cached",
    ),
)
async def test_bash_git_forms_cross_real_inspection_audit_and_policy(
    tmp_path: Path,
    command: str,
) -> None:
    workspace, host = _synthetic_bash_host(tmp_path, {"git"})
    (workspace / ".git").mkdir()

    result = await _gateway(workspace, host, level="read-only").call(_call(command))

    assert result.status == "success"
    assert result.confirmation is None
    assert len(host.executions) == 1
    assert host.executions[0][1] == workspace.resolve()
    assert host.executions[0][3]["GIT_CONFIG_NOSYSTEM"] == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "command"),
    (
        ("workspace-write", "rg --files ../outside.txt"),
        ("workspace-write", "cat ./inside.txt | grep -e needle ../outside.txt"),
        ("workspace-write", "cat ./inside.txt | rg -e needle ../outside.txt"),
        ("workspace-write", "ls --color ../outside.txt"),
        ("workspace-write", "touch --reference ../outside.txt ./target.txt"),
        ("read-only", "uniq ./inside.txt ./output.txt"),
        ("workspace-write", "tail --follow ./inside.txt"),
        ("workspace-write", "grep -R needle ."),
        ("workspace-write", "rg -L needle ."),
        ("workspace-write", "sort --random-source ../outside.txt ./inside.txt"),
        ("workspace-write", "file --magic-file ../outside.txt ./inside.txt"),
        ("workspace-write", "diff --from-file ../outside.txt ./inside.txt"),
        ("workspace-write", "cat $'../outside.txt'"),
        ("workspace-write", 'cat $"../outside.txt"'),
        ("workspace-write", "cat {./inside.txt,../outside.txt}"),
    ),
)
async def test_bash_real_inspection_closes_grammar_and_path_bypasses(
    tmp_path: Path,
    level: Literal["read-only", "workspace-write"],
    command: str,
) -> None:
    workspace, host = _synthetic_bash_host(
        tmp_path,
        {"cat", "diff", "file", "grep", "ls", "rg", "sort", "tail", "touch", "uniq"},
    )
    (workspace / "inside.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")
    requests: list[ConfirmationRequest] = []

    result = await _gateway(workspace, host, level=level).call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details == {
        "command": command,
        "cwd": str(workspace.resolve()),
        "timeout": 7,
    }
    assert host.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ("cat 'file[1]'", r"cat file\[1\]", "cat '$HOME'"))
async def test_bash_static_quoted_or_escaped_literals_run_directly(
    tmp_path: Path,
    command: str,
) -> None:
    workspace, host = _synthetic_bash_host(tmp_path, {"cat"})
    for name in ("file[1]", "$HOME"):
        (workspace / name).write_text("inside\n", encoding="utf-8")

    result = await _gateway(workspace, host, level="read-only").call(_call(command))

    assert result.status == "success"
    assert result.confirmation is None
    assert len(host.executions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "ls --color=always .",
        "grep --color=always needle ./inside.txt",
        "sort --batch-size 2 ./inside.txt",
        "touch --reference ./inside.txt ./target.txt",
        "uniq ./inside.txt ./output.txt",
        "cat -- ./inside.txt",
        "cp -- ./inside.txt ./copy.txt",
    ),
)
async def test_bash_common_option_values_and_separator_keep_path_roles(
    tmp_path: Path,
    command: str,
) -> None:
    workspace, host = _synthetic_bash_host(
        tmp_path,
        {"cat", "cp", "grep", "ls", "sort", "touch", "uniq"},
    )
    (workspace / "inside.txt").write_text("needle\n", encoding="utf-8")

    result = await _gateway(workspace, host, level="workspace-write").call(_call(command))

    assert result.status == "success"
    assert result.confirmation is None
    assert len(host.executions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "level",
    ("read-only", "workspace-write", "full-access"),
)
async def test_bash_workspace_root_recursive_force_delete_is_catastrophic(
    tmp_path: Path,
    level: Literal["read-only", "workspace-write", "full-access"],
) -> None:
    workspace, host = _synthetic_bash_host(tmp_path, {"rm"})
    command = f"rm -rf {shlex.quote(workspace.as_posix())}"
    requests: list[ConfirmationRequest] = []

    result = await _gateway(workspace, host, level=level).call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert "catastrophic" in requests[0].reason.lower()
    assert host.executions == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    (
        "rm --rec --for /",
        "systemctl --no-wall reboot",
        "sudo -u root systemctl --no-wall reboot",
        "eval 'shutdown now'",
    ),
)
async def test_bash_catastrophic_wrappers_and_options_confirm_in_full_access(
    tmp_path: Path,
    command: str,
) -> None:
    workspace, host = _synthetic_bash_host(
        tmp_path,
        {"eval", "rm", "sudo", "systemctl"},
    )
    requests: list[ConfirmationRequest] = []

    result = await _gateway(workspace, host, level="full-access").call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert host.executions == []


@pytest.mark.asyncio
async def test_bash_identity_rejects_plain_script_and_duplicate_path(
    tmp_path: Path,
) -> None:
    workspace, host = _synthetic_bash_host(tmp_path, {"cat", "plain"})
    (workspace / "inside.txt").write_text("inside\n", encoding="utf-8")
    sentinel = tmp_path / "identity-executed.txt"
    plain = tmp_path / "bin" / "plain"
    plain.write_text(f"touch {sentinel.as_posix()}\n", encoding="utf-8")
    plain.chmod(0o755)

    script = await host.inspect("plain", workspace)
    duplicate_shell = ResolvedExecShell(
        selector="auto",
        platform="posix",
        family="bash",
        executable="/usr/bin/bash",
        flags=("--noprofile", "--norc"),
        environment=(("PATH", os.pathsep.join((str(tmp_path / "bin"),) * 2)),),
        available=True,
    )
    duplicate = await BashExecHost(duplicate_shell).inspect("cat ./inside.txt", workspace)

    assert script.command_identities[0].kind == "script"
    assert not sentinel.exists()
    assert duplicate.command_identities[0].kind == "ambiguous"
    assert duplicate.command_identities[0].resolution_count == 2


@pytest.mark.asyncio
async def test_bash_identity_rejects_symlink(tmp_path: Path) -> None:
    workspace, _host = _synthetic_bash_host(tmp_path, {"linked"})
    target_dir = tmp_path / "target-bin"
    target_dir.mkdir()
    target = target_dir / "linked"
    target.write_bytes(b"\x7fELF")
    target.chmod(0o755)
    link_dir = tmp_path / "link-bin"
    link_dir.mkdir()
    link = link_dir / "linked"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this host")
    linked_shell = ResolvedExecShell(
        selector="auto",
        platform="posix",
        family="bash",
        executable="/usr/bin/bash",
        flags=("--noprofile", "--norc"),
        environment=(("PATH", str(link_dir)),),
        available=True,
    )
    linked = await BashExecHost(linked_shell).inspect("linked", workspace)

    assert linked.command_identities[0].kind == "shim"


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ("read-only", "workspace-write"))
@pytest.mark.parametrize(
    ("case", "expected_kind", "expected_hits"),
    (
        ("single", "native", 1),
        ("symlinked-directory", "native", 1),
        ("duplicate-entry", "ambiguous", 2),
        ("directory-alias", "ambiguous", 2),
        ("final-file-symlink", "shim", 1),
        ("distinct-targets", "ambiguous", 2),
    ),
)
async def test_bash_path_identity_crosses_gateway(
    tmp_path: Path,
    level: Literal["read-only", "workspace-write"],
    case: str,
    expected_kind: str,
    expected_hits: int,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "cat"
    executable.write_bytes(b"\x7fELF")
    executable.chmod(0o755)
    path_entries: tuple[Path, ...] = (bin_dir,)

    if case in {"symlinked-directory", "directory-alias"}:
        alias_dir = tmp_path / "alias-bin"
        try:
            alias_dir.symlink_to(bin_dir, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symbolic links are unavailable on this host")
        path_entries = (alias_dir,) if case == "symlinked-directory" else (bin_dir, alias_dir)
    elif case == "duplicate-entry":
        path_entries = (bin_dir, bin_dir)
    elif case == "final-file-symlink":
        link_dir = tmp_path / "link-bin"
        link_dir.mkdir()
        try:
            (link_dir / "cat").symlink_to(executable)
        except (OSError, NotImplementedError):
            pytest.skip("symbolic links are unavailable on this host")
        path_entries = (link_dir,)
    elif case == "distinct-targets":
        second_dir = tmp_path / "second-bin"
        second_dir.mkdir()
        second_executable = second_dir / "cat"
        second_executable.write_bytes(b"\x7fELF")
        second_executable.chmod(0o755)
        path_entries = (bin_dir, second_dir)

    workspace, host = _synthetic_bash_host(tmp_path, set(), path_entries=path_entries)
    (workspace / "inside.txt").write_text("inside\n", encoding="utf-8")
    command = "cat ./inside.txt"
    identity = (await host.inspect(command, workspace)).command_identities[0]
    assert identity.kind == expected_kind
    assert identity.resolution_count == expected_hits
    if expected_kind == "native":
        assert identity.resolved == str(executable.resolve())

    requests: list[ConfirmationRequest] = []
    result = await _gateway(workspace, host, level=level).call(
        _call(command),
        confirmation=lambda request: _decline(requests, request),
    )

    if expected_kind == "native":
        assert result.status == "success"
        assert result.confirmation is None
        assert requests == []
        assert len(host.executions) == 1
    else:
        assert result.status == "refused"
        assert len(requests) == 1
        assert requests[0].details == {
            "command": command,
            "cwd": str(workspace.resolve()),
            "timeout": 7,
        }
        assert host.executions == []


def test_windows_shell_resolution_never_discovers_git_bash() -> None:
    requested: list[str] = []

    def which(name: str) -> str | None:
        requested.append(name)
        return r"C:\Program Files\Git\bin\bash.exe" if name == "bash" else None

    resolved = resolve_exec_shell(
        "auto",
        platform="windows",
        which=which,
        version_probe=lambda executable, environment: None,
        environment={"PATH": r"C:\Program Files\Git\bin"},
    )

    assert resolved.available is False
    assert resolved.family == "powershell"
    assert "bash" not in requested


@pytest.mark.skipif(os.name != "posix", reason="requires a real POSIX production host")
@pytest.mark.asyncio
async def test_real_posix_bash_inspect_policy_execute_smoke(tmp_path: Path) -> None:
    resolved = resolve_exec_shell("pwsh")
    assert resolved.available is True
    assert resolved.family == "bash"
    host = BashExecHost(resolved)

    result = await _gateway(tmp_path, host, level="read-only").call(_call("pwd"))

    assert result.status == "success"
    assert str(tmp_path.resolve()) in result.content
