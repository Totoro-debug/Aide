"""Portable Host Exec adapters with one process-lifetime shell resolution."""

from __future__ import annotations

import asyncio
import base64
import json
import ntpath
import os
import posixpath
import re
import shlex
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Protocol, cast

from myclaw.tools.exec.policy import (
    BASH_APPROVED_BUILTINS,
    ExecAssessment,
    ExecCommandIdentity,
    ExecDynamicConstruct,
    ExecIdentityKind,
    ExecOutcome,
    ExecPlatform,
    ExecShellFamily,
    ExecShellSelector,
    ResolvedExecShell,
    assess_command,
    bash_git_audit_targets,
    classify_bash_command,
    classify_powershell_command,
    powershell_git_audit_targets,
)
from myclaw.utils.async_tasks import await_task_preserving_cancellation
from myclaw.utils.host_filesystem import host_path_is_within

EXEC_CAPABILITY_ERROR: Final = "Exec capability is unavailable because the selected shell is missing."
_PROCESS_REAP_TIMEOUT: Final[float] = 5.0
_INSPECTION_TIMEOUT: Final[int] = 5
_INSPECTION_MAX_COMMAND_LENGTH: Final[int] = 1_048_576
_POSIX_ENVIRONMENT: Final[tuple[str, ...]] = ("HOME", "LANG", "TERM", "PATH")
_WINDOWS_ENVIRONMENT: Final[tuple[str, ...]] = (
    "HOME",
    "LANG",
    "TERM",
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "TEMP",
    "TMP",
    "USERPROFILE",
)
_PS_FLAGS: Final[tuple[str, ...]] = ("-NoLogo", "-NoProfile", "-NonInteractive")
_BASH_FLAGS: Final[tuple[str, ...]] = ("--noprofile", "--norc")
_NATIVE_EXECUTABLE_MAGICS: Final[frozenset[bytes]] = frozenset(
    {
        b"\xbe\xba\xfe\xca",
        b"\xbf\xba\xfe\xca",
        b"\xca\xfe\xba\xbe",
        b"\xca\xfe\xba\xbf",
        b"\xce\xfa\xed\xfe",
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xce",
        b"\xfe\xed\xfa\xcf",
    }
)
_BACKGROUND_CLEANUPS: Final[set[asyncio.Task[None]]] = set()
_PS_VERSION_PREFIX: Final[str] = "MYCLAW_PS_VERSION:"
_GIT_HARDENED_FORM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<leading>\s*)(?P<requested>git(?:\.exe)?)"
    r"(?P<global>(?:\s+-C\s+(?:\"[^\"]*\"|'[^']*'|[^\s]+))*)\s+"
    r"(?P<form>status|diff|log|show|branch|rev-parse|ls-files)"
    r"(?P<rest>(?:\s.*)?)$",
    re.IGNORECASE,
)
_GIT_HARDENED_FLAGS: Final[str] = " --no-ext-diff --no-textconv"
_GIT_DELEGATION_CONFIG_PATTERN: Final[str] = (
    r"^(include\.|includeif\.|filter\..*\.(clean|process)$)"
)
_PS_VERSION_COMMAND: Final[str] = rf"""
$text = '{_PS_VERSION_PREFIX}' + $PSVersionTable.PSVersion.ToString()
$bytes = [Text.Encoding]::UTF8.GetBytes($text)
$stdout = [Console]::OpenStandardOutput()
$stdout.Write($bytes, 0, $bytes.Length)
"""
_PS_SESSION_PREAMBLE: Final[str] = r"""
$PSModuleAutoLoadingPreference = 'None'
$trustedModuleNames = @(
    'Microsoft.PowerShell.Management',
    'Microsoft.PowerShell.Utility'
)
foreach ($trustedModuleName in $trustedModuleNames) {
    $trustedModulePath = [IO.Path]::Combine(
        $PSHOME,
        'Modules',
        $trustedModuleName,
        $trustedModuleName + '.psd1'
    )
    if ([IO.File]::Exists($trustedModulePath)) {
        Import-Module -Name $trustedModulePath -ErrorAction Stop
    }
}
"""
_PS_INSPECTOR_SCRIPT: Final[str] = (
    r"""
param([string]$encodedSource)
"""
    + _PS_SESSION_PREAMBLE
    + r"""
$source = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($encodedSource))
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $source,
    [ref]$tokens,
    [ref]$errors
)
$commandAsts = @($ast.FindAll(
    { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
    $true
))
$commands = @($commandAsts | ForEach-Object { $_.GetCommandName() } | Where-Object { $_ })
$sessionCommands = @(Get-Command -All -ListImported)
$identities = @(
    foreach ($commandAst in $commandAsts) {
        $requested = $commandAst.GetCommandName()
        if (-not $requested) {
            continue
        }
        $resolutions = @($sessionCommands | Where-Object { $_.Name -ieq $requested })
        if ($requested -ieq 'git' -or $requested -ieq 'git.exe') {
            $resolutions = @(
                Get-Command -Name $requested -All -CommandType Application,ExternalScript `
                    -ErrorAction SilentlyContinue
            )
        }
        $resolution = if ($resolutions.Count -eq 1) { $resolutions[0] } else { $null }
        $commandType = if ($null -eq $resolution) { '' } else { [string]$resolution.CommandType }
        $module = if ($null -eq $resolution) { $null } elseif ($resolution.ModuleName) {
            [string]$resolution.ModuleName
        } elseif ($resolution.Source) {
            [string]$resolution.Source
        } else { $null }
        $resolved = if ($null -eq $resolution) { $null } elseif ($resolution.Path) {
            [string]$resolution.Path
        } elseif ($module) { $module } else { $null }
        $canonical = if ($null -eq $resolution) { $null } elseif ($resolution.ResolvedCommandName) {
            [string]$resolution.ResolvedCommandName
        } elseif ($resolution.Name) { [string]$resolution.Name } else { $null }
        $kind = if ($resolutions.Count -gt 1) {
            'ambiguous'
        } else {
            switch ($commandType) {
                'Alias' { 'alias'; break }
                'Function' { 'function'; break }
                'Filter' { 'function'; break }
                'ExternalScript' { 'script'; break }
                'Script' { 'script'; break }
                'Application' { 'native'; break }
                'Cmdlet' { 'cmdlet'; break }
                default { 'unknown' }
            }
        }
        [ordered]@{
            requested = [string]$requested
            canonical = $canonical
            resolved = $resolved
            module = $module
            kind = $kind
            resolution_count = [int]$resolutions.Count
        }
    }
)
$json = [ordered]@{
    syntax_ok = ($errors.Count -eq 0)
    command_names = $commands
    identities = $identities
} | ConvertTo-Json -Compress
$bytes = [Text.Encoding]::UTF8.GetBytes($json)
$stdout = [Console]::OpenStandardOutput()
$stdout.Write($bytes, 0, $bytes.Length)
"""
)


class ExecHostError(Exception):
    """A host-level execution failure that is not a Tool Result."""


class ExecCapabilityUnavailable(ExecHostError):
    """The selected shell is absent and cannot execute any command."""

    def __init__(self) -> None:
        super().__init__(EXEC_CAPABILITY_ERROR)


class ExecProcess(Protocol):
    @property
    def returncode(self) -> int | None: ...

    async def communicate(self) -> tuple[bytes | None, bytes | None]: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ExecProcessSpec:
    """Shared process inputs used by both Host inspection and execution."""

    executable: str
    flags: tuple[str, ...]
    cwd: Path
    environment: tuple[tuple[str, str], ...]

    @property
    def env(self) -> dict[str, str]:
        return dict(self.environment)


class ExecHost(Protocol):
    """The raw Host boundary consumed by the Exec Tool adapter."""

    resolved_shell: ResolvedExecShell

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment: ...

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome: ...

    def process_spec(self, cwd: Path) -> ExecProcessSpec: ...


def resolve_exec_shell(
    selector: ExecShellSelector,
    *,
    platform: str | None = None,
    which: Callable[[str], str | None] | None = None,
    version_probe: Callable[[str, dict[str, str]], tuple[int, ...] | None] | None = None,
    environment: Mapping[str, str] | None = None,
) -> ResolvedExecShell:
    """Resolve one effective selector without ever crossing an explicit fallback."""
    if selector not in {"auto", "powershell", "pwsh"}:
        raise ValueError("Exec shell selector is invalid")
    host_platform = _normalize_platform(os.name if platform is None else platform)
    env = _minimal_environment(
        os.environ if environment is None else environment,
        platform=host_platform,
    )
    locator = which or (lambda name: shutil.which(name, path=env.get("PATH")))

    def locate(name: str) -> str | None:
        return _canonical_shell_path(
            locator(name),
            expected=name,
            platform=host_platform,
            require_exists=which is None,
        )

    if host_platform == "posix":
        executable = locate("bash")
        return _resolved(
            selector=selector,
            platform="posix",
            family="bash",
            executable=executable,
            flags=_BASH_FLAGS,
            environment=env,
        )

    probe = version_probe or _probe_powershell_version
    if selector == "pwsh":
        executable = locate("pwsh")
        version = None if executable is None else probe(executable, env)
        if executable is None or not _is_pwsh_version(version):
            return _unavailable(
                selector=selector,
                platform="windows",
                family="pwsh",
                flags=_PS_FLAGS,
                environment=env,
            )
        return _resolved(
            selector=selector,
            platform="windows",
            family="pwsh",
            executable=executable,
            flags=_PS_FLAGS,
            environment=env,
            version=version,
        )

    if selector == "powershell":
        executable = locate("powershell")
        version = None if executable is None else probe(executable, env)
        if executable is None or not _is_windows_powershell_version(version):
            return _unavailable(
                selector=selector,
                platform="windows",
                family="powershell",
                flags=_PS_FLAGS,
                environment=env,
            )
        return _resolved(
            selector=selector,
            platform="windows",
            family="powershell",
            executable=executable,
            flags=_PS_FLAGS,
            environment=env,
            version=version,
        )

    pwsh = locate("pwsh")
    if pwsh is not None:
        version = probe(pwsh, env)
        if _is_pwsh_version(version):
            return _resolved(
                selector=selector,
                platform="windows",
                family="pwsh",
                executable=pwsh,
                flags=_PS_FLAGS,
                environment=env,
                version=version,
            )

    powershell = locate("powershell")
    version = None if powershell is None else probe(powershell, env)
    if powershell is not None and _is_windows_powershell_version(version):
        return _resolved(
            selector=selector,
            platform="windows",
            family="powershell",
            executable=powershell,
            flags=_PS_FLAGS,
            environment=env,
            version=version,
        )
    return _unavailable(
        selector=selector,
        platform="windows",
        family="powershell",
        flags=_PS_FLAGS,
        environment=env,
    )


def create_exec_host(resolved_shell: ResolvedExecShell) -> ExecHost:
    """Construct the adapter for an already resolved process-lifetime shell."""
    if not resolved_shell.available:
        return _UnavailableExecHost(resolved_shell)
    if resolved_shell.family == "bash":
        return BashExecHost(resolved_shell)
    return PowerShellExecHost(resolved_shell)


def _is_lexically_within(path: Path, root: Path) -> bool:
    try:
        absolute = os.path.abspath(path)
        return os.path.commonpath(
            (os.path.normcase(absolute), os.path.normcase(str(root)))
        ) == os.path.normcase(str(root))
    except ValueError:
        return False


def _normalize_platform(value: str) -> ExecPlatform:
    if value in {"nt", "windows", "win32"}:
        return "windows"
    if value in {"posix", "linux", "darwin", "macos"}:
        return "posix"
    raise ValueError("Exec host platform is unsupported")


def _resolved(
    *,
    selector: ExecShellSelector,
    platform: ExecPlatform,
    family: ExecShellFamily,
    executable: str | None,
    flags: tuple[str, ...],
    environment: dict[str, str],
    version: tuple[int, ...] | None = None,
) -> ResolvedExecShell:
    if executable is None:
        return _unavailable(
            selector=selector,
            platform=platform,
            family=family,
            flags=flags,
            environment=environment,
        )
    return ResolvedExecShell(
        selector=selector,
        platform=platform,
        family=family,
        executable=executable,
        flags=flags,
        environment=tuple(environment.items()),
        available=True,
        version=version,
    )


def _unavailable(
    *,
    selector: ExecShellSelector,
    platform: ExecPlatform,
    family: ExecShellFamily,
    flags: tuple[str, ...],
    environment: dict[str, str],
) -> ResolvedExecShell:
    return ResolvedExecShell(
        selector=selector,
        platform=platform,
        family=family,
        executable=None,
        flags=flags,
        environment=tuple(environment.items()),
        available=False,
        diagnostic=EXEC_CAPABILITY_ERROR,
    )


def _minimal_environment(
    source: Mapping[str, str],
    *,
    platform: ExecPlatform,
) -> dict[str, str]:
    result: dict[str, str] = {}
    allowed = _WINDOWS_ENVIRONMENT if platform == "windows" else _POSIX_ENVIRONMENT
    for expected in allowed:
        for name, value in source.items():
            if name.upper() == expected:
                result[expected] = value
                break
    return result


def _canonical_shell_path(
    executable: str | None,
    *,
    expected: str,
    platform: ExecPlatform,
    require_exists: bool,
) -> str | None:
    if executable is None:
        return None
    path_module = ntpath if platform == "windows" else posixpath
    normalized: str = path_module.normpath(executable)
    if not path_module.isabs(normalized):
        return None
    basename = path_module.basename(normalized).lower()
    allowed_names = {expected.lower()}
    if platform == "windows":
        allowed_names.add(f"{expected.lower()}.exe")
    if basename not in allowed_names:
        return None
    current_platform = _normalize_platform(os.name)
    if require_exists and current_platform == platform:
        try:
            normalized = str(Path(normalized).resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            return None
        if path_module.basename(normalized).lower() not in allowed_names:
            return None
    return normalized


def _is_pwsh_version(version: tuple[int, ...] | None) -> bool:
    return version is not None and version >= (7,)


def _is_windows_powershell_version(version: tuple[int, ...] | None) -> bool:
    return version is not None and version[:1] == (5,) and version >= (5, 1)


def _probe_powershell_version(executable: str, environment: dict[str, str]) -> tuple[int, ...] | None:
    try:
        completed = subprocess.run(
            [executable, *_PS_FLAGS, "-Command", _PS_VERSION_COMMAND],
            check=False,
            cwd=None,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    match = re.fullmatch(
        rf"\s*{re.escape(_PS_VERSION_PREFIX)}(\d+(?:\.\d+){{1,3}})\s*",
        completed.stdout,
    )
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


class _BaseExecHost:
    def __init__(self, resolved_shell: ResolvedExecShell) -> None:
        if not resolved_shell.available or resolved_shell.executable is None:
            raise ValueError("An available shell is required for a concrete Exec Host")
        self.resolved_shell = resolved_shell

    def process_spec(self, cwd: Path) -> ExecProcessSpec:
        return ExecProcessSpec(
            executable=cast(str, self.resolved_shell.executable),
            flags=self.resolved_shell.flags,
            cwd=Path(cwd),
            environment=self.resolved_shell.environment,
        )

    def environment_for_command(self, command: str) -> dict[str, str]:
        environment = self.resolved_shell.env
        if self.resolved_shell.family not in {"powershell", "pwsh", "bash"}:
            return environment
        if re.search(r"(?:^|[|;\s])git(?:\.exe)?(?:\s|$)", command, re.IGNORECASE):
            null_device = "NUL" if self.resolved_shell.platform == "windows" else "/dev/null"
            environment.update(
                {
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_SYSTEM": null_device,
                    "GIT_CONFIG_GLOBAL": null_device,
                    "GIT_PAGER": "cat",
                    "PAGER": "cat",
                    "GIT_EXTERNAL_DIFF": "",
                    "GIT_DIFF_OPTS": "",
                    "GIT_OPTIONAL_LOCKS": "0",
                    "GIT_TERMINAL_PROMPT": "0",
                    "GIT_CONFIG_COUNT": "4",
                    "GIT_CONFIG_KEY_0": "core.pager",
                    "GIT_CONFIG_VALUE_0": "cat",
                    "GIT_CONFIG_KEY_1": "core.fsmonitor",
                    "GIT_CONFIG_VALUE_1": "false",
                    "GIT_CONFIG_KEY_2": "diff.external",
                    "GIT_CONFIG_VALUE_2": "",
                    "GIT_CONFIG_KEY_3": "core.hooksPath",
                    "GIT_CONFIG_VALUE_3": null_device,
                }
            )
        return environment

    def command_for_execution(
        self,
        command: str,
        *,
        git_executable: str | None = None,
    ) -> str:
        """Add fixed Git flags that disable configuration-driven diff execution."""
        if self.resolved_shell.family not in {"powershell", "pwsh", "bash"}:
            return command
        match = _GIT_HARDENED_FORM_PATTERN.fullmatch(command)
        if match is None:
            return command
        requested = match.group("requested")
        invocation = requested
        if git_executable is not None:
            if self.resolved_shell.family == "bash":
                invocation = shlex.quote(git_executable)
            else:
                quoted = git_executable.replace("'", "''")
                invocation = f"& '{quoted}'"
        flags = _GIT_HARDENED_FLAGS if match.group("form").casefold() in {"diff", "show"} else ""
        return (
            f"{match.group('leading')}{invocation}{match.group('global')} "
            f"{match.group('form')}{flags}{match.group('rest')}"
        )

    async def _run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        timeout: int,
        environment: dict[str, str] | None = None,
    ) -> ExecOutcome:
        process: ExecProcess | None = None
        communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None = None
        try:
            process = await _spawn_process(
                argv=argv,
                cwd=cwd,
                environment=(
                    self.resolved_shell.env if environment is None else environment
                ),
            )
            communication = asyncio.create_task(process.communicate())
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=timeout,
                )
            except TimeoutError:
                stdout, stderr = await _cleanup_preserving_cancellation(process, communication)
                return ExecOutcome(
                    exit_code=process.returncode,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=True,
                )
            except asyncio.CancelledError:
                await _cleanup_without_replacing_cancellation(process, communication)
                raise
            except Exception as error:
                await _cleanup_without_replacing_cancellation(process, communication)
                raise ExecHostError("Exec failed while reading process output.") from error
            return ExecOutcome(
                exit_code=process.returncode,
                stdout=_as_bytes(stdout),
                stderr=_as_bytes(stderr),
            )
        except asyncio.CancelledError:
            if process is not None and communication is None:
                await _cleanup_without_replacing_cancellation(process, None)
            raise
        except ExecHostError:
            raise
        except Exception as error:
            raise ExecHostError(
                f"Exec failed to start {self.resolved_shell.family}: {error}"
            ) from error

class BashExecHost(_BaseExecHost):
    """POSIX Bash Host using a no-profile, no-rc process policy."""

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        try:
            source_bytes = command.encode("utf-8")
        except UnicodeEncodeError:
            return ExecAssessment.uncertain_result(
                "Bash AST input encoding was invalid.",
                status="uncertain",
            )
        if len(source_bytes) > _INSPECTION_MAX_COMMAND_LENGTH:
            return ExecAssessment.uncertain_result(
                "Bash AST input exceeded the inspection limit.",
                status="uncertain",
            )
        try:
            import tree_sitter_bash
            from tree_sitter import Language, Parser

            parser = Parser(Language(tree_sitter_bash.language()))
            tree = parser.parse(source_bytes)
        except (ImportError, OSError, TypeError, ValueError):
            return ExecAssessment.uncertain_result(
                "Bash AST inspection is unavailable.",
                status="uncertain",
            )
        except Exception:
            return ExecAssessment.uncertain_result(
                "Bash AST inspection failed.",
                status="failed",
            )

        root = tree.root_node
        command_names = _tree_command_names(root, command)
        if not _bash_tree_is_complete(root, len(source_bytes)):
            return assess_command(
                command,
                family="bash",
                syntax_confidence="unknown",
                syntax_uncertain=True,
                command_names=command_names,
                diagnostics=("Bash AST syntax was malformed.",),
                inspector_status="uncertain",
            )
        assessment = assess_command(
            command,
            family="bash",
            syntax_confidence="high",
            syntax_uncertain=False,
            command_names=command_names,
            command_identities=_tree_command_identities(
                command_names,
                cwd=cwd,
                environment=self.resolved_shell.env,
            ),
            dynamic_constructs=_bash_dynamic_constructs(root),
        )
        grammar = classify_bash_command(command, assessment)
        return replace(assessment, file_accesses=grammar.file_accesses)

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
        return await self.execute_assessed(command, cwd, timeout, assessment=None)

    async def execute_assessed(
        self,
        command: str,
        cwd: Path,
        timeout: int,
        *,
        assessment: ExecAssessment | None,
    ) -> ExecOutcome:
        spec = self.process_spec(cwd)
        git_executable = self._assessed_git_executable(assessment)
        return await self._run(
            (
                spec.executable,
                *spec.flags,
                "-c",
                self.command_for_execution(command, git_executable=git_executable),
            ),
            cwd=spec.cwd,
            timeout=timeout,
            environment=self.environment_for_command(command),
        )

    async def audit_git_delegation(
        self,
        command: str,
        cwd: Path,
        workspace_root: Path,
        assessment: ExecAssessment,
    ) -> ExecAssessment:
        """Complete Git facts only after Workspace boundaries are available."""
        targets = bash_git_audit_targets(command, str(cwd))
        if not targets:
            return assessment
        provisional = replace(assessment, git_delegation_safe=True)
        if not classify_bash_command(command, provisional).accepted:
            return assessment
        try:
            root = workspace_root.resolve(strict=True)
            for identity_index, target in targets:
                identity = assessment.command_identities[identity_index]
                if (
                    identity.kind != "native"
                    or identity.resolution_count != 1
                    or identity.resolved is None
                    or not os.path.isabs(identity.resolved)
                ):
                    return assessment
                executable = Path(identity.resolved)
                if _is_lexically_within(executable, root):
                    return assessment
                canonical_executable = executable.resolve(strict=False)
                if host_path_is_within(canonical_executable, root):
                    return assessment
                candidate = Path(target)
                if not _is_lexically_within(candidate, root):
                    return assessment
                canonical_target = candidate.resolve(strict=True)
                if not canonical_target.is_dir() or not host_path_is_within(
                    canonical_target,
                    root,
                ):
                    return assessment
        except (IndexError, OSError, RuntimeError, ValueError):
            return assessment
        git_delegation_safe, audit_failed = await self._audit_git_delegation(
            command,
            cwd,
            assessment.command_identities,
        )
        if audit_failed:
            return ExecAssessment.uncertain_result(
                "Git repository delegation inspection failed.",
                status="failed",
                catastrophic=assessment.catastrophic_matches,
            )
        return replace(assessment, git_delegation_safe=git_delegation_safe)

    @staticmethod
    def _assessed_git_executable(assessment: ExecAssessment | None) -> str | None:
        if assessment is None or not assessment.command_identities:
            return None
        identity = assessment.command_identities[0]
        if (
            identity.requested not in {"git", "git.exe"}
            or identity.kind != "native"
            or identity.resolution_count != 1
            or identity.resolved is None
            or not os.path.isabs(identity.resolved)
        ):
            return None
        return identity.resolved

    async def _audit_git_delegation(
        self,
        command: str,
        cwd: Path,
        identities: tuple[ExecCommandIdentity, ...],
    ) -> tuple[bool | None, bool]:
        targets = bash_git_audit_targets(command, str(cwd))
        if not targets:
            return None, False
        audited = False
        for identity_index, target in targets:
            if identity_index >= len(identities):
                return None, True
            identity = identities[identity_index]
            if (
                identity.kind != "native"
                or identity.resolution_count != 1
                or identity.resolved is None
                or not os.path.isabs(identity.resolved)
            ):
                continue
            audited = True
            try:
                outcome = await self._run(
                    (
                        identity.resolved,
                        "-C",
                        target,
                        "config",
                        "--no-includes",
                        "--name-only",
                        "--get-regexp",
                        _GIT_DELEGATION_CONFIG_PATTERN,
                    ),
                    cwd=cwd,
                    timeout=_INSPECTION_TIMEOUT,
                    environment=self.environment_for_command("git"),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return None, True
            if outcome.timed_out or outcome.stderr:
                return None, True
            if outcome.exit_code == 0:
                if outcome.stdout.strip():
                    return False, False
                continue
            if outcome.exit_code != 1:
                return None, True
        return (True if audited else None), False


class PowerShellExecHost(_BaseExecHost):
    """Windows PowerShell Host with a static Parser.ParseInput inspector."""

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        try:
            source_bytes = command.encode("utf-8")
        except UnicodeEncodeError:
            return ExecAssessment.uncertain_result(
                "PowerShell AST input encoding was invalid.",
                status="uncertain",
            )
        if len(source_bytes) > _INSPECTION_MAX_COMMAND_LENGTH:
            return ExecAssessment.uncertain_result(
                "PowerShell AST input exceeded the inspection limit.",
                status="uncertain",
            )
        encoded = base64.b64encode(source_bytes).decode("ascii")
        inspector_command = f"& {{\n{_PS_INSPECTOR_SCRIPT}\n}} '{encoded}'"
        spec = self.process_spec(cwd)
        try:
            outcome = await self._run(
                (spec.executable, *spec.flags, "-Command", inspector_command),
                cwd=spec.cwd,
                timeout=_INSPECTION_TIMEOUT,
                environment=self.environment_for_command(command),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ExecAssessment.uncertain_result(
                "PowerShell AST inspection failed.",
                status="failed",
            )
        if outcome.timed_out:
            return ExecAssessment.uncertain_result(
                "PowerShell AST inspection timed out.",
                status="timeout",
            )
        if outcome.exit_code != 0 or outcome.stderr:
            return ExecAssessment.uncertain_result(
                "PowerShell AST inspection failed.",
                status="failed",
            )
        try:
            payload = json.loads(outcome.stdout.decode("utf-8"))
            if not isinstance(payload, dict) or set(payload) not in (
                {"syntax_ok", "command_names"},
                {"syntax_ok", "command_names", "identities"},
            ):
                raise ValueError("parser output is not an object")
            syntax_ok = payload.get("syntax_ok")
            raw_names = payload.get("command_names")
            if not isinstance(syntax_ok, bool) or not isinstance(raw_names, list):
                raise ValueError("parser output is malformed")
            if any(not isinstance(name, str) or not name for name in raw_names):
                raise ValueError("parser command names are malformed")
            raw_identities = payload.get("identities")
            command_identities: tuple[ExecCommandIdentity, ...] = ()
            identities_complete = raw_identities is not None
            if raw_identities is None:
                raw_identities = []
            if not isinstance(raw_identities, list) or (
                identities_complete and len(raw_identities) != len(raw_names)
            ):
                raise ValueError("parser command identities are malformed")
            parsed_identities: list[ExecCommandIdentity] = []
            for expected_name, raw_identity in zip(
                raw_names,
                raw_identities,
                strict=identities_complete,
            ):
                if not isinstance(raw_identity, dict) or set(raw_identity) != {
                    "requested",
                    "canonical",
                    "resolved",
                    "module",
                    "kind",
                    "resolution_count",
                }:
                    raise ValueError("parser command identity is malformed")
                requested = raw_identity.get("requested")
                canonical = raw_identity.get("canonical")
                resolved = raw_identity.get("resolved")
                module = raw_identity.get("module")
                kind = raw_identity.get("kind")
                resolution_count = raw_identity.get("resolution_count")
                if (
                    not isinstance(requested, str)
                    or not requested
                    or requested.casefold() != expected_name.casefold()
                ):
                    raise ValueError("parser command identity name is malformed")
                if canonical is not None and (
                    not isinstance(canonical, str) or not canonical
                ):
                    raise ValueError("parser command identity canonical name is malformed")
                if resolved is not None and (not isinstance(resolved, str) or not resolved):
                    raise ValueError("parser command identity resolution is malformed")
                if module is not None and (not isinstance(module, str) or not module):
                    raise ValueError("parser command identity module is malformed")
                if kind == "native" and isinstance(resolved, str) and resolved.casefold().endswith(
                    (".cmd", ".bat", ".com")
                ):
                    kind = "shim"
                if kind not in {
                    "builtin",
                    "cmdlet",
                    "native",
                    "alias",
                    "function",
                    "script",
                    "shim",
                    "workspace",
                    "ambiguous",
                    "unknown",
                }:
                    raise ValueError("parser command identity kind is malformed")
                if (
                    not isinstance(resolution_count, int)
                    or isinstance(resolution_count, bool)
                    or resolution_count < 0
                ):
                    raise ValueError("parser command identity count is malformed")
                populated_identity_fields = (canonical, resolved, module)
                if resolution_count == 0 and (
                    kind != "unknown" or any(value is not None for value in populated_identity_fields)
                ):
                    raise ValueError("unknown parser command identity is inconsistent")
                if resolution_count > 1 and (
                    kind != "ambiguous"
                    or any(value is not None for value in populated_identity_fields)
                ):
                    raise ValueError("ambiguous parser command identity is inconsistent")
                if resolution_count == 1 and (kind in {"unknown", "ambiguous"} or canonical is None):
                    raise ValueError("unique parser command identity is inconsistent")
                if kind == "cmdlet" and (resolved is None or module is None):
                    raise ValueError("cmdlet parser command identity is incomplete")
                if kind in {"native", "script", "shim", "workspace"} and resolved is None:
                    raise ValueError("external parser command identity is incomplete")
                if kind in {"native", "script", "shim", "workspace"} and not ntpath.isabs(
                    cast(str, resolved)
                ):
                    raise ValueError("external parser command identity is not absolute")
                parsed_identities.append(
                    ExecCommandIdentity(
                        requested=requested,
                        resolved=resolved,
                        canonical=canonical,
                        module=module,
                        resolution_count=resolution_count,
                        kind=cast(Any, kind),
                    )
                )
            command_identities = tuple(parsed_identities)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            return ExecAssessment.uncertain_result(
                "PowerShell AST inspection returned malformed output.",
                status="uncertain",
            )
        if not syntax_ok:
            return assess_command(
                command,
                family="powershell",
                syntax_confidence="unknown",
                syntax_uncertain=True,
                command_names=tuple(cast(list[str], raw_names)),
                command_identities=command_identities,
                diagnostics=("PowerShell AST syntax was malformed.",),
                inspector_status="uncertain",
            )
        if not identities_complete:
            return assess_command(
                command,
                family="powershell",
                syntax_confidence="high",
                syntax_uncertain=False,
                command_names=tuple(cast(list[str], raw_names)),
                diagnostics=("PowerShell identity output was incomplete.",),
                inspector_status="uncertain",
            )
        return assess_command(
            command,
            family="powershell",
            syntax_confidence="high",
            syntax_uncertain=False,
            command_names=tuple(cast(list[str], raw_names)),
            command_identities=command_identities,
        )

    async def audit_git_delegation(
        self,
        command: str,
        cwd: Path,
        workspace_root: Path,
        assessment: ExecAssessment,
    ) -> ExecAssessment:
        """Complete Git facts only after Workspace boundaries are available."""
        targets = powershell_git_audit_targets(command, str(cwd))
        if not targets:
            return assessment
        provisional = replace(assessment, git_delegation_safe=True)
        if not classify_powershell_command(command, provisional).accepted:
            return assessment
        try:
            root = workspace_root.resolve(strict=True)
            for identity_index, target in targets:
                identity = assessment.command_identities[identity_index]
                if (
                    identity.kind != "native"
                    or identity.resolution_count != 1
                    or identity.resolved is None
                    or not ntpath.isabs(identity.resolved)
                ):
                    return assessment
                executable = Path(identity.resolved)
                if _is_lexically_within(executable, root):
                    return assessment
                canonical_executable = executable.resolve(strict=False)
                if host_path_is_within(canonical_executable, root):
                    return assessment
                candidate = Path(target)
                if not _is_lexically_within(candidate, root):
                    return assessment
                canonical_target = candidate.resolve(strict=True)
                if not canonical_target.is_dir() or not host_path_is_within(
                    canonical_target,
                    root,
                ):
                    return assessment
        except (IndexError, OSError, RuntimeError, ValueError):
            return assessment
        git_delegation_safe, audit_failed = await self._audit_git_delegation(
            command,
            cwd,
            assessment.command_identities,
        )
        if audit_failed:
            return ExecAssessment.uncertain_result(
                "Git repository delegation inspection failed.",
                status="failed",
                catastrophic=assessment.catastrophic_matches,
            )
        return replace(assessment, git_delegation_safe=git_delegation_safe)

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
        return await self.execute_assessed(command, cwd, timeout, assessment=None)

    async def execute_assessed(
        self,
        command: str,
        cwd: Path,
        timeout: int,
        *,
        assessment: ExecAssessment | None,
    ) -> ExecOutcome:
        spec = self.process_spec(cwd)
        git_executable = self._assessed_git_executable(assessment)
        execution_command = self.command_for_execution(
            command,
            git_executable=git_executable,
        )
        wrapped_command = f"& {{\n{_PS_SESSION_PREAMBLE}\n{execution_command}\n}}"
        return await self._run(
            (
                spec.executable,
                *spec.flags,
                "-Command",
                wrapped_command,
            ),
            cwd=spec.cwd,
            timeout=timeout,
            environment=self.environment_for_command(command),
        )

    @staticmethod
    def _assessed_git_executable(assessment: ExecAssessment | None) -> str | None:
        if assessment is None or not assessment.command_identities:
            return None
        identity = assessment.command_identities[0]
        if (
            identity.requested.casefold() not in {"git", "git.exe"}
            or identity.kind != "native"
            or identity.resolution_count != 1
            or identity.resolved is None
            or not ntpath.isabs(identity.resolved)
        ):
            return None
        return identity.resolved

    async def _audit_git_delegation(
        self,
        command: str,
        cwd: Path,
        identities: tuple[ExecCommandIdentity, ...],
    ) -> tuple[bool | None, bool]:
        targets = powershell_git_audit_targets(command, str(cwd))
        if not targets:
            return None, False
        audited = False
        for identity_index, target in targets:
            if identity_index >= len(identities):
                return None, True
            identity = identities[identity_index]
            if (
                identity.kind != "native"
                or identity.resolution_count != 1
                or identity.resolved is None
                or not ntpath.isabs(identity.resolved)
            ):
                continue
            audited = True
            try:
                outcome = await self._run(
                    (
                        identity.resolved,
                        "-C",
                        target,
                        "config",
                        "--no-includes",
                        "--name-only",
                        "--get-regexp",
                        _GIT_DELEGATION_CONFIG_PATTERN,
                    ),
                    cwd=cwd,
                    timeout=_INSPECTION_TIMEOUT,
                    environment=self.environment_for_command("git"),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return None, True
            if outcome.timed_out:
                return None, True
            if outcome.stderr:
                return None, True
            if outcome.exit_code == 0:
                if outcome.stdout.strip():
                    return False, False
                continue
            if outcome.exit_code != 1:
                return None, True
        return (True if audited else None), False


class _UnavailableExecHost:
    def __init__(self, resolved_shell: ResolvedExecShell) -> None:
        self.resolved_shell = resolved_shell

    def process_spec(self, cwd: Path) -> ExecProcessSpec:
        del cwd
        raise ExecCapabilityUnavailable

    async def inspect(self, command: str, cwd: Path) -> ExecAssessment:
        del command, cwd
        return ExecAssessment.uncertain_result(
            EXEC_CAPABILITY_ERROR,
            status="uncertain",
        )

    async def execute(self, command: str, cwd: Path, timeout: int) -> ExecOutcome:
        del command, cwd, timeout
        raise ExecCapabilityUnavailable


async def _spawn_process(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environment: dict[str, str],
) -> ExecProcess:
    spawning = asyncio.create_task(
        _create_process(
            argv=argv,
            cwd=cwd,
            environment=environment,
        )
    )
    try:
        return await asyncio.shield(spawning)
    except asyncio.CancelledError as cancellation:
        cleanup = asyncio.create_task(_cleanup_cancelled_spawn(spawning))
        try:
            await await_task_preserving_cancellation(cleanup)
        except asyncio.CancelledError:
            raise
        except BaseException:
            pass
        raise cancellation


async def _create_process(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environment: dict[str, str],
) -> ExecProcess:
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=os.fspath(cwd),
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _cleanup_cancelled_spawn(spawning: asyncio.Task[ExecProcess]) -> None:
    try:
        process = await asyncio.wait_for(
            asyncio.shield(spawning),
            timeout=_PROCESS_REAP_TIMEOUT,
        )
    except TimeoutError:
        _defer_spawn_cleanup(spawning)
        return
    except BaseException:
        return
    await _cleanup_process(process, None)


def _defer_spawn_cleanup(spawning: asyncio.Task[ExecProcess]) -> None:
    spawning.cancel()
    cleanup = asyncio.create_task(_cleanup_late_spawn(spawning))
    _BACKGROUND_CLEANUPS.add(cleanup)
    cleanup.add_done_callback(_BACKGROUND_CLEANUPS.discard)


async def _cleanup_late_spawn(spawning: asyncio.Task[ExecProcess]) -> None:
    try:
        process = await spawning
        await _cleanup_process(process, None)
    except BaseException:
        pass


async def _cleanup_process(
    process: ExecProcess,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None,
) -> tuple[bytes, bytes]:
    try:
        if process.returncode is None:
            process.kill()
    except Exception:
        pass

    wait_task = asyncio.create_task(process.wait())
    tasks: tuple[asyncio.Task[object], ...] = (
        wait_task,
        *((communication,) if communication is not None else ()),
    )
    joined = asyncio.create_task(_join_cleanup_tasks(tasks))
    try:
        await asyncio.wait_for(asyncio.shield(joined), timeout=_PROCESS_REAP_TIMEOUT)
    except TimeoutError:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if not joined.done():
            joined.cancel()
        await asyncio.gather(joined, return_exceptions=True)
    return _communication_output(communication)


async def _join_cleanup_tasks(tasks: tuple[asyncio.Task[object], ...]) -> None:
    await asyncio.gather(*tasks, return_exceptions=True)


async def _cleanup_without_replacing_cancellation(
    process: ExecProcess,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None,
) -> None:
    cleanup = asyncio.create_task(_cleanup_process(process, communication))
    try:
        await await_task_preserving_cancellation(cleanup)
    except asyncio.CancelledError:
        raise
    except BaseException:
        pass


async def _cleanup_preserving_cancellation(
    process: ExecProcess,
    communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None,
) -> tuple[bytes, bytes]:
    cleanup = asyncio.create_task(_cleanup_process(process, communication))
    return await await_task_preserving_cancellation(cleanup)


def _communication_output(
    communication: asyncio.Task[tuple[bytes | None, bytes | None]] | None,
) -> tuple[bytes, bytes]:
    if communication is None or not communication.done() or communication.cancelled():
        return b"", b""
    try:
        stdout, stderr = communication.result()
    except BaseException:
        return b"", b""
    return _as_bytes(stdout), _as_bytes(stderr)


def _as_bytes(value: bytes | None) -> bytes:
    return b"" if value is None else value


def _bash_tree_is_complete(root: Any, source_length: int) -> bool:
    if (
        getattr(root, "type", None) != "program"
        or getattr(root, "start_byte", None) != 0
        or getattr(root, "end_byte", None) != source_length
        or bool(getattr(root, "has_error", True))
    ):
        return False
    stack = [root]
    while stack:
        node = stack.pop()
        if (
            getattr(node, "type", None) == "ERROR"
            or bool(getattr(node, "is_error", False))
            or bool(getattr(node, "is_missing", False))
        ):
            return False
        stack.extend(getattr(node, "children", ()))
    return True


def _tree_command_identities(
    command_names: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[ExecCommandIdentity, ...]:
    return tuple(
        _resolve_bash_identity(name, cwd=cwd, environment=environment)
        for name in command_names
    )


def _resolve_bash_identity(
    requested: str,
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> ExecCommandIdentity:
    if requested in BASH_APPROVED_BUILTINS:
        return ExecCommandIdentity(
            requested=requested,
            canonical=requested,
            kind="builtin",
            resolution_count=1,
        )
    candidates = _bash_executable_candidates(requested, cwd=cwd, environment=environment)
    if not candidates:
        return ExecCommandIdentity(requested=requested, resolution_count=0, kind="unknown")
    if len(candidates) > 1:
        return ExecCommandIdentity(requested=requested, resolution_count=len(candidates), kind="ambiguous")
    resolved, is_symlink = candidates[0]
    return ExecCommandIdentity(
        requested=requested,
        resolved=resolved,
        canonical=Path(resolved).name,
        kind="shim" if is_symlink else _bash_executable_kind(resolved, cwd=cwd),
        resolution_count=1,
    )


def _bash_executable_candidates(
    requested: str,
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[tuple[str, bool], ...]:
    if not requested or requested in {".", ".."}:
        return ()
    raw_candidates: list[Path] = []
    if "/" in requested:
        candidate = Path(requested)
        raw_candidates.append(candidate if candidate.is_absolute() else cwd / candidate)
    else:
        for raw_entry in environment.get("PATH", "").split(os.pathsep):
            entry = Path(raw_entry) if raw_entry else cwd
            if not entry.is_absolute():
                entry = cwd / entry
            raw_candidates.append(entry / requested)
    resolved: list[tuple[str, bool]] = []
    for candidate in raw_candidates:
        try:
            if not candidate.is_file() or not os.access(candidate, os.X_OK):
                continue
            is_symlink = candidate.is_symlink()
            value = str(candidate.resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            continue
        resolved.append((value, is_symlink))
    return tuple(resolved)


def _bash_executable_kind(resolved: str, *, cwd: Path) -> ExecIdentityKind:
    path = Path(resolved)
    try:
        if host_path_is_within(path, cwd.resolve(strict=True)):
            return "workspace"
    except (OSError, RuntimeError, ValueError):
        return "unknown"
    if path.suffix.casefold() in {".shim", ".cmd", ".bat", ".com"}:
        return "shim"
    if path.suffix.casefold() in {".sh", ".bash", ".zsh", ".fish", ".py", ".pl", ".rb"}:
        return "script"
    try:
        with path.open("rb") as stream:
            header = stream.read(4)
    except OSError:
        return "unknown"
    if header.startswith(b"#!"):
        return "script"
    if header.startswith(b"\x7fELF") or header in _NATIVE_EXECUTABLE_MAGICS:
        return "native"
    return "script"


def _bash_dynamic_constructs(root: Any) -> tuple[ExecDynamicConstruct, ...]:
    constructs: list[ExecDynamicConstruct] = []
    seen: set[tuple[str, str]] = set()
    node_kinds = {
        "variable_assignment": ("assignment", "variable assignment"),
        "command_substitution": ("substitution", "command substitution"),
        "process_substitution": ("substitution", "process substitution"),
        "simple_expansion": ("variable", "variable expansion"),
        "parameter_expansion": ("variable", "variable expansion"),
        "arithmetic_expansion": ("variable", "arithmetic expansion"),
        "ansi_c_string": ("syntax", "ANSI-C quoted string"),
        "$": ("variable", "locale-translated or variable string"),
        "file_redirect": ("redirection", "stream redirection"),
        "heredoc_redirect": ("redirection", "here-document redirection"),
        "heredoc_body": ("redirection", "here-document body"),
        "herestring_redirect": ("redirection", "here-string redirection"),
        "function_definition": ("control-flow", "function definition"),
        "if_statement": ("control-flow", "conditional statement"),
        "for_statement": ("control-flow", "for loop"),
        "c_style_for_statement": ("control-flow", "C-style for loop"),
        "while_statement": ("control-flow", "while loop"),
        "until_statement": ("control-flow", "until loop"),
        "case_statement": ("control-flow", "case statement"),
        "subshell": ("control-flow", "subshell"),
        "compound_statement": ("control-flow", "compound statement"),
        "comment": ("syntax", "shell comment"),
        ";": ("command-list", "command list"),
        "&&": ("command-list", "conditional command list"),
        "||": ("command-list", "conditional command list"),
        "&": ("command-list", "background command list"),
    }
    indirect_names = {
        ".",
        "bash",
        "builtin",
        "command",
        "dash",
        "doas",
        "env",
        "eval",
        "exec",
        "make",
        "nice",
        "node",
        "nohup",
        "npm",
        "npx",
        "perl",
        "python",
        "python3",
        "ruby",
        "sh",
        "source",
        "sudo",
        "time",
        "xargs",
        "zsh",
    }

    def add(kind: str, expression: str) -> None:
        key = (kind, expression)
        if key not in seen:
            seen.add(key)
            constructs.append(ExecDynamicConstruct(kind, expression))

    stack = [root]
    while stack:
        node = stack.pop()
        node_type = getattr(node, "type", "")
        expression = _tree_node_text(node)
        classification = node_kinds.get(node_type)
        if classification is not None:
            add(classification[0], classification[1])
        if node_type == "word" and _bash_word_has_dynamic_expansion(expression):
            add("glob", "unquoted glob or home expansion")
        if node_type == "command":
            command_name = next(
                (
                    _tree_node_text(child)
                    for child in getattr(node, "children", ())
                    if getattr(child, "type", None) == "command_name"
                ),
                "",
            )
            if command_name.casefold() in indirect_names:
                add("indirect-invocation", "indirect or delegated command invocation")
        stack.extend(reversed(tuple(getattr(node, "children", ()))))
    return tuple(constructs)


def _bash_word_has_dynamic_expansion(value: str) -> bool:
    if value.startswith("~"):
        return True
    escaped = False
    for character in value:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character in "*?[]":
            return True
    return False


def _tree_node_text(node: Any) -> str:
    value = getattr(node, "text", b"")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _tree_command_names(root: Any, source: str) -> tuple[str, ...]:
    names: list[str] = []
    stack = [root]
    encoded = source.encode("utf-8")
    while stack:
        node = stack.pop()
        if getattr(node, "type", None) == "command":
            for child in getattr(node, "children", ()):
                if getattr(child, "type", None) == "command_name":
                    name = encoded[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
                    if name:
                        names.append(name)
                    break
        stack.extend(reversed(tuple(getattr(node, "children", ()))))
    return tuple(names)


__all__ = [
    "EXEC_CAPABILITY_ERROR",
    "BashExecHost",
    "ExecCapabilityUnavailable",
    "ExecHost",
    "ExecHostError",
    "ExecOutcome",
    "ExecProcess",
    "ExecProcessSpec",
    "PowerShellExecHost",
    "ResolvedExecShell",
    "create_exec_host",
    "resolve_exec_shell",
]
