"""Exec Core Catalog Tool."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Final, Literal
from urllib.parse import urlsplit

from omni.agent.tools.base import BaseTool, ToolError, ToolParam, truncate_text
from omni.agent.tools.core.exec_host import (
    ExecCapabilityUnavailable,
    ExecHost,
    ExecHostError,
    ExecProcess,
    create_exec_host,
    resolve_exec_shell,
)
from omni.agent.tools.core.exec_policy import (
    CatastrophicMatch,
    ExecAssessment,
    catastrophic_matches,
    destructive_matches,
)
from omni.agent.tools.network_safety import DNSResolver, SocketDNSResolver, assess_target
from omni.agent.tools.permission import (
    NetworkAssessment,
    NetworkTargetRisk,
    NormalizedNetworkTarget,
    ToolAuthorizationSession,
    ToolInvocationFacts,
)

_OUTPUT_LIMIT: Final[int] = 4000
class ExecTool(BaseTool):
    """Adapt one normalized Tool call to the process-lifetime Host Exec boundary."""

    name = "exec"
    description = "Run one host-shell command with captured output in the selected directory."
    required = ("command",)

    command: Annotated[str, ToolParam(description="Shell command to execute.", min_length=1)]
    cwd: Annotated[str, ToolParam(description="Working directory.", min_length=1)] = "."
    timeout: Annotated[
        int,
        ToolParam(description="Execution timeout in seconds.", minimum=1, maximum=600),
    ] = 60

    def __init__(
        self,
        *,
        workspace: Path,
        resolver: DNSResolver | None = None,
        host: ExecHost | None = None,
    ) -> None:
        self._workspace = workspace
        self._resolver = SocketDNSResolver() if resolver is None else resolver
        self._host = (
            create_exec_host(resolve_exec_shell("auto"))
            if host is None
            else host
        )

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Freeze one canonical cwd for validation, inspection, confirmation, and execution."""
        prepared = await super().prepare_arguments(arguments)
        if not self._host.resolved_shell.available:
            return prepared
        cwd = prepared.get("cwd")
        if not isinstance(cwd, str):
            raise ToolError("Exec working directory is invalid.")
        prepared["cwd"] = str(
            self.resolve_path_argument(workspace=self._workspace, requested=cwd)
        )
        return prepared

    def validate_arguments(  # type: ignore[override]
        self,
        *,
        command: str,
        cwd: str,
        timeout: int,
    ) -> str | None:
        del timeout
        if not command.strip():
            return "Exec command must not be blank."
        if "\x00" in command:
            return "Exec command must not contain a NUL character."
        if not self._host.resolved_shell.available:
            return self._host.resolved_shell.diagnostic or (
                "Exec capability is unavailable because the selected shell is missing."
            )
        try:
            target = self.resolve_path_argument(workspace=self._workspace, requested=cwd)
        except ToolError as error:
            return error.message
        if not target.is_dir():
            return "Exec working directory must be a directory."
        return None

    async def execute(self, *, command: str, cwd: str, timeout: int) -> str:
        return await self._execute_host(
            command=command,
            cwd=cwd,
            timeout=timeout,
            assessment=None,
        )

    async def execute_authorized(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
    ) -> str:
        """Execute with the immutable assessment owned by this authorization call."""
        command = arguments.get("command")
        cwd = arguments.get("cwd")
        timeout = arguments.get("timeout")
        if not isinstance(command, str) or not isinstance(cwd, str) or not isinstance(timeout, int):
            raise ToolError("Exec arguments are invalid.")
        assessment = authorization.exec_assessment
        return await self._execute_host(
            command=command,
            cwd=cwd,
            timeout=timeout,
            assessment=assessment if isinstance(assessment, ExecAssessment) else None,
        )

    async def _execute_host(
        self,
        *,
        command: str,
        cwd: str,
        timeout: int,
        assessment: ExecAssessment | None,
    ) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=cwd)
        if not target.is_dir():
            raise ToolError("Exec working directory must be a directory.")
        try:
            execute_assessed = getattr(self._host, "execute_assessed", None)
            if callable(execute_assessed):
                outcome = await execute_assessed(
                    command,
                    target,
                    timeout,
                    assessment=assessment,
                )
            else:
                outcome = await self._host.execute(command, target, timeout)
        except asyncio.CancelledError:
            raise
        except ExecCapabilityUnavailable as error:
            raise ToolError(str(error)) from error
        except ExecHostError as error:
            raise ToolError(str(error)) from error
        except Exception as error:
            raise ToolError("Exec failed to start the selected shell.") from error
        if outcome.timed_out:
            raise ToolError(
                _format_timeout(timeout=timeout, stdout=outcome.stdout, stderr=outcome.stderr)
            )
        return _format_result(
            exit_code=outcome.exit_code,
            stdout=outcome.stdout,
            stderr=outcome.stderr,
        )

    async def collect_invocation_facts(
        self,
        prepared_arguments: dict[str, Any],
    ) -> ToolInvocationFacts:
        """Attach one detached Host assessment to the shared authorization facts."""
        command = prepared_arguments["command"]
        cwd = prepared_arguments["cwd"]
        if not isinstance(command, str) or not isinstance(cwd, str):
            raise ToolError("Exec arguments are invalid.")
        target = self.resolve_path_argument(workspace=self._workspace, requested=cwd)
        try:
            assessment = await self._host.inspect(command, target)
            audit_git_delegation = getattr(self._host, "audit_git_delegation", None)
            if callable(audit_git_delegation):
                assessment = await audit_git_delegation(
                    command,
                    target,
                    self._workspace,
                    assessment,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            assessment = ExecAssessment.uncertain_result(
                "Exec inspection failed.",
                status="failed",
            )
        assessment = _complete_command_risk_facts(assessment, command)
        network_targets = tuple(
            [await self._assess_url(url) for url in assessment.network_targets]
        )
        cwd_access = self.canonical_file_access(
            workspace=self._workspace,
            base=self._workspace,
            requested=target,
            role="read",
        )
        return ToolInvocationFacts(
            tool_name=self.name,
            normalized_arguments=prepared_arguments,
            file_accesses=(cwd_access,),
            exec_assessment=assessment,
            network_targets=network_targets,
        )

    async def _assess_url(self, url: str) -> NetworkAssessment:
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
            port = parsed.port
        except ValueError:
            return NetworkAssessment(
                target=NormalizedNetworkTarget(
                    url=url,
                    scheme="http",
                    host="invalid",
                    port=80,
                ),
                static_risk="dns_failure",
            )
        scheme: Literal["http", "https"] = (
            "https" if parsed.scheme.lower() == "https" else "http"
        )
        if hostname is None:
            hostname = "invalid"
            risk: NetworkTargetRisk | None = "dns_failure"
        else:
            risk = None
        effective_port = (
            port if port is not None else (443 if parsed.scheme.lower() == "https" else 80)
        )
        target = NormalizedNetworkTarget(
            url=url,
            scheme=scheme,
            host=hostname,
            port=effective_port,
        )
        if risk is None:
            risk = (await assess_target(hostname, effective_port, self._resolver)).risk
        return NetworkAssessment(target=target, static_risk=risk)


def _complete_command_risk_facts(
    assessment: ExecAssessment,
    command: str,
) -> ExecAssessment:
    """Keep command-level risk facts complete across custom Host adapters."""
    catastrophic = _merge_matches(assessment.catastrophic_matches, catastrophic_matches(command))
    destructive = tuple(
        dict.fromkeys((*assessment.destructive_matches, *destructive_matches(command)))
    )
    if (
        catastrophic == assessment.catastrophic_matches
        and destructive == assessment.destructive_matches
    ):
        return assessment
    return replace(
        assessment,
        catastrophic_matches=catastrophic,
        destructive_matches=destructive,
    )


def _merge_matches(
    existing: tuple[CatastrophicMatch, ...],
    additions: tuple[CatastrophicMatch, ...],
) -> tuple[CatastrophicMatch, ...]:
    seen = {item.rule for item in existing}
    return (*existing, *(item for item in additions if item.rule not in seen))

def _format_result(*, exit_code: int | None, stdout: bytes, stderr: bytes) -> str:
    return _format_streams(
        heading=f"Exit code: {exit_code}",
        stdout=stdout,
        stderr=stderr,
    )


def _format_timeout(*, timeout: int, stdout: bytes, stderr: bytes) -> str:
    return _format_streams(
        heading=f"Exec timed out after {timeout} seconds.",
        stdout=stdout,
        stderr=stderr,
    )


def _format_streams(*, heading: str, stdout: bytes, stderr: bytes) -> str:
    blocks = [heading]
    decoded_stdout = stdout.decode("utf-8", errors="replace")
    decoded_stderr = stderr.decode("utf-8", errors="replace")
    if decoded_stdout:
        blocks.append(f"stdout:\n{decoded_stdout}")
    if decoded_stderr:
        blocks.append(f"stderr:\n{decoded_stderr}")
    return truncate_text("\n".join(blocks), limit=_OUTPUT_LIMIT)

__all__ = ["ExecProcess", "ExecTool"]
