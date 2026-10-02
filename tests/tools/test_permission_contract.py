from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from myclaw.agent.confirmation import ConfirmationAborted
from myclaw.agent.permission import ToolPermissionLevel
from myclaw.agent.tools.base import BaseTool, ToolError
from myclaw.agent.tools.core.exec_policy import (
    ExecAssessment,
    ExecCommandIdentity,
    ExecShellSelector,
    ResolvedExecShell,
)
from myclaw.agent.tools.permission import (
    NetworkAssessment,
    NetworkTargetRisk,
    NormalizedNetworkTarget,
    PermissionContext,
    PermissionDecision,
    ScheduleAction,
    ToolAuthorizationSession,
    ToolInvocationFacts,
    ToolPermissionPolicy,
)
from myclaw.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
)


class _RecordingSession:
    def __init__(self, decision: PermissionDecision) -> None:
        self.decision = decision
        self.exec_assessment: ExecAssessment | None = None

    def initial_decision(self) -> PermissionDecision:
        return self.decision

    def confirmation_reason(self) -> str:
        return "Tool confirmation is required."

    async def authorize_network_target(
        self,
        target: object,
        resolved_addresses: tuple[str, ...],
    ) -> None:
        del target, resolved_addresses


class _RecordingPolicy(ToolPermissionPolicy):
    def __init__(self) -> None:
        self.facts: list[ToolInvocationFacts] = []
        self.contexts: list[PermissionContext] = []
        self.sessions: list[ToolAuthorizationSession] = []

    def open(
        self,
        facts: ToolInvocationFacts,
        context: PermissionContext,
    ) -> ToolAuthorizationSession:
        session = _RecordingSession("direct")
        self.facts.append(facts)
        self.contexts.append(context)
        self.sessions.append(session)
        return session


class _FixedDecisionPolicy(_RecordingPolicy):
    def __init__(self, decision: PermissionDecision) -> None:
        super().__init__()
        self.decision = decision

    def open(
        self,
        facts: ToolInvocationFacts,
        context: PermissionContext,
    ) -> ToolAuthorizationSession:
        del context
        session = _RecordingSession(self.decision)
        self.facts.append(facts)
        self.sessions.append(session)
        return session


class _PreparingTool(BaseTool):
    name = "preparing"
    description = "A Tool used to inspect the authorization contract."
    required = ("count",)
    count: int

    def __init__(self) -> None:
        self.calls: list[int] = []

    async def execute(self, *, count: int) -> str:
        self.calls.append(count)
        return str(count)


class _NoSafetyTool(BaseTool):
    name = "no_safety"
    description = "A Tool with no additional authorization facts."
    required = ("value",)
    value: str

    def __init__(self, observed_sessions: list[ToolAuthorizationSession] | None = None) -> None:
        self.calls: list[str] = []
        self.observed_sessions = observed_sessions

    async def execute(self, *, value: str) -> str:
        self.calls.append(value)
        return value

    async def execute_authorized(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
    ) -> str:
        if self.observed_sessions is not None:
            self.observed_sessions.append(authorization)
        return await super().execute_authorized(arguments, authorization)


class _ValidationTool(BaseTool):
    name = "validation_error"
    description = "A Tool with a validation error."

    def validate_arguments(self) -> str:  # type: ignore[override]
        return "validation failed"

    async def execute(self) -> str:
        raise AssertionError("validation must prevent execution")


class _RefusingTool(BaseTool):
    name = "business_refusal"
    description = "A Tool with a business refusal."
    required = ("value",)
    value: str

    def refusal_reason(self, *, value: str) -> str:
        del value
        return "business refusal"

    async def execute(self, *, value: str) -> str:
        del value
        raise AssertionError("business refusal must prevent execution")


class _ExecutionErrorTool(BaseTool):
    name = "execution_error"
    description = "A Tool with an execution error."

    async def execute(self) -> str:
        raise ToolError("execution failed")


class _AbortedExecutionTool(BaseTool):
    name = "aborted_execution"
    description = "A Tool whose execution-time confirmation is lifecycle-aborted."

    async def execute(self) -> str:
        raise AssertionError("the authorized execution seam must be used")

    async def execute_authorized(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
    ) -> str:
        del arguments, authorization
        raise ConfirmationAborted("confirmation lifecycle cancelled")


@pytest.mark.asyncio
async def test_gateway_opens_a_fresh_session_with_structured_call_facts() -> None:
    policy = _RecordingPolicy()
    context = PermissionContext(origin="foreground")
    tool = _PreparingTool()
    gateway = ToolGateway._for_memory(
        (tool,),
        permission_policy=policy,
        permission_context=context,
    )

    first = await gateway.call(
        ModelToolCall(id="call-1", name=tool.name, arguments=json.dumps({"count": "7"}))
    )
    second = await gateway.call(
        ModelToolCall(id="call-2", name=tool.name, arguments=json.dumps({"count": "8"}))
    )

    assert (first.status, first.content) == ("success", "7")
    assert (second.status, second.content) == ("success", "8")
    assert [facts.tool_name for facts in policy.facts] == [tool.name, tool.name]
    assert [facts.normalized_arguments for facts in policy.facts] == [
        {"count": 7},
        {"count": 8},
    ]
    assert all(not hasattr(facts, "safety_reason") for facts in policy.facts)
    assert policy.contexts == [context, context]
    assert policy.sessions[0] is not policy.sessions[1]
    assert tool.calls == [7, 8]
    assert not any(
        name in vars(tool)
        for name in ("authorization_session", "permission_state", "_authorization_session")
    )


@pytest.mark.asyncio
async def test_policy_confirm_decision_uses_one_confirmation_and_passes_session_to_execution() -> None:
    policy = _FixedDecisionPolicy("confirm")
    observed_sessions: list[ToolAuthorizationSession] = []
    tool = _NoSafetyTool(observed_sessions)
    gateway = ToolGateway._for_memory((tool,), permission_policy=policy)
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(id="call-confirm", name=tool.name, arguments='{"value":"payload"}'),
        confirmation=decline,
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].reason == "Tool confirmation is required."
    assert requests[0].details == {"value": "payload"}
    assert observed_sessions == []
    assert tool.calls == []


@pytest.mark.asyncio
async def test_direct_policy_session_reaches_the_execution_boundary() -> None:
    policy = _FixedDecisionPolicy("direct")
    observed_sessions: list[ToolAuthorizationSession] = []
    tool = _NoSafetyTool(observed_sessions)
    gateway = ToolGateway._for_memory((tool,), permission_policy=policy)

    result = await gateway.call(
        ModelToolCall(id="call-authorized", name=tool.name, arguments='{"value":"payload"}')
    )

    assert result.status == "success"
    assert len(observed_sessions) == 1
    assert observed_sessions[0] is policy.sessions[0]
    assert tool.calls == ["payload"]


@pytest.mark.asyncio
async def test_approved_confirm_session_reaches_the_execution_boundary() -> None:
    policy = _FixedDecisionPolicy("confirm")
    observed_sessions: list[ToolAuthorizationSession] = []
    tool = _NoSafetyTool(observed_sessions)
    gateway = ToolGateway._for_memory((tool,), permission_policy=policy)

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        return "approved"

    result = await gateway.call(
        ModelToolCall(id="call-approved", name=tool.name, arguments='{"value":"payload"}'),
        confirmation=approve,
    )

    assert result.status == "success"
    assert result.confirmation is not None
    assert result.confirmation.decision == "approved"
    assert observed_sessions == [policy.sessions[0]]
    assert tool.calls == ["payload"]


@pytest.mark.asyncio
async def test_execution_time_confirmation_abort_remains_typed() -> None:
    tool = _AbortedExecutionTool()
    gateway = ToolGateway._for_memory(
        (tool,),
        permission_policy=_FixedDecisionPolicy("direct"),
    )

    with pytest.raises(ConfirmationAborted, match="lifecycle cancelled"):
        await gateway.call(
            ModelToolCall(id="call-aborted", name=tool.name, arguments="{}")
        )


@pytest.mark.asyncio
async def test_hard_and_business_errors_do_not_open_permission_or_confirmation() -> None:
    policy = _RecordingPolicy()
    gateway = ToolGateway._for_memory(
        (_ValidationTool(), _RefusingTool(), _ExecutionErrorTool()),
        permission_policy=policy,
    )

    invalid_json = await gateway.call(
        ModelToolCall(id="call-invalid-json", name="validation_error", arguments="not json")
    )
    unknown_tool = await gateway.call(
        ModelToolCall(id="call-unknown", name="unknown", arguments="{}")
    )
    validation = await gateway.call(
        ModelToolCall(id="call-validation", name="validation_error", arguments="{}")
    )
    refusal = await gateway.call(
        ModelToolCall(
            id="call-refusal",
            name="business_refusal",
            arguments='{"value":"payload"}',
        )
    )
    execution = await gateway.call(
        ModelToolCall(id="call-execution", name="execution_error", arguments="{}")
    )

    assert [result.status for result in (invalid_json, unknown_tool, validation, execution)] == [
        "error",
        "error",
        "error",
        "error",
    ]
    assert refusal.status == "refused"
    assert [facts.tool_name for facts in policy.facts] == ["execution_error"]
    assert len(policy.sessions) == 1


@pytest.mark.asyncio
async def test_invalid_policy_decision_is_a_hard_error_without_confirmation() -> None:
    class InvalidSession:
        def initial_decision(self) -> str:
            return "defer"

        async def authorize_network_target(
            self,
            target: object,
            resolved_addresses: tuple[str, ...],
        ) -> None:
            del target, resolved_addresses

    class InvalidPolicy(ToolPermissionPolicy):
        def open(
            self,
            facts: ToolInvocationFacts,
            context: PermissionContext,
        ) -> ToolAuthorizationSession:
            del facts, context
            return InvalidSession()  # type: ignore[return-value]

    tool = _NoSafetyTool()
    gateway = ToolGateway._for_memory((tool,), permission_policy=InvalidPolicy())
    result = await gateway.call(
        ModelToolCall(id="call-invalid-decision", name=tool.name, arguments='{"value":"x"}')
    )

    assert result.status == "error"
    assert result.content == "Tool authorization returned an invalid decision."
    assert tool.calls == []


def test_invocation_facts_detach_normalized_arguments() -> None:
    arguments: dict[str, Any] = {"nested": {"value": 1}}
    facts = ToolInvocationFacts(tool_name="tool", normalized_arguments=arguments)

    arguments["nested"] = {"value": 2}
    detached = facts.normalized_arguments
    detached["nested"]["value"] = 3

    assert facts.normalized_arguments == {"nested": {"value": 1}}


@pytest.mark.parametrize(
    ("risk", "expected_reason"),
    [
        (
            "dns_failure",
            "Exec URL DNS resolution is unavailable or returned no addresses "
            "and requires confirmation.",
        ),
        (
            "dns_non_global",
            "Exec URL resolves to a private or non-global address and requires confirmation.",
        ),
    ],
)
def test_exec_network_confirmation_reason_uses_exec_subject(
    risk: NetworkTargetRisk,
    expected_reason: str,
) -> None:
    facts = ToolInvocationFacts(
        tool_name="exec",
        normalized_arguments={"command": "curl http://private.example", "cwd": "."},
        exec_assessment=ExecAssessment(syntax_confidence="high", syntax_uncertain=False),
        network_targets=(
            NetworkAssessment(
                target=NormalizedNetworkTarget(
                    url="http://private.example",
                    scheme="http",
                    host="private.example",
                    port=80,
                ),
                static_risk=risk,
            ),
        ),
    )

    authorization = ToolPermissionPolicy().open(
        facts,
        PermissionContext(level="read-only", origin="foreground"),
    )

    assert authorization.initial_decision() == "confirm"
    assert authorization.confirmation_reason() == expected_reason


@pytest.mark.parametrize("shell_family", ["pwsh", "powershell"])
@pytest.mark.parametrize(
    ("case", "level", "expected_decision", "expected_reason"),
    [
        ("grammar_rejection", "read-only", "confirm", None),
        ("workspace_read", "read-only", "direct", "Tool confirmation is required."),
        (
            "readonly_write",
            "read-only",
            "confirm",
            "Write access requires confirmation in read-only mode.",
        ),
        ("workspace_write", "workspace-write", "direct", "Tool confirmation is required."),
        (
            "external_read",
            "workspace-write",
            "confirm",
            "The requested path resolves outside the Workspace and requires confirmation.",
        ),
        (
            "external_write_readonly",
            "read-only",
            "confirm",
            "The requested path resolves outside the Workspace and requires confirmation.",
        ),
    ],
)
def test_exec_post_grammar_policy_preserves_shell_parity(
    tmp_path: Path,
    shell_family: ExecShellSelector,
    case: str,
    level: ToolPermissionLevel,
    expected_decision: PermissionDecision,
    expected_reason: str | None,
) -> None:
    (tmp_path / "inside.txt").write_text("inside", encoding="utf-8")
    (tmp_path.parent / "outside.txt").write_text("outside", encoding="utf-8")
    commands = {
        "grammar_rejection": "Write-Host hello",
        "workspace_read": r"Get-Content -LiteralPath .\inside.txt",
        "readonly_write": r"Clear-Content -LiteralPath .\inside.txt",
        "workspace_write": r"Clear-Content -LiteralPath .\inside.txt",
        "external_read": r"Get-Content -LiteralPath ..\outside.txt",
        "external_write_readonly": r"Clear-Content -LiteralPath ..\outside.txt",
    }
    command = commands[case]
    command_name = command.split(maxsplit=1)[0]
    identity = ExecCommandIdentity(
        requested=command_name,
        canonical=command_name,
        resolved="Microsoft.PowerShell.Management",
        module="Microsoft.PowerShell.Management",
        kind="cmdlet",
        resolution_count=1,
    )
    assessment = ExecAssessment(
        syntax_confidence="high",
        syntax_uncertain=False,
        command_identities=(identity,),
    )
    shell = ResolvedExecShell(
        selector=shell_family,
        platform="windows",
        family="pwsh" if shell_family == "pwsh" else "powershell",
        executable=shell_family,
        flags=(),
        environment=(),
        available=True,
    )
    facts = ToolInvocationFacts(
        tool_name="exec",
        normalized_arguments={"command": command, "cwd": str(tmp_path)},
        exec_assessment=assessment,
    )

    authorization = ToolPermissionPolicy().open(
        facts,
        PermissionContext(
            level=level,
            origin="foreground",
            workspace_root=tmp_path,
            exec_shell=shell,
        ),
    )

    if expected_reason is None:
        expected_reason = "The PowerShell command is not on the fixed candidate list."
    assert authorization.initial_decision() == expected_decision
    assert authorization.confirmation_reason() == expected_reason
    assert authorization.exec_assessment == assessment


@pytest.mark.parametrize("action", ["list", "add", "remove"])
@pytest.mark.parametrize("level", ["read-only", "workspace-write", "full-access"])
def test_schedule_policy_maps_every_action_and_current_level(
    action: str,
    level: str,
) -> None:
    facts = ToolInvocationFacts(
        tool_name="schedule",
        normalized_arguments={"action": action},
        schedule_action=ScheduleAction(action=action),  # type: ignore[arg-type]
    )
    authorization = ToolPermissionPolicy().open(
        facts,
        PermissionContext(
            level=level,  # type: ignore[arg-type]
            configured_schedule_level="read-only",
            origin="foreground",
        ),
    )

    expected = "direct" if action == "list" or level != "read-only" else "confirm"
    assert authorization.initial_decision() == expected


@pytest.mark.parametrize("action", ["add", "remove"])
def test_schedule_policy_without_run_snapshot_preserves_direct_behavior(action: str) -> None:
    facts = ToolInvocationFacts(
        tool_name="schedule",
        normalized_arguments={"action": action},
        schedule_action=ScheduleAction(action=action),  # type: ignore[arg-type]
    )

    authorization = ToolPermissionPolicy().open(
        facts,
        PermissionContext(
            configured_schedule_level="full-access",
            origin="foreground",
        ),
    )

    assert authorization.initial_decision() == "direct"


@pytest.mark.parametrize(
    ("configured", "current", "expects_escalation"),
    [
        (configured, current, configured_index > current_index)
        for configured_index, configured in enumerate(
            ("read-only", "workspace-write", "full-access")
        )
        for current_index, current in enumerate(
            ("read-only", "workspace-write", "full-access")
        )
    ],
)
def test_schedule_add_policy_compares_all_configured_and_current_levels(
    configured: str,
    current: str,
    expects_escalation: bool,
) -> None:
    facts = ToolInvocationFacts(
        tool_name="schedule",
        normalized_arguments={"action": "add"},
        schedule_action=ScheduleAction(action="add"),
    )
    authorization = ToolPermissionPolicy().open(
        facts,
        PermissionContext(
            level=current,  # type: ignore[arg-type]
            configured_schedule_level=configured,  # type: ignore[arg-type]
            origin="foreground",
        ),
    )

    expected = "confirm" if current == "read-only" or expects_escalation else "direct"
    assert authorization.initial_decision() == expected
    reason = getattr(authorization, "confirmation_reason", lambda: "")()
    assert ("configured Schedule level" in reason) is expects_escalation


def test_schedule_add_merges_crud_and_escalation_reasons_once() -> None:
    facts = ToolInvocationFacts(
        tool_name="schedule",
        normalized_arguments={"action": "add"},
        schedule_action=ScheduleAction(action="add"),
    )
    authorization = ToolPermissionPolicy().open(
        facts,
        PermissionContext(
            level="read-only",
            configured_schedule_level="full-access",
            origin="foreground",
        ),
    )

    assert authorization.initial_decision() == "confirm"
    reason = authorization.confirmation_reason()
    assert "persistent scheduled work" in reason
    assert "configured Schedule level 'full-access'" in reason
    assert reason.count("requires confirmation") == 2
