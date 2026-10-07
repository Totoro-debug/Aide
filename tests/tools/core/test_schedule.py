from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from aide.agent.tools.core.schedule import ScheduleTool
from aide.agent.tools.permission import PermissionContext
from aide.agent.tools.tool_gateway import ConfirmationDecision, ConfirmationRequest, ModelToolCall
from aide.agent.workspace_state import WorkspaceState
from aide.schedule.model import JobSchedule, ScheduleJob
from aide.schedule.service import ScheduleService, ScheduleStaleRemovalError
from aide.schedule.store import WorkspaceScheduleStore
from tests.fixtures import SingleToolGateway, write_schedule_state
from tests.fixtures.gateway import contextual_tool

JOB_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


class _ToolClock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


def _store(
    workspace: Path,
    agent_home: Path,
    *persisted_jobs: ScheduleJob,
) -> WorkspaceScheduleStore:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    if persisted_jobs:
        write_schedule_state(state, *persisted_jobs)
    return WorkspaceScheduleStore(state)


def _gateway(tool: ScheduleTool) -> SingleToolGateway:
    return SingleToolGateway((tool,))


def _service(store: WorkspaceScheduleStore) -> ScheduleService:
    async def execute_user_job(job: ScheduleJob) -> None:
        del job

    async def execute_dream() -> object:
        return None

    service = ScheduleService(
        workspace_state=store.workspace_state,
        clock=_ToolClock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    service._store = store
    return service


def test_schema_exposes_the_three_actions_and_optional_schedule_branches(
    workspace: Path,
    agent_home: Path,
) -> None:
    schema = contextual_tool(ScheduleTool, schedule_service=_service(_store(workspace, agent_home))).to_schema()[
        "function"
    ]["parameters"]

    assert schema["required"] == ["action"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {
        "action",
        "message",
        "title",
        "every_seconds",
        "cron_expr",
        "timezone",
        "at_time",
        "job_id",
    }
    assert properties["title"]["type"] == "string"
    assert "default" not in properties["title"]


@pytest.mark.asyncio
async def test_add_uses_the_common_gateway_without_confirmation_and_ignores_lower_priority_fields(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(
        contextual_tool(ScheduleTool,
            schedule_service=_service(store),
            now=lambda: NOW,
            new_uuid=lambda: JOB_UUID,
        )
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_add",
            name="schedule",
            arguments=json.dumps(
                {
                    "action": "add",
                    "message": "  Run it  ",
                    "every_seconds": "60",
                    "cron_expr": {"invalid": True},
                    "timezone": ["invalid"],
                    "at_time": {"invalid": True},
                    "job_id": "ignored",
                }
            ),
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert json.loads(result.content) == {
        "action": "add",
        "job": {
            "job_id": str(JOB_UUID),
            "message": "Run it",
            "title": "Run it",
            "schedule": {"type": "every", "every_seconds": 60},
        },
    }
    jobs = await store.snapshot()
    assert len(jobs) == 1
    assert jobs[0].message == "Run it"


@pytest.mark.asyncio
async def test_add_confirmation_uses_the_canonical_invocation_and_decline_does_not_mutate(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    requests = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    gateway = SingleToolGateway(
        (
            contextual_tool(ScheduleTool,
                schedule_service=_service(store),
                now=lambda: NOW,
                new_uuid=lambda: JOB_UUID,
            ),
        ),
        confirmation=decline,
        permission_context=PermissionContext(
            level="read-only",
            configured_schedule_level="full-access",
            origin="foreground",
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_add_declined",
            name="schedule",
            arguments=json.dumps(
                {
                    "action": "add",
                    "message": "  Run it  ",
                    "title": "  Weekly run  ",
                    "every_seconds": 60,
                    "job_id": "ignored",
                }
            ),
        )
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details == {
        "action": "add",
        "message": "Run it",
        "title": "Weekly run",
        "schedule": {"type": "every", "every_seconds": 60},
    }
    assert await store.snapshot() == ()


@pytest.mark.parametrize(
    ("action", "level", "expects_confirmation"),
    [
        (action, level, action != "list" and level == "read-only")
        for action in ("list", "add", "remove")
        for level in ("read-only", "workspace-write", "full-access")
    ],
)
@pytest.mark.asyncio
async def test_schedule_gateway_covers_every_action_and_current_level(
    workspace: Path,
    agent_home: Path,
    action: str,
    level: str,
    expects_confirmation: bool,
) -> None:
    store = _store(workspace, agent_home)
    if action == "remove":
        await store.add_user_job(
            ScheduleJob(
                job_id=str(JOB_UUID),
                message="Remove this Job",
                schedule=JobSchedule.every(60),
                created_at_ms=1,
                updated_at_ms=1,
            )
        )
    requests = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    arguments: dict[str, object]
    if action == "list":
        arguments = {"action": "list"}
    elif action == "add":
        arguments = {"action": "add", "message": "Create this Job", "every_seconds": 60}
    else:
        arguments = {"action": "remove", "job_id": str(JOB_UUID)}
    gateway = SingleToolGateway(
        (
            contextual_tool(ScheduleTool,
                schedule_service=_service(store),
                now=lambda: NOW,
                new_uuid=lambda: JOB_UUID,
            ),
        ),
        confirmation=approve,
        permission_context=PermissionContext(
            level=level,  # type: ignore[arg-type]
            configured_schedule_level=level,  # type: ignore[arg-type]
            origin="foreground",
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id=f"call_{action}_{level}",
            name="schedule",
            arguments=json.dumps(arguments),
        )
    )

    assert result.status == "success"
    assert len(requests) == int(expects_confirmation)


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
@pytest.mark.asyncio
async def test_schedule_gateway_compares_every_configured_and_current_level(
    workspace: Path,
    agent_home: Path,
    configured: str,
    current: str,
    expects_escalation: bool,
) -> None:
    store = _store(workspace, agent_home)
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = SingleToolGateway(
        (
            contextual_tool(ScheduleTool,
                schedule_service=_service(store),
                now=lambda: NOW,
                new_uuid=lambda: JOB_UUID,
            ),
        ),
        confirmation=approve,
        permission_context=PermissionContext(
            level=current,  # type: ignore[arg-type]
            configured_schedule_level=configured,  # type: ignore[arg-type]
            origin="foreground",
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id=f"call_add_{configured}_{current}",
            name="schedule",
            arguments=json.dumps(
                {"action": "add", "message": "  Create this Job  ", "every_seconds": 60}
            ),
        )
    )

    expects_confirmation = current == "read-only" or expects_escalation
    assert result.status == "success"
    assert len(requests) == int(expects_confirmation)
    if requests:
        assert requests[0].details == {
            "action": "add",
            "message": "Create this Job",
            "title": "Create this Job",
            "schedule": {"type": "every", "every_seconds": 60},
        }
        assert ("persistent scheduled work" in requests[0].reason) is (
            current == "read-only"
        )
        assert ("configured Schedule level" in requests[0].reason) is expects_escalation
    assert len(await store.snapshot()) == 1


@pytest.mark.asyncio
async def test_remove_decline_does_not_mutate_the_store(
    workspace: Path,
    agent_home: Path,
) -> None:
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Keep this Job",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    store = _store(workspace, agent_home)
    await store.add_user_job(job)
    requests = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    gateway = SingleToolGateway(
        (contextual_tool(ScheduleTool, schedule_service=_service(store), now=lambda: NOW),),
        confirmation=decline,
        permission_context=PermissionContext(
            level="read-only",
            configured_schedule_level="read-only",
            origin="foreground",
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_remove_declined",
            name="schedule",
            arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
        )
    )

    assert result.status == "refused"
    assert len(requests) == 1
    assert requests[0].details == {"action": "remove", "job_id": str(JOB_UUID)}
    assert await store.snapshot() == (job,)


@pytest.mark.asyncio
async def test_approved_remove_stale_failure_is_canonical_and_not_retried(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Keep this Job",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    store = _store(workspace, agent_home)
    await store.add_user_job(job)
    service = _service(store)
    requests: list[ConfirmationRequest] = []
    removal_calls = 0

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    async def stale_remove(
        job_id: str,
        *,
        expected: ScheduleJob | None = None,
    ) -> bool:
        nonlocal removal_calls
        del job_id, expected
        removal_calls += 1
        raise ScheduleStaleRemovalError("changed")

    monkeypatch.setattr(service, "remove_user_job", stale_remove)
    gateway = SingleToolGateway(
        (contextual_tool(ScheduleTool, schedule_service=service, now=lambda: NOW),),
        confirmation=approve,
        permission_context=PermissionContext(
            level="read-only",
            configured_schedule_level="read-only",
            origin="foreground",
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_remove_stale",
            name="schedule",
            arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
        )
    )

    assert result.status == "error"
    assert result.content == "Schedule Job changed before removal. Request removal again."
    assert len(requests) == 1
    assert requests[0].details == {"action": "remove", "job_id": str(JOB_UUID)}
    assert removal_calls == 1
    assert await store.snapshot() == (job,)


@pytest.mark.asyncio
async def test_concurrent_schedule_gateways_keep_permission_contexts_isolated(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    tool = contextual_tool(ScheduleTool,
        schedule_service=_service(store),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    low_requests: list[ConfirmationRequest] = []
    low_requested = asyncio.Event()
    release_low = asyncio.Event()

    async def approve_low(request: ConfirmationRequest) -> ConfirmationDecision:
        low_requests.append(request)
        low_requested.set()
        await release_low.wait()
        return "approved"

    async def unexpected_high(request: ConfirmationRequest) -> ConfirmationDecision:
        raise AssertionError(f"Full-Access call must remain direct: {request}")

    low_gateway = SingleToolGateway(
        (tool,),
        confirmation=approve_low,
        permission_context=PermissionContext(
            level="read-only",
            configured_schedule_level="full-access",
            origin="foreground",
        ),
    )
    high_gateway = SingleToolGateway(
        (tool,),
        confirmation=unexpected_high,
        permission_context=PermissionContext(
            level="full-access",
            configured_schedule_level="read-only",
            origin="foreground",
        ),
    )
    low_call = asyncio.create_task(
        low_gateway.call(
            ModelToolCall(
                id="call_low",
                name="schedule",
                arguments='{"action":"add","message":"low","every_seconds":60}',
            )
        )
    )
    await asyncio.wait_for(low_requested.wait(), timeout=1)
    try:
        high_result = await high_gateway.call(
            ModelToolCall(
                id="call_high",
                name="schedule",
                arguments='{"action":"add","message":"high","every_seconds":60}',
            )
        )
    finally:
        release_low.set()
    low_result = await asyncio.wait_for(low_call, timeout=1)

    assert high_result.status == low_result.status == "success"
    assert len(low_requests) == 1
    assert "persistent scheduled work" in low_requests[0].reason
    assert "configured Schedule level 'full-access'" in low_requests[0].reason
    assert {job.message for job in await store.snapshot()} == {"low", "high"}


@pytest.mark.asyncio
async def test_known_store_failure_precedes_schedule_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    service = _service(store)
    store._faulted = True
    requests = []

    async def unexpected(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        raise AssertionError("A known Store failure must not request confirmation")

    gateway = SingleToolGateway(
        (contextual_tool(ScheduleTool, schedule_service=service, now=lambda: NOW),),
        confirmation=unexpected,
        permission_context=PermissionContext(
            level="read-only",
            configured_schedule_level="full-access",
            origin="foreground",
        ),
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_store_failure",
            name="schedule",
            arguments='{"action":"add","message":"message","every_seconds":60}',
        )
    )

    assert result.status == "error"
    assert result.content == "Schedule state could not be updated."
    assert requests == []


@pytest.mark.parametrize("level", ["read-only", "workspace-write", "full-access"])
@pytest.mark.asyncio
async def test_schedule_hard_errors_precede_permission_at_every_level(
    workspace: Path,
    agent_home: Path,
    level: str,
) -> None:
    store = _store(workspace, agent_home)
    requests = []

    async def unexpected(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        raise AssertionError("Schedule hard errors must not request confirmation")

    gateway = SingleToolGateway(
        (contextual_tool(ScheduleTool, schedule_service=_service(store), now=lambda: NOW),),
        confirmation=unexpected,
        permission_context=PermissionContext(
            level=level,  # type: ignore[arg-type]
            configured_schedule_level=level,  # type: ignore[arg-type]
            origin="foreground",
        ),
    )
    invalid_json = await gateway.call(ModelToolCall("invalid-json", "schedule", "not-json"))
    invalid_schema = await gateway.call(
        ModelToolCall("invalid-schema", "schedule", '{"action":"add"}')
    )
    invalid_title = await gateway.call(
        ModelToolCall(
            "invalid-title",
            "schedule",
            '{"action":"add","message":"message","title":null,"every_seconds":60}',
        )
    )
    invalid_schedule = await gateway.call(
        ModelToolCall(
            "invalid-schedule",
            "schedule",
            '{"action":"add","message":"message","every_seconds":0}',
        )
    )
    invalid_message = await gateway.call(
        ModelToolCall(
            "invalid-message",
            "schedule",
            '{"action":"add","message":"   ","every_seconds":60}',
        )
    )
    unknown_action = await gateway.call(
        ModelToolCall("unknown-action", "schedule", '{"action":"change"}')
    )
    invalid_job_id = await gateway.call(
        ModelToolCall("invalid-job-id", "schedule", '{"action":"remove","job_id":"bad"}')
    )
    missing_job = await gateway.call(
        ModelToolCall(
            "missing-job",
            "schedule",
            json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
        )
    )

    assert all(
        result.status == "error"
        for result in (
            invalid_json,
            invalid_schema,
            invalid_title,
            invalid_schedule,
            invalid_message,
            unknown_action,
            invalid_job_id,
        )
    )
    assert missing_job.status == "error"
    assert missing_job.content == "Schedule Job was not found."
    assert requests == []
    assert await store.snapshot() == ()


@pytest.mark.asyncio
async def test_add_normalizes_explicit_title_and_rejects_invalid_explicit_titles(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(
        contextual_tool(ScheduleTool,
            schedule_service=_service(store),
            now=lambda: NOW,
            new_uuid=lambda: JOB_UUID,
        )
    )

    explicit = await gateway.call(
        ModelToolCall(
            id="call_explicit_title",
            name="schedule",
            arguments=json.dumps(
                {
                    "action": "add",
                    "message": "Run it",
                    "title": "\n \u300c Weekly\t review \u300d\nignored",
                    "every_seconds": 60,
                }
            ),
        )
    )

    assert explicit.status == "success"
    assert json.loads(explicit.content)["job"]["title"] == "Weekly review"
    assert (await store.snapshot())[0].title == "Weekly review"

    empty = await gateway.call(
        ModelToolCall(
            id="call_empty_title",
            name="schedule",
            arguments=json.dumps(
                {"action": "add", "message": "Another", "title": '""', "every_seconds": 60}
            ),
        )
    )

    assert empty.status == "error"
    assert empty.content == "Invalid arguments for schedule."

    null = await gateway.call(
        ModelToolCall(
            id="call_null_title",
            name="schedule",
            arguments=json.dumps(
                {"action": "add", "message": "Another", "title": None, "every_seconds": 60}
            ),
        )
    )

    assert null.status == "error"
    assert null.content == "Invalid arguments for schedule."
    assert len(await store.snapshot()) == 1


@pytest.mark.asyncio
async def test_add_normalizes_cron_timezone_and_at_time_without_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    uuids = iter((JOB_UUID, UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")))
    gateway = _gateway(
        contextual_tool(ScheduleTool,
            schedule_service=_service(store),
            now=lambda: NOW,
            new_uuid=lambda: next(uuids),
        )
    )

    cron = await gateway.call(
        ModelToolCall(
            id="call_cron",
            name="schedule",
            arguments='{"action":"add","message":"Cron","cron_expr":" 0 9 * * 1 ","timezone":"Asia/Shanghai"}',
        )
    )
    at = await gateway.call(
        ModelToolCall(
            id="call_at",
            name="schedule",
            arguments='{"action":"add","message":"Past","at_time":"2020-01-02T03:04:05.123456Z"}',
        )
    )

    assert cron.status == "success"
    assert json.loads(cron.content)["job"]["schedule"] == {
        "type": "cron",
        "cron_expr": "0 9 * * 1",
        "timezone": "Asia/Shanghai",
    }
    assert at.status == "success"
    assert json.loads(at.content)["job"]["schedule"] == {
        "type": "at",
        "at_time": "2020-01-02T03:04:05.123+00:00",
    }


@pytest.mark.asyncio
async def test_cron_accepts_valid_iana_timezone_alias(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(
        contextual_tool(ScheduleTool, schedule_service=_service(store), now=lambda: NOW, new_uuid=lambda: JOB_UUID)
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_alias_cron",
            name="schedule",
            arguments=(
                '{"action":"add","message":"Alias","cron_expr":"0 9 * * *","timezone":"US/Eastern"}'
            ),
        )
    )

    assert result.status == "success"
    assert json.loads(result.content)["job"]["schedule"]["timezone"] == "US/Eastern"


@pytest.mark.asyncio
async def test_cron_defaults_to_utc_and_invalid_at_time_is_rejected(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(
        contextual_tool(ScheduleTool, schedule_service=_service(store), now=lambda: NOW, new_uuid=lambda: JOB_UUID)
    )

    cron = await gateway.call(
        ModelToolCall(
            id="call_utc_cron",
            name="schedule",
            arguments='{"action":"add","message":"UTC","cron_expr":"0 9 * * 1"}',
        )
    )
    naive = await gateway.call(
        ModelToolCall(
            id="call_naive_at",
            name="schedule",
            arguments='{"action":"add","message":"Naive","at_time":"2026-08-07T13:00:00"}',
        )
    )

    assert cron.status == "success"
    assert json.loads(cron.content)["job"]["schedule"]["timezone"] == "UTC"
    assert naive.status == "error"
    assert naive.content == "Invalid arguments for schedule."


@pytest.mark.parametrize(
    "schedule_arguments",
    [
        {"cron_expr": "@daily"},
        {"cron_expr": "0 9 * * *", "timezone": "Not/A_Timezone"},
    ],
)
@pytest.mark.asyncio
async def test_add_rejects_invalid_cron_inputs_without_mutating_the_store(
    workspace: Path,
    agent_home: Path,
    schedule_arguments: dict[str, object],
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(
        contextual_tool(ScheduleTool, schedule_service=_service(store), now=lambda: NOW, new_uuid=lambda: JOB_UUID)
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_invalid_cron",
            name="schedule",
            arguments=json.dumps(
                {"action": "add", "message": "Invalid", **schedule_arguments},
                separators=(",", ":"),
            ),
        )
    )

    assert result.status == "error"
    assert result.content == "Invalid arguments for schedule."
    assert await store.snapshot() == ()


@pytest.mark.asyncio
async def test_list_returns_only_public_jobs_in_creation_then_id_order(
    workspace: Path,
    agent_home: Path,
) -> None:
    first_id = JOB_UUID
    second_id = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
    earlier_id = UUID("9ba7b810-9dad-41d1-80b4-00c04fd430c8")
    first = ScheduleJob(
        job_id=str(first_id),
        message="First",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=20,
        updated_at_ms=20,
    )
    second = ScheduleJob(
        job_id=str(second_id),
        message="Second",
        schedule=JobSchedule.at("2026-08-07T14:00:00.000+00:00"),
        created_at_ms=20,
        updated_at_ms=20,
    )
    earlier = ScheduleJob(
        job_id=str(earlier_id),
        message="Earlier",
        schedule=JobSchedule.at("2026-08-07T12:30:00.000+00:00"),
        created_at_ms=10,
        updated_at_ms=10,
    )
    hidden = ScheduleJob(
        job_id="dream",
        source="system",
        message="Hidden",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    store = _store(workspace, agent_home, hidden)
    await store.add_user_job(second)
    await store.add_user_job(first)
    await store.add_user_job(earlier)

    result = await _gateway(
        contextual_tool(ScheduleTool, schedule_service=_service(store), now=lambda: NOW, new_uuid=lambda: JOB_UUID)
    ).call(
        ModelToolCall(
            id="call_list",
            name="schedule",
            arguments='{"action":"list","title":null}',
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert json.loads(result.content) == {
        "jobs": [
            {
                "job_id": str(earlier_id),
                "message": "Earlier",
                "title": "Earlier",
                "schedule": {"type": "at", "at_time": "2026-08-07T12:30:00.000+00:00"},
            },
            {
                "job_id": str(first_id),
                "message": "First",
                "title": "First",
                "schedule": {"type": "at", "at_time": "2026-08-07T13:00:00.000+00:00"},
            },
            {
                "job_id": str(second_id),
                "message": "Second",
                "title": "Second",
                "schedule": {"type": "at", "at_time": "2026-08-07T14:00:00.000+00:00"},
            },
        ]
    }


@pytest.mark.asyncio
async def test_remove_requires_canonical_uuid_and_hides_unknown_or_system_jobs(
    workspace: Path,
    agent_home: Path,
) -> None:
    public = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Remove me",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    hidden = ScheduleJob(
        job_id="dream",
        source="system",
        message="Internal",
        schedule=JobSchedule.every(60),
        created_at_ms=2,
        updated_at_ms=2,
    )
    store = _store(workspace, agent_home, hidden)
    await store.add_user_job(public)
    gateway = _gateway(
        contextual_tool(ScheduleTool, schedule_service=_service(store), now=lambda: NOW, new_uuid=lambda: JOB_UUID)
    )

    invalid = await gateway.call(
        ModelToolCall(
            id="call_invalid_remove",
            name="schedule",
            arguments='{"action":"remove","job_id":"550E8400-E29B-41D4-A716-446655440000"}',
        )
    )
    unknown = await gateway.call(
        ModelToolCall(
            id="call_unknown_remove",
            name="schedule",
            arguments='{"action":"remove","job_id":"8ba7b810-9dad-41d1-80b4-00c04fd430c8"}',
        )
    )
    system = await gateway.call(
        ModelToolCall(
            id="call_system_remove",
            name="schedule",
            arguments='{"action":"remove","job_id":"9ba7b810-9dad-41d1-80b4-00c04fd430c8"}',
        )
    )
    removed = await gateway.call(
        ModelToolCall(
            id="call_remove",
            name="schedule",
            arguments=json.dumps(
                {"action": "remove", "job_id": str(JOB_UUID), "title": None}
            ),
        )
    )

    assert invalid.status == "error"
    assert invalid.content == "Invalid arguments for schedule."
    assert unknown.status == "error"
    assert unknown.content == "Schedule Job was not found."
    assert system.status == "error"
    assert system.content == "Schedule Job was not found."
    assert removed.status == "success"
    assert removed.confirmation is None
    assert json.loads(removed.content) == {
        "action": "remove",
        "job": {
            "job_id": str(JOB_UUID),
            "message": "Remove me",
            "schedule": {"type": "at", "at_time": "2026-08-07T13:00:00.000+00:00"},
        },
    }
    assert await store.snapshot() == (hidden,)


@pytest.mark.asyncio
async def test_schedule_tool_supports_add_list_and_remove_without_context_guard(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(
        contextual_tool(ScheduleTool,
            schedule_service=_service(store),
            now=lambda: NOW,
            new_uuid=lambda: JOB_UUID,
        )
    )

    invalid_add = await gateway.call(
        ModelToolCall(
            id="call_invalid_add",
            name="schedule",
            arguments='{"action":"add","every_seconds":60}',
        )
    )
    add = await gateway.call(
        ModelToolCall(
            id="call_add",
            name="schedule",
            arguments='{"action":"add","message":"Created","every_seconds":60}',
        )
    )
    listed = await gateway.call(
        ModelToolCall(id="call_list", name="schedule", arguments='{"action":"list"}')
    )
    removed = await gateway.call(
        ModelToolCall(
            id="call_remove",
            name="schedule",
            arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
        )
    )

    assert invalid_add.status == "error"
    assert invalid_add.confirmation is None
    assert add.status == "success"
    assert add.confirmation is None
    assert json.loads(add.content)["action"] == "add"
    assert listed.status == "success"
    assert removed.status == "success"
    assert await store.snapshot() == ()
