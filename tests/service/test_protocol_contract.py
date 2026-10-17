"""Contract evidence for the service protocol before transport implementation."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, NotRequired, cast, get_origin, get_type_hints

import pytest
from jsonschema import Draft202012Validator, ValidationError

from aide.agent.memory.dream import DreamResult
from aide.management.commands import ManagementCommandResult
from aide.management.service import RuntimeStatus
from aide.provider.session_configuration import REASONING_EFFORT_LEVELS
from aide.service.client import _management_result
from aide.service.runtime.projections import _encode_management_result
from aide.trace import types as trace_types
from aide.utils.errors import ErrorInfo
from tests.fixtures.trace import trace_contract_fixtures


def _schema() -> dict[str, Any]:
    schema_path = resources.files("aide.service.protocol").joinpath("v1.schema.json")
    return cast(dict[str, Any], json.loads(schema_path.read_text(encoding="utf-8")))


def _validator(definition: str) -> Draft202012Validator:
    schema = _schema()
    schema["$ref"] = f"#/$defs/{definition}"
    return Draft202012Validator(schema)


@pytest.mark.parametrize("field", ("tool_keywords", "tool_keyword_rows"))
def test_keyword_patch_contract_accepts_mixed_items_but_requires_arrays(field: str) -> None:
    validator = _validator("config_patch_mcp_fields")
    mixed: list[object] = [None, 1, True, {}, "搜索", " Pull Request "]
    value = (
        {"search": mixed} if field == "tool_keywords" else [{"name": "search", "keywords": mixed}]
    )

    validator.validate({field: value})
    invalid = (
        {"search": "not-array"}
        if field == "tool_keywords"
        else [{"name": "search", "keywords": "not-array"}]
    )
    with pytest.raises(ValidationError):
        validator.validate({field: invalid})


@pytest.mark.parametrize(
    ("definition", "value"),
    [
        (
            "submit_user_input_result",
            {
                "kind": "conversation_input",
                "run_id": "run-1",
                "live_state": None,
            },
        ),
        (
            "submit_user_input_result",
            {
                "kind": "management",
                "management_result": _encode_management_result(
                    ManagementCommandResult(handled=True, output="status")
                ),
            },
        ),
        (
            "recall_queued_inputs_result",
            {
                "recalled_inputs": [{"run_id": "run-b", "text": " original B\n"}],
                "live_state": None,
            },
        ),
        (
            "input_capabilities_response",
            {
                "management_commands": ["/status"],
                "skill_metadata": [],
            },
        ),
    ],
)
def test_input_operation_dtos_reject_incomplete_results(
    definition: str, value: dict[str, object]
) -> None:
    validator = _validator(definition)
    validator.validate(value)
    for field in value:
        with pytest.raises(ValidationError):
            validator.validate({key: item for key, item in value.items() if key != field})


@pytest.mark.parametrize(
    ("definition", "selection"),
    [
        ("runtime_status_request", {}),
        ("runtime_permission_request", {"permission_level": "read-only"}),
        ("runtime_effort_request", {"effort": "high"}),
    ],
)
def test_typed_runtime_requests_require_claim_and_reject_slash_commands(
    definition: str, selection: dict[str, str]
) -> None:
    request = {
        "request_id": "runtime-1",
        "current_session_id": "session-1",
        "claim_version": 1,
        **selection,
    }
    validator = _validator(definition)
    validator.validate(request)
    for field in request:
        with pytest.raises(ValidationError):
            validator.validate({key: value for key, value in request.items() if key != field})
    with pytest.raises(ValidationError):
        validator.validate({**request, "command": "/status"})
    with pytest.raises(ValidationError):
        validator.validate({**request, "claim_version": 0})
    for field in selection:
        with pytest.raises(ValidationError):
            validator.validate({**request, field: "unsupported"})


@pytest.mark.parametrize(
    ("definition", "action"),
    [
        ("runtime_memory_request", "memory"),
        ("runtime_dream_request", "dream"),
        ("runtime_skills_reload_request", "skills/reload"),
    ],
)
def test_web_memory_dream_and_skill_requests_are_typed_and_claim_scoped(
    definition: str,
    action: str,
) -> None:
    request = {
        "request_id": "management-1",
        "current_session_id": "session-1",
        "claim_version": 1,
    }
    validator = _validator(definition)
    validator.validate(request)
    with pytest.raises(ValidationError):
        validator.validate({**request, "command": f"/{action}"})
    with pytest.raises(ValidationError):
        validator.validate({key: value for key, value in request.items() if key != "claim_version"})


@pytest.mark.parametrize("cached", [None, 0, 250])
def test_runtime_management_wire_results_validate_and_remote_decode_nullable_selections(
    cached: int | None,
) -> None:
    status = RuntimeStatus(
        version="test",
        chat_model="primary/small-model",
        chat_reasoning_effort="high",
        uptime_seconds=2,
        context_window=1000,
        max_output=100,
        available_context=900,
        compact_ratio=0.8,
        compact_context_window=720,
        projected_next_request_tokens=300,
        projection_source="reported_delta",
        input_budget_used_percent=300 / 900 * 100,
        session_message_count=4,
        last_compacted=0,
        cumulative_usage={"input_tokens": 300, "output_tokens": 10},
        historical_usage={"input_tokens": 300, "cached_input_tokens": cached, "output_tokens": 10},
        current_permission_level="read-only",
        schedule={"status": "available", "active_job_count": 1},
    )
    results = (
        ManagementCommandResult(handled=True, output=None, status_view=status),
        ManagementCommandResult(handled=True, output=None, effort_selection="high"),
        ManagementCommandResult(handled=True, output=None, permission_selection="read-only"),
        ManagementCommandResult(
            handled=True, output="Chat reasoning effort: max", published_effort="max"
        ),
        ManagementCommandResult(
            handled=True,
            output="Foreground permission level: full-access",
            published_permission_level="full-access",
        ),
        ManagementCommandResult(
            handled=True,
            output=None,
            memory_content="# Current memory\n",
        ),
        ManagementCommandResult(
            handled=True,
            output=None,
            dream_result=DreamResult(
                status="Memory Task complete.",
                processed_count=2,
                memory_updated=True,
                cursor=4,
            ),
        ),
        ManagementCommandResult(
            handled=True,
            output=None,
            skill_metadata=(),
            management_error=ErrorInfo("skill_reload_failed", "Skill reload failed."),
        ),
        ManagementCommandResult(handled=True, output="config_invalid: Selection is invalid."),
    )
    validator = _validator("management_response")
    for result in results:
        encoded = _encode_management_result(result)
        validator.validate({"request_id": "runtime-1", "result": encoded})
        decoded = _management_result(encoded)
        assert decoded.handled == result.handled
        assert decoded.output == result.output
        assert decoded.status_view == result.status_view
        assert decoded.effort_selection == result.effort_selection
        assert decoded.permission_selection == result.permission_selection
        assert decoded.published_effort == result.published_effort
        assert decoded.published_permission_level == result.published_permission_level
        assert decoded.memory_content == result.memory_content
        assert decoded.dream_result == result.dream_result
        assert decoded.skill_metadata == result.skill_metadata
        assert decoded.management_error == result.management_error
    encoded_status = _encode_management_result(results[0])
    for invalid in ("unsupported", True, 1):
        with pytest.raises(ValidationError):
            validator.validate(
                {
                    "request_id": "runtime-1",
                    "result": {**encoded_status, "published_effort": invalid},
                }
            )
    with pytest.raises(ValidationError):
        validator.validate(
            {"request_id": "runtime-1", "result": {**encoded_status, "credential": "secret"}}
        )
    with pytest.raises(ValidationError):
        validator.validate(
            {
                "request_id": "runtime-1",
                "result": {
                    **encoded_status,
                    "status_view": {**status.to_dict(), "messages": ["private"]},
                },
            }
        )


def test_versioned_protocol_schema_is_packaged_and_keeps_secrets_write_only() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    definitions = schema["$defs"]

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert definitions["event"]["properties"]["protocol_version"] == {"const": 1}
    assert set(definitions["event"]["required"]) == {
        "protocol_version",
        "service_instance_id",
        "stream_id",
        "seq",
        "type",
        "workspace_id",
        "project_id",
        "session_id",
        "run_id",
        "payload",
    }
    assert definitions["error"]["additionalProperties"] is False
    assert "value" not in definitions["redacted_secret"]["properties"]
    assert {
        choice["properties"]["action"]["const"]
        for choice in definitions["config_secret_change"]["oneOf"]
    } == {"keep", "replace", "clear"}


def test_event_and_error_envelopes_reject_missing_identity_and_unknown_fields() -> None:
    event = {
        "protocol_version": 1,
        "service_instance_id": "instance-1",
        "stream_id": "stream-1",
        "seq": 1,
        "type": "run.completed",
        "workspace_id": "workspace-1",
        "project_id": None,
        "session_id": "session-1",
        "run_id": "run-1",
        "payload": {},
    }
    event_validator = _validator("event")
    event_validator.validate(event)
    with pytest.raises(ValidationError):
        event_validator.validate({**event, "seq": 0})
    with pytest.raises(ValidationError):
        event_validator.validate({**event, "credential": "must-not-be-published"})

    error = {
        "code": "session_claimed",
        "message": "Session is in use.",
        "field_errors": {},
        "retryable": True,
        "request_id": "request-1",
    }
    error_validator = _validator("error")
    error_validator.validate(error)
    with pytest.raises(ValidationError):
        error_validator.validate({**error, "raw_exception": "secret"})


def test_config_patch_requires_explicit_secret_operation_and_revision() -> None:
    patch = {
        "request_id": "request-1",
        "revision": "sha256:old",
        "fields": {"runtime": {"compact_ratio": 0.7}},
        "secrets": {"models.providers.primary.api_key": {"action": "replace", "value": "new"}},
    }
    validator = _validator("config_patch")
    validator.validate(patch)
    with pytest.raises(ValidationError):
        validator.validate({**patch, "revision": ""})
    with pytest.raises(ValidationError):
        validator.validate({**patch, "secrets": {"key": {"action": "clear", "value": "old"}}})


def test_session_deletion_response_is_explicit_and_does_not_publish_credentials() -> None:
    deletion = {
        "request_id": "delete-1",
        "project_id": "project-1",
        "workspace_id": "workspace-1",
        "session_id": "session-1",
        "deleted": True,
    }
    validator = _validator("session_deletion")
    validator.validate(deletion)
    with pytest.raises(ValidationError):
        validator.validate({**deletion, "reconnect_credential": "secret"})


def test_project_http_shapes_are_versioned_and_preserve_saved_job_review_data() -> None:
    project = {
        "project_id": "project-1",
        "path": "C:/Projects/example",
        "name": "example",
        "schedule_state": "awaiting_resume",
        "available": True,
        "saved_jobs": [
            {
                "job_id": "job-1",
                "title": "Review",
                "schedule": {
                    "kind": "every",
                    "at_time": None,
                    "every_seconds": 3600,
                    "cron_expr": None,
                    "timezone": None,
                },
                "due_at": "2026-10-01T00:00:00+00:00",
                "review_status": "upcoming",
            }
        ],
        "schedule_status": {
            "admitted": False,
            "status": "available",
            "active_job_count": 0,
        },
    }
    projects_validator = _validator("projects_response")
    projects_validator.validate({"projects": [project]})
    projects_validator.validate(
        {
            "projects": [
                {
                    **project,
                    "schedule_state": "removing",
                    "removal_operation_id": "operation-1",
                    "removal_error": "Project work could not be stopped; retry removal.",
                }
            ]
        }
    )
    with pytest.raises(ValidationError):
        projects_validator.validate({"projects": [{**project, "credential": "secret"}]})

    removal_validator = _validator("project_removal")
    removal_validator.validate(
        {
            "request_id": "request-1",
            "project_id": "project-1",
            "operation_id": "operation-1",
            "status": "removing",
        }
    )
    with pytest.raises(ValidationError):
        removal_validator.validate(
            {
                "request_id": "request-1",
                "project_id": "project-1",
                "operation_id": "operation-1",
                "status": "removing",
                "path": "must-not-be-returned",
            }
        )

    status_validator = _validator("project_removal_status")
    status_validator.validate(
        {"project_id": "project-1", "operation_id": "operation-1", "status": "completed"}
    )
    with pytest.raises(ValidationError):
        status_validator.validate(
            {"project_id": "project-1", "operation_id": "operation-1", "status": "unknown"}
        )

    registration_validator = _validator("project_registration")
    registration_validator.validate(
        {
            "request_id": "request-1",
            "project_id": "project-1",
            "workspace_id": "workspace-1",
            "schedule_state": "awaiting_resume",
            "saved_jobs": project["saved_jobs"],
        }
    )


def test_client_commands_require_claim_identity_and_typed_payloads() -> None:
    command = {
        "request_id": "request-1",
        "type": "input",
        "workspace_id": "workspace-1",
        "session_id": "session-1",
        "claim_version": 2,
        "payload": {"text": "Inspect status."},
    }
    validator = _validator("client_command")
    validator.validate(command)
    with pytest.raises(ValidationError):
        validator.validate({**command, "claim_version": None})
    with pytest.raises(ValidationError):
        validator.validate({**command, "payload": {"text": ""}})

    validator.validate({**command, "type": "recall_queued_inputs", "payload": {}})
    with pytest.raises(ValidationError):
        validator.validate(
            {**command, "type": "recall_queued_inputs", "claim_version": None, "payload": {}}
        )

    decision = {
        **command,
        "type": "confirmation_decide",
        "session_id": None,
        "claim_version": None,
        "payload": {"token": "confirmation-1", "decision": "declined"},
    }
    validator.validate(decision)
    with pytest.raises(ValidationError):
        validator.validate(
            {**decision, "payload": {"token": "confirmation-1", "decision": "later"}}
        )


def test_session_deletion_recovery_contract_never_contains_history_or_paths() -> None:
    identity = {"project_id": "project-1", "workspace_id": "workspace-1", "session_id": "session-1"}
    status = _validator("session_deletion_status")
    for state in ("deleted", "deleting", "present"):
        status.validate({**identity, "state": state})
    with pytest.raises(ValidationError):
        status.validate({**identity, "state": "deleted", "path": "user-file"})
    cleanup = _validator("session_deletion_claim")
    claim = {
        **identity,
        "request_id": "request-1",
        "claim": {
            "workspace_id": "workspace-1",
            "session_id": "session-1",
            "claim_version": 2,
            "reconnect_credential": "credential-1",
        },
    }
    cleanup.validate(claim)
    with pytest.raises(ValidationError):
        cleanup.validate({**claim, "snapshot": {"messages": []}})


def test_execution_trace_wire_fixtures_match_the_protocol_schema() -> None:
    fixtures = trace_contract_fixtures()

    for record in fixtures.records.values():
        _validator("trace_record").validate(record)
    _validator("trace_record_detail").validate(fixtures.detail)
    _validator("trace_page").validate(fixtures.page)
    _validator("trace_sessions_response").validate(fixtures.sessions)
    _validator("trace_stats_response").validate(fixtures.stats)


def test_execution_trace_schema_rejects_unknown_status_and_ambiguous_usage() -> None:
    record = trace_contract_fixtures().records["success"]
    record_validator = _validator("trace_record")

    with pytest.raises(ValidationError):
        record_validator.validate({**record, "status": "cancelled"})
    with pytest.raises(ValidationError):
        record_validator.validate({**record, "started_at": "2026-10-11T02:00:00+01:00"})
    with pytest.raises(ValidationError):
        record_validator.validate({**record, "status": "failed", "error": None})
    with pytest.raises(ValidationError):
        _validator("trace_token_usage").validate(
            {
                "state": "unknown",
                "input_tokens": 0,
                "output_tokens": None,
                "total_tokens": None,
                "cached_input_tokens": None,
                "cache_creation_input_tokens": None,
            }
        )
    detail = trace_contract_fixtures().detail
    with pytest.raises(ValidationError):
        _validator("trace_record_detail").validate(
            {**detail, "capture_issues": [{"state": "complete", "message": "", "source": None}]}
        )


@pytest.mark.parametrize(
    ("dto_name", "definition"),
    [
        ("TraceUsageDTO", "trace_token_usage"),
        ("TraceErrorDTO", "trace_error"),
        ("TraceRecordDTO", "trace_record"),
        ("TraceToolLinkDTO", "trace_tool_link"),
        ("TraceCaptureIssueDTO", "trace_capture_issue"),
        ("TraceContentDTO", "trace_content"),
        ("TraceRecordDetailDTO", "trace_record_detail"),
        ("TracePageDTO", "trace_page"),
        ("TraceSessionSummaryDTO", "trace_session_summary"),
        ("TraceSessionsResponseDTO", "trace_sessions_response"),
        ("TraceTokenTotalsDTO", "trace_token_totals"),
        ("TraceStatsTotalsDTO", "trace_stats_totals"),
        ("TraceBreakdownDTO", "trace_breakdown"),
        ("TraceErrorBreakdownDTO", "trace_error_breakdown"),
        ("TraceTimelinePointDTO", "trace_timeline_point"),
        ("TraceStatsResponseDTO", "trace_stats_response"),
    ],
)
def test_execution_trace_python_dto_fields_match_wire_contract(
    dto_name: str, definition: str
) -> None:
    dto = getattr(trace_types, dto_name)
    annotations = get_type_hints(dto, include_extras=True)
    wire = _schema()["$defs"][definition]

    assert set(annotations) == set(wire["properties"])
    required = {
        name
        for name, annotation in annotations.items()
        if dto.__total__ and get_origin(annotation) is not NotRequired
    }
    assert required == set(wire.get("required", ()))


@pytest.mark.parametrize(
    ("enum_name", "definition"),
    [
        ("TraceOrigin", "trace_origin"),
        ("TracePurpose", "trace_purpose"),
        ("TraceKind", "trace_kind"),
        ("TraceStatus", "trace_status"),
        ("TraceErrorCategory", "trace_error_category"),
        ("TraceCaptureState", "trace_capture_state"),
        ("TraceUsageState", "trace_usage_state"),
    ],
)
def test_execution_trace_python_enums_match_wire_contract(enum_name: str, definition: str) -> None:
    values = {item.value for item in getattr(trace_types, enum_name)}
    assert values == set(_schema()["$defs"][definition]["enum"])


def test_execution_trace_query_dtos_pin_filters_and_page_limits() -> None:
    _validator("trace_list_query").validate(
        {
            "workspace_key": "sha256:workspace",
            "kind": "model",
            "purpose": "task_framing",
            "from_at": "2026-10-10T00:00:00Z",
            "limit": 50,
        }
    )
    _validator("trace_stats_query").validate(
        {"workspace_key": "sha256:workspace", "purpose": "dream"}
    )
    _validator("trace_sessions_query").validate({"cursor": "cursor-1", "limit": 50})
    with pytest.raises(ValidationError):
        _validator("trace_list_query").validate({"limit": 201})


def test_reasoning_effort_levels_match_wire_contract() -> None:
    assert set(REASONING_EFFORT_LEVELS) == set(_schema()["$defs"]["reasoning_effort"]["enum"])
