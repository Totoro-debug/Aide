"""Contract evidence for the service protocol before transport implementation."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator, ValidationError


def _schema() -> dict[str, Any]:
    schema_path = resources.files("myclaw.service.protocol").joinpath("v1.schema.json")
    return cast(dict[str, Any], json.loads(schema_path.read_text(encoding="utf-8")))


def _validator(definition: str) -> Draft202012Validator:
    schema = _schema()
    schema["$ref"] = f"#/$defs/{definition}"
    return Draft202012Validator(schema)


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
    } == {"replace", "clear"}


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
        "fields": {"runtime.compact_ratio": 0.7},
        "secrets": {"models.providers.primary.api_key": {"action": "replace", "value": "new"}},
    }
    validator = _validator("config_patch")
    validator.validate(patch)
    with pytest.raises(ValidationError):
        validator.validate({**patch, "revision": ""})
    with pytest.raises(ValidationError):
        validator.validate({**patch, "secrets": {"key": {"action": "clear", "value": "old"}}})


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
