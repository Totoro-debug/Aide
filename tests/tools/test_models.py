import pytest

from aide.agent.tools.base import ArtifactReference
from aide.agent.tools.tool_gateway import ModelToolCall, ToolResult

SESSION_ID = "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"


def test_normalized_tool_result_serializes_the_exact_artifact_shape() -> None:
    result = ToolResult(
        tool_call_id="call_123",
        name="read_file",
        status="success",
        content="preview",
        artifact=ArtifactReference(
            path=(
                ".aide/artifacts/20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000/call_123.txt"
            ),
            total_chars=73421,
            preview_chars=2000,
        ),
    )

    assert result.to_dict() == {
        "tool_call_id": "call_123",
        "name": "read_file",
        "status": "success",
        "content": "preview",
        "artifact": {
            "path": (
                ".aide/artifacts/20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000/call_123.txt"
            ),
            "total_chars": 73421,
            "preview_chars": 2000,
        },
    }


def test_model_tool_call_preserves_raw_json_argument_text() -> None:
    raw_arguments = '{"path":"CONTEXT.md","offset":1}'
    call = ModelToolCall(id="call_123", name="read_file", arguments=raw_arguments)

    assert call.arguments is raw_arguments
    assert call.to_dict() == {
        "id": "call_123",
        "name": "read_file",
        "arguments": raw_arguments,
    }


def test_tool_records_reject_values_outside_the_normalized_contract() -> None:
    with pytest.raises(ValueError, match="artifact path contract"):
        ArtifactReference(path="C:/secret.txt", total_chars=10, preview_chars=10)
    with pytest.raises(ValueError, match="preview_chars"):
        ArtifactReference(
            path=f".aide/artifacts/{SESSION_ID}/call.txt",
            total_chars=10,
            preview_chars=11,
        )

    result = ToolResult(
        tool_call_id="call_123",
        name="read_file",
        status="error",
        content="Tool failed.",
        artifact=None,
    )
    assert result.to_dict() == {
        "tool_call_id": "call_123",
        "name": "read_file",
        "status": "error",
        "content": "Tool failed.",
        "artifact": None,
    }


@pytest.mark.parametrize(
    "path",
    [
        f"C:/agent/artifacts/{SESSION_ID}/call_123.txt",
        f"artifacts/../{SESSION_ID}/call_123.txt",
        f"artifacts\\{SESSION_ID}\\call_123.txt",
        f"results/{SESSION_ID}/call_123.txt",
        "artifacts/session-id/call_123.txt",
        f".aide/artifacts/{SESSION_ID}/call:123.txt",
        f".aide/artifacts/{SESSION_ID}/.txt",
        f".aide/artifacts/{SESSION_ID}/%43.txt",
    ],
)
def test_artifact_reference_rejects_paths_outside_the_exact_persisted_shape(path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactReference(path=path, total_chars=10, preview_chars=10)


def test_artifact_reference_accepts_legal_ascii_filename() -> None:
    reference = ArtifactReference(
        path=f".aide/artifacts/{SESSION_ID}/CON.txt",
        total_chars=10,
        preview_chars=10,
    )

    assert reference.path.endswith("/CON.txt")
    with pytest.raises(ValueError, match="artifact path contract"):
        ArtifactReference(
            path=f".aide/artifacts/{SESSION_ID}/%43.txt",
            total_chars=10,
            preview_chars=10,
        )
