from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Annotated, Any, ClassVar, cast
from uuid import UUID

import pytest

from omni.agent.tools.base import BaseTool, ToolError, ToolParam
from omni.agent.tools.context import ToolRunContext
from omni.agent.tools.permission import ToolAuthorizationSession, ToolInvocationFacts


class _RepresentativeTool(BaseTool):
    name = "representative"
    description = "Exercise the complete supported Tool declaration surface."
    required = ("text", "empty_text")

    text: Annotated[
        str,
        ToolParam(
            description="Text with explicit constraints.",
            min_length=1,
            max_length=40,
            format="hostname",
        ),
    ]
    empty_text: str
    count: Annotated[
        int | None,
        ToolParam(description="Optional bounded count.", minimum=0, maximum=10),
    ] = None
    enabled: bool = False
    _private: str = "hidden"
    cache: ClassVar[str] = "hidden"

    async def execute(
        self,
        *,
        text: str,
        empty_text: str,
        count: int | None,
        enabled: bool,
    ) -> str:
        return f"{text}:{empty_text}:{count}:{enabled}"


@pytest.mark.asyncio
async def test_default_hooks_accept_a_property_named_self() -> None:
    tool = _RepresentativeTool()
    assert tool.validate_arguments(**{"self": "ok"}) is None


def test_base_tool_generates_complete_openai_function_calling_schema() -> None:
    assert _RepresentativeTool().to_schema() == {
        "type": "function",
        "function": {
            "name": "representative",
            "description": "Exercise the complete supported Tool declaration surface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text with explicit constraints.",
                        "minLength": 1,
                        "maxLength": 40,
                        "format": "hostname",
                    },
                    "empty_text": {"type": "string"},
                    "count": {
                        "type": ["integer", "null"],
                        "description": "Optional bounded count.",
                        "minimum": 0,
                        "maximum": 10,
                        "default": None,
                    },
                    "enabled": {"type": "boolean", "default": False},
                },
                "required": ["text", "empty_text"],
            },
        },
    }


def test_schema_uses_only_direct_public_parameter_annotations() -> None:
    class ParentTool(BaseTool):
        name = "parent"
        description = "Parent Tool."
        inherited: str = "not inherited"

        async def execute(self, *, inherited: str) -> str:
            return inherited

    class ChildTool(ParentTool):
        name = "child"
        description = "Child Tool."
        required = ("direct",)
        direct: int
        metadata: ClassVar[str] = "not a parameter"

        async def execute(self, *, direct: int) -> str:  # type: ignore[override]
            return str(direct)

    properties = ChildTool().to_schema()["function"]
    assert isinstance(properties, dict)
    parameters = properties["parameters"]
    assert isinstance(parameters, dict)
    assert parameters["properties"] == {"direct": {"type": "integer"}}


def test_required_string_is_not_implicitly_nonempty() -> None:
    schema = _RepresentativeTool().to_schema()
    function = schema["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert properties["empty_text"] == {"type": "string"}


def test_schema_exports_are_detached() -> None:
    tool = _RepresentativeTool()
    first = tool.to_schema()
    first_function = first["function"]
    assert isinstance(first_function, dict)
    first_function["name"] = "mutated"
    first_parameters = first_function["parameters"]
    assert isinstance(first_parameters, dict)
    first_required = first_parameters["required"]
    assert isinstance(first_required, list)
    first_required.append("mutated")

    second = tool.to_schema()
    assert second["function"] != first_function
    assert second == tool.to_schema()


def test_tool_schema_rejects_unsupported_parameter_annotations() -> None:
    class UnsupportedTool(BaseTool):
        name = "unsupported"
        description = "Unsupported parameter."
        values: tuple[str, ...] = ()

        async def execute(self, *, values: tuple[str, ...]) -> str:
            return ",".join(values)

    with pytest.raises(TypeError, match="unsupported annotation"):
        UnsupportedTool().to_schema()

    class FloatTool(BaseTool):
        name = "float"
        description = "Unsupported float parameter."
        ratio: float

        async def execute(self, *, ratio: float) -> str:
            return str(ratio)

    with pytest.raises(TypeError, match="unsupported annotation"):
        FloatTool().to_schema()


def test_tool_error_contains_only_a_public_safe_message() -> None:
    error = ToolError("The path could not be read.")

    assert error.message == "The path could not be read."
    assert str(error) == error.message
    assert not hasattr(error, "code")


@pytest.mark.asyncio
async def test_base_tool_prepare_returns_normalized_arguments_and_structured_facts() -> None:
    observed: list[tuple[str, int, bool]] = []

    class PreparingTool(BaseTool):
        name = "preparing"
        description = "Exercise the public preparation interface."
        required = ("count",)

        count: int
        enabled: bool = False

        def validate_arguments(  # type: ignore[override]
            self, *, count: int, enabled: bool
        ) -> None:
            observed.append(("validation", count, enabled))

        async def execute(self, *, count: int, enabled: bool) -> str:
            return f"{count}:{enabled}"

    facts = await PreparingTool().prepare({"count": "3"})

    assert isinstance(facts, ToolInvocationFacts)
    assert facts.normalized_arguments == {"count": 3, "enabled": False}
    assert facts.tool_name == "preparing"
    assert observed == [("validation", 3, False)]


@pytest.mark.asyncio
async def test_base_tool_prepare_casts_integer_text() -> None:
    class IntegerTool(BaseTool):
        name = "integer"
        description = "Cast an integer."
        required = ("count",)
        count: int

        async def execute(self, *, count: int) -> str:
            return str(count)

    facts = await IntegerTool().prepare({"count": "7"})

    assert facts.normalized_arguments == {"count": 7}


@pytest.mark.asyncio
async def test_base_tool_prepare_casts_boolean_text() -> None:
    class BooleanTool(BaseTool):
        name = "boolean"
        description = "Cast a boolean."
        required = ("enabled",)
        enabled: bool

        async def execute(self, *, enabled: bool) -> str:
            return str(enabled)

    facts = await BooleanTool().prepare({"enabled": "true"})

    assert facts.normalized_arguments == {"enabled": True}


@pytest.mark.asyncio
async def test_base_tool_prepare_applies_declared_defaults() -> None:
    class DefaultTool(BaseTool):
        name = "default"
        description = "Apply a default."
        value: str = "fallback"

        async def execute(self, *, value: str) -> str:
            return value

    facts = await DefaultTool().prepare({})

    assert facts.normalized_arguments == {"value": "fallback"}


@pytest.mark.asyncio
async def test_base_tool_prepare_filters_unknown_fields() -> None:
    class FilterTool(BaseTool):
        name = "filter"
        description = "Filter unknown fields."
        value: str

        async def execute(self, *, value: str) -> str:
            return value

    facts = await FilterTool().prepare({"value": "kept", "extra": "removed"})

    assert facts.normalized_arguments == {"value": "kept"}


@pytest.mark.asyncio
async def test_base_tool_prepare_rejects_missing_required_argument() -> None:
    class RequiredTool(BaseTool):
        name = "required"
        description = "Require a value."
        required = ("value",)
        value: str

        async def execute(self, *, value: str) -> str:
            return value

    with pytest.raises(ToolError, match="value"):
        await RequiredTool().prepare({})


@pytest.mark.asyncio
async def test_base_tool_prepare_rejects_invalid_integer() -> None:
    class IntegerTool(BaseTool):
        name = "invalid_integer"
        description = "Reject an invalid integer."
        required = ("count",)
        count: int

        async def execute(self, *, count: int) -> str:
            return str(count)

    with pytest.raises(ToolError, match="integer"):
        await IntegerTool().prepare({"count": "not-an-integer"})


@pytest.mark.asyncio
async def test_base_tool_prepare_rejects_schema_constraints() -> None:
    class BoundedTool(BaseTool):
        name = "bounded"
        description = "Enforce a lower bound."
        count: Annotated[int, ToolParam(minimum=1)]

        async def execute(self, *, count: int) -> str:
            return str(count)

    with pytest.raises(ToolError, match="greater than or equal to 1"):
        await BoundedTool().prepare({"count": 0})


@pytest.mark.asyncio
async def test_base_tool_prepare_rejects_string_constraints() -> None:
    class NamedTool(BaseTool):
        name = "named"
        description = "Enforce a minimum name length."
        value: Annotated[str, ToolParam(min_length=2)]

        async def execute(self, *, value: str) -> str:
            return value

    with pytest.raises(ToolError, match="at least 2 characters"):
        await NamedTool().prepare({"value": "x"})


@pytest.mark.asyncio
async def test_base_tool_prepare_runs_validation_after_cast() -> None:
    observed: list[int] = []

    class ValidatingTool(BaseTool):
        name = "validating"
        description = "Observe normalized validation."
        required = ("count",)
        count: int

        def validate_arguments(self, *, count: int) -> None:  # type: ignore[override]
            observed.append(count)

        async def execute(self, *, count: int) -> str:
            return str(count)

    await ValidatingTool().prepare({"count": "9"})

    assert observed == [9]


@pytest.mark.asyncio
async def test_base_tool_prepare_collects_facts_after_validation() -> None:
    observed: list[tuple[str, int]] = []

    class SafeTool(BaseTool):
        name = "safe"
        description = "Observe safety ordering."
        required = ("count",)
        count: int

        def validate_arguments(self, *, count: int) -> None:  # type: ignore[override]
            observed.append(("validate", count))

        def build_invocation_facts(self, prepared_arguments: dict[str, Any]) -> ToolInvocationFacts:
            observed.append(("facts", prepared_arguments["count"]))
            return super().build_invocation_facts(prepared_arguments)

        async def execute(self, *, count: int) -> str:
            return str(count)

    await SafeTool().prepare({"count": "4"})

    assert observed == [("validate", 4), ("facts", 4)]


@pytest.mark.asyncio
async def test_base_tool_prepare_propagates_cancellation_from_argument_preparation() -> None:
    class CancelledPreparationTool(BaseTool):
        name = "cancelled_preparation"
        description = "Cancel while preparing."

        async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
            del arguments
            raise asyncio.CancelledError

        async def execute(self) -> str:
            return "unreachable"

    with pytest.raises(asyncio.CancelledError):
        await CancelledPreparationTool().prepare({})


@pytest.mark.asyncio
async def test_base_tool_prepare_propagates_cancellation_from_fact_collection() -> None:
    class CancelledFactsTool(BaseTool):
        name = "cancelled_facts"
        description = "Cancel while collecting facts."

        async def collect_invocation_facts(
            self,
            prepared_arguments: dict[str, Any],
        ) -> ToolInvocationFacts:
            del prepared_arguments
            raise asyncio.CancelledError

        async def execute(self) -> str:
            return "unreachable"

    with pytest.raises(asyncio.CancelledError):
        await CancelledFactsTool().prepare({})


@pytest.mark.asyncio
async def test_default_execute_prepared_expands_prepared_keywords() -> None:
    class KeywordTool(BaseTool):
        name = "keyword"
        description = "Expand prepared keywords."
        required = ("value",)
        value: str

        async def execute(self, *, value: str) -> str:
            return f"received:{value}"

    result = await KeywordTool().execute_prepared({"value": "payload"})

    assert result == "received:payload"


@pytest.mark.asyncio
async def test_default_execute_prepared_expands_multiple_keywords() -> None:
    class MultipleKeywordTool(BaseTool):
        name = "multiple_keyword"
        description = "Expand multiple prepared keywords."
        left: str
        right: int

        async def execute(self, *, left: str, right: int) -> str:
            return f"{left}:{right}"

    result = await MultipleKeywordTool().execute_prepared({"left": "left", "right": 3})

    assert result == "left:3"


@pytest.mark.asyncio
async def test_default_execute_prepared_propagates_execute_result() -> None:
    class ResultTool(BaseTool):
        name = "result"
        description = "Return the prepared execution result."
        required = ("value",)
        value: str

        async def execute(self, *, value: str) -> str:
            return f"complete:{value}"

    result = await ResultTool().execute_prepared({"value": "payload"})

    assert result == "complete:payload"


@pytest.mark.asyncio
async def test_tool_with_only_execute_prepared_is_concrete() -> None:
    class PreparedOnlyTool(BaseTool):
        name = "prepared_only"
        description = "Implement only prepared execution."
        parameters: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        async def execute_prepared(self, arguments: dict[str, Any]) -> str:
            assert arguments == {}
            return "prepared only"

    tool = PreparedOnlyTool()

    assert await tool.execute_prepared({}) == "prepared only"


def test_tool_without_a_declaration_or_execution_remains_abstract() -> None:
    class IncompleteTool(BaseTool):
        name = "incomplete"
        description = "An incomplete Tool."

    assert inspect.isabstract(IncompleteTool)


def test_tool_with_parameters_but_without_execution_remains_abstract() -> None:
    class ParametersOnlyTool(BaseTool):
        name = "parameters_only"
        description = "A Tool without execution."
        value: str

    assert inspect.isabstract(ParametersOnlyTool)
    with pytest.raises(TypeError, match="abstract method 'execute'"):
        cast(Any, ParametersOnlyTool)()


def test_tool_with_plain_mixin_but_without_execution_remains_abstract() -> None:
    class PlainMixin:
        pass

    class IncompleteTool(BaseTool, PlainMixin):
        name = "incomplete_mixin"
        description = "A Tool whose mixin does not provide execution."
        parameters: ClassVar[dict[str, Any]] = {
            "type": "object",
            "properties": {},
        }

    assert inspect.isabstract(IncompleteTool)
    with pytest.raises(TypeError, match="abstract method 'execute'"):
        cast(Any, IncompleteTool)()


def test_tool_cannot_override_schema_projection() -> None:
    with pytest.raises(TypeError, match="cannot override"):

        class CustomSchemaTool(BaseTool):
            name = "custom_schema"
            description = "Attempt to override schema projection."
            parameters: ClassVar[dict[str, Any]] = {
                "type": "object",
                "properties": {},
            }

            def to_schema(self) -> dict[str, Any]:  # type: ignore[misc]
                return {}

            async def execute(self) -> str:
                return "unreachable"


def test_base_tool_result_handler_writes_a_bounded_workspace_artifact(
    tmp_path: Path,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    workspace = workspace_path
    content = "0123456789" * 80
    limit = 160

    output = _RepresentativeTool().handle_result(
        content,
        workspace=workspace,
        session_id="session-1",
        tool_call_id="call-1",
        limit=limit,
    )

    assert output.artifact is not None
    assert output.artifact.path == ".omni/artifacts/session-1/call-1.txt"
    marker = "\n\n...[truncated; full result stored at .omni/artifacts/session-1/call-1.txt]"
    assert output.content == content[: limit - len(marker)] + marker
    assert len(output.content) == limit
    assert output.artifact.to_dict() == {
        "path": ".omni/artifacts/session-1/call-1.txt",
        "total_chars": len(content),
        "preview_chars": limit - len(marker),
    }
    assert (workspace_path / ".omni" / "artifacts" / "session-1" / "call-1.txt").read_text(
        encoding="utf-8"
    ) == content


def test_base_tool_result_handler_keeps_exact_limit_inline_and_overwrites_targets(
    tmp_path: Path,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    workspace = workspace_path
    target = workspace_path / ".omni" / "artifacts" / "session-1" / "call-1.txt"
    target.parent.mkdir(parents=True)
    target.write_text("old", encoding="utf-8")

    exact = _RepresentativeTool().handle_result(
        "x" * 12,
        workspace=workspace,
        session_id="session-1",
        tool_call_id="call-1",
        limit=12,
    )
    overwritten = _RepresentativeTool().handle_result(
        "new oversized content",
        workspace=workspace,
        session_id="session-1",
        tool_call_id="call-1",
        limit=12,
    )

    assert exact.content == "x" * 12
    assert exact.artifact is None
    assert overwritten.artifact is not None
    assert target.read_text(encoding="utf-8") == "new oversized content"


def test_base_tool_result_handler_uses_uuid_for_an_illegal_tool_call_id(
    tmp_path: Path,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    workspace = workspace_path

    output = _RepresentativeTool().handle_result(
        "oversized",
        workspace=workspace,
        session_id="session-1",
        tool_call_id="../unsafe id",
        limit=4,
    )

    assert output.artifact is not None
    artifact_name = output.artifact.path.rsplit("/", maxsplit=1)[-1].removesuffix(".txt")
    assert UUID(artifact_name).version == 4
    assert (workspace_path / output.artifact.path).read_text(encoding="utf-8") == ("oversized")


def test_base_tool_result_handler_retains_success_when_artifact_write_fails(
    tmp_path: Path,
) -> None:
    workspace_path = tmp_path / "workspace"
    workspace_path.mkdir()
    workspace = workspace_path
    failed_target = workspace_path / ".omni" / "artifacts" / "session-1" / "failed.txt"
    failed_target.mkdir(parents=True)

    output = _RepresentativeTool().handle_result(
        "private oversized result" * 4,
        workspace=workspace,
        session_id="session-1",
        tool_call_id="failed",
        limit=40,
    )

    assert output.artifact is None
    assert len(output.content) <= 40
    assert "artifact write failed" in output.content.lower()


@pytest.mark.asyncio
async def test_context_execution_implementation_is_a_concrete_tool(tmp_path: Path) -> None:
    class ContextTool(BaseTool):
        name = "context_fixture"
        description = "Return the explicit Workspace."
        _contextual = True
        parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

        async def execute_authorized_for_context(
            self, arguments: dict[str, Any], authorization: ToolAuthorizationSession,
            *, context: ToolRunContext, **kwargs: Any,
        ) -> str:
            del arguments, authorization, kwargs
            return str(context.workspace)

    from omni.agent.tools.tool_gateway import ModelToolCall, ToolGateway

    assert not inspect.isabstract(ContextTool)
    tool = ContextTool()
    gateway = ToolGateway._for_memory((tool,), tool_context=ToolRunContext(workspace=tmp_path))
    result = await gateway.call(ModelToolCall("context", tool.name, "{}"))
    assert result.status == "success"
    assert result.content == str(tmp_path)


@pytest.mark.parametrize("default_alias", [False, True])
def test_context_declaration_without_execution_remains_abstract(default_alias: bool) -> None:
    class ContextOnlyTool(BaseTool):
        name = "context_only"
        description = "An incomplete context Tool."
        _contextual = True
        parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}
        if default_alias:
            execute_authorized_for_context = BaseTool.execute_authorized_for_context

    assert inspect.isabstract(ContextOnlyTool)
    with pytest.raises(TypeError, match="abstract method 'execute'"):
        cast(Any, ContextOnlyTool)()
