"""The nominal BaseTool declaration contract."""

from __future__ import annotations

import inspect
import re
from abc import ABC, ABCMeta, update_abstractmethods
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from ipaddress import IPv6Address, ip_address
from pathlib import Path
from types import NoneType, UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Union,
    cast,
    final,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import uuid4

from loguru import logger

from omni.agent.tools.permission import (
    FileAccess,
    FileAccessRole,
    ToolAuthorizationSession,
    ToolInvocationFacts,
    canonicalize_file_access,
)
from omni.agent.tools.schema import Schema, ToolParam
from omni.utils.validation import require_nonnegative_int

_METADATA_NAMES = frozenset({"name", "description", "required", "parameters"})
_CLASS_VAR_ORIGIN: object = ClassVar
_MISSING = object()
_INVALID_SCHEMA_MARKER = "__schema_declaration_invalid__"
_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_ARTIFACT_SESSION_PATTERN = re.compile(r"^[^./\\]+$")
_DEFAULT_TRUNCATION_MARKER = "\n\n...[truncated]"
_ARTIFACT_WRITE_FAILURE_MARKER = "\n\n...[artifact write failed; full result was not stored]"
class ToolError(Exception):
    """An expected Tool failure whose message is safe to return to the model."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def resolve_tool_path(workspace: Path, requested: str | Path) -> Path:
    """Resolve a model-provided path using the current host's path semantics."""
    if not isinstance(requested, (str, Path)):
        raise TypeError("requested Tool path must be a string or Path")
    if not isinstance(workspace, Path):
        raise TypeError("Tool workspace must be a Path")
    try:
        root = workspace.resolve(strict=True)
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(str(error)) from error
    if not root.is_dir():
        raise ValueError("Tool Workspace is not a directory")
    return resolved


def normalize_public_ip(value: str) -> str:
    """Normalize one globally routable IPv4/IPv6 address or reject it."""
    if not isinstance(value, str) or "%" in value:
        raise ValueError("address is not a public IP")
    address = ip_address(value)
    if isinstance(address, IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if (
        not address.is_global
        or address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_unspecified
        or address.is_multicast
        or address.is_reserved
        or (isinstance(address, IPv6Address) and address.is_site_local)
    ):
        raise ValueError("address is not a public IP")
    return str(address)


def is_public_ip(value: str) -> bool:
    """Return whether a value is a globally routable IP address."""
    try:
        normalize_public_ip(value)
    except (TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """A Workspace-relative reference to one persisted Tool Artifact."""

    path: str
    total_chars: int
    preview_chars: int

    def __post_init__(self) -> None:
        parts = self.path.split("/")
        if (
            len(parts) != 4
            or parts[0] != ".omni"
            or parts[1] != "artifacts"
            or not _ARTIFACT_SESSION_PATTERN.fullmatch(parts[2])
            or not parts[3].endswith(".txt")
            or not _ARTIFACT_ID_PATTERN.fullmatch(parts[3][:-4])
        ):
            raise ValueError("path must match the Workspace-relative artifact path contract")
        require_nonnegative_int(self.total_chars, field="total_chars")
        require_nonnegative_int(self.preview_chars, field="preview_chars")
        if self.preview_chars > self.total_chars:
            raise ValueError("preview_chars must not exceed total_chars")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "total_chars": self.total_chars,
            "preview_chars": self.preview_chars,
        }


@dataclass(frozen=True, slots=True)
class ToolResultContent:
    """Inline Tool content with an optional external Artifact reference."""

    content: str
    artifact: ArtifactReference | None = None


type ArtifactWriter = Callable[[Path, str], None]


def truncate_text(content: str, *, limit: int, marker: str = _DEFAULT_TRUNCATION_MARKER) -> str:
    """Keep a prefix and marker within one configured character limit."""
    if not isinstance(content, str):
        raise TypeError("Tool result content must be a string")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Tool result limit must be a positive integer")
    if len(content) <= limit:
        return content
    if len(marker) >= limit:
        return marker[:limit]
    return content[: limit - len(marker)] + marker


class _BaseToolMeta(ABCMeta):
    """Keep incomplete Tool declarations abstract while supporting prepared-only Tools."""

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        tool_type = super().__new__(mcls, name, bases, namespace, **kwargs)
        if any(isinstance(base, _BaseToolMeta) for base in bases):
            base_tool = globals().get("BaseTool")
            default_execute = getattr(base_tool, "execute", None)
            default_execute_prepared = getattr(base_tool, "execute_prepared", None)
            resolved_execute = getattr(tool_type, "execute", None)
            resolved_execute_prepared = getattr(tool_type, "execute_prepared", None)
            has_execution = (
                callable(resolved_execute) and resolved_execute is not default_execute
            ) or (
                callable(resolved_execute_prepared)
                and resolved_execute_prepared is not default_execute_prepared
            )
            if not has_execution:
                tool_type.__abstractmethods__ = frozenset(
                    {*tool_type.__abstractmethods__, "execute"}
                )
        return tool_type


class BaseTool(ABC, metaclass=_BaseToolMeta):
    """Declare one Tool and expose its model-visible parameter contract."""

    name: str
    description: str
    parameters: ClassVar[dict[str, Any]]
    required: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "to_schema" in cls.__dict__:
            raise TypeError("Concrete Tools cannot override BaseTool.to_schema()")
        execute = cls.__dict__.get("execute")
        execute_prepared = cls.__dict__.get("execute_prepared")
        if execute is not None:
            _validate_execute(execute)
        if execute_prepared is not None:
            _validate_execute_prepared(execute_prepared)
        if "parameters" not in cls.__dict__:
            try:
                legacy = _schema_from_class(cls, execute)
            except (TypeError, ValueError):
                # Keep declaration errors at the public to_schema boundary for the bridge.
                if _has_legacy_declaration(cls, execute):
                    cast(Any, cls).parameters = {}
                    cast(Any, cls).__schema_declaration_invalid__ = True
            else:
                cast(Any, cls).parameters = legacy.to_json_schema()
        update_abstractmethods(cls)

    async def execute(self, *args: Any, **kwargs: Any) -> str:
        """Execute one already prepared Tool invocation and return text."""
        raise NotImplementedError

    async def execute_prepared(self, arguments: dict[str, Any]) -> str:
        """Execute one prepared argument dictionary through the Built-in Tool seam."""
        return await self.execute(**deepcopy(arguments))

    async def execute_authorized(
        self,
        arguments: dict[str, Any],
        authorization: ToolAuthorizationSession,
    ) -> str:
        """Execute one prepared call while retaining its per-call authorization boundary."""
        del authorization
        return await self.execute_prepared(arguments)

    @final
    def resolve_path_argument(
        self,
        *,
        workspace: Path,
        requested: str | Path,
    ) -> Path:
        """Resolve one Tool path and map expected failures to the public result contract."""
        try:
            return resolve_tool_path(workspace, requested)
        except (OSError, RuntimeError, ValueError) as error:
            raise self._path_resolution_error(error) from error

    @final
    def canonical_file_access(
        self,
        *,
        workspace: Path,
        base: Path,
        requested: str | Path,
        role: FileAccessRole,
    ) -> FileAccess:
        """Build one canonical host path fact for a File Tool invocation."""
        try:
            return canonicalize_file_access(
                workspace=workspace,
                base=base,
                requested=requested,
                role=role,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise self._path_resolution_error(error) from error

    @final
    def _path_resolution_error(self, error: Exception) -> ToolError:
        operation = self.name.replace("_", " ").title()
        return ToolError(f"{operation} path could not be resolved: {error}")

    @staticmethod
    def handle_result(
        content: str,
        *,
        workspace: Path,
        session_id: str,
        tool_call_id: str,
        limit: int,
        write_text: ArtifactWriter | None = None,
    ) -> ToolResultContent:
        """Externalize oversized successful content beneath Workspace State."""
        if len(content) <= limit:
            return ToolResultContent(content=content)
        if not isinstance(workspace, Path):
            raise TypeError("Tool Artifact workspace must be a Path")
        if not isinstance(session_id, str) or not _ARTIFACT_SESSION_PATTERN.fullmatch(session_id):
            raise ValueError("Tool Artifact session ID must be a single path component")

        filename = tool_call_id if _ARTIFACT_ID_PATTERN.fullmatch(tool_call_id) else str(uuid4())
        relative_path = f".omni/artifacts/{session_id}/{filename}.txt"
        artifact_path = workspace / Path(*relative_path.split("/"))
        marker = f"\n\n...[truncated; full result stored at {relative_path}]"
        preview = truncate_text(content, limit=limit, marker=marker)
        preview_chars = max(0, len(preview) - len(marker))
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if write_text is None:
                with artifact_path.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(content)
            else:
                write_text(artifact_path, content)
        except Exception as error:
            logger.opt(exception=error).error(
                "Tool Artifact persistence failed code=persistence_error "
                "session={} tool_call_id={} type={}",
                session_id,
                tool_call_id,
                type(error).__name__,
            )
            return ToolResultContent(
                content=truncate_text(
                    content,
                    limit=limit,
                    marker=_ARTIFACT_WRITE_FAILURE_MARKER,
                )
            )

        return ToolResultContent(
            content=preview,
            artifact=ArtifactReference(
                path=relative_path,
                total_chars=len(content),
                preview_chars=preview_chars,
            ),
        )

    @final
    async def prepare(
        self,
        arguments: dict[str, Any],
    ) -> ToolInvocationFacts:
        """Normalize, validate, and collect detached authorization facts."""
        prepared_arguments = await self.prepare_arguments(arguments)
        if not isinstance(prepared_arguments, dict):
            raise TypeError("Tool argument preparation must return a dictionary")

        validation = self.validate_arguments(**deepcopy(prepared_arguments))
        if inspect.isawaitable(validation):
            validation = await validation
        if isinstance(validation, str):
            raise ToolError(validation)
        if validation is False:
            raise ToolError("Tool arguments are invalid.")

        return await self.collect_invocation_facts(prepared_arguments)

    def build_invocation_facts(
        self,
        prepared_arguments: dict[str, Any],
    ) -> ToolInvocationFacts:
        """Build detached authorization facts after normalization and validation."""
        return ToolInvocationFacts(
            tool_name=self.name,
            normalized_arguments=prepared_arguments,
            file_accesses=self.build_file_accesses(prepared_arguments),
        )

    def build_file_accesses(
        self,
        prepared_arguments: dict[str, Any],
    ) -> tuple[FileAccess, ...]:
        """Return structured host path facts for this invocation when applicable."""
        del prepared_arguments
        return ()

    async def collect_invocation_facts(
        self,
        prepared_arguments: dict[str, Any],
    ) -> ToolInvocationFacts:
        """Collect detached facts, allowing Host-backed Tools to inspect asynchronously."""
        return self.build_invocation_facts(prepared_arguments)

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Cast, default, filter, and validate one Built-in Tool argument object."""
        schema = self._build_preparation_schema()
        casted = schema.cast(arguments)
        if not isinstance(casted, dict):
            raise ToolError("Tool arguments must be an object.")
        prepared_arguments: dict[str, Any] = {}
        for name, property_schema in schema.properties.items():
            if name in casted:
                prepared_arguments[name] = deepcopy(casted[name])
            elif property_schema.has_default:
                prepared_arguments[name] = property_schema.default
        errors = schema.validate(prepared_arguments)
        if errors:
            raise ToolError("; ".join(str(error) for error in errors))
        return prepared_arguments

    def _build_preparation_schema(self) -> Schema:
        """Build the temporary restricted Schema used by Built-in preparation."""
        return _schema_from_class(type(self), getattr(type(self), "execute", None))

    def validate_arguments(self, /, **arguments: Any) -> Any:
        """Validate normalized Tool-specific arguments before fact collection.

        Concrete Tools may raise ``ToolError`` with a model-safe domain message.  The
        default accepts the normalized arguments without adding validation.
        """
        del arguments

    @final
    def to_schema(self) -> dict[str, Any]:
        """Generate a detached OpenAI Function Calling schema."""
        name = getattr(self, "name", None)
        description = getattr(self, "description", None)
        if not isinstance(name, str) or not name:
            raise TypeError("Tool name must be a non-empty string")
        if not isinstance(description, str) or not description:
            raise TypeError("Tool description must be a non-empty string")
        tool_type = type(self)
        if getattr(tool_type, _INVALID_SCHEMA_MARKER, False):
            _schema_from_class(tool_type, getattr(tool_type, "execute", None))
        parameters = getattr(self, "parameters", _MISSING)
        if not isinstance(parameters, dict):
            raise TypeError("Tool parameters must be a dictionary")
        if parameters.get("type") != "object":
            raise TypeError("Tool parameters must use an object root Schema")
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": deepcopy(parameters),
            },
        }


def _has_legacy_declaration(tool_type: type[object], execute: object) -> bool:
    return bool(tool_type.__dict__.get("__annotations__")) or execute is not None


def _validate_execute(execute: object) -> None:
    if not inspect.iscoroutinefunction(execute):
        raise TypeError("Concrete Tool execute() must be asynchronous")
    parameters = tuple(inspect.signature(cast(Any, execute)).parameters.values())
    if (
        not parameters
        or parameters[0].name != "self"
        or parameters[0].kind
        not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        or any(parameter.kind is not inspect.Parameter.KEYWORD_ONLY for parameter in parameters[1:])
    ):
        raise TypeError("Concrete Tool execute() must accept only self and keyword parameters")
    try:
        return_annotation = get_type_hints(cast(Any, execute)).get("return")
    except (NameError, TypeError) as error:
        raise TypeError(
            "Concrete Tool execute() return annotation could not be resolved"
        ) from error
    if return_annotation is not str:
        raise TypeError("Concrete Tool execute() must declare a string return")


def _validate_execute_prepared(execute_prepared: object) -> None:
    if not inspect.iscoroutinefunction(execute_prepared):
        raise TypeError("Concrete Tool execute_prepared() must be asynchronous")
    parameters = tuple(inspect.signature(cast(Any, execute_prepared)).parameters.values())
    if (
        len(parameters) != 2
        or parameters[0].name != "self"
        or parameters[0].kind
        not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        or parameters[1].name != "arguments"
        or parameters[1].kind
        not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ):
        raise TypeError(
            "Concrete Tool execute_prepared() must accept self and one arguments parameter"
        )
    try:
        return_annotation = get_type_hints(cast(Any, execute_prepared)).get("return")
    except (NameError, TypeError) as error:
        raise TypeError(
            "Concrete Tool execute_prepared() return annotation could not be resolved"
        ) from error
    if return_annotation is not str:
        raise TypeError("Concrete Tool execute_prepared() must declare a string return")


def _schema_from_class(tool_type: type[object], execute: object) -> Schema:
    try:
        annotations = inspect.get_annotations(tool_type, eval_str=True)
    except (NameError, TypeError) as error:
        name = getattr(tool_type, "name", tool_type.__name__)
        raise TypeError(f"Tool annotations for {name} could not be resolved") from error

    required = getattr(tool_type, "required", ())
    if not isinstance(required, tuple) or any(
        not isinstance(item, str) or not item for item in required
    ):
        raise TypeError("Tool required parameters must be a tuple of non-empty strings")
    if len(set(required)) != len(required):
        raise ValueError("Tool required parameters must be unique")

    if not annotations and execute is None:
        raise TypeError("Tool parameters are not declared")
    if not annotations:
        annotations = _execute_annotations(execute)
    properties: dict[str, Schema] = {}
    for parameter_name, annotation in annotations.items():
        if (
            parameter_name.startswith("_")
            or parameter_name in _METADATA_NAMES
            or get_origin(annotation) is _CLASS_VAR_ORIGIN
        ):
            continue
        metadata = _annotation_metadata(parameter_name, annotation)
        base_annotation, nullable = _unwrap_annotation(annotation)
        default = tool_type.__dict__.get(parameter_name, _MISSING)
        properties[parameter_name] = _schema_for_annotation(
            parameter_name,
            base_annotation,
            metadata,
            nullable=nullable,
            default=default,
        )
    return Schema.object(properties, required=required)


def _execute_annotations(execute: object) -> dict[str, object]:
    if execute is None:
        return {}
    try:
        signature = inspect.signature(cast(Any, execute))
    except (TypeError, ValueError) as error:
        raise TypeError("Tool execute() signature could not be inspected") from error
    try:
        resolved_annotations = get_type_hints(cast(Any, execute))
    except (NameError, TypeError) as error:
        raise TypeError("Tool execute() annotations could not be resolved") from error
    annotations: dict[str, object] = {}
    for execute_parameter in tuple(signature.parameters.values())[1:]:
        if execute_parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        annotation = resolved_annotations.get(execute_parameter.name, execute_parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Tool parameter {execute_parameter.name} has no annotation")
        annotations[execute_parameter.name] = annotation
    return annotations


def _annotation_metadata(parameter_name: str, annotation: object) -> ToolParam:
    if get_origin(annotation) is not Annotated:
        return ToolParam()
    _, *extras = get_args(annotation)
    metadata = next((item for item in extras if isinstance(item, ToolParam)), ToolParam())
    if any(not isinstance(item, ToolParam) for item in extras):
        raise TypeError(f"Tool parameter {parameter_name} has unsupported Annotated metadata")
    return metadata


def _unwrap_annotation(annotation: object) -> tuple[object, bool]:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) in {Union, UnionType}:
        members = get_args(annotation)
        non_none = tuple(member for member in members if member is not NoneType)
        if len(members) != 2 or len(non_none) != 1:
            raise TypeError("Tool parameter has an unsupported union")
        return non_none[0], True
    return annotation, False


def _schema_for_annotation(
    parameter_name: str,
    annotation: object,
    metadata: ToolParam,
    *,
    nullable: bool,
    default: object,
) -> Schema:
    default_kwargs: dict[str, object] = {}
    if default is not _MISSING:
        default_kwargs["default"] = default
    common: dict[str, Any] = {
        "description": metadata.description,
        "nullable": nullable,
        **default_kwargs,
    }
    try:
        if annotation is str:
            return Schema.string(
                **common,
                min_length=metadata.min_length,
                max_length=metadata.max_length,
                format=metadata.format,
                pattern=metadata.pattern,
            )
        if annotation is int:
            if (
                metadata.min_length is not None
                or metadata.max_length is not None
                or metadata.format is not None
            ):
                raise TypeError(
                    f"Tool parameter {parameter_name} has string constraints on a non-string"
                )
            return Schema.integer(
                **common,
                minimum=metadata.minimum,
                maximum=metadata.maximum,
            )
        if annotation is bool:
            if any(
                value is not None
                for value in (
                    metadata.min_length,
                    metadata.max_length,
                    metadata.minimum,
                    metadata.maximum,
                    metadata.format,
                )
            ):
                raise TypeError(f"Tool parameter {parameter_name} has incompatible constraints")
            return Schema.boolean(**common)
    except TypeError:
        raise
    raise TypeError(f"Tool parameter {parameter_name} has an unsupported annotation")


__all__ = [
    "ArtifactReference",
    "ArtifactWriter",
    "BaseTool",
    "Schema",
    "ToolError",
    "ToolParam",
    "ToolResultContent",
    "is_public_ip",
    "normalize_public_ip",
    "resolve_tool_path",
    "truncate_text",
]
