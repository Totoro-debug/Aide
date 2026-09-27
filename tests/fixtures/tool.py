"""Scripted nominal Tools for deterministic offline tests."""

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, cast

from myclaw.tools.base import BaseTool, ToolParam


@dataclass(frozen=True, slots=True)
class FakeToolCall:
    """One normalized keyword call observed by a FakeTool."""

    arguments: Mapping[str, object]


class _FakeToolBase(BaseTool):
    name = "fake"
    description = "Scripted fake Tool."

    path: str | None = None
    item: str | None = None
    message: str | None = None
    query: str | None = None
    notice_id: str | None = None
    recipient: Annotated[str | None, ToolParam(format="email")] = None

    def __init__(self, *, outcomes: Iterable[str | BaseException]) -> None:
        self._outcomes = deque(outcomes)
        self.calls: list[FakeToolCall] = []

    async def execute(
        self,
        *,
        path: str | None = None,
        item: str | None = None,
        message: str | None = None,
        query: str | None = None,
        notice_id: str | None = None,
        recipient: str | None = None,
    ) -> str:
        arguments = {
            name: value
            for name, value in (
                ("path", path),
                ("item", item),
                ("message", message),
                ("query", query),
                ("notice_id", notice_id),
                ("recipient", recipient),
            )
            if value is not None
        }
        self.calls.append(FakeToolCall(arguments=arguments))
        if not self._outcomes:
            msg = "No scripted Tool outcome remains"
            raise AssertionError(msg)
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def FakeTool(
    *,
    name: str,
    description: str,
    outcomes: Iterable[str | BaseException],
    required: tuple[str, ...] = (),
) -> _FakeToolBase:
    """Build one independently declared scripted BaseTool instance."""
    annotations = dict(_FakeToolBase.__annotations__)
    namespace: dict[str, object] = {
        "__annotations__": annotations,
        "name": name,
        "description": description,
        "required": required,
        "execute": _FakeToolBase.execute,
    }
    for parameter_name in annotations:
        if parameter_name not in required:
            namespace[parameter_name] = None
    tool_type = type(f"Fake_{name}_Tool", (_FakeToolBase,), namespace)
    return cast(_FakeToolBase, tool_type(outcomes=outcomes))
