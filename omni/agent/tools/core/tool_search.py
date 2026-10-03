"""BaseTool adapter for local deferred Tool Search retrieval."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Annotated, cast

from omni.agent.tools.base import BaseTool, ToolParam
from omni.agent.tools.search import MAX_TOOL_SEARCH_RESULTS

type SearchAndActivate = Callable[
    [str],
    Iterable[str] | Awaitable[Iterable[str]],
]


class ToolSearchTool(BaseTool):
    """Search and activate deferred Tools through an injected run-local callback."""

    name = "tool_search"
    description = (
        "Find deferred Tools with English keywords. Matching Tools become available "
        "to the next model request."
    )
    required = ("query",)

    query: Annotated[
        str,
        ToolParam(description="English keywords describing the capability to find."),
    ]

    def __init__(self, search_and_activate: SearchAndActivate) -> None:
        if not callable(search_and_activate):
            raise TypeError("Tool Search requires a callable search-and-activate callback")
        self._search_and_activate = search_and_activate

    async def execute(self, *, query: str) -> str:
        result: object = self._search_and_activate(query)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, (str, bytes)) or not isinstance(result, Iterable):
            raise TypeError("Tool Search callback must return an iterable of Tool names")

        names = tuple(cast(Iterable[object], result))
        if len(names) > MAX_TOOL_SEARCH_RESULTS:
            raise ValueError(
                f"Tool Search callback returned more than {MAX_TOOL_SEARCH_RESULTS} names"
            )
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("Tool Search callback must return non-empty string names")
        return json.dumps(list(names))
