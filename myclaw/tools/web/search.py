"""Web Search Core Catalog Tool."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Annotated, Final, cast

from ddgs import DDGS
from ddgs.exceptions import DDGSException

from myclaw.tools.base import BaseTool, ToolError, ToolParam

_SEARCH_TIMEOUT_SECONDS: Final[float] = 30.0


class WebSearchTool(BaseTool):
    """Search the public web through the synchronous DDGS client."""

    name = "web_search"
    description = "Search the public web and return normalized result summaries."
    required = ("query",)

    query: Annotated[str, ToolParam(description="Public web search query.", min_length=1)]
    count: Annotated[
        int,
        ToolParam(description="Maximum normalized results.", minimum=1, maximum=10),
    ] = 5

    def validate_arguments(self, *, query: str, count: int) -> str | None:  # type: ignore[override]
        del count
        if not query.strip():
            return "Web Search query must not be blank."
        return None

    async def execute(self, *, query: str, count: int) -> str:
        try:
            raw_results = await asyncio.wait_for(
                asyncio.to_thread(_search_sync, query.strip(), count),
                timeout=_SEARCH_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise ToolError(
                f"Web Search timed out after {_SEARCH_TIMEOUT_SECONDS:g} seconds."
            ) from error

        return _format_results(raw_results[:count])


def _search_sync(query: str, count: int) -> list[Mapping[str, object]]:
    """Run the blocking DDGS call in the worker thread supplied by the caller."""
    try:
        with DDGS() as client:
            results = client.text(query, max_results=count, backend="duckduckgo")
    except DDGSException as error:
        if str(error) == "No results found.":
            return []
        raise
    return cast(list[Mapping[str, object]], results)


def _format_results(results: list[Mapping[str, object]]) -> str:
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        title = _collapse_whitespace(result.get("title"))
        url = _url_value(result)
        snippet = _collapse_whitespace(_first_text(result, "body", "snippet"))
        blocks.append(f"{index}. Title: {title}\n   URL: {url}\n   Snippet: {snippet}")
    return "\n\n".join(blocks)


def _collapse_whitespace(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _url_value(result: Mapping[str, object]) -> str:
    value = _first_text(result, "href", "url")
    return value.strip()


def _first_text(result: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = result.get(name)
        if isinstance(value, str):
            return value
    return ""


__all__ = ["WebSearchTool"]
