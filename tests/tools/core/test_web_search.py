from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import ClassVar

import pytest
from ddgs.exceptions import DDGSException

from myclaw.tools.base import BaseTool
from myclaw.tools.gateway import ModelToolCall
from myclaw.tools.web.search import WebSearchTool
from tests.fixtures import SingleToolGateway


def _call(arguments: dict[str, object], *, call_id: str = "call_search") -> ModelToolCall:
    return ModelToolCall(
        id=call_id,
        name="web_search",
        arguments=json.dumps(arguments),
    )


def _gateway(tool: BaseTool | None = None) -> SingleToolGateway:
    return SingleToolGateway((WebSearchTool() if tool is None else tool,))


class FakeDDGS:
    records: ClassVar[list[dict[str, object]]] = []
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []
    failure: ClassVar[BaseException | None] = None

    def __enter__(self) -> FakeDDGS:
        return self

    def __exit__(self, *errors: object) -> None:
        del errors

    def text(self, query: str, **kwargs: object) -> list[dict[str, object]]:
        self.calls.append((query, kwargs))
        if self.failure is not None:
            raise self.failure
        return self.records


@pytest.fixture
def fake_ddgs(monkeypatch: pytest.MonkeyPatch) -> type[FakeDDGS]:
    FakeDDGS.records = []
    FakeDDGS.calls = []
    FakeDDGS.failure = None
    monkeypatch.setattr("myclaw.tools.web.search.DDGS", FakeDDGS)
    return FakeDDGS


def test_web_search_schema_has_no_generic_retry_metadata() -> None:
    tool = WebSearchTool()

    assert not hasattr(tool, "max_retries")
    assert tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the public web and return normalized result summaries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Public web search query.",
                        "minLength": 1,
                    },
                    "count": {
                        "type": "integer",
                        "description": "Maximum normalized results.",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    }


@pytest.mark.asyncio
async def test_web_search_trims_query_passes_count_and_uses_one_worker_call(
    fake_ddgs: type[FakeDDGS],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_calls: list[tuple[Callable[..., object], tuple[object, ...]]] = []

    async def to_thread(function: Callable[..., object], *arguments: object) -> object:
        worker_calls.append((function, arguments))
        return function(*arguments)

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    fake_ddgs.records = [{"title": "A", "href": "https://example.com", "body": "B"}]

    result = await _gateway().call(_call({"query": "  runtime search  ", "count": 3}))

    assert result.status == "success"
    assert result.content == "1. Title: A\n   URL: https://example.com\n   Snippet: B"
    assert len(worker_calls) == 1
    assert fake_ddgs.calls == [("runtime search", {"max_results": 3, "backend": "duckduckgo"})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "call_id"),
    (
        ({"query": ""}, "empty-query"),
        ({"query": "   \t\n"}, "blank-query"),
        ({"query": "valid", "count": 0}, "low-count"),
        ({"query": "valid", "count": 11}, "high-count"),
    ),
)
async def test_web_search_rejects_invalid_query_and_count_before_ddgs(
    fake_ddgs: type[FakeDDGS],
    arguments: dict[str, object],
    call_id: str,
) -> None:
    result = await _gateway().call(_call(arguments, call_id=call_id))

    assert result.status == "error"
    assert fake_ddgs.calls == []


@pytest.mark.asyncio
async def test_web_search_normalizes_fields_preserves_order_and_uses_empty_missing_values(
    fake_ddgs: type[FakeDDGS],
) -> None:
    fake_ddgs.records = [
        {
            "title": " First\n  result ",
            "href": "  https://example.com/first  ",
            "body": " first\tresult\nwith whitespace ",
        },
        {"url": "https://example.com/second", "snippet": " second\nsummary "},
        {},
    ]

    result = await _gateway().call(_call({"query": "ordered", "count": 3}))

    assert result.status == "success"
    assert result.content == (
        "1. Title: First result\n"
        "   URL: https://example.com/first\n"
        "   Snippet: first result with whitespace\n\n"
        "2. Title: \n"
        "   URL: https://example.com/second\n"
        "   Snippet: second summary\n\n"
        "3. Title: \n"
        "   URL: \n"
        "   Snippet: "
    )


@pytest.mark.asyncio
async def test_web_search_returns_empty_content_without_results(fake_ddgs: type[FakeDDGS]) -> None:
    fake_ddgs.failure = DDGSException("No results found.")

    result = await _gateway().call(_call({"query": "nothing"}))

    assert result.status == "success"
    assert result.content == ""
    assert fake_ddgs.calls == [("nothing", {"max_results": 5, "backend": "duckduckgo"})]


@pytest.mark.asyncio
async def test_web_search_limits_output_to_count_even_if_ddgs_returns_more(
    fake_ddgs: type[FakeDDGS],
) -> None:
    fake_ddgs.records = [
        {"title": "First"},
        {"title": "Second"},
    ]

    result = await _gateway().call(_call({"query": "limited", "count": 1}))

    assert result.status == "success"
    assert result.content == "1. Title: First\n   URL: \n   Snippet: "


@pytest.mark.asyncio
async def test_web_search_failure_is_not_retried_or_leaked(
    fake_ddgs: type[FakeDDGS],
) -> None:
    fake_ddgs.failure = RuntimeError("private upstream response")

    result = await _gateway().call(_call({"query": "failure"}))

    assert result.status == "error"
    assert result.content == "web_search could not complete the request."
    assert "private upstream response" not in result.content
    assert fake_ddgs.calls == [("failure", {"max_results": 5, "backend": "duckduckgo"})]


@pytest.mark.asyncio
async def test_web_search_timeout_is_bounded_to_thirty_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, float] = {}

    async def to_thread(function: Callable[..., object], *arguments: object) -> object:
        del function, arguments
        return None

    async def wait_for(awaitable: object, *, timeout: float) -> object:
        observed["timeout"] = timeout
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    monkeypatch.setattr(asyncio, "wait_for", wait_for)

    result = await _gateway().call(_call({"query": "slow"}))

    assert result.status == "error"
    assert result.content == "Web Search timed out after 30 seconds."
    assert observed == {"timeout": 30.0}


@pytest.mark.asyncio
async def test_web_search_cancellation_propagates_from_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def to_thread(function: Callable[..., object], *arguments: object) -> object:
        del function, arguments
        started.set()
        await release.wait()
        return []

    monkeypatch.setattr(asyncio, "to_thread", to_thread)
    task = asyncio.create_task(_gateway().call(_call({"query": "cancel"})))
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
