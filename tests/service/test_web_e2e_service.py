"""Keep provider fixture replies scoped to the current conversation turn."""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.scripts import e2e_service


@pytest.mark.asyncio
async def test_mcp_fixture_searches_current_turn_despite_prior_generation_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(e2e_service, "PROVIDER_OBSERVATION_PATH", None)
    app = web.Application()
    app.router.add_post("/completion", e2e_service._fixture_completion)
    messages = [
        {"role": "user", "content": "model MCP generation barrier"},
        {"role": "tool", "tool_call_id": "call-model-mcp-search-v1", "content": "found v1"},
        {"role": "tool", "tool_call_id": "call-model-mcp-v1", "content": "old"},
        {"role": "assistant", "content": "Old model and MCP resource completed."},
        {"role": "user", "content": "## User Input\nmodel MCP resource"},
    ]
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/completion",
            json={
                "model": "small-model",
                "messages": messages,
                "tools": [{"type": "function", "function": {"name": "tool_search"}}],
            },
        )
        assert response.status == 200
        chunks = [
            json.loads(line[6:])
            for line in (await response.text()).splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    calls = [
        call for chunk in chunks for call in chunk["choices"][0]["delta"].get("tool_calls", [])
    ]
    assert [call["id"] for call in calls] == ["call-model-mcp-search-v2"]
