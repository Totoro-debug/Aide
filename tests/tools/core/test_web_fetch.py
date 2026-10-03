from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest

from omni.agent.tools.core.web_fetch import (
    HTTPClientBoundary,
    HTTPResponseBoundary,
    WebFetchTool,
)
from omni.agent.tools.network_safety import DNSResolver
from omni.agent.tools.permission import PermissionContext
from omni.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ModelToolCall,
)
from tests.fixtures import SingleToolGateway


def _call(arguments: dict[str, object], *, call_id: str = "call_fetch") -> ModelToolCall:
    return ModelToolCall(
        id=call_id,
        name="web_fetch",
        arguments=json.dumps(arguments),
    )


class FakeResolver:
    def __init__(self, answers: dict[str, tuple[str, ...]] | tuple[str, ...]) -> None:
        self._answers = answers
        self.calls: list[tuple[str, int]] = []
        self.failure: BaseException | None = None

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        self.calls.append((hostname, port))
        if self.failure is not None:
            raise self.failure
        if isinstance(self._answers, dict):
            return self._answers[hostname]
        return self._answers


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (),
    ) -> None:
        self.status_code = status_code
        self.headers = {} if headers is None else headers
        self._chunks = chunks
        self.closed = False
        self.iterated = False

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        self.iterated = True
        for chunk in self._chunks:
            yield chunk

    async def close(self) -> None:
        self.closed = True


class FakeHTTPClient:
    def __init__(self, responses: tuple[HTTPResponseBoundary, ...]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, float, float]] = []

    async def get(
        self,
        url: str,
        *,
        resolved_addresses: tuple[str, ...],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        del resolved_addresses
        self.calls.append((url, connect_timeout_seconds, total_timeout_seconds))
        response = next(self._responses)
        return response


def _gateway(
    *,
    resolver: DNSResolver,
    http: HTTPClientBoundary | None = None,
    confirmation: ConfirmationRequester | None = None,
) -> SingleToolGateway:
    tool = WebFetchTool(resolver=resolver, http_client=http)
    return SingleToolGateway(
        (tool,),
        confirmation=confirmation,
        permission_context=PermissionContext(
            level="workspace-write",
            workspace_root=Path.cwd(),
        ),
    )


def test_web_fetch_schema_declares_format_and_max_chars() -> None:
    schema = WebFetchTool().to_schema()

    assert schema == {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch readable content from an HTTP or HTTPS URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "HTTP or HTTPS URL to fetch.",
                        "minLength": 1,
                        "format": "uri",
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format: markdown or text.",
                        "minLength": 1,
                        "default": "markdown",
                    },
                    "maxChars": {
                        "type": "integer",
                        "description": "Maximum returned characters.",
                        "minimum": 1,
                        "default": 50000,
                    },
                },
                "required": ["url"],
            },
        },
    }


@pytest.mark.asyncio
async def test_web_fetch_uses_the_audited_direct_client_for_public_targets() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"Public page",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "  https://public.example/page  "})
    )

    assert result.status == "success"
    assert result.content == "Public page"
    assert resolver.calls == [("public.example", 443)]
    assert http.calls == [("https://public.example/page", 10.0, 30.0)]
    assert response.closed


@pytest.mark.asyncio
async def test_web_fetch_direct_text_preserves_declared_charset() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    response = FakeResponse(
        headers={"Content-Type": "text/plain; charset=utf-8"},
        chunks=(b"Direct ", b"content"),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://public.example/page", "format": "text"})
    )

    assert result.status == "success"
    assert result.content == "Direct content"
    assert http.calls == [("https://public.example/page", 10.0, 30.0)]
    assert response.closed


@pytest.mark.asyncio
async def test_web_fetch_cancellation_propagates_from_the_audited_client() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class HangingHTTP:
        async def get(
            self,
            url: str,
            *,
            resolved_addresses: tuple[str, ...],
            connect_timeout_seconds: float,
            total_timeout_seconds: float,
        ) -> HTTPResponseBoundary:
            del url, resolved_addresses, connect_timeout_seconds, total_timeout_seconds
            started.set()
            await release.wait()
            raise AssertionError("unreachable")

    resolver = FakeResolver(("93.184.216.34",))
    task = asyncio.create_task(
        _gateway(resolver=resolver, http=HangingHTTP()).call(
            _call({"url": "https://public.example/page"})
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    (
        {"url": ""},
        {"url": "  \t"},
        {"url": "ftp://public.example/page"},
        {"url": "https://user:password@public.example/page"},
        {"url": "https://public.example:bad/page"},
        {"url": "https://public.example:0/page"},
        {"url": "https:///page"},
        {"url": "https://public.example/page", "format": "html"},
        {"url": "https://public.example/page", "maxChars": 0},
    ),
)
async def test_web_fetch_rejects_invalid_parameters_before_dns(
    arguments: dict[str, object],
) -> None:
    resolver = FakeResolver(("93.184.216.34",))
    http = FakeHTTPClient(())

    result = await _gateway(resolver=resolver, http=http).call(_call(arguments))

    assert result.status == "error"
    assert resolver.calls == []
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_uses_direct_http_response_content() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"direct",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://public.example/page"})
    )

    assert result.status == "success"
    assert result.content == "direct"
    assert http.calls == [("https://public.example/page", 10.0, 30.0)]
    assert response.closed


@pytest.mark.asyncio
async def test_web_fetch_approved_private_target_uses_direct_http() -> None:
    resolver = FakeResolver(("127.0.0.1",))
    response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"private",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(
        resolver=resolver,
        http=http,
        confirmation=approve,
    ).call(_call({"url": "http://private.example/status"}))

    assert result.status == "success"
    assert result.content == "private"
    assert http.calls == [("http://private.example/status", 10.0, 30.0)]
    assert len(requests) == 1
    assert "private" in requests[0].reason


@pytest.mark.asyncio
async def test_concurrent_web_fetch_calls_keep_target_evaluations_isolated() -> None:
    class SequencedResolver:
        def __init__(self) -> None:
            self._answers = iter(
                (
                    ("127.0.0.1",),
                    ("93.184.216.34",),
                )
            )

        async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
            del hostname, port
            return next(self._answers)

    public_response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"public-direct",),
    )
    private_response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"confirmed-private-direct",),
    )
    http = FakeHTTPClient(
        (
            cast(HTTPResponseBoundary, public_response),
            cast(HTTPResponseBoundary, private_response),
        )
    )
    confirmation_requested = asyncio.Event()
    approve_private = asyncio.Event()
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        confirmation_requested.set()
        await approve_private.wait()
        return "approved"

    gateway = _gateway(
        resolver=SequencedResolver(),
        http=http,
        confirmation=approve,
    )
    private_call = asyncio.create_task(
        gateway.call(
            _call(
                {"url": "https://same.example/page"},
                call_id="call_private",
            )
        )
    )
    await confirmation_requested.wait()

    public_result = await gateway.call(
        _call(
            {"url": "https://same.example/page"},
            call_id="call_public",
        )
    )
    approve_private.set()
    private_result = await private_call

    assert public_result.status == "success"
    assert public_result.content == "public-direct"
    assert private_result.status == "success"
    assert private_result.content == "confirmed-private-direct"
    assert private_result.confirmation is not None
    assert private_result.confirmation.request.tool_call_id == "call_private"
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_web_fetch_dns_failure_requests_confirmation_then_returns_tool_error() -> None:
    resolver = FakeResolver(())
    resolver.failure = OSError("DNS unavailable")
    response = FakeResponse(headers={"content-type": "text/plain"}, chunks=(b"approved",))
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(
        resolver=resolver,
        http=http,
        confirmation=approve,
    ).call(_call({"url": "https://missing.example/page"}))

    assert result.status == "error"
    assert "DNS" in result.content
    assert len(requests) == 1
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_maps_ipv4_mapped_ipv6_before_requesting_confirmation() -> None:
    resolver = FakeResolver(("::ffff:10.0.0.7",))
    response = FakeResponse(headers={"content-type": "text/plain"}, chunks=(b"approved",))
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(
        resolver=resolver,
        http=http,
        confirmation=approve,
    ).call(_call({"url": "https://mapped.example/page"}))

    assert result.status == "success"
    assert result.content == "approved"
    assert len(requests) == 1
    assert "private" in requests[0].reason


@pytest.mark.asyncio
async def test_web_fetch_refuses_private_target_without_confirmation_channel() -> None:
    resolver = FakeResolver(("192.168.1.7",))
    http = FakeHTTPClient(())

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://private.example/page"})
    )

    assert result.status == "refused"
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_follows_public_redirects_and_rechecks_each_target() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "next.example": ("8.8.8.8",),
        }
    )
    redirect = FakeResponse(
        status_code=302,
        headers={"location": "https://next.example/final"},
    )
    final = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"redirected",),
    )
    http = FakeHTTPClient(
        (
            cast(HTTPResponseBoundary, redirect),
            cast(HTTPResponseBoundary, final),
        )
    )

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://public.example/start"})
    )

    assert result.status == "success"
    assert result.content == "redirected"
    assert resolver.calls == [
        ("public.example", 443),
        ("next.example", 443),
    ]
    assert [call[0] for call in http.calls] == [
        "https://public.example/start",
        "https://next.example/final",
    ]
    assert redirect.closed
    assert final.closed


@pytest.mark.asyncio
async def test_web_fetch_authorizes_a_newly_unsafe_redirect_in_the_same_call() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "internal.example": ("10.0.0.7",),
        }
    )
    redirect = FakeResponse(
        status_code=302,
        headers={"location": "http://internal.example/admin"},
    )
    final = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"private result",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, redirect), cast(HTTPResponseBoundary, final)))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _gateway(
        resolver=resolver,
        http=http,
        confirmation=approve,
    ).call(
        _call({"url": "https://public.example/start"})
    )

    assert result.status == "success"
    assert result.content == "private result"
    assert len(requests) == 1
    assert resolver.calls == [
        ("public.example", 443),
        ("internal.example", 80),
    ]
    assert len(http.calls) == 2
    assert redirect.closed
    assert final.closed


@pytest.mark.asyncio
async def test_web_fetch_follows_at_most_five_redirects() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    redirects = tuple(
        FakeResponse(
            status_code=302,
            headers={"location": f"/hop-{index + 1}"},
        )
        for index in range(6)
    )
    http = FakeHTTPClient(tuple(cast(HTTPResponseBoundary, response) for response in redirects))

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://public.example/hop-0"})
    )

    assert result.status == "error"
    assert "redirect limit" in result.content
    assert len(http.calls) == 6
    assert all(response.closed for response in redirects)
    assert len(resolver.calls) == 6


@pytest.mark.asyncio
async def test_web_fetch_accepts_textual_media_and_declared_charset() -> None:
    cases = (
        ("application/json", b'{"ok":true}', '{"ok":true}'),
        ("application/xml", b"<ok>true</ok>", "<ok>true</ok>"),
        ("application/javascript", b"const ok = true;", "const ok = true;"),
        ("text/plain; charset=iso-8859-1", b"caf\xe9", "café"),
    )
    for content_type, body, expected in cases:
        resolver = FakeResolver(("93.184.216.34",))
        response = FakeResponse(headers={"content-type": content_type}, chunks=(body,))
        http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

        result = await _gateway(resolver=resolver, http=http).call(
            _call({"url": "https://public.example/data"})
        )

        assert result.status == "success"
        assert result.content == expected
        assert response.closed


@pytest.mark.asyncio
async def test_web_fetch_decodes_missing_content_type_with_utf8_replacement() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    response = FakeResponse(chunks=(b"valid\xff",))
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://public.example/data"})
    )

    assert result.status == "success"
    assert result.content == "valid�"


@pytest.mark.asyncio
async def test_web_fetch_rejects_explicit_binary_media() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    response = FakeResponse(
        headers={"content-type": "application/octet-stream"},
        chunks=(b"binary",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://public.example/download"})
    )

    assert result.status == "error"
    assert "media type" in result.content
    assert response.closed
    assert not response.iterated


@pytest.mark.asyncio
async def test_web_fetch_extracts_readable_html_without_ignored_elements() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    response = FakeResponse(
        headers={"content-type": "text/html; charset=utf-8"},
        chunks=(
            b"<html><head><title>Example</title><style>hidden style</style>",
            b"<script>hidden script</script></head><body><h1>Public page</h1>",
            b"<p>Hello &amp; <strong>world</strong>.</p><template>hidden template</template>",
            b"</body></html>",
        ),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://public.example/page", "format": "text"})
    )

    assert result.status == "success"
    assert result.content == "Example\nPublic page\nHello & world."
    assert "hidden" not in result.content
    assert "<" not in result.content


@pytest.mark.asyncio
async def test_web_fetch_applies_final_shared_prefix_truncation_to_direct_output() -> None:
    resolver = FakeResolver(("93.184.216.34",))
    response = FakeResponse(
        headers={"content-type": "text/plain"},
        chunks=(b"abcdefghijklmnopqrstuvwxyz",),
    )
    http = FakeHTTPClient((cast(HTTPResponseBoundary, response),))

    result = await _gateway(resolver=resolver, http=http).call(
        _call({"url": "https://public.example/page", "maxChars": 20})
    )

    assert result.status == "success"
    assert result.content == "abcd\n\n...[truncated]"
    assert len(result.content) == 20
    assert len(http.calls) == 1


@pytest.mark.asyncio
async def test_web_fetch_whole_call_timeout_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowHTTP:
        async def get(
            self,
            url: str,
            *,
            resolved_addresses: tuple[str, ...],
            connect_timeout_seconds: float,
            total_timeout_seconds: float,
        ) -> HTTPResponseBoundary:
            del url, resolved_addresses, connect_timeout_seconds, total_timeout_seconds
            started.set()
            await release.wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr("omni.agent.tools.core.web_fetch.TOTAL_TIMEOUT_SECONDS", 0.01)
    resolver = FakeResolver(("93.184.216.34",))
    result = await _gateway(
        resolver=resolver,
        http=SlowHTTP(),
    ).call(_call({"url": "https://public.example/slow"}))

    assert result.status == "error"
    assert result.content == "Web Fetch timed out after 0.01 seconds."
    assert started.is_set()


@pytest.mark.asyncio
async def test_web_fetch_cancellation_propagates_from_direct_client() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class HangingHTTP:
        async def get(
            self,
            url: str,
            *,
            resolved_addresses: tuple[str, ...],
            connect_timeout_seconds: float,
            total_timeout_seconds: float,
        ) -> HTTPResponseBoundary:
            del url, resolved_addresses, connect_timeout_seconds, total_timeout_seconds
            started.set()
            await release.wait()
            raise AssertionError("unreachable")

    resolver = FakeResolver(("93.184.216.34",))
    task = asyncio.create_task(
        _gateway(resolver=resolver, http=HangingHTTP()).call(
            _call({"url": "https://public.example/slow"})
        )
    )
    await started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
