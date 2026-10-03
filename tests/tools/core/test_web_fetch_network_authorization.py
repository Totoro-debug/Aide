from __future__ import annotations

import asyncio
import json
import socket
import ssl
from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest
from aiohttp import ClientConnectorCertificateError

from omni.agent.permission import ToolPermissionLevel
from omni.agent.tools.base import ToolError
from omni.agent.tools.core.web_fetch import (
    AioHttpWebFetchClient,
    HTTPClientBoundary,
    HTTPResponseBoundary,
    WebFetchTool,
)
from omni.agent.tools.core.web_search import WebSearchTool
from omni.agent.tools.network_safety import DNSResolver
from omni.agent.tools.permission import PermissionContext
from omni.agent.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationRequester,
    ModelToolCall,
    ToolGateway,
    ToolResult,
)

_TLS_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIC/zCCAeegAwIBAgIUZ+U95pMk7qZehIoDeGwFbm0fc0kwDQYJKoZIhvcNAQEL
BQAwGjEYMBYGA1UEAwwPYXVkaXRlZC5leGFtcGxlMB4XDTI2MDkyMTEzMDgzNloX
DTM2MDkxODEzMDgzNlowGjEYMBYGA1UEAwwPYXVkaXRlZC5leGFtcGxlMIIBIjAN
BgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAyzlKLsWg6vWLJGwppIf276mMaWek
lxsWX6I9uFXwnQVlLgY9iKoJHda2avfVx0nvN6AzcdASTesV7jJAKd0zbYh1Xozj
642L9YJ+TRblIEJyRroBtgIwwhOb/4ocfy/8DPb+qBbcFmZWCsDhlL6At0O3k1AD
IxZAqRFHlCCvMWBTNZobMYvnk8vLaIOxq6z25AXq2RFp1REqIBL4mQR7Pc3gGXWV
WW/UZXw5i9x/88u2LXDH2aYEW2Gg5iH+Mhf748ZZ1ejFxNkGDROm0z+s8twCAWeO
rOPd+seto8GGbrec+G9FMfUre2EqaVr8PUfYL84qo7+LbUKHpfiZuoNCYwIDAQAB
oz0wOzAaBgNVHREEEzARgg9hdWRpdGVkLmV4YW1wbGUwHQYDVR0OBBYEFJ4EXZ2l
KQwnznp3038jnIbtASsbMA0GCSqGSIb3DQEBCwUAA4IBAQCDRfOYOOq39sr2VRAo
tgGLFW93viNTCOvBUmaBwf74titUCqkYbicdeBYEzD9tljzENgRfIg0Wk0A9LoAc
A9i/2IlP36jvRFwZ5CCX3rNS/fdMULMTbuElwa5Kcl5A0qvPAt35LU+sXACSTZ8h
GIKLNpWWm1qFzddd9KOhWGFpecSp/0YPyKl4ndAjza/QHvDHGdW7XxHeOEcg/blK
8D3eCKy3oj8ItYJH38IqWOxL7otvLnedrT+oFOID8hqRWYBMfX3v8TuJQzcQKw2L
fSVbQchdWL3XsLUDNTciCLHeRqGepabyT8jHeSIdmmxAInKWxuiOj4nP8TX+ZJc/
zIfU
-----END CERTIFICATE-----
"""
_TLS_PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDLOUouxaDq9Ysk
bCmkh/bvqYxpZ6SXGxZfoj24VfCdBWUuBj2Iqgkd1rZq99XHSe83oDNx0BJN6xXu
MkAp3TNtiHVejOPrjYv1gn5NFuUgQnJGugG2AjDCE5v/ihx/L/wM9v6oFtwWZlYK
wOGUvoC3Q7eTUAMjFkCpEUeUIK8xYFM1mhsxi+eTy8tog7GrrPbkBerZEWnVESog
EviZBHs9zeAZdZVZb9RlfDmL3H/zy7YtcMfZpgRbYaDmIf4yF/vjxlnV6MXE2QYN
E6bTP6zy3AIBZ46s4936x62jwYZut5z4b0Ux9St7YSppWvw9R9gvziqjv4ttQoel
+Jm6g0JjAgMBAAECggEAEjECKXaqZW3ucye1gJNlMOXp+kN7UcVsdsoUoUwcGkox
2PFZD8M8xq2CLcganFjLb5zJDh6UjOIG2AgqgzTYVi05aGnPOzYz+ZmhSbBLeVxJ
U3hyD8NZbv4HYFQSIfZ/Jv/zIsPNFro5aIQEjWaSKhWHOMoYRctHpXq1ABb+57nx
UZoh5tVRez955wGyNjT8a33Q6jbzKXW51U5mKmmgHsgeT97O6tctLQJ9MbOa+UKk
gMlptLGNATy0Nuc7yZe7rGhw6an+z9qxd2z65lQJ0UnORcDdZpDVrlxfEoXBfpRs
vuhoQQUPQGTpRLB97ioGUnfiRBW4fClc5LeDJ/qGeQKBgQDn5S07hJCsK6iFigK2
HyUKbAKT8RKQ7NvEd+asLiyUs5dmE3iGCKIu8UW1uXjSQh7po3S8g2Da7wiCBQTO
0by3q8wCp9j0oVZ440Az0AXqTF57C3ZRxARqBHibHNHnUZjoR/849JUu3d49dYaA
qoI24Rhq0aii/qSZY+jGAHXbuwKBgQDgWSeqEYNHxnDESb0r0P1eaFY7smSBqFmP
VFXG+1YqHeTTnKRgK2xxXUYxmatS49ppHSJbwaau5enTNQJy3pllElXg1IV8noYu
YJfvaPKUwEvuLtFlx4Br29y5ExuFLamAuWBwt/B1g+ku6APCQmpRq0zKZ3YqcGmm
g6zRoe9FeQKBgCnjLb54faF76V7lxQOcsJYnWHfcrdvbzP66IcKsPIVHw2s+zSB4
4sLT9iGTNQ3Vv7u4ONfsa0xgrQq/WVT6cbpDoABCzV+y3OnNMsWpJ8hgrxhOw7qV
S67Sy+5I0GmWRaZ/isyA8Ymbrg8v8XHAWvEKy9xPrsRydsz2TQ+m+aMNAoGBALwL
fqeaTkOXHWYpuJpFbln3cnBPMtdK2Oa+dbd3a92ZePe2UEEbpKXQ3MkuWN/9hFCe
zvHB+4iVxcv2nrrRwhlpqPnuqISwCyBMbo2JlesA06QtMe7xrb66ZuPqFCMpBu6S
czeHtdGKY6Wha6UkLiGOR6tP1Uf1OVkM/YopBXlhAoGBAMnGtyI/P5X4/QE9IwjO
J3VBwpTkxpmAPjfcMIVIOVaY0qk/kEqoRSk5/3B7BhLvi9N4Lq/3BhbbideH/dCJ
UttRrA4aEQpa6Qj2AjRInOeNmN8fBTKlzlam2r09LShxnO3on86+hCtsY6dKhY7S
0lAqiLbMyUzmiUvaEz/jMZ1T
-----END PRIVATE KEY-----
"""


def _call(url: str, *, call_id: str = "network-call") -> ModelToolCall:
    return ModelToolCall(
        id=call_id,
        name="web_fetch",
        arguments=json.dumps({"url": url, "format": "text"}),
    )


class FakeResolver:
    def __init__(self, answers: dict[str, tuple[str, ...]] | BaseException) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    async def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        self.calls.append((hostname, port))
        if isinstance(self.answers, BaseException):
            raise self.answers
        return self.answers[hostname]


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: tuple[bytes, ...] = (b"ok",),
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


class AuditedHTTPClient:
    def __init__(
        self,
        responses: tuple[FakeResponse, ...],
        *,
        expected_peers: tuple[str, ...] | None = None,
    ) -> None:
        self.responses = iter(responses)
        self.expected_peers = expected_peers
        self.calls: list[tuple[str, tuple[str, ...], float, float]] = []
        self.bytes_sent = 0

    async def get(
        self,
        url: str,
        *,
        resolved_addresses: tuple[str, ...],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        if self.expected_peers is not None and resolved_addresses != self.expected_peers:
            raise ToolError("connected peer was not in the audited address set")
        self.calls.append(
            (url, resolved_addresses, connect_timeout_seconds, total_timeout_seconds)
        )
        self.bytes_sent += 1
        return cast(HTTPResponseBoundary, next(self.responses))


def _gateway(
    *,
    resolver: DNSResolver,
    http: HTTPClientBoundary,
    level: ToolPermissionLevel,
    tool: WebFetchTool | None = None,
) -> ToolGateway:
    fetch = (
        WebFetchTool(resolver=resolver, http_client=http)
        if tool is None
        else tool
    )
    gateway = ToolGateway._for_memory(
        (fetch,),
        permission_context=PermissionContext(
            level=level,
            workspace_root=Path.cwd(),
        ),
    )
    return gateway


async def _call_gateway(
    gateway: ToolGateway,
    call: ModelToolCall,
    confirmation: ConfirmationRequester | None,
) -> ToolResult:
    return await gateway.call(call, confirmation=confirmation)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "addresses"),
    (
        ("public", ("93.184.216.34",)),
        ("private", ("10.0.0.7",)),
        ("loopback", ("127.0.0.1",)),
        ("link-local", ("169.254.1.7",)),
        ("reserved", ("192.0.2.7",)),
        ("unspecified", ("0.0.0.0",)),
        ("multicast", ("224.0.0.7",)),
        ("ipv6-public", ("2001:4860:4860::8888",)),
        ("ipv6-site-local", ("fec0::1",)),
        ("mapped-private", ("::ffff:10.0.0.7",)),
        ("mixed", ("93.184.216.34", "10.0.0.7")),
        ("empty", ()),
    ),
)
@pytest.mark.parametrize("level", ("read-only", "workspace-write", "full-access"))
async def test_web_fetch_target_level_matrix(
    label: str,
    addresses: tuple[str, ...],
    level: ToolPermissionLevel,
) -> None:
    resolver = FakeResolver({"target.example": addresses})
    http = AuditedHTTPClient((FakeResponse(),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level=level),
        _call("https://target.example/page"),
        approve,
    )

    if label == "empty":
        assert result.status == "error"
        assert http.calls == []
        if level == "full-access":
            assert requests == []
        else:
            assert len(requests) == 1
        return
    assert result.status == "success", label
    if label in {"public", "ipv6-public"} or level == "full-access":
        assert requests == []
    else:
        assert len(requests) == 1
    assert http.calls == [
        ("https://target.example/page", addresses, 10.0, 30.0)
    ]


@pytest.mark.asyncio
async def test_web_fetch_dns_failure_is_confirmed_then_remains_an_ordinary_error() -> None:
    resolver = FakeResolver(OSError("DNS unavailable"))
    http = AuditedHTTPClient(())
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="workspace-write"),
        _call("https://missing.example/page"),
        approve,
    )

    assert result.status == "error"
    assert "DNS" in result.content
    assert len(requests) == 1
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_declining_an_unsafe_initial_target_sends_zero_request_bytes() -> None:
    resolver = FakeResolver({"private.example": ("192.168.1.7",)})
    http = AuditedHTTPClient(())
    requests: list[ConfirmationRequest] = []

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call("https://private.example/admin"),
        decline,
    )

    assert result.status == "refused"
    assert result.confirmation is not None
    assert result.confirmation.decision == "declined"
    assert http.calls == []
    assert http.bytes_sent == 0


@pytest.mark.asyncio
async def test_web_fetch_public_to_private_redirect_uses_one_popup_and_audited_addresses() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "private.example": ("10.0.0.7",),
        }
    )
    redirect = FakeResponse(status_code=302, headers={"location": "http://private.example/x"})
    final = FakeResponse(chunks=(b"private result",))
    http = AuditedHTTPClient((redirect, final))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call("https://public.example/start"),
        approve,
    )

    assert result.status == "success"
    assert len(requests) == 1
    assert [call[:2] for call in http.calls] == [
        ("https://public.example/start", ("93.184.216.34",)),
        ("http://private.example/x", ("10.0.0.7",)),
    ]
    assert redirect.closed
    assert final.closed


@pytest.mark.asyncio
async def test_web_fetch_redirect_decline_sends_no_bytes_to_unsafe_target() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "private.example": ("10.0.0.7",),
        }
    )
    redirect = FakeResponse(status_code=302, headers={"location": "http://private.example/x"})
    http = AuditedHTTPClient((redirect,))

    async def decline(request: ConfirmationRequest) -> ConfirmationDecision:
        del request
        return "declined"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="workspace-write"),
        _call("https://public.example/start"),
        decline,
    )

    assert result.status == "refused"
    assert len(http.calls) == 1
    assert http.bytes_sent == 1
    assert redirect.closed


@pytest.mark.asyncio
async def test_web_fetch_approved_call_does_not_prompt_again_for_multiple_unsafe_redirects() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "private-one.example": ("10.0.0.7",),
            "private-two.example": ("192.168.1.7",),
        }
    )
    responses = tuple(
        [
            FakeResponse(status_code=302, headers={"location": "http://private-one.example/one"}),
            FakeResponse(status_code=302, headers={"location": "http://private-two.example/two"}),
            FakeResponse(chunks=(b"done",)),
        ]
    )
    http = AuditedHTTPClient(responses)
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="workspace-write"),
        _call("https://public.example/start"),
        approve,
    )

    assert result.status == "success"
    assert len(requests) == 1
    assert [call[1] for call in http.calls] == [
        ("93.184.216.34",),
        ("10.0.0.7",),
        ("192.168.1.7",),
    ]


@pytest.mark.asyncio
async def test_web_fetch_private_to_public_redirect_keeps_the_single_call_approval() -> None:
    resolver = FakeResolver(
        {
            "private.example": ("10.0.0.7",),
            "public.example": ("93.184.216.34",),
        }
    )
    redirect = FakeResponse(status_code=302, headers={"location": "https://public.example/final"})
    final = FakeResponse(chunks=(b"public result",))
    http = AuditedHTTPClient((redirect, final))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call("https://private.example/start"),
        approve,
    )

    assert result.status == "success"
    assert len(requests) == 1
    assert [call[1] for call in http.calls] == [("10.0.0.7",), ("93.184.216.34",)]


@pytest.mark.asyncio
async def test_web_fetch_does_not_cache_approval_between_calls() -> None:
    resolver = FakeResolver({"private.example": ("10.0.0.7",)})
    http = AuditedHTTPClient((FakeResponse(), FakeResponse()))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    gateway = _gateway(resolver=resolver, http=http, level="workspace-write")
    first = await _call_gateway(gateway, _call("https://private.example/one", call_id="one"), approve)
    second = await _call_gateway(
        gateway,
        _call("https://private.example/two", call_id="two"),
        approve,
    )

    assert first.status == "success"
    assert second.status == "success"
    assert [request.tool_call_id for request in requests] == ["one", "two"]
    assert resolver.calls == [
        ("private.example", 443),
        ("private.example", 443),
    ]


@pytest.mark.asyncio
async def test_web_fetch_concurrent_calls_keep_authorization_sessions_isolated() -> None:
    resolver = FakeResolver({"private.example": ("10.0.0.7",)})
    http = AuditedHTTPClient((FakeResponse(),))
    pending: dict[str, asyncio.Event] = {"first": asyncio.Event(), "second": asyncio.Event()}
    requests: list[ConfirmationRequest] = []

    async def decide(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        if request.tool_call_id == "first":
            pending["first"].set()
            return "approved"
        pending["second"].set()
        return "declined"

    gateway = _gateway(resolver=resolver, http=http, level="workspace-write")
    first = asyncio.create_task(
        _call_gateway(gateway, _call("https://private.example/one", call_id="first"), decide)
    )
    second = asyncio.create_task(
        _call_gateway(gateway, _call("https://private.example/two", call_id="second"), decide)
    )
    await asyncio.gather(pending["first"].wait(), pending["second"].wait())
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result.status == "success"
    assert second_result.status == "refused"
    assert [request.tool_call_id for request in requests] == ["first", "second"]
    assert len(http.calls) == 1


@pytest.mark.asyncio
async def test_web_fetch_binds_to_the_audited_dns_answer_without_a_second_resolution() -> None:
    resolver = FakeResolver({"rebind.example": ("93.184.216.34",)})
    http = AuditedHTTPClient((FakeResponse(),), expected_peers=("93.184.216.34",))

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call("https://rebind.example/page"),
        None,
    )

    assert result.status == "success"
    assert resolver.calls == [("rebind.example", 443)]
    assert http.calls[0][1] == ("93.184.216.34",)
    assert http.bytes_sent == 1


@pytest.mark.asyncio
async def test_web_fetch_rejects_an_unaudited_peer_before_sending_bytes() -> None:
    resolver = FakeResolver({"rebind.example": ("93.184.216.34",)})
    http = AuditedHTTPClient((FakeResponse(),), expected_peers=("203.0.113.9",))

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="full-access"),
        _call("https://rebind.example/page"),
        None,
    )

    assert result.status == "error"
    assert http.calls == []
    assert http.bytes_sent == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ("read-only", "workspace-write", "full-access"))
async def test_web_fetch_connection_errors_remain_errors_at_every_level(
    level: ToolPermissionLevel,
) -> None:
    resolver = FakeResolver({"public.example": ("93.184.216.34",)})

    class FailingHTTP:
        async def get(
            self,
            url: str,
            *,
            resolved_addresses: tuple[str, ...],
            connect_timeout_seconds: float,
            total_timeout_seconds: float,
        ) -> HTTPResponseBoundary:
            del url, resolved_addresses, connect_timeout_seconds, total_timeout_seconds
            raise OSError("connection failed")

    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=FailingHTTP(), level=level),
        _call("https://public.example/page"),
        approve,
    )

    assert result.status == "error"
    assert requests == []


@pytest.mark.asyncio
async def test_web_fetch_normalizes_idna_case_trailing_dot_and_explicit_port() -> None:
    resolver = FakeResolver({"xn--bcher-kva.example": ("93.184.216.34",)})
    http = AuditedHTTPClient((FakeResponse(),))

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call("HTTPS://BÜCHER.Example.:443/path"),
        None,
    )

    assert result.status == "success"
    assert resolver.calls == [("xn--bcher-kva.example", 443)]
    assert http.calls[0][0] == "https://xn--bcher-kva.example:443/path"


@pytest.mark.asyncio
async def test_web_fetch_literal_ipv6_preserves_port_and_uses_exact_address_without_dns() -> None:
    resolver = FakeResolver({})
    http = AuditedHTTPClient((FakeResponse(),))

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="full-access"),
        _call("http://[2001:4860:4860::8888]:8080/page"),
        None,
    )

    assert result.status == "success"
    assert resolver.calls == []
    assert http.calls[0][0] == "http://[2001:4860:4860::8888]:8080/page"
    assert http.calls[0][1] == ("2001:4860:4860::8888",)


@pytest.mark.asyncio
async def test_web_fetch_literal_private_target_is_static_and_does_not_resolve() -> None:
    resolver = FakeResolver(OSError("must not resolve a literal"))
    http = AuditedHTTPClient((FakeResponse(),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call("http://127.0.0.1:8080/admin"),
        approve,
    )

    assert result.status == "success"
    assert len(requests) == 1
    assert "private or non-global" in requests[0].reason
    assert resolver.calls == []
    assert http.calls[0][1] == ("127.0.0.1",)


@pytest.mark.asyncio
async def test_web_fetch_redirect_rejects_credentials_before_resolving_next_hop() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "private.example": ("10.0.0.7",),
        }
    )
    redirect = FakeResponse(
        status_code=302,
        headers={"location": "https://user:secret@private.example/admin"},
    )
    http = AuditedHTTPClient((redirect,))

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="full-access"),
        _call("https://public.example/start"),
        None,
    )

    assert result.status == "error"
    assert "redirect URL is invalid" in result.content
    assert resolver.calls == [("public.example", 443)]
    assert len(http.calls) == 1
    assert redirect.closed


@pytest.mark.asyncio
async def test_web_fetch_redirect_binds_changed_scheme_and_explicit_port() -> None:
    resolver = FakeResolver(
        {
            "public.example": ("93.184.216.34",),
            "next.example": ("8.8.8.8",),
        }
    )
    redirect = FakeResponse(
        status_code=302,
        headers={"location": "http://next.example:8080/final"},
    )
    final = FakeResponse(chunks=(b"changed",))
    http = AuditedHTTPClient((redirect, final))

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call("https://public.example/start"),
        None,
    )

    assert result.status == "success"
    assert resolver.calls == [
        ("public.example", 443),
        ("next.example", 8080),
    ]
    assert [call[0] for call in http.calls] == [
        "https://public.example/start",
        "http://next.example:8080/final",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ("read-only", "workspace-write", "full-access"))
async def test_web_fetch_dns_failure_is_an_error_at_every_permission_level(
    level: ToolPermissionLevel,
) -> None:
    resolver = FakeResolver(OSError("resolver failed"))
    http = AuditedHTTPClient(())
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level=level),
        _call("https://missing.example/page"),
        approve,
    )

    assert result.status == "error"
    assert len(requests) == (0 if level == "full-access" else 1)
    assert http.calls == []


@pytest.mark.asyncio
async def test_schedule_keeps_legacy_fail_closed_behavior_for_unsafe_web_fetch() -> None:
    resolver = FakeResolver({"private.example": ("10.0.0.7",)})
    http = AuditedHTTPClient(())
    gateway = ToolGateway._for_memory(
        (WebFetchTool(resolver=resolver, http_client=http),),
        permission_context=PermissionContext(
            origin="schedule",
            workspace_root=Path.cwd(),
        ),
    )

    result = await gateway.call(_call("https://private.example/admin"))

    assert result.status == "refused"
    assert result.content == "Tool confirmation is unavailable."
    assert resolver.calls == [("private.example", 443)]
    assert http.calls == []
    assert http.bytes_sent == 0


@pytest.mark.asyncio
async def test_web_fetch_popup_uses_the_original_normalized_invocation() -> None:
    resolver = FakeResolver({})
    http = AuditedHTTPClient((FakeResponse(),))
    requests: list[ConfirmationRequest] = []

    async def approve(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call(" HTTP://127.0.0.1:80/admin#credentials "),
        approve,
    )

    assert result.status == "success"
    assert len(requests) == 1
    assert requests[0].details == {
        "url": "http://127.0.0.1:80/admin",
        "format": "text",
        "maxChars": 50000,
    }
    assert resolver.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "normalized"),
    (
        ("127.1", "127.0.0.1"),
        ("2130706433", "127.0.0.1"),
        ("0177.0.0.1", "127.0.0.1"),
        ("0x7f000001", "127.0.0.1"),
    ),
)
async def test_web_fetch_canonicalizes_legacy_ipv4_literals_without_dns(
    source: str,
    normalized: str,
) -> None:
    resolver = FakeResolver({})
    http = AuditedHTTPClient((FakeResponse(),))

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="full-access"),
        _call(f"http://{source}:8080/status"),
        None,
    )

    assert result.status == "success"
    assert resolver.calls == []
    assert http.calls[0][:2] == (
        f"http://{normalized}:8080/status",
        (normalized,),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "http://[fe80::1%25Ethernet]/",
        "http://[fe80::1%Ethernet]/",
        "http://08.0.0.1/",
        "http://exa_mple.example/",
    ),
)
async def test_web_fetch_rejects_ambiguous_hosts_before_permission_or_dns(url: str) -> None:
    resolver = FakeResolver({})
    http = AuditedHTTPClient(())
    requests: list[ConfirmationRequest] = []

    async def unexpected(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "approved"

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="read-only"),
        _call(url),
        unexpected,
    )

    assert result.status == "error"
    assert requests == []
    assert resolver.calls == []
    assert http.calls == []


@pytest.mark.asyncio
async def test_web_fetch_resolves_relative_redirect_and_strips_fragments_per_hop() -> None:
    resolver = FakeResolver({"public.example": ("93.184.216.34",)})
    redirect = FakeResponse(
        status_code=302,
        headers={"location": "../final?view=text#server-fragment"},
    )
    final = FakeResponse(chunks=(b"done",))
    http = AuditedHTTPClient((redirect, final))

    result = await _call_gateway(
        _gateway(resolver=resolver, http=http, level="workspace-write"),
        _call("https://public.example/a/start#client-fragment"),
        None,
    )

    assert result.status == "success"
    assert resolver.calls == [
        ("public.example", 443),
        ("public.example", 443),
    ]
    assert [item[0] for item in http.calls] == [
        "https://public.example/a/start",
        "https://public.example/final?view=text",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ("read-only", "workspace-write", "full-access"))
async def test_web_search_is_direct_with_unchanged_schema_and_result(
    monkeypatch: pytest.MonkeyPatch,
    level: ToolPermissionLevel,
) -> None:
    calls: list[tuple[str, int]] = []

    def search(query: str, count: int) -> list[dict[str, str]]:
        calls.append((query, count))
        return [{"title": "Result", "href": "https://example.test/", "body": "Body"}]

    monkeypatch.setattr("omni.agent.tools.core.web_search._search_sync", search)
    tool = WebSearchTool()
    gateway = ToolGateway._for_memory(
        (tool,),
        permission_context=PermissionContext(level=level, workspace_root=Path.cwd()),
    )
    requests: list[ConfirmationRequest] = []

    async def unexpected(request: ConfirmationRequest) -> ConfirmationDecision:
        requests.append(request)
        return "declined"

    result = await gateway.call(
        ModelToolCall(
            id=f"search-{level}",
            name="web_search",
            arguments=json.dumps({"query": "  topic  ", "count": 1}),
        ),
        confirmation=unexpected,
    )

    assert gateway.schemas == [tool.to_schema()]
    assert result.status == "success"
    assert result.content == (
        "1. Title: Result\n   URL: https://example.test/\n   Snippet: Body"
    )
    assert calls == [("topic", 1)]
    assert requests == []


@pytest.mark.asyncio
async def test_aiohttp_client_uses_audited_ipv4_and_preserves_host_and_port() -> None:
    requests: list[bytes] = []

    async def serve(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 2\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            b"ok"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(serve, host="127.0.0.1", port=0)
    socket = server.sockets[0]
    assert socket is not None
    port = socket.getsockname()[1]
    try:
        response = await AioHttpWebFetchClient().get(
            f"http://audited.example:{port}/page",
            resolved_addresses=("127.0.0.1",),
            connect_timeout_seconds=1.0,
            total_timeout_seconds=5.0,
        )
        body = b"".join([chunk async for chunk in response.iter_bytes()])
        await response.close()
    finally:
        server.close()
        await server.wait_closed()

    assert body == b"ok"
    assert len(requests) == 1
    assert f"Host: audited.example:{port}".encode() in requests[0]


@pytest.mark.asyncio
async def test_aiohttp_client_ignores_system_rebinding_and_environment_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audited_requests: list[bytes] = []
    unaudited_bytes: list[bytes] = []

    async def serve_audited(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        audited_requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 1\r\n"
            b"Connection: close\r\n\r\nA"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def serve_unaudited(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        unaudited_bytes.append(await reader.read(4096))
        writer.close()
        await writer.wait_closed()

    audited_server = await asyncio.start_server(serve_audited, "127.0.0.1", 0)
    audited_socket = audited_server.sockets[0]
    assert audited_socket is not None
    port = int(audited_socket.getsockname()[1])
    unaudited_server = await asyncio.start_server(serve_unaudited, "127.0.0.2", port)

    loop = asyncio.get_running_loop()
    system_dns_calls: list[tuple[object, ...]] = []

    async def rebinding_dns(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
        del kwargs
        system_dns_calls.append(args)
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("127.0.0.2", port),
            )
        ]

    monkeypatch.setattr(loop, "getaddrinfo", rebinding_dns)
    proxy = f"http://127.0.0.2:{port}"
    monkeypatch.setenv("HTTP_PROXY", proxy)
    monkeypatch.setenv("http_proxy", proxy)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    try:
        response = await AioHttpWebFetchClient().get(
            f"http://rebind.example:{port}/resource",
            resolved_addresses=("127.0.0.1",),
            connect_timeout_seconds=1.0,
            total_timeout_seconds=5.0,
        )
        body = b"".join([chunk async for chunk in response.iter_bytes()])
        await response.close()
    finally:
        audited_server.close()
        unaudited_server.close()
        await audited_server.wait_closed()
        await unaudited_server.wait_closed()

    assert body == b"A"
    assert len(audited_requests) == 1
    assert system_dns_calls == []
    assert unaudited_bytes == []


@pytest.mark.asyncio
async def test_same_origin_redirect_uses_the_new_hop_audited_set() -> None:
    first_requests: list[bytes] = []
    second_requests: list[bytes] = []
    first_peer_closed = asyncio.Event()

    async def serve_first(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        first_requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            b"HTTP/1.1 302 Found\r\n"
            b"Location: /final\r\n"
            b"Content-Length: 0\r\n"
            b"Connection: keep-alive\r\n\r\n"
        )
        await writer.drain()
        await reader.read()
        first_peer_closed.set()
        writer.close()
        await writer.wait_closed()

    async def serve_second(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        second_requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 6\r\n"
            b"Connection: close\r\n\r\nsecond"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    first_server = await asyncio.start_server(serve_first, "127.0.0.1", 0)
    first_socket = first_server.sockets[0]
    assert first_socket is not None
    port = int(first_socket.getsockname()[1])
    second_server = await asyncio.start_server(serve_second, "127.0.0.2", port)

    class SequencedResolver:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []
            self._answers = iter((("127.0.0.1",), ("127.0.0.2",)))

        async def resolve(self, hostname: str, target_port: int) -> tuple[str, ...]:
            self.calls.append((hostname, target_port))
            return next(self._answers)

    resolver = SequencedResolver()
    try:
        result = await _call_gateway(
            _gateway(
                resolver=resolver,
                http=AioHttpWebFetchClient(),
                level="full-access",
            ),
            _call(f"http://same-origin.example:{port}/start"),
            None,
        )
        await asyncio.wait_for(first_peer_closed.wait(), timeout=1.0)
    finally:
        first_server.close()
        second_server.close()
        await first_server.wait_closed()
        await second_server.wait_closed()

    assert result.status == "success"
    assert result.content == "second"
    assert resolver.calls == [
        ("same-origin.example", port),
        ("same-origin.example", port),
    ]
    assert first_requests[0].startswith(b"GET /start HTTP/1.1\r\n")
    assert second_requests[0].startswith(b"GET /final HTTP/1.1\r\n")


@pytest.mark.asyncio
async def test_aiohttp_client_preserves_ipv6_literal_host_header() -> None:
    requests: list[bytes] = []

    async def serve(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(serve, "::1", 0)
    server_socket = server.sockets[0]
    assert server_socket is not None
    port = int(server_socket.getsockname()[1])
    try:
        response = await AioHttpWebFetchClient().get(
            f"http://[::1]:{port}/ipv6",
            resolved_addresses=("::1",),
            connect_timeout_seconds=1.0,
            total_timeout_seconds=5.0,
        )
        await response.close()
    finally:
        server.close()
        await server.wait_closed()

    assert len(requests) == 1
    assert f"Host: [::1]:{port}".encode() in requests[0]


@pytest.mark.asyncio
async def test_https_direct_ip_dial_preserves_sni_host_and_certificate_validation(
    tmp_path: Path,
) -> None:
    certificate_path = tmp_path / "certificate.pem"
    private_key_path = tmp_path / "private-key.pem"
    certificate_path.write_text(_TLS_CERTIFICATE, encoding="ascii", newline="\n")
    private_key_path.write_text(_TLS_PRIVATE_KEY, encoding="ascii", newline="\n")

    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(certificate_path, private_key_path)
    server_names: list[str | None] = []

    def capture_server_name(
        ssl_socket: ssl.SSLSocket | ssl.SSLObject,
        server_name: str | None,
        context: ssl.SSLSocket,
    ) -> int | None:
        del ssl_socket, context
        server_names.append(server_name)
        return None

    server_context.set_servername_callback(capture_server_name)
    requests: list[bytes] = []

    async def serve(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        requests.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0, ssl=server_context)
    server_socket = server.sockets[0]
    assert server_socket is not None
    port = int(server_socket.getsockname()[1])
    client_context = ssl.create_default_context(cafile=str(certificate_path))
    client = AioHttpWebFetchClient(ssl_context=client_context)
    try:
        response = await client.get(
            f"https://audited.example:{port}/secure",
            resolved_addresses=("127.0.0.1",),
            connect_timeout_seconds=1.0,
            total_timeout_seconds=5.0,
        )
        body = b"".join([chunk async for chunk in response.iter_bytes()])
        await response.close()

        with pytest.raises(ClientConnectorCertificateError):
            await client.get(
                f"https://wrong.example:{port}/secure",
                resolved_addresses=("127.0.0.1",),
                connect_timeout_seconds=1.0,
                total_timeout_seconds=5.0,
            )
    finally:
        server.close()
        await server.wait_closed()

    assert body == b"ok"
    assert server_names == ["audited.example", "wrong.example"]
    assert len(requests) == 1
    assert f"Host: audited.example:{port}".encode() in requests[0]
    assert client_context.check_hostname is True
    assert client_context.verify_mode == ssl.CERT_REQUIRED


@pytest.mark.asyncio
async def test_cancelling_web_fetch_closes_the_production_socket() -> None:
    request_received = asyncio.Event()
    peer_closed = asyncio.Event()

    async def serve(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 100000\r\n\r\npartial"
        )
        await writer.drain()
        request_received.set()
        await reader.read()
        peer_closed.set()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    server_socket = server.sockets[0]
    assert server_socket is not None
    port = int(server_socket.getsockname()[1])
    resolver = FakeResolver({"cancel.example": ("127.0.0.1",)})
    task = asyncio.create_task(
        _call_gateway(
            _gateway(
                resolver=resolver,
                http=AioHttpWebFetchClient(),
                level="full-access",
            ),
            _call(f"http://cancel.example:{port}/slow"),
            None,
        )
    )
    try:
        await asyncio.wait_for(request_received.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(peer_closed.wait(), timeout=1.0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        server.close()
        await server.wait_closed()
