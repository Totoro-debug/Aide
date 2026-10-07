"""Web Fetch Core Catalog Tool."""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from email.message import Message
from html import unescape
from ipaddress import IPv4Address, IPv6Address, ip_address
from socket import AF_INET, AF_INET6, AF_UNSPEC, IPPROTO_TCP
from ssl import SSLContext
from typing import Annotated, Any, Literal, Protocol, cast
from urllib.parse import SplitResult, urljoin, urlsplit, urlunsplit

from aiohttp import ClientResponse, ClientSession, ClientTimeout, TCPConnector
from aiohttp.abc import AbstractResolver, ResolveResult

from aide.agent.tools.base import (
    BaseTool,
    ToolError,
    ToolParam,
    is_public_ip,
    truncate_text,
)
from aide.agent.tools.core.exec_policy import ExecAssessment
from aide.agent.tools.network_safety import (
    DNSResolver,
    SocketDNSResolver,
    TargetResolution,
    resolve_target,
)
from aide.agent.tools.permission import (
    NetworkAssessment,
    NetworkTargetRisk,
    NormalizedNetworkTarget,
    ToolAuthorizationSession,
    ToolInvocationFacts,
)

CONNECT_TIMEOUT_SECONDS = 10.0
TOTAL_TIMEOUT_SECONDS = 30.0
MAX_REDIRECTS = 5
DEFAULT_MAX_CHARS = 50000

_TEXTUAL_APPLICATION_MEDIA_TYPES = frozenset(
    {
        "application/ecmascript",
        "application/javascript",
        "application/json",
        "application/sql",
        "application/x-javascript",
        "application/x-yaml",
        "application/xml",
        "application/yaml",
    }
)
_HTML_MEDIA_TYPES = frozenset({"application/xhtml+xml", "text/html"})
_HTML_IGNORED_ELEMENTS = ("noscript", "script", "style", "template")
_HTML_BLOCK_TAG_PATTERN = re.compile(
    r"</?(?:address|article|aside|blockquote|br|div|footer|h[1-6]|header|li|main|nav|p|"
    r"pre|section|table|td|th|title|tr)\b[^>]*>",
    re.IGNORECASE,
)
_HTML_TAG_PATTERN = re.compile(r"<[^>]*>")


class HTTPResponseBoundary(Protocol):
    status_code: int
    headers: Mapping[str, str]

    def iter_bytes(self) -> AsyncIterator[bytes]: ...

    async def close(self) -> None: ...


class HTTPClientBoundary(Protocol):
    async def get(
        self,
        url: str,
        *,
        resolved_addresses: tuple[str, ...],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary: ...


class _AuditedResolver(AbstractResolver):
    """Expose only the address set audited by the Web Fetch Tool."""

    def __init__(self, *, hostname: str, addresses: tuple[str, ...]) -> None:
        self._hostname = hostname
        self._addresses = addresses

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: int = AF_UNSPEC,
    ) -> list[ResolveResult]:
        if host.casefold().rstrip(".") != self._hostname.casefold().rstrip("."):
            raise OSError("Web Fetch connector hostname did not match the audited target")
        results: list[ResolveResult] = []
        for address in self._addresses:
            try:
                parsed = ip_address(address)
            except ValueError as error:
                raise OSError("Web Fetch resolver returned an invalid address") from error
            address_family = AF_INET6 if isinstance(parsed, IPv6Address) else AF_INET
            if family not in {AF_UNSPEC, address_family}:
                continue
            results.append(
                {
                    "hostname": self._hostname,
                    "host": address,
                    "port": port,
                    "family": address_family,
                    "proto": IPPROTO_TCP,
                    "flags": 0,
                }
            )
        if not results:
            raise OSError("Web Fetch audited address set has no compatible address")
        return results

    async def close(self) -> None:
        return None


class _AioHttpResponse:
    def __init__(self, *, session: ClientSession, response: ClientResponse) -> None:
        self._session = session
        self._response = response
        self.status_code = response.status
        self.headers: Mapping[str, str] = response.headers

    async def iter_bytes(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.content.iter_chunked(64 * 1024):
            yield chunk

    async def close(self) -> None:
        self._response.close()
        await self._session.close()


class AioHttpWebFetchClient:
    """Perform one no-redirect GET with the caller's bounded timeout."""

    def __init__(self, *, ssl_context: SSLContext | None = None) -> None:
        self._ssl_context = ssl_context

    async def get(
        self,
        url: str,
        *,
        resolved_addresses: tuple[str, ...],
        connect_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> HTTPResponseBoundary:
        parsed = _parse_url(url)
        hostname = parsed.hostname
        if hostname is None:
            raise ToolError("Web Fetch URL must contain a hostname.")
        session = ClientSession(
            timeout=ClientTimeout(
                total=total_timeout_seconds,
                connect=connect_timeout_seconds,
                sock_connect=connect_timeout_seconds,
            ),
            auto_decompress=True,
            trust_env=False,
            connector=TCPConnector(
                resolver=_AuditedResolver(
                    hostname=hostname,
                    addresses=resolved_addresses,
                ),
                use_dns_cache=False,
                force_close=True,
                ssl=True if self._ssl_context is None else self._ssl_context,
            ),
        )
        try:
            response = await session.get(url, allow_redirects=False)
        except BaseException:
            await _close_session(session)
            raise
        return _AioHttpResponse(session=session, response=response)


@dataclass(frozen=True, slots=True)
class _TargetEvaluation:
    target: NormalizedNetworkTarget
    static_risk: NetworkTargetRisk | None


class WebFetchTool(BaseTool):
    """Fetch readable web content through per-hop audited connections."""

    name = "web_fetch"
    description = "Fetch readable content from an HTTP or HTTPS URL."
    required = ("url",)

    url: Annotated[
        str,
        ToolParam(description="HTTP or HTTPS URL to fetch.", min_length=1, format="uri"),
    ]
    format: Annotated[
        str,
        ToolParam(description="Output format: markdown or text.", min_length=1),
    ] = "markdown"
    maxChars: Annotated[
        int,
        ToolParam(description="Maximum returned characters.", minimum=1),
    ] = DEFAULT_MAX_CHARS

    def __init__(
        self,
        *,
        resolver: DNSResolver | None = None,
        http_client: HTTPClientBoundary | None = None,
    ) -> None:
        self._resolver = SocketDNSResolver() if resolver is None else resolver
        self._http_client = AioHttpWebFetchClient() if http_client is None else http_client

    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        prepared = await super().prepare_arguments(arguments)
        url = prepared.get("url")
        if not isinstance(url, str):
            raise ToolError("Web Fetch URL is invalid.")
        try:
            prepared["url"] = _normalize_url(url.strip()).url
        except ValueError as error:
            raise ToolError(f"Web Fetch URL is invalid: {error}") from error
        return prepared

    def validate_arguments(  # type: ignore[override]
        self,
        *,
        url: str,
        format: str,
        maxChars: int,
    ) -> str | None:
        del maxChars
        try:
            _normalize_url(url.strip())
        except ValueError as error:
            return f"Web Fetch URL is invalid: {error}"
        if format not in {"markdown", "text"}:
            return "Web Fetch format must be either markdown or text."
        return None

    def build_invocation_facts(
        self,
        prepared_arguments: dict[str, Any],
    ) -> ToolInvocationFacts:
        url = prepared_arguments.get("url")
        if not isinstance(url, str):
            raise ToolError("Web Fetch URL is invalid.")
        evaluation = self._evaluate_target(url)
        return ToolInvocationFacts(
            tool_name=self.name,
            normalized_arguments=prepared_arguments,
            network_targets=(
                NetworkAssessment(
                    target=evaluation.target,
                    static_risk=evaluation.static_risk,
                ),
            ),
        )

    async def execute_authorized(
        self,
        arguments: dict[str, object],
        authorization: ToolAuthorizationSession,
    ) -> str:
        url = arguments.get("url")
        output_format = arguments.get("format")
        max_chars = arguments.get("maxChars")
        if (
            not isinstance(url, str)
            or not isinstance(output_format, str)
            or isinstance(max_chars, bool)
            or not isinstance(max_chars, int)
        ):
            raise ToolError("Web Fetch arguments are invalid.")
        return await self._execute_fetch(
            url,
            output_format,
            max_chars,
            authorization,
        )

    async def execute(self, *, url: str, format: str, maxChars: int) -> str:
        return await self._execute_fetch(
            url,
            format,
            maxChars,
            _DirectNetworkAuthorization(),
        )

    async def _execute_fetch(
        self,
        url: str,
        output_format: str,
        max_chars: int,
        authorization: ToolAuthorizationSession,
    ) -> str:
        evaluation = self._evaluate_target(url)
        resolution = await self._resolve_target(evaluation.target)
        await authorization.authorize_network_target(
            evaluation.target,
            resolution.addresses,
        )
        _raise_resolution_error(evaluation.target, resolution)

        try:
            async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
                content = await self._fetch_direct(
                    evaluation.target,
                    resolution,
                    authorization,
                )
                return truncate_text(content, limit=max_chars)
        except asyncio.CancelledError:
            raise
        except TimeoutError as error:
            raise ToolError(
                f"Web Fetch timed out after {TOTAL_TIMEOUT_SECONDS:g} seconds."
            ) from error

    def _evaluate_target(self, url: str) -> _TargetEvaluation:
        try:
            target = _normalize_url(url.strip())
        except ValueError as error:
            raise ToolError(f"Web Fetch URL is invalid: {error}") from error

        return _TargetEvaluation(
            target=target,
            static_risk=(
                "literal_non_global"
                if not is_public_ip(target.host) and _is_ip(target.host)
                else None
            ),
        )

    async def _resolve_target(self, target: NormalizedNetworkTarget) -> TargetResolution:
        try:
            return await asyncio.wait_for(
                resolve_target(target.host, target.port, self._resolver),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return TargetResolution(
                addresses=(),
                risk="dns_failure",
                error_message="DNS resolution timed out.",
            )

    async def _fetch_direct(
        self,
        target: NormalizedNetworkTarget,
        resolution: TargetResolution,
        authorization: ToolAuthorizationSession,
    ) -> str:
        redirects_followed = 0
        while True:
            response = await self._http_client.get(
                target.url,
                resolved_addresses=resolution.addresses,
                connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS,
                total_timeout_seconds=TOTAL_TIMEOUT_SECONDS,
            )
            try:
                location = _header(response.headers, "location")
                if 300 <= response.status_code < 400 and location:
                    if redirects_followed >= MAX_REDIRECTS:
                        raise ToolError("Web Fetch redirect limit exceeded.")
                    try:
                        next_target = _normalize_url(urljoin(target.url, location))
                    except ValueError as error:
                        raise ToolError(f"Web Fetch redirect URL is invalid: {error}") from error
                    next_resolution = await self._resolve_target(next_target)
                    await authorization.authorize_network_target(
                        next_target,
                        next_resolution.addresses,
                    )
                    _raise_resolution_error(next_target, next_resolution)
                    redirects_followed += 1
                    target = next_target
                    resolution = next_resolution
                    continue

                content_type = _header(response.headers, "content-type")
                if not _is_textual_media_type(_media_type(content_type)):
                    raise ToolError("Web Fetch response media type is unsupported.")
                body = await _read_body(response)
                return _decode_response(
                    body,
                    content_type=content_type,
                )
            finally:
                await response.close()


class _DirectNetworkAuthorization:
    exec_assessment: ExecAssessment | None = None

    def initial_decision(self) -> Literal["direct"]:
        return "direct"

    def confirmation_reason(self) -> str:
        return "Tool confirmation is required."

    async def authorize_network_target(
        self,
        target: NormalizedNetworkTarget,
        resolved_addresses: tuple[str, ...],
    ) -> None:
        del target, resolved_addresses


async def _read_body(response: HTTPResponseBoundary) -> bytes:
    chunks = [chunk async for chunk in response.iter_bytes()]
    return b"".join(chunks)


async def _close_session(session: ClientSession) -> None:
    try:
        await session.close()
    except BaseException:
        return None


def _parse_url(url: str) -> SplitResult:
    if not url:
        raise ValueError("URL must not be blank")
    if any(character.isspace() or ord(character) < 32 for character in url):
        raise ValueError("URL must not contain whitespace or control characters")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("URL is malformed") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("URL scheme must be HTTP or HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL userinfo is not supported")
    if not parsed.netloc or hostname is None:
        raise ValueError("URL must contain a hostname")
    if port == 0:
        raise ValueError("URL port must not be zero")
    return parsed


def _normalize_url(url: str) -> NormalizedNetworkTarget:
    parsed = _parse_url(url)
    if parsed.hostname is None:
        raise ValueError("URL must contain a hostname")
    scheme = cast(Literal["http", "https"], parsed.scheme.lower())
    host = _normalize_hostname(parsed.hostname)
    port = _effective_port(parsed)
    host_for_url = f"[{host}]" if ":" in host else host
    authority = host_for_url if parsed.port is None else f"{host_for_url}:{port}"
    normalized_url = urlunsplit((scheme, authority, parsed.path, parsed.query, ""))
    return NormalizedNetworkTarget(
        url=normalized_url,
        scheme=scheme,
        host=host,
        port=port,
    )


def _normalize_hostname(hostname: str) -> str:
    if "%" in hostname:
        raise ValueError("URL IPv6 zone identifiers are not supported")
    try:
        return str(ip_address(hostname))
    except ValueError:
        pass

    ipv4 = _parse_legacy_ipv4(hostname)
    if ipv4 is not None:
        return str(ipv4)

    hostname = hostname.rstrip(".")
    if not hostname:
        raise ValueError("URL hostname is invalid") from None
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as error:
        raise ValueError("URL hostname is invalid") from error
    labels = ascii_hostname.split(".")
    if (
        len(ascii_hostname) > 253
        or any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or re.fullmatch(r"[a-z0-9-]+", label) is None
            for label in labels
        )
    ):
        raise ValueError("URL hostname is invalid")
    return ascii_hostname


def _parse_legacy_ipv4(hostname: str) -> IPv4Address | None:
    parts = hostname.split(".")
    if not 1 <= len(parts) <= 4 or any(not part for part in parts):
        return None

    values: list[int] = []
    for part in parts:
        try:
            if part.lower().startswith("0x"):
                if len(part) == 2:
                    return None
                value = int(part[2:], 16)
            elif len(part) > 1 and part.startswith("0"):
                value = int(part[1:], 8) if part[1:] else 0
            elif part.isdecimal():
                value = int(part, 10)
            else:
                return None
        except ValueError as error:
            raise ValueError("URL IPv4 address is invalid") from error
        values.append(value)

    widths = {
        1: (32,),
        2: (8, 24),
        3: (8, 8, 16),
        4: (8, 8, 8, 8),
    }[len(values)]
    if any(value >= 1 << width for value, width in zip(values, widths, strict=True)):
        raise ValueError("URL IPv4 address is invalid")

    packed = 0
    for value, width in zip(values, widths, strict=True):
        packed = (packed << width) | value
    return IPv4Address(packed)


def _is_ip(value: str) -> bool:
    try:
        ip_address(value)
    except ValueError:
        return False
    return True


def _raise_resolution_error(
    target: NormalizedNetworkTarget,
    resolution: TargetResolution,
) -> None:
    if resolution.risk == "dns_failure":
        detail = resolution.error_message or "DNS resolution failed."
        raise ToolError(f"Web Fetch DNS resolution failed: {detail}")
    if resolution.risk == "dns_empty":
        raise ToolError(
            f"Web Fetch DNS resolution returned no addresses for {target.host}."
        )


def _effective_port(parsed: SplitResult) -> int:
    port = parsed.port
    if port is not None:
        return port
    return 443 if parsed.scheme.lower() == "https" else 80


def _header(headers: Mapping[str, str], name: str) -> str | None:
    normalized_name = name.lower()
    return next(
        (value for key, value in headers.items() if key.lower() == normalized_name),
        None,
    )


def _media_type(content_type: str | None) -> str:
    if content_type is None:
        return ""
    return content_type.split(";", maxsplit=1)[0].strip().lower()


def _is_textual_media_type(media_type: str) -> bool:
    return (
        not media_type
        or media_type.startswith("text/")
        or media_type in _TEXTUAL_APPLICATION_MEDIA_TYPES
        or media_type.endswith("+json")
        or media_type.endswith("+xml")
    )


def _charset(content_type: str | None) -> str:
    if content_type is None:
        return "utf-8"
    message = Message()
    message["content-type"] = content_type
    return message.get_content_charset() or "utf-8"


def _decode_response(
    body: bytes,
    *,
    content_type: str | None,
) -> str:
    media_type = _media_type(content_type)
    if not _is_textual_media_type(media_type):
        raise ToolError("Web Fetch response media type is unsupported.")
    try:
        charset = _charset(content_type)
    except (LookupError, ValueError):
        charset = "utf-8"
    try:
        content = body.decode(charset, errors="replace")
    except LookupError:
        content = body.decode("utf-8", errors="replace")
    if media_type in _HTML_MEDIA_TYPES:
        return _readable_html(content)
    return content


def _readable_html(content: str) -> str:
    for tag in _HTML_IGNORED_ELEMENTS:
        content = re.sub(
            rf"<{tag}\b[^>]*>.*?(?:</{tag}\s*>|$)",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
    content = _HTML_BLOCK_TAG_PATTERN.sub("\n", content)
    content = _HTML_TAG_PATTERN.sub("", content)
    content = unescape(content)
    lines = (" ".join(line.split()) for line in content.splitlines())
    return "\n".join(line for line in lines if line)


__all__ = [
    "CONNECT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_CHARS",
    "MAX_REDIRECTS",
    "TOTAL_TIMEOUT_SECONDS",
    "AioHttpWebFetchClient",
    "HTTPClientBoundary",
    "HTTPResponseBoundary",
    "WebFetchTool",
]
