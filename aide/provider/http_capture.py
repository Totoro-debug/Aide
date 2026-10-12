"""Best-effort capture at the OpenAI-compatible HTTP transport boundary."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import cast
from urllib.parse import urldefrag

import httpx

from aide.trace.sanitization import is_sensitive_trace_name, sanitize_trace_url
from aide.trace.scope import TraceProviderAttempt, current_trace_provider_attempt
from aide.trace.types import TraceCaptureLayer, TraceDirection


@dataclass(frozen=True, slots=True)
class _TraceHTTPExchange:
    attempt: TraceProviderAttempt
    index: int


_EXCHANGE_EXTENSION = "aide.trace_http_exchange"


async def capture_http_request(request: httpx.Request) -> None:
    """Capture serialized bytes without replacing the client's transport routing."""
    request.extensions.pop(_EXCHANGE_EXTENSION, None)
    attempt = current_trace_provider_attempt()
    if attempt is None:
        return
    try:
        exchange = attempt.begin_http_exchange()
        request.extensions[_EXCHANGE_EXTENSION] = _TraceHTTPExchange(attempt, exchange)
        request_source = _exchange_source("http_request", exchange)
        request_metadata_source = _exchange_source("http_request_metadata", exchange)
        endpoint = urldefrag(sanitize_trace_url(str(request.url)))[0]
        attempt.set_endpoint(endpoint)
        try:
            request_body = request.content
        except httpx.RequestNotRead:
            request_body = None
        if request_body is None:
            attempt.report_capture_issue("Provider request body was unavailable for capture.")
            attempt.request_captured = False
        else:
            metadata_captured = _append_json(
                attempt,
                source=request_metadata_source,
                direction=TraceDirection.REQUEST,
                data={
                    "method": request.method,
                    "url": endpoint,
                    "headers": _safe_headers(request.headers),
                },
            )
            body_captured = attempt.append_raw(
                source=request_source,
                direction=TraceDirection.REQUEST,
                layer=TraceCaptureLayer.HTTP_BODY,
                sequence=0,
                data=request_body,
                content_type=request.headers.get("content-type"),
                content_encoding=request.headers.get("content-encoding"),
            )
            attempt.request_captured = metadata_captured and body_captured

    except Exception:
        attempt.report_capture_issue("Provider HTTP request capture failed.")


async def capture_http_response(response: httpx.Response) -> None:
    """Wrap raw response consumption without reading ahead of the SDK."""
    exchange = response.request.extensions.get(_EXCHANGE_EXTENSION)
    if not isinstance(exchange, _TraceHTTPExchange):
        return
    attempt = exchange.attempt
    try:
        response_metadata_captured = _append_json(
            attempt,
            source=_exchange_source("http_response_metadata", exchange.index),
            direction=TraceDirection.RESPONSE,
            data={
                "status_code": response.status_code,
                "headers": _safe_headers(response.headers),
            },
        )
        attempt.response_metadata_captured = response_metadata_captured
        response.stream = _TraceCaptureByteStream(
            cast(httpx.AsyncByteStream, response.stream),
            attempt,
            source=_exchange_source("http_response", exchange.index),
            content_type=response.headers.get("content-type"),
            content_encoding=response.headers.get("content-encoding"),
            metadata_captured=response_metadata_captured,
        )
    except Exception:
        attempt.report_capture_issue("Provider HTTP response capture failed.")


class _TraceCaptureByteStream(httpx.AsyncByteStream):
    def __init__(
        self,
        stream: httpx.AsyncByteStream,
        attempt: TraceProviderAttempt,
        *,
        source: str,
        content_type: str | None,
        content_encoding: str | None,
        metadata_captured: bool,
    ) -> None:
        self._stream = stream
        self._attempt = attempt
        self._source = source
        self._content_type = content_type
        self._content_encoding = content_encoding
        self._metadata_captured = metadata_captured

    async def __aiter__(self) -> AsyncIterator[bytes]:
        sequence = 0
        captured = self._metadata_captured
        async for chunk in self._stream:
            chunk_captured = self._attempt.append_raw(
                source=self._source,
                direction=TraceDirection.RESPONSE,
                layer=TraceCaptureLayer.HTTP_BODY,
                sequence=sequence,
                data=chunk,
                content_type=self._content_type,
                content_encoding=self._content_encoding,
            )
            self._attempt.response_body_started |= chunk_captured
            captured = chunk_captured and captured
            sequence += 1
            yield chunk
        if sequence == 0:
            empty_captured = self._attempt.append_raw(
                source=self._source,
                direction=TraceDirection.RESPONSE,
                layer=TraceCaptureLayer.HTTP_BODY,
                sequence=0,
                data=b"",
                content_type=self._content_type,
                content_encoding=self._content_encoding,
            )
            self._attempt.response_body_started = empty_captured
            captured = empty_captured and captured
        self._attempt.response_captured = captured

    async def aclose(self) -> None:
        await self._stream.aclose()


def _append_json(
    attempt: TraceProviderAttempt,
    *,
    source: str,
    direction: TraceDirection,
    data: object,
) -> bool:
    try:
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        attempt.report_capture_issue(f"Provider {source} metadata could not be serialized.")
        return False
    return attempt.append_raw(
        source=source,
        direction=direction,
        layer=TraceCaptureLayer.HTTP_METADATA,
        sequence=0,
        data=encoded,
        content_type="application/json",
    )


def _exchange_source(source: str, exchange: int) -> str:
    return source if exchange == 0 else f"{source}_{exchange + 1}"


def _safe_headers(headers: httpx.Headers) -> dict[str, object]:
    values: dict[str, list[str]] = {}
    for name, value in headers.multi_items():
        normalized = name.casefold()
        if is_sensitive_trace_name(normalized):
            continue
        if normalized in {"location", "content-location", "referer"}:
            value = urldefrag(sanitize_trace_url(value))[0]
        elif normalized == "link":
            value = re.sub(
                r"<([^<>]+)>",
                lambda match: f"<{urldefrag(sanitize_trace_url(match[1]))[0]}>",
                value,
            )
        values.setdefault(normalized, []).append(value)
    return {
        name: items[0] if len(items) == 1 else items
        for name, items in values.items()
    }


__all__ = ["capture_http_request", "capture_http_response"]
